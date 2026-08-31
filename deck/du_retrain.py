"""Phase 2b: retrain the CLEAN reproducible DU-MFOCUSS (front-cone dictionary, budget parity,
reduce-to-MFOCUSS init) on the recorded ULA3 @150 manifold, mix of single + non-coherent +
coherent front-cone pairs. Trains the K=20 learned layers (extend applied only at eval). Saves a
single clean checkpoint. Smoke mode (--smoke) runs 3 epochs / small N to verify it trains."""
import sys, os, copy, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r"c:/GitHub/DUNCS"); os.chdir(r"c:/GitHub/DUNCS")
import numpy as np, torch
from src.config.simulation_config import load_simulation_config
from src.system_model import SystemModel
from src.signal_creation import Samples
from src.models import ModelGenerator
from src.utils import device

SMOKE = "--smoke" in sys.argv
DATASIM = "--datasim" in sys.argv
EPOCHS = 3 if SMOKE else 80
N_TR = 400 if SMOKE else 5000
N_VA = 200 if SMOKE else 1000
BATCH = 128
OUT = ("c:/GitHub/DUNCS/data/weights/DUMFOCUSS_clean_K20_grid901_frontcone_datasim_N=5_T=8.pt" if DATASIM
       else "c:/GitHub/DUNCS/data/weights/DUMFOCUSS_clean_K20_grid901_frontcone_N=5_T=8.pt")

cfg = load_simulation_config("src/config/subspaceNet.yaml")
cfg.system_model.M = 2
SM = SystemModel(cfg.system_model)
SAMP = Samples(cfg.system_model, cfg.system_model.antenna_pattern)

def steer(a_deg):
    return np.asarray(SM.steering_vec(float(np.deg2rad(a_deg))))

def scene(angles_deg, coherent, powers, rng):
    M = len(angles_deg)
    SAMP.set_doa([float(a) for a in angles_deg], M)
    SAMP.params.signal_nature = "coherent" if coherent else "non-coherent"
    clean, noise = SAMP.samples_creation(noise_variance=1, signal_variance=1, source_number=M)
    clean = np.asarray(clean); noise = np.asarray(noise)
    Amat = np.stack([steer(a) for a in angles_deg], axis=1)
    base = np.linalg.lstsq(Amat, clean, rcond=None)[0]
    if powers is not None:
        clean = Amat @ (base * np.sqrt(np.asarray(powers))[:, None]); base = np.linalg.lstsq(Amat, clean, rcond=None)[0]
    if DATASIM:                                    # 1-3 weak coherent multipath reflections per source
        for si, a in enumerate(angles_deg):
            for _ in range(int(rng.integers(1, 4))):
                ref = float(np.clip(a + rng.uniform(-40, 40), -70, 70))
                g = rng.uniform(0.2, 0.6) * np.exp(1j * rng.uniform(0, 2 * np.pi))
                clean = clean + (g * steer(ref)[:, None]) * base[si:si + 1]
    snr = 10 ** (rng.uniform(25, 30) / 10)
    x = clean + noise * np.sqrt(1.0 / snr)
    return torch.tensor(x, dtype=torch.complex128), M, np.deg2rad(np.sort(angles_deg))

def make_set(n, seed):
    rng = np.random.default_rng(seed); data = []
    for _ in range(n):
        u = rng.random()
        if u < 0.34:                              # single
            data.append(scene([rng.uniform(-65, 65)], False, None, rng))
        else:                                     # pair (half non-coherent reuse, half coherent multipath)
            a = rng.uniform(-65, 45); b = min(a + rng.uniform(15, 45), 68)
            coh = u > 0.67
            powers = None if coh else [rng.uniform(0.3, 1.0), 1.0]
            data.append(scene([a, b], coh, powers, rng))
    return data

print(f"SMOKE={SMOKE} epochs={EPOCHS} N_tr={N_TR}", flush=True)
TR = make_set(N_TR, 1); VA = make_set(N_VA, 2)
print(f"built {len(TR)} train / {len(VA)} val scenes", flush=True)

du = (ModelGenerator().set_model_type("DUMFOCUSS").set_system_model(SM)
      .set_model_params(dict(num_iterations=20, grid_size=901, grid_range_deg=[-70, 70],
                             p_init_decay=0.2, peak_lim_deg=70.0, angle_dependent_reg=True))
      .set_model()).model.to(device)
du.extend_iters = 0            # train the K=20 layers; extend tail is eval-only (budget parity)

def batches(data, bs, shuffle):
    idx = np.arange(len(data))
    if shuffle: np.random.shuffle(idx)
    for i in range(0, len(data), bs):
        sub = [data[j] for j in idx[i:i + bs]]
        M = sub[0][1]
        x = torch.stack([s[0] for s in sub]).to(device)
        ang = torch.tensor(np.stack([s[2] for s in sub]), dtype=torch.float64, device=device)
        srcnum = torch.full((x.shape[0],), M, dtype=torch.long, device=device)
        yield x, srcnum, ang

# group by source count so each batch is homogeneous
def grouped(data):
    g1 = [d for d in data if d[1] == 1]; g2 = [d for d in data if d[1] == 2]
    return g1, g2

opt = torch.optim.AdamW(du.parameters(), lr=1e-3, weight_decay=1e-4)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
g1tr, g2tr = grouped(TR); g1va, g2va = grouped(VA)

def run_epoch(g1, g2, train):
    du.train() if train else du.eval()
    tot, nb = 0.0, 0
    for g in (g1, g2):
        for x, M, ang in batches(g, BATCH, train):
            if train:
                opt.zero_grad()
                out = du.training_step((x, M, ang)); loss = out[0] if isinstance(out, tuple) else out
                try:
                    loss.backward(); torch.nn.utils.clip_grad_norm_(du.parameters(), 1.0); opt.step()
                except RuntimeError:
                    opt.zero_grad(); continue
            else:
                with torch.no_grad():
                    out = du.validation_step((x, M, ang)); loss = out[0] if isinstance(out, tuple) else out
            tot += float(loss); nb += 1
    return tot / max(nb, 1)

best = 1e9; best_sd = None
for ep in range(EPOCHS):
    tl = run_epoch(g1tr, g2tr, True); sch.step()
    vl = run_epoch(g1va, g2va, False)
    if vl < best: best = vl; best_sd = copy.deepcopy(du.state_dict())
    if ep % 5 == 0 or ep == EPOCHS - 1: print(f"ep{ep+1:3d} train={tl:.4f} val={vl:.4f} (best {best:.4f})", flush=True)

torch.save(best_sd, OUT)
print(f"SAVED {OUT}  best-val={best:.4f}", flush=True)
print("DONE", flush=True)
