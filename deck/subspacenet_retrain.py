"""Train SubspaceNet (MUSIC readout) on the DataSim manifold (heavy multipath) for the Sim-> columns.
Synth-trained MUSIC already exists (deployed music_angle weights); this adds the DataSim-trained one."""
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
EPOCHS = 3 if SMOKE else 60
N_TR = 400 if SMOKE else 5000
N_VA = 200 if SMOKE else 1000
BATCH = 128
OUT = "c:/GitHub/DUNCS/data/weights/SubspaceNet_clean_music_datasim_N=5_T=8.pt"

cfg = load_simulation_config("src/config/subspaceNet.yaml"); cfg.system_model.M = 2
SM = SystemModel(cfg.system_model); SAMP = Samples(cfg.system_model, cfg.system_model.antenna_pattern)
def steer(a): return np.asarray(SM.steering_vec(float(np.deg2rad(a))))

def scene(angles_deg, coherent, powers, rng):
    M = len(angles_deg); SAMP.set_doa([float(a) for a in angles_deg], M)
    SAMP.params.signal_nature = "coherent" if coherent else "non-coherent"
    clean, noise = SAMP.samples_creation(noise_variance=1, signal_variance=1, source_number=M)
    clean = np.asarray(clean); noise = np.asarray(noise)
    Amat = np.stack([steer(a) for a in angles_deg], axis=1); base = np.linalg.lstsq(Amat, clean, rcond=None)[0]
    if powers is not None:
        clean = Amat @ (base * np.sqrt(np.asarray(powers))[:, None]); base = np.linalg.lstsq(Amat, clean, rcond=None)[0]
    for si, a in enumerate(angles_deg):                # DataSim: 1-3 weak coherent multipath reflections
        for _ in range(int(rng.integers(1, 4))):
            ref = float(np.clip(a + rng.uniform(-40, 40), -70, 70))
            g = rng.uniform(0.2, 0.6) * np.exp(1j * rng.uniform(0, 2 * np.pi))
            clean = clean + (g * steer(ref)[:, None]) * base[si:si + 1]
    snr = 10 ** (rng.uniform(25, 30) / 10)
    return torch.tensor(clean + noise * np.sqrt(1.0 / snr), dtype=torch.complex128), M, np.deg2rad(np.sort(angles_deg))

def make_set(n, seed):
    rng = np.random.default_rng(seed); d = []
    for _ in range(n):
        u = rng.random()
        if u < 0.34: d.append(scene([rng.uniform(-65, 65)], False, None, rng))
        else:
            a = rng.uniform(-65, 45); b = min(a + rng.uniform(15, 45), 68); coh = u > 0.67
            d.append(scene([a, b], coh, None if coh else [rng.uniform(0.3, 1.0), 1.0], rng))
    return d

print(f"SMOKE={SMOKE} epochs={EPOCHS}", flush=True)
TR = make_set(N_TR, 1); VA = make_set(N_VA, 2); print(f"built {len(TR)}/{len(VA)}", flush=True)
m = (ModelGenerator().set_model_type("SubspaceNet").set_system_model(SM)
     .set_model_params(dict(tau=7, diff_method="music_1D")).set_model()).model.to(device)

def grouped(d): return [x for x in d if x[1] == 1], [x for x in d if x[1] == 2]
def batches(g, bs, sh):
    idx = np.arange(len(g))
    if sh: np.random.shuffle(idx)
    for i in range(0, len(g), bs):
        sub = [g[j] for j in idx[i:i + bs]]; M = sub[0][1]
        x = torch.stack([s[0] for s in sub]).to(device)
        ang = torch.tensor(np.stack([s[2] for s in sub]), dtype=torch.float64, device=device)
        yield x, torch.full((x.shape[0],), M, dtype=torch.long, device=device), ang

opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS); g1t, g2t = grouped(TR); g1v, g2v = grouped(VA)
def epoch(g1, g2, tr):
    m.train() if tr else m.eval(); tot = nb = 0
    for g in (g1, g2):
        for x, sn, ang in batches(g, BATCH, tr):
            if tr:
                opt.zero_grad(); o = m.training_step((x, sn, ang)); loss = o[0] if isinstance(o, tuple) else o
                try: loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
                except RuntimeError: opt.zero_grad(); continue
            else:
                with torch.no_grad(): o = m.validation_step((x, sn, ang)); loss = o[0] if isinstance(o, tuple) else o
            tot += float(loss); nb += 1
    return tot / max(nb, 1)
best = 1e9; bsd = None
for ep in range(EPOCHS):
    tl = epoch(g1t, g2t, True); sch.step(); vl = epoch(g1v, g2v, False)
    if vl < best: best = vl; bsd = copy.deepcopy(m.state_dict())
    if ep % 5 == 0 or ep == EPOCHS - 1: print(f"ep{ep+1:3d} train={tl:.4f} val={vl:.4f} (best {best:.4f})", flush=True)
torch.save(bsd, OUT); print(f"SAVED {OUT} best-val={best:.4f}", flush=True); print("DONE", flush=True)
