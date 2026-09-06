"""Retrain every learned model at the TRUE 150 MHz carrier using the canonical scene generator.

All previous checkpoints were trained with the steering manifold silently sliced at mid-band
(343 MHz), a 2.3x larger electrical aperture, so they must be retrained. Scenes come from
deck/doa_scenes.py (partial coherence rho=0.9, calibrated -20..-10 dB multipath, explicit RNG).

Usage:  python deck/retrain_150.py <du|music|doaformer> [--datasim] [--smoke]
"""
import sys, os, copy, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"c:/GitHub/DUNCS"); os.chdir(r"c:/GitHub/DUNCS")
import numpy as np, torch
from src.config.simulation_config import load_simulation_config
from src.system_model import SystemModel
from src.models import ModelGenerator
from src.utils import device
from deck.doa_scenes import make_scene, draw_angles, RHO_PARTIAL, POWER_IMBALANCE

WHICH = (sys.argv[1] if len(sys.argv) > 1 else "du").lower()
DATASIM = "--datasim" in sys.argv
SMOKE = "--smoke" in sys.argv
# MUSIC soft-argmax half-window as a fraction of the grid. Training reads the spectrum out with a
# soft-argmax over +/-cell_size while EVAL takes hard peaks; at the 0.3 default that window is
# +/-42 deg, so for a 25-40 deg pair it spans BOTH sources and the training gradient is never
# forced to separate them. --cell runs the controlled A/B against the 0.3 checkpoint.
CELL = float(sys.argv[sys.argv.index("--cell") + 1]) if "--cell" in sys.argv else None
EPOCHS = 3 if SMOKE else {"du": 80, "music": 60, "doaformer": 100}[WHICH]
N_TR, N_VA = (400, 200) if SMOKE else (5000, 1000)
BATCH = 128
SINGLE_FRAC, COH_FRAC = 0.34, 0.33          # 34% single / 33% independent pair / 33% coherent pair
TAG = "datasim" if DATASIM else "synth"
OUT = (f"c:/GitHub/DUNCS/data/weights/{WHICH}_150MHz_{TAG}"
       + (f"_cell{CELL:g}" if CELL is not None else "") + ".pt")

cfg = load_simulation_config("src/config/subspaceNet.yaml"); cfg.system_model.M = 2
SM = SystemModel(cfg.system_model)
print(f"carrier = {getattr(cfg.system_model,'carrier_freq_mhz','?')} MHz | model={WHICH} datasim={DATASIM} epochs={EPOCHS}", flush=True)
steer = lambda a: np.asarray(SM.steering_vec(float(np.deg2rad(a))))

def build_set(n, seed):
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        u = rng.random()
        if u < SINGLE_FRAC:
            gt = draw_angles(rng, 1); rho, pw = 0.0, None
        else:
            gap = 15.0 if rng.random() < 0.5 else 25.0
            gt = draw_angles(rng, 2, gap_deg=gap)
            coh = u > SINGLE_FRAC + COH_FRAC
            rho = RHO_PARTIAL if coh else 0.0
            pw = None if coh else [float(rng.uniform(*POWER_IMBALANCE)), 1.0]
        x, _ = make_scene(steer, gt, rng, T=int(SM.params.T), rho=rho, powers=pw, datasim=DATASIM)
        out.append((torch.tensor(x, dtype=torch.complex128), len(gt), np.deg2rad(np.sort(gt))))
    return out

TR, VA = build_set(N_TR, 101), build_set(N_VA, 202)
print(f"built {len(TR)} train / {len(VA)} val", flush=True)

SPEC = {
    "du":        ("DUMFOCUSS", dict(num_iterations=20, grid_size=901, grid_range_deg=[-70, 70],
                                    p_init_decay=0.2, peak_lim_deg=70.0, angle_dependent_reg=True)),
    "music":     ("SubspaceNet", dict(tau=7, diff_method="music_1D")),
    "doaformer": ("DoAFormer", dict(d_model=96, nhead=4, num_encoder_layers=3, num_decoder_layers=2,
                                    dim_feedforward=192, input_mode="both")),
}[WHICH]
model = (ModelGenerator().set_model_type(SPEC[0]).set_system_model(SM)
         .set_model_params(SPEC[1]).set_model()).model.to(device)
if WHICH == "du":
    model.extend_iters = 0                    # train K=20 layers; extension tail is eval-only
if CELL is not None:
    dm = model.diff_method
    dm.cell_size_frac = CELL
    dm.cell_size = max(1, int(dm.angels.shape[0] * CELL))
    print(f"MUSIC soft-argmax window = +/-{np.rad2deg(dm.cell_size * float(dm.angels[1]-dm.angels[0])):.2f} deg "
          f"({dm.cell_size} cells)", flush=True)

def batches(data, bs, shuffle, rng):
    g1 = [d for d in data if d[1] == 1]; g2 = [d for d in data if d[1] == 2]
    for g in (g1, g2):
        idx = np.arange(len(g))
        if shuffle: rng.shuffle(idx)
        for i in range(0, len(g), bs):
            sub = [g[j] for j in idx[i:i + bs]]
            x = torch.stack([s[0] for s in sub]).to(device)
            ang = torch.tensor(np.stack([s[2] for s in sub]), dtype=torch.float64, device=device)
            yield x, torch.full((x.shape[0],), sub[0][1], dtype=torch.long, device=device), ang

opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
brng = np.random.default_rng(7)

def run(data, train):
    model.train() if train else model.eval(); tot = nb = 0
    for x, sn, ang in batches(data, BATCH, train, brng):
        if train:
            opt.zero_grad()
            o = model.training_step((x, sn, ang)); loss = o[0] if isinstance(o, tuple) else o
            try:
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            except RuntimeError:
                opt.zero_grad(); continue
        else:
            with torch.no_grad():
                o = model.validation_step((x, sn, ang)); loss = o[0] if isinstance(o, tuple) else o
        tot += float(loss); nb += 1
    return tot / max(nb, 1)

best, bsd = 1e9, None
for ep in range(EPOCHS):
    tl = run(TR, True); sch.step(); vl = run(VA, False)
    if vl < best: best, bsd = vl, copy.deepcopy(model.state_dict())
    if ep % 5 == 0 or ep == EPOCHS - 1:
        print(f"ep{ep+1:3d} train={tl:.4f} val={vl:.4f} (best {best:.4f})", flush=True)
torch.save(bsd, OUT)
print(f"SAVED {OUT} best-val={best:.4f}", flush=True)
print("DONE", flush=True)
