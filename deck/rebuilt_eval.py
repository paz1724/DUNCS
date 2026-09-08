"""Phase 3: rebuilt evaluation across the SYNTHETIC scenarios for the reproducible methods +
the ML baseline + CRB + the RETRAINED clean DU-MFOCUSS + a classical-floor-guarded DU. Self-
consistent convention (recorded ULA3 @150, per-source SNR U(25,30) matching model training).
"""
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r"c:/GitHub/DUNCS"); os.chdir(r"c:/GitHub/DUNCS")
import numpy as np, torch
from scipy.optimize import linear_sum_assignment
from src.config.simulation_config import load_simulation_config
from src.system_model import SystemModel
from src.signal_creation import Samples
from src.models import ModelGenerator
from src.utils import device

torch.set_grad_enabled(False)
cfg = load_simulation_config("src/config/subspaceNet.yaml"); cfg.system_model.M = 2
SM = SystemModel(cfg.system_model); SAMP = Samples(cfg.system_model, cfg.system_model.antenna_pattern)
DOA = (-70.0, 70.0); THRESH = 10.0
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
CLEAN_W = "data/weights/DUMFOCUSS_clean_K20_grid901_frontcone_N=5_T=8.pt"

GRID = np.arange(DOA[0], DOA[1] + 1e-6, 0.2); GRID_RAD = np.deg2rad(GRID)
A = np.stack([np.asarray(SM.steering_vec(float(t))) for t in GRID_RAD], axis=1)
Anorm2 = np.sum(np.abs(A) ** 2, axis=0)
def steer(a): return np.asarray(SM.steering_vec(float(np.deg2rad(a))))

def scene(angles_deg, coherent=False, powers=None, seed=None):
    if seed is not None: np.random.seed(seed)
    M = len(angles_deg); SAMP.set_doa([float(a) for a in angles_deg], M)
    SAMP.params.signal_nature = "coherent" if coherent else "non-coherent"
    clean, noise = SAMP.samples_creation(noise_variance=1, signal_variance=1, source_number=M)
    clean = np.asarray(clean); noise = np.asarray(noise)
    if powers is not None:
        Am = np.stack([steer(a) for a in angles_deg], axis=1)
        sig = np.linalg.lstsq(Am, clean, rcond=None)[0] * np.sqrt(np.asarray(powers))[:, None]
        clean = Am @ sig
    snr = 10 ** (np.random.uniform(25, 30) / 10)
    return clean + noise * np.sqrt(1.0 / snr), snr

def ml_doa(x, M):
    R = (x @ x.conj().T) / x.shape[1]
    if M == 1:
        p = np.real(np.sum(A.conj() * (R @ A), axis=0)) / Anorm2
        return np.array([GRID[int(np.argmax(p))]])
    cg = np.arange(DOA[0], DOA[1] + 1e-6, 1.0); Bc = np.stack([steer(t) for t in cg], axis=1)
    BhRB = (Bc.conj().T @ R) @ Bc; BhB = Bc.conj().T @ Bc; Gc = len(cg)
    gd = np.real(np.diag(BhB)); md = np.real(np.diag(BhRB))       # vectorized tr((B^HB)^-1 B^HRB) over all pairs
    num = md[:, None] * gd[None, :] + gd[:, None] * md[None, :] - BhB * BhRB.T - BhB.T * BhRB
    det = gd[:, None] * gd[None, :] - np.abs(BhB) ** 2
    val = np.real(num) / (det + 1e-12)
    iu = np.triu_indices(Gc, k=2); flat = val[iu]; b = int(np.argmax(flat))
    return np.sort([cg[iu[0][b]], cg[iu[1][b]]])

def crb_deg(angles_deg, snr_lin):
    M = len(angles_deg); T = int(SM.params.T); Am = np.stack([steer(a) for a in angles_deg], axis=1)
    d = 0.05; D = np.stack([(steer(a + d) - steer(a - d)) / (2 * np.deg2rad(d)) for a in angles_deg], axis=1)
    Pp = np.eye(Am.shape[0]) - Am @ np.linalg.pinv(Am.conj().T @ Am) @ Am.conj().T
    F = (2 * T / (1.0 / snr_lin)) * np.real((D.conj().T @ Pp @ D) * np.eye(M).T)
    return np.rad2deg(np.sqrt(np.abs(np.diag(np.linalg.inv(F)))))

def residual(x, angles_deg):
    Ath = np.stack([steer(a) for a in np.atleast_1d(angles_deg)], axis=1)
    S = np.linalg.lstsq(Ath, x, rcond=None)[0]
    return float(np.linalg.norm(x - Ath @ S) / (np.linalg.norm(x) + 1e-12))

def score_scene(pred, gt):
    pred = np.atleast_1d(pred); gt = np.atleast_1d(gt)
    C = np.abs(gt[:, None] - pred[None, :]); ri, ci = linear_sum_assignment(C); e = C[ri, ci]
    det = e <= THRESH
    return e[det].tolist(), int((~det).sum()), len(gt)

def build(mt, pr, wf, extra=None):
    m = (ModelGenerator().set_model_type(mt).set_system_model(SM).set_model_params(pr).set_model()).model.to(device).eval()
    if wf and os.path.exists("data/weights/" + wf): m.load_state_dict(torch.load("data/weights/" + wf, map_location=device), strict=False)
    if extra: extra(m)
    return m
MUSIC_W = "SubspaceNet_tau=7_diff_method=music_angle_N=5_M=[1,-2]_T=8_NarrowBand_SNR=30_Far_field_non-coherent_eta=0.0_sv_var=0.0"
mus = build("SubspaceNet", dict(tau=7, diff_method="music_1D"), MUSIC_W)
mfo = build("MFOCUSS", dict(num_iterations=100, grid_size=901, grid_range_deg=[-180, 180]), None)
spi = build("SPICE", dict(num_iterations=100, grid_size=901), None)
du = build("DUMFOCUSS", dict(num_iterations=20, grid_size=901, grid_range_deg=[-70, 70], p_init_decay=0.2,
                            peak_lim_deg=70.0, angle_dependent_reg=True), None)
if os.path.exists(CLEAN_W): du.load_state_dict(torch.load(CLEAN_W, map_location=device), strict=False); print("loaded retrained clean DU")
du.extend_iters = 80
def mdoa(model, x, M):
    return np.rad2deg(model(torch.tensor(x, dtype=torch.complex128, device=device)[None], M)[0].cpu().numpy()).ravel()

METHOD_NAMES = ["ML", "MFOCUSS", "SPICE", "SubspaceNet-MUSIC", "DU-MFOCUSS", "DU-MFOCUSS-guarded"]

def eval_sample(x, M):                       # compute each model ONCE; guard reuses DU + MFOCUSS
    o = {"ML": ml_doa(x, M), "MFOCUSS": mdoa(mfo, x, M), "SPICE": mdoa(spi, x, M),
         "SubspaceNet-MUSIC": mdoa(mus, x, M), "DU-MFOCUSS": mdoa(du, x, M)}
    o["DU-MFOCUSS-guarded"] = o["DU-MFOCUSS"] if residual(x, o["DU-MFOCUSS"]) <= residual(x, o["MFOCUSS"]) else o["MFOCUSS"]
    return o

def gen_single(i):
    r = np.random.default_rng(i); gt = [float(r.uniform(-65, 65))]; x, s = scene(gt, seed=1000 + i); return gt, x, s
def make_pair(i, gap, coh):
    r = np.random.default_rng(10000 * (gap == 25) + 20000 * coh + i)
    a = r.uniform(-65, 68 - gap); b = min(a + r.uniform(gap, gap + 25), 68)
    powers = None if coh else [float(r.uniform(0.3, 1.0)), 1.0]
    gt = sorted([float(a), float(b)]); x, s = scene(gt, coherent=coh, powers=powers, seed=2000 + i); return gt, x, s

SCENARIOS = [("single", gen_single, 1),
             ("reuse_noncoh (>=15)", lambda i: make_pair(i, 15, False), 2),
             ("multipath  (>=15,coh)", lambda i: make_pair(i, 15, True), 2),
             ("reuse_noncoh25", lambda i: make_pair(i, 25, False), 2),
             ("multipath25 (coh)", lambda i: make_pair(i, 25, True), 2)]

print(f"N={N} per scenario  (current-table cells in parentheses)")
REF = {"single": {"MFOCUSS": "0.9/0", "SubspaceNet-MUSIC": "0.4/0", "DU-MFOCUSS": "0.9/0"},
       "reuse_noncoh (>=15)": {"MFOCUSS": "2.5/4", "SubspaceNet-MUSIC": "1.4/2", "DU-MFOCUSS": "2.2/3"},
       "multipath  (>=15,coh)": {"MFOCUSS": "3.3/9", "SubspaceNet-MUSIC": "2.6/8", "DU-MFOCUSS": "2.8/6"},
       "reuse_noncoh25": {"MFOCUSS": "2.3/3", "SubspaceNet-MUSIC": "1.1/1", "DU-MFOCUSS": "2.0/2"},
       "multipath25 (coh)": {"MFOCUSS": "2.9/5", "SubspaceNet-MUSIC": "2.1/4", "DU-MFOCUSS": "2.5/3"}}
for name, gen, M in SCENARIOS:
    acc = {k: ([], 0, 0) for k in METHOD_NAMES}; crbs = []
    for i in range(N):
        gt, x, snr = gen(i); crbs.append(np.sqrt(np.mean(crb_deg(gt, snr) ** 2)))
        try:
            preds = eval_sample(x, M)
        except Exception as ex:
            if i == 0: print(f"   eval err: {ex}")
            continue
        for k in METHOD_NAMES:
            e, md, ng = score_scene(preds[k], gt); a, b, c = acc[k]; acc[k] = (a + e, b + md, c + ng)
    print(f"\n[{name}]  CRB={np.mean(crbs):.2f}")
    for k, (e, md, ng) in acc.items():
        rms = np.sqrt(np.mean(np.square(e))) if e else float("nan")
        ref = REF.get(name, {}).get(k, ""); ref = f"  (was {ref})" if ref else ""
        print(f"   {k:22s} RMS={rms:.2f}  MD={100*md/max(ng,1):.0f}%{ref}")
print("\nDONE")
