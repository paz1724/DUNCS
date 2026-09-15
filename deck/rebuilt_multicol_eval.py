"""Phase 4: the four train->test columns (Synth->Synth, Synth->Real, Sim->Real, Sim->Sim) x
scenarios (Single, Reuse>=15, Reuse>=25) for the correct method set. Classical (ML/MFOCUSS/SPICE)
depend only on the TEST manifold; learned (MUSIC/DU/DoAFormer) use synth-trained for Synth-> and
DataSim-trained for Sim-> columns. Test manifolds: synth (clean), real (measured-azimuth samples),
datasim (heavy multipath). Writes rebuilt_multicol.json."""
import sys, os, json, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"c:/GitHub/DUNCS"); os.chdir(r"c:/GitHub/DUNCS")
import numpy as np, torch
from scipy.optimize import linear_sum_assignment
from src.config.simulation_config import load_simulation_config
from src.system_model import SystemModel
from src.signal_creation import Samples
from src.models import ModelGenerator
from src.utils import device
from deck.doa_scenes import DATASIM_REFL_COHERENCE

torch.set_grad_enabled(False)
cfg = load_simulation_config("src/config/subspaceNet.yaml"); cfg.system_model.M = 2
SM = SystemModel(cfg.system_model); SAMP = Samples(cfg.system_model, cfg.system_model.antenna_pattern)
THRESH = 10.0; N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
def steer(a): return np.asarray(SM.steering_vec(float(np.deg2rad(a))))
MG = np.arange(-70, 70 + 1e-6, 0.2); A = np.stack([steer(t) for t in MG], axis=1); An2 = np.sum(np.abs(A) ** 2, axis=0)
REC_AZ = np.arange(-69.0, 69.0 + 1e-6, 3.0)                      # measured-manifold azimuth samples (the "Real" proxy)

def make_scene(angles, coherent, powers, datasim, rng):
    M = len(angles); SAMP.set_doa([float(a) for a in angles], M)
    SAMP.params.signal_nature = "coherent" if coherent else "non-coherent"
    clean, noise = SAMP.samples_creation(noise_variance=1, signal_variance=1, source_number=M)
    clean = np.asarray(clean); noise = np.asarray(noise)
    Am = np.stack([steer(a) for a in angles], axis=1); base = np.linalg.lstsq(Am, clean, rcond=None)[0]
    if powers is not None:
        clean = Am @ (base * np.sqrt(np.asarray(powers))[:, None]); base = np.linalg.lstsq(Am, clean, rcond=None)[0]
    if datasim:
        # Echo waveform: decorrelated from its own direct path, at the SHARED coherence constant
        # (deck/doa_scenes.py::DATASIM_REFL_COHERENCE). This generator is a SECOND copy of the
        # multipath model -- it builds signals through the Samples pipeline at SNR U(25,30) rather
        # than doa_scenes' own generation, so it cannot simply call the shared make_scene, but it
        # MUST share the coherence. Reusing base[si] verbatim (=1) fused direct+echo into one
        # rank-1 wavefront that genuinely arrives off the ground truth, so every method returned
        # the same displaced answer -- the Sim->Sim column read 1.6-2.1 deg while every other
        # column read 0.1-0.3 deg. Same artifact doa_scenes fixed; this copy was missed.
        Tn = clean.shape[1]
        for si, a in enumerate(angles):
            for _ in range(int(rng.integers(1, 4))):
                ref = float(np.clip(a + rng.uniform(-40, 40), -70, 70)); g = rng.uniform(0.2, 0.6) * np.exp(1j * rng.uniform(0, 2 * np.pi))
                amp = np.sqrt(np.mean(np.abs(base[si]) ** 2))
                indep = amp * (rng.standard_normal((1, Tn)) + 1j * rng.standard_normal((1, Tn))) / np.sqrt(2)
                s_ref = (DATASIM_REFL_COHERENCE * base[si:si + 1]
                         + np.sqrt(max(0.0, 1.0 - DATASIM_REFL_COHERENCE ** 2)) * indep)
                clean = clean + (g * steer(ref)[:, None]) * s_ref
    snr = 10 ** (rng.uniform(25, 30) / 10)
    return clean + noise * np.sqrt(1.0 / snr)

def gt_for(scenario, testman, rng):                             # draw GT angles for a scenario on a test manifold
    pool = REC_AZ if testman == "real" else None                # real uses measured-azimuth samples
    def draw():
        return float(rng.choice(pool)) if pool is not None else float(rng.uniform(-65, 65))
    if scenario == "single": return [draw()]
    gap = 25.0 if "25" in scenario else 15.0
    for _ in range(200):
        a, b = sorted([draw(), draw()])
        if b - a >= gap and b - a <= gap + 30: return [a, b]
    return [-20.0, -20.0 + gap]

def ml_doa(x, M):
    R = (x @ x.conj().T) / x.shape[1]
    if M == 1: return np.array([MG[int(np.argmax(np.real(np.sum(A.conj() * (R @ A), axis=0)) / An2))]])
    cg = np.arange(-70, 70 + 1e-6, 1.0); Bc = np.stack([steer(t) for t in cg], axis=1)
    BhRB = (Bc.conj().T @ R) @ Bc; BhB = Bc.conj().T @ Bc; gd = np.real(np.diag(BhB)); md = np.real(np.diag(BhRB))
    num = md[:, None] * gd[None, :] + gd[:, None] * md[None, :] - BhB * BhRB.T - BhB.T * BhRB
    det = gd[:, None] * gd[None, :] - np.abs(BhB) ** 2; iu = np.triu_indices(len(cg), k=2)
    b = int(np.argmax((np.real(num) / (det + 1e-12))[iu])); return np.sort([cg[iu[0][b]], cg[iu[1][b]]])

def score(pred, gt):
    pred = np.atleast_1d(pred); gt = np.atleast_1d(gt); C = np.abs(gt[:, None] - pred[None, :])
    ri, ci = linear_sum_assignment(C); e = C[ri, ci]; det = e <= THRESH
    return e[det].tolist(), int((~det).sum()), len(gt)

def build(mt, pr, wf):
    m = (ModelGenerator().set_model_type(mt).set_system_model(SM).set_model_params(pr).set_model()).model.to(device).eval()
    if wf and os.path.exists("data/weights/" + wf): m.load_state_dict(torch.load("data/weights/" + wf, map_location=device), strict=False)
    return m
mfo = build("MFOCUSS", dict(num_iterations=100, grid_size=901, grid_range_deg=[-180, 180]), None)
spi = build("SPICE", dict(num_iterations=100, grid_size=901), None)
DUP = dict(num_iterations=20, grid_size=901, grid_range_deg=[-70, 70], p_init_decay=0.2, peak_lim_deg=70.0, angle_dependent_reg=True)
MUP = dict(tau=7, diff_method="music_1D"); DFP = dict(d_model=96, nhead=4, num_encoder_layers=3, num_decoder_layers=2, dim_feedforward=192, input_mode="both")
mus_s = build("SubspaceNet", MUP, "SubspaceNet_tau=7_diff_method=music_angle_N=5_M=[1,-2]_T=8_NarrowBand_SNR=30_Far_field_non-coherent_eta=0.0_sv_var=0.0")
mus_d = build("SubspaceNet", MUP, "SubspaceNet_clean_music_datasim_N=5_T=8.pt")
du_s = build("DUMFOCUSS", DUP, "DUMFOCUSS_clean_K20_grid901_frontcone_N=5_T=8.pt"); du_s.extend_iters = 80
du_d = build("DUMFOCUSS", DUP, "DUMFOCUSS_clean_K20_grid901_frontcone_datasim_N=5_T=8.pt"); du_d.extend_iters = 80
dfm_s = build("DoAFormer", DFP, "DoAFormer_clean_dmodel96_Q2_synth_N=5_T=8.pt")
dfm_d = build("DoAFormer", DFP, "DoAFormer_clean_dmodel96_Q2_datasim_N=5_T=8.pt")
def md_(m, x, M): return np.rad2deg(m(torch.tensor(x, dtype=torch.complex128, device=device)[None], M)[0].cpu().numpy()).ravel()
def resid(x, ang):
    At = np.stack([steer(a) for a in np.atleast_1d(ang)], axis=1); S = np.linalg.lstsq(At, x, rcond=None)[0]
    return float(np.linalg.norm(x - At @ S) / (np.linalg.norm(x) + 1e-12))

METHODS = ["ML", "MFOCUSS", "SPICE (IAA)", "SubspaceNet-MUSIC", "DoAFormer", "DU-MFOCUSS", "DU-MFOCUSS-guarded"]
def estimate(x, M, train):                                      # train in {"synth","datasim"}
    mus, du, dfm = (mus_s, du_s, dfm_s) if train == "synth" else (mus_d, du_d, dfm_d)
    o = {"ML": ml_doa(x, M), "MFOCUSS": md_(mfo, x, M), "SPICE (IAA)": md_(spi, x, M),
         "SubspaceNet-MUSIC": md_(mus, x, M), "DoAFormer": md_(dfm, x, M), "DU-MFOCUSS": md_(du, x, M)}
    o["DU-MFOCUSS-guarded"] = o["DU-MFOCUSS"] if resid(x, o["DU-MFOCUSS"]) <= resid(x, o["MFOCUSS"]) else o["MFOCUSS"]
    return o

COLUMNS = [("Synth\u2192Synth", "synth", "synth"), ("Synth\u2192Real", "synth", "real"),
           ("Sim\u2192Real", "datasim", "real"), ("Sim\u2192Sim", "datasim", "datasim")]
SCEN = ["single", "reuse15", "reuse25"]
out = {"scenarios": SCEN, "columns": [c[0] for c in COLUMNS], "cells": {}}
for scen in SCEN:
    for cname, train, testman in COLUMNS:
        M = 1 if scen == "single" else 2
        acc = {k: ([], 0, 0) for k in METHODS}
        rng = np.random.default_rng(hash((scen, cname)) % 99999)
        for i in range(N):
            gt = gt_for(scen, testman, rng)
            x = make_scene(gt, coherent=(testman == "datasim"), powers=None, datasim=(testman == "datasim"), rng=rng)
            est = estimate(x, M, train)
            for k in METHODS:
                e, mdc, ng = score(est[k], gt); a, b, c = acc[k]; acc[k] = (a + e, b + mdc, c + ng)
        for k in METHODS:
            e, mdc, ng = acc[k]; rms = float(np.sqrt(np.mean(np.square(e)))) if e else 9.9
            out["cells"][f"{scen}|{cname}|{k}"] = [round(rms, 1), round(100 * mdc / max(ng, 1))]
        print(f"{scen:8s} {cname:14s} " + "  ".join(f"{k.split('-')[0][:4]}={out['cells'][f'{scen}|{cname}|{k}'][0]}/{out['cells'][f'{scen}|{cname}|{k}'][1]}" for k in METHODS), flush=True)
json.dump(out, open("data/simulations/results/rebuilt_multicol.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("SAVED rebuilt_multicol.json\nDONE", flush=True)
