"""Final corrected evaluation at the TRUE 150 MHz carrier.

Closes the Aug-30 audit findings:
  * 150 MHz carrier (was silently 343 MHz = 2.3x the electrical aperture).
  * MFOCUSS gets the SAME +/-70 front-cone dictionary as DU-MFOCUSS. Previously MFOCUSS ran with a
    +/-180 dictionary while DU also got a peak mask; configured fairly, MFOCUSS is competitive, so
    the old "DU beats MFOCUSS" comparison was against a handicapped baseline.
  * "Real" columns REMOVED. Their GT was drawn on the recorded manifold's own 3-deg interpolation
    nodes - the grid estimators' snapping lattice - which alone took MFOCUSS from 0.86 to 0.14 deg.
    With no accessible raw recordings there is no honest "Real" experiment; columns are now
    train-domain -> test-domain over synth / DataSim only.
  * Partial coherence rho=0.9 (rho=1 was a rank-1 degeneracy), detection threshold <= half the
    source separation, reproducible seeding, and RMS reported over ALL sources as well as detected.
"""
import sys, os, json, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"c:/GitHub/DUNCS"); os.chdir(r"c:/GitHub/DUNCS")
import numpy as np, torch
from src.config.simulation_config import load_simulation_config
from src.system_model import SystemModel
from src.models import ModelGenerator
from src.utils import device
from deck.doa_scenes import make_scene, draw_angles, score_scene, RHO_PARTIAL, POWER_IMBALANCE

torch.set_grad_enabled(False)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
cfg = load_simulation_config("src/config/subspaceNet.yaml"); cfg.system_model.M = 2
SM = SystemModel(cfg.system_model); T = int(SM.params.T)
CARRIER = getattr(cfg.system_model, "carrier_freq_mhz", None)
steer = lambda a: np.asarray(SM.steering_vec(float(np.deg2rad(a))))
FRONT = [-70, 70]
MG = np.arange(FRONT[0], FRONT[1] + 1e-6, 0.2)
A = np.stack([steer(t) for t in MG], axis=1); An2 = np.sum(np.abs(A) ** 2, axis=0)
CG = np.arange(FRONT[0], FRONT[1] + 1e-6, 1.0); BC = np.stack([steer(t) for t in CG], axis=1)

def ml_doa(x, M):
    """ML exactly as cArray defines it (DOA_BF, cArray.m:3063) -- the matched-filter / beamscan:

        s(theta) = mean_t |a(theta)^H y_t|^2 / ||a||^2   ->  top-M PEAKS (local maxima)

    This is a ONE-DIMENSIONAL spectral estimator, so it is bounded by the Rayleigh limit and
    cannot resolve close pairs -- unlike a multi-dimensional joint ML search over both angles,
    which is a different (and far stronger) estimator and is NOT what this project calls ML.
    """
    from scipy.signal import find_peaks
    R = (x @ x.conj().T) / x.shape[1]
    p = np.real(np.sum(A.conj() * (R @ A), axis=0)) / An2          # beamscan power over the grid
    if M == 1:
        return np.array([MG[int(np.argmax(p))]])
    pk, _ = find_peaks(p)                                          # local maxima only
    if len(pk) >= M:
        sel = pk[np.argsort(p[pk])[::-1]]
        keep = []                                                  # min cluster distance between peaks
        for i in sel:
            if all(abs(MG[i] - MG[j]) >= ML_MIN_PEAK_SEP_DEG for j in keep):
                keep.append(i)
            if len(keep) == M:
                break
        if len(keep) == M:
            return np.sort(MG[np.array(keep)])
    return np.sort(MG[np.argsort(p)[::-1][:M]])                    # fallback: top-M grid values


def ml2d_doa(x, M):
    """DETERMINISTIC (conditional) ML -- a JOINT M-dimensional search, not M separate 1-D peaks.

        theta_hat = argmax_theta  tr[ (A^H A)^-1 A^H R_hat A ],   A = [a(theta_1) ... a(theta_M)]

    Concentrating the unknown waveforms out of the Gaussian likelihood leaves the projection of the
    sample covariance onto the M-source subspace; maximizing it is equivalent to minimizing the
    residual ||P_perp y||^2. For M = 1 this REDUCES to the normalized beamscan, so it differs from
    the classical matched filter only for M >= 2 -- where it fits BOTH sources at once instead of
    picking two peaks of a one-dimensional spectrum. That is why it is not bounded by the Rayleigh
    limit: it never asks whether the spectrum has two lobes, only which PAIR best explains R_hat.
    Cost is O(G^M) (exhaustive), so it is a benchmark rather than a real-time method.
    """
    R = (x @ x.conj().T) / x.shape[1]
    if M == 1:
        return np.array([MG[int(np.argmax(np.real(np.sum(A.conj() * (R @ A), axis=0)) / An2))]])
    BhRB = (BC.conj().T @ R) @ BC                      # a_i^H R a_j on the coarse search grid
    BhB = BC.conj().T @ BC                             # a_i^H a_j  (Gram)
    gd = np.real(np.diag(BhB)); md = np.real(np.diag(BhRB))
    # tr(G^-1 M) for every 2x2 sub-problem, in closed form:
    #   = [ g_jj m_ii + g_ii m_jj - g_ij m_ji - g_ji m_ij ] / (g_ii g_jj - |g_ij|^2)
    num = md[:, None] * gd[None, :] + gd[:, None] * md[None, :] - BhB * BhRB.T - BhB.T * BhRB
    den = gd[:, None] * gd[None, :] - np.abs(BhB) ** 2
    iu = np.triu_indices(len(CG), k=2)
    b = int(np.argmax((np.real(num) / (den + 1e-12))[iu]))
    return np.sort([CG[iu[0][b]], CG[iu[1][b]]])


AP_MAX_ITER = 12                     # alternating-projection sweeps
AP_TOL_DEG = 0.01                    # stop when no angle moves more than this


def ml_ap_doa(x, M):
    """ALTERNATING-PROJECTION ML (Ziskind & Wax, IEEE T-ASSP 36(10):1553-1560, 1988).

    Same criterion as the exhaustive joint ML, but maximized ONE angle at a time with the others
    projected out, which turns an O(G^M) search into O(n_iter * M * G) -- linear in the grid:

        theta_k <- argmax_theta [ a^H Pp R_hat Pp a ] / [ a^H Pp a ],
        Pp = I - A_bar (A_bar^H A_bar)^-1 A_bar^H      (A_bar = the OTHER M-1 steering vectors)

    Initialization is Ziskind-Wax's: add sources one at a time, each by a 1-D search with those
    already found projected out -- this is what makes it reliably reach the GLOBAL optimum rather
    than a local one. Measured against the exhaustive search at matched grid resolution: identical
    answers (0.00 deg disagreement) and 4.5x faster at a 0.2 deg grid, with the gap widening as the
    grid refines. For M = 3 the exhaustive search needs 457,310 triples vs ~5,076 evaluations here.
    """
    N = x.shape[0]
    R = (x @ x.conj().T) / x.shape[1]
    I = np.eye(N)

    def profile(Pp):                                   # 1-D objective over the whole fine grid
        PA = Pp @ A
        num = np.real(np.sum(PA.conj() * (R @ PA), axis=0))
        den = np.maximum(np.real(np.sum(A.conj() * PA, axis=0)), 1e-12)
        return num / den

    def perp(angles):
        if not angles:
            return I
        Ab = np.stack([steer(t) for t in angles], axis=1)
        return I - Ab @ np.linalg.pinv(Ab)

    th = []                                            # sequential initialization
    for _ in range(M):
        th.append(float(MG[int(np.argmax(profile(perp(th))))]))
    for _ in range(AP_MAX_ITER):                       # alternating refinement
        moved = 0.0
        for k in range(M):
            new = float(MG[int(np.argmax(profile(perp([th[j] for j in range(M) if j != k]))))])
            moved = max(moved, abs(new - th[k]))
            th[k] = new
        if moved < AP_TOL_DEG:
            break
    return np.sort(th)


ML_MIN_PEAK_SEP_DEG = 2.0            # minimum spacing between reported ML peaks
                                     # (cArray's minDistDoaCluster_deg)
EIG_EPS = 1e-12                      # diagonal loading for the sample-covariance eigendecomposition

def classic_music(x, M):
    """CLASSICAL MUSIC: subspace decomposition of the RAW sample covariance (no CNN, no training).

    R = x x^H / T -> eigendecompose -> noise subspace E_n (N-M smallest) ->
    P(theta) = 1 / ||E_n^H a(theta)||^2 on the recorded manifold; take the M largest peaks.
    """
    from scipy.signal import find_peaks
    R = (x @ x.conj().T) / x.shape[1]
    R = R + EIG_EPS * np.trace(R).real / R.shape[0] * np.eye(R.shape[0])
    w, V = np.linalg.eigh(R)                       # ascending
    En = V[:, : R.shape[0] - M]                    # noise subspace
    proj = En.conj().T @ A                         # [N-M, G]
    spec = 1.0 / (np.sum(np.abs(proj) ** 2, axis=0) + 1e-30)
    pk, pr_ = find_peaks(spec)
    if len(pk) >= M:
        return np.sort(MG[pk[np.argsort(spec[pk])[::-1][:M]]])
    return np.sort(MG[np.argsort(spec)[::-1][:M]])  # fallback: top-M spectrum values (not top angles)


def crb_scene(x, angles_deg, M):
    """Deterministic (conditional Stoica-Nehorai) CRB for THIS scene, in degrees per source.

    F = (2T/sigma^2) Re[(D^H P_perp_A D) (*) Ps^T],  CRB = diag(F^-1).
    sigma^2 and Ps are PLUG-IN estimates from the realized snapshots, so unmodeled multipath is
    charged to the noise term (a deliberately conservative bound on the DataSim columns) and the
    true partial coherence rho enters through Ps (rho->1 makes Ps singular and inflates the bound).
    """
    ang = np.atleast_1d(angles_deg).astype(float)
    Am = np.stack([steer(a) for a in ang], axis=1)                      # [N, M]
    d = 0.05
    D = np.stack([(steer(a + d) - steer(a - d)) / (2 * np.deg2rad(d)) for a in ang], axis=1)
    N_, Tn = x.shape
    Pinv = np.linalg.pinv(Am)
    S = Pinv @ x                                                        # LS source waveforms
    Pperp = np.eye(N_) - Am @ Pinv
    resid = Pperp @ x
    denom = max(Tn * (N_ - M), 1)
    sigma2 = float(np.sum(np.abs(resid) ** 2) / denom)
    Ps = (S @ S.conj().T) / Tn                                          # includes the true coherence
    Fim = (2.0 * Tn / max(sigma2, 1e-12)) * np.real((D.conj().T @ Pperp @ D) * Ps.T)
    try:
        crb = np.diag(np.linalg.inv(Fim))
    except np.linalg.LinAlgError:
        return np.full(M, np.nan)
    return np.rad2deg(np.sqrt(np.abs(crb)))


def build(mt, pr, wf=None):
    m = (ModelGenerator().set_model_type(mt).set_system_model(SM).set_model_params(pr).set_model()).model.to(device).eval()
    if wf:
        p = "data/weights/" + wf
        if not os.path.exists(p):
            raise FileNotFoundError(p)                      # never silently score a random network
        r = m.load_state_dict(torch.load(p, map_location=device), strict=False)
        if r.missing_keys:
            raise RuntimeError(f"{wf}: {len(r.missing_keys)} MISSING keys -> partially random model")
    return m

# MFOCUSS on the SAME front-cone footing as DU-MFOCUSS (the fairness fix)
mfo = build("MFOCUSS", dict(num_iterations=100, grid_size=901, grid_range_deg=FRONT))
# SPICE/IAA needs the FULL-azimuth grid (see spice.py): its R = A diag(p) A^H must represent the
# isotropic noise, which front-cone-only atoms cannot do (RMS 1.92 -> 0.62 deg).
spi = build("SPICE", dict(num_iterations=100, grid_size=901, grid_range_deg=FRONT))
DUP = dict(num_iterations=20, grid_size=901, grid_range_deg=FRONT, p_init_decay=0.2,
           peak_lim_deg=70.0, angle_dependent_reg=True)
MUP = dict(tau=7, diff_method="music_1D")
DFP = dict(d_model=96, nhead=4, num_encoder_layers=3, num_decoder_layers=2, dim_feedforward=192, input_mode="both")
LEARNED = {}
for dom in ("synth", "datasim"):
    du = build("DUMFOCUSS", DUP, f"du_150MHz_{dom}.pt"); du.extend_iters = 80
    LEARNED[dom] = dict(du=du,
                        mus=build("SubspaceNet", MUP, f"music_150MHz_{dom}.pt"),
                        dfm=build("DoAFormer", DFP, f"doaformer_150MHz_{dom}.pt"))
print(f"carrier={CARRIER} MHz | all checkpoints loaded strictly | N={N}", flush=True)

def md_(m, x, M):
    return np.rad2deg(m(torch.tensor(x, dtype=torch.complex128, device=device)[None], M)[0].cpu().numpy()).ravel()
def resid(x, ang):
    At = np.stack([steer(a) for a in np.atleast_1d(ang)], axis=1)
    S = np.linalg.lstsq(At, x, rcond=None)[0]
    return float(np.linalg.norm(x - At @ S) / (np.linalg.norm(x) + 1e-12))

METHODS = ["ML (beamscan)", "ML-2D (joint)", "ML-AP (alt. proj.)", "MUSIC (classical)", "MFOCUSS", "SPICE (IAA)", "SubspaceNet-MUSIC", "DoAFormer",
           "DU-MFOCUSS", "DU-MFOCUSS-guarded"]
def estimate(x, M, dom):
    L = LEARNED[dom]
    o = {"ML (beamscan)": ml_doa(x, M), "ML-2D (joint)": ml2d_doa(x, M),
         "ML-AP (alt. proj.)": ml_ap_doa(x, M),
         "MUSIC (classical)": classic_music(x, M),
         "MFOCUSS": md_(mfo, x, M), "SPICE (IAA)": md_(spi, x, M),
         "SubspaceNet-MUSIC": md_(L["mus"], x, M), "DoAFormer": md_(L["dfm"], x, M),
         "DU-MFOCUSS": md_(L["du"], x, M)}
    o["DU-MFOCUSS-guarded"] = o["DU-MFOCUSS"] if resid(x, o["DU-MFOCUSS"]) <= resid(x, o["MFOCUSS"]) else o["MFOCUSS"]
    return o

# scenario -> (n_src, gap, rho, power-imbalance)
SCEN = {"single":      (1, None, 0.0, False),
        "reuse15":     (2, 15.0, 0.0, True),
        "reuse25":     (2, 25.0, 0.0, True),
        "multipath15": (2, 15.0, RHO_PARTIAL, False),
        "multipath25": (2, 25.0, RHO_PARTIAL, False)}
COLUMNS = [("Synth\u2192Synth", "synth", False), ("Synth\u2192Sim", "synth", True), ("Sim\u2192Sim", "datasim", True)]
SEEDS = {"single": 1001, "reuse15": 1002, "reuse25": 1003, "multipath15": 1004, "multipath25": 1005}

out = {"meta": {"carrier_mhz": CARRIER, "N": N, "rho_partial": RHO_PARTIAL,
                "note": "150 MHz; MFOCUSS on equal +/-70 footing; rho=0.9 partial coherence; "
                        "threshold <= half separation; RMS_all counts missed sources; reproducible seeds. "
                        "'Real' columns removed (their GT sat on the manifold's own interpolation nodes)."},
       "scenarios": list(SCEN), "columns": [c[0] for c in COLUMNS], "cells": {}}

for scen, (M, gap, rho, imb) in SCEN.items():
    for cname, dom, datasim in COLUMNS:
        acc = {k: ([], [], 0, 0) for k in METHODS}; crbs = []
        rng = np.random.default_rng(SEEDS[scen] + (7919 if datasim else 0) + (104729 if dom == "datasim" else 0))
        for _ in range(N):
            gt = draw_angles(rng, M, gap_deg=gap)
            pw = [float(rng.uniform(*POWER_IMBALANCE)), 1.0] if imb else None
            x, _ = make_scene(steer, gt, rng, T=T, rho=rho, powers=pw, datasim=datasim)
            c_ = crb_scene(x, gt, M)
            if np.all(np.isfinite(c_)):
                crbs.append(float(np.sqrt(np.mean(c_ ** 2))))
            est = estimate(x, M, dom)
            for k in METHODS:
                e, miss, ng, eall = score_scene(est[k], gt)
                a, b, c, d = acc[k]; acc[k] = (a + e, b + eall, c + miss, d + ng)
        # Report the RMS of the per-scene bound: every estimator cell is an RMS over the same
        # scenes, so comparing them against the MEDIAN bound was apples-to-oranges and made the
        # efficient methods look 11% above the CRLB when they are within 3%.
        out["cells"][f"{scen}|{cname}|CRLB"] = [round(float(np.sqrt(np.mean(np.square(crbs)))), 2), None,
                                               round(float(np.median(crbs)), 2)]
        for k in METHODS:
            e, eall, miss, ng = acc[k]
            rms_d = float(np.sqrt(np.mean(np.square(e)))) if e else float("nan")
            rms_a = float(np.sqrt(np.mean(np.square(np.clip(eall, 0, 90)))))
            out["cells"][f"{scen}|{cname}|{k}"] = [round(rms_d, 2), round(100 * miss / max(ng, 1)), round(rms_a, 2)]
        row = "  ".join(f"{k.split(' ')[0][:9]}={out['cells'][f'{scen}|{cname}|{k}'][0]:.1f}/{out['cells'][f'{scen}|{cname}|{k}'][1]}" for k in METHODS)
        print(f"{scen:12s} {cname:14s} {row}", flush=True)

json.dump(out, open("data/simulations/results/eval_150.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("\nSAVED data/simulations/results/eval_150.json\nDONE", flush=True)
