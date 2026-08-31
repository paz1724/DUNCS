"""Correct DOA power-spectrum plots: ML (beamscan) + MFOCUSS + SPICE + SubspaceNet-MUSIC +
retrained DU-MFOCUSS spectra, plus retrained-DoAFormer angle markers. Each scene -> its own .mat
for the MATLAB Plot_DOA renderer. Scenes: single, reuse-15, multipath-15(coherent), 2 realizations each."""
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r"c:/GitHub/DUNCS"); os.chdir(r"c:/GitHub/DUNCS")
import numpy as np, torch
from scipy.io import savemat
from scipy.signal import find_peaks
from src.config.simulation_config import load_simulation_config
from src.system_model import SystemModel
from src.signal_creation import Samples
from src.models import ModelGenerator
from src.utils import device

torch.set_grad_enabled(False)
OUT = r"C:/Users/Daniel/AppData/Local/Temp/claude/c--GitHub-DUNCS/64e8dd82-9435-48e6-a804-bb899502d269/scratchpad/doa_png3"
os.makedirs(OUT, exist_ok=True)
GRID = np.round(np.arange(-90.0, 90.0 + 1e-6, 0.1), 4)
cfg = load_simulation_config("src/config/subspaceNet.yaml"); cfg.system_model.M = 2
SM = SystemModel(cfg.system_model); SAMP = Samples(cfg.system_model, cfg.system_model.antenna_pattern)
def steer(a): return np.asarray(SM.steering_vec(float(np.deg2rad(a))))
MG = np.arange(-70, 70 + 1e-6, 0.2); A = np.stack([steer(t) for t in MG], axis=1); An2 = np.sum(np.abs(A) ** 2, axis=0)

def ml2d(x, M):                              # 2-D deterministic conditional ML (the table's ML estimate)
    R = (x @ x.conj().T) / x.shape[1]
    if M == 1:
        return np.array([MG[int(np.argmax(np.real(np.sum(A.conj() * (R @ A), axis=0)) / An2))]])
    cg = np.arange(-70, 70 + 1e-6, 1.0); Bc = np.stack([steer(t) for t in cg], axis=1)
    BhRB = (Bc.conj().T @ R) @ Bc; BhB = Bc.conj().T @ Bc; Gc = len(cg)
    gd = np.real(np.diag(BhB)); md = np.real(np.diag(BhRB))
    num = md[:, None] * gd[None, :] + gd[:, None] * md[None, :] - BhB * BhRB.T - BhB.T * BhRB
    det = gd[:, None] * gd[None, :] - np.abs(BhB) ** 2
    iu = np.triu_indices(Gc, k=2); b = int(np.argmax((np.real(num) / (det + 1e-12))[iu]))
    return np.sort([cg[iu[0][b]], cg[iu[1][b]]])

def scene(angles_deg, coherent=False, powers=None, seed=0):
    np.random.seed(seed); M = len(angles_deg); SAMP.set_doa([float(a) for a in angles_deg], M)
    SAMP.params.signal_nature = "coherent" if coherent else "non-coherent"
    clean, noise = SAMP.samples_creation(noise_variance=1, signal_variance=1, source_number=M)
    clean = np.asarray(clean); noise = np.asarray(noise)
    if powers is not None:
        Am = np.stack([steer(a) for a in angles_deg], axis=1)
        clean = Am @ (np.linalg.lstsq(Am, clean, rcond=None)[0] * np.sqrt(np.asarray(powers))[:, None])
    snr = 10 ** (np.random.uniform(25, 30) / 10)
    return clean + noise * np.sqrt(1.0 / snr)

def to_grid(spec_lin, gdeg):
    o = np.argsort(gdeg); r = np.interp(GRID, np.asarray(gdeg)[o], np.asarray(spec_lin, float).ravel()[o], left=np.nan, right=np.nan)
    r = r / (np.nanmax(r) + 1e-30)
    with np.errstate(divide="ignore", invalid="ignore"): db = 10 * np.log10(np.clip(r, 1e-6, None))
    db[np.abs(GRID) > 70] = np.nan; return db

def peaks(psd, M):
    y = np.nan_to_num(psd, nan=-80.0).copy(); y[np.abs(GRID) > 68] = -80.0
    pk, pr = find_peaks(y, height=-40)
    if not len(pk): return np.array([GRID[int(np.nanargmax(y))]])
    return np.sort(GRID[pk[np.argsort(pr["peak_heights"])[::-1][:M]]])

def build(mt, pr, wf):
    m = (ModelGenerator().set_model_type(mt).set_system_model(SM).set_model_params(pr).set_model()).model.to(device).eval()
    if wf and os.path.exists("data/weights/" + wf): m.load_state_dict(torch.load("data/weights/" + wf, map_location=device), strict=False)
    return m
mfo = build("MFOCUSS", dict(num_iterations=100, grid_size=901, p=0.8, lam=0.05, grid_range_deg=[-180, 180]), None)
spi = build("SPICE", dict(num_iterations=100, grid_size=901), None)
mus = build("SubspaceNet", dict(tau=7, diff_method="music_1D"), "SubspaceNet_tau=7_diff_method=music_angle_N=5_M=[1,-2]_T=8_NarrowBand_SNR=30_Far_field_non-coherent_eta=0.0_sv_var=0.0")
du = build("DUMFOCUSS", dict(num_iterations=20, grid_size=901, grid_range_deg=[-70, 70], p_init_decay=0.2, peak_lim_deg=70.0, angle_dependent_reg=True), "DUMFOCUSS_clean_K20_grid901_frontcone_N=5_T=8.pt")
du.extend_iters = 80
dfm = build("DoAFormer", dict(d_model=96, nhead=4, num_encoder_layers=3, num_decoder_layers=2, dim_feedforward=192, input_mode="both"), "DoAFormer_clean_dmodel96_Q2_synth_N=5_T=8.pt")
mg_deg = np.rad2deg(mus.diff_method.angels.detach().cpu().numpy())

def specs(x, M):
    xt = torch.tensor(x, dtype=torch.complex128, device=device)[None]; R = (x @ x.conj().T) / x.shape[1]
    out = {}
    # ML beamscan power spectrum
    ml = np.real(np.sum(A.conj() * (R @ A), axis=0)) / An2; out["ML"] = (to_grid(ml, MG), ml2d(x, M))
    mfo._A_use = mfo.A_fine if M == 1 else mfo.A; mfo._grid_use = mfo.grid_fine if M == 1 else mfo.grid
    out["MFOCUSS"] = (to_grid(mfo._spectrum(xt).cpu().numpy().ravel(), np.rad2deg((mfo.grid_fine if M == 1 else mfo.grid).cpu().numpy())), None)
    spi._A_use = spi.A_fine; spi._grid_use = spi.grid_fine
    out["SPICE_IAA"] = (to_grid(spi._spectrum(xt).cpu().numpy().ravel(), np.rad2deg(spi.grid_fine.cpu().numpy())), None)
    mus(xt, M); out["SubspaceNet_MUSIC"] = (to_grid(mus.diff_method.music_spectrum.cpu().numpy().ravel(), mg_deg), None)
    du(xt, M); out["DU_MFOCUSS_retrained"] = (to_grid(du._last_spectrum.cpu().numpy().ravel(), np.rad2deg((du.grid_fine if M == 1 else du.grid).cpu().numpy())), None)
    out["DoAFormer"] = (np.full_like(GRID, np.nan), np.rad2deg(dfm(xt, M)[0].cpu().numpy()).ravel())   # angle-only
    return out

SCENES = [("single", [12.0], False, None, 1724), ("single", [-28.0], False, None, 2025),
          ("reuse15", [-20.0, -5.0], False, [0.7, 1.0], 1724), ("reuse15", [8.0, 25.0], False, [0.6, 1.0], 2025),
          ("multipath15", [-15.0, 3.0], True, None, 1724), ("multipath15", [18.0, 34.0], True, None, 2025)]
ri = {}
for case, gt, coh, pw, seed in SCENES:
    ri[case] = ri.get(case, 0) + 1; r = ri[case]; M = len(gt)
    x = scene(gt, coh, pw, seed); sp = specs(x, M); results = {}
    for k, (psd, ang) in sp.items():
        a = ang if ang is not None else peaks(psd, M)
        df = float(np.mean([min(abs(np.asarray(a) - g)) for g in gt]))
        results[k] = dict(psd_dB=psd.astype(float), estimated_DoA=np.asarray(a, float), df_error=df)
    savemat(os.path.join(OUT, f"results_{case}_r{r}.mat"), dict(angles=GRID.astype(float), anglesGT=np.asarray(gt, float), results=results))
    print(f"wrote {case}_r{r} GT={gt} methods={list(results)}", flush=True)
print("DONE", flush=True)
