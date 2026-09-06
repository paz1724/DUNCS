"""DOA power-spectrum sanity plots for the performance tables.

Spectra: ML (beamscan) + classical MUSIC + MFOCUSS + SPICE/IAA + SubspaceNet-MUSIC + retrained
DU-MFOCUSS, plus retrained-DoAFormer angle markers (gridless, so no spectrum). Every marker is the
method's OWN forward() readout, and every scene comes from deck/doa_scenes.py, so the figures show
the same estimators on the same scenarios that the tables score. Each scene -> its own .mat for the
MATLAB Plot_DOA renderer. Scenes: single, reuse-15, multipath-15 (rho=0.9), 2 realizations each."""
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r"c:/GitHub/DUNCS"); os.chdir(r"c:/GitHub/DUNCS")
import numpy as np, torch
from scipy.io import savemat
from scipy.signal import find_peaks
from src.config.simulation_config import load_simulation_config
from src.system_model import SystemModel
from src.models import ModelGenerator
from src.utils import device
from deck.doa_scenes import make_scene, RHO_PARTIAL

torch.set_grad_enabled(False)
OUT = r"C:/Users/Daniel/AppData/Local/Temp/claude/c--GitHub-DUNCS/64e8dd82-9435-48e6-a804-bb899502d269/scratchpad/doa_png4"
os.makedirs(OUT, exist_ok=True)
GRID = np.round(np.arange(-90.0, 90.0 + 1e-6, 0.1), 4)
cfg = load_simulation_config("src/config/subspaceNet.yaml"); cfg.system_model.M = 2
SM = SystemModel(cfg.system_model)
def steer(a): return np.asarray(SM.steering_vec(float(np.deg2rad(a))))
MG = np.arange(-70, 70 + 1e-6, 0.2); A = np.stack([steer(t) for t in MG], axis=1); An2 = np.sum(np.abs(A) ** 2, axis=0)

def ml_beamscan(x):
    """The project's ML (cArray DOA_BF): s(theta) = mean_t |a^H y_t|^2 / ||a||^2."""
    R = (x @ x.conj().T) / x.shape[1]
    return np.real(np.sum(A.conj() * (R @ A), axis=0)) / An2

def music_classic(x, M):
    """Classical MUSIC on the RAW sample covariance: 1/||E_n^H a||^2 over the noise subspace."""
    R = (x @ x.conj().T) / x.shape[1]
    R = R + 1e-10 * np.trace(R).real / R.shape[0] * np.eye(R.shape[0])
    _, V = np.linalg.eigh(R)                                   # ascending eigenvalues
    En = V[:, : R.shape[0] - M]
    return 1.0 / (np.sum(np.abs(En.conj().T @ A) ** 2, axis=0) + 1e-30)

def scene(angles_deg, coherent=False, powers=None, seed=0):
    """Scenes come from the CANONICAL generator, so the sanity plots show the same scenarios the
    performance tables score. In particular multipath is rho=0.9 PARTIAL coherence, not rho=1:
    at exactly rho=1 the source covariance is rank-1 (the two sources are the same signal), which
    is a degenerate special case no subspace or sparse method can resolve, and plotting it would
    show a failure of the scenario rather than of the algorithms."""
    x, _ = make_scene(steer, angles_deg, np.random.default_rng(seed), T=int(SM.params.T),
                      rho=RHO_PARTIAL if coherent else 0.0, powers=powers)
    return x

def to_grid(spec_lin, gdeg):
    o = np.argsort(gdeg); r = np.interp(GRID, np.asarray(gdeg)[o], np.asarray(spec_lin, float).ravel()[o], left=np.nan, right=np.nan)
    r = r / (np.nanmax(r) + 1e-30)
    with np.errstate(divide="ignore", invalid="ignore"): db = 10 * np.log10(np.clip(r, 1e-6, None))
    db[np.abs(GRID) > 70] = np.nan; return db

def peak_idx(spec, M, min_sep_deg=2.0):
    """Top-M LOCAL MAXIMA of a linear spectrum on MG, kept min_sep_deg apart (the DOA_BF readout)."""
    pk, _ = find_peaks(spec)
    if len(pk) >= M:
        keep = []
        for i in pk[np.argsort(spec[pk])[::-1]]:
            if all(abs(MG[i] - MG[j]) >= min_sep_deg for j in keep):
                keep.append(i)
            if len(keep) == M:
                return np.array(keep)
    return np.argsort(spec)[::-1][:M]

def build(mt, pr, wf):
    m = (ModelGenerator().set_model_type(mt).set_system_model(SM).set_model_params(pr).set_model()).model.to(device).eval()
    if wf and os.path.exists("data/weights/" + wf): m.load_state_dict(torch.load("data/weights/" + wf, map_location=device), strict=False)
    return m
mfo = build("MFOCUSS", dict(num_iterations=100, grid_size=901, p=0.8, lam=0.05, grid_range_deg=[-70, 70]), None)
spi = build("SPICE", dict(num_iterations=100, grid_size=901), None)
mus = build("SubspaceNet", dict(tau=7, diff_method="music_1D"), "music_150MHz_synth.pt")
du = build("DUMFOCUSS", dict(num_iterations=20, grid_size=901, grid_range_deg=[-70, 70], p_init_decay=0.2, peak_lim_deg=70.0, angle_dependent_reg=True), "du_150MHz_synth.pt")
du.extend_iters = 80
dfm = build("DoAFormer", dict(d_model=96, nhead=4, num_encoder_layers=3, num_decoder_layers=2, dim_feedforward=192, input_mode="both"), "doaformer_150MHz_synth.pt")
mg_deg = np.rad2deg(mus.diff_method.angels.detach().cpu().numpy())

def deg(t):
    return np.sort(np.rad2deg(np.asarray(t.detach().cpu().numpy(), float)).ravel())

def specs(x, M):
    """Each entry is (power spectrum on GRID, the method's OWN forward() angle readout).

    The markers are every model's real readout -- NOT peak-picked off the plotted curve -- so the
    DF errors in the legend are produced the same way the performance tables are.
    """
    xt = torch.tensor(x, dtype=torch.complex128, device=device)[None]
    out = {}
    ml = ml_beamscan(x)
    out["ML"] = (to_grid(ml, MG), np.sort(MG[peak_idx(ml, M)]))
    mu = music_classic(x, M)
    out["MUSIC_classical"] = (to_grid(mu, MG), np.sort(MG[peak_idx(mu, M)]))
    # MFOCUSS: forward() uses a SOURCE-ADAPTIVE lambda -- the Hof high-lambda schedule for a single
    # source (smooth -> precise) and lam_multi for >=2 (sharp -> resolves the pair). Calling
    # _spectrum() without it silently used lam=0.99 on pairs, which over-regularizes so hard that
    # the recovery never sparsifies: the plotted curve stayed a 69 deg-wide blob (100% of the grid
    # above -20 dB) and read out 8.50 deg of DF error, instead of the 1.0 deg-wide spikes and
    # 1.00 deg error the algorithm actually produces. A sparse method must plot as spikes.
    mfo._A_use, mfo._grid_use = mfo.A_fine, mfo.grid_fine
    mfo_spec = mfo._spectrum(xt, None if M == 1 else mfo.lam_multi)
    out["MFOCUSS"] = (to_grid(mfo_spec.cpu().numpy().ravel(), np.rad2deg(mfo.grid_fine.cpu().numpy())),
                      deg(mfo(xt, M)[0]))
    spi._A_use, spi._grid_use = spi.A_fine, spi.grid_fine
    out["SPICE_IAA"] = (to_grid(spi._spectrum(xt).cpu().numpy().ravel(), np.rad2deg(spi.grid_fine.cpu().numpy())),
                        deg(spi(xt, M)[0]))
    mus_doa = deg(mus(xt, M)[0])
    out["SubspaceNet_MUSIC"] = (to_grid(mus.diff_method.music_spectrum.cpu().numpy().ravel(), mg_deg), mus_doa)
    du_doa = deg(du(xt, M)[0])
    out["DU_MFOCUSS_retrained"] = (to_grid(du._last_spectrum.cpu().numpy().ravel(),
                                           np.rad2deg((du.grid_fine if M == 1 else du.grid).cpu().numpy())), du_doa)
    out["DoAFormer"] = (np.full_like(GRID, np.nan), deg(dfm(xt, M)[0]))   # gridless: angle-only
    return out

SCENES = [("single", [12.0], False, None, 1724), ("single", [-28.0], False, None, 2025),
          ("reuse15", [-20.0, -5.0], False, [0.7, 1.0], 1724), ("reuse15", [8.0, 25.0], False, [0.6, 1.0], 2025),
          ("multipath15", [-15.0, 3.0], True, None, 1724), ("multipath15", [18.0, 34.0], True, None, 2025)]
ri = {}
for case, gt, coh, pw, seed in SCENES:
    ri[case] = ri.get(case, 0) + 1; r = ri[case]; M = len(gt)
    x = scene(gt, coh, pw, seed); sp = specs(x, M); results = {}
    for k, (psd, ang) in sp.items():
        a = np.atleast_1d(ang)
        df = float(np.mean([min(abs(np.asarray(a) - g)) for g in gt]))
        results[k] = dict(psd_dB=psd.astype(float), estimated_DoA=np.asarray(a, float), df_error=df)
    savemat(os.path.join(OUT, f"results_{case}_r{r}.mat"), dict(angles=GRID.astype(float), anglesGT=np.asarray(gt, float), results=results))
    print(f"wrote {case}_r{r} GT={gt} methods={list(results)}", flush=True)
print("DONE", flush=True)
