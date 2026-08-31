"""Bridge: run every table algorithm on the SAME recorded-ULA3 @150 Sim scenes, extract each
one's DOA spectrum (or angle estimates) onto a shared -90:0.1:90 grid, and export per-scene
.mat files for the MATLAB Plot_DOA overlay. Angle-only methods (DoAFormer) carry NaN spectra +
their estimate markers; the root/ESPRIT subspace methods get a MUSIC pseudo-spectrum from their
learned covariance so they appear as curves on a common footing."""
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
from src.methods_pack.music import MUSIC
from src.utils import device

WD = r"c:/GitHub/DUNCS/data/weights/"
OUT = r"C:/Users/Daniel/AppData/Local/Temp/claude/c--GitHub-DUNCS/64e8dd82-9435-48e6-a804-bb899502d269/scratchpad/doa_png2"
os.makedirs(OUT, exist_ok=True)
GRID = np.round(np.arange(-90.0, 90.0 + 1e-6, 0.1), 4)   # common degree grid (1801 pts)

cfg = load_simulation_config("src/config/subspaceNet.yaml")
try:
    cfg.system_model.M = 2            # max sources = 2 (DoAFormer Q=max(M)=2; SubspaceNet M-agnostic)
except Exception as e:
    print("M set warn:", e)
sm = SystemModel(cfg.system_model)
samples = Samples(cfg.system_model, cfg.system_model.antenna_pattern)
snr_lin = 10 ** (float(cfg.system_model.snr) / 10)
music_ref = MUSIC(system_model=sm, estimation_parameter="angle")   # for subspace pseudo-spectra
music_grid_deg = np.rad2deg(music_ref.angels.detach().cpu().numpy())


def make_scene(angles_deg, seed):
    np.random.seed(int(seed))
    M = len(angles_deg)
    samples.set_doa([float(a) for a in angles_deg], M)
    clean, noise = samples.samples_creation(noise_variance=1, signal_variance=1, source_number=M)
    x = np.asarray(clean) + np.asarray(noise) * np.sqrt(1.0 / snr_lin)
    return torch.tensor(x, dtype=torch.complex128, device=device)[None], M


def to_grid(spec_lin, grid_deg):
    spec = np.asarray(spec_lin, float).ravel()
    gd = np.asarray(grid_deg, float).ravel()
    o = np.argsort(gd)
    r = np.interp(GRID, gd[o], spec[o], left=np.nan, right=np.nan)
    r = r / (np.nanmax(r) + 1e-30)
    with np.errstate(divide="ignore", invalid="ignore"):
        db = 10 * np.log10(np.clip(r, 1e-6, None))
    db[np.abs(GRID) > 70.0] = np.nan          # only display the configured DOA range [-70,70]
    return db


def build(mtype, params, wfile):
    mg = (ModelGenerator().set_model_type(mtype).set_system_model(sm)
          .set_model_params(params).set_model())
    m = mg.model.to(device).eval()
    info = ""
    if wfile:
        p = WD + wfile
        if not os.path.exists(p):
            return m, f"NO WEIGHTS ({wfile})"
        r = m.load_state_dict(torch.load(p, map_location=device), strict=False)
        info = f"miss={len(r.missing_keys)} unexp={len(r.unexpected_keys)}"
    return m, info


def angles_deg(model, x, M):
    with torch.no_grad():
        out = model(x, M)
    return np.rad2deg(np.asarray(out[0].detach().cpu().numpy())).ravel()


MUSIC_W = "SubspaceNet_tau=7_diff_method=music_angle_N=5_M=[1,-2]_T=8_NarrowBand_SNR=30_Far_field_non-coherent_eta=0.0_sv_var=0.0"
ESPRIT_W = "SubspaceNet_tau=7_diff_method=esprit_N=5_M=[1,-2]_T=8_NarrowBand_SNR=30_Far_field_non-coherent_eta=0.0_sv_var=0.0"
ROOT_W = "SubspaceNet_tau=7_diff_method=RootMusic()_N=5_M=[1,-2]_T=8_NarrowBand_SNR=30_Far_field_coherent_eta=0.0_sv_var=0.0"
DAF_W = "DoAFormer_dmodel=96_Q=2_N=5_M=[1,-2]_T=8_NarrowBand_SNR=30_Far_field_non-coherent_eta=0.0_sv_var=0.0"
DU_W = "DUMFOCUSS_K=20_grid=721_N=5_M=[1,-2]_T=8_NarrowBand_SNR=30_Far_field_coherent_eta=0.0_sv_var=0.0"

# (display name, sanitized field, builder, extractor kind)
SPECS = [
    ("MFOCUSS", "MFOCUSS", ("MFOCUSS", dict(num_iterations=100, grid_size=901, p=0.8, lam=0.05, grid_range_deg=[-180, 180]), None), "focuss"),
    ("SPICE (IAA)", "SPICE_IAA", ("SPICE", dict(num_iterations=100, grid_size=901), None), "focuss"),
    ("SubspaceNet-MUSIC", "SubspaceNet_MUSIC", ("SubspaceNet", dict(tau=7, diff_method="music_1D"), MUSIC_W), "ss_music"),
    ("SubspaceNet-MVDR", "SubspaceNet_MVDR", ("SubspaceNet", dict(tau=7, diff_method="mvdr"), MUSIC_W), "ss_music"),
    ("SubspaceNet-RootMUSIC", "SubspaceNet_RootMUSIC", ("SubspaceNet", dict(tau=7, diff_method="root_music"), ROOT_W), "pseudo"),
    ("SubspaceNet-ESPRIT", "SubspaceNet_ESPRIT", ("SubspaceNet", dict(tau=7, diff_method="esprit"), ESPRIT_W), "pseudo"),
    # DoAFormer (gridless set-prediction — no power spectrum; checkpoint architecture doesn't match a
    # fresh build) and DU-MFOCUSS (its deployed row needs compare_v3's eval-time readout — standalone
    # its full-azimuth unrolled dictionary shows edge artifacts) are omitted; their rows are in the tables.
]

# build all models once
MODELS = {}
for disp, fld, (mtype, params, wfile), kind in SPECS:
    try:
        m, info = build(mtype, params, wfile)
        MODELS[fld] = (disp, m, kind)
        print(f"BUILT {disp:24s} {info}")
    except Exception as e:
        print(f"FAIL  {disp:24s} {type(e).__name__}: {e}")

SCENES = {
    "single": [([12.0], 1724), ([-23.0], 2025), ([34.0], 4242)],
    "reuse15": [([-20.0, -5.0], 1724), ([5.0, 20.0], 2025), ([28.0, 43.0], 4242)],
}


def peaks_from_psd(psd_dB, M):
    """Top-M spectrum peaks (deg) on the common GRID within the configured DOA range [-70,70]
    — the DOA readout each spectrum implies (config-agnostic, avoids ±180 back-lobe edge peaks)."""
    inrange = np.abs(GRID) <= 68.0        # 2 deg margin so the ±70 boundary isn't a spurious peak
    y = np.nan_to_num(psd_dB, nan=-80.0).copy()
    y[~inrange] = -80.0
    pk, props = find_peaks(y, height=-40)
    if len(pk) == 0:
        idx = np.where(inrange)[0]
        return np.array([GRID[idx[int(np.nanargmax(np.nan_to_num(psd_dB[idx], nan=-80.0)))]]])
    top = pk[np.argsort(props["peak_heights"])[::-1][:M]]
    return np.sort(GRID[top])


def extract(kind, disp, model, x, M):
    ang = angles_deg(model, x, M)          # forward (also populates the SubspaceNet MUSIC spectrum)
    psd = np.full_like(GRID, np.nan)
    if kind == "focuss":
        single = (M == 1)
        model._A_use = model.A_fine if single else model.A
        model._grid_use = model.grid_fine if single else model.grid
        spec = model._spectrum(x).detach().cpu().numpy().ravel()
        gdeg = np.rad2deg((model.grid_fine if single else model.grid).detach().cpu().numpy())
        psd = to_grid(spec, gdeg)
    elif kind == "du":
        spec = model._last_spectrum.detach().cpu().numpy().ravel()
        single = (M == 1)
        gdeg = np.rad2deg((model.grid_fine if single else model.grid).detach().cpu().numpy())
        psd = to_grid(spec, gdeg)
    elif kind == "ss_music":
        spec = model.diff_method.music_spectrum.detach().cpu().numpy().ravel()
        psd = to_grid(spec, music_grid_deg)
    elif kind == "pseudo":
        Rz = model.get_learned_covariance(x)
        if getattr(model, "calibration", None) is not None:
            C = model.calibration
            Rz = C @ Rz @ C.conj().transpose(-1, -2)
        ns = music_ref.get_noise_subspace(Rz, M)
        spec = music_ref.get_music_spectrum_from_noise_subspace(ns).detach().cpu().numpy().ravel()
        psd = to_grid(spec, music_grid_deg)
    # DoAFormer (kind == "angles") has NO spectrum → keep its forward angles; everyone else reads
    # the estimate off the plotted spectrum peak (config-agnostic, self-consistent with the curve).
    if kind != "angles":
        ang = peaks_from_psd(psd, M)
    return psd, ang


for case, scenes in SCENES.items():
    for ri, (gt, seed) in enumerate(scenes, 1):
        x, M = make_scene(gt, seed)
        results = {}
        for fld, (disp, model, kind) in MODELS.items():
            try:
                psd, ang = extract(kind, disp, model, x, M)
                dferr = float(np.mean([min(abs(np.asarray(ang) - g)) for g in gt])) if len(ang) else 9.9
                results[fld] = dict(psd_dB=psd.astype(float), estimated_DoA=np.asarray(ang, float),
                                    df_error=dferr)
            except Exception as e:
                print(f"  [{case} r{ri}] {disp} extract FAIL: {type(e).__name__}: {e}")
        mat = os.path.join(OUT, f"results_{case}_r{ri}.mat")
        savemat(mat, dict(angles=GRID.astype(float), anglesGT=np.asarray(gt, float),
                          results=results))
        dfs = " ".join(f"{k}={results[k]['df_error']:.1f}" for k in results)
        print(f"[{case} r{ri}] GT={gt} | {dfs}")
print("DONE")
