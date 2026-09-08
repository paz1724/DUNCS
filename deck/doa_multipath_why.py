"""Why MFOCUSS / SPICE-IAA / SubspaceNet-MUSIC lose on multipath -- 3 worked examples.

Each figure overlays, on the SAME coherent scene, the method that works next to the three that
do not, so the failure mode is visible rather than inferred:

  MUSIC (raw cov)        classical MUSIC on R_hat = x x^H / T -- the control. Works (4% MD).
  SubspaceNet-MUSIC      the SAME MUSIC algorithm on the CNN's surrogate covariance (65% MD).
                         Holding the readout fixed and swapping only the covariance reproduces
                         the whole gap, so the CNN -- not the peak picker -- is what fails.
  MFOCUSS / SPICE-IAA    sparse / covariance-fitting recovery; both merge the coherent pair.

Scenes: multipath >=25 deg (rho = 0.9) from deck/doa_scenes.py, the generator the tables score.
Writes one .mat per scene for the MATLAB Plot_DOA renderer.
"""
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"c:/GitHub/DUNCS"); os.chdir(r"c:/GitHub/DUNCS")
import numpy as np, torch
from scipy.io import savemat
from scipy.signal import find_peaks
from src.config.simulation_config import load_simulation_config
from src.system_model import SystemModel
from src.models import ModelGenerator
from src.utils import device
from deck.doa_scenes import make_scene, draw_angles, RHO_PARTIAL

torch.set_grad_enabled(False)
OUT = r"C:/Users/Daniel/AppData/Local/Temp/claude/c--GitHub-DUNCS/64e8dd82-9435-48e6-a804-bb899502d269/scratchpad/doa_mp"
os.makedirs(OUT, exist_ok=True)
GRID = np.round(np.arange(-90.0, 90.0 + 1e-6, 0.1), 4)
MIN_SEP_DEG = 4.0                      # matches MUSIC.peak_min_sep_deg

cfg = load_simulation_config("src/config/subspaceNet.yaml"); cfg.system_model.M = 2
SM = SystemModel(cfg.system_model)
steer = lambda a: np.asarray(SM.steering_vec(float(np.deg2rad(a))))
MG = np.arange(-70, 70 + 1e-6, 0.2)
A = np.stack([steer(t) for t in MG], axis=1)


def build(mt, pr, wf=None):
    m = (ModelGenerator().set_model_type(mt).set_system_model(SM).set_model_params(pr).set_model()).model.to(device).eval()
    if wf:
        m.load_state_dict(torch.load("data/weights/" + wf, map_location=device), strict=False)
    return m


mus = build("SubspaceNet", dict(tau=7, diff_method="music_1D"), "music_150MHz_synth.pt")
mfo = build("MFOCUSS", dict(num_iterations=100, grid_size=901, grid_range_deg=[-70, 70]))
spi = build("SPICE", dict(num_iterations=100, grid_size=901))
cap = {}
_orig = mus.diff_method.forward
def _cap(cov, *a, **k):
    cap["Rz"] = cov.detach().cpu().numpy()
    return _orig(cov, *a, **k)
mus.diff_method.forward = _cap


def music_spec(R, M):
    R = R + 1e-10 * np.trace(R).real / R.shape[0] * np.eye(R.shape[0])
    _, V = np.linalg.eigh(R)
    En = V[:, : R.shape[0] - M]
    return 1.0 / (np.sum(np.abs(En.conj().T @ A) ** 2, axis=0) + 1e-30)


def pick(spec, grid, M, min_sep=MIN_SEP_DEG):
    pk = find_peaks(spec)[0]
    order = list(pk[np.argsort(spec[pk])[::-1]]) if len(pk) else []
    order += [i for i in np.argsort(spec)[::-1]]
    keep = []
    for i in order:
        if all(abs(grid[i] - grid[j]) >= min_sep for j in keep):
            keep.append(int(i))
        if len(keep) == M:
            break
    return np.sort(grid[keep[:M]])


def to_grid(spec_lin, gdeg):
    o = np.argsort(gdeg)
    r = np.interp(GRID, np.asarray(gdeg)[o], np.asarray(spec_lin, float).ravel()[o], left=np.nan, right=np.nan)
    r = r / (np.nanmax(r) + 1e-30)
    with np.errstate(divide="ignore", invalid="ignore"):
        db = 10 * np.log10(np.clip(r, 1e-6, None))
    db[np.abs(GRID) > 70] = np.nan
    return db


rng = np.random.default_rng(5)
for k in range(3):
    gt = draw_angles(rng, 2, gap_deg=25.0)
    x, _ = make_scene(steer, gt, rng, T=int(SM.params.T), rho=RHO_PARTIAL)
    xt = torch.tensor(x, dtype=torch.complex128, device=device)[None]
    Rraw = (x @ x.conj().T) / x.shape[1]
    mus(xt, 2)                                                   # populates cap["Rz"]
    Rz = cap["Rz"][0]
    res = {}
    s = music_spec(Rraw, 2); res["MUSIC_raw_cov"] = (to_grid(s, MG), pick(s, MG, 2))
    s = music_spec(Rz, 2);   res["SubspaceNet_MUSIC_CNNcov"] = (to_grid(s, MG), pick(s, MG, 2))
    mfo._A_use, mfo._grid_use = mfo.A_fine, mfo.grid_fine
    sp = mfo._spectrum(xt, mfo.lam_multi).cpu().numpy().ravel()
    gf = np.rad2deg(mfo.grid_fine.cpu().numpy())
    res["MFOCUSS"] = (to_grid(sp, gf), np.sort(np.rad2deg(mfo(xt, 2)[0].cpu().numpy().ravel())))
    spi._A_use, spi._grid_use = spi.A_fine, spi.grid_fine
    sp = spi._spectrum(xt).cpu().numpy().ravel()
    res["SPICE_IAA"] = (to_grid(sp, np.rad2deg(spi.grid_fine.cpu().numpy())),
                        np.sort(np.rad2deg(spi(xt, 2)[0].cpu().numpy().ravel())))
    out = {}
    for name, (psd, ang) in res.items():
        ang = np.atleast_1d(np.asarray(ang, float))
        out[name] = dict(psd_dB=psd.astype(float), estimated_DoA=ang,
                         df_error=float(np.mean([min(abs(ang - g)) for g in gt])))
    savemat(os.path.join(OUT, f"results_mp_r{k+1}.mat"),
            dict(angles=GRID.astype(float), anglesGT=np.asarray(gt, float), results=out))
    print(f"scene {k+1}: GT={np.round(gt,1)} sep={gt[1]-gt[0]:.1f}deg  "
          + "  ".join(f"{n.split('_')[0]}={out[n]['df_error']:.2f}" for n in out), flush=True)
print("DONE", flush=True)
