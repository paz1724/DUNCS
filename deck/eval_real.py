"""Evaluate on the REAL-PROPAGATION DataSim recordings, not on a synthetic stand-in.

deck/doa_scenes.py generates multipath from a parametric model. This script instead scores the
methods on C:/GitHub/DOA_AI_Data/DataSim -- the Hof data-simulator set the project calls "real
propagation from the other machine": 88,950 files, each holding ~17 single-source instances of a
[5 sensors x 8 snapshots] complex snapshot matrix with a ground-truth azimuth and its own SNR.
Nothing about the propagation, the SNR spread or the multipath is chosen by us here.

Azimuth convention is verified rather than assumed: the sign/offset that minimizes error is
reported for both polarities, so a convention mismatch shows up as a large constant bias instead
of silently inflating every method equally.
"""
import sys, os, glob, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np, torch, scipy.io as sio, scipy.signal as sig
from src.config.simulation_config import load_simulation_config
from src.system_model import SystemModel
from src.models import ModelGenerator
from src.utils import device

REAL_DIR = r"C:/GitHub/DOA_AI_Data/DataSim"
N_INST = int(os.environ.get("N_INST", 400))
CONE_DEG = 65.0                     # keep to the front cone the models were trained on

torch.set_grad_enabled(False)
cfg = load_simulation_config("src/config/subspaceNet.yaml"); cfg.system_model.M = 2
SM = SystemModel(cfg.system_model)
steer = lambda a: np.asarray(SM.steering_vec(float(np.deg2rad(a))))
MG = np.arange(-70, 70 + 1e-6, 0.2)
A = np.stack([steer(t) for t in MG], axis=1); An2 = np.sum(np.abs(A) ** 2, axis=0)


def load_instances(n_want, seed=0):
    """Pull n_want single-source instances (x [5,8], az_deg, snr_db) from random files."""
    files = sorted(glob.glob(os.path.join(REAL_DIR, "*.mat")))
    rng = np.random.default_rng(seed)
    rng.shuffle(files)
    out = []
    for f in files:
        try:
            t = sio.loadmat(f, squeeze_me=True, struct_as_record=False)["sTrainingData"]
        except Exception:
            continue
        sigs = np.atleast_1d(t.inputSignal)
        az = np.atleast_1d(t.sGT.Az).astype(float)
        snr = np.atleast_1d(t.SNR).astype(float)
        for i in range(min(len(sigs), len(az))):
            x = np.asarray(sigs[i])
            if x.ndim != 2 or x.shape[0] != SM.params.N:
                continue
            if abs(az[i]) > CONE_DEG:                 # outside the trained field of view
                continue
            out.append((x.astype(complex), float(az[i]), float(snr[i])))
            if len(out) >= n_want:
                return out
    return out


def ml(x):
    R = (x @ x.conj().T) / x.shape[1]
    return MG[int(np.argmax(np.real(np.sum(A.conj() * (R @ A), axis=0)) / An2))]


def music(x, M=1):
    R = (x @ x.conj().T) / x.shape[1]
    R = R + 1e-10 * np.trace(R).real / R.shape[0] * np.eye(R.shape[0])
    _, V = np.linalg.eigh(R); En = V[:, : R.shape[0] - M]
    sp = 1.0 / (np.sum(np.abs(En.conj().T @ A) ** 2, axis=0) + 1e-30)
    return MG[int(np.argmax(sp))]


def build(mt, pr, wf=None):
    m = (ModelGenerator().set_model_type(mt).set_system_model(SM)
         .set_model_params(pr).set_model()).model.to(device).eval()
    if wf:
        p = "data/weights/" + wf
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        r = m.load_state_dict(torch.load(p, map_location=device), strict=False)
        if r.missing_keys:
            raise RuntimeError(f"{wf}: {len(r.missing_keys)} missing keys -> partially random model")
    return m


if __name__ == "__main__":
    inst = load_instances(N_INST)
    az = np.array([a for _, a, _ in inst]); snr = np.array([s for _, _, s in inst])
    print(f"loaded {len(inst)} real-propagation instances from {REAL_DIR}")
    print(f"  azimuth  {az.min():7.1f} .. {az.max():7.1f} deg   (front cone |az| <= {CONE_DEG})")
    print(f"  SNR      {snr.min():7.1f} .. {snr.max():7.1f} dB   median {np.median(snr):.1f}\n")

    mfo = build("MFOCUSS", dict(num_iterations=100, grid_size=901, grid_range_deg=[-70, 70]))
    spi = build("SPICE", dict(num_iterations=100, grid_size=901))
    mus = build("SubspaceNet", dict(tau=7, diff_method="music_1D"), "music_150MHz_synth.pt")
    DFP = dict(d_model=96, nhead=4, num_encoder_layers=3, num_decoder_layers=2,
               dim_feedforward=192, input_mode="both")
    dfm = build("DoAFormer", DFP, "doaformer_150MHz_synth.pt")
    dfm_r = build("DoAFormer", DFP, "doaformer_150MHz_real.pt") if os.path.exists(
        "data/weights/doaformer_150MHz_real.pt") else None
    tt = lambda x: torch.tensor(x, dtype=torch.complex128, device=device)[None]
    METH = {
        "ML (beamscan)":     lambda x: ml(x),
        "MUSIC (classical)": lambda x: music(x),
        "MFOCUSS":           lambda x: float(np.rad2deg(mfo(tt(x), 1)[0].cpu().numpy().ravel()[0])),
        "SPICE (IAA)":       lambda x: float(np.rad2deg(spi(tt(x), 1)[0].cpu().numpy().ravel()[0])),
        "SubspaceNet-MUSIC": lambda x: float(np.rad2deg(mus(tt(x), 1)[0].cpu().numpy().ravel()[0])),
        "DoAFormer (synth-tr)": lambda x: float(np.rad2deg(dfm(tt(x), 1)[0].cpu().numpy().ravel()[0])),
    }
    if dfm_r is not None:
        METH["DoAFormer (real-tr)"] = lambda x: float(np.rad2deg(dfm_r(tt(x), 1)[0].cpu().numpy().ravel()[0]))
    print(f"{'method':22s} {'RMS':>7s} {'median':>8s} {'p90':>7s} {'bias':>7s}   {'RMS(-az)':>9s}")
    for nm, fn in METH.items():
        est = np.array([fn(x) for x, _, _ in inst])
        e = est - az                                  # signed, to expose a convention bias
        e2 = est + az                                 # if the azimuth sign convention were flipped
        print(f"{nm:22s} {np.sqrt(np.mean(e**2)):7.2f} {np.median(np.abs(e)):8.2f} "
              f"{np.percentile(np.abs(e),90):7.2f} {np.mean(e):7.2f}   {np.sqrt(np.mean(e2**2)):9.2f}")
    print("\nDONE")
