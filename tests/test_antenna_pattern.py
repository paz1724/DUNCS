"""Validity tests for antenna-pattern steering generation.

Two flavors are exercised and compared:
  * "simulated" — the analytic / ideal ULA steering vector
    (``antenna_pattern=False`` → SteeringVectorGenerator._generate_far_field).
  * "field"     — the real measured ULA3 pattern loaded from the experiment
    (``antenna_pattern=True`` → interpolated sSteering.A, Mid band 136-550 MHz).

The tests confirm both flavors load/run, produce valid samples and covariances,
and that the measured field pattern actually differs from the ideal model.

Run:  pytest tests/test_antenna_pattern.py -v
"""
import os
import sys

import numpy as np
import pytest
import torch

# Make `src` importable regardless of pytest's rootdir.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.config.simulation_config import SystemModelParams
from src.signal_creation import Samples
from src.data_handler import create_dataset
from src.methods_pack.cov_reconstruct import sample_covariance
from src.steering_vector_generator import SteeringVectorGenerator

# Real measured steering data shipped with the Hof repo (Mid band: 136-550 MHz).
FIELD_FILE = r"C:\GitHub\Hof\Auxiliary\Data\Steering\ULA3\SteeringData_Mid.mat"

# Angle grid used for steering-vector comparisons (within the DoA range).
ANGLES_DEG = np.linspace(-70.0, 70.0, 15)
ANGLES_RAD = np.deg2rad(ANGLES_DEG)

pytestmark = pytest.mark.skipif(
    not os.path.exists(FIELD_FILE),
    reason=f"Hof ULA3 steering data not available at {FIELD_FILE}",
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _make_params(antenna_pattern: bool, pattern_file=None) -> SystemModelParams:
    """DUNCS default scenario (N=5 ULA, M=1, T=8, SNR=30, far-field)."""
    return SystemModelParams(
        M=1, T=8, snr=30, field_type="Far", signal_nature="non-coherent",
        array_form="ula", N=5, doa_range=[-70, 70], min_gap=5,
        antenna_pattern=antenna_pattern, antenna_pattern_file=pattern_file,
        eta=0.0, bias=0.0, sv_noise_var=0.0,
    )


def _build_samples(antenna_pattern: bool, pattern_file=None):
    """Build a Samples model for one flavor (resets the SV singleton first)."""
    SteeringVectorGenerator.reset_instance()
    params = _make_params(antenna_pattern, pattern_file)
    return Samples(params, antenna_pattern), params


def _steering_matrix(samples, angles_rad) -> np.ndarray:
    """Stack per-angle steering vectors into an (N, K) matrix."""
    cols = [np.asarray(samples.steering_vec(float(t))).reshape(-1) for t in angles_rad]
    return np.stack(cols, axis=1)


def _simulated():
    return _build_samples(False, None)


def _field():
    return _build_samples(True, FIELD_FILE)


# --------------------------------------------------------------------------- #
# 1. the field pattern loads and matches the expected format
# --------------------------------------------------------------------------- #
def test_field_pattern_loads():
    sm, _ = _field()
    assert sm.pattern_data is not None, "field pattern failed to load"

    A = np.asarray(sm.pattern_data["A"])
    freq = np.asarray(sm.pattern_data["freq"])
    phi = np.asarray(sm.pattern_data["phi"])

    assert A.ndim == 3 and A.shape[0] == 5, f"unexpected A shape {A.shape}"
    assert np.iscomplexobj(A), "A must be complex phasors"
    assert A.shape[1] == phi.size and A.shape[2] == freq.size

    # Mid band: 136-550 MHz.
    assert freq.min() == pytest.approx(136, abs=1)
    assert freq.max() == pytest.approx(550, abs=1)
    # azimuth coverage must span the DoA range we sample over.
    assert phi.min() <= -70 and phi.max() >= 70


# --------------------------------------------------------------------------- #
# 2. both flavors produce steering vectors of the same shape, all finite
# --------------------------------------------------------------------------- #
def test_steering_shapes_match_between_flavors():
    a_sim = _steering_matrix(*( (_simulated()[0], ANGLES_RAD) ))
    a_fld = _steering_matrix(*( (_field()[0], ANGLES_RAD) ))

    assert a_sim.shape == (5, ANGLES_RAD.size)
    assert a_fld.shape == (5, ANGLES_RAD.size)
    assert np.isfinite(a_sim).all()
    assert np.isfinite(a_fld).all()


# --------------------------------------------------------------------------- #
# 3. the comparison: measured field pattern DIFFERS from the ideal model
# --------------------------------------------------------------------------- #
def test_simulated_vs_field_differ():
    sim, _ = _simulated()
    fld, _ = _field()
    a_sim = _steering_matrix(sim, ANGLES_RAD)
    a_fld = _steering_matrix(fld, ANGLES_RAD)

    # phase- and scale-invariant similarity per angle: |a_sim^H a_fld| / (|a_sim||a_fld|)
    corr = np.empty(ANGLES_RAD.size)
    for k in range(ANGLES_RAD.size):
        x, y = a_sim[:, k], a_fld[:, k]
        nx, ny = np.linalg.norm(x), np.linalg.norm(y)
        assert nx > 0 and ny > 0, "degenerate steering vector"
        corr[k] = np.abs(np.vdot(x, y)) / (nx * ny)

    assert np.isfinite(corr).all()
    # They are the same physical array → still broadly correlated...
    assert corr.mean() > 0.1
    # ...but the measured pattern is NOT identical to the ideal model.
    assert np.max(np.abs(corr - 1.0)) > 1e-3, "field pattern is identical to ideal"

    print(f"\n[compare] mean |corr| sim-vs-field = {corr.mean():.4f} "
          f"(min={corr.min():.4f}, max={corr.max():.4f})")


# --------------------------------------------------------------------------- #
# 4. sample generation runs for both flavors and yields valid observations
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("antenna_pattern,pattern_file",
                         [(False, None), (True, FIELD_FILE)],
                         ids=["simulated", "field"])
def test_sample_generation_both_flavors(antenna_pattern, pattern_file):
    sm, params = _build_samples(antenna_pattern, pattern_file)
    ds = create_dataset(sm, samples_size=8)
    ds.set_system_model_params(params)

    x, M, Y = ds[0]
    assert tuple(x.shape) == (5, 8), f"unexpected sample shape {tuple(x.shape)}"
    assert x.dtype == torch.complex128
    assert torch.isfinite(x).all(), "non-finite values in generated sample"
    assert int(M) == 1
    assert Y.numel() == 1


# --------------------------------------------------------------------------- #
# 5. the sample covariance is a valid (Hermitian PSD) matrix for both flavors
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("antenna_pattern,pattern_file",
                         [(False, None), (True, FIELD_FILE)],
                         ids=["simulated", "field"])
def test_sample_covariance_valid_both_flavors(antenna_pattern, pattern_file):
    sm, params = _build_samples(antenna_pattern, pattern_file)
    ds = create_dataset(sm, samples_size=8)
    ds.set_system_model_params(params)

    x, _, _ = ds[0]
    R = sample_covariance(x)[0]  # (N, N)

    assert torch.isfinite(R).all()
    # Hermitian
    assert torch.allclose(R, R.conj().transpose(-2, -1), atol=1e-6)
    # PSD (a rank-1 covariance from one source has eigenvalues >= ~0)
    eig = torch.linalg.eigvalsh(0.5 * (R + R.conj().transpose(-2, -1)))
    assert eig.min().item() > -1e-6, f"covariance not PSD (min eig {eig.min().item()})"
