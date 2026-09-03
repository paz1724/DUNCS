"""Canonical scene generator + scorer for the DUNCS rebuilt evaluation.

Shared by BOTH training and evaluation so the two can never drift apart. Fixes several defects
found in the Aug-30 adversarial audit of the previous ad-hoc generators:

  * FREQUENCY  - the recorded manifold slice is now the configured carrier (150 MHz), not the
                 middle of the 136-550 MHz band (343 MHz), which had silently evaluated a 2.3x
                 larger electrical aperture (aperture/lambda 1.15 -> 2.63).
  * COHERENCE  - pairs use PARTIAL coherence rho (default 0.9). Using signal_nature="coherent"
                 gave rho = 1.0 exactly (both sources sharing ONE waveform), a rank-1 degeneracy:
                 toggling it alone moved SubspaceNet-MUSIC's miss-rate 71% -> 2%, so the old
                 "multipath" column measured that degeneracy rather than multipath.
  * MULTIPATH  - specular reflections at -20..-10 dB (amplitude 0.10-0.30) with the TOTAL
                 reflected power capped, instead of the unphysical -5 dB that made the composite
                 wavefront arrive off the direct-path ground truth (every estimator then agreed
                 on the same "wrong" answer).
  * RNG        - every random draw comes from an explicitly-seeded Generator passed in by the
                 caller. Nothing touches the global np.random state, so runs are reproducible.
  * POWERS     - per-source power imbalance is applied in BOTH training and evaluation.
"""
import numpy as np

# ---- signal / scene defaults (no magic numbers inline) ----
SNR_DB_RANGE = (25.0, 30.0)      # per-source SNR drawn per scene
RHO_PARTIAL = 0.9                # partial coherence for "multipath" pairs (rho=1 is degenerate)
POWER_IMBALANCE = (0.3, 1.0)     # weaker source's power relative to the stronger one
DATASIM_GAIN = (0.10, 0.30)      # specular reflection AMPLITUDE = -20..-10 dB power
DATASIM_NREFL = (1, 3)           # inclusive number of reflections per source
DATASIM_SPREAD_DEG = 40.0        # reflection angular offset from its source
DATASIM_POWER_BUDGET = 0.30      # total reflected power <= 30% of the direct path
SEP_MAX_DEG = 40.0               # pair separation ~ U(gap, SEP_MAX): the project's definition,
                                 # i.e. reuse>=15 is U(15,40) and reuse>=25 is U(25,40)
FRONT_CONE_DEG = 70.0            # |theta| <= 70 (the configured doa_range)
EDGE_MARGIN_DEG = 2.0            # keep sources/reflections off the very cone edge


def draw_angles(rng, n_src, gap_deg=None, lo=-65.0, hi=65.0, sep_max=SEP_MAX_DEG):
    """Draw n_src angles (deg), continuous (never snapped to the manifold's 3-deg nodes).

    The previous 'Real' pool drew GT from exactly the recorded manifold's interpolation nodes,
    which is the grid estimators' own snapping lattice -> it handed them a free exact answer
    (MFOCUSS single RMS 0.86 -> 0.14 deg from that alone). Always draw continuously.

    SEPARATION follows the project definition: separation ~ U(gap_deg, SEP_MAX_DEG), so
    'reuse >= 15' is U(15, 40) and 'reuse >= 25' is U(25, 40). Both training and evaluation use it,
    so the two never drift apart.
    """
    if n_src == 1:
        return np.array([rng.uniform(lo, hi)])
    gap = float(gap_deg)
    sep = rng.uniform(gap, float(sep_max))       # U(gap, 40) -- the project's reuse definition
    a = rng.uniform(lo, hi - sep)
    return np.sort([a, a + sep])


def make_scene(steer, angles_deg, rng, *, T=8, rho=0.0, powers=None, datasim=False):
    """Build one received snapshot matrix x [N, T] from explicit signals (no global RNG).

    Args:
        steer: callable(angle_deg) -> complex steering vector [N] on the RECORDED manifold.
        angles_deg: ground-truth source angles.
        rng: np.random.Generator (the ONLY randomness source).
        rho: 0 = independent sources; 0<rho<1 = partially coherent; 1 = rank-1 degenerate.
        powers: per-source linear power weights, or None for equal power.
        datasim: add calibrated specular multipath reflections.
    Returns:
        x [N, T] complex, and the realized per-source SNR in dB.
    """
    angles_deg = np.atleast_1d(angles_deg).astype(float)
    M = len(angles_deg)
    A = np.stack([np.asarray(steer(a)) for a in angles_deg], axis=1)          # [N, M]
    N = A.shape[0]

    def cn(shape):                                                            # unit-power complex normal
        return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2.0)

    s = cn((M, T))
    if M > 1 and rho > 0:                                                     # PARTIAL coherence
        s[1] = rho * s[0] + np.sqrt(max(0.0, 1.0 - rho ** 2)) * s[1]
    if powers is not None:
        s = s * np.sqrt(np.asarray(powers, float))[:, None]

    clean = A @ s
    if datasim:                                                               # calibrated multipath
        for si, a in enumerate(angles_deg):
            k = int(rng.integers(DATASIM_NREFL[0], DATASIM_NREFL[1] + 1))
            g = rng.uniform(*DATASIM_GAIN, size=k)
            if (g ** 2).sum() > DATASIM_POWER_BUDGET:
                g = g * np.sqrt(DATASIM_POWER_BUDGET / (g ** 2).sum())
            for gm in g:
                lim = FRONT_CONE_DEG - EDGE_MARGIN_DEG
                ref = float(np.clip(a + rng.uniform(-DATASIM_SPREAD_DEG, DATASIM_SPREAD_DEG), -lim, lim))
                clean = clean + (gm * np.exp(2j * np.pi * rng.random()) * np.asarray(steer(ref))[:, None]) * s[si:si + 1]

    snr_db = rng.uniform(*SNR_DB_RANGE)
    noise = cn((N, T)) * np.sqrt(1.0 / (10 ** (snr_db / 10.0)))
    return clean + noise, snr_db


def detection_threshold(gt_deg, cap_deg=10.0):
    """Detection threshold: never more than HALF the true source separation.

    With a fixed 10-deg threshold a 15-deg pair that is completely unresolved (both estimates on
    the midpoint, 7.5 deg from each source) scored as TWO correct detections.
    """
    gt = np.atleast_1d(gt_deg)
    if len(gt) < 2:
        return cap_deg
    return float(min(cap_deg, 0.5 * np.min(np.diff(np.sort(gt)))))


def score_scene(pred_deg, gt_deg, cap_deg=10.0):
    """Hungarian-match predictions to GT.

    Returns (errs_detected, n_missed, n_gt, errs_all) where errs_all keeps the true error of
    EVERY source (missed ones included). RMS over `errs_detected` alone is conditional on a
    method's own detection subset -- a method that misses its hard sources reports a smaller RMS
    -- so `errs_all` is what may be compared across methods.
    """
    from scipy.optimize import linear_sum_assignment
    pred = np.atleast_1d(pred_deg).astype(float)
    gt = np.atleast_1d(gt_deg).astype(float)
    thr = detection_threshold(gt, cap_deg)
    C = np.abs(gt[:, None] - pred[None, :])
    if C.shape[1] == 0:
        return [], len(gt), len(gt), [cap_deg] * len(gt)
    ri, ci = linear_sum_assignment(C)
    errs_all = np.full(len(gt), np.nan)
    errs_all[ri] = C[ri, ci]
    errs_all = np.where(np.isnan(errs_all), 180.0, errs_all)                  # unmatched GT = gross error
    det = errs_all <= thr
    return errs_all[det].tolist(), int((~det).sum()), len(gt), errs_all.tolist()
