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
SNR_DB_RANGE = (6.0, 30.0)       # per-source SNR drawn per scene. Widened from (25,30) on
                                 # 2026-09-18. At 25-30 dB a single source is noise-limited
                                 # right at the CRLB, so every consistent estimator lands on
                                 # the same ML solution and the column cannot tell them apart:
                                 # measured, only 21-28% of each method's error was its own,
                                 # the rest was the shared noise realization. A 6 dB floor puts
                                 # seeds where estimators actually diverge. NOTE this is the
                                 # SHARED generator: training and evaluation both move, so every
                                 # learned checkpoint must be retrained or it is domain-mismatched.
RHO_PARTIAL = 0.9                # partial coherence for "multipath" pairs (rho=1 is degenerate)
POWER_IMBALANCE = (0.3, 1.0)     # weaker source's power relative to the stronger one
DATASIM_GAIN = (0.10, 0.30)      # specular reflection AMPLITUDE = -20..-10 dB power
DATASIM_NREFL = (1, 3)           # inclusive number of reflections per source
DATASIM_SPREAD_DEG = 40.0        # reflection angular offset from its source
DATASIM_POWER_BUDGET = 0.06      # TOTAL reflected power across all echoes, relative to the direct
                                 # path (-12.2 dB). Was 0.30 (-5.2 dB), which NEVER BOUND: with
                                 # 1-3 echoes each drawn at -20..-10 dB power the sum can only
                                 # reach 0.27, so the 'cap' was inert and the realized total
                                 # averaged 0.087 (-10.6 dB) -- at the top of, and often above,
                                 # the -20..-10 dB band this module documents for a specular echo.
                                 # Calibrated instead against the REAL DataSim recordings these
                                 # scenes stand in for: 400 real single-source instances score
                                 # 1.17-1.18 deg (ML / MUSIC / MFOCUSS, deck/eval_real.py).
                                 # Synthetic single-source beamscan RMS vs this budget
                                 # (800 scenes x 3 seeds, spread of the 3 seeds in brackets):
                                 #   0.30 -> 1.61   0.10 -> 1.41   0.08 -> 1.32
                                 #   0.06 -> 1.18 [1.18-1.21]      0.05 -> 1.09   0.04 -> 1.00
                                 # 0.06 reproduces the measured real-data error, so the synthetic
                                 # channel is no longer HARSHER than the recordings it emulates.
                                 # The echo error is irreducible for ANY estimator: with a ~50 deg
                                 # Rayleigh limit an echo <=40 deg away is inside the main lobe,
                                 # so it cannot be resolved away -- giving MUSIC the TRUE arrival
                                 # count made it WORSE (M=1 3.53 deg -> M=1+k 6.20), because
                                 # over-modelling at T=8 snapshots costs more than the bias it
                                 # removes. The only honest lever is the channel's own strength.
DATASIM_REFL_COHERENCE = 0.0     # correlation between an echo's waveform and its OWN direct path.
                                 # A specular echo travels a LONGER path, so it arrives delayed;
                                 # over the snapshot burst that delay decorrelates it and it acts
                                 # as an interferer (adds variance) rather than as part of one
                                 # wavefront. Reusing the direct waveform verbatim (=1.0) instead
                                 # makes direct+echo exactly rank-1 with an effective steering
                                 # vector a(th) + sum_k g_k e^{j phi_k} a(th_k) -- so the composite
                                 # genuinely ARRIVES off the ground truth, the direct path is not
                                 # identifiable even in principle, and every estimator returns the
                                 # same displaced answer. Measured on noiseless single-source
                                 # scenes, that displacement was 3.60 deg RMS, which is exactly the
                                 # 3.2-3.7 deg every method used to report on the DataSim columns.
                                 # Single-source DataSim RMS vs this value (400 scenes, beamscan):
                                 #   1.0 -> 3.39   0.7 -> 2.62   0.5 -> 2.19   0.3 -> 1.85   0 -> 1.57
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


def make_scene(steer, angles_deg, rng, *, T=8, rho=0.0, powers=None, datasim=False,
               refl_coherence=DATASIM_REFL_COHERENCE):
    """Build one received snapshot matrix x [N, T] from explicit signals (no global RNG).

    Args:
        steer: callable(angle_deg) -> complex steering vector [N] on the RECORDED manifold.
        angles_deg: ground-truth source angles.
        rng: np.random.Generator (the ONLY randomness source).
        rho: 0 = independent sources; 0<rho<1 = partially coherent; 1 = rank-1 degenerate.
        powers: per-source linear power weights, or None for equal power.
        datasim: add calibrated specular multipath reflections.
        refl_coherence: correlation of each echo's waveform with its own direct path; see
            DATASIM_REFL_COHERENCE. 0 = delayed/decorrelated echo (default), 1 = zero-delay echo
            that fuses with the direct path into a single displaced wavefront.
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
                # Echo waveform: partially/fully decorrelated from its direct path (see
                # DATASIM_REFL_COHERENCE). This is what keeps the DIRECT path identifiable -- with
                # a verbatim copy the two fuse into one wavefront and the ground truth stops being
                # a property of the data at all.
                s_ref = (refl_coherence * s[si:si + 1]
                         + np.sqrt(max(0.0, 1.0 - refl_coherence ** 2)) * cn((1, T)))
                clean = clean + (gm * np.exp(2j * np.pi * rng.random()) * np.asarray(steer(ref))[:, None]) * s_ref

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
    # ---- 1. Detection threshold and the all-pairs error matrix ----
    # The threshold is HALF the source separation, so an estimate can only be credited to the
    # source it is genuinely closer to -- it is impossible for one estimate to "detect" both.
    from scipy.optimize import linear_sum_assignment
    pred = np.atleast_1d(pred_deg).astype(float)
    gt = np.atleast_1d(gt_deg).astype(float)
    thr = detection_threshold(gt, cap_deg)
    C = np.abs(gt[:, None] - pred[None, :])
    if C.shape[1] == 0:
        return [], len(gt), len(gt), [cap_deg] * len(gt)
    # ---- 2. Optimal one-to-one assignment ----
    # Hungarian rather than nearest-neighbour: greedy matching lets a single good estimate be
    # claimed by two ground-truth sources, which would hide a miss. One-to-one makes that
    # impossible, so a method that reports one source twice is scored as missing the other.
    ri, ci = linear_sum_assignment(C)
    errs_all = np.full(len(gt), np.nan)
    errs_all[ri] = C[ri, ci]
    errs_all = np.where(np.isnan(errs_all), 180.0, errs_all)                  # unmatched GT = gross error
    # ---- 3. Split into detected and missed ----
    # Both are returned because RMS over DETECTED sources alone is conditional on each method's own
    # detection subset: a method that misses its hard scenes reports a flatteringly small RMS. The
    # tables therefore always show RMS alongside MD%, and errs_all is what is comparable across
    # methods regardless of how many sources each one found.
    det = errs_all <= thr
    return errs_all[det].tolist(), int((~det).sum()), len(gt), errs_all.tolist()
