"""DF-error diagnostics: one panel per algorithm, three views of the same ensemble.

Every realization is one SEED. A seed draws a fresh source azimuth, a fresh SNR, and fresh
signals and noise. The carrier is fixed at 150 MHz and is never swept. Three figures per
scenario ask three different questions of that ensemble:

  df_traces_<scen>.png   average DF error vs REALIZATIONS -- does the number converge?
  df_az_<scen>.png       average DF error vs SOURCE AZIMUTH -- worse near the cone edge?
  df_snr_<scen>.png      average DF error vs SNR -- driven by the weak-signal seeds?

The seed-average alone cannot answer the last two. It collapses the whole ensemble into one
number, and hides where the error actually lives.

Input is the npz written by deck/eval_150.py, so these are exactly the scenes the table scored.

Usage:  python deck/df_error_plots.py [traces.npz]
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path(r"C:/GitHub/DUNCS/data/simulations/results")
OUTDIR = Path(r"C:/GitHub/DUNCS/data/simulations/Plots")
NPZ = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS / "eval_150_traces.npz"

SCENARIOS = ["single", "reuse15", "reuse25", "multipath15", "multipath25"]
COLUMNS = ["Synth-to-Synth", "Synth-to-Sim", "Sim-to-Sim"]
COL_C = {"Synth-to-Synth": "#4C72B0", "Synth-to-Sim": "#DD8452", "Sim-to-Sim": "#55A868"}
# Deployed set only. The two unrefined twins are diagnostics, and including them would stretch
# every shared axis to accommodate deliberate outliers.
METHODS = ["ML (beamscan)", "ML-2D (joint)", "MUSIC (classical)", "MFOCUSS",
           "SPICE (IAA)", "SubspaceNet-MUSIC", "DoAFormer", "DU-MFOCUSS"]
NCOL = 4
AZ_EDGES = np.arange(-70.0, 70.1, 14.0)        # 10 bins across the front cone
SNR_EDGES = np.arange(6.0, 30.01, 3.0)         # 8 bins across the drawn SNR range
PREVIEW_SEEDS = 80                             # seeds drawn individually; 400 overplots into a
                                               # solid block where nothing seed-for-seed is visible


def per_realization(a):
    """Collapse a stored [scenes, M] error block to one value per seed (mean over Tx sources)."""
    a = np.asarray(a, float)
    return a.mean(axis=1) if a.ndim > 1 else a


def running_mean(v):
    return np.cumsum(v) / np.arange(1, len(v) + 1)


def binned(x, y, edges):
    """Mean of y inside each bin of x. Empty bins return NaN, so the line simply breaks."""
    idx = np.digitize(x, edges) - 1
    mid, val = [], []
    for b in range(len(edges) - 1):
        m = idx == b
        mid.append(0.5 * (edges[b] + edges[b + 1]))
        val.append(y[m].mean() if m.any() else np.nan)
    return np.array(mid), np.array(val)


def grid(nplots):
    nrow = int(np.ceil(nplots / NCOL))
    fig, axes = plt.subplots(nrow, NCOL, figsize=(15.0, 3.3 * nrow + 1.0), squeeze=False)
    return fig, axes, nrow


def finish(fig, axes, have, nrow, title, sub, out):
    for j in range(len(have), nrow * NCOL):
        axes[j // NCOL][j % NCOL].axis("off")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.995)
    fig.text(0.5, 0.960, sub, ha="center", fontsize=9.5, style="italic", color="#444444")
    fig.tight_layout(rect=[0, 0.06, 1, 0.935])
    fig.savefig(out, dpi=120)
    plt.close(fig)


def fig_realizations(data, scen, have, out):
    """RAW error per realization -- no averaging over realizations.

    A running mean smooths away the very thing this figure exists to show: two methods that
    return the SAME estimate on the SAME seed produce identical spikes, seed for seed. An
    average hides that behind a smooth curve. The dotted reference is still a mean, but it is
    taken ACROSS METHODS within each seed, never across seeds.
    """
    fig, axes, nrow = grid(len(have))
    crowd = {}
    for c in COLUMNS:
        cur = [per_realization(data[f"{scen}__{c}__{m}"]) for m in have
               if f"{scen}__{c}__{m}" in data]
        if cur:
            n = min(len(x) for x in cur)
            crowd[c] = np.mean([x[:n] for x in cur], axis=0)   # across methods, per seed
    nreal = 0
    for i, m in enumerate(have):
        ax = axes[i // NCOL][i % NCOL]
        rms, hi = [], []
        for c in COLUMNS:
            k = f"{scen}__{c}__{m}"
            if k not in data:
                continue
            v = per_realization(data[k])
            nreal = max(nreal, len(v))
            w = min(PREVIEW_SEEDS, len(v))            # a window of seeds, each one still individual
            x = np.arange(1, w + 1)
            ax.plot(x, v[:w], lw=0.8, marker="o", ms=2.6, color=COL_C[c], alpha=0.95,
                    label=c if i == 0 else None)
            if c in crowd:
                n = min(w, len(crowd[c]))
                ax.plot(x[:n], crowd[c][:n], lw=0.9, ls=":", color=COL_C[c], alpha=0.65,
                        label=(c + " (mean across methods)" if i == 0 else None))
            a = np.asarray(data[k], float)
            rms.append("%.2f" % np.sqrt((a ** 2).mean()))
            hi.append(np.percentile(v[:w], 99))
        ax.set_title("%s   RMS_all %s" % (m, " / ".join(rms)), fontsize=9.5, fontweight="bold")
        ax.set_xlabel("realization (seed)", fontsize=8)
        if i % NCOL == 0:
            ax.set_ylabel("|DF error| per realization [deg]", fontsize=8)
        if hi:
            ax.set_ylim(0, max(max(hi) * 1.15, 1e-3))      # p99; misses at 90 deg run off-scale
        ax.grid(alpha=0.25, lw=0.5)
        ax.tick_params(labelsize=8)
    finish(fig, axes, have, nrow,
           "DF error per realization  -  %s  (seeds 1-%d of %d, 150 MHz, NOT averaged)"
           % (scen, min(PREVIEW_SEEDS, nreal), nreal),
           "one point per seed, no averaging   |   identical spikes at identical seeds = identical "
           "estimates   |   y clipped at p99   |   RMS_all uses all seeds", out)
    return nreal


def fig_binned(data, scen, have, key, edges, xlabel, title, sub, out):
    fig, axes, nrow = grid(len(have))
    for i, m in enumerate(have):
        ax = axes[i // NCOL][i % NCOL]
        for c in COLUMNS:
            k = "%s__%s__%s" % (scen, c, m)
            mk = "%s__%s__%s" % (scen, c, key)
            if k not in data or mk not in data:
                continue
            err = np.asarray(data[k], float)
            meta = np.asarray(data[mk], float)
            if key == "__az":
                # one sample per SOURCE, placed at that source's own true azimuth
                x, y = meta.ravel(), err.ravel()
            else:
                # one SNR per seed, shared by every source in that seed
                reps = err.shape[1] if err.ndim > 1 else 1
                x, y = np.repeat(meta, reps), err.ravel()
            n = min(len(x), len(y))
            mid, val = binned(x[:n], y[:n], edges)
            ax.plot(mid, val, lw=1.7, marker="o", ms=3, color=COL_C[c],
                    label=c if i == 0 else None)
        ax.set_title(m, fontsize=9.5, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=8)
        if i % NCOL == 0:
            ax.set_ylabel("mean |DF error| [deg]", fontsize=8)
        ax.grid(alpha=0.25, lw=0.5)
        ax.tick_params(labelsize=8)
    finish(fig, axes, have, nrow, title, sub, out)


def main():
    if not NPZ.exists():
        sys.exit("traces not found: %s  (run deck/eval_150.py first)" % NPZ)
    d = np.load(NPZ, allow_pickle=False)
    data = {k: d[k] for k in d.files}
    OUTDIR.mkdir(parents=True, exist_ok=True)
    has_meta = any(k.endswith("____az") for k in data)
    for scen in SCENARIOS:
        have = [m for m in METHODS if any("%s__%s__%s" % (scen, c, m) in data for c in COLUMNS)]
        if not have:
            print("skipped %s: no traces" % scen)
            continue
        n = fig_realizations(data, scen, have, OUTDIR / ("df_traces_%s.png" % scen))
        print("wrote df_traces_%s.png  (%d panels, %d seeds)" % (scen, len(have), n))
        if not has_meta:
            print("  no azimuth/SNR recorded -- rerun deck/eval_150.py for the binned views")
            continue
        fig_binned(data, scen, have, "__az", AZ_EDGES, "source azimuth [deg]",
                   "Average DF error vs source azimuth  -  %s  (150 MHz)" % scen,
                   "each point averages every source whose true azimuth falls in that bin   |   "
                   "a rise at the edges is the manifold, not the estimator",
                   OUTDIR / ("df_az_%s.png" % scen))
        fig_binned(data, scen, have, "__snr", SNR_EDGES, "per-source SNR [dB]",
                   "Average DF error vs SNR  -  %s  (150 MHz)" % scen,
                   "each point averages every seed whose drawn SNR falls in that bin   |   "
                   "SNR is drawn U(6,30) dB per seed",
                   OUTDIR / ("df_snr_%s.png" % scen))
        print("wrote df_az_%s.png and df_snr_%s.png" % (scen, scen))


if __name__ == "__main__":
    main()
