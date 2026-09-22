"""Export the per-realization traces + the published table as one JSON for the verification GUI.

The GUI recomputes every statistic from the RAW per-source errors rather than trusting a
pre-aggregated number, so it can show where a table value comes from -- and disagree with it.
"""
import json, sys
from pathlib import Path
import numpy as np

RES = Path(r"C:/GitHub/DUNCS/data/simulations/results")
# Canonical traces by default, so the GUI and the performance tables come from the SAME run.
# It previously read the 150-seed _diag file while the tables were at 400, which made every
# "agreement with published table" check disagree for a reason that was not a defect.
NPZ = Path(sys.argv[1]) if len(sys.argv) > 1 else RES / "eval_150_traces.npz"
TABLE = RES / "eval_150.json"
OUT = Path(r"C:/GitHub/DUNCS/deck/gui/doa_verify_data.json")

SCEN = ["single", "reuse15", "reuse25", "multipath15", "multipath25"]
COLS = ["Synth-to-Synth", "Synth-to-Sim", "Sim-to-Sim"]
COL_TABLE = {"Synth-to-Synth": "Synth\u2192Synth", "Synth-to-Sim": "Synth\u2192Sim",
             "Sim-to-Sim": "Sim\u2192Sim"}
# The two "own readout" twins are NOT exported. They exist to show how much of SPICE's and
# DoAFormer's accuracy comes from the local-ML refinement both bolt on at the end -- useful once,
# for diagnosing why those two rows agreed with each other on ~91% of seeds. That question is
# answered and recorded in the deck. The GUI is for comparing the methods you would actually
# deploy, so it now carries only those. deck/eval_150.py still scores the twins, and the npz
# still holds them; re-add a name here to bring one back.
METHODS = ["ML (beamscan)", "ML-2D (joint)", "ML-AP (alt. proj.)", "MUSIC (classical)",
           "MFOCUSS", "SPICE (IAA)", "SubspaceNet-MUSIC", "DoAFormer", "DU-MFOCUSS",
           "DU-MFOCUSS-guarded"]


def main():
    d = np.load(NPZ, allow_pickle=False)
    table = json.loads(TABLE.read_text(encoding="utf-8"))
    out = {"meta": {}, "errors": {}, "table": {}, "crlb": {}}
    nreal = 0
    for sc in SCEN:
        out["errors"][sc] = {}
        out["table"][sc] = {}
        out["crlb"][sc] = {}
        for c in COLS:
            out["errors"][sc][c] = {}
            out["table"][sc][c] = {}
            tc = COL_TABLE[c]
            cr = table["cells"].get(f"{sc}|{tc}|CRLB")
            out["crlb"][sc][c] = cr[0] if cr else None
            for m in METHODS:
                k = f"{sc}__{c}__{m}"
                if k in d.files:
                    a = np.asarray(d[k], float)
                    if a.ndim == 1:
                        a = a[:, None]
                    nreal = max(nreal, a.shape[0])
                    out["errors"][sc][c][m] = [[round(float(x), 3) for x in row] for row in a]
            for mk, field in (("__az", "az"), ("__snr", "snr")):
                kk = f"{sc}__{c}__{mk}"
                if kk in d.files:
                    v = np.asarray(d[kk], float)
                    out.setdefault(field, {}).setdefault(sc, {})[c] = (
                        [[round(float(x), 2) for x in row] for row in v] if v.ndim > 1
                        else [round(float(x), 2) for x in v])
                t = table["cells"].get(f"{sc}|{tc}|{m}")
                if t:
                    out["table"][sc][c][m] = {"rms": t[0], "md": t[1], "rms_all": t[2]}
    out["meta"] = {"realizations": nreal, "scenarios": SCEN, "columns": COLS,
                   "methods": [m for m in METHODS if m in out["errors"]["single"][COLS[0]]],
                   "table_realizations": table["meta"].get("N"),
                   "carrier_mhz": table["meta"].get("carrier_mhz"),
                   "note": table["meta"].get("note", "")}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    kb = OUT.stat().st_size / 1024
    print(f"wrote {OUT}  ({kb:.0f} KB, {nreal} realizations, {len(out['meta']['methods'])} methods)")


if __name__ == "__main__":
    main()
