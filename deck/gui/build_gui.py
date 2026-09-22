"""Inline the verification data into the GUI page.

The artifact sandbox blocks fetch/XHR even for a file published beside the page, so the data has
to ship inside the HTML. It is injected as a plain JS constant at the top of the page's own
script rather than as a <script type="application/json"> block: a non-executable script tag is
the kind of thing a sanitizer may drop or relocate, and if it did, getElementById would return
null and the page would fail exactly like a failed fetch. A const cannot fail that way.

doa_verify.html stays the editable source (it fetches, so serve it over http while editing);
this emits doa_verify_inline.html for publishing. Rerun after deck/export_gui_data.py.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "doa_verify.html"
DATA = HERE / "doa_verify_data.json"
OUT = HERE / "doa_verify_inline.html"
# A second copy lands beside the decks, because that is where these results are actually
# read. It is a plain file: open it by double-click, no server, no hosting, no account.
SHIP = Path(r"G:/My Drive/DOA_AI/DoA_Verification_Bench.html")

FETCH = '''    res = await fetch("doa_verify_data.json");
    if (!res.ok) throw new Error(res.status);
    DATA = await res.json();'''
INLINE = '''    DATA = PAYLOAD;
    if (!DATA || !DATA.meta) throw new Error("payload missing or malformed");'''


def main():
    html = SRC.read_text(encoding="utf-8")
    data = DATA.read_text(encoding="utf-8").strip()
    if FETCH not in html:
        raise SystemExit("loader block not found in doa_verify.html -- did boot() change?")
    html = html.replace(FETCH, INLINE, 1)
    html = html.replace("  let res;\n", "", 1)
    html = html.replace(
        'Could not read <code>doa_verify_data.json</code>. Regenerate it with '
        '<code>python deck/eval_150.py 150 --tag _diag</code> then <code>python deck/export_gui_data.py</code>.',
        'The embedded dataset did not load: <code>${err && err.message ? err.message : err}</code>. '
        'Rebuild with <code>python deck/export_gui_data.py</code> then <code>python deck/gui/build_gui.py</code>.')
    # "</" cannot appear outside a JSON string here, but escaping it anyway keeps a method name
    # from ever terminating the <script> element early.
    marker = "<script>\n"
    i = html.index(marker) + len(marker)
    html = html[:i] + "const PAYLOAD = " + data.replace("</", r"<\/") + ";\n" + html[i:]
    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT}  ({OUT.stat().st_size / 1024:.0f} KB, data {len(data) / 1024:.0f} KB)")
    try:
        SHIP.parent.mkdir(parents=True, exist_ok=True)
        SHIP.write_text(html, encoding="utf-8")
        print(f"wrote {SHIP}")
    except OSError as e:
        print(f"could not write {SHIP}: {e}")   # the Drive copy is a convenience, not a gate


if __name__ == "__main__":
    main()
