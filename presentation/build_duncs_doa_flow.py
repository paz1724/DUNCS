"""Build DUNCS_DOA_flow.pptx — describes the DoA simulation flow in DUNCS.

Sections:
  1. Title
  2. Abstract / What is DUNCS
  3. End-to-end pipeline diagram
  4. System model (sparse array geometry)
  5. Signal & sample generation
  6. Forward pass overview (DUNCS model)
  7. ADMM unfolding — measured projection
  8. ADMM unfolding — the five updates
  9. Subspace step (ESPRIT/MUSIC) + RMSPE loss
 10. Training loop
 11. Evaluation: classical baselines + CRB
 12. How to run / file map

Generates: presentation/DUNCS_DOA_flow.pptx
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.dml.color import RGBColor

OUT_PATH = Path(__file__).parent / "DUNCS_DOA_flow.pptx"

# ---------------------- style ----------------------
COL_TITLE_BG = RGBColor(0x1F, 0x49, 0x8A)   # deep navy
COL_TITLE_FG = RGBColor(0xFF, 0xFF, 0xFF)
COL_ACCENT   = RGBColor(0xE6, 0x4F, 0x1F)   # warm accent
COL_CARD_BG  = RGBColor(0xF4, 0xF6, 0xFA)
COL_CARD_BD  = RGBColor(0xCB, 0xD3, 0xDD)
COL_TEXT     = RGBColor(0x1B, 0x21, 0x2A)
COL_SUB      = RGBColor(0x55, 0x5F, 0x6A)
COL_OK       = RGBColor(0x2D, 0x8A, 0x4E)
COL_MATH_BG  = RGBColor(0xEE, 0xF2, 0xF8)

SLIDE_W, SLIDE_H = Inches(13.333), Inches(7.5)


def _new_pres():
    prs = Presentation()
    prs.slide_width  = SLIDE_W
    prs.slide_height = SLIDE_H
    return prs


def _blank_slide(prs):
    blank = prs.slide_layouts[6]
    return prs.slides.add_slide(blank)


def _add_title_bar(slide, text):
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_W, Inches(0.85))
    bar.line.fill.background()
    bar.fill.solid()
    bar.fill.fore_color.rgb = COL_TITLE_BG
    tf = bar.text_frame
    tf.margin_left = Inches(0.4)
    tf.margin_top  = Inches(0.1)
    tf.margin_bottom = Inches(0.05)
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    r = p.add_run()
    r.text = text
    r.font.size = Pt(28)
    r.font.bold = True
    r.font.color.rgb = COL_TITLE_FG


def _add_text_box(slide, x, y, w, h, text, size=14, bold=False, color=COL_TEXT,
                  align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, italic=False):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = Inches(0.05)
    tf.margin_right = Inches(0.05)
    tf.margin_top = Inches(0.02)
    tf.margin_bottom = Inches(0.02)
    lines = text.split("\n") if isinstance(text, str) else text
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run()
        r.text = line
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.italic = italic
        r.font.color.rgb = color
    return tb


def _add_card(slide, x, y, w, h, title=None, bullets=None,
              title_size=15, body_size=13, accent=COL_ACCENT):
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    card.fill.solid()
    card.fill.fore_color.rgb = COL_CARD_BG
    card.line.color.rgb = COL_CARD_BD
    card.line.width = Pt(0.75)
    card.shadow.inherit = False
    if card.text_frame:
        card.text_frame.text = ""

    inner_x = x + Inches(0.15)
    inner_y = y + Inches(0.10)
    inner_w = w - Inches(0.30)
    inner_h = h - Inches(0.20)

    if title is not None:
        title_h = Inches(0.45)
        tb = slide.shapes.add_textbox(inner_x, inner_y, inner_w, title_h)
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = Inches(0.0)
        tf.margin_top = Inches(0.0)
        p = tf.paragraphs[0]
        r = p.add_run()
        r.text = title
        r.font.size = Pt(title_size)
        r.font.bold = True
        r.font.color.rgb = accent
        body_y = inner_y + title_h - Inches(0.05)
        body_h = inner_h - title_h + Inches(0.05)
    else:
        body_y = inner_y
        body_h = inner_h

    if bullets:
        tb = slide.shapes.add_textbox(inner_x, body_y, inner_w, body_h)
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = Inches(0.0)
        for i, line in enumerate(bullets):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.alignment = PP_ALIGN.LEFT
            r = p.add_run()
            r.text = ("• " + line) if not line.startswith("•") else line
            r.font.size = Pt(body_size)
            r.font.color.rgb = COL_TEXT
            p.space_after = Pt(2)
    return card


def _add_math(slide, x, y, w, h, latex_text):
    """Render a single block of pseudo-math as monospace text on a light
    background. python-pptx has no LaTeX rendering, but a courier-styled
    block with Unicode operators reads as the equation."""
    box = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    box.fill.solid()
    box.fill.fore_color.rgb = COL_MATH_BG
    box.line.color.rgb = COL_CARD_BD
    box.line.width = Pt(0.5)
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = Inches(0.15)
    tf.margin_right = Inches(0.15)
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = latex_text
    r.font.name = "Cambria Math"
    r.font.size = Pt(15)
    r.font.color.rgb = COL_TEXT


def _add_chevron(slide, x, y, w, h, text, bg=COL_TITLE_BG, fg=COL_TITLE_FG, size=13):
    sh = slide.shapes.add_shape(MSO_SHAPE.CHEVRON, x, y, w, h)
    sh.fill.solid()
    sh.fill.fore_color.rgb = bg
    sh.line.fill.background()
    tf = sh.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = Inches(0.05)
    tf.margin_right = Inches(0.1)
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.bold = True
    r.font.color.rgb = fg
    return sh


def _add_arrow(slide, x1, y1, x2, y2):
    conn = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, x1, y1, x2, y2)
    conn.line.color.rgb = COL_SUB
    conn.line.width = Pt(2)
    line = conn.line
    # arrowhead
    from pptx.oxml.ns import qn
    ln = line._get_or_add_ln()
    tail = ln.makeelement(qn('a:tailEnd'), {'type': 'triangle', 'w': 'med', 'len': 'med'})
    ln.append(tail)


# ============================================================
# SLIDES
# ============================================================

def slide_title(prs):
    s = _blank_slide(prs)
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_W, SLIDE_H)
    bar.fill.solid()
    bar.fill.fore_color.rgb = COL_TITLE_BG
    bar.line.fill.background()
    _add_text_box(s, Inches(0.5), Inches(2.4), Inches(12.3), Inches(1.3),
                  "DUNCS — DoA Simulation Flow",
                  size=44, bold=True, color=COL_TITLE_FG,
                  align=PP_ALIGN.CENTER)
    _add_text_box(s, Inches(0.5), Inches(3.7), Inches(12.3), Inches(0.7),
                  "Deep-Unfolded Sparse-Covariance ADMM for direction-of-arrival recovery",
                  size=22, color=COL_TITLE_FG, italic=True,
                  align=PP_ALIGN.CENTER)
    _add_text_box(s, Inches(0.5), Inches(5.8), Inches(12.3), Inches(0.5),
                  "End-to-end walkthrough: config → system model → samples → model → subspace → RMSPE",
                  size=15, color=COL_TITLE_FG,
                  align=PP_ALIGN.CENTER)


def slide_abstract(prs):
    s = _blank_slide(prs)
    _add_title_bar(s, "What is DUNCS?")
    _add_text_box(s, Inches(0.5), Inches(1.05), Inches(12.3), Inches(0.55),
                  "Deep-Unfolded Sparse-Covariance ADMM — a learned algorithm for DoA estimation from sparse arrays.",
                  size=17, italic=True, color=COL_SUB)

    _add_card(s, Inches(0.4), Inches(1.85), Inches(4.1), Inches(5.3),
              title="The problem",
              bullets=[
                  "Estimate angles of M radio sources from",
                  "    N antenna snapshots over T time samples.",
                  "Sparse arrays (MRA) save hardware but break",
                  "    classical subspace methods designed for ULA.",
                  "Low SNR / few snapshots → the sample",
                  "    covariance R̂ₓ becomes noisy & non-Toeplitz."
              ], accent=COL_ACCENT)

    _add_card(s, Inches(4.6), Inches(1.85), Inches(4.1), Inches(5.3),
              title="The DUNCS idea",
              bullets=[
                  "Reconstruct a clean Hermitian-Toeplitz-PSD",
                  "    covariance R on the virtual co-array.",
                  "Solve the convex reconstruction with ADMM,",
                  "    but unroll K iterations as a network.",
                  "Make ADMM penalty/step parameters",
                  "    (ρ, τ, μ) per-iteration learnable.",
                  "Feed the reconstructed R into a classical",
                  "    subspace method (ESPRIT / MUSIC).",
                  "Train end-to-end on RMSPE between predicted",
                  "    and true DoA — a differentiable pipeline."
              ], accent=COL_ACCENT)

    _add_card(s, Inches(8.8), Inches(1.85), Inches(4.1), Inches(5.3),
              title="Why this presentation",
              bullets=[
                  "Trace one full DoA-simulation pass through",
                  "    the DUNCS code base, slide by slide.",
                  "From YAML config to the final RMSPE number.",
                  "Pointers to the exact files / classes that",
                  "    implement each stage.",
                  "Companion to README.md and CLAUDE.md.",
                  "",
                  "Audience: anyone (or any Claude session)",
                  "    spinning up on the DUNCS code base."
              ], accent=COL_OK)


def slide_pipeline(prs):
    s = _blank_slide(prs)
    _add_title_bar(s, "End-to-End Pipeline")
    _add_text_box(s, Inches(0.5), Inches(1.05), Inches(12.3), Inches(0.5),
                  "main.py → SimulationRunner.run() — every box is a real class/function.",
                  size=15, italic=True, color=COL_SUB)

    # Row of chevrons
    stages = [
        ("YAML\nconfig",        "load_simulation_config()"),
        ("SystemModel",         "src/system_model.py"),
        ("Samples\n(signals)",  "src/signal_creation.py"),
        ("Dataset",             "src/data_handler.py"),
        ("Model\n(DUNCS)",      "models_pack/sparse_cov_admm_unfold.py"),
        ("Subspace\n(ESPRIT)",  "methods_pack/esprit.py"),
        ("Loss / Eval\n(RMSPE)","metrics/criterions.py")
    ]
    y_top = Inches(2.0)
    h     = Inches(1.0)
    n     = len(stages)
    total_w = Inches(12.5)
    gap = Inches(0.05)
    w = Emu(int((total_w - gap * (n - 1)) / n))
    x = Inches(0.42)
    for i, (label, _file) in enumerate(stages):
        bg = COL_TITLE_BG if i % 2 == 0 else COL_ACCENT
        _add_chevron(s, x, y_top, w, h, label, bg=bg, size=12)
        # file caption under the chevron
        _add_text_box(s, x, y_top + h + Inches(0.05), w, Inches(0.5),
                      stages[i][1], size=8, color=COL_SUB,
                      align=PP_ALIGN.CENTER)
        x = x + w + gap

    # Detail boxes below
    detail_y = Inches(3.8)
    _add_card(s, Inches(0.4), detail_y, Inches(6.2), Inches(3.4),
              title="Data path (per epoch)",
              bullets=[
                  "1. Samples.samples_creation() draws random DoAs (gap-respecting)",
                  "      and synthesises the steering matrix A(θ).",
                  "2. x[t] = A(θ) s[t] + n[t]  for t = 1..T snapshots.",
                  "3. R̂ₓ = (1/T) Σₜ x[t] x[t]ᴴ   (sample covariance)",
                  "4. DataLoader batches (B,N,T) tensors with collate_fn.",
                  "5. Model receives x → returns predicted DoAs (θ̂)."
              ], accent=COL_ACCENT)

    _add_card(s, Inches(6.75), detail_y, Inches(6.2), Inches(3.4),
              title="Loss path (per batch)",
              bullets=[
                  "1. DUNCS forward → reconstructed R̂ of size |U|×|U|.",
                  "2. ESPRIT(R̂, M) → estimated angles θ̂.",
                  "3. RMSPELoss(θ̂, θ)  with permutation-invariant",
                  "      matching of the M sources.",
                  "4. backward() through every ADMM iteration: ρ, τ,",
                  "      μ are nn.Parameter tensors of length K.",
                  "5. AdamW + ReduceLROnPlateau schedule the learning rate."
              ], accent=COL_OK)


def slide_system_model(prs):
    s = _blank_slide(prs)
    _add_title_bar(s, "System Model — Array Geometry & Signal")
    _add_text_box(s, Inches(0.5), Inches(1.05), Inches(12.3), Inches(0.5),
                  "src/system_model.py defines the physical array and the steering manifold.",
                  size=15, italic=True, color=COL_SUB)

    _add_card(s, Inches(0.4), Inches(1.75), Inches(6.2), Inches(2.4),
              title="Array geometry",
              bullets=[
                  "ULA   — uniform linear array, N sensors at λ/2 spacing.",
                  "MRA-N — minimum-redundancy array: N physical sensors,",
                  "        virtual ULA of size |U| via the difference co-array.",
                  "Sparse arrays use sparse_array.py to compute U (virtual",
                  "        sensor indices) and the selection operator Φ.",
                  "Optional perturbations: eta (position jitter),",
                  "        bias (uniform offset), sv_noise_var."
              ])

    _add_card(s, Inches(6.75), Inches(1.75), Inches(6.2), Inches(2.4),
              title="Signal model",
              bullets=[
                  "M sources at angles θ = (θ₁..θₘ), coherent or non-coherent.",
                  "Far-field plane wave → steering vector a(θ) ∈ ℂ^N.",
                  "Stack into steering matrix A(θ) = [a(θ₁)..a(θ_M)].",
                  "Snapshots: x[t] = A(θ) s[t] + n[t],  t = 1..T.",
                  "s[t]: source amplitudes, n[t]: complex AWGN at given SNR.",
                  "Output observation tensor: (B, N, T) complex64."
              ], accent=COL_OK)

    _add_math(s, Inches(0.4), Inches(4.35), Inches(12.5), Inches(0.9),
              "x[t]  =  A(θ) · s[t]  +  n[t]      →      R̂ₓ  =  (1/T) Σₜ x[t] x[t]ᴴ   "
              "∈ ℂ^{N×N}")

    _add_card(s, Inches(0.4), Inches(5.45), Inches(12.5), Inches(1.7),
              title="Defaults from src/config/DUNCS.yaml",
              bullets=[
                  "N = 5 sensors      M = 1 source       T = 8 snapshots       SNR = 30 dB",
                  "array_form = 'ula'    field_type = 'Far'    signal_nature = 'non-coherent'",
                  "doa_range = [−70°, 70°]      min_gap = 5° between DoAs at training time"
              ], body_size=13)


def slide_signal_creation(prs):
    s = _blank_slide(prs)
    _add_title_bar(s, "Signal & Sample Generation")
    _add_text_box(s, Inches(0.5), Inches(1.05), Inches(12.3), Inches(0.5),
                  "src/signal_creation.py — Samples class produces every training/test observation.",
                  size=15, italic=True, color=COL_SUB)

    _add_card(s, Inches(0.4), Inches(1.75), Inches(6.2), Inches(5.3),
              title="Samples.samples_creation() — per-sample recipe",
              bullets=[
                  "1. Sample M DoAs:",
                  "    • set_doa() draws angles in [−70°, +70°]",
                  "    • enforces ≥ 5° gap (create_doa_with_gap)",
                  "    • picks a random extra offset per source",
                  "",
                  "2. Build steering matrix:",
                  "    • SteeringVectorGenerator (singleton)",
                  "    • cached per (array geometry, frequency)",
                  "    • supports optional .mat antenna pattern",
                  "",
                  "3. Synthesise snapshots x[t]:",
                  "    • source signals s[t]: complex Gaussian",
                  "    • noise n[t]: scaled by SNR",
                  "    • coherent mode → shared s across sources",
                  "",
                  "4. Return tuple (x, θ, range, ...) into Dataset."
              ])

    _add_card(s, Inches(6.75), Inches(1.75), Inches(6.2), Inches(5.3),
              title="Dataset → DataLoader",
              bullets=[
                  "create_dataset(samples, N_samples, save?, paths, ...)",
                  "    • builds train and test splits separately",
                  "    • train_test_ratio = 0.1 by default",
                  "",
                  "Sample sizes (DUNCS.yaml defaults):",
                  "    • samples_size = 7000 training samples",
                  "    • 700 test samples",
                  "    • batch_size  = 128",
                  "",
                  "collate_fn batches variable-length tensors;",
                  "SameLengthBatchSampler groups same-T samples.",
                  "",
                  "dataset.materialize(system_model) re-computes",
                  "    derived quantities (R̂ₓ, masks) at the moment",
                  "    of use, instead of caching them on disk."
              ], accent=COL_OK)


def slide_forward_overview(prs):
    s = _blank_slide(prs)
    _add_title_bar(s, "DUNCS Forward Pass — Overview")
    _add_text_box(s, Inches(0.5), Inches(1.05), Inches(12.3), Inches(0.5),
                  "src/models_pack/sparse_cov_admm_unfold.py — DUNCS.forward()",
                  size=15, italic=True, color=COL_SUB)

    # 4 chevron stages
    stages = [
        ("Sample\ncovariance",       "R̂ₓ = (1/T) X Xᴴ"),
        ("Measured\nprojection",     "M = Φᴴ R̂ₓ Φ"),
        ("Unrolled ADMM\n(K iter.)", "R → S → T  +  duals"),
        ("Subspace +\nDoA estimate", "ESPRIT / MUSIC")
    ]
    y_top = Inches(2.0)
    h = Inches(1.4)
    total_w = Inches(12.5)
    gap = Inches(0.15)
    n = len(stages)
    w = Emu(int((total_w - gap * (n - 1)) / n))
    x = Inches(0.42)
    for i, (label, _math) in enumerate(stages):
        bg = COL_TITLE_BG if i % 2 == 0 else COL_ACCENT
        _add_chevron(s, x, y_top, w, h, label, bg=bg, size=14)
        _add_text_box(s, x, y_top + h + Inches(0.1), w, Inches(0.5),
                      stages[i][1], size=12, color=COL_SUB,
                      align=PP_ALIGN.CENTER)
        x = x + w + gap

    _add_card(s, Inches(0.4), Inches(4.6), Inches(12.5), Inches(2.6),
              title="What gets learned",
              bullets=[
                  "Per-iteration ADMM step / penalty parameters become nn.Parameter:",
                  "    • rho_r[k], rho_m[k]  — R-update inversion coefficients (|U|² each, K iters)",
                  "    • tau[k]              — soft-threshold for the SVT step (|U|-vector, K iters)",
                  "    • mu_u[k], mu_v[k]    — dual ascent step sizes (scalars, K iters)",
                  "The selection matrix Φ (built once from the array geometry) is frozen.",
                  "Subspace method (ESPRIT) is parameter-free — it just does eigen-decomp."
              ])


def slide_admm_projection(prs):
    s = _blank_slide(prs)
    _add_title_bar(s, "Step 1 — Measured Projection onto the Co-Array")
    _add_text_box(s, Inches(0.5), Inches(1.05), Inches(12.3), Inches(0.5),
                  "Lift the noisy N×N sample covariance onto the |U|×|U| virtual co-array.",
                  size=15, italic=True, color=COL_SUB)

    _add_math(s, Inches(0.4), Inches(1.85), Inches(12.5), Inches(0.9),
              "R̂ₓ  =  (1/T) X Xᴴ   ∈ ℂ^{N×N}      "
              "➜       M  =  Φᴴ R̂ₓ Φ   ∈ ℂ^{|U|×|U|}")

    _add_card(s, Inches(0.4), Inches(2.95), Inches(6.2), Inches(4.2),
              title="What Φ is",
              bullets=[
                  "Φ (built by build_phi() from the sparse-array",
                  "    geometry) maps physical sensor indices",
                  "    to virtual co-array indices.",
                  "Sparse layouts (e.g. MRA-5) yield |U| > N —",
                  "    a virtual ULA larger than the physical array.",
                  "P = Φᴴ Φ   (diagonal mask used in the R-update)",
                  "is precomputed once and reused across all iters."
              ])

    _add_card(s, Inches(6.75), Inches(2.95), Inches(6.2), Inches(4.2),
              title="What we want to recover",
              bullets=[
                  "An |U|×|U| matrix R that is simultaneously:",
                  "    • Hermitian   (R = Rᴴ)",
                  "    • Toeplitz    (depends only on i−j)",
                  "    • PSD         (R ⪰ 0)",
                  "    • Consistent with the measurements:",
                  "             Φᴴ R Φ  ≈  M",
                  "Subspace methods (ESPRIT) assume exactly",
                  "this structure — they fail on R̂ₓ directly.",
                  "DUNCS “denoises” into this feasible set."
              ], accent=COL_OK)


def slide_admm_iterations(prs):
    s = _blank_slide(prs)
    _add_title_bar(s, "Step 2 — Unrolled ADMM (K iterations)")
    _add_text_box(s, Inches(0.5), Inches(1.05), Inches(12.3), Inches(0.5),
                  "Each k = 1..K block runs five updates. K is fixed (default 20). Every ρ, τ, μ is learnable.",
                  size=15, italic=True, color=COL_SUB)

    # 5 small cards side-by-side
    items = [
        ("R-update",
         "Diagonal solve\nvec(R) = (P + 2ρ₝)⁻¹ · (vec(M) + ρ₝(S−U + T−V))"),
        ("S-update (SVT)",
         "Singular Value\nThresholding\nS = SVT_τ₊(R + U)"),
        ("T-update",
         "PSD ∘ Toeplitz ∘ Herm\nT = Π_PSD(Π_Toep(Π_Herm(R+V)))"),
        ("U dual",
         "U ← U + μ_u (R − S)"),
        ("V dual",
         "V ← V + μ_v (R − T)")
    ]
    y_top = Inches(1.95)
    h     = Inches(2.4)
    total_w = Inches(12.5)
    gap = Inches(0.1)
    n = len(items)
    w = Emu(int((total_w - gap * (n - 1)) / n))
    x = Inches(0.42)
    accents = [COL_ACCENT, COL_OK, COL_TITLE_BG, COL_SUB, COL_SUB]
    for i, (title, body) in enumerate(items):
        _add_card(s, x, y_top, w, h, title=title,
                  bullets=body.split("\n"),
                  body_size=11, title_size=13, accent=accents[i])
        x = x + w + gap

    _add_card(s, Inches(0.4), Inches(4.55), Inches(12.5), Inches(2.6),
              title="Why unroll instead of solving the convex problem to convergence?",
              bullets=[
                  "Fixed compute budget  — K iterations → deterministic runtime, easy to deploy.",
                  "Learned hyperparameters  — ρ, τ, μ adapt to the SNR / snapshot regime.",
                  "End-to-end gradient  — the loss is on DoA, not on ADMM objective → the network",
                  "    can sacrifice optimisation purity for better downstream estimation.",
                  "Test-time control  — evaluate at K=20 (matched) or K=500 (classical ADMM)",
                  "    to confirm the learned net beats the converged baseline."
              ])


def slide_subspace_loss(prs):
    s = _blank_slide(prs)
    _add_title_bar(s, "Step 3 — Subspace Method + RMSPE Loss")
    _add_text_box(s, Inches(0.5), Inches(1.05), Inches(12.3), Inches(0.5),
                  "After K iterations, the reconstructed T is fed to a parameter-free subspace estimator.",
                  size=15, italic=True, color=COL_SUB)

    _add_card(s, Inches(0.4), Inches(1.75), Inches(6.2), Inches(2.7),
              title="Subspace step — ESPRIT (default)",
              bullets=[
                  "Eigen-decompose T  →  signal subspace U_s of size |U|×M.",
                  "Split U_s into two overlapping sub-arrays U₁, U₂.",
                  "Solve U₁ Ψ = U₂; eigenvalues of Ψ → angles θ̂.",
                  "Alternatives: music.py, root_music.py — same interface."
              ])

    _add_card(s, Inches(6.75), Inches(1.75), Inches(6.2), Inches(2.7),
              title="RMSPE — Root Mean-Squared Phase Error",
              bullets=[
                  "Pairs each predicted θ̂ with the closest true θ via",
                  "    permutation-invariant Hungarian matching.",
                  "Wraps the residual to [−π, π) before squaring —",
                  "    avoids spurious 2π penalties.",
                  "Differentiable wrt the subspace inputs → gradient",
                  "    flows back through ADMM into ρ, τ, μ."
              ], accent=COL_OK)

    _add_math(s, Inches(0.4), Inches(4.65), Inches(12.5), Inches(0.9),
              "L_RMSPE  =  √( (1/M) Σᵢ  wrap(θ̂_π(i) − θ_i)² )"
              "      with  π  =  arg min over permutations")

    _add_card(s, Inches(0.4), Inches(5.65), Inches(12.5), Inches(1.5),
              title="Alternative criterion: ADMM objective",
              bullets=[
                  "Set criterion='admm_objective' to train on the ADMM",
                  "    residual itself rather than RMSPE — stable on hard regimes",
                  "    where supervised DoA matching is ambiguous (e.g. coherent sources)."
              ])


def slide_training_loop(prs):
    s = _blank_slide(prs)
    _add_title_bar(s, "Training Loop")
    _add_text_box(s, Inches(0.5), Inches(1.05), Inches(12.3), Inches(0.5),
                  "src/training.py — train() consumes the TrainingParams builder.",
                  size=15, italic=True, color=COL_SUB)

    _add_card(s, Inches(0.4), Inches(1.75), Inches(6.2), Inches(5.3),
              title="Per-epoch loop",
              bullets=[
                  "for epoch in 1..E:",
                  "    for batch (x, θ) in DataLoader:",
                  "        θ̂ = model(x, M, phase='train')",
                  "        loss = RMSPELoss(θ̂, θ)",
                  "        loss.backward()",
                  "        optimizer.step()",
                  "    validate on held-out split",
                  "    scheduler.step(val_loss)",
                  "",
                  "Saving:",
                  "    • final state_dict → data/weights/final_models/",
                  "    • per-config filename built by",
                  "      model.get_model_file_name()"
              ])

    _add_card(s, Inches(6.75), Inches(1.75), Inches(6.2), Inches(5.3),
              title="Defaults (DUNCS.yaml)",
              bullets=[
                  "Optimizer            AdamW",
                  "Learning rate        5e-4    weight decay 1e-5",
                  "Scheduler            ReduceLROnPlateau",
                  "    step_size=40    gamma=0.5",
                  "Epochs               90",
                  "Batch size           128",
                  "samples_size         7000  (train) + 700 (test)",
                  "Training objective   'angle'",
                  "",
                  "All five ADMM parameter vectors are jointly",
                  "optimised end-to-end with the rest of the model;",
                  "ESPRIT contributes only via the chain rule on θ̂."
              ], accent=COL_OK)


def slide_evaluation(prs):
    s = _blank_slide(prs)
    _add_title_bar(s, "Evaluation — Baselines and CRB")
    _add_text_box(s, Inches(0.5), Inches(1.05), Inches(12.3), Inches(0.5),
                  "src/evaluation.py — evaluate() compares the learned model against fair baselines.",
                  size=15, italic=True, color=COL_SUB)

    _add_card(s, Inches(0.4), Inches(1.75), Inches(4.1), Inches(5.3),
              title="What gets scored",
              bullets=[
                  "Trained DUNCS @ K=20  —",
                  "    the headline number.",
                  "Trained DUNCS @ K=500 —",
                  "    sanity check that learned",
                  "    rates still beat plain ADMM.",
                  "Classical ADMM @ K=20, 500",
                  "    on the same sample covariance.",
                  "Plain sample-covariance ESPRIT —",
                  "    no reconstruction at all.",
                  "Other models (SubspaceNet,",
                  "    DA-MUSIC, TransMUSIC, ...)",
                  "    if enabled in evaluation.models."
              ])

    _add_card(s, Inches(4.6), Inches(1.75), Inches(4.1), Inches(5.3),
              title="Metric: test RMSPE",
              bullets=[
                  "Reported per (model, K, scenario).",
                  "Averaged over Monte-Carlo runs",
                  "    if monte_carlo_simulations > 1.",
                  "Scenario sweeps in YAML let you",
                  "    plot RMSPE vs SNR or T",
                  "    with a single config run.",
                  "Outputs:",
                  "    • score logs in data/simulations/",
                  "                /results/scores/",
                  "    • plots in .../results/plots/"
              ], accent=COL_OK)

    _add_card(s, Inches(8.8), Inches(1.75), Inches(4.1), Inches(5.3),
              title="The CRB floor",
              bullets=[
                  "src/metrics/crb.py computes the",
                  "    Cramér–Rao Bound on the DoA",
                  "    variance for the current",
                  "    (N, M, T, SNR) configuration.",
                  "Drawn as a dashed line on every",
                  "    RMSPE-vs-X sweep.",
                  "Any model below CRB is a bug;",
                  "    a model close to CRB is",
                  "    statistically efficient."
              ], accent=COL_ACCENT)


def slide_how_to_run(prs):
    s = _blank_slide(prs)
    _add_title_bar(s, "How to Run / File Map")
    _add_text_box(s, Inches(0.5), Inches(1.05), Inches(12.3), Inches(0.5),
                  "Quick reference for landing in the code base.",
                  size=15, italic=True, color=COL_SUB)

    _add_card(s, Inches(0.4), Inches(1.75), Inches(6.2), Inches(5.3),
              title="Commands",
              bullets=[
                  "pip install -r requirements.txt   (Python 3.10+, PyTorch 2.6)",
                  "",
                  "python main.py duncs              # single DUNCS sim",
                  "python main.py sparsenet          # SparseNet baseline",
                  "python compare_models.py          # DUNCS vs SparseNet, shared dataset",
                  "python train_single_model.py      # standalone trainer",
                  "",
                  "Output goes under data/:",
                  "    • data/datasets/   — cached if save_dataset: true",
                  "    • data/weights/    — model state_dicts",
                  "    • data/simulations/results/  — score logs + plots"
              ])

    _add_card(s, Inches(6.75), Inches(1.75), Inches(6.2), Inches(5.3),
              title="Key files for the DoA flow",
              bullets=[
                  "main.py                              — entry, picks YAML config",
                  "run_simulation.py                    — SimulationRunner.run()",
                  "src/config/DUNCS.yaml                — all hyperparameters",
                  "src/config/simulation_config.py      — OmegaConf dataclasses",
                  "src/system_model.py                  — array, steering vectors",
                  "src/signal_creation.py               — Samples (generates x[t])",
                  "src/data_handler.py                  — Dataset + DataLoader",
                  "src/models_pack/sparse_cov_admm_unfold.py  — the DUNCS model",
                  "src/methods_pack/esprit.py           — subspace estimator",
                  "src/metrics/criterions.py            — RMSPELoss",
                  "src/metrics/crb.py                   — Cramér–Rao Bound",
                  "src/training.py                      — train() loop",
                  "src/evaluation.py                    — evaluate() loop"
              ], accent=COL_OK)


# ============================================================
# main
# ============================================================

def main():
    prs = _new_pres()
    slide_title(prs)
    slide_abstract(prs)
    slide_pipeline(prs)
    slide_system_model(prs)
    slide_signal_creation(prs)
    slide_forward_overview(prs)
    slide_admm_projection(prs)
    slide_admm_iterations(prs)
    slide_subspace_loss(prs)
    slide_training_loop(prs)
    slide_evaluation(prs)
    slide_how_to_run(prs)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT_PATH)
    print(f"wrote {OUT_PATH}  ({OUT_PATH.stat().st_size//1024} KB, {len(prs.slides)} slides)")


if __name__ == "__main__":
    main()
