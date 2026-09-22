# DUNCS instructions for Codex

## Start here

This is a Python/PyTorch research project for direction-of-arrival (DoA)
estimation with sparse arrays and deep unfolding. The current work centers on
MFOCUSS, DU-MFOCUSS, SubspaceNet-MUSIC, DoAFormer, and matched classical
baselines. DUNCS and SparseNet remain implemented.

Read [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) before continuing prior work. It
maps the architecture, recovered Claude session, current changes, experiments,
and unfinished requests. Treat it as a dated snapshot: check current source,
Git state, logs, and artifacts before relying on its status or numbers.
`tasks.md` is the older task ledger, not a complete account of recent work.

These instructions replace `CLAUDE.md` for Codex. The original Claude files
remain available for historical reference. Keep durable guidance here and
update the context snapshot when completing a substantial continuation task.
Current user instructions take precedence over migrated preferences.

## Working agreements

- State material assumptions, choose a simple implementation, and complete
  authorized work with suitable verification. Ask only when missing information
  materially blocks progress; do not repeatedly seek approval already given.
- Keep edits focused. Match existing style; avoid unrelated cleanup, new
  abstractions, and speculative features. Preserve existing staged, unstaged,
  untracked, and ignored research artifacts.
- Inspect `git status --short --branch`, `git diff`, and `git diff --cached`
  before editing. Do not stage or commit unrelated changes.
- Put tunables, thresholds, and numerical tolerances in named defaults
  (constructor parameters/attributes or module constants), not buried literals.
  Preserve numerical behavior when only moving existing values.
- Use titled, numbered subsections for meaningful algorithm steps and concise
  comments explaining purpose and scientific reasoning. Avoid adding a header
  to every trivial statement.
- Report what actually ran and its outcome. A checkpoint or historical success
  message does not establish that a current experiment passed.

## Scientific and experiment conventions

- Match carrier frequency, measured manifold, angle range, source count,
  snapshots, SNR, data split, and scoring when comparing algorithms. Current
  canonical deck experiments use **150 MHz**, five sensors, eight snapshots,
  and a common **[-70, 70] degree** search grid. Other YAML workflows differ.
- Keep analytic ULA synthesis, synthesis from the measured ULA3 dictionary,
  parametric propagation scenes, external Hof **DataSim** files, and physical
  **Mitvah** field recordings distinct. The flag `--real` currently means the
  external DataSim files; it does not prove physical field provenance.
- Reuse-15 samples separation uniformly from [15, 40] degrees; Reuse-25 from
  [25, 40]. Distinguish source-to-source coherence from echo-to-direct-path
  coherence. A chosen scene assumption is not a universal propagation fact.
- Reset `SteeringVectorGenerator` between different system configurations.
  Preserve the recorded manifold's frequency and azimuth conventions.
- Inspect data/checkpoint provenance. Keep training, validation, and evaluation
  inputs disjoint under the stated protocol. Do not tune on held-out data.
- Distinguish radians/modulo-pi RMSPE training loss from degree error, detection
  error, and MD/FA. Report all-source error and detection/miss metrics together;
  a low detected-only error can conceal missed targets. Standard evaluations
  often supply the true source count, which is not count-estimation accuracy.
- Distinguish ML beamscan, joint ML, and alternating-projection ML. The user's
  original "ML" baseline means beamscan.
- For suspicious metrics, investigate data, conventions, training, checkpoint
  loading, inference, and scoring with controlled comparisons. Seek
  classical-equivalent initialization and measured improvement for unfolding;
  do not claim that model expressiveness or a lower reconstruction residual
  guarantees a lower DoA error.
- Validate a candidate on a bounded run before expensive training, with output
  isolation. `deck/retrain_150.py --smoke` currently overwrites normal checkpoint
  names. Generic training can write checkpoints even with `save_model: false`.

## Presentations

For tasks that change study results, figures, or presentation source, update
the affected explanations and progress/bug history, regenerate all applicable
deck variants, and verify the written files as part of finishing that task.
Keep failed experiments and their measured outcomes traceable. The current
builder is `deck/build_deck.py`; outputs go to `G:/My Drive/DOA_AI`.
The context map lists the nine known variants and environment requirements.
Documentation-only onboarding does not require rerunning studies or decks.

Table highlighting compares RMS and MD at displayed precision: no worse on
either metric and strictly better on at least one. Do not substitute a good
median or detected-only score for the complete comparison.

## Local workflows and skills

- Use the repository root as the working directory. Prefer explicit interpreter
  paths; `.venv/Scripts/python.exe` exists on this Windows checkout. Its verified
  migration-time package limitations are in the context map.
- Read the selected `src/config/*.yaml` before running `main.py`; it can train
  and overwrite artifacts. `main.py` defaults to `mfocuss`.
- The existing focused test file is `tests/test_antenna_pattern.py`; it requires
  pytest and skips as a whole when the external Hof manifold is absent.
- For Hof/DOA1 or Exhaustive MATLAB creation/refactoring, use
  [.agents/skills/matlab-coding-conventions/SKILL.md](.agents/skills/matlab-coding-conventions/SKILL.md),
  or invoke `$matlab-coding-conventions`. Apply its MATLAB style elsewhere when
  explicitly requested. Its external paths and classifier task rotation apply
  only to those repositories, not to DUNCS Python/tasks.md.
- The old `.claude/agents/fable5.md` selects a Claude-specific model. It contains
  no project algorithm guidance; use available Codex delegation for suitable
  independent subtasks rather than treating that model name as Codex config.
