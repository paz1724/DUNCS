# Hof / Exhaustive conventions retained from the Claude skill

Apply these rules only when the user's task involves Hof/DOA1 or Exhaustive.
They are reference context, not authorization to change another repository.
Read that repository's current `AGENTS.md` (or its maintained `CLAUDE.md` if
no Codex instructions exist) before editing it.

## Code location and naming

- In Hof, edit `DOA1/`, not `DOA/` or `DebugCode/`.
- Exhaustive scheduler changes belong in `C:\GitHub\Exhaustive\Code\`,
  the copy used on the MATLAB path.
- Prefer an additive wrapper to changing an existing entry function when that
  satisfies the task.
- Prefix conventions: `c` for class, `s` for struct, `b` for boolean, `o` for
  object. Use `Pascal_Case` functions and the existing static utility style.
- The original examples are `cArray`, `cEstimator`, `cHofEstimator`, and
  `Run_test_xls_exhaustive_scheduler_HH_and_presentation.m`. These belong to
  those repositories, not DUNCS.

## Classifier task tracking

The live list is `classifierTasks.md`; the archive is
`oldClassifierTasks.md`. Numbering is global and monotonic across both files.
Determine the next number from the current files. The old skill's last known
task was 337 and its next number was 338; those are historical values, not a
current numbering instruction.

After completing a listed task, append these lines directly under its body:

```markdown
Summary: One short line describing what was actually done.
Status: Done
```

Use `Partial`, `Blocked`, or `Skipped` with a brief reason when appropriate.
Preserve the original task description and number.

When the live list reaches 11 tasks, move its first 10 tasks (oldest by
document order) to the end of the archive in the same order. Leave the newest
task in the live file. Never renumber tasks during rotation.
