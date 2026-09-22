---
name: matlab-coding-conventions
description: >-
  Apply Hof/DOA1 and Exhaustive MATLAB conventions when creating or refactoring
  .m functions, wrappers, or scripts: thin top-level orchestration, named local
  steps, and section banners. Also covers their classifierTasks.md workflow.
  Use elsewhere when explicitly asked to apply these conventions; do not
  automatically impose them on DUNCS MATLAB, Python, or DUNCS tasks.md.
---

# MATLAB coding conventions

Converted from `.claude/skills/matlab-coding-conventions/SKILL.md`.
Follow the current user request and the target repository's `AGENTS.md`.
Automatic scope is Hof/DOA1 and Exhaustive, as in the original skill. Apply
these conventions elsewhere only when the user asks for them.

## Thin orchestrator and named steps

Keep a multi-step top-level function short: an ordered list of named,
high-level operations. Put each cohesive operation in a local subfunction
after the main function's `end`.

- Use numbered `%% n) Step` comments for the top-level calls.
- Name step functions for what they do; use the established `Pascal_Case`
  convention with underscores for new MATLAB helpers.
- Introduce each local step function with the section banner below.
- Pass state through explicit arguments; do not use shared globals.
- Extract meaningful operations, not single trivial lines.
- Preserve public function names, ordering, and behavior unless the requested
  refactor requires changing them. Apply the convention surgically.
- Keep task numbers out of source code and comments.

```matlab
function [] = Run_analysis_and_presentation()
    %% 1) Run the analysis
    Run_analysis();

    %% 2) Build the presentation from the updated figures
    Build_presentation();
end

%% ~~~~~~~~~~~~~~~~~~~~ Build_presentation ~~~~~~~~~~~~~~~~~~~~ %%
function [] = Build_presentation()
    % Implementation of this cohesive step.
end
```

For existing scripts, preserve script behavior unless conversion to a function
is part of the task. Match surrounding MATLAB conventions where they differ.

## Hof / Exhaustive workflows

Read [references/hof-exhaustive.md](references/hof-exhaustive.md) only when
working in those codebases or editing `classifierTasks.md` /
`oldClassifierTasks.md`. Their paths and task rotation are specific to those
repositories; DUNCS uses `tasks.md` and its own source layout.
