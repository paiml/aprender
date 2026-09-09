---
name: project_pmat_cognitive_gate_decomposition
description: how to decompose a Python function that fails pmat's cognitive-complexity pre-commit gate via pure code motion
metadata:
  type: project
---

`pmat analyze complexity --file <f> --max-cyclomatic N --max-cognitive M` (and the
PMAT-managed `pre-commit` hook that runs the same check on staged files only) counts
comprehensions, generator expressions, ternaries, and `sum(...)`/`any(...)` calls with
a generator argument as complexity contributors *per enclosing function*. A function
that builds one big dict literal where several values are themselves comprehensions/
ternaries/sums racks up cognitive complexity fast even with shallow nesting.

**How to apply:** lift each such expression into its own tiny module-level helper
(`_something(...)` returning the value), then reference the helper by name inside the
dict literal (or via `**helper(...)` spread if the helper returns several keys that
need to land in a specific position — this preserves key ORDER for JSON output, which
matters when a script's stdout/file is compared byte-for-byte elsewhere). Repeat for
early-return / if-elif-else dispatch chains: extract the arms into named helpers and
call them from a slim dispatcher. This reliably drives a function from `Cognitive 35`
down under 10 without changing behavior — verified on `scripts/lib/parity_block.py`
(PMAT-972/#2887): `_executor_side` 35→~5, `_executor_lane` 43→~8, `build` 31→~6,
`main` 27→~2, all via decomposition alone, re-verified with
`pmat analyze complexity ... | grep -c ...` after each step.

The repo's PMAT-managed pre-commit hook (`.git/hooks/pre-commit`, chained from a
`core.hooksPath` wrapper) only runs complexity + SATD + doc + task-id checks on
*staged* source files — it does NOT run repo-specific shell guards like
`check_thresholds_in_matrix.sh` or `check_perf_receipt_fields_have_producers.sh`.
Those are separate CI-level gates; a pre-existing false-positive in one of them
(e.g. a docstring containing a float literal beside a comparison operator, like
"0 / positive == 0.0") does not block a commit and is out of scope for a
complexity-decomposition ticket unless the ticket's own acceptance/gate commands
name that script.
