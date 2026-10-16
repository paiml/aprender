# EG-0 (d) — kind-totality guard: first green and mutation RED

Target: `cargo test -p apr-cli --test cli_commands` (an existing required target; no new job).
Binary tree: f745e7abc + this branch's registry and test edits. Run on the CPU test host, 2026-10-07.

| Run | Contract state | `every_command_declares_one_verdict_per_model_kind` |
|---|---|---|
| green | as committed | ok (17/17 in the target) |
| mutation 1 | `run`/`chat`/`code` embedding verdict `refuse` → `advisory` | FAILED: `run: kind embedding has verdict advisory, not one of ["receipt", "refuse", "no-model"]` |
| mutation 2 | first `kinds:` line deleted (`validate-manifest`) | FAILED: `validate-manifest: no kinds: map` |
| restore | as committed | ok |

`kind_totality_guard_case_table` carries 12 planted must-fail cases (zero commands, no list,
missing map, missing kind, fourth state, third kind, model verb says no-model, refusal without
`use_instead`, `use_instead` naming an unregistered verb, naming a verb that refuses the same
kind, `use_instead` on a receipt, no-model for one kind only) and one must-pass registry.

## Re-run after the complexity split

The guard was split into five functions (max cyclomatic 6) to meet the pre-commit
complexity gate. Mutation 1 re-run on the split code: RED, ``run: kind `embedding` has
verdict `advisory`, not one of ["receipt", "refuse", "no-model"]``; restore GREEN. All 17
tests in the target pass.
