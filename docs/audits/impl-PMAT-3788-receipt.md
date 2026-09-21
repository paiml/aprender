# impl receipt — PMAT-3788 (#3788)

## Identity
- ticket: PMAT-3788 (GitHub #3788), kind: code, 0.69.1 (cop triage)
- branch: `PMAT-3788-eval-diagnostics-no-wallclock`, cut from origin/main `a9502d992`
- owner: aprender-fc; routed by cop aprender-04 ("Owner: aprender-fc, after #3773")
- the issue's root cause and the cop's triage are quoted verbatim in `docs/roadmaps/entries/PMAT-3788.yaml` `notes:`. The triage supersedes the issue's own fix direction (a 60 s timeout): a larger wall-clock number is still a wall-clock assertion.

## What was wrong, measured
- Every test in `execute_python_test_diagnostics_tests` gave its python3 child a 5 s (or 10 s) wall-clock deadline, and asserted on exit codes that a killed child cannot have. **Reproduced deterministically, with no load:** a `python3` shim on PATH that sleeps 6 s before exec-ing the real interpreter (a slow host by construction) turns origin/main's tests RED: 3 failed / 2 passed, `assertion_failure_reports_nonzero_and_traceback` `left: None right: Some(1)`. That is the #3788 failure text verbatim.
- A defect found in reading the code: the "drain" was one `read` of stderr AFTER the child exited. A child writing more than the 64 KiB pipe buffer blocks on the write and never exits, so only the deadline stood between that bug and a hang. Dropping the clock without fixing the drain would have traded a flake for a hang. FALSIFY-HEH-002's 10 KB program could not see this.

## What landed, criterion by criterion
| criterion | where | evidence |
|---|---|---|
| 1. no test passes a wall-clock budget | `ExecBudget` trait. `execute_python_with_budget(program, &mut dyn ExecBudget)`. Production keeps `WallClockBudget` behind the unchanged `execute_python_test_with_diagnostics(program, secs)`, so `run_humaneval_inference` and `execute_python_test` are unchanged. The behaviour tests run the child to its own exit (`NeverExpires`). FALSIFY-HEH-001..004 keep their names | with the 6 s-startup shim, the new tests give 9 passed / 0 failed (12.2 s, which proves the shim engaged; 0.5 s without it). The same shim on origin/main gives 3 FAILED, as above |
| 2. timeout path deterministic | the poll loop is `wait_within_budget(&mut impl PollChild, budget, interval)`. `the_poll_loop_kills_on_the_poll_its_budget_expires` drives it with a fake child that never exits, under `ExpiresAfterPolls(3)`: killed on exactly poll 4, `timed_out`. It needs no python3, so it runs in the CI container, where every python test early-returns. The fake panics past 1000 polls, so a loop that ignores its budget fails on a count instead of hanging. Real-process paths: `an_expired_budget_kills_the_child_and_reports_timed_out` (a never-exiting program) and `the_production_budget_is_a_wall_clock_deadline` (`secs = 0`, no elapsed time asserted) | 9/9 pass |
| 3. the regression the deadline stood in for, caught by a count | stderr is read to EOF on its own thread while the child runs (`drain_capped`, keeps the first 64 KiB). `stderr_drain_consumes_everything_and_keeps_the_cap`: a 1 MiB stream is consumed in full (1048576 bytes) and 65536 are kept. `verbose_stderr_does_not_deadlock_on_success` now writes 256 KiB (4× the pipe) under a budget that never expires | mutants below |
| 4. mutants | `scratchpad/mut3788.sh` applies each mutant, runs the module and restores | **5/5 RED**: drain stops after one read → `stderr_drain_consumes_everything…`, `verbose_stderr…`, `assertion_failure…`. Loop does not report the kill → `the_poll_loop…` plus both real-process timeout tests. Result drops `timed_out` → both real-process timeout tests. Exit code not propagated → HEH-001/003/004. Budget never consulted → `the_poll_loop…` RED on the poll count (`polled 1001 times`), fast. Under that last mutant the two real-python timeout tests cannot return (their child never exits and nothing kills it), so that run is filtered to the fake-child test. The first draft had no fake child: this mutant HUNG the suite past a 10-minute tool timeout, and that is why the loop was extracted |
| 5. contract | `contracts/apr-eval-humaneval-harness-invariant-v1.yaml` v1.1.0 → v1.2.0. PO-HEH-002 is restated as a real drain; new PO-HEH-004 (timeout is an injected budget); FALSIFY-HEH-002 restated at 256 KiB; new FALSIFY-HEH-005 (drain count), 006 (fake-child poll loop), 007 (real-process expired budget) | `pv validate`: 0 errors (2 pre-existing SCHEMA-012 kani `bound` warnings in sections untouched). `check_contract_test_binding.sh` PASS, baseline did not grow |

## Verification (re-run by the orchestrator on lambda-vector, CARGO_TARGET_DIR per worktree)
| command | result |
|---|---|
| `cargo test -p apr-cli --lib` | 7304 passed, 0 failed, 12 ignored |
| `cargo test -p aprender-contracts --lib` | 1689 passed |
| `cargo clippy -p apr-cli --lib -- -D warnings` | clean. `--profile=test` flags only `commands/nf4_classifier.rs` (2, pre-existing, untouched); `inference.rs` and its test module are clean |
| `check_complexity_ratchet.sh` | PASS, none new, none grown |
| `scripts/guard_tree.sh --no-cargo` | 81 checks, 0 failed |
| `check_roadmap_sorted.sh` · `pmat work validate` | PASS · passed |
| `cargo fmt --all -- --check` · `cargo deny check advisories` | clean · advisories ok |

## Gaps, stated
- `drain_capped` joins its thread after the child exits or is killed. If the python program spawns a grandchild that inherits stderr and outlives it, the join waits on the grandchild; the old single `read` had the same exposure. HumanEval programs do not spawn.
- The python-backed tests still early-return when python3 is absent (the CI container), as before. The two python-free tests (drain count, poll loop) are what CI actually exercises.

## Routing
All phases ran direct: one function and its test module in one file, plus the contract.

## Quorum
- Pending. The agy pool is empty for every non-author family (gemini 429 until ~2026-09-23 04:30Z; gpt-oss 429 until ~02:13Z, measured on the PMAT-3749 rounds at 22:16–22:22Z). A seat-fill is requested from the cop (never park).

verdict: PARTIAL — gates green, quorum pending (agy quota)
