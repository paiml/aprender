# impl receipt — PMAT-2730 (#2730)

| Claim | Evidence |
|-------|----------|
| Empty set is FAIL/red, failed 1/total 1 | `brick_verdict` in `crates/apr-cli/src/commands/cbtop_get_cpu_memory.rs`; test `test_brick_verdict_empty_set_is_red` |
| Both producers use it | `generate_headless_report_simulated` (same file) and `build_and_output_report` (`cbtop_measure_batch.rs`, cuda-gated; `cargo check -p apr-cli --lib --features cuda` rc=0) |
| `--ci` fails on it without explicit thresholds | `test_empty_brick_report_is_red_and_fails_ci` |
| Non-empty unchanged | `test_brick_verdict_counts_each_brick` (epsilon edge 1+1e-12 passes; 0.5/1.5 is 1/1 red) |
| Existing test asserted the defect | `test_headless_report_simulated_no_bricks` said `"PASS" // all pass vacuously`; now FAIL/red |
| Mutant | restore `.all()` → 3 tests RED (231 pass / 3 fail) |
| Suite | `cargo test -p apr-cli --lib cbtop` 234/234 on lambda x86; `cargo clippy -p apr-cli --lib --tests` clean |

Not run: the live `apr cbtop --ci` repro on a GPU (cuda clippy is blocked by pre-existing
`aprender-serve` `borrow_deref_ref` errors in `cuda/executor/residual.rs`, not touched here).
