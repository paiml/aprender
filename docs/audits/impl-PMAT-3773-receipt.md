# impl receipt — PMAT-3773 (#3773)

## Identity
- ticket: PMAT-3773 (GitHub #3773), kind: code, release-blocking for 0.69.1
- branch: `PMAT-3773-showcase-no-fabricated-baseline`, cut from origin/main `a9502d992`
- owner: aprender-fc; routed by cop aprender-04
- done_when, the cop's triage, the widening and the scope ruling are quoted verbatim in `docs/roadmaps/entries/PMAT-3773.yaml` `notes:`. The five acceptance criteria below are that fragment's.

## What landed, criterion by criterion
| criterion | where | evidence |
|---|---|---|
| 1. showcase never prints an unmeasured baseline | `crates/apr-cli/src/commands/showcase/`: `run_llama_cpp_bench` makes no PATH probe and returns `UNMEASURED: <reason>`, because the only comparator measurement is the pinned parity lane (one client for both engines, #2696/#3563). `build_comparison` records the reason in `BenchmarkComparison.unmeasured` (serde, exported to JSON) and reports no speedup. `ollama_tps_ttft` reads only the eval counters, and a missing field is UNMEASURED with its name; the 200.0 / 150.0 fallbacks are gone. The no-inference `run_benchmark` refuses by name. `generate_jitter` is deleted | `tests_tests_no_fabricated_baseline.rs`: `llama_cpp_baseline_is_unmeasured_never_a_number`, `an_unmeasured_baseline_is_recorded_and_yields_no_speedup`, `an_unrequested_baseline_is_absent_not_unmeasured`, `ollama_counters_or_unmeasured_never_a_fallback_constant`. Live mutant, re-run on c754ca471: `run_llama_cpp_bench` returning `Ok((35.0, 120.0))` turns `llama_cpp_baseline_is_unmeasured_never_a_number` and `an_unmeasured_baseline_is_recorded_and_yields_no_speedup` RED (165 passed, 2 failed); restored, green |
| 2. pmat_benchmark_matrix | `crates/aprender-serve/examples/pmat_benchmark_matrix.rs`: `llama_bench_tg64(model, ngl)` runs `$LLAMA_BENCH -p 0 -n 64 -ngl N -o json` (exported by `scripts/llama_bin.sh` once it proves the build; per f0's #3740 ruling, no PATH lookup) and reads the tg64 `avg_ts`. Otherwise it returns UNMEASURED with the reason. A cell without a measurement gets no speedup and no verdict. The six literal "verified" baselines are deleted | compiles under `--features cuda` |
| 3. every executed literal competitor throughput in Rust is measured or deleted; the ledger is retired into a ban | All 36 ledgered sites were resolved. **Receipted** (dated `// receipt:` comments): `ch22_vs_llamacpp.rs` (corrected to its own receipts: llama.cpp 431.1 from `bootstrap-llama-cpp-b7746-fair-20260405-180222.json`, apr 351.97 from `bootstrap-realizr-0.8.6-20260406-080315.json`; it had said 285), `ch23_training_bench.rs`, `ch27_switch_unsloth.rs`. **Comparison deleted**: the other examples and benches. `gpu_showcase_benchmark`'s invented Stats became `Option`s, with an INCOMPLETE verdict. **11 further shipped sites that the ledger's shape could not see**, found by the new scanner: `aprender-core` `bench_viz::{grid_builder,rendering}` and `showcase::{mod,profiling}` defaulted an absent Ollama / llama.cpp to 318 / 200 tok/s and judged Point 41 / 2x-Ollama PASS against it; `render_compact` divided by `max(1.0)` and printed apr's own tok/s as the "speedup"; the `showcase_benchmark` example synthesized both competitors; `aprender-serve` `bench_viz_render_profiling_benchmark` did the same. All now print UNMEASURED and judge nothing. `scripts/fabricated_baseline_rust_sites.txt` is deleted, along with its `classify` row in `check_baseline_ratchets.sh` | `test_showcase_runner_check_{2x_ollama,point_41}_unmeasured_does_not_pass`, `test_pmat_verification_unmeasured_comparators_do_not_pass`, `test_unmeasured_comparators_render_no_ratio` (core and serve). Tests that asserted a PASS against the invented default now record a comparator. Live mutant: `check_2x_ollama` with `.map_or(318.0, …)` restored makes `…_unmeasured_does_not_pass` RED |
| 4. published docs dated and receipted, or deleted | `crates/apr-cli/README.md`: "2.9x faster than Ollama", the "vs Ollama" table column and the "(2.6x Ollama)" sample output are removed; comparisons point to BEATS.md. `crates/aprender-serve/README.md`: llama.cpp 256 / Ollama 228 are removed. `docs/BEATS.md`: the llama.cpp figure cites its bootstrap JSON (2026-04-05). The Ollama parity rows, the withdrawn 1.371× claim and its four-row history each name their date and the beat contract that records them (`beat-ollama-decode-throughput-speed-v1.yaml:17-20` holds those rows verbatim) | `readme_contract` 15 passed; `check_no_claim_literals.sh` PASS (ratchet did not grow) |
| 5. guard | `scripts/check_no_fabricated_baselines.sh`, still wired at `.github/workflows/ci.yml` "No fabricated comparator baselines". **Rust** over `git ls-files -co 'crates/*.rs' 'src/*.rs'` (10367 files, floor 1000). Rules: R1, a competitor-named binding to a literal; R2, "default <competitor> baseline"; R3, inside a competitor-named fn, a throughput-named binding with a non-zero literal, or a non-zero `.map_or`/`.unwrap_or` fallback standing in for throughput in a competitor statement. Legal only with a dated `receipt:` within 3 lines. Tests are out of scope by ruling. **Docs** over the root README.md, `crates/*/README.md` and `docs/BEATS.md` (86 files, floor 20). Rules: D1, a competitor named with `N tok/s` or a ratio fastened to its name; D2, a figure under a table column whose header names a competitor (main's "vs Ollama" column). Legal only with a date AND a pointer on the same line. One scanner function per universe, called by the sweep and by the case table | case table: 27 rust + 21 docs rows as fixture files through the shipped scanners, plus the unchanged 16+13 shell and 13 cfg rows. **20 rule mutants, all RED** (R1/R2/R3-bind/R3-fallback dropped, zero-init as claim, undated receipt, receipt window widened, test path excludes all, cfg(test) cut ignored, fallback ignores context, scanner crash, D1 ratio dropped, D2 dropped, table state leaks, D2 needs no number, date-only receipt, pointer-only receipt, bare "llama" as competitor, any ratio as competitor, docs scanner crash). **4 live-tree mutants, all RED**: M1 `let tps = 35.0 + 0.0;` in `run_llama_cpp_bench`; M2 an untracked `examples/zz_probe_3773.rs`; M3 `GIT_DIR=/nonexistent` (universe collapses → floor FAIL, not PASS); M4 "(2.6x Ollama)" restored in the README |

## The control: the old guard on the old tree
On origin/main `a9502d992`, main's own guard prints `ok rust 36 ledgered site(s)` and PASS, while `benchmark_helpers.rs:55` `let tps = 35.0 + generate_jitter() * 1.5;` is live (the #3773 birth site; it is named `tps`, which is invisible to a name-keyed ledger). The new guard run over that same tree fails on **44 Rust sites**: all 36 ledgered ones, the birth site and its TTFT twin, the Ollama 200.0 / 150.0 fallbacks, and the bench_viz / showcase defaults.

## Verification (re-run by the orchestrator on lambda-vector, CARGO_TARGET_DIR per worktree)
| command | result |
|---|---|
| `cargo test -p apr-cli --lib` | 7302 passed, 0 failed, 12 ignored (HEAD c754ca471) |
| `cargo test -p aprender-contracts --lib` | 1689 passed |
| `cargo test -p aprender-core -p aprender-serve --lib` | aprender-core 14282 passed (2 ignored); aprender-serve 15915 passed (59 ignored); 0 failed |
| `cargo test -p aprender-core --test readme_contract` | 15 passed |
| `cargo check -p aprender-core -p aprender-serve --lib --examples`; `cargo check -p aprender-serve --features cuda --example pmat_benchmark_matrix` (all examples/benches were also checked under `--features cuda` in a28e23241) | rc 0 |
| `bash scripts/check_no_fabricated_baselines.sh` | PASS: shell 322, config 99, rust 10367 (0 unreceipted, 4 receipted), docs 86 (0 unreceipted, 9 receipted) |
| `check_complexity_ratchet.sh` | PASS, none new, none grown. Before the helper extraction it had flagged `render_scientific` / `render_profiling_log` (core, NEW), serve `render_profiling_log` 30→42 and `pmat_benchmark_matrix::main` 31→32 |
| `check_no_claim_literals.sh` · `check_baseline_ratchets.sh` (+ `--self-test`) · `check_comparator_one_client.sh --selftest` | PASS · PASS (21 files, 17 ratcheted) · 48 passed |
| `scripts/guard_tree.sh --no-cargo` | 81 checks, 0 failed |
| `cargo fmt --all -- --check` · `cargo deny check advisories` | clean · advisories ok |

## Gaps, stated
- The docs rule is line-scoped. BEATS.md's history paragraphs wrap a figure over lines (`(1.371× median,` / `412.3 vs 300.7 tok/s`); they are dated prose about a withdrawn claim. A paragraph window was rejected because it lets one table row receipt its neighbour. This is recorded under RESIDUAL in the guard.
- The Rust rule is not a parser. A figure built through a const table, a builder or a match arm, or a generic-named literal in a fn whose name does not name the competitor, is RESIDUAL.
- Internal docs and specs that quote competitor figures are a filed follow-up (cop ruling: "Internal docs/specs as a filed follow-up is fine; test fixtures stay"). `crates/aprender-serve/CLAUDE.md` still quotes Ollama/llama.cpp tok/s and is part of that follow-up.
- `showcase_benchmark` still renders SYNTHETIC apr numbers, and now says so on every line it prints. It records no competitor.

## Coordination
- #3740 (aprender-f0) baselines the `which llama-server` probe that this branch removes. Whoever folds second drops that baseline line.

## Routing
All phases ran direct (the orchestrator implemented; no worker subagent). The work was a sweep over sites that the scanner enumerated, and every one was re-verified by compile and test.

## Quorum
- Pending. At 22:16–22:22Z the agy pool was empty for every non-author family: gemini returned 429 "Resets in 30h14m" and gpt-oss-120b-medium returned 429 "Resets in 3h51m" (probes from the PMAT-3749 round, same account). A seat-fill was requested from the cop (never park).

verdict: PARTIAL — gates green, quorum pending (agy quota)
