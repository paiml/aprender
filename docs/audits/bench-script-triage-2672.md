# Triage before delete — the four dark benchmark scripts (#2672)

**Required artifact, produced BEFORE any deletion.** #2640 taught that all nine
hunks of an 86-line "drift" were real hardenings; deleting first would have
destroyed the record of which. The bar is genuinely lower here — these scripts
fabricate their comparator — but 1,086 lines is enough to hide something, and
`ThermalGuard`-adjacent or discard logic was the specific concern raised.

So: every line classified, then deleted.

## Subjects

| script | lines | referenced by any workflow or the Makefile |
|---|---:|---|
| `scripts/benchmark-2x-ollama.sh` | 481 | none |
| `scripts/benchmark-matrix.sh` | 408 | none |
| `scripts/gpu_2x_benchmark.sh` | 171 | none |
| `scripts/verify-parity.sh` | 26 | none |

`grep -rl` across `.github/workflows/` and `Makefile` returns empty for all
four. All four were last touched 2026-05-07 in one unrelated mass-touch commit.

## Salvage scan

Every line matching `ThermalGuard|thermal|temp|discard|warmup|WARMUP|cooldown|stddev|median|percentile`:

| script | hits | what they are |
|---|---:|---|
| `verify-parity.sh` | 0 | — |
| `gpu_2x_benchmark.sh` | 0 | — |
| `benchmark-2x-ollama.sh` | 0 | — |
| `benchmark-matrix.sh` | 8 | **all eight are one `WARMUP_ITERATIONS` counter** (`:36` default, `:91-92` flag, `:125` help text, `:287` banner, `:292` JSON, `:341` the loop) |

**No thermal handling. No discard logic. No percentile or median computation.**
The concern was specific and the answer is measured: there is nothing of that
kind in these files.

## Disposition — all four DISCARD, with the reason per file

| # | script | classification | reason |
|---|---|---|---|
| 1 | `benchmark-2x-ollama.sh` | **discard** | F12. `:27-29` hardcodes `OLLAMA_BASELINE=291`, `OLLAMA_SINGLE=120`, `OLLAMA_CPU=15` and **never invokes ollama**. A ratio against a frozen literal of unknown date, host and build. |
| 2 | `benchmark-matrix.sh` | **discard** | F12. `:396` emits the same three literals into its JSON as `ollama_baselines`. Its only statistic (`:179`) is `sqrt(sq_sum / n)` — a **population** stddev over n=3, which understates dispersion; and `apr bench` already has `--warmup`/`--iterations` (`bench.rs:11-12,102`), so the sole salvageable line is redundant. |
| 3 | `gpu_2x_benchmark.sh` | **discard** | Structurally dead. `:15` shells into `REALIZAR_DIR=../realizar`, a sibling checkout APR-MONO consolidated into `crates/aprender-serve`. It cannot run. |
| 4 | `verify-parity.sh` | **discard** | Benchmarks nothing — 26 lines wrapping `cargo run --example qa_run -- --matrix`. Its **name** is the highest-value trap in the tree for anyone implementing #2667. |

**Zero `hardening-port`. Four `drift-discard`.** The inverse of #2640's
finding, and stated as such so the comparison is honest rather than rhetorical:
there the copies had each gained a real measured hardening, and here the
"measurements" have nothing behind them.

## What replaces them

Not a rewrite of these files. `apr bench --json` (#2668) now emits raw
`samples_ms`, `compute_class`, model `sha256` and a provenance block — the
inputs a correct statistic needs — and `scripts/lib/bench_receipt.py` (#2669)
validates them, including a constant-sample-vector check that would have
rejected `OLLAMA_BASELINE=291` had it ever been written into a receipt.

## Ratchet

`scripts/check_no_fabricated_baselines.sh` bans the construct rather than the
files, so the pattern cannot return under a new name.
