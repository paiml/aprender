# PMAT-3598 (#3606) implementation receipt: adopted, brought current, and #4105 folded in

#3606 was open, DIRTY and labelled `needs-owner`. The cop (aprender-cf, 2026-09-24) ruled that aprender-30
adopts it: bring it current with main, fold in #4105 (the falsifier through the real call sites), and run one
quorum round over the combined diff. The original row's claims (StageTimings, absent ≠ zero, the books close
exactly) are #3606's and are unchanged here. This receipt covers what the adoption added.

## 1. Brought current by MERGE, not rebase (c0ee0e5da)
A rebase would have force-pushed over a published head, so this is a merge. Main had moved under #3606 through
three release batches. Every resolution keeps BOTH features:

| main brought | #3606 had | resolution |
|---|---|---|
| #3718 `usage` on RunResult / InferenceOutput | `stages` on the same structs | both fields; RunResult's `Default` gains `usage` |
| `run_gguf_inference` returns `(InferenceResult, RunReport)` | `stages` in the InferenceResult | main's return, with `stages` set |
| #3602 `backend: {requested, ran, fell_back}` JSON object | wrote `backend: "<path>"` over the same key | #3602's object KEPT; the stage path goes inside it as `backend.path` |
| #3604 F2 receipt cache (`f2_validate_qwen35_receipted`, `F2Verdict`) | `stages` threaded into `f2_validate_qwen35` | the whole receipted guard is timed as `validate`; `stages` is threaded through the receipted wrapper; main's `F2Verdict` returns |
| `pub mod run_report` | `pub mod stage_timings` | both |
| new tests building `RunResult` literally | `..RunResult::default()` style | main's new tests take `..RunResult::default()`; #3606's pass `build_final_json`'s new `accel_forced` |

`cargo check --all-targets` is clean for aprender-serve (with and without `cuda`) and apr-cli, with one exception.
`tests/driver_cuda_gguf.rs` fails under `cuda`, but it is byte-identical to main and already broken there
(`query_pre_attn_scalar`, `post_*_norm_weight`).

## 2. #4105: the falsifier through the call sites that ship
`crates/aprender-serve/src/infer/stage_timings_callsite_tests.rs` runs `run_inference` on a REAL GGUF once per
planted stage, against a clean baseline. Rule: the planted field must CONTAIN the plant (≥ `PLANT_MS` = 1000)
and move by ≥ `PLANT_MS − TOL_MS`; every other field, `unattributed_ms` included, may move by at most
`TOL_MS` = 300. The tolerance is sized from a measurement: `load_ms` alone varied by 113.7 ms between two clean
runs on the CUDA leg (a 400 MB mmap against the page cache), which failed an earlier 100 ms tolerance on noise.

| leg | stages | where | result (lambda, RTX 4090 sm_89) |
|---|---|---|---|
| CPU, generated 256-aligned Q4_K llama fixture | clean control (stable, measures `load` only); `load` moves load only; h2d/validate/validate_ref/validate_probe/prefill/decode land NOWHERE (no field moves, wall does not grow) | CI (`cargo test -p aprender-serve --lib`) | 15/15 stage_timings tests pass |
| CUDA dense, qwen2.5-coder-0.5b q4_k_m, `--release` | load, h2d, validate, prefill, decode; every run must stay `backend == "cuda"` | `#[ignore]`, GPU runner, `APR_STAGE_CUDA_GGUF` | PASS twice (30.6 s, 29.9 s) under gpu-q |
| CUDA qwen35, Qwen3.5-0.8B Q4_K_M, `--release` | + validate_ref, validate_probe (inside validate); fresh F2 guard in a private receipt dir | `#[ignore]`, `APR_STAGE_CUDA_QWEN35_GGUF` | PASS (27.9 s) under gpu-q |

Measured caveats:
- **FP8 prefill:** with the default FP8 prefill, the F2 guard REJECTS the GPU for qwen2.5-coder-0.5b on
  sm_89 (#3602/#3483). The run falls back to CPU, and the leg failed loudly (`backend "cpu"`), as designed. The
  dense leg now sets `FP8_PREFILL=0` unless the caller set it. Same stage wiring, FP16 prefill.
- **Debug builds:** the guard's CPU reference takes 43 s in debug, which is noise far above `TOL_MS`, so the GPU
  legs are run from `--release` (stated in the module doc).

## 3. MUST-RED: misattribution turns the test red (both planted, both KILLED)
- **CPU, the historical bug.** The `load` sleep was moved before `load_start` (outside the window). Result:
  `cpu_a_load_plant_moves_load_and_only_load` FAILED with "load_ms is 0.7 ms, less than the plant … unattributed_ms
  moved 253.8 ms". Restored → green.
- **GPU, plant attributed to its NEIGHBOUR.** `split_generate_ms` was passed `0.0` for the prefill plant (the
  quorum-lane bug from #3606's round 1). Result: `cuda_dense_each_plant_moves_its_own_stage` FAILED with
  "plant `prefill:1000`: prefill_ms is 11.5 ms, less than the 1000 ms plant … decode_ms moved 1000.2 ms". That was
  the only failure line. Restored → green.
- **The checker's own case table.** A plant in a neighbour, in the residual, or slept on a path with no site is
  RED; a validate half may move only its parent.

## 4. done_when 2: the wall-clock boundary is stated
`wall_ms` is the engine's clock. It starts just before the load stage in `run_gguf_inference`, and the stages
close against it exactly. `inference_time_ms` is the CLI's clock and also covers resolving the model and
tokenization. `apr run --json` now emits `outside_wall_ms = inference_time_ms − wall_ms`, which is `null`
(never 0) when the engine never closed its books. Test: `run_json_states_the_gap_between_the_cli_clock_and_the_engine_clock`.

## 5. Checks
- aprender-serve `infer::` 757 passed (after the merge); `stage_timings` 15 passed.
- apr-cli `commands::run` 231 passed.
- `cargo fmt --all --check` clean.
- `clippy --all-targets --no-deps -D warnings` is clean on every file this PR touches. The 11 files it still
  flags are byte-identical to main.
- Binaries measured, sha256 prefixes: release test binary 455917122af39789; GPU mutant 43491a88f041edfa.

## Not done
- A `load` plant on the CPU **qwen35** path is covered by the same `run_gguf_inference` site as dense, and is not
  separately fixtured: no small generated Qwen3.5 GGUF exists in-tree.
- The CUDA legs are not wired into CI. They are `#[ignore]` with the env var named, per #4105's "gated on a GPU
  runner".
