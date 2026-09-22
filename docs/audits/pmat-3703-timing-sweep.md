# PMAT-3703: the timing-assertion sweep, every remaining hit listed with its reason

This is the listing #3703 done_when 2 asks for ("each hit fixed or listed with its reason"), as the issue author scoped it on 2026-09-21: the slow-host `elapsed < N` class is the named remainder in #3705. It is a **snapshot** of the tree at `8c08dd8dd`, the head the quorum judged. Line numbers drift; regenerate it with the commands below. What is *enforced* is `scripts/check_no_timing_in_required.sh` PART 4 and its ledger `scripts/wallclock_assert_baseline.txt` (101 hits, 32 texts), not this file.

Scope: `git ls-files crates/*/src/*.rs`, the `--lib` surface that every required lane and the clean-room B2 gate run. Comment lines are excluded.

| Class | Where | Count |
|---|---|---|
| Integer duration asserted `> 0` / `>= 1` / `!= 0` (the #3703 shape) | 3 FIXED here + 101 in the PART 4 ledger with reasons | 104 |
| An upper bound on a duration (`elapsed < N`), the slow-host twin | §1 below, reason `#3705` | 58 |
| Any other lower bound on a duration (`elapsed >= N`, `> Duration::ZERO`, `as_nanos() > 0`, `> 0.0`) | §2 below, each with a reason | 45 |
| `monotonic_ns()` read at epoch initialization | FIXED here (inference_monitor_tests.rs) | 1 |

## Regenerate

```bash
UB='assert[a-z_]*!\(.*(\.elapsed\(\)|\belapsed[a-z_]*\b|\.as_millis\(\)|\.as_micros\(\)|\.as_nanos\(\)|\.as_secs(_f64|_f32)?\(\)|\b[a-z_]*(_ms|_us|_ns|_secs)\b)[[:space:]]*<=?[[:space:]]*[0-9A-Za-z(]'
LB='assert[a-z_]*!\(.*(\.elapsed\(\)|\belapsed[a-z_]*\b|\.as_millis\(\)|\.as_micros\(\)|\.as_nanos\(\)|\.as_secs(_f64|_f32)?\(\))[[:space:]]*>=?[[:space:]]*[0-9A-Za-z(]'
WRE=$(sed -n "s/^WALLCLOCK_RE='\(.*\)'$/\1/p" scripts/check_no_timing_in_required.sh)   # PART 4's own pattern
git ls-files -z -- 'crates/*/src/*.rs' | xargs -0 grep -nHE "$UB" | grep -vE '^[^:]+:[0-9]+:[[:space:]]*//'                       # §1
git ls-files -z -- 'crates/*/src/*.rs' | xargs -0 grep -nHE "$LB" | grep -vE '^[^:]+:[0-9]+:[[:space:]]*//' | grep -vE "$WRE"   # §2
```

## §1: upper bounds, the slow-host class (58). Reason for every line: `#3705`

Each holds on a fast idle host and can fail on a slow or loaded one. Many are not wall-clock at all (a parsed value, a constructed Duration within a tolerance, a configured limit). Classifying them one by one is #3705's done_when, not this row's.

| # | Site | Assertion |
|---|---|---|
| 1 | `crates/aprender-cbtop/src/incremental_snapshot/tests.rs:159` | `assert!(snapshot.timestamp_ns >= 1200 && snapshot.timestamp_ns <= 1700);` |
| 2 | `crates/aprender-cgp/src/analysis/compete.rs:171` | `assert!(result.wall_time_ms < 1000.0);` |
| 3 | `crates/aprender-cgp/src/doctor.rs:406` | `assert!(elapsed.as_secs() < 2, "doctor checks took {:?}", elapsed);` |
| 4 | `crates/aprender-compute/src/brick/tests/phases/phase11_profiling.rs:26` | `assert!(avg_ns < 50.0, "cpu_cycles() overhead should be < 50ns, got {:.1}ns", avg_ns);` |
| 5 | `crates/aprender-compute/src/brick/tests/phases/phase11_profiling.rs:104` | `assert!(avg_ns < 20.0, "cached_nanos() overhead should be < 20ns, got {:.1}ns", avg_ns);` |
| 6 | `crates/aprender-core/src/audio/mod.rs:171` | `assert!(audio.duration_ms < 1);` |
| 7 | `crates/aprender-core/src/automl/tuner_tests.rs:47` | `assert!(result.elapsed.as_secs() <= 3);` |
| 8 | `crates/aprender-core/src/cache/tests_eviction.rs:326` | `assert!(age.as_secs() < 1);` |
| 9 | `crates/aprender-core/src/citl/metrics_tests_tracker.rs:289` | `assert!(duration.as_secs() < 10);` |
| 10 | `crates/aprender-core/src/demo/demo_tests.rs:75` | `assert!(metrics.load_time_ms < 5000);` |
| 11 | `crates/aprender-core/src/demo/demo_tests.rs:92` | `assert!(metrics.first_token_ms < 2000);` |
| 12 | `crates/aprender-core/src/qa/robustness_tests.rs:123` | `assert!(start.elapsed().as_millis() < 1000);` |
| 13 | `crates/aprender-core/src/wasm/wasm_tests.rs:43` | `assert!(estimated_load_ms < 500);` |
| 14 | `crates/aprender-orchestrate/src/falsification/invariants_tests.rs:102` | `assert!(result.duration_ms < 60000); // Less than 1 minute` |
| 15 | `crates/aprender-orchestrate/src/oracle/cookbook/recipes.rs:1011` | `assert!(overlap_ms < chunk_size_ms);` |
| 16 | `crates/aprender-orchestrate/src/serve/banco/state_tests.rs:47` | `assert!(health.uptime_secs < 2);` |
| 17 | `crates/aprender-present-core/src/animation.rs:1405` | `assert!(eased.elapsed <= eased.duration);` |
| 18 | `crates/aprender-present-terminal/src/tools/bench.rs:1141` | `assert!(stats.max_us <= 199);` |
| 19 | `crates/aprender-profile/src/analysis/anti_pattern.rs:600` | `assert!(*interval_ms < 10, "Interval should be < 10ms, got {}", interval_ms);` |
| 20 | `crates/aprender-profile/src/brick_tracer.rs:965` | `assert!(result.duration_us < 1000000); // Should be fast` |
| 21 | `crates/aprender-profile/src/metrics/histogram.rs:555` | `assert!(h.created_at().elapsed().as_secs() < 1);` |
| 22 | `crates/aprender-qa-runner/src/dimensional_check_tests.rs:101` | `assert!(result.duration_ms < 5000, "should complete quickly");` |
| 23 | `crates/aprender-serve/src/api/tests/gpu_batch_02.rs:293` | `assert!(low_latency.window_ms < default_config.window_ms);` |
| 24 | `crates/aprender-serve/src/api/tests/imp_137a.rs:446` | `assert!(elapsed < 10.0, "IMP-140b: Elapsed should be small (< 10s)");` |
| 25 | `crates/aprender-serve/src/api/tests/partial_batch.rs:164` | `assert!(stats.avg_wait_ms < 50.0);` |
| 26 | `crates/aprender-serve/src/api/tests/tests_19.rs:34` | `assert!(config.window_ms <= 10); // Low latency = short window` |
| 27 | `crates/aprender-serve/src/api/tests/tests_23.rs:24` | `assert!(config.window_ms < 100);` |
| 28 | `crates/aprender-serve/src/api/tests/tests_25.rs:97` | `assert!(result.avg_latency_ms < 10.0);` |
| 29 | `crates/aprender-serve/src/bench/statistics_measurement_protocol.rs:110` | `assert!(stats.p50.as_millis() >= 49 && stats.p50.as_millis() <= 51);` |
| 30 | `crates/aprender-serve/src/bench/statistics_measurement_protocol.rs:112` | `assert!(stats.p90.as_millis() >= 89 && stats.p90.as_millis() <= 91);` |
| 31 | `crates/aprender-serve/src/bench/statistics_measurement_protocol.rs:114` | `assert!(stats.p99.as_millis() >= 98 && stats.p99.as_millis() <= 100);` |
| 32 | `crates/aprender-serve/src/bench/tests_distributed_bench.rs:30` | `assert!(gather_1kb.latency_us < reduce_1kb.latency_us);` |
| 33 | `crates/aprender-serve/src/bench/tests_distributed_bench_02.rs:208` | `assert!(gather_1kb.latency_us < reduce_1kb.latency_us);` |
| 34 | `crates/aprender-serve/src/bench/tests_dynamic_sampler.rs:185` | `assert!(metrics.median_ms > 11.0 && metrics.median_ms < 13.0);` |
| 35 | `crates/aprender-serve/src/bench/tests_dynamic_sampler.rs:187` | `assert!(metrics.std_dev_ms < 5.0);` |
| 36 | `crates/aprender-serve/src/bench/tests_dynamic_sampler.rs:425` | `assert!(result.max_hol_blocking_ms < 500.0);` |
| 37 | `crates/aprender-serve/src/brick/profiler_basic_profiling.rs:41` | `assert!(stats.min_us <= stats.avg_us);` |
| 38 | `crates/aprender-serve/src/brick/profiler_basic_profiling.rs:42` | `assert!(stats.avg_us <= stats.max_us);` |
| 39 | `crates/aprender-serve/src/brick/tests_f083_timing_f084.rs:12` | `assert!(elapsed < 500_000, "Timing should not drift too much");` |
| 40 | `crates/aprender-serve/src/gguf/batch_scheduler_tests.rs:180` | `assert!(config.timeout_ms < 50); // Shorter than default` |
| 41 | `crates/aprender-serve/src/gguf/inference_types_tests_dispatch_metrics.rs:144` | `assert!(elapsed < 1.0);` |
| 42 | `crates/aprender-serve/src/gguf/tests/parity035a_chunked.rs:215` | `assert!(stats.ttft_ms < 500.0, "TTFT should be < 500ms");` |
| 43 | `crates/aprender-serve/src/gpu/tests/quantized_dot_matvec.rs:342` | `assert!(delay.as_secs() <= 2);` |
| 44 | `crates/aprender-serve/src/scheduler/tests_chunked_prefill.rs:125` | `assert!(wait.as_millis() < 100);` |
| 45 | `crates/aprender-serve/src/serve_state_predict.rs:175` | `assert!(response.latency_ms < 10.0); // Should be sub-10ms` |
| 46 | `crates/aprender-serve/src/serve_state_predict.rs:224` | `assert!(response.total_latency_ms < 10.0);` |
| 47 | `crates/aprender-test-lib/src/audio_quality/silence.rs:123` | `assert!(report.total_silence_secs < f64::EPSILON);` |
| 48 | `crates/aprender-test-lib/src/av_sync/comparison.rs:216` | `assert!(report.max_delta_ms < f64::EPSILON);` |
| 49 | `crates/aprender-test-lib/src/av_sync/detection.rs:377` | `assert!(pair[0].time_secs <= pair[1].time_secs);` |
| 50 | `crates/aprender-test-lib/src/brick/deterministic.rs:1764` | `assert!(duration.as_secs_f64() > 0.99 && duration.as_secs_f64() < 1.01);` |
| 51 | `crates/aprender-test-lib/src/browser.rs:5096` | `assert!(report.timestamp_ms <= after);` |
| 52 | `crates/aprender-test-lib/src/driver.rs:1228` | `assert!(duration.as_secs() < 1);` |
| 53 | `crates/aprender-test-lib/src/tui/compute_block.rs:556` | `assert!(duration.as_micros() < 1_000_000);` |
| 54 | `crates/aprender-test-lib/src/websocket.rs:1098` | `assert!(elapsed < 1000); // Should be very small, just created` |
| 55 | `crates/aprender-train/src/dashboard/wasm/tests.rs:111` | `assert!(loading_time.as_millis() < 1000, "wasm_bench: {loading_time:?}");` |
| 56 | `crates/aprender-train/src/io/load.rs:497` | `assert!(loading_time.as_millis() < 5000, "load_bench: {loading_time:?}");` |
| 57 | `crates/aprender-train/src/monitor/tui/state.rs:739` | `assert!(rem_ms > 5000 && rem_ms < 30000);` |
| 58 | `crates/aprender-zram-core/src/samefill.rs:401` | `assert!(elapsed.as_millis() < 1000);` |

## §2: other lower bounds on a duration (45), each with its reason

### A. a real measurement, but a sleep of at least the bound runs inside the measured span, so a monotonic clock cannot read below it (10)

| Site | Assertion |
|---|---|
| `crates/aprender-core/src/automl/tuner_tests.rs:150` | `assert!(budget.elapsed() > Duration::ZERO);` |
| `crates/aprender-core/src/demo/reliable/tests.rs:113` | `assert!(timer.elapsed() >= Duration::from_millis(20));` |
| `crates/aprender-profile/src/metrics/histogram.rs:652` | `assert!(elapsed > 0.0);` |
| `crates/aprender-test-lib/src/perf/span.rs:368` | `assert!(elapsed > 0);` |
| `crates/aprender-test-lib/src/perf/trace.rs:417` | `assert!(duration.unwrap().as_millis() >= 10);` |
| `crates/aprender-test-lib/src/performance.rs:1481` | `assert!(elapsed >= 10);` |
| `crates/aprender-test-lib/src/pixel_coverage/wasm_demo.rs:1650` | `assert!(elapsed.as_millis() >= 10);` |
| `crates/aprender-test-lib/src/wait.rs:1055` | `assert!(start.elapsed() >= Duration::from_millis(50));` |
| `crates/aprender-train/src/hf_pipeline/trainer/tests.rs:273` | `assert!(state.elapsed().as_millis() >= 4);` |
| `crates/aprender-verify-ml/src/ml/experiment.rs:665` | `assert!(experiment.total_duration.as_millis() >= 10);` |

### B. a tautology (`>= 0`, `>= 0.0`, `>= Duration::ZERO`) that no clock can fail (8)

| Site | Assertion |
|---|---|
| `crates/aprender-core/src/demo/reliable/tests.rs:263` | `assert!(timer.elapsed() >= Duration::ZERO);` |
| `crates/aprender-core/src/demo/reliable/tests.rs:349` | `assert!(timer.elapsed() >= Duration::ZERO);` |
| `crates/aprender-orchestrate/src/oracle/rag/profiling.rs:695` | `assert!(elapsed >= Duration::ZERO);` |
| `crates/aprender-serve/src/api/tests/imp_137a.rs:445` | `assert!(elapsed >= 0.0, "IMP-140b: Elapsed should be >= 0");` |
| `crates/aprender-serve/src/gguf/inference_types_config_default.rs:409` | `assert!(elapsed >= 0.0);` |
| `crates/aprender-serve/src/gguf/inference_types_tests_dispatch_metrics.rs:143` | `assert!(elapsed >= 0.0);` |
| `crates/aprender-train/src/hf_pipeline/trainer/tests.rs:302` | `assert!(eta.as_secs_f32() >= 0.0);` |
| `crates/aprender-train/src/train/trainer/train_loop/tests.rs:86` | `assert!(result.elapsed_secs >= 0.0);` |

### C. not a clock reading: the Duration is constructed, computed (ETA, back-off), or comes from a synthetic clock (8)

| Site | Assertion |
|---|---|
| `crates/aprender-serve/src/bench/statistics_measurement_protocol.rs:110` | `assert!(stats.p50.as_millis() >= 49 && stats.p50.as_millis() <= 51);` |
| `crates/aprender-serve/src/bench/statistics_measurement_protocol.rs:112` | `assert!(stats.p90.as_millis() >= 89 && stats.p90.as_millis() <= 91);` |
| `crates/aprender-serve/src/bench/statistics_measurement_protocol.rs:114` | `assert!(stats.p99.as_millis() >= 98 && stats.p99.as_millis() <= 100);` |
| `crates/aprender-serve/src/gpu/tests/quantized_dot_matvec.rs:331` | `assert!(delay.as_millis() >= 200);` |
| `crates/aprender-test-lib/src/brick/deterministic.rs:1764` | `assert!(duration.as_secs_f64() > 0.99 && duration.as_secs_f64() < 1.01);` |
| `crates/aprender-train/src/ecosystem/batuta/tests.rs:170` | `assert!(adjusted.as_secs() >= 4500);` |
| `crates/aprender-train/src/ecosystem/batuta/tests.rs:180` | `assert!(adjusted.as_secs() > base);` |
| `crates/aprender-train/src/ecosystem/batuta/tests.rs:189` | `assert!(adjusted.as_secs() >= 7200);` |

### D. a nanosecond or float reading over real work. It does not truncate to milliseconds, and reads 0 only if the work finishes inside one tick of the monotonic clock. The closest to that is trueno-ublk adaptive_batch.rs:494, which times two `add_items` calls: tens of nanoseconds of work (17)

| Site | Assertion |
|---|---|
| `crates/aprender-core/src/automl/tuner_tests_callbacks.rs:56` | `assert!(result.elapsed.as_nanos() > 0);` |
| `crates/aprender-core/src/optim/tests/advanced.rs:246` | `assert!(result.elapsed_time.as_nanos() > 0);` |
| `crates/aprender-core/src/optim/tests/advanced.rs:422` | `assert!(result.elapsed_time.as_nanos() > 0);` |
| `crates/aprender-core/src/optim/tests/advanced_coordinate_descent.rs:312` | `assert!(result.elapsed_time.as_nanos() > 0);` |
| `crates/aprender-core/src/optim/tests/advanced_interior_point.rs:141` | `assert!(result.elapsed_time.as_nanos() > 0);` |
| `crates/aprender-core/src/optim/tests/advanced_line_search.rs:394` | `assert!(result.elapsed_time.as_nanos() > 0);` |
| `crates/aprender-core/src/optim/tests/advanced_projected_gradient.rs:207` | `assert!(result.elapsed_time.as_nanos() > 0);` |
| `crates/aprender-core/src/optim/tests/advanced_projected_gradient.rs:393` | `assert!(result.elapsed_time.as_nanos() > 0);` |
| `crates/aprender-core/src/optim/tests/core_lbfgs.rs:147` | `assert!(result.elapsed_time.as_nanos() > 0);` |
| `crates/aprender-distribute/src/executor/simd.rs:937` | `assert!(result.duration().as_nanos() > 0);` |
| `crates/aprender-gpu/src/monitor/compute/tests.rs:187` | `assert!(kernel.elapsed_ms > 0.0);` |
| `crates/aprender-serve/src/quantize/tests/tests_25.rs:273` | `assert!(simd_time.as_nanos() > 0);` |
| `crates/aprender-test-lib/src/perf/metrics.rs:507` | `assert!(metrics.duration.as_nanos() > 0);` |
| `crates/aprender-train/src/train/trainer/train_loop/tests.rs:38` | `assert!(result.elapsed_secs > 0.0);` |
| `crates/aprender-zram-core/src/benchmark.rs:247` | `assert!(result.compress_time.as_nanos() > 0);` |
| `crates/aprender-zram-core/src/benchmark.rs:248` | `assert!(result.decompress_time.as_nanos() > 0);` |
| `crates/aprender-zram/bins/trueno-ublk/src/perf/tenx/adaptive_batch.rs:494` | `assert!(duration.as_nanos() > 0);` |

### E. #3705 (a hidden UPPER bound: `remaining >= 999 s` of a 1000 s budget fails if the host stalls over 1 s between on_start and remaining) (1)

| Site | Assertion |
|---|---|
| `crates/aprender-core/src/automl/tuner_tests_callbacks.rs:191` | `assert!(remaining.as_secs() >= 999);` |

### F. `#[ignore]`d, so B2 does not run it (its `elapsed < 500_000` ns upper bound is #3705 material if it is ever re-enabled) (1)

| Site | Assertion |
|---|---|
| `crates/aprender-serve/src/brick/tests_f083_timing_f084.rs:11` | `assert!(elapsed > 50_000, "Timing should measure > 50µs");` |
