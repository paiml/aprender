# Realizar Benchmarking QA Checklist

**Version:** 0.2.1
**Date:** _______
**Tester:** _______
**Target:** `realizar` CLI benchmarking commands
**Platform:** _______

## Overview

This QA checklist validates the benchmarking functionality of Realizar, including:
- CLI `bench` and `viz` commands
- Criterion.rs integration
- trueno-viz visualization
- Statistical analysis modules
- Performance targets compliance

## Status Legend

- `[ ]` = Not tested
- `[x]` = Passed
- `[!]` = Failed (document issue in Notes)
- `[~]` = Partial/needs investigation

---

## 1. CLI Bench Command (10 points)

| # | Test Case | Expected Result | Status | Notes |
|---|-----------|-----------------|--------|-------|
| 1.1 | `realizar bench --list` | Lists 6 benchmark suites with descriptions | [ ] | |
| 1.2 | `realizar bench` (no args) | Runs all benchmarks via `cargo bench` | [ ] | |
| 1.3 | `realizar bench tensor_ops` | Runs only tensor_ops benchmark suite | [ ] | |
| 1.4 | `realizar bench cache` | Runs only cache benchmark suite | [ ] | |
| 1.5 | `realizar bench inference` | Runs only inference benchmark suite | [ ] | |
| 1.6 | `realizar bench tokenizer` | Runs only tokenizer benchmark suite | [ ] | |
| 1.7 | `realizar bench quantize` | Runs only quantize benchmark suite | [ ] | |
| 1.8 | `realizar bench lambda` | Runs only lambda benchmark suite | [ ] | Requires `lambda` feature |
| 1.9 | `realizar bench invalid_suite` | Error: "Unknown benchmark suite 'invalid_suite'" | [ ] | Lists valid suites |
| 1.10 | `realizar bench -l` | Same as `--list` (short flag works) | [ ] | |

---

## 2. CLI Viz Command (10 points)

| # | Test Case | Expected Result | Status | Notes |
|---|-----------|-----------------|--------|-------|
| 2.1 | `realizar viz` | Displays visualization demo with 100 samples | [ ] | |
| 2.2 | `realizar viz --samples 50` | Uses 50 samples for visualization | [ ] | |
| 2.3 | `realizar viz -s 500` | Uses 500 samples (short flag) | [ ] | |
| 2.4 | `realizar viz --color` | Enables ANSI color output | [ ] | |
| 2.5 | `realizar viz -c` | Same as `--color` (short flag) | [ ] | |
| 2.6 | `realizar viz --samples 10 --color` | Combined flags work | [ ] | |
| 2.7 | Viz output contains sparkline section | Shows Unicode bar characters (▁▂▃▄▅▆▇█) | [ ] | |
| 2.8 | Viz output contains histogram section | Shows ASCII histogram with bars | [ ] | |
| 2.9 | Viz output contains statistics | Shows mean, std_dev, p50, p95, p99 | [ ] | |
| 2.10 | Viz output contains comparison table | Shows benchmark comparison with trends | [ ] | |

---

## 3. Criterion.rs Integration (10 points)

| # | Test Case | Expected Result | Status | Notes |
|---|-----------|-----------------|--------|-------|
| 3.1 | `cargo bench --bench tensor_ops` | Criterion runs with warm-up phase | [ ] | |
| 3.2 | Benchmark output shows confidence intervals | Format: `[lower bound, estimate, upper bound]` | [ ] | |
| 3.3 | Benchmark output shows sample count | "Collecting 100 samples" message | [ ] | |
| 3.4 | Benchmark detects regressions | "Performance has regressed" when applicable | [ ] | |
| 3.5 | Benchmark detects improvements | "Performance has improved" when applicable | [ ] | |
| 3.6 | `target/criterion/` directory created | Contains benchmark results | [ ] | |
| 3.7 | HTML report generated | `target/criterion/report/index.html` exists | [ ] | |
| 3.8 | Parameterized benchmarks work | tensor_creation/10, /100, /1000, /10000 | [ ] | |
| 3.9 | `cargo bench -- --save-baseline main` | Saves baseline for comparison | [ ] | |
| 3.10 | `cargo bench -- --baseline main` | Compares against saved baseline | [ ] | |

---

## 4. Benchmark Suites - tensor_ops (10 points)

| # | Test Case | Expected Result | Status | Notes |
|---|-----------|-----------------|--------|-------|
| 4.1 | tensor_creation/10 benchmark | Completes in <50ns | [ ] | Target: ~18ns |
| 4.2 | tensor_creation/100 benchmark | Completes in <50ns | [ ] | Target: ~21ns |
| 4.3 | tensor_creation/1000 benchmark | Completes in <100ns | [ ] | Target: ~66ns |
| 4.4 | tensor_creation/10000 benchmark | Completes in <1µs | [ ] | Target: ~643ns |
| 4.5 | tensor_shape benchmark | Completes in <5ns | [ ] | Target: ~0.8ns |
| 4.6 | tensor_ndim benchmark | Completes in <5ns | [ ] | Target: ~0.6ns |
| 4.7 | tensor_size benchmark | Completes in <5ns | [ ] | Target: ~0.6ns |
| 4.8 | Benchmarks use `black_box()` | Prevents compiler optimization | [ ] | Code review |
| 4.9 | No outliers >20% | Stable measurements | [ ] | |
| 4.10 | Benchmarks reproducible | <5% variance between runs | [ ] | |

---

## 5. Benchmark Suites - cache (10 points)

| # | Test Case | Expected Result | Status | Notes |
|---|-----------|-----------------|--------|-------|
| 5.1 | cache_hit benchmark | Completes in <100ns | [ ] | Target: ~39ns |
| 5.2 | cache_miss_with_load benchmark | Completes in <50µs | [ ] | Target: ~15µs |
| 5.3 | cache_eviction/2 benchmark | Completes successfully | [ ] | |
| 5.4 | cache_eviction/5 benchmark | Completes successfully | [ ] | |
| 5.5 | cache_eviction/10 benchmark | Completes successfully | [ ] | |
| 5.6 | cache_concurrent_access benchmark | No deadlocks or panics | [ ] | |
| 5.7 | cache_metrics_access benchmark | Completes in <10ns | [ ] | Target: ~4.5ns |
| 5.8 | cache_key/from_string benchmark | Completes in <20ns | [ ] | |
| 5.9 | cache_key/from_config benchmark | Completes in <100ns | [ ] | |
| 5.10 | hit_rate_calculation benchmark | Completes in <1ns | [ ] | Target: ~407ps |

---

## 6. Visualization Module (10 points)

| # | Test Case | Expected Result | Status | Notes |
|---|-----------|-----------------|--------|-------|
| 6.1 | `render_sparkline(&[1.0, 5.0, 3.0], 10)` | Returns 10-char string with ▁-█ chars | [ ] | |
| 6.2 | `render_sparkline(&[], 10)` | Returns empty string | [ ] | |
| 6.3 | `render_sparkline(&[5.0; 10], 10)` | All same character (constant input) | [ ] | |
| 6.4 | `render_ascii_histogram(&data, 10, 50)` | Returns 10-line histogram | [ ] | |
| 6.5 | `render_ascii_histogram(&[], 10, 50)` | Returns empty string | [ ] | |
| 6.6 | `BenchmarkData::new("test", vec![1.0, 2.0, 3.0])` | Creates valid data struct | [ ] | |
| 6.7 | `data.stats()` returns correct mean | Mean of [1,2,3,4,5] = 3.0 | [ ] | |
| 6.8 | `data.stats()` returns correct p50 | Median calculated correctly | [ ] | |
| 6.9 | `data.stats()` returns correct p99 | 99th percentile calculated correctly | [ ] | |
| 6.10 | `print_benchmark_results(&data, false)` | Prints formatted report | [ ] | |

---

## 7. Statistical Analysis (10 points)

| # | Test Case | Expected Result | Status | Notes |
|---|-----------|-----------------|--------|-------|
| 7.1 | `mann_whitney_u(&control, &treatment)` | Returns valid U statistic | [ ] | |
| 7.2 | Mann-Whitney handles equal samples | p-value ≈ 1.0 for identical data | [ ] | |
| 7.3 | Mann-Whitney handles ties correctly | Tie correction applied | [ ] | |
| 7.4 | Mann-Whitney effect size (r) valid | -1.0 ≤ r ≤ 1.0 | [ ] | |
| 7.5 | Mann-Whitney p-value valid | 0.0 ≤ p ≤ 1.0 | [ ] | |
| 7.6 | `analyze_with_auto_select()` detects normal | Uses t-test for normal data | [ ] | |
| 7.7 | `analyze_with_auto_select()` detects skewed | Uses Mann-Whitney for skewed data | [ ] | |
| 7.8 | Effect size interpretation | Negligible/Small/Medium/Large | [ ] | |
| 7.9 | Log-transform applied to latency data | Handles log-normal distributions | [ ] | |
| 7.10 | Sample size validation | Rejects n < 5 samples | [ ] | |

---

## 8. Performance Targets (10 points)

| # | Test Case | Expected Result | Status | Notes |
|---|-----------|-----------------|--------|-------|
| 8.1 | 1-token generation | p50 < 50µs | [ ] | Target: ~17.5µs |
| 8.2 | 5-token generation | p50 < 1ms | [ ] | Target: ~504µs |
| 8.3 | 10-token generation | p50 < 2ms | [ ] | Target: ~1.54ms |
| 8.4 | Tensor creation (10 elements) | < 50ns | [ ] | |
| 8.5 | Tensor creation (10K elements) | < 1µs | [ ] | |
| 8.6 | Cache hit | < 100ns | [ ] | |
| 8.7 | Cache miss with load | < 50µs | [ ] | |
| 8.8 | Tokenizer encode (100 chars) | < 1ms | [ ] | |
| 8.9 | Q4_0 dequantize (1K elements) | < 10µs | [ ] | |
| 8.10 | Lambda cold start | < 100ms | [ ] | |

---

## 9. Error Handling (10 points)

| # | Test Case | Expected Result | Status | Notes |
|---|-----------|-----------------|--------|-------|
| 9.1 | `realizar bench nonexistent` | Error with valid suite list | [ ] | |
| 9.2 | `render_histogram_terminal` with empty data | Returns error | [ ] | |
| 9.3 | `render_histogram_ansi` with empty data | Returns error | [ ] | |
| 9.4 | `BenchmarkData::new("", vec![])` stats | Returns default stats (count=0) | [ ] | |
| 9.5 | Benchmark suite missing required feature | Clear error message | [ ] | lambda needs feature |
| 9.6 | `cargo bench` with missing criterion | Compile error, not runtime panic | [ ] | |
| 9.7 | Invalid samples count `--samples 0` | Handles gracefully | [ ] | |
| 9.8 | Negative values in histogram | No panic, handles gracefully | [ ] | |
| 9.9 | NaN values in statistics | No panic, filters or errors | [ ] | |
| 9.10 | Infinity values in sparkline | Handles without panic | [ ] | |

---

## 10. Edge Cases & Integration (10 points)

| # | Test Case | Expected Result | Status | Notes |
|---|-----------|-----------------|--------|-------|
| 10.1 | Very large sample count (1M samples) | Completes without OOM | [ ] | |
| 10.2 | Very small latencies (nanoseconds) | Displays correctly | [ ] | |
| 10.3 | Very large latencies (seconds) | Displays correctly | [ ] | |
| 10.4 | Unicode in benchmark names | Handled correctly | [ ] | |
| 10.5 | Concurrent benchmark runs | No file conflicts | [ ] | |
| 10.6 | CI environment (no TTY) | No ANSI codes when piped | [ ] | |
| 10.7 | `make bench` Makefile target | Runs `cargo bench` | [ ] | |
| 10.8 | `make bench-tensor` Makefile target | Runs tensor_ops only | [ ] | |
| 10.9 | Benchmark results JSON export | Valid JSON in target/criterion | [ ] | |
| 10.10 | Book documentation accurate | CLI help matches docs | [ ] | |

---

## Summary

| Section | Points | Passed | Failed | Pending |
|---------|--------|--------|--------|---------|
| 1. CLI Bench Command | 10 | | | |
| 2. CLI Viz Command | 10 | | | |
| 3. Criterion.rs Integration | 10 | | | |
| 4. Benchmark Suites - tensor_ops | 10 | | | |
| 5. Benchmark Suites - cache | 10 | | | |
| 6. Visualization Module | 10 | | | |
| 7. Statistical Analysis | 10 | | | |
| 8. Performance Targets | 10 | | | |
| 9. Error Handling | 10 | | | |
| 10. Edge Cases & Integration | 10 | | | |
| **TOTAL** | **100** | | | |

**Final Score:** ___/100

---

## Sign-off

| Role | Name | Date | Signature |
|------|------|------|-----------|
| QA Lead | | | |
| Dev Lead | | | |
| Performance Engineer | | | |

---

## Appendix A: Quick Smoke Test Commands

```bash
# 1. Build release binary
cargo build --release --bin realizar

# 2. Test bench command
./target/release/realizar bench --list
./target/release/realizar bench tensor_ops

# 3. Test viz command
./target/release/realizar viz
./target/release/realizar viz --color --samples 50

# 4. Run all benchmarks
cargo bench

# 5. Run specific benchmark
cargo bench --bench cache

# 6. Check Makefile targets
make bench
```

---

## Appendix B: Test Data Generation

```rust
// Generate log-normal latency data (typical for benchmarks)
fn generate_latencies(n: usize, median_us: f64) -> Vec<f64> {
    use std::f64::consts::E;
    let log_median = median_us.ln();
    let log_std = 0.5; // Typical variance

    (0..n).map(|i| {
        let z = (i as f64 / n as f64) * 2.0 - 1.0;
        (log_median + log_std * z).exp()
    }).collect()
}

// Example: 100 samples around 20µs median
let latencies = generate_latencies(100, 20.0);
```

---

## Appendix C: Performance Baseline Reference

| Metric | Baseline (v0.2.1) | Target | Tolerance |
|--------|-------------------|--------|-----------|
| tensor_creation/10 | 18ns | <50ns | ±20% |
| tensor_creation/10K | 643ns | <1µs | ±20% |
| cache_hit | 39ns | <100ns | ±20% |
| cache_miss | 15µs | <50µs | ±20% |
| 1-token gen | 17.5µs | <50µs | ±20% |
| 5-token gen | 504µs | <1ms | ±20% |

---

## Appendix D: Known Issues

| Issue | Severity | Status | Workaround |
|-------|----------|--------|------------|
| | | | |

---

## Appendix E: Test Environment Requirements

### Hardware
- CPU: Multi-core x86_64 or ARM64
- RAM: ≥8GB
- Storage: ≥1GB free (for benchmark artifacts)

### Software
- Rust: 1.75+
- Cargo: Latest stable
- OS: Linux, macOS, or Windows with WSL2

### Configuration
```bash
# Disable CPU frequency scaling for accurate benchmarks
sudo cpupower frequency-set --governor performance

# Close other applications
# Run in isolation (not in VM or container for timing tests)
```

---

## Appendix F: Regression Testing Protocol

1. **Before changes:**
   ```bash
   cargo bench -- --save-baseline before
   ```

2. **After changes:**
   ```bash
   cargo bench -- --baseline before
   ```

3. **Review regressions:**
   - Any `Performance has regressed` with p < 0.05
   - Document in PR with justification if intentional

4. **Acceptance criteria:**
   - No regressions >10% without justification
   - All performance targets met
   - No new panics or errors

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-11-27 | QA Team | Initial checklist |
