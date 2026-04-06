# QA Checklist: PyTorch vs Aprender Benchmark Validation

**Document ID:** QA-BENCH-001
**Version:** 1.0.0
**Date:** 2025-11-27
**Status:** Active
**Auditor:** [Red Team Lead]

---

## Purpose

This document provides **100 validation points** for a red team to audit the PyTorch vs Aprender MNIST inference benchmark. The goal is to prove the benchmark is:

- **Not "cooked"** (results not manipulated)
- **Not "fake"** (actual measurements, not fabricated)
- **Not "outrageous"** (results are plausible and explainable)

Each point should be verified independently. Mark as PASS/FAIL with evidence.

---

## Validation Summary

| Category | Points | Pass | Fail | Skip |
|----------|--------|------|------|------|
| Methodology | 15 | | | |
| Statistical Rigor | 15 | | | |
| Code Correctness | 15 | | | |
| Data Integrity | 10 | | | |
| Environment Fairness | 15 | | | |
| Reproducibility | 10 | | | |
| Result Plausibility | 10 | | | |
| Bias Detection | 10 | | | |
| **Total** | **100** | | | |

---

## Category 1: Methodology (15 points)

### 1.1 Benchmark Design

| # | Validation Point | How to Verify | Pass/Fail |
|---|-----------------|---------------|-----------|
| 1 | Warmup iterations are excluded from measurements | Read `WARMUP_ITERATIONS` in both scripts, verify loop structure | |
| 2 | Warmup count is identical (100) in both implementations | `grep -n "WARMUP" mnist_benchmark.py mnist_apr_benchmark.rs` | |
| 3 | Measurement iterations are identical (10,000) | `grep -n "BENCHMARK_ITERATIONS" *.py *.rs` | |
| 4 | Same random seed (42) used in both | `grep -n "SEED\|seed" *.py *.rs` | |
| 5 | Measurement uses high-resolution timer | Python: `time.perf_counter_ns()`, Rust: `Instant::now()` | |

### 1.2 Task Equivalence

| # | Validation Point | How to Verify | Pass/Fail |
|---|-----------------|---------------|-----------|
| 6 | Same model architecture (784 -> 2) | Check `LogisticRegression` in both, verify input/output dims | |
| 7 | Same task: binary classification | Both use `NUM_CLASSES = 2` | |
| 8 | Same input dimensions (784 floats) | `grep -n "INPUT_DIM\|input_dim" *.py *.rs` | |
| 9 | Single sample inference (batch_size=1) | Verify sample shape is (1, 784) in both | |
| 10 | No batching tricks in either implementation | Read inference loop, ensure single sample per call | |

### 1.3 Measurement Scope

| # | Validation Point | How to Verify | Pass/Fail |
|---|-----------------|---------------|-----------|
| 11 | Timer starts immediately before inference | Check `start = ...` placement in both | |
| 12 | Timer stops immediately after inference | Check `end = ...` or `elapsed()` placement | |
| 13 | No I/O operations inside timed section | Verify no print/file/network in timed loop | |
| 14 | No memory allocation logging in timed section | Verify no allocator hooks during measurement | |
| 15 | Model is in eval/inference mode | Python: `model.eval()`, Rust: n/a (always inference) | |

---

## Category 2: Statistical Rigor (15 points)

### 2.1 Sample Size

| # | Validation Point | How to Verify | Pass/Fail |
|---|-----------------|---------------|-----------|
| 16 | 10,000 iterations is statistically sufficient | n > 30 for CLT, 10,000 >> 30 | |
| 17 | Both report same iteration count in output | Check JSON files for `iterations: 10000` | |
| 18 | No early termination in either benchmark | No `break` or conditional exit in measurement loop | |

### 2.2 Statistical Calculations

| # | Validation Point | How to Verify | Pass/Fail |
|---|-----------------|---------------|-----------|
| 19 | Mean calculation is correct | `mean = sum / n`, verify formula in both | |
| 20 | Standard deviation uses n-1 (sample std) | Check `/ (n - 1.0)` in variance calculation | |
| 21 | 95% CI uses correct z-value (1.96) | `grep -n "1.96" *.py *.rs` | |
| 22 | Percentiles calculated correctly | p50 = sorted[n/2], p99 = sorted[0.99*n] | |
| 23 | Throughput = 1_000_000 / mean_us | Verify formula in both implementations | |

### 2.3 Significance Testing

| # | Validation Point | How to Verify | Pass/Fail |
|---|-----------------|---------------|-----------|
| 24 | Welch's t-test used (not Student's t) | Check for unequal variance handling in compare_mnist.py | |
| 25 | p-value < 0.05 for claimed significance | Verify in comparison_summary.json | |
| 26 | Cohen's d calculation is correct | d = (mean1 - mean2) / pooled_std | |
| 27 | Effect size interpretation follows standards | <0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, >0.8 large | |
| 28 | Confidence intervals don't overlap | Aprender CI and PyTorch CI are disjoint | |
| 29 | Results reported with appropriate precision | 2 decimal places for microseconds | |
| 30 | No cherry-picking of runs | Single run, no "best of N" selection | |

---

## Category 3: Code Correctness (15 points)

### 3.1 Python Implementation

| # | Validation Point | How to Verify | Pass/Fail |
|---|-----------------|---------------|-----------|
| 31 | PyTorch model is minimal LogisticRegression | `nn.Linear(784, 2)` only, no hidden layers | |
| 32 | No CUDA/GPU accidentally enabled | Verify tensors are on CPU, no `.cuda()` calls | |
| 33 | `torch.no_grad()` used during inference | Check context manager around benchmark loop | |
| 34 | Model is in eval mode | `model.eval()` called before benchmark | |
| 35 | No JIT compilation (`torch.jit`) | No `@torch.jit.script` or `torch.jit.trace` | |

### 3.2 Rust Implementation

| # | Validation Point | How to Verify | Pass/Fail |
|---|-----------------|---------------|-----------|
| 36 | Aprender LogisticRegression is standard impl | Check aprender source, no custom optimizations | |
| 37 | Release mode compilation | `--release` flag in cargo command | |
| 38 | No LTO or PGO special optimizations | Check Cargo.toml for default profile | |
| 39 | Uses `Instant::now()` not `SystemTime` | Verify in mnist_apr_benchmark.rs | |
| 40 | No `#[inline(always)]` on benchmark code | Check for forced inlining in example | |

### 3.3 Data Generation

| # | Validation Point | How to Verify | Pass/Fail |
|---|-----------------|---------------|-----------|
| 41 | Same formula: `((i * 17 + j * 31) % 256) / 255.0` | Compare both implementations character by character | |
| 42 | Same label generation: `i % 10 == 0 ? 0 : 1` | Verify binary label logic identical | |
| 43 | Same training sample count (1000) | `TRAINING_SAMPLES = 1000` in both | |
| 44 | Inference sample uses same formula | `(j % 256) / 255.0` for single sample | |
| 45 | Data types match: f32 in both | Python: torch.float32, Rust: f32 | |

---

## Category 4: Data Integrity (10 points)

### 4.1 Input Validation

| # | Validation Point | How to Verify | Pass/Fail |
|---|-----------------|---------------|-----------|
| 46 | Training data shape is (1000, 784) | Print/log data.shape in both | |
| 47 | Inference sample shape is (1, 784) | Verify sample dimensions | |
| 48 | All pixel values in [0, 1] range | Check formula bounds: 0/255=0, 255/255=1 | |
| 49 | Labels are only 0 or 1 | Binary classification constraint | |
| 50 | No NaN or Inf in generated data | Deterministic formula cannot produce NaN | |

### 4.2 Model State

| # | Validation Point | How to Verify | Pass/Fail |
|---|-----------------|---------------|-----------|
| 51 | Model is trained before benchmarking | `fit()` / training loop completes before measurement | |
| 52 | Model weights are non-zero | Print model.linear.weight sum, verify != 0 | |
| 53 | Model produces valid predictions | Output is 0 or 1, not garbage | |
| 54 | Same model used for all iterations | No re-initialization in benchmark loop | |
| 55 | Model not modified during benchmark | Inference only, no gradient updates | |

---

## Category 5: Environment Fairness (15 points)

### 5.1 Hardware

| # | Validation Point | How to Verify | Pass/Fail |
|---|-----------------|---------------|-----------|
| 56 | Same CPU for both benchmarks | Compare CPU field in both JSON outputs | |
| 57 | Same machine (not cloud instance variance) | Run sequentially on same physical machine | |
| 58 | No CPU throttling during benchmark | Check governor is 'performance', not 'powersave' | |
| 59 | No thermal throttling | Monitor CPU temp, should be stable | |
| 60 | Sufficient RAM available | No swap usage during benchmark | |

### 5.2 Software Environment

| # | Validation Point | How to Verify | Pass/Fail |
|---|-----------------|---------------|-----------|
| 61 | PyTorch version is recent/stable | 2.x series, not ancient version | |
| 62 | Python version is recent | 3.10+ | |
| 63 | Rust uses stable toolchain | `rustc --version` shows stable | |
| 64 | No debug symbols in Rust build | Release profile strips debug info | |
| 65 | No MKL/OpenBLAS advantage for PyTorch | Check if special BLAS linked | |

### 5.3 Process Isolation

| # | Validation Point | How to Verify | Pass/Fail |
|---|-----------------|---------------|-----------|
| 66 | No other CPU-intensive processes | Check `top` during benchmark | |
| 67 | GC not triggered during Python benchmark | Warmup should stabilize memory | |
| 68 | No background compilation in Rust | Benchmark runs pre-compiled binary | |
| 69 | Both run single-threaded | No parallelism in inference | |
| 70 | No NUMA effects | Single socket or NUMA-aware | |

---

## Category 6: Reproducibility (10 points)

### 6.1 Determinism

| # | Validation Point | How to Verify | Pass/Fail |
|---|-----------------|---------------|-----------|
| 71 | Results reproducible on re-run | Run 3 times, compare means within 10% | |
| 72 | Same seed produces same training data | Hash data bytes, verify identical | |
| 73 | Inference output is deterministic | Same input → same prediction every time | |

### 6.2 Documentation

| # | Validation Point | How to Verify | Pass/Fail |
|---|-----------------|---------------|-----------|
| 74 | All commands documented in README | Can reproduce from README alone | |
| 75 | Dependencies specified (pyproject.toml) | `uv sync` installs correct versions | |
| 76 | Rust dependencies locked (Cargo.lock) | Committed and used | |
| 77 | Environment captured in JSON output | Platform, CPU, versions recorded | |

### 6.3 Third-Party Verification

| # | Validation Point | How to Verify | Pass/Fail |
|---|-----------------|---------------|-----------|
| 78 | Benchmark can run on different machine | Test on separate hardware | |
| 79 | Results consistent across machines | Speedup ratio similar (within 2x) | |
| 80 | No machine-specific optimizations | Code doesn't detect specific CPU | |

---

## Category 7: Result Plausibility (10 points)

### 7.1 Absolute Performance

| # | Validation Point | How to Verify | Pass/Fail |
|---|-----------------|---------------|-----------|
| 81 | PyTorch ~5µs is plausible for 784→2 linear | Compare to published PyTorch benchmarks | |
| 82 | Aprender ~0.5µs is plausible for pure Rust | Compare to similar Rust ML libraries | |
| 83 | Neither result is suspiciously fast (<100ns) | Sub-100ns would indicate measurement error | |
| 84 | Neither result is suspiciously slow (>1ms) | Would indicate I/O or allocation issue | |

### 7.2 Relative Performance

| # | Validation Point | How to Verify | Pass/Fail |
|---|-----------------|---------------|-----------|
| 85 | 10x speedup is plausible for Rust vs Python | Consistent with other Rust/Python comparisons | |
| 86 | Speedup not >100x (would indicate bug) | 100x would suggest PyTorch misconfiguration | |
| 87 | Speedup not <2x (would suggest Rust issue) | <2x would indicate Rust not optimized | |

### 7.3 Consistency

| # | Validation Point | How to Verify | Pass/Fail |
|---|-----------------|---------------|-----------|
| 88 | p50 < p95 < p99 in both | Percentile ordering is correct | |
| 89 | Mean close to p50 (low skew) | For stable systems, mean ≈ median | |
| 90 | Std dev << mean (low variance) | CV < 50% indicates stable measurement | |

---

## Category 8: Bias Detection (10 points)

### 8.1 Favorable Conditions

| # | Validation Point | How to Verify | Pass/Fail |
|---|-----------------|---------------|-----------|
| 91 | PyTorch not deliberately slowed | No `time.sleep()`, no busy loops | |
| 92 | Aprender not using undocumented optimizations | Check aprender source is standard | |
| 93 | Neither uses SIMD when other doesn't | Both use scalar or both use SIMD | |
| 94 | No precomputed results in Aprender | Actual matrix multiplication occurs | |

### 8.2 Unfair Comparisons

| # | Validation Point | How to Verify | Pass/Fail |
|---|-----------------|---------------|-----------|
| 95 | Not comparing debug Rust vs release Python | Rust uses `--release` | |
| 96 | Not comparing GPU PyTorch vs CPU Rust | Both on CPU | |
| 97 | Not comparing batched vs single inference | Both single sample | |
| 98 | Training time not included in inference | Only `predict()` timed | |

### 8.3 Reporting Honesty

| # | Validation Point | How to Verify | Pass/Fail |
|---|-----------------|---------------|-----------|
| 99 | Worst-case (p99) reported, not just best | p99 in results table | |
| 100 | Limitations acknowledged | Binary classification noted, not full MNIST | |

---

## Red Team Verification Protocol

### Phase 1: Code Review (Points 1-45)
```bash
# Clone and inspect
git clone <repo>
cd realizar/benches/comparative

# Verify code equivalence
diff -u <(grep -E "(SEED|WARMUP|BENCHMARK|INPUT_DIM|NUM_CLASSES)" mnist_benchmark.py) \
        <(grep -E "(SEED|WARMUP|BENCHMARK|INPUT_DIM|NUM_CLASSES)" ../../examples/mnist_apr_benchmark.rs)
```

### Phase 2: Independent Execution (Points 46-80)
```bash
# Run benchmarks independently
cargo run --example mnist_apr_benchmark --release --features aprender-serve
cd benches/comparative && uv run mnist_benchmark.py

# Compare JSON outputs
jq '.config' aprender_mnist_results.json
jq '.config' pytorch_mnist_results.json
```

### Phase 3: Statistical Validation (Points 16-30)
```bash
# Verify statistics with independent calculation
python3 -c "
import json
import statistics

with open('pytorch_mnist_results.json') as f:
    data = json.load(f)

# Recalculate from raw data if available
# Or verify reported statistics are mathematically consistent
r = data['results'][0]
print(f'Reported mean: {r[\"mean_us\"]}')
print(f'Throughput check: {1_000_000 / r[\"mean_us\"]:.0f} vs reported {r[\"throughput_per_sec\"]:.0f}')
"
```

### Phase 4: Cross-Machine Verification (Points 78-80)
```bash
# Run on different hardware
ssh other-machine
git clone <repo> && cd realizar
cargo run --example mnist_apr_benchmark --release --features aprender-serve
# Compare speedup ratio
```

---

## Known Limitations (Acknowledged)

1. **Binary classification only** - Aprender's LogisticRegression is binary; full 10-class MNIST would require different model
2. **test data** - Not real MNIST images, but mathematically equivalent workload
3. **Single-threaded** - No parallelism comparison
4. **CPU only** - No GPU comparison
5. **Inference only** - Training performance not measured

---

## Acceptance Criteria

- **Minimum 95/100 points PASS** for benchmark to be considered valid
- **All Category 1-3 points must PASS** (methodology, statistics, code)
- **Any FAIL in bias detection (91-100) invalidates benchmark**

---

## Sign-Off

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Benchmark Author | | | |
| Red Team Lead | | | |
| Independent Reviewer | | | |

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-11-27 | Realizar Team | Initial QA checklist |
