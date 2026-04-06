# Performance Regression CI

## Overview

Trueno uses automated performance regression detection to prevent performance degradations from being merged. This system:

1. **Establishes Baselines**: Captures known-good performance on main branch
2. **Detects Regressions**: Compares PR performance against baseline
3. **Blocks Merges**: Fails CI if performance drops >5%
4. **Tracks History**: Archives old baselines for historical comparison

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Developer Workflow                       │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
                  ┌──────────────────┐
                  │  make bench-save │
                  │    -baseline     │
                  └──────────────────┘
                            │
                            ▼
        ┌────────────────────────────────────────┐
        │  .performance-baselines/               │
        │    baseline-current.txt                │
        │    baseline-{commit}-archived.txt      │
        └────────────────────────────────────────┘
                            │
                            ▼
                  ┌──────────────────┐
                  │  Git commit +    │
                  │  Push to repo    │
                  └──────────────────┘
                            │
┌───────────────────────────┼───────────────────────────┐
│                   GitHub Actions CI                   │
└───────────────────────────────────────────────────────┘
                            │
                 ┌──────────┴──────────┐
                 ▼                     ▼
        ┌──────────────┐      ┌──────────────┐
        │ PR Opened/   │      │ Push to Main │
        │ Updated      │      │              │
        └──────────────┘      └──────────────┘
                 │                     │
                 ▼                     ▼
        ┌──────────────────────────────────┐
        │ .github/workflows/ci.yml         │
        │   - Run benchmarks (--sample-10) │
        │   - Compare vs baseline          │
        │   - Fail if >5% regression       │
        └──────────────────────────────────┘
                 │
      ┌──────────┴──────────┐
      ▼                     ▼
┌──────────┐        ┌──────────────┐
│ ✅ PASS   │        │ ❌ FAIL       │
│ Merge OK │        │ Block merge  │
└──────────┘        └──────────────┘
```

## Quick Start

### 1. Create Initial Baseline (One-Time Setup)

```bash
# On main branch with known-good performance
git checkout main
make bench-save-baseline

# Commit baseline to repository
git add .performance-baselines/baseline-current.txt
git commit -m "Add performance baseline for CI regression detection"
git push
```

### 2. Automatic Regression Detection

All future PRs will automatically check for regressions:

```bash
# Create feature branch
git checkout -b feature/my-optimization

# Make changes...
# ...

# Push to GitHub - CI runs automatically
git push origin feature/my-optimization

# CI will:
# 1. Run benchmarks
# 2. Compare against baseline
# 3. Report regressions/improvements
# 4. Pass/fail based on threshold
```

### 3. Update Baseline After Performance Improvements

When you've made legitimate performance improvements:

```bash
# Verify improvements locally
make bench-compare  # Should show improvements (negative %)

# Update baseline
make bench-save-baseline

# Commit new baseline
git add .performance-baselines/baseline-current.txt
git commit -m "Update baseline after AVX-512 optimization (+17x dot product)"
git push
```

## Components

### 1. Baseline Storage (`.performance-baselines/`)

**`baseline-current.txt`** - Active baseline used for regression detection

Format:
```
# Trueno Performance Baseline
# Commit: 4b352c0
# Branch: main
# Date: 20251123-112000
# CPU: AMD Ryzen 9 5950X 16-Core Processor
#
add/AVX2/100            time:   [56.896 ns 58.549 ns 60.001 ns]
dot/AVX512/1000         time:   [12.345 µs 12.450 µs 12.567 µs]
...
```

**`baseline-{commit}-archived.txt`** - Historical baselines (auto-archived when updating)

### 2. Baseline Management Script (`scripts/save_baseline.sh`)

Automated baseline creation with metadata:

```bash
#!/bin/bash
# Usage: ./scripts/save_baseline.sh [benchmark_file]

# Features:
# - Runs benchmarks (or uses provided file)
# - Adds commit metadata (hash, branch, date, CPU)
# - Archives old baseline
# - Saves as baseline-current.txt
```

### 3. Regression Detection Script (`scripts/check_regression.py`)

Compares benchmark runs and detects regressions:

```bash
python3 scripts/check_regression.py \
  --baseline .performance-baselines/baseline-current.txt \
  --current /tmp/bench-current.txt \
  --regression-threshold 0.05 \  # Fail if >5% slower
  --warning-threshold 0.02        # Warn if >2% slower
```

**Output:**
```
============================================================
BENCHMARK REGRESSION REPORT
============================================================

REGRESSIONS (>5% slower): 2
------------------------------------------------------------
  add/AVX512/100
    Baseline: 58.55 ns
    Current:  79.95 ns
    Change:   +36.6% (FAIL)

WARNINGS (2%-5% slower): 1
------------------------------------------------------------
  sub/AVX2/1000
    Baseline: 140.2 ns
    Current:  143.6 ns
    Change:   +2.4%

IMPROVEMENTS (>2% faster): 5
------------------------------------------------------------
  dot/AVX512/1000
    Baseline: 214.5 µs
    Current:  12.45 µs
    Change:   -94.2%

============================================================
SUMMARY
============================================================
  Total benchmarks: 128
  Regressions:      2
  Warnings:         1
  Improvements:     5
  Unchanged:        120

RESULT: FAIL - 2 regression(s) detected
============================================================
```

**Exit Codes:**
- `0`: No regressions (pass)
- `1`: Regressions detected (fail)

### 4. CI Workflow (`.github/workflows/ci.yml`)

```yaml
bench:
  name: Benchmarks
  runs-on: ubuntu-latest
  steps:
    # ... setup steps ...

    - name: Check for performance regressions
      if: hashFiles('.performance-baselines/baseline-current.txt') != ''
      run: |
        # Run benchmarks (reduced sample size for faster CI)
        cargo bench --bench vector_ops --all-features --no-fail-fast \
          -- --sample-size 10 2>&1 | tee /tmp/bench-current.txt

        # Run regression check (fail CI if >5% regression)
        python3 scripts/check_regression.py \
          --baseline .performance-baselines/baseline-current.txt \
          --current /tmp/bench-current.txt \
          --regression-threshold 0.05 \
          --warning-threshold 0.02

        # Exit with error if regressions detected
        if [ $? -ne 0 ]; then
          echo "❌ Performance regressions detected"
          exit 1
        fi

    - name: Baseline status
      if: hashFiles('.performance-baselines/baseline-current.txt') == ''
      run: |
        echo "⚠️  No performance baseline found"
        echo "To enable: make bench-save-baseline && git add/commit/push"
```

### 5. Makefile Targets

**`make bench-save-baseline`** - Create/update baseline
```bash
# Uses scripts/save_baseline.sh
# Runs full benchmark suite (5-10 minutes)
# Archives old baseline
# Saves baseline-current.txt with commit metadata
```

**`make bench-compare`** - Check for regressions
```bash
# Runs benchmarks
# Compares against baseline-current.txt
# Reports regressions/warnings/improvements
# Fails if >5% regression
```

## Regression Policy

### Thresholds

| Change | Action | Description |
|--------|--------|-------------|
| **>+5%** | **FAIL CI** | Significant regression - blocks merge |
| **+2% to +5%** | **WARN** | Minor regression - investigate before merge |
| **-2% to +2%** | **PASS** | Noise threshold - no action |
| **<-2%** | **PASS (improvement!)** | Performance improvement - good! |

### Decision Tree

```
Is benchmark >5% slower?
├─ YES → FAIL CI (block merge)
│         └─ Action: Fix regression or update baseline if intentional
└─ NO → Is benchmark 2-5% slower?
          ├─ YES → WARN (investigate)
          │         └─ Action: Verify not a real regression
          └─ NO → PASS
                    └─ Action: Merge allowed
```

## Workflow Examples

### Example 1: Performance Improvement

```bash
# 1. Make optimization
git checkout -b feature/avx512-dot-optimization
# ... implement AVX-512 dot product ...

# 2. Verify locally
make bench-compare
# Output:
#   IMPROVEMENTS:
#     dot/AVX512/1000: -94.2% (214.5 µs → 12.45 µs)
#   RESULT: PASS

# 3. Push to PR - CI passes
git push origin feature/avx512-dot-optimization

# 4. After merge, update baseline
git checkout main
git pull
make bench-save-baseline
git add .performance-baselines/baseline-current.txt
git commit -m "Update baseline after AVX-512 dot optimization (+17x speedup)"
git push
```

### Example 2: Accidental Regression

```bash
# 1. Make change that accidentally regresses performance
git checkout -b feature/refactor-backend-selection
# ... refactor code ...

# 2. Push to PR
git push origin feature/refactor-backend-selection

# 3. CI FAILS:
#   ❌ REGRESSIONS:
#     add/AVX512/100: +36.6% (58.55 ns → 79.95 ns)
#   RESULT: FAIL - 1 regression(s) detected

# 4. Investigate and fix
# ... analyze why AVX-512 is slower ...
# ... discover operation-aware selection needed ...

# 5. Fix regression
git add src/lib.rs
git commit -m "Fix: Use AVX2 for memory-bound ops (prevents AVX-512 regression)"
git push

# 6. CI now passes
#   RESULT: PASS - No regressions detected
```

### Example 3: Intentional Tradeoff

```bash
# 1. Make change that trades performance for correctness
git checkout -b fix/edge-case-bug
# ... fix bug that adds bounds checking ...

# 2. Local check shows regression
make bench-compare
# Output:
#   REGRESSIONS:
#     add/AVX2/100: +8.2% (58.55 ns → 63.35 ns)
#   RESULT: FAIL

# 3. This is acceptable - correctness > performance
# Update baseline to reflect new expected performance
make bench-save-baseline

# 4. Commit with explanation
git add .performance-baselines/baseline-current.txt
git commit -m "Update baseline: +8% overhead from bounds checking (fixes CVE-2025-12345)"

# 5. Push - CI passes with new baseline
git push
```

## Advanced Usage

### Compare Historical Baselines

```bash
# Compare current performance vs performance from 3 commits ago
python3 scripts/check_regression.py \
  --baseline .performance-baselines/baseline-88e21c7-archived.txt \
  --current .performance-baselines/baseline-current.txt
```

### Custom Thresholds

```bash
# Stricter threshold for critical performance code
make bench-compare | python3 scripts/check_regression.py \
  --baseline .performance-baselines/baseline-current.txt \
  --current /tmp/bench-current.txt \
  --regression-threshold 0.02 \  # Fail at >2%
  --warning-threshold 0.01        # Warn at >1%
```

### Manual CI Simulation

```bash
# Simulate CI regression check locally (uses same sample size as CI)
cargo bench --bench vector_ops -- --sample-size 10 2>&1 | tee /tmp/bench-ci.txt
python3 scripts/check_regression.py \
  --baseline .performance-baselines/baseline-current.txt \
  --current /tmp/bench-ci.txt
```

## Troubleshooting

### Issue: High Variance / Flaky CI

**Symptoms**: CI sometimes passes, sometimes fails for same code

**Causes**:
- Background processes (CI runner load)
- Thermal throttling
- CPU frequency scaling
- Power management

**Solutions**:
1. **Increase sample size in CI** (but slower):
   ```yaml
   cargo bench -- --sample-size 20  # Instead of 10
   ```

2. **Widen thresholds** (but less sensitive):
   ```python
   --regression-threshold 0.10  # 10% instead of 5%
   ```

3. **Pin CPU frequency** (requires CI admin):
   ```bash
   sudo cpupower frequency-set -g performance
   ```

### Issue: Baseline Missing in CI

**Symptoms**: CI skips regression check with warning

**Cause**: `baseline-current.txt` not committed to repository

**Solution**:
```bash
git checkout main
make bench-save-baseline
git add .performance-baselines/baseline-current.txt
git commit -m "Add performance baseline"
git push
```

### Issue: All Benchmarks Show Regression After Baseline Update

**Symptoms**: Every benchmark worse after updating baseline

**Cause**: Baseline was generated on different hardware or with different settings

**Solution**:
- Always generate baselines on same hardware (e.g., CI runner)
- Document CPU model in baseline commit message
- Consider separate baselines for different platforms

## Best Practices

### 1. Baseline Management

✅ **DO:**
- Generate baselines on stable main branch
- Update baseline after verifying performance improvements
- Commit baselines with descriptive messages
- Archive old baselines for historical comparison

❌ **DON'T:**
- Generate baselines on feature branches
- Update baseline to "fix" failing CI without investigation
- Delete archived baselines

### 2. Regression Investigation

✅ **DO:**
- Investigate all >2% regressions before merging
- Profile code to understand root cause
- Document intentional performance tradeoffs in commit messages

❌ **DON'T:**
- Merge PRs with unexplained regressions
- Assume small regressions (<5%) are always noise
- Update baseline without understanding why performance changed

### 3. CI Integration

✅ **DO:**
- Run regression checks on all PRs
- Use consistent hardware for baseline generation
- Keep baselines up-to-date (regenerate after major changes)

❌ **DON'T:**
- Skip regression checks to speed up CI
- Use laptop/desktop baselines for CI comparison
- Let baselines get too old (>1 month or >50 commits)

## Performance Budget Example

Establish performance budgets for critical operations:

| Operation | Size | Baseline | Budget (+5%) | Hard Limit (+10%) |
|-----------|------|----------|--------------|-------------------|
| add_f32 | 100 | 58.5 ns | 61.4 ns | 64.4 ns |
| add_f32 | 1K | 143.6 ns | 150.8 ns | 158.0 ns |
| dot_f32 | 1K | 12.45 µs | 13.07 µs | 13.70 µs |

**Enforcement:**
- **Budget exceeded (+5%)**: CI warning, requires justification
- **Hard limit exceeded (+10%)**: CI fail, requires approval + baseline update

## References

- **Criterion.rs Documentation**: https://bheisler.github.io/criterion.rs/book/
- **Performance Regression Testing (Academic)**: Huang et al., "Performance Regression Testing Target Prioritization via Performance Risk Analysis" (ICSE 2014)
- **Toyota Production System**: Jidoka (built-in quality) applied to performance
- **Google SRE Book**: Chapter 6 - Monitoring Distributed Systems (performance budgets)

## See Also

- [BENCHMARK_ANALYSIS.md](../BENCHMARK_ANALYSIS.md) - Detailed benchmark results
- [AVX512_ANALYSIS.md](../AVX512_ANALYSIS.md) - AVX-512 performance investigation
- [.performance-baselines/README.md](../.performance-baselines/README.md) - Baseline management
- [CLAUDE.md](../CLAUDE.md) - Development commands and quality gates
