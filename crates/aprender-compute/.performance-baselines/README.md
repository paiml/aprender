# Performance Baseline Storage

This directory contains performance baselines for regression detection.

## Quick Start

### Create a New Baseline

```bash
# Recommended: Use the Makefile target (includes commit metadata)
make bench-save-baseline

# Manual: Run benchmarks and save output
./scripts/save_baseline.sh
```

This will:
1. Run full benchmark suite (5-10 minutes)
2. Add commit metadata (commit hash, branch, date, CPU info)
3. Archive any existing baseline
4. Save as `.performance-baselines/baseline-current.txt`

### Check for Regressions

```bash
# Compare current performance vs baseline
make bench-compare
```

This will:
1. Run benchmarks
2. Compare against `baseline-current.txt`
3. Report regressions (>5% slower), warnings (2-5% slower), and improvements
4. Fail if any benchmark shows >5% regression

## Format

Baselines are stored as raw Criterion benchmark output with metadata header:

```
# Trueno Performance Baseline
# Commit: 4b352c0
# Branch: claude/continue-next-step-01NEN2Jw5zVsNK9DWCE1Hwqz
# Date: 20251123-112000
# CPU: AMD Ryzen 9 5950X 16-Core Processor
#
add/AVX2/100            time:   [56.896 ns 58.549 ns 60.001 ns]
                        thrpt:  [1.6666 Gelem/s 1.7080 Gelem/s 1.7576 Gelem/s]
...
```

The check_regression.py script parses these lines and extracts mean times.

## Files

- **`baseline-current.txt`** - Active baseline used for CI/CD regression detection
- **`baseline-{commit}-archived.txt`** - Historical baselines (auto-created when updating)
- **`baseline-template.json`** - Legacy JSON format (not currently used)

## CI Integration

The CI pipeline (`.github/workflows/ci.yml`) automatically:

1. Checks if `baseline-current.txt` exists
2. If yes: Runs benchmarks and compares against baseline
3. Fails CI if any benchmark shows >5% regression
4. If no: Displays warning but continues (baseline creation is optional)

### Setting Up CI Regression Detection

```bash
# 1. Generate baseline on main branch
git checkout main
make bench-save-baseline

# 2. Commit baseline to repository
git add .performance-baselines/baseline-current.txt
git commit -m "Add performance baseline for regression detection"
git push

# 3. All future PRs will check against this baseline
```

## Regression Policy

**CI Fails if**:
- Any benchmark >5% slower than baseline
- Required for: Merging to main

**CI Warns if**:
- Any benchmark 2-5% slower than baseline
- Action: Investigate before merging

**No Action if**:
- Benchmark unchanged (±2%)
- Benchmark faster (improvement!)

## Advanced Usage

### Update Baseline After Performance Improvements

When you've made performance improvements and want to update the baseline:

```bash
# 1. Verify improvements
make bench-compare  # Should show improvements

# 2. Update baseline
make bench-save-baseline

# 3. Commit new baseline
git add .performance-baselines/baseline-current.txt
git commit -m "Update baseline after AVX-512 optimization (17x speedup for dot product)"
```

### Manual Regression Check

```bash
# Run benchmarks
cargo bench --bench vector_ops 2>&1 | tee /tmp/my-bench.txt

# Compare against baseline
python3 scripts/check_regression.py \
  --baseline .performance-baselines/baseline-current.txt \
  --current /tmp/my-bench.txt \
  --regression-threshold 0.05 \
  --warning-threshold 0.02
```

### Compare Two Benchmark Runs

```bash
# Run baseline
cargo bench 2>&1 | tee /tmp/baseline.txt

# Make changes...

# Run current
cargo bench 2>&1 | tee /tmp/current.txt

# Compare
python3 scripts/check_regression.py \
  --baseline /tmp/baseline.txt \
  --current /tmp/current.txt
```

## Troubleshooting

### Baseline Too Old

If the baseline is from an old commit and the codebase has changed significantly:

```bash
# Regenerate baseline
make bench-save-baseline
git add .performance-baselines/baseline-current.txt
git commit -m "Update baseline after major refactor"
```

### High Variance

If benchmarks show high variance (>10% between runs):

- **Cause**: Background processes, thermal throttling, power management
- **Solution**: Run on dedicated CI hardware or increase sample size

### False Positives

If regression check fails but you're confident there's no regression:

1. Verify on dedicated hardware (not laptop)
2. Check CPU frequency scaling/power management
3. Increase sample size: `cargo bench -- --sample-size 50`
4. Check if Criterion baseline is stale: `rm -rf target/criterion && make bench-compare`

## Historical Baselines

When you run `make bench-save-baseline`, the old baseline is automatically archived:

```bash
.performance-baselines/
├── baseline-current.txt           # Active baseline
├── baseline-4b352c0-archived.txt  # Previous baseline (commit 4b352c0)
├── baseline-88e21c7-archived.txt  # Older baseline (commit 88e21c7)
└── README.md
```

This allows you to compare against historical performance:

```bash
python3 scripts/check_regression.py \
  --baseline .performance-baselines/baseline-88e21c7-archived.txt \
  --current .performance-baselines/baseline-current.txt
```
