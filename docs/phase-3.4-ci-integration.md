# Phase 3.4: PMAT Integration and CI/CD Automation

**Status**: Implemented
**Date**: 2025-11-18
**Dependencies**: Phase 3.2 (Infrastructure), Phase 3.3 (Reporting)

## Overview

Phase 3.4 integrates benchmark automation into CI/CD pipelines with PMAT quality gates,
automated regression detection, and performance tracking dashboards.

## Components

### 1. GitHub Actions Workflow (`.github/workflows/benchmarks.yml`)

Comprehensive CI/CD workflow with 5 jobs:

#### Job 1: Run Benchmarks
- **Triggers**:
  - Push to main branch
  - Pull requests
  - Weekly schedule (Mondays 00:00 UTC)
  - Manual workflow dispatch
- **Actions**:
  - Checkout code with full history
  - Setup Rust 1.82.0 + Deno
  - Build release binaries
  - Generate reproducibility manifest
  - Run benchmarks (critical or all)
  - Upload results as artifacts

#### Job 2: Regression Check (PR only)
- **Purpose**: Detect performance regressions in pull requests
- **Actions**:
  - Download current benchmark results
  - Fetch baseline from main branch
  - Run regression detection with thresholds
  - Generate comparison report
  - Comment on PR with results
  - Fail build if critical regressions detected
- **Exit Codes**:
  - 0: No regressions
  - 1: Warnings (don't fail build)
  - 2: Critical regressions (fail build)

#### Job 3: Generate Reports (main branch + schedule)
- **Purpose**: Create comprehensive reports after merging
- **Outputs**:
  - Markdown report
  - CSV exports (summary + metadata + raw timings)
  - Upload as artifacts (90-day retention)

#### Job 4: Update Baseline (main branch only)
- **Purpose**: Automatically update main baseline after merge
- **Actions**:
  - Save current results as `benchmarks/baselines/main.json`
  - Commit and push baseline update
  - Uses `[skip ci]` to avoid triggering another build

#### Job 5: Performance Tracking (scheduled only)
- **Purpose**: Long-term performance trend tracking
- **Actions**:
  - Save weekly snapshots
  - Create named baselines (weekly-YYYY-WNN)
  - Archive results for historical analysis

### 2. PMAT Quality Gates Configuration

Added `[benchmarks]` section to `.pmat-gates.toml`:

```toml
[benchmarks]
# Performance benchmarking quality gates
run_benchmarks_on_pr = true
run_benchmarks_on_main = true

# Regression thresholds
regression_warning_threshold = 0.05  # 5% slowdown
regression_critical_threshold = 0.10  # 10% slowdown
regression_significance_level = 0.05  # p-value
regression_min_effect_size = 0.2  # Cohen's d

# Requirements
min_iterations = 10
min_warmup = 3
require_baseline = true
fail_on_critical_regression = true
warn_on_regression = true

# Reporting
generate_reports = true
comment_on_pr = true
upload_artifacts = true
```

### 3. Makefile Integration

Added 7 new benchmark targets:

| Target | Description |
|--------|-------------|
| `make benchmark` | Run critical benchmarks (default) |
| `make benchmark-all` | Run comprehensive benchmark suite |
| `make benchmark-report` | Generate markdown + CSV reports |
| `make benchmark-compare` | Compare against baseline, detect regressions |
| `make benchmark-baseline-save` | Save current results as baseline |
| `make benchmark-clean` | Clean benchmark artifacts |

**Integration with existing workflow:**
- All targets check for Deno availability
- Auto-creates baseline if none exists
- Follows existing Make conventions (quiet, helpful messages)

### 4. Performance Dashboard (`generate_dashboard.ts`)

Interactive HTML dashboard with Chart.js visualizations:

**Features:**
- Bar chart: Mean execution time per benchmark
- Bar chart: Coefficient of variation (stability indicator)
- Detailed statistics table
- Color-coded CV indicators:
  - Green: CV < 5% (excellent stability)
  - Yellow: CV 5-10% (acceptable)
  - Red: CV > 10% (unstable)
- Responsive dark theme matching GitHub's design

**Usage:**
```bash
deno run --allow-read --allow-write scripts/generate_dashboard.ts \
  --input benchmarks/results/latest.json \
  --output dashboard.html
```

## Workflow Diagrams

### Pull Request Flow

```
PR Opened/Updated
      │
      ├─> Run Benchmarks (critical)
      │         │
      │         ├─> Generate results.json
      │         └─> Upload artifact
      │
      ├─> Regression Check
      │         │
      │         ├─> Download results.json
      │         ├─> Fetch baseline from main
      │         ├─> Run check_regression.ts
      │         ├─> Generate comparison.md
      │         ├─> Comment on PR
      │         │
      │         └─> Exit:
      │               0 → ✅ Pass
      │               1 → ⚠️ Pass with warnings
      │               2 → 🔴 Fail build
      │
      └─> If passed → Merge
```

### Main Branch Flow

```
Merge to Main
      │
      ├─> Run Benchmarks (critical)
      │         │
      │         └─> Upload artifact
      │
      ├─> Generate Reports
      │         │
      │         ├─> Markdown report
      │         ├─> CSV exports
      │         └─> Upload artifacts (90 days)
      │
      └─> Update Baseline
                │
                ├─> Save as main.json
                ├─> Commit [skip ci]
                └─> Push to main
```

### Weekly Schedule Flow

```
Monday 00:00 UTC
      │
      ├─> Run Benchmarks (all)
      │
      ├─> Generate Reports
      │
      └─> Performance Tracking
                │
                ├─> Save weekly snapshot
                ├─> Create weekly-YYYY-WNN baseline
                └─> Archive for trend analysis
```

## Usage Examples

### Local Development

```bash
# Run benchmarks locally
make benchmark

# Generate reports
make benchmark-report

# Check for regressions
make benchmark-compare

# Save as baseline
make benchmark-baseline-save

# Generate dashboard
deno run --allow-read --allow-write scripts/generate_dashboard.ts \
  --input benchmarks/results/latest.json \
  --output dashboard.html
```

### CI/CD Integration

The workflow automatically runs on:
- Every push to main
- Every pull request
- Weekly (Mondays)
- Manual trigger

No manual configuration needed after initial setup.

### Customizing Thresholds

Edit `.pmat-gates.toml`:

```toml
[benchmarks]
# Make regressions more/less strict
regression_warning_threshold = 0.10  # 10% instead of 5%
regression_critical_threshold = 0.20  # 20% instead of 10%
```

Then re-run:

```bash
make benchmark-compare
```

## Directory Structure

```
certeza/
├── .github/
│   └── workflows/
│       └── benchmarks.yml          # CI/CD workflow
├── .pmat-gates.toml                # Updated with [benchmarks] section
├── Makefile                        # Updated with benchmark targets
├── benchmarks/
│   ├── baselines/
│   │   ├── index.json              # Baseline registry
│   │   ├── main.json               # Auto-updated from main branch
│   │   └── weekly-*.json           # Weekly snapshots
│   ├── history/
│   │   └── YYYY-WNN/               # Weekly historical data
│   │       ├── results.json
│   │       └── report.md
│   ├── metadata/
│   │   └── toolchain_manifest.txt
│   └── results/
│       ├── latest.json             # Most recent run
│       ├── report.md               # Markdown report
│       ├── comparison.md           # Baseline comparison
│       ├── regression_report.json  # Regression detection results
│       └── csv/                    # CSV exports
├── docs/
│   └── phase-3.4-ci-integration.md # This document
└── scripts/
    ├── generate_dashboard.ts       # HTML dashboard generator
    ├── check_regression.ts         # Regression detection (Phase 3.3)
    ├── baseline_manager.ts         # Baseline management (Phase 3.3)
    └── ... (other Phase 3.3 tools)
```

## Best Practices

### 1. Baseline Management

**Do:**
- Update baseline after every release
- Keep named baselines for major versions
- Archive baselines before major refactors

**Don't:**
- Update baseline without review
- Delete baselines without archiving
- Commit unstable benchmarks as baseline

### 2. Regression Thresholds

**Conservative (recommended):**
- Warning: 5% slowdown
- Critical: 10% slowdown
- Requires statistical significance (p < 0.05)

**Aggressive (for critical paths):**
- Warning: 2% slowdown
- Critical: 5% slowdown
- Lower significance threshold

**Lenient (for experimental code):**
- Warning: 10% slowdown
- Critical: 20% slowdown
- Higher significance threshold

### 3. PR Workflow

1. Open PR
2. Wait for benchmark job to complete (~5 min)
3. Review bot comment on PR
4. If regressions:
   - Investigate cause
   - Optimize if necessary
   - Update PR
5. If no regressions or justified:
   - Merge to main
6. Baseline auto-updates after merge

### 4. Performance Tracking

**Weekly snapshots:**
- Saved every Monday
- Retained indefinitely
- Used for trend analysis

**How to analyze trends:**
```bash
# Compare against previous week
deno run --allow-read --allow-write scripts/baseline_manager.ts compare \
  --baseline weekly-2025-W46 \
  --current weekly-2025-W47 \
  --format markdown \
  --output trend-analysis.md
```

## Troubleshooting

### Issue: Benchmark job times out

**Solution:**
```yaml
# In .github/workflows/benchmarks.yml
jobs:
  benchmark:
    timeout-minutes: 120  # Increase from 60
```

### Issue: No baseline available

**Solution:**
```bash
# Create initial baseline
make benchmark
make benchmark-baseline-save
git add benchmarks/baselines/main.json
git commit -m "chore: add initial benchmark baseline"
git push
```

### Issue: Flaky regression detection

**Possible causes:**
1. Insufficient warmup iterations
2. High system variance
3. Unstable benchmarks

**Solutions:**
```bash
# Increase iterations
./scripts/run_benchmarks.sh \
  --benchmarks critical \
  --warmup 10 \
  --iterations 50 \
  --output results.json

# Adjust thresholds in .pmat-gates.toml
regression_min_effect_size = 0.5  # Larger effect required
```

### Issue: Dashboard not displaying correctly

**Check:**
1. Input JSON is valid
2. All required fields present
3. Browser console for errors

**Regenerate:**
```bash
deno run --allow-read --allow-write scripts/generate_dashboard.ts \
  --input benchmarks/results/latest.json \
  --output dashboard.html
```

## Integration with PMAT

Phase 3.4 is fully compliant with PMAT quality gates:

- **Tier 1 (ON-SAVE)**: Not applicable (benchmarks are expensive)
- **Tier 2 (ON-COMMIT)**: Critical benchmarks on PR
- **Tier 3 (ON-MERGE)**: Full benchmark suite + baseline update

Quality gates enforce:
- Minimum iterations (10)
- Minimum warmup (3)
- Statistical significance
- Baseline comparison
- Automated regression detection

## Future Enhancements (Phase 3.5+)

Potential improvements:

1. **Trend Analysis**:
   - Automated detection of performance degradation over time
   - Alert if performance degrades >5% over 4 weeks

2. **Benchmark Coverage**:
   - Track which code paths are benchmarked
   - Report on un-benchmarked critical functions

3. **Multi-Platform Benchmarks**:
   - Run on Linux, macOS, Windows
   - Compare cross-platform performance

4. **Benchmark Budget**:
   - Set maximum allowed execution time per benchmark
   - Fail if budget exceeded

5. **Historical Dashboard**:
   - Show performance trends over months
   - Identify optimization opportunities

## References

- **GitHub Actions**: https://docs.github.com/en/actions
- **PMAT Documentation**: https://github.com/paiml/paiml-mcp-agent-toolkit
- **Chart.js**: https://www.chartjs.org/
- **Scientific Benchmarking**: Phase 3.2 + 3.3 documentation

## Changelog

- **2025-11-18**: Initial Phase 3.4 implementation
  - GitHub Actions workflow
  - PMAT quality gates
  - Makefile integration
  - Performance dashboard
  - Documentation

---

**Status**: ✅ Phase 3.4 Complete

**Next Phase**: Phase 3.5 - Reproducibility and Archival
