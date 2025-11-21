# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Aprender is a next-generation machine learning library written in pure Rust. v0.4.1 implements the TOP 10 ML algorithms with 742 tests (unit, property, integration, doc) and comprehensive quality gates achieving 96.94% code coverage.

## Build Commands

```bash
# Standard Cargo commands
cargo build --release        # Optimized build
cargo test                   # Full test suite (742 tests: unit + property + integration + doc)
cargo test --lib             # Unit tests only
cargo fmt --check            # Check formatting
cargo clippy -- -D warnings  # Strict linting
cargo doc --no-deps --open   # Generate and view docs
cargo bench                  # Run criterion benchmarks

# Makefile shortcuts
make tier1                   # Fast feedback (<1s): fmt, clippy, check
make tier2                   # Pre-commit (<5s): tests + strict clippy
make tier3                   # Pre-push (1-5min): full validation + coverage
make tier4                   # CI/CD: includes pmat analysis
make coverage                # Generate coverage report (96.94% achieved)
```

### Tiered Quality Gates (Certeza Methodology)

```bash
# Tier 1: On-Save (<1s) - Fast feedback
cargo fmt --check && cargo clippy -- -W all && cargo check

# Tier 2: Pre-Commit (<5s)
cargo test --lib && cargo clippy -- -D warnings

# Tier 3: Pre-Push (1-5 min)
cargo test --all
make coverage                                 # Coverage (achieved: 96.94%, target: ≥95%)
pmat analyze complexity                       # Max 10 cyclomatic/function
pmat analyze satd                            # Zero TODO/FIXME/HACK

# Tier 4: CI/CD (5-60 min)
cargo mutants --no-times                     # Mutation testing (target: 85%)
pmat tdg . --include-components              # TDG score (target: A+ = 95.0+)
```

## Architecture

### Core Design Patterns

1. **Trait-Based Multiple Dispatch** - Julia-inspired pattern for different data types/backends
2. **Backend Agnostic** - Algorithms transparent to CPU (SIMD), GPU, WASM via Trueno
3. **Three-Tier API**:
   - High: `Estimator` trait (sklearn-like `fit`/`predict`/`score`)
   - Mid: `Optimizer`, `Loss`, `Regularizer` abstractions
   - Low: Direct Trueno primitives

### Dependencies

**Runtime:** `trueno = "0.4.0"` (SIMD-accelerated tensor operations)

**Dev/Quality Tools:**
- `proptest` - Property-based testing (10K+ cases)
- `criterion` - Benchmarking
- `pmat` v2.200.0 - Comprehensive quality analysis (see PMAT section below)
- `renacer` - Profiling
- `cargo-mutants` - Mutation testing

### Banned Dependencies
serde, rayon, tokio, thiserror, ndarray, polars, arrow - see spec for rationale

## Testing Strategy

Target distribution: 60% unit, 30% property tests, 10% integration

### Test Execution

```bash
cargo test                              # All tests (742 unit + property + doc tests)
cargo test --lib                        # Unit tests only
cargo test --test integration           # Integration tests
cargo test --test property_tests        # Property-based tests (proptest)
cargo test --doc                        # Doctests
```

### Coverage Analysis

**Current Achievement: 96.94% line coverage** (Target: ≥95%)

```bash
# Generate coverage report (recommended)
make coverage                           # Full coverage with HTML + lcov output

# Manual coverage commands
cargo llvm-cov --all-features --workspace --lcov --output-path lcov.info
cargo llvm-cov report --html --output-dir target/coverage/html

# View results
open target/coverage/html/html/index.html  # macOS
xdg-open target/coverage/html/html/index.html  # Linux

# Coverage summary
cargo llvm-cov report | grep TOTAL
```

**Coverage by Module:**
- optim/mod.rs: 100.00%
- loss/mod.rs: 99.79%
- graph/mod.rs: 99.62%
- classification/mod.rs: 98.71%
- All other modules: >92%
- **TOTAL: 96.94% line, 95.46% region, 96.62% function**

**CI Integration:**
- Automated coverage reports on every PR
- Codecov integration with PR comments
- Configured in `.github/workflows/ci.yml` (coverage job)
- Target thresholds: 95% project, 90% patch (codecov.yml)

### Mutation Testing

**Current Setup:** Integrated in CI pipeline (`.github/workflows/ci.yml`)

```bash
# CI automatically runs mutation tests on every PR/push
# Results stored as artifacts for 30 days

# View CI results
gh run list --workflow=ci.yml --limit 5
gh run download <run-id> -n mutants-results

# Local execution (when working):
cargo mutants --no-times --timeout 300 --in-place -- --all-features

# Test specific file:
cargo mutants --no-times --timeout 120 --file src/loss/mod.rs

# List mutants without running:
cargo mutants --list --file src/loss/mod.rs
```

**Known Issue:** Local mutation testing may fail with package ambiguity errors for published crates. Use CI for mutation testing or temporarily bump version for local testing.

**Mutation Stats:**
- Total mutants: ~13,705 across entire codebase
- Target mutation score: ≥80% (PMAT recommendation)
- CI timeout: 300s per mutant
- Results: Available as CI artifacts for 30 days

**Configuration:** See `mutation-testing-setup.md` for detailed documentation.

## Linting Configuration

Aprender enforces high-quality code standards through comprehensive **workspace-level** Rust and Clippy lints configured in `Cargo.toml`.

### Workspace-Level Lints (GH-42)

As of v0.4.1, lints are defined at workspace level for consistency and future multi-crate support:
- `[workspace.lints.rust]` - Workspace-wide Rust lints
- `[workspace.lints.clippy]` - Workspace-wide Clippy lints
- `[lints] workspace = true` - Package inherits workspace lints

This approach provides:
- Centralized lint configuration
- Consistent enforcement across all crates
- Easy addition of future crates to workspace
- Improved PMAT Code Quality score

### Rust Lints

**Safety:**
- `unsafe_code = "forbid"` - Zero tolerance for unsafe code
- `unsafe_op_in_unsafe_fn = "warn"` - Enforce safety blocks

**Code Quality:**
- `unreachable_pub = "warn"` - Public items must be actually reachable
- `missing_debug_implementations = "warn"` - All public types must be debuggable

**Best Practices:**
- `rust_2018_idioms = "warn"` - Enforce modern Rust idioms
- `trivial_casts/trivial_numeric_casts = "warn"` - Remove unnecessary casts
- `unused_*` - Warn on unused imports, lifetimes, qualifications

### Clippy Lints

**Base Levels:**
- `all = "warn"` - All clippy lints enabled
- `pedantic = "warn"` - Strict code quality checks

**High-Priority:**
- `checked_conversions = "warn"` - Use checked arithmetic
- `inefficient_to_string = "warn"` - Optimize string conversions
- `redundant_closure_for_method_calls = "warn"` - Simplify closures

**ML-Specific Allows:**
- `cast_*` - Allow numeric conversions common in ML algorithms
- `float_cmp` - Allow float comparisons (with proper epsilon checks)
- `many_single_char_names` - Allow mathematical notation (x, y, z, i, j)
- `unreadable_literal` - Allow long test data literals

### Running Lints

```bash
cargo clippy                    # Check all lints
cargo clippy -- -D warnings     # Fail on any warning (CI mode)
cargo fmt --check               # Verify formatting
```

**Current State:** ~140 pedantic warnings (mostly style improvements like format string inlining). Core production code is lint-clean.

## CI/CD Workflows

Aprender uses GitHub Actions for comprehensive CI/CD automation in `.github/workflows/`:

### ci.yml - Main CI Pipeline
Runs on every push and PR to main:
- **Check**: `cargo check --all-features`
- **Format**: `cargo fmt --all -- --check`
- **Clippy**: `cargo clippy -- -D warnings` (zero tolerance)
- **Test**: Unit, integration, property, doc, and full test suite
- **Coverage**: `cargo llvm-cov` with Codecov upload (achieved: 96.94%, target: ≥95%)
- **Mutation Testing**: `cargo mutants` (sample run, continue-on-error)
- **Security**: `cargo audit` and `cargo deny` checks
- **Docs**: `cargo doc` with -Dwarnings
- **Shellcheck**: Script linting
- **Build**: Release build + example runs

### benchmark.yml - Performance Monitoring
Runs on manual trigger, PR (for performance-sensitive changes), and weekly schedule:
- **Benchmarks**: All criterion benchmarks (linear_regression, kmeans, dataframe)
- **Artifacts**: Results stored for 90 days with historical tracking
- **PR Comments**: Automatic benchmark result summaries on PRs
- **Triggers**:
  - Manual: `workflow_dispatch` for on-demand runs
  - PR: When `src/**/*.rs` or `benches/**/*.rs` changes
  - Schedule: Weekly on Sundays at 2 AM UTC

**Running benchmarks locally:**
```bash
cargo bench                     # Run all benchmarks
cargo bench --bench kmeans      # Run specific benchmark
```

**Triggering in CI:**
1. Go to Actions → Benchmarks → Run workflow
2. Or wait for weekly scheduled run
3. Or modify performance-sensitive code and create PR

### security.yml - Dependency Security & Policy
Runs on weekly schedule (Mondays 3 AM UTC), PR (dependency changes), and manual trigger:
- **cargo-audit**: Scans for known security vulnerabilities (CVEs) in dependencies
- **cargo-deny**: Enforces dependency policies from deny.toml:
  - License compliance (only approved licenses)
  - Banned crates (security/maintenance concerns)
  - Source verification (crates.io only)
- **cargo-outdated**: Reports outdated dependencies for proactive updates
- **Artifacts**: Outdated dependency reports stored for 30 days

**Running security checks locally:**
```bash
cargo audit                     # Check for vulnerabilities
cargo deny check                # Validate dependency policies
cargo outdated                  # List outdated dependencies
```

### dependabot.yml - Automated Dependency Updates
- **Rust dependencies**: Weekly updates (Mondays 3 AM UTC)
  - Groups minor/patch updates to reduce PR noise
  - Separate PRs for major version updates
- **GitHub Actions**: Monthly updates for workflow actions
- Automatic labeling: `dependencies`, `rust`, `github-actions`
- Auto-assignment to `paiml/aprender-maintainers` team

### book.yml - Documentation CI
Builds and deploys the EXTREME TDD book to GitHub Pages.

### release.yml - Release Automation
Automated releases on version tags.

## v0.4.0 - TOP 10 ML Algorithms Complete

**Supervised Learning:**
- Linear Regression (OLS)
- Logistic Regression (gradient descent)
- Decision Tree Classifier (GINI impurity)
- Random Forest Classifier (bootstrap aggregating)
- Gradient Boosting Machine (adaptive boosting)
- Naive Bayes (Gaussian)
- K-Nearest Neighbors (distance-based)
- Support Vector Machine (linear kernel)

**Unsupervised Learning:**
- K-Means (Lloyd's + k-means++ initialization)
- PCA (dimensionality reduction via eigendecomposition)

**Model Selection:** train_test_split, KFold, cross_validate
**Persistence:** serde + bincode serialization
**Metrics:** R², MSE, RMSE, MAE, accuracy, silhouette, inertia

## Key Files

- `src/lib.rs` - Library entry point with module exports
- `src/traits.rs` - Core traits (Estimator, UnsupervisedEstimator, Transformer)
- `src/primitives/` - Vector and Matrix types with Cholesky solver
- `src/data/mod.rs` - DataFrame implementation (~250 LOC)
- `src/linear_model/mod.rs` - Linear regression with OLS
- `src/cluster/mod.rs` - K-Means with k-means++ initialization
- `src/metrics/mod.rs` - R², MSE, MAE, inertia, silhouette
- `docs/specifications/aprender-spec-v1.md` - Full v1.0 specification

## Integration

Ruchy integration for Python-like syntax. Data flow:
```
polars::DataFrame → ruchy bridge → aprender::DataFrame → .to_matrix() → trueno primitives
```

## PMAT Quality Analysis (v2.200.0)

Aprender uses PMAT (Professional Maintenance Analysis Tool) v2.200.0 for comprehensive quality analysis and enforcement.

### Current Quality Metrics

**Rust Project Score:** 124.0/134 (92.5%, Grade: A+)
**TDG Score:** 95.2/100 (Grade: A+)
**Test Coverage:** 96.94% line, 95.46% region, 96.62% function (target: ≥95%)
**Test Count:** 742 tests (unit + property + integration + doc)
**Mutation Score:** 85.3% (target: ≥85%)
**Cyclomatic Complexity:** Max 9 (target: ≤10)
**SATD Violations:** 0 (zero tolerance)
**Known Defects:** 0 unwrap() calls (100% score, zero tolerance)

### Key PMAT Commands

```bash
# Quality gates (configured in .pmat-gates.toml)
pmat quality-gates              # Run all quality gates
pmat quality-gates --report     # Generate markdown report
make pmat-gates                 # Makefile shortcut

# Project scoring
pmat rust-project-score         # Comprehensive Rust project analysis
make pmat-score                 # Makefile shortcut

# Code analysis
pmat analyze complexity         # Cyclomatic/cognitive complexity
pmat analyze satd               # Self-admitted technical debt (TODO/FIXME/HACK)
pmat tdg . --include-components # Technical debt grading

# Semantic code search (PMAT-SEARCH-011)
pmat semantic                   # Interactive semantic search
pmat embed                      # Manage search embeddings
make semantic-search            # Makefile shortcut

# Mutation testing (Sprint 61)
pmat mutate <file>              # Run mutation testing on specific file

# Comprehensive reporting
make quality-report             # Generate full quality report
```

### Quality Gates Configuration

Configuration file: `.pmat-gates.toml`

Key thresholds:
- **Test Coverage:** ≥95% (achieved: 96.94% line coverage)
- **Cyclomatic Complexity:** ≤10 per function (current max: 9)
- **SATD Comments:** 0 (zero tolerance)
- **TDG Score:** ≥95.0 (A+ grade required)
- **Mutation Score:** ≥85% (current: 85.3%)
- **Clippy Warnings:** 0 (strict mode: `-D warnings`)
- **Known Defects:** 0 unwrap() calls (eliminated all 1,066 in v0.4.1)

Pre-commit hooks enforce these thresholds automatically.

### Critical Issues Tracked

1. **1,066 unwrap() calls in src/** (Cloudflare-class defect, Issue #41)
   - **Severity:** CRITICAL (P0)
   - **Defect Class:** Cloudflare 2025-11-18 outage class (unwrap panic caused 3+ hour downtime)
   - **Current Count:** 1,066 unwrap() calls in production code (src/)
   - **Target:** 0 unwrap() calls
   - **Timeline:** 6-8 weeks, 80-120 hours effort

   **Top 5 Offenders:**
   - `src/cluster/mod.rs`: 280 unwrap() (26.3%)
   - `src/tree/mod.rs`: 178 unwrap() (16.7%)
   - `src/classification/mod.rs`: 152 unwrap() (14.3%)
   - `src/linear_model/mod.rs`: 136 unwrap() (12.8%)
   - `src/preprocessing/mod.rs`: 121 unwrap() (11.4%)

   **Enforcement:** `.clippy.toml` configured with `disallowed-methods` to ban unwrap()

   **Remediation Strategy:**
   ```rust
   // ❌ DANGEROUS: Can panic and crash the process
   let value = some_option.unwrap();

   // ✅ ACCEPTABLE: Descriptive error message
   let value = some_option.expect("User configuration must have name field");

   // ✅ BEST: Proper error handling
   let value = some_option.ok_or_else(|| AprenderError::MissingField("name"))?;
   ```

   **Testing Enforcement:**
   ```bash
   # This will now FAIL due to .clippy.toml disallowed-methods
   cargo clippy -- -D clippy::disallowed-methods
   ```

   See Issue #41 for complete remediation plan.

### PMAT Workflow Integration

**Pre-Commit (Tier 2):**
```bash
# Automated via git hooks
pmat analyze complexity --max-cyclomatic 10
pmat analyze satd --max-count 0
cargo fmt --check
cargo clippy -- -D warnings
cargo test --lib
```

**Pre-Push (Tier 3):**
```bash
make tier3                      # Full validation
cargo test --all
make coverage                   # Coverage check (96.94% achieved)
```

**CI/CD (Tier 4):**
```bash
make tier4                      # Heavyweight analysis
pmat tdg . --include-components
pmat rust-project-score
pmat quality-gates --report
cargo mutants --no-times        # Mutation testing
```

### Semantic Search Usage

PMAT v2.200.0 includes semantic code search for finding similar code patterns:

```bash
# First run (builds embeddings, ~2-3 minutes)
pmat embed                      # Build embeddings

# Interactive search
pmat semantic                   # Launch interactive search
# Or via Makefile
make semantic-search

# Example queries:
# - "error handling patterns"
# - "matrix multiplication implementations"
# - "test fixtures for ML models"
```

### Resources

- **Configuration:** `.pmat-gates.toml`
- **Reports:** `docs/quality-reports/latest.md` (auto-generated)
- **PMAT Documentation:** https://github.com/paiml/pmat
- **Toyota Way Standards:** See `aprender-spec-v1.md`
