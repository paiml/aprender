# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## CRITICAL: Contract-First Design

**NEVER write code before writing a provable contract.**

All code changes MUST have a corresponding contract (YAML in ../provable-contracts/contracts/<project>/ or .pmat-work/<TICKET>/contract.json) BEFORE implementation. This is enforced by `pmat comply` CB-1400.

- Use `pmat comply check` to verify contract coverage
- Minimum verification level: L1 (recommended L3+)
- See docs/agent-instructions/provable-contract-first-agents.md for the full workflow

## Project Overview

**certeza** is a scientific experiment into realistic provability with Rust. This is a research project developing a comprehensive framework for approaching asymptotic test effectiveness in Rust software systems.

### Core Concept
The project explores achieving practical maximum confidence in software testing through tiered verification approaches, acknowledging that complete verification is theoretically impossible (Dijkstra's observation: "testing can only prove the presence of bugs, not their absence").

### Reference Implementation
The framework targets vector-based data structures using the **trueno project** (https://github.com/paiml/trueno) as a reference implementation, with the **paiml-mcp-agent-toolkit (PMAT)** (https://github.com/paiml/paiml-mcp-agent-toolkit) for test orchestration.

## Current Project State

**Status**: Active development with PMAT compliance
- Rust library project scaffolded with full testing framework
- Contains comprehensive testing framework specification (~14K words)
- PMAT-compliant configuration and quality gates implemented
- Example functions with unit tests and property-based tests

## Testing Philosophy: Tiered TDD-X Framework

This project implements a three-tiered testing approach that balances rigor with developer productivity:

### Tier 1: ON-SAVE (Sub-second feedback)
- Unit tests and focused property tests
- Static analysis (`cargo check`, `cargo clippy`)
- Enables rapid iteration in flow state

### Tier 2: ON-COMMIT (1-5 minutes)
- Full property-based test suite with proptest
- Coverage analysis (target: 95%+ line coverage)
- Integration tests
- Pre-commit hook enforcement

### Tier 3: ON-MERGE/NIGHTLY (Hours)
- Comprehensive mutation testing with cargo-mutants (target: >85% mutation score)
- Formal verification for critical paths (using Kani)
- Performance benchmarks
- CI/CD gate for main branch

**Critical Principle**: Different verification techniques operate at different time scales. Fast feedback enables flow; slow feedback causes context switching waste. Never run mutation testing or formal verification in the inner development loop.

## Testing Pyramid Distribution

```
┌─────────────────┐
│  Formal (Kani)  │  ~1-5% code (invariant proofs)
├─────────────────┤
│   Integration   │  ~10% tests (system properties)
├─────────────────┤
│  Property-Based │  ~30% tests (algorithmic correctness)
├─────────────────┤
│   Unit Tests    │  ~60% tests (basic functionality)
└─────────────────┘
```

## Risk-Based Verification Strategy

Not all code requires the same verification intensity. Apply rigorous techniques based on risk:

| Risk Level | Components | Verification Approach |
|------------|------------|----------------------|
| **Very High** | `unsafe` blocks, memory allocators, crypto, concurrency primitives | Full framework: Property + Coverage + Mutation (90%) + Formal |
| **High** | Core algorithms, data structure internals, parsers | Property + Coverage + Mutation (85-90%) |
| **Medium** | Business logic, API handlers, utilities | Property + Coverage + Mutation (80%) |
| **Low** | Simple accessors, config, CLI parsing | Unit tests + Coverage (90%) |

**Resource Allocation**: Spend 40% of verification time on the 5-10% highest-risk code.

## Expected Cargo Commands

When Rust code is implemented, the project will use:

### Development
- `cargo check` - Type checking (Tier 1, sub-second)
- `cargo clippy` - Linting (Tier 1, sub-second)
- `cargo test` - Run unit tests (Tier 1, sub-second for focused tests)
- `cargo test --all` - Run full test suite (Tier 2, 1-5 min)

### Coverage Analysis (Tier 2)
- `cargo tarpaulin` or `cargo llvm-cov` - Generate coverage reports
- Target: 95%+ line coverage

### Mutation Testing (Tier 3)
- `cargo mutants` - Run mutation testing
- Target: >85% mutation score
- Analyze surviving mutants for test gaps

### Formal Verification (Tier 3)
- `cargo kani` - Formal verification for critical invariants
- Applied selectively to highest-risk code paths

### Property-Based Testing
Uses **proptest** crate for property-based testing (see specification for detailed examples)

## PMAT Compliance

This project is fully compliant with the **Pragmatic AI Labs Multi-Language Agent Toolkit (PMAT)** standards.

### Makefile Targets (PMAT-Aligned)

The project uses a comprehensive Makefile for all quality operations:

**Tiered Workflow:**
- `make tier1` - Tier 1: ON-SAVE checks (sub-second)
- `make tier2` - Tier 2: ON-COMMIT checks (1-5 minutes)
- `make tier3` - Tier 3: ON-MERGE/NIGHTLY checks (hours)

**Quality Gates:**
- `make quality-gate` - Run all PMAT quality gates
- `make quality-gate-tier2` - Tier 2 quality gates (default for commits)
- `make quality-gate-tier3` - Tier 3 quality gates (pre-merge)

**Testing:**
- `make test` - Run all tests
- `make test-quick` - Run unit tests only (fast)
- `make test-property` - Run property-based tests
- `make coverage` - Generate coverage report (target: 85%+)
- `make mutation` - Run mutation testing (target: 85%+ score)

**Code Quality:**
- `make clippy` - Run clippy linter
- `make clippy-strict` - Run clippy with pedantic/nursery lints
- `make fmt` - Format code
- `make fmt-check` - Check formatting

**Analysis:**
- `make complexity` - Analyze code complexity with PMAT
- `make tdg` - Technical debt grading
- `make security` - Security audit (cargo-audit + cargo-deny)
- `make repo-score` - Calculate repository health score

**Documentation:**
- `make docs` - Generate documentation
- `make validate-docs` - Validate documentation with PMAT

**Setup:**
- `make install-tools` - Install all required tooling
- `make install-hooks` - Install PMAT git hooks

### PMAT Configuration Files

The project includes three PMAT configuration files:

1. **pmat.toml** - Main PMAT configuration
   - Complexity limits: max_cyclomatic=10, max_cognitive=10
   - Coverage requirements: min_coverage=85%
   - SATD: zero tolerance (max_satd=0)
   - Mutation testing: min_mutation_score=85%
   - Documentation: min_rustdoc_coverage=90%

2. **.pmat-gates.toml** - Quality gate enforcement
   - Clippy strict mode enabled
   - Rustfmt checking enabled
   - Coverage threshold: 85%
   - Complexity checking enabled
   - Security audits (cargo-audit, cargo-deny)
   - SATD checking with zero tolerance

3. **pmat-quality.toml** - Detailed quality thresholds
   - Tiered testing configuration aligned with certeza spec
   - Component-level grading thresholds
   - Risk-based verification settings

### PMAT Quality Standards (EXTREME TDD)

The project enforces **EXTREME TDD** standards:

**Coverage Requirements:**
- Line coverage: ≥85% (minimum), 95% (target)
- Branch coverage: ≥80% (minimum), 90% (target)
- Function coverage: ≥90%

**Complexity Limits:**
- Cyclomatic complexity: ≤10 per function
- Cognitive complexity: ≤10 per function
- Nesting depth: ≤5
- Lines per function: ≤50

**Testing Requirements:**
- Minimum 20 unit tests
- Minimum 10 integration tests
- Minimum 5 property-based tests
- Proptest iterations: 256-10,000

**SATD (Self-Admitted Technical Debt):**
- Zero tolerance for TODO, FIXME, HACK comments
- All technical debt must link to GitHub issues
- Fail build on unlinked SATD

**Security:**
- cargo-audit: deny vulnerabilities
- cargo-deny: deny unmaintained/deprecated dependencies
- Unsafe code: max_unsafe_blocks=0 (forbid unsafe)

**Documentation:**
- ≥90% public items documented
- All public functions require examples
- Module and crate documentation required
- Safety documentation for any unsafe code (≥3 lines)

### CI/CD Integration

GitHub Actions workflow (`.github/workflows/ci.yml`) enforces quality gates:

- **Tier 1**: Quick checks on every push (check, clippy, unit tests)
- **Tier 2**: Full test suite + coverage on PR (all tests, coverage ≥85%)
- **Security**: Parallel security audit (cargo-audit, cargo-deny)
- **Tier 3**: Mutation testing on merge to main (≥85% mutation score)

### PMAT Commands

Use PMAT directly for advanced analysis:

```bash
# Generate AI-ready context
pmat context --output context.md --format llm-optimized

# Analyze technical debt
pmat analyze tdg --include-components

# Check complexity
pmat analyze complexity --path src/

# Repository health score (0-110 scale)
pmat repo-score .
pmat repo-score . --deep  # Include git history

# Run mutation testing
pmat mutate --target src/ --threshold 85

# Validate documentation accuracy
pmat validate-readme --targets README.md

# Install pre-commit hooks
pmat hooks install
pmat hooks status

# Run quality gates
pmat quality-gates
```

### Project Scoring (Rust Project Score)

PMAT evaluates the project across 6 dimensions (total: 100 points):

1. **Rust Tooling Compliance** (25 points): Clippy, rustfmt, cargo-deny, cargo-audit
2. **Code Quality** (20 points): Complexity, unsafe code, dead code, SATD
3. **Testing Excellence** (20 points): Unit tests, integration tests, property tests, mutation tests
4. **Documentation** (15 points): Rustdoc coverage, examples, architecture docs
5. **Performance & Security** (10 points): Benchmarks, security analysis
6. **Community & DevOps** (10 points): CI/CD, release process

**Target Grade**: A (90-94) or A+ (95-100)

## Architecture Insights

### Testing Framework Components

1. **Structural Coverage**: Instrumentation-based measurement of code execution
2. **Property-Based Testing**: Specification verification using proptest strategies
3. **Mutation Testing**: Test suite quality assessment (detect test gaps)
4. **Selective Formal Verification**: Mathematical proofs for critical invariants

### Key Design Principles

- **Sustainable Workflows**: Tiered feedback loops prevent burnout and maintain flow state
- **Risk-Based Resource Allocation**: Focus expensive verification on high-risk components
- **Human-Centered Analysis**: Mutation analysis as learning exercise, not just metrics
- **Economic Realism**: Acknowledge costs and diminishing returns of verification techniques

### Theoretical Bounds

The specification acknowledges fundamental limits:
- Coverage ceiling: 100% coverage doesn't guarantee correctness
- Mutation score asymptote: Typically plateaus at 80-95% (equivalent mutants are undecidable)
- Property space incompleteness: Infinite meaningful properties, finite testing
- Formal verification tractability: State explosion limits verification scope

## Documentation Structure

- `docs/specifications/theoretical-max-testing-spec.md` - Main framework specification (v1.1, ~14K words)
- `docs/specifications/IMPROVEMENTS_v1.1.md` - Changelog showing philosophy shift from "theoretical maximum" to "asymptotic effectiveness"
- `docs/specifications/scientific-reporting-benchmarking-spec.md` - Scientific benchmarking framework (v1.0, ~12.5K words)
- `ROADMAP.md` - Project roadmap and implementation phases

## Scientific Benchmarking Framework

**Status**: Phase 3 implementation in progress (see ROADMAP.md)

This project includes a comprehensive scientific benchmarking framework for reproducible performance measurement and reporting. The framework emphasizes statistical rigor, multi-format reporting, and integration with the tiered testing philosophy.

### Benchmarking Philosophy

Performance is a quality attribute that requires the same rigor as functional correctness. Performance regressions are bugs. Scientific benchmarking provides the evidence to prevent, detect, and fix them systematically.

**Key Principles**:
1. **Reproducibility First**: Complete environmental metadata and toolchain pinning
2. **Statistical Rigor**: Confidence intervals, significance testing, effect sizes
3. **Transparency**: Full disclosure of methodology and negative results
4. **Fitness for Purpose**: Multiple report formats for different audiences
5. **Integration with Quality Gates**: Performance gates in CI/CD pipelines

### Tiered Benchmarking

Benchmarks align with the three-tier testing framework:

**Tier 1: ON-SAVE**
- Not applicable (benchmarks require release builds)
- Alternative: Smoke tests verify benchmark binaries compile

**Tier 2: ON-COMMIT (1-5 Minutes)**
- Quick regression check with critical benchmarks only
- Threshold gates: Fail commit if >10% slower than baseline
- Integration: Pre-commit hook via PMAT
- Tools: bashrs with 3 warmup + 10 measured iterations

**Tier 3: ON-MERGE/NIGHTLY (Hours)**
- Comprehensive benchmark suite across all optimization profiles
- Statistical analysis and full reporting pipeline
- Integration: GitHub Actions CI/CD workflow
- Tools: bashrs with 5 warmup + 20 measured iterations

### Makefile Targets (Benchmarking)

**Quick Commands:**
```bash
# Run critical benchmarks (Tier 2: ~5 min)
make benchmark

# Run comprehensive suite (Tier 3: ~30 min)
make benchmark-all

# Generate all report formats (JSON, CSV, Markdown, HTML)
make benchmark-report

# Compare against baseline (regression detection)
make benchmark-compare

# Save current as new baseline
make benchmark-baseline-save

# Clean benchmark artifacts
make benchmark-clean
```

**Full Workflow Example:**
```bash
# 1. Run benchmarks
make benchmark-all

# 2. Generate reports
make benchmark-report

# 3. Compare against baseline
make benchmark-compare

# 4. If no regressions, update baseline
make benchmark-baseline-save --name=v0.2.0
```

### Expected Benchmark Commands

```bash
# Run critical benchmarks (Tier 2)
./scripts/run_benchmarks.sh \
    --benchmarks critical \
    --output benchmarks/quick_results.json \
    --warmup 3 \
    --iterations 10

# Run comprehensive suite (Tier 3)
./scripts/run_benchmarks.sh \
    --benchmarks all \
    --profiles all \
    --output benchmarks/comprehensive_results.json \
    --warmup 5 \
    --iterations 20

# Compare against baseline
python3 scripts/check_regression.py \
    --baseline benchmarks/baseline.json \
    --current benchmarks/latest.json \
    --max-regression 10.0

# Generate all report formats
python3 scripts/generate_report.py \
    --input benchmarks/comprehensive_results.json \
    --format all \
    --output benchmarks/reports/
```

### Report Formats

The framework generates five output formats from a single measurement run:

1. **JSON** (machine-readable): Complete structured data for archival and programmatic analysis
2. **CSV** (spreadsheet-compatible): Tabular data for R, Python pandas, Excel
3. **Markdown** (human-readable): GitHub documentation and technical blogs
4. **LaTeX** (publication-quality): IEEE/ACM formatted tables for academic papers
5. **HTML** (interactive dashboard): Chart.js visualizations with drill-down capabilities

### Statistical Methodology

**Measurement Protocol**:
- Warmup phase (3-5 iterations) to eliminate cold-start effects
- Measured iterations (10-20) with adaptive stopping based on CV
- IQR-based outlier detection and removal
- Normality testing (Shapiro-Wilk) to select appropriate statistics

**Comparative Analysis**:
- Welch's t-test (parametric) or Mann-Whitney U (non-parametric)
- Effect size calculation (Cohen's d)
- Bootstrap confidence intervals for speedup ratios
- Significance threshold: α = 0.05

**Quality Metrics**:
- Coefficient of variation (CV) < 10% for reproducibility
- Statistical power analysis for regression detection
- Change-point detection for long-running time series

### Reproducibility Mechanisms

**Toolchain Pinning**:
```toml
# rust-toolchain.toml
[toolchain]
channel = "1.75.0"
components = ["rustfmt", "clippy", "rust-src"]
targets = ["x86_64-unknown-linux-gnu"]
profile = "minimal"
```

**Dependency Locking**:
- Exact version pinning in Cargo.toml (not semver ranges)
- Cargo.lock committed to repository
- SHA256 validation of source tree and dependencies

**Containerized Builds**:
- Multi-stage Dockerfile with pinned Rust toolchain
- Hermetic build environment isolating from host system
- Byte-identical build verification across environments

**Metadata Capture**:
- Complete hardware specifications (CPU, memory, storage)
- Software environment (OS, kernel, Rust version, LLVM version)
- Runtime configuration (CPU governor, turbo boost, isolated cores)

### Performance Quality Gates

**.pmat-gates.toml Configuration**:
```toml
[performance]
enabled = true
tier = "tier2"
max_regression_percent = 10.0
min_improvement_percent = 3.0
baseline_file = "benchmarks/baseline.json"
critical_benchmarks = [
    "vector_push_capacity_growth",
    "vector_iteration_sum",
    "vector_binary_search"
]
```

**Pre-Commit Hook Behavior**:
- Runs critical benchmarks automatically on commit
- Blocks commit if performance regression >10%
- Provides actionable feedback: "Run `make benchmark-analyze` for details"

**CI/CD Integration**:
- PR benchmarks (Tier 2): Run on every pull request, comment results
- Nightly benchmarks (Tier 3): Run comprehensive suite, publish reports
- Artifact publishing: Upload to GitHub Pages and Zenodo

### Benchmark Design Guidelines

**Good Benchmarks**:
- Measure single, well-defined operation
- Stable runtime (CV < 10%)
- Representative of real-world usage
- Isolated from environmental noise

**Benchmark Categories**:
- **CPU-bound**: Fibonacci recursion, prime sieve, Ackermann function
- **Memory-intensive**: Matrix multiplication, quicksort, large allocations
- **I/O-bound**: File operations, network calls, serialization

**Anti-Patterns**:
- Benchmarking debug builds (use `--release` always)
- Single-measurement anecdotes (use statistical sampling)
- Ignoring warmup (JIT, cache population, OS resource allocation)
- Comparing across different hardware without normalization

### Optimization Profile Matrix

Explore performance/size tradeoffs systematically:

```toml
# Cargo.toml profiles
[profile.dev]
opt-level = 0  # Fast compilation, slow runtime

[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1

[profile.release-size]
opt-level = "z"  # Optimize for binary size
lto = "fat"
strip = true
```

**Pathfinder Algorithm**: Reduces combinatorial explosion of optimization flags from 800+ configurations to ~150 targeted experiments while maintaining 98.6% confidence in identifying optimal profiles.

### Artifact Archival

**Zenodo Integration**:
- Publish complete benchmarking artifacts with DOI
- Includes: source code, results, metadata, reproduction scripts
- Enables long-term citation and independent replication

**Archive Structure**:
```
certeza-benchmark-artifact-v1.0.0.tar.gz
├── README.md                    # Reproduction instructions
├── src/                         # Complete source code
├── Cargo.toml & Cargo.lock     # Dependency specifications
├── rust-toolchain.toml         # Toolchain pin
├── benchmarks/
│   ├── results.json            # Raw measurement data
│   ├── results.csv             # Tabular export
│   ├── report.md               # Human-readable report
│   └── metadata/               # Hardware/software specs
└── scripts/
    ├── run_benchmarks.sh       # Execution protocol
    └── validate_reproduction.sh # Verification script
```

### Reference Implementations

**Methodology Sources**:
- [compiled-rust-benchmarking](https://github.com/paiml/compiled-rust-benchmarking): Pathfinder optimization algorithm
- [ruchy-docker](https://github.com/paiml/ruchy-docker): Containerized benchmarking with bashrs
- [ruchy-lambda](https://github.com/paiml/ruchy-lambda): Serverless cold-start performance measurement

**Statistical Tools**:
- bashrs: Statistical command-line benchmarking tool
- scipy/numpy: Python statistical analysis libraries
- Chart.js: Interactive web-based visualizations

## Benchmarking Best Practices

### Development Workflow

**When to Benchmark:**
- Before/after performance optimizations
- Before major releases
- When investigating performance regressions
- During code reviews for performance-critical PRs

**Local Benchmarking:**
```bash
# Quick sanity check (Tier 2)
make benchmark

# Before committing optimization
make benchmark-compare

# Full validation before PR
make benchmark-all && make benchmark-report
```

**Interpreting Results:**
- **CV < 5%**: Excellent stability, trust the measurements
- **CV 5-10%**: Acceptable, but investigate outliers
- **CV > 10%**: Unstable, check system load, thermal throttling

### CI/CD Integration

The project includes automated benchmarking workflows:

**On Pull Request** (`.github/workflows/benchmarks.yml`):
- Runs critical benchmarks automatically
- Comments regression results on PR
- Fails build if >10% slowdown detected
- Statistical significance required (p < 0.05, Cohen's d ≥ 0.2)

**On Merge to Main**:
- Updates baseline automatically
- Generates comprehensive reports
- Publishes to benchmarks/baselines/main.json

**Weekly Scheduled**:
- Tracks long-term performance trends
- Saves snapshots to benchmarks/history/

### Reproducibility

**For Exact Reproduction:**
```bash
# Use Docker for hermetic builds
docker build -t certeza:reproducible .
docker run --rm -v $(pwd)/benchmarks:/app/benchmarks certeza:reproducible

# Validate statistical equivalence
./scripts/validate_reproduction.sh baseline.json reproduced.json
```

**Metadata Requirements:**
All benchmark results include complete environmental metadata:
- Hardware: CPU model, cores, frequency, memory
- Software: OS, kernel, rustc, cargo, LLVM versions
- Configuration: CPU governor, turbo boost, swap status
- Git: commit hash, branch name

### Report Formats

The framework generates 5 output formats from each benchmark run:

1. **JSON** (`benchmarks/results/latest.json`):
   - Machine-readable, complete structured data
   - Schema version 1.0
   - Use for programmatic analysis and archival

2. **CSV** (`benchmarks/results/report.csv`):
   - Spreadsheet-compatible tabular data
   - Import to R, Python pandas, Excel
   - Single-file or multi-file modes

3. **Markdown** (`benchmarks/results/report.md`):
   - GitHub-flavored markdown
   - Human-readable, suitable for documentation
   - Includes statistical methodology

4. **HTML** (`benchmarks/results/dashboard.html`):
   - Interactive Chart.js visualizations
   - Self-contained, open in browser
   - Performance trends and distributions

5. **LaTeX** (future):
   - Publication-quality tables
   - IEEE/ACM paper formatting

**Generating Reports:**
```bash
# All formats at once
make benchmark-report

# Individual formats
deno run --allow-read --allow-write scripts/generate_csv_report.ts input.json output.csv
deno run --allow-read --allow-write scripts/generate_markdown_report.ts input.json report.md
deno run --allow-read --allow-write scripts/generate_dashboard.ts input.json dashboard.html
```

### Baseline Management

**Save Baseline:**
```bash
# Via Makefile
make benchmark-baseline-save --name=v1.0.0

# Via script
deno run --allow-read --allow-write scripts/baseline_manager.ts save \
    --input benchmarks/results/latest.json \
    --name v1.0.0 \
    --description "Release 1.0.0 baseline"
```

**List Baselines:**
```bash
deno run --allow-read scripts/baseline_manager.ts list
```

**Compare Against Baseline:**
```bash
# Detect regressions
deno run --allow-read --allow-write scripts/check_regression.ts \
    --baseline benchmarks/baselines/v1.0.0.json \
    --current benchmarks/results/latest.json \
    --max-regression 10.0

# Exit codes:
# 0 = No regressions
# 1 = Warning (5-10% slower)
# 2 = Critical (>10% slower)
# 3 = Error
```

### Troubleshooting Performance Variance

**High Coefficient of Variation (CV > 10%)**:

1. **Check CPU Governor:**
   ```bash
   # Linux
   cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
   sudo cpupower frequency-set --governor performance
   ```

2. **Disable Turbo Boost** (for consistency):
   ```bash
   # Intel
   echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo

   # AMD
   echo 0 | sudo tee /sys/devices/system/cpu/cpufreq/boost
   ```

3. **Close Background Processes:**
   ```bash
   # Check system load
   top
   htop

   # Stop unnecessary services
   systemctl stop <service-name>
   ```

4. **Increase Iterations:**
   ```bash
   # More iterations reduce variance
   ./scripts/run_benchmarks.sh --warmup 10 --iterations 50
   ```

### Archival and Publication

**Zenodo Integration:**

The project includes `.zenodo.json` for DOI assignment:

```bash
# Prepare archive
git archive --format=tar.gz HEAD > certeza-v0.1.0.tar.gz

# Include benchmark artifacts
tar -czf certeza-benchmarks-v0.1.0.tar.gz benchmarks/

# Upload to Zenodo (manual or via API)
# Zenodo will assign DOI for permanent citation
```

**Citation:**
See `.zenodo.json` for complete metadata. Generated DOI enables academic citation.

## Development Anti-Patterns

Based on the specification's emphasis on sustainable practices:

1. **Never** run mutation testing on every file save (destroys flow, 10-100x productivity loss)
2. **Never** chase metrics without understanding (Goodhart's Law warning)
3. **Never** apply full verification framework to low-risk code (over-processing waste)
4. **Never** ignore cognitive load limits (use batching, time-boxing, pairing for mutation analysis)
5. **Never** benchmark debug builds (always use `--release`)
6. **Never** trust single-measurement anecdotes (use statistical sampling with n ≥ 10)
7. **Never** compare benchmarks across different hardware without normalization

## Quality Standards

When implementing code:
- Strong type safety leveraging Rust's ownership model
- Memory safety violations prevented by language (focus testing on algorithmic correctness)
- Scientific rigor with empirical validation of testing approaches
- Comprehensive documentation with academic-style citations


## Stack Documentation Search

Query this component's documentation and the entire Sovereign AI Stack using batuta's RAG Oracle:

```bash
# Index all stack documentation (run once, persists to ~/.cache/batuta/rag/)
batuta oracle --rag-index

# Search across the entire stack
batuta oracle --rag "your question here"

# Examples
batuta oracle --rag "SIMD matrix multiplication"
batuta oracle --rag "how to train a model"
batuta oracle --rag "tokenization for BERT"

# Check index status
batuta oracle --rag-stats
```

The RAG index includes CLAUDE.md, README.md, and source files from all stack components plus Python ground truth corpora for cross-language pattern matching.

Index auto-updates via post-commit hooks and `ora-fresh` on shell login.
To manually check freshness: `ora-fresh`
To force full reindex: `batuta oracle --rag-index --force`
