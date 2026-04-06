# Contributing to Trueno-Viz

## The Lean-Scientific Code Review Protocol

Trueno-Viz follows the **Lean-Scientific Code Review Protocol (LSCRP)**, which synthesizes Toyota Production System principles with scientific peer review standards. This is not bureaucracy—it is mindful process designed to eliminate waste and build quality in.

### Core Principles

| Toyota Principle | Software Application |
|------------------|---------------------|
| **Genchi Genbutsu** | Go and see the running code, not just the diff |
| **Jidoka** | Stop the line on quality issues—every engineer has this authority |
| **Kaizen** | Continuously improve process through retrospection |
| **Muda Elimination** | Remove waste: waiting, overprocessing, defects |

---

## Development Setup

### Prerequisites

```bash
# Install Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install PMAT (quality enforcement)
cargo install pmat

# Clone repository
git clone https://github.com/paiml/trueno-viz.git
cd trueno-viz
```

### Environment Verification

```bash
# Verify toolchain
rustc --version    # >= 1.75.0
cargo --version
pmat --version     # >= 2.200.0

# Run diagnostics
pmat diagnose --full
```

### Initial Quality Baseline

```bash
# Generate quality baseline (first-time setup)
make pmat-baseline

# Verify project health
make pmat-score
```

---

## Git Workflow

### CRITICAL: Master-Only Development

**Zero tolerance for feature branches.** All work happens directly on `master`:

```bash
# Verify you're on master
git branch --show-current  # MUST output: master

# If not on master, return immediately
git checkout master
```

### Commit Protocol

1. **Pre-commit validation** (automatic via hooks):
   - O(1) quality gates (<30ms)
   - TDG regression check
   - Clippy linting

2. **Commit message format**:
   ```
   <type>(<scope>): <description>

   [optional body]

   [optional footer]
   ```

   Types: `feat`, `fix`, `refactor`, `test`, `docs`, `perf`, `chore`

3. **Push directly to master**:
   ```bash
   git push origin master
   ```

---

## Quality Standards

### Minimum Thresholds (Enforced)

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| Test Coverage | >= 95% | NASA-grade reliability |
| Mutation Score | >= 80% | Test suite quality validation |
| TDG Grade | >= A- (88+) | No technical debt accumulation |
| Cyclomatic Complexity | <= 15 | Cognitive load management |
| Dead Code | <= 5% | Inventory waste elimination |
| Unwrap Calls | <= 50 | Cloudflare-class defect prevention |

### Quality Gates (Pre-Commit)

```bash
# Run manually
make quality

# Or individual checks
pmat quality-gate --fail-on-violation
pmat analyze complexity --top-files 10
pmat analyze dead-code --max-percentage 5.0
pmat mutate --target src/ --threshold 80
```

### Rust Project Score Target

Target: **180+/211 points** (85%+)

```bash
# Full assessment
pmat rust-project-score --full

# Categories:
# - Tooling & CI/CD: 130 pts
# - Code Quality: 26 pts
# - Testing Excellence: 20 pts
# - Documentation: 15 pts
# - Performance: 10 pts
# - Dependencies: 12 pts
# - Formal Verification: 8 pts (bonus)
```

---

## The Review Process

### 1. Pre-Submission (Author's Lab)

Before requesting review, authors MUST:

```bash
# Self-check checklist
make self-check

# This runs:
# - cargo fmt --check
# - cargo clippy -- -D warnings
# - cargo test
# - pmat quality-gate --fail-on-violation
# - pmat analyze satd  # Technical debt grading
```

### 2. Reproducibility Audit (Reviewer's First Step)

Reviewers MUST verify reproducibility before reading code:

| Audit Item | Command | Pass Criteria |
|------------|---------|---------------|
| Dependencies | `cargo build` | No unresolved deps |
| Environment | `cargo test` | All tests pass |
| Determinism | `cargo test` (3x) | No flaky tests |
| Results Match | Manual verification | Behavior matches PR description |

```bash
# Reviewer checkout and verify
git fetch origin
git checkout <commit>
make verify-reproducibility
```

### 3. Scientific Annotation (Code Review)

All review comments MUST follow the LSCRP taxonomy:

```
[CATEGORY:SEVERITY]: <Narrative>
Evidence: <citation or data>
```

See [CODE_REVIEW.md](./docs/CODE_REVIEW.md) for full taxonomy.

### 4. Stop the Line Authority

**Any engineer can block a merge** for:

- `[CORRECTNESS:BLOCKING]` - Logic errors
- `[SECURITY:BLOCKING]` - Vulnerabilities
- `[REPRODUCIBILITY:BLOCKING]` - Cannot verify behavior
- `[TESTABILITY:BLOCKING]` - Insufficient test coverage

When the line is stopped:
1. **Signal immediately** (Slack/email)
2. **Swarm to resolve** (pair programming if needed)
3. **No blame**—focus on systemic fixes

---

## PMAT Integration

### Daily Workflow Commands

```bash
# Generate AI-ready context for Claude
make pmat-context

# Quick quality check
make pmat-quality

# Full quality assessment
make pmat-score

# Mutation testing (test quality validation)
make pmat-mutate

# Technical debt analysis
make pmat-tdg

# Dead code detection
make pmat-dead-code

# Complexity hotspots
make pmat-complexity
```

### CI/CD Pipeline

Every push triggers:

1. **Build verification**: `cargo build --release`
2. **Test suite**: `cargo test --all`
3. **Quality gates**: `pmat quality-gate`
4. **Rust score**: `pmat rust-project-score`
5. **TDG regression**: `pmat tdg check-regression`

### MCP Integration (Claude Code)

PMAT provides 19 MCP tools for AI-assisted development:

```bash
# Start MCP server
pmat mcp

# Available tools:
# - analyze_technical_debt
# - complexity_analysis
# - mutation_testing_analysis
# - deep_context_generation
# - validate_documentation
# - defect_prediction
# ... and 13 more
```

---

## Code Style

### Automated Enforcement

Style is **automated, not debated**:

```bash
# Format code
cargo fmt

# Lint code
cargo clippy -- -D warnings

# Both are enforced in pre-commit hooks
```

### Design Principles

1. **Zero JavaScript/HTML**: Pure Rust only
2. **SIMD-first**: Use trueno dispatch macros for hot paths
3. **Memory alignment**: 64-byte alignment for SIMD operations
4. **No unnecessary abstractions**: 3 similar lines > premature abstraction
5. **Fail fast**: Validate at boundaries, trust internal code

### Documentation Requirements

- Public APIs: `///` rustdoc with examples
- Complex algorithms: Cite academic papers
- Non-obvious code: Explain "why", not "what"
- No TODOs without tickets

---

## Testing Requirements

### Coverage Targets

| Test Type | Minimum | Target |
|-----------|---------|--------|
| Unit | 90% | 95% |
| Integration | 80% | 90% |
| Doc tests | 100% | 100% |
| Mutation | 80% | 85% |

### Test Structure

```rust
#[cfg(test)]
mod tests {
    use super::*;

    // Unit tests: isolated, fast, deterministic
    #[test]
    fn test_scatter_plot_basic() {
        let plot = ScatterPlot::new()
            .x(&[1.0, 2.0, 3.0])
            .y(&[4.0, 5.0, 6.0])
            .build();

        assert_eq!(plot.point_count(), 3);
    }

    // Property-based tests for invariants
    #[test]
    fn test_framebuffer_dimensions_preserved() {
        proptest!(|(w in 1u32..4096, h in 1u32..4096)| {
            let fb = Framebuffer::new(w, h);
            prop_assert_eq!(fb.width, w);
            prop_assert_eq!(fb.height, h);
        });
    }

    // Benchmark-backed performance tests
    #[test]
    fn test_scatter_10k_under_5ms() {
        let start = Instant::now();
        let plot = ScatterPlot::new()
            .x(&random_vec(10_000))
            .y(&random_vec(10_000))
            .build();
        plot.render(&mut Framebuffer::new(800, 600));
        assert!(start.elapsed() < Duration::from_millis(5));
    }
}
```

### Mutation Testing

```bash
# Run mutation testing
pmat mutate --target src/ --threshold 80

# If score < 80%, add tests to kill surviving mutants
# Common survivors:
# - Boundary conditions (off-by-one)
# - Boolean logic inversions
# - Arithmetic operator swaps
```

---

## Continuous Improvement (Kaizen)

### Weekly Calibration Sessions

The team reviews:
1. Escaped defects (bugs found post-merge)
2. Review cycle time metrics
3. "Stop the Line" frequency
4. TDG trend analysis

### Retrospective Questions

- Why did this defect escape review?
- What systemic change prevents recurrence?
- Is this check automatable?

### Metrics Dashboard

```bash
# View quality trends
pmat show-metrics --trend

# Historical analysis
pmat analyze churn --days 30
```

---

## Getting Help

- **Documentation issues**: File in GitHub Issues with `[DOCS]` prefix
- **Quality gate failures**: Run `pmat diagnose --full`
- **Review disputes**: Escalate to Design Review meeting
- **Process improvements**: Propose in weekly Kaizen session

---

## Summary: The 5S of Code Contribution

| 5S | Action |
|----|--------|
| **Seiri (Sort)** | Only relevant changes in PR |
| **Seiton (Set in Order)** | Logical commit organization |
| **Seiso (Shine)** | Lint/format before review |
| **Seiketsu (Standardize)** | Use review checklist |
| **Shitsuke (Sustain)** | Participate in calibration |

---

*"The goal is not to ship, but to build quality in."* — Toyota Way
