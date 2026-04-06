# Quality Roadmap: Trueno-Viz

## Quality Philosophy

Trueno-Viz is built with **zero tolerance for defects**, following:

1. **Toyota Production System**: Jidoka, Kaizen, Genchi Genbutsu
2. **Scientific Peer Review**: Evidence-based, reproducible, verifiable
3. **NASA-Grade Standards**: 95% coverage, mutation testing, formal methods

---

## PMAT Integration Overview

**PMAT (paiml-mcp-agent-toolkit)** is the quality enforcement backbone for trueno-viz.

### Capabilities Utilized

| PMAT Feature | trueno-viz Use |
|--------------|----------------|
| **O(1) Quality Gates** | Pre-commit validation (<30ms) |
| **Rust Project Score** | 211-point comprehensive assessment |
| **Technical Debt Grading** | A+ to F grade enforcement |
| **Mutation Testing** | Test suite quality validation |
| **Complexity Analysis** | Cognitive load management |
| **Dead Code Detection** | Inventory waste elimination |
| **Documentation Validation** | Hallucination detection |
| **MCP Tools** | Claude Code AI integration |

### Quick Reference

```bash
# Daily development
make pmat-quality      # Quick quality check
make pmat-tdg          # Technical debt grade
make self-check        # Pre-submission checklist

# Deep analysis
make pmat-score        # Full 211-point assessment
make pmat-mutate       # Mutation testing
make pmat-complexity   # Complexity hotspots

# CI/CD
make ci                # Full CI pipeline
make ci-quick          # Quick validation
```

---

## Quality Targets by Phase

### Phase 1: Foundation (v0.1.0)

| Metric | Target | Rationale |
|--------|--------|-----------|
| Test Coverage | >= 85% | Initial baseline |
| Mutation Score | >= 70% | Bootstrap test quality |
| TDG Grade | >= B+ | Allow initial debt |
| Rust Score | >= 120/211 | Foundation |
| Complexity | <= 20 | Initial tolerance |

**Milestone Gates:**
- [ ] Framebuffer with SIMD operations
- [ ] Basic geometric primitives
- [ ] PNG output encoder
- [ ] ScatterPlot implementation

### Phase 2: Statistical Plots (v0.2.0)

| Metric | Target | Rationale |
|--------|--------|-----------|
| Test Coverage | >= 90% | Growing codebase |
| Mutation Score | >= 75% | Improving tests |
| TDG Grade | >= A- | Debt paydown |
| Rust Score | >= 150/211 | Maturity |
| Complexity | <= 18 | Tightening |

**Milestone Gates:**
- [ ] Heatmap with color scales
- [ ] Box plot, violin plot
- [ ] Line chart with simplification
- [ ] SVG and terminal outputs

### Phase 3: ML/DL Visualizations (v0.3.0)

| Metric | Target | Rationale |
|--------|--------|-----------|
| Test Coverage | >= 93% | Critical paths |
| Mutation Score | >= 78% | High reliability |
| TDG Grade | >= A | Minimal debt |
| Rust Score | >= 170/211 | Strong |
| Complexity | <= 16 | Optimizing |

**Milestone Gates:**
- [ ] Confusion matrix
- [ ] Loss curves (streaming)
- [ ] Integration with aprender
- [ ] Integration with entrenar

### Phase 4: Graph Visualizations (v0.4.0)

| Metric | Target | Rationale |
|--------|--------|-----------|
| Test Coverage | >= 94% | Graph algorithms critical |
| Mutation Score | >= 80% | Target reached |
| TDG Grade | >= A | Maintained |
| Rust Score | >= 175/211 | High quality |
| Complexity | <= 15 | Target |

**Milestone Gates:**
- [ ] Force-directed layout
- [ ] Barnes-Hut optimization
- [ ] Integration with trueno-graph
- [ ] Adjacency matrix visualization

### Phase 5: Advanced Features (v0.5.0)

| Metric | Target | Rationale |
|--------|--------|-----------|
| Test Coverage | >= 95% | NASA standard |
| Mutation Score | >= 82% | Exceeding target |
| TDG Grade | >= A | Maintained |
| Rust Score | >= 180/211 | Production ready |
| Complexity | <= 15 | Maintained |

**Milestone Gates:**
- [ ] Text prompt interface
- [ ] Grammar of Graphics API
- [ ] Faceting/small multiples
- [ ] WASM package

### Phase 6: Production (v1.0.0)

| Metric | Target | Rationale |
|--------|--------|-----------|
| Test Coverage | >= 95% | Maintained |
| Mutation Score | >= 85% | Excellent |
| TDG Grade | >= A+ | Exemplary |
| Rust Score | >= 190/211 | Production excellence |
| Complexity | <= 12 | Highly maintainable |

**Milestone Gates:**
- [ ] Comprehensive test suite
- [ ] Benchmark suite
- [ ] Full documentation
- [ ] crates.io publication

---

## Quality Gate Enforcement

### Pre-Commit (Automatic)

```
Hook: .git/hooks/pre-commit
├── Format check (cargo fmt --check)
├── Clippy lints (cargo clippy -- -D warnings)
├── Fast tests (cargo test --lib)
├── O(1) quality gate (pmat quality-gate)
└── TDG regression check (pmat tdg check-regression)
```

**Install:** `make hooks-install`

### Pre-Merge (CI/CD)

```yaml
GitHub Actions: .github/workflows/pmat-quality.yml
├── quick-check       # Format, lint, build, test
├── pmat-quality      # O(1) gates, TDG, complexity, dead code
├── rust-score        # 211-point assessment
├── coverage          # Test coverage >= 95%
├── tdg-regression    # No grade drops
└── quality-status    # Final gate
```

### Nightly (Comprehensive)

```
Scheduled: 2:00 AM UTC
├── Full test suite with coverage
├── Mutation testing (80%+ kill rate)
├── Dependency audit (cargo audit)
├── Performance regression tests
└── Documentation validation
```

---

## PMAT MCP Integration

### Claude Code Configuration

Add to `.claude/settings.json`:

```json
{
  "mcpServers": {
    "pmat": {
      "command": "pmat",
      "args": ["mcp"]
    }
  }
}
```

### Available MCP Tools

| Tool | Use Case |
|------|----------|
| `analyze_technical_debt` | Check TDG grade during development |
| `complexity_analysis` | Find complexity hotspots |
| `mutation_testing_analysis` | Validate test quality |
| `deep_context_generation` | Generate AI-ready context |
| `validate_documentation` | Check README accuracy |
| `defect_prediction` | ML-based defect risk |
| `semantic_search` | Natural language code search |

### Example Usage in Claude Code

```
User: "What's the current quality score?"
Claude: [Uses analyze_technical_debt MCP tool]
Claude: "Current TDG grade is A- (89/100). Main penalties:
         - 3 SATD comments (-6 pts)
         - 2 TODOs without issues (-3 pts)"

User: "Find complex functions in the render module"
Claude: [Uses complexity_analysis MCP tool]
Claude: "Top complexity hotspots in src/render/:
         - rasterizer.rs:draw_polygon (CC: 14)
         - text.rs:layout_glyphs (CC: 12)"
```

---

## Continuous Improvement (Kaizen)

### Weekly Review

1. **Escaped Defects**: Bugs found post-merge
2. **Review Cycle Time**: Time from PR open to merge
3. **TDG Trend**: Grade trajectory over time
4. **Mutation Score Trend**: Test quality improvement

### Monthly Retrospective

1. **Quality Gate Effectiveness**: Are we catching bugs?
2. **Process Friction**: What's slowing development?
3. **Threshold Calibration**: Adjust targets based on data

### Quarterly Goals

| Q1 | Q2 | Q3 | Q4 |
|----|----|----|-----|
| v0.1-0.2 | v0.3 | v0.4-0.5 | v1.0 |
| 85% → 90% cov | 90% → 93% cov | 93% → 95% cov | Maintain 95% |
| B+ → A- TDG | A- → A TDG | A TDG | A+ TDG |

---

## Metrics Dashboard

### View Current Metrics

```bash
# Quick health check
pmat repo-score

# Detailed Rust assessment
pmat rust-project-score --full

# Historical trend
pmat show-metrics --trend

# Generate report
pmat report --format markdown --output quality-report.md
```

### Key Metrics Tracked

| Metric | Source | Frequency |
|--------|--------|-----------|
| Test Coverage | cargo-tarpaulin | Every PR |
| Mutation Score | pmat mutate | Nightly |
| TDG Grade | pmat analyze satd | Every commit |
| Rust Project Score | pmat rust-project-score | Every PR |
| Complexity | pmat analyze complexity | Every PR |
| Dead Code % | pmat analyze dead-code | Weekly |
| Dependency Count | cargo tree | Weekly |

---

## Emergency Procedures

### Quality Gate Failure

1. **DO NOT** force-push or skip hooks
2. Run `pmat diagnose --full` to understand failure
3. Fix the violation or request exception with justification
4. Document exception in PR description

### TDG Regression

1. Check `pmat analyze satd` for new debt
2. Either fix the debt or update baseline
3. Baseline updates require team approval
4. Document why debt was acceptable

### Coverage Drop

1. Run `cargo tarpaulin --out Html` for detailed report
2. Identify uncovered lines
3. Add tests or mark as intentionally uncovered
4. Never merge with unexplained coverage drop

---

## References

- [CONTRIBUTING.md](../CONTRIBUTING.md) - Development workflow
- [CODE_REVIEW.md](./CODE_REVIEW.md) - Review taxonomy
- [trueno-viz-spec.md](./specifications/trueno-viz-spec.md) - Technical specification
- [PMAT Documentation](https://github.com/paiml/paiml-mcp-agent-toolkit)

---

*"Quality is not an act, it is a habit."* — Aristotle
