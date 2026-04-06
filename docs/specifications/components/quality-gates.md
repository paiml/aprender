# Quality Gates

> Parent: [presentar-spec.md](../presentar-spec.md)

**Scope:** TUI quality scoring system, coverage enforcement, mutation testing, CI/CD pipeline, quality metrics.

---

## TUI Quality Scoring System (0-100, Grades F-A)

| Dimension | Weight | Max | Description |
|-----------|--------|-----|-------------|
| Performance (SIMD/GPU) | 25% | 25 | Frame latency, vectorization coverage, ComputeBrick, zero-alloc |
| Testing (Probador) | 20% | 20 | Pixel tests, playbook scenarios, regression detection, mutation |
| Widget Reuse | 15% | 15 | Library coverage, custom widget justification, composition |
| Code Coverage | 15% | 15 | Line (llvm-cov), branch, function coverage |
| Quality Metrics | 15% | 15 | Clippy warnings, rustfmt, certeza score |
| Falsifiability | 10% | 10 | Explicit failure criteria, automated tests, SelfDescribingBrick |

### Grade Scale

| Grade | Score | Status |
|-------|-------|--------|
| A | 90-100 | Production Ready |
| B | 80-89 | Release Candidate |
| C | 70-79 | Beta Quality |
| D | 60-69 | Alpha Quality |
| F | < 60 | Not Releasable |

## Performance Scoring (25 pts)

| Metric | Points | Criteria |
|--------|--------|----------|
| Frame Latency | 0-10 | < 16ms = 10, < 33ms = 5, > 33ms = 0 |
| SIMD Coverage | 0-8 | % of hot paths using SIMD |
| ComputeBrick Usage | 0-5 | Proper batch rendering |
| Zero-Alloc Rendering | 0-2 | No allocations in render loop |

## Testing Scoring (20 pts)

| Metric | Points | Criteria |
|--------|--------|----------|
| Pixel Test Coverage | 0-8 | % of widgets with pixel-perfect assertions |
| Playbook Scenarios | 0-6 | % of user flows covered |
| Regression Detection | 0-4 | Golden master comparison working |
| Mutation Coverage | 0-2 | % of mutants killed |

## Coverage Enforcement (ZERO TOLERANCE)

| Metric | Threshold | Enforcement |
|--------|-----------|-------------|
| Line Coverage | >= 95% | CI blocks merge |
| Region Coverage | >= 95% | CI blocks merge |
| Function Coverage | >= 95% | CI blocks merge |
| Branch Coverage | >= 90% | CI warns, tracked |

```bash
# Verification (REQUIRED before commit)
cargo llvm-cov --all-features --workspace --fail-under-lines 95
```

No file may fall below 95% coverage. Exceptions require technical lead approval with documented justification and expiry date in `.coverage-exceptions.toml`.

## Mutation Testing Gate

```bash
cargo mutants --minimum-mutants-tested 80
cargo mutants --package presentar-terminal --minimum-mutants-tested 90  # Critical paths
```

## PMAT Quality Gates Configuration

```toml
[gates]
min_grade = "A"
min_coverage = 95
min_mutation_score = 80
max_complexity = 10
max_nesting = 3
max_function_lines = 40

[thresholds]
min_tdg_score = 85
min_repo_score = 90
min_rust_score = 90

[enforcement]
fail_on_regression = true
max_score_drop = 0
require_all_gates = true
block_merge_on_failure = true

[satd]
allow_todo = false
allow_fixme = false
allow_hack = false
```

## Pre-Commit Hook

```bash
#!/bin/bash
set -euo pipefail
# 1. Coverage check (min 95%)
cargo llvm-cov --quiet --fail-under-lines 95
# 2. Clippy (deny all warnings)
cargo clippy --all-features -- -D warnings
# 3. SATD check (no TODO/FIXME/HACK)
grep -rn "TODO\|FIXME\|HACK\|XXX" src/ && exit 1 || true
```

## CI Pipeline

```yaml
name: PMAT Quality Gates
on: [push, pull_request]
jobs:
  coverage:
    steps:
      - name: Coverage (95% REQUIRED)
        run: cargo llvm-cov --all-features --workspace --fail-under-lines 95
  mutation:
    steps:
      - name: Mutation Testing (80% REQUIRED)
        run: cargo mutants --minimum-mutants-tested 80
  quality-score:
    steps:
      - name: PMAT Score (90+ REQUIRED)
        run: score --ci --threshold 90 --output json > quality-report.json
```

## Regression Prevention

Every commit records quality metrics. Any drop in coverage or mutation score triggers CI failure.

```json
{
  "commit": "abc123",
  "metrics": {
    "line_coverage": 96.2,
    "mutation_score": 82.5,
    "complexity_avg": 4.2,
    "quality_grade": "A"
  }
}
```

## pmat Quality Scorer CLI

```
score [OPTIONS] [PATH]
  -o, --output <FORMAT>   text, json, yaml (default: text)
  --ci                    Exit 1 if score < threshold
  --threshold <N>         Minimum passing score (default: 80)
```

Analyzes crate and produces scores for each dimension via AST analysis, test counting, import analysis, llvm-cov integration, clippy, and pattern matching.

## App Quality Score (Presentar Apps)

Six-dimension scoring for Presentar applications:

| Category | Points | Focus |
|----------|--------|-------|
| Structural | 25 | Widget complexity, layout depth, component count |
| Performance | 20 | Render time p95, memory, bundle size |
| Accessibility | 20 | WCAG AA, keyboard nav, screen reader |
| Data Quality | 15 | Completeness, freshness, schema validation |
| Documentation | 10 | Manifest fields, model/data cards |
| Consistency | 10 | Theme adherence, naming conventions |

### App Quality Gates

```toml
[gates]
min_grade = "B+"
min_score = 80.0

[performance]
max_render_time_ms = 16
max_bundle_size_kb = 500

[accessibility]
wcag_level = "AA"
min_contrast_ratio = 4.5
require_keyboard_nav = true

[documentation]
require_model_cards = true
require_data_cards = true
```

## Quality Pipeline Tiers

| Tier | Timing | Scope |
|------|--------|-------|
| Tier 1 | < 1s | `cargo check`, YAML lint |
| Tier 2 | < 5min | fmt, clippy, unit tests, integration, score check |
| Tier 3 | Hours | Visual regression, coverage, mutation testing, benchmarks |

## Acceptance Criteria

- [ ] Line coverage >= 95%
- [ ] Region coverage >= 95%
- [ ] Mutation score >= 80%
- [ ] Quality grade = A
- [ ] Zero SATD markers
- [ ] All falsification tests pass
- [ ] Pre-commit hook installed
- [ ] CI pipeline configured

## References

- Jia, Y. & Harman, M. (2011). Mutation Testing. *IEEE TSE*, 37(5).
- Fog, A. (2023). Optimizing software in C++.
- Popper, K. (1959). *The Logic of Scientific Discovery*. Routledge.
- Nagappan, N. & Ball, T. (2005). Use of relative code churn measures to predict system defect density. *ICSE '05*.
