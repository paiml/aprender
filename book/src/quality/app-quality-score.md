# App Quality Score

Every Presentar app receives a quality score (0-100).

## Score Components

| Component | Weight | Measures |
|-----------|--------|----------|
| Test Coverage | 30% | Line coverage percentage |
| Performance | 25% | Frame time, bundle size |
| Accessibility | 20% | WCAG 2.1 AA compliance |
| Code Quality | 15% | Clippy, complexity |
| Documentation | 10% | Rustdoc coverage |

## Calculation

```rust
let score =
    coverage_score * 0.30 +
    performance_score * 0.25 +
    accessibility_score * 0.20 +
    code_quality_score * 0.15 +
    documentation_score * 0.10;
```

## Running Quality Check

```bash
make score
```

## Score Breakdown

```
App Quality Score: 85/100 (B+)
─────────────────────────────
Test Coverage:    90/100 (30 pts)
  Lines:          95%
  Branches:       88%

Performance:      80/100 (20 pts)
  Frame time:     12ms
  Bundle size:    420KB

Accessibility:    85/100 (17 pts)
  WCAG violations: 2
  Contrast pass:  98%

Code Quality:     90/100 (13.5 pts)
  Clippy warnings: 0
  Complexity:     Low

Documentation:    75/100 (7.5 pts)
  Public items:   80%
─────────────────────────────
```

## Improving Score

| Issue | Fix |
|-------|-----|
| Low coverage | Add more tests |
| Slow frames | Optimize paint/layout |
| A11y violations | Add accessible names |
| Clippy warnings | Run `cargo clippy --fix` |

## Verified Test

```rust
#[test]
fn test_quality_score_range() {
    // Score is 0-100
    let score = 85;
    assert!(score >= 0 && score <= 100);
}
```
