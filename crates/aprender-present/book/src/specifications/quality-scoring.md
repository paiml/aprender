# Quality Scoring

Comprehensive app quality measurement.

## Score Components

| Component | Weight | Description |
|-----------|--------|-------------|
| Tests | 30% | Coverage + mutation |
| Performance | 25% | Frame time + bundle |
| Accessibility | 25% | WCAG compliance |
| Structure | 20% | Code quality |

## Overall Formula

```rust
fn quality_score(m: &Metrics) -> f32 {
    m.test_score * 0.30
        + m.performance_score * 0.25
        + m.accessibility_score * 0.25
        + m.structure_score * 0.20
}
```

## Grade Thresholds

| Score | Grade | Status |
|-------|-------|--------|
| 90-100 | A | Excellent |
| 80-89 | B+ | Production ready |
| 70-79 | B | Good |
| 60-69 | C | Acceptable |
| < 60 | F | Failing |

## Test Score

```rust
fn test_score(coverage: f32, mutation: f32) -> f32 {
    let coverage_score = (coverage / 95.0 * 100.0).min(100.0);
    let mutation_score = (mutation / 80.0 * 100.0).min(100.0);
    coverage_score * 0.6 + mutation_score * 0.4
}
```

## Performance Score

```rust
fn performance_score(frame_ms: f32, bundle_kb: f32) -> f32 {
    let frame_score = if frame_ms <= 16.0 { 100.0 }
        else { (16.0 / frame_ms) * 100.0 };
    let bundle_score = if bundle_kb <= 500.0 { 100.0 }
        else { (500.0 / bundle_kb) * 100.0 };
    frame_score * 0.7 + bundle_score * 0.3
}
```

## Report Format

```
Quality Report
==============
Test Score:         92/100
Performance Score:  88/100
Accessibility:      95/100
Structure:          85/100
--------------------------
Overall:            90/100 (A)
```

## Verified Test

```rust
#[test]
fn test_quality_scoring_formula() {
    // Quality score calculation
    let test_score = 92.0;
    let perf_score = 88.0;
    let a11y_score = 95.0;
    let struct_score = 85.0;

    let overall = test_score * 0.30
        + perf_score * 0.25
        + a11y_score * 0.25
        + struct_score * 0.20;

    // 27.6 + 22.0 + 23.75 + 17.0 = 90.35
    assert!((overall - 90.35).abs() < 0.01);

    // Grade assignment
    let grade = match overall as u32 {
        90..=100 => "A",
        80..=89 => "B+",
        70..=79 => "B",
        60..=69 => "C",
        _ => "F",
    };
    assert_eq!(grade, "A");
}
```
