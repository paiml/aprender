# Grade Thresholds

Quality scores map to letter grades.

## Grade Scale

| Grade | Score | Description |
|-------|-------|-------------|
| A | 90-100 | Excellent |
| A- | 85-89 | Very Good |
| B+ | 80-84 | Good |
| B | 75-79 | Above Average |
| B- | 70-74 | Average |
| C+ | 65-69 | Below Average |
| C | 60-64 | Needs Work |
| D | 50-59 | Poor |
| F | 0-49 | Failing |

## Production Requirements

**Minimum: B+ (80+)**

```bash
# Gate check enforces minimum grade
make tier2

# Fails if score < 80
```

## Grade Calculation

```rust
fn grade_from_score(score: u8) -> Grade {
    match score {
        90..=100 => Grade::A,
        85..=89 => Grade::AMinus,
        80..=84 => Grade::BPlus,
        75..=79 => Grade::B,
        70..=74 => Grade::BMinus,
        65..=69 => Grade::CPlus,
        60..=64 => Grade::C,
        50..=59 => Grade::D,
        _ => Grade::F,
    }
}
```

## Grade Components

Each component can block deployment:

| Component | Minimum | Blocker |
|-----------|---------|---------|
| Test Coverage | 85% | < 70% blocks |
| Performance | Frame < 16ms | > 32ms blocks |
| Accessibility | 0 critical | Any critical blocks |
| Clippy | 0 warnings | Any warning blocks |

## Verified Test

```rust
#[test]
fn test_grade_thresholds() {
    use presentar_test::Grade;

    assert_eq!(Grade::from_score(95), Grade::A);
    assert_eq!(Grade::from_score(82), Grade::BPlus);
    assert_eq!(Grade::from_score(45), Grade::F);
}
```
