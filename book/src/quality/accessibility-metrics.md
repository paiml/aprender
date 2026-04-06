# Accessibility Metrics

Measuring WCAG compliance.

## Score Components

| Component | Weight | Description |
|-----------|--------|-------------|
| Contrast | 30% | Text/background ratio |
| Focus | 25% | Keyboard navigation |
| Labels | 20% | Form accessibility |
| Structure | 15% | Semantic HTML |
| ARIA | 10% | Role/state attributes |

## Contrast Requirements

| Element | Minimum Ratio |
|---------|---------------|
| Normal text | 4.5:1 |
| Large text (18pt+) | 3.0:1 |
| UI components | 3.0:1 |
| Non-text content | 3.0:1 |

## Calculating Contrast

```rust
fn contrast_ratio(fg: &Color, bg: &Color) -> f32 {
    let l1 = relative_luminance(fg);
    let l2 = relative_luminance(bg);

    let lighter = l1.max(l2);
    let darker = l1.min(l2);

    (lighter + 0.05) / (darker + 0.05)
}

fn relative_luminance(c: &Color) -> f32 {
    let r = linearize(c.r);
    let g = linearize(c.g);
    let b = linearize(c.b);
    0.2126 * r + 0.7152 * g + 0.0722 * b
}
```

## Focus Indicators

| Requirement | Pass Criteria |
|-------------|---------------|
| Visible | 2px+ outline |
| Contrast | 3:1 against adjacent |
| Persistent | Doesn't disappear |

## Grading

| Score | Grade | Status |
|-------|-------|--------|
| 90-100 | A | Excellent |
| 80-89 | B | Good |
| 70-79 | C | Acceptable |
| < 70 | F | Failing |

## Verified Test

```rust
#[test]
fn test_accessibility_contrast_ratio() {
    use presentar_test::A11yChecker;
    use presentar_core::Color;

    // Black on white: maximum contrast
    let result = A11yChecker::check_contrast(
        &Color::BLACK,
        &Color::WHITE,
        false  // not large text
    );

    assert!(result.passes_aa);
    assert!((result.ratio - 21.0).abs() < 0.5);

    // Gray on white: lower contrast
    let gray = Color::new(0.5, 0.5, 0.5, 1.0);
    let result2 = A11yChecker::check_contrast(
        &gray,
        &Color::WHITE,
        false
    );

    // ~4.0:1 ratio - borderline
    assert!(result2.ratio > 3.0);
}
```
