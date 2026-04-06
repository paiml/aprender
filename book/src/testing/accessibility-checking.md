# Accessibility Checking

WCAG 2.1 AA compliance checking built into the test harness.

## Basic Usage

```rust
use presentar_test::A11yChecker;

let report = A11yChecker::check(&widget);
report.assert_pass();
```

## Check Report

```rust
let report = A11yChecker::check(&widget);

if !report.is_passing() {
    for violation in &report.violations {
        println!("{}: {} (WCAG {})",
            violation.rule,
            violation.message,
            violation.wcag
        );
    }
}
```

## Critical Violations

```rust
let critical = report.critical();
assert!(critical.is_empty(), "No critical a11y issues");
```

## Contrast Checking

```rust
use presentar_core::Color;

let result = A11yChecker::check_contrast(
    &Color::BLACK,    // foreground
    &Color::WHITE,    // background
    false             // large_text
);

assert!(result.passes_aa);   // WCAG AA (4.5:1)
assert!(result.passes_aaa);  // WCAG AAA (7:1)
println!("Ratio: {:.2}:1", result.ratio);
```

## Large Text Thresholds

| Level | Normal Text | Large Text (14pt bold / 18pt) |
|-------|-------------|-------------------------------|
| AA | 4.5:1 | 3.0:1 |
| AAA | 7.0:1 | 4.5:1 |

## Rules Checked

| Rule | WCAG | Description |
|------|------|-------------|
| `aria-label` | 4.1.2 | Interactive elements need accessible names |
| `keyboard` | 2.1.1 | Interactive elements must be focusable |

## Impact Levels

```rust
use presentar_test::a11y::Impact;

match violation.impact {
    Impact::Critical => { /* Must fix */ }
    Impact::Serious => { /* Should fix */ }
    Impact::Moderate => { /* Consider fixing */ }
    Impact::Minor => { /* Optional */ }
}
```

## Verified Test

```rust
#[test]
fn test_contrast_check() {
    use presentar_test::A11yChecker;
    use presentar_core::Color;

    let result = A11yChecker::check_contrast(
        &Color::BLACK,
        &Color::WHITE,
        false
    );

    assert!(result.passes_aa);
    assert!((result.ratio - 21.0).abs() < 0.5);
}
```
