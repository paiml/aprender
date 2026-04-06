# WCAG 2.1 AA Checklist

Accessibility requirements for Presentar apps.

## Perceivable

### 1.1 Text Alternatives

- [ ] Images have alt text
- [ ] Icons have accessible names

### 1.3 Adaptable

- [ ] Content is semantic (proper roles)
- [ ] Reading order is logical

### 1.4 Distinguishable

| Requirement | Threshold |
|-------------|-----------|
| Text contrast | 4.5:1 |
| Large text contrast | 3.0:1 |
| Focus visible | Required |

## Operable

### 2.1 Keyboard Accessible

- [ ] All functions keyboard accessible
- [ ] No keyboard traps
- [ ] Focus order logical

### 2.4 Navigable

- [ ] Skip links available
- [ ] Focus visible
- [ ] Heading structure logical

## Understandable

### 3.2 Predictable

- [ ] Navigation consistent
- [ ] Identification consistent

### 3.3 Input Assistance

- [ ] Error identification
- [ ] Labels provided
- [ ] Error prevention

## Robust

### 4.1 Compatible

- [ ] Valid markup
- [ ] Name/role/value for all controls

## Presentar A11y API

```rust
// Set accessible name
button.with_accessible_name("Submit form");

// Set role
fn accessible_role(&self) -> AccessibleRole {
    AccessibleRole::Button
}

// Check focusable
fn is_focusable(&self) -> bool {
    !self.disabled
}
```

## Testing

```rust
use presentar_test::A11yChecker;

let report = A11yChecker::check(&widget);
report.assert_pass();
```

## Verified Test

```rust
#[test]
fn test_wcag_contrast() {
    use presentar_test::A11yChecker;
    use presentar_core::Color;

    let result = A11yChecker::check_contrast(
        &Color::BLACK,
        &Color::WHITE,
        false
    );

    assert!(result.passes_aa);  // 4.5:1
    assert!((result.ratio - 21.0).abs() < 0.5);
}
```
