# Testing Strategy

Comprehensive testing approach for Presentar apps.

## Test Pyramid

```
        /\
       /  \     E2E (few)
      /----\
     /      \   Integration
    /--------\
   /          \  Unit (many)
  /------------\
```

## Test Categories

| Category | Purpose | Count |
|----------|---------|-------|
| Unit | Single function | Many |
| Integration | Cross-module | Some |
| Visual | Pixel-perfect | Few |
| A11y | Accessibility | All screens |
| Performance | Frame time | Critical paths |

## Unit Tests

```rust
#[test]
fn test_button_is_interactive() {
    let button = Button::new("OK");
    assert!(button.is_interactive());
}
```

## Integration Tests

```rust
#[test]
fn test_form_submission() {
    let form = Form::new()
        .field("name")
        .submit_button();

    let harness = Harness::new(form);
    harness.type_text("[data-testid='name']", "Alice");
    harness.click("[data-testid='submit']");
    harness.assert_exists("[data-testid='success']");
}
```

## Visual Tests

```rust
#[test]
fn test_button_visual() {
    let button = Button::new("OK");
    let screenshot = render_to_image(&button);
    Snapshot::assert_match("button-default", &screenshot, 0.001);
}
```

## A11y Tests

```rust
#[test]
fn test_form_accessibility() {
    let form = build_form();
    let report = A11yChecker::check(&form);
    report.assert_pass();
}
```

## Performance Tests

```rust
#[test]
fn test_paint_performance() {
    let widget = build_complex_ui();
    let start = Instant::now();
    widget.paint(&mut canvas);
    assert!(start.elapsed().as_millis() < 16);
}
```

## Coverage Target

**Minimum: 85%**

```bash
make coverage
```

## Verified Test

```rust
#[test]
fn test_testing_strategy() {
    // All widgets are testable
    use presentar_widgets::Button;
    use presentar_core::Widget;

    let button = Button::new("Test");
    assert!(button.is_interactive());  // Unit test
}
```
