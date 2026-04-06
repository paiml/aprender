# Assertions

Built-in assertions for verifying widget state.

## Existence

```rust
// Assert widget exists
harness.assert_exists("[data-testid='header']");

// Assert widget does not exist
harness.assert_not_exists("[data-testid='error']");
```

## Text Content

```rust
// Exact match
harness.assert_text("[data-testid='title']", "Welcome");

// Substring match
harness.assert_text_contains("[data-testid='msg']", "success");
```

## Count

```rust
// Verify exact count
harness.assert_count("[data-testid='item']", 5);
```

## Chaining

```rust
harness
    .assert_exists("[data-testid='form']")
    .assert_exists("[data-testid='submit']")
    .assert_count("[data-testid='field']", 3);
```

## Custom Assertions

```rust
fn assert_enabled(harness: &Harness, selector: &str) {
    let widget = harness.query(selector).expect("Not found");
    assert!(widget.is_interactive(), "Should be enabled");
}
```

## Verified Test

```rust
#[test]
fn test_assertions() {
    use presentar_test::Harness;
    use presentar_widgets::{Column, Button};

    let ui = Column::new()
        .child(Button::new("A").with_test_id("btn"))
        .child(Button::new("B").with_test_id("btn"));

    let harness = Harness::new(ui);

    harness
        .assert_count("[data-testid='btn']", 2)
        .assert_not_exists("[data-testid='missing']");
}
```
