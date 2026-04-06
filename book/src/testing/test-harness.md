# Test Harness

The `Harness` struct provides a pure-Rust testing interface for Presentar widgets.

## Creating a Harness

```rust
use presentar_test::Harness;
use presentar_widgets::Button;

let harness = Harness::new(Button::new("Click me"));
```

## Viewport Configuration

```rust
let harness = Harness::new(widget)
    .viewport(1920.0, 1080.0);
```

## Event Simulation

| Method | Description |
|--------|-------------|
| `click(selector)` | Simulate mouse click |
| `type_text(selector, text)` | Type text into widget |
| `press_key(key)` | Simulate key press |
| `scroll(selector, delta)` | Simulate scroll |
| `tick(ms)` | Advance simulated time |

## Queries

| Method | Returns | Description |
|--------|---------|-------------|
| `query(sel)` | `Option<&dyn Widget>` | Single widget |
| `query_all(sel)` | `Vec<&dyn Widget>` | All matches |
| `text(sel)` | `String` | Text content |
| `exists(sel)` | `bool` | Existence check |

## Assertions

| Method | Panics when |
|--------|-------------|
| `assert_exists(sel)` | Widget not found |
| `assert_not_exists(sel)` | Widget found |
| `assert_text(sel, expected)` | Text mismatch |
| `assert_text_contains(sel, sub)` | Substring missing |
| `assert_count(sel, n)` | Count mismatch |

## Verified Example

```rust
#[test]
fn test_harness_basics() {
    use presentar_test::Harness;
    use presentar_widgets::Button;

    let button = Button::new("OK").with_test_id("ok-btn");
    let harness = Harness::new(button);

    harness.assert_exists("[data-testid='ok-btn']");
    assert!(harness.exists("[data-testid='ok-btn']"));
}
```
