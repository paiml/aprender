# Selectors

CSS-like selectors for querying widgets.

## Selector Types

| Syntax | Match Type | Example |
|--------|-----------|---------|
| `Type` | Widget type | `"Button"` |
| `#id` | Widget ID | `"#submit"` |
| `.class` | Widget class | `".primary"` |
| `[attr='val']` | Attribute | `"[data-testid='x']"` |

## Test ID Selector (Recommended)

```rust
// Widget code
let btn = Button::new("OK").with_test_id("confirm");

// Test code
harness.assert_exists("[data-testid='confirm']");
```

## Attribute Selectors

```rust
// By test ID
harness.query("[data-testid='my-widget']");

// By aria-label
harness.query("[aria-label='Close']");
```

## Parsing API

```rust
use presentar_test::Selector;

let sel = Selector::parse("[data-testid='btn']").unwrap();
assert_eq!(sel, Selector::TestId("btn".to_string()));

let sel = Selector::parse("#submit-btn").unwrap();
assert_eq!(sel, Selector::Id("submit-btn".to_string()));

let sel = Selector::parse(".primary").unwrap();
assert_eq!(sel, Selector::Class("primary".to_string()));
```

## Error Handling

```rust
use presentar_test::selector::SelectorError;

assert_eq!(Selector::parse(""), Err(SelectorError::Empty));
```

## Best Practices

1. Use `[data-testid='...']` for reliable selection
2. Avoid type selectors in production tests
3. Keep selectors specific and stable
