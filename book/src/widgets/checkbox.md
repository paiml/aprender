# Checkbox

Boolean toggle input widget.

## Basic Usage

```rust
use presentar_widgets::Checkbox;

let checkbox = Checkbox::new("Accept terms");
```

## Builder Methods

| Method | Description |
|--------|-------------|
| `checked(bool)` | Initial checked state |
| `indeterminate(bool)` | Indeterminate state |
| `disabled(bool)` | Disable interaction |
| `check_color(Color)` | Checkmark color |
| `with_test_id(str)` | Test identifier |

## States

```rust
use presentar_widgets::checkbox::CheckState;

// Unchecked
let cb = Checkbox::new("Option").checked(false);

// Checked
let cb = Checkbox::new("Option").checked(true);

// Indeterminate (partial selection)
let cb = Checkbox::new("Select all").indeterminate(true);
```

## Event Handling

```rust
use presentar_widgets::CheckboxToggled;

if let Some(msg) = checkbox.event(&event) {
    if let Some(toggled) = msg.downcast_ref::<CheckboxToggled>() {
        println!("Now: {:?}", toggled.state);
    }
}
```

## Getters

```rust
let state = checkbox.get_state();
let is_checked = checkbox.is_checked();
```

## Example: Form

```rust
let form = Column::new()
    .gap(8.0)
    .child(Checkbox::new("Subscribe to newsletter").with_test_id("subscribe"))
    .child(Checkbox::new("Accept terms").with_test_id("terms"));
```

## Accessibility

- Role: `Checkbox`
- States: checked, unchecked, mixed
- Keyboard: Space toggles

## Verified Test

```rust
#[test]
fn test_checkbox_toggle() {
    use presentar_widgets::Checkbox;

    let mut cb = Checkbox::new("Test").checked(false);
    assert!(!cb.is_checked());

    cb.toggle();
    assert!(cb.is_checked());
}
```
