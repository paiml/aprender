# TextInput

Text entry field widget.

## Basic Usage

```rust
use presentar_widgets::TextInput;

let input = TextInput::new()
    .placeholder("Enter name...");
```

## Builder Methods

| Method | Description |
|--------|-------------|
| `value(str)` | Initial text value |
| `placeholder(str)` | Placeholder text |
| `password(bool)` | Mask input as password |
| `max_length(usize)` | Maximum characters |
| `disabled(bool)` | Disable interaction |
| `background(Color)` | Background color |
| `text_color(Color)` | Text color |

## Example

```rust
let email = TextInput::new()
    .placeholder("email@example.com")
    .value("")
    .with_test_id("email-input");

let password = TextInput::new()
    .placeholder("Password")
    .password(true)
    .with_test_id("password-input");
```

## Event Handling

```rust
use presentar_widgets::TextInputChanged;

if let Some(msg) = input.event(&event) {
    if let Some(changed) = msg.downcast_ref::<TextInputChanged>() {
        println!("New text: {}", changed.value);
    }
}
```

## Getters

```rust
let text = input.get_value();
let is_empty = input.is_empty();
```

## Focus

```rust
use presentar_core::Event;

// Focus event
input.event(&Event::FocusIn);

// Blur event
input.event(&Event::FocusOut);
```

## Accessibility

- Role: `TextInput`
- Keyboard: Full text editing
- Focus indicator shown

## Verified Test

```rust
#[test]
fn test_text_input_value() {
    use presentar_widgets::TextInput;

    let input = TextInput::new().value("hello");
    assert_eq!(input.get_value(), "hello");
}
```
