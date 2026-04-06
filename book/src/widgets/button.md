# Button

The `Button` widget provides a clickable interactive element with label text and visual feedback for hover/press states.

## Basic Usage

```rust
use presentar_widgets::Button;

let button = Button::new("Click me");
```

## Builder Pattern

Button supports a fluent builder pattern for customization:

```rust
use presentar_widgets::Button;
use presentar_core::Color;

let button = Button::new("Submit")
    .padding(16.0)
    .font_size(16.0)
    .background(Color::from_hex("#4f46e5").unwrap())
    .text_color(Color::WHITE)
    .disabled(false)
    .with_test_id("submit-btn")
    .with_accessible_name("Submit form");
```

## Customization Options

| Method | Description | Default |
|--------|-------------|---------|
| `padding(f32)` | Inner padding around text | `12.0` |
| `font_size(f32)` | Text size in pixels | `14.0` |
| `background(Color)` | Normal state background | `#6366f1` |
| `background_hover(Color)` | Hover state background | `#4f46e5` |
| `background_pressed(Color)` | Pressed state background | `#4338ca` |
| `text_color(Color)` | Label text color | `WHITE` |
| `corner_radius(CornerRadius)` | Rounded corners | `uniform(4.0)` |
| `disabled(bool)` | Disable interaction | `false` |

## Event Handling

Button emits `ButtonClicked` when activated:

```rust
use presentar_widgets::{Button, ButtonClicked};
use presentar_core::{Event, Widget};

let mut button = Button::new("OK");
button.layout(bounds);

// Mouse click
if let Some(msg) = button.event(&Event::MouseUp { position, button: MouseButton::Left }) {
    if msg.downcast_ref::<ButtonClicked>().is_some() {
        // Handle click
    }
}

// Keyboard activation (Enter or Space)
if let Some(msg) = button.event(&Event::key_up(Key::Enter)) {
    // Handle keyboard activation
}
```

## Visual States

Button automatically manages visual states:

- **Normal**: Default `background` color
- **Hovered**: `background_hover` color when mouse enters
- **Pressed**: `background_pressed` color when clicked
- **Disabled**: Grayscale background, gray text, no interaction

## Accessibility

Button is fully accessible:

- Role: `AccessibleRole::Button`
- Focusable via Tab navigation (when not disabled)
- Keyboard activation with Enter or Space
- `accessible_name` defaults to label text

```rust
let button = Button::new("OK")
    .with_accessible_name("Confirm action");

assert_eq!(button.accessible_role(), AccessibleRole::Button);
assert_eq!(button.accessible_name(), Some("Confirm action"));
```

## Testing

Use `test_id` for testing:

```rust
let button = Button::new("Submit").with_test_id("submit-btn");

// In tests, select by test ID
harness.find("[data-testid='submit-btn']").click();
```

## YAML Definition

```yaml
- Button:
    label: "Click me"
    background: "#6366f1"
    padding: 12
    on_click: submit_form
```
