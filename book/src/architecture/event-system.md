# Event System

How user input flows through widgets.

## Event Types

| Event | Description |
|-------|-------------|
| `MouseMove` | Cursor position change |
| `MouseDown` | Button pressed |
| `MouseUp` | Button released |
| `Scroll` | Wheel scroll |
| `KeyDown` | Key pressed |
| `KeyUp` | Key released |
| `TextInput` | Character typed |
| `FocusIn` | Widget gained focus |
| `FocusOut` | Widget lost focus |
| `MouseEnter` | Cursor entered bounds |
| `MouseLeave` | Cursor left bounds |
| `Resize` | Window resized |

## Event Handling

```rust
impl Widget for MyWidget {
    fn event(&mut self, event: &Event) -> Option<Box<dyn Any + Send>> {
        match event {
            Event::MouseUp { position, button } => {
                if *button == MouseButton::Left {
                    Some(Box::new(Clicked))
                } else {
                    None
                }
            }
            Event::KeyDown { key } => {
                if *key == Key::Enter {
                    Some(Box::new(Activated))
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}
```

## Messages

Widgets return messages, not state:

```rust
// Define message type
pub struct ButtonClicked;

// Return from event
fn event(&mut self, e: &Event) -> Option<Box<dyn Any + Send>> {
    if let Event::MouseUp { .. } = e {
        Some(Box::new(ButtonClicked))
    } else {
        None
    }
}

// Handle in parent
if let Some(msg) = widget.event(&event) {
    if msg.downcast_ref::<ButtonClicked>().is_some() {
        state.count += 1;
    }
}
```

## Event Propagation

```
Event → Root → Child → ... → Target
                              │
                              ▼
                           Message
```

## Focus Management

```rust
// Check if focusable
if widget.is_focusable() {
    widget.event(&Event::FocusIn);
}

// Tab navigation
if key == Key::Tab {
    current.event(&Event::FocusOut);
    next.event(&Event::FocusIn);
}
```

## Verified Test

```rust
#[test]
fn test_event_handling() {
    use presentar_widgets::Button;
    use presentar_core::{Event, MouseButton, Point, Rect, Widget};

    let mut button = Button::new("Test");
    button.layout(Rect::new(0.0, 0.0, 100.0, 40.0));

    // MouseUp returns message
    let msg = button.event(&Event::MouseUp {
        position: Point::new(50.0, 20.0),
        button: MouseButton::Left,
    });

    assert!(msg.is_some());
}
```
