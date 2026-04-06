# Unidirectional Data Flow

Events flow one direction through the system.

## Flow Diagram

```
┌──────────┐   ┌─────────┐   ┌────────┐   ┌──────────┐
│  EVENT   │──▶│  STATE  │──▶│ WIDGET │──▶│   DRAW   │
│ (input)  │   │(update) │   │ (tree) │   │(commands)│
└──────────┘   └─────────┘   └────────┘   └──────────┘
     ▲                                          │
     └──────────────────────────────────────────┘
                    (next frame)
```

## Event Phase

User generates input:

```rust
Event::MouseDown { position, button: MouseButton::Left }
Event::key_down(Key::Enter)
Event::TextInput { text: "hello" }
```

## State Phase

State updates from events:

```rust
struct AppState {
    counter: i32,
}

impl AppState {
    fn handle_event(&mut self, event: &Event) {
        if let Event::MouseUp { .. } = event {
            self.counter += 1;
        }
    }
}
```

## Widget Phase

Widgets rebuild from state:

```rust
fn build_ui(state: &AppState) -> impl Widget {
    Column::new()
        .child(Text::new(format!("Count: {}", state.counter)))
        .child(Button::new("+1"))
}
```

## Draw Phase

Canvas receives commands:

```rust
fn render(widget: &impl Widget, canvas: &mut impl Canvas) {
    widget.paint(canvas);
    // canvas now has: FillRect, DrawText, etc.
}
```

## Benefits

| Benefit | Description |
|---------|-------------|
| Predictability | Same state = same UI |
| Testability | State can be mocked |
| Debugging | Event log shows history |
| Time-travel | Replay events for debugging |

## Anti-Pattern: Two-Way Binding

```rust
// BAD: Widget directly modifies state
impl Widget for Counter {
    fn event(&mut self, e: &Event) -> Option<Box<dyn Any + Send>> {
        self.state.counter += 1;  // Wrong!
        None
    }
}

// GOOD: Widget emits message
impl Widget for Counter {
    fn event(&mut self, e: &Event) -> Option<Box<dyn Any + Send>> {
        Some(Box::new(Increment))  // Message handled by state
    }
}
```

## Verified Test

```rust
#[test]
fn test_unidirectional_flow() {
    use presentar_widgets::Button;
    use presentar_core::{Event, MouseButton, Point, Rect, Widget};

    let mut button = Button::new("Click");
    button.layout(Rect::new(0.0, 0.0, 100.0, 40.0));

    // Event → Widget → Message
    let msg = button.event(&Event::MouseUp {
        position: Point::new(50.0, 20.0),
        button: MouseButton::Left,
    });

    // Message flows back to state handler
    assert!(msg.is_some());
}
```
