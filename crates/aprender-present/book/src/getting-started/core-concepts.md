# Core Concepts

Essential concepts for understanding Presentar.

## Widget

The fundamental building block. Everything on screen is a widget.

```rust
pub trait Widget {
    fn measure(&self, constraints: Constraints) -> Size;
    fn layout(&mut self, bounds: Rect) -> LayoutResult;
    fn paint(&self, canvas: &mut dyn Canvas);
    fn event(&mut self, event: &Event) -> Option<Box<dyn Any + Send>>;
}
```

## Unidirectional Data Flow

```
Event → State → Widget → Draw
  │                        │
  └────────────────────────┘
```

1. **Event**: User interaction (click, type, scroll)
2. **State**: Application data updates
3. **Widget**: UI tree rebuilds
4. **Draw**: Canvas receives commands

## Constraints

Minimum and maximum size bounds:

```rust
// Tight: exact size
Constraints::tight(Size::new(100.0, 50.0))

// Loose: 0 to maximum
Constraints::loose(Size::new(400.0, 300.0))

// Unbounded
Constraints::unbounded()
```

## Layout Phases

| Phase | Direction | Purpose |
|-------|-----------|---------|
| Measure | Bottom-up | Compute sizes |
| Layout | Top-down | Position widgets |
| Paint | Any | Emit draw commands |

## Canvas

Abstract drawing surface:

```rust
canvas.fill_rect(bounds, color);
canvas.draw_text(text, position, style);
canvas.fill_circle(center, radius, color);
```

## Events

User interactions:

```rust
Event::MouseDown { position, button }
Event::MouseUp { position, button }
Event::KeyDown { key }
Event::TextInput { text }
Event::FocusIn
Event::FocusOut
```

## Messages

Widgets emit messages on interaction:

```rust
// Button emits ButtonClicked
if let Some(msg) = button.event(&event) {
    if msg.downcast_ref::<ButtonClicked>().is_some() {
        // Handle click
    }
}
```

## Verified Test

```rust
#[test]
fn test_core_concepts() {
    use presentar_core::{Constraints, Size};

    // Constraints work
    let c = Constraints::loose(Size::new(100.0, 100.0));
    assert_eq!(c.biggest(), Size::new(100.0, 100.0));
    assert_eq!(c.smallest(), Size::new(0.0, 0.0));
}
```
