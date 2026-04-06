# Widget Spec

Complete Widget trait specification.

## Required Methods

```rust
pub trait Widget {
    /// Measure intrinsic size within constraints
    fn measure(&self, constraints: &Constraints) -> Size;

    /// Position widget and children at given size
    fn layout(&mut self, size: Size);

    /// Emit draw commands to canvas
    fn paint(&self, canvas: &mut dyn Canvas);
}
```

## Optional Methods

```rust
pub trait Widget {
    /// Child widgets (default: empty)
    fn children(&self) -> &[Box<dyn Widget>] { &[] }

    /// Accessible role (default: None)
    fn accessible_role(&self) -> Option<AccessibleRole> { None }

    /// Accessible name (default: None)
    fn accessible_name(&self) -> Option<&str> { None }

    /// Whether widget can receive focus
    fn is_focusable(&self) -> bool { false }

    /// Handle event, return true if consumed
    fn on_event(&mut self, event: &Event) -> bool { false }
}
```

## Lifecycle

```
┌─────────┐     ┌─────────┐     ┌─────────┐
│ measure │ ──> │ layout  │ ──> │  paint  │
└─────────┘     └─────────┘     └─────────┘
     ▲               │               │
     │               ▼               ▼
     │         Position set    Draw commands
     │
Constraints from parent
```

## Contract

| Method | Pre-condition | Post-condition |
|--------|---------------|----------------|
| measure | Valid constraints | Size within constraints |
| layout | Size from measure | Children positioned |
| paint | Layout complete | Commands emitted |

## Example Implementation

```rust
struct MyWidget {
    text: String,
    bounds: Rect,
}

impl Widget for MyWidget {
    fn measure(&self, constraints: &Constraints) -> Size {
        let width = (self.text.len() * 10) as f32;
        constraints.constrain(Size::new(width, 20.0))
    }

    fn layout(&mut self, size: Size) {
        self.bounds = Rect::from_size(size);
    }

    fn paint(&self, canvas: &mut dyn Canvas) {
        canvas.draw_text(&self.text, self.bounds.origin(), &TextStyle::default());
    }
}
```

## Verified Test

```rust
#[test]
fn test_widget_spec_lifecycle() {
    use presentar_core::{Constraints, Size};

    // Simulate widget lifecycle
    struct TestWidget { size: Size }

    impl TestWidget {
        fn measure(&self, c: &Constraints) -> Size {
            c.constrain(Size::new(100.0, 50.0))
        }

        fn layout(&mut self, size: Size) {
            self.size = size;
        }
    }

    let mut widget = TestWidget { size: Size::ZERO };
    let constraints = Constraints::new(0.0, 200.0, 0.0, 100.0);

    // 1. Measure
    let size = widget.measure(&constraints);
    assert_eq!(size.width, 100.0);
    assert_eq!(size.height, 50.0);

    // 2. Layout
    widget.layout(size);
    assert_eq!(widget.size, size);
}
```
