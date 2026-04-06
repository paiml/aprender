# Measure-Layout-Paint

The three-phase rendering cycle for all widgets.

## Overview

```
┌──────────┐    ┌──────────┐    ┌──────────┐
│ MEASURE  │───▶│  LAYOUT  │───▶│  PAINT   │
│(bottom-up)│   │(top-down)│    │(any order)│
└──────────┘    └──────────┘    └──────────┘
```

## Phase 1: Measure (Bottom-Up)

Compute intrinsic size given constraints.

```rust
fn measure(&self, constraints: Constraints) -> Size {
    // Leaf widget
    let text_width = self.text.len() as f32 * 8.0;
    constraints.constrain(Size::new(text_width, 24.0))
}
```

Parent widgets measure children first:

```rust
fn measure(&self, constraints: Constraints) -> Size {
    let mut total_height = 0.0;
    let mut max_width = 0.0;

    for child in &self.children {
        let child_size = child.measure(constraints);
        total_height += child_size.height;
        max_width = max_width.max(child_size.width);
    }

    constraints.constrain(Size::new(max_width, total_height))
}
```

## Phase 2: Layout (Top-Down)

Position children within allocated bounds.

```rust
fn layout(&mut self, bounds: Rect) -> LayoutResult {
    self.bounds = bounds;
    let mut y = bounds.y;

    for child in &mut self.children {
        let child_bounds = Rect::new(bounds.x, y, bounds.width, 50.0);
        child.layout(child_bounds);
        y += 50.0;
    }

    LayoutResult { size: bounds.size() }
}
```

## Phase 3: Paint (Any Order)

Emit draw commands to canvas.

```rust
fn paint(&self, canvas: &mut dyn Canvas) {
    // Paint self
    canvas.fill_rect(self.bounds, self.background);

    // Paint children
    for child in &self.children {
        child.paint(canvas);
    }
}
```

## Key Rules

| Phase | Direction | Mutation | Purpose |
|-------|-----------|----------|---------|
| Measure | Bottom-up | Read-only | Compute sizes |
| Layout | Top-down | Writes bounds | Position widgets |
| Paint | Any | Read-only | Emit draw commands |

## Verified Test

```rust
#[test]
fn test_measure_layout_paint_cycle() {
    use presentar_widgets::Button;
    use presentar_core::{Constraints, Rect, Size, Widget, RecordingCanvas};

    let mut button = Button::new("Test");

    // 1. Measure
    let size = button.measure(Constraints::loose(Size::new(1000.0, 1000.0)));
    assert!(size.width > 0.0);

    // 2. Layout
    let result = button.layout(Rect::new(0.0, 0.0, size.width, size.height));
    assert_eq!(result.size, size);

    // 3. Paint
    let mut canvas = RecordingCanvas::new();
    button.paint(&mut canvas);
    assert!(canvas.command_count() > 0);
}
```
