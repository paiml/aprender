# Layout Engine

Flexbox-inspired layout system.

## Overview

```
Constraints → Measure → Layout → Paint
    ↓            ↓         ↓        ↓
  bounds     sizes   positions  commands
```

## Constraints System

```rust
pub struct Constraints {
    pub min_width: f32,
    pub max_width: f32,
    pub min_height: f32,
    pub max_height: f32,
}
```

## Measure Phase

Bottom-up size computation:

```rust
fn measure(&self, constraints: Constraints) -> Size {
    // Children measure first
    let child_sizes: Vec<Size> = self.children()
        .iter()
        .map(|c| c.measure(constraints))
        .collect();

    // Parent uses child sizes
    let width = child_sizes.iter().map(|s| s.width).sum();
    constraints.constrain(Size::new(width, 50.0))
}
```

## Layout Phase

Top-down positioning:

```rust
fn layout(&mut self, bounds: Rect) -> LayoutResult {
    let mut x = bounds.x;

    for child in self.children_mut() {
        let child_bounds = Rect::new(x, bounds.y, 100.0, bounds.height);
        child.layout(child_bounds);
        x += 100.0;
    }

    LayoutResult { size: bounds.size() }
}
```

## Main/Cross Axis

| Layout | Main Axis | Cross Axis |
|--------|-----------|------------|
| Row | Horizontal | Vertical |
| Column | Vertical | Horizontal |

## Alignment

```rust
// Main axis: distribute along direction
MainAxisAlignment::Start
MainAxisAlignment::Center
MainAxisAlignment::SpaceBetween

// Cross axis: perpendicular alignment
CrossAxisAlignment::Start
CrossAxisAlignment::Center
CrossAxisAlignment::Stretch
```

## Verified Test

```rust
#[test]
fn test_layout_engine() {
    use presentar_widgets::Column;
    use presentar_core::{Constraints, Size, Rect, Widget};

    let mut col = Column::new();
    col.measure(Constraints::loose(Size::new(400.0, 300.0)));
    col.layout(Rect::new(0.0, 0.0, 400.0, 300.0));
}
```
