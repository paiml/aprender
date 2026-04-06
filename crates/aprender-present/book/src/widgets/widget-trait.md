# Widget Trait

The `Widget` trait is the foundation of Presentar's UI system. Every visual element implements this trait.

## Core Definition

```rust
pub trait Widget: Send + Sync {
    /// Unique type identifier for diffing
    fn type_id(&self) -> TypeId;

    /// Compute intrinsic size constraints
    fn measure(&self, constraints: Constraints) -> Size;

    /// Position children within allocated bounds
    fn layout(&mut self, bounds: Rect) -> LayoutResult;

    /// Generate draw commands
    fn paint(&self, canvas: &mut dyn Canvas);

    /// Handle input events, return state mutations
    fn event(&mut self, event: &Event) -> Option<Box<dyn Any + Send>>;

    /// Child widgets for tree traversal
    fn children(&self) -> &[Box<dyn Widget>];

    /// Mutable child access
    fn children_mut(&mut self) -> &mut [Box<dyn Widget>];

    // Optional methods with defaults
    fn is_interactive(&self) -> bool { false }
    fn is_focusable(&self) -> bool { self.is_interactive() }
    fn accessible_name(&self) -> Option<&str> { None }
    fn accessible_role(&self) -> AccessibleRole { AccessibleRole::None }
    fn test_id(&self) -> Option<&str> { None }
}
```

## The Widget Lifecycle

Every widget goes through three phases each frame:

```
┌──────────┐    ┌──────────┐    ┌──────────┐
│ MEASURE  │───▶│  LAYOUT  │───▶│  PAINT   │
│(bottom-up)│   │(top-down)│    │(any order)│
└──────────┘    └──────────┘    └──────────┘
```

### 1. Measure Phase (Bottom-Up)

Widgets compute their intrinsic size given constraints:

```rust
fn measure(&self, constraints: Constraints) -> Size {
    // Children are measured first
    let child_sizes: Vec<Size> = self.children()
        .iter()
        .map(|c| c.measure(constraints))
        .collect();

    // Parent computes its size based on children
    let total_height: f32 = child_sizes.iter().map(|s| s.height).sum();
    let max_width = child_sizes.iter().map(|s| s.width).max();

    constraints.constrain(Size::new(max_width, total_height))
}
```

### 2. Layout Phase (Top-Down)

Widgets position themselves and their children:

```rust
fn layout(&mut self, bounds: Rect) -> LayoutResult {
    let mut y = bounds.y;

    for child in self.children_mut() {
        let child_bounds = Rect::new(bounds.x, y, bounds.width, child_height);
        child.layout(child_bounds);
        y += child_height;
    }

    LayoutResult { size: bounds.size() }
}
```

### 3. Paint Phase

Widgets emit draw commands to the canvas:

```rust
fn paint(&self, canvas: &mut dyn Canvas) {
    // Paint self
    canvas.fill_rect(self.bounds, self.background_color);

    // Paint children
    for child in self.children() {
        child.paint(canvas);
    }
}
```

## Implementing a Custom Widget

```rust
use presentar_core::*;

pub struct MyWidget {
    text: String,
    bounds: Rect,
}

impl Widget for MyWidget {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        // Compute size based on text length
        let width = self.text.len() as f32 * 8.0;
        let height = 24.0;
        constraints.constrain(Size::new(width, height))
    }

    fn layout(&mut self, bounds: Rect) -> LayoutResult {
        self.bounds = bounds;
        LayoutResult { size: bounds.size() }
    }

    fn paint(&self, canvas: &mut dyn Canvas) {
        canvas.draw_text(&self.text, self.bounds.origin(), &TextStyle::default());
    }

    fn event(&mut self, _event: &Event) -> Option<Box<dyn Any + Send>> {
        None
    }

    fn children(&self) -> &[Box<dyn Widget>] {
        &[]
    }

    fn children_mut(&mut self) -> &mut [Box<dyn Widget>] {
        &mut []
    }
}
```

## Testing Widgets

Use the test harness for widget testing:

```rust
use presentar_test::*;

#[test]
fn test_my_widget() {
    let widget = MyWidget::new("Hello");
    let harness = Harness::new(widget);

    harness.assert_exists("[data-testid='my-widget']");
    harness.assert_text("[data-testid='my-widget']", "Hello");
}
```

## Accessibility

Widgets should provide accessibility information:

```rust
impl Widget for Button {
    fn is_interactive(&self) -> bool { true }
    fn is_focusable(&self) -> bool { !self.disabled }
    fn accessible_name(&self) -> Option<&str> { Some(&self.label) }
    fn accessible_role(&self) -> AccessibleRole { AccessibleRole::Button }
}
```

## Next Steps

- [Core Widgets](./core-widgets.md) - Built-in widget library
- [Custom Widgets](./custom-widgets.md) - Building your own widgets
