# Custom Widgets

Build your own widgets by implementing the Widget trait.

## Minimal Implementation

```rust
use presentar_core::*;
use std::any::Any;

pub struct MyWidget {
    text: String,
    bounds: Rect,
}

impl MyWidget {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            bounds: Rect::default(),
        }
    }
}

impl Widget for MyWidget {
    fn type_id(&self) -> TypeId { TypeId::of::<Self>() }

    fn measure(&self, c: Constraints) -> Size {
        let width = self.text.len() as f32 * 8.0;
        c.constrain(Size::new(width, 24.0))
    }

    fn layout(&mut self, b: Rect) -> LayoutResult {
        self.bounds = b;
        LayoutResult { size: b.size() }
    }

    fn paint(&self, canvas: &mut dyn Canvas) {
        canvas.draw_text(&self.text, self.bounds.origin(), &TextStyle::default());
    }

    fn event(&mut self, _: &Event) -> Option<Box<dyn Any + Send>> { None }
    fn children(&self) -> &[Box<dyn Widget>] { &[] }
    fn children_mut(&mut self) -> &mut [Box<dyn Widget>] { &mut [] }
}
```

## Required Methods

| Method | Purpose |
|--------|---------|
| `type_id()` | Unique type identifier |
| `measure()` | Compute intrinsic size |
| `layout()` | Position within bounds |
| `paint()` | Emit draw commands |
| `event()` | Handle user input |
| `children()` | Return child widgets |

## Optional Methods

| Method | Default | Purpose |
|--------|---------|---------|
| `is_interactive()` | `false` | Can receive events |
| `is_focusable()` | `is_interactive()` | Can receive focus |
| `accessible_name()` | `None` | Screen reader label |
| `accessible_role()` | `None` | ARIA role |
| `test_id()` | `None` | Test selector |

## With Builder Pattern

```rust
pub struct Badge {
    text: String,
    color: Color,
    test_id: Option<String>,
}

impl Badge {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            color: Color::RED,
            test_id: None,
        }
    }

    pub fn color(mut self, color: Color) -> Self {
        self.color = color;
        self
    }

    pub fn with_test_id(mut self, id: impl Into<String>) -> Self {
        self.test_id = Some(id.into());
        self
    }
}
```

## Verified Test

```rust
#[test]
fn test_custom_widget() {
    use presentar_core::{Constraints, Size, Widget, TypeId};

    struct Custom;
    impl Widget for Custom {
        fn type_id(&self) -> TypeId { TypeId::of::<Self>() }
        fn measure(&self, c: Constraints) -> Size { c.smallest() }
        fn layout(&mut self, b: presentar_core::Rect) -> presentar_core::widget::LayoutResult {
            presentar_core::widget::LayoutResult { size: b.size() }
        }
        fn paint(&self, _: &mut dyn presentar_core::Canvas) {}
        fn event(&mut self, _: &presentar_core::Event) -> Option<Box<dyn std::any::Any + Send>> { None }
        fn children(&self) -> &[Box<dyn Widget>] { &[] }
        fn children_mut(&mut self) -> &mut [Box<dyn Widget>] { &mut [] }
    }

    let w = Custom;
    assert_eq!(w.measure(Constraints::unbounded()), Size::new(0.0, 0.0));
}
```
