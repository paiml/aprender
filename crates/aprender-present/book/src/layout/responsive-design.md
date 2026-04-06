# Responsive Design

Adapt layouts to different screen sizes.

## Breakpoints

| Name | Width | Use Case |
|------|-------|----------|
| `xs` | <640px | Mobile |
| `sm` | 640-768px | Large mobile |
| `md` | 768-1024px | Tablet |
| `lg` | 1024-1280px | Desktop |
| `xl` | >1280px | Wide desktop |

## Constraint-Based Layouts

Layout naturally adapts via constraints:

```rust
// Container fills available width
let card = Container::new()
    .max_width(400.0)  // Cap at 400px
    .child(content);

// Row wraps when narrow
let grid = Row::new()
    .wrap(true)
    .child(item1)
    .child(item2);
```

## Conditional Layouts

```rust
fn build_ui(width: f32) -> impl Widget {
    if width < 768.0 {
        // Mobile: stack vertically
        Column::new()
            .child(nav)
            .child(content)
    } else {
        // Desktop: side by side
        Row::new()
            .child(nav)
            .child(content)
    }
}
```

## Resize Event

```rust
impl Widget for App {
    fn event(&mut self, event: &Event) -> Option<Box<dyn Any + Send>> {
        if let Event::Resize { width, height } = event {
            self.viewport_width = *width;
            self.needs_rebuild = true;
        }
        None
    }
}
```

## Flexible Sizing

```rust
// Flexible width
let sidebar = Container::new()
    .min_width(200.0)
    .max_width(300.0)
    .child(nav);

// Fixed width
let header = Container::new()
    .min_width(100.0)
    .max_width(100.0)
    .child(logo);
```

## Best Practices

| Practice | Description |
|----------|-------------|
| Use constraints | Let layout engine handle sizing |
| Test breakpoints | Verify at each breakpoint |
| Mobile first | Start with smallest, add complexity |

## Verified Test

```rust
#[test]
fn test_responsive_constraints() {
    use presentar_core::{Constraints, Size};

    // Mobile constraints
    let mobile = Constraints::loose(Size::new(375.0, 812.0));
    assert!(mobile.max_width < 400.0);

    // Desktop constraints
    let desktop = Constraints::loose(Size::new(1920.0, 1080.0));
    assert!(desktop.max_width > 1000.0);
}
```
