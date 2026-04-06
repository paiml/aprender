# Widget Tree

The UI is represented as a tree of widgets.

## Structure

```
Root
├── Column
│   ├── Text ("Title")
│   ├── Row
│   │   ├── Button ("OK")
│   │   └── Button ("Cancel")
│   └── Text ("Footer")
```

## Building Trees

```rust
let tree = Column::new()
    .child(Text::new("Title"))
    .child(
        Row::new()
            .child(Button::new("OK"))
            .child(Button::new("Cancel"))
    )
    .child(Text::new("Footer"));
```

## Traversal

### Depth-First

```rust
fn visit_all(widget: &dyn Widget) {
    // Process current
    println!("Widget: {:?}", widget.type_id());

    // Process children
    for child in widget.children() {
        visit_all(child.as_ref());
    }
}
```

### Finding by Selector

```rust
fn find_by_test_id<'a>(widget: &'a dyn Widget, id: &str) -> Option<&'a dyn Widget> {
    if widget.test_id() == Some(id) {
        return Some(widget);
    }
    for child in widget.children() {
        if let Some(found) = find_by_test_id(child.as_ref(), id) {
            return Some(found);
        }
    }
    None
}
```

## Lifecycle

| Phase | Direction | Action |
|-------|-----------|--------|
| Measure | Bottom-up | Leaf→Root size computation |
| Layout | Top-down | Root→Leaf positioning |
| Paint | Any | Emit draw commands |
| Event | Top-down | Route to target |

## Children Access

```rust
// Immutable access
fn children(&self) -> &[Box<dyn Widget>];

// Mutable access (for layout)
fn children_mut(&mut self) -> &mut [Box<dyn Widget>];
```

## Verified Test

```rust
#[test]
fn test_widget_tree() {
    use presentar_widgets::{Column, Button};
    use presentar_core::Widget;

    let tree = Column::new()
        .child(Button::new("A"))
        .child(Button::new("B"));

    assert_eq!(tree.children().len(), 2);
}
```
