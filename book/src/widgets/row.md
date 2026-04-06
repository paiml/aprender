# Row

Horizontal layout widget for arranging children.

## Basic Usage

```rust
use presentar_widgets::Row;

let row = Row::new()
    .child(Button::new("A"))
    .child(Button::new("B"))
    .child(Button::new("C"));
```

## Builder Methods

| Method | Description |
|--------|-------------|
| `gap(f32)` | Space between children |
| `main_axis_alignment(MainAxisAlignment)` | Horizontal alignment |
| `cross_axis_alignment(CrossAxisAlignment)` | Vertical alignment |
| `child(widget)` | Add child widget |

## MainAxisAlignment

```rust
use presentar_widgets::row::MainAxisAlignment;

Row::new()
    .main_axis_alignment(MainAxisAlignment::Start)        // Left
    .main_axis_alignment(MainAxisAlignment::Center)       // Center
    .main_axis_alignment(MainAxisAlignment::End)          // Right
    .main_axis_alignment(MainAxisAlignment::SpaceBetween) // Spread
    .main_axis_alignment(MainAxisAlignment::SpaceAround)  // Equal space
    .main_axis_alignment(MainAxisAlignment::SpaceEvenly)  // Even gaps
```

## CrossAxisAlignment

```rust
use presentar_widgets::row::CrossAxisAlignment;

Row::new()
    .cross_axis_alignment(CrossAxisAlignment::Start)   // Top
    .cross_axis_alignment(CrossAxisAlignment::Center)  // Middle
    .cross_axis_alignment(CrossAxisAlignment::End)     // Bottom
    .cross_axis_alignment(CrossAxisAlignment::Stretch) // Fill height
```

## Example

```rust
let toolbar = Row::new()
    .gap(8.0)
    .main_axis_alignment(MainAxisAlignment::SpaceBetween)
    .cross_axis_alignment(CrossAxisAlignment::Center)
    .child(Text::new("Title"))
    .child(Button::new("Action"));
```

## Verified Test

```rust
#[test]
fn test_row_layout() {
    use presentar_widgets::Row;
    use presentar_widgets::row::MainAxisAlignment;
    use presentar_core::{Constraints, Size, Widget};

    let row = Row::new()
        .gap(10.0)
        .main_axis_alignment(MainAxisAlignment::Start);

    assert_eq!(row.children().len(), 0);
}
```
