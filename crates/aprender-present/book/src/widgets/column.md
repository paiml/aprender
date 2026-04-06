# Column

Vertical layout widget for arranging children.

## Basic Usage

```rust
use presentar_widgets::Column;

let column = Column::new()
    .child(Text::new("First"))
    .child(Text::new("Second"))
    .child(Text::new("Third"));
```

## Builder Methods

| Method | Description |
|--------|-------------|
| `gap(f32)` | Space between children |
| `main_axis_alignment(MainAxisAlignment)` | Vertical alignment |
| `cross_axis_alignment(CrossAxisAlignment)` | Horizontal alignment |
| `child(widget)` | Add child widget |

## MainAxisAlignment

```rust
use presentar_widgets::row::MainAxisAlignment;

Column::new()
    .main_axis_alignment(MainAxisAlignment::Start)        // Top
    .main_axis_alignment(MainAxisAlignment::Center)       // Middle
    .main_axis_alignment(MainAxisAlignment::End)          // Bottom
    .main_axis_alignment(MainAxisAlignment::SpaceBetween) // Spread
    .main_axis_alignment(MainAxisAlignment::SpaceEvenly)  // Even gaps
```

## CrossAxisAlignment

```rust
use presentar_widgets::row::CrossAxisAlignment;

Column::new()
    .cross_axis_alignment(CrossAxisAlignment::Start)   // Left
    .cross_axis_alignment(CrossAxisAlignment::Center)  // Center
    .cross_axis_alignment(CrossAxisAlignment::End)     // Right
    .cross_axis_alignment(CrossAxisAlignment::Stretch) // Fill width
```

## Example

```rust
let form = Column::new()
    .gap(16.0)
    .cross_axis_alignment(CrossAxisAlignment::Stretch)
    .child(TextInput::new().placeholder("Username"))
    .child(TextInput::new().placeholder("Password"))
    .child(Button::new("Login"));
```

## Verified Test

```rust
#[test]
fn test_column_layout() {
    use presentar_widgets::Column;
    use presentar_widgets::row::MainAxisAlignment;

    let column = Column::new()
        .gap(10.0)
        .main_axis_alignment(MainAxisAlignment::Center);

    assert_eq!(column.children().len(), 0);
}
```
