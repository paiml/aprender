# Container

Single-child wrapper with padding, decoration, and constraints.

## Basic Usage

```rust
use presentar_widgets::Container;

let container = Container::new()
    .padding(16.0)
    .child(Text::new("Hello"));
```

## Builder Methods

| Method | Description |
|--------|-------------|
| `padding(f32)` | Uniform padding |
| `padding_horizontal(f32)` | Left/right padding |
| `padding_vertical(f32)` | Top/bottom padding |
| `background(Color)` | Background color |
| `border(Color, f32)` | Border color and width |
| `corner_radius(f32)` | Rounded corners |
| `min_width(f32)` | Minimum width |
| `min_height(f32)` | Minimum height |
| `max_width(f32)` | Maximum width |
| `max_height(f32)` | Maximum height |

## Example

```rust
let card = Container::new()
    .padding(24.0)
    .background(Color::WHITE)
    .corner_radius(8.0)
    .border(Color::from_hex("#e5e7eb").unwrap(), 1.0)
    .child(
        Column::new()
            .gap(8.0)
            .child(Text::new("Title").font_size(18.0))
            .child(Text::new("Description"))
    );
```

## Constraints

```rust
let fixed = Container::new()
    .min_width(200.0)
    .max_width(400.0)
    .child(content);
```

## Verified Test

```rust
#[test]
fn test_container_padding() {
    use presentar_widgets::Container;
    use presentar_core::{Constraints, Size, Widget};

    let container = Container::new()
        .padding(10.0)
        .min_width(100.0)
        .min_height(50.0);

    let size = container.measure(Constraints::unbounded());
    assert!(size.width >= 100.0);
    assert!(size.height >= 50.0);
}
```
