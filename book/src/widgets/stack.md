# Stack

Overlay layout widget for stacking children on top of each other.

## Basic Usage

```rust
use presentar_widgets::Stack;

let stack = Stack::new()
    .child(Image::new("background.png"))
    .child(Text::new("Overlay"));
```

## Builder Methods

| Method | Description |
|--------|-------------|
| `alignment(Alignment)` | Position of children |
| `child(widget)` | Add child (later = on top) |

## Alignment

```rust
use presentar_widgets::stack::Alignment;

Stack::new()
    .alignment(Alignment::TopLeft)
    .alignment(Alignment::TopCenter)
    .alignment(Alignment::TopRight)
    .alignment(Alignment::CenterLeft)
    .alignment(Alignment::Center)
    .alignment(Alignment::CenterRight)
    .alignment(Alignment::BottomLeft)
    .alignment(Alignment::BottomCenter)
    .alignment(Alignment::BottomRight)
```

## Z-Order

Children added later appear on top:

```rust
let stack = Stack::new()
    .child(background)  // Bottom
    .child(content)     // Middle
    .child(overlay);    // Top
```

## Example: Badge

```rust
let badge = Stack::new()
    .alignment(Alignment::TopRight)
    .child(Button::new("Inbox"))
    .child(
        Container::new()
            .background(Color::RED)
            .corner_radius(10.0)
            .padding(4.0)
            .child(Text::new("3").color(Color::WHITE))
    );
```

## Example: Loading Overlay

```rust
let loading = Stack::new()
    .alignment(Alignment::Center)
    .child(content)
    .child(
        Container::new()
            .background(Color::rgba(0.0, 0.0, 0.0, 0.5))
            .child(Text::new("Loading...").color(Color::WHITE))
    );
```

## Verified Test

```rust
#[test]
fn test_stack_children() {
    use presentar_widgets::Stack;

    let stack = Stack::new();
    assert_eq!(stack.children().len(), 0);
}
```
