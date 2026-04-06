# Slider

Numeric range input widget.

## Basic Usage

```rust
use presentar_widgets::Slider;

let slider = Slider::new(0.0, 100.0)
    .value(50.0);
```

## Builder Methods

| Method | Description |
|--------|-------------|
| `value(f32)` | Current value |
| `step(f32)` | Step increment |
| `track_color(Color)` | Track background |
| `active_color(Color)` | Filled portion |
| `thumb_color(Color)` | Thumb color |
| `disabled(bool)` | Disable interaction |

## Example

```rust
let volume = Slider::new(0.0, 100.0)
    .value(75.0)
    .step(5.0)
    .active_color(Color::from_hex("#4f46e5").unwrap())
    .with_test_id("volume-slider");
```

## Event Handling

```rust
use presentar_widgets::SliderChanged;

if let Some(msg) = slider.event(&event) {
    if let Some(changed) = msg.downcast_ref::<SliderChanged>() {
        println!("New value: {}", changed.value);
    }
}
```

## Getters

```rust
let value = slider.get_value();
let (min, max) = slider.get_range();
```

## Accessibility

- Role: `Slider`
- Keyboard: Arrow keys adjust value
- Accessible name from label

## Verified Test

```rust
#[test]
fn test_slider_value() {
    use presentar_widgets::Slider;

    let slider = Slider::new(0.0, 100.0).value(50.0);
    assert_eq!(slider.get_value(), 50.0);
}
```
