# Performance Testing

Measure and validate widget performance.

## Frame Time

Target: **<16ms** for 60fps

```rust
use std::time::Instant;

let start = Instant::now();
widget.paint(&mut canvas);
let elapsed = start.elapsed();

assert!(elapsed.as_millis() < 16, "Paint exceeded 16ms budget");
```

## Measure Performance

```rust
fn measure_layout_time<W: Widget>(widget: &mut W, bounds: Rect) -> u128 {
    let start = Instant::now();
    widget.layout(bounds);
    start.elapsed().as_micros()
}
```

## Benchmark Pattern

```rust
#[test]
fn bench_button_paint() {
    use presentar_widgets::Button;
    use presentar_core::{RecordingCanvas, Rect, Widget};

    let mut button = Button::new("Test");
    button.layout(Rect::new(0.0, 0.0, 100.0, 40.0));

    let iterations = 1000;
    let start = std::time::Instant::now();

    for _ in 0..iterations {
        let mut canvas = RecordingCanvas::new();
        button.paint(&mut canvas);
    }

    let elapsed = start.elapsed();
    let per_paint = elapsed.as_nanos() / iterations;

    println!("Paint: {}ns/iter", per_paint);
    assert!(per_paint < 100_000, "Paint should be <100µs");
}
```

## Performance Budgets

| Operation | Budget |
|-----------|--------|
| Widget measure | <100µs |
| Widget layout | <100µs |
| Widget paint | <100µs |
| Full frame | <16ms |
| Initial load | <100ms |

## Memory Testing

```rust
fn measure_widget_size<W: Widget>(widget: &W) -> usize {
    std::mem::size_of_val(widget)
}
```

## Draw Command Count

```rust
use presentar_core::RecordingCanvas;

let mut canvas = RecordingCanvas::new();
widget.paint(&mut canvas);

assert!(canvas.command_count() < 100, "Too many draw commands");
```

## Verified Test

```rust
#[test]
fn test_button_performance() {
    use presentar_widgets::Button;
    use presentar_core::{Constraints, Size, Widget};

    let button = Button::new("Test");
    let start = std::time::Instant::now();

    for _ in 0..1000 {
        button.measure(Constraints::loose(Size::new(1000.0, 1000.0)));
    }

    let elapsed = start.elapsed();
    assert!(elapsed.as_millis() < 100, "1000 measures < 100ms");
}
```
