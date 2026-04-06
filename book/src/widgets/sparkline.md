# Sparkline

Sparklines are minimalist line charts designed to be embedded inline with text or in table cells. They're commonly used to show loss/accuracy trends over training epochs.

## Basic Usage

```rust
use trueno_viz::prelude::*;

// Create a sparkline from loss values
let loss_values = vec![0.9, 0.7, 0.5, 0.3, 0.2, 0.15, 0.1];
let sparkline = Sparkline::new(&loss_values)
    .dimensions(100, 20)
    .color(Rgba::rgb(66, 133, 244));

// Render to framebuffer
let fb = sparkline.to_framebuffer()?;
```

## Trend Indicator

Sparklines can automatically detect and display trend direction:

```rust
use trueno_viz::prelude::*;

let sparkline = Sparkline::new(&[0.9, 0.7, 0.5, 0.3, 0.1])
    .with_trend_indicator();

// Get trend direction
let trend = sparkline.trend();
println!("Trend: {} {}", trend.indicator(), match trend {
    TrendDirection::Rising => "Rising",
    TrendDirection::Falling => "Falling",
    TrendDirection::Stable => "Stable",
});
// Output: Trend: ↓ Falling
```

## Trend Direction

| Direction | Indicator | Meaning |
|-----------|-----------|---------|
| `Rising` | ↑ | Values increasing |
| `Falling` | ↓ | Values decreasing |
| `Stable` | → | Values relatively constant |

## Stability Threshold

The stability threshold controls how sensitive trend detection is:

```rust
// Default threshold (5% of range)
let sparkline = Sparkline::new(&data);

// Custom threshold - 10% of range considered "stable"
let sparkline = Sparkline::new(&data)
    .stability_threshold(0.1);
```

## API Reference

### Sparkline

| Method | Description |
|--------|-------------|
| `new(data)` | Create from slice of `f64` values |
| `dimensions(w, h)` | Set width and height in pixels |
| `color(rgba)` | Set line color |
| `with_trend_indicator()` | Enable trend arrow |
| `stability_threshold(f)` | Set stability threshold (0.0-1.0) |
| `trend()` | Get `TrendDirection` |
| `render(fb)` | Render to existing framebuffer |
| `to_framebuffer()` | Render to new framebuffer |

### TrendDirection

| Variant | `indicator()` |
|---------|---------------|
| `Rising` | `"↑"` |
| `Falling` | `"↓"` |
| `Stable` | `"→"` |
