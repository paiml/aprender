# ResourceBar

ResourceBar displays a horizontal bar comparing planned vs actual resource usage. Ideal for tracking GPU hours, training time, compute costs, or any budgeted metric.

## Basic Usage

```rust
use trueno_viz::prelude::*;

// Track GPU hours: planned 100, used 75
let bar = ResourceBar::new("GPU Hours", 100.0, 75.0, "hours")
    .dimensions(200, 20);

println!("Usage: {:.1}%", bar.percentage()); // 75.0%
println!("Over budget: {}", bar.is_over_budget()); // false

let fb = bar.to_framebuffer()?;
```

## Over Budget Detection

The bar automatically changes color when actual exceeds planned:

```rust
use trueno_viz::prelude::*;

let under = ResourceBar::new("Time", 10.0, 5.0, "hours");
assert!(!under.is_over_budget()); // Green bar

let over = ResourceBar::new("Time", 10.0, 15.0, "hours");
assert!(over.is_over_budget()); // Red bar with planned marker
```

## Visual Representation

| State | Bar Color | Planned Marker |
|-------|-----------|----------------|
| Under budget | Green | None |
| At budget | Green | None |
| Over budget | Red | Black vertical line |

## Custom Colors

```rust
use trueno_viz::prelude::*;

let bar = ResourceBar::new("Cost", 1000.0, 1200.0, "$")
    .under_budget_color(Rgba::rgb(76, 175, 80))   // Material Green
    .over_budget_color(Rgba::rgb(244, 67, 54))    // Material Red
    .background_color(Rgba::rgb(224, 224, 224));  // Light Gray
```

## Percentage Calculation

```rust
use trueno_viz::prelude::*;

// Normal case
let bar = ResourceBar::new("Test", 100.0, 50.0, "units");
assert_eq!(bar.percentage(), 50.0);

// Over budget
let bar = ResourceBar::new("Test", 100.0, 150.0, "units");
assert_eq!(bar.percentage(), 150.0);

// Edge case: zero planned, non-zero actual
let bar = ResourceBar::new("Test", 0.0, 50.0, "units");
assert!(bar.percentage().is_infinite());
```

## API Reference

### ResourceBar

| Method | Description |
|--------|-------------|
| `new(label, planned, actual, unit)` | Create a new resource bar |
| `dimensions(w, h)` | Set width and height in pixels |
| `under_budget_color(rgba)` | Set color when under budget |
| `over_budget_color(rgba)` | Set color when over budget |
| `background_color(rgba)` | Set background bar color |
| `percentage()` | Get actual/planned as percentage |
| `is_over_budget()` | Check if actual > planned |
| `label()` | Get the label |
| `planned()` | Get planned value |
| `actual()` | Get actual value |
| `unit()` | Get the unit string |
| `fill_color()` | Get current fill color based on status |
| `render(fb)` | Render to existing framebuffer |
| `to_framebuffer()` | Render to new framebuffer |
