# Dashboard Widgets

Trueno-viz provides lightweight, composable widgets designed for ML experiment dashboards. These widgets are optimized for embedding in tables, status panels, and real-time monitoring interfaces.

## Available Widgets

| Widget | Purpose | Use Case |
|--------|---------|----------|
| [Sparkline](./sparkline.md) | Mini line charts | Loss/accuracy trends |
| [ResourceBar](./resource-bar.md) | Plan vs actual bars | Budget tracking |
| [RunTable](./run-table.md) | Sortable run tables | Experiment status |

## Quick Example

```rust
use trueno_viz::prelude::*;

// Create a sparkline showing loss decreasing
let sparkline = Sparkline::new(&[0.9, 0.7, 0.5, 0.3, 0.2])
    .dimensions(100, 20)
    .with_trend_indicator();

// Check the trend direction
match sparkline.trend() {
    TrendDirection::Falling => println!("Loss is decreasing!"),
    TrendDirection::Rising => println!("Loss is increasing"),
    TrendDirection::Stable => println!("Loss is stable"),
}

// Create a resource bar for GPU hours
let gpu_bar = ResourceBar::new("GPU Hours", 100.0, 75.0, "hours")
    .dimensions(200, 20);

if gpu_bar.is_over_budget() {
    println!("Warning: Over budget!");
} else {
    println!("Usage: {:.1}%", gpu_bar.percentage());
}
```

## Running the Example

```bash
cargo run --example dashboard_widgets
```

This demonstrates all three widgets with sample ML experiment data.

## Design Philosophy

These widgets follow the trueno-viz design principles:

1. **Minimal footprint**: Small dimensions suitable for tables and dashboards
2. **Zero dependencies**: Pure Rust rendering with no external libraries
3. **Consistent API**: Builder pattern matching other trueno-viz components
4. **Hardware accelerated**: SIMD-optimized rendering via trueno core
