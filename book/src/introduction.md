# Introduction

**Trueno-Viz** is a hardware-accelerated visualization library for Rust, designed for
data science and machine learning workflows. It provides a Grammar of Graphics API
with pure Rust implementation - no JavaScript, HTML, or browser dependencies.

## Philosophy

Trueno-Viz follows three core principles:

1. **Literate Visualization**: Code should read like documentation
2. **Test-Driven Development**: Every feature is test-backed
3. **Zero Dependencies**: Pure Rust rendering to PNG, SVG, or terminal

## Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                    Grammar of Graphics                       │
│  ┌─────────┐ ┌─────┐ ┌──────┐ ┌──────┐ ┌───────┐ ┌───────┐ │
│  │  Data   │ │ Aes │ │ Geom │ │ Stat │ │ Scale │ │ Theme │ │
│  └────┬────┘ └──┬──┘ └──┬───┘ └──┬───┘ └───┬───┘ └───┬───┘ │
│       └─────────┴───────┴────────┴─────────┴─────────┘     │
│                           │                                 │
│                    ┌──────▼──────┐                          │
│                    │   Render    │                          │
│                    └──────┬──────┘                          │
│       ┌──────────────────┼─────────────────┐               │
│  ┌────▼────┐        ┌────▼────┐       ┌────▼────┐          │
│  │   PNG   │        │   SVG   │       │Terminal │          │
│  │ Encoder │        │ Writer  │       │ Output  │          │
│  └─────────┘        └─────────┘       └─────────┘          │
└─────────────────────────────────────────────────────────────┘
```

## Quick Example

Every example in this book is reproducible. Here's your first visualization:

```rust
use trueno_viz::prelude::*;
use trueno_viz::plots::ScatterPlot;

// Create sample data
let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
let y = vec![2.1, 3.9, 6.2, 7.8, 10.1];

// Build a scatter plot
let plot = ScatterPlot::new()
    .x(&x)
    .y(&y)
    .color(Rgba::BLUE)
    .title("Linear Relationship")
    .build();

// Verify the plot was constructed correctly
assert_eq!(plot.len(), 5);
```

**Test Verification**: This example is extracted from `src/plots/scatter.rs` tests.

## Feature Matrix

| Feature | Status | Acceleration |
|---------|--------|--------------|
| Scatter Plot | Complete | SIMD |
| Line Chart | Complete | SIMD |
| Histogram | Complete | SIMD |
| Heatmap | Complete | SIMD/GPU |
| Box Plot | Complete | SIMD |
| Violin Plot | Complete | SIMD |
| ROC Curve | Complete | SIMD |
| PR Curve | Complete | SIMD |
| Confusion Matrix | Complete | SIMD |

## Why Trueno-Viz?

1. **Performance**: SIMD-accelerated rendering using the trueno core
2. **Portability**: Compile to native, WASM, or embedded targets
3. **Integration**: Seamless ML pipeline integration with aprender
4. **Quality**: 95%+ test coverage, property-based testing

## Book Structure

This book follows literate programming principles (Knuth, 1984). Each chapter:

- Explains concepts with working code
- Provides complete, runnable examples
- References the test suite for verification
- Builds upon previous chapters

Continue to [Quick Start](./getting-started/quickstart.md) to create your first visualization.

## References

- Wilkinson, L. (2005). *The Grammar of Graphics*. Springer.
- Knuth, D. E. (1984). *Literate Programming*. The Computer Journal.
