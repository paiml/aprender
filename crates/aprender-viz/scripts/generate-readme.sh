#!/usr/bin/env bash
# Generate README.md with live output from examples
# Run via: make readme
set -euo pipefail

DEMO_OUTPUT=$(cargo run --example readme_demo --quiet 2>&1)

cat << 'HEADER'
# trueno-viz

SIMD/GPU/WASM-accelerated visualization for Data Science, Physics, and ML/DL.

**Pure Rust** - zero JavaScript, zero browser dependencies.

## Quick Example

```rust
use trueno_viz::prelude::*;
use trueno_viz::output::{TerminalEncoder, TerminalMode};

let plot = ScatterPlot::new()
    .x(&[1.0, 2.0, 3.0, 4.0])
    .y(&[1.0, 4.0, 2.0, 8.0])
    .build()?;

let fb = plot.to_framebuffer()?;
TerminalEncoder::new()
    .mode(TerminalMode::Ascii)
    .print(&fb);
```

## ASCII Output Demo

```
$ cargo run --example readme_demo --quiet
HEADER

echo '```'
echo "$DEMO_OUTPUT"
echo '```'

cat << 'FOOTER'

## Features

- **SIMD-aligned Framebuffer** - 64-byte alignment for AVX-512
- **Plot Types**: Scatter, Line, Heatmap, Histogram, Confusion Matrix, ROC/PR, Loss curves
- **Output Formats**: PNG, Terminal (ASCII/Unicode/ANSI 24-bit color)
- **ML-focused**: Built-in metrics, smoothing, normalization

## Installation

```toml
[dependencies]
trueno-viz = "0.1"
```

## Examples

```bash
cargo run --example scatter_basic
cargo run --example heatmap_correlation
cargo run --example loss_training
cargo run --example confusion_matrix_ml
cargo run --example roc_pr_curves
cargo run --example terminal_output
```

## License

MIT OR Apache-2.0
FOOTER
