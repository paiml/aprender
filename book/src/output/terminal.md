# Terminal Rendering

Terminal rendering produces ASCII or Unicode art for quick visualization
directly in the command line.

## Basic Terminal Output

```rust
use trueno_viz::prelude::*;
use trueno_viz::plots::ScatterPlot;
use trueno_viz::output::TerminalRenderer;

#[cfg(feature = "terminal")]
{
    let plot = ScatterPlot::new()
        .x(&[1.0, 2.0, 3.0, 4.0, 5.0])
        .y(&[1.0, 4.0, 9.0, 16.0, 25.0])
        .build();

    let ascii = TerminalRenderer::new(80, 24)
        .render(&plot);

    println!("{}", ascii);
}
```

Output:
```text
                    Scatter Plot
  25 ┤                                            ●
     │
  20 ┤
     │                                  ●
  15 ┤
     │
  10 ┤                        ●
     │
   5 ┤              ●
     │    ●
   0 ┼────┬────┬────┬────┬────┬────┬────┬────┬────┬
     0    1    2    3    4    5    6    7    8    9
```

## Character Sets

### ASCII Only

```rust
use trueno_viz::output::{TerminalRenderer, CharSet};

let renderer = TerminalRenderer::new(80, 24)
    .charset(CharSet::Ascii);
```

Output uses: `+`, `-`, `|`, `*`, `.`

### Unicode Box Drawing

```rust
use trueno_viz::output::{TerminalRenderer, CharSet};

let renderer = TerminalRenderer::new(80, 24)
    .charset(CharSet::Unicode);
```

Output uses: `│`, `─`, `┌`, `┐`, `└`, `┘`, `┼`, `●`, `○`

### Braille Patterns

High resolution using 2x4 Braille characters:

```rust
use trueno_viz::output::{TerminalRenderer, CharSet};

let renderer = TerminalRenderer::new(80, 24)
    .charset(CharSet::Braille);
```

**Test Reference**: `src/output/terminal.rs::test_terminal_charsets`

## Custom Dimensions

```rust
use trueno_viz::output::TerminalRenderer;

// Standard terminal
let standard = TerminalRenderer::new(80, 24);

// Wide terminal
let wide = TerminalRenderer::new(120, 40);

// Compact
let compact = TerminalRenderer::new(40, 12);
```

## Line Charts

```rust
use trueno_viz::plots::LineChart;
use trueno_viz::output::TerminalRenderer;

let chart = LineChart::new()
    .x(&x)
    .y(&y)
    .build();

let ascii = TerminalRenderer::new(60, 20)
    .render(&chart);
```

Output:
```text
  Value
  100 ┤                              ╭─────
      │                         ╭────╯
   80 ┤                    ╭────╯
      │               ╭────╯
   60 ┤          ╭────╯
      │     ╭────╯
   40 ┤╭────╯
      ├─────┬─────┬─────┬─────┬─────┬─────
      0     20    40    60    80    100
                     Time
```

## Histograms

```rust
use trueno_viz::plots::Histogram;
use trueno_viz::output::TerminalRenderer;

let hist = Histogram::new(&data)
    .bins(10)
    .build();

let ascii = TerminalRenderer::new(60, 15)
    .render(&hist);
```

Output:
```text
  Frequency
  15 ┤      ██
     │      ██
  10 ┤   ██ ██ ██
     │   ██ ██ ██ ██
   5 ┤██ ██ ██ ██ ██ ██
     │██ ██ ██ ██ ██ ██ ██
   0 ┼──┴──┴──┴──┴──┴──┴──┴──┴──┴──
     0  1  2  3  4  5  6  7  8  9
                  Value
```

## Heatmaps

Using shade characters:

```rust
use trueno_viz::plots::Heatmap;
use trueno_viz::output::TerminalRenderer;

let heatmap = Heatmap::new(&data, 5, 5).build();

let ascii = TerminalRenderer::new(30, 15)
    .render(&heatmap);
```

Output:
```text
  ░░▒▒▓▓██░░
  ▒▒▓▓████▒▒
  ▓▓████████
  ██████████
  ░░▓▓██████
```

## Color Support

For terminals with ANSI color support:

```rust
use trueno_viz::output::{TerminalRenderer, ColorMode};

// No colors (default)
let mono = TerminalRenderer::new(80, 24)
    .color_mode(ColorMode::None);

// 16 ANSI colors
let ansi = TerminalRenderer::new(80, 24)
    .color_mode(ColorMode::Ansi16);

// 256 colors
let extended = TerminalRenderer::new(80, 24)
    .color_mode(ColorMode::Ansi256);

// True color (24-bit)
let true_color = TerminalRenderer::new(80, 24)
    .color_mode(ColorMode::TrueColor);
```

## Inline Rendering

For quick debugging:

```rust
use trueno_viz::plots::ScatterPlot;

let plot = ScatterPlot::new()
    .x(&x)
    .y(&y)
    .build();

// Quick terminal output
plot.print_terminal();
```

## Complete Example

```rust
use trueno_viz::prelude::*;
use trueno_viz::plots::ScatterPlot;
use trueno_viz::output::{TerminalRenderer, CharSet, ColorMode};

fn main() {
    let x: Vec<f32> = (0..20).map(|i| i as f32).collect();
    let y: Vec<f32> = x.iter().map(|x| x * x / 20.0).collect();

    let plot = ScatterPlot::new()
        .x(&x)
        .y(&y)
        .title("Quadratic Growth")
        .build();

    // Unicode with colors
    let renderer = TerminalRenderer::new(70, 20)
        .charset(CharSet::Unicode)
        .color_mode(ColorMode::Ansi16);

    let output = renderer.render(&plot);
    println!("{}", output);
}
```

## Use Cases

| Scenario | Best Option |
|----------|-------------|
| SSH session | ASCII |
| Modern terminal | Unicode |
| High detail | Braille |
| Quick debug | Default |
| CI/CD logs | ASCII |

## Next Chapter

Continue to [SIMD Acceleration](../accel/simd.md) for performance optimization.
