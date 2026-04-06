# Pipeline TUI Example

The `pipeline_tui.rs` example provides a terminal user interface for visualizing the inference pipeline in real-time.

## Overview

This example demonstrates:
- Real-time token generation visualization
- Latency sparklines showing per-token timing
- Memory usage tracking
- ANSI 256-color rendering with Unicode box-drawing characters

## Running the Example

```bash
cargo run --example pipeline_tui
```

## Features

### Real-Time Token Display

As tokens are generated, they appear in the terminal with color-coding:
- Generated tokens highlighted
- Latency displayed per token
- Running throughput calculation

### Latency Sparklines

Visual representation of per-token latency using Unicode block characters:

```
Token Latency: ▁▂▃▄▅▆▇█▇▆▅▄▃▂▁
               └─ Each bar represents one token's generation time
```

### Memory Tracking

Displays current memory usage:
- Model weights
- KV cache
- Activation memory
- Total VRAM (if GPU)

### Color Scheme

The TUI uses ANSI 256-color for clear visualization:
- Cyan: Headers and borders
- Green: Success/completed tokens
- Yellow: In-progress operations
- Red: Errors or warnings

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  REALIZAR PIPELINE TUI                          │
├─────────────────────────────────────────────────────────────────┤
│ Prompt: "Hello, world!"                                         │
│                                                                 │
│ Generated: Hello, world! This is a demonstration of...          │
│                                                                 │
│ Latency: ▁▂▃▄▅▆▇█▇▆▅▄▃▂▁  (avg: 15.2ms/tok)                   │
│                                                                 │
│ Throughput: 65.8 tok/s                                          │
│ Memory: 2.4 GB (Model: 2.1 GB, KV: 0.3 GB)                     │
└─────────────────────────────────────────────────────────────────┘
```

## Code Highlights

### Box Drawing with Unicode

```rust
const BOLD: &str = "\x1b[1m";
const CYAN: &str = "\x1b[36m";
const RESET: &str = "\x1b[0m";

println!(
    "{BOLD}{CYAN}═══════════════════════════════════════{RESET}"
);
```

### Sparkline Generation

```rust
fn latency_sparkline(latencies: &[f64]) -> String {
    const BARS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    let max = latencies.iter().cloned().fold(0.0, f64::max);
    latencies.iter()
        .map(|&l| {
            let idx = ((l / max) * 7.0).round() as usize;
            BARS[idx.min(7)]
        })
        .collect()
}
```

### Real-Time Refresh

```rust
// Clear and redraw
print!("\x1b[2J\x1b[H");  // Clear screen, move to home

// Update display
render_pipeline_state(&state);
```

## Integration with Inference

The TUI wraps the standard inference loop:

```rust
let model = OwnedQuantizedModel::from_mapped(&mapped)?;
let config = QuantizedGenerateConfig::default()
    .with_max_tokens(100)
    .with_temperature(0.7);

for token in model.generate_streaming(&prompt, &config)? {
    update_display(token);
    record_latency(token.latency);
}
```

## Customization

### Adjusting Refresh Rate

```rust
const REFRESH_MS: u64 = 100;  // 10 FPS
```

### Changing Color Scheme

```rust
// Use your own ANSI colors
const MY_HEADER: &str = "\x1b[38;5;123m";  // Light cyan
const MY_SUCCESS: &str = "\x1b[38;5;46m"; // Bright green
```

## Requirements

- Terminal with Unicode support (most modern terminals)
- ANSI color support
- For best experience: true-color terminal

## Related Examples

- `observability_demo.rs` - Prometheus metrics
- `performance_parity.rs` - Benchmark suite
- `imp_700_realworld_verification.rs` - HTTP benchmarking
