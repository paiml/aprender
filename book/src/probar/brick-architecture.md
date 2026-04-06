# Brick Architecture

Brick Architecture enables widget-level testing with built-in assertions, performance budgets, and verification. Each "brick" is a testable widget component that declares its testing requirements upfront.

## Overview

The Brick trait allows widgets to:
- Declare **assertions** they must satisfy (e.g., text visible, contrast ratio)
- Set **performance budgets** (e.g., 16ms for 60fps rendering)
- Define **verification** logic that runs before rendering

## Quick Start

```rust
use jugar_probar::brick::{Brick, BrickAssertion, BrickBudget, BrickVerification};

struct MyButton {
    label: String,
}

impl Brick for MyButton {
    fn brick_name(&self) -> &'static str {
        "MyButton"
    }

    fn assertions(&self) -> &[BrickAssertion] {
        &[
            BrickAssertion::TextVisible,
            BrickAssertion::ContrastRatio(4.5),
        ]
    }

    fn budget(&self) -> BrickBudget {
        BrickBudget::uniform(16) // 60fps target
    }

    fn verify(&self) -> BrickVerification {
        BrickVerification::pass()
    }
}
```

## Assertions

Built-in assertions include:

| Assertion | Description |
|-----------|-------------|
| `TextVisible` | Text content is visible |
| `ContrastRatio(f64)` | WCAG contrast ratio minimum |
| `MinSize { w, h }` | Minimum dimensions |
| `Accessible` | Meets accessibility requirements |
| `Custom { name, validator_id }` | Custom validation logic |

## Performance Budgets

```rust
// Uniform budget: all operations share 16ms
BrickBudget::uniform(16)

// Tiered budget: layout gets more time than paint
BrickBudget {
    layout_ms: 8,
    paint_ms: 4,
    total_ms: 16,
}
```

## Verification

```rust
fn verify(&self) -> BrickVerification {
    let mut passed = Vec::new();
    let mut failed = Vec::new();

    // Check contrast ratio
    if self.contrast_ratio() >= 4.5 {
        passed.push(BrickAssertion::ContrastRatio(4.5));
    } else {
        failed.push((
            BrickAssertion::ContrastRatio(4.5),
            format!("Contrast {} < 4.5", self.contrast_ratio()),
        ));
    }

    BrickVerification {
        passed,
        failed,
        verification_time: Duration::from_micros(100),
    }
}
```

## TUI Integration

Brick Architecture integrates seamlessly with ratatui widgets:

```rust
use jugar_probar::brick::tui::BrickWidget;

let button = BrickWidget::new("Submit")
    .with_assertion(BrickAssertion::TextVisible)
    .with_budget(BrickBudget::uniform(8));

// Verify before render
assert!(button.can_render());

// Render to terminal
frame.render_widget(button, area);
```

## Best Practices

1. **Declare assertions upfront** - Define what each widget must satisfy
2. **Set realistic budgets** - 16ms for 60fps, 8ms for 120fps
3. **Verify before render** - Call `can_render()` to ensure constraints are met
4. **Use presets** - Standard button, input, and container presets available

## BrickHouse: Budgeted Composition

Compose multiple bricks with a total performance budget. The BrickHouse enforces Jidoka (stop-the-line) principles - if any brick exceeds its budget, rendering halts.

```rust
use jugar_probar::brick_house::{BrickHouse, BrickHouseBuilder};
use std::sync::Arc;

// Build a house with bricks and budgets
let house = BrickHouseBuilder::new("whisper-app")
    .budget_ms(1000)  // 1 second total
    .brick(Arc::new(status_brick), 50)      // 50ms for status
    .brick(Arc::new(waveform_brick), 100)   // 100ms for waveform
    .brick(Arc::new(transcription_brick), 600) // 600ms for transcription
    .build()?;

// Verify all bricks can render
assert!(house.can_render());

// Render with budget tracking
let html = house.render()?;

// Check budget report
if let Some(report) = house.last_report() {
    println!("Utilization: {:.1}%", report.utilization());
    assert!(report.within_budget());
}
```

## web_sys_gen: Zero Hand-Written web_sys

The `web_sys_gen` module provides generated abstractions that replace hand-written `web_sys` calls. This ensures:

1. **Traceability** - All code is derived from brick specifications
2. **Consistency** - Error handling is uniform
3. **No hand-written web_sys** - Application code stays clean

### Performance Timing

```rust
use jugar_probar::brick::web_sys_gen::PerformanceTiming;

// Get high-resolution timestamp
let start = PerformanceTiming::now();

// Measure an operation with automatic timing
let (result, duration_ms) = PerformanceTiming::measure(|| {
    expensive_computation()
});

println!("Operation took {:.2}ms", duration_ms);
```

### Custom Events

```rust
use jugar_probar::brick::web_sys_gen::{CustomEventDispatcher, EventDetail};

// Create a dispatcher for your event type
let dispatcher = CustomEventDispatcher::new("transcription-complete");

// Dispatch with various detail types
dispatcher.dispatch()?;  // No detail
dispatcher.dispatch_with_detail(EventDetail::string("Done"))?;
dispatcher.dispatch_with_detail(EventDetail::number(42.0))?;
dispatcher.dispatch_with_detail(EventDetail::json(&my_data))?;
```

### Blob URLs

```rust
use jugar_probar::brick::web_sys_gen::BlobUrl;

// Create a blob URL from JavaScript code
let worker_code = "self.onmessage = (e) => self.postMessage(e.data * 2);";
let url = BlobUrl::from_js_code(worker_code)?;

// Use the URL to create a Worker
// ...

// Clean up when done
BlobUrl::revoke(&url)?;
```

### Fetch Client

```rust
use jugar_probar::brick::web_sys_gen::FetchClient;

let client = FetchClient::new();

// Fetch bytes from a URL (async)
let bytes = client.fetch_bytes("https://example.com/data.bin").await?;
```

## Examples

Run the brick examples:

```bash
# Basic brick architecture demo
cargo run --example brick_demo -p jugar-probar

# Visual TUI demo - shows bricks "lighting up" as tests pass
cargo run --example brick_tui_demo -p jugar-probar

# web_sys_gen utilities demo
cargo run --example web_sys_gen_demo -p jugar-probar
```

### Visual Brick Demo Output

The `brick_tui_demo` shows a live visualization of bricks being verified:

```
  ========================================
    BRICK ARCHITECTURE - VERIFICATION COMPLETE
  ========================================

    +-------+-------+-------+-------+-------+-------+
    |  S S  |  W W  |  A A  |  T T  |  E E  |  M M  |
    | Statu | Wave  | Audio | Trans | Error | Model |
    +-------+-------+-------+-------+-------+-------+
         [GREEN]  [GREEN]  [GREEN]  [GREEN]  [RED]   [GREEN]

  Final Verification Results
  -------------------------
    Status       [PASS] budget:  50ms actual:  12ms assertions: 3/3
    Wave         [PASS] budget: 100ms actual:  45ms assertions: 3/3
    Audio        [PASS] budget: 150ms actual:  67ms assertions: 3/3
    Trans        [PASS] budget: 600ms actual: 234ms assertions: 3/3
    Error        [FAIL] budget:  50ms actual:  15ms assertions: 1/2
    Model        [PASS] budget: 200ms actual:  89ms assertions: 3/3

  Summary
  -------
    Bricks: 5 passed, 1 failed
    Budget Utilization: 40.2% (healthy)
```

## ComputeBlock Testing (presentar-terminal)

The `compute-blocks` feature enables testing of SIMD-optimized panel elements from presentar-terminal. ComputeBlocks are high-performance widgets like sparklines, gauges, and thermal displays that use SIMD instructions for efficient computation.

### Enabling ComputeBlock Support

```toml
[dev-dependencies]
jugar-probar = { version = "1.0", features = ["compute-blocks"] }
```

### ComputeBlockAssertion (Playwright-style API)

```rust
use jugar_probar::tui::{ComputeBlockAssertion, SimdInstructionSet};
use presentar_terminal::SparklineBlock;

let mut block = SparklineBlock::new(60);
block.push(50.0);

// Fluent assertions
ComputeBlockAssertion::new(&block)
    .to_have_simd_support()
    .to_have_latency_under(100)  // microseconds
    .to_use_simd(SimdInstructionSet::Sse4);
```

### Soft Assertions (Collect Errors)

```rust
let mut assertion = ComputeBlockAssertion::new(&block).soft();
assertion.to_have_simd_support();
assertion.to_have_latency_under(50);

// Check all errors at once
if !assertion.errors().is_empty() {
    println!("Warnings: {:?}", assertion.errors());
}
```

### Latency Budget Validation

```rust
use jugar_probar::tui::assert_compute_latency;

// Assert actual computation time is within the block's budget
let duration = assert_compute_latency(&mut sparkline, &75.0)?;
println!("Computed in {:?}", duration);
```

### SIMD Detection

```rust
use jugar_probar::tui::{detect_simd, simd_available, assert_simd_available};
use presentar_terminal::SimdInstructionSet;

// Check what SIMD is available
let simd = detect_simd();
println!("Detected: {} ({}-bit vectors)", simd.name(), simd.vector_width() * 8);

// Quick check
if simd_available() {
    println!("SIMD acceleration available");
}

// Assert minimum SIMD level
assert_simd_available(SimdInstructionSet::Avx2)?;
```

### BrickTestAssertion for Verification Gates

```rust
use jugar_probar::tui::{BrickTestAssertion, assert_brick_valid, assert_brick_budget};

// Playwright-style fluent API
BrickTestAssertion::new(&my_widget)
    .to_be_valid()
    .to_have_budget_under(16)  // milliseconds
    .to_be_renderable();

// Standalone assertions
assert_brick_valid(&my_widget)?;

// Measure phase timing
let duration = assert_brick_budget(&my_widget, || {
    my_widget.paint(&mut buffer);
}, "paint")?;
```

### Available ComputeBlock Types

The following presentar-terminal types are re-exported:

| Block Type | Description | SIMD Optimized |
|------------|-------------|----------------|
| `SparklineBlock` | Sparkline graph | Yes |
| `CpuFrequencyBlock` | CPU frequency display | Yes |
| `CpuGovernorBlock` | CPU governor status | No |
| `GpuThermalBlock` | GPU temperature | Yes |
| `GpuVramBlock` | GPU VRAM usage | Yes |
| `LoadTrendBlock` | System load trend | Yes |
| `MemPressureBlock` | Memory pressure indicator | Yes |
| `HugePagesBlock` | HugePages status | No |

### SIMD Instruction Sets

Probar detects and validates these instruction sets:

| Set | Vector Width | Platforms |
|-----|--------------|-----------|
| `Scalar` | 1 (no SIMD) | All |
| `Sse4` | 128-bit | x86_64 |
| `Avx2` | 256-bit | x86_64 |
| `Avx512` | 512-bit | x86_64 (server) |
| `Neon` | 128-bit | ARM64 |
| `WasmSimd128` | 128-bit | WASM |

### Example: Testing a Dashboard

```rust
use jugar_probar::tui::{
    BrickTestAssertion, ComputeBlockAssertion,
    assert_brick_valid, assert_compute_latency,
};
use presentar_terminal::{SparklineBlock, GpuThermalBlock};

#[test]
fn test_dashboard_widgets() {
    // Test sparkline
    let mut sparkline = SparklineBlock::new(60);
    for i in 0..60 {
        sparkline.push(50.0 + (i as f64).sin() * 20.0);
    }

    ComputeBlockAssertion::new(&sparkline)
        .to_have_simd_support()
        .to_have_latency_under(100);

    // Test thermal display
    let thermal = GpuThermalBlock::new(75.0, 90.0);
    assert_brick_valid(&thermal).expect("Thermal widget must be valid");
}
```

## See Also

- [PROBAR-SPEC-009: Brick Architecture](../specs/brick-architecture.md)
- [TUI Testing](./tui-testing.md)
- [Accessibility Testing](./accessibility.md)
