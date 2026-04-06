# Real-Time Terminal Monitoring and Visualization Specification

**Version:** 1.0.0
**Status:** Draft
**Author:** PAIML Team
**Date:** 2025-11-28

## Abstract

This specification defines a real-time terminal-based monitoring and visualization system for neural network training in
Entrenar. The system leverages **trueno-viz** exclusively for hardware-accelerated rendering of training metrics, loss
curves, and diagnostic information directly to terminal output. The design prioritizes low-latency updates, minimal
memory overhead, and compatibility with headless server environments.

## 1. Introduction

### 1.1 Background

Real-time visualization of training metrics is essential for debugging, hyperparameter tuning, and understanding model
convergence behavior [1]. Traditional approaches rely on browser-based dashboards (TensorBoard, Weights & Biases) which
introduce latency, require network connectivity, and cannot operate in headless environments.

Terminal-based visualization provides immediate feedback within the training environment itself, enabling rapid
iteration cycles critical to modern ML development workflows [2].

### 1.2 Design Goals

1. **Zero external dependencies**: Use trueno-viz exclusively; no JavaScript, HTML, or browser requirements
2. **Sub-100ms update latency**: Real-time feedback for interactive training sessions
3. **Headless compatibility**: Full functionality over SSH, tmux, and screen sessions
4. **Memory efficiency**: O(1) memory for streaming metrics visualization
5. **Accessibility**: Support for ASCII-only terminals alongside Unicode/ANSI-capable terminals

### 1.3 Academic References

This specification draws upon the following peer-reviewed research:

> **[1]** Lipton, Z. C., & Steinhardt, J. (2019). "Troubling Trends in Machine Learning Scholarship." *Queue*, 17(1),
45-77. ACM. https://doi.org/10.1145/3317287.3328534
>
> *Annotation: Highlights the importance of interpretability and debugging tools in ML systems, motivating real-time
monitoring capabilities.*

> **[2]** Sculley, D., et al. (2015). "Hidden Technical Debt in Machine Learning Systems." *Advances in Neural
Information Processing Systems (NeurIPS)*, 28, 2503-2511.
>
> *Annotation: Identifies configuration and monitoring as critical infrastructure for sustainable ML systems, justifying
investment in training observability.*

## 2. Architecture

### 2.1 System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Entrenar Training Loop                       │
├─────────────────────────────────────────────────────────────────┤
│  Trainer → CallbackManager → TerminalMonitorCallback            │
│                                    │                            │
│                                    ▼                            │
│                          ┌─────────────────┐                    │
│                          │  MetricsBuffer  │  (Ring buffer)     │
│                          └────────┬────────┘                    │
│                                   │                             │
│                                   ▼                             │
│                    ┌──────────────────────────┐                 │
│                    │     trueno-viz           │                 │
│                    │  ┌─────────────────────┐ │                 │
│                    │  │ LossCurve           │ │                 │
│                    │  │ LineChart           │ │                 │
│                    │  │ Framebuffer         │ │                 │
│                    │  └─────────────────────┘ │                 │
│                    │  ┌─────────────────────┐ │                 │
│                    │  │ TerminalEncoder     │ │                 │
│                    │  │ - ASCII mode        │ │                 │
│                    │  │ - Unicode halfblock │ │                 │
│                    │  │ - ANSI true color   │ │                 │
│                    │  └─────────────────────┘ │                 │
│                    └───────────┬──────────────┘                 │
│                                │                                │
│                                ▼                                │
│                           stdout/stderr                         │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Responsibilities

| Component | Responsibility |
|-----------|----------------|
| `TerminalMonitorCallback` | Training loop integration, event handling |
| `MetricsBuffer` | Fixed-size ring buffer for streaming data |
| `trueno_viz::LossCurve` | Loss curve rendering with EMA smoothing |
| `trueno_viz::TerminalEncoder` | Framebuffer to terminal string conversion |
| `LayoutEngine` | Multi-panel dashboard composition |

### 2.3 Data Flow

The visualization pipeline implements a **producer-consumer pattern** with backpressure handling [3]:

> **[3]** Welsh, M., Culler, D., & Brewer, E. (2001). "SEDA: An Architecture for Well-Conditioned, Scalable Internet
Services." *ACM SIGOPS Operating Systems Review*, 35(5), 230-243.
>
> *Annotation: SEDA's staged event-driven architecture informs our callback-based design with bounded buffers to prevent
memory exhaustion under high update rates.*

```rust
// Simplified data flow
on_step_end(ctx) {
    metrics_buffer.push(ctx.loss);  // O(1) ring buffer insert

    if should_refresh() {
        let fb = loss_curve.to_framebuffer();  // SIMD-accelerated
        let output = encoder.render(&fb);      // Terminal encoding
        clear_and_print(output);               // ANSI cursor control
    }
}
```

## 3. Terminal Rendering Modes

### 3.1 Mode Selection

trueno-viz provides three terminal rendering modes with automatic capability detection:

| Mode | Characters | Resolution | Color | Compatibility |
|------|------------|------------|-------|---------------|
| ASCII | ` .:-=+*#%@` | 1x1 | Grayscale | Universal |
| UnicodeHalfBlock | `▀ ▄ █` | 1x2 | 24-bit ANSI | Modern terminals |
| AnsiTrueColor | `█` (full block) | 1x1 | 24-bit ANSI | Modern terminals |

### 3.2 Grayscale Perception

The ASCII ramp implements **perceptual uniformity** based on Weber-Fechner law [4]:

> **[4]** Stevens, S. S. (1957). "On the psychophysical law." *Psychological Review*, 64(3), 153-181. American
Psychological Association.
>
> *Annotation: Human brightness perception is logarithmic. Our grayscale ramp is designed for perceptually uniform
steps, ensuring training curves are visually accurate.*

The grayscale conversion uses **Rec. 709 luminance coefficients**:

```rust
let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
```

### 3.3 Unicode Half-Block Rendering

Unicode half-block characters (`▀` U+2580, `▄` U+2584) achieve **2x vertical resolution** by encoding two pixels per
character cell:

```
┌───┐
│▀▀▀│  ← Upper pixel (foreground color)
│▄▄▄│  ← Lower pixel (background color)
└───┘
```

This technique doubles effective vertical resolution without increasing character count, critical for displaying
detailed loss curves in constrained terminal heights [5].

> **[5]** Baudisch, P., & Gutwin, C. (2004). "Multiblending: Displaying overlapping windows simultaneously without the
drawbacks of alpha blending." *Proceedings of CHI 2004*, 367-374. ACM.
>
> *Annotation: While focused on windowing systems, this work establishes principles for maximizing information density
in constrained display areas, applicable to our terminal constraints.*

## 4. Visualization Components

### 4.1 Loss Curve (`LossCurve`)

The loss curve component provides streaming visualization of training metrics with exponential moving average (EMA)
smoothing.

#### 4.1.1 EMA Smoothing

Smoothing reduces visual noise while preserving trend information [6]:

```rust
smoothed[t] = α * smoothed[t-1] + (1 - α) * raw[t]
```

Where `α ∈ [0.0, 0.99]` controls smoothing strength.

> **[6]** Hunter, J. S. (1986). "The Exponentially Weighted Moving Average." *Journal of Quality Technology*, 18(4),
203-210.
>
> *Annotation: EMA is optimal for online trend estimation with bounded memory, matching our O(1) memory requirement for
streaming visualization.*

#### 4.1.2 Best Value Markers

Visual markers indicate optimal values (minimum loss or maximum accuracy):

```rust
pub struct LossCurve {
    lower_is_better: bool,  // true for loss, false for accuracy
    show_best_markers: bool,
    marker_size: f32,
}
```

#### 4.1.3 Reference Curves (Standardized Work)

To enable **Standardized Work** comparisons, the loss curve supports overlaying a "Golden Trace" from a known good run:

```rust
pub struct LossCurve {
    reference_curve: Option<Vec<Point>>, // "Shadow" curve drawn in faint gray
    // ...
}
```

This allows immediate visual detection of deviation from expected convergence behavior (anomaly detection).

### 4.2 Sparklines

Compact inline sparklines for status bars using **miniature bar charts** [7]:

> **[7]** Tufte, E. R. (2006). *Beautiful Evidence*. Graphics Press. pp. 46-63.
>
> *Annotation: Tufte's sparkline concept—"intense, simple, word-sized graphics"—directly informs our compact inline
metric display design.*

```
Loss: 0.0234 ▁▂▃▄▅▆▇█▇▆▅▄▃  Best: 0.0189 @ epoch 47
```

Implementation using Unicode block elements:

```rust
const SPARK_CHARS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

fn sparkline(values: &[f32], width: usize) -> String {
    let (min, max) = extent(values);
    let range = max - min;

    values.iter()
        .map(|v| {
            let idx = (((v - min) / range) * 7.0).round() as usize;
            SPARK_CHARS[idx.min(7)]
        })
        .collect()
}
```

### 4.3 Progress Bar

Training progress with ETA estimation using **Kalman filtering** for stable predictions [8]:

> **[8]** Welch, G., & Bishop, G. (1995). "An Introduction to the Kalman Filter." *Technical Report TR 95-041*,
University of North Carolina at Chapel Hill.
>
> *Annotation: Kalman filtering provides optimal ETA estimates by modeling step duration as a noisy linear process,
avoiding erratic ETA jumps common in naive implementations.*

```
Epoch 15/100 [████████████░░░░░░░░░░░░░░░░░░] 40% │ 12.3 steps/s │ ETA: 2h 15m
```

### 4.4 Multi-Panel Dashboard

Composable dashboard layout with ANSI cursor control for flicker-free updates:

```
╔═══════════════════════════════════════════════════════════════════╗
║  ENTRENAR TRAINING MONITOR                           [RUNNING]    ║
╠═══════════════════════════════════════════════════════════════════╣
║  Model: llama-7b-lora    │ Dataset: alpaca-52k │ GPU: RTX 4090    ║
╠═══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  Loss Curve (last 100 epochs)                                     ║
║  ┌─────────────────────────────────────────────────────────────┐  ║
║  │                                                             │  ║
║  │  ▓▓▓▓▓▓▓▓▓▓▓▓▓                                              │  ║
║  │              ▓▓▓▓▓▓                                         │  ║
║  │                    ▓▓▓▓▓▓▓                                  │  ║
║  │                           ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓                │  ║
║  │                                              ▓▓▓▓▓▓▓▓▓▓▓▓▓  │  ║
║  └─────────────────────────────────────────────────────────────┘  ║
║                                                                   ║
║  Metrics:                                                         ║
║    Train Loss: 0.0234 ▁▂▃▄▅▆▇█▇▆▅▄▃   Val Loss: 0.0456 ▂▃▄▅▄▃▂  ║
║    LR: 1.2e-4 (cosine)                 Grad Norm: 0.89            ║
║                                                                   ║
║  Progress: [████████████████░░░░░░░░░░░░░░] 53% │ ETA: 1h 23m     ║
╚═══════════════════════════════════════════════════════════════════╝
```

### 4.5 Health Monitoring (Andon Cord)

In the spirit of **Jidoka** (automation with a human touch), the monitor actively detects training abnormalities and
"pulls the cord" (flashing alert):

- **NaN/Inf Detection**: Immediate red flashing alert if loss becomes invalid.
- **Divergence Check**: Alert if loss exceeds 3σ of EMA.
- **Stall Detection**: Alert if loss doesn't improve for N steps.

```rust
if ctx.loss.is_nan() || ctx.loss.is_infinite() {
    monitor.trigger_andon("Training Diverged: NaN Loss Detected");
    return CallbackAction::Stop;
}
```

## 5. Update Strategies

### 5.1 Refresh Rate Control

To prevent terminal flooding while maintaining responsiveness, we implement **adaptive refresh rate** [9]:

> **[9]** Card, S. K., Robertson, G. G., & Mackinlay, J. D. (1991). "The Information Visualizer: An Information
Workspace." *Proceedings of CHI 1991*, 181-186. ACM.
>
> *Annotation: Establishes 100ms as the threshold for perceived instantaneous response, guiding our minimum refresh
interval.*

```rust
pub struct RefreshPolicy {
    min_interval_ms: u64,   // Minimum 50ms between refreshes
    max_interval_ms: u64,   // Maximum 1000ms (force refresh)
    step_threshold: usize,  // Refresh every N steps
}

impl RefreshPolicy {
    pub fn should_refresh(&self, ctx: &CallbackContext) -> bool {
        let elapsed = self.last_refresh.elapsed();

        // Always refresh after max interval
        if elapsed >= Duration::from_millis(self.max_interval_ms) {
            return true;
        }

        // Rate-limit to min interval
        if elapsed < Duration::from_millis(self.min_interval_ms) {
            return false;
        }

        // Step-based refresh
        ctx.global_step % self.step_threshold == 0
    }
}
```

### 5.2 Flicker-Free Updates

ANSI escape sequences enable **in-place updates** without screen clearing:

```rust
const CURSOR_HOME: &str = "\x1b[H";        // Move cursor to (0,0)
const CLEAR_SCREEN: &str = "\x1b[2J";      // Clear entire screen
const CURSOR_SAVE: &str = "\x1b[s";        // Save cursor position
const CURSOR_RESTORE: &str = "\x1b[u";     // Restore cursor position
const HIDE_CURSOR: &str = "\x1b[?25l";     // Hide cursor
const SHOW_CURSOR: &str = "\x1b[?25h";     // Show cursor

fn update_display(content: &str) {
    print!("{HIDE_CURSOR}{CURSOR_HOME}{content}{SHOW_CURSOR}");
    std::io::stdout().flush().unwrap();
}
```

### 5.3 Terminal Size Detection

Dynamic layout adaptation using environment variables (CI/Headless) or TIOCGWINSZ ioctl:

```rust
fn get_terminal_size() -> (u16, u16) {
    use std::env;
    use std::io::{stdout, IsTerminal};

    // 1. Check Environment Variables (CI/Headless)
    if let (Ok(cols), Ok(rows)) = (env::var("COLUMNS"), env::var("LINES")) {
        if let (Ok(c), Ok(r)) = (cols.parse(), rows.parse()) {
            return (c, r);
        }
    }

    // 2. Check TTY (Interactive)
    if stdout().is_terminal() {
        #[cfg(unix)]
        {
            use libc::{ioctl, winsize, TIOCGWINSZ};
            let mut ws = winsize { ws_row: 0, ws_col: 0, ws_xpixel: 0, ws_ypixel: 0 };
            if unsafe { ioctl(1, TIOCGWINSZ, &mut ws) } == 0 {
                return (ws.ws_col, ws.ws_row);
            }
        }
    }

    // 3. Fallback
    (80, 24)
}
```

## 6. API Design

### 6.1 Callback Interface

```rust
use entrenar::train::callback::{TrainerCallback, CallbackContext, CallbackAction};
use trueno_viz::prelude::*;
use trueno_viz::output::terminal::{TerminalEncoder, TerminalMode};

/// Real-time terminal monitoring callback
pub struct TerminalMonitorCallback {
    loss_curve: LossCurve,
    encoder: TerminalEncoder,
    refresh_policy: RefreshPolicy,
    layout: DashboardLayout,
    last_refresh: Instant,
}

impl TerminalMonitorCallback {
    pub fn new() -> Self {
        Self {
            loss_curve: LossCurve::new()
                .train_loss()
                .val_loss()
                .dimensions(80, 20),
            encoder: TerminalEncoder::new()
                .mode(TerminalMode::UnicodeHalfBlock)
                .width(80),
            refresh_policy: RefreshPolicy::default(),
            layout: DashboardLayout::Compact,
            last_refresh: Instant::now(),
        }
    }

    pub fn mode(mut self, mode: TerminalMode) -> Self {
        self.encoder = self.encoder.mode(mode);
        self
    }

    pub fn layout(mut self, layout: DashboardLayout) -> Self {
        self.layout = layout;
        self
    }

    pub fn refresh_interval(mut self, ms: u64) -> Self {
        self.refresh_policy.min_interval_ms = ms;
        self
    }
}

impl TrainerCallback for TerminalMonitorCallback {
    fn on_step_end(&mut self, ctx: &CallbackContext) -> CallbackAction {
        // Record metrics
        self.loss_curve.push(0, ctx.loss);
        if let Some(val) = ctx.val_loss {
            self.loss_curve.push(1, val);
        }

        // Rate-limited refresh
        if self.refresh_policy.should_refresh(ctx) {
            self.render_dashboard(ctx);
            self.last_refresh = Instant::now();
        }

        CallbackAction::Continue
    }

    fn on_train_end(&mut self, _ctx: &CallbackContext) {
        // Final render and cleanup
        print!("\x1b[?25h");  // Show cursor
    }

    fn name(&self) -> &str {
        "TerminalMonitorCallback"
    }
}
```

### 6.2 Layout Options

```rust
pub enum DashboardLayout {
    /// Minimal single-line progress bar
    Minimal,
    /// Compact 5-line summary with sparklines
    Compact,
    /// Full dashboard with loss curve plot
    Full,
    /// Custom dimensions
    Custom { width: u32, height: u32 },
}
```

### 6.3 Configuration

```yaml
# entrenar.yaml
monitor:
  enabled: true
  layout: full
  terminal_mode: unicode_halfblock  # ascii | unicode_halfblock | ansi_true_color
  refresh_ms: 100
  metrics:
    - train_loss
    - val_loss
    - learning_rate
    - gradient_norm
  sparkline_width: 20
  show_eta: true
  show_gpu_memory: true
```

## 7. Performance Requirements

### 7.1 Latency Bounds

| Operation | Target | Maximum |
|-----------|--------|---------|
| Metrics buffer insert | 100ns | 1μs |
| Framebuffer render (80x24) | 500μs | 2ms |
| Terminal encode | 200μs | 1ms |
| Full refresh cycle | 1ms | 5ms |

### 7.2 Memory Bounds

| Component | Memory |
|-----------|--------|
| MetricsBuffer (1000 points) | 4KB |
| Framebuffer (80x24 RGBA) | 7.5KB |
| LossCurve state | 8KB |
| Total overhead | <32KB |

### 7.3 Line Simplification

For large epoch counts (>1000), Douglas-Peucker simplification reduces rendering complexity [10]:

> **[10]** Douglas, D. H., & Peucker, T. K. (1973). "Algorithms for the reduction of the number of points required to
represent a digitized line or its caricature." *Cartographica: The International Journal for Geographic Information and
Geovisualization*, 10(2), 112-122.
>
> *Annotation: The Douglas-Peucker algorithm achieves O(n log n) complexity for polyline simplification, essential for
rendering 10,000+ epoch curves in real-time.*

```rust
// Automatic simplification for large datasets
let points = if epochs > 1000 {
    douglas_peucker(&raw_points, epsilon: 1.0)
} else {
    raw_points
};
```

## 8. Error Handling

### 8.1 Graceful Degradation

| Condition | Behavior |
|-----------|----------|
| Non-TTY output | Disable visualization, log metrics to file |
| Terminal too small | Switch to Minimal layout |
| ANSI not supported | Fall back to ASCII mode |
| Rendering error | Skip frame, continue training |

### 8.2 Signal Handling

```rust
// Handle SIGWINCH for terminal resize
signal::signal(Signal::SIGWINCH, || {
    let (cols, rows) = get_terminal_size();
    monitor.resize(cols, rows);
});

// Handle SIGINT gracefully
signal::signal(Signal::SIGINT, || {
    print!("\x1b[?25h\x1b[0m");  // Restore cursor and colors
    std::process::exit(0);
});
```

## 9. Testing Strategy

### 9.1 Unit Tests

- Metrics buffer ring behavior (wrap-around, capacity)
- Sparkline generation for edge cases (empty, constant, NaN)
- EMA smoothing numerical accuracy
- Terminal encoding correctness

### 9.2 Property Tests

```rust
proptest! {
    #[test]
    fn sparkline_length_matches_input(values in prop::collection::vec(0.0f32..1.0, 1..100)) {
        let spark = sparkline(&values, values.len());
        prop_assert_eq!(spark.chars().count(), values.len());
    }

    #[test]
    fn ema_bounded(values in prop::collection::vec(-100.0f32..100.0, 1..1000)) {
        let smoothed = ema(&values, 0.9);
        let (min, max) = extent(&values);
        for v in smoothed {
            prop_assert!(v >= min && v <= max);
        }
    }
}
```

### 9.3 Visual Regression Tests

Snapshot testing for terminal output consistency across modes.

## 10. Real-Time Explainability

### 10.1 Feature Attribution Visualization

Integrate `ExplainabilityCallback` with terminal display for live feature importance:

```
┌─ Feature Importance (Permutation) ──────────────────────────────┐
│  token_embedding  ████████████████████████████████  0.847       │
│  position_bias    ████████████████████             0.523       │
│  attention_mask   ███████████████                  0.401       │
│  layer_norm_gain  ██████████                       0.267       │
│  dropout_prob     ████                             0.112       │
└─────────────────────────────────────────────────────────────────┘
```

### 10.2 Gradient Flow Heatmap

Per-layer gradient magnitude using trueno-viz `Heatmap`:

```
Layer Gradients (log scale):
         Q    K    V    O   FFN
Layer 0  ▓▓   ▓▓   ▓▓   ▓    ▓▓
Layer 1  ▓▓▓  ▓▓▓  ▓▓   ▓▓   ▓▓▓
Layer 2  ▓▓▓▓ ▓▓▓▓ ▓▓▓  ▓▓▓  ▓▓▓▓
Layer 3  ▓▓▓  ▓▓▓  ▓▓▓▓ ▓▓▓▓ ▓▓▓
```

### 10.3 Academic Reference

> **[11]** Sundararajan, M., Taly, A., & Yan, Q. (2017). "Axiomatic Attribution for Deep Networks." *ICML 2017*,
3319-3328.
>
> *Annotation: Integrated Gradients provides theoretically grounded feature attribution, already implemented in
aprender::interpret.*

### 10.4 API Extension

```rust
impl TerminalMonitorCallback {
    pub fn with_explainability(mut self, method: ExplainMethod) -> Self {
        self.explain = Some(ExplainabilityCallback::new(method).with_top_k(5));
        self
    }
}
```

| Ticket | Description | Estimate |
|--------|-------------|----------|
| ENT-064 | Real-time feature importance display | 3h |
| ENT-065 | Gradient flow heatmap | 2h |

## 11. Implementation Roadmap

| Ticket | Description | Estimate |
|--------|-------------|----------|
| ENT-054 | TerminalMonitorCallback skeleton | 2h |
| ENT-055 | MetricsBuffer ring buffer | 2h |
| ENT-056 | trueno-viz LossCurve integration | 4h |
| ENT-057 | Sparkline generation | 2h |
| ENT-058 | Progress bar with ETA | 3h |
| ENT-059 | Multi-panel dashboard layout | 4h |
| ENT-060 | Adaptive refresh policy | 2h |
| ENT-061 | Terminal capability detection | 2h |
| ENT-062 | YAML configuration support | 2h |
| ENT-063 | Property tests and documentation | 3h |
| ENT-066 | Health monitoring (Andon) | 3h |
| ENT-067 | Reference curve overlay | 3h |

**Total: 32 hours**

## 12. References

1. Lipton, Z. C., & Steinhardt, J. (2019). "Troubling Trends in Machine Learning Scholarship." *Queue*, 17(1), 45-77.

2. Sculley, D., et al. (2015). "Hidden Technical Debt in Machine Learning Systems." *NeurIPS*, 28, 2503-2511.

3. Welsh, M., Culler, D., & Brewer, E. (2001). "SEDA: An Architecture for Well-Conditioned, Scalable Internet Services."
   *ACM SIGOPS Operating Systems Review*, 35(5), 230-243.

4. Stevens, S. S. (1957). "On the psychophysical law." *Psychological Review*, 64(3), 153-181.

5. Baudisch, P., & Gutwin, C. (2004). "Multiblending: Displaying overlapping windows simultaneously without the
   drawbacks of alpha blending." *CHI 2004*, 367-374.

6. Hunter, J. S. (1986). "The Exponentially Weighted Moving Average." *Journal of Quality Technology*, 18(4), 203-210.

7. Tufte, E. R. (2006). *Beautiful Evidence*. Graphics Press.

8. Welch, G., & Bishop, G. (1995). "An Introduction to the Kalman Filter." *TR 95-041*, UNC Chapel Hill.

9. Card, S. K., Robertson, G. G., & Mackinlay, J. D. (1991). "The Information Visualizer: An Information Workspace."
   *CHI 1991*, 181-186.

10. Douglas, D. H., & Peucker, T. K. (1973). "Algorithms for the reduction of the number of points required to represent
    a digitized line or its caricature." *Cartographica*, 10(2), 112-122.

11. Sundararajan, M., Taly, A., & Yan, Q. (2017). "Axiomatic Attribution for Deep Networks." *ICML 2017*, 3319-3328.

## Appendix A: trueno-viz Feature Summary

| Feature | Module | Description |
|---------|--------|-------------|
| `LossCurve` | `plots::loss_curve` | Streaming loss visualization with EMA |
| `LineChart` | `plots::line` | Multi-series line charts |
| `TerminalEncoder` | `output::terminal` | ASCII/Unicode/ANSI encoding |
| `Framebuffer` | `framebuffer` | RGBA pixel buffer |
| `LinearScale` | `scale` | Data-to-visual coordinate mapping |
| `douglas_peucker` | `plots::line` | Line simplification algorithm |

## Appendix B: ANSI Escape Code Reference

| Code | Purpose |
|------|---------|
| `\x1b[H` | Cursor to home (0,0) |
| `\x1b[2J` | Clear screen |
| `\x1b[?25l` | Hide cursor |
| `\x1b[?25h` | Show cursor |
| `\x1b[38;2;R;G;Bm` | Set foreground color (24-bit) |
| `\x1b[48;2;R;G;Bm` | Set background color (24-bit) |
| `\x1b[0m` | Reset all attributes |
