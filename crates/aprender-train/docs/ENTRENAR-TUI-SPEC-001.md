# ENTRENAR-TUI-SPEC-001: Probar-Compliant TUI Monitor

**Status:** Implemented
**Version:** 2.0.0
**Date:** 2026-01-25

## Overview

The entrenar TUI monitor implements a btop/ptop-style training visualization with probar compliance for testing and
falsification. Features epoch-by-epoch history table and run configuration display.

## Probar Integration

### Features Used

| Feature | Module | Purpose |
|---------|--------|---------|
| TuiFrame | `tui` | Frame capture and assertions |
| TuiSnapshot | `tui` | Hash-based snapshot comparison |
| UxCoverageTracker | `ux_coverage` | Panel interaction coverage |
| InteractionType | `ux_coverage` | Element hover/click tracking |
| ElementId | `ux_coverage` | Panel identification |

### Falsification Protocol (F001-F025)

Per PROBAR-SPEC-015, we implement 25-point falsification for TUI robustness:

| ID | Test | Assertion |
|----|------|-----------|
| F001 | Epoch overflow | epoch clamped to total_epochs |
| F002 | Step overflow | step clamped to steps_per_epoch |
| F003 | NaN loss | Shows "???" with error color |
| F004 | Inf loss | Shows "???" with error color |
| F005 | Negative LR | Clamped to 0.0 |
| F006 | VRAM overflow | Clamped to 100% |
| F007 | Extreme temp | Shows warning indicator |
| F008 | Empty processes | Descriptive message |
| F009 | Missing GPU | Shows N/A placeholder |
| F010 | Missing sample | Graceful degradation |
| F011 | Empty model name | Shows "N/A" |
| F012 | Empty optimizer | Shows "N/A" |
| F013 | Zero batch size | Shows "N/A" |
| F014 | Empty executable path | Falls back to GPU process or "N/A" |
| F015 | Empty lr_history | Uses current learning_rate |
| F016 | NaN in loss_history | Filtered from epoch summaries |
| F017 | Negative gradient norm | Clamped to 0.0 |
| F018 | Zero steps_per_epoch | No division by zero |
| F019 | Long model name | Truncated with "..." |
| F020 | Long executable path | Truncated with "..." |
| F021 | Negative tokens_per_second | Clamped to 0.0 |
| F022 | Inf in loss_history | Filtered from epoch summaries |
| F023 | Zero total_epochs | No division by zero |
| F024 | Extreme loss value | Displays without overflow |
| F025 | Empty loss_history | Shows waiting message |

## State Schema

### TrainingSnapshot Fields

```rust
pub struct TrainingSnapshot {
    // Core training state
    pub timestamp_ms: u64,
    pub epoch: usize,
    pub total_epochs: usize,
    pub step: usize,
    pub steps_per_epoch: usize,

    // Metrics
    pub loss: f32,
    pub loss_history: Vec<f32>,
    pub learning_rate: f32,
    pub lr_history: Vec<f32>,        // NEW: Per-step LR for schedulers
    pub gradient_norm: f32,
    pub tokens_per_second: f32,

    // Timing
    pub start_timestamp_ms: u64,

    // Hardware
    pub gpu: Option<GpuTelemetry>,

    // Sample preview
    pub sample: Option<SamplePeek>,

    // Run metadata
    pub status: TrainingStatus,
    pub experiment_id: String,
    pub model_name: String,
    pub model_path: String,           // NEW: Path to model weights
    pub optimizer_name: String,       // NEW: "AdamW", "SGD", etc.
    pub batch_size: usize,           // NEW: Training batch size
    pub checkpoint_path: String,      // NEW: Checkpoint save location
    pub executable_path: String,      // NEW: Training binary path
}
```

## Quantitative Metrics

### Required Thresholds

| Metric | Minimum | Current |
|--------|---------|---------|
| Pixel coverage | 40% | 51.2% |
| Unicode richness | 10% | 17.0% |
| UX coverage | 70% | 85.0% |
| Panel score | 6/7 | 7/7 |
| Render time | <1ms | 0.046ms |
| Large history (10K) | <100ms | 1.01ms |

### Panel Coverage

All 7 panels must be present and functional:

1. **Loss Curve Panel** - Sparkline with min/max/avg stats
2. **GPU Panel** - Utilization, VRAM, temperature
3. **Process Panel** - GPU processes with memory usage
4. **Epoch History Table** - Per-epoch loss, LR, tok/s, trend
5. **Run Config Panel** - Model, optimizer, batch, paths
6. **Sample Preview** - Input/target/generated previews
7. **Training Metrics** - Epoch/step progress, ETA

## Epoch History Table

Row-based training history like standard ML frameworks:

```
📊 EPOCH HISTORY
Epoch     Loss      Min      Max         LR      Tok/s Trend
───────────────────────────────────────────────────────────────────────────
    1    9.250    8.500   10.000   0.000100      110.5
    2    7.650    6.900    8.400   0.000100      112.3 ↓
    3    6.050    5.300    6.800   0.000080      115.2 ↓
    4    4.450    3.700    5.200   0.000060      118.1 ↓
  ↑ 5 more epochs above
```

### Columns

| Column | Description | Color |
|--------|-------------|-------|
| Epoch | Epoch number | Light blue |
| Loss | Average loss for epoch | Gradient (green→red) |
| Min | Minimum loss in epoch | Light green |
| Max | Maximum loss in epoch | Light red |
| LR | Learning rate (from lr_history) | Cyan |
| Tok/s | Tokens per second | Purple |
| Trend | ↓ improving, ↑ worsening, → stable | Green/Red/Gray |

## Run Config Panel

Displays training configuration:

```
⚙️  RUN CONFIG
Model: Qwen2.5-Coder-0.5B
Optimizer: AdamW  Batch: 4
Exe: .../finetune_real
Checkpoint: ./experiments/finetune-real/
```

## Architecture

### Panel Verification (panel.rs)

```rust
pub trait Panel {
    fn name(&self) -> &'static str;
    fn can_render(&self) -> bool;  // Jidoka gate
    fn verify(&self) -> PanelVerification;
    fn budget_ms(&self) -> u32 { 16 }  // 60fps default
}
```

Implemented panels:
- `LossCurvePanel`
- `GpuPanel`
- `ProcessPanel`
- `SamplePanel`
- `MetricsPanel`
- `HistoryPanel` (NEW)
- `ConfigPanel` (NEW)

### Color System (color.rs)

btop-style gradient coloring:
- 0-25%: Cyan → Green
- 25-50%: Green → Yellow
- 50-75%: Yellow → Orange
- 75-90%: Orange → Red
- 90-100%: Critical Red

### Rendering (render.rs)

Unicode characters:
- Block bars: `█░`
- Braille sparklines: `⣿⣷⣶⣴⣤⣄⣀⡀`
- Trend arrows: `↑↓→`
- Status indicators: `●◐◔○⚡`
- Box drawing: `╭╮╰╯─│┬┴├┤┼`

## Data Sanitization

All numeric values are sanitized before display:

```rust
// Loss
if !loss.is_finite() { "???" } else { format!("{:.3}", loss) }

// Learning rate
lr.max(0.0)  // Clamp negative

// VRAM
vram_used.min(vram_total)  // Clamp overflow

// Progress
epoch.min(total_epochs)
step.min(steps_per_epoch)

// Config fields
if model_name.is_empty() { "N/A" }
if optimizer_name.is_empty() { "N/A" }
if batch_size == 0 { "N/A" }
```

## Testing

### Running Tests

```bash
# Full probar compliance test
cargo test --test probar_tui_compliance -- --nocapture

# Unit tests only
cargo test --lib -- tui::panel

# Render tests
cargo test --lib -- tui::render::tests
```

### Test Output

```
════════════════════════════════════════════════════════════
              PROBAR TUI COMPLIANCE METRICS
════════════════════════════════════════════════════════════

PIXEL COVERAGE:
  Content density: 51.2%
  Unicode richness: 17.0%
  Total characters: 3150

FRAME DIMENSIONS:
  Width: 188 chars
  Height: 17 lines
  Total area: 3196 cells

PANEL COVERAGE:
  Loss curve:      ✓
  GPU panel:       ✓
  Process panel:   ✓
  Epoch history:   ✓
  Run config:      ✓
  Sample preview:  ✓
  Training metrics: ✓

OVERALL PANEL SCORE: 7/7 (100%)
════════════════════════════════════════════════════════════
```

## Files

- `src/monitor/tui/mod.rs` - Module exports
- `src/monitor/tui/render.rs` - TUI rendering (incl. history table, config panel)
- `src/monitor/tui/panel.rs` - Panel verification
- `src/monitor/tui/color.rs` - Color system
- `src/monitor/tui/state.rs` - State types (incl. new config fields)
- `tests/probar_tui_compliance.rs` - Probar tests (F001-F015)
