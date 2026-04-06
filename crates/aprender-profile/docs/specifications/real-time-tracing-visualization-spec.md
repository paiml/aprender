# Real-Time Tracing Visualization Specification

**Version:** 1.0
**Date:** 2026-01-07
**Status:** Specification - Phase 1
**Sprint Target:** 52-55 (Unified Visualization System)
**Quality Framework:** Iron Lotus + Toyota Way + Popperian Falsification
**Depends On:** trueno-viz v0.1.14, trueno v0.3.6, probador v0.2, renacer::isolation_forest

## Executive Summary

This specification defines a **world-class real-time tracing visualization system** for Renacer, providing terminal-based visualization of syscall traces, anomalies, ML clustering, and OpenTelemetry spans using the exact architecture, primitives, and performance characteristics of **ttop** (trueno-viz).

**Business Value:**
- **Observability**: Real-time visibility into traced program behavior without external tools
- **Performance**: 8ms frame target with 50ms event tick (identical to ttop)
- **Ergonomics**: Single command invocation (`renacer visualize`) for immediate value
- **Integration**: Native OTLP span visualization, ML anomaly correlation, DWARF source mapping
- **Sister Project Synergy**: Dogfooding trueno-viz primitives within PAIML ecosystem

**Key Principle:**
> *"Genchi Genbutsu (現地現物) - Go and see for yourself. Real-time visualization eliminates the gap between execution and understanding."* — Toyota Way [5]

---

## Table of Contents

1. [Goals and Requirements](#1-goals-and-requirements)
2. [Architecture Overview](#2-architecture-overview)
3. [CLI Integration](#3-cli-integration)
4. [Visualization Primitives](#4-visualization-primitives)
5. [Panel System](#5-panel-system)
6. [Data Collection Architecture](#6-data-collection-architecture)
7. [Performance Specifications](#7-performance-specifications)
8. [Testing Strategy (Probador)](#8-testing-strategy-probador)
9. [Implementation Phases](#9-implementation-phases)
10. [References](#10-references)
11. [Popperian Falsification Checklist (100 Points)](#11-popperian-falsification-checklist-100-points)
12. [Approval](#12-approval)

---

## 1. Goals and Requirements

### 1.1 Primary Goals

| ID | Goal | Success Metric |
|----|------|----------------|
| G1 | Trivial invocation | Single command `renacer visualize` starts TUI |
| G2 | ttop-identical performance | 8ms avg frame time, <16ms P99 |
| G3 | Real-time syscall visualization | <100ms latency from syscall to display |
| G4 | Anomaly highlighting | Z-score outliers visible within 1 frame |
| G5 | ML cluster visualization | PCA/t-SNE scatter in braille mode |
| G6 | OTLP span waterfall | Gantt-style trace timeline |
| G7 | Source correlation | DWARF file:line displayed for anomalies |
| G8 | 100% probador coverage | All TUI components tested deterministically |

### 1.2 Non-Goals

| ID | Non-Goal | Rationale |
|----|----------|-----------|
| NG1 | Web-based visualization | Out of scope; use Jaeger for web UI |
| NG2 | Historical replay from files | Focus on real-time; offline analysis via JSON |
| NG3 | Custom color themes | Single btop-style theme matches ttop |
| NG4 | Mouse interaction | Keyboard-only; consistent with ttop |

### 1.3 Design Principles

Per the Iron Lotus Framework [5, 12]:

1. **Genchi Genbutsu**: Direct syscall observation, no intermediaries
2. **Jidoka**: Automatic anomaly detection with visual alerts (Andon)
3. **Muda**: Zero-copy ring buffers, no allocation in hot path
4. **Poka-Yoke**: Type-safe panel IDs prevent configuration errors
5. **SIMD Everywhere**: All numeric data processing (aggregations, scaling, color mapping) MUST use SIMD primitives (AVX2/NEON) following `trueno-viz` patterns.

---

## 2. Architecture Overview

### 2.1 System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           renacer visualize                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Syscall   │  │   Anomaly   │  │     ML      │  │    OTLP     │        │
│  │  Collector  │  │  Detector   │  │  Pipeline   │  │   Exporter  │        │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │
│         │                │                │                │               │
│         └────────────────┴────────────────┴────────────────┘               │
│                                   │                                         │
│                          ┌────────▼────────┐                               │
│                          │  Metrics Store  │  ← Ring buffers (300 elem)    │
│                          │  (Normalized)   │                               │
│                          └────────┬────────┘                               │
│                                   │                                         │
│  ┌────────────────────────────────┼────────────────────────────────────┐   │
│  │                          Panel Router                                │   │
│  ├─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬────────┤   │
│  │ Syscall │ Anomaly │   ML    │  Trace  │ Process │  Stats  │  Help  │   │
│  │ Heatmap │Timeline │ Scatter │Waterfall│  Tree   │ Summary │ Overlay│   │
│  └────┬────┴────┬────┴────┬────┴────┬────┴────┬────┴────┬────┴───┬────┘   │
│       │         │         │         │         │         │        │        │
│       └─────────┴─────────┴─────────┴─────────┴─────────┴────────┘        │
│                                   │                                         │
│                          ┌────────▼────────┐                               │
│                          │    ratatui      │  ← Crossterm backend          │
│                          │   (trueno-viz)  │                               │
│                          └────────┬────────┘                               │
│                                   │                                         │
│                          ┌────────▼────────┐                               │
│                          │    Terminal     │  ← 8ms frame target           │
│                          └─────────────────┘                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow

```
ptrace syscall intercept
         │
         ▼
┌─────────────────┐
│ SyscallEvent    │ ← name, args, result, duration_us, source
└────────┬────────┘
         │
    ┌────┴────┬──────────────┬─────────────────┐
    │         │              │                 │
    ▼         ▼              ▼                 ▼
┌───────┐ ┌───────┐    ┌──────────┐    ┌────────────┐
│ Stats │ │Anomaly│    │   ML     │    │    OTLP    │
│Tracker│ │Detect │    │ Pipeline │    │  Exporter  │
└───┬───┘ └───┬───┘    └────┬─────┘    └─────┬──────┘
    │         │             │                │
    └─────────┴─────────────┴────────────────┘
                     │
                     ▼
           ┌─────────────────┐
           │  VisualizeApp   │ ← ttop-identical App struct
           │  (ring buffers) │
           └────────┬────────┘
                    │
                    ▼
           ┌─────────────────┐
           │   Panel Draw    │ ← 50ms tick, 8ms frame
           └─────────────────┘
```

### 2.3 Module Structure

```
src/
├── visualize/
│   ├── mod.rs              # Module exports
│   ├── app.rs              # VisualizeApp state (ttop::App pattern)
│   ├── ui.rs               # Layout and draw dispatcher
│   ├── panels/
│   │   ├── mod.rs          # Panel trait + registry
│   │   ├── syscall_heatmap.rs
│   │   ├── anomaly_timeline.rs
│   │   ├── ml_scatter.rs
│   │   ├── trace_waterfall.rs
│   │   ├── process_syscalls.rs
│   │   └── stats_summary.rs
│   ├── collectors/
│   │   ├── mod.rs          # Collector trait (trueno-viz pattern)
│   │   ├── syscall.rs      # Real-time syscall metrics
│   │   ├── anomaly.rs      # Z-score anomaly stream
│   │   └── span.rs         # OTLP span receiver
│   ├── widgets/
│   │   ├── braille_scatter.rs  # 2D scatter in braille mode
│   │   ├── gantt.rs            # Horizontal span bars
│   │   └── heatmap.rs          # Category × time heatmap
│   └── theme.rs            # Color gradients (ttop-identical)
├── cli.rs                  # Add `visualize` subcommand
└── lib.rs                  # Re-export visualize module
```

---

## 3. CLI Integration

### 3.1 Command Syntax

```bash
# Trivial invocation - trace and visualize
renacer visualize -- ./program [args...]

# Attach to running process
renacer visualize -p <PID>

# With ML analysis enabled
renacer visualize --ml-anomaly --ml-clusters 5 -- ./program

# With OTLP export + visualization
renacer visualize --otlp-endpoint http://localhost:4317 -- ./program

# Visualization-only options
renacer visualize --tick-rate 100 --no-anomaly -- ./program
```

### 3.2 CLI Arguments (Addition to cli.rs)

```rust
/// Real-time TUI visualization of syscall traces
#[derive(Parser, Debug)]
pub struct VisualizeArgs {
    /// Tick rate in milliseconds (default: 50, matches ttop)
    #[arg(long, default_value = "50")]
    pub tick_rate: u64,

    /// Disable anomaly detection panel
    #[arg(long)]
    pub no_anomaly: bool,

    /// Disable ML clustering panel
    #[arg(long)]
    pub no_ml: bool,

    /// Number of ML clusters (default: 3)
    #[arg(long, default_value = "3")]
    pub ml_clusters: usize,

    /// Anomaly Z-score threshold (default: 3.0)
    #[arg(long, default_value = "3.0")]
    pub anomaly_threshold: f32,

    /// History buffer size (default: 300, matches ttop)
    #[arg(long, default_value = "300")]
    pub history_size: usize,

    /// Panel layout: "default", "compact", "wide"
    #[arg(long, default_value = "default")]
    pub layout: String,

    /// Enable deterministic mode for testing
    #[arg(long)]
    pub deterministic: bool,
}
```

### 3.3 Subcommand Registration

```rust
#[derive(Subcommand)]
pub enum Commands {
    /// Validate trace against golden file
    Validate(ValidateArgs),

    /// Real-time TUI visualization of syscall traces
    Visualize(VisualizeArgs),
}
```

---

## 4. Visualization Primitives

### 4.1 Rendering Modes (ttop-Identical)

Per trueno-viz architecture [15], three rendering modes are supported:

| Mode | Resolution | Characters | Use Case |
|------|------------|------------|----------|
| **Braille** | 2×4 per cell | U+2800-28FF | High-density graphs, scatter plots |
| **Block** | 2×2 per cell | ▗▄▖▟▌▙█ | Medium-density, wide compatibility |
| **TTY** | 1×1 per cell | ASCII only | Pure terminal, SSH |

**Selection Logic:**
```rust
fn select_render_mode(term: &Terminal) -> RenderMode {
    if term.supports_unicode() && term.width() >= 80 {
        RenderMode::Braille
    } else if term.supports_unicode() {
        RenderMode::Block
    } else {
        RenderMode::TTY
    }
}
```

### 4.2 Color Gradients (ttop-Identical)

**Percentage Gradient** (syscall frequency, CPU utilization):
```rust
pub fn percentage_color(pct: f64) -> Color {
    match pct {
        p if p < 20.0 => Color::Rgb(64, 180, 220),   // Cyan
        p if p < 40.0 => Color::Rgb(100, 220, 100),  // Green
        p if p < 60.0 => Color::Rgb(220, 220, 80),   // Yellow
        p if p < 80.0 => Color::Rgb(255, 180, 100),  // Orange
        _ => Color::Rgb(255, 64, 64),                // Red
    }
}
```

**Severity Gradient** (anomaly Z-scores):
```rust
pub fn severity_color(z_score: f32) -> Color {
    match z_score {
        z if z < 3.0 => Color::Rgb(100, 220, 100),   // Green (normal)
        z if z < 4.0 => Color::Rgb(220, 220, 80),    // Yellow (low)
        z if z < 5.0 => Color::Rgb(255, 180, 100),   // Orange (medium)
        _ => Color::Rgb(255, 64, 64),                // Red (high)
    }
}
```

**Panel Border Colors** (btop-style):
```rust
pub const PANEL_COLORS: &[(PanelId, Color)] = &[
    (PanelId::SyscallHeatmap, Color::Rgb(100, 200, 255)),  // Cyan
    (PanelId::AnomalyTimeline, Color::Rgb(255, 100, 100)), // Red
    (PanelId::MlScatter, Color::Rgb(180, 120, 255)),       // Purple
    (PanelId::TraceWaterfall, Color::Rgb(255, 150, 100)),  // Orange
    (PanelId::ProcessSyscalls, Color::Rgb(220, 180, 100)), // Gold
    (PanelId::StatsSummary, Color::Rgb(100, 255, 150)),    // Green
];
```

### 4.3 Sparklines (ttop-Identical)

8-level Unicode block characters for inline trends:
```rust
const SPARKLINE_CHARS: &[char] = &['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

pub fn sparkline(values: &[f64], width: usize) -> String {
    let (min, max) = (values.min(), values.max());
    values.iter()
        .map(|v| {
            let idx = ((v - min) / (max - min) * 7.0).round() as usize;
            SPARKLINE_CHARS[idx.min(7)]
        })
        .take(width)
        .collect()
}
```

### 4.4 Braille Scatter Plot (New Widget)

For ML cluster visualization:
```rust
pub struct BrailleScatter {
    points: Vec<(f64, f64, u8)>,  // (x, y, cluster_id)
    x_range: (f64, f64),
    y_range: (f64, f64),
}

impl Widget for BrailleScatter {
    fn render(self, area: Rect, buf: &mut Buffer) {
        // Each cell is 2×4 braille dots
        // Map points to dot coordinates
        // Use cluster_id for color selection
        // Outliers (cluster_id = 255) rendered as '×' in red
    }
}
```

### 4.5 Gantt Chart (New Widget)

For OTLP span waterfall:
```rust
pub struct GanttChart<'a> {
    spans: &'a [SpanRecord],
    time_range: (u64, u64),  // nanoseconds
    selected: Option<usize>,
}

impl Widget for GanttChart<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        // Horizontal bars proportional to duration
        // Nested spans indented by parent relationship
        // Color by span_kind (Internal, Server, Client)
        // Critical path highlighted in bold
    }
}
```

---

## 5. Panel System

### 5.1 Panel Layout

**Default Layout** (2-row adaptive grid, ttop-identical):
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ ┌─ Syscalls ─────────────┐ ┌─ Anomalies ───────────┐ ┌─ ML Clusters ──────┐│
│ │ file   ▆▅▄▆▇█▇▆▄▃▂▄▅▆▇ │ │ Latency (μs)         │ │         ⠁⠁        ││
│ │ net    ▂▃▄▃▂▁▂▃▄▅▄▃▂▁▁ │ │ ▁▁▂▁▁▁█▁▁▃▁▁▁▁▂▁▁▁█▁ │ │   ⠂⠂⠂     ×       ││
│ │ mem    ▁▁▂▁▁▁▁▂▃▂▁▁▁▂█ │ │        ↑ 4.2σ    ↑ 5σ│ │ ⠂⠂⠂⠂⠂⠂     ⣿⣿   ││
│ │ proc   ▃▃▄▃▃▃▃▄▅▄▃▃▃▄▆ │ │ read 52ms 5.1σ ██▌   │ │   ⠂⠂⠂⠂            ││
│ └────────────────────────┘ └───────────────────────┘ └────────────────────┘│
│ ┌─ Process Syscalls ───────────────────────────────────────────────────────┐│
│ │ PID    Process       CPU%  Calls/s  Top Syscall     Trend     Errors    ││
│ │ 1234   nginx         12.3  4,521    read  ████▌     ▃▅▆▄▃▅       0      ││
│ │ 5678   postgres       8.7  3,102    futex ███▌      ▄▅▆▇▆       12      ││
│ │ 9012   myapp         24.1  5,224    write █████     ▆▇█▇▆        3      ││
│ └──────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Panel Descriptions

| Panel | Key | Description | Data Source |
|-------|-----|-------------|-------------|
| **Syscall Heatmap** | `1` | Category × time heatmap with sparklines | `SyscallCollector` |
| **Anomaly Timeline** | `2` | Z-score graph + anomaly table with source | `AnomalyDetector` |
| **ML Clusters** | `3` | PCA scatter plot in braille, outliers marked | `MLPipeline` |
| **Trace Waterfall** | `4` | OTLP span Gantt chart with critical path | `OtlpExporter` |
| **Process Syscalls** | `5` | Per-process syscall breakdown, sortable | `ProcessCollector` |
| **Stats Summary** | `6` | Aggregate statistics, percentiles | `StatsTracker` |

### 5.3 Panel Trait

```rust
pub trait Panel {
    /// Unique identifier for this panel
    fn id(&self) -> PanelId;

    /// Short title for panel header
    fn title(&self) -> &'static str;

    /// Generate header string with live metrics (btop-style)
    fn header(&self, app: &VisualizeApp) -> String;

    /// Render panel contents
    fn draw(&self, f: &mut Frame, area: Rect, app: &VisualizeApp);

    /// Handle panel-specific key events
    fn handle_key(&mut self, key: KeyCode) -> bool;
}
```

### 5.4 Panel Header Format (btop-Style)

```rust
impl Panel for SyscallHeatmapPanel {
    fn header(&self, app: &VisualizeApp) -> String {
        format!(
            " Syscalls {} calls/s │ {} anomalies │ {} errors │ top: {} ",
            app.syscall_rate,
            app.anomaly_count,
            app.error_count,
            app.top_syscall
        )
    }
}
```

---

## 6. Data Collection Architecture

### 6.1 Collector Trait (trueno-viz Pattern)

```rust
pub trait Collector: Send {
    /// Collect current metrics
    fn collect(&mut self) -> Result<Metrics>;

    /// Check if collector is available
    fn is_available(&self) -> bool;

    /// Collector name for logging
    fn name(&self) -> &'static str;
}
```

### 6.2 SyscallCollector

```rust
pub struct SyscallCollector {
    /// Receiver for syscall events from tracer
    rx: Receiver<SyscallEvent>,

    /// Per-syscall statistics
    stats: HashMap<String, SyscallStats>,

    /// Category aggregates
    categories: HashMap<SyscallCategory, CategoryStats>,

    /// Collection interval (default: 1000ms)
    interval: Duration,
}

impl Collector for SyscallCollector {
    fn collect(&mut self) -> Result<Metrics> {
        // Drain all pending syscall events
        while let Ok(event) = self.rx.try_recv() {
            self.update_stats(&event);
        }

        // Calculate rates (calls/sec)
        let metrics = self.stats.iter()
            .map(|(name, stats)| {
                (name.clone(), MetricValue::Gauge(stats.rate()))
            })
            .collect();

        Ok(Metrics::new(metrics))
    }
}
```

### 6.3 Ring Buffers (ttop-Identical)

```rust
pub struct RingBuffer<T> {
    data: Vec<T>,
    capacity: usize,
    head: usize,
    len: usize,
}

impl<T: Default + Clone> RingBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            data: vec![T::default(); capacity],
            capacity,
            head: 0,
            len: 0,
        }
    }

    /// O(1) push, automatically evicts oldest
    pub fn push(&mut self, value: T) {
        self.data[self.head] = value;
        self.head = (self.head + 1) % self.capacity;
        self.len = (self.len + 1).min(self.capacity);
    }

    /// Iterate from oldest to newest
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        let start = if self.len < self.capacity {
            0
        } else {
            self.head
        };
        (0..self.len).map(move |i| &self.data[(start + i) % self.capacity])
    }
}
```

### 6.4 VisualizeApp State

```rust
pub struct VisualizeApp {
    // Collectors
    pub syscall_collector: SyscallCollector,
    pub anomaly_detector: AnomalyDetector,
    pub ml_pipeline: Option<MLPipeline>,
    pub span_receiver: Option<SpanReceiver>,

    // History buffers (300 elements, normalized 0-1)
    pub syscall_history: HashMap<String, RingBuffer<f64>>,
    pub category_history: HashMap<SyscallCategory, RingBuffer<f64>>,
    pub anomaly_history: RingBuffer<f64>,  // Z-scores
    pub latency_history: RingBuffer<f64>,  // Avg latency

    // Current metrics
    pub syscall_rate: u64,
    pub anomaly_count: usize,
    pub error_count: u64,
    pub top_syscall: String,

    // ML state
    pub cluster_points: Vec<(f64, f64, u8)>,  // (x, y, cluster_id)
    pub outlier_indices: Vec<usize>,

    // OTLP spans (last 100)
    pub spans: VecDeque<SpanRecord>,

    // UI state
    pub panels: PanelVisibility,
    pub selected_panel: PanelId,
    pub process_selected: usize,
    pub sort_column: SortColumn,
    pub filter: String,
    pub show_help: bool,

    // Frame timing
    pub frame_id: u64,
    pub avg_frame_time_us: u64,
    pub max_frame_time_us: u64,
    pub frame_times: RingBuffer<u64>,  // Last 60 frames
}
```

---

## 7. Performance Specifications

### 7.1 Frame Timing Targets (ttop-Identical)

| Metric | Target | Measurement |
|--------|--------|-------------|
| Event tick rate | 50ms | `event::poll(Duration::from_millis(50))` |
| Average frame time | 8ms | 60-frame rolling average |
| P99 frame time | <16ms | Must maintain 60fps equivalent |
| Max frame time | <33ms | No visible stuttering |
| Collection interval | 1000ms | Metrics update rate |

### 7.2 Memory Budgets

| Component | Budget | Rationale |
|-----------|--------|-----------|
| Ring buffers (per metric) | 2.4KB | 300 × 8 bytes (f64) |
| Total ring buffers | <1MB | ~400 metrics max |
| Span buffer | <1MB | 100 spans × ~10KB each |
| ML cluster points | <100KB | 1000 points × 24 bytes |
| **Total RSS** | <10MB | Matches ttop target |

### 7.3 CPU Budget

| State | Target | Measurement |
|-------|--------|-------------|
| Idle (no tracing) | <1% | No active syscalls |
| Active (light load) | <5% | ~100 syscalls/sec |
| Active (heavy load) | <15% | ~10,000 syscalls/sec |
| ML analysis enabled | <25% | Clustering overhead |

### 7.4 Latency Specifications

| Operation | Target | Method |
|-----------|--------|--------|
| Syscall → display | <100ms | End-to-end latency |
| Anomaly detection | <10ms | Z-score computation |
| ML clustering | <500ms | K-means convergence |
| Panel switch | <1ms | Instant feedback |
| Filter application | <10ms | Process list filtering |

### 7.5 Scalability Limits

| Dimension | Limit | Behavior at Limit |
|-----------|-------|-------------------|
| Syscalls/sec | 100,000 | Sampling kicks in |
| Unique syscall types | 335 | All Linux syscalls |
| Traced processes | 1,000 | Hierarchical collapse |
| OTLP spans | 10,000/sec | Oldest evicted |
| ML data points | 10,000 | PCA dimensionality reduction |

### 7.6 SIMD Acceleration

| Operation | Target | Implementation |
|-----------|--------|----------------|
| Buffer Aggregation | 8x speedup | `std::simd` or `trueno` vectorization |
| Color Mapping | 4x speedup | Vectorized gradient interpolation |
| Anomaly Detection | 8x speedup | AVX2/NEON intrinsics for Z-score calc |
| Ring Buffer Writes | Zero-copy | `ptr::copy_nonoverlapping` (vectorized memcpy) |

---

## 8. Testing Strategy (Probador)

### 8.1 Test Architecture

Per Probador framework [16], testing follows three layers:

**Layer 1: Unit Tests (ON-SAVE)**
- Widget rendering (braille, block, TTY modes)
- Color gradient functions
- Ring buffer operations
- Sparkline generation

**Layer 2: Integration Tests (ON-COMMIT)**
- Collector → App state flow
- Panel rendering with mock data
- Keyboard navigation
- Layout constraint calculations

**Layer 3: E2E Tests (ON-MERGE)**
- Full TUI with synthetic syscall stream
- Deterministic mode for reproducibility
- Frame timing verification
- Memory leak detection

### 8.2 Deterministic Mode

```rust
impl VisualizeApp {
    pub fn new_deterministic(seed: u64) -> Self {
        Self {
            // Use seeded RNG for any randomness
            rng: StdRng::seed_from_u64(seed),
            // Fixed timestamp for frame timing
            start_time: Instant::now(),
            // Disable real collectors
            syscall_collector: SyscallCollector::mock(),
            ..Default::default()
        }
    }
}
```

### 8.3 Frame Capture Testing

```rust
#[test]
fn test_syscall_heatmap_renders_correctly() {
    let backend = TestBackend::new(80, 24);
    let mut terminal = Terminal::new(backend).unwrap();
    let mut app = VisualizeApp::new_deterministic(42);

    // Inject synthetic syscall data
    app.inject_syscalls(vec![
        SyscallEvent::new("read", 100),
        SyscallEvent::new("write", 200),
        SyscallEvent::new("open", 50),
    ]);

    terminal.draw(|f| draw_ui(f, &mut app)).unwrap();

    let frame = TuiFrame::from_buffer(terminal.backend().buffer(), 0);
    expect_frame(&frame)
        .to_contain_text("Syscalls")
        .to_contain_text("read")
        .to_contain_text("write")
        .unwrap();
}
```

### 8.4 Performance Testing

```rust
#[test]
fn test_frame_time_under_16ms() {
    let mut app = VisualizeApp::new_deterministic(42);

    // Inject heavy load
    for _ in 0..10_000 {
        app.inject_syscall(random_syscall());
    }

    let backend = TestBackend::new(120, 40);
    let mut terminal = Terminal::new(backend).unwrap();

    let mut frame_times = Vec::with_capacity(100);
    for _ in 0..100 {
        let start = Instant::now();
        terminal.draw(|f| draw_ui(f, &mut app)).unwrap();
        frame_times.push(start.elapsed().as_micros() as u64);
    }

    let avg = frame_times.iter().sum::<u64>() / frame_times.len() as u64;
    let p99 = frame_times.iter().sorted().nth(99).copied().unwrap();

    assert!(avg < 8000, "Average frame time {} > 8ms", avg);
    assert!(p99 < 16000, "P99 frame time {} > 16ms", p99);
}
```

### 8.5 Invariant Testing

```rust
#[test]
fn test_visualization_invariants() {
    let invariants = vec![
        Invariant::new("ring_buffer_bounded", |app| {
            app.syscall_history.values().all(|rb| rb.len() <= 300)
        }),
        Invariant::new("no_nan_in_display", |app| {
            app.latency_history.iter().all(|v| v.is_finite())
        }),
        Invariant::new("z_scores_positive", |app| {
            app.anomaly_history.iter().all(|z| *z >= 0.0)
        }),
        Invariant::new("cluster_ids_valid", |app| {
            app.cluster_points.iter().all(|(_, _, id)| *id < 10 || *id == 255)
        }),
    ];

    fuzz_with_invariants(100_000, invariants, |app| {
        app.inject_syscall(random_syscall());
        app.collect_metrics();
    });
}
```

### 8.6 Coverage Requirements

| Category | Target | Method |
|----------|--------|--------|
| Line coverage | ≥95% | `cargo llvm-cov` |
| Branch coverage | ≥90% | `cargo llvm-cov --branch` |
| Mutation score | ≥80% | `cargo mutants` |
| Panel coverage | 100% | Each panel rendered in tests |
| Keyboard coverage | 100% | Each key binding tested |

### 8.7 Test Matrix

| Test Category | Count | Coverage |
|---------------|-------|----------|
| Widget rendering | 15 | 100% |
| Color functions | 8 | 100% |
| Ring buffer ops | 12 | 100% |
| Panel layout | 10 | 100% |
| Collector flow | 8 | 100% |
| Keyboard handling | 20 | 100% |
| Performance | 5 | 100% |
| Invariants | 10 | 100% |
| Edge cases | 12 | 100% |
| **Total** | **100** | **100%** |

---

## 9. Implementation Phases

### Phase 1: Core Infrastructure (Sprint 52)
- [ ] CLI `visualize` subcommand
- [ ] `VisualizeApp` state management
- [ ] Ring buffer implementation
- [ ] Basic event loop (50ms tick)
- [ ] ttop theme integration

### Phase 2: Syscall Panels (Sprint 53)
- [ ] `SyscallCollector` implementation
- [ ] Syscall heatmap panel
- [ ] Process syscalls panel (table + sparklines)
- [ ] Stats summary panel
- [ ] Keyboard navigation (j/k, 1-6, ?, f)

### Phase 3: Anomaly & ML (Sprint 54)
- [ ] Anomaly timeline panel
- [ ] Braille scatter plot widget
- [ ] ML scatter panel
- [ ] Z-score overlay on heatmap
- [ ] Source location display
- [ ] Integration with `renacer::isolation_forest` (XAI features)

### Phase 4: OTLP & Polish (Sprint 55)
- [ ] Gantt chart widget
- [ ] Trace waterfall panel
- [ ] Critical path highlighting
- [ ] Deterministic mode
- [ ] 100% probador coverage

### Phase 5: GPU/HPU & Future Work (Post-Sprint 55)
- [ ] GPU Accelerator Panel (Kernel tracing stats)
- [ ] HPU Profiling integration
- [ ] Memory transfer tracking visualization
- [ ] Multi-node tracing support

---

## 10. References

[1] J. Nickolls, I. Buck, M. Garland, and K. Skadron, "Scalable Parallel Programming with CUDA," *ACM Queue*, vol. 6, no. 2, pp. 40-53, 2008. doi:10.1145/1365490.1365500

[2] V. J. Reddi et al., "MLPerf Inference Benchmark," in *Proc. ACM/IEEE Int. Symp. Computer Architecture (ISCA)*, 2020, pp. 446-459. doi:10.1109/ISCA45697.2020.00045

[3] B. H. Sigelman et al., "Dapper, a Large-Scale Distributed Systems Tracing Infrastructure," Google Technical Report, 2010.

[4] P. Barham et al., "Magpie: Online Modelling and Performance-aware Systems," in *Proc. 9th Workshop on Hot Topics in Operating Systems (HotOS)*, 2003.

[5] J. K. Liker, *The Toyota Way: 14 Management Principles from the World's Greatest Manufacturer*, 2nd ed. McGraw-Hill, 2021. ISBN: 978-1260468519

[6] L. Lamport, "Time, Clocks, and the Ordering of Events in a Distributed System," *Commun. ACM*, vol. 21, no. 7, pp. 558-565, 1978. doi:10.1145/359545.359563

[7] R. D. Blumofe and C. E. Leiserson, "Scheduling Multithreaded Computations by Work Stealing," *J. ACM*, vol. 46, no. 5, pp. 720-748, 1999. doi:10.1145/324133.324234

[8] V. Chandola, A. Banerjee, and V. Kumar, "Anomaly Detection: A Survey," *ACM Computing Surveys*, vol. 41, no. 3, pp. 1-58, 2009. doi:10.1145/1541880.1541882

[9] F. T. Liu, K. M. Ting, and Z.-H. Zhou, "Isolation Forest," in *Proc. IEEE Int. Conf. Data Mining (ICDM)*, 2008, pp. 413-422. doi:10.1109/ICDM.2008.17

[10] M. M. Breunig, H.-P. Kriegel, R. T. Ng, and J. Sander, "LOF: Identifying Density-Based Local Outliers," in *Proc. ACM SIGMOD Int. Conf. Management of Data*, 2000, pp. 93-104. doi:10.1145/342009.335388

[11] L. van der Maaten and G. Hinton, "Visualizing Data using t-SNE," *J. Machine Learning Research*, vol. 9, pp. 2579-2605, 2008.

[12] K. R. Popper, *The Logic of Scientific Discovery*. Routledge, 1959. ISBN: 978-0415278447

[13] W3C Trace Context Specification, "Trace Context Level 2," W3C Recommendation, 2021. https://www.w3.org/TR/trace-context-2/

[14] OpenTelemetry Authors, "OpenTelemetry Specification v1.0," CNCF, 2021. https://opentelemetry.io/docs/specs/otel/

[15] trueno-viz Contributors, "ttop: Terminal System Monitor," GitHub, 2025. https://github.com/paiml/trueno-viz

[16] Probador Contributors, "Probador: Pure Rust Testing Framework for WASM," GitHub, 2025. https://github.com/paiml/jugar

[17] D. E. Knuth, *The Art of Computer Programming, Volume 2: Seminumerical Algorithms*, 3rd ed. Addison-Wesley, 1997. ISBN: 978-0201896848

[18] J. L. Hennessy and D. A. Patterson, *Computer Architecture: A Quantitative Approach*, 6th ed. Morgan Kaufmann, 2017. ISBN: 978-0128119051

[19] N. Matsakis and F. Klock, "The Rust Language," in *Proc. ACM SIGAda Annual Conf. High Integrity Language Technology (HILT)*, 2014, pp. 103-104. doi:10.1145/2663171.2663188

[20] R. Jung et al., "RustBelt: Securing the Foundations of the Rust Programming Language," *Proc. ACM Program. Lang.*, vol. 2, no. POPL, pp. 66:1-66:34, 2018. doi:10.1145/3158154

---

## 11. Popperian Falsification Checklist (100 Points)

Per Popper's philosophy of science [12], a specification is only meaningful if it makes falsifiable predictions. Each item below describes a testable property that, if violated, falsifies the specification's claims.

### 11.1 CLI Integration (10 points)

| # | Falsifiable Claim | Test Method | Points |
|---|-------------------|-------------|--------|
| 1 | `renacer visualize -- ./prog` starts TUI without errors | Run command, verify terminal mode | 2 |
| 2 | `renacer visualize -p <PID>` attaches to running process | Attach to `sleep`, verify syscalls | 2 |
| 3 | `--tick-rate 100` changes event poll to 100ms | Measure poll interval | 1 |
| 4 | `--no-anomaly` hides anomaly panel | Verify panel not rendered | 1 |
| 5 | `--no-ml` hides ML panel | Verify panel not rendered | 1 |
| 6 | `--ml-clusters 5` creates 5 clusters | Count distinct cluster IDs | 1 |
| 7 | `--deterministic` produces identical output | Run twice, compare frame hashes | 2 |

### 11.2 Frame Timing (15 points)

| # | Falsifiable Claim | Test Method | Points |
|---|-------------------|-------------|--------|
| 8 | Average frame time < 8ms | Measure 100 frames, compute mean | 3 |
| 9 | P99 frame time < 16ms | Measure 100 frames, compute percentile | 3 |
| 10 | Max frame time < 33ms | Measure 100 frames, find max | 2 |
| 11 | Event tick rate is 50ms ± 5ms | Measure poll timing | 2 |
| 12 | FPS overlay shows accurate timing | Compare displayed vs measured | 2 |
| 13 | Frame timing stable under 10K syscalls/sec | Load test with benchmark | 3 |

### 11.3 Rendering Modes (10 points)

| # | Falsifiable Claim | Test Method | Points |
|---|-------------------|-------------|--------|
| 14 | Braille mode uses U+2800-28FF characters | Inspect buffer for braille | 2 |
| 15 | Block mode uses ▗▄▖▟▌▙█ characters | Inspect buffer for blocks | 2 |
| 16 | TTY mode uses ASCII only | Inspect buffer, all chars < 128 | 2 |
| 17 | Mode auto-selects based on terminal | Test each terminal type | 2 |
| 18 | Scatter plot renders in braille mode | Render ML panel, verify braille | 2 |

### 11.4 Color Gradients (8 points)

| # | Falsifiable Claim | Test Method | Points |
|---|-------------------|-------------|--------|
| 19 | 0% utilization renders cyan (64,180,220) | Call percentage_color(0.0) | 1 |
| 20 | 50% utilization renders yellow (220,220,80) | Call percentage_color(50.0) | 1 |
| 21 | 100% utilization renders red (255,64,64) | Call percentage_color(100.0) | 1 |
| 22 | Z-score 3.0 renders yellow | Call severity_color(3.0) | 1 |
| 23 | Z-score 5.0+ renders red | Call severity_color(5.0) | 1 |
| 24 | Panel borders match specified colors | Render each panel, verify color | 2 |
| 25 | Gradient interpolates smoothly | Check intermediate values | 1 |

### 11.5 Ring Buffers (8 points)

| # | Falsifiable Claim | Test Method | Points |
|---|-------------------|-------------|--------|
| 26 | Ring buffer capacity is 300 | Create buffer, verify capacity | 1 |
| 27 | Push is O(1) | Benchmark 10K pushes | 2 |
| 28 | Oldest value evicted at capacity | Push 301 values, verify first gone | 2 |
| 29 | Iteration order is oldest-to-newest | Push 1,2,3, verify iter order | 1 |
| 30 | No allocation after initialization | Monitor allocator during push | 2 |

### 11.6 Syscall Heatmap Panel (10 points)

| # | Falsifiable Claim | Test Method | Points |
|---|-------------------|-------------|--------|
| 31 | Panel title shows "Syscalls" | Render panel, check title | 1 |
| 32 | Header shows calls/sec rate | Inject syscalls, verify rate | 2 |
| 33 | Categories (file, net, mem, proc) displayed | Render panel, verify categories | 2 |
| 34 | Sparklines update with new data | Push data, verify sparkline changes | 2 |
| 35 | Press `1` toggles panel visibility | Press key, verify toggle | 1 |
| 36 | Panel respects layout constraints | Resize terminal, verify adaptive | 2 |

### 11.7 Anomaly Timeline Panel (10 points)

| # | Falsifiable Claim | Test Method | Points |
|---|-------------------|-------------|--------|
| 37 | Panel title shows "Anomalies" | Render panel, check title | 1 |
| 38 | Header shows anomaly count | Inject anomalies, verify count | 2 |
| 39 | Z-score graph renders spikes | Inject high Z-score, verify spike | 2 |
| 40 | Anomaly table shows syscall name | Inject anomaly, verify name | 1 |
| 41 | Anomaly table shows source file:line | Inject with source, verify display | 2 |
| 42 | Press `2` toggles panel visibility | Press key, verify toggle | 1 |
| 43 | High severity (>5σ) renders in red | Inject 5.5σ anomaly, verify color | 1 |

### 11.8 ML Scatter Panel (10 points)

| # | Falsifiable Claim | Test Method | Points |
|---|-------------------|-------------|--------|
| 44 | Panel title shows "ML Clusters" | Render panel, check title | 1 |
| 45 | Points rendered in braille characters | Inject points, verify braille | 2 |
| 46 | Different clusters have different colors | Inject 3 clusters, verify 3 colors | 2 |
| 47 | Outliers rendered as '×' in red | Inject outlier (cluster 255), verify | 2 |
| 48 | Header shows silhouette score | Run clustering, verify score | 1 |
| 49 | Press `3` toggles panel visibility | Press key, verify toggle | 1 |
| 50 | Panel handles 0 points gracefully | Render with empty data | 1 |

### 11.9 Trace Waterfall Panel (10 points)

| # | Falsifiable Claim | Test Method | Points |
|---|-------------------|-------------|--------|
| 51 | Panel title shows "Trace Waterfall" | Render panel, check title | 1 |
| 52 | Spans rendered as horizontal bars | Inject spans, verify bars | 2 |
| 53 | Nested spans are indented | Inject parent/child, verify indent | 2 |
| 54 | Duration proportional to bar width | Inject spans, measure widths | 2 |
| 55 | Critical path highlighted in bold | Calculate path, verify bold | 2 |
| 56 | Press `4` toggles panel visibility | Press key, verify toggle | 1 |

### 11.10 Process Syscalls Panel (10 points)

| # | Falsifiable Claim | Test Method | Points |
|---|-------------------|-------------|--------|
| 57 | Panel shows process table | Render panel, verify table | 1 |
| 58 | Columns: PID, Process, CPU%, Calls/s | Verify column headers | 1 |
| 59 | Sparklines in Trend column | Inject data, verify sparklines | 2 |
| 60 | j/k navigates rows | Press keys, verify selection | 2 |
| 61 | s cycles sort column | Press s, verify sort changes | 1 |
| 62 | f opens filter input | Press f, verify input shown | 1 |
| 63 | Filter applies to process list | Type filter, verify filtered | 2 |

### 11.11 Keyboard Navigation (7 points)

| # | Falsifiable Claim | Test Method | Points |
|---|-------------------|-------------|--------|
| 64 | q exits application | Press q, verify exit | 1 |
| 65 | ? shows help overlay | Press ?, verify help | 1 |
| 66 | 1-6 toggle respective panels | Press each, verify toggle | 2 |
| 67 | 0 resets all panels to visible | Press 0, verify reset | 1 |
| 68 | Tab cycles between panels | Press Tab, verify focus change | 1 |
| 69 | Esc clears overlays | Show help, press Esc, verify clear | 1 |

### 11.12 Memory & Performance (12 points)

| # | Falsifiable Claim | Test Method | Points |
|---|-------------------|-------------|--------|
| 70 | RSS < 10MB after 1 hour | Long-running test, monitor RSS | 3 |
| 71 | No memory leaks (RSS stable) | Monitor RSS over time | 3 |
| 72 | CPU < 5% when idle | Monitor CPU with no syscalls | 2 |
| 73 | CPU < 15% at 10K syscalls/sec | Load test, monitor CPU | 2 |
| 74 | Syscall → display latency < 100ms | Measure end-to-end | 2 |

### 11.13 Collectors (10 points)

| # | Falsifiable Claim | Test Method | Points |
|---|-------------------|-------------|--------|
| 75 | SyscallCollector receives events | Inject event, verify received | 2 |
| 76 | AnomalyDetector calculates Z-scores & IF scores | Inject data, verify scores | 2 |
| 77 | MLPipeline produces cluster assignments | Run pipeline, verify clusters | 2 |
| 78 | SpanReceiver handles OTLP spans | Send span, verify received | 2 |
| 79 | Collectors implement Collector trait | Compile-time check | 1 |
| 80 | Collectors report availability | Call is_available() | 1 |

### 11.14 Edge Cases (10 points)

| # | Falsifiable Claim | Test Method | Points |
|---|-------------------|-------------|--------|
| 81 | Empty syscall stream renders gracefully | Start with no syscalls | 1 |
| 82 | 100K syscalls/sec sampled, not crashed | Generate high load | 2 |
| 83 | NaN values don't crash renderer | Inject NaN, verify handled | 1 |
| 84 | Inf values don't crash renderer | Inject Inf, verify handled | 1 |
| 85 | Zero-width terminal handled | Set width=0, verify no panic | 1 |
| 86 | Very long syscall names truncated | Inject 1000-char name | 1 |
| 87 | Unicode in process names works | Inject emoji in name | 1 |
| 88 | Rapid panel toggles don't crash | Toggle 1000 times quickly | 1 |
| 89 | Terminal resize handled | SIGWINCH, verify re-layout | 1 |

### 11.15 Probador Coverage (10 points)

| # | Falsifiable Claim | Test Method | Points |
|---|-------------------|-------------|--------|
| 90 | Line coverage ≥ 95% | `cargo llvm-cov` | 3 |
| 91 | Branch coverage ≥ 90% | `cargo llvm-cov --branch` | 2 |
| 92 | Mutation score ≥ 80% | `cargo mutants` | 2 |
| 93 | All panels have render tests | Grep test files | 1 |
| 94 | All key bindings have tests | Grep test files | 1 |
| 95 | Deterministic mode reproduces frames | Run twice, compare hashes | 1 |

### 11.16 API Correctness (5 points)

| # | Falsifiable Claim | Test Method | Points |
|---|-------------------|-------------|--------|
| 96 | All public types implement Debug | Compile-time check | 1 |
| 97 | All public types implement Send + Sync | Compile-time check | 1 |
| 98 | No unwrap() in library code | Static analysis | 1 |
| 99 | Documentation examples compile | `cargo test --doc` | 1 |
| 100 | Error types include context | Inspect error messages | 1 |

### 11.17 Bridge Integration - QA Falsification Findings (5 bonus points)

**QA Protocol Executed:** 2026-01-07 "The Bridge Stress Test"

| # | Issue Discovered | Root Cause | Solution | Points |
|---|------------------|------------|----------|--------|
| 101 | TUI hangs on `curl` trace | Tracer stdout output corrupts TUI | Redirect stdout → /dev/null in tracer thread | 2 |
| 102 | 'q' key doesn't exit after trace | Child process may reset terminal raw mode | Re-enable raw_mode() after tracer thread joins | 1 |
| 103 | UI "stops" after trace completes | Channel polling dead receiver | Drop receiver when trace completes, set `trace_complete` flag | 1 |
| 104 | No indication trace finished | Auto-exit too fast for user | Display "TRACE COMPLETE - Press q to exit" banner | 1 |

**Architectural Decision Record (ADR):**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Bridge Architecture (Post-QA)                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Main Thread (TUI)                    Tracer Thread                          │
│  ─────────────────                    ─────────────                          │
│                                                                              │
│  1. Create mpsc channel ──────────────┐                                      │
│  2. Spawn tracer thread ──────────────┼──▶ suppress_stdout()                 │
│  3. enable_raw_mode()                 │    └─▶ dup2(/dev/null, STDOUT)       │
│  4. EnterAlternateScreen              │                                      │
│  5. Main loop:                        ├──▶ trace_command() / attach_to_pid() │
│     ├─ draw()                         │    ├─ ptrace syscalls                │
│     ├─ event::poll(50ms)              │    └─ tx.send(VisualizerEvent)       │
│     ├─ collect_metrics()              │                                      │
│     │  └─ rx.try_recv() ◀─────────────┘                                      │
│     ├─ check is_finished() ◀──────────────── thread exits                    │
│     │  ├─ drain remaining events                                             │
│     │  ├─ app.trace_complete = true                                          │
│     │  ├─ app.event_receiver = None                                          │
│     │  ├─ handle.join()                                                      │
│     │  └─ enable_raw_mode() // re-enable if child reset it                   │
│     └─ continue until 'q' pressed                                            │
│  6. disable_raw_mode()                                                       │
│  7. LeaveAlternateScreen                                                     │
│                                                                              │
│  Panic Safety: TerminalGuard Drop impl ensures cleanup on any exit path      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key Invariants:**
1. Tracer thread MUST NOT write to stdout (corrupts TUI)
2. Channel is unbounded (backpressure handled by 50ms tick drain)
3. UI MUST remain responsive after trace completes (user presses 'q')
4. Terminal raw mode MUST be restored if child process affected it

### 11.17 Integration Hardening (The Missing Link) (15 points)

| # | Falsifiable Claim | Test Method | Points |
|---|-------------------|-------------|--------|
| 101 | **Data Pipeline**: Tracer-to-UI bridge is active | Run `renacer visualize -- touch /tmp/qa_marker`, verify marker exists | 5 |
| 102 | **Memory Stability**: Unbounded channel backpressure handled | Run `yes | head -n 1M`, verify RSS < 50MB | 5 |
| 103 | **Lifecycle Management**: Clean exit on tracer completion | Run `ls`, verify TUI exits or prompts (no zombie threads) | 5 |

### 11.18 SIMD Acceleration (10 points)

| # | Falsifiable Claim | Test Method | Points |
|---|-------------------|-------------|--------|
| 104 | **SIMD Usage**: Binary contains AVX2/NEON instructions | `objdump -d` or `readelf` checks for `vaddps`, etc. | 3 |
| 105 | **Vectorization**: Aggregation uses `std::simd` | Benchmark vs scalar implementation (>4x speedup) | 4 |
| 106 | **Fallback**: Works on non-SIMD hardware | Run with `RUSTFLAGS="-C target-feature=-avx2"` | 3 |

---

**Total: 125 points**

Minimum passing score: **115 points** (per Iron Lotus quality standards)

---

## 12. Approval
**Specification Author:** Claude Code
**Review Required Before Implementation:** Yes

- [ ] Architecture review
- [ ] Performance review
- [ ] API design review
- [ ] Testing strategy review
- [ ] Falsification checklist review

**Awaiting user approval to proceed with implementation.**

---

## 13. Appendix D: Documentation Integration Strategy

**Mandate:** All functional examples and recipes in the documentation MUST use `{{#include ...}}` to directly link to the verified Test-Driven Development (TDD) source files.

**Rationale:**
1.  **Single Source of Truth:** Code examples in documentation are guaranteed to match the actual codebase.
2.  **Verified Correctness:** The included files are subject to `cargo test` and Probador QA, ensuring they compile and run as expected.
3.  **Maintenance:** Changes to the API are automatically reflected in the documentation, preventing "documentation drift."

**Implementation:**

1.  **Source File Structure:**
    Create a `examples/tui_recipes` directory for specific visualization recipes.
    ```rust
    // examples/tui_recipes/custom_panel.rs
    // ... verified code ...
    ```

2.  **MD Book Integration:**
    In `docs/book/src/recipes/custom_panel.md`:
    ```markdown
    # Creating a Custom Panel

    Here is a verified example of implementing the `Panel` trait:

    ```rust
    {{#include ../../../../examples/tui_recipes/custom_panel.rs}}
    ```
    ```

3.  **Validation:**
    The `book.yml` CI workflow must ensure that `mdbook test` or a custom script verifies the existence of all included files.
