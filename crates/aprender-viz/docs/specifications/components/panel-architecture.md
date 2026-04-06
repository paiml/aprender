# Panel Architecture

> Parent: [ttop-v2-spec.md](../ttop-v2-spec.md) Section 7

**Scope:** PanelBrick trait, widget composition, ComputeBlock SIMD,
detail levels, layout grid, color system.

---

## 1. Panel Trait Hierarchy

```
Brick (jugar-probar)
  └── PanelBrick (ttop)
        └── ComputeBlock (presentar-terminal, optional SIMD)
```

### 1.1 PanelBrick

```rust
/// A renderable panel in the ttop monitor.
///
/// Each panel combines Brick (assertions + budget) with rendering
/// into a CellBuffer region. Panels are composed into a BrickHouse
/// with a total 16ms frame budget.
pub trait PanelBrick: Brick {
    /// Render panel content into the buffer at the given area.
    ///
    /// Contract: all writes MUST be within `area` bounds.
    /// Contract: render time MUST be within `self.budget().render_ms`.
    fn render(&self, app: &App, buf: &mut CellBuffer, area: Rect);

    /// Generate title string with current metrics.
    /// Format: " PanelName | metric1 | metric2 | ... "
    fn title(&self, app: &App) -> String;

    /// Minimum (width, height) below which the panel cannot render.
    /// If area is smaller, render() should be a no-op.
    fn min_size(&self) -> (u16, u16);

    /// Compute appropriate detail level from available height.
    fn detail_level(&self, height: u16) -> DetailLevel;
}
```

### 1.2 DetailLevel

```rust
pub enum DetailLevel {
    /// 0-5 rows: title + single utilization bar
    Minimal,
    /// 6-8 rows: + secondary bar, basic stats
    Compact,
    /// 9-14 rows: + thermal, power, clock, breakdown
    Normal,
    /// 15-19 rows: + process list, history graphs
    Expanded,
    /// 20+ rows (or fullscreen): everything
    Exploded,
}
```

---

## 2. Panel Inventory

### 2.1 P0 Panels (Core)

| Panel | Module | Widgets | Budget | Min Size |
|-------|--------|---------|--------|----------|
| CPU | `panels/cpu.rs` | CpuGrid, Gauge, Sparkline | 2ms | 30x6 |
| Memory | `panels/memory.rs` | MemoryBar, Gauge | 2ms | 30x6 |
| Disk | `panels/disk.rs` | Gauge, Sparkline | 1ms | 30x6 |
| Network | `panels/network.rs` | Sparkline, LineChart | 1ms | 30x6 |
| Process | `panels/process.rs` | ProcessTable | 3ms | 40x8 |

### 2.2 P1 Panels (Extended)

| Panel | Module | Widgets | Budget | Min Size |
|-------|--------|---------|--------|----------|
| **GPU** | `panels/gpu.rs` | **GpuPanel, Gauge, Sparkline** | 1ms | 30x6 |
| Sensors | `panels/sensors.rs` | Heatmap, Gauge | 1ms | 30x6 |
| PSI | `panels/psi.rs` | Gauge | 1ms | 30x4 |
| Connections | `panels/connections.rs` | ConnectionsPanel | 1ms | 40x6 |
| Containers | `panels/containers.rs` | ContainersPanel | 1ms | 40x6 |

### 2.3 P2-P3 Panels (Optional)

| Panel | Module | Widgets | Budget | Min Size |
|-------|--------|---------|--------|----------|
| Battery | `panels/battery.rs` | Gauge | 1ms | 30x4 |
| System | `panels/system.rs` | Text | 1ms | 30x4 |
| Treemap | `panels/treemap.rs` | Treemap | 1ms | 30x6 |
| Files | `panels/files.rs` | FilesPanel | 1ms | 30x6 |

---

## 3. Widget Inventory (from presentar-terminal)

### 3.1 System Monitor Widgets

| Widget | Source | Description |
|--------|--------|-------------|
| `Border` | `widgets/border.rs` | btop-style rounded corners |
| `Gauge` | `widgets/gauge.rs` | Horizontal progress bar with percent |
| `Sparkline` | `widgets/sparkline.rs` | Mini line chart (braille/block) |
| `LineChart` | `widgets/line_chart.rs` | Full line chart with axes |
| `Histogram` | `widgets/histogram.rs` | Vertical bar chart |
| `Heatmap` | `widgets/heatmap.rs` | 2D color matrix |
| `CpuGrid` | `widgets/cpu_grid.rs` | Per-core meter grid |
| `CpuExploded` | `widgets/cpu_exploded.rs` | Fullscreen CPU detail |
| `MemoryBar` | `widgets/memory_bar.rs` | Stacked memory bar |
| `ProcessDataframe` | `widgets/process_dataframe.rs` | Sortable process table |
| `DiskPanel` | `widgets/disk_panel.rs` | Disk I/O + usage |
| `GpuPanel` | `widgets/gpu_panel.rs` | GPU utilization + VRAM |
| `BatteryPanel` | `widgets/battery_panel.rs` | Battery status |
| `ConnectionsPanel` | `widgets/connections_panel.rs` | Network connections |
| `ContainersPanel` | `widgets/containers_panel.rs` | Docker containers |
| `FilesPanel` | `widgets/files_panel.rs` | File browser |
| `Treemap` | `widgets/treemap.rs` | Squarified treemap |
| `SegmentedMeter` | `widgets/segmented_meter.rs` | Multi-segment bar |
| `InfoDense` | `widgets/info_dense.rs` | Key-value info block |
| `SemanticLabel` | `widgets/semantic_label.rs` | Typed label with color |

### 3.2 Widget → CellBuffer API

All widgets render directly to CellBuffer:

```rust
pub trait TerminalWidget {
    fn render(&self, buf: &mut CellBuffer, area: Rect);
}
```

No intermediate `Frame` or `Buffer` type. This eliminates the ratatui
double-buffering overhead.

---

## 4. ComputeBlock (SIMD Optimization)

### 4.1 Trait

```rust
/// SIMD-optimized panel element (SPEC-024 Section 21.6).
///
/// ComputeBlocks process f32 arrays into rendered character sequences
/// using SIMD instructions where available.
pub trait ComputeBlock {
    /// Process input data into rendered output.
    fn compute(&self, input: &[f32], output: &mut [char], width: usize);

    /// Detect best available SIMD instruction set.
    fn simd_level(&self) -> SimdInstructionSet;
}
```

### 4.2 SIMD Dispatch

| Platform | ISA | Vector Width | Use Case |
|----------|-----|-------------|----------|
| x86_64 | AVX2 | 8xf32 | CPU history sparkline |
| x86_64 | SSE4.1 | 4xf32 | Fallback |
| aarch64 | NEON | 4xf32 | Jetson/ARM servers |
| wasm32 | SIMD128 | 4xf32 | Future WASM target |
| Any | Scalar | 1xf32 | Universal fallback |

### 4.3 Example: Sparkline ComputeBlock

```rust
impl ComputeBlock for SparklineCompute {
    fn compute(&self, input: &[f32], output: &mut [char], width: usize) {
        let blocks = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
        let max = input.iter().cloned().fold(f32::MIN, f32::max);
        let scale = if max > 0.0 { 7.0 / max } else { 0.0 };

        // AVX2: process 8 values at once
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") {
            unsafe { sparkline_avx2(input, output, width, scale, &blocks) };
            return;
        }

        // Scalar fallback
        for (i, &val) in input.iter().take(width).enumerate() {
            let idx = ((val * scale) as usize).min(7);
            output[i] = blocks[idx];
        }
    }
}
```

---

## 5. Layout Grid

### 5.1 Default Layout

```
┌──────────────────┬──────────────────┐  ─┐
│       CPU        │     Memory       │   │ 45% height
├──────────────────┼──────────────────┤   │ (top panels, 2-col adaptive)
│       Disk       │     Network      │   │
├──────────────────┤──────────────────┤  ─┘
│   Processes      │  Connections     │  ─┐
│   (40%)          │  (30%)           │   │ 55% height
│                  ├──────────────────┤   │ (bottom row, 3-col)
│                  │  Treemap/Files   │   │
│                  │  (30%)           │   │
└──────────────────┴──────────────────┘  ─┘
```

### 5.2 Layout Algorithm

```rust
fn calculate_layout(w: u16, h: u16, visible: &PanelVisibility) -> Vec<PanelRect> {
    let top_height = (h as f32 * 0.45) as u16;
    let bottom_height = h - top_height;

    let top_panels = count_visible_top(visible);
    let cols = if top_panels <= 2 { top_panels } else { 2 };
    let rows = (top_panels + cols - 1) / cols;

    // ... grid subdivision ...
}
```

### 5.3 GPU Panel — MANDATORY (NVIDIA + AMD)

The GPU panel is P0 MUST. Both NVIDIA and AMD GPUs are supported.

**NVIDIA detection chain (priority order):**
1. `nvidia-smi --query-gpu=...` (most reliable)
2. NVML library (if available)
3. `/sys/bus/pci/devices/*/` with vendor=0x10de

**AMD detection chain:**
1. `/sys/class/drm/card*/device/gpu_busy_percent` (AMDGPU driver)
2. `/sys/class/drm/card*/device/mem_info_vram_used`
3. `/sys/class/drm/card*/device/hwmon/*/temp1_input`

**GPU panel layout (Normal detail):**
```
╭ NVIDIA RTX 4090 │ 45°C │ 120W ──────────────────╮
│ GPU  ████████████████░░░░░░ 72%                   │
│ VRAM ██████████░░░░░░░░░░░░ 8.2G/24G (34%)       │
│ Clock: 2520MHz  Fan: 65%  PCIe: Gen4 x16         │
│ PID   TYPE  MEM     COMMAND                       │
│ 1234  G     2.1G    Xorg                          │
│ 5678  C     4.0G    python3                       │
╰───────────────────────────────────────────────────╯
```

**Temperature color coding:**
- < 60°C: green (safe)
- 60-79°C: yellow (warm)
- >= 80°C: red (thermal throttle risk)

**Process type badges:**
- `G` (magenta): Graphics process
- `C` (cyan): Compute process

**No GPU fallback:**
```
╭ GPU │ No GPU detected ───────────────────────────╮
│ No NVIDIA or AMD GPU found                        │
│ Install nvidia-smi or check AMDGPU driver         │
╰───────────────────────────────────────────────────╯
```

### 5.4 Exploded Mode

Tab key cycles through panels in fullscreen mode:

```
ExplodedView(panel: PanelType)
  → render single panel at full terminal dimensions
  → DetailLevel::Exploded
  → Tab to next, Esc to exit
```

---

## 6. Color System

### 6.1 percent_color Gradient

```rust
/// Maps 0.0-100.0 to green→yellow→red gradient.
/// Reference: Tufte (1983) — color encodes data, not decoration.
fn percent_color(pct: f64) -> Color {
    match pct {
        p if p < 50.0 => Color::Rgb(
            (p * 5.1) as u8,       // 0→255
            255,                     // full green
            0,
        ),
        p => Color::Rgb(
            255,                     // full red
            (255.0 - (p - 50.0) * 5.1) as u8, // 255→0
            0,
        ),
    }
}
```

### 6.2 Panel Border Colors

Each panel type has a designated border color for visual identification:

| Panel | Color | RGB |
|-------|-------|-----|
| CPU | Cyan | (100, 200, 220) |
| Memory | Green | (100, 220, 140) |
| Disk | Orange | (220, 180, 100) |
| Network | Blue | (100, 140, 220) |
| Process | White | (200, 200, 200) |
| GPU | Magenta | (200, 100, 220) |

### 6.3 DisplayRules Enforcement

```rust
/// Tufte's data-ink ratio: every colored cell must carry information.
/// No decorative gradients, no chartjunk.
pub struct DisplayRules;

impl DisplayRules {
    /// Verify panel output meets Tufte data-ink criteria.
    pub fn verify(buf: &CellBuffer, area: Rect) -> Vec<Violation> {
        let mut violations = Vec::new();
        // Check: no cells with color but empty symbol
        // Check: border chars use muted colors (not data colors)
        // Check: data colors map to data values (not arbitrary)
        violations
    }
}
```

---

## 7. References

- presentar-terminal widgets: reference implementation
- PROBAR-SPEC-009: Brick/ComputeBlock architecture
- SPEC-024: presentar ptop panel specifications
- Tufte, E. (1983). Data-ink ratio. *Visual Display of Quantitative Information*.
- Fog, A. (2023). SIMD optimization patterns.
