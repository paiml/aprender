# Presentar: Sovereign AI Visualization & App Framework

**Version:** 0.1.0 | **Status:** Specification Complete

Presentar is a **PURE WASM** visualization and rapid application framework built entirely on Sovereign AI Stack primitives. It replaces Streamlit, Gradio, and Panel with 60fps GPU-accelerated rendering, compile-time type safety, and deterministic reproducibility.

## Position in the Stack

```
┌─────────────────────────────────────────────────────────────────┐
│  Presentar (Visualization & Apps)           ← YOU ARE HERE     │
├─────────────────────────────────────────────────────────────────┤
│  Trueno-Viz (GPU Rendering Primitives)                         │
├─────────────────────────────────────────────────────────────────┤
│  Trueno (SIMD/GPU Compute) v0.7.3                               │
├─────────────────────────────────────────────────────────────────┤
│  Aprender (ML) | Realizar (Inference) | Alimentar (Data)       │
└─────────────────────────────────────────────────────────────────┘
```

## Core Principles

| Principle | Implementation |
|-----------|----------------|
| **80% Pure Stack** | All rendering via `trueno-viz` GPU primitives |
| **20% Minimal External** | Only `winit` (windowing) + `fontdue` (fonts) |
| **WASM-First** | Browser deployment without server dependencies |
| **YAML-Driven** | Declarative app configuration |
| **Graded Quality** | Every app receives F-A score via TDG metrics |

## Auto-Display: Convention Over Configuration

Presentar auto-generates UIs from Sovereign AI Stack file formats:

| File Type | Generated UI |
|-----------|--------------|
| `.apr` (Aprender model) | ModelCard + inference panel |
| `.ald` (Alimentar dataset) | DataCard + DataTable |
| `app.yaml` | Custom layout from YAML |
| Mixed `.apr`/`.ald` | Split-view grid |

```bash
# Point at a directory, get an app
presentar --serve ./fraud-detector/

# Bundle for deployment
presentar --bundle ./fraud-detector/ -o app.wasm
```

## YAML App Configuration

```yaml
presentar: "0.1"
name: "fraud-detection-dashboard"
version: "1.0.0"

# Data sources (Alimentar .ald files)
data:
  transactions:
    source: "pacha://datasets/transactions:latest"
    format: "ald"
    refresh: "5m"

# Model references (Aprender .apr files)
models:
  fraud_detector:
    source: "pacha://models/fraud-detector:1.2.0"
    format: "apr"

# Layout definition (12-column responsive grid)
layout:
  type: "dashboard"
  columns: 12
  sections:
    - id: "metrics"
      span: [1, 4]
      widgets:
        - type: "metric"
          label: "Fraud Rate"
          value: "{{ data.predictions | filter(fraud=true) | percentage }}"

    - id: "main-chart"
      span: [5, 12]
      widgets:
        - type: "chart"
          chart_type: "line"
          data: "{{ data.transactions }}"
          x: "timestamp"
          y: "amount"
```

## Quality Scoring

Every Presentar app receives a TDG score (0-100, F-A):

| Category | Weight | Metrics |
|----------|--------|---------|
| Structural | 25 | Widget complexity, layout depth |
| Performance | 20 | Frame time, memory, bundle size |
| Accessibility | 20 | WCAG AA, keyboard nav, ARIA |
| Data Quality | 15 | Completeness, freshness, schema |
| Documentation | 10 | Manifest, model/data cards |
| Consistency | 10 | Theme adherence, naming |

## Integration with Batuta Workflow

Presentar apps integrate with Batuta's 5-phase workflow:

```
Phase 1: Analysis    → presentar analyze app.yaml
Phase 2: Transpile   → (N/A - pure Rust)
Phase 3: Optimize    → presentar optimize --wasm-opt
Phase 4: Validate    → presentar test (zero-dep harness)
Phase 5: Deploy      → presentar --bundle → pacha publish
```

## presentar-test: Zero-Dependency E2E Testing

**Critical constraint:** No playwright, selenium, npm, or C bindings.

```rust
use presentar_test::*;

#[presentar_test]
fn inference_flow() {
    let mut h = Harness::new(include_bytes!("fixtures/app.tar"));
    h.type_text("[data-testid='input-amount']", "1500")
     .click("[data-testid='predict-btn']");
    h.assert_text_contains("[data-testid='result']", "Fraud Score:");
}

#[presentar_test]
fn visual_regression() {
    let mut h = Harness::new(include_bytes!("fixtures/app.tar"));
    Snapshot::assert_match("app-default", h.screenshot("[data-testid='app-root']"), 0.001);
}
```

**Determinism guarantees:**
- Fixed DPI: 1.0
- Font antialiasing: Grayscale only
- Fixed viewport: 1280x720
- Embedded test font (Inter)

## Trueno-Viz GPU Primitives

Presentar renders via Trueno-Viz draw commands:

```rust
pub enum DrawCommand {
    Path { points: Vec<Point>, closed: bool, style: StrokeStyle },
    Fill { path: PathRef, color: Color, rule: FillRule },
    Rect { bounds: Rect, radius: CornerRadius, style: BoxStyle },
    Text { content: String, position: Point, style: TextStyle },
    Image { tensor: TensorRef, bounds: Rect, sampling: Sampling },
}
```

**Anti-aliasing strategy:**
- Hardware MSAA (4x) for fills
- Analytical AA for lines/curves
- SDF for text rendering

## Pacha Registry Integration

```yaml
# Fetch models and datasets from Pacha
models:
  classifier:
    source: "pacha://models/mnist-cnn:1.0.0"

data:
  training:
    source: "pacha://datasets/mnist:latest"
```

Lineage tracking follows W3C PROV-DM for full provenance.

## Performance Targets

| Operation | Target | Backend |
|-----------|--------|---------|
| Path tessellation (1K points) | <1ms | Trueno SIMD |
| Fill rendering (10K triangles) | <2ms | WebGPU |
| Full frame (complex dashboard) | <16ms | 60fps |
| Bundle size | <500KB | WASM |

## Ruchy Script Integration (Future)

Embedded scripting for dynamic behavior:

```yaml
scripts:
  on_load: |
    let data = load_dataset("transactions")
    let filtered = data.filter(|row| row.amount > 100)
    set_state("filtered_data", filtered)
```

**Security:** Resource limits (1M instructions, 16MB memory, 10ms slice) prevent DoS.

## Comparison with Alternatives

| Feature | Presentar | Streamlit | Gradio |
|---------|-----------|-----------|--------|
| Runtime | WASM (no server) | Python | Python |
| Performance | 60fps GPU | ~10fps | ~10fps |
| Type Safety | Compile-time | Runtime | Runtime |
| Bundle Size | <500KB | ~50MB | ~30MB |
| Testing | Zero-dep harness | Manual | Manual |
| Reproducibility | Deterministic | Non-deterministic | Non-deterministic |

## presentar-terminal: Native TUI Backend

For terminal-based applications, **presentar-terminal** provides efficient character-cell rendering with the same Brick Architecture as the WASM stack.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  presentar-terminal (TUI)                                       │
├─────────────────────────────────────────────────────────────────┤
│  CellBuffer + DiffRenderer (efficient updates)                  │
├─────────────────────────────────────────────────────────────────┤
│  crossterm 0.28 (terminal control)                              │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Purpose |
|-----------|---------|
| `CellBuffer` | Character-cell buffer with RGBA colors |
| `DiffRenderer` | Efficient partial updates (only changed cells) |
| `Modifiers` | Text styling (bold, italic, underline) |
| `Color` | RGBA colors with transparency support |

### Example Usage

```rust
use presentar_terminal::{CellBuffer, Color, DiffRenderer, Modifiers};

// Create buffer
let mut buffer = CellBuffer::new(80, 24);

// Write colored text
buffer.update(0, 0, "H", Color::GREEN, Color::TRANSPARENT, Modifiers::NONE);
buffer.update(1, 0, "i", Color::GREEN, Color::TRANSPARENT, Modifiers::NONE);

// Render to terminal with diff optimization
let mut renderer = DiffRenderer::new();
renderer.flush(&mut buffer, &mut std::io::stdout())?;
```

### Widgets Available

- **Table**: Data tables with sorting and selection
- **Gauge**: Progress bars and meters
- **Sparkline**: Inline mini-charts
- **ForceGraph**: Force-directed network visualization
- **Treemap**: Hierarchical data visualization
- **Heatmap**: 2D density visualization
- **BoxPlot/ViolinPlot**: Statistical distributions

### Stack Dashboards

Batuta uses presentar-terminal for its TUI dashboards:

```bash
# Stack health dashboard
cargo run --example stack_graph_tui --features native

# Oracle RAG dashboard
cargo run --example rag_oracle_demo --features native
```

### Why Not ratatui?

presentar-terminal replaces ratatui for stack consistency:

| Feature | presentar-terminal | ratatui |
|---------|-------------------|---------|
| Stack native | Yes | No |
| Diff rendering | Built-in | Manual |
| Color model | RGBA f32 | Limited |
| Brick Architecture | Yes | No |
| PROBAR-SPEC-009 | Compliant | N/A |

## Agent Dashboard Integration

Presentar provides the visualization layer for the [Agent Runtime](./agent-runtime.md)
TUI dashboard. The `AgentDashboard` widget renders real-time agent loop state:

| Widget | Display | Source |
|--------|---------|--------|
| Loop progress | Iteration / max, phase indicator | `AgentDashboardState` |
| Tool call log | Tool name, result, latency | `ToolLogEntry` |
| Token usage | Input/output tokens, cost | `TokenUsage` |
| Guard status | Ping-pong detection, budget | `LoopGuard` state |

**Terminal mode:** `presentar-terminal` renders the dashboard in-terminal
(used by `batuta agent run --stream` and `batuta agent chat --stream`).

**WASM mode:** When targeting wos, presentar renders via Canvas2D in
the browser. Agents can screenshot their own dashboards via BrowserTool
for visual regression testing.

See [Agent Runtime: TUI Dashboard](./agent-runtime.md#tui-dashboard) for details.

## Academic Foundation

Key references (see full spec for 30+ citations):
- Czaplicki (2012): Elm Architecture
- Haas et al. (2017): WebAssembly performance model
- Mitchell et al. (2019): Model Cards
- Ohno (1988): Toyota Production System (Jidoka)

---

**Navigate:** [Table of Contents](../SUMMARY.md) | [Trueno-Viz](./trueno-viz.md) | [Trueno](./trueno.md)
