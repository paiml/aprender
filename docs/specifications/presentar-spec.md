# Presentar Specification

**Version:** 2.0.0
**Status:** Active
**Date:** 2026-03-18

## Overview

Presentar is a **pure-Rust visualization and application framework** for the Sovereign AI Stack. It provides GPU-accelerated rendering, WASM-first deployment, and a TUI system monitor (`ptop`) that achieves pixel-perfect parity with ttop/btop.

Unlike Streamlit/Gradio (Python GIL, runtime overhead), Presentar delivers 60fps rendering, compile-time safety, and deterministic reproducibility via YAML-driven configuration.

## Architecture

### Layer Hierarchy

```
Layer 9: App Runtime
  YAML parser, .apr/.ald loaders, Pacha integration
Layer 8: Presentar (Reactive UI Framework)
  Widget tree, layout engine, event dispatch, state management
Layer 7: Trueno-Viz (GPU Rendering Primitives)
  Paths, fills, strokes, text, charts, WGSL shaders
Layer 6: Trueno (SIMD/GPU Compute)
  Tensor ops, backend dispatch, memory management
```

### Data Flow (Unidirectional, Elm Architecture)

```
Event Input --> State Update --> Widget Diff --> Draw Commands --> GPU Render
```

## Crate Workspace

| Crate | Purpose |
|-------|---------|
| `presentar-core` | Widget trait, layout engine, Canvas abstraction |
| `presentar-yaml` | YAML manifest parser, `.prs` scene format, expression engine |
| `presentar-terminal` | Direct crossterm TUI backend, `ptop` system monitor |
| `presentar-test` | Pure-Rust test harness (no Selenium/Playwright) |

## Widget System

All widgets implement the core `Widget` trait (Composite pattern):

```rust
pub trait Widget: Send + Sync {
    fn type_id(&self) -> TypeId;
    fn measure(&self, constraints: Constraints) -> Size;
    fn layout(&mut self, bounds: Rect) -> LayoutResult;
    fn paint(&self, canvas: &mut Canvas);
    fn event(&mut self, event: &Event) -> Option<Message>;
    fn children(&self) -> &[Box<dyn Widget>];
}
```

**Built-in widgets:** Container, Row, Column, Stack, Text, Button, Slider, TextInput, Select, Checkbox, DataTable, Chart, ModelCard, DataCard.

**TUI widgets (ptop):** Border, Gauge, Graph, LineChart, Histogram, Heatmap, ScatterPlot, BoxPlot, ViolinPlot, ForceGraph, Sparkline, MemoryBar, CpuGrid, ProcessTable, NetworkPanel, ConnectionsPanel, GpuPanel, SensorsPanel, ContainersPanel, Treemap, ConfusionMatrix.

## State Management (Elm Pattern)

```rust
pub trait State: Clone + Serialize + Deserialize {
    type Message;
    fn update(&mut self, msg: Self::Message) -> Command<Self::Message>;
}
```

Commands enable side effects: `Task(Future)`, `LoadModel`, `LoadDataset`, `SaveState`, `Navigate`. Widgets are "dumb" renderers receiving data via props from State.

## YAML Configuration

Declarative app manifests (`app.yaml`) define layout, data sources, model references, interactions, and themes. Expression language (`{{ source | transform }}`) enables reactive data binding without imperative code.

Scene sharing uses the `.prs` format -- a portable, content-addressed manifest referencing external models/datasets by URL with BLAKE3 hashes.

## Component Specifications

| Document | Scope |
|----------|-------|
| [Framework Architecture](components/framework-architecture.md) | Layer 6-9 architecture, rendering pipeline, GPU shaders, performance targets |
| [Scene Format](components/scene-format.md) | `.prs` v1.0 specification, schema, expression language, security model |
| [TUI Rendering](components/tui-rendering.md) | Direct crossterm backend, CellBuffer, DiffRenderer, zero-alloc design |
| [ptop Panels](components/ptop-panels.md) | 14 panel implementations, layout, widget inventory, color system |
| [ptop Analyzers](components/ptop-analyzers.md) | 13 system analyzers, data sources, analyzer trait, parity metrics |
| [ptop Falsification](components/ptop-falsification.md) | F-series tests, pixel comparison framework, headless QA protocol |
| [Testing Philosophy](components/testing-philosophy.md) | Popperian falsificationism, severity levels, anti-patterns |
| [Examples Catalog](components/examples-catalog.md) | 50 executable examples with 15-point QA checklist |
| [Showcase Demos](components/showcase-demos.md) | Shell autocomplete demo, WASM integration, QA verification |
| [Quality Gates](components/quality-gates.md) | Scoring system, coverage enforcement, CI/CD pipeline |

## Design Principles

### Popperian Testing
Tests do not prove correctness. They fail to falsify incorrectness. Every feature has explicit falsifiable claims. All tests must be severity S3+ (likely to fail if bug exists).

### Brick Architecture
Widgets implement `Brick + Send + Sync` with performance assertions, budget enforcement, and self-describing diagnostics. ComputeBlock enables SIMD-optimized panel elements.

### Toyota Production System
- **Jidoka:** Stop-on-error in pipelines; schema validation before execution
- **Muda:** No embedded data in `.prs`; zero-alloc steady-state rendering
- **Heijunka:** Lazy resource loading; 60fps render cap
- **Poka-Yoke:** Required fields enforced; invalid states unrepresentable
- **Kaizen:** Tiered quality pipeline (Tier 1: <1s, Tier 2: <5s, Tier 3: hours)

## Quality Standards

| Metric | Target |
|--------|--------|
| Line coverage | >= 95% |
| Mutation score | >= 80% |
| Frame time | < 16ms (60fps) |
| WASM bundle | < 500KB |
| Clippy warnings | 0 |
| WCAG compliance | AA |
| Quality grade | A (90+) |

## References

- Popper, K. (1963). *Conjectures and Refutations*. Routledge.
- Wilkinson, L. (2005). *The Grammar of Graphics*. Springer.
- Satyanarayan, A. et al. (2017). Vega-Lite. *IEEE TVCG*, 23(1).
- Haas, A. et al. (2017). WebAssembly. *PLDI '17*.
- Elliott, C. & Hudak, P. (1997). Functional Reactive Animation. *ICFP '97*.
- Liker, J.K. (2004). *The Toyota Way*. McGraw-Hill.
- Ohno, T. (1988). *Toyota Production System*. Productivity Press.
