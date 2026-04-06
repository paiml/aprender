# Framework Architecture

> Parent: [presentar-spec.md](../presentar-spec.md)

**Scope:** Layers 6-9, rendering pipeline, GPU shaders, chart primitives, layout engine, performance targets.

---

## Layer Architecture

| Layer | Component | Responsibility |
|-------|-----------|---------------|
| 9 | App Runtime | YAML parser, `.apr`/`.ald` loaders, Pacha integration, auto-display |
| 8 | Presentar | Widget tree, layout engine, event dispatch, state management |
| 7 | Trueno-Viz | GPU rendering: paths, fills, strokes, text, charts, WGSL shaders |
| 6 | Trueno | SIMD/GPU compute: tensor ops, backend dispatch, memory management |

```rust
// Layer 6: Trueno
pub struct Tensor<'a, T> { data: &'a [T], backend: Backend }

// Layer 7: Trueno-Viz
pub struct Canvas<'gpu> { context: &'gpu GpuContext, commands: Vec<DrawCommand>, viewport: Viewport }

// Layer 8: Presentar
pub struct App<S: State> { root: WidgetTree, state: S, layout: LayoutEngine, renderer: TruenoVizRenderer }

// Layer 9: Runtime
pub struct AppConfig { manifest: Manifest, data_sources: Vec<DataSource>, model_refs: Vec<ModelRef> }
```

## GPU Rendering Primitives (Trueno-Viz)

### DrawCommand

```rust
pub enum DrawCommand {
    Path { points: Vec<Point>, closed: bool, style: StrokeStyle },
    Fill { path: PathRef, color: Color, rule: FillRule },
    Rect { bounds: Rect, radius: CornerRadius, style: BoxStyle },
    Circle { center: Point, radius: f32, style: BoxStyle },
    Text { content: String, position: Point, style: TextStyle },
    Image { tensor: TensorRef, bounds: Rect, sampling: Sampling },
    Group { children: Vec<DrawCommand>, transform: Transform2D },
    Clip { bounds: Rect, child: Box<DrawCommand> },
    Opacity { alpha: f32, child: Box<DrawCommand> },
}
```

### Anti-Aliasing Strategy

| Technique | Use Case |
|-----------|----------|
| Hardware MSAA (4x) | Solid fills, basic shapes |
| SDF (Signed Distance Fields) | Text, icons, thin lines |
| Analytical AA | Chart lines, curves |

### Chart Primitives (Grammar of Graphics)

```rust
pub enum ChartType {
    Line { series: Vec<Series>, interpolation: Interpolation },
    Bar { series: Vec<Series>, orientation: Orientation, grouped: bool },
    Scatter { series: Vec<Series>, size_encoding: Option<String> },
    Heatmap { matrix: TensorRef, color_scale: ColorScale },
    Histogram { data: TensorRef, bins: BinStrategy },
    BoxPlot { groups: Vec<BoxPlotData> },
    Pie { slices: Vec<Slice>, donut_ratio: Option<f32> },
}
```

Charts convert to `DrawCommand` via `to_commands()`, using Trueno SIMD for interpolation.

## Layout Engine

Flexbox-inspired with Trueno SIMD acceleration. Two-phase: Measure (bottom-up) then Layout (top-down). O(n) with memoization.

```rust
impl LayoutEngine {
    pub fn compute(&mut self, root: &dyn Widget, viewport: Size) -> LayoutTree {
        let sizes = self.measure_tree(root, Constraints::loose(viewport));
        let positions = self.position_tree(root, Rect::from_size(viewport), &sizes);
        LayoutTree { sizes, positions }
    }
}
```

## State Management

Elm Architecture: `Event -> State -> Widget -> Draw`. All widgets are dumb renderers. Side effects via `Command` enum (`Task`, `LoadModel`, `LoadDataset`, `SaveState`, `Navigate`).

## YAML App Configuration

12-column responsive grid layout. Expression language for data binding:

```
{{ source | transform | transform }}
Transforms: filter, select, sort, limit, count, sum, mean, rate, percentage, join
```

All transforms execute client-side in WASM via Trueno.

### Auto-Display Rules

| Directory Contents | Generated UI |
|--------------------|--------------|
| `app.yaml` present | Custom layout from YAML |
| Single `.apr` file | ModelCard + inference panel |
| Single `.ald` file | DataCard + DataTable |
| Multiple files | Split-view grid |

## Model/Data Cards

Follows Mitchell et al. (2019) Model Cards and Gebru et al. (2021) Datasheets standards. Embedded in `app.yaml` or standalone. Includes metrics, training info, ethical considerations, lineage.

## Performance Targets

| Operation | Target |
|-----------|--------|
| Path tessellation (1K points) | < 1ms |
| Fill rendering (10K triangles) | < 2ms |
| Text layout (1K glyphs) | < 5ms |
| Chart update (100K points) | < 16ms |
| Full frame (complex dashboard) | < 16ms (60fps) |

## Build & Deployment

Two modes: `presentar --serve ./app/` (dev) and `presentar --bundle ./app/ -o app.wasm` (production).

Bundle contents (~300KB base): trueno-viz runtime (150KB), presentar widgets (100KB), embedded YAML (2KB), schemas only (not weights/rows).

### Quality Pipeline

| Tier | Timing | Scope |
|------|--------|-------|
| Tier 1 | < 1s | `cargo check`, YAML lint |
| Tier 2 | 1-5min | fmt, clippy, unit tests, integration, score check |
| Tier 3 | Hours | Visual regression, coverage, mutation testing, benchmarks |

## Test Harness (presentar-test)

Pure Rust, zero external dependencies. Event simulation, CSS-like widget selectors, state inspection, framebuffer with software rasterizer for visual regression. Fixed DPI (1.0), embedded test font (Inter), fixed viewport (1280x720) for determinism.

## References

- Elliott, C. & Hudak, P. (1997). Functional Reactive Animation. *ICFP '97*.
- Wilkinson, L. (2005). *The Grammar of Graphics*. Springer.
- Gamma, E. et al. (1994). *Design Patterns*. Addison-Wesley.
- Meyerovich, L. & Bodik, R. (2010). Parallel layout. *PLDI '10*.
- Satyanarayan, A. et al. (2017). Vega-Lite. *IEEE TVCG*, 23(1).
- Haas, A. et al. (2017). WebAssembly. *PLDI '17*.
