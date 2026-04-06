# Presentar: Sovereign AI Visualization & App Framework

**Version:** 0.1.2 (Draft - Genchi Genbutsu Review)
**Status:** Specification
**Last Updated:** 2025-11-29

## Executive Summary

Presentar is a **PURE WASM** visualization and rapid application framework built entirely on **Sovereign AI Stack** primitives—a vertically integrated Rust ecosystem (Trueno, Aprender, Realizar, Pacha, etc.) that eliminates Python/CUDA/cloud dependencies for fully self-hosted AI workloads. Unlike Streamlit, Gradio, or Panel which suffer from Python's GIL, poor testability, and runtime overhead, Presentar delivers **60fps GPU-accelerated rendering**, **compile-time type safety**, and **deterministic reproducibility**.

> **Toyota Principle (Muda):** Elimination of Waste. By removing the Python GIL and runtime interpretation overhead, we eliminate "waiting" waste. Deterministic reproducibility further eliminates waste from debugging non-reproducible issues—every run produces identical output given identical input. We strictly adhere to the **Data-Ink Ratio** (Tufte, 1983), ensuring every pixel rendered conveys information, minimizing "chart junk."

> **Toyota Principle (Standardized Work):** Standardization is the basis for continuous improvement (Liker, 2004). Presentar enforces standardized architectural patterns via the compiler.

**Core Principles:**
- **80% Pure Stack**: All rendering via `trueno-viz` GPU primitives
- **20% Minimal External**: Only windowing (`winit`) and font rasterization (`fontdue`)—chosen because WASM lacks native window/event loop APIs and font hinting requires platform-specific complexity that would bloat the stack without clear benefit
- **WASM-First**: Browser deployment without server dependencies (Haas et al., 2017)
- **YAML-Driven**: Declarative app configuration, no code required for common patterns
- **Graded Quality**: Every app receives F-A score via TDG metrics
- **Reproducible**: Guaranteed distinct computational state (Peng, 2011)

## 1. Architecture

### 1.1 Layer Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│  Layer 9: App Runtime                                           │
│  - YAML parser, .apr/.ald loaders, Pacha integration            │
├─────────────────────────────────────────────────────────────────┤
│  Layer 8: Presentar (Reactive UI Framework)                     │
│  - Widget tree, layout engine, event dispatch, state management │
├─────────────────────────────────────────────────────────────────┤
│  Layer 7: Trueno-Viz (GPU Rendering Primitives)                 │
│  - Paths, fills, strokes, text, charts, WGSL shaders            │
├─────────────────────────────────────────────────────────────────┤
│  Layer 6: Trueno (SIMD/GPU Compute)                             │
│  - Tensor ops, backend dispatch, memory management              │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Component Boundaries

```rust
// Layer 6: Trueno (existing)
// Foundational Array Programming model (Iverson, 1962)
// Utilizes columnar data layout for SIMD efficiency (Stonebraker et al., 2005)
// Lifetime 'a enables zero-copy access to underlying data buffers,
// critical for GPU upload without intermediate allocations.
pub struct Tensor<'a, T> {
    data: &'a [T],
    backend: Backend,
}

// Layer 7: Trueno-Viz (new)
// Data Locality pattern for GPU throughput (Nystrom, 2014)
// Lifetime 'gpu enforces that Canvas cannot outlive its GPU context,
// preventing use-after-free of hardware resources.
pub struct Canvas<'gpu> {
    context: &'gpu GpuContext,
    commands: Vec<DrawCommand>,
    viewport: Viewport,
}

// Resource references are indices into per-frame resource buffers,
// enabling efficient batching and preventing dangling pointers.
// NOTE: Simple u32 indices risk "ABA problems" if slots are reused across frames.
// Future hardening (1.x): Consider generation indices (SlotMap pattern) for
// cross-frame handle validation. For 0.1, per-frame buffer reset mitigates this.
pub type PathRef = u32;    // Index into path buffer
pub type TensorRef = u32;  // Index into tensor buffer

// Layer 8: Presentar (new)
pub struct App<S: State> {
    root: WidgetTree,
    state: S,
    layout: LayoutEngine,
    renderer: TruenoVizRenderer,
}

// Layer 9: Runtime (new)
pub struct AppConfig {
    manifest: Manifest,
    data_sources: Vec<DataSource>,
    model_refs: Vec<ModelRef>,
}
```

### 1.3 Data Flow (Unidirectional)

```
┌─────────┐    ┌─────────┐    ┌──────────┐    ┌─────────┐
│  Event  │───▶│  State  │───▶│  Widget  │───▶│  Draw   │
│  Input  │    │  Update │    │  Diff    │    │  Cmds   │
└─────────┘    └─────────┘    └──────────┘    └─────────┘
     │                                              │
     │              ┌───────────┐                   │
     └──────────────│  GPU      │◀──────────────────┘
                    │  Render   │
                    └───────────┘
```

**Academic Foundation:** Functional Reactive Animation (Elliott & Hudak, 1997) adapted to the Elm Architecture (Czaplicki, 2012) with GPU-accelerated virtual DOM diffing inspired by React Fiber (Facebook, 2017).

## 2. Trueno-Viz: GPU Rendering Primitives

### 2.1 Core Types

```rust
/// Drawing primitive - all rendering reduces to these
pub enum DrawCommand {
    // Geometry
    Path { points: Vec<Point>, closed: bool, style: StrokeStyle },
    Fill { path: PathRef, color: Color, rule: FillRule },
    Rect { bounds: Rect, radius: CornerRadius, style: BoxStyle },
    Circle { center: Point, radius: f32, style: BoxStyle },

    // Text (fontdue rasterization, GPU compositing)
    Text { content: String, position: Point, style: TextStyle },

    // Images (Trueno tensor → GPU texture)
    Image { tensor: TensorRef, bounds: Rect, sampling: Sampling },

    // Compositing
    Group { children: Vec<DrawCommand>, transform: Transform2D },
    Clip { bounds: Rect, child: Box<DrawCommand> },
    Opacity { alpha: f32, child: Box<DrawCommand> },
}

/// Style types built on Trueno color math
pub struct Color {
    pub r: f32, pub g: f32, pub b: f32, pub a: f32,
}

impl Color {
    /// Perceptually uniform color space transforms (Levkowitz & Herman, 1992)
    pub fn to_lab(&self) -> LabColor { /* ... */ }

    /// WCAG 2.1 contrast ratio calculation
    pub fn contrast_ratio(&self, other: &Color) -> f32 {
        let l1 = self.relative_luminance();
        let l2 = other.relative_luminance();
        (l1.max(l2) + 0.05) / (l1.min(l2) + 0.05)
    }
}
```

### 2.2 WGSL Shader Pipeline

```wgsl
// trueno_viz_fill.wgsl - GPU fill shader
struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4<f32>(in.position, 0.0, 1.0);
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
```

#### 2.2.1 Anti-Aliasing Strategy

Vector graphics require explicit anti-aliasing for quality rendering:

| Technique | Use Case | Implementation |
|-----------|----------|----------------|
| **Hardware MSAA** | Solid fills, basic shapes | 4x MSAA via WebGPU `multisample` |
| **SDF (Signed Distance Fields)** | Text, icons, thin lines | Shader-based, resolution-independent |
| **Analytical AA** | Chart lines, curves | Edge distance in fragment shader |

```wgsl
// trueno_viz_line_aa.wgsl - Analytical anti-aliased line
@fragment
fn fs_line(in: LineVertexOutput) -> @location(0) vec4<f32> {
    // Distance from fragment to line center (in pixels)
    let dist = abs(in.edge_distance);
    // Smooth falloff over 1 pixel for AA
    let alpha = 1.0 - smoothstep(in.line_width - 1.0, in.line_width, dist);
    return vec4<f32>(in.color.rgb, in.color.a * alpha);
}
```

**Default:** 4x MSAA for fills + analytical AA for lines/curves. SDF for text (via fontdue rasterization at 2x then downscale).

### 2.3 Chart Primitives

All charts are compositions of `DrawCommand` primitives, adhering to the Grammar of Graphics (Wilkinson, 2005):

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

impl ChartType {
    /// Convert chart to GPU draw commands
    pub fn to_commands(&self, bounds: Rect, theme: &Theme) -> Vec<DrawCommand> {
        match self {
            Self::Line { series, interpolation } => {
                // Trueno SIMD for interpolation calculations
                let points = trueno::interpolate(series, interpolation);
                vec![DrawCommand::Path {
                    points,
                    closed: false,
                    style: theme.line_style()
                }]
            }
            // ... other chart types
        }
    }
}
```

### 2.4 Performance Targets

| Operation | Target | Backend |
|-----------|--------|---------|
| Path tessellation (1K points) | <1ms | Trueno SIMD |
| Fill rendering (10K triangles) | <2ms | WebGPU |
| Text layout (1K glyphs) | <5ms | fontdue + GPU |
| Chart update (100K points) | <16ms | Full pipeline |
| Full frame (complex dashboard) | <16ms | 60fps target |

*Validation: The decision to dispatch specific compute-heavy visualization tasks to GPU vs CPU is critical for maintaining high throughput (Gregg & Hazelwood, 2011).*

*Enforcement: These targets are validated by `presentar-test` (Section 6.5) via the `#[presentar_test] fn performance_under_16ms()` pattern, making performance regressions CI-blocking.*

## 3. Presentar: Reactive UI Framework

### 3.1 Widget System

Implements the **Composite Pattern** (Gamma et al., 1994) to treat individual objects and compositions uniformly.
Input handling respects **Fitts's Law** (Fitts, 1954) for touch target sizing.

```rust
/// Core widget trait - all UI elements implement this
pub trait Widget: Send + Sync {
    /// Unique type identifier for diffing
    fn type_id(&self) -> TypeId;

    /// Compute intrinsic size constraints
    fn measure(&self, constraints: Constraints) -> Size;

    /// Position children within allocated bounds
    fn layout(&mut self, bounds: Rect) -> LayoutResult;

    /// Generate draw commands
    fn paint(&self, canvas: &mut Canvas);

    /// Handle input events, return state mutations
    fn event(&mut self, event: &Event) -> Option<Message>;

    /// Child widgets for tree traversal
    fn children(&self) -> &[Box<dyn Widget>];
}

/// Built-in widgets
/// Grouped to minimize Cognitive Load (Hick, 1952)
pub mod widgets {
    pub struct Container { /* layout, padding, decoration */ }
    pub struct Row { /* horizontal flex layout */ }
    pub struct Column { /* vertical flex layout */ }
    pub struct Stack { /* z-order stacking */ }
    pub struct Text { /* styled text display */ }
    pub struct Button { /* interactive button */ }
    pub struct Slider { /* value slider */ }
    pub struct TextInput { /* text entry */ }
    pub struct Select { /* dropdown selection */ }
    pub struct Checkbox { /* boolean toggle */ }
    pub struct DataTable { /* virtualized table, renders data from State (see note) */ }
    pub struct Chart { /* Trueno-Viz chart wrapper */ }
    pub struct ModelCard { /* displays .apr (Aprender) metadata from Pacha registry */ }
    pub struct DataCard { /* displays .ald (Alimentar) metadata from Pacha registry */ }
}

// NOTE on "smart" vs "dumb" widgets:
// All widgets are "dumb" renderers—they receive data via props from State.
// Loading .apr/.ald files happens via Command::LoadModel/LoadDataset, which
// updates State, which triggers re-render. Widgets never fetch their own data.
// This preserves unidirectional data flow: Event → State → Widget → Draw.
```

### 3.2 State Management

Deprecating the Observer pattern in favor of Functional Reactive Programming signals (Maier & Odersky, 2010).
Async tasks utilize Cooperative Task Management (Adya et al., 2002) to prevent UI blocking on the main WASM thread.

```rust
/// Application state with automatic persistence
pub trait State: Clone + Serialize + Deserialize {
    type Message;

    fn update(&mut self, msg: Self::Message) -> Command<Self::Message>;
}

/// Commands for side effects
pub enum Command<M> {
    None,
    Batch(Vec<Command<M>>),
    // Pin ensures the Future won't move in memory (required for self-referential async).
    // Send is required for potential wasm-bindgen-rayon multi-threading.
    // In single-threaded WASM, executor polls futures cooperatively on the main thread.
    Task(Pin<Box<dyn Future<Output = M> + Send>>),
    LoadModel { path: String, on_load: fn(Model) -> M },
    LoadDataset { path: String, on_load: fn(Dataset) -> M },
    SaveState { key: String },
    Navigate { route: String },
}
```

### 3.3 Layout Engine

Flexbox-inspired layout with Trueno SIMD acceleration:

```rust
pub struct LayoutEngine {
    cache: LayoutCache,
}

impl LayoutEngine {
    /// O(n) layout pass with memoization
    pub fn compute(&mut self, root: &dyn Widget, viewport: Size) -> LayoutTree {
        // Phase 1: Measure (bottom-up)
        let sizes = self.measure_tree(root, Constraints::loose(viewport));

        // Phase 2: Layout (top-down)
        let positions = self.position_tree(root, Rect::from_size(viewport), &sizes);

        LayoutTree { sizes, positions }
    }

    /// SIMD-accelerated flex distribution
    fn distribute_flex(&self, items: &[FlexItem], available: f32) -> Vec<f32> {
        // Trueno SIMD for parallel flex calculations
        trueno::flex_distribute(items, available)
    }
}
```

**Academic Foundation:** Parallel layout algorithms with SIMD (Meyerovich & Bodik, 2010) and CSS Flexbox specification (W3C, 2018).

## 4. App Configuration (YAML)

> **Toyota Principle (Poka-yoke):** Mistake Proofing. Strict schema validation at compile/load time prevents invalid configurations from reaching runtime.
> **Foundation:** Validates Satyanarayan et al. (2017) demonstrating that high-level declarative specifications (like Vega-Lite) reduce implementation errors vs imperative code.

### 4.1 Manifest Schema

```yaml
# app.yaml - Presentar application manifest
presentar: "0.1"
name: "fraud-detection-dashboard"
version: "1.0.0"
description: "Real-time fraud detection monitoring"

# Quality metadata (auto-computed)
score:
  grade: "A"      # F-A scale
  value: 92.3     # 0-100 TDG
  coverage: 94.1  # Test coverage %

# Data sources (Alimentar .ald files)
data:
  transactions:
    source: "pacha://datasets/transactions:latest"
    format: "ald"
    refresh: "5m"

  predictions:
    source: "./predictions.ald"
    format: "ald"

# Model references (Aprender .apr files)
models:
  fraud_detector:
    source: "pacha://models/fraud-detector:1.2.0"
    format: "apr"

# Layout definition (12-column responsive grid, Bootstrap-inspired)
layout:
  type: "dashboard"
  columns: 12              # 12-column grid; span: [start, end] where 1-12
  rows: auto
  gap: 16

  sections:
    - id: "header"
      span: [1, 12]
      widgets:
        - type: "text"
          content: "Fraud Detection Dashboard"
          style: "heading-1"
        - type: "model-card"
          model: "fraud_detector"

    - id: "metrics"
      span: [1, 4]
      widgets:
        - type: "metric"
          label: "Transactions/sec"
          value: "{{ data.transactions | count | rate(1m) }}"
          format: "number"

        - type: "metric"
          label: "Fraud Rate"
          value: "{{ data.predictions | filter(fraud=true) | percentage }}"
          format: "percent"
          threshold:
            warning: 0.05
            critical: 0.10
            # Colors mapped to pre-attentive processing channels (Ware, 2012)

    - id: "main-chart"
      span: [5, 12]
      widgets:
        - type: "chart"
          chart_type: "line"
          data: "{{ data.transactions }}"
          x: "timestamp"
          y: "amount"
          color: "{{ data.predictions.fraud }}"

    - id: "table"
      span: [1, 12]
      widgets:
        - type: "data-table"
          data: "{{ data.transactions | join(data.predictions, on='id') }}"
          columns:
            - field: "id"
              label: "Transaction ID"
            - field: "amount"
              label: "Amount"
              format: "currency"
            - field: "fraud"
              label: "Fraud Score"
              format: "percent"
              conditional:
                - condition: "> 0.8"
                  style: "danger"
                - condition: "> 0.5"
                  style: "warning"
          pagination: 50
          sortable: true
          filterable: true

# Interactivity
interactions:
  - trigger: "table.row.click"
    action: "navigate"
    target: "/transaction/{{ row.id }}"

  - trigger: "chart.point.hover"
    action: "tooltip"
    content: "Amount: {{ point.amount }}"

# Theme (Trueno-Viz color primitives)
theme:
  preset: "dark"
  colors:
    primary: "#6366f1"
    danger: "#ef4444"
    warning: "#f59e0b"
    success: "#10b981"
```

### 4.2 Expression Language

Minimal expression syntax for data binding. **All transforms execute client-side in WASM**—no server round-trips. For large datasets, pre-aggregate server-side or use Alimentar's streaming.

```
{{ source | transform | transform }}

Transforms (all execute in-browser via Trueno):
- filter(field=value)     - Filter rows
- select(field1, field2)  - Select columns
- sort(field, desc=true)  - Sort rows
- limit(n)                - Take first n
- count                   - Row count
- sum(field)              - Sum column
- mean(field)             - Average column
- rate(window)            - Rate over sliding window (client-side, requires timestamp column)
- percentage              - As percentage
- join(other, on=field)   - Join datasets
```

### 4.3 Data/Model Cards

```yaml
# Embedded in app.yaml or standalone .card.yaml
model_card:
  name: "fraud-detector"
  version: "1.2.0"

  # Mitchell et al. (2019) Model Cards standard
  description: "XGBoost fraud detection model"

  metrics:
    auc: 0.94
    precision: 0.89
    recall: 0.91
    f1: 0.90

  training:
    dataset: "pacha://datasets/transactions-2024:1.0.0"
    recipe: "pacha://recipes/fraud-training:1.0.0"
    date: "2024-11-15"
    duration_hours: 4.2

  intended_use:
    primary: ["Real-time fraud scoring", "Batch fraud analysis"]
    out_of_scope: ["Credit scoring", "Identity verification"]

  limitations:
    - "Trained on US transaction data only"
    - "May underperform on transactions < $10"

  ethical_considerations:
    - "Monitor for demographic bias in false positive rates"
    - "Human review required for high-value blocks"

  # Structured tags for automated auditing (taxonomy-based)
  ethical_tags:
    - "bias:demographic"
    - "safety:human_in_loop"
    - "fairness:disparate_impact_tested"

  lineage:
    parent: "pacha://models/fraud-detector:1.1.0"
    type: "fine_tuned"

data_card:
  name: "transactions-2024"
  version: "1.0.0"

  # Gebru et al. (2021) Datasheets standard
  purpose: "Training data for fraud detection models"

  composition:
    rows: 10_000_000
    features: 42
    time_range: ["2024-01-01", "2024-10-31"]

  collection:
    method: "Production transaction logs"
    sampling: "All transactions > $1"

  preprocessing:
    - "PII removed via k-anonymity (k=5)"
    - "Timestamps normalized to UTC"
    - "Currency converted to USD"

  sensitive_features:
    - "merchant_category"
    - "zip_code"

  license: "Internal use only"

  quality:
    completeness: 0.99
    duplicates: 0.001
    outliers: 0.02
```

## 5. Quality Scoring System

> **Toyota Principle (Visual Control):** Making problems visible. The "App Quality Score" acts as an Andon board, immediately highlighting metrics that deviate from the standard.

### 5.1 App Quality Score (0-100, F-A)

Every Presentar app receives a quality grade based on six orthogonal metrics:

```rust
pub struct AppQualityScore {
    pub overall: f64,           // 0-100
    pub grade: Grade,           // F, D, C-, C, C+, B-, B, B+, A-, A, A+

    pub breakdown: ScoreBreakdown,
}

pub struct ScoreBreakdown {
    // Structural (25 points)
    pub widget_complexity: f64,      // Cyclomatic complexity (McCabe, 1976)
    pub layout_depth: f64,           // Nesting depth penalty
    pub component_count: f64,        // Widget count vs viewport

    // Performance (20 points)
    pub render_time_p95: f64,        // 95th percentile frame time
    pub memory_usage: f64,           // Peak memory vs baseline
    pub bundle_size: f64,            // WASM binary size

    // Accessibility (20 points)
    pub wcag_aa_compliance: f64,     // WCAG 2.1 AA checklist (Caldwell et al., 2008)
    pub keyboard_navigation: f64,   // Full keyboard support
    pub screen_reader: f64,          // ARIA labels coverage

    // Data Quality (15 points)
    pub data_completeness: f64,      // Missing value ratio
    pub data_freshness: f64,         // Staleness penalty
    pub schema_validation: f64,      // Type errors

    // Documentation (10 points)
    pub manifest_completeness: f64,  // Required fields coverage
    pub card_coverage: f64,          // Model/data cards present

    // Consistency (10 points)
    pub theme_adherence: f64,        // Design system compliance
    pub naming_conventions: f64,     // ID/class naming
}
// Validation: Relative code churn and complexity measures predict system defect density (Nagappan & Ball, 2005).
```

### 5.2 Grade Thresholds

| Grade | Score Range | Status |
|-------|-------------|--------|
| A+ | 95-100 | Production Excellence |
| A | 90-94 | Production Ready |
| A- | 85-89 | Release Candidate |
| B+ | 80-84 | Beta Quality |
| B | 75-79 | Alpha Quality |
| B- | 70-74 | Development |
| C+ | 65-69 | Prototype |
| C | 60-64 | Draft |
| C- | 55-59 | Sketch |
| D | 50-54 | Incomplete |
| F | 0-49 | Failing |

### 5.3 Quality Gates

```toml
# .presentar-gates.toml
[gates]
min_grade = "B+"
min_score = 80.0

[performance]
max_render_time_ms = 16      # 60fps
max_bundle_size_kb = 500
max_memory_mb = 100

[accessibility]
wcag_level = "AA"
min_contrast_ratio = 4.5
require_keyboard_nav = true
require_aria_labels = true

[data]
max_staleness_minutes = 60
require_schema_validation = true

[documentation]
require_model_cards = true
require_data_cards = true
min_manifest_fields = ["name", "version", "description"]
```

## 6. Build System (Makefile)

### 6.1 Target Overview

```makefile
.PHONY: all build dev test lint fmt coverage score deploy clean

# Default target
all: fmt lint test build score

# Development server with hot reload
dev:
	@echo "Starting Presentar dev server..."
	@cargo watch -x "build --target wasm32-unknown-unknown" \
		-s "wasm-bindgen target/wasm32-unknown-unknown/debug/$(APP).wasm --out-dir pkg --target web" \
		-s "python3 -m http.server 8080 -d pkg"

# Production WASM build
build:
	@echo "Building WASM bundle..."
	@cargo build --target wasm32-unknown-unknown --release
	@wasm-bindgen target/wasm32-unknown-unknown/release/$(APP).wasm \
		--out-dir pkg --target web
	@wasm-opt -O3 -o pkg/$(APP)_bg_opt.wasm pkg/$(APP)_bg.wasm
	@echo "Bundle size: $$(du -h pkg/$(APP)_bg_opt.wasm | cut -f1)"

# Run all tests
test: test-unit test-integration test-visual

test-unit:
	@echo "Running unit tests..."
	@cargo nextest run --lib

test-integration:
	@echo "Running integration tests..."
	@cargo nextest run --test '*'

test-visual:
	@echo "Running visual regression tests..."
	@cargo test --features visual-regression

# Linting
lint: lint-rust lint-yaml lint-a11y

lint-rust:
	@cargo clippy --target wasm32-unknown-unknown -- -D warnings

lint-yaml:
	@yamllint app.yaml

lint-a11y:
	@presentar check-a11y app.yaml --level AA

# Format
fmt:
	@cargo fmt --check
	@prettier --check "**/*.yaml"

# Coverage
coverage:
	@cargo llvm-cov --target wasm32-unknown-unknown --html

# Quality score
score:
	@echo "Computing app quality score..."
	@presentar score app.yaml --output score.json
	@presentar badge score.json --output badge.svg

# Deploy to Pacha
deploy: build test score
	@echo "Deploying to Pacha..."
	@pacha publish app.yaml --type app
	@echo "Published: pacha://apps/$(APP):$(VERSION)"

# Clean
clean:
	@cargo clean
	@rm -rf pkg/ score.json badge.svg
```

### 6.2 Three-Tier Quality Pipeline

> **Toyota Principle (Jidoka):** Automation with a human touch. We implement "Jidoka" by automatically stopping the build (line) when quality standards are not met (Ohno, 1988).
> **Toyota Principle (Kaizen):** Continuous Improvement. This tiered structure allows for rapid feedback loops (Tier 1) while ensuring deep quality verification (Tier 3) (Womack et al., 1990).
> **Foundation:** Validates the effectiveness of automated Continuous Integration in reducing integration risks (Duvall et al., 2007).

```makefile
# TIER 1: On-save (<1 second)
tier1:
	@cargo check --target wasm32-unknown-unknown
	@yamllint app.yaml 2>/dev/null || true

# TIER 2: Pre-commit (1-5 minutes)
tier2: fmt lint test-unit test-integration score
	@presentar gate-check .presentar-gates.toml

# TIER 3: Nightly (hours)
tier3: tier2 test-visual coverage
	@cargo mutants --timeout 300
	@presentar benchmark app.yaml --output bench.json
```

### 6.3 CI/CD Integration

```yaml
# .github/workflows/presentar.yml
name: Presentar CI

on: [push, pull_request]

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install Rust
        uses: dtolnay/rust-action@stable
        with:
          targets: wasm32-unknown-unknown

      - name: Install tools
        run: |
          cargo install wasm-bindgen-cli wasm-opt
          cargo install cargo-nextest cargo-llvm-cov

      - name: Tier 2 Quality
        run: make tier2

      - name: Upload Score
        uses: actions/upload-artifact@v4
        with:
          name: quality-score
          path: score.json
```

### 6.4 Deployment (1.0)

> **Toyota Principle (Genchi Genbutsu):** Go and see. The simplest deployment is running locally and observing the app directly. No cloud abstractions between you and your application.

#### 6.4.1 Two Modes Only

| Mode | Command | Output |
|------|---------|--------|
| **Serve** | `presentar --serve ./app/` | HTTP server on :8080 |
| **Bundle** | `presentar --bundle ./app/ -o app.wasm` | Self-contained WASM |

```bash
# Development: serve with hot reload
presentar --serve ./fraud-detector/ --watch

# Production: bundle and run anywhere
presentar --bundle ./fraud-detector/ -o fraud-detector.wasm

# Run the bundle (pick your runtime)
wasmtime --serve :8080 fraud-detector.wasm
```

#### 6.4.2 Auto-Display Rules

When no `app.yaml` is present, Presentar auto-generates a UI:

| Directory Contents | Generated UI |
|--------------------|--------------|
| `app.yaml` present | Custom layout from YAML |
| Single `.apr` file | ModelCard + inference panel |
| Single `.ald` file | DataCard + DataTable |
| Multiple `.apr`/`.ald` | Split-view grid |
| Mixed files | Model panel (left) + Data panel (right) |

```rust
/// Auto-display resolution order
pub fn resolve_layout(dir: &Path) -> Layout {
    if dir.join("app.yaml").exists() {
        return Layout::from_yaml(dir.join("app.yaml"));
    }

    let models: Vec<_> = glob(dir, "*.apr").collect();
    let datasets: Vec<_> = glob(dir, "*.ald").collect();

    match (models.len(), datasets.len()) {
        (0, 0) => Layout::Empty,
        (1, 0) => Layout::SingleModel(models[0]),
        (0, 1) => Layout::SingleDataset(datasets[0]),
        (1, 1) => Layout::SplitView { left: models[0], right: datasets[0] },
        _ => Layout::Grid { models, datasets },
    }
}
```

#### 6.4.3 Bundle Contents

```
app.wasm (self-contained, ~300KB base)
├── trueno-viz runtime       (150KB)
├── presentar widgets        (100KB)
├── app.yaml (embedded)      (2KB)
├── model.apr schema         (5KB)   ← schema only, not weights
└── data.ald schema          (3KB)   ← schema only, not rows
```

Large assets loaded at runtime via fetch from `file://`, `pacha://`, or `https://`.

#### 6.4.4 What 1.0 Defers

| 1.0 Has | Deferred to 2.0 |
|---------|-----------------|
| Local serve | CDN distribution |
| Single binary | Preview environments |
| `file://` + `pacha://` | Cloud storage backends |
| Manual versioning | Automatic rollbacks |

**Rationale:** Sovereign-first. Users own their deployment.

### 6.5 Presentar Test Harness (presentar-test)

> **CRITICAL DESIGN CONSTRAINT:** Zero external dependencies. No playwright. No selenium. No puppeteer. No npm. No C bindings. Pure Rust + WASM only.

This is **non-negotiable**. External browser automation tools introduce:
- Security vulnerabilities (C/C++ codebases)
- Non-deterministic behavior
- Platform-specific failures
- Dependency hell
- License contamination

**presentar-test** is a first-party testing framework built on Trueno primitives, shipping with Presentar 0.1.

#### 6.5.1 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  presentar-test (Pure Rust, ~50KB)                              │
├─────────────────────────────────────────────────────────────────┤
│  TestRunner                                                     │
│  ├── Discovers #[presentar_test] functions                      │
│  ├── Spawns isolated App instances                              │
│  └── Collects results, generates reports                        │
├─────────────────────────────────────────────────────────────────┤
│  Harness                                                        │
│  ├── Event simulation (click, type, scroll, keyboard)           │
│  ├── Widget queries (CSS-like selectors)                        │
│  ├── State inspection                                           │
│  └── Async waiting (for animations, data loading)               │
├─────────────────────────────────────────────────────────────────┤
│  Framebuffer                                                    │
│  ├── Software rasterizer (Trueno SIMD)                          │
│  ├── Pixel capture for visual regression                        │
│  ├── **Fixed DPI (1.0) and font AA mode for determinism**       │
│  └── PNG encode/decode (pure Rust)                              │
├─────────────────────────────────────────────────────────────────┤
│  A11yChecker                                                    │
│  ├── WCAG 2.1 AA rule engine                                    │
│  ├── Contrast ratio (Trueno color math)                         │
│  └── Focus order, ARIA validation                               │
└─────────────────────────────────────────────────────────────────┘
```

#### 6.5.2 Core Harness API

```rust
pub struct Harness {
    app: App<TestState>,
    framebuffer: Framebuffer,
    event_queue: VecDeque<Event>,
}

impl Harness {
    /// Load from fixture bytes (compile-time embedded)
    pub fn new(fixture: &[u8]) -> Self;

    // === Event Simulation ===
    pub fn click(&mut self, selector: &str) -> &mut Self;
    pub fn type_text(&mut self, selector: &str, text: &str) -> &mut Self;
    pub fn press_key(&mut self, key: Key) -> &mut Self;
    pub fn scroll(&mut self, selector: &str, delta: f32) -> &mut Self;

    // === Queries ===
    pub fn query(&self, selector: &str) -> Option<WidgetRef>;
    pub fn query_all(&self, selector: &str) -> Vec<WidgetRef>;
    pub fn text(&self, selector: &str) -> String;
    pub fn exists(&self, selector: &str) -> bool;

    // === Assertions ===
    pub fn assert_exists(&self, selector: &str) -> &Self;
    pub fn assert_text(&self, selector: &str, expected: &str) -> &Self;
    pub fn assert_text_contains(&self, selector: &str, substring: &str) -> &Self;
    pub fn assert_count(&self, selector: &str, expected: usize) -> &Self;

    // === Async ===
    pub async fn wait_for(&mut self, selector: &str) -> Result<(), Timeout>;
    pub fn tick(&mut self, ms: u64);

    // === Visual ===
    pub fn render(&mut self) -> &Framebuffer;
    pub fn screenshot(&mut self, selector: &str) -> Image;
}
```

#### 6.5.3 Selector Syntax

Minimal CSS-like selectors (hand-written parser, no regex crate):

```
"Button"                    // By widget type
"#submit-btn"               // By ID
".primary"                  // By class
"[data-testid='login']"     // By test ID (recommended)
"Row > Button"              // Child combinator
"Container Button"          // Descendant
```

#### 6.5.4 Visual Regression (Pure Rust)

**Determinism Guarantees:** To ensure pixel-perfect reproducibility across platforms:
- Fixed DPI: `1.0` (no system scaling)
- Font antialiasing: Grayscale only (no subpixel/ClearType)
- Fixed viewport: `1280x720` default
- No system fonts: Embedded test font (Inter, Apache 2.0)

```rust
pub struct Snapshot;

impl Snapshot {
    /// Compare against baseline, panic on diff > threshold
    pub fn assert_match(name: &str, actual: &Image, threshold: f64) {
        let baseline = Self::load_baseline(name);
        let diff_ratio = Self::diff(&baseline, actual);

        if diff_ratio > threshold {
            Self::save_actual(name, actual);
            Self::save_diff(name, &baseline, actual);
            panic!("Visual regression '{}': {:.2}% diff", name, diff_ratio * 100.0);
        }
    }

    /// Pixel diff using Trueno SIMD
    fn diff(a: &Image, b: &Image) -> f64 {
        let changed = trueno::simd::count_diff_u8(a.as_bytes(), b.as_bytes());
        changed as f64 / a.as_bytes().len() as f64
    }
}
```

#### 6.5.5 Accessibility Checker (Built-in)

```rust
pub struct A11yChecker;

impl A11yChecker {
    pub fn check(app: &App) -> A11yReport {
        let mut violations = vec![];

        for widget in app.walk_tree() {
            // 1.4.3 Contrast (Minimum) - WCAG AA requires 4.5:1
            if let Some(style) = widget.text_style() {
                let ratio = style.color.contrast_ratio(&widget.background());
                if ratio < 4.5 {
                    violations.push(violation("color-contrast", widget,
                        format!("Contrast {:.1}:1 < 4.5:1", ratio)));
                }
            }

            // 2.1.1 Keyboard
            if widget.is_interactive() && !widget.is_focusable() {
                violations.push(violation("keyboard", widget,
                    "Interactive element not focusable"));
            }

            // 4.1.2 Name, Role, Value
            if widget.is_interactive() && widget.accessible_name().is_none() {
                violations.push(violation("aria-label", widget,
                    "Missing accessible name"));
            }
        }

        A11yReport { violations }
    }
}
```

#### 6.5.6 Example Tests

```rust
use presentar_test::*;

#[presentar_test]
fn app_renders() {
    let h = Harness::new(include_bytes!("fixtures/app.tar"));
    h.assert_exists("[data-testid='app-root']");
    h.assert_exists("[data-testid='model-card']");
}

#[presentar_test]
fn auto_display_model() {
    let h = Harness::new(include_bytes!("fixtures/model.apr"));
    h.assert_exists("[data-testid='model-card']");
    h.assert_exists("[data-testid='inference-panel']");
}

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

#[presentar_test]
fn accessibility() {
    let h = Harness::new(include_bytes!("fixtures/app.tar"));
    A11yChecker::check(&h.app).assert_pass();
}

#[presentar_test]
fn performance_under_16ms() {
    let mut h = Harness::new(include_bytes!("fixtures/app.tar"));
    let start = std::time::Instant::now();
    h.render();
    assert!(start.elapsed().as_millis() < 16);
}
```

#### 6.5.7 Makefile Targets

```makefile
test: test-unit test-integration test-e2e

test-e2e:
	@echo "Running E2E tests..."
	@cargo test --test '*' --features presentar-test

test-e2e-visual:
	@cargo test --test '*' --features presentar-test,visual

test-e2e-a11y:
	@cargo test --test '*' --features presentar-test,a11y

snapshot-update:
	@SNAPSHOT_UPDATE=1 cargo test --test '*' --features presentar-test,visual
```

#### 6.5.8 What We Build (Not Import)

| Component | Lines | Dependencies |
|-----------|-------|--------------|
| Selector parser | ~100 | None |
| PNG encode/decode | ~300 | None (Trueno) |
| Pixel diff | ~50 | Trueno SIMD |
| WCAG contrast | ~30 | Trueno color |
| Event simulation | ~200 | None |
| **Total** | **~700** | **Zero external** |

#### 6.5.9 Why Not Alternatives?

| Tool | Problem |
|------|---------|
| playwright | npm + Chrome DevTools Protocol + 200MB |
| selenium | Java + WebDriver + non-deterministic |
| wasm-bindgen-test | Requires browser install, no event sim |
| cypress | npm + Electron + 500MB |

**presentar-test** runs entirely in Rust, uses Trueno's software rasterizer, produces deterministic results.

#### 6.5.10 Roadmap

| Version | Features |
|---------|----------|
| **0.1** | Harness, selectors, assertions, snapshots, a11y |
| **0.2** | Async waiting, animation testing, perf metrics |
| **0.3** | Coverage integration, mutation hooks |
| **1.0** | Parallel execution, distributed fixtures |

## 7. Pacha Integration

### 7.1 Content Registry

```rust
/// Pacha content types supported by Presentar
pub enum PachaContent {
    Model(ModelRef),      // .apr files
    Dataset(DatasetRef),  // .ald files
    Recipe(RecipeRef),    // Training recipes
    App(AppRef),          // Presentar apps
}

/// Load content from Pacha registry
pub async fn load_from_pacha(uri: &str) -> Result<PachaContent> {
    // pacha://models/fraud-detector:1.2.0
    // pacha://datasets/transactions:latest
    // pacha://apps/dashboard:1.0.0

    let parsed = PachaUri::parse(uri)?;
    let registry = PachaRegistry::connect().await?;

    match parsed.content_type {
        "models" => {
            let model = registry.get_model(&parsed.name, &parsed.version).await?;
            Ok(PachaContent::Model(model))
        }
        "datasets" => {
            let dataset = registry.get_dataset(&parsed.name, &parsed.version).await?;
            Ok(PachaContent::Dataset(dataset))
        }
        // ...
    }
}
```

### 7.2 Lineage Tracking

```rust
/// Track app dependencies in Pacha lineage graph
pub struct AppLineage {
    pub app_id: AppId,
    pub models: Vec<ModelRef>,
    pub datasets: Vec<DatasetRef>,
    pub parent_app: Option<AppRef>,
}

impl AppLineage {
    /// Register app in Pacha with full provenance (Moreau et al., 2013)
    pub async fn register(&self, registry: &PachaRegistry) -> Result<()> {
        // W3C PROV-DM compliant lineage
        for model in &self.models {
            registry.add_lineage_edge(
                LineageEdge::Used {
                    entity: self.app_id.into(),
                    activity: model.into(),
                }
            ).await?;
        }
        // ...
    }
}
```

## 8. Ruchy Script Integration (Future)

### 8.1 Embedded Scripting

```yaml
# app.yaml with Ruchy scripts
scripts:
  on_load: |
    // Ruchy script executed on app load
    let data = load_dataset("transactions")
    let filtered = data.filter(|row| row.amount > 100)
    set_state("filtered_data", filtered)

  on_refresh: |
    // Periodic refresh script
    let fresh = fetch_dataset("transactions", refresh=true)
    if fresh.count() != state.data.count() {
      notify("New transactions available")
      set_state("data", fresh)
    }

  custom_transform: |
    // User-defined data transform
    fun enrich_with_risk(row) {
      let score = models.fraud_detector.predict(row)
      row.with("risk_score", score)
    }
```

### 8.2 Resource Limits (Security)

Scripts execute in a sandboxed environment with hard limits to prevent DoS:

```rust
pub struct ScriptLimits {
    /// Max instructions before forced termination (prevents infinite loops)
    pub max_instructions: u64,       // Default: 1_000_000
    /// Max memory allocation in bytes
    pub max_memory_bytes: usize,     // Default: 16MB
    /// Max execution time before yield (cooperative)
    pub max_slice_ms: u64,           // Default: 10ms, then yield to event loop
}

impl Default for ScriptLimits {
    fn default() -> Self {
        Self {
            max_instructions: 1_000_000,
            max_memory_bytes: 16 * 1024 * 1024,
            max_slice_ms: 10,
        }
    }
}
```

**Enforcement:** Ruchy VM checks instruction count at loop back-edges and function calls. Exceeding limits raises `ScriptError::ResourceExhausted`, which the app can handle gracefully (e.g., show error toast, abort operation).

### 8.3 Reactive Bindings

```rust
/// Ruchy runtime integration
pub struct RuchyRuntime {
    env: Arc<Mutex<Environment>>,
    state: Arc<RwLock<AppState>>,
    limits: ScriptLimits,
}

impl RuchyRuntime {
    /// Execute script with state access
    pub async fn eval(&self, script: &str) -> Result<Value> {
        let mut env = self.env.lock().await;

        // Inject state bindings
        env.bind("state", self.state.read().await.to_value());
        env.bind("models", self.model_bindings());
        env.bind("data", self.data_bindings());

        let result = ruchy::eval(script, &mut env)?;

        // Apply state mutations
        if let Some(mutations) = env.get_mutations() {
            self.state.write().await.apply(mutations);
        }

        Ok(result)
    }
}
```

## 9. WASM Constraints & Optimizations

### 9.1 Pure WASM Requirements

```rust
// NO std::fs - all I/O via fetch
// NO std::thread - use wasm-bindgen-rayon for parallelism
// NO std::time::Instant - use web_sys::Performance

#[cfg(target_arch = "wasm32")]
pub fn now() -> f64 {
    web_sys::window()
        .expect("no window")
        .performance()
        .expect("no performance")
        .now()
}

// Memory limits
const MAX_WASM_MEMORY: usize = 4 * 1024 * 1024 * 1024; // 4GB
const RECOMMENDED_MEMORY: usize = 256 * 1024 * 1024;   // 256MB
```

### 9.2 Bundle Size Budget

| Component | Budget | Actual |
|-----------|--------|--------|
| Trueno-Viz core | 100KB | - |
| Presentar widgets | 150KB | - |
| YAML parser | 50KB | - |
| Expression engine | 30KB | - |
| Ruchy runtime | 100KB | - |
| **Total** | **<500KB** | - |

### 9.3 Performance Optimizations

```rust
/// Lazy widget initialization
pub struct LazyWidget<W: Widget> {
    factory: Box<dyn Fn() -> W>,
    instance: OnceCell<W>,
}

/// Virtualized list for large datasets
pub struct VirtualList {
    item_height: f32,
    visible_range: Range<usize>,
    recycled_widgets: Vec<Box<dyn Widget>>,
}

/// GPU texture atlas for icons/images
pub struct TextureAtlas {
    texture: GpuTexture,
    regions: HashMap<String, Rect>,
}
```

*Validation: Static analysis and optimization passes (e.g., `wasm-opt`) are essential for mitigating the performance gap between WASM and native code (Jangda et al., 2019).*

## 10. Academic Foundations

### 10.1 Core References (Peer-Reviewed)

1. **Czaplicki, E. (2012).** "Elm: Concurrent FRP for Functional GUIs." *Senior Thesis, Harvard University.*
   **Validates:** Unidirectional data flow architecture (Section 1.3).

2. **Meyerovich, L. A., & Bodik, R. (2010).** "Fast and Parallel Webpage Layout." *Proceedings of the 19th International Conference on World Wide Web (WWW)*, 711-720.
   **Validates:** Parallel layout algorithms with SIMD (Section 3.3).

3. **Haas, A., et al. (2017).** "Bringing the Web up to Speed with WebAssembly." *ACM SIGPLAN Conference on Programming Language Design and Implementation (PLDI)*, 185-200.
   **Validates:** WASM performance model (Section 9).

4. **Mitchell, M., et al. (2019).** "Model Cards for Model Reporting." *Proceedings of the Conference on Fairness, Accountability, and Transparency (FAT*)*, 220-229.
   **Validates:** Model card schema (Section 4.3).

5. **Gebru, T., et al. (2021).** "Datasheets for Datasets." *Communications of the ACM*, 64(12), 86-92.
   **Validates:** Datasheet schema (Section 4.3).

6. **Caldwell, B., et al. (2008).** "Web Content Accessibility Guidelines (WCAG) 2.0." *W3C Recommendation.*
   **Validates:** Accessibility scoring (Section 5.1).

7. **Nagappan, N., & Ball, T. (2005).** "Use of Relative Code Churn Measures to Predict System Defect Density." *Proceedings of the 27th International Conference on Software Engineering (ICSE)*, 284-292.
   **Validates:** Churn-based quality metrics (Section 5.1).

8. **McCabe, T. J. (1976).** "A Complexity Measure." *IEEE Transactions on Software Engineering*, SE-2(4), 308-320.
   **Validates:** Cyclomatic complexity in widget scoring (Section 5.1).

9. **Ohno, T. (1988).** "Toyota Production System: Beyond Large-Scale Production." *Productivity Press.*
   **Validates:** Jidoka (stop-on-error) in quality gates (Section 6.2).

10. **Gregg, C., & Hazelwood, K. (2011).** "Where is the Data? Why You Cannot Debate CPU vs. GPU Performance Without the Answer." *ISPASS*, 134-144.
    **Validates:** GPU dispatch decisions for rendering (Section 2.4).

11. **Wilkinson, L. (2005).** "The Grammar of Graphics." *Springer-Verlag New York.*
    **Validates:** Compositional chart primitives (Section 2.3).

12. **Stonebraker, M., et al. (2005).** "C-Store: A Column-oriented DBMS." *Proceedings of the 31st VLDB Conference*, 553-564.
    **Validates:** Columnar memory layout for tensor operations (Section 1.2).

13. **Satyanarayan, A., et al. (2017).** "Vega-Lite: A Grammar of Interactive Graphics." *IEEE Transactions on Visualization and Computer Graphics*, 23(1).
    **Validates:** Declarative specification benefits (Section 4).

14. **Liker, J. K. (2004).** "The Toyota Way: 14 Management Principles from the World's Greatest Manufacturer." *McGraw-Hill.*
    **Validates:** Standardization and continuous improvement (Executive Summary).

15. **Gamma, E., et al. (1994).** "Design Patterns: Elements of Reusable Object-Oriented Software." *Addison-Wesley.*
    **Validates:** Composite pattern in widget systems (Section 3.1).

16. **Maier, I., & Odersky, M. (2010).** "Deprecating the Observer Pattern." *Technical Report, EPFL.*
    **Validates:** Reactive data flow over callbacks (Section 3.2).

17. **Bostock, M., et al. (2011).** "D³: Data-Driven Documents." *IEEE Transactions on Visualization and Computer Graphics*, 17(12).
    **Validates:** Data binding and selection mechanics (Section 1.3).

18. **Ware, C. (2012).** "Information Visualization: Perception for Design." *Morgan Kaufmann.*
    **Validates:** Pre-attentive processing for alerts (Section 4.1).

19. **Moreau, L., et al. (2013).** "PROV-DM: The PROV Data Model." *W3C Recommendation.*
    **Validates:** Lineage tracking standards (Section 7.2).

20. **Jangda, A., et al. (2019).** "Not So Fast: Analyzing the Performance of WebAssembly vs. Native Code." *USENIX ATC*, 107-120.
    **Validates:** Necessity of optimization passes (Section 9.3).

21. **Womack, J. P., et al. (1990).** "The Machine That Changed the World." *Free Press.*
    **Validates:** Kaizen and rapid feedback loops (Section 6.2).

22. **Tufte, E. R. (1983).** "The Visual Display of Quantitative Information." *Graphics Press.*
    **Validates:** Data-Ink Ratio and elimination of chart junk (Executive Summary).

23. **Iverson, K. E. (1962).** "A Programming Language." *Wiley.*
    **Validates:** Foundational array programming concepts for Trueno (Section 1.2).

24. **Levkowitz, H., & Herman, G. T. (1992).** "Color scales for image data." *IEEE Computer Graphics and Applications*, 12(1), 72-80.
    **Validates:** Perceptually uniform color spaces in Trueno-Viz (Section 2.1).

25. **Nystrom, R. (2014).** "Game Programming Patterns." *Genever Benning.*
    **Validates:** Data Locality patterns for rendering performance (Section 1.2).

26. **Elliott, C., & Hudak, P. (1997).** "Functional Reactive Animation." *ACM SIGPLAN International Conference on Functional Programming (ICFP)*, 263-273.
    **Validates:** Theoretical basis for Reactive UI framework (Section 1.3).

27. **Fitts, P. M. (1954).** "The information capacity of the human motor system in controlling the amplitude of movement." *Journal of Experimental Psychology*, 47(6), 381-391.
    **Validates:** Input handling and target sizing (Section 3.1).

28. **Hick, W. E. (1952).** "On the rate of gain of information." *Quarterly Journal of Experimental Psychology*, 4(1), 11-26.
    **Validates:** Minimizing cognitive load in widget grouping (Section 3.1).

29. **Adya, A., et al. (2002).** "Cooperative task management without manual stack management." *USENIX Annual Technical Conference.*
    **Validates:** Async task management strategies (Section 3.2).

30. **Peng, R. D. (2011).** "Reproducible research in computational science." *Science*, 334(6060), 1226-1227.
    **Validates:** Deterministic reproducibility guarantees (Executive Summary).

31. **Duvall, P. M., et al. (2007).** "Continuous Integration: Improving Software Quality and Reducing Risk." *Addison-Wesley.*
    **Validates:** Automated quality pipelines (Section 6.2).

### 10.2 Implementation References

- **React Fiber Architecture** (Facebook, 2017) - Virtual DOM diffing
- **Yoga Layout Engine** (Facebook, 2016) - Flexbox implementation
- **wgpu** (gfx-rs, 2023) - WebGPU abstraction
- **fontdue** (mooman, 2021) - Font rasterization

## 11. Roadmap

### Phase 1: Foundation (Q1 2025)
- [ ] Trueno-Viz core primitives (paths, fills, text)
- [ ] Basic widget system (container, row, column, text, button)
- [ ] YAML manifest parser
- [ ] WASM build pipeline

### Phase 2: Charts & Data (Q2 2025)
- [ ] Chart primitives (line, bar, scatter, heatmap)
- [ ] Alimentar integration (.ald loading)
- [ ] Data table widget with virtualization
- [ ] Expression language

### Phase 3: ML Integration (Q3 2025)
- [ ] Aprender integration (.apr loading)
- [ ] Model/data card widgets
- [ ] Pacha registry integration
- [ ] Quality scoring system

### Phase 4: Scripting (Q4 2025)
- [ ] Ruchy runtime integration
- [ ] Reactive bindings
- [ ] Custom transform support
- [ ] Full Makefile toolchain

## 12. Example Application

```yaml
# examples/mnist-explorer/app.yaml
presentar: "0.1"
name: "mnist-explorer"
version: "1.0.0"
description: "Interactive MNIST digit classifier"

score:
  grade: "A"
  value: 94.2

data:
  mnist:
    source: "alimentar://mnist:latest"
    format: "ald"

models:
  classifier:
    source: "pacha://models/mnist-cnn:1.0.0"
    format: "apr"

layout:
  type: "app"

  sections:
    - id: "header"
      widgets:
        - type: "text"
          content: "MNIST Digit Explorer"
          style: "heading-1"
        - type: "model-card"
          model: "classifier"
          compact: true

    - id: "canvas"
      widgets:
        - type: "drawing-canvas"
          id: "input_canvas"
          size: [280, 280]
          stroke_width: 20

    - id: "prediction"
      widgets:
        - type: "chart"
          chart_type: "bar"
          data: "{{ state.predictions }}"
          x: "digit"
          y: "probability"

    - id: "samples"
      widgets:
        - type: "data-table"
          data: "{{ data.mnist | sample(100) }}"
          columns:
            - field: "image"
              type: "image"
              size: [28, 28]
            - field: "label"
              label: "True Label"

interactions:
  - trigger: "input_canvas.stroke_end"
    action: "predict"
    script: |
      let pixels = canvas.to_tensor(28, 28)
      let probs = models.classifier.predict(pixels)
      set_state("predictions", probs.to_chart_data())

  - trigger: "samples.row.click"
    action: "load_sample"
    script: |
      canvas.load_image(row.image)
```

## 13. License

MIT License - Pragmatic AI Labs

---

**Repository:** https://github.com/paiml/presentar
**Documentation:** https://presentar.paiml.com
**Pacha Registry:** pacha://apps/presentar

```