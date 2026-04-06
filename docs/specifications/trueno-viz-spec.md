# Trueno-Viz: SIMD/GPU/WASM-Accelerated Visualization Library

**Version:** 0.1.0
**Status:** Draft Specification
**Authors:** PAIML Team
**Date:** 2024-11-24

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Design Philosophy](#2-design-philosophy)
3. [Architecture Overview](#3-architecture-overview)
4. [Core Visualization Types](#4-core-visualization-types)
5. [Rendering Pipeline](#5-rendering-pipeline)
6. [SIMD/GPU/WASM Acceleration Strategy](#6-simdgpuwasm-acceleration-strategy)
7. [Text Prompt Interface](#7-text-prompt-interface)
8. [Integration with Trueno Ecosystem](#8-integration-with-trueno-ecosystem)
9. [API Design](#9-api-design)
10. [Performance Targets](#10-performance-targets)
11. [Academic Foundations](#11-academic-foundations)
12. [Implementation Roadmap](#12-implementation-roadmap)

---

## 1. Executive Summary

Trueno-Viz is a pure Rust visualization library designed for data science, machine learning, and deep learning workflows. Built on the trueno core library, it leverages SIMD (SSE2/AVX2/AVX512/NEON), GPU compute, and WebAssembly for hardware-accelerated rendering of statistical and scientific visualizations.

### Key Differentiators

- **Zero JavaScript/HTML**: Pure Rust from data ingestion to pixel output
- **Hardware Acceleration**: Automatic backend dispatch (CPU SIMD → GPU → WASM)
- **Text Prompt Interface**: Natural language visualization specification
- **Ecosystem Integration**: Native interoperability with trueno-db, trueno-graph, aprender, entrenar, realizar, and repartir
- **Declarative Grammar**: Inspired by Grammar of Graphics [1] with Rust type safety

---

## 2. Design Philosophy

### 2.1 Pure Rust Rendering Stack

Traditional visualization libraries (matplotlib, plotly, D3.js) rely on browser/GUI dependencies. Trueno-Viz takes a different approach:

```
┌─────────────────────────────────────────────────────────────┐
│                    Traditional Stack                         │
├─────────────────────────────────────────────────────────────┤
│  Data → Python/JS → Canvas/SVG/WebGL → Browser → Pixels     │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    Trueno-Viz Stack                          │
├─────────────────────────────────────────────────────────────┤
│  Data → Rust → trueno SIMD/GPU → Framebuffer → PNG/SVG/TTY  │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Guiding Principles

1. **Composability**: Visualizations are algebraic compositions of geometric primitives
2. **Determinism**: Same input produces identical output across platforms
3. **Performance**: Sub-millisecond rendering for interactive exploration
4. **Accessibility**: Terminal (ASCII/Unicode), raster (PNG), and vector (SVG) outputs
5. **Type Safety**: Compile-time verification of aesthetic mappings

---

## 3. Architecture Overview

### 3.1 Layer Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     Layer 5: Outputs                          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐ │
│  │ PNG/JPEG │ │   SVG    │ │ Terminal │ │ Raw Framebuffer  │ │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────────┘ │
├──────────────────────────────────────────────────────────────┤
│                     Layer 4: Rendering                        │
│  ┌───────────────────────────────────────────────────────┐   │
│  │              Software Rasterizer (Pure Rust)           │   │
│  │  - Anti-aliased line drawing (Wu's algorithm) [2]      │   │
│  │  - Polygon fill (scanline with SIMD edge testing)      │   │
│  │  - Text rendering (embedded font rasterization)        │   │
│  └───────────────────────────────────────────────────────┘   │
├──────────────────────────────────────────────────────────────┤
│                     Layer 3: Geometry                         │
│  ┌───────────────────────────────────────────────────────┐   │
│  │              Geometric Primitives                       │   │
│  │  Points, Lines, Polygons, Arcs, Beziers, Text Glyphs   │   │
│  └───────────────────────────────────────────────────────┘   │
├──────────────────────────────────────────────────────────────┤
│                     Layer 2: Scales & Transforms              │
│  ┌───────────────────────────────────────────────────────┐   │
│  │  Linear, Log, Sqrt, Time, Ordinal, Color Scales        │   │
│  │  Affine transforms, Projections (geo), Coordinate sys  │   │
│  └───────────────────────────────────────────────────────┘   │
├──────────────────────────────────────────────────────────────┤
│                     Layer 1: Data Abstraction                 │
│  ┌───────────────────────────────────────────────────────┐   │
│  │  trueno::Vector, trueno::Matrix, Arrow RecordBatch     │   │
│  │  trueno-db queries, trueno-graph adjacency, tensors    │   │
│  └───────────────────────────────────────────────────────┘   │
├──────────────────────────────────────────────────────────────┤
│                     Layer 0: Acceleration                     │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐    │
│  │   SSE2   │ │   AVX2   │ │  AVX512  │ │    NEON      │    │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────┘    │
│  ┌──────────┐ ┌──────────────────────────────────────────┐   │
│  │   WASM   │ │              GPU Compute                 │   │
│  └──────────┘ └──────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

### 3.2 Core Modules

```rust
trueno-viz/
├── src/
│   ├── lib.rs                 // Public API surface
│   ├── grammar/               // Grammar of Graphics implementation
│   │   ├── aesthetic.rs       // Aesthetic mappings (x, y, color, size, shape)
│   │   ├── geom.rs            // Geometric objects
│   │   ├── stat.rs            // Statistical transformations
│   │   ├── scale.rs           // Scale functions
│   │   ├── coord.rs           // Coordinate systems
│   │   └── facet.rs           // Faceting/small multiples
│   ├── plots/                 // High-level plot types
│   │   ├── scatter.rs         // Scatter plots
│   │   ├── heatmap.rs         // Heatmaps and density plots
│   │   ├── histogram.rs       // Histograms and distributions
│   │   ├── line.rs            // Line charts and time series
│   │   ├── bar.rs             // Bar charts
│   │   ├── box_plot.rs        // Box and violin plots
│   │   ├── contour.rs         // Contour plots
│   │   ├── parallel.rs        // Parallel coordinates
│   │   ├── radar.rs           // Radar/spider charts
│   │   └── network.rs         // Network/graph visualizations
│   ├── render/                // Rendering backends
│   │   ├── framebuffer.rs     // RGBA pixel buffer
│   │   ├── rasterizer.rs      // SIMD-accelerated rasterization
│   │   ├── text.rs            // Font rendering
│   │   └── antialiasing.rs    // AA algorithms
│   ├── output/                // Output encoders
│   │   ├── png.rs             // PNG encoding (no libpng)
│   │   ├── svg.rs             // SVG generation
│   │   ├── terminal.rs        // ASCII/Unicode/Sixel output
│   │   └── raw.rs             // Raw buffer export
│   ├── accel/                 // Acceleration layer
│   │   ├── dispatch.rs        // Backend selection
│   │   ├── simd_ops.rs        // SIMD kernels
│   │   └── gpu_ops.rs         // GPU compute shaders
│   ├── prompt/                // Text prompt interface
│   │   ├── parser.rs          // Natural language parsing
│   │   ├── intent.rs          // Visualization intent detection
│   │   └── codegen.rs         // Rust code generation
│   └── interop/               // Ecosystem integration
│       ├── trueno_db.rs       // Query result visualization
│       ├── trueno_graph.rs    // Graph layout algorithms
│       ├── aprender.rs        // ML model visualization
│       └── entrenar.rs        // Training metrics plots
```

---

## 4. Core Visualization Types

### 4.1 Statistical Plots

| Plot Type | Use Case | SIMD Acceleration Point |
|-----------|----------|------------------------|
| **Scatter Plot** | Bivariate relationships, clustering | Point binning, alpha blending |
| **Heatmap** | Correlation matrices, 2D histograms | Matrix operations, color mapping |
| **Histogram** | Univariate distributions | Binning with SIMD comparisons |
| **Density Plot** | Kernel density estimation | KDE convolution [3] |
| **Box Plot** | Distribution summaries | Quantile computation |
| **Violin Plot** | Distribution + density | KDE + symmetric reflection |

### 4.2 Relationship Plots

| Plot Type | Use Case | SIMD Acceleration Point |
|-----------|----------|------------------------|
| **Line Chart** | Time series, trends | Polyline simplification [4] |
| **Area Chart** | Cumulative quantities | Polygon triangulation |
| **Contour Plot** | 3D surface projections | Marching squares [5] |
| **Parallel Coordinates** | High-dimensional data | Line intersection culling |
| **Radar Chart** | Multivariate comparison | Polar coordinate transform |

### 4.3 ML/DL Specific Plots

| Plot Type | Use Case | SIMD Acceleration Point |
|-----------|----------|------------------------|
| **Loss Curves** | Training progress | Streaming aggregation |
| **Confusion Matrix** | Classification performance | Matrix rendering |
| **ROC/PR Curves** | Threshold analysis | Sorted array operations |
| **t-SNE/UMAP** | Dimensionality reduction | Force-directed layout [6] |
| **Attention Maps** | Transformer interpretability | Matrix heatmap |
| **Gradient Flow** | Backprop visualization | Layer-wise statistics |
| **Network Architecture** | Model structure | Graph layout |

### 4.4 Graph Visualizations

| Plot Type | Use Case | SIMD Acceleration Point |
|-----------|----------|------------------------|
| **Force-Directed** | General graph layout | N-body simulation [7] |
| **Hierarchical** | Trees, DAGs | Reingold-Tilford layout |
| **Circular** | Cyclic relationships | Arc bundling |
| **Adjacency Matrix** | Dense graphs | Sparse matrix rendering |

---

## 5. Rendering Pipeline

### 5.1 Pipeline Stages

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Data      │───▶│   Scale     │───▶│   Geometry  │───▶│  Rasterize  │
│  Binding    │    │  Transform  │    │  Generation │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
      │                  │                  │                  │
      ▼                  ▼                  ▼                  ▼
  Map columns       Apply log/sqrt    Points → coords    Coords → pixels
  to aesthetics     normalization     Lines → segments   + anti-aliasing
```

### 5.2 SIMD-Accelerated Rasterization

```rust
/// Anti-aliased line drawing using Wu's algorithm with SIMD
pub fn draw_line_aa_simd(
    fb: &mut Framebuffer,
    x0: f32, y0: f32,
    x1: f32, y1: f32,
    color: Rgba,
) {
    // Process 8 pixels simultaneously with AVX2
    dispatch_unary_op!(
        wu_line_kernel,
        &mut fb.pixels,
        (x0, y0, x1, y1, color)
    );
}
```

### 5.3 Framebuffer Design

```rust
/// SIMD-aligned framebuffer for efficient pixel operations
#[repr(C, align(64))]
pub struct Framebuffer {
    /// Width in pixels
    pub width: u32,
    /// Height in pixels
    pub height: u32,
    /// RGBA pixels in row-major order, 64-byte aligned
    pub pixels: Vec<u8>,  // [R, G, B, A, R, G, B, A, ...]
}

impl Framebuffer {
    /// Create with alignment for SIMD operations
    pub fn new(width: u32, height: u32) -> Self {
        let size = (width * height * 4) as usize;
        let mut pixels = Vec::with_capacity(size + 64);
        pixels.resize(size, 0);
        Self { width, height, pixels }
    }

    /// SIMD-accelerated clear to color
    pub fn clear(&mut self, color: Rgba) {
        dispatch_unary_op!(fill_color, &mut self.pixels, color);
    }

    /// SIMD-accelerated alpha blending
    pub fn blend(&mut self, other: &Framebuffer, alpha: f32) {
        dispatch_binary_op!(alpha_blend, &mut self.pixels, &other.pixels, alpha);
    }
}
```

---

## 6. SIMD/GPU/WASM Acceleration Strategy

### 6.1 Operation Classification

Following trueno's dispatch model, operations are classified by arithmetic intensity:

| Operation | AI Ratio | Preferred Backend |
|-----------|----------|-------------------|
| Point plotting | < 1 | AVX2 (memory-bound) |
| Color mapping | 1-2 | AVX2/AVX512 |
| Polygon fill | 2-5 | AVX512 |
| KDE convolution | > 10 | GPU |
| Force-directed layout | > 50 | GPU |
| Matrix heatmap | 5-20 | GPU (large), AVX2 (small) |

### 6.2 Backend Dispatch Macro

```rust
/// Visualization-specific dispatch considering output size
macro_rules! viz_dispatch {
    ($op:ident, $data:expr, $output_size:expr) => {{
        let ai = compute_arithmetic_intensity::<$op>();
        let pixels = $output_size.0 * $output_size.1;

        match (ai, pixels) {
            // Small outputs: always SIMD (GPU launch overhead dominates)
            (_, p) if p < 100_000 => simd::$op($data),

            // Large outputs with high AI: GPU
            (ai, _) if ai > 10.0 => gpu::$op($data),

            // Memory-bound on large data: AVX2 (better cache utilization)
            (ai, _) if ai < 2.0 => avx2::$op($data),

            // Compute-bound: AVX512 if available
            _ => avx512_or_fallback::$op($data),
        }
    }};
}
```

### 6.3 GPU Compute Kernels

For operations exceeding the GPU threshold (based on trueno benchmarks: ~25,000 elements for compute-bound ops):

```rust
/// GPU kernel for kernel density estimation (2D)
#[gpu_kernel]
fn kde_2d_kernel(
    points: &[f32],      // [x0, y0, x1, y1, ...]
    output: &mut [f32],  // density grid
    bandwidth: f32,
    grid_size: (u32, u32),
) {
    let (gx, gy) = global_id();
    let (w, h) = grid_size;

    if gx >= w || gy >= h { return; }

    let x = (gx as f32) / (w as f32);
    let y = (gy as f32) / (h as f32);

    let mut density = 0.0f32;
    for i in 0..(points.len() / 2) {
        let px = points[i * 2];
        let py = points[i * 2 + 1];
        let dx = (x - px) / bandwidth;
        let dy = (y - py) / bandwidth;
        density += (-0.5 * (dx * dx + dy * dy)).exp();
    }

    output[(gy * w + gx) as usize] = density;
}
```

### 6.4 WASM Compilation Strategy

```rust
#[cfg(target_arch = "wasm32")]
mod wasm {
    use wasm_bindgen::prelude::*;

    /// WASM-exported visualization function
    #[wasm_bindgen]
    pub fn render_scatter(
        x_data: &[f32],
        y_data: &[f32],
        width: u32,
        height: u32,
    ) -> Vec<u8> {
        // Uses WASM SIMD128 when available
        let fb = Framebuffer::new(width, height);
        let plot = ScatterPlot::new()
            .x(x_data)
            .y(y_data)
            .build();
        plot.render(&mut fb);
        fb.to_png()
    }
}
```

---

## 7. Text Prompt Interface

### 7.1 Natural Language Specification

Trueno-Viz supports natural language visualization specification, enabling LLM integration and rapid prototyping:

```
User: "Show me a scatter plot of loss vs accuracy colored by epoch"

Parsed Intent:
  - Plot type: scatter
  - X-axis: loss (numeric)
  - Y-axis: accuracy (numeric)
  - Color: epoch (sequential)
  - Output: default (PNG)
```

### 7.2 Prompt Grammar

```ebnf
prompt ::= [show_verb] [article] plot_type [of] mapping+ [style_clause]*

show_verb ::= "show" | "plot" | "visualize" | "create" | "draw"
article ::= "a" | "an" | "the"
plot_type ::= "scatter" | "heatmap" | "histogram" | "line" | "bar" | ...

mapping ::= variable [preposition variable]* [aesthetic_binding]
preposition ::= "vs" | "versus" | "by" | "against" | "over"
aesthetic_binding ::= "colored by" variable
                    | "sized by" variable
                    | "grouped by" variable

style_clause ::= "with" style_option
style_option ::= "log scale" | "dark theme" | "no legend" | ...
```

### 7.3 Intent Detection Model

```rust
/// Lightweight intent classifier using trueno's SIMD embeddings
pub struct VizIntentClassifier {
    embeddings: Matrix<f32>,  // Pre-computed plot type embeddings
    threshold: f32,
}

impl VizIntentClassifier {
    pub fn classify(&self, prompt: &str) -> VizIntent {
        // Tokenize and embed using SIMD-accelerated ops
        let prompt_vec = self.embed_prompt(prompt);

        // Cosine similarity with plot type centroids
        let similarities = dispatch_binary_op!(
            cosine_similarity,
            &self.embeddings,
            &prompt_vec
        );

        // Return highest-scoring intent
        let (idx, score) = argmax_simd(&similarities);
        VizIntent::from_index(idx, score)
    }
}
```

### 7.4 Code Generation

For complex visualizations, the prompt interface generates Rust code:

```rust
// Input prompt: "heatmap of correlation matrix for columns a, b, c, d"

// Generated code:
let data = dataframe.select(&["a", "b", "c", "d"])?;
let corr = data.correlation_matrix()?;  // Uses trueno SIMD

let plot = Heatmap::new()
    .data(&corr)
    .color_scale(ColorScale::Diverging(Palette::RdBu))
    .annotate(true)
    .build();

plot.render_to_file("correlation.png")?;
```

---

## 8. Integration with Trueno Ecosystem

### 8.1 trueno-db Integration

```rust
use trueno_db::{Database, Query};
use trueno_viz::prelude::*;

// Direct query result visualization
let db = Database::open("analytics.db")?;
let results = db.query("
    SELECT date, revenue, region
    FROM sales
    WHERE year = 2024
")?;

// Arrow RecordBatch flows directly to visualization
let plot = LinePlot::new()
    .from_recordbatch(&results)
    .x("date")
    .y("revenue")
    .color("region")
    .build();
```

### 8.2 trueno-graph Integration

```rust
use trueno_graph::{Graph, PageRank};
use trueno_viz::prelude::*;

let graph = Graph::from_edges(&edges)?;

// Compute PageRank with GPU acceleration
let ranks = graph.pagerank(0.85, 100)?;

// Visualize with force-directed layout
let plot = NetworkPlot::new()
    .graph(&graph)
    .node_size(&ranks)  // Size by PageRank
    .layout(Layout::ForceAtlas2 {
        iterations: 1000,
        gravity: 1.0,
        scaling: 10.0,
    })
    .build();
```

### 8.3 aprender Integration

```rust
use aprender::{KMeans, PCA, TSNE};
use trueno_viz::prelude::*;

// Dimensionality reduction + clustering visualization
let pca = PCA::new(2).fit(&high_dim_data)?;
let reduced = pca.transform(&high_dim_data)?;

let kmeans = KMeans::new(5).fit(&reduced)?;
let labels = kmeans.predict(&reduced)?;

let plot = ScatterPlot::new()
    .x(&reduced.column(0))
    .y(&reduced.column(1))
    .color(&labels)
    .title("PCA + K-Means Clustering")
    .build();
```

### 8.4 entrenar Integration

```rust
use entrenar::{Trainer, TrainingCallback};
use trueno_viz::prelude::*;

// Real-time training visualization
struct VizCallback {
    loss_plot: StreamingLinePlot,
}

impl TrainingCallback for VizCallback {
    fn on_epoch_end(&mut self, epoch: usize, metrics: &Metrics) {
        self.loss_plot.push(epoch as f32, metrics.loss);
        self.loss_plot.render_to_terminal();  // Live ASCII plot
    }
}

let trainer = Trainer::new(model, optimizer)
    .callback(VizCallback::new())
    .train(&data, 100)?;
```

---

## 9. API Design

### 9.1 Builder Pattern API

```rust
use trueno_viz::prelude::*;

// Fluent builder API
let plot = ScatterPlot::new()
    .data(&dataset)
    .x("sepal_length")
    .y("sepal_width")
    .color("species")
    .size(5.0)
    .alpha(0.7)
    .title("Iris Dataset")
    .xlabel("Sepal Length (cm)")
    .ylabel("Sepal Width (cm)")
    .legend(Position::TopRight)
    .theme(Theme::Dark)
    .build();

// Multiple output formats
plot.render_to_file("scatter.png")?;
plot.render_to_svg("scatter.svg")?;
plot.render_to_terminal()?;
```

### 9.2 Grammar of Graphics API

```rust
use trueno_viz::grammar::*;

// ggplot2-style layered grammar
let plot = Plot::new(&data)
    .aes(Aes::new().x("x").y("y").color("group"))
    .geom(Geom::Point { size: 3.0, alpha: 0.8 })
    .geom(Geom::Smooth { method: SmoothMethod::Loess, se: true })
    .scale_x(Scale::Log10)
    .scale_color(Scale::Discrete(Palette::Set1))
    .facet(Facet::Wrap { var: "category", ncol: 3 })
    .coord(Coord::Cartesian)
    .theme(Theme::Minimal)
    .build();
```

### 9.3 Functional Composition API

```rust
use trueno_viz::functional::*;

// Point-free functional composition
let pipeline = scatter()
    | map_x(log10)
    | map_color(from_column("label"))
    | add_regression(Linear)
    | set_theme(dark())
    | annotate_outliers(zscore(3.0));

let plot = pipeline.apply(&data);
```

### 9.4 Macro DSL

```rust
use trueno_viz::dsl::*;

// Declarative macro DSL
let plot = viz! {
    scatter {
        data: &iris,
        x: sepal_length,
        y: sepal_width,
        color: species,

        scales {
            x: linear(0.0, 8.0),
            color: categorical(["setosa", "versicolor", "virginica"]),
        }

        theme: dark,
        output: png(800, 600),
    }
};
```

---

## 10. Performance Targets

### 10.1 Rendering Benchmarks

| Operation | Target Latency | Data Size | Backend |
|-----------|---------------|-----------|---------|
| Scatter plot (10K points) | < 5ms | 80 KB | AVX2 |
| Scatter plot (1M points) | < 50ms | 8 MB | GPU |
| Heatmap (1000x1000) | < 10ms | 4 MB | AVX512 |
| KDE density (100K points) | < 100ms | 800 KB | GPU |
| Force-directed (10K nodes) | < 1s | varies | GPU |
| Line chart (1M points) | < 20ms | 4 MB | AVX2 + decimation |

### 10.2 Memory Efficiency

| Component | Memory Budget |
|-----------|--------------|
| Framebuffer (1920x1080 RGBA) | 8.3 MB |
| Point buffer (1M f32 pairs) | 8 MB |
| Color LUT (256 RGBA) | 1 KB |
| Font atlas (ASCII) | 64 KB |
| Total overhead | < 20 MB |

### 10.3 Comparison with Existing Libraries

Based on published benchmarks and trueno's SIMD advantages:

| Library | 100K Scatter | Notes |
|---------|-------------|-------|
| matplotlib | ~500ms | Python + Agg backend |
| plotly | ~200ms | JS + WebGL |
| Vega-Lite | ~150ms | JS + Canvas |
| **trueno-viz (target)** | **< 20ms** | Rust + SIMD/GPU |

---

## 11. Academic Foundations

This specification is grounded in peer-reviewed computer science research:

### [1] Grammar of Graphics
**Wilkinson, L. (2005).** *The Grammar of Graphics*. Springer.
DOI: 10.1007/0-387-28695-0

Foundational framework for declarative visualization specification. Trueno-Viz implements the layered grammar with aesthetic mappings, geometric objects, scales, and coordinate systems as first-class Rust types.

### [2] Anti-Aliased Line Drawing
**Wu, X. (1991).** "An Efficient Antialiasing Technique." *Computer Graphics (SIGGRAPH '91 Proceedings)*, 25(4), 143-152.
DOI: 10.1145/127719.122734

Wu's algorithm achieves sub-pixel accuracy with only integer arithmetic, making it ideal for SIMD vectorization. Our implementation processes 8 pixels per AVX2 instruction.

### [3] Kernel Density Estimation
**Silverman, B. W. (1986).** *Density Estimation for Statistics and Data Analysis*. Chapman and Hall.
ISBN: 978-0412246203

KDE provides non-parametric density estimates essential for distribution visualization. GPU-accelerated convolution enables real-time density plots for million-point datasets.

### [4] Line Simplification
**Douglas, D. H., & Peucker, T. K. (1973).** "Algorithms for the Reduction of the Number of Points Required to Represent a Digitized Line or its Caricature." *Cartographica*, 10(2), 112-122.
DOI: 10.3138/FM57-6770-U75U-7727

Douglas-Peucker algorithm enables efficient rendering of time series with millions of points by reducing to visually equivalent simplified polylines.

### [5] Marching Squares
**Lorensen, W. E., & Cline, H. E. (1987).** "Marching Cubes: A High Resolution 3D Surface Construction Algorithm." *Computer Graphics (SIGGRAPH '87)*, 21(4), 163-169.
DOI: 10.1145/37401.37422

The 2D variant (marching squares) generates contour lines for heatmaps and topographic visualizations with SIMD-parallelizable cell classification.

### [6] Force-Directed Graph Layout
**Fruchterman, T. M. J., & Reingold, E. M. (1991).** "Graph Drawing by Force-Directed Placement." *Software: Practice and Experience*, 21(11), 1129-1164.
DOI: 10.1002/spe.4380211102

Force-directed algorithms model graphs as physical systems with attractive and repulsive forces. GPU acceleration enables real-time layout of graphs with 10K+ nodes.

### [7] Barnes-Hut N-Body Simulation
**Barnes, J., & Hut, P. (1986).** "A Hierarchical O(N log N) Force-Calculation Algorithm." *Nature*, 324(6096), 446-449.
DOI: 10.1038/324446a0

Octree-based approximation reduces force calculation complexity from O(N²) to O(N log N), critical for large-scale graph visualization and t-SNE implementations.

### [8] SIMD-Accelerated Database Operations
**Polychroniou, O., Raghavan, A., & Ross, K. A. (2015).** "Rethinking SIMD Vectorization for In-Memory Databases." *SIGMOD '15*, 1493-1508.
DOI: 10.1145/2723372.2747645

Demonstrates that SIMD vectorization provides 3-10x speedups for analytical workloads. Trueno-viz applies these principles to visualization data pipelines.

### [9] GPU-Accelerated Visualization
**Fang, H., Huang, T., & Zhou, K. (2008).** "Real-Time Continuous Level of Detail Rendering of Point Clouds." *2008 IEEE Symposium on Interactive Ray Tracing*, 103-110.
DOI: 10.1109/RT.2008.4634627

GPU-based point cloud rendering techniques applicable to large-scale scatter plots. Hierarchical LOD enables interactive exploration of massive datasets.

### [10] Perceptually Uniform Color Spaces
**Sharma, G., Wu, W., & Dalal, E. N. (2005).** "The CIEDE2000 Color-Difference Formula: Implementation Notes, Supplementary Test Data, and Mathematical Observations." *Color Research & Application*, 30(1), 21-30.
DOI: 10.1002/col.20070

Perceptually uniform color spaces (CIELAB, CIEDE2000) ensure visual accuracy in heatmaps and continuous color scales. Trueno-viz implements these for scientific accuracy.

---

## 12. Implementation Roadmap

### Phase 1: Core Foundation (v0.1.0)
- [ ] Framebuffer with SIMD clear/blend operations
- [ ] Basic geometric primitives (point, line, rectangle)
- [ ] Wu's anti-aliased line drawing
- [ ] Linear and log scales
- [ ] PNG output encoder
- [ ] Scatter plot implementation
- [ ] Histogram implementation

### Phase 2: Statistical Plots (v0.2.0)
- [ ] Heatmap with color scales
- [ ] Box plot with quartile computation
- [ ] Violin plot with KDE
- [ ] Line chart with Douglas-Peucker simplification
- [ ] Contour plot with marching squares
- [ ] SVG output encoder
- [ ] Terminal output (ASCII/Unicode)

### Phase 3: ML/DL Visualizations (v0.3.0)
- [ ] Confusion matrix
- [ ] ROC/PR curves
- [ ] Loss curves with streaming update
- [ ] Attention map visualization
- [ ] Integration with aprender
- [ ] Integration with entrenar

### Phase 4: Graph Visualizations (v0.4.0)
- [ ] Force-directed layout (Fruchterman-Reingold)
- [ ] Barnes-Hut optimization
- [ ] Hierarchical layouts
- [ ] Integration with trueno-graph
- [ ] Adjacency matrix visualization

### Phase 5: Advanced Features (v0.5.0)
- [ ] Text prompt interface
- [ ] Grammar of Graphics API
- [ ] Faceting/small multiples
- [ ] Interactive mode (for WASM)
- [ ] Sixel terminal output
- [ ] GPU compute kernels

### Phase 6: Production Hardening (v1.0.0)
- [ ] Comprehensive test suite (95%+ coverage)
- [ ] Benchmark suite
- [ ] Documentation and examples
- [ ] WASM package publication
- [ ] crates.io publication

---

## 13. Book Examples with TDD

### 13.1 Example-Driven Development

Following the aprender library pattern, trueno-viz enforces **book-quality examples** that serve as both documentation and integration tests. Each example must:

1. **Be Self-Contained**: Run with `cargo run --example <name>`
2. **Be Educational**: Include clear comments explaining concepts
3. **Be Tested**: Examples are compiled and run in CI
4. **Follow TDD**: Write the example first, then implement features

### 13.2 Example Structure

```
trueno-viz/
├── examples/
│   ├── scatter_basic.rs           # Basic scatter plot
│   ├── scatter_iris.rs            # Iris dataset visualization
│   ├── heatmap_correlation.rs     # Correlation matrix heatmap
│   ├── histogram_distribution.rs  # Distribution analysis
│   ├── line_time_series.rs        # Time series visualization
│   ├── loss_training.rs           # ML training loss curves
│   ├── confusion_matrix_ml.rs     # Classification evaluation
│   ├── roc_pr_curves.rs           # ROC and PR curve analysis
│   ├── terminal_output.rs         # ASCII/Unicode rendering
│   └── complete_workflow.rs       # End-to-end ML visualization
```

### 13.3 Example Template

Each example follows this structure (based on aprender style):

```rust
//! Example Name - Brief Description
//!
//! Demonstrates [specific feature] using [data/scenario].
//!
//! Run with: `cargo run --example example_name`

use trueno_viz::prelude::*;

fn main() {
    println!("Example Name");
    println!("============\n");

    // Step 1: Prepare data
    println!("Step 1: Preparing data...");
    let data = prepare_example_data();

    // Step 2: Create visualization
    println!("Step 2: Creating visualization...");
    let plot = create_visualization(&data);

    // Step 3: Render output
    println!("Step 3: Rendering...");
    render_and_display(&plot);

    // Step 4: Show results/metrics
    println!("\nResults:");
    display_metrics(&plot);
}

fn prepare_example_data() -> ExampleData {
    // Clear, documented data preparation
}

fn create_visualization(data: &ExampleData) -> Plot {
    // Builder pattern with comments
}

fn render_and_display(plot: &Plot) {
    // Multiple output formats demonstrated
}

fn display_metrics(plot: &Plot) {
    // Educational output showing what was visualized
}
```

### 13.4 Example Categories

| Category | Examples | Purpose |
|----------|----------|---------|
| **Basics** | scatter_basic, histogram_basic | Core API introduction |
| **Statistical** | heatmap_correlation, box_violin | Statistical analysis |
| **ML/DL** | loss_training, confusion_matrix, roc_pr | ML workflow integration |
| **Time Series** | line_time_series, streaming_data | Temporal data |
| **Output Formats** | terminal_output, svg_export | Multi-format rendering |
| **Integration** | aprender_workflow, trueno_graph_viz | Ecosystem integration |

### 13.5 CI/CD Example Testing

```yaml
# .github/workflows/examples.yml
name: Examples CI

on: [push, pull_request]

jobs:
  examples:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable

      - name: Build all examples
        run: cargo build --examples

      - name: Run all examples
        run: |
          for example in examples/*.rs; do
            name=$(basename "$example" .rs)
            echo "Running example: $name"
            cargo run --example "$name" || exit 1
          done

      - name: Verify example outputs
        run: |
          # Check that PNG files were generated
          ls -la *.png 2>/dev/null || echo "No PNG outputs (may be expected)"
```

### 13.6 Example Quality Requirements

1. **Compilation**: All examples must compile without warnings
2. **Execution**: All examples must run successfully
3. **Output**: Examples should produce meaningful console output
4. **Documentation**: Each example requires header documentation
5. **Idiomatic Code**: Follow Rust best practices and trueno-viz patterns
6. **Educational Value**: Comments explain the "why" not just the "what"

### 13.7 Adding New Examples

When adding a new feature:

1. **Write the example first** (TDD approach)
2. Example should initially fail to compile
3. Implement the feature to make the example work
4. Add tests for edge cases discovered while writing the example
5. Update this documentation with the new example

---

## Appendix A: Dependency Graph

```
trueno-viz
├── trueno (0.7.x)          # SIMD/GPU acceleration
├── trueno-db (optional)     # Database integration
├── trueno-graph (optional)  # Graph integration
├── aprender (optional)      # ML integration
├── png (pure Rust)          # PNG encoding
├── ttf-parser               # Font parsing
└── (no other external deps)
```

## Appendix B: Feature Flags

```toml
[features]
default = ["simd", "png"]
simd = ["trueno/simd"]
gpu = ["trueno/gpu"]
wasm = ["trueno/wasm"]
db = ["trueno-db"]
graph = ["trueno-graph"]
ml = ["aprender", "entrenar"]
terminal = []
svg = []
full = ["simd", "gpu", "db", "graph", "ml", "terminal", "svg"]
```

## Appendix C: Output Format Comparison

| Format | Use Case | Size (1920x1080) | Generation Time |
|--------|----------|------------------|-----------------|
| PNG | Publication, web | ~500 KB | ~10ms |
| SVG | Scalable, editing | ~50-500 KB | ~5ms |
| Terminal (ASCII) | Quick inspection | N/A | < 1ms |
| Terminal (Sixel) | High-fidelity TTY | ~200 KB | ~20ms |
| Raw RGBA | Embedding, IPC | 8.3 MB | ~1ms |

---

*This specification is a living document. Contributions and feedback are welcome.*
