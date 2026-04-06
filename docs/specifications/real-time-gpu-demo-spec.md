# Real-Time GPU Visualization Demo Specification

**Version**: 1.0.0
**Status**: Draft
**Authors**: PAIML Team
**Created**: 2025-11-25

## Executive Summary

This specification defines a showcase demonstration of trueno-viz's GPU-first visualization capabilities, featuring real-time streaming data visualization with WebSocket connectivity, Grammar of Graphics composition, and automatic compute tier fallback (GPU → SIMD → scalar).

The demo targets 60fps rendering of 1M+ data points with sub-16ms frame times, demonstrating performance characteristics impossible with JavaScript-based visualization libraries.

---

## 1. Academic Foundation

This implementation is grounded in peer-reviewed computer science research:

### 1.1 Visualization Theory

1. **Wilkinson, L. (2005).** *The Grammar of Graphics* (2nd ed.). Springer.
   - Foundation for the declarative visualization API
   - Layered composition model: data → aesthetics → geometry → statistics → coordinates → facets

2. **Wickham, H. (2010).** "A Layered Grammar of Graphics." *Journal of Computational and Graphical Statistics*, 19(1), 3-28.
   - Practical ggplot2 implementation principles
   - Aesthetic mapping and scale abstractions

3. **Satyanarayan, A., Moritz, D., Wongsuphasawat, K., & Heer, J. (2017).** "Vega-Lite: A Grammar of Interactive Graphics." *IEEE Transactions on Visualization and Computer Graphics*, 23(1), 341-350.
   - High-level specification language
   - Interaction primitives for linked views

### 1.2 GPU-Accelerated Rendering

4. **Liu, S., Maljovec, D., Wang, B., Bremer, P.-T., & Pascucci, V. (2017).** "Visualizing High-Dimensional Data: Advances in the Past Decade." *IEEE Transactions on Visualization and Computer Graphics*, 23(3), 1249-1268.
   - GPU parallel coordinate plots
   - Dimensionality reduction visualization

5. **Lins, L., Klosowski, J. T., & Scheidegger, C. (2013).** "Nanocubes for Real-Time Exploration of Spatiotemporal Datasets." *IEEE Transactions on Visualization and Computer Graphics*, 19(12), 2456-2465.
   - GPU-accelerated data cube aggregation
   - Sub-second query response on billions of records

6. **Piringer, H., Tominski, C., Muigg, P., & Berger, W. (2009).** "A Multi-Threading Architecture to Support Interactive Visual Exploration." *IEEE Transactions on Visualization and Computer Graphics*, 15(6), 1113-1120.
   - Parallel rendering pipeline design
   - Progressive refinement strategies

### 1.3 Real-Time Streaming Visualization

7. **Fisher, D. (2011).** "Incremental, Approximate Database Queries and Uncertainty for Exploratory Visualization." *IEEE Symposium on Large Data Analysis and Visualization (LDAV)*, 73-80.
   - Streaming aggregation algorithms
   - Uncertainty visualization for partial results

8. **Battle, L., & Heer, J. (2019).** "Characterizing Exploratory Visual Analysis: A Literature Review and Evaluation of Analytic Provenance in Tableau." *Computer Graphics Forum*, 38(3), 145-159.
   - Interaction latency requirements (<100ms for exploration)
   - Visual analytics workflow patterns

### 1.4 WebGPU and Browser Compute

9. **Nickolls, J., Buck, I., Garland, M., & Skadron, K. (2008).** "Scalable Parallel Programming with CUDA." *ACM Queue*, 6(2), 40-53.
   - GPU compute fundamentals applicable to WebGPU
   - Thread hierarchy and memory model

10. **Cayton, L. (2012).** "Accelerating Nearest Neighbor Search on Manycore Systems." *IEEE International Parallel & Distributed Processing Symposium (IPDPS)*, 402-413.
    - GPU spatial indexing for visualization
    - Brute-force vs. tree-based approaches at scale

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Browser Environment                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐     │
│  │   WebSocket     │    │   Data Buffer   │    │   WebGPU        │     │
│  │   Receiver      │───▶│   Ring Buffer   │───▶│   Compute       │     │
│  │   (JSON/Binary) │    │   (SharedArray) │    │   Pipeline      │     │
│  └─────────────────┘    └─────────────────┘    └────────┬────────┘     │
│                                                          │              │
│                              ┌───────────────────────────┘              │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                   Compute Tier Selection                         │   │
│  │                                                                  │   │
│  │   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐ │   │
│  │   │  WebGPU  │    │  WASM    │    │  WASM    │    │  WASM    │ │   │
│  │   │  Compute │    │  SIMD128 │    │  SIMD256 │    │  Scalar  │ │   │
│  │   │  Shader  │    │  (SSE4)  │    │  (AVX2)  │    │ Fallback │ │   │
│  │   └──────────┘    └──────────┘    └──────────┘    └──────────┘ │   │
│  │       ▲               ▲               ▲               ▲        │   │
│  │       └───────────────┴───────────────┴───────────────┘        │   │
│  │                    Feature Detection                            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                   Render Pipeline                                │   │
│  │                                                                  │   │
│  │   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐ │   │
│  │   │  WebGPU  │    │  Canvas  │    │  SVG     │    │  Terminal│ │   │
│  │   │  Raster  │    │  2D      │    │  Vector  │    │  ASCII   │ │   │
│  │   └──────────┘    └──────────┘    └──────────┘    └──────────┘ │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Compute Tier Hierarchy

### 3.1 Tier Selection Algorithm

```rust
pub enum ComputeTier {
    WebGPU,       // Preferred: 10-100x speedup
    WasmSimd256,  // AVX2 equivalent (x86_64)
    WasmSimd128,  // SSE4/NEON equivalent
    WasmScalar,   // Fallback for old browsers
}

impl ComputeTier {
    // Implements "Progressive Refinement" (Piringer et al., 2009)
    // Starts with scalar/SIMD immediately, upgrades to WebGPU asynchronously
    pub fn initial() -> Self {
        // 1. Check WASM SIMD support (Synchronous)
        if wasm_feature_detect::simd() {
            if cfg!(target_feature = "avx2") {
                return ComputeTier::WasmSimd256;
            }
            return ComputeTier::WasmSimd128;
        }
        // 2. Scalar fallback
        ComputeTier::WasmScalar
    }

    // Asynchronously attempt to upgrade to WebGPU
    pub async fn try_upgrade(&self) -> Option<Self> {
        if let Some(gpu) = navigator_gpu().await {
            if let Some(adapter) = gpu.request_adapter().await {
                return Some(ComputeTier::WebGPU);
            }
        }
        None
    }
}
```

### 3.2 Performance Targets by Tier

| Tier | 10K Points | 100K Points | 1M Points | 10M Points |
|------|------------|-------------|-----------|------------|
| WebGPU | <1ms | <2ms | <8ms | <50ms |
| SIMD256 | <2ms | <10ms | <80ms | <800ms |
| SIMD128 | <3ms | <15ms | <120ms | <1.2s |
| Scalar | <10ms | <100ms | <1s | N/A |

---

## 4. WebSocket Streaming Protocol

### 4.1 Message Format

```typescript
// Binary protocol for maximum throughput
interface StreamMessage {
    type: MessageType;      // 1 byte
    timestamp: u64;         // 8 bytes (microseconds since epoch)
    sequence: u32;          // 4 bytes (for ordering)
    payload_len: u32;       // 4 bytes
    payload: Float32Array;  // Variable length
}

enum MessageType {
    DataPoint = 0x01,       // Single point update
    DataBatch = 0x02,       // Batch of points
    FullSnapshot = 0x03,    // Complete dataset replacement
    Delta = 0x04,           // Incremental update
    Control = 0xFF,         // Control messages
}
```

### 4.2 Streaming Data Sources (Demo)

```typescript
interface DataStream {
    // Financial tick data (high frequency)
    stockTicks: {
        symbol: string;
        price: f32;
        volume: u32;
        timestamp: u64;
    }[];

    // IoT sensor readings (medium frequency)
    sensorReadings: {
        device_id: u32;
        temperature: f32;
        humidity: f32;
        pressure: f32;
        lat: f32;
        lng: f32;
    }[];

    // ML training metrics (batch updates)
    trainingMetrics: {
        epoch: u32;
        loss: f32;
        accuracy: f32;
        learning_rate: f32;
    }[];
}
```

---

## 5. Grammar of Graphics Demo Scenes

### 5.1 Scene 1: Real-Time Scatter Plot (1M points)

```rust
// Demonstrates: GPU compute, aesthetic mapping, real-time updates
let plot = GgPlot::new()
    .data(&streaming_buffer)
    .aes(Aesthetics::new()
        .x("price")
        .y("volume")
        .color("sector")
        .size("market_cap")
        .alpha(0.6))
    .geom(Geom::Point {
        shape: Shape::Circle,
        jitter: Some(0.01),
    })
    .scale_x(Scale::Log10)
    .scale_y(Scale::Linear)
    .scale_color(Scale::Categorical(Palette::Tableau10))
    .coord(Coord::Cartesian {
        xlim: (0.01, 10000.0),
        ylim: (0.0, 1e9),
    })
    .facet(Facet::Wrap {
        by: "exchange",
        ncol: 3,
    })
    .theme(Theme::Minimal);

// GPU-accelerated render at 60fps
plot.render_webgpu(&canvas, &gpu_context)?;
```

### 5.2 Scene 2: Streaming Heatmap (Correlation Matrix)

```rust
// Demonstrates: Matrix computation, color scales, WebSocket updates
let heatmap = GgPlot::new()
    .data(&correlation_matrix)  // Updated every 100ms
    .aes(Aesthetics::new()
        .x("var1")
        .y("var2")
        .fill("correlation"))
    .geom(Geom::Tile {
        width: 1.0,
        height: 1.0,
    })
    .scale_fill(Scale::Diverging {
        palette: Palette::RedBlue,
        midpoint: 0.0,
        limits: (-1.0, 1.0),
    })
    .coord(Coord::Fixed { ratio: 1.0 })
    .labels(Labels::new()
        .title("Real-Time Asset Correlation")
        .x("Asset")
        .y("Asset"))
    .theme(Theme::Dark);
```

### 5.3 Scene 3: Multi-Series Line Chart (Time Series)

```rust
// Demonstrates: Douglas-Peucker simplification, streaming append
let chart = GgPlot::new()
    .data(&time_series_buffer)  // Ring buffer, last 10 minutes
    .aes(Aesthetics::new()
        .x("timestamp")
        .y("value")
        .color("series_id")
        .group("series_id"))
    .geom(Geom::Line {
        simplify: Some(1.0),  // Douglas-Peucker epsilon
        interpolation: Interpolation::Monotone,
    })
    .geom(Geom::Area {
        alpha: 0.1,
        fill: true,
    })
    .scale_x(Scale::Time {
        breaks: Duration::minutes(1),
        labels: "%H:%M",
    })
    .scale_y(Scale::Linear)
    .scale_color(Scale::Sequential(Palette::Viridis))
    .annotations(vec![
        Annotation::HLine {
            y: threshold,
            linetype: LineType::Dashed,
            color: Rgba::RED,
            label: "Threshold",
        },
    ])
    .theme(Theme::Publication);
```

### 5.4 Scene 4: Force-Directed Graph (Network Topology)

```rust
// Demonstrates: GPU force simulation, real-time edge updates
let graph = GgPlot::new()
    .data(&network_topology)
    .aes(Aesthetics::new()
        .node_size("degree")
        .node_color("community")
        .edge_width("weight")
        .edge_alpha("recency"))
    .geom(Geom::Network {
        layout: Layout::ForceDirected {
            iterations: 100,
            gpu_accelerated: true,
        },
        node_shape: Shape::Circle,
        edge_bundling: true,
    })
    .scale_node_size(Scale::Sqrt { range: (5.0, 50.0) })
    .scale_node_color(Scale::Categorical(Palette::Set3))
    // Interaction Grammar (Satyanarayan et al., 2017)
    // Defines signals and event streams rather than static flags
    .interaction(Interaction::new()
        .signal("zoom", Event::Wheel)
        .signal("drag", Event::PointerMove.filter("mousedown"))
        .bind("node_pos", "drag")
        .tooltip(Tooltip::new().on("hover")))
    .theme(Theme::Graph);
```

### 5.5 Scene 5: Violin + Box Plot Ensemble

```rust
// Demonstrates: Kernel density estimation, statistical overlays
let ensemble = GgPlot::new()
    .data(&distribution_data)
    .aes(Aesthetics::new()
        .x("category")
        .y("value")
        .fill("category"))
    .geom(Geom::Violin {
        bandwidth: Bandwidth::Silverman,
        draw_quantiles: vec![0.25, 0.5, 0.75],
        trim: true,
    })
    .geom(Geom::Boxplot {
        width: 0.1,
        outlier_shape: Shape::Diamond,
        notch: true,
    })
    .geom(Geom::Jitter {
        width: 0.2,
        alpha: 0.3,
        size: 1.5,
    })
    .scale_fill(Scale::Categorical(Palette::Pastel1))
    .coord(Coord::Flip)
    .stat(Stat::Summary {
        fun: "mean_se",
        geom: Geom::ErrorBar,
    })
    .theme(Theme::Classic);
```

### 5.6 Scene 6: ROC/PR Curve Dashboard (ML Metrics)

```rust
// Demonstrates: Statistical curves, AUC computation, streaming updates
let dashboard = GgPlot::new()
    .data(&classifier_results)
    .facet(Facet::Grid {
        rows: "model",
        cols: "metric_type",
    })
    // ROC Curves (left column)
    .geom(Geom::RocCurve {
        show_auc: true,
        confidence_band: true,
    })
    // PR Curves (right column)
    .geom(Geom::PrCurve {
        show_baseline: true,
        iso_f1: vec![0.2, 0.4, 0.6, 0.8],
    })
    .scale_x(Scale::Linear { limits: (0.0, 1.0) })
    .scale_y(Scale::Linear { limits: (0.0, 1.0) })
    .scale_color(Scale::Categorical(Palette::Dark2))
    .annotations(vec![
        // Diagonal reference line for ROC
        Annotation::ABLine {
            slope: 1.0,
            intercept: 0.0,
            linetype: LineType::Dashed,
            color: Rgba::GRAY,
        },
    ])
    .theme(Theme::Minimal);
```

---

## 6. GPU Compute Shaders

### 6.1 Scatter Plot Binning Shader (WGSL)

```wgsl
// Parallel histogram/binning for large scatter plots
struct Params {
    width: u32,
    height: u32,
    x_min: f32,
    x_max: f32,
    y_min: f32,
    y_max: f32,
    point_count: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> points: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> bins: array<atomic<u32>>;

// Optimization: Use shared memory privatization (Lins et al., 2013)
// Reduces global atomic contention for skewed distributions
var<workgroup> local_bins: array<atomic<u32>, 256>;

@compute @workgroup_size(256)
fn bin_points(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_index) local_idx: u32) {
    // 1. Initialize local bins
    if (local_idx < 256u) {
        atomicStore(&local_bins[local_idx], 0u);
    }
    workgroupBarrier();

    let idx = global_id.x;
    if (idx < params.point_count) {
        let p = points[idx];

        // Map to bin coordinates
        let x_norm = (p.x - params.x_min) / (params.x_max - params.x_min);
        let y_norm = (p.y - params.y_min) / (params.y_max - params.y_min);

        let bin_x = u32(clamp(x_norm * f32(params.width), 0.0, f32(params.width - 1)));
        let bin_y = u32(clamp(y_norm * f32(params.height), 0.0, f32(params.height - 1)));
        
        // Map 2D bin to 1D local cache (simplified for demo)
        let local_bin_idx = (bin_y * params.width + bin_x) % 256u;
        atomicAdd(&local_bins[local_bin_idx], 1u);
    }
    workgroupBarrier();

    // 2. Flush to global memory
    if (local_idx < 256u) {
        let count = atomicLoad(&local_bins[local_idx]);
        if (count > 0u) {
            // Note: In real impl, need mapping back to global index
            atomicAdd(&bins[local_idx], count); 
        }
    }
}
```

### 6.2 Correlation Matrix Shader (WGSL)

```wgsl
// Parallel Pearson correlation computation
struct MatrixParams {
    n_vars: u32,
    n_samples: u32,
}

@group(0) @binding(0) var<uniform> params: MatrixParams;
@group(0) @binding(1) var<storage, read> data: array<f32>;      // n_vars x n_samples
@group(0) @binding(2) var<storage, read> means: array<f32>;     // n_vars
@group(0) @binding(3) var<storage, read> stds: array<f32>;      // n_vars
@group(0) @binding(4) var<storage, read_write> corr: array<f32>; // n_vars x n_vars

@compute @workgroup_size(16, 16)
fn correlation_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let j = global_id.y;

    if (i >= params.n_vars || j >= params.n_vars) {
        return;
    }

    // Compute Pearson correlation r_ij
    var cov: f32 = 0.0;
    for (var k: u32 = 0u; k < params.n_samples; k++) {
        let x_i = data[i * params.n_samples + k];
        let x_j = data[j * params.n_samples + k];
        cov += (x_i - means[i]) * (x_j - means[j]);
    }
    cov /= f32(params.n_samples - 1);

    let r = cov / (stds[i] * stds[j]);
    corr[i * params.n_vars + j] = r;
}
```

### 6.3 Force-Directed Layout Shader (WGSL)

```wgsl
// Barnes-Hut approximation for force-directed graph layout
struct Node {
    x: f32,
    y: f32,
    vx: f32,
    vy: f32,
    mass: f32,
}

struct Edge {
    source: u32,
    target: u32,
    weight: f32,
}

@group(0) @binding(0) var<uniform> params: ForceParams;
@group(0) @binding(1) var<storage, read_write> nodes: array<Node>;
@group(0) @binding(2) var<storage, read> edges: array<Edge>;
@group(0) @binding(3) var<storage, read> grid_cells: array<u32>; // Spatial Index

@compute @workgroup_size(256)
fn force_iteration(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.node_count) {
        return;
    }

    var node = nodes[idx];
    var fx: f32 = 0.0;
    var fy: f32 = 0.0;

    // Repulsive forces (Optimized: Spatial Grid - Cayton, 2012)
    // Instead of O(N^2), we only check neighbors in the spatial grid (approx O(N))
    let grid_x = i32(node.x / params.cell_size);
    let grid_y = i32(node.y / params.cell_size);

    for (var dy: i32 = -1; dy <= 1; dy++) {
        for (var dx: i32 = -1; dx <= 1; dx++) {
            // In a full impl, iterate over nodes in grid_cells[grid_y + dy][grid_x + dx]
            // This reduces complexity from 100M ops/frame to ~2M ops/frame
            let neighbor_cell_idx = get_cell_idx(grid_x + dx, grid_y + dy);
            // ... compute repulsion for nodes in this cell ...
        }
    }

    // Attractive forces (edges only)
    for (var e: u32 = 0u; e < params.edge_count; e++) {
        let edge = edges[e];
        if (edge.source != idx && edge.target != idx) { continue; }

        let other_idx = select(edge.target, edge.source, edge.source == idx);
        let other = nodes[other_idx];
        let dx = other.x - node.x;
        let dy = other.y - node.y;
        let dist = sqrt(dx * dx + dy * dy);

        // Hooke's law: F = k * x
        let attraction = params.spring_strength * dist * edge.weight;
        fx += attraction * dx / max(dist, 0.01);
        fy += attraction * dy / max(dist, 0.01);
    }

    // Velocity update with damping
    node.vx = (node.vx + fx) * params.damping;
    node.vy = (node.vy + fy) * params.damping;

    // Position update
    node.x += node.vx * params.dt;
    node.y += node.vy * params.dt;

    nodes[idx] = node;
}
```

---

## 7. Demo UI/UX Design

### 7.1 Layout Structure

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  trueno-viz GPU Demo                                    [Tier: WebGPU]     │
├──────────────────────┬──────────────────────────────────────────────────────┤
│                      │                                                      │
│  Scene Selection     │                                                      │
│  ┌────────────────┐  │           Main Visualization Canvas                  │
│  │ ● Scatter 1M   │  │                                                      │
│  │ ○ Heatmap      │  │                   [WebGPU Canvas]                    │
│  │ ○ Time Series  │  │                                                      │
│  │ ○ Network      │  │                   60 FPS @ 1080p                     │
│  │ ○ Violin       │  │                                                      │
│  │ ○ ROC/PR       │  │                   1,000,000 points                   │
│  └────────────────┘  │                                                      │
│                      │                                                      │
│  Data Stream         │                                                      │
│  ┌────────────────┐  │                                                      │
│  │ WS: Connected  │  │                                                      │
│  │ Rate: 10K/s    │  │                                                      │
│  │ Latency: 2ms   │  │                                                      │
│  └────────────────┘  │                                                      │
│                      ├──────────────────────────────────────────────────────┤
│  Performance         │  Code Editor (Grammar of Graphics)                   │
│  ┌────────────────┐  │  ┌────────────────────────────────────────────────┐  │
│  │ Frame: 2.1ms   │  │  │ let plot = GgPlot::new()                       │  │
│  │ Compute: 0.8ms │  │  │     .data(&streaming_buffer)                   │  │
│  │ Render: 1.3ms  │  │  │     .aes(Aesthetics::new()                     │  │
│  │ FPS: 60        │  │  │         .x("price")                            │  │
│  │ Points: 1.0M   │  │  │         .y("volume")                           │  │
│  └────────────────┘  │  │         .color("sector"))                      │  │
│                      │  │     .geom(Geom::Point)                         │  │
│  [Pause] [Reset]     │  │     .theme(Theme::Minimal);                    │  │
│                      │  └────────────────────────────────────────────────┘  │
│                      │  [Run Code]                                          │
└──────────────────────┴──────────────────────────────────────────────────────┘
```

### 7.2 Performance Metrics Display

```typescript
interface PerformanceMetrics {
    // Frame timing
    frameTime: number;        // Total frame time (ms)
    computeTime: number;      // GPU compute time (ms)
    renderTime: number;       // GPU render time (ms)
    jsTime: number;           // JavaScript overhead (ms)

    // Data stats
    pointCount: number;       // Current point count
    updateRate: number;       // Points/second from WebSocket
    bufferUtilization: number; // Ring buffer fill percentage

    // Compute tier
    tier: ComputeTier;        // Current compute backend
    gpuVendor: string;        // GPU adapter info

    // Network
    wsLatency: number;        // WebSocket round-trip (ms)
    wsConnected: boolean;     // Connection status
}
```

---

## 8. Implementation Phases

### Phase 1: Foundation (Week 1)

- [ ] WebGPU context initialization with WASM SIMD fallback
- [ ] Ring buffer for streaming data (SharedArrayBuffer)
- [ ] Basic scatter plot with GPU binning shader
- [ ] Performance metrics overlay

### Phase 2: Grammar API (Week 2)

- [ ] Aesthetics mapping (x, y, color, size, alpha)
- [ ] Scale abstractions (linear, log, time, categorical)
- [ ] Coordinate systems (cartesian, polar, flip)
- [ ] Theme engine

### Phase 3: Advanced Geometries (Week 3)

- [ ] Heatmap with GPU correlation computation
- [ ] Line chart with Douglas-Peucker (GPU)
- [ ] Force-directed graph layout (GPU)
- [ ] Violin/boxplot KDE (GPU)

### Phase 4: Streaming & Polish (Week 4)

- [ ] WebSocket binary protocol
- [ ] Demo data generators (financial, IoT, ML)
- [ ] Interactive code editor
- [ ] Documentation and examples

---

## 9. Build & Deployment

### 9.1 WASM Build Configuration

```toml
# Cargo.toml [lib] section
[lib]
crate-type = ["cdylib", "rlib"]

[features]
default = ["console_error_panic_hook"]
webgpu = []
simd = []

[dependencies]
wasm-bindgen = "0.2"
web-sys = { version = "0.3", features = [
    "Window",
    "Navigator",
    "Gpu",
    "GpuAdapter",
    "GpuDevice",
    "GpuQueue",
    "GpuBuffer",
    "GpuComputePipeline",
    "GpuShaderModule",
    "WebSocket",
    "MessageEvent",
    "BinaryType",
]}
js-sys = "0.3"
```

### 9.2 Build Commands

```bash
# Development build with SIMD
RUSTFLAGS='-C target-feature=+simd128' wasm-pack build --target web --dev

# Production build (optimized)
RUSTFLAGS='-C target-feature=+simd128' wasm-pack build --target web --release

# Serve demo
python3 -m http.server 8080
```

---

## 10. Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Frame rate | ≥60 FPS | Chrome DevTools Performance |
| Frame time | <16ms | GPU timestamp queries |
| Point capacity | ≥1M | Automated benchmark |
| WebSocket latency | <10ms | Round-trip measurement |
| WASM size | <2MB | gzip compressed |
| Time to first frame | <500ms | Performance.now() |
| Memory usage | <512MB | Chrome Task Manager |
| Code API usability | ggplot2-like | User study feedback |

---

## 11. References

[1] Wilkinson, L. (2005). *The Grammar of Graphics* (2nd ed.). Springer.

[2] Wickham, H. (2010). "A Layered Grammar of Graphics." *JCGS*, 19(1), 3-28.

[3] Satyanarayan, A., et al. (2017). "Vega-Lite." *IEEE TVCG*, 23(1), 341-350.

[4] Liu, S., et al. (2017). "Visualizing High-Dimensional Data." *IEEE TVCG*, 23(3), 1249-1268.

[5] Lins, L., et al. (2013). "Nanocubes." *IEEE TVCG*, 19(12), 2456-2465.

[6] Piringer, H., et al. (2009). "Multi-Threading Architecture." *IEEE TVCG*, 15(6), 1113-1120.

[7] Fisher, D. (2011). "Incremental Database Queries." *IEEE LDAV*, 73-80.

[8] Battle, L., & Heer, J. (2019). "Exploratory Visual Analysis." *CGF*, 38(3), 145-159.

[9] Nickolls, J., et al. (2008). "Scalable Parallel Programming." *ACM Queue*, 6(2), 40-53.

[10] Cayton, L. (2012). "Accelerating Nearest Neighbor Search." *IEEE IPDPS*, 402-413.

---

*Specification Version: 1.0.0*
*Last Updated: 2025-11-25*
