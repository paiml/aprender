# Load Testing Visualization and Reporting Specification

**Document ID:** PROBAR-SPEC-006
**Parent:** PROBAR-SPEC-005 (Enhanced Serving and Debugging)
**Version:** 1.0.0
**Status:** Approved (TUI/WASM-only, no HTML/JS)
**Author:** Claude Code
**Date:** 2025-12-14
**Reviewed:** 2025-12-14 (confirmed no HTML/JavaScript output)

## Abstract

This specification extends PROBAR-SPEC-005 Section D (Load Testing) with advanced visualization, statistical analysis, deep tracing, and simulation playback capabilities. The design integrates with the Sovereign AI Stack components (trueno, renacer, simular, aprender) to provide industry-leading load testing for WASM client-side applications.

## Motivation

Current load testing tools (Locust, k6, Gatling) focus on server-side metrics. For WASM applications, client-side performance is equally critical:

1. **WASM boot sequence latency** affects user experience
2. **Linear memory pressure** can cause OOM on constrained devices
3. **Rendering performance** degrades under concurrent load
4. **Deep tracing** is needed to identify syscall-level bottlenecks

This specification addresses the gap between server-side load testing and client-side WASM performance analysis.

## Peer-Reviewed Citations

### [C8] Statistical Analysis of Latency

> Huang, J., et al. "Statistical Analysis of Latency Through Semantic Profiling." *Proceedings of the Twelfth European Conference on Computer Systems (EuroSys)*, ACM, 2017. https://dl.acm.org/doi/10.1145/3064176.3064179

**Relevance:** Introduces VProfiler for analyzing latency variance using "variance trees." Their technique reduces 99th percentile latency by 50% through systematic variance decomposition. Our statistical analysis module applies similar variance attribution to identify WASM performance bottlenecks.

### [C9] Tail Latency Attribution

> Zhang, Y., et al. "Treadmill: Attributing the Source of Tail Latency Through Precise Load Testing and Statistical Inference." *ACM SIGARCH Computer Architecture News*, Vol. 44, No. 3, 2016. https://dl.acm.org/doi/10.1145/3007787.3001186

**Relevance:** Treadmill uses quantile regression to attribute tail latency sources. Their methodology achieves 43% reduction in p99 latency through precise attribution. Our deep tracing integration with renacer follows this attribution approach for WASM-specific bottlenecks.

### [C10] WebAssembly Performance Testing

> Jangda, A., et al. "Revealing Performance Issues in Server-side WebAssembly Runtimes via Differential Testing." *Proceedings of the 38th IEEE/ACM International Conference on Automated Software Engineering (ASE)*, 2023. https://conf.researchr.org/details/ase-2023/ase-2023-papers/133/

**Relevance:** WarpDiff identifies performance issues through differential testing across WASM runtimes. Their oracle ratio methodology detects abnormal latency. We apply similar differential analysis when testing across Chrome/Firefox/Safari WASM engines.

### [C11] Record-Reduce-Replay for WASM

> Lehmann, D., et al. "Wasm-R3: Record-Reduce-Replay for Realistic and Standalone WebAssembly Benchmarks." *Proceedings of the ACM on Programming Languages (OOPSLA)*, 2024. https://2024.splashcon.org/details/splash-2024-oopsla/130/

**Relevance:** Wasm-R3 demonstrates record-replay for WASM with 99.53% trace reduction. Their technique produces standalone replay modules. Our simulation playback from simular applies similar principles for deterministic load test reproduction.

### [C12] The Tail at Scale

> Dean, J., & Barroso, L.A. "The Tail at Scale." *Communications of the ACM*, Vol. 56, No. 2, 2013, pp. 74-80. https://cacm.acm.org/research/the-tail-at-scale/

**Relevance:** Foundational paper on tail latency in distributed systems. Shows that 99th percentile latency with 100 parallel requests approaches worst-case behavior. Our visualization applies their "hedged requests" and "tied requests" patterns for WASM boot optimization.

---

## Industry Tool Analysis

### Locust (Python)

**Key Features:**
- Real-time web UI with throughput/latency charts
- OpenTelemetry integration (v2.42.4+) for distributed tracing
- Bokeh-based custom visualizations
- Azure Load Testing multi-region support

**Applicable Patterns:**
- Stage-based load profiles (ramp-up, steady, spike, recovery)
- User-centric scenarios (task weights, think times)
- Real-time metrics streaming

### k6 (Grafana)

**Key Features:**
- Grafana Cloud integration with pre-built dashboards
- xk6-dashboard for local web visualization
- InfluxDB streaming for time-series analysis
- Built-in trend detection ("knee" pattern identification)

**Applicable Patterns:**
- Metrics correlation (load vs system resources)
- Historical comparison across test runs
- Threshold-based pass/fail criteria

### Gatling (Scala)

**Key Features:**
- Feature-rich HTML reports with interactive JavaScript charts
- Response time distribution (p50/p75/p95/p99)
- Assertions overview with pass/fail summary
- Differential reports for A/B comparison

**Applicable Patterns:**
- Comprehensive percentile reporting
- Visual bottleneck identification
- CI/CD gate integration

---

## Stack Integration Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    PROBAR LOAD TESTING ENHANCED                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │                         Load Generator                              │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐           │  │
│  │  │  Ramp-up │  │  Steady  │  │  Spike   │  │ Recovery │           │  │
│  │  │  Stage   │──│  State   │──│  Test    │──│  Phase   │           │  │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘           │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                              │                                            │
│           ┌──────────────────┼──────────────────┐                        │
│           ▼                  ▼                  ▼                        │
│  ┌────────────────┐ ┌────────────────┐ ┌────────────────┐               │
│  │    trueno      │ │    renacer     │ │    simular     │               │
│  │   Statistics   │ │  Deep Tracing  │ │   Simulation   │               │
│  │                │ │                │ │   Playback     │               │
│  │ • Percentiles  │ │ • Syscall log  │ │ • Record/      │               │
│  │ • Variance     │ │ • Source corr  │ │   Replay       │               │
│  │ • Histograms   │ │ • Flamegraph   │ │ • Monte Carlo  │               │
│  │ • Regression   │ │ • WASM memory  │ │ • Scenarios    │               │
│  └───────┬────────┘ └───────┬────────┘ └───────┬────────┘               │
│          │                  │                  │                        │
│          └──────────────────┼──────────────────┘                        │
│                             ▼                                            │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │                      trueno-viz Visualization                       │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │  │
│  │  │  Real-time   │  │   Latency    │  │  Flamegraph  │              │  │
│  │  │  Dashboard   │  │  Heatmaps    │  │   Viewer     │              │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘              │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │  │
│  │  │   Memory     │  │  Throughput  │  │   Trend      │              │  │
│  │  │   Timeline   │  │    Charts    │  │   Analysis   │              │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘              │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Feature Specifications

### H. Enhanced Visualization (`probador load-test --viz`)

#### H.1 Command Interface

```bash
# Real-time TUI dashboard during load test
probador load-test --scenario boot.yaml --viz

# Export binary report (MessagePack format)
probador load-test --scenario boot.yaml --report results.msgpack

# Export JSON report (for tooling integration)
probador load-test --scenario boot.yaml --report results.json

# Stream metrics to file (append mode for live monitoring)
probador load-test --scenario boot.yaml --stream metrics.ndjson
```

**NOTE:** No HTML or JavaScript output. All visualization is TUI-based or binary format for WASM consumption.

#### H.2 Real-Time Dashboard (trueno-viz)

```
┌─────────────────────────────────────────────────────────────────────────┐
│ LOAD TEST: WASM Boot Sequence                    Stage: spike (45s/60s) │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  THROUGHPUT (req/s)                       ACTIVE USERS                   │
│  ┌────────────────────────────┐          ┌────────────────────────────┐ │
│  │                    ╭───╮   │          │                    ╭───────│ │
│  │               ╭────╯   │   │          │          ╭─────────╯       │ │
│  │         ╭─────╯        │   │          │    ╭─────╯                 │ │
│  │    ╭────╯              │   │          │────╯                       │ │
│  │────╯                   ╰───│          │                            │ │
│  └────────────────────────────┘          └────────────────────────────┘ │
│   0        50       100      130s         0        50       100      130s│
│   Peak: 892 req/s @ t=45s                 Current: 200 users             │
│                                                                          │
│  LATENCY PERCENTILES (ms)                 ERROR RATE                     │
│  ┌────────────────────────────┐          ┌────────────────────────────┐ │
│  │ p99 ████████████████ 456   │          │                            │ │
│  │ p95 ██████████ 234         │          │    0.03%  ────────────     │ │
│  │ p50 ████ 78                │          │                            │ │
│  │ min │ 12                   │          │                            │ │
│  └────────────────────────────┘          └────────────────────────────┘ │
│                                                                          │
│  ENDPOINT BREAKDOWN                                                      │
│  ┌──────────────┬────────┬────────┬────────┬────────┬────────┐         │
│  │ Endpoint     │ Count  │ p50    │ p95    │ p99    │ Errors │         │
│  ├──────────────┼────────┼────────┼────────┼────────┼────────┤         │
│  │ load_html    │ 15,080 │ 12ms   │ 45ms   │ 89ms   │ 0      │         │
│  │ load_wasm    │ 15,075 │ 78ms   │ 234ms  │ 456ms  │ 5      │         │
│  │ load_model   │ 15,075 │ 890ms  │ 1.8s   │ 3.2s   │ 7      │         │
│  └──────────────┴────────┴────────┴────────┴────────┴────────┘         │
│                                                                          │
│  [q] Quit  [p] Pause  [r] Reset  [e] Export  [d] Deep trace             │
└─────────────────────────────────────────────────────────────────────────┘
```

#### H.3 TUI Report Viewer

Terminal-based report viewer (no browser required):

```bash
# View report in TUI
probador report view results.msgpack

# Compare two reports in TUI
probador report compare baseline.msgpack current.msgpack
```

**Report Structure (MessagePack/JSON):**

1. **Summary Statistics**
   - Total requests, success rate, duration
   - Throughput peak/average
   - Resource usage (CPU, memory)

2. **Response Time Distribution**
   - Histogram data with percentile markers
   - Per-endpoint breakdown
   - Time-series data with stage markers

3. **Error Analysis**
   - Error categorization (timeout, 4xx, 5xx, connection)
   - Timestamped error log
   - Sample request/response data for debugging

4. **Assertion Results**
   - Pass/fail summary with actual vs expected
   - Trend data (when comparing multiple runs)
   - Actionable recommendations

**TUI Report Display:**
```
LOAD TEST REPORT VIEWER                                    [q]uit [n]ext [p]rev
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ┌─ SUMMARY ──────────────────────────────────────────────────────────────┐
  │ Duration: 130s │ Requests: 45,230 │ Success: 99.97% │ Apdex: 0.89     │
  └────────────────────────────────────────────────────────────────────────┘

  ┌─ LATENCY DISTRIBUTION ─────────────────────────────────────────────────┐
  │   ▁▂▄▆█▆▄▂▁                                                            │
  │  12ms    78ms    156ms    234ms    456ms                               │
  │  min     p50      p90      p95      p99                                │
  └────────────────────────────────────────────────────────────────────────┘

  ┌─ ASSERTIONS ───────────────────────────────────────────────────────────┐
  │  ✓ load_html latency_p95 < 100ms (actual: 45ms)                       │
  │  ✓ load_wasm latency_p95 < 500ms (actual: 234ms)                      │
  │  ✗ load_model latency_p95 < 1500ms (actual: 1.8s) ← FAILED            │
  └────────────────────────────────────────────────────────────────────────┘
```

#### H.4 Implementation Requirements

```rust
use trueno_viz::{Chart, Dashboard, Widget};

pub struct LoadTestVisualization {
    pub dashboard: Dashboard,
    pub metrics_stream: MetricsStream,
    pub export_format: ExportFormat,
}

/// Export formats - NO HTML/JavaScript
pub enum ExportFormat {
    /// MessagePack binary (compact, WASM-friendly)
    MessagePack,
    /// JSON (human-readable, tooling integration)
    Json,
    /// NDJSON stream (newline-delimited, append-friendly)
    NdJsonStream { path: PathBuf },
    /// Binary stream (for real-time TUI consumption)
    BinaryStream,
}

pub struct MetricsStream {
    pub throughput: TimeSeries<f64>,
    pub latency_histogram: StreamingHistogram,
    pub error_rate: TimeSeries<f64>,
    pub active_users: TimeSeries<u32>,
}

impl LoadTestVisualization {
    pub fn new(config: &VisualizationConfig) -> Self {
        let dashboard = Dashboard::new()
            .add_widget(Widget::line_chart("Throughput").stream("throughput"))
            .add_widget(Widget::histogram("Latency").stream("latency"))
            .add_widget(Widget::gauge("Error Rate").stream("error_rate"))
            .add_widget(Widget::table("Endpoints").stream("endpoints"));

        Self {
            dashboard,
            metrics_stream: MetricsStream::new(),
            export_format: ExportFormat::MessagePack, // Default: binary
        }
    }
}
```

---

### I. Statistical Analysis (trueno Integration)

#### I.1 Command Interface

```bash
# Run load test with statistical analysis
probador load-test --scenario boot.yaml --stats

# Compare two load test results
probador load-test compare baseline.json current.json

# Generate statistical report (TUI or binary)
probador load-test --scenario boot.yaml --stats-report stats.msgpack
```

#### I.2 Statistical Metrics (Based on [C8], [C9], [C12])

| Metric | Description | Formula |
|--------|-------------|---------|
| **Variance Tree** | Hierarchical latency variance decomposition | Var(total) = Σ Var(component) + 2Σ Cov(i,j) |
| **Quantile Regression** | Tail latency attribution | Q_τ(Y\|X) = X'β_τ |
| **Coefficient of Variation** | Latency stability measure | CV = σ/μ |
| **Apdex Score** | User satisfaction index | Apdex = (Satisfied + Tolerating/2) / Total |
| **Throughput Knee** | Inflection point detection | d²(latency)/d(load)² = 0 |

#### I.3 Statistical Report Output

```
STATISTICAL ANALYSIS: WASM Boot Sequence
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

VARIANCE DECOMPOSITION (following [C8] VProfiler methodology)
┌───────────────────────────────────────────────────────────────┐
│ Total Latency Variance: 15,234 ms²                            │
│                                                                │
│ ├── Network I/O:     8,123 ms² (53.3%)  ████████████████████  │
│ │   ├── DNS:           412 ms² ( 2.7%)  ██                    │
│ │   ├── TCP:         1,234 ms² ( 8.1%)  ████                  │
│ │   └── Transfer:    6,477 ms² (42.5%)  ████████████████      │
│ │                                                              │
│ ├── WASM Execution:  4,567 ms² (30.0%)  ████████████          │
│ │   ├── Compile:     2,345 ms² (15.4%)  ██████                │
│ │   └── Init:        2,222 ms² (14.6%)  ██████                │
│ │                                                              │
│ └── Rendering:       2,544 ms² (16.7%)  ███████               │
│     ├── Layout:        890 ms² ( 5.8%)  ███                   │
│     └── Paint:       1,654 ms² (10.9%)  ████                  │
└───────────────────────────────────────────────────────────────┘

TAIL LATENCY ANALYSIS (following [C9] Treadmill methodology)
┌───────────────────────────────────────────────────────────────┐
│ Percentile │ Latency │ Attribution                            │
│────────────┼─────────┼────────────────────────────────────────│
│ p50        │ 78ms    │ Typical case (WASM cached)             │
│ p90        │ 156ms   │ Cold cache (WASM recompile)            │
│ p95        │ 234ms   │ Network congestion                     │
│ p99        │ 456ms   │ GC pause + network timeout retry       │
│ p99.9      │ 1.2s    │ Full page reload (cache invalidation)  │
└───────────────────────────────────────────────────────────────┘

THROUGHPUT KNEE DETECTION (following [C12] Tail at Scale)
┌───────────────────────────────────────────────────────────────┐
│ Knee detected at: 150 concurrent users                        │
│                                                                │
│ latency  │                                                     │
│  500ms   │                                          ╭─────    │
│  400ms   │                                     ╭────╯         │
│  300ms   │                                ╭────╯              │
│  200ms   │                           ╭────╯                   │
│  100ms   │────────────────────────────╯ ← knee at 150 users   │
│    0ms   └───────────────────────────────────────────          │
│           0    50   100   150   200   250   300  users        │
│                                                                │
│ Recommendation: Scale horizontally at 120 users (80% of knee) │
└───────────────────────────────────────────────────────────────┘

APDEX SCORE
┌───────────────────────────────────────────────────────────────┐
│ Target: 100ms (Satisfied), 400ms (Tolerating)                 │
│                                                                │
│ Satisfied:  12,345 requests (82.3%)                           │
│ Tolerating:  2,100 requests (14.0%)                           │
│ Frustrated:    555 requests ( 3.7%)                           │
│                                                                │
│ Apdex Score: 0.89 (Good)                                      │
│                                                                │
│ Industry Benchmark: E-commerce 0.85, Gaming 0.90, Enterprise 0.80│
└───────────────────────────────────────────────────────────────┘
```

#### I.4 Implementation Requirements

```rust
use trueno::stats::{Histogram, Percentile, Variance, Regression};

pub struct StatisticalAnalysis {
    pub variance_tree: VarianceTree,
    pub quantile_regression: QuantileRegression,
    pub apdex: ApdexCalculator,
    pub knee_detector: KneeDetector,
}

pub struct VarianceTree {
    pub total_variance: f64,
    pub components: Vec<VarianceComponent>,
}

pub struct VarianceComponent {
    pub name: String,
    pub variance: f64,
    pub percentage: f64,
    pub children: Vec<VarianceComponent>,
}

pub struct QuantileRegression {
    pub quantiles: Vec<f64>,  // [0.5, 0.9, 0.95, 0.99, 0.999]
    pub coefficients: HashMap<String, Vec<f64>>,
    pub attributions: Vec<TailAttribution>,
}

pub struct TailAttribution {
    pub percentile: f64,
    pub latency: Duration,
    pub primary_cause: String,
    pub contributing_factors: Vec<(String, f64)>,
}

pub struct KneeDetector {
    pub knee_point: Option<(f64, f64)>,  // (users, latency)
    pub second_derivative: Vec<(f64, f64)>,
    pub recommended_capacity: f64,
}
```

---

### J. Deep Tracing (renacer Integration)

#### J.1 Command Interface

```bash
# Enable deep tracing during load test
probador load-test --scenario boot.yaml --trace

# Analyze trace file
probador trace analyze trace.renacer

# Generate flamegraph from trace
probador trace flamegraph trace.renacer --output flame.svg

# Correlate trace with source code
probador trace correlate trace.renacer --source ./src
```

#### J.2 Trace Categories

| Category | Captured Events | Use Case |
|----------|-----------------|----------|
| **Syscall** | read, write, mmap, futex | I/O bottleneck detection |
| **WASM** | compile, instantiate, call, memory_grow | WASM-specific profiling |
| **Network** | connect, send, recv, dns | Network latency attribution |
| **Memory** | malloc, free, mprotect | Memory leak detection |
| **GPU** | shader_compile, draw_call | Rendering performance |

#### J.3 Deep Trace Output

```
DEEP TRACE ANALYSIS: WASM Boot Sequence (request #4523)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TIMELINE (total: 234ms)
┌──────────────────────────────────────────────────────────────────────────┐
│ 0ms                           100ms                          234ms       │
│ ├────────────────────────────────┼────────────────────────────┤          │
│                                                                          │
│ [DNS]    ██ 12ms                                                         │
│ [TCP]       ████ 23ms                                                    │
│ [TLS]           ██████ 34ms                                              │
│ [HTTP]                 ██████████████████████████ 89ms                   │
│ [WASM]                                          ████████████ 56ms        │
│ [Render]                                                    ████ 20ms    │
│                                                                          │
│ Critical Path: HTTP (38%) → WASM (24%) → TLS (15%)                      │
└──────────────────────────────────────────────────────────────────────────┘

SYSCALL BREAKDOWN
┌─────────────────────────────────────────────────────────────────────────┐
│ Syscall        │ Count │ Total Time │ Avg Time │ Max Time │ % of Total │
│────────────────┼───────┼────────────┼──────────┼──────────┼────────────│
│ recvfrom       │ 1,234 │ 89.2ms     │ 72μs     │ 12ms     │ 38.1%      │
│ mmap           │ 45    │ 34.5ms     │ 767μs    │ 8ms      │ 14.7%      │
│ futex          │ 234   │ 23.1ms     │ 99μs     │ 5ms      │ 9.9%       │
│ write          │ 89    │ 12.3ms     │ 138μs    │ 2ms      │ 5.3%       │
│ other          │ 567   │ 75.0ms     │ 132μs    │ 15ms     │ 32.0%      │
└─────────────────────────────────────────────────────────────────────────┘

WASM-SPECIFIC EVENTS
┌─────────────────────────────────────────────────────────────────────────┐
│ Event              │ Duration │ Memory Impact │ Source Location         │
│────────────────────┼──────────┼───────────────┼─────────────────────────│
│ wasm_compile       │ 34ms     │ +12MB         │ (V8 internal)           │
│ wasm_instantiate   │ 12ms     │ +4MB          │ (V8 internal)           │
│ wasm_call(init)    │ 8ms      │ +2MB          │ src/lib.rs:45           │
│ wasm_memory_grow   │ 2ms      │ +16MB         │ src/allocator.rs:123    │
└─────────────────────────────────────────────────────────────────────────┘

SOURCE CORRELATION (top 5 hot spots)
┌─────────────────────────────────────────────────────────────────────────┐
│ File:Line             │ Function            │ Time  │ Calls │ Suggestion│
│───────────────────────┼─────────────────────┼───────┼───────┼───────────│
│ src/decoder.rs:234    │ decode_frame        │ 45ms  │ 1,234 │ ⚠ SIMD    │
│ src/allocator.rs:123  │ alloc_buffer        │ 23ms  │ 89    │ ⚠ Pool    │
│ src/network.rs:567    │ fetch_chunk         │ 18ms  │ 45    │ ✓ OK      │
│ src/render.rs:89      │ draw_waveform       │ 12ms  │ 234   │ ⚠ Batch   │
│ src/model.rs:345      │ load_weights        │ 8ms   │ 1     │ ✓ OK      │
└─────────────────────────────────────────────────────────────────────────┘
```

#### J.4 Implementation Requirements

```rust
use renacer::{Tracer, TraceEvent, Flamegraph, SourceCorrelator};

pub struct DeepTraceConfig {
    pub categories: Vec<TraceCategory>,
    pub sample_rate: f64,          // 0.0-1.0
    pub max_events: usize,
    pub source_map: Option<PathBuf>,
    pub wasm_source_map: Option<PathBuf>,
}

pub enum TraceCategory {
    Syscall,
    Wasm,
    Network,
    Memory,
    Gpu,
    Custom(String),
}

pub struct TraceAnalysis {
    pub timeline: Vec<TraceSpan>,
    pub syscall_breakdown: HashMap<String, SyscallStats>,
    pub wasm_events: Vec<WasmEvent>,
    pub source_hotspots: Vec<SourceHotspot>,
    pub flamegraph: Flamegraph,
}

pub struct SourceHotspot {
    pub file: PathBuf,
    pub line: u32,
    pub function: String,
    pub total_time: Duration,
    pub call_count: u64,
    pub suggestion: Option<OptimizationSuggestion>,
}

pub enum OptimizationSuggestion {
    UseSIMD { expected_speedup: f64 },
    UsePool { current_allocs: u64 },
    BatchOperations { current_calls: u64 },
    AsyncIO { blocking_time: Duration },
}
```

---

### K. Simulation Playback (simular Integration)

#### K.1 Command Interface

```bash
# Record load test session
probador load-test --scenario boot.yaml --record session.simular

# Replay with different parameters
probador simulate replay session.simular --users 2x --network "3G"

# Monte Carlo analysis (1000 iterations)
probador simulate monte-carlo session.simular --iterations 1000

# Generate scenario variations
probador simulate variations session.simular --output variations/
```

#### K.2 Simulation Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| **Deterministic Replay** | Exact reproduction of recorded session | Bug reproduction |
| **Parameterized Replay** | Replay with modified parameters | Capacity planning |
| **Monte Carlo** | Randomized variations with statistical analysis | Risk assessment |
| **Chaos Engineering** | Inject failures and delays | Resilience testing |

#### K.3 Monte Carlo Analysis Output

```
MONTE CARLO SIMULATION: WASM Boot Sequence
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Configuration:
  Base session: session_20241214.simular
  Iterations: 1,000
  Varied parameters:
    - Network latency: ±50ms (normal distribution)
    - Packet loss: 0-5% (uniform distribution)
    - User count: 80-120 (uniform distribution)

LATENCY DISTRIBUTION (p95 across iterations)
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  count                                                                   │
│   200 │                    ████                                          │
│   150 │                  ████████                                        │
│   100 │                ████████████                                      │
│    50 │            ██████████████████                                    │
│     0 └──────────────────────────────────────────────────────────────    │
│        150ms    200ms    250ms    300ms    350ms    400ms    p95         │
│                                                                          │
│  Mean: 234ms    StdDev: 45ms    95% CI: [189ms, 312ms]                  │
└─────────────────────────────────────────────────────────────────────────┘

FAILURE PROBABILITY
┌─────────────────────────────────────────────────────────────────────────┐
│ SLA Target           │ Probability of Meeting │ Risk Level              │
│──────────────────────┼────────────────────────┼─────────────────────────│
│ p95 < 200ms          │ 23.4%                  │ ███████████████ HIGH    │
│ p95 < 300ms          │ 89.2%                  │ ████ LOW                │
│ p95 < 500ms          │ 99.1%                  │ █ MINIMAL               │
│ Error rate < 1%      │ 97.8%                  │ █ MINIMAL               │
│ Error rate < 0.1%    │ 45.6%                  │ ██████████ MEDIUM       │
└─────────────────────────────────────────────────────────────────────────┘

SENSITIVITY ANALYSIS (which parameters matter most)
┌─────────────────────────────────────────────────────────────────────────┐
│ Parameter            │ Correlation with p95 │ Impact                    │
│──────────────────────┼──────────────────────┼───────────────────────────│
│ Network latency      │ r = 0.82             │ ████████████████████ HIGH │
│ User count           │ r = 0.45             │ ████████████ MEDIUM       │
│ Packet loss          │ r = 0.34             │ █████████ MEDIUM          │
│ Server load          │ r = 0.12             │ ███ LOW                   │
└─────────────────────────────────────────────────────────────────────────┘

RECOMMENDATIONS
1. Network latency dominates p95 variance - consider CDN for WASM assets
2. Current configuration has 76.6% chance of missing 200ms SLA
3. To achieve 95% confidence of meeting 200ms SLA:
   - Reduce network latency by 30% (CDN edge caching)
   - OR reduce WASM binary size by 40%
```

#### K.4 Implementation Requirements

```rust
use simular::{Simulation, MonteCarlo, Distribution, Scenario};

pub struct SimulationConfig {
    pub base_session: PathBuf,
    pub mode: SimulationMode,
    pub parameters: HashMap<String, ParameterVariation>,
    pub output: SimulationOutput,
}

pub enum SimulationMode {
    DeterministicReplay,
    Parameterized { multipliers: HashMap<String, f64> },
    MonteCarlo { iterations: u32, seed: Option<u64> },
    Chaos { failure_injections: Vec<FailureInjection> },
}

pub struct ParameterVariation {
    pub distribution: Distribution,
    pub min: f64,
    pub max: f64,
}

pub enum Distribution {
    Normal { mean: f64, std_dev: f64 },
    Uniform,
    Exponential { lambda: f64 },
    Poisson { lambda: f64 },
}

pub struct MonteCarloResult {
    pub iterations: u32,
    pub latency_distribution: Histogram,
    pub sla_probabilities: HashMap<String, f64>,
    pub sensitivity_analysis: Vec<SensitivityFactor>,
    pub recommendations: Vec<String>,
}

pub struct SensitivityFactor {
    pub parameter: String,
    pub correlation: f64,
    pub impact_level: ImpactLevel,
}
```

---

## QA Checklist Extension (20 points)

### H. Enhanced Visualization (5 points)

| # | Falsification Attempt | Expected Result | Pass/Fail |
|---|----------------------|-----------------|-----------|
| H01 | Run `--viz` during load test | Real-time TUI updates | |
| H02 | Export MessagePack report | Valid binary, loads in TUI viewer | |
| H03 | Stream to NDJSON file | Metrics appendable, parseable | |
| H04 | Resize terminal during `--viz` | Dashboard resizes correctly | |
| H05 | Export JSON report | Valid JSON, schema-compliant | |

### I. Statistical Analysis (5 points)

| # | Falsification Attempt | Expected Result | Pass/Fail |
|---|----------------------|-----------------|-----------|
| I01 | Generate variance tree | Components sum to total | |
| I02 | Calculate Apdex score | Score between 0.0 and 1.0 | |
| I03 | Detect throughput knee | Knee point identified | |
| I04 | Compare two test runs | Statistical significance reported | |
| I05 | Quantile regression | Tail attributions provided | |

### J. Deep Tracing (5 points)

| # | Falsification Attempt | Expected Result | Pass/Fail |
|---|----------------------|-----------------|-----------|
| J01 | Enable tracing during load test | Trace file created | |
| J02 | Generate flamegraph | Valid SVG output | |
| J03 | Correlate with source | Line numbers accurate | |
| J04 | Capture WASM-specific events | memory_grow, compile visible | |
| J05 | Syscall breakdown | Counts match strace | |

### K. Simulation Playback (5 points)

| # | Falsification Attempt | Expected Result | Pass/Fail |
|---|----------------------|-----------------|-----------|
| K01 | Record and replay session | Deterministic results | |
| K02 | Parameterized replay with 2x users | Latency scales appropriately | |
| K03 | Monte Carlo with 1000 iterations | Distribution statistics valid | |
| K04 | Sensitivity analysis | Correlations mathematically correct | |
| K05 | Chaos injection (50% packet loss) | System degrades gracefully | |

---

## Implementation Priority

| Phase | Feature | Stack Integration | Effort | Impact |
|-------|---------|-------------------|--------|--------|
| 1 | H (Visualization) | trueno-viz | Medium | High |
| 2 | I (Statistics) | trueno | Medium | High |
| 3 | J (Deep Tracing) | renacer | High | Critical |
| 4 | K (Simulation) | simular | High | High |

---

## References

1. Huang, J., et al. "Statistical Analysis of Latency Through Semantic Profiling." EuroSys 2017.
2. Zhang, Y., et al. "Treadmill: Attributing the Source of Tail Latency." SIGARCH 2016.
3. Jangda, A., et al. "Revealing Performance Issues in Server-side WebAssembly Runtimes." ASE 2023.
4. Lehmann, D., et al. "Wasm-R3: Record-Reduce-Replay for WASM Benchmarks." OOPSLA 2024.
5. Dean, J., & Barroso, L.A. "The Tail at Scale." CACM 2013.
6. [Locust Documentation](https://locust.io/)
7. [k6 Documentation](https://k6.io/)
8. [Gatling Documentation](https://gatling.io/docs/)
