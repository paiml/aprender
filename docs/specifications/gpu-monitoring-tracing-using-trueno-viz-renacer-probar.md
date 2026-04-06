# TRUENO-SPEC-010: GPU Monitoring, Tracing, and Visualization

**Status**: DRAFT
**Version**: 1.1.0
**Created**: 2024-12-14
**Updated**: 2024-12-14
**Authors**: PAIML Team
**Related Projects**: trueno, trueno-gpu, renacer, probar, trueno-viz

---

## Abstract

This specification defines a comprehensive GPU monitoring, tracing, and visualization system for the trueno ecosystem. The system supports **two GPU backends**:

1. **trueno** (wgpu-based) - Cross-platform WebGPU abstraction (Vulkan/Metal/DX12)
2. **trueno-gpu** (native CUDA) - Direct NVIDIA CUDA Driver API with PTX generation

Following industry best practices from btop, llama.cpp, and NVIDIA's NVML, the system provides real-time hardware telemetry, distributed tracing integration via renacer/OpenTelemetry, and visual coverage analysis through probar. The design prioritizes zero-overhead instrumentation when disabled, full NVIDIA feature support via trueno-gpu, cross-platform compatibility via trueno, and integration with existing Rust observability ecosystems.

---

## 1. Introduction

### 1.1 Problem Statement

Current GPU compute libraries, including trueno, lack visibility into runtime hardware utilization. Developers cannot:

1. **Identify which GPU** is being used (critical for multi-GPU systems)
2. **Monitor resource consumption** (VRAM, compute utilization, temperature)
3. **Trace kernel execution** for performance debugging
4. **Correlate GPU operations** with application-level spans

This gap is significant: NVIDIA's RTX 4090 costs $1,599+ and draws 450W TDP. Without monitoring, developers waste both money and energy on suboptimal GPU utilization.

### 1.2 Industry Context

Leading projects have established monitoring patterns:

| Project | Approach | Metrics Collected |
|---------|----------|-------------------|
| **btop** | NVML/ROCm SMI via dlopen | utilization, memory, temp, power, clocks |
| **llama.cpp** | Vulkan properties + memory logging | device name, VRAM, allocation tracking |
| **PyTorch** | CUDA events + profiler | kernel timing, memory snapshots |
| **TensorRT** | NVTX annotations | layer-by-layer profiling |

### 1.3 Dual-Backend Architecture

The trueno ecosystem provides two complementary GPU backends:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Application Layer                                  │
│                  (probar, renacer, user applications)                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                         trueno-monitor (this spec)                          │
│                    Unified API for GPU metrics & tracing                    │
├────────────────────────────────┬────────────────────────────────────────────┤
│          trueno (wgpu)         │            trueno-gpu (CUDA)               │
│   Cross-platform abstraction   │      Native NVIDIA PTX/SASS execution      │
├────────────────────────────────┼────────────────────────────────────────────┤
│   • Vulkan (Linux/Windows)     │   • cuDeviceGetName → "RTX 4090"           │
│   • Metal (macOS/iOS)          │   • cuMemGetInfo → free/total VRAM         │
│   • DX12 (Windows)             │   • cuMemAlloc/cuMemFree → tracking        │
│   • WebGPU (WASM)              │   • cuLaunchKernel → kernel timing         │
│   • adapter.get_info()         │   • cuCtxSynchronize → sync points         │
├────────────────────────────────┴────────────────────────────────────────────┤
│                              Hardware Layer                                  │
│        NVIDIA (RTX 4090, A100, H100)  |  AMD  |  Intel  |  Apple Silicon    │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Backend Selection Strategy:**

| Scenario | Recommended Backend | Rationale |
|----------|---------------------|-----------|
| NVIDIA production ML | trueno-gpu | Full CUDA features, kernel profiling, PTX optimization |
| Cross-platform app | trueno | Works on all platforms via wgpu |
| WASM deployment | trueno | WebGPU native |
| Multi-vendor fleet | trueno | Consistent API across vendors |
| Maximum NVIDIA perf | trueno-gpu | Direct driver access, no abstraction overhead |

### 1.4 Scope

This specification covers:

1. **Dual-Backend Device Discovery** (Section 3)
2. **Real-Time Metrics Collection** (Section 4)
3. **Distributed Tracing Integration** (Section 5)
4. **Visualization with probar** (Section 6)
5. **Cross-Platform Support Matrix** (Section 7)
6. **trueno-gpu CUDA Integration** (Section 8)

---

## 2. References

### 2.1 Peer-Reviewed Citations

1. **Nickolls, J., Buck, I., Garland, M., & Skadron, K. (2008)**. "Scalable Parallel Programming with CUDA." *ACM Queue*, 6(2), 40-53. https://doi.org/10.1145/1365490.1365500
   - Foundational GPU computing model; establishes kernel-based execution paradigm for monitoring

2. **Gregg, B. (2016)**. "The Flame Graph." *Communications of the ACM*, 59(6), 48-57. https://doi.org/10.1145/2909476
   - Visualization methodology for hierarchical performance data; applicable to GPU kernel traces

3. **Arafa, Y., Badawy, A.-H., Chennupati, G., Santhi, N., & Eidenbenz, S. (2020)**. "PPT-GPU: Scalable GPU Performance Modeling." *IEEE Computer Architecture Letters*, 19(1), 55-58. https://doi.org/10.1109/LCA.2020.2988392
   - GPU performance modeling techniques for predictive monitoring

4. **Jeon, M., Venkataraman, S., Phanishayee, A., Qian, J., Xiao, W., & Yang, F. (2019)**. "Analysis of Large-Scale Multi-Tenant GPU Clusters for DNN Training Workloads." *USENIX ATC '19*, 947-960.
   - Multi-GPU monitoring strategies for shared systems; informs device enumeration

5. **Burtscher, M., Zecena, I., & Zong, Z. (2014)**. "Measuring GPU Power with the K20 Built-in Sensor." *Workshop on General Purpose Processing Using GPUs (GPGPU-7)*, 28-36. https://doi.org/10.1145/2588768.2576783
   - Hardware power measurement validation; establishes accuracy requirements for power monitoring

6. **Hong, S., & Kim, H. (2009)**. "An Analytical Model for a GPU Architecture with Memory-Level and Thread-Level Parallelism Awareness." *ACM SIGARCH Computer Architecture News*, 37(3), 152-163. https://doi.org/10.1145/1555815.1555775
   - Memory bandwidth modeling; informs VRAM utilization metrics

7. **Kaldewey, T., Lohman, G., Mueller, R., & Volk, P. (2012)**. "GPU Join Processing Revisited." *ACM SIGMOD '12*, 55-66. https://doi.org/10.1145/2213836.2213844
   - GPU kernel execution patterns; validates timing measurement approaches

8. **Sigelman, B. H., Barroso, L. A., Burrows, M., et al. (2010)**. "Dapper, a Large-Scale Distributed Systems Tracing Infrastructure." *Google Technical Report*. https://research.google/pubs/pub36356/
   - Distributed tracing fundamentals; basis for OpenTelemetry integration design

9. **Micikevicius, P. (2010)**. "Analysis-Driven Optimization." *GPU Technology Conference 2010*, NVIDIA.
   - Profiling-driven optimization methodology; establishes feedback loop requirements

10. **Che, S., Boyer, M., Meng, J., Tarjan, D., Sheaffer, J. W., Lee, S.-H., & Skadron, K. (2009)**. "Rodinia: A Benchmark Suite for Heterogeneous Computing." *IEEE International Symposium on Workload Characterization (IISWC)*, 44-54. https://doi.org/10.1109/IISWC.2009.5306797
    - GPU benchmarking methodology; validates performance measurement approaches

### 2.2 Industry Standards

- **NVIDIA NVML**: NVIDIA Management Library v12.x
- **AMD ROCm SMI**: ROCm System Management Interface v6.x
- **WebGPU Specification**: W3C Working Draft (2024)
- **OpenTelemetry Specification**: CNCF v1.x
- **Vulkan 1.3**: Khronos Group (device properties queries)

---

## 3. GPU Device Discovery & Information

### 3.1 Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| GPU-001 | Query device name (e.g., "NVIDIA GeForce RTX 4090") | P0 |
| GPU-002 | Query total VRAM capacity | P0 |
| GPU-003 | Query compute capability/architecture | P1 |
| GPU-004 | Enumerate all available GPUs | P1 |
| GPU-005 | Detect vendor (NVIDIA/AMD/Intel/Apple) | P0 |
| GPU-006 | Query driver version | P2 |
| GPU-007 | Query PCI bus ID (for multi-GPU systems) | P2 |

### 3.2 API Design

```rust
/// GPU device information (TRUENO-SPEC-010)
#[derive(Debug, Clone)]
pub struct GpuDeviceInfo {
    /// Device index (0-based)
    pub index: u32,
    /// Device name (e.g., "NVIDIA GeForce RTX 4090")
    pub name: String,
    /// Vendor identifier
    pub vendor: GpuVendor,
    /// Total VRAM in bytes
    pub vram_total: u64,
    /// Compute capability (NVIDIA) or architecture info
    pub compute_capability: Option<(u32, u32)>,
    /// Driver version string
    pub driver_version: Option<String>,
    /// PCI bus ID (e.g., "0000:01:00.0")
    pub pci_bus_id: Option<String>,
    /// wgpu backend being used
    pub backend: GpuBackend,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuVendor {
    Nvidia,
    Amd,
    Intel,
    Apple,
    Unknown(u32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuBackend {
    Vulkan,
    Metal,
    Dx12,
    Dx11,
    WebGpu,
}

impl GpuDevice {
    /// Query device information
    pub fn device_info(&self) -> GpuDeviceInfo;

    /// Enumerate all available GPU devices
    pub fn enumerate_devices() -> Vec<GpuDeviceInfo>;

    /// Get the currently active device info
    pub fn current_device_info() -> Option<GpuDeviceInfo>;
}
```

### 3.3 Implementation Strategy

**Primary Path (wgpu adapter info)**:
```rust
// wgpu provides adapter info directly
let adapter_info = adapter.get_info();
GpuDeviceInfo {
    name: adapter_info.name,
    vendor: match adapter_info.vendor {
        0x10de => GpuVendor::Nvidia,
        0x1002 => GpuVendor::Amd,
        0x8086 => GpuVendor::Intel,
        0x106b => GpuVendor::Apple,
        v => GpuVendor::Unknown(v),
    },
    backend: match adapter_info.backend {
        wgpu::Backend::Vulkan => GpuBackend::Vulkan,
        wgpu::Backend::Metal => GpuBackend::Metal,
        wgpu::Backend::Dx12 => GpuBackend::Dx12,
        // ...
    },
    // ...
}
```

**Enhanced Path (NVML for NVIDIA)**:
For NVIDIA GPUs, optionally load libnvidia-ml.so at runtime (btop pattern) for extended metrics not available via wgpu.

---

## 4. Real-Time Metrics Collection

### 4.1 Metrics Categories

#### 4.1.1 Utilization Metrics
| Metric | Unit | Source | Update Frequency |
|--------|------|--------|------------------|
| GPU Compute Utilization | % (0-100) | NVML/ROCm | 100ms |
| Memory Controller Utilization | % (0-100) | NVML/ROCm | 100ms |
| Encoder Utilization | % (0-100) | NVML | 100ms |
| Decoder Utilization | % (0-100) | NVML | 100ms |

#### 4.1.2 Memory Metrics
| Metric | Unit | Source | Update Frequency |
|--------|------|--------|------------------|
| VRAM Used | bytes | wgpu/NVML | 100ms |
| VRAM Free | bytes | wgpu/NVML | 100ms |
| VRAM Reserved | bytes | NVML | 100ms |
| Allocation Count | count | trueno internal | per-op |

#### 4.1.3 Thermal & Power Metrics
| Metric | Unit | Source | Update Frequency |
|--------|------|--------|------------------|
| GPU Temperature | Celsius | NVML/ROCm | 500ms |
| Power Draw | Watts | NVML/ROCm | 100ms |
| Power Limit | Watts | NVML/ROCm | static |
| Fan Speed | % RPM | NVML/ROCm | 500ms |

#### 4.1.4 Clock Metrics
| Metric | Unit | Source | Update Frequency |
|--------|------|--------|------------------|
| Graphics Clock | MHz | NVML/ROCm | 100ms |
| Memory Clock | MHz | NVML/ROCm | 100ms |
| SM Clock | MHz | NVML | 100ms |

#### 4.1.5 PCIe Metrics
| Metric | Unit | Source | Update Frequency |
|--------|------|--------|------------------|
| PCIe TX Throughput | bytes/s | NVML/ROCm | 100ms |
| PCIe RX Throughput | bytes/s | NVML/ROCm | 100ms |
| PCIe Link Gen | 1-5 | NVML | static |
| PCIe Link Width | lanes | NVML | static |

### 4.2 API Design

```rust
/// Real-time GPU metrics snapshot
#[derive(Debug, Clone)]
pub struct GpuMetrics {
    /// Timestamp of measurement
    pub timestamp: std::time::Instant,
    /// Device index
    pub device_index: u32,
    /// Utilization metrics
    pub utilization: GpuUtilization,
    /// Memory metrics
    pub memory: GpuMemoryMetrics,
    /// Thermal metrics (if available)
    pub thermal: Option<GpuThermalMetrics>,
    /// Power metrics (if available)
    pub power: Option<GpuPowerMetrics>,
    /// Clock metrics (if available)
    pub clocks: Option<GpuClockMetrics>,
    /// PCIe metrics (if available)
    pub pcie: Option<GpuPcieMetrics>,
}

#[derive(Debug, Clone)]
pub struct GpuUtilization {
    pub gpu_percent: u32,
    pub memory_percent: u32,
    pub encoder_percent: Option<u32>,
    pub decoder_percent: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct GpuMemoryMetrics {
    pub total: u64,
    pub used: u64,
    pub free: u64,
    pub allocations: u64,
}

#[derive(Debug, Clone)]
pub struct GpuThermalMetrics {
    pub temperature_celsius: u32,
    pub temperature_threshold_shutdown: Option<u32>,
    pub fan_speed_percent: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct GpuPowerMetrics {
    pub power_draw_watts: f32,
    pub power_limit_watts: f32,
    pub power_state: u32,
}

#[derive(Debug, Clone)]
pub struct GpuClockMetrics {
    pub graphics_mhz: u32,
    pub memory_mhz: u32,
    pub sm_mhz: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct GpuPcieMetrics {
    pub tx_bytes_per_sec: u64,
    pub rx_bytes_per_sec: u64,
    pub link_gen: u32,
    pub link_width: u32,
}

/// GPU metrics collector
pub struct GpuMonitor {
    // ...
}

impl GpuMonitor {
    /// Create a new monitor for the specified device
    pub fn new(device_index: u32) -> Result<Self, GpuMonitorError>;

    /// Collect current metrics snapshot
    pub fn collect(&self) -> Result<GpuMetrics, GpuMonitorError>;

    /// Start background collection at specified interval
    pub fn start_background(
        &self,
        interval: Duration,
        callback: impl Fn(GpuMetrics) + Send + 'static,
    ) -> GpuMonitorHandle;

    /// Get metrics history (ring buffer)
    pub fn history(&self, duration: Duration) -> Vec<GpuMetrics>;
}
```

### 4.3 NVML Integration (btop pattern)

```rust
/// NVML bindings loaded at runtime (optional feature)
#[cfg(feature = "nvml")]
mod nvml {
    use std::ffi::c_void;

    type NvmlReturn = i32;
    type NvmlDevice = *mut c_void;

    pub struct NvmlHandle {
        lib: libloading::Library,
        // Function pointers
        init: unsafe extern "C" fn() -> NvmlReturn,
        shutdown: unsafe extern "C" fn() -> NvmlReturn,
        device_get_count: unsafe extern "C" fn(*mut u32) -> NvmlReturn,
        device_get_handle: unsafe extern "C" fn(u32, *mut NvmlDevice) -> NvmlReturn,
        device_get_name: unsafe extern "C" fn(NvmlDevice, *mut i8, u32) -> NvmlReturn,
        device_get_utilization: unsafe extern "C" fn(NvmlDevice, *mut NvmlUtilization) -> NvmlReturn,
        device_get_memory_info: unsafe extern "C" fn(NvmlDevice, *mut NvmlMemory) -> NvmlReturn,
        device_get_temperature: unsafe extern "C" fn(NvmlDevice, u32, *mut u32) -> NvmlReturn,
        device_get_power_usage: unsafe extern "C" fn(NvmlDevice, *mut u32) -> NvmlReturn,
        // ... additional function pointers
    }

    impl NvmlHandle {
        pub fn load() -> Result<Self, NvmlError> {
            // Try multiple library paths (btop pattern)
            let lib_paths = [
                "libnvidia-ml.so.1",
                "libnvidia-ml.so",
                "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1",
            ];

            for path in lib_paths {
                if let Ok(lib) = unsafe { libloading::Library::new(path) } {
                    return Self::init_from_lib(lib);
                }
            }

            Err(NvmlError::LibraryNotFound)
        }
    }
}
```

---

## 5. Distributed Tracing Integration

### 5.1 OpenTelemetry Integration (via renacer)

The monitoring system integrates with OpenTelemetry for distributed tracing, allowing GPU operations to be correlated with application-level spans.

```rust
use opentelemetry::{global, trace::Tracer};
use tracing_opentelemetry::OpenTelemetryLayer;

/// GPU operation span attributes
pub struct GpuSpanAttributes {
    /// Operation name (e.g., "matmul", "relu", "softmax")
    pub operation: String,
    /// Input tensor dimensions
    pub input_dims: Vec<usize>,
    /// Output tensor dimensions
    pub output_dims: Vec<usize>,
    /// Device index
    pub device_index: u32,
    /// VRAM allocated for this operation
    pub vram_allocated: u64,
    /// Kernel execution time (microseconds)
    pub kernel_time_us: u64,
}

/// Instrumented GPU device with tracing
pub struct TracedGpuDevice {
    inner: GpuDevice,
    tracer: BoxedTracer,
}

impl TracedGpuDevice {
    /// Execute operation with tracing span
    #[tracing::instrument(
        name = "gpu.matmul",
        fields(
            gpu.device = %self.device_info().name,
            gpu.vram_used = tracing::field::Empty,
            gpu.kernel_time_us = tracing::field::Empty,
        )
    )]
    pub fn matmul(&self, a: &[f32], b: &[f32], result: &mut [f32], m: usize, k: usize, n: usize) -> Result<(), GpuError> {
        let span = tracing::Span::current();
        let start = std::time::Instant::now();

        // Execute operation
        let result = self.inner.matmul(a, b, result, m, k, n);

        // Record metrics
        let elapsed = start.elapsed();
        span.record("gpu.kernel_time_us", elapsed.as_micros() as u64);
        span.record("gpu.vram_used", self.inner.vram_used());

        result
    }
}
```

### 5.2 Span Hierarchy

```
application_request (HTTP/gRPC)
└── model_inference
    ├── gpu.preprocess
    │   ├── gpu.resize [device=0, kernel_time=120us]
    │   └── gpu.normalize [device=0, kernel_time=45us]
    ├── gpu.forward_pass
    │   ├── gpu.matmul [device=0, kernel_time=2.3ms, vram=128MB]
    │   ├── gpu.relu [device=0, kernel_time=89us]
    │   ├── gpu.matmul [device=0, kernel_time=1.8ms, vram=64MB]
    │   └── gpu.softmax [device=0, kernel_time=156us]
    └── gpu.postprocess
        └── gpu.argmax [device=0, kernel_time=23us]
```

### 5.3 Exporters

| Exporter | Protocol | Use Case |
|----------|----------|----------|
| Jaeger | gRPC/HTTP | Development debugging |
| Tempo | gRPC | Production at scale |
| Zipkin | HTTP | Legacy systems |
| OTLP | gRPC/HTTP | Cloud-native (preferred) |

### 5.4 Integration with renacer

Renacer provides syscall-level tracing that can be correlated with GPU operations:

```rust
// GPU operation triggers syscalls that renacer captures
// Correlation via trace context propagation

use renacer::TraceContext;

impl TracedGpuDevice {
    pub fn execute_with_syscall_correlation<F, R>(
        &self,
        ctx: &TraceContext,
        op: F,
    ) -> R
    where
        F: FnOnce(&GpuDevice) -> R,
    {
        // Inject trace context for syscall correlation
        renacer::with_context(ctx, || op(&self.inner))
    }
}
```

---

## 6. Visualization with probar

### 6.1 Real-Time TUI Dashboard

Integration with probar's terminal heatmap capabilities for GPU monitoring visualization:

```rust
use jugar_probar::pixel_coverage::{
    GpuPixelBuffer, TerminalHeatmap, ansi,
};

/// GPU metrics dashboard using probar's TUI
pub struct GpuDashboard {
    buffer: GpuPixelBuffer,
    metrics_history: Vec<GpuMetrics>,
    config: DashboardConfig,
}

impl GpuDashboard {
    /// Render real-time GPU utilization heatmap
    pub fn render_utilization_heatmap(&self) -> String {
        // Map utilization history to pixel coverage format
        let width = self.config.heatmap_width;
        let height = self.config.heatmap_height;

        // Time on X-axis, utilization on Y-axis
        for (x, metrics) in self.metrics_history.iter().enumerate() {
            let y = (metrics.utilization.gpu_percent as usize * height) / 100;
            self.buffer.set_pixel(x, height - y - 1, 1.0);
        }

        self.buffer.render_terminal()
    }

    /// Render VRAM usage bar
    pub fn render_vram_bar(&self, width: usize) -> String {
        if let Some(metrics) = self.metrics_history.last() {
            let used_pct = (metrics.memory.used as f64 / metrics.memory.total as f64) * 100.0;
            let filled = (used_pct as usize * width) / 100;

            let color = if used_pct > 90.0 {
                ansi::FAIL
            } else if used_pct > 70.0 {
                ansi::WARN
            } else {
                ansi::PASS
            };

            format!(
                "VRAM: [{}{:█<filled$}{:░<empty$}{}] {:.1}% ({}/{})",
                color,
                "",
                "",
                ansi::RESET,
                used_pct,
                format_bytes(metrics.memory.used),
                format_bytes(metrics.memory.total),
                filled = filled,
                empty = width - filled,
            )
        } else {
            "VRAM: [No data]".to_string()
        }
    }

    /// Render temperature gauge
    pub fn render_temperature(&self) -> String {
        if let Some(metrics) = self.metrics_history.last() {
            if let Some(thermal) = &metrics.thermal {
                let temp = thermal.temperature_celsius;
                let (color, status) = match temp {
                    0..=50 => (ansi::PASS, "COOL"),
                    51..=70 => (ansi::INFO, "WARM"),
                    71..=85 => (ansi::WARN, "HOT"),
                    _ => (ansi::FAIL, "CRITICAL"),
                };

                format!("Temp: {}{}°C ({}){}", color, temp, status, ansi::RESET)
            } else {
                "Temp: N/A".to_string()
            }
        } else {
            "Temp: [No data]".to_string()
        }
    }
}
```

### 6.2 Dashboard Layout (btop-inspired)

```
┌─ GPU: NVIDIA GeForce RTX 4090 ────────────────────────────────────────────┐
│                                                                           │
│  Utilization                      Memory                                  │
│  ┌────────────────────────┐       ┌────────────────────────┐             │
│  │ GPU:  [████████░░] 78% │       │ Used:    18.2 GB / 24 GB             │
│  │ Mem:  [██████░░░░] 62% │       │ Free:     5.8 GB                     │
│  │ Enc:  [░░░░░░░░░░]  0% │       │ Allocs:  1,247                       │
│  │ Dec:  [░░░░░░░░░░]  0% │       │ [████████████████░░░░░░░░] 75.8%     │
│  └────────────────────────┘       └────────────────────────┘             │
│                                                                           │
│  Power & Thermal                  Clocks                                  │
│  ┌────────────────────────┐       ┌────────────────────────┐             │
│  │ Power: 287W / 450W TDP │       │ Graphics: 2520 MHz                   │
│  │ Temp:  67°C (WARM)     │       │ Memory:   10501 MHz                  │
│  │ Fan:   45%             │       │ SM:       2520 MHz                   │
│  └────────────────────────┘       └────────────────────────┘             │
│                                                                           │
│  Utilization History (last 60s)                                          │
│  100% ┤                                                                   │
│   75% ┤        ████                    ██████████                        │
│   50% ┤   █████████████       █████████████████████████                  │
│   25% ┤████████████████████████████████████████████████████████          │
│    0% ┼────────────────────────────────────────────────────────→ time    │
│                                                                           │
│  PCIe: Gen4 x16 | TX: 12.4 GB/s | RX: 8.7 GB/s                          │
└───────────────────────────────────────────────────────────────────────────┘
```

### 6.3 Kernel Flamegraph Export

```rust
/// Export GPU kernel execution as flamegraph-compatible format
pub fn export_flamegraph(spans: &[GpuSpan]) -> String {
    let mut output = String::new();

    for span in spans {
        // Format: stack;frame;frame count
        let stack = span.stack.join(";");
        let count = span.duration_us;
        writeln!(output, "{} {}", stack, count).ok();
    }

    output
}

// Example output:
// main;model_inference;gpu.forward_pass;gpu.matmul 2300
// main;model_inference;gpu.forward_pass;gpu.relu 89
// main;model_inference;gpu.forward_pass;gpu.softmax 156
```

---

## 7. Cross-Platform Support Matrix

### 7.1 Feature Availability

| Feature | NVIDIA (Linux) | NVIDIA (Windows) | AMD (Linux) | AMD (Windows) | Intel | Apple Silicon |
|---------|---------------|------------------|-------------|---------------|-------|---------------|
| Device Name | ✅ wgpu + NVML | ✅ wgpu + NVML | ✅ wgpu + ROCm | ✅ wgpu | ✅ wgpu | ✅ wgpu |
| VRAM Total/Used | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| GPU Utilization | ✅ NVML | ✅ NVML | ✅ ROCm SMI | ❌ | ❌ | ❌ |
| Temperature | ✅ NVML | ✅ NVML | ✅ ROCm SMI | ❌ | ❌ | ❌ |
| Power Draw | ✅ NVML | ✅ NVML | ✅ ROCm SMI | ❌ | ❌ | ❌ |
| Clock Speeds | ✅ NVML | ✅ NVML | ✅ ROCm SMI | ❌ | ❌ | ❌ |
| PCIe Throughput | ✅ NVML | ✅ NVML | ✅ ROCm SMI | ❌ | ❌ | ❌ |

### 7.2 Backend Selection

```rust
/// Automatic backend selection based on platform
pub fn select_optimal_backend() -> GpuBackend {
    #[cfg(target_os = "macos")]
    return GpuBackend::Metal;

    #[cfg(target_os = "windows")]
    {
        // Prefer Vulkan on Windows for consistency, fall back to DX12
        if vulkan_available() {
            return GpuBackend::Vulkan;
        }
        return GpuBackend::Dx12;
    }

    #[cfg(target_os = "linux")]
    return GpuBackend::Vulkan;

    #[cfg(target_arch = "wasm32")]
    return GpuBackend::WebGpu;
}
```

---

## 8. Configuration

### 8.1 Feature Flags

```toml
[features]
default = []
# Enable NVML monitoring for NVIDIA GPUs
nvml = []
# Enable ROCm SMI monitoring for AMD GPUs
rocm = []
# Enable distributed tracing integration
tracing = ["opentelemetry", "tracing-opentelemetry"]
# Enable TUI dashboard with probar
dashboard = ["jugar-probar"]
# Enable all monitoring features
full-monitoring = ["nvml", "rocm", "tracing", "dashboard"]
```

### 8.2 Runtime Configuration

```rust
/// Monitoring configuration
#[derive(Debug, Clone)]
pub struct MonitorConfig {
    /// Polling interval for metrics collection
    pub poll_interval: Duration,
    /// History buffer size (number of samples to retain)
    pub history_size: usize,
    /// Enable background collection thread
    pub background_collection: bool,
    /// Export format for traces
    pub trace_exporter: TraceExporter,
    /// Dashboard refresh rate
    pub dashboard_refresh: Duration,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            poll_interval: Duration::from_millis(100),
            history_size: 600,  // 60 seconds at 100ms
            background_collection: true,
            trace_exporter: TraceExporter::None,
            dashboard_refresh: Duration::from_millis(500),
        }
    }
}
```

---

## 9. Performance Considerations

### 9.1 Overhead Budget

| Component | Target Overhead | Measurement Method |
|-----------|----------------|---------------------|
| Metrics Collection | < 0.1% CPU | Background thread, 100ms polling |
| NVML Queries | < 1ms per call | Direct library calls |
| Span Creation | < 100ns | Async batching |
| Dashboard Render | < 10ms | Incremental updates |

### 9.2 Zero-Cost Abstraction

When monitoring is disabled, all instrumentation compiles to no-ops:

```rust
#[cfg(feature = "tracing")]
macro_rules! gpu_span {
    ($name:expr, $($field:tt)*) => {
        tracing::span!(tracing::Level::INFO, $name, $($field)*)
    };
}

#[cfg(not(feature = "tracing"))]
macro_rules! gpu_span {
    ($name:expr, $($field:tt)*) => {
        // Compiles to nothing
    };
}
```

---

## 10. Testing Requirements

### 10.1 Unit Tests

- [ ] Device enumeration returns valid device list
- [ ] Metrics collection doesn't panic on missing data
- [ ] History buffer correctly implements ring buffer semantics
- [ ] Span attributes are correctly recorded

### 10.2 Integration Tests

- [ ] NVML integration works on NVIDIA systems
- [ ] ROCm integration works on AMD systems
- [ ] Traces export correctly to Jaeger
- [ ] Dashboard renders correctly with live data

### 10.3 Benchmarks

- [ ] Metrics collection overhead < 0.1%
- [ ] NVML query latency < 1ms
- [ ] Span creation overhead < 100ns

---

## 8. trueno-gpu CUDA Integration

### 8.1 Native CUDA Driver API

trueno-gpu provides direct access to NVIDIA's CUDA Driver API through a hand-written FFI layer (~400 lines). This enables:

```rust
use trueno_gpu::driver::{CudaContext, CudaModule, CudaStream, GpuBuffer};

// Initialize CUDA context (auto-detects GPU)
let ctx = CudaContext::new(0)?;

// Get device information - REAL hardware info, not abstraction
println!("GPU: {}", ctx.device_name()?);  // "NVIDIA GeForce RTX 4090"
println!("Total VRAM: {} MB", ctx.total_memory()? / (1024 * 1024));  // 24045 MB
```

### 8.2 trueno-gpu Device Discovery API

```rust
/// Existing trueno-gpu API (from driver/context.rs)
impl CudaContext {
    /// Get device name (e.g., "NVIDIA GeForce RTX 4090")
    pub fn device_name(&self) -> Result<String, GpuError>;

    /// Get total device memory in bytes
    pub fn total_memory(&self) -> Result<usize, GpuError>;

    /// Synchronize all GPU operations
    pub fn synchronize(&self) -> Result<(), GpuError>;
}

/// Device enumeration (from driver/context.rs)
pub fn device_count() -> Result<usize, GpuError>;
pub fn cuda_available() -> bool;
```

### 8.3 Extended Monitoring API (New)

The following API extensions are proposed for trueno-gpu monitoring:

```rust
/// Extended monitoring for trueno-gpu (TRUENO-SPEC-010)
impl CudaContext {
    /// Get current memory usage (free, total) in bytes
    /// Uses cuMemGetInfo internally
    pub fn memory_info(&self) -> Result<(usize, usize), GpuError> {
        let driver = get_driver()?;
        let mut free: usize = 0;
        let mut total: usize = 0;
        unsafe {
            let result = (driver.cuMemGetInfo)(&mut free, &mut total);
            CudaDriver::check(result)?;
        }
        Ok((free, total))
    }

    /// Get device compute capability (major, minor)
    /// Uses cuDeviceGetAttribute with CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_*
    pub fn compute_capability(&self) -> Result<(u32, u32), GpuError>;

    /// Get SM count for occupancy calculations
    pub fn multiprocessor_count(&self) -> Result<u32, GpuError>;

    /// Get memory clock rate in kHz
    pub fn memory_clock_rate(&self) -> Result<u32, GpuError>;

    /// Get core clock rate in kHz
    pub fn clock_rate(&self) -> Result<u32, GpuError>;
}

/// Kernel execution metrics
pub struct KernelMetrics {
    /// Kernel name (from PTX)
    pub name: String,
    /// Grid dimensions (blocks)
    pub grid_dim: (u32, u32, u32),
    /// Block dimensions (threads per block)
    pub block_dim: (u32, u32, u32),
    /// Shared memory bytes
    pub shared_mem_bytes: u32,
    /// Execution time in microseconds
    pub execution_time_us: u64,
    /// Theoretical occupancy (0.0-1.0)
    pub occupancy: f32,
}

/// Instrumented kernel launch with metrics collection
impl CudaModule {
    pub fn launch_instrumented(
        &self,
        func: &str,
        config: LaunchConfig,
        args: &[*mut std::ffi::c_void],
        stream: &CudaStream,
    ) -> Result<KernelMetrics, GpuError>;
}
```

### 8.4 Example: GPU Pixel Rendering with Monitoring

This example shows trueno-gpu's native CUDA monitoring (from `gpu_pixels_render` example):

```rust
use trueno_gpu::driver::{CudaContext, CudaModule, CudaStream, GpuBuffer, LaunchConfig};
use trueno_gpu::ptx::{PtxModule, PtxKernel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔════════════════════════════════════════════════════════╗");
    println!("║       trueno-gpu: REAL GPU PIXEL RENDERING             ║");
    println!("╚════════════════════════════════════════════════════════╝");

    // [1/6] Initialize CUDA with full device info
    println!("\n[1/6] Initializing CUDA...");
    let ctx = CudaContext::new(0)?;
    let device_name = ctx.device_name()?;
    let (free, total) = ctx.memory_info()?;

    println!("       ✓ GPU: {}", device_name);
    println!("       Memory: {} MB free / {} MB total",
        free / (1024 * 1024), total / (1024 * 1024));

    // [2/6] Generate PTX for target architecture
    println!("[2/6] Generating gradient kernel PTX...");
    let ptx = PtxModule::new()
        .version(8, 0)
        .target("sm_89")  // RTX 4090 = Ada Lovelace (sm_89)
        .address_size(64)
        .add_kernel(build_gradient_kernel())
        .emit();
    println!("       PTX size: {} bytes", ptx.len());

    // [3/6] JIT compile PTX to SASS
    println!("[3/6] JIT compiling PTX to SASS...");
    let module = CudaModule::load_ptx(&ctx, &ptx)?;
    println!("       ✓ Compiled to device code");

    // [4/6] Allocate GPU memory with tracking
    let width = 80;
    let height = 30;
    let pixels = width * height;
    let bytes = pixels * std::mem::size_of::<f32>();

    println!("[4/6] Allocating GPU memory...");
    let buffer = GpuBuffer::alloc(&ctx, bytes)?;
    println!("       Allocated {} bytes for {}x{} pixels", bytes, width, height);

    // [5/6] Launch kernel with timing
    println!("[5/6] Launching kernel on GPU...");
    let stream = CudaStream::new(&ctx)?;
    let start = std::time::Instant::now();

    module.launch(
        "gradient_pixel",
        LaunchConfig::for_2d(width, height, 16, 16),
        &[&buffer.ptr(), &(width as u32), &(height as u32)],
        &stream,
    )?;
    stream.synchronize()?;

    let kernel_time = start.elapsed();
    println!("       ✓ Kernel executed in {:?}", kernel_time);

    // [6/6] Copy results and display
    let mut host_pixels = vec![0.0f32; pixels];
    buffer.copy_to_host(&mut host_pixels)?;

    // ... render to terminal ...

    // Statistics with REAL GPU metrics
    println!("═══ STATISTICS ═══");
    println!("┌────────────────────┬──────────────────────┐");
    println!("│ Pixels computed    │ {:>20} │", pixels);
    println!("│ GPU execution time │ {:>16.3}µs │", kernel_time.as_micros());
    println!("│ Throughput         │ {:>14.2} Mpx/s │",
        pixels as f64 / kernel_time.as_secs_f64() / 1_000_000.0);
    println!("│ Device             │ {:>20} │", &device_name[..device_name.len().min(20)]);
    println!("└────────────────────┴──────────────────────┘");

    Ok(())
}
```

### 8.5 trueno-gpu Feature Flags

```toml
[features]
default = []
# Enable CUDA driver FFI for actual GPU execution (requires NVIDIA driver)
cuda = ["dep:libloading"]
# Visual testing with trueno-viz
viz = ["dep:trueno-viz"]
# Stress testing with randomized inputs (native only)
stress-test = ["dep:simular", "dep:renacer"]
# TUI monitoring mode for stress tests
tui-monitor = ["stress-test", "dep:presentar", "dep:crossterm"]
# GPU pixel testing with probar TUI visualization
gpu-pixels = ["dep:jugar-probar", "dep:presentar", "dep:crossterm"]
```

### 8.6 trueno vs trueno-gpu Monitoring Comparison

| Capability | trueno (wgpu) | trueno-gpu (CUDA) |
|------------|---------------|-------------------|
| Device name | Generic adapter info | ✅ "NVIDIA GeForce RTX 4090" |
| VRAM total | ✅ Via wgpu | ✅ cuDeviceTotalMem |
| VRAM free/used | ❌ Not available | ✅ cuMemGetInfo |
| Kernel timing | ❌ Abstracted | ✅ Event-based timing |
| Compute capability | ❌ Not exposed | ✅ cuDeviceGetAttribute |
| PTX inspection | ❌ N/A (SPIR-V) | ✅ Full PTX access |
| Occupancy metrics | ❌ Not available | ✅ SM count, registers |
| Cross-platform | ✅ All platforms | ❌ NVIDIA only |

---

## 9. Integration Examples

### 9.1 probar + trueno-gpu Real-Time Dashboard

```rust
use jugar_probar::pixel_coverage::{GpuPixelBuffer, ansi};
use trueno_gpu::driver::CudaContext;

/// GPU-accelerated pixel coverage with REAL monitoring
pub struct TruenoGpuPixelBuffer {
    ctx: CudaContext,
    buffer: GpuBuffer,
    width: u32,
    height: u32,
}

impl TruenoGpuPixelBuffer {
    pub fn new(width: u32, height: u32) -> Result<Self, GpuError> {
        let ctx = CudaContext::new(0)?;
        let bytes = (width * height * 4) as usize;  // f32 per pixel
        let buffer = GpuBuffer::alloc(&ctx, bytes)?;

        Ok(Self { ctx, buffer, width, height })
    }

    /// Get GPU info for dashboard header
    pub fn gpu_header(&self) -> String {
        let name = self.ctx.device_name().unwrap_or_else(|_| "Unknown".to_string());
        let (free, total) = self.ctx.memory_info().unwrap_or((0, 0));
        let used = total - free;
        let pct = if total > 0 { (used as f64 / total as f64) * 100.0 } else { 0.0 };

        format!(
            "{}GPU: {} | VRAM: {:.1} GB / {:.1} GB ({:.0}%){}",
            ansi::PASS,
            name,
            used as f64 / (1024.0 * 1024.0 * 1024.0),
            total as f64 / (1024.0 * 1024.0 * 1024.0),
            pct,
            ansi::RESET
        )
    }
}
```

### 9.2 renacer + trueno-gpu Profiling

```rust
use renacer::{Tracer, TracerConfig, AnomalyDetector};
use trueno_gpu::driver::{CudaContext, CudaModule, CudaStream};

/// Profile GPU kernel execution with renacer anomaly detection
pub async fn profile_gpu_workload() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracer
    let tracer = Tracer::new(TracerConfig::default())?;

    // Initialize GPU
    let ctx = CudaContext::new(0)?;
    let module = CudaModule::load_ptx(&ctx, include_str!("kernel.ptx"))?;
    let stream = CudaStream::new(&ctx)?;

    // Collect kernel timing samples
    let mut timings = Vec::new();
    for _ in 0..1000 {
        let start = std::time::Instant::now();
        module.launch("compute", config, args, &stream)?;
        stream.synchronize()?;
        timings.push(start.elapsed().as_micros() as f64);
    }

    // Detect anomalies (outlier kernel executions)
    let detector = AnomalyDetector::new(2.0);  // 2 sigma threshold
    let anomalies = detector.detect(&timings);

    if !anomalies.is_empty() {
        println!("⚠️  Detected {} anomalous kernel executions", anomalies.len());
        for (idx, timing) in anomalies {
            println!("   Sample {}: {:.1}µs (outlier)", idx, timing);
        }
    }

    Ok(())
}
```

---

## 10. Performance Considerations

(See Section 9 above for overhead budgets)

---

## 11. Migration Path

### 11.1 For trueno Users

```rust
// Before: No visibility
let device = GpuDevice::new()?;
device.matmul(&a, &b, &mut result, m, k, n)?;

// After: Full observability
let device = GpuDevice::with_monitoring(MonitorConfig::default())?;
println!("Using GPU: {}", device.device_info().name);

device.matmul(&a, &b, &mut result, m, k, n)?;

let metrics = device.metrics();
println!("VRAM used: {} / {}",
    format_bytes(metrics.memory.used),
    format_bytes(metrics.memory.total));
```

### 11.2 For probar Integration

```rust
// Create monitored GPU buffer for probar demo
let mut buffer = GpuPixelBuffer::new_monitored(1920, 1080)?;

// Access monitoring during operations
buffer.random_fill_pass(0.05);
println!("GPU: {} - Utilization: {}%",
    buffer.gpu_name().unwrap_or("Unknown"),
    buffer.gpu_utilization().unwrap_or(0));
```

---

## 12. Open Questions

1. **Should NVML be a hard or soft dependency?**
   - Soft (dlopen at runtime) provides better portability
   - Hard (link-time) provides compile-time guarantees

2. **How to handle multi-GPU workload distribution monitoring?**
   - Per-device metrics vs. aggregate view
   - Load balancing visibility

3. **Should dashboard be a separate binary or library component?**
   - Separate binary: easier to use, larger deployment
   - Library: more flexible, requires application integration

---

## 13. QA Checklist

### 13.1 Functional Verification
- [ ] **Dual-Backend Selection**: Verify `select_optimal_backend()` correctly identifies NVIDIA vs non-NVIDIA environments.
- [ ] **trueno-gpu Initialization**: Confirm `CudaContext::new(0)` succeeds on NVIDIA hardware and fails gracefully otherwise.
- [ ] **Device Identification**: Verify `device_name()` returns the full model name (e.g., "NVIDIA GeForce RTX 4090") via `trueno-gpu`.
- [ ] **Memory Reporting**: Compare `ctx.memory_info()` output against `nvidia-smi` to ensure accuracy within 1MB.
- [ ] **Kernel Execution**: Run `gpu_pixels_render` example and verify kernel executes without error.

### 13.2 Integration Verification
- [ ] **Dashboard Visualization**: Run `probar` dashboard and verify real-time updates of VRAM and Utilization bars.
- [ ] **Trace Correlation**: Generate a trace using `renacer` and confirm GPU spans appear nested within application spans.
- [ ] **Anomaly Detection**: Force a kernel delay and verify `renacer` anomaly detector flags the outlier.

### 13.3 Performance & Stability
- [ ] **Overhead Test**: Measure CPU usage of `GpuMonitor` background thread (target < 0.1%).
- [ ] **Long-Run Stability**: Run monitoring for 1 hour to check for memory leaks (especially with FFI calls).
- [ ] **Concurrency**: Verify thread safety when multiple threads access `CudaContext` or `GpuMonitor`.

---

## Appendix A: NVML Function Reference

| Function | Purpose | btop Usage |
|----------|---------|------------|
| `nvmlInit()` | Initialize NVML | Startup |
| `nvmlDeviceGetCount()` | Enumerate GPUs | Device discovery |
| `nvmlDeviceGetName()` | Get device name | Display |
| `nvmlDeviceGetUtilizationRates()` | GPU/Memory util | Real-time |
| `nvmlDeviceGetMemoryInfo()` | VRAM stats | Real-time |
| `nvmlDeviceGetTemperature()` | Thermal reading | Real-time |
| `nvmlDeviceGetPowerUsage()` | Power draw (mW) | Real-time |
| `nvmlDeviceGetClockInfo()` | Clock speeds | Real-time |
| `nvmlDeviceGetPcieThroughput()` | PCIe bandwidth | Real-time |

---

## Appendix B: trueno-gpu CUDA Driver Functions

| Function | Purpose | trueno-gpu Usage |
|----------|---------|------------------|
| `cuInit()` | Initialize driver | Startup |
| `cuDeviceGetCount()` | Enumerate GPUs | Device discovery |
| `cuDeviceGet()` | Get device handle | Context creation |
| `cuDeviceGetName()` | Get device name | "NVIDIA GeForce RTX 4090" |
| `cuDeviceTotalMem()` | Total VRAM | Capacity reporting |
| `cuMemGetInfo()` | Free/total VRAM | Real-time monitoring |
| `cuDevicePrimaryCtxRetain()` | Get context | Resource management |
| `cuModuleLoadData()` | Load PTX | JIT compilation |
| `cuLaunchKernel()` | Execute kernel | Timed execution |
| `cuStreamSynchronize()` | Sync point | Timing boundary |

---

## Appendix C: Related Specifications

- **TRUENO-SPEC-001**: Core tensor operations (trueno)
- **TRUENO-SPEC-002**: GPU compute backend (trueno)
- **TRUENO-GPU-SPEC-001**: Complete CUDA runtime specification (trueno-gpu)
- **PROBAR-SPEC-009**: WASM pixel GUI demo (probar)
- **RENACER-SPEC-001**: Syscall tracing fundamentals (renacer)

---

## Appendix D: Crate Locations

| Crate | Path | Description |
|-------|------|-------------|
| trueno | `../trueno` | wgpu-based tensor ops |
| trueno-gpu | `../trueno/trueno-gpu` | Native CUDA PTX generation |
| trueno-viz | Published crate | Visualization utilities |
| jugar-probar | `../probar/crates/probar` | TUI testing framework |
| renacer | Published crate | System call tracing |

---

**Document Status**: DRAFT - Awaiting team review before implementation.

**Version History**:
- v1.0.0 (2024-12-14): Initial specification
- v1.1.0 (2024-12-14): Added trueno-gpu dual-backend support, Section 8 CUDA integration
