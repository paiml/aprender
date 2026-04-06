# GPU Monitoring

This chapter covers trueno's GPU monitoring capabilities as defined in **TRUENO-SPEC-010**.

## Overview

Trueno provides comprehensive GPU monitoring through two complementary approaches:

1. **Cross-platform wgpu backend** - Works on any system with Vulkan, Metal, or DX12
2. **Native CUDA backend** - Direct access to NVIDIA GPU information via CUDA Driver API

## Quick Start

```rust
use trueno::monitor::{GpuMonitor, GpuDeviceInfo, MonitorConfig};

// Enumerate all available GPUs
let devices = GpuDeviceInfo::enumerate()?;
for dev in &devices {
    println!("[{}] {} ({:.2} GB)", dev.index, dev.name, dev.vram_gb());
}

// Create a monitor with history buffer
let monitor = GpuMonitor::new(0, MonitorConfig::default())?;

// Collect metrics over time
for _ in 0..10 {
    let metrics = monitor.collect()?;
    println!("Memory: {:.1}% used", metrics.memory.usage_percent());
}
```

## Feature Flags

| Feature | Description |
|---------|-------------|
| `gpu` | Enable wgpu-based GPU monitoring (cross-platform) |
| `cuda-monitor` | Enable native CUDA monitoring (NVIDIA only) |

Enable features in your `Cargo.toml`:

```toml
[dependencies]
trueno = { version = "0.8", features = ["gpu", "cuda-monitor"] }
```

## Device Discovery

### GpuDeviceInfo

Represents a discovered GPU device:

```rust
pub struct GpuDeviceInfo {
    pub index: usize,
    pub name: String,
    pub vendor: GpuVendor,
    pub backend: GpuBackend,
    pub vram_total: u64,
    pub compute_capability: Option<(u32, u32)>,
    pub driver_version: Option<String>,
}
```

**Methods:**

- `enumerate() -> Result<Vec<GpuDeviceInfo>, MonitorError>` - List all GPUs
- `vram_gb() -> f64` - Get VRAM in gigabytes
- `supports_cuda() -> bool` - Check CUDA support

### GpuVendor

GPU manufacturer identification:

```rust
pub enum GpuVendor {
    Nvidia,
    Amd,
    Intel,
    Apple,
    Unknown(u32),
}
```

**PCI Vendor ID Mapping:**

| Vendor ID | Vendor |
|-----------|--------|
| `0x10de` | NVIDIA |
| `0x1002` | AMD |
| `0x8086` | Intel |
| `0x106b` | Apple |

### GpuBackend

Graphics/compute backend:

```rust
pub enum GpuBackend {
    Vulkan,
    Metal,
    Dx12,
    Cuda,
    WebGpu,
    OpenGl,
    Cpu,
}
```

## Memory Monitoring

### GpuMemoryMetrics

Real-time memory statistics:

```rust
pub struct GpuMemoryMetrics {
    pub total: u64,      // Total VRAM in bytes
    pub used: u64,       // Used VRAM in bytes
    pub free: u64,       // Free VRAM in bytes
}
```

**Methods:**

- `usage_percent() -> f64` - Memory utilization (0.0-100.0)
- `available_gb() -> f64` - Free memory in GB

## GpuMonitor

The `GpuMonitor` provides continuous monitoring with a ring buffer for history:

```rust
// Configure monitoring
let config = MonitorConfig {
    poll_interval: Duration::from_millis(100),
    history_size: 1000,
};

// Create monitor for device 0
let monitor = GpuMonitor::new(0, config)?;

// Collect a sample
let metrics = monitor.collect()?;

// Get sample age
println!("Sample age: {:?}", metrics.age());

// Check history
println!("History size: {}", monitor.sample_count());
```

### MonitorConfig

```rust
pub struct MonitorConfig {
    pub poll_interval: Duration,  // Default: 100ms
    pub history_size: usize,      // Default: 1000
}
```

### GpuMetrics

Complete metrics snapshot:

```rust
pub struct GpuMetrics {
    pub memory: GpuMemoryMetrics,
    pub utilization: GpuUtilization,
    pub thermal: GpuThermalMetrics,
    pub power: GpuPowerMetrics,
    pub clock: GpuClockMetrics,
    pub pcie: GpuPcieMetrics,
    pub timestamp: Instant,
}
```

## CUDA Native Monitoring

For NVIDIA GPUs, enable `cuda-monitor` for accurate device information via the CUDA Driver API:

```rust
use trueno::monitor::{
    cuda_monitor_available,
    enumerate_cuda_devices,
    query_cuda_memory,
};

// Check availability
if cuda_monitor_available() {
    // Enumerate CUDA devices
    let devices = enumerate_cuda_devices()?;

    // Query real-time memory
    let mem = query_cuda_memory(0)?;
    println!("CUDA Memory: {:.1}% used", mem.usage_percent());
}
```

### Why CUDA Native?

| Aspect | wgpu | CUDA Native |
|--------|------|-------------|
| Device Name | Generic ("NVIDIA GPU") | Exact ("GeForce RTX 4090") |
| Memory Info | Estimated | Accurate (cuMemGetInfo) |
| Portability | Cross-platform | NVIDIA only |
| Dependencies | wgpu | libcuda.so/nvcuda.dll |

## trueno-gpu Module

For direct CUDA access without the trueno facade:

```rust
use trueno_gpu::monitor::{CudaDeviceInfo, CudaMemoryInfo};
use trueno_gpu::driver::CudaContext;

// Query device info
let info = CudaDeviceInfo::query(0)?;
println!("GPU: {} ({:.2} GB)", info.name, info.total_memory_gb());

// Create context and query memory
let ctx = CudaContext::new(0)?;
let mem = CudaMemoryInfo::query(&ctx)?;
println!("Memory: {}", mem);  // "8192 / 24576 MB (33.3% used)"
```

## Examples

### Run the GPU Monitor Demo

```bash
# Cross-platform (wgpu)
cargo run --example gpu_monitor_demo --features gpu

# With CUDA (NVIDIA)
cargo run --example gpu_monitor_demo --features "gpu,cuda-monitor"
```

### Run the CUDA Monitor Example

```bash
cargo run -p trueno-gpu --example cuda_monitor --features cuda
```

## Error Handling

```rust
pub enum MonitorError {
    NoDevice,           // No GPU found
    DeviceNotFound(u32), // Specific device not found
    BackendError(String), // Backend-specific error
    ContextError(String), // Context creation failed
}
```

## Performance Considerations

- **Poll Interval**: Set `poll_interval` based on your monitoring needs. 100ms is good for visualization; 1s is sufficient for logging.
- **History Size**: The ring buffer is fixed-size. Larger sizes consume more memory but allow longer history analysis.
- **CUDA Context**: Creating a CUDA context has overhead. Reuse `GpuMonitor` instances when possible.

## References

- **TRUENO-SPEC-010**: GPU Monitoring, Tracing, and Visualization
- **Nickolls et al. (2008)**: GPU parallel computing model
- **CUDA Driver API**: cuDeviceGetName, cuDeviceTotalMem, cuMemGetInfo
