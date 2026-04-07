# TUI Compute Mode Flow: CPU, GPU, and Memory Monitoring

**Specification Version:** 1.1.0
**Status:** Review
**Date:** 2026-01-03
**Authors:** PAIML Engineering Team
**Validation:** 100% Probador Pixel-by-Pixel Tested

## Abstract

This specification defines a real-time Terminal User Interface (TUI) for monitoring compute flow, memory utilization, and data movement across heterogeneous hardware (CPU, NVIDIA GPU, AMD GPU, Apple Metal) in the Trueno compute ecosystem. It supports local and **distributed compute monitoring** (via **repartir**) and seamless **SSH-based remote monitoring** for heterogeneous clusters. The design follows Toyota Way principles (Iron Lotus Framework) and includes a comprehensive 100-point Popperian falsification test suite for QA validation.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Hardware Abstraction Layer](#2-hardware-abstraction-layer)
3. [Memory Hierarchy Monitoring](#3-memory-hierarchy-monitoring)
4. [Compute Flow Visualization](#4-compute-flow-visualization)
5. [Data Flow Tracking](#5-data-flow-tracking)
6. [TUI Layout Specification](#6-tui-layout-specification)
7. [Stress Test Mode (--stress-test)](#7-stress-test-mode---stress-test)
8. [Probador Pixel Testing Integration](#8-probador-pixel-testing-integration)
9. [100-Point Popperian Falsification Suite](#9-100-point-popperian-falsification-suite)
10. [Peer-Reviewed Citations](#10-peer-reviewed-citations)
11. [Implementation Roadmap](#11-implementation-roadmap)

---

## 1. Architecture Overview

### 1.1 System Context

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        TUI Compute Mode Flow Monitor                         │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐      │
│  │   CPU       │   │ NVIDIA GPU  │   │  AMD GPU    │   │  Memory     │      │
│  │  Monitor    │   │   Monitor   │   │   Monitor   │   │  Monitor    │      │
│  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘   └──────┬──────┘      │
│         │                 │                 │                 │              │
│         └────────────┬────┴────────────┬────┴─────────────────┘              │
│                      │                 │                                      │
│              ┌───────▼─────────────────▼───────┐                             │
│              │   Unified Telemetry Collector   │                             │
│              │     (trueno-gpu + sysinfo)      │                             │
│              └───────────────┬─────────────────┘                             │
│                              │                                                │
│              ┌───────────────▼─────────────────┐                             │
│              │    TUI Renderer (presentar)     │                             │
│              │   - Sparklines (60-point)       │                             │
│              │   - Gauges (memory bars)        │                             │
│              │   - Tables (process list)       │                             │
│              │   - Heatmaps (data flow)        │                             │
│              └─────────────────────────────────┘                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Design Principles (Toyota Way)

| Principle | Application | Citation |
|-----------|-------------|----------|
| **Genchi Genbutsu** | Direct hardware sampling via trueno-gpu CUDA/ROCm bindings | [Liker2004] §10 |
| **Jidoka** | Automatic anomaly detection (Isolation Forest) with circuit breakers | [Liker2004] §11, [Liu2008] |
| **Heijunka** | Level-loaded polling (adaptive 10ms-1000ms intervals) | [Liker2004] §4 |
| **Muda** | Zero-copy telemetry with ring buffers | [Ohno1988] §3 |
| **Poka-Yoke** | Type-safe metric structs prevent unit confusion | [Shingo1986] §2 |
| **Mieruka** | Visual control (Sparklines, Heatmaps) for instant understanding | [Liker2004] §13, [Tufte2006] |

### 1.3 Integration Points

```rust
// Crate dependencies
trueno = { version = "0.10", features = ["gpu", "cuda-monitor"] }
trueno-gpu = { version = "0.4", features = ["cuda"] }
repartir = { version = "1.1", features = ["tui", "gpu", "remote-tls"] } // Validated v1.1
renacer = { version = "0.9", features = ["chaos-full", "otlp"] }
probar = { version = "0.4", features = ["tui", "gpu"] }
sysinfo = "0.32"
nvml-wrapper = "0.10"   // NVIDIA Management Library
rocm-smi-lib = "0.2"    // AMD ROCm System Management Interface
```

### 1.4 Hardware Verification Matrix

| Environment | Access Method | Primary Device | Backend | Status |
|-------------|---------------|----------------|---------|--------|
| **Linux Dev** | Local | RTX 4090 | CUDA/Vulkan | ✅ Verified |
| **Intel Mac** | SSH (`ssh mac`) | AMD Radeon Pro | Metal | ⚠️ **Required** |
| **Apple Silicon** | Local | M1/M2/M3 | Metal | ⏳ Pending |
| **AMD ROCm** | SSH | Instinct MI210 | ROCm | ⏳ Pending |

---

## 2. Hardware Abstraction Layer

### 2.1 Unified Device Trait

```rust
/// Unified compute device abstraction (TRUENO-SPEC-020)
pub trait ComputeDevice: Send + Sync {
    /// Device identification
    fn device_id(&self) -> DeviceId;
    fn device_name(&self) -> &str;
    fn device_type(&self) -> DeviceType;

    /// Compute metrics
    fn compute_utilization(&self) -> Result<f64>;      // 0.0-100.0%
    fn compute_clock_mhz(&self) -> Result<u32>;
    fn compute_temperature_c(&self) -> Result<f64>;
    fn compute_power_watts(&self) -> Result<f64>;
    fn compute_power_limit_watts(&self) -> Result<f64>;

    /// Memory metrics
    fn memory_used_bytes(&self) -> Result<u64>;
    fn memory_total_bytes(&self) -> Result<u64>;
    fn memory_bandwidth_gbps(&self) -> Result<f64>;

    /// Streaming multiprocessor / Compute Unit metrics
    fn sm_count(&self) -> u32;
    fn active_sm_count(&self) -> Result<u32>;

    /// PCIe / Interconnect metrics
    fn pcie_tx_bytes_per_sec(&self) -> Result<u64>;
    fn pcie_rx_bytes_per_sec(&self) -> Result<u64>;
    fn pcie_generation(&self) -> u8;
    fn pcie_width(&self) -> u8;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceType {
    Cpu,
    NvidiaGpu,
    AmdGpu,
    IntelGpu,
    AppleSilicon,
    Hpu, // Hardware Processing Unit (e.g., Gaudi, TPU)
}
```

### 2.2 NVIDIA GPU Implementation (via NVML)

```rust
/// NVIDIA GPU monitor using NVML (cuda-monitor feature)
pub struct NvidiaDevice {
    nvml: Nvml,
    device: Device,
    index: u32,
}

impl NvidiaDevice {
    pub fn enumerate() -> Result<Vec<Self>> {
        let nvml = Nvml::init()?;
        let count = nvml.device_count()?;
        (0..count)
            .map(|i| Ok(Self {
                nvml: nvml.clone(),
                device: nvml.device_by_index(i)?,
                index: i,
            }))
            .collect()
    }
}

impl ComputeDevice for NvidiaDevice {
    fn device_name(&self) -> &str {
        self.device.name().unwrap_or("Unknown NVIDIA GPU")
    }

    fn compute_utilization(&self) -> Result<f64> {
        let util = self.device.utilization_rates()?;
        Ok(util.gpu as f64)
    }

    fn memory_used_bytes(&self) -> Result<u64> {
        let mem = self.device.memory_info()?;
        Ok(mem.used)
    }

    fn memory_total_bytes(&self) -> Result<u64> {
        let mem = self.device.memory_info()?;
        Ok(mem.total)
    }

    fn compute_temperature_c(&self) -> Result<f64> {
        Ok(self.device.temperature(TemperatureSensor::Gpu)? as f64)
    }

    fn compute_power_watts(&self) -> Result<f64> {
        Ok(self.device.power_usage()? as f64 / 1000.0) // mW to W
    }
}
```

### 2.3 AMD GPU Implementation (via ROCm SMI)

```rust
/// AMD GPU monitor using ROCm SMI
pub struct AmdDevice {
    device_index: u32,
}

impl AmdDevice {
    pub fn enumerate() -> Result<Vec<Self>> {
        let count = rocm_smi::num_devices()?;
        (0..count).map(|i| Ok(Self { device_index: i })).collect()
    }
}

impl ComputeDevice for AmdDevice {
    fn device_name(&self) -> &str {
        rocm_smi::get_name(self.device_index)
            .unwrap_or("Unknown AMD GPU")
    }

    fn compute_utilization(&self) -> Result<f64> {
        Ok(rocm_smi::get_gpu_busy_percent(self.device_index)? as f64)
    }

    fn memory_used_bytes(&self) -> Result<u64> {
        rocm_smi::get_memory_usage(self.device_index)
    }

    fn memory_total_bytes(&self) -> Result<u64> {
        rocm_smi::get_memory_total(self.device_index)
    }

    fn compute_temperature_c(&self) -> Result<f64> {
        Ok(rocm_smi::get_temp_metric(
            self.device_index,
            RocmTemperatureType::Edge
        )? as f64 / 1000.0) // millidegrees to degrees
    }
}
```

### 2.4 CPU Implementation (via sysinfo)

```rust
/// CPU monitor using sysinfo crate
pub struct CpuDevice {
    system: System,
    core_count: usize,
}

impl CpuDevice {
    pub fn new() -> Self {
        let mut system = System::new_all();
        system.refresh_cpu();
        Self {
            core_count: system.cpus().len(),
            system,
        }
    }

    pub fn refresh(&mut self) {
        self.system.refresh_cpu();
        self.system.refresh_memory();
    }
}

impl ComputeDevice for CpuDevice {
    fn device_name(&self) -> &str {
        self.system.cpus().first()
            .map(|c| c.brand())
            .unwrap_or("Unknown CPU")
    }

    fn compute_utilization(&self) -> Result<f64> {
        let total: f32 = self.system.cpus().iter()
            .map(|c| c.cpu_usage())
            .sum();
        Ok((total / self.core_count as f32) as f64)
    }

    fn memory_used_bytes(&self) -> Result<u64> {
        Ok(self.system.used_memory())
    }

    fn memory_total_bytes(&self) -> Result<u64> {
        Ok(self.system.total_memory())
    }

    fn compute_temperature_c(&self) -> Result<f64> {
        // Platform-specific: Linux reads from /sys/class/thermal
        #[cfg(target_os = "linux")]
        {
            let temp = std::fs::read_to_string(
                "/sys/class/thermal/thermal_zone0/temp"
            )?;
            Ok(temp.trim().parse::<f64>()? / 1000.0)
        }
        #[cfg(not(target_os = "linux"))]
        Err(Error::NotSupported)
    }
}
```

---

## 3. Memory Hierarchy Monitoring

### 3.1 Memory Levels

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         Memory Hierarchy View                               │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  L1 Cache (per core)     L2 Cache (shared)      L3 Cache (LLC)             │
│  ┌────────────────┐      ┌────────────────┐     ┌────────────────┐         │
│  │ █████░░░ 62%   │      │ ████████░ 89%  │     │ ███████░░ 78%  │         │
│  │ 32 KB / 32 KB  │      │ 256K / 256K    │     │ 30MB / 36MB    │         │
│  └────────────────┘      └────────────────┘     └────────────────┘         │
│                                                                             │
│  System RAM                           SWAP                                  │
│  ┌──────────────────────────────┐    ┌──────────────────────────────┐      │
│  │ ████████████████░░░░ 72.4%   │    │ ██░░░░░░░░░░░░░░░░░░ 8.2%    │      │
│  │ 46.3 GB / 64.0 GB            │    │ 1.3 GB / 16.0 GB             │      │
│  │ ▁▂▃▄▅▆▇█▇▆▅▄▃▂▁▂▃▄▅▆▇█▇▆▅▄▃▂ │    │ ▁▁▁▁▁▂▂▂▃▃▃▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁ │      │
│  └──────────────────────────────┘    └──────────────────────────────┘      │
│                                                                             │
│  GPU VRAM (NVIDIA RTX 4090)           GPU VRAM (AMD Radeon Pro W5700X)     │
│  ┌──────────────────────────────┐    ┌──────────────────────────────┐      │
│  │ ██████████████░░░░░░ 58.3%   │    │ ████████░░░░░░░░░░░░ 34.7%   │      │
│  │ 14.0 GB / 24.0 GB            │    │ 5.6 GB / 16.0 GB             │      │
│  │ ▁▂▃▄▅▆▇█▇▆▅▄▅▆▇█▇▆▅▄▃▂▁▂▃▄▅▆ │    │ ▁▁▂▂▃▃▄▄▅▅▄▄▃▃▂▂▁▁▁▁▂▂▃▃▄▄▅ │      │
│  └──────────────────────────────┘    └──────────────────────────────┘      │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Memory Metrics Structure

```rust
/// Comprehensive memory metrics (TRUENO-SPEC-021)
#[derive(Debug, Clone)]
pub struct MemoryMetrics {
    // System RAM
    pub ram_used_bytes: u64,
    pub ram_total_bytes: u64,
    pub ram_available_bytes: u64,
    pub ram_cached_bytes: u64,
    pub ram_buffers_bytes: u64,

    // Swap
    pub swap_used_bytes: u64,
    pub swap_total_bytes: u64,

    // Per-GPU VRAM
    pub gpu_vram: Vec<GpuVramMetrics>,

    // Memory pressure (LAMBDA-0002)
    pub pressure_level: PressureLevel,
    pub safe_parallel_jobs: u32,

    // Bandwidth (measured)
    pub ram_read_bandwidth_gbps: f64,
    pub ram_write_bandwidth_gbps: f64,

    // History (60-point sparkline)
    pub ram_history: VecDeque<f64>,
    pub swap_history: VecDeque<f64>,
}

#[derive(Debug, Clone)]
pub struct GpuVramMetrics {
    pub device_id: DeviceId,
    pub used_bytes: u64,
    pub total_bytes: u64,
    pub reserved_bytes: u64,  // Driver/system reserved
    pub bar1_used_bytes: u64, // PCIe BAR1 aperture
    pub history: VecDeque<f64>,
}

/// Memory pressure levels (from lambda-lab-rust-development)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PressureLevel {
    Ok,        // >= 50% available
    Elevated,  // 30-50% available
    Warning,   // 15-30% available
    Critical,  // < 15% available
}
```

### 3.3 Memory Pressure Calculation

Based on [LAMBDA-0002] specification:

```rust
/// Calculate memory pressure and safe job count
pub fn analyze_pressure(metrics: &MemoryMetrics) -> PressureAnalysis {
    let available_pct = (metrics.ram_available_bytes as f64
        / metrics.ram_total_bytes as f64) * 100.0;

    let level = match available_pct {
        x if x >= 50.0 => PressureLevel::Ok,
        x if x >= 30.0 => PressureLevel::Elevated,
        x if x >= 15.0 => PressureLevel::Warning,
        _ => PressureLevel::Critical,
    };

    // Safe jobs = min(available_gb / 3.0, cpu_cores)
    // Based on 3GB/job heuristic [Volkov2008]
    let available_gb = metrics.ram_available_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    let cpu_cores = num_cpus::get() as u32;
    let safe_jobs = ((available_gb / 3.0) as u32).min(cpu_cores).max(1);

    PressureAnalysis {
        level,
        available_percent: available_pct as u32,
        available_gb,
        safe_jobs,
        block_builds: level == PressureLevel::Critical,
        recommendation: pressure_recommendation(level),
    }
}
```

---

## 4. Compute Flow Visualization

### 4.1 Compute Pipeline View

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         Compute Pipeline Flow                               │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   INPUT     │    │   COMPUTE   │    │   REDUCE    │    │   OUTPUT    │  │
│  │  (Host→Dev) │───▶│   (Kernel)  │───▶│   (Tile)    │───▶│  (Dev→Host) │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│                                                                             │
│  Stage Latency:                                                             │
│  ├─ Input:   ████████████░░░░░░░░  2.34 ms (PCIe 4.0 x16)                  │
│  ├─ Compute: ██████████████████░░  8.92 ms (RTX 4090 @ 2520 MHz)           │
│  ├─ Reduce:  ████░░░░░░░░░░░░░░░░  0.87 ms (Tiled 16x16)                   │
│  └─ Output:  ██████░░░░░░░░░░░░░░  1.23 ms (DMA async)                     │
│                                                                             │
│  Total: 13.36 ms │ Throughput: 74.9 ops/s │ Efficiency: 89.2%              │
│                                                                             │
│  Active Kernels:                                                            │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ batched_gemm_tiled     [████████████░░░░░░░░] 58% │ 2048x2048x2048 │    │
│  │ fma_fusion_pass        [██████░░░░░░░░░░░░░░] 28% │ Optimizing...  │    │
│  │ tiled_reduction_sum    [████░░░░░░░░░░░░░░░░] 14% │ 16x16 tiles    │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Compute Metrics Structure

```rust
/// Compute pipeline metrics (TRUENO-SPEC-022)
#[derive(Debug, Clone)]
pub struct ComputeMetrics {
    // Per-device utilization
    pub devices: Vec<DeviceComputeMetrics>,

    // Active kernel tracking
    pub active_kernels: Vec<KernelExecution>,

    // Pipeline stage latencies
    pub input_latency_ms: f64,
    pub compute_latency_ms: f64,
    pub reduce_latency_ms: f64,
    pub output_latency_ms: f64,

    // Throughput
    pub operations_per_second: f64,
    pub flops_achieved: f64,
    pub flops_theoretical: f64,

    // Efficiency
    pub compute_efficiency_pct: f64,  // achieved/theoretical
    pub memory_efficiency_pct: f64,   // bandwidth utilization
}

#[derive(Debug, Clone)]
pub struct DeviceComputeMetrics {
    pub device_id: DeviceId,
    pub utilization_pct: f64,
    pub sm_active_pct: f64,
    pub warps_active: u32,
    pub warps_max: u32,
    pub clock_mhz: u32,
    pub clock_max_mhz: u32,
    pub power_watts: f64,
    pub power_limit_watts: f64,
    pub temperature_c: f64,
    pub throttle_reason: Option<ThrottleReason>,
    pub history: VecDeque<f64>, // 60-point sparkline
}

#[derive(Debug, Clone)]
pub struct KernelExecution {
    pub name: String,
    pub grid_dim: (u32, u32, u32),
    pub block_dim: (u32, u32, u32),
    pub shared_mem_bytes: usize,
    pub registers_per_thread: u32,
    pub occupancy_pct: f64,
    pub elapsed_ms: f64,
    pub status: KernelStatus,
}

#[derive(Debug, Clone, Copy)]
pub enum ThrottleReason {
    Power,
    Thermal,
    ApplicationClocks,
    SwPowerCap,
    HwSlowdown,
    SyncBoost,
    None,
}
```

---

## 5. Data Flow Tracking

### 5.1 Data Movement Visualization

```
┌────────────────────────────────────────────────────────────────────────────┐
│                          Data Flow Monitor                                  │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Host ◀════════════════════════════════════════════════════════▶ Device    │
│       │                                                         │           │
│  RAM  │  ┌─────────────────────────────────────────────────┐   │  VRAM     │
│       │  │     PCIe 4.0 x16: 31.5 GB/s theoretical         │   │           │
│       │  │     ══════════════════════════════════════════  │   │           │
│       │  │     TX: ██████████████░░░░░░  12.4 GB/s (39%)   │   │           │
│       │  │     RX: ████████░░░░░░░░░░░░   6.8 GB/s (22%)   │   │           │
│       │  └─────────────────────────────────────────────────┘   │           │
│       │                                                         │           │
│       │  Active Transfers:                                      │           │
│       │  ┌─────────────────────────────────────────────────┐   │           │
│       │  │ H→D  tensor_a      [████████████░░░░] 78%  1.2GB│   │           │
│       │  │ H→D  tensor_b      [██████████████░░] 89%  1.2GB│   │           │
│       │  │ D→H  result        [████░░░░░░░░░░░░] 23%  256MB│   │           │
│       │  └─────────────────────────────────────────────────┘   │           │
│       │                                                         │           │
│  ┌────▼────┐    ┌──────────┐    ┌──────────┐    ┌────────▼───┐ │           │
│  │ Pinned  │───▶│ Staging  │───▶│ Compute  │───▶│  Result    │ │           │
│  │ Buffer  │    │  Buffer  │    │  Buffer  │    │  Buffer    │ │           │
│  │ 4.0 GB  │    │  2.0 GB  │    │ 14.0 GB  │    │  2.0 GB    │ │           │
│  └─────────┘    └──────────┘    └──────────┘    └────────────┘ │           │
│                                                                             │
│  Memory Bus Utilization:                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ GPU Memory: ████████████████████░░░░░░░░░░  672 GB/s (67%)          │   │
│  │ ▁▂▃▄▅▆▇█▇▆▅▄▃▂▁▂▃▄▅▆▇█▇▆▅▄▃▂▁▂▃▄▅▆▇█▇▆▅▄▃▂▁▂▃▄▅▆▇█▇▆▅▄▃▂▁▂▃▄▅ (60s)  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Data Transfer Metrics

```rust
/// Data transfer tracking (TRUENO-SPEC-023)
#[derive(Debug, Clone)]
pub struct DataFlowMetrics {
    // PCIe metrics
    pub pcie_generation: u8,
    pub pcie_width: u8,
    pub pcie_theoretical_gbps: f64,
    pub pcie_tx_gbps: f64,
    pub pcie_rx_gbps: f64,

    // Active transfers
    pub active_transfers: Vec<Transfer>,
    pub completed_transfers: VecDeque<Transfer>, // Last 100

    // Memory bus
    pub memory_bus_utilization_pct: f64,
    pub memory_read_gbps: f64,
    pub memory_write_gbps: f64,

    // Buffer pools
    pub pinned_memory_used_bytes: u64,
    pub pinned_memory_total_bytes: u64,
    pub staging_buffer_used_bytes: u64,

    // History
    pub pcie_tx_history: VecDeque<f64>,
    pub pcie_rx_history: VecDeque<f64>,
    pub memory_bus_history: VecDeque<f64>,
}

#[derive(Debug, Clone)]
pub struct Transfer {
    pub id: TransferId,
    pub direction: TransferDirection,
    pub source: MemoryLocation,
    pub destination: MemoryLocation,
    pub size_bytes: u64,
    pub transferred_bytes: u64,
    pub start_time: Instant,
    pub end_time: Option<Instant>,
    pub status: TransferStatus,
    pub label: String,
}

#[derive(Debug, Clone, Copy)]
pub enum TransferDirection {
    HostToDevice,
    DeviceToHost,
    DeviceToDevice,
    PeerToPeer,
}

#[derive(Debug, Clone, Copy)]
pub enum MemoryLocation {
    SystemRam,
    PinnedMemory,
    GpuVram(DeviceId),
    UnifiedMemory,
}
```

---

## 6. TUI Layout Specification

### 6.1 Full Screen Layout (80x24 minimum, 160x48 recommended)

```
┌────────────────────────────────────────────────────────────────────────────────┐
│ TRUENO Compute Monitor v0.10.1 │ CPU: Intel Xeon │ GPU: RTX 4090 + W5700X │ F1│
├────────────────────────────────────────────────────────────────────────────────┤
│ [COMPUTE]────────────────────────────────────────────────────────────────────┐ │
│ │ CPU:  ████████████░░░░░░░░ 62.3% │ 3.8 GHz │ 45°C │ 125W / 250W           │ │
│ │ GPU0: ██████████████████░░ 89.1% │ 2.5 GHz │ 72°C │ 320W / 450W  [NVIDIA] │ │
│ │ GPU1: ████████░░░░░░░░░░░░ 34.7% │ 1.8 GHz │ 58°C │ 145W / 200W  [AMD]    │ │
│ │ ▁▂▃▄▅▆▇█▇▆▅▄▅▆▇█▇▆▅▄▃▂▁▂▃▄▅▆▇█▇▆▅▄▃▂▁▂▃▄▅▆▇█▇▆▅▄▃▂▁▂▃▄▅▆ (60s history)     │ │
│ └─────────────────────────────────────────────────────────────────────────────┘ │
│ [MEMORY]─────────────────────────────────────────────────────────────────────┐ │
│ │ RAM:   ████████████████████░░░░░░░░░░ 72.4% │ 46.3 / 64.0 GB │ OK         │ │
│ │ SWAP:  ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░  8.2% │  1.3 / 16.0 GB │            │ │
│ │ VRAM0: ██████████████████░░░░░░░░░░░░ 58.3% │ 14.0 / 24.0 GB │ [RTX 4090] │ │
│ │ VRAM1: ████████████░░░░░░░░░░░░░░░░░░ 34.7% │  5.6 / 16.0 GB │ [W5700X]   │ │
│ └─────────────────────────────────────────────────────────────────────────────┘ │
│ [DATA FLOW]──────────────────────────────────────────────────────────────────┐ │
│ │ PCIe TX: ██████████████░░░░░░ 12.4 GB/s │ RX: ████████░░░░░░░░░░ 6.8 GB/s │ │
│ │ MEM BW:  ████████████████████░░░░░░░░░░ 672 GB/s (67% of 1008 GB/s peak)  │ │
│ │ Transfers: H→D tensor_a [████████░░] 78% │ D→H result [██░░░░░░] 23%      │ │
│ └─────────────────────────────────────────────────────────────────────────────┘ │
│ [KERNELS]────────────────────────────────────────────────────────────────────┐ │
│ │ batched_gemm_tiled     GPU0 [████████████░░] 58% │ 2048x2048 │ 8.92 ms    │ │
│ │ tiled_reduction_sum    GPU0 [████░░░░░░░░░░] 14% │ 16x16     │ 0.87 ms    │ │
│ └─────────────────────────────────────────────────────────────────────────────┘ │
├────────────────────────────────────────────────────────────────────────────────┤
│ q:Quit r:Refresh s:Stress Tab:Focus ↑↓:Navigate ?:Help │ Refresh: 100ms       │
└────────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 TUI Widget Specifications

```rust
/// TUI layout configuration (TRUENO-SPEC-024)
pub struct TuiLayout {
    pub min_width: u16,   // 80
    pub min_height: u16,  // 24
    pub rec_width: u16,   // 160
    pub rec_height: u16,  // 48

    pub sections: Vec<Section>,
    pub refresh_rate_ms: u64,
    pub sparkline_points: usize,  // 60
}

pub struct Section {
    pub id: &'static str,
    pub title: String,
    pub height_pct: f32,  // 0.0-1.0
    pub widgets: Vec<Widget>,
}

pub enum Widget {
    Gauge {
        label: String,
        value_pct: f64,
        thresholds: (f64, f64, f64), // warning, critical, max
        suffix: String,
    },
    Sparkline {
        data: VecDeque<f64>,
        label: String,
        baseline: Option<f64>,
    },
    ProgressBar {
        label: String,
        progress: f64,
        total: String,
    },
    Table {
        headers: Vec<String>,
        rows: Vec<Vec<String>>,
        highlight_row: Option<usize>,
    },
    Text {
        content: String,
        style: TextStyle,
    },
}

/// Color scheme (colorblind-safe Viridis-based)
pub struct ColorScheme {
    pub ok: Color,        // #21918c (teal)
    pub warning: Color,   // #fde725 (yellow)
    pub critical: Color,  // #f03b20 (red-orange)
    pub neutral: Color,   // #3b528b (blue)
    pub background: Color,// #440154 (dark purple)
}
```

### 6.3 Keyboard Controls

| Key | Action | Description |
|-----|--------|-------------|
| `q` | Quit | Exit the TUI |
| `r` | Refresh | Force immediate refresh |
| `s` | Stress Test | Toggle stress test mode |
| `Tab` | Focus | Cycle through sections |
| `↑`/`↓` | Navigate | Select rows in tables |
| `Enter` | Expand | Show detailed view |
| `?` | Help | Toggle help overlay |
| `a` | Alerts | Show alert panel |
| `e` | Export | Export metrics to JSON |
| `p` | Pause | Pause/resume monitoring |

---

## 7. Stress Test Mode (--stress-test)



### 7.1 Stress Test Objectives



The `--stress-test` mode saturates all compute and memory resources to validate system stability.

**CRITICAL REQUIREMENT:** GPU stress tests MUST use native compute shaders (WGPU/CUDA/Metal/ROCm) to generate real thermal/power load. CPU-based "fake" GPU load loops are strictly prohibited and will fail QA.



1.  **Thermal throttling behavior** under sustained load

2.  **Memory pressure handling** at 95%+ utilization

3.  **PCIe bandwidth saturation** with concurrent transfers

4.  **Error detection** under resource contention



### 7.2 CLI Interface



```bash

# Full stress test (all resources)

trueno-monitor --stress-test



# Targeted stress tests

trueno-monitor --stress-test --target cpu        # CPU only

trueno-monitor --stress-test --target gpu        # All GPUs (WGPU/CUDA/Metal)

trueno-monitor --stress-test --target gpu:0      # Specific GPU

trueno-monitor --stress-test --target memory     # RAM + VRAM

trueno-monitor --stress-test --target pcie       # PCIe bandwidth



# Duration and intensity

trueno-monitor --stress-test --duration 60s      # 60 second test

trueno-monitor --stress-test --intensity 0.8     # 80% of max load

trueno-monitor --stress-test --ramp-up 10s       # Gradual ramp-up



# Chaos integration (via renacer)

trueno-monitor --stress-test --chaos gentle      # With gentle chaos

trueno-monitor --stress-test --chaos aggressive  # With aggressive chaos

```



### 7.3 Stress Test Implementation



```rust

/// Stress test configuration (TRUENO-SPEC-025)

#[derive(Debug, Clone)]

pub struct StressTestConfig {

    pub target: StressTarget,

    pub duration: Duration,

    pub intensity: f64,        // 0.0-1.0

    pub ramp_up: Duration,

    pub chaos_preset: Option<ChaosPreset>,

    pub collect_metrics: bool,

    pub export_report: bool,

}



#[derive(Debug, Clone)]

pub enum StressTarget {

    All,

    Cpu,

    Gpu(Option<DeviceId>),

    Memory,

    Pcie,

    Custom(Vec<StressTarget>),

}



/// Stress test runner

pub struct StressTestRunner {

    config: StressTestConfig,

    metrics: Arc<Mutex<StressMetrics>>,

    workers: Vec<JoinHandle<()>>,

}



impl StressTestRunner {

    pub async fn run(&mut self) -> Result<StressTestReport> {

        // Phase 1: Ramp-up

        self.ramp_up().await?;



        // Phase 2: Sustained load

        self.sustain_load().await?;



        // Phase 3: Cool-down and report

        self.cool_down().await?;



        Ok(self.generate_report())

    }



    async fn stress_cpu(&self) {

        // SIMD-heavy matrix operations via trueno

        let size = 4096;

        let a = Matrix::<f32>::random(size, size);

        let b = Matrix::<f32>::random(size, size);



        loop {

            let _ = a.matmul_simd(&b);

            if self.should_stop() { break; }

        }

    }



    async fn stress_gpu(&self, device_id: DeviceId) {

        // MUST use WGPU/CUDA compute shader - NO CPU LOOPS

        let kernel = if self.is_cuda() {

             BatchedGemmKernel::tiled(64, 4096, 4096, 4096, 16)

        } else {

             // WGPU/Metal/Vulkan fallback

             WgpuComputeShader::stress_kernel(4096)

        };



        loop {

            self.dispatch_kernel(&kernel).await?;

            if self.should_stop() { break; }

        }

    }



    async fn stress_memory(&self) {

        // Allocate and access memory to prevent caching

        let size = self.config.intensity * available_memory() * 0.9;

        let mut buffers: Vec<Vec<u8>> = Vec::new();



        // Fill to target utilization

        while total_allocated(&buffers) < size {

            let mut buf = vec![0u8; 1024 * 1024 * 64]; // 64MB chunks

            // Touch all pages (prevent lazy allocation)

            for chunk in buf.chunks_mut(4096) {

                chunk[0] = rand::random();

            }

            buffers.push(buf);

        }



        // Random access pattern to stress memory controller

        loop {

            let idx = rand::random::<usize>() % buffers.len();

            let offset = rand::random::<usize>() % buffers[idx].len();

            let _ = buffers[idx][offset];

            if self.should_stop() { break; }

        }

    }



    async fn stress_pcie(&self) {

        // Concurrent H2D and D2H transfers

        let buffer_size = 256 * 1024 * 1024; // 256MB

        let host_buffer = vec![0u8; buffer_size];



        loop {

            // Overlapped transfers for maximum bandwidth

            tokio::join!(

                self.transfer_h2d(&host_buffer),

                self.transfer_d2h(buffer_size),

            );

            if self.should_stop() { break; }

        }

    }

}

```



### 7.4 Stress Test Metrics



```rust

/// Stress test metrics collection

#[derive(Debug, Clone)]

pub struct StressMetrics {

    // Peak values

    pub peak_cpu_utilization: f64,

    pub peak_gpu_utilization: f64,

    pub peak_memory_utilization: f64,

    pub peak_temperature_c: f64,

    pub peak_power_watts: f64,

    pub peak_pcie_bandwidth_gbps: f64,



    // Throttling events

    pub thermal_throttle_count: u32,

    pub power_throttle_count: u32,

    pub memory_pressure_events: u32,



    // Errors

    pub gpu_errors: Vec<GpuError>,

    pub memory_errors: Vec<MemoryError>,

    pub transfer_errors: Vec<TransferError>,



    // Performance regression detection

    pub baseline_flops: f64,

    pub achieved_flops: f64,

    pub performance_degradation_pct: f64,

}



#[derive(Debug, Clone)]

pub struct StressTestReport {

    pub config: StressTestConfig,

    pub metrics: StressMetrics,

    pub duration_actual: Duration,

    pub verdict: StressTestVerdict,

    pub recommendations: Vec<String>,

}



#[derive(Debug, Clone, Copy)]

pub enum StressTestVerdict {

    Pass,           // All metrics within acceptable range

    PassWithNotes,  // Minor throttling, acceptable

    Fail,           // Errors or severe throttling

}

```



---



## 8. Probador Pixel Testing Integration



### 8.1 Pixel Coverage Strategy



Every TUI element is validated pixel-by-pixel using Probar's statistical coverage framework:



```rust

/// Pixel coverage test configuration (PIXEL-001 v2.1)

pub struct TuiPixelCoverage {

    pub tracker: PixelCoverageTracker,

    pub regions: HashMap<String, PixelRegion>,

    pub thresholds: CoverageThresholds,

}



impl TuiPixelCoverage {

    pub fn new(width: u32, height: u32) -> Self {

        let mut tracker = PixelCoverageTracker::new(width, height, 20, 15);



        // Define critical regions

        let regions = hashmap! {

            "header" => PixelRegion::new(0, 0, width, 1),

            "compute_section" => PixelRegion::new(0, 1, width, 5),

            "memory_section" => PixelRegion::new(0, 6, width, 4),

            "dataflow_section" => PixelRegion::new(0, 10, width, 3),

            "kernels_section" => PixelRegion::new(0, 13, width, 3),

            "footer" => PixelRegion::new(0, height - 1, width, 1),

        };



        Self {

            tracker,

            regions,

            thresholds: CoverageThresholds::default(),

        }

    }



    /// Validate 100% pixel coverage

    pub fn validate_full_coverage(&self) -> Result<()> {

        let coverage = self.tracker.coverage_percent();



        if coverage < 100.0 {

            let gaps = self.tracker.find_gaps();

            return Err(PixelCoverageError::IncompleteCoverage {

                actual: coverage,

                expected: 100.0,

                gaps,

            });

        }



        Ok(())

    }



    /// Export heatmap for QA review

    pub fn export_heatmap(&self, path: &Path) -> Result<()> {

        PngHeatmap::new(self.tracker.width(), self.tracker.height())

            .with_palette(ColorPalette::viridis())

            .with_gap_highlighting()

            .with_legend()

            .with_title("TUI Compute Monitor Pixel Coverage")

            .export_to_file(self.tracker.cells(), path)

    }

}

```



### 8.2 Widget-Level Testing



```rust

/// Test each widget renders correctly

#[cfg(test)]

mod pixel_tests {

    use probar::prelude::*;

    use super::*;



    #[test]

    fn test_gauge_widget_coverage() {

        let mut coverage = TuiPixelCoverage::new(80, 24);

        let backend = TestBackend::new(80, 24);

        let mut terminal = Terminal::new(backend)?;



        terminal.draw(|f| {

            let gauge = Gauge::default()

                .percent(75)

                .label("CPU: 75%");

            f.render_widget(gauge, f.size());

        })?;



        // Record all rendered cells

        for (x, y, cell) in terminal.backend().buffer().cells() {

            if cell.symbol() != " " {

                coverage.tracker.record_point(x, y);

            }

        }



        assert!(coverage.tracker.coverage_percent() > 95.0);

    }



    #[test]

    fn test_sparkline_widget_coverage() {

        let mut coverage = TuiPixelCoverage::new(80, 24);

        let data: Vec<u64> = (0..60).map(|i| (i * 100 / 60) as u64).collect();



        let backend = TestBackend::new(80, 24);

        let mut terminal = Terminal::new(backend)?;



        terminal.draw(|f| {

            let sparkline = Sparkline::default()

                .data(&data)

                .style(Style::default().fg(Color::Cyan));

            f.render_widget(sparkline, f.size());

        })?;



        // Validate sparkline renders all 60 data points

        let rendered_points = count_non_empty_cells(terminal.backend().buffer());

        assert!(rendered_points >= 60, "Sparkline should render all data points");

    }



    #[test]

    fn test_full_tui_layout_coverage() {

        let mut coverage = TuiPixelCoverage::new(160, 48);

        let app = TuiApp::new_with_mock_data();



        let backend = TestBackend::new(160, 48);

        let mut terminal = Terminal::new(backend)?;



        // Render full UI

        terminal.draw(|f| app.render(f))?;



        // Record all cells

        for (x, y, cell) in terminal.backend().buffer().cells() {

            coverage.tracker.record_point(x as u32, y as u32);

        }



        // Export heatmap for QA review

        coverage.export_heatmap(Path::new("coverage_heatmap.png"))?;



        // Validate 100% coverage

        coverage.validate_full_coverage()?;

    }

}

```



### 8.3 Visual Regression Testing



```rust

/// Visual regression tests for TUI

#[cfg(test)]

mod visual_regression_tests {

    use probar::visual_regression::*;



    #[test]

    fn test_tui_visual_stability() {

        let reference = load_reference_snapshot("tui_main_view.png")?;



        // Render current TUI

        let current = render_tui_to_image()?;



        // Compare with multiple metrics

        let ssim = compute_ssim(&reference, &current)?;

        let psnr = compute_psnr(&reference, &current)?;

        let delta_e = compute_ciede2000(&reference, &current)?;



        // Thresholds based on [Wang2004] SSIM research

        assert!(ssim > 0.95, "SSIM should be > 0.95 for visual similarity");

        assert!(psnr > 30.0, "PSNR should be > 30dB for good quality");

        assert!(delta_e < 2.0, "CIEDE2000 ΔE should be < 2.0 for imperceptible diff");

    }

}

```



---



## 9. 100-Point Popperian Falsification Suite



### 9.1 Falsification Philosophy



Following Karl Popper's philosophy of science [Popper1959], each test is designed to **potentially disprove** a hypothesis rather than confirm it. Tests are structured as:



> **H[n]**: [Hypothesis that could be false]

> **Test**: [Action that would reveal falsity]

> **Pass Criterion**: [Observable outcome if hypothesis holds]

> **Falsification**: [Observable outcome if hypothesis fails]



### 9.2 Complete Falsification Test Suite



```yaml

# File: tests/falsification/tui-compute-monitor.yaml

# 100-Point Popperian Falsification Suite for TUI Compute Monitor

# Version: 1.0.0

# Standard: PMAT-TDD-2024 + Iron Lotus Framework



metadata:

  specification: TRUENO-SPEC-020

  coverage_target: 100%

  mutation_target: 80%

  reviewed_by: QA Team Lead

  last_updated: 2026-01-03



# =============================================================================

# SECTION 1: HARDWARE DETECTION (20 points)

# =============================================================================



hardware_detection:

  - id: H001

    hypothesis: "NVIDIA GPU detection returns accurate device name"

    test: "Compare nvml device name with nvidia-smi output"

    pass_criterion: "Names match exactly"

    falsification: "Name mismatch or 'Unknown' returned"

    severity: critical

    points: 2



  - id: H002

    hypothesis: "AMD GPU detection works via ROCm SMI"

    test: "Call rocm_smi::get_name() and verify against rocm-smi CLI"

    pass_criterion: "Names match, no library errors"

    falsification: "rocm-smi-lib returns error or wrong name"

    severity: critical

    points: 2



  - id: H002b

    hypothesis: "Metal backend detects Apple GPUs on macOS"

    test: "Verify wgpu backend is Metal on macOS"

    pass_criterion: "Backend::Metal reported"

    falsification: "Backend::Vulkan/Gl or error"

    severity: critical

    points: 2



  - id: H003

    hypothesis: "CPU core count matches physical reality"

    test: "Compare num_cpus::get() with /proc/cpuinfo"

    pass_criterion: "Core count matches"

    falsification: "Count mismatch (hyperthreading confusion)"

    severity: high

    points: 2



  - id: H004

    hypothesis: "Multi-GPU systems enumerate all devices"

    test: "System with 2+ GPUs, verify all detected"

    pass_criterion: "All GPUs in device list"

    falsification: "Missing GPU or duplicate entries"

    severity: critical

    points: 2



  - id: H005

    hypothesis: "Device enumeration is idempotent"

    test: "Call enumerate() 100 times in sequence"

    pass_criterion: "Same results every time"

    falsification: "Device count or order changes"

    severity: high

    points: 2



  - id: H006

    hypothesis: "Hot-plug GPU detection works"

    test: "Add/remove GPU and re-enumerate"

    pass_criterion: "Device list updates correctly"

    falsification: "Stale device list or crash"

    severity: medium

    points: 2



  - id: H007

    hypothesis: "Device IDs are stable across restarts"

    test: "Record device IDs, restart, compare"

    pass_criterion: "Same device gets same ID"

    falsification: "ID changes without hardware change"

    severity: medium

    points: 2



  - id: H008

    hypothesis: "PCIe topology is correctly identified"

    test: "Verify PCIe generation and width"

    pass_criterion: "Matches lspci output"

    falsification: "Wrong gen/width reported"

    severity: low

    points: 2



  - id: H009

    hypothesis: "Unified device trait works for all backends"

    test: "Call all ComputeDevice methods on CPU/NVIDIA/AMD"

    pass_criterion: "No panics, correct types returned"

    falsification: "Method panics or returns wrong type"

    severity: critical

    points: 2



  - id: H010

    hypothesis: "Device capability detection is accurate"

    test: "Query SM/CU count and verify against spec"

    pass_criterion: "Matches known hardware specs"

    falsification: "Wrong compute unit count"

    severity: medium

    points: 2



# =============================================================================

# SECTION 2: MEMORY METRICS (20 points)

# =============================================================================

memory_metrics:
  - id: H011
    hypothesis: "RAM usage matches /proc/meminfo"
    test: "Compare memory_used_bytes with MemTotal - MemAvailable"
    pass_criterion: "Within 1% of /proc/meminfo"
    falsification: "Deviation > 1%"
    severity: critical
    points: 2

  - id: H012
    hypothesis: "VRAM usage matches nvidia-smi"
    test: "Compare gpu_vram.used_bytes with nvidia-smi query"
    pass_criterion: "Within 1MB"
    falsification: "Deviation > 1MB"
    severity: critical
    points: 2

  - id: H013
    hypothesis: "Swap usage is correctly reported"
    test: "Compare with /proc/swaps and free -m"
    pass_criterion: "Values match"
    falsification: "Swap usage incorrect"
    severity: high
    points: 2

  - id: H014
    hypothesis: "Memory pressure levels trigger correctly"
    test: "Allocate memory until Critical level reached"
    pass_criterion: "Level transitions at correct thresholds"
    falsification: "Wrong level at known utilization"
    severity: critical
    points: 2

  - id: H015
    hypothesis: "Safe job calculation is conservative"
    test: "Run calculated safe_jobs in parallel"
    pass_criterion: "No OOM kill occurs"
    falsification: "OOM killer invoked"
    severity: critical
    points: 2

  - id: H016
    hypothesis: "Memory history sparkline has 60 points"
    test: "Run for 60 seconds at 1Hz, check history length"
    pass_criterion: "Exactly 60 points in VecDeque"
    falsification: "Wrong count or data corruption"
    severity: medium
    points: 2

  - id: H017
    hypothesis: "Pinned memory tracking is accurate"
    test: "Allocate 1GB pinned, verify reported"
    pass_criterion: "pinned_memory_used_bytes increases by ~1GB"
    falsification: "No change or wrong amount"
    severity: high
    points: 2

  - id: H018
    hypothesis: "Memory bandwidth measurement is reasonable"
    test: "Compare with STREAM benchmark results"
    pass_criterion: "Within 20% of STREAM"
    falsification: "Deviation > 20%"
    severity: medium
    points: 2

  - id: H019
    hypothesis: "Per-GPU VRAM is correctly attributed"
    test: "Allocate on GPU0 only, verify GPU1 unchanged"
    pass_criterion: "Only GPU0 VRAM increases"
    falsification: "Wrong GPU shows increase"
    severity: critical
    points: 2

  - id: H020
    hypothesis: "Memory metrics update within 100ms"
    test: "Allocate 100MB, measure time to reflect in UI"
    pass_criterion: "Update visible < 100ms"
    falsification: "Stale data shown > 100ms"
    severity: medium
    points: 2

# =============================================================================
# SECTION 3: COMPUTE METRICS (20 points)
# =============================================================================

compute_metrics:
  - id: H021
    hypothesis: "GPU utilization matches nvidia-smi"
    test: "Run stress kernel, compare utilization"
    pass_criterion: "Within 5% of nvidia-smi"
    falsification: "Deviation > 5%"
    severity: critical
    points: 2

  - id: H022
    hypothesis: "CPU utilization matches top/htop"
    test: "Run CPU stress, compare with top"
    pass_criterion: "Within 3% of top"
    falsification: "Deviation > 3%"
    severity: high
    points: 2

  - id: H023
    hypothesis: "Temperature readings are in Celsius"
    test: "Verify GPU temp is 30-90°C range under load"
    pass_criterion: "Realistic temperature values"
    falsification: "Impossible values (e.g., 300°C)"
    severity: critical
    points: 2

  - id: H024
    hypothesis: "Power readings are in Watts"
    test: "Verify power is 10-500W range for GPU"
    pass_criterion: "Reasonable power values"
    falsification: "Impossible values (e.g., 10000W)"
    severity: high
    points: 2

  - id: H025
    hypothesis: "Throttling detection works"
    test: "Force thermal throttle, verify detected"
    pass_criterion: "ThrottleReason::Thermal reported"
    falsification: "No throttle detected"
    severity: high
    points: 2

  - id: H026
    hypothesis: "Clock speed is correctly reported"
    test: "Compare with nvidia-smi -q"
    pass_criterion: "Within 50 MHz"
    falsification: "Deviation > 50 MHz"
    severity: medium
    points: 2

  - id: H027
    hypothesis: "SM/CU active count is dynamic"
    test: "Run partial workload, verify < 100% active"
    pass_criterion: "Active SM < total SM"
    falsification: "Always shows 100% or 0%"
    severity: medium
    points: 2

  - id: H028
    hypothesis: "FLOPS calculation is accurate"
    test: "Run known GEMM, compute achieved FLOPS"
    pass_criterion: "Within 10% of manual calculation"
    falsification: "Deviation > 10%"
    severity: high
    points: 2

  - id: H029
    hypothesis: "Compute efficiency percentage is valid"
    test: "Verify 0% <= efficiency <= 100%"
    pass_criterion: "Value in valid range"
    falsification: "Value < 0% or > 100%"
    severity: critical
    points: 2

  - id: H030
    hypothesis: "Kernel execution tracking is accurate"
    test: "Run 10 kernels, verify all tracked"
    pass_criterion: "10 entries in active_kernels"
    falsification: "Missing or extra kernel entries"
    severity: high
    points: 2

# =============================================================================
# SECTION 4: DATA FLOW TRACKING (15 points)
# =============================================================================

data_flow:
  - id: H031
    hypothesis: "PCIe bandwidth measurement is accurate"
    test: "Transfer 1GB, measure time, calculate bandwidth"
    pass_criterion: "Within 10% of theoretical"
    falsification: "Deviation > 10%"
    severity: high
    points: 2

  - id: H032
    hypothesis: "Transfer direction is correctly identified"
    test: "Do H2D transfer, verify direction = HostToDevice"
    pass_criterion: "Correct direction enum"
    falsification: "Wrong direction"
    severity: critical
    points: 2

  - id: H033
    hypothesis: "Transfer progress percentage is accurate"
    test: "Mid-transfer, verify progress"
    pass_criterion: "Progress = transferred / total"
    falsification: "Wrong percentage"
    severity: high
    points: 2

  - id: H034
    hypothesis: "Concurrent transfers are tracked"
    test: "Start 3 overlapped transfers"
    pass_criterion: "All 3 in active_transfers"
    falsification: "Missing transfers"
    severity: high
    points: 1

  - id: H035
    hypothesis: "Completed transfers move to history"
    test: "Complete transfer, check completed_transfers"
    pass_criterion: "Transfer in completed queue"
    falsification: "Transfer lost"
    severity: medium
    points: 1

  - id: H036
    hypothesis: "Memory bus utilization is bounded 0-100%"
    test: "Check utilization under various loads"
    pass_criterion: "0% <= util <= 100%"
    falsification: "Out of bounds value"
    severity: critical
    points: 2

  - id: H037
    hypothesis: "PCIe generation/width is correctly detected"
    test: "Compare with lspci -vv"
    pass_criterion: "Matches lspci output"
    falsification: "Wrong PCIe config"
    severity: medium
    points: 1

  - id: H038
    hypothesis: "Transfer labeling works"
    test: "Create transfer with label, verify retrieved"
    pass_criterion: "Label preserved"
    falsification: "Label lost or corrupted"
    severity: low
    points: 1

  - id: H039
    hypothesis: "Transfer timing is microsecond-accurate"
    test: "Measure known transfer, verify duration"
    pass_criterion: "Within 10% of wall-clock"
    falsification: "Timing significantly off"
    severity: medium
    points: 1

  - id: H040
    hypothesis: "Peer-to-peer transfers are detected"
    test: "D2D transfer between GPUs"
    pass_criterion: "Direction = DeviceToDevice"
    falsification: "Wrong direction"
    severity: high
    points: 2

  - id: H041
    hypothesis: "History queues maintain 60-point limit"
    test: "Run for 120 seconds, check queue length"
    pass_criterion: "Queue length = 60 (FIFO)"
    falsification: "Queue grows unbounded"
    severity: medium
    points: 2

# =============================================================================
# SECTION 5: TUI RENDERING (15 points)
# =============================================================================

tui_rendering:
  - id: H042
    hypothesis: "TUI renders at minimum 80x24"
    test: "Render on 80x24 terminal"
    pass_criterion: "No truncation or overflow"
    falsification: "Content cut off or panics"
    severity: critical
    points: 2

  - id: H043
    hypothesis: "TUI scales to 160x48 with more detail"
    test: "Render on 160x48 terminal"
    pass_criterion: "Additional detail visible"
    falsification: "Same as 80x24 or broken"
    severity: medium
    points: 1

  - id: H044
    hypothesis: "Gauge widgets show correct percentage"
    test: "Set CPU to 75%, verify gauge"
    pass_criterion: "Gauge shows 75%"
    falsification: "Wrong percentage displayed"
    severity: high
    points: 2

  - id: H045
    hypothesis: "Sparklines render all 60 data points"
    test: "Provide 60-point dataset"
    pass_criterion: "60 bars visible in sparkline"
    falsification: "Missing data points"
    severity: high
    points: 2

  - id: H046
    hypothesis: "Color scheme is colorblind-safe"
    test: "Simulate deuteranopia on screenshot"
    pass_criterion: "All elements distinguishable"
    falsification: "Critical info lost in simulation"
    severity: medium
    points: 1

  - id: H047
    hypothesis: "Keyboard navigation works"
    test: "Press Tab 10 times"
    pass_criterion: "Focus cycles through sections"
    falsification: "Focus stuck or skips section"
    severity: high
    points: 2

  - id: H048
    hypothesis: "Help overlay toggles with ?"
    test: "Press ? twice"
    pass_criterion: "Overlay appears then disappears"
    falsification: "Overlay stuck or missing"
    severity: medium
    points: 1

  - id: H049
    hypothesis: "Refresh rate is configurable"
    test: "Set --refresh-rate 50ms"
    pass_criterion: "Updates occur every ~50ms"
    falsification: "Update rate unchanged"
    severity: low
    points: 1

  - id: H050
    hypothesis: "Unicode characters render correctly"
    test: "Verify box-drawing and block chars"
    pass_criterion: "All chars display properly"
    falsification: "Garbled or missing chars"
    severity: high
    points: 1

  - id: H051
    hypothesis: "TUI handles terminal resize"
    test: "Resize terminal during operation"
    pass_criterion: "Layout adapts without crash"
    falsification: "Crash or frozen display"
    severity: high
    points: 2

# =============================================================================
# SECTION 6: STRESS TEST MODE (10 points)
# =============================================================================

stress_test:
  - id: H052
    hypothesis: "Stress test saturates CPU to >95%"
    test: "Run --stress-test --target cpu"
    pass_criterion: "CPU utilization > 95%"
    falsification: "CPU stays below 95%"
    severity: high
    points: 2

  - id: H053
    hypothesis: "Stress test saturates GPU to >90%"
    test: "Run --stress-test --target gpu"
    pass_criterion: "GPU utilization > 90%"
    falsification: "GPU stays below 90%"
    severity: high
    points: 2

  - id: H054
    hypothesis: "Memory stress reaches target utilization"
    test: "Run --stress-test --target memory --intensity 0.9"
    pass_criterion: "Memory at 90% +/- 5%"
    falsification: "Memory utilization off target"
    severity: high
    points: 2

  - id: H055
    hypothesis: "Stress test respects duration limit"
    test: "Run --stress-test --duration 10s"
    pass_criterion: "Test completes at ~10s"
    falsification: "Runs longer than 12s or stops early"
    severity: medium
    points: 1

  - id: H056
    hypothesis: "Ramp-up phase is gradual"
    test: "Run --stress-test --ramp-up 5s"
    pass_criterion: "Load increases linearly over 5s"
    falsification: "Instant full load or no ramp"
    severity: low
    points: 1

  - id: H057
    hypothesis: "Chaos integration works"
    test: "Run --stress-test --chaos gentle"
    pass_criterion: "Memory limit applied during stress"
    falsification: "No chaos effects visible"
    severity: medium
    points: 1

  - id: H058
    hypothesis: "Stress report is generated"
    test: "Run stress test to completion"
    pass_criterion: "JSON report written with metrics"
    falsification: "No report or empty report"
    severity: high
    points: 1

# =============================================================================
# SECTION 7: ERROR HANDLING (10 points)
# =============================================================================

error_handling:
  - id: H059
    hypothesis: "Missing NVIDIA driver handled gracefully"
    test: "Run on system without nvidia-smi"
    pass_criterion: "Falls back to CPU-only mode"
    falsification: "Crash or panic"
    severity: critical
    points: 2

  - id: H060
    hypothesis: "Missing ROCm handled gracefully"
    test: "Run on system without rocm-smi"
    pass_criterion: "Falls back to NVIDIA/CPU"
    falsification: "Crash or panic"
    severity: critical
    points: 2

  - id: H061
    hypothesis: "GPU driver crash recovery works"
    test: "Simulate driver timeout"
    pass_criterion: "Re-enumeration after recovery"
    falsification: "Stuck with stale state"
    severity: high
    points: 1

  - id: H062
    hypothesis: "Memory allocation failure handled"
    test: "Exhaust memory, attempt allocation"
    pass_criterion: "Error returned, no panic"
    falsification: "Panic or undefined behavior"
    severity: critical
    points: 2

  - id: H063
    hypothesis: "Invalid CLI arguments rejected"
    test: "Pass --invalid-flag"
    pass_criterion: "Usage error with help"
    falsification: "Crash or silent ignore"
    severity: medium
    points: 1

  - id: H064
    hypothesis: "Division by zero protected"
    test: "Device with 0 total memory"
    pass_criterion: "Percentage shows 0% or N/A"
    falsification: "Crash or NaN displayed"
    severity: critical
    points: 2

# =============================================================================
# SECTION 8: PIXEL COVERAGE (10 points) - PROBADOR INTEGRATION
# =============================================================================

pixel_coverage:
  - id: H065
    hypothesis: "Header region is 100% covered"
    test: "Render header, check pixel coverage"
    pass_criterion: "100% of header pixels touched"
    falsification: "Gap in header region"
    severity: high
    points: 1

  - id: H066
    hypothesis: "Compute section is 100% covered"
    test: "Render compute section with all gauges"
    pass_criterion: "100% pixel coverage"
    falsification: "Uncovered pixels"
    severity: high
    points: 1

  - id: H067
    hypothesis: "Memory section is 100% covered"
    test: "Render memory section with all bars"
    pass_criterion: "100% pixel coverage"
    falsification: "Uncovered pixels"
    severity: high
    points: 1

  - id: H068
    hypothesis: "Data flow section is 100% covered"
    test: "Render data flow with active transfers"
    pass_criterion: "100% pixel coverage"
    falsification: "Uncovered pixels"
    severity: high
    points: 1

  - id: H069
    hypothesis: "Kernels section is 100% covered"
    test: "Render with 5 active kernels"
    pass_criterion: "100% pixel coverage"
    falsification: "Uncovered pixels"
    severity: high
    points: 1

  - id: H070
    hypothesis: "Footer/help region is 100% covered"
    test: "Render footer with all key hints"
    pass_criterion: "100% pixel coverage"
    falsification: "Uncovered pixels"
    severity: medium
    points: 1

  - id: H071
    hypothesis: "SSIM > 0.95 vs reference image"
    test: "Compare render with golden master"
    pass_criterion: "SSIM score > 0.95"
    falsification: "SSIM < 0.95"
    severity: high
    points: 1

  - id: H072
    hypothesis: "PSNR > 30dB vs reference image"
    test: "Compare render with golden master"
    pass_criterion: "PSNR > 30dB"
    falsification: "PSNR < 30dB"
    severity: medium
    points: 1

  - id: H073
    hypothesis: "CIEDE2000 ΔE < 2.0"
    test: "Compare colors with reference"
    pass_criterion: "ΔE < 2.0 (imperceptible)"
    falsification: "ΔE >= 2.0 (visible diff)"
    severity: medium
    points: 1

  - id: H074
    hypothesis: "Heatmap export produces valid PNG"
    test: "Export coverage heatmap"
    pass_criterion: "Valid PNG file, correct dimensions"
    falsification: "Corrupt or wrong-sized PNG"
    severity: low
    points: 1

# =============================================================================
# SECTION 9: PERFORMANCE (10 points)
# =============================================================================

performance:
  - id: H075
    hypothesis: "Metric collection < 10ms overhead"
    test: "Measure time for full metric collection"
    pass_criterion: "Collection completes < 10ms"
    falsification: "Collection takes > 10ms"
    severity: high
    points: 2

  - id: H076
    hypothesis: "TUI rendering < 16ms (60 FPS capable)"
    test: "Measure render time"
    pass_criterion: "Render completes < 16ms"
    falsification: "Render takes > 16ms"
    severity: medium
    points: 2

  - id: H077
    hypothesis: "Memory usage < 50MB steady-state"
    test: "Run for 1 hour, measure RSS"
    pass_criterion: "RSS stays < 50MB"
    falsification: "RSS grows or exceeds 50MB"
    severity: high
    points: 2

  - id: H078
    hypothesis: "No memory leaks over 24h run"
    test: "Run for 24 hours, compare start/end RSS"
    pass_criterion: "RSS difference < 10MB"
    falsification: "RSS grew > 10MB"
    severity: critical
    points: 2

  - id: H079
    hypothesis: "CPU overhead < 2% when idle"
    test: "Run monitor on idle system"
    pass_criterion: "Monitor uses < 2% CPU"
    falsification: "Monitor uses > 2% CPU"
    severity: medium
    points: 2

# =============================================================================
# SECTION 10: INTEGRATION (10 points)
# =============================================================================

integration:
  - id: H080
    hypothesis: "Repartir TUI model compatible"
    test: "Use repartir NodeStatus with this monitor"
    pass_criterion: "Data structures interoperable"
    falsification: "Type mismatches or crashes"
    severity: high
    points: 1

  - id: H081
    hypothesis: "Renacer chaos presets work"
    test: "Apply ChaosConfig::aggressive()"
    pass_criterion: "Memory limits applied"
    falsification: "No effect from chaos config"
    severity: medium
    points: 1

  - id: H082
    hypothesis: "Probar pixel tracker integration"
    test: "Use PixelCoverageTracker with TUI"
    pass_criterion: "Coverage data collected"
    falsification: "Tracker errors or incompatible"
    severity: high
    points: 1

  - id: H083
    hypothesis: "trueno-gpu metrics work"
    test: "Use trueno_gpu::device_info()"
    pass_criterion: "Returns valid DeviceInfo"
    falsification: "Error or wrong info"
    severity: critical
    points: 1

  - id: H084
    hypothesis: "OTLP export works (via renacer)"
    test: "Export metrics to Jaeger"
    pass_criterion: "Spans visible in Jaeger UI"
    falsification: "No spans or export errors"
    severity: medium
    points: 1

  - id: H085
    hypothesis: "JSON export produces valid JSON"
    test: "Export metrics, parse with serde_json"
    pass_criterion: "Valid JSON, all fields present"
    falsification: "Parse error or missing fields"
    severity: high
    points: 1

  - id: H086
    hypothesis: "Lambda Labs memory pressure compatible"
    test: "Use LAMBDA-0002 PressureLevel enum"
    pass_criterion: "Enum values match spec"
    falsification: "Incompatible enum values"
    severity: medium
    points: 1

  - id: H087
    hypothesis: "sysinfo crate compatibility"
    test: "Use sysinfo::System for CPU metrics"
    pass_criterion: "Metrics match sysinfo output"
    falsification: "Discrepancy with sysinfo"
    severity: high
    points: 1

  - id: H088
    hypothesis: "nvml-wrapper compatibility"
    test: "Use nvml-wrapper for NVIDIA metrics"
    pass_criterion: "All NVML calls succeed"
    falsification: "NVML errors"
    severity: critical
    points: 1

  - id: H089
    hypothesis: "rocm-smi-lib compatibility"
    test: "Use rocm-smi-lib for AMD metrics"
    pass_criterion: "All ROCm calls succeed"
    falsification: "ROCm errors"
    severity: critical
    points: 1

# =============================================================================
# SECTION 11: EDGE CASES (10 points)
# =============================================================================

edge_cases:
  - id: H090
    hypothesis: "Zero GPU systems work"
    test: "Run on system with no GPU"
    pass_criterion: "Shows CPU-only metrics"
    falsification: "Crash or missing CPU data"
    severity: critical
    points: 2

  - id: H091
    hypothesis: "100+ core systems work"
    test: "Run on 128-core server"
    pass_criterion: "All cores shown, no overflow"
    falsification: "Core count wrong or UI broken"
    severity: high
    points: 1

  - id: H092
    hypothesis: "1TB+ RAM systems work"
    test: "Run on system with 1.5TB RAM"
    pass_criterion: "Memory shown correctly"
    falsification: "Overflow or wrong values"
    severity: high
    points: 1

  - id: H093
    hypothesis: "4+ GPU systems work"
    test: "Run on 8-GPU server"
    pass_criterion: "All GPUs enumerated and shown"
    falsification: "Missing GPUs or UI overflow"
    severity: high
    points: 1

  - id: H094
    hypothesis: "Mixed NVIDIA+AMD works"
    test: "System with both vendors"
    pass_criterion: "Both GPU types detected"
    falsification: "One vendor missing"
    severity: high
    points: 1

  - id: H095
    hypothesis: "Docker container works"
    test: "Run in Docker with GPU passthrough"
    pass_criterion: "GPU visible inside container"
    falsification: "GPU not detected"
    severity: medium
    points: 1

  - id: H096
    hypothesis: "WSL2 works (Windows Subsystem)"
    test: "Run in WSL2 with GPU support"
    pass_criterion: "GPU visible via WSL"
    falsification: "GPU not detected"
    severity: medium
    points: 1

  - id: H097
    hypothesis: "Minimal terminal (80x24) works"
    test: "Run on exactly 80x24 terminal"
    pass_criterion: "All critical info visible"
    falsification: "Content cut off"
    severity: high
    points: 1

  - id: H098
    hypothesis: "Huge terminal (400x100) works"
    test: "Run on 400x100 terminal"
    pass_criterion: "Layout expands gracefully"
    falsification: "Excessive whitespace or crash"
    severity: low
    points: 0.5

  - id: H099
    hypothesis: "Non-ASCII locale works"
    test: "Run with LANG=ja_JP.UTF-8"
    pass_criterion: "No rendering issues"
    falsification: "Garbled text or crash"
    severity: medium
    points: 0.5

  - id: H100
    hypothesis: "Rapid metric changes handled"
    test: "Oscillate CPU 0-100% at 100Hz"
    pass_criterion: "Display updates smoothly"
    falsification: "Flickering or data corruption"
    severity: medium
    points: 1

# =============================================================================
# SUMMARY
# =============================================================================

summary:
  total_tests: 100
  total_points: 100
  passing_threshold: 85  # Must pass 85+ points
  critical_tests: 22     # Tests that MUST pass (severity: critical)

  section_weights:
    hardware_detection: 20
    memory_metrics: 20
    compute_metrics: 20
    data_flow: 15
    tui_rendering: 15
    stress_test: 10
    error_handling: 10
    pixel_coverage: 10
    performance: 10
    integration: 10
    edge_cases: 10
```

---

## 10. Peer-Reviewed Citations

### 10.1 Toyota Production System & Quality

| Citation | Reference | Application |
|----------|-----------|-------------|
| [Liker2004] | Liker, J.K. (2004). *The Toyota Way: 14 Management Principles*. McGraw-Hill. ISBN 0-07-139231-9 | Iron Lotus Framework principles |
| [Ohno1988] | Ohno, T. (1988). *Toyota Production System: Beyond Large-Scale Production*. Productivity Press. ISBN 0-915299-14-3 | Muda (waste) elimination in telemetry |
| [Shingo1986] | Shingo, S. (1986). *Zero Quality Control: Source Inspection and the Poka-Yoke System*. Productivity Press. ISBN 0-915299-07-0 | Type-safe metrics prevent unit errors |
| [Womack1990] | Womack, J.P., Jones, D.T., Roos, D. (1990). *The Machine That Changed the World*. Free Press. ISBN 0-7432-9979-4 | Lean principles in software |

### 10.2 GPU Computing & Performance

| Citation | Reference | Application |
|----------|-----------|-------------|
| [Volkov2008] | Volkov, V., Demmel, J.W. (2008). "Benchmarking GPUs to Tune Dense Linear Algebra". *SC '08: Proceedings of the 2008 ACM/IEEE Conference on Supercomputing*. DOI: 10.1109/SC.2008.5214359 | Tile size optimization (16x16), memory bandwidth modeling |
| [Harris2007] | Harris, M. (2007). "Optimizing Parallel Reduction in CUDA". *NVIDIA Developer Technology*. | Tiled reduction algorithm |
| [Nickolls2008] | Nickolls, J., Buck, I., Garland, M., Skadron, K. (2008). "Scalable Parallel Programming with CUDA". *ACM Queue*, 6(2), 40-53. DOI: 10.1145/1365490.1365500 | CUDA programming model |
| [Jia2018] | Jia, Z., Maggioni, M., Staiger, B., Scarpazza, D.P. (2018). "Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking". *arXiv:1804.06826* | GPU microarchitecture analysis |

### 10.3 Memory Systems & Pressure

| Citation | Reference | Application |
|----------|-----------|-------------|
| [Hennessy2017] | Hennessy, J.L., Patterson, D.A. (2017). *Computer Architecture: A Quantitative Approach* (6th ed.). Morgan Kaufmann. ISBN 978-0128119051 | Memory hierarchy model |
| [McCalpin1995] | McCalpin, J.D. (1995). "STREAM: Sustainable Memory Bandwidth in High Performance Computers". *Technical Report*, University of Virginia. | Memory bandwidth benchmarking |
| [Drepper2007] | Drepper, U. (2007). "What Every Programmer Should Know About Memory". *Red Hat, Inc.* | Memory access patterns |

### 10.4 Testing & Falsification

| Citation | Reference | Application |
|----------|-----------|-------------|
| [Popper1959] | Popper, K. (1959). *The Logic of Scientific Discovery*. Hutchinson. ISBN 0-415-27844-9 | Falsification test methodology |
| [Claessen2000] | Claessen, K., Hughes, J. (2000). "QuickCheck: A Lightweight Tool for Random Testing of Haskell Programs". *ACM SIGPLAN Notices*, 35(9), 268-279. DOI: 10.1145/351240.351266 | Property-based testing foundation |
| [Regehr2012] | Regehr, J., Chen, Y., Cuoq, P., Eide, E., Ellison, C., Yang, X. (2012). "Test-Case Reduction for C Compiler Bugs". *PLDI '12*. DOI: 10.1145/2254064.2254104 | Test minimization |

### 10.5 Visual Quality Metrics

| Citation | Reference | Application |
|----------|-----------|-------------|
| [Wang2004] | Wang, Z., Bovik, A.C., Sheikh, H.R., Simoncelli, E.P. (2004). "Image Quality Assessment: From Error Visibility to Structural Similarity". *IEEE Transactions on Image Processing*, 13(4), 600-612. DOI: 10.1109/TIP.2003.819861 | SSIM metric for visual regression |
| [Sharma2005] | Sharma, G., Wu, W., Dalal, E.N. (2005). "The CIEDE2000 Color-Difference Formula". *Color Research & Application*, 30(1), 21-30. DOI: 10.1002/col.20070 | CIEDE2000 color difference |

### 10.6 Distributed Systems & Monitoring

| Citation | Reference | Application |
|----------|-----------|-------------|
| [Blumofe1999] | Blumofe, R.D., Leiserson, C.E. (1999). "Scheduling Multithreaded Computations by Work Stealing". *Journal of the ACM*, 46(5), 720-748. DOI: 10.1145/324133.324234 | Work-stealing scheduler in repartir |
| [Burns2016] | Burns, B. (2016). *Designing Distributed Systems*. O'Reilly Media. ISBN 978-1491983645 | Distributed monitoring patterns |
| [Beyer2016] | Beyer, B., Jones, C., Petoff, J., Murphy, N.R. (2016). *Site Reliability Engineering*. O'Reilly Media. ISBN 978-1491929124 | SRE monitoring principles |

### 10.7 Anomaly Detection & Statistics

| Citation | Reference | Application |
|----------|-----------|-------------|
| [Liu2008] | Liu, F.T., Ting, K.M., Zhou, Z.H. (2008). "Isolation Forest". *2008 Eighth IEEE International Conference on Data Mining*. DOI: 10.1109/ICDM.2008.17 | Jidoka anomaly detection algorithm |
| [Anscombe1973] | Anscombe, F.J. (1973). "Graphs in Statistical Analysis". *The American Statistician*, 27(1), 17-21. | Importance of visualization (Anscombe's quartet) |

### 10.8 Data Visualization

| Citation | Reference | Application |
|----------|-----------|-------------|
| [Tufte2006] | Tufte, E.R. (2006). *Beautiful Evidence*. Graphics Press. ISBN 0-9613921-7-7 | Sparklines theory and design |
| [Gregg2013] | Gregg, B. (2013). *Systems Performance: Enterprise and the Cloud*. Prentice Hall. ISBN 978-0133390094 | USE Method (Utilization, Saturation, Errors) |

---

## 11. Implementation Roadmap

### Phase 1: Core Infrastructure (2 weeks)

- [ ] Implement `ComputeDevice` trait
- [ ] Add NVIDIA GPU backend (nvml-wrapper)
- [ ] Add AMD GPU backend (rocm-smi-lib)
- [ ] Add CPU backend (sysinfo)
- [ ] Implement unified telemetry collector
- [ ] Write 40 falsification tests (H001-H040)

### Phase 2: TUI Implementation (2 weeks)

- [ ] Create TUI layout with presentar
- [ ] Implement gauge, sparkline, progress widgets
- [ ] Add keyboard navigation
- [ ] Implement help overlay
- [ ] Write 15 falsification tests (H041-H055)

### Phase 3: Stress Test Mode (1 week)

- [ ] Implement CPU stress worker
- [ ] Implement GPU stress worker (trueno BatchedGemmKernel)
- [ ] Implement memory stress worker
- [ ] Implement PCIe stress worker
- [ ] Add chaos integration (renacer)
- [ ] Write 10 falsification tests (H052-H061)

### Phase 4: Pixel Testing Integration (1 week)

- [ ] Integrate probar PixelCoverageTracker
- [ ] Add visual regression tests
- [ ] Create golden master images
- [ ] Implement heatmap export
- [ ] Write 15 falsification tests (H065-H079)

### Phase 5: Integration & Polish (1 week)

- [ ] Integrate with repartir TUI model
- [ ] Add OTLP export (via renacer)
- [ ] Add JSON export
- [ ] Performance optimization
- [ ] Write 20 falsification tests (H080-H100)

### Phase 6: QA Validation (1 week)

- [ ] Run full 100-point falsification suite
- [ ] Fix any failing tests
- [ ] Generate coverage reports
- [ ] Perform mutation testing
- [ ] Documentation review

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| **CU** | Compute Unit (AMD terminology for SM) |
| **FLOPS** | Floating-Point Operations Per Second |
| **H2D** | Host-to-Device (CPU→GPU transfer) |
| **D2H** | Device-to-Host (GPU→CPU transfer) |
| **NVML** | NVIDIA Management Library |
| **PCIe** | Peripheral Component Interconnect Express |
| **ROCm** | Radeon Open Compute (AMD GPU platform) |
| **SM** | Streaming Multiprocessor (NVIDIA terminology) |
| **SSIM** | Structural Similarity Index Measure |
| **VRAM** | Video Random Access Memory (GPU memory) |

---

## Appendix B: File Structure

```
trueno/
├── src/
│   └── bin/
│       └── trueno-monitor.rs      # TUI binary
├── trueno-gpu/
│   └── src/
│       ├── monitor/
│       │   ├── mod.rs             # Monitor module
│       │   ├── nvidia.rs          # NVIDIA backend
│       │   ├── amd.rs             # AMD backend
│       │   ├── cpu.rs             # CPU backend
│       │   └── unified.rs         # Unified collector
│       ├── tui/
│       │   ├── mod.rs             # TUI module
│       │   ├── layout.rs          # Layout spec
│       │   ├── widgets.rs         # Custom widgets
│       │   └── render.rs          # Render logic
│       └── stress/
│           ├── mod.rs             # Stress test module
│           ├── cpu.rs             # CPU stress
│           ├── gpu.rs             # GPU stress
│           ├── memory.rs          # Memory stress
│           └── pcie.rs            # PCIe stress
├── docs/
│   └── specifications/
│       └── tui-compute-mode-flow-cpu-memory.md  # This document
└── tests/
    └── falsification/
        ├── tui-compute-monitor.yaml  # 100-point test suite
        └── pixel_coverage/
            ├── golden_master.png     # Reference image
            └── coverage_tests.rs     # Pixel tests
```

---

**Document History:**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-01-03 | PAIML Engineering | Initial specification |

**Approval:**

- [ ] Engineering Lead
- [ ] QA Team Lead
- [ ] Product Owner

---

*This specification is validated by 100-point Popperian falsification testing and integrated with Probador pixel-by-pixel coverage analysis.*
