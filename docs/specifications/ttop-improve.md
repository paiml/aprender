# TTOP Enhancement Specification: World's Best Terse Systems Monitor

**Version**: 2.1.0
**Status**: Draft
**Target Component**: `crates/ttop` & `trueno-viz/monitor`
**Authors**: Sovereign AI Stack Team
**Last Updated**: 2026-01-08
**Goal**: 100% accuracy, SIMD-first, cross-platform (Linux + SSH Mac)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Swap Analysis](#2-swap-analysis)
3. [Disk I/O Analysis](#3-disk-io-analysis)
4. [Disk Storage (GB) & Large File Anomaly Detection](#4-disk-storage-gb--large-file-anomaly-detection)
5. [Feature Enhancements](#5-feature-enhancements)
6. [Bug Fixes & Stability](#6-bug-fixes--stability)
7. [Architecture: SIMD-First with trueno](#7-architecture-simd-first-with-trueno)
8. [Cross-Platform: SSH Mac Support](#8-cross-platform-ssh-mac-support)
9. [Testing: probador Integration](#9-testing-probador-integration)
10. [100-Point Popperian Falsification Checklist](#10-100-point-popperian-falsification-checklist)
11. [Peer-Reviewed References](#11-peer-reviewed-references)

---

## 1. Executive Summary

This specification defines `ttop` as the **world's best terse systems monitor** with three non-negotiable requirements:

1. **100% Accuracy**: Every metric matches kernel/hardware ground truth within defined tolerances
2. **SIMD-First**: All compute-intensive operations leverage `trueno` SIMD backends (AVX2/AVX-512/NEON)
3. **Cross-Platform**: Native Linux + transparent SSH Mac monitoring (Apple Silicon via `sysctl`/`IOKit`)

Key innovations:
- **Swap thrashing detection** via PSI (Pressure Stall Information) + page fault rate analysis
- **Disk I/O latency percentiles** using Little's Law estimation without eBPF
- **Real-time large file anomaly detection** via `fanotify` + statistical outlier detection
- **ZRAM integration** from `trueno-zram` for compression ratio visualization

---

## 2. Swap Analysis

### 2.1 Data Sources

| Metric | Linux Source | Mac Source (SSH) | Update Interval |
|--------|--------------|------------------|-----------------|
| Swap Total/Used/Free | `/proc/meminfo` | `sysctl vm.swapusage` | 1s |
| Pages In/Out (rate) | `/proc/vmstat` (`pswpin`, `pswpout`) | `vm_stat` | 1s |
| Swap Pressure (PSI) | `/proc/pressure/memory` | N/A (derive from page faults) | 100ms |
| Page Fault Rate | `/proc/vmstat` (`pgfault`, `pgmajfault`) | `vm_stat` (`Pageins`, `Pageouts`) | 1s |
| ZRAM Compression | `/sys/block/zram*/mm_stat` | N/A | 1s |

### 2.2 Swap Thrashing Detection Algorithm

Thrashing occurs when the system spends more time swapping than executing. Detection uses a **multi-signal approach** (Denning, 1968):

```rust
/// Thrashing detection per Denning's Working Set Model
pub struct SwapAnalyzer {
    pswpin_history: RingBuffer<u64, 60>,   // 60s window
    pswpout_history: RingBuffer<u64, 60>,
    pgmajfault_history: RingBuffer<u64, 60>,
    psi_some_avg10: f64,
}

impl SwapAnalyzer {
    /// Returns thrashing severity: None, Mild, Moderate, Severe
    pub fn detect_thrashing(&self) -> ThrashingSeverity {
        let swap_rate = self.pswpin_history.rate_per_sec() + self.pswpout_history.rate_per_sec();
        let fault_rate = self.pgmajfault_history.rate_per_sec();
        let psi_pressure = self.psi_some_avg10;

        match (swap_rate, fault_rate, psi_pressure) {
            (s, f, p) if p > 50.0 || (s > 1000 && f > 100) => ThrashingSeverity::Severe,
            (s, f, p) if p > 25.0 || (s > 500 && f > 50) => ThrashingSeverity::Moderate,
            (s, f, p) if p > 10.0 || (s > 100 && f > 10) => ThrashingSeverity::Mild,
            _ => ThrashingSeverity::None,
        }
    }
}
```

### 2.3 ZRAM Integration (from trueno-zram)

Leverage `trueno-zram` patterns for compression monitoring:

```rust
// From trueno-zram: /home/noah/src/trueno-zram/crates/trueno-zram-core/src/zram/device.rs
pub struct ZramStats {
    pub orig_data_size: u64,      // Uncompressed bytes
    pub compr_data_size: u64,     // Compressed bytes
    pub mem_used_total: u64,      // Total memory including metadata
    pub comp_algorithm: String,   // lz4, zstd, etc.
}

impl ZramStats {
    pub fn compression_ratio(&self) -> f64 {
        if self.compr_data_size == 0 { return 1.0; }
        self.orig_data_size as f64 / self.compr_data_size as f64
    }

    pub fn space_savings_percent(&self) -> f64 {
        if self.orig_data_size == 0 { return 0.0; }
        (1.0 - (self.compr_data_size as f64 / self.orig_data_size as f64)) * 100.0
    }
}
```

### 2.4 Visualization

```
┌─ Swap ────────────────────────────────────────────────────┐
│ Total: 32.0G │ Used: 2.1G (6.5%) │ ZRAM: 4:1 ratio       │
│ ▁▂▃▄▅▆▇█▇▆▅▄▃▂▁▂▃▄▅▆ Pages In/Out (60s)                  │
│ PSI: some=2.3% full=0.1% │ Status: OK                    │
│ Thrashing: None │ Major Faults: 3/s                      │
└───────────────────────────────────────────────────────────┘
```

---

## 3. Disk I/O Analysis

### 3.1 Data Sources

| Metric | Linux Source | Mac Source (SSH) | Calculation |
|--------|--------------|------------------|-------------|
| Read/Write Throughput (MB/s) | `/proc/diskstats` | `iostat -d` | Δsectors × 512 / Δt |
| IOPS | `/proc/diskstats` | `iostat -d` | Δios / Δt |
| Queue Depth | `/proc/diskstats` (field 11) | N/A | Direct read |
| Latency (estimated) | Derived | Derived | Little's Law |
| Per-mount breakdown | `/proc/mounts` + diskstats | `df` + iostat | Join on device |

### 3.2 Latency Estimation via Little's Law

Without eBPF, estimate I/O latency using **Little's Law** (Little, 1961):

$$L = \lambda W$$

Where:
- $L$ = average queue length (from `in_flight` field)
- $\lambda$ = arrival rate (IOPS)
- $W$ = average wait time (latency)

```rust
/// Little's Law latency estimation (Little, 1961)
pub fn estimate_latency_ms(queue_depth: f64, iops: f64) -> f64 {
    if iops < 1.0 { return 0.0; }
    (queue_depth / iops) * 1000.0  // Convert to ms
}

/// Percentile estimation using exponential distribution assumption
/// Valid for random I/O workloads (Chen et al., 1994)
pub fn estimate_p99_latency_ms(avg_latency_ms: f64) -> f64 {
    // P99 ≈ avg × ln(100) for exponential distribution
    avg_latency_ms * 4.605  // ln(100)
}
```

### 3.3 IOPS vs Throughput Classification

Distinguish workload types (Ruemmler & Wilkes, 1994):

| Workload | IOPS Pattern | Throughput Pattern | Typical Use |
|----------|--------------|-------------------|-------------|
| Sequential | Low (<100) | High (>500 MB/s) | Video, backups |
| Random | High (>1000) | Low (<100 MB/s) | Databases, VMs |
| Mixed | Medium | Medium | General use |

```rust
pub enum IoWorkloadType {
    Sequential,  // High throughput, low IOPS
    Random,      // Low throughput, high IOPS
    Mixed,       // Balanced
    Idle,        // Minimal activity
}

pub fn classify_workload(iops: f64, throughput_mbps: f64) -> IoWorkloadType {
    let ratio = if iops > 0.0 { throughput_mbps * 1024.0 / iops } else { 0.0 }; // KB per IO
    match ratio {
        r if r > 128.0 => IoWorkloadType::Sequential,  // >128KB per IO
        r if r < 16.0 && iops > 100.0 => IoWorkloadType::Random,  // <16KB per IO
        _ if iops < 10.0 && throughput_mbps < 1.0 => IoWorkloadType::Idle,
        _ => IoWorkloadType::Mixed,
    }
}
```

### 3.4 Visualization

```
┌─ Disk I/O ────────────────────────────────────────────────┐
│ nvme0n1 │ R: 251.0 MB/s │ W: 129.1 MB/s │ IOPS: 45K      │
│ Latency │ avg: 0.8ms │ p50: 0.5ms │ p99: 3.7ms           │
│ Queue: 12 │ Workload: Sequential │ Util: 67%             │
├───────────────────────────────────────────────────────────┤
│ Mount      │ Device    │ Read      │ Write     │ Usage   │
│ /          │ nvme0n1p2 │ 120 MB/s  │ 80 MB/s   │ 54%     │
│ /home      │ nvme1n1p1 │ 131 MB/s  │ 49 MB/s   │ 24%     │
└───────────────────────────────────────────────────────────┘
```

---

## 4. Disk Storage (GB) & Large File Anomaly Detection

### 4.1 Storage Monitoring

| Metric | Linux Source | Mac Source (SSH) | Update Interval |
|--------|--------------|------------------|-----------------|
| Total/Used/Free (GB) | `statvfs()` syscall | `df -g` | 5s |
| Inode Usage | `statvfs()` | `df -i` | 30s |
| File Count by Size | `find` + histogram | `find` + histogram | On-demand |
| Recent Large Files | `fanotify` / `inotify` | `fsevents` | Real-time |

### 4.2 Large File Anomaly Detection

Detect anomalous large file creation using **statistical outlier detection** (Grubbs, 1969):

```rust
/// Large file anomaly detection using Modified Z-Score (Iglewicz & Hoaglin, 1993)
pub struct LargeFileDetector {
    size_history: RingBuffer<u64, 1000>,  // Last 1000 file sizes
    median: u64,
    mad: u64,  // Median Absolute Deviation
}

impl LargeFileDetector {
    /// Modified Z-Score: more robust than standard Z-score for outliers
    pub fn is_anomaly(&self, file_size: u64) -> bool {
        if self.mad == 0 { return file_size > self.median * 10; }
        let modified_z = 0.6745 * (file_size as f64 - self.median as f64) / self.mad as f64;
        modified_z.abs() > 3.5  // Threshold per Iglewicz & Hoaglin
    }

    /// Real-time detection via fanotify (Linux) or FSEvents (Mac)
    pub fn on_file_created(&mut self, path: &Path, size: u64) -> Option<Anomaly> {
        self.size_history.push(size);
        self.update_statistics();

        if self.is_anomaly(size) {
            Some(Anomaly {
                path: path.to_owned(),
                size,
                z_score: self.calculate_z_score(size),
                timestamp: Instant::now(),
            })
        } else {
            None
        }
    }
}
```

### 4.3 fanotify Integration (Linux)

```rust
use nix::sys::fanotify::{Fanotify, FanotifyMask, FanotifyResponse};

pub struct FileSystemWatcher {
    fanotify: Fanotify,
    detector: LargeFileDetector,
    anomalies: VecDeque<Anomaly>,
}

impl FileSystemWatcher {
    pub fn new(watch_paths: &[&Path]) -> Result<Self> {
        let fanotify = Fanotify::init(
            FanotifyInitFlags::FAN_CLASS_NOTIF,
            OFlag::O_RDONLY,
        )?;

        for path in watch_paths {
            fanotify.mark(
                FanotifyMarkFlags::FAN_MARK_ADD | FanotifyMarkFlags::FAN_MARK_MOUNT,
                FanotifyMask::FAN_CREATE | FanotifyMask::FAN_CLOSE_WRITE,
                None,
                Some(path),
            )?;
        }

        Ok(Self { fanotify, detector: LargeFileDetector::new(), anomalies: VecDeque::new() })
    }
}
```

### 4.4 Visualization

```
┌─ Storage ─────────────────────────────────────────────────┐
│ Mount      │ Total    │ Used     │ Free     │ Use%       │
│ /          │ 500.0 GB │ 270.0 GB │ 230.0 GB │ ████░ 54%  │
│ /home      │ 2.0 TB   │ 480.0 GB │ 1.5 TB   │ ██░░░ 24%  │
│ /var       │ 100.0 GB │ 45.0 GB  │ 55.0 GB  │ ██░░░ 45%  │
├───────────────────────────────────────────────────────────┤
│ Large File Anomalies (last hour):                         │
│ ! /var/log/app.log grew 2.3 GB (z=4.2) 5m ago            │
│ ! /tmp/core.12345 created 8.1 GB (z=6.8) 12m ago         │
└───────────────────────────────────────────────────────────┘
```

---

## 5. Feature Enhancements

### 5.1 CPU Analytics

| Feature | Implementation | Source |
|---------|----------------|--------|
| Per-Core Temperature | Map hwmon sensors to cores | `/sys/class/hwmon` |
| Frequency Histogram | SIMD-accelerated binning | `/proc/cpuinfo` |
| Process Affinity | Visual indicator per core | `sched_getaffinity` |

### 5.2 Memory & Swap

| Feature | Implementation | Source |
|---------|----------------|--------|
| PSI Metrics | Real-time pressure display | `/proc/pressure/memory` |
| OOM Score | Per-process risk indicator | `/proc/[pid]/oom_score` |
| ZRAM Ratio | Compression efficiency | `trueno-zram` integration |

### 5.3 GPU & AI Accelerators

| Feature | Implementation | Source |
|---------|----------------|--------|
| Per-Process VRAM | Process memory breakdown | `nvmlDeviceGetComputeRunningProcesses` |
| Codec Utilization | NVDEC/NVENC bars | `nvmlDeviceGetDecoderUtilization` |
| PCIe Bandwidth | Bus throughput graph | `nvmlDeviceGetPcieThroughput` |

### 5.4 Network Telemetry

| Feature | Implementation | Source |
|---------|----------------|--------|
| Connection States | TCP/UDP socket counts | `/proc/net/tcp`, `/proc/net/udp` |
| Error/Drop Rate | Sparkline overlay | `/proc/net/dev` |
| Session Totals | 64-bit accumulators | Counter wrap handling |

### 5.5 Sensors

| Feature | Implementation | Source |
|---------|----------------|--------|
| Fan RPM | Speed with thresholds | `lm-sensors` / sysfs |
| Thermal Throttling | Alert on trip point | `/sys/class/thermal` |
| Session Min/Max/Avg | Ring buffer aggregation | In-memory tracking |

### 5.6 Process Management

| Feature | Implementation | Source |
|---------|----------------|--------|
| Tree View | Collapsible hierarchy | PPID map O(n) |
| Fuzzy Filter | Real-time search | `nucleo-matcher` |
| Inline Sparklines | Per-PID history | Circular buffer |

---

## 6. Bug Fixes & Stability

### 6.1 Network Session Overflow

**Symptom**: Session totals reset on long uptimes
**Root Cause**: u32 accumulators, incorrect wrap handling
**Fix**: u64 accumulators with wrap detection

```rust
pub fn handle_counter_wrap(prev: u64, curr: u64) -> u64 {
    if curr >= prev {
        curr - prev
    } else {
        // Counter wrapped
        u64::MAX - prev + curr + 1
    }
}
```

### 6.2 GPU VRAM Under-reporting

**Symptom**: VRAM shows 3% when higher
**Root Cause**: Missing Graphics/Display contexts
**Fix**: Use `nvmlDeviceGetMemoryInfo` for total

### 6.3 Sensor Label Truncation

**Symptom**: "PHY Temperat" cut off
**Root Cause**: Fixed-width columns
**Fix**: Dynamic column sizing with ellipsis

---

## 7. Architecture: SIMD-First with trueno

### 7.1 Backend Selection (from trueno)

```rust
// Adapted from /home/noah/src/trueno/src/lib.rs
pub fn select_backend() -> Backend {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
            return Backend::Avx512;  // 16x f32 parallelism
        }
        if is_x86_feature_detected!("avx2") {
            return Backend::Avx2;    // 8x f32 parallelism (default)
        }
        if is_x86_feature_detected!("sse2") {
            return Backend::Sse2;    // 4x f32 parallelism
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return Backend::Neon;        // 4x f32 parallelism
    }
    Backend::Scalar
}
```

### 7.2 SIMD-Accelerated Operations

| Operation | SIMD Benefit | Implementation |
|-----------|--------------|----------------|
| Histogram binning | 8x throughput | AVX2 comparison + mask |
| Ring buffer stats | 4-8x throughput | Vectorized sum/min/max |
| Sparkline rendering | 8x throughput | Parallel value mapping |
| String matching | 4x throughput | SIMD substring search |

### 7.3 trueno Integration

```toml
# Cargo.toml
[dependencies]
trueno = "0.11"  # SIMD-accelerated operations

[features]
simd = ["trueno/avx2"]
gpu = ["trueno/gpu"]
```

---

## 8. Cross-Platform: SSH Mac Support

### 8.1 Architecture

```
┌─────────────┐     SSH      ┌─────────────────┐
│ Linux Host  │ ──────────── │ Mac (Apple M*)  │
│ (ttop TUI)  │   Commands   │ sysctl/IOKit    │
└─────────────┘              └─────────────────┘
```

### 8.2 Mac Metric Collection (from lambda-lab-rust-development)

```rust
// Adapted from /home/noah/src/lambda-lab-rust-development/src/intel_mac.rs
pub struct MacCollector {
    ssh_session: SshSession,
    hostname: String,
}

impl MacCollector {
    pub async fn collect_cpu(&self) -> Result<CpuStats> {
        let output = self.ssh_exec("sysctl -n machdep.cpu.core_count hw.cpufrequency_max").await?;
        // Parse sysctl output
        Ok(CpuStats { cores, frequency, .. })
    }

    pub async fn collect_memory(&self) -> Result<MemoryStats> {
        let vm_stat = self.ssh_exec("vm_stat").await?;
        let page_size: u64 = 16384;  // Apple Silicon page size
        // Parse vm_stat: Pages free, active, inactive, speculative, wired
        Ok(MemoryStats { total, used, free, .. })
    }

    pub async fn collect_gpu(&self) -> Result<GpuStats> {
        // Apple Silicon: Metal performance via powermetrics (requires sudo)
        let output = self.ssh_exec("sudo powermetrics -n 1 -i 1000 --samplers gpu_power").await?;
        Ok(GpuStats { utilization, power, .. })
    }

    pub async fn collect_disk(&self) -> Result<DiskStats> {
        let iostat = self.ssh_exec("iostat -d -c 1").await?;
        let df = self.ssh_exec("df -g").await?;
        Ok(DiskStats { read_mbps, write_mbps, .. })
    }
}
```

### 8.3 Platform Abstraction

```rust
pub trait PlatformCollector: Send + Sync {
    async fn collect_cpu(&self) -> Result<CpuMetrics>;
    async fn collect_memory(&self) -> Result<MemoryMetrics>;
    async fn collect_disk(&self) -> Result<DiskMetrics>;
    async fn collect_gpu(&self) -> Result<Option<GpuMetrics>>;
    async fn collect_network(&self) -> Result<NetworkMetrics>;
    async fn collect_sensors(&self) -> Result<SensorMetrics>;
}

pub struct LinuxCollector { /* /proc, /sys access */ }
pub struct MacSshCollector { /* SSH + sysctl/IOKit */ }
pub struct HybridCollector {
    linux: LinuxCollector,
    mac: Option<MacSshCollector>,
}
```

---

## 9. Testing: probador Integration

### 9.1 GUI Coverage (from probador)

```rust
// Adapted from /home/noah/src/probar/crates/probar/src/gui_coverage.rs
gui_coverage! {
    buttons: [
        "sort_cpu", "sort_mem", "sort_pid",
        "kill_process", "nice_process",
        "toggle_tree", "toggle_filter",
    ],
    screens: [
        "overview", "cpu_detail", "memory_detail",
        "disk_detail", "gpu_detail", "process_list",
        "sensors", "network",
    ],
    interactions: [
        "scroll_up", "scroll_down", "page_up", "page_down",
        "select_process", "expand_tree", "collapse_tree",
    ],
}

#[test]
fn test_gui_coverage_minimum() {
    let coverage = run_gui_tests();
    assert!(coverage.meets(95.0), "GUI coverage must be ≥95%");
}
```

### 9.2 TUI Testing

```rust
use jugar_probar::tui::{Terminal, Frame, assert_frame};

#[test]
fn test_cpu_panel_renders() {
    let mut terminal = Terminal::new(80, 24);
    let metrics = mock_cpu_metrics();

    render_cpu_panel(&mut terminal, &metrics);

    assert_frame!(terminal, contains: "CPU");
    assert_frame!(terminal, contains: "cores");
    assert_frame!(terminal, row: 0, col: 0..10, equals: "CPU 7%");
}
```

### 9.3 Property-Based Testing

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn swap_thrashing_detection_is_monotonic(
        psi in 0.0..100.0f64,
        swap_rate in 0u64..10000,
        fault_rate in 0u64..1000,
    ) {
        let severity1 = detect_thrashing(psi, swap_rate, fault_rate);
        let severity2 = detect_thrashing(psi + 10.0, swap_rate + 100, fault_rate + 10);

        // Higher pressure should never decrease severity
        prop_assert!(severity2 >= severity1);
    }

    #[test]
    fn latency_estimation_is_positive(
        queue_depth in 0.0..1000.0f64,
        iops in 1.0..100000.0f64,
    ) {
        let latency = estimate_latency_ms(queue_depth, iops);
        prop_assert!(latency >= 0.0);
        prop_assert!(latency.is_finite());
    }
}
```

---

## 10. 100-Point Popperian Falsification Checklist

### 10.0 Falsification Response Protocol: The Five Whys

If any falsifiable claim (C01-R10) is disproven during testing, the **Five Whys** root cause analysis (Ohno, 1988) MUST be performed and documented.

**Protocol:**
1.  **State the Failure**: Explicitly state which claim failed and the observed value.
2.  **Ask Why (1)**: Why did the metric deviate? (e.g., "The CPU calculation was off by 5%").
3.  **Ask Why (2)**: Why was the calculation incorrect? (e.g., "We used integer division instead of float").
4.  **Ask Why (3)**: Why was integer division used? (e.g., "The variable type was u64").
5.  **Ask Why (4)**: Why was the type u64? (e.g., "The kernel returns raw jiffies as u64").
6.  **Ask Why (5)**: Why did we not cast to f64 before division? (e.g., "The conversion logic was missing in the collector trait").
7.  **Corrective Action**: Implement the fix (e.g., "Cast to f64 in collector") and add a regression test.

**Example Artifact:**
> **Failure**: C01 (CPU %) - Observed 5.0%, Expected 10.0%
> **Root Cause**: Jiffies delta calculation overflowed on 32-bit counter wrap.
> **Fix**: Implemented `handle_counter_wrap` with u64 upcasting.

### 10.1 CPU Metrics (C01-C10)

| ID | Falsifiable Claim | Test Method | Falsification |
|----|-------------------|-------------|---------------|
| C01 | CPU % matches `/proc/stat` within 0.1% | Compare 1000 samples | Any sample differs >0.1% |
| C02 | Core count equals `nproc` output | Direct comparison | Values differ |
| C03 | Frequency matches `cpufreq` within 1MHz | Cross-reference | Deviation >1MHz |
| C04 | Temperature matches `sensors` within 1°C | Cross-reference | Deviation >1°C |
| C05 | Load average matches `uptime` exactly | String comparison | Any digit differs |
| C06 | Per-core usage sums to total (±1%) | Arithmetic check | Sum differs >1% |
| C07 | Frequency histogram bins are correct | Manual verification | Any bin miscounted |
| C08 | Process affinity display is accurate | Compare `taskset -p` | Mismatch found |
| C09 | CPU panel renders in <1ms | Benchmark | Render time >1ms |
| C10 | Hot-plug detection within 100ms | Add/remove CPU | Detection >100ms |

### 10.2 Memory Metrics (M01-M15)

| ID | Falsifiable Claim | Test Method | Falsification |
|----|-------------------|-------------|---------------|
| M01 | Total RAM matches `free -b` exactly | Byte comparison | Any difference |
| M02 | Used RAM matches `MemTotal - MemAvailable` | Calculation | Values differ |
| M03 | Cached RAM matches `/proc/meminfo Cached` | Direct read | Values differ |
| M04 | Buffers matches `/proc/meminfo Buffers` | Direct read | Values differ |
| M05 | Swap total matches `swapon -s` | Cross-reference | Values differ |
| M06 | Swap used matches `/proc/meminfo SwapFree` delta | Calculation | Values differ |
| M07 | PSI some% matches `/proc/pressure/memory` | Direct read | Deviation >0.1% |
| M08 | PSI full% matches `/proc/pressure/memory` | Direct read | Deviation >0.1% |
| M09 | ZRAM ratio matches `mm_stat` calculation | Manual verify | Ratio differs |
| M10 | ZRAM algorithm matches `/sys/block/zram*/comp_algorithm` | String compare | Mismatch |
| M11 | Page fault rate matches `vmstat` delta | Compare rates | Deviation >1/s |
| M12 | Thrashing detection triggers at PSI >50% | Inject pressure | No trigger |
| M13 | OOM score matches `/proc/[pid]/oom_score` | Per-process check | Any mismatch |
| M14 | Memory panel renders in <2ms | Benchmark | Render time >2ms |
| M15 | Memory leak: RSS stable over 24h run | Long-running test | RSS grows >1MB |

### 10.3 Disk I/O Metrics (D01-D15)

| ID | Falsifiable Claim | Test Method | Falsification |
|----|-------------------|-------------|---------------|
| D01 | Read MB/s matches `iostat` within 1% | Cross-reference | Deviation >1% |
| D02 | Write MB/s matches `iostat` within 1% | Cross-reference | Deviation >1% |
| D03 | IOPS matches `iostat` within 1% | Cross-reference | Deviation >1% |
| D04 | Queue depth matches diskstats field 11 | Direct read | Values differ |
| D05 | Latency estimate within 2x of `ioping` | Benchmark | Ratio >2x |
| D06 | P99 latency within 3x of measured | Statistical test | Ratio >3x |
| D07 | Utilization % matches `iostat %util` | Cross-reference | Deviation >2% |
| D08 | Per-mount breakdown sums to device total | Arithmetic | Sum differs >1% |
| D09 | Workload classification is correct | Manual verify | Misclassified |
| D10 | Device hotplug detected in <1s | Add/remove device | Detection >1s |
| D11 | NVMe vs SATA correctly identified | Check device type | Misidentified |
| D12 | RAID arrays show aggregate stats | mdadm comparison | Values differ |
| D13 | LVM volumes correctly attributed | lvs comparison | Misattributed |
| D14 | Disk panel renders in <2ms | Benchmark | Render time >2ms |
| D15 | Counter wrap handled at 2^64 boundary | Unit test | Overflow/panic |

### 10.4 Storage Metrics (S01-S10)

| ID | Falsifiable Claim | Test Method | Falsification |
|----|-------------------|-------------|---------------|
| S01 | Total GB matches `df` exactly | Cross-reference | Values differ |
| S02 | Used GB matches `df` exactly | Cross-reference | Values differ |
| S03 | Free GB matches `df` exactly | Cross-reference | Values differ |
| S04 | Percentage matches `df` within 0.1% | Cross-reference | Deviation >0.1% |
| S05 | Inode count matches `df -i` | Cross-reference | Values differ |
| S06 | Mount points all listed | Compare `mount` | Any missing |
| S07 | Large file detection triggers at z>3.5 | Statistical test | False negative |
| S08 | Anomaly timestamp accurate to 1s | Clock comparison | Deviation >1s |
| S09 | fanotify events received in <100ms | Latency test | Latency >100ms |
| S10 | Storage panel renders in <1ms | Benchmark | Render time >1ms |

### 10.5 Network Metrics (N01-N10)

| ID | Falsifiable Claim | Test Method | Falsification |
|----|-------------------|-------------|---------------|
| N01 | RX bytes matches `ip -s link` | Cross-reference | Values differ |
| N02 | TX bytes matches `ip -s link` | Cross-reference | Values differ |
| N03 | RX rate matches `iftop` within 5% | Cross-reference | Deviation >5% |
| N04 | TX rate matches `iftop` within 5% | Cross-reference | Deviation >5% |
| N05 | TCP connection count matches `ss -t` | Cross-reference | Count differs |
| N06 | UDP socket count matches `ss -u` | Cross-reference | Count differs |
| N07 | Error count matches `/proc/net/dev` | Direct read | Values differ |
| N08 | Drop count matches `/proc/net/dev` | Direct read | Values differ |
| N09 | Session total survives 10TB transfer | Long test | Overflow/reset |
| N10 | Network panel renders in <1ms | Benchmark | Render time >1ms |

### 10.6 GPU Metrics (G01-G15)

| ID | Falsifiable Claim | Test Method | Falsification |
|----|-------------------|-------------|---------------|
| G01 | GPU % matches `nvidia-smi` within 1% | Cross-reference | Deviation >1% |
| G02 | VRAM total matches `nvidia-smi` exactly | Cross-reference | Values differ |
| G03 | VRAM used matches `nvidia-smi` within 1MB | Cross-reference | Deviation >1MB |
| G04 | Temperature matches `nvidia-smi` within 1°C | Cross-reference | Deviation >1°C |
| G05 | Power draw matches `nvidia-smi` within 1W | Cross-reference | Deviation >1W |
| G06 | Clock speed matches `nvidia-smi` | Cross-reference | Values differ |
| G07 | PCIe bandwidth matches NVML | API comparison | Values differ |
| G08 | Encoder utilization matches NVML | API comparison | Values differ |
| G09 | Decoder utilization matches NVML | API comparison | Values differ |
| G10 | Per-process VRAM sums to total (±5%) | Arithmetic | Sum differs >5% |
| G11 | Multi-GPU indexing correct | Manual verify | Wrong GPU shown |
| G12 | GPU hotplug detected in <1s | Add/remove GPU | Detection >1s |
| G13 | Apple Silicon GPU (SSH) reports utilization | Mac test | No data |
| G14 | AMD GPU (ROCm) metrics accurate | Cross-reference | Values differ |
| G15 | GPU panel renders in <2ms | Benchmark | Render time >2ms |

### 10.7 Sensor Metrics (T01-T10)

| ID | Falsifiable Claim | Test Method | Falsification |
|----|-------------------|-------------|---------------|
| T01 | All sensors from `sensors` appear | Cross-reference | Any missing |
| T02 | Temperature values match within 1°C | Cross-reference | Deviation >1°C |
| T03 | Fan RPM matches within 10 RPM | Cross-reference | Deviation >10 RPM |
| T04 | Throttling alert at trip point | Inject heat | No alert |
| T05 | Session min is correct | Manual verify | Wrong value |
| T06 | Session max is correct | Manual verify | Wrong value |
| T07 | Session avg is mathematically correct | Calculate | Wrong value |
| T08 | Sensor labels not truncated | Visual check | Any truncation |
| T09 | Sensor hotplug detected | Add USB sensor | Not detected |
| T10 | Sensor panel renders in <1ms | Benchmark | Render time >1ms |

### 10.8 Process Metrics (P01-P10)

| ID | Falsifiable Claim | Test Method | Falsification |
|----|-------------------|-------------|---------------|
| P01 | Process count matches `ps aux | wc -l` | Cross-reference | Count differs |
| P02 | PID values are correct | Spot check | Any wrong PID |
| P03 | CPU % per process matches `top` within 1% | Cross-reference | Deviation >1% |
| P04 | MEM % per process matches `top` within 1% | Cross-reference | Deviation >1% |
| P05 | Tree view parent-child correct | Manual verify | Wrong hierarchy |
| P06 | Filter finds all matching processes | Grep comparison | Any missing |
| P07 | Sort by CPU is stable and correct | Manual verify | Wrong order |
| P08 | Sort by MEM is stable and correct | Manual verify | Wrong order |
| P09 | Sparklines show correct history | Replay test | Wrong values |
| P10 | Process panel renders 10K procs in <50ms | Benchmark | Render time >50ms |

### 10.9 Cross-Platform (X01-X10)

| ID | Falsifiable Claim | Test Method | Falsification |
|----|-------------------|-------------|---------------|
| X01 | SSH Mac: CPU cores correct | Compare `sysctl` | Values differ |
| X02 | SSH Mac: Memory total correct | Compare `sysctl` | Values differ |
| X03 | SSH Mac: Disk space correct | Compare `df` | Values differ |
| X04 | SSH Mac: Network stats correct | Compare `netstat` | Values differ |
| X05 | SSH timeout handled gracefully | Kill SSH | App crashes |
| X06 | SSH reconnect automatic | Restore network | No reconnect |
| X07 | Linux/Mac metrics unified format | API comparison | Format differs |
| X08 | M1/M2/M3 all supported | Test each | Any fails |
| X09 | Intel Mac supported | Test Intel Mac | Fails |
| X10 | SSH latency <100ms for metrics | Measure RTT | Latency >100ms |

### 10.10 Performance & Stability (R01-R10)

| ID | Falsifiable Claim | Test Method | Falsification |
|----|-------------------|-------------|---------------|
| R01 | Full render <16ms (60 FPS capable) | Benchmark | Render >16ms |
| R02 | Memory usage <50MB baseline | Measure RSS | RSS >50MB |
| R03 | CPU usage <2% when idle | Measure | Usage >2% |
| R04 | No memory leaks over 24h | Valgrind/heaptrack | Leaks detected |
| R05 | No file descriptor leaks | `lsof` count | FD count grows |
| R06 | Graceful degradation on permission error | Test non-root | Crash/hang |
| R07 | SIGTERM handled cleanly | Send signal | Unclean exit |
| R08 | SIGHUP triggers config reload | Send signal | No reload |
| R09 | Panic-free: no unwrap on Option/Result | Code audit | Any unwrap |
| R10 | 100% test coverage on core modules | `cargo llvm-cov` | Coverage <100% |

---

## 11. Peer-Reviewed References

### Memory & Virtual Memory

1. **Denning, P.J.** (1968). "The Working Set Model for Program Behavior." *Communications of the ACM*, 11(5), 323-333. DOI: 10.1145/363095.363141
   - Foundation for swap thrashing detection

2. **Denning, P.J.** (1980). "Working Sets Past and Present." *IEEE Transactions on Software Engineering*, SE-6(1), 64-84. DOI: 10.1109/TSE.1980.230464
   - Working set theory refinements

3. **Carr, R.W., & Hennessy, J.L.** (1981). "WSCLOCK—A Simple and Effective Algorithm for Virtual Memory Management." *ACM SIGOPS Operating Systems Review*, 15(5), 87-95. DOI: 10.1145/1067627.806593
   - Page replacement algorithms

### Disk I/O & Performance

4. **Little, J.D.C.** (1961). "A Proof for the Queuing Formula: L = λW." *Operations Research*, 9(3), 383-387. DOI: 10.1287/opre.9.3.383
   - Foundation for latency estimation

5. **Ruemmler, C., & Wilkes, J.** (1994). "An Introduction to Disk Drive Modeling." *IEEE Computer*, 27(3), 17-28. DOI: 10.1109/2.268881
   - Disk workload characterization

6. **Chen, P.M., Lee, E.K., Gibson, G.A., Katz, R.H., & Patterson, D.A.** (1994). "RAID: High-Performance, Reliable Secondary Storage." *ACM Computing Surveys*, 26(2), 145-185. DOI: 10.1145/176979.176981
   - Storage system fundamentals

7. **Ousterhout, J.K., Da Costa, H., Harrison, D., Kunze, J.A., Kupfer, M., & Thompson, J.G.** (1985). "A Trace-Driven Analysis of the UNIX 4.2 BSD File System." *ACM SIGOPS Operating Systems Review*, 19(5), 15-24. DOI: 10.1145/323647.323631
   - File system workload analysis

### Statistical Methods

8. **Grubbs, F.E.** (1969). "Procedures for Detecting Outlying Observations in Samples." *Technometrics*, 11(1), 1-21. DOI: 10.1080/00401706.1969.10490657
   - Outlier detection methodology

9. **Iglewicz, B., & Hoaglin, D.C.** (1993). "How to Detect and Handle Outliers." *ASQC Quality Press*. ISBN: 978-0873892476
   - Modified Z-score for robust outlier detection

10. **Rosner, B.** (1983). "Percentage Points for a Generalized ESD Many-Outlier Procedure." *Technometrics*, 25(2), 165-172. DOI: 10.1080/00401706.1983.10487848
    - Multiple outlier detection

### Systems Monitoring

11. **Gregg, B.** (2020). "Systems Performance: Enterprise and the Cloud." 2nd Edition. *Addison-Wesley*. ISBN: 978-0136820154
    - Comprehensive systems monitoring methodology

12. **Gregg, B.** (2016). "The Flame Graph." *Communications of the ACM*, 59(6), 48-57. DOI: 10.1145/2909476
    - Performance visualization

13. **Zadok, E., & Nieh, J.** (2000). "FiST: A Language for Stackable File Systems." *USENIX Annual Technical Conference*, 55-70.
    - File system monitoring architecture

### GPU & Heterogeneous Computing

14. **Nickolls, J., Buck, I., Garland, M., & Skadron, K.** (2008). "Scalable Parallel Programming with CUDA." *ACM Queue*, 6(2), 40-53. DOI: 10.1145/1365490.1365500
    - GPU monitoring fundamentals

15. **Jia, Z., Maggioni, M., Staber, B., & Scarpazza, D.P.** (2018). "Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking." *arXiv:1804.06826*.
    - GPU performance characterization

### Testing & Verification

16. **Jia, Y., & Harman, M.** (2011). "An Analysis and Survey of the Development of Mutation Testing." *IEEE Transactions on Software Engineering*, 37(5), 649-678. DOI: 10.1109/TSE.2010.62
    - Mutation testing methodology

17. **Popper, K.** (1959). "The Logic of Scientific Discovery." *Routledge*. ISBN: 978-0415278447
    - Falsificationism philosophy

18. **Claessen, K., & Hughes, J.** (2000). "QuickCheck: A Lightweight Tool for Random Testing of Haskell Programs." *ACM SIGPLAN Notices*, 35(9), 268-279. DOI: 10.1145/357766.351266
    - Property-based testing

### SIMD & Vectorization

19. **Lemire, D., & Boytsov, L.** (2015). "Decoding Billions of Integers Per Second Through Vectorization." *Software: Practice and Experience*, 45(1), 1-29. DOI: 10.1002/spe.2203
    - SIMD optimization techniques

20. **Fog, A.** (2021). "Optimizing Software in C++." *Technical University of Denmark*. Available: https://www.agner.org/optimize/
    - Low-level optimization reference

### Visualization

21. **Shneiderman, B.** (1996). "The Eyes Have It: A Task by Data Type Taxonomy for Information Visualizations." *IEEE Symposium on Visual Languages*, 336-343. DOI: 10.1109/VL.1996.545307
    - Overview first, zoom and filter, details on demand

22. **Tufte, E.R.** (2001). "The Visual Display of Quantitative Information." 2nd Edition. *Graphics Press*. ISBN: 978-1930824133
    - Data visualization principles

### Compression & ZRAM

23. **Collet, Y.** (2016). "LZ4: Extremely Fast Compression Algorithm." GitHub. Available: https://github.com/lz4/lz4
    - LZ4 compression algorithm

24. **Collet, Y., & Kucherawy, M.** (2021). "Zstandard Compression and the 'application/zstd' Media Type." *RFC 8878*. DOI: 10.17487/RFC8878
    - Zstandard compression standard

25. **Jennings, N.** (2013). "zram: Compressed RAM-based Block Devices." *Linux Kernel Documentation*.
    - ZRAM implementation reference

### General Management

26. **Ohno, T.** (1988). *Toyota Production System: Beyond Large-Scale Production*. Productivity Press. ISBN: 978-0915299140.
    - Source of the "Five Whys" root cause analysis technique.

---

## Appendix A: Integration with Sovereign AI Stack

### A.1 trueno (SIMD Operations)

```toml
[dependencies.trueno]
version = "0.11"
features = ["avx2", "monitor"]
```

### A.2 trueno-zram (Compression Monitoring)

```toml
[dependencies.trueno-zram-core]
version = "0.1"
features = ["stats"]
```

### A.3 probador (Testing)

```toml
[dev-dependencies.jugar-probar]
version = "0.5"
features = ["tui", "coverage"]
```

### A.4 lambda-lab Patterns

Memory pressure monitoring adapted from `/home/noah/src/lambda-lab-rust-development/src/memory_pressure.rs`.

---

*Document generated following Popperian falsificationism: every claim herein is designed to be disprovable through empirical testing.*