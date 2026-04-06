//! Type definitions for trueno_monitor.
//!
//! Contains all struct/enum types used across the monitor binary,
//! extracted from main.rs for file health (TRUENO-SPEC-020).

use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use trueno_gpu::monitor::{CudaDeviceInfo, MemoryMetrics, StressTestConfig};

#[cfg(feature = "cuda")]
use trueno_gpu::driver::CudaContext;

use trueno_gpu::monitor::CpuDevice;

/// Stress test worker handle
pub(crate) struct StressWorker {
    pub(crate) running: Arc<AtomicBool>,
    pub(crate) ops_count: Arc<AtomicU64>,
    pub(crate) thread: Option<thread::JoinHandle<()>>,
}

/// Stress test verdict (TRUENO-SPEC-025)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StressTestVerdict {
    /// All metrics within acceptable range
    Pass,
    /// Minor throttling, acceptable
    PassWithNotes,
    /// Errors or severe throttling
    Fail,
}

impl std::fmt::Display for StressTestVerdict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pass => write!(f, "PASS"),
            Self::PassWithNotes => write!(f, "PASS (with notes)"),
            Self::Fail => write!(f, "FAIL"),
        }
    }
}

/// Stress test report (TRUENO-SPEC-025 Section 7.4)
#[derive(Debug, Clone)]
pub(crate) struct StressTestReport {
    /// Test duration
    pub(crate) duration_secs: u64,
    /// Peak CPU ops/sec
    pub(crate) peak_cpu_ops: u64,
    /// Peak memory ops/sec
    pub(crate) peak_mem_ops: u64,
    /// Peak GPU ops/sec
    pub(crate) peak_gpu_ops: u64,
    /// Peak CPU utilization
    pub(crate) peak_cpu_util: f64,
    /// Peak RAM utilization
    pub(crate) peak_ram_util: f64,
    /// Peak GPU VRAM utilization (max across all GPUs)
    pub(crate) peak_vram_util: f64,
    /// Number of CPU workers
    pub(crate) cpu_workers: usize,
    /// Number of GPU workers
    pub(crate) gpu_workers: usize,
    /// Test verdict
    pub(crate) verdict: StressTestVerdict,
    /// Recommendations
    pub(crate) recommendations: Vec<String>,
}

/// GPU state from real hardware
pub(crate) struct GpuState {
    pub(crate) info: CudaDeviceInfo,
    #[cfg(feature = "cuda")]
    pub(crate) ctx: CudaContext,
    pub(crate) vram_used_gb: f64,
    pub(crate) vram_total_gb: f64,
    pub(crate) vram_percent: f64,
}

/// Application state
pub(crate) struct App {
    /// CPU device monitor
    pub(crate) cpu: CpuDevice,
    /// Memory metrics
    pub(crate) memory: MemoryMetrics,
    /// CPU usage history (60 points for sparkline)
    pub(crate) cpu_history: Vec<u64>,
    /// Memory usage history
    pub(crate) mem_history: Vec<u64>,
    /// Currently selected tab
    pub(crate) selected_tab: usize,
    /// Is stress test running
    pub(crate) stress_running: bool,
    /// Stress test config
    pub(crate) stress_config: Option<StressTestConfig>,
    /// Show help overlay
    pub(crate) show_help: bool,
    /// Tick count for animations
    pub(crate) tick: u64,
    /// Real GPU states (from CUDA hardware)
    pub(crate) gpus: Vec<GpuState>,
    /// GPU VRAM history per device (sparkline data)
    pub(crate) gpu_vram_history: Vec<Vec<u64>>,
    /// CPU stress workers (one per core)
    pub(crate) cpu_workers: Vec<StressWorker>,
    /// Memory stress worker
    pub(crate) mem_worker: Option<StressWorker>,
    /// GPU stress workers (one per GPU)
    pub(crate) gpu_workers: Vec<StressWorker>,
    /// Total CPU ops/sec
    pub(crate) cpu_ops_per_sec: u64,
    /// CPU ops history for sparkline
    pub(crate) cpu_ops_history: Vec<u64>,
    /// Memory ops/sec
    pub(crate) mem_ops_per_sec: u64,
    /// Memory ops history
    pub(crate) mem_ops_history: Vec<u64>,
    /// GPU ops/sec (FLOPS)
    pub(crate) gpu_ops_per_sec: u64,
    /// GPU ops history
    pub(crate) gpu_ops_history: Vec<u64>,
    /// Stress test start time
    pub(crate) stress_start: Option<Instant>,
    /// Peak CPU ops/sec
    pub(crate) peak_cpu_ops: u64,
    /// Peak memory ops/sec
    pub(crate) peak_mem_ops: u64,
    /// Peak GPU ops/sec
    pub(crate) peak_gpu_ops: u64,
    /// Peak CPU utilization during stress
    pub(crate) peak_cpu_util: f64,
    /// Peak RAM utilization during stress
    pub(crate) peak_ram_util: f64,
    /// Peak VRAM utilization during stress
    pub(crate) peak_vram_util: f64,
    /// Last stress test report (via renacer)
    pub(crate) stress_report: Option<StressTestReport>,
}
