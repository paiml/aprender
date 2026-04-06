//! TUI render state types for compute monitoring display.

use std::collections::VecDeque;

use crate::monitor::device::DeviceId;
use crate::monitor::memory::PressureLevel;

// ============================================================================
// TUI Render State
// ============================================================================

/// Complete TUI render state
#[derive(Debug, Clone)]
pub struct TuiRenderState {
    /// CPU device metrics
    pub cpu: Option<DeviceRenderState>,
    /// GPU device metrics
    pub gpus: Vec<DeviceRenderState>,
    /// Memory metrics
    pub memory: MemoryRenderState,
    /// Data flow metrics
    pub data_flow: DataFlowRenderState,
    /// Active kernels
    pub kernels: Vec<KernelRenderState>,
    /// Current pressure level
    pub pressure: PressureLevel,
    /// Stress test active
    pub stress_active: bool,
    /// Paused
    pub paused: bool,
    /// Current focus
    pub focused_section: usize,
    /// Error message (if any)
    pub error: Option<String>,
}

impl Default for TuiRenderState {
    fn default() -> Self {
        Self {
            cpu: None,
            gpus: Vec::new(),
            memory: MemoryRenderState::default(),
            data_flow: DataFlowRenderState::default(),
            kernels: Vec::new(),
            pressure: PressureLevel::Ok,
            stress_active: false,
            paused: false,
            focused_section: 0,
            error: None,
        }
    }
}

/// Device render state
#[derive(Debug, Clone)]
pub struct DeviceRenderState {
    /// Device ID
    pub device_id: DeviceId,
    /// Device name
    pub name: String,
    /// Utilization percentage
    pub utilization_pct: f64,
    /// Temperature in Celsius
    pub temperature_c: f64,
    /// Power in Watts
    pub power_watts: f64,
    /// Power limit in Watts
    pub power_limit_watts: f64,
    /// Clock speed in MHz
    pub clock_mhz: u32,
    /// Utilization history
    pub history: VecDeque<f64>,
}

/// Memory render state
#[derive(Debug, Clone, Default)]
pub struct MemoryRenderState {
    /// RAM usage percentage
    pub ram_pct: f64,
    /// RAM used in GB
    pub ram_used_gb: f64,
    /// RAM total in GB
    pub ram_total_gb: f64,
    /// Swap usage percentage
    pub swap_pct: f64,
    /// Swap used in GB
    pub swap_used_gb: f64,
    /// Swap total in GB
    pub swap_total_gb: f64,
    /// VRAM metrics per GPU
    pub vram: Vec<(DeviceId, f64, f64, f64)>, // (id, pct, used_gb, total_gb)
    /// RAM history
    pub ram_history: VecDeque<f64>,
}

/// Data flow render state
#[derive(Debug, Clone, Default)]
pub struct DataFlowRenderState {
    /// PCIe TX in GB/s
    pub pcie_tx_gbps: f64,
    /// PCIe RX in GB/s
    pub pcie_rx_gbps: f64,
    /// PCIe theoretical in GB/s
    pub pcie_theoretical_gbps: f64,
    /// Memory bus utilization percentage
    pub memory_bus_pct: f64,
    /// Active transfers
    pub transfers: Vec<(String, String, f64)>, // (label, direction, progress)
}

/// Kernel render state
#[derive(Debug, Clone)]
pub struct KernelRenderState {
    /// Kernel name
    pub name: String,
    /// Device ID
    pub device_id: DeviceId,
    /// Progress percentage
    pub progress_pct: f64,
    /// Grid dimensions
    pub grid: String,
    /// Elapsed time in ms
    pub elapsed_ms: f64,
}
