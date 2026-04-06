#![allow(missing_docs)]
//! Unified Compute Device Abstraction (TRUENO-SPEC-020)

/// Device identification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeviceId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceType {
    Cpu,
    NvidiaGpu,
    AmdGpu,
    IntelGpu,
    AppleSilicon,
    Hpu, // Hardware Processing Unit (e.g., Gaudi, TPU)
}

/// Unified compute device abstraction (TRUENO-SPEC-020)
pub trait ComputeDevice: Send + Sync {
    /// Device identification
    fn device_id(&self) -> DeviceId;
    fn device_name(&self) -> &str;
    fn device_type(&self) -> DeviceType;

    /// Compute metrics
    fn compute_utilization(&self) -> anyhow::Result<f64>; // 0.0-100.0%
    fn compute_clock_mhz(&self) -> anyhow::Result<u32>;
    fn compute_temperature_c(&self) -> anyhow::Result<f64>;
    fn compute_power_watts(&self) -> anyhow::Result<f64>;
    fn compute_power_limit_watts(&self) -> anyhow::Result<f64>;

    /// Memory metrics
    fn memory_used_bytes(&self) -> anyhow::Result<u64>;
    fn memory_total_bytes(&self) -> anyhow::Result<u64>;
    fn memory_bandwidth_gbps(&self) -> anyhow::Result<f64>;

    /// Streaming multiprocessor / Compute Unit metrics
    fn sm_count(&self) -> u32;
    fn active_sm_count(&self) -> anyhow::Result<u32>;

    /// PCIe / Interconnect metrics
    fn pcie_tx_bytes_per_sec(&self) -> anyhow::Result<u64>;
    fn pcie_rx_bytes_per_sec(&self) -> anyhow::Result<u64>;
    fn pcie_generation(&self) -> u8;
    fn pcie_width(&self) -> u8;
}
