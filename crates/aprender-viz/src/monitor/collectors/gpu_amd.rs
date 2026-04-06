//! AMD GPU metrics collector via ROCm SMI.
//!
//! Dynamically loads `librocm_smi64.so` to collect GPU metrics from AMD GPUs.
//!
//! ## Feature Flag
//!
//! Requires `monitor-amd` feature to be enabled.
//!
//! ## Metrics Collected
//!
//! - GPU utilization percentage
//! - Memory utilization percentage
//! - Memory used/total (VRAM)
//! - GPU temperature
//! - Power draw (watts)
//! - GPU clock speed (MHz)
//! - Memory clock speed (MHz)
//! - PCIe throughput

use crate::monitor::error::{MonitorError, Result};
use crate::monitor::ring_buffer::RingBuffer;
use crate::monitor::types::{Collector, MetricValue, Metrics};
use std::ffi::{c_void, CStr};
use std::os::raw::c_char;
use std::time::Duration;

/// ROCm SMI status codes
#[allow(dead_code)]
mod rsmi_status {
    pub(super) const RSMI_STATUS_SUCCESS: i32 = 0;
}

/// ROCm SMI memory types
#[allow(dead_code)]
mod rsmi_memory_type {
    pub(super) const RSMI_MEM_TYPE_VRAM: u32 = 0;
    pub(super) const RSMI_MEM_TYPE_VIS_VRAM: u32 = 1;
    pub(super) const RSMI_MEM_TYPE_GTT: u32 = 2;
}

/// ROCm SMI clock types
#[allow(dead_code)]
mod rsmi_clk_type {
    pub(super) const RSMI_CLK_TYPE_SYS: u32 = 0; // GPU clock
    pub(super) const RSMI_CLK_TYPE_MEM: u32 = 4; // Memory clock
}

/// ROCm SMI temperature types
#[allow(dead_code)]
mod rsmi_temp {
    pub(super) const RSMI_TEMP_TYPE_EDGE: u32 = 0;
    pub(super) const RSMI_TEMP_CURRENT: u32 = 0;
    pub(super) const RSMI_TEMP_MAX: u32 = 1;
}

/// Information about a single AMD GPU.
#[derive(Debug, Clone)]
pub struct AmdGpuInfo {
    /// GPU index.
    pub index: u32,
    /// GPU name/model.
    pub name: String,
    /// GPU utilization percentage (0-100).
    pub gpu_util: f64,
    /// Memory utilization percentage (0-100).
    pub mem_util: f64,
    /// VRAM used in bytes.
    pub vram_used: u64,
    /// VRAM total in bytes.
    pub vram_total: u64,
    /// GPU temperature in Celsius.
    pub temperature: f64,
    /// Max temperature threshold in Celsius.
    pub temp_max: f64,
    /// Power draw in watts.
    pub power_watts: f64,
    /// Power cap in watts.
    pub power_cap_watts: f64,
    /// GPU clock speed in MHz.
    pub gpu_clock_mhz: u64,
    /// Memory clock speed in MHz.
    pub mem_clock_mhz: u64,
    /// PCIe TX throughput in KB/s.
    pub pcie_tx_kbps: u64,
    /// PCIe RX throughput in KB/s.
    pub pcie_rx_kbps: u64,
}

/// ROCm SMI library wrapper.
struct RocmSmi {
    #[allow(dead_code)]
    handle: *mut c_void,
    // Function pointers
    rsmi_init: unsafe extern "C" fn(u64) -> i32,
    rsmi_shut_down: unsafe extern "C" fn() -> i32,
    rsmi_num_monitor_devices: unsafe extern "C" fn(*mut u32) -> i32,
    rsmi_dev_name_get: unsafe extern "C" fn(u32, *mut c_char, usize) -> i32,
    rsmi_dev_busy_percent_get: unsafe extern "C" fn(u32, *mut u32) -> i32,
    rsmi_dev_memory_busy_percent_get: unsafe extern "C" fn(u32, *mut u32) -> i32,
    rsmi_dev_memory_total_get: unsafe extern "C" fn(u32, u32, *mut u64) -> i32,
    rsmi_dev_memory_usage_get: unsafe extern "C" fn(u32, u32, *mut u64) -> i32,
    rsmi_dev_temp_metric_get: unsafe extern "C" fn(u32, u32, u32, *mut i64) -> i32,
    rsmi_dev_power_ave_get: unsafe extern "C" fn(u32, u32, *mut u64) -> i32,
    rsmi_dev_power_cap_get: unsafe extern "C" fn(u32, u32, *mut u64) -> i32,
    rsmi_dev_pci_throughput_get: unsafe extern "C" fn(u32, *mut u64, *mut u64, *mut u64) -> i32,
}

// SAFETY: All unsafe blocks in this impl are FFI calls to the ROCm SMI library.
// The function signatures match the library ABI. Pointers are valid for the
// duration of each call. Library handle is managed in try_load/Drop.
#[allow(unsafe_code)]
impl RocmSmi {
    /// Attempts to load the ROCm SMI library.
    fn load() -> Option<Self> {
        // Library paths to try (matching btop's order)
        let lib_paths = [
            "/opt/rocm/lib/librocm_smi64.so",
            "librocm_smi64.so",
            "librocm_smi64.so.5",   // Fedora
            "librocm_smi64.so.1.0", // Debian
            "librocm_smi64.so.6",
            "librocm_smi64.so.7",
        ];

        for path in &lib_paths {
            if let Some(rsmi) = Self::try_load(path) {
                return Some(rsmi);
            }
        }

        None
    }

    #[allow(clippy::missing_transmute_annotations)]
    fn try_load(path: &str) -> Option<Self> {
        use std::ffi::CString;

        let path_cstr = CString::new(path).ok()?;

        // SAFETY: Loading dynamic library and resolving function pointers.
        // All function signatures match ROCm SMI library ABI.
        // Library handle is stored and closed on Drop.
        unsafe {
            let handle = libc::dlopen(path_cstr.as_ptr(), libc::RTLD_LAZY);
            if handle.is_null() {
                return None;
            }

            macro_rules! load_fn {
                ($name:ident) => {{
                    let sym_name = CString::new(stringify!($name)).ok()?;
                    let sym = libc::dlsym(handle, sym_name.as_ptr());
                    if sym.is_null() {
                        libc::dlclose(handle);
                        return None;
                    }
                    std::mem::transmute(sym)
                }};
            }

            let rsmi = Self {
                handle,
                rsmi_init: load_fn!(rsmi_init),
                rsmi_shut_down: load_fn!(rsmi_shut_down),
                rsmi_num_monitor_devices: load_fn!(rsmi_num_monitor_devices),
                rsmi_dev_name_get: load_fn!(rsmi_dev_name_get),
                rsmi_dev_busy_percent_get: load_fn!(rsmi_dev_busy_percent_get),
                rsmi_dev_memory_busy_percent_get: load_fn!(rsmi_dev_memory_busy_percent_get),
                rsmi_dev_memory_total_get: load_fn!(rsmi_dev_memory_total_get),
                rsmi_dev_memory_usage_get: load_fn!(rsmi_dev_memory_usage_get),
                rsmi_dev_temp_metric_get: load_fn!(rsmi_dev_temp_metric_get),
                rsmi_dev_power_ave_get: load_fn!(rsmi_dev_power_ave_get),
                rsmi_dev_power_cap_get: load_fn!(rsmi_dev_power_cap_get),
                rsmi_dev_pci_throughput_get: load_fn!(rsmi_dev_pci_throughput_get),
            };

            // Initialize the library
            if (rsmi.rsmi_init)(0) != rsmi_status::RSMI_STATUS_SUCCESS {
                libc::dlclose(handle);
                return None;
            }

            Some(rsmi)
        }
    }

    fn device_count(&self) -> u32 {
        let mut count: u32 = 0;
        // SAFETY: rsmi_num_monitor_devices writes to a valid &mut u32 pointer.
        // Function pointer is validated non-null during library loading.
        unsafe {
            if (self.rsmi_num_monitor_devices)(&mut count) == rsmi_status::RSMI_STATUS_SUCCESS {
                count
            } else {
                0
            }
        }
    }

    fn device_name(&self, index: u32) -> String {
        let mut name_buf = [0 as c_char; 256];
        // SAFETY: rsmi_dev_name_get writes at most 256 bytes into name_buf.
        // CStr::from_ptr is safe because the buffer is null-terminated by ROCm SMI.
        unsafe {
            if (self.rsmi_dev_name_get)(index, name_buf.as_mut_ptr(), 256)
                == rsmi_status::RSMI_STATUS_SUCCESS
            {
                CStr::from_ptr(name_buf.as_ptr()).to_string_lossy().into_owned()
            } else {
                format!("AMD GPU {index}")
            }
        }
    }

    fn gpu_utilization(&self, index: u32) -> u32 {
        let mut util: u32 = 0;
        // SAFETY: FFI call with valid pointer to stack-allocated u32.
        unsafe {
            if (self.rsmi_dev_busy_percent_get)(index, &mut util)
                == rsmi_status::RSMI_STATUS_SUCCESS
            {
                util
            } else {
                0
            }
        }
    }

    fn mem_utilization(&self, index: u32) -> u32 {
        let mut util: u32 = 0;
        // SAFETY: FFI call with valid pointer to stack-allocated u32.
        unsafe {
            if (self.rsmi_dev_memory_busy_percent_get)(index, &mut util)
                == rsmi_status::RSMI_STATUS_SUCCESS
            {
                util
            } else {
                0
            }
        }
    }

    fn vram_total(&self, index: u32) -> u64 {
        let mut total: u64 = 0;
        // SAFETY: FFI call with valid pointer to stack-allocated u64.
        unsafe {
            if (self.rsmi_dev_memory_total_get)(
                index,
                rsmi_memory_type::RSMI_MEM_TYPE_VRAM,
                &mut total,
            ) == rsmi_status::RSMI_STATUS_SUCCESS
            {
                total
            } else {
                0
            }
        }
    }

    fn vram_used(&self, index: u32) -> u64 {
        let mut used: u64 = 0;
        // SAFETY: FFI call with valid pointer to stack-allocated u64.
        unsafe {
            if (self.rsmi_dev_memory_usage_get)(
                index,
                rsmi_memory_type::RSMI_MEM_TYPE_VRAM,
                &mut used,
            ) == rsmi_status::RSMI_STATUS_SUCCESS
            {
                used
            } else {
                0
            }
        }
    }

    fn temperature(&self, index: u32) -> (f64, f64) {
        let mut current: i64 = 0;
        let mut max: i64 = 0;
        // SAFETY: FFI calls with valid pointers to stack-allocated i64 values.
        unsafe {
            (self.rsmi_dev_temp_metric_get)(
                index,
                rsmi_temp::RSMI_TEMP_TYPE_EDGE,
                rsmi_temp::RSMI_TEMP_CURRENT,
                &mut current,
            );
            (self.rsmi_dev_temp_metric_get)(
                index,
                rsmi_temp::RSMI_TEMP_TYPE_EDGE,
                rsmi_temp::RSMI_TEMP_MAX,
                &mut max,
            );
        }
        // Temperature is in millidegrees Celsius
        (current as f64 / 1000.0, max as f64 / 1000.0)
    }

    fn power(&self, index: u32) -> (f64, f64) {
        let mut power: u64 = 0;
        let mut cap: u64 = 0;
        // SAFETY: FFI calls with valid pointers to stack-allocated u64 values.
        unsafe {
            (self.rsmi_dev_power_ave_get)(index, 0, &mut power);
            (self.rsmi_dev_power_cap_get)(index, 0, &mut cap);
        }
        // Power is in microwatts
        (power as f64 / 1_000_000.0, cap as f64 / 1_000_000.0)
    }

    fn pcie_throughput(&self, index: u32) -> (u64, u64) {
        let mut tx: u64 = 0;
        let mut rx: u64 = 0;
        // SAFETY: FFI call with valid u64 pointers. Null pointer for unused size_sent param.
        unsafe {
            (self.rsmi_dev_pci_throughput_get)(index, &mut tx, &mut rx, std::ptr::null_mut());
        }
        (tx, rx)
    }
}

#[allow(unsafe_code)]
impl Drop for RocmSmi {
    fn drop(&mut self) {
        // SAFETY: Shutting down library and closing handle loaded in try_load.
        unsafe {
            (self.rsmi_shut_down)();
            if !self.handle.is_null() {
                libc::dlclose(self.handle);
            }
        }
    }
}

// SAFETY: ROCm SMI library functions are thread-safe.
// The handle is only used for dlclose in Drop and is never dereferenced.
#[allow(unsafe_code)]
unsafe impl Send for RocmSmi {}
#[allow(unsafe_code)]
unsafe impl Sync for RocmSmi {}

/// Collector for AMD GPU metrics via ROCm SMI.
pub struct AmdGpuCollector {
    /// ROCm SMI library instance.
    rsmi: Option<RocmSmi>,
    /// Number of GPUs detected.
    gpu_count: u32,
    /// GPU utilization history per GPU.
    gpu_history: Vec<RingBuffer<f64>>,
    /// Memory utilization history per GPU.
    mem_history: Vec<RingBuffer<f64>>,
    /// Temperature history per GPU.
    temp_history: Vec<RingBuffer<f64>>,
    /// Power history per GPU (normalized 0-1).
    power_history: Vec<RingBuffer<f64>>,
    /// Cached GPU info.
    gpus: Vec<AmdGpuInfo>,
}

impl std::fmt::Debug for AmdGpuCollector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AmdGpuCollector")
            .field("gpu_count", &self.gpu_count)
            .field("gpus", &self.gpus)
            .finish()
    }
}

impl AmdGpuCollector {
    /// Creates a new AMD GPU collector.
    #[must_use]
    pub fn new() -> Self {
        let rsmi = RocmSmi::load();
        let gpu_count = rsmi.as_ref().map_or(0, RocmSmi::device_count);

        let mut gpu_history = Vec::with_capacity(gpu_count as usize);
        let mut mem_history = Vec::with_capacity(gpu_count as usize);
        let mut temp_history = Vec::with_capacity(gpu_count as usize);
        let mut power_history = Vec::with_capacity(gpu_count as usize);

        for _ in 0..gpu_count {
            gpu_history.push(RingBuffer::new(300));
            mem_history.push(RingBuffer::new(300));
            temp_history.push(RingBuffer::new(300));
            power_history.push(RingBuffer::new(300));
        }

        Self {
            rsmi,
            gpu_count,
            gpu_history,
            mem_history,
            temp_history,
            power_history,
            gpus: Vec::new(),
        }
    }

    /// Returns the number of GPUs detected.
    #[must_use]
    pub fn gpu_count(&self) -> u32 {
        self.gpu_count
    }

    /// Returns GPU utilization history for a specific GPU.
    #[must_use]
    pub fn gpu_history(&self, index: usize) -> Option<&RingBuffer<f64>> {
        self.gpu_history.get(index)
    }

    /// Returns memory utilization history for a specific GPU.
    #[must_use]
    pub fn mem_history(&self, index: usize) -> Option<&RingBuffer<f64>> {
        self.mem_history.get(index)
    }

    /// Returns cached GPU information.
    #[must_use]
    pub fn gpus(&self) -> &[AmdGpuInfo] {
        &self.gpus
    }

    fn collect_all(&mut self) -> Result<Vec<AmdGpuInfo>> {
        let rsmi = self.rsmi.as_ref().ok_or_else(|| MonitorError::CollectionFailed {
            collector: "amd_gpu",
            message: "ROCm SMI not initialized".to_string(),
        })?;

        let mut gpus = Vec::with_capacity(self.gpu_count as usize);

        for i in 0..self.gpu_count {
            let name = rsmi.device_name(i);
            let gpu_util = f64::from(rsmi.gpu_utilization(i));
            let mem_util = f64::from(rsmi.mem_utilization(i));
            let vram_total = rsmi.vram_total(i);
            let vram_used = rsmi.vram_used(i);
            let (temperature, temp_max) = rsmi.temperature(i);
            let (power_watts, power_cap_watts) = rsmi.power(i);
            let (pcie_tx_kbps, pcie_rx_kbps) = rsmi.pcie_throughput(i);

            gpus.push(AmdGpuInfo {
                index: i,
                name,
                gpu_util,
                mem_util,
                vram_used,
                vram_total,
                temperature,
                temp_max,
                power_watts,
                power_cap_watts,
                gpu_clock_mhz: 0, // Would need additional API calls
                mem_clock_mhz: 0,
                pcie_tx_kbps,
                pcie_rx_kbps,
            });
        }

        Ok(gpus)
    }
}

impl Default for AmdGpuCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl Collector for AmdGpuCollector {
    fn id(&self) -> &'static str {
        "amd_gpu"
    }

    fn collect(&mut self) -> Result<Metrics> {
        let gpus = self.collect_all()?;
        let mut metrics = Metrics::new();

        metrics.insert("gpu.count", MetricValue::Counter(u64::from(self.gpu_count)));

        for (i, gpu) in gpus.iter().enumerate() {
            let prefix = format!("gpu.{i}");

            metrics.insert(format!("{prefix}.util"), gpu.gpu_util);
            metrics.insert(format!("{prefix}.mem_util"), gpu.mem_util);
            metrics.insert(format!("{prefix}.vram_used"), MetricValue::Counter(gpu.vram_used));
            metrics.insert(format!("{prefix}.vram_total"), MetricValue::Counter(gpu.vram_total));
            metrics.insert(format!("{prefix}.temp"), gpu.temperature);
            metrics.insert(format!("{prefix}.power_watts"), gpu.power_watts);
            metrics
                .insert(format!("{prefix}.pcie_tx_kbps"), MetricValue::Counter(gpu.pcie_tx_kbps));
            metrics
                .insert(format!("{prefix}.pcie_rx_kbps"), MetricValue::Counter(gpu.pcie_rx_kbps));

            // Update history
            if let Some(history) = self.gpu_history.get_mut(i) {
                history.push(gpu.gpu_util / 100.0);
            }
            if let Some(history) = self.mem_history.get_mut(i) {
                history.push(gpu.mem_util / 100.0);
            }
            if let Some(history) = self.temp_history.get_mut(i) {
                history.push(gpu.temperature);
            }
            if let Some(history) = self.power_history.get_mut(i) {
                let power_norm = if gpu.power_cap_watts > 0.0 {
                    gpu.power_watts / gpu.power_cap_watts
                } else {
                    0.0
                };
                history.push(power_norm);
            }
        }

        self.gpus = gpus;
        Ok(metrics)
    }

    fn is_available(&self) -> bool {
        self.rsmi.is_some() && self.gpu_count > 0
    }

    fn interval_hint(&self) -> Duration {
        Duration::from_millis(1000)
    }

    fn display_name(&self) -> &'static str {
        "AMD GPU"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_amd_collector_new() {
        let collector = AmdGpuCollector::new();
        // May or may not have AMD GPUs, but shouldn't panic
        let _ = collector.gpu_count();
    }

    #[test]
    fn test_amd_gpu_info_struct() {
        let info = AmdGpuInfo {
            index: 0,
            name: "AMD RX 7900 XTX".to_string(),
            gpu_util: 50.0,
            mem_util: 30.0,
            vram_used: 8 * 1024 * 1024 * 1024,
            vram_total: 24 * 1024 * 1024 * 1024,
            temperature: 65.0,
            temp_max: 110.0,
            power_watts: 200.0,
            power_cap_watts: 355.0,
            gpu_clock_mhz: 2500,
            mem_clock_mhz: 2500,
            pcie_tx_kbps: 1000,
            pcie_rx_kbps: 2000,
        };

        assert_eq!(info.index, 0);
        assert_eq!(info.gpu_util, 50.0);
    }
}
