//! GPU Monitor for real-time metrics collection (TRUENO-SPEC-010 Section 8.2)

use std::collections::VecDeque;
use std::sync::{Arc, Mutex, RwLock};

use super::{GpuDeviceInfo, GpuMemoryMetrics, GpuMetrics, MonitorConfig, MonitorError};

/// GPU Monitor for real-time metrics collection (TRUENO-SPEC-010)
///
/// Provides both on-demand and background metric collection with configurable
/// polling intervals and history retention.
///
/// # Example
///
/// ```rust,ignore
/// use trueno::monitor::{GpuMonitor, MonitorConfig};
///
/// // Create monitor for device 0
/// let monitor = GpuMonitor::new(0, MonitorConfig::default())?;
///
/// // Get latest metrics
/// let metrics = monitor.latest()?;
/// println!("GPU usage: {}%", metrics.utilization.gpu_percent);
///
/// // Get history (ring buffer)
/// let history = monitor.history();
/// println!("Samples: {}", history.len());
/// ```
pub struct GpuMonitor {
    /// Device info
    device_info: GpuDeviceInfo,
    /// Configuration
    config: MonitorConfig,
    /// Metrics history (ring buffer)
    history: Arc<RwLock<VecDeque<GpuMetrics>>>,
    /// Background thread handle
    #[cfg(feature = "gpu")]
    _background_handle: Option<std::thread::JoinHandle<()>>,
    /// Stop signal for background thread
    stop_signal: Arc<Mutex<bool>>,
}

impl GpuMonitor {
    /// Create a new GPU monitor for the specified device
    ///
    /// # Errors
    ///
    /// Returns error if device is not found or initialization fails.
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    pub fn new(device_index: u32, config: MonitorConfig) -> Result<Self, MonitorError> {
        let device_info = GpuDeviceInfo::query_device(device_index)?;
        let history = Arc::new(RwLock::new(VecDeque::with_capacity(config.history_size)));
        let stop_signal = Arc::new(Mutex::new(false));

        let monitor = Self { device_info, config, history, _background_handle: None, stop_signal };

        Ok(monitor)
    }

    /// Create monitor without GPU feature (for testing)
    #[cfg(not(feature = "gpu"))]
    pub fn new(_device_index: u32, _config: MonitorConfig) -> Result<Self, MonitorError> {
        Err(MonitorError::NotAvailable("GPU feature not enabled".to_string()))
    }

    /// Create a mock monitor for testing (no GPU required)
    #[must_use]
    pub fn mock(device_info: GpuDeviceInfo, config: MonitorConfig) -> Self {
        Self {
            device_info,
            config,
            history: Arc::new(RwLock::new(VecDeque::with_capacity(16))),
            #[cfg(feature = "gpu")]
            _background_handle: None,
            stop_signal: Arc::new(Mutex::new(false)),
        }
    }

    /// Get device info
    #[must_use]
    pub fn device_info(&self) -> &GpuDeviceInfo {
        &self.device_info
    }

    /// Get current configuration
    #[must_use]
    pub fn config(&self) -> &MonitorConfig {
        &self.config
    }

    /// Collect metrics sample now
    ///
    /// This performs an immediate collection and adds to history.
    pub fn collect(&self) -> Result<GpuMetrics, MonitorError> {
        // For now, return basic memory metrics
        // Full implementation would query NVML/wgpu for utilization, thermal, etc.
        let memory = GpuMemoryMetrics::new(
            self.device_info.vram_total,
            0, // Would query actual usage
            self.device_info.vram_total,
        );

        let metrics = GpuMetrics::new(self.device_info.index, memory);

        // Add to history
        if let Ok(mut history) = self.history.write() {
            if history.len() >= self.config.history_size {
                history.pop_front();
            }
            history.push_back(metrics.clone());
        }

        Ok(metrics)
    }

    /// Get the latest metrics snapshot (without collecting)
    pub fn latest(&self) -> Result<GpuMetrics, MonitorError> {
        self.history
            .read()
            .ok()
            .and_then(|h| h.back().cloned())
            .ok_or(MonitorError::QueryFailed("No metrics available".to_string()))
    }

    /// Get history buffer (read-only snapshot)
    #[must_use]
    pub fn history(&self) -> Vec<GpuMetrics> {
        self.history.read().map(|h| h.iter().cloned().collect()).unwrap_or_default()
    }

    /// Get number of samples in history
    #[must_use]
    pub fn sample_count(&self) -> usize {
        self.history.read().map(|h| h.len()).unwrap_or(0)
    }

    /// Clear history buffer
    pub fn clear_history(&self) {
        if let Ok(mut history) = self.history.write() {
            history.clear();
        }
    }

    /// Check if background collection is active
    #[must_use]
    pub fn is_collecting(&self) -> bool {
        #[cfg(feature = "gpu")]
        {
            self._background_handle.is_some()
        }
        #[cfg(not(feature = "gpu"))]
        {
            false
        }
    }

    /// Stop background collection (if running)
    pub fn stop(&self) {
        if let Ok(mut stop) = self.stop_signal.lock() {
            *stop = true;
        }
    }
}

impl Drop for GpuMonitor {
    fn drop(&mut self) {
        self.stop();
    }
}
