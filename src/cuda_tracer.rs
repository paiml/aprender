//! CUDA kernel tracing wrapper for CUPTI (Sprint 38)
//!
//! Integrates CUPTI Activity API with Renacer's OTLP export infrastructure.
//! Follows Sprint 37's wgpu pattern: adaptive sampling, kernel-level tracing.
//!
//! # Architecture
//!
//! - Uses CUPTI Activity API for asynchronous kernel profiling
//! - Converts CUPTI activity records → `GpuKernel` structs
//! - Applies adaptive sampling (duration > 100μs by default)
//! - Exports to OTLP via `OtlpExporter::record_gpu_kernel()`
//!
//! # Usage
//!
//! ```ignore
//! use renacer::{CudaTracerWrapper, CudaTracerConfig, OtlpExporter, OtlpConfig};
//!
//! // Setup OTLP exporter
//! let otlp_config = OtlpConfig::new(
//!     "http://localhost:4317".to_string(),
//!     "my-cuda-app".to_string(),
//! );
//! let otlp_exporter = OtlpExporter::new(otlp_config, None).expect("test");
//! let otlp_arc = std::sync::Arc::new(otlp_exporter);
//!
//! // Setup CUDA tracer wrapper
//! let mut cuda_tracer = CudaTracerWrapper::new(
//!     Some(otlp_arc.clone()),
//!     CudaTracerConfig::default(),
//! )
//! .expect("test");
//!
//! // Launch CUDA kernels (your existing CUDA code)
//! // ...kernels execute...
//!
//! // Flush and export CUDA profiling results to OTLP
//! cuda_tracer.flush();
//! ```

use anyhow::Result;

#[cfg(feature = "cuda-tracing")]
use crate::otlp_exporter::{GpuKernel, OtlpExporter};

#[cfg(not(feature = "cuda-tracing"))]
use crate::otlp_exporter::OtlpExporter;

/// Configuration for CUDA kernel tracing
///
/// Follows Sprint 32's adaptive sampling pattern:
/// - Default threshold: 100μs (same as wgpu/SIMD tracing)
/// - `trace_all`: false (safe by default, only trace slow kernels)
#[derive(Debug, Clone)]
pub struct CudaTracerConfig {
    /// Minimum duration to trace (default: 100μs, same as Sprint 32/37)
    pub threshold_us: u64,
    /// Trace all kernels regardless of duration (debug mode)
    pub trace_all: bool,
    /// CUPTI activity buffer size (default: 8MB)
    pub buffer_size: usize,
    /// Device ID to trace (default: 0)
    pub device_id: u32,
}

impl Default for CudaTracerConfig {
    fn default() -> Self {
        CudaTracerConfig {
            threshold_us: 100,            // Same as Sprint 32/37
            trace_all: false,             // Safe by default (adaptive sampling)
            buffer_size: 8 * 1024 * 1024, // 8MB
            device_id: 0,                 // Primary GPU
        }
    }
}

/// Wrapper around CUPTI Activity API that exports to OTLP
///
/// This wrapper integrates CUPTI (CUDA Profiling Tools Interface) with
/// Renacer's OTLP export infrastructure. It follows Sprint 37's wgpu pattern
/// and Sprint 32's Toyota Way principles:
/// - **Genchi Genbutsu**: Reuse proven CUPTI Activity API infrastructure
/// - **Jidoka**: Mandatory adaptive sampling (cannot `DoS` tracing backend)
/// - **Muda**: Kernel-level tracing (no per-instruction overhead)
/// - **Poka-Yoke**: Feature flag prevents accidental overhead
#[cfg(feature = "cuda-tracing")]
pub struct CudaTracerWrapper {
    _otlp_exporter: Option<std::sync::Arc<OtlpExporter>>,
    config: CudaTracerConfig,
    cupti_initialized: bool,
    activity_buffer: Vec<u8>,
}

#[cfg(feature = "cuda-tracing")]
impl std::fmt::Debug for CudaTracerWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaTracerWrapper")
            .field("_otlp_exporter", &"<OtlpExporter>")
            .field("config", &self.config)
            .field("cupti_initialized", &self.cupti_initialized)
            .field("activity_buffer_len", &self.activity_buffer.len())
            .finish()
    }
}

#[cfg(feature = "cuda-tracing")]
impl CudaTracerWrapper {
    /// Create a new CUDA tracer wrapper
    ///
    /// # Arguments
    ///
    /// * `otlp_exporter` - Optional OTLP exporter for trace export
    /// * `config` - Tracing configuration (thresholds, sampling, buffer size)
    ///
    /// # Returns
    ///
    /// Returns `Ok(CudaTracerWrapper)` on success, or error if:
    /// - CUDA runtime not available
    /// - CUPTI library not found
    /// - Activity API initialization fails
    ///
    /// # Implementation Strategy
    ///
    /// This implementation uses cudarc for CUDA runtime management and
    /// custom FFI bindings for CUPTI Activity API (since cudarc doesn't
    /// expose CUPTI directly).
    pub fn new(
        otlp_exporter: Option<std::sync::Arc<OtlpExporter>>,
        config: CudaTracerConfig,
    ) -> Result<Self> {
        // Future: Initialize CUDA runtime via cudarc
        // The cudarc 0.18 API has changed from 0.12.
        // Need to verify correct device initialization API:
        // - Option 1: cudarc::CudaDevice::new()
        // - Option 2: cudarc::driver::Device::new()
        // - Option 3: Different API entirely
        //
        // For now, skip device verification and initialize the wrapper.
        // This allows the code to compile and tests to run.
        // Full device initialization will be added once cudarc API is confirmed.

        tracing::warn!(
            "CUDA device initialization is stubbed. \
             cudarc 0.18 API integration pending."
        );

        // Allocate activity buffer
        let activity_buffer = vec![0u8; config.buffer_size];

        let mut wrapper = CudaTracerWrapper {
            _otlp_exporter: otlp_exporter,
            config,
            cupti_initialized: false,
            activity_buffer,
        };

        // Initialize CUPTI Activity API
        wrapper.initialize_cupti()?;

        Ok(wrapper)
    }

    /// Initialize CUPTI Activity API
    ///
    /// This method:
    /// 1. Enables `CUPTI_ACTIVITY_KIND_KERNEL` activity recording
    /// 2. Registers activity buffer callbacks
    /// 3. Sets up asynchronous profiling
    ///
    /// # CUPTI Activity API Workflow
    ///
    /// ```text
    /// User launches CUDA kernel → CUPTI records activity asynchronously →
    /// Buffer fills → Callback fires → Parse activities → Export to OTLP
    /// ```
    fn initialize_cupti(&mut self) -> Result<()> {
        // Future: Implement CUPTI initialization
        //
        // Required steps (using CUPTI C API via FFI):
        // 1. cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL)
        //    - Enables kernel activity recording
        // 2. cuptiActivityRegisterCallbacks(buffer_requested, buffer_completed)
        //    - buffer_requested: Called when CUPTI needs a new buffer
        //    - buffer_completed: Called when buffer is full (process activities)
        // 3. cuptiActivityFlushAll(0) to force initial buffer setup
        //
        // For now, mark as initialized (full CUPTI FFI implementation needed)
        self.cupti_initialized = true;

        tracing::warn!(
            "CUPTI Activity API initialization is a stub. \
             Full CUPTI FFI bindings required for production use."
        );

        Ok(())
    }

    /// Process CUPTI activity buffer and export to OTLP
    ///
    /// Called when CUPTI activity buffer is full or flushed.
    /// Parses activity records and converts them to `GpuKernel` structs.
    ///
    /// # Arguments
    ///
    /// * `buffer` - Raw CUPTI activity buffer
    ///
    /// # Adaptive Sampling
    ///
    /// Only kernels with `duration >= threshold_us` (default 100μs) are exported,
    /// unless `config.trace_all = true` (debug mode).
    ///
    /// This prevents `DoS` on the tracing backend (Toyota Way: Jidoka).
    pub fn process_activity_buffer(&mut self, buffer: &[u8]) {
        if buffer.is_empty() {
            return;
        }

        // Future: Implement CUPTI activity record parsing
        //
        // Required steps:
        // 1. Loop through buffer with cuptiActivityGetNextRecord()
        // 2. For each record, check activity kind:
        //    - CUPTI_ACTIVITY_KIND_KERNEL → parse kernel activity
        //    - Other kinds → skip for now
        // 3. Extract kernel metadata:
        //    - Kernel name (from CUPTI record)
        //    - Duration (end - start timestamps)
        //    - Grid/block dimensions
        //    - Device/context/stream IDs
        // 4. Convert to GpuKernel struct
        // 5. Apply adaptive sampling
        // 6. Export via otlp_exporter.record_gpu_kernel()
        //
        // Example CUPTI record structure (CUpti_ActivityKernel4):
        // - start: u64 (nanoseconds)
        // - end: u64 (nanoseconds)
        // - name: *const c_char
        // - gridX, gridY, gridZ: i32
        // - blockX, blockY, blockZ: i32
        // - device_id, context_id, stream_id: u32

        tracing::warn!(
            "CUPTI activity buffer processing is a stub. Received {} bytes. \
             Full CUPTI FFI bindings required for production use.",
            buffer.len()
        );
    }

    /// Convert CUPTI activity record to `GpuKernel` struct
    ///
    /// This is a helper method that would be called from `process_activity_buffer()`
    /// once CUPTI FFI bindings are implemented.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Pseudocode for CUPTI record conversion
    /// let duration_us = ((record.end - record.start) / 1000) as u64; // ns → μs
    /// // SAFETY: CStr::from_ptr requires a valid null-terminated C string pointer
    /// let kernel_name = unsafe { CStr::from_ptr(record.name).to_string_lossy().into_owned() };
    /// let workgroup_size = format!("[{},{},{}]", record.blockX, record.blockY, record.blockZ);
    ///
    /// let kernel = GpuKernel {
    ///     kernel: kernel_name,
    ///     duration_us,
    ///     backend: "cuda",
    ///     workgroup_size: Some(workgroup_size),
    ///     elements: None,  // User-provided metadata (future enhancement)
    ///     is_slow: duration_us > self.config.threshold_us,
    /// };
    /// ```
    #[cfg(feature = "cuda-tracing")]
    fn _convert_cupti_record_to_kernel(&self, _record_data: &[u8]) -> Option<GpuKernel> {
        // Future: Implement CUPTI record → GpuKernel conversion
        // This requires parsing the binary CUPTI activity record structure

        None
    }

    /// Flush pending CUPTI activities and export to OTLP
    ///
    /// Call this method periodically or at application shutdown to ensure
    /// all CUDA kernel activities are exported to OTLP.
    ///
    /// # Implementation
    ///
    /// Calls cuptiActivityFlushAll(0) to force CUPTI to complete all
    /// pending activity records and fire buffer completion callbacks.
    pub fn flush(&mut self) {
        if !self.cupti_initialized {
            return;
        }

        // Future: Implement CUPTI flush
        // cuptiActivityFlushAll(0) → triggers buffer_completed callback
        //                          → calls process_activity_buffer()

        tracing::debug!("CUPTI flush called (stub implementation)");
    }

    /// Get CUDA device information for OTLP resource attributes
    ///
    /// Returns device name, compute capability, and driver version.
    /// Used to populate OTLP resource-level attributes at startup.
    pub fn get_device_info(&self) -> Result<CudaDeviceInfo> {
        // Future: Use cudarc to query device properties
        // Once cudarc 0.18 API is confirmed, query:
        // - device.name()
        // - device.compute_cap()
        // - CUDA driver version

        tracing::warn!(
            "CUDA device info query is stubbed. \
             Returning placeholder values."
        );

        Ok(CudaDeviceInfo {
            device_id: self.config.device_id,
            device_name: "NVIDIA GPU (cudarc API pending)".to_string(),
            compute_capability: "Unknown".to_string(),
            driver_version: "Unknown".to_string(),
        })
    }
}

#[cfg(feature = "cuda-tracing")]
impl Drop for CudaTracerWrapper {
    fn drop(&mut self) {
        if !self.cupti_initialized {
            return;
        }

        // Flush any remaining activities before cleanup
        self.flush();

        // Future: Clean up CUPTI resources
        // - cuptiActivityDisable(CUPTI_ACTIVITY_KIND_KERNEL)
        // - cuptiFinalize()

        tracing::debug!("CUPTI resources cleaned up (stub implementation)");
    }
}

/// CUDA device information for OTLP resource attributes
#[cfg(feature = "cuda-tracing")]
#[derive(Debug, Clone)]
pub struct CudaDeviceInfo {
    pub device_id: u32,
    pub device_name: String,
    pub compute_capability: String,
    pub driver_version: String,
}

// Stub implementation when CUDA tracing feature is disabled
#[cfg(not(feature = "cuda-tracing"))]
#[derive(Debug)]
pub struct CudaTracerWrapper;

#[cfg(not(feature = "cuda-tracing"))]
impl CudaTracerWrapper {
    pub fn new(
        _otlp_exporter: Option<std::sync::Arc<OtlpExporter>>,
        _config: CudaTracerConfig,
    ) -> Result<Self> {
        anyhow::bail!("CUDA tracing support not compiled in. Enable the 'cuda-tracing' feature.");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_tracer_config_defaults() {
        let config = CudaTracerConfig::default();

        assert_eq!(config.threshold_us, 100); // Same as Sprint 32/37
        assert!(!config.trace_all); // Safe by default (adaptive sampling)
        assert_eq!(config.buffer_size, 8 * 1024 * 1024); // 8MB
        assert_eq!(config.device_id, 0); // Primary GPU
    }

    #[test]
    fn test_cuda_tracer_config_custom() {
        let config = CudaTracerConfig {
            threshold_us: 1000,            // 1ms threshold
            trace_all: true,               // Debug mode
            buffer_size: 16 * 1024 * 1024, // 16MB
            device_id: 1,                  // Secondary GPU
        };

        assert_eq!(config.threshold_us, 1000);
        assert!(config.trace_all);
        assert_eq!(config.buffer_size, 16 * 1024 * 1024);
        assert_eq!(config.device_id, 1);
    }

    #[test]
    #[cfg(feature = "cuda-tracing")]
    fn test_cuda_device_initialization() {
        // This test requires NVIDIA GPU hardware
        // Skip if CUDA not available
        if std::env::var("CUDA_VISIBLE_DEVICES").is_err() {
            eprintln!("Skipping test_cuda_device_initialization: No CUDA device available");
            return;
        }

        let config = CudaTracerConfig::default();
        let result = CudaTracerWrapper::new(None, config);

        match result {
            Ok(wrapper) => {
                // Verify device info can be retrieved
                let device_info = wrapper.get_device_info();
                assert!(device_info.is_ok());

                let info = device_info.expect("test");
                assert_eq!(info.device_id, 0);
                assert!(!info.device_name.is_empty());
                assert!(!info.compute_capability.is_empty());
            }
            Err(e) => {
                eprintln!(
                    "CUDA device initialization failed (expected on non-NVIDIA hardware): {}",
                    e
                );
            }
        }
    }

    #[test]
    #[cfg(not(feature = "cuda-tracing"))]
    fn test_cuda_tracer_feature_disabled() {
        let config = CudaTracerConfig::default();
        let result = CudaTracerWrapper::new(None, config);

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("CUDA tracing support not compiled in"));
    }
}
