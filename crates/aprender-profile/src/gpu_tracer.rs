//! GPU kernel tracing wrapper for wgpu-profiler (Sprint 37)
//!
//! Integrates wgpu timestamp queries with Renacer's OTLP export infrastructure.
//! Follows Sprint 32's adaptive sampling and block-level tracing patterns.
//!
//! # Architecture
//!
//! - Wraps `wgpu-profiler::GpuProfiler` for timestamp query management
//! - Converts wgpu profiling results → `GpuKernel` structs
//! - Applies adaptive sampling (duration > 100μs by default)
//! - Exports to OTLP via `OtlpExporter::record_gpu_kernel()`
//!
//! # Usage
//!
//! ```ignore
//! use renacer::{GpuProfilerWrapper, GpuTracerConfig, OtlpExporter, OtlpConfig};
//!
//! // Setup OTLP exporter
//! let otlp_config = OtlpConfig::new(
//!     "http://localhost:4317".to_string(),
//!     "my-gpu-app".to_string(),
//! );
//! let otlp_exporter = OtlpExporter::new(otlp_config, None).expect("test");
//! let otlp_arc = std::sync::Arc::new(otlp_exporter);
//!
//! // Setup GPU profiler wrapper
//! let mut gpu_tracer = GpuProfilerWrapper::new(
//!     Some(otlp_arc.clone()),
//!     GpuTracerConfig::default(),
//! )
//! .expect("test");
//!
//! // Instrument GPU code (standard wgpu-profiler API)
//! let mut encoder = device.create_command_encoder(&Default::default());
//! {
//!     let mut scope = gpu_tracer.profiler_mut().scope("kernel_name", &mut encoder);
//!     let mut compute_pass = scope.scoped_compute_pass("compute");
//!     // ... GPU commands ...
//! }
//!
//! gpu_tracer.profiler_mut().resolve_queries(&mut encoder);
//! queue.submit(Some(encoder.finish()));
//! gpu_tracer.profiler_mut().end_frame().expect("test");
//!
//! // Export GPU profiling results to OTLP
//! let timestamp_period = queue.get_timestamp_period();
//! gpu_tracer.export_frame(timestamp_period);
//! ```

use anyhow::Result;

#[cfg(feature = "gpu-tracing")]
use crate::otlp_exporter::GpuKernel;
use crate::otlp_exporter::OtlpExporter;

/// Configuration for GPU kernel tracing
///
/// Follows Sprint 32's adaptive sampling pattern:
/// - Default threshold: 100μs (same as SIMD compute tracing)
/// - `trace_all`: false (safe by default, only trace slow kernels)
#[derive(Debug, Clone)]
pub struct GpuTracerConfig {
    /// Minimum duration to trace (default: 100μs, same as Sprint 32 SIMD tracing)
    pub threshold_us: u64,
    /// Trace all kernels regardless of duration (debug mode)
    pub trace_all: bool,
}

impl Default for GpuTracerConfig {
    fn default() -> Self {
        GpuTracerConfig {
            threshold_us: 100, // Same as Sprint 32 SIMD tracing
            trace_all: false,  // Safe by default (adaptive sampling)
        }
    }
}

/// Wrapper around wgpu-profiler that exports to OTLP
///
/// This wrapper integrates wgpu-profiler (community-standard GPU profiling
/// library) with Renacer's OTLP export infrastructure. It follows Sprint 32's
/// Toyota Way principles:
/// - **Genchi Genbutsu**: Reuse proven wgpu-profiler infrastructure
/// - **Jidoka**: Mandatory adaptive sampling (cannot `DoS` tracing backend)
/// - **Muda**: Kernel-level tracing (no per-instruction overhead)
/// - **Poka-Yoke**: Feature flag prevents accidental overhead
#[cfg(feature = "gpu-tracing")]
pub struct GpuProfilerWrapper {
    profiler: wgpu_profiler::GpuProfiler,
    otlp_exporter: Option<std::sync::Arc<OtlpExporter>>,
    config: GpuTracerConfig,
}

#[cfg(feature = "gpu-tracing")]
impl GpuProfilerWrapper {
    /// Create a new GPU profiler wrapper
    ///
    /// # Arguments
    ///
    /// * `otlp_exporter` - Optional OTLP exporter for trace export
    /// * `config` - Tracing configuration (thresholds, sampling)
    ///
    /// # Returns
    ///
    /// Returns `Ok(GpuProfilerWrapper)` on success, or error if:
    /// - wgpu-profiler initialization fails
    ///
    /// # Note
    ///
    /// In wgpu-profiler 0.18, `GpuProfiler::new()` doesn't require a device.
    /// The profiler is initialized lazily when first used.
    pub fn new(
        otlp_exporter: Option<std::sync::Arc<OtlpExporter>>,
        config: GpuTracerConfig,
    ) -> Result<Self> {
        let settings = wgpu_profiler::GpuProfilerSettings::default();
        let profiler = wgpu_profiler::GpuProfiler::new(settings)?;

        Ok(GpuProfilerWrapper { profiler, otlp_exporter, config })
    }

    /// Get a reference to the underlying wgpu-profiler
    ///
    /// Users instrument their wgpu code with standard wgpu-profiler API:
    /// ```ignore
    /// let mut scope = wrapper.profiler_mut().scope("kernel_name", &mut encoder);
    /// let mut compute_pass = scope.scoped_compute_pass("compute");
    /// // ... GPU commands ...
    /// ```
    pub fn profiler_mut(&mut self) -> &mut wgpu_profiler::GpuProfiler {
        &mut self.profiler
    }

    /// Process finished GPU profiling frame and export to OTLP
    ///
    /// Call this after `queue.submit()` and `profiler.end_frame()`.
    ///
    /// # Arguments
    ///
    /// * `timestamp_period` - GPU timestamp period from `queue.get_timestamp_period()`
    ///
    /// # Adaptive Sampling
    ///
    /// Only kernels with `duration >= threshold_us` (default 100μs) are exported,
    /// unless `config.trace_all = true` (debug mode).
    ///
    /// This prevents `DoS` on the tracing backend (Toyota Way: Jidoka).
    pub fn export_frame(&mut self, timestamp_period: f32) {
        if let Some(frame_data) = self.profiler.process_finished_frame(timestamp_period) {
            // Convert wgpu-profiler results to GpuKernel structs
            // Note: In wgpu-profiler 0.18, `time` is Option<Range<f64>> representing
            // the start and end timestamps in seconds
            for scope in &frame_data {
                // Calculate duration from time range
                let duration_us = if let Some(ref time_range) = scope.time {
                    // Duration = end - start, converted to microseconds
                    ((time_range.end - time_range.start) * 1_000_000.0) as u64
                } else {
                    // If time is None, skip this scope (no timing data available)
                    continue;
                };

                // Adaptive sampling: Only export if duration > threshold OR debug mode
                if self.config.trace_all || duration_us >= self.config.threshold_us {
                    if let Some(ref exporter) = self.otlp_exporter {
                        let kernel = GpuKernel {
                            kernel: scope.label.clone(),
                            duration_us,
                            backend: "wgpu",
                            workgroup_size: None, // Future: Extract from wgpu metadata
                            elements: None,       // Future: User-provided via scope metadata
                            is_slow: duration_us > self.config.threshold_us,
                        };

                        exporter.record_gpu_kernel(kernel);
                    }
                }
            }
        }
    }
}

// Stub implementation when GPU tracing feature is disabled
#[cfg(not(feature = "gpu-tracing"))]
pub struct GpuProfilerWrapper;

#[cfg(not(feature = "gpu-tracing"))]
impl GpuProfilerWrapper {
    pub fn new(
        _otlp_exporter: Option<std::sync::Arc<OtlpExporter>>,
        _config: GpuTracerConfig,
    ) -> Result<Self> {
        anyhow::bail!("GPU tracing support not compiled in. Enable the 'gpu-tracing' feature.");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_tracer_config_defaults() {
        let config = GpuTracerConfig::default();

        assert_eq!(config.threshold_us, 100); // Same as Sprint 32 SIMD tracing
        assert!(!config.trace_all); // Safe by default (adaptive sampling)
    }

    #[test]
    fn test_gpu_tracer_config_custom() {
        let config = GpuTracerConfig {
            threshold_us: 1000, // 1ms threshold
            trace_all: true,    // Debug mode
        };

        assert_eq!(config.threshold_us, 1000);
        assert!(config.trace_all);
    }
}
