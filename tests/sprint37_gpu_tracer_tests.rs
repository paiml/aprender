//! Integration tests for GPU tracer (Sprint 37)
//!
//! Tests wgpu-profiler integration and adaptive sampling.

use renacer::gpu_tracer::GpuTracerConfig;

#[test]
fn test_gpu_config_defaults() {
    let config = GpuTracerConfig::default();
    assert_eq!(config.threshold_us, 100);
    assert!(!config.trace_all);
}

#[test]
fn test_gpu_config_custom_threshold() {
    let config = GpuTracerConfig { threshold_us: 500, trace_all: false };
    assert_eq!(config.threshold_us, 500);
    assert!(!config.trace_all);
}

#[test]
fn test_gpu_config_debug_mode() {
    let config = GpuTracerConfig { threshold_us: 0, trace_all: true };
    assert_eq!(config.threshold_us, 0);
    assert!(config.trace_all);
}

#[test]
fn test_gpu_config_clone() {
    let config1 = GpuTracerConfig { threshold_us: 1000, trace_all: true };
    let config2 = config1.clone();
    assert_eq!(config1.threshold_us, config2.threshold_us);
    assert_eq!(config1.trace_all, config2.trace_all);
}

#[test]
fn test_gpu_config_debug_format() {
    let config = GpuTracerConfig::default();
    let debug_str = format!("{:?}", config);
    assert!(debug_str.contains("GpuTracerConfig"));
    assert!(debug_str.contains("threshold_us"));
    assert!(debug_str.contains("trace_all"));
}

#[cfg(feature = "gpu-tracing")]
mod gpu_tracing_tests {
    use super::*;
    use renacer::gpu_tracer::GpuProfilerWrapper;

    #[test]
    fn test_gpu_profiler_creation_without_otlp() {
        let config = GpuTracerConfig::default();
        let result = GpuProfilerWrapper::new(None, config);

        // Should succeed even without OTLP exporter
        assert!(result.is_ok());
    }

    #[test]
    fn test_gpu_profiler_with_custom_config() {
        let config = GpuTracerConfig { threshold_us: 1000, trace_all: true };
        let result = GpuProfilerWrapper::new(None, config);
        assert!(result.is_ok());
    }
}

#[cfg(not(feature = "gpu-tracing"))]
mod gpu_tracing_disabled_tests {
    use super::*;
    use renacer::gpu_tracer::GpuProfilerWrapper;

    #[test]
    fn test_gpu_profiler_feature_disabled() {
        let config = GpuTracerConfig::default();
        let result = GpuProfilerWrapper::new(None, config);

        assert!(result.is_err());
        if let Err(err) = result {
            let err_msg = err.to_string();
            assert!(err_msg.contains("GPU tracing support not compiled in"));
            assert!(err_msg.contains("gpu-tracing"));
        }
    }
}

#[test]
fn test_gpu_config_extreme_values() {
    let config = GpuTracerConfig { threshold_us: u64::MAX, trace_all: false };
    assert_eq!(config.threshold_us, u64::MAX);
}

#[test]
fn test_gpu_config_zero_threshold() {
    let config = GpuTracerConfig { threshold_us: 0, trace_all: false };
    assert_eq!(config.threshold_us, 0);
}
