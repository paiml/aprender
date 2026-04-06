//! Integration tests for CUDA tracer (Sprint 38)
//!
//! Tests CUPTI Activity API integration and adaptive sampling.

use renacer::cuda_tracer::CudaTracerConfig;

#[test]
fn test_cuda_config_defaults() {
    let config = CudaTracerConfig::default();
    assert_eq!(config.threshold_us, 100);
    assert!(!config.trace_all);
    assert_eq!(config.buffer_size, 8 * 1024 * 1024);
    assert_eq!(config.device_id, 0);
}

#[test]
fn test_cuda_config_custom_threshold() {
    let config = CudaTracerConfig {
        threshold_us: 500,
        trace_all: false,
        buffer_size: 16 * 1024 * 1024,
        device_id: 0,
    };
    assert_eq!(config.threshold_us, 500);
    assert!(!config.trace_all);
    assert_eq!(config.buffer_size, 16 * 1024 * 1024);
}

#[test]
fn test_cuda_config_debug_mode() {
    let config = CudaTracerConfig {
        threshold_us: 0,
        trace_all: true,
        buffer_size: 4 * 1024 * 1024,
        device_id: 0,
    };
    assert_eq!(config.threshold_us, 0);
    assert!(config.trace_all);
}

#[test]
fn test_cuda_config_multi_gpu() {
    let config = CudaTracerConfig {
        threshold_us: 100,
        trace_all: false,
        buffer_size: 8 * 1024 * 1024,
        device_id: 3,
    };
    assert_eq!(config.device_id, 3);
}

#[test]
fn test_cuda_config_large_buffer() {
    let config = CudaTracerConfig {
        threshold_us: 100,
        trace_all: false,
        buffer_size: 64 * 1024 * 1024, // 64MB
        device_id: 0,
    };
    assert_eq!(config.buffer_size, 64 * 1024 * 1024);
}

#[test]
fn test_cuda_config_small_buffer() {
    let config = CudaTracerConfig {
        threshold_us: 100,
        trace_all: false,
        buffer_size: 1024 * 1024, // 1MB
        device_id: 0,
    };
    assert_eq!(config.buffer_size, 1024 * 1024);
}

#[test]
fn test_cuda_config_clone() {
    let config1 = CudaTracerConfig {
        threshold_us: 1000,
        trace_all: true,
        buffer_size: 16 * 1024 * 1024,
        device_id: 2,
    };
    let config2 = config1.clone();
    assert_eq!(config1.threshold_us, config2.threshold_us);
    assert_eq!(config1.trace_all, config2.trace_all);
    assert_eq!(config1.buffer_size, config2.buffer_size);
    assert_eq!(config1.device_id, config2.device_id);
}

#[test]
fn test_cuda_config_debug_format() {
    let config = CudaTracerConfig::default();
    let debug_str = format!("{:?}", config);
    assert!(debug_str.contains("CudaTracerConfig"));
    assert!(debug_str.contains("threshold_us"));
    assert!(debug_str.contains("trace_all"));
    assert!(debug_str.contains("buffer_size"));
    assert!(debug_str.contains("device_id"));
}

#[cfg(feature = "cuda-tracing")]
mod cuda_tracing_tests {
    use super::*;
    use renacer::cuda_tracer::CudaTracerWrapper;

    #[test]
    fn test_cuda_tracer_creation() {
        let config = CudaTracerConfig::default();
        let result = CudaTracerWrapper::new(None, config);

        // May fail if no CUDA hardware, but should compile
        match result {
            Ok(mut wrapper) => {
                // Test flush
                wrapper.flush();

                // Test device info
                let device_info = wrapper.get_device_info();
                assert!(device_info.is_ok());

                let info = device_info.expect("test");
                assert_eq!(info.device_id, 0);
            }
            Err(e) => {
                eprintln!("CUDA not available (expected on non-NVIDIA hardware): {}", e);
            }
        }
    }

    #[test]
    fn test_cuda_tracer_process_empty_buffer() {
        let config = CudaTracerConfig::default();
        if let Ok(mut wrapper) = CudaTracerWrapper::new(None, config) {
            // Processing empty buffer should not panic
            wrapper.process_activity_buffer(&[]);
        }
    }

    #[test]
    fn test_cuda_tracer_process_small_buffer() {
        let config = CudaTracerConfig::default();
        if let Ok(mut wrapper) = CudaTracerWrapper::new(None, config) {
            // Processing small buffer should not panic
            let buffer = vec![0u8; 1024];
            wrapper.process_activity_buffer(&buffer);
        }
    }

    #[test]
    fn test_cuda_tracer_debug_format() {
        let config = CudaTracerConfig::default();
        if let Ok(wrapper) = CudaTracerWrapper::new(None, config) {
            let debug_str = format!("{:?}", wrapper);
            assert!(debug_str.contains("CudaTracerWrapper"));
        }
    }

    #[test]
    fn test_cuda_tracer_custom_buffer_size() {
        let config = CudaTracerConfig {
            threshold_us: 100,
            trace_all: false,
            buffer_size: 16 * 1024 * 1024,
            device_id: 0,
        };
        let result = CudaTracerWrapper::new(None, config);

        match result {
            Ok(_wrapper) => {
                // Success with custom buffer size
            }
            Err(e) => {
                eprintln!("CUDA not available: {}", e);
            }
        }
    }

    #[test]
    fn test_cuda_tracer_multiple_flushes() {
        let config = CudaTracerConfig::default();
        if let Ok(mut wrapper) = CudaTracerWrapper::new(None, config) {
            // Multiple flushes should not panic
            wrapper.flush();
            wrapper.flush();
            wrapper.flush();
        }
    }
}

#[cfg(not(feature = "cuda-tracing"))]
mod cuda_tracing_disabled_tests {
    use super::*;
    use renacer::cuda_tracer::CudaTracerWrapper;

    #[test]
    fn test_cuda_tracer_feature_disabled() {
        let config = CudaTracerConfig::default();
        let result = CudaTracerWrapper::new(None, config);

        assert!(result.is_err());
        if let Err(err) = result {
            let err_msg = err.to_string();
            assert!(err_msg.contains("CUDA tracing support not compiled in"));
            assert!(err_msg.contains("cuda-tracing"));
        }
    }

    #[test]
    fn test_cuda_tracer_debug_format() {
        // Debug format should work even without feature
        let wrapper = CudaTracerWrapper;
        let debug_str = format!("{:?}", wrapper);
        assert!(debug_str.contains("CudaTracerWrapper"));
    }
}

#[test]
fn test_cuda_config_extreme_values() {
    let config = CudaTracerConfig {
        threshold_us: u64::MAX,
        trace_all: false,
        buffer_size: usize::MAX / 2, // Avoid overflow
        device_id: u32::MAX,
    };
    assert_eq!(config.threshold_us, u64::MAX);
    assert_eq!(config.device_id, u32::MAX);
}

#[test]
fn test_cuda_config_zero_values() {
    let config = CudaTracerConfig {
        threshold_us: 0,
        trace_all: false,
        buffer_size: 1024, // Minimum reasonable size
        device_id: 0,
    };
    assert_eq!(config.threshold_us, 0);
    assert_eq!(config.device_id, 0);
}

#[test]
fn test_cuda_config_consistent_with_gpu() {
    // CUDA and GPU configs should have same defaults for threshold
    let cuda_config = CudaTracerConfig::default();
    let gpu_config = renacer::gpu_tracer::GpuTracerConfig::default();

    assert_eq!(cuda_config.threshold_us, gpu_config.threshold_us);
    assert_eq!(cuda_config.trace_all, gpu_config.trace_all);
}
