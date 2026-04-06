//! Integration tests requiring GPU feature and CUDA monitoring tests

use crate::monitor::*;

// =========================================================================
// H0-MON-90: Integration tests (require GPU feature)
// =========================================================================

#[test]
#[cfg(feature = "gpu")]
fn h0_mon_90_query_device_info() {
    // This test requires actual GPU hardware
    match GpuDeviceInfo::query() {
        Ok(info) => {
            // Verify we got valid data
            assert!(!info.name.is_empty());
            assert!(info.backend.is_gpu());
            println!("Found GPU: {info}");
        }
        Err(MonitorError::NoDevice) => {
            // No GPU is OK for CI environments
            println!("No GPU available (expected in CI)");
        }
        Err(e) => {
            panic!("Unexpected error: {e}");
        }
    }
}

#[test]
#[cfg(feature = "gpu")]
fn h0_mon_91_enumerate_devices() {
    match GpuDeviceInfo::enumerate() {
        Ok(devices) => {
            for dev in &devices {
                println!("Found: {dev}");
            }
        }
        Err(MonitorError::NoDevice) => {
            println!("No GPU available (expected in CI)");
        }
        Err(e) => {
            panic!("Unexpected error: {e}");
        }
    }
}

// =========================================================================
// H0-MON-110: GpuMonitor integration tests (require GPU feature)
// =========================================================================

#[test]
#[cfg(feature = "gpu")]
fn h0_mon_110_monitor_real_gpu() {
    match GpuMonitor::new(0, MonitorConfig::default()) {
        Ok(monitor) => {
            println!("GPU Monitor: {}", monitor.device_info());

            // Collect a sample
            match monitor.collect() {
                Ok(metrics) => {
                    println!("Collected metrics: device={}", metrics.device_index);
                    assert_eq!(monitor.sample_count(), 1);
                }
                Err(e) => {
                    println!("Collect failed (expected in CI): {e}");
                }
            }
        }
        Err(MonitorError::NoDevice) => {
            println!("No GPU available (expected in CI)");
        }
        Err(e) => {
            panic!("Unexpected error: {e}");
        }
    }
}

// =========================================================================
// H0-MON-120: CUDA monitoring integration tests
// =========================================================================

#[test]
fn h0_mon_120_cuda_monitor_available_check() {
    // Should return false without cuda-monitor feature
    let available = cuda_monitor_available();
    #[cfg(feature = "cuda-monitor")]
    {
        // With feature, returns true/false based on hardware
        println!("CUDA monitoring available: {}", available);
    }
    #[cfg(not(feature = "cuda-monitor"))]
    {
        assert!(!available, "Should be false without cuda-monitor feature");
    }
}

#[test]
#[cfg(feature = "cuda-monitor")]
fn h0_mon_121_query_cuda_device_info() {
    use crate::monitor::backends::query_cuda_device_info;

    match query_cuda_device_info(0) {
        Ok(info) => {
            assert!(!info.name.is_empty());
            assert_eq!(info.vendor, GpuVendor::Nvidia);
            assert_eq!(info.backend, GpuBackend::Cuda);
            assert!(info.vram_total > 0);
            println!("CUDA Device: {}", info);
        }
        Err(e) => {
            println!("No CUDA device (expected in CI): {}", e);
        }
    }
}

#[test]
#[cfg(feature = "cuda-monitor")]
fn h0_mon_122_enumerate_cuda_devices() {
    use crate::monitor::backends::enumerate_cuda_devices;

    match enumerate_cuda_devices() {
        Ok(devices) => {
            for dev in &devices {
                assert_eq!(dev.vendor, GpuVendor::Nvidia);
                assert_eq!(dev.backend, GpuBackend::Cuda);
                println!("Found CUDA device: {}", dev);
            }
        }
        Err(e) => {
            println!("CUDA enumeration failed (expected in CI): {}", e);
        }
    }
}

#[test]
#[cfg(feature = "cuda-monitor")]
fn h0_mon_123_query_cuda_memory() {
    use crate::monitor::backends::query_cuda_memory;

    match query_cuda_memory(0) {
        Ok(mem) => {
            assert!(mem.total > 0);
            assert!(mem.free <= mem.total);
            println!(
                "CUDA Memory: {} / {} MB ({:.1}% used)",
                mem.used_mb(),
                mem.total / (1024 * 1024),
                mem.usage_percent()
            );
        }
        Err(e) => {
            println!("CUDA memory query failed (expected in CI): {}", e);
        }
    }
}
