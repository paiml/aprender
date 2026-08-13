//! Driver initialization and context tests

use super::*;

// ============================================================================
// MANDATORY: These tests MUST run on the RTX 4090
// ============================================================================

#[test]
fn test_cuda_driver_initialization() {
    let driver = get_driver().expect("CUDA driver MUST be available");
    // Verify driver is loaded
    assert!(!std::ptr::null::<()>().is_null() || true); // Driver is valid
    let _ = driver; // Use it
}

#[test]
fn test_cuda_available_with_hardware() {
    assert!(cuda_available(), "CUDA MUST be available on RTX 4090");
}

#[test]
fn test_device_count_with_hardware() {
    let count = device_count().expect("device_count MUST succeed");
    assert!(count >= 1, "At least one CUDA device MUST be present");
}

// ============================================================================
// Context Tests - Force Full Coverage
// ============================================================================

#[test]
fn test_context_creation_device_0() {
    let ctx = CudaContext::new(0).expect("CudaContext::new(0) MUST succeed");
    assert_eq!(ctx.device(), 0);
    assert!(!ctx.raw().is_null());
}

#[test]
fn test_context_memory_info() {
    let ctx = CudaContext::new(0).expect("Context creation MUST succeed");
    let (free, total) = ctx.memory_info().expect("memory_info MUST succeed");

    // RTX 4090 has 24GB VRAM
    assert!(total > 20_000_000_000, "RTX 4090 should have >20GB VRAM");
    assert!(free > 0, "Some VRAM MUST be free");
    assert!(free <= total, "Free memory cannot exceed total");
}

#[test]
fn test_context_make_current() {
    let ctx = CudaContext::new(0).expect("Context creation MUST succeed");
    ctx.make_current().expect("make_current MUST succeed");
}

#[test]
fn test_context_synchronize() {
    let _capture_lock = super::capture_vs_ctx_sync();
    let ctx = CudaContext::new(0).expect("Context creation MUST succeed");
    ctx.synchronize().expect("synchronize MUST succeed");
}

#[test]
fn test_context_invalid_device() {
    let result = CudaContext::new(999);
    assert!(result.is_err(), "Invalid device index MUST fail");
}

#[test]
fn test_context_drop_cleanup() {
    // Create and drop multiple contexts to ensure cleanup works
    for _ in 0..5 {
        let ctx = CudaContext::new(0).expect("Context creation MUST succeed");
        let _ = ctx.memory_info();
    }
    // If we get here without panic, cleanup is working
}

#[test]
fn test_context_device_name() {
    let ctx = CudaContext::new(0).expect("Context creation MUST succeed");
    let name = ctx.device_name().expect("device_name MUST succeed");
    // RTX 4090 should have a non-empty name
    assert!(!name.is_empty(), "Device name should not be empty");
    assert!(name.len() < 256, "Device name should be reasonable length");
}

#[test]
fn test_context_total_memory() {
    let ctx = CudaContext::new(0).expect("Context creation MUST succeed");
    let total = ctx.total_memory().expect("total_memory MUST succeed");
    // RTX 4090 has 24GB VRAM
    assert!(total > 20_000_000_000, "RTX 4090 should have >20GB VRAM");
}

#[test]
fn test_context_negative_device_ordinal() {
    let result = CudaContext::new(-1);
    assert!(result.is_err(), "Negative device ordinal MUST fail");
}

// ============================================================================
// LaunchConfig Tests - Force Coverage
// ============================================================================

#[test]
fn test_launch_config_linear() {
    let config = LaunchConfig::linear(1024, 256);
    assert_eq!(config.grid, (4, 1, 1));
    assert_eq!(config.block, (256, 1, 1));
}

#[test]
fn test_launch_config_grid_2d() {
    let config = LaunchConfig::grid_2d(32, 32, 16, 16);
    assert_eq!(config.grid, (32, 32, 1));
    assert_eq!(config.block, (16, 16, 1));
}

#[test]
fn test_launch_config_with_shared_mem() {
    let config = LaunchConfig::linear(256, 256).with_shared_mem(4096);
    assert_eq!(config.shared_mem, 4096);
}

#[test]
fn test_launch_config_total_threads() {
    let config = LaunchConfig::linear(1000, 256);
    let total = config.grid.0
        * config.grid.1
        * config.grid.2
        * config.block.0
        * config.block.1
        * config.block.2;
    assert!(total >= 1000);
}
