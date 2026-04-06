use super::*;
use crate::memory::resident::stats::{record_d2h_transfer, record_h2d_transfer};

// =========================================================================
// GpuResidentTensor Lifecycle Tests (Titan Duel Strategy - PMAT-018)
// =========================================================================

/// Test GpuResidentTensor lifecycle: allocate, write, read, drop
///
/// This test verifies the complete lifecycle path to ensure coverage
/// of allocation, transfer tracking, and deallocation paths.
#[cfg(feature = "cuda")]
#[test]
fn test_gpu_resident_tensor_lifecycle() {
    use crate::driver::CudaContext;

    // Skip gracefully if no CUDA context available
    let ctx = match CudaContext::new(0) {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("Skipping CUDA lifecycle test: {:?}", e);
            return;
        }
    };

    // Reset counters for clean test
    reset_transfer_counters();

    // 1. Create tensor from host data (1 H2D transfer)
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut tensor =
        GpuResidentTensor::from_host(&ctx, &data).expect("Failed to create GpuResidentTensor");

    // Verify initial state
    assert!(tensor.is_device_resident());
    assert_eq!(tensor.len(), 8);
    assert_eq!(tensor.h2d_transfers(), 1);
    assert_eq!(tensor.d2h_transfers(), 0);
    assert_eq!(tensor.kernel_launches(), 0);

    // 2. Verify global transfer counters
    assert_eq!(total_h2d_transfers(), 1);
    assert_eq!(total_d2h_transfers(), 0);
    assert_eq!(total_h2d_bytes(), 32); // 8 * sizeof(f32) = 32

    // 3. Read data back (1 D2H transfer)
    let result = tensor.to_host().expect("Failed to read from GPU");
    assert_eq!(result, data);
    assert_eq!(tensor.d2h_transfers(), 1);
    assert_eq!(total_d2h_transfers(), 1);
    assert_eq!(total_d2h_bytes(), 32);

    // 4. Tensor drops automatically at end of scope
    // This tests the Drop implementation for GpuBuffer
}

/// Test new_uninit path for output buffers
#[cfg(feature = "cuda")]
#[test]
fn test_gpu_resident_tensor_uninit() {
    use crate::driver::CudaContext;

    let ctx = match CudaContext::new(0) {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("Skipping CUDA uninit test: {:?}", e);
            return;
        }
    };

    reset_transfer_counters();

    // Create uninitialized tensor (no transfer)
    let tensor: GpuResidentTensor<f32> =
        GpuResidentTensor::new_uninit(&ctx, 16).expect("Failed to create uninit GpuResidentTensor");

    // No transfers for uninitialized buffer
    assert_eq!(tensor.h2d_transfers(), 0);
    assert_eq!(tensor.d2h_transfers(), 0);
    assert!(tensor.is_device_resident());
    assert_eq!(tensor.len(), 16);
    assert_eq!(tensor.size_bytes(), 64); // 16 * sizeof(f32)

    // Global counters unchanged
    assert_eq!(total_h2d_transfers(), 0);
    assert_eq!(total_d2h_transfers(), 0);
}

/// Test peek_host doesn't affect transfer counters
#[cfg(feature = "cuda")]
#[test]
fn test_gpu_resident_tensor_peek() {
    use crate::driver::CudaContext;

    let ctx = match CudaContext::new(0) {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("Skipping CUDA peek test: {:?}", e);
            return;
        }
    };

    reset_transfer_counters();

    let data = vec![42.0f32; 4];
    let tensor =
        GpuResidentTensor::from_host(&ctx, &data).expect("Failed to create GpuResidentTensor");

    // Initial state: 1 H2D, 0 D2H
    let before_h2d = total_h2d_transfers();
    let before_d2h = total_d2h_transfers();

    // Peek doesn't update counters
    let peeked = tensor.peek_host().expect("Failed to peek");
    assert_eq!(peeked, data);

    // Counters unchanged after peek
    assert_eq!(total_h2d_transfers(), before_h2d);
    assert_eq!(total_d2h_transfers(), before_d2h);
    assert_eq!(tensor.d2h_transfers(), 0); // Instance counter also unchanged
}

/// Test buffer accessor methods
#[cfg(feature = "cuda")]
#[test]
fn test_gpu_resident_tensor_buffer_access() {
    use crate::driver::CudaContext;

    let ctx = match CudaContext::new(0) {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("Skipping CUDA buffer access test: {:?}", e);
            return;
        }
    };

    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let mut tensor =
        GpuResidentTensor::from_host(&ctx, &data).expect("Failed to create GpuResidentTensor");

    // Test immutable buffer access
    let buf = tensor.buffer();
    assert_eq!(buf.len(), 4);

    // Test mutable buffer access
    let buf_mut = tensor.buffer_mut();
    assert_eq!(buf_mut.len(), 4);
}

// =========================================================================
// Original Transfer Stats Tests
// =========================================================================

#[test]
fn test_transfer_stats_capture_and_delta() {
    reset_transfer_counters();

    let before = TransferStats::capture();
    assert_eq!(before.total_transfers(), 0);

    // Simulate some transfers using the record functions
    record_h2d_transfer(1024);
    record_h2d_transfer(2048);
    record_h2d_transfer(512);
    record_d2h_transfer(512);

    let after = TransferStats::capture();
    let delta = after.delta_from(&before);

    assert_eq!(delta.h2d_transfers, 3);
    assert_eq!(delta.d2h_transfers, 1);
    assert_eq!(delta.h2d_bytes, 3584); // 1024 + 2048 + 512
    assert_eq!(delta.d2h_bytes, 512);
    assert_eq!(delta.total_transfers(), 4);
    assert_eq!(delta.total_bytes(), 4096);
}

#[test]
fn test_transfer_stats_display() {
    let stats = TransferStats {
        h2d_transfers: 5,
        d2h_transfers: 2,
        h2d_bytes: 1024 * 1024 * 10, // 10 MB
        d2h_bytes: 1024 * 1024 * 5,  // 5 MB
    };

    let display = format!("{}", stats);
    assert!(display.contains("H2D: 5"));
    assert!(display.contains("D2H: 2"));
    assert!(display.contains("10.00 MB"));
    assert!(display.contains("5.00 MB"));
}

#[test]
fn test_reset_counters() {
    record_h2d_transfer(100);
    record_d2h_transfer(50);

    reset_transfer_counters();

    assert_eq!(total_h2d_transfers(), 0);
    assert_eq!(total_d2h_transfers(), 0);
    assert_eq!(total_h2d_bytes(), 0);
    assert_eq!(total_d2h_bytes(), 0);
}

// =========================================================================
// GPU Memory Pressure Test (PMAT-018: Coverage Killer Remediation)
// =========================================================================

/// Test GPU behavior under memory pressure
///
/// This test exercises the allocation failure path by:
/// 1. Allocating tensors until memory is exhausted
/// 2. Verifying that allocation failures are graceful (no panic)
/// 3. Demonstrating that memory is reclaimed after dropping tensors
///
/// Note: This test does NOT assert automatic eviction since no eviction
/// policy is currently implemented. It tests graceful degradation.
#[cfg(feature = "cuda")]
#[test]
fn test_gpu_allocation_under_pressure() {
    use crate::driver::CudaContext;

    let ctx = match CudaContext::new(0) {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("Skipping GPU pressure test: {:?}", e);
            return;
        }
    };

    reset_transfer_counters();

    // Allocate 64MB chunks until we hit an allocation failure
    const CHUNK_SIZE: usize = 64 * 1024 * 1024 / 4; // 64MB in f32s
    const MAX_CHUNKS: usize = 1024; // Safety limit (64GB max)

    let mut tensors: Vec<GpuResidentTensor<f32>> = Vec::new();
    let mut allocation_count = 0;
    let mut hit_limit = false;

    // Phase 1: Allocate until we hit memory limit
    for _ in 0..MAX_CHUNKS {
        let data = vec![0.0f32; CHUNK_SIZE];
        match GpuResidentTensor::from_host(&ctx, &data) {
            Ok(tensor) => {
                tensors.push(tensor);
                allocation_count += 1;
            }
            Err(_) => {
                // Expected: CUDA_ERROR_OUT_OF_MEMORY or similar
                hit_limit = true;
                break;
            }
        }
    }

    // We should have allocated at least one tensor
    assert!(allocation_count > 0, "Should have allocated at least one tensor");

    // Record how many we allocated before hitting the limit
    let tensors_at_limit = tensors.len();
    eprintln!(
        "GPU pressure test: Allocated {} tensors ({} MB) before limit",
        tensors_at_limit,
        tensors_at_limit * 64
    );

    // Phase 2: Free half the tensors
    let drop_count = tensors.len() / 2;
    for _ in 0..drop_count {
        tensors.pop();
    }

    // Phase 3: Verify we can allocate again after freeing
    let data = vec![0.0f32; CHUNK_SIZE];
    let recovery_result = GpuResidentTensor::from_host(&ctx, &data);

    // If we hit the limit, we should be able to recover after freeing
    if hit_limit {
        assert!(recovery_result.is_ok(), "Should be able to allocate after freeing tensors");
    }

    // Verify transfer tracking still works under pressure
    let total_transfers = total_h2d_transfers();
    assert!(
        total_transfers >= allocation_count as u64,
        "Transfer counter should track all allocations"
    );
}

/// Test MemoryPool behavior under pressure (CPU-side simulation)
///
/// This tests the MemoryPool allocator's behavior when full,
/// verifying that allocation failures are properly reported.
#[test]
fn test_memory_pool_exhaustion() {
    use crate::memory::pool::{MemoryPool, PoolConfig};

    // Create a tiny pool (1MB with 64KB pages = 16 pages)
    let config = PoolConfig { total_bytes: 1024 * 1024, page_size: 64 * 1024 };
    let mut pool = MemoryPool::new(config);

    // Allocate all pages
    let mut allocations = Vec::new();
    for _ in 0..16 {
        if let Some(id) = pool.allocate(64 * 1024) {
            allocations.push(id);
        }
    }

    // Pool should now be full
    let stats = pool.stats();
    assert_eq!(stats.free_pages, 0, "Pool should be completely full");

    // Next allocation should fail
    let failed_alloc = pool.allocate(64 * 1024);
    assert!(failed_alloc.is_none(), "Allocation should fail when pool is exhausted");

    // Free one allocation
    if let Some(id) = allocations.pop() {
        assert!(pool.free(id), "Free should succeed");
    }

    // Now allocation should succeed
    let recovered_alloc = pool.allocate(64 * 1024);
    assert!(recovered_alloc.is_some(), "Allocation should succeed after freeing");
}
