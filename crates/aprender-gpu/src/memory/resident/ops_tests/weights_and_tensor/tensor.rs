//! PMAT-018: mod.rs GpuResidentTensor method coverage tests
//!
//! Tests for transfer aliases, kernel launches, is_empty, size_bytes,
//! as_ptr, is_device_resident, buffer/buffer_mut, from_buffer_internal,
//! peek vs to_host, new_uninit, TransferStats, and kernel cache stats.

use crate::driver::CudaContext;
use crate::memory::resident::{reset_transfer_counters, GpuResidentTensor};

/// Helper to create CUDA context, skipping test if unavailable
macro_rules! cuda_ctx {
    () => {
        match CudaContext::new(0) {
            Ok(ctx) => ctx,
            Err(e) => {
                eprintln!("Skipping CUDA test: {:?}", e);
                return;
            }
        }
    };
}

// ============================================================================
// Transfer Alias Tests
// ============================================================================

#[test]
fn test_gpu_resident_tensor_transfer_aliases() {
    use crate::memory::resident::clear_kernel_cache;
    clear_kernel_cache();
    let ctx = cuda_ctx!();
    reset_transfer_counters();

    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut tensor = GpuResidentTensor::from_host(&ctx, &data).unwrap();

    // Test alias methods: host_to_device_transfers() == h2d_transfers()
    assert_eq!(tensor.host_to_device_transfers(), tensor.h2d_transfers());
    assert_eq!(tensor.host_to_device_transfers(), 1);

    // Initially no D2H transfers
    assert_eq!(tensor.device_to_host_transfers(), tensor.d2h_transfers());
    assert_eq!(tensor.device_to_host_transfers(), 0);

    // After to_host(), D2H counter increments
    let _ = tensor.to_host().unwrap();
    assert_eq!(tensor.device_to_host_transfers(), 1);
    assert_eq!(tensor.device_to_host_transfers(), tensor.d2h_transfers());
}

// ============================================================================
// Kernel Launch Tests
// ============================================================================

#[test]
fn test_gpu_resident_tensor_record_kernel_launch() {
    use crate::memory::resident::clear_kernel_cache;
    clear_kernel_cache();
    let ctx = cuda_ctx!();

    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let mut tensor = GpuResidentTensor::from_host(&ctx, &data).unwrap();

    // Initial kernel launches should be 0
    assert_eq!(tensor.kernel_launches(), 0);

    // Call record_kernel_launch() multiple times
    tensor.record_kernel_launch();
    assert_eq!(tensor.kernel_launches(), 1);

    tensor.record_kernel_launch();
    tensor.record_kernel_launch();
    assert_eq!(tensor.kernel_launches(), 3);
}

// ============================================================================
// Tensor Property Tests
// ============================================================================

#[test]
fn test_gpu_resident_tensor_is_empty() {
    use crate::memory::resident::clear_kernel_cache;
    clear_kernel_cache();
    let ctx = cuda_ctx!();

    // Non-empty tensor
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let tensor = GpuResidentTensor::from_host(&ctx, &data).unwrap();
    assert!(!tensor.is_empty());
    assert_eq!(tensor.len(), 4);

    // Empty tensor via new_uninit with 0 length
    let empty_tensor: GpuResidentTensor<f32> = GpuResidentTensor::new_uninit(&ctx, 0).unwrap();
    assert!(empty_tensor.is_empty());
    assert_eq!(empty_tensor.len(), 0);
}

#[test]
fn test_gpu_resident_tensor_size_bytes() {
    use crate::memory::resident::clear_kernel_cache;
    clear_kernel_cache();
    let ctx = cuda_ctx!();

    let data = vec![1.0f32; 100];
    let tensor = GpuResidentTensor::from_host(&ctx, &data).unwrap();

    // 100 f32s = 100 * 4 = 400 bytes
    assert_eq!(tensor.size_bytes(), 400);
}

#[test]
fn test_gpu_resident_tensor_as_ptr() {
    use crate::memory::resident::clear_kernel_cache;
    clear_kernel_cache();
    let ctx = cuda_ctx!();

    let data = vec![1.0f32; 16];
    let tensor = GpuResidentTensor::from_host(&ctx, &data).unwrap();

    // as_ptr() should return non-zero device pointer
    let ptr = tensor.as_ptr();
    assert!(ptr != 0, "Device pointer should be non-zero");
}

#[test]
fn test_gpu_resident_tensor_is_device_resident() {
    use crate::memory::resident::clear_kernel_cache;
    clear_kernel_cache();
    let ctx = cuda_ctx!();

    let data = vec![1.0f32; 8];
    let tensor = GpuResidentTensor::from_host(&ctx, &data).unwrap();

    // Tensor should be device resident
    assert!(tensor.is_device_resident());
}

// ============================================================================
// Buffer Access Tests
// ============================================================================

#[test]
fn test_gpu_resident_tensor_buffer_and_buffer_mut() {
    use crate::memory::resident::clear_kernel_cache;
    clear_kernel_cache();
    let ctx = cuda_ctx!();

    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let mut tensor = GpuResidentTensor::from_host(&ctx, &data).unwrap();

    // Test buffer() immutable reference
    {
        let buf = tensor.buffer();
        assert_eq!(buf.len(), 4);
        assert_eq!(buf.size_bytes(), 16);
    }

    // Test buffer_mut() mutable reference
    {
        let buf_mut = tensor.buffer_mut();
        assert_eq!(buf_mut.len(), 4);
    }
}

#[test]
fn test_gpu_resident_tensor_from_buffer_internal() {
    use crate::driver::GpuBuffer;
    use crate::memory::resident::clear_kernel_cache;
    clear_kernel_cache();
    let ctx = cuda_ctx!();

    // Create a buffer directly
    let buf = GpuBuffer::<f32>::new(&ctx, 32).unwrap();

    // Create tensor from buffer (internal API used by operations)
    let tensor = GpuResidentTensor::from_buffer_internal(buf, 5);

    // Verify initial state
    assert_eq!(tensor.len(), 32);
    assert_eq!(tensor.h2d_transfers(), 0); // No H2D since created from buffer
    assert_eq!(tensor.d2h_transfers(), 0);
    assert_eq!(tensor.kernel_launches(), 5);
    assert!(tensor.is_device_resident());
}

// ============================================================================
// Peek vs To-Host Tests
// ============================================================================

#[test]
fn test_gpu_resident_tensor_peek_vs_to_host() {
    use crate::memory::resident::clear_kernel_cache;
    clear_kernel_cache();
    let ctx = cuda_ctx!();
    reset_transfer_counters();

    let data = vec![42.0f32; 16];
    let mut tensor = GpuResidentTensor::from_host(&ctx, &data).unwrap();

    // peek_host() should NOT increment counters
    let before_d2h = tensor.d2h_transfers();
    let peeked = tensor.peek_host().unwrap();
    assert_eq!(peeked, data);
    assert_eq!(tensor.d2h_transfers(), before_d2h); // Counter unchanged

    // to_host() SHOULD increment counters
    let result = tensor.to_host().unwrap();
    assert_eq!(result, data);
    assert_eq!(tensor.d2h_transfers(), before_d2h + 1); // Counter incremented
}

// ============================================================================
// New Uninit Tests
// ============================================================================

#[test]
fn test_gpu_resident_tensor_new_uninit_various_sizes() {
    use crate::memory::resident::clear_kernel_cache;
    clear_kernel_cache();
    let ctx = cuda_ctx!();

    // Test various sizes
    for size in [0, 1, 16, 256, 1024, 4096] {
        let tensor: GpuResidentTensor<f32> = GpuResidentTensor::new_uninit(&ctx, size).unwrap();
        assert_eq!(tensor.len(), size);
        assert_eq!(tensor.h2d_transfers(), 0); // No transfer for uninit
        assert_eq!(tensor.d2h_transfers(), 0);
        assert!(tensor.is_device_resident());
    }
}

// ============================================================================
// TransferStats Tests
// ============================================================================

#[test]
fn test_transfer_stats_default() {
    use crate::memory::resident::TransferStats;

    let stats = TransferStats::default();
    assert_eq!(stats.h2d_transfers, 0);
    assert_eq!(stats.d2h_transfers, 0);
    assert_eq!(stats.h2d_bytes, 0);
    assert_eq!(stats.d2h_bytes, 0);
    assert_eq!(stats.total_transfers(), 0);
    assert_eq!(stats.total_bytes(), 0);
}

#[test]
fn test_transfer_stats_clone() {
    use crate::memory::resident::TransferStats;

    let stats = TransferStats {
        h2d_transfers: 10,
        d2h_transfers: 5,
        h2d_bytes: 1000,
        d2h_bytes: 500,
    };

    let cloned = stats.clone();
    assert_eq!(cloned.h2d_transfers, 10);
    assert_eq!(cloned.d2h_transfers, 5);
    assert_eq!(cloned.h2d_bytes, 1000);
    assert_eq!(cloned.d2h_bytes, 500);
}

#[test]
fn test_transfer_stats_debug() {
    use crate::memory::resident::TransferStats;

    let stats = TransferStats {
        h2d_transfers: 100,
        d2h_transfers: 50,
        h2d_bytes: 10240,
        d2h_bytes: 5120,
    };

    let debug_str = format!("{:?}", stats);
    assert!(debug_str.contains("TransferStats"));
    assert!(debug_str.contains("100"));
    assert!(debug_str.contains("50"));
}

// ============================================================================
// Kernel Cache Stats Tests
// ============================================================================

/// GPU-ORD-2: this observation spans two kernel launches and is only true if
/// nothing empties the shared module map in between. Holding exclusivity for
/// the whole body makes that so — before, a neighbouring test's
/// `clear_kernel_cache()` landed between the two `gelu` calls and the second
/// one recompiled, printing `[KERNEL-CACHE] Compiling: gelu:16` twice.
#[test]
fn test_kernel_cache_stats_after_operations() {
    use crate::memory::resident::{
        kernel_cache_exclusive, kernel_cache_hits, kernel_cache_misses, reset_kernel_cache_stats,
    };
    let exclusive = kernel_cache_exclusive();
    exclusive.clear();
    let ctx = cuda_ctx!();

    reset_kernel_cache_stats();

    // First operation should be a cache miss
    let data = vec![1.0f32; 16];
    let tensor = GpuResidentTensor::from_host(&ctx, &data).unwrap();
    let _ = tensor.gelu(&ctx).unwrap();

    let first_misses = kernel_cache_misses();
    assert!(first_misses >= 1, "Should have at least 1 cache miss");

    // Same operation again should be a cache hit
    let tensor2 = GpuResidentTensor::from_host(&ctx, &data).unwrap();
    let _ = tensor2.gelu(&ctx).unwrap();

    let hits = kernel_cache_hits();
    assert!(
        hits >= 1,
        "Should have at least 1 cache hit on repeated operation"
    );
}
