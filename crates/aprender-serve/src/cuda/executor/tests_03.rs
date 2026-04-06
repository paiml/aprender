//! CudaExecutor tests Part 03 - COV-007 through COV-012
//!
//! Coverage tests for:
//! - COV-007: activations.rs coverage (silu_gpu, gelu_async, elementwise_mul, swiglu)
//! - COV-008: workspace.rs coverage (init, batch, clear, buffer stats)
//! - COV-009: gemm.rs coverage (optimized, fused, tiled)
//! - COV-010: core.rs coverage (synchronize, device info, profiling)
//! - COV-011: layer.rs additional coverage (transformer_layer, indexed weights)
//! - COV-012: Additional quantized.rs batched operations

use super::*;
use serial_test::serial;

// ============================================================================
// COV-008: workspace.rs coverage tests
// Target: Increase coverage from 9.73% to 50%+
// Focus: init_workspace, init_batched_workspace, has_workspace,
//        workspace_batch_size, has_decode_graph, clear_workspace,
//        clear_decode_graph, gemv_buffer_stats, clear_gemv_buffers
// ============================================================================

#[test]
#[serial]
fn test_cov008_init_workspace_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Set KV cache params first (required by init_workspace)
    let _ = executor.init_kv_cache_gpu(1, 4, 4, 8, 16);

    let hidden_dim = 64usize;
    let intermediate_dim = 128usize;

    let result = executor.init_workspace(hidden_dim, intermediate_dim);
    assert!(result.is_ok(), "init_workspace failed: {:?}", result.err());

    assert!(executor.has_workspace(), "Workspace should be initialized");
    assert_eq!(
        executor.workspace_batch_size(),
        1,
        "Default batch size should be 1"
    );
}

#[test]
#[serial]
fn test_cov008_init_workspace_large() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Set KV cache params first
    let _ = executor.init_kv_cache_gpu(1, 8, 8, 64, 512);

    let hidden_dim = 512usize;
    let intermediate_dim = 2048usize;

    let result = executor.init_workspace(hidden_dim, intermediate_dim);
    assert!(
        result.is_ok(),
        "init_workspace large failed: {:?}",
        result.err()
    );

    assert!(executor.has_workspace());
}

#[test]
#[serial]
fn test_cov008_init_batched_workspace_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Set KV cache params first
    let _ = executor.init_kv_cache_gpu(1, 4, 4, 8, 16);

    let hidden_dim = 64usize;
    let intermediate_dim = 128usize;
    let batch_size = 4usize;

    let result = executor.init_batched_workspace(hidden_dim, intermediate_dim, batch_size);
    assert!(
        result.is_ok(),
        "init_batched_workspace failed: {:?}",
        result.err()
    );

    assert!(executor.has_workspace());
    assert_eq!(executor.workspace_batch_size(), 4, "Batch size should be 4");
}

#[test]
#[serial]
fn test_cov008_init_batched_workspace_max_batch() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Set KV cache params first
    let _ = executor.init_kv_cache_gpu(1, 4, 4, 8, 16);

    // Test maximum batch size (32)
    let result = executor.init_batched_workspace(64, 128, 32);
    assert!(
        result.is_ok(),
        "init_batched_workspace max batch failed: {:?}",
        result.err()
    );
    assert_eq!(executor.workspace_batch_size(), 32);
}

#[test]
#[serial]
fn test_cov008_init_batched_workspace_zero_batch_error() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Set KV cache params first
    let _ = executor.init_kv_cache_gpu(1, 4, 4, 8, 16);

    // Test zero batch size (should fail)
    let result = executor.init_batched_workspace(64, 128, 0);
    assert!(
        result.is_err(),
        "init_batched_workspace with batch=0 should fail"
    );
}

#[test]
#[serial]
fn test_cov008_init_batched_workspace_too_large_batch_error() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Set KV cache params first
    let _ = executor.init_kv_cache_gpu(1, 4, 4, 8, 16);

    // Test batch size > 32 (should fail)
    let result = executor.init_batched_workspace(64, 128, 33);
    assert!(
        result.is_err(),
        "init_batched_workspace with batch=33 should fail"
    );
}

#[test]
#[serial]
fn test_cov008_has_workspace_false_initially() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    assert!(
        !executor.has_workspace(),
        "Workspace should not be initialized initially"
    );
}

#[test]
#[serial]
fn test_cov008_has_decode_graph_false_initially() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    assert!(
        !executor.has_decode_graph(),
        "Decode graph should not exist initially"
    );
}

#[test]
#[serial]
fn test_cov008_clear_workspace() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Set KV cache and init workspace
    let _ = executor.init_kv_cache_gpu(1, 4, 4, 8, 16);
    let _ = executor.init_workspace(64, 128);
    assert!(executor.has_workspace());

    // Clear workspace
    executor.clear_workspace();
    assert!(!executor.has_workspace(), "Workspace should be cleared");
}

#[test]
#[serial]
fn test_cov008_clear_decode_graph() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Clear decode graph (even without capturing one)
    executor.clear_decode_graph();
    assert!(
        !executor.has_decode_graph(),
        "Decode graph should be cleared"
    );
}

#[test]
#[serial]
fn test_cov008_gemv_buffer_stats_initial() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    let (input_bytes, output_bytes) = executor.gemv_buffer_stats();
    assert_eq!(
        input_bytes, 0,
        "Initial GEMV input buffer should be 0 bytes"
    );
    assert_eq!(
        output_bytes, 0,
        "Initial GEMV output buffer should be 0 bytes"
    );
}

#[test]
#[serial]
fn test_cov008_clear_gemv_buffers() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Clear GEMV buffers (even without allocating any)
    executor.clear_gemv_buffers();
    let (input_bytes, output_bytes) = executor.gemv_buffer_stats();
    assert_eq!(input_bytes, 0);
    assert_eq!(output_bytes, 0);
}

#[test]
#[serial]
fn test_cov008_ensure_gemv_input_buffer() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Ensure GEMV input buffer
    let result = executor.ensure_gemv_input_buffer(256);
    assert!(
        result.is_ok(),
        "ensure_gemv_input_buffer failed: {:?}",
        result.err()
    );

    let (input_bytes, _) = executor.gemv_buffer_stats();
    assert_eq!(
        input_bytes,
        256 * 4,
        "GEMV input buffer should be 1024 bytes (256 * 4)"
    );
}

#[test]
#[serial]
fn test_cov008_ensure_gemv_output_buffer() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Ensure GEMV output buffer
    let result = executor.ensure_gemv_output_buffer(128);
    assert!(
        result.is_ok(),
        "ensure_gemv_output_buffer failed: {:?}",
        result.err()
    );

    let (_, output_bytes) = executor.gemv_buffer_stats();
    assert_eq!(
        output_bytes,
        128 * 4,
        "GEMV output buffer should be 512 bytes (128 * 4)"
    );
}

#[test]
#[serial]
fn test_cov008_ensure_gemv_buffers_reuse() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // First allocation
    let ptr1 = executor.ensure_gemv_input_buffer(256).expect("first alloc");

    // Same size - should reuse
    let ptr2 = executor
        .ensure_gemv_input_buffer(256)
        .expect("second alloc");
    assert_eq!(ptr1, ptr2, "Same size should reuse buffer");

    // Different size - should reallocate
    let ptr3 = executor.ensure_gemv_input_buffer(512).expect("third alloc");
    assert_ne!(ptr1, ptr3, "Different size should create new buffer");
}

#[test]
#[serial]
fn test_cov008_copy_gemv_buffers() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 64usize;
    let input: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let mut output = vec![0.0f32; n];

    // Ensure both buffers
    executor.ensure_gemv_input_buffer(n).expect("ensure input");
    executor
        .ensure_gemv_output_buffer(n)
        .expect("ensure output");

    // Copy to input buffer
    let result = executor.copy_to_gemv_input(&input);
    assert!(
        result.is_ok(),
        "copy_to_gemv_input failed: {:?}",
        result.err()
    );

    // Copy from output buffer (note: output buffer won't have the input data,
    // this just tests the copy path works)
    let result = executor.copy_from_gemv_output(&mut output);
    assert!(
        result.is_ok(),
        "copy_from_gemv_output failed: {:?}",
        result.err()
    );
}

// ============================================================================
// COV-009: gemm.rs coverage tests
// Target: Increase coverage from 60.92% to 75%+
// Focus: synchronize_compute, synchronize_transfer, synchronize_all,
//        allocate_buffer, softmax, gemm
// ============================================================================

#[test]
#[serial]
fn test_cov009_synchronize_compute() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    let result = executor.synchronize_compute();
    assert!(
        result.is_ok(),
        "synchronize_compute failed: {:?}",
        result.err()
    );
}

#[test]
#[serial]
fn test_cov009_synchronize_transfer() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    let result = executor.synchronize_transfer();
    assert!(
        result.is_ok(),
        "synchronize_transfer failed: {:?}",
        result.err()
    );
}

#[test]
#[serial]
fn test_cov009_synchronize_all() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    let result = executor.synchronize_all();
    assert!(result.is_ok(), "synchronize_all failed: {:?}", result.err());
}

#[test]
#[serial]
fn test_cov009_allocate_buffer_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    let result = executor.allocate_buffer(256);
    assert!(result.is_ok(), "allocate_buffer failed: {:?}", result.err());

    let buffer = result.unwrap();
    assert!(buffer.len() == 256, "Buffer should have 256 elements");
}

#[test]
#[serial]
fn test_cov009_allocate_buffer_large() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    // Allocate 1MB buffer (262144 f32 elements)
    let result = executor.allocate_buffer(262144);
    assert!(
        result.is_ok(),
        "allocate_buffer large failed: {:?}",
        result.err()
    );
}

include!("tests_cov009_softmax.rs");
include!("tests_cov010_memory_context.rs");
include!("tests_cov012_batched.rs");
include!("tests_cov015_has.rs");
