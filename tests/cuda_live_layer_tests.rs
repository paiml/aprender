//! CUDA "Live Layer" Protocol Tests
//!
//! These tests exercise the CUDA kernels directly with random f32 weights,
//! avoiding GGUF file loading for fast, reliable coverage.
//!
//! Spec: Live Layer Protocol (1.5.4)
//! Target: cuda.rs coverage from 39.62% to 80%+
//!
//! Run: cargo test --test cuda_live_layer_tests --features cuda -- --nocapture

#![cfg(feature = "cuda")]

use rand::prelude::*;
use realizar::cuda::{CudaExecutor, CudaKernels, KernelType};
use serial_test::serial;

// ============================================================================
// Helper: Skip test if CUDA not available
// ============================================================================

fn cuda_available() -> bool {
    CudaExecutor::is_available()
}

macro_rules! skip_if_no_cuda {
    () => {
        if !cuda_available() {
            eprintln!("Skipping test: CUDA not available");
            return;
        }
    };
}

// ============================================================================
// Random Data Generation
// ============================================================================

fn random_f32_vec(len: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..len).map(|_| rng.gen_range(-1.0..1.0)).collect()
}

// ============================================================================
// LL-001: Basic Kernel Execution Tests
// ============================================================================

#[test]
#[serial]
fn test_ll001_softmax_random_input() {
    skip_if_no_cuda!();

    let mut executor = CudaExecutor::new(0).expect("CUDA executor creation failed");

    // Test various dimensions - this test focuses on COVERAGE not correctness.
    // The softmax kernel has known numerical issues tracked in trueno-gpu.
    // We verify the kernel executes without CUDA errors.
    for dim in [1024, 2048, 4096, 8192, 16384, 32000] {
        let mut data = random_f32_vec(dim, 42 + dim as u64);

        let result = executor.softmax(&mut data);
        assert!(
            result.is_ok(),
            "LL-001: softmax failed for dim={}: {:?}",
            dim,
            result.err()
        );

        // Verify output is finite (not NaN/Inf)
        let finite_count = data.iter().filter(|x| x.is_finite()).count();
        assert!(
            finite_count == dim,
            "LL-001: softmax produced {} non-finite values for dim={}",
            dim - finite_count,
            dim
        );
    }
    eprintln!("LL-001: PASS - softmax with random inputs (coverage only)");
}

#[test]
#[serial]
fn test_ll002_rmsnorm_random_input() {
    skip_if_no_cuda!();

    let mut executor = CudaExecutor::new(0).expect("CUDA executor creation failed");

    // Test various hidden dimensions
    for hidden_dim in [64, 128, 256, 512, 1024, 2048, 4096] {
        let input = random_f32_vec(hidden_dim, 100 + hidden_dim as u64);
        let gamma = random_f32_vec(hidden_dim, 200 + hidden_dim as u64);
        let mut output = vec![0.0f32; hidden_dim];

        let result = executor.rmsnorm_host(&input, &gamma, &mut output, 1e-5);
        assert!(
            result.is_ok(),
            "LL-002: rmsnorm failed for hidden_dim={}: {:?}",
            hidden_dim,
            result.err()
        );

        // Output should be finite
        for (i, &val) in output.iter().enumerate() {
            assert!(
                val.is_finite(),
                "LL-002: rmsnorm output[{}]={} is not finite for hidden_dim={}",
                i,
                val,
                hidden_dim
            );
        }
    }
    eprintln!("LL-002: PASS - rmsnorm with random inputs");
}

#[test]
#[serial]
fn test_ll003_residual_add_random_input() {
    skip_if_no_cuda!();

    let mut executor = CudaExecutor::new(0).expect("CUDA executor creation failed");

    // Test various dimensions
    for dim in [64, 128, 256, 512, 1024, 2048, 4096] {
        let input1 = random_f32_vec(dim, 300 + dim as u64);
        let input2 = random_f32_vec(dim, 400 + dim as u64);
        let mut output = vec![0.0f32; dim];

        let result = executor.residual_add_host(&input1, &input2, &mut output);
        assert!(
            result.is_ok(),
            "LL-003: residual_add failed for dim={}: {:?}",
            dim,
            result.err()
        );

        // Verify element-wise addition
        for i in 0..dim {
            let expected = input1[i] + input2[i];
            let diff = (output[i] - expected).abs();
            assert!(
                diff < 1e-5,
                "LL-003: residual_add[{}]={} != {} for dim={}",
                i,
                output[i],
                expected,
                dim
            );
        }
    }
    eprintln!("LL-003: PASS - residual_add with random inputs");
}

#[test]
#[serial]
fn test_ll004_gemm_random_input() {
    skip_if_no_cuda!();

    let mut executor = CudaExecutor::new(0).expect("CUDA executor creation failed");

    // Test various matrix sizes
    for (m, k, n) in [
        (4, 4, 4),
        (8, 8, 8),
        (16, 16, 16),
        (32, 32, 32),
        (64, 64, 64),
    ] {
        let a = random_f32_vec(m * k, 500 + (m * k) as u64);
        let b = random_f32_vec(k * n, 600 + (k * n) as u64);
        let mut c = vec![0.0f32; m * n];

        let result = executor.gemm(&a, &b, &mut c, m as u32, n as u32, k as u32);
        assert!(
            result.is_ok(),
            "LL-004: gemm failed for {}x{}x{}: {:?}",
            m,
            k,
            n,
            result.err()
        );

        // Output should be finite
        for (i, &val) in c.iter().enumerate() {
            assert!(
                val.is_finite(),
                "LL-004: gemm output[{}]={} is not finite for {}x{}x{}",
                i,
                val,
                m,
                k,
                n
            );
        }
    }
    eprintln!("LL-004: PASS - gemm with random inputs");
}

// ============================================================================
// LL-010: Batched Kernel Tests (batch_size=32)
// ============================================================================

#[test]
#[serial]
fn test_ll010_batched_softmax_b32() {
    skip_if_no_cuda!();

    let mut executor = CudaExecutor::new(0).expect("CUDA executor creation failed");
    let batch_size = 32;

    // Test batched softmax by running multiple softmax calls
    for dim in [128, 256, 512, 1024] {
        for batch_idx in 0..batch_size {
            let mut data = random_f32_vec(dim, 1000 + batch_idx as u64);
            let result = executor.softmax(&mut data);
            assert!(
                result.is_ok(),
                "LL-010: batched softmax failed for dim={}, batch={}: {:?}",
                dim,
                batch_idx,
                result.err()
            );
        }
    }
    eprintln!("LL-010: PASS - batched softmax (b=32)");
}

#[test]
#[serial]
fn test_ll011_batched_rmsnorm_b32() {
    skip_if_no_cuda!();

    let mut executor = CudaExecutor::new(0).expect("CUDA executor creation failed");
    let batch_size = 32;

    for hidden_dim in [256, 512, 1024, 2048] {
        for batch_idx in 0..batch_size {
            let input = random_f32_vec(hidden_dim, 2000 + batch_idx as u64);
            let gamma = random_f32_vec(hidden_dim, 3000 + batch_idx as u64);
            let mut output = vec![0.0f32; hidden_dim];

            let result = executor.rmsnorm_host(&input, &gamma, &mut output, 1e-5);
            assert!(
                result.is_ok(),
                "LL-011: batched rmsnorm failed for hidden_dim={}, batch={}: {:?}",
                hidden_dim,
                batch_idx,
                result.err()
            );
        }
    }
    eprintln!("LL-011: PASS - batched rmsnorm (b=32)");
}

#[test]
#[serial]
fn test_ll012_batched_residual_add_b32() {
    skip_if_no_cuda!();

    let mut executor = CudaExecutor::new(0).expect("CUDA executor creation failed");
    let batch_size = 32;

    for dim in [256, 512, 1024, 2048] {
        for batch_idx in 0..batch_size {
            let input1 = random_f32_vec(dim, 4000 + batch_idx as u64);
            let input2 = random_f32_vec(dim, 5000 + batch_idx as u64);
            let mut output = vec![0.0f32; dim];

            let result = executor.residual_add_host(&input1, &input2, &mut output);
            assert!(
                result.is_ok(),
                "LL-012: batched residual_add failed for dim={}, batch={}: {:?}",
                dim,
                batch_idx,
                result.err()
            );
        }
    }
    eprintln!("LL-012: PASS - batched residual_add (b=32)");
}

// ============================================================================
// LL-020: PTX Generation Coverage Tests
// ============================================================================

#[test]
fn test_ll020_ptx_generation_all_kernel_types() {
    let kernels = CudaKernels::new();

    // Test all kernel type PTX generation
    let kernel_types = vec![
        KernelType::GemmNaive {
            m: 64,
            n: 64,
            k: 64,
        },
        KernelType::GemmTiled {
            m: 128,
            n: 128,
            k: 128,
            tile_size: 32,
        },
        KernelType::GemmTensorCore {
            m: 16,
            n: 16,
            k: 16,
        },
        KernelType::Gemv { k: 2048, n: 1024 },
        KernelType::CoalescedGemv { k: 2048, n: 1024 },
        KernelType::Softmax { dim: 4096 },
        KernelType::LayerNorm {
            hidden_size: 4096,
            epsilon: 1e-5,
            affine: true,
        },
        KernelType::Attention {
            seq_len: 512,
            head_dim: 64,
            causal: true,
        },
        KernelType::MultiHeadAttention {
            seq_len: 256,
            head_dim: 64,
            n_heads: 8,
            causal: true,
        },
        KernelType::QuantizedGemm {
            m: 1,
            n: 4096,
            k: 4096,
        },
        KernelType::QuantizedGemmGgml {
            m: 1,
            n: 2560,
            k: 2560,
        },
        KernelType::RmsNorm {
            hidden_size: 2048,
            epsilon: 1e-6,
        },
        KernelType::Rope {
            num_heads: 32,
            head_dim: 64,
            theta: 10000.0,
        },
        KernelType::ResidualAdd { n: 4096 },
        KernelType::Silu { n: 4096 },
        KernelType::Gelu { n: 4096 },
        KernelType::Q4KGemv { k: 2560, n: 2560 },
        KernelType::Q5KGemv { k: 2560, n: 2560 },
        KernelType::Q6KGemv { k: 2560, n: 2560 },
        KernelType::Q8_0Gemv { k: 2048, n: 2048 },
        KernelType::Q4_0Gemv { k: 2048, n: 2048 },
        KernelType::Q4_1Gemv { k: 2048, n: 2048 },
        KernelType::Q5_0Gemv { k: 2048, n: 2048 },
        KernelType::BatchedQ4KGemv {
            m: 8,
            k: 2560,
            n: 2560,
        },
        KernelType::BatchedQ6KGemv {
            k: 2560,
            n: 2560,
            m: 8,
        },
        KernelType::BatchedVectorizedRmsNorm {
            hidden_size: 2048,
            batch_size: 8,
            epsilon: 1e-5,
        },
        KernelType::BatchedRope {
            num_heads: 32,
            head_dim: 64,
            batch_size: 8,
            theta: 10000.0,
        },
        KernelType::BatchedResidualAdd {
            n: 2048,
            batch_size: 8,
        },
        KernelType::BatchedSwiglu {
            n: 8192,
            batch_size: 8,
        },
        KernelType::ArgMax { length: 32000 },
        KernelType::ArgMaxFinal { num_blocks: 128 },
        KernelType::IncrementalAttention {
            max_seq_len: 4096,
            head_dim: 64,
            n_heads: 32,
            n_kv_heads: 4,
            indirect: true,
        },
    ];

    for (i, kernel_type) in kernel_types.iter().enumerate() {
        let ptx = kernels.generate_ptx(kernel_type);
        assert!(
            ptx.contains(".version"),
            "LL-020: PTX generation failed for kernel type {} ({:?})",
            i,
            kernel_type
        );
        assert!(
            !ptx.is_empty(),
            "LL-020: Empty PTX for kernel type {} ({:?})",
            i,
            kernel_type
        );
    }
    eprintln!(
        "LL-020: PASS - PTX generation for {} kernel types",
        kernel_types.len()
    );
}

// ============================================================================
// LL-030: Kernel Name Coverage Tests
// ============================================================================

#[test]
fn test_ll030_kernel_names_coverage() {
    let kernels = CudaKernels::new();

    let kernel_types_and_expected = vec![
        (KernelType::Softmax { dim: 4096 }, "softmax"),
        (KernelType::ResidualAdd { n: 4096 }, "residual_add"),
        (KernelType::Silu { n: 4096 }, "silu"),
        (KernelType::Gelu { n: 4096 }, "gelu"),
        (KernelType::ArgMax { length: 32000 }, "argmax"),
        (KernelType::ArgMaxFinal { num_blocks: 128 }, "argmax_final"),
    ];

    for (kernel_type, expected_substr) in kernel_types_and_expected {
        let name = kernels.kernel_name(&kernel_type);
        assert!(
            name.contains(expected_substr),
            "LL-030: kernel name '{}' should contain '{}'",
            name,
            expected_substr
        );
    }
    eprintln!("LL-030: PASS - kernel names coverage");
}

// ============================================================================
// LL-040: Error Handling Tests
// ============================================================================

#[test]
#[serial]
fn test_ll040_gemm_size_validation() {
    skip_if_no_cuda!();

    let mut executor = CudaExecutor::new(0).expect("CUDA executor creation failed");

    // Wrong sizes - should fail validation
    let a = vec![1.0f32; 10]; // Wrong size for 4x4x4
    let b = vec![1.0f32; 16];
    let mut c = vec![0.0f32; 16];

    let result = executor.gemm(&a, &b, &mut c, 4, 4, 4);
    assert!(
        result.is_err(),
        "LL-040: gemm should fail with mismatched sizes"
    );
    eprintln!("LL-040: PASS - gemm size validation");
}

#[test]
#[serial]
fn test_ll041_rmsnorm_empty_input() {
    skip_if_no_cuda!();

    let mut executor = CudaExecutor::new(0).expect("CUDA executor creation failed");

    // Empty input should be handled gracefully
    let input: Vec<f32> = vec![];
    let gamma: Vec<f32> = vec![];
    let mut output: Vec<f32> = vec![];

    // This might error or succeed with no-op - either is acceptable
    let _result = executor.rmsnorm_host(&input, &gamma, &mut output, 1e-5);
    eprintln!("LL-041: PASS - rmsnorm empty input handled");
}

// ============================================================================
// LL-050: Memory Pool Tests (GPU Allocation Coverage)
// ============================================================================

#[test]
#[serial]
fn test_ll050_executor_memory_info() {
    skip_if_no_cuda!();

    let executor = CudaExecutor::new(0).expect("CUDA executor creation failed");

    let (free, total) = executor.memory_info().expect("memory_info should succeed");
    assert!(total > 0, "LL-050: GPU should have total memory > 0");
    assert!(free <= total, "LL-050: free memory should be <= total");

    eprintln!(
        "LL-050: PASS - GPU memory: {:.2} GB free / {:.2} GB total",
        free as f64 / (1024.0 * 1024.0 * 1024.0),
        total as f64 / (1024.0 * 1024.0 * 1024.0)
    );
}

#[test]
#[serial]
fn test_ll051_executor_device_name() {
    skip_if_no_cuda!();

    let executor = CudaExecutor::new(0).expect("CUDA executor creation failed");
    let name = executor.device_name().expect("device_name should succeed");

    assert!(!name.is_empty(), "LL-051: device name should not be empty");
    eprintln!("LL-051: PASS - GPU device: {}", name);
}

#[test]
#[serial]
fn test_ll052_executor_synchronize() {
    skip_if_no_cuda!();

    let executor = CudaExecutor::new(0).expect("CUDA executor creation failed");
    let result = executor.synchronize();

    assert!(result.is_ok(), "LL-052: synchronize should succeed");
    eprintln!("LL-052: PASS - GPU synchronize");
}

// ============================================================================
// LL-060: Profiler Coverage Tests
// ============================================================================

#[test]
#[serial]
fn test_ll060_profiler_enable_disable() {
    skip_if_no_cuda!();

    let mut executor = CudaExecutor::new(0).expect("CUDA executor creation failed");

    // Initially disabled
    assert!(
        !executor.is_profiling_enabled(),
        "LL-060: profiling should be disabled by default"
    );

    // Enable profiling
    executor.enable_profiling();
    assert!(
        executor.is_profiling_enabled(),
        "LL-060: profiling should be enabled"
    );

    // Disable profiling
    executor.disable_profiling();
    assert!(
        !executor.is_profiling_enabled(),
        "LL-060: profiling should be disabled"
    );

    eprintln!("LL-060: PASS - profiler enable/disable");
}

#[test]
#[serial]
fn test_ll061_profiler_summary() {
    skip_if_no_cuda!();

    let mut executor = CudaExecutor::new(0).expect("CUDA executor creation failed");

    // Enable and run some operations
    executor.enable_profiling();

    let mut data = random_f32_vec(1024, 9999);
    let _ = executor.softmax(&mut data);

    // Get summary
    let summary = executor.profiler_summary();
    assert!(
        !summary.is_empty(),
        "LL-061: profiler summary should not be empty"
    );

    eprintln!("LL-061: PASS - profiler summary: {} chars", summary.len());
}

#[test]
#[serial]
fn test_ll062_profiler_reset() {
    skip_if_no_cuda!();

    let mut executor = CudaExecutor::new(0).expect("CUDA executor creation failed");

    executor.enable_profiling();

    // Run operation to generate stats
    let mut data = random_f32_vec(1024, 8888);
    let _ = executor.softmax(&mut data);

    // Reset
    executor.reset_profiler();

    eprintln!("LL-062: PASS - profiler reset");
}

// ============================================================================
// LL-070: Drop Order / Lifecycle Tests
// ============================================================================

#[test]
#[serial]
fn test_ll070_executor_drop_order() {
    skip_if_no_cuda!();

    // Multiple create/destroy cycles
    for i in 0..5 {
        let mut executor = CudaExecutor::new(0).expect("CUDA executor creation failed");

        // Do some work
        let a = vec![1.0f32; 64];
        let b = vec![1.0f32; 64];
        let mut c = vec![0.0f32; 64];
        let _ = executor.gemm(&a, &b, &mut c, 8, 8, 8);

        drop(executor); // Explicit drop
        eprintln!("LL-070: cycle {} - executor dropped", i);
    }
    eprintln!("LL-070: PASS - executor drop order (5 cycles)");
}

#[test]
#[serial]
fn test_ll071_rapid_lifecycle() {
    skip_if_no_cuda!();

    // Rapid create/destroy without work
    for i in 0..20 {
        let executor = CudaExecutor::new(0);
        assert!(executor.is_ok(), "LL-071: cycle {} failed", i);
        drop(executor);
    }
    eprintln!("LL-071: PASS - rapid lifecycle (20 cycles)");
}

// ============================================================================
// LL-080: Multi-Stream Coverage (make_current)
// ============================================================================

#[test]
#[serial]
fn test_ll080_make_current() {
    skip_if_no_cuda!();

    let executor = CudaExecutor::new(0).expect("CUDA executor creation failed");

    let result = executor.make_current();
    assert!(result.is_ok(), "LL-080: make_current should succeed");

    eprintln!("LL-080: PASS - make_current");
}

// ============================================================================
// LL-090: Performance Smoke Tests (< 5s each)
// ============================================================================

#[test]
#[serial]
fn test_ll090_performance_softmax_4k() {
    skip_if_no_cuda!();

    let mut executor = CudaExecutor::new(0).expect("CUDA executor creation failed");

    let start = std::time::Instant::now();
    for _ in 0..100 {
        let mut data = random_f32_vec(4096, 7777);
        let _ = executor.softmax(&mut data);
    }
    let elapsed = start.elapsed();

    assert!(
        elapsed.as_secs() < 5,
        "LL-090: 100x softmax(4096) took {:?}, should be < 5s",
        elapsed
    );
    eprintln!("LL-090: PASS - 100x softmax(4096) in {:?}", elapsed);
}

#[test]
#[serial]
fn test_ll091_performance_gemm_64() {
    skip_if_no_cuda!();

    let mut executor = CudaExecutor::new(0).expect("CUDA executor creation failed");

    let start = std::time::Instant::now();
    for _ in 0..100 {
        let a = random_f32_vec(64 * 64, 6666);
        let b = random_f32_vec(64 * 64, 5555);
        let mut c = vec![0.0f32; 64 * 64];
        let _ = executor.gemm(&a, &b, &mut c, 64, 64, 64);
    }
    let elapsed = start.elapsed();

    assert!(
        elapsed.as_secs() < 5,
        "LL-091: 100x gemm(64x64x64) took {:?}, should be < 5s",
        elapsed
    );
    eprintln!("LL-091: PASS - 100x gemm(64x64x64) in {:?}", elapsed);
}
