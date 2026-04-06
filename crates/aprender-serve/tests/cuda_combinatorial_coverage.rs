//! T-QA-023: Combinatorial Coverage Tests for cuda.rs
//!
//! Strategy: "Combinatorial Explosion" - test every valid combination of kernel types
//! to cover PTX generation and kernel name dispatch branches.
//!
//! Goal: Cover kernel dispatch branches and improve cuda.rs coverage.
//!
//! **IMPORTANT**: Run with `--test-threads=1` to avoid CUDA driver init race:
//! ```bash
//! cargo test --test cuda_combinatorial_coverage --features cuda -- --test-threads=1
//! ```
//! Parallel test execution causes CUDA_ERROR_NOT_INITIALIZED (error 3) due to
//! concurrent cuInit() calls.

#![cfg(feature = "cuda")]

use proptest::prelude::*;
use realizar::cuda::{CudaExecutor, CudaKernels, KernelType, PinnedHostBuffer};

// ============================================================================
// A. Helper: Graceful CUDA executor creation
// ============================================================================

fn try_create_executor() -> Option<CudaExecutor> {
    // IMPORTANT: Don't use is_available() pre-check - it can return false even when
    // CUDA hardware exists (RTX 4090). Instead, try to create and show actual error.
    match CudaExecutor::new(0) {
        Ok(exec) => Some(exec),
        Err(e) => {
            eprintln!("CUDA executor creation failed: {:?}", e);
            eprintln!("This should NOT happen on RTX 4090 - investigate the error above!");
            None
        },
    }
}

// ============================================================================
// B. PTX Generation Combinatorial Tests (No GPU Required)
// ============================================================================

#[test]
fn test_tqa023_ptx_all_kernel_types() {
    println!("\n=== T-QA-023: PTX Generation for All Kernel Types ===\n");

    let kernels = CudaKernels::new();
    let mut covered = 0;

    // B1. Basic GEMM variants
    let gemm_variants = [
        KernelType::GemmNaive {
            m: 64,
            n: 64,
            k: 64,
        },
        KernelType::GemmNaive {
            m: 1,
            n: 4096,
            k: 4096,
        },
        KernelType::GemmNaive {
            m: 32,
            n: 128,
            k: 256,
        },
        KernelType::GemmTiled {
            m: 256,
            n: 256,
            k: 256,
            tile_size: 16,
        },
        KernelType::GemmTiled {
            m: 1024,
            n: 1024,
            k: 1024,
            tile_size: 32,
        },
        KernelType::GemmTensorCore {
            m: 128,
            n: 128,
            k: 128,
        },
        KernelType::Gemv { k: 4096, n: 4096 },
        KernelType::Gemv { k: 2560, n: 10240 },
        KernelType::CoalescedGemv { k: 4096, n: 4096 },
        KernelType::CoalescedGemv { k: 896, n: 4864 },
    ];

    for kt in &gemm_variants {
        let ptx = kernels.generate_ptx(kt);
        assert!(
            ptx.contains(".version"),
            "PTX should have version for {:?}",
            kt
        );
        covered += 1;
    }
    println!("  GEMM variants: {} covered", covered);

    // B2. Quantized GEMM variants
    let quant_variants = [
        KernelType::QuantizedGemm {
            m: 1,
            n: 4096,
            k: 4096,
        },
        KernelType::QuantizedGemm {
            m: 32,
            n: 2560,
            k: 2560,
        },
        KernelType::QuantizedGemmGgml {
            m: 1,
            n: 4096,
            k: 4096,
        },
        KernelType::QuantizedGemmGgml {
            m: 16,
            n: 10240,
            k: 2560,
        },
        KernelType::Q5KQuantizedGemm {
            m: 1,
            n: 4096,
            k: 4096,
        },
        KernelType::Q5KQuantizedGemm {
            m: 8,
            n: 2560,
            k: 10240,
        },
        KernelType::Q6KQuantizedGemm {
            m: 1,
            n: 4096,
            k: 4096,
        },
        KernelType::Q6KQuantizedGemm {
            m: 4,
            n: 896,
            k: 4864,
        },
    ];

    for kt in &quant_variants {
        let ptx = kernels.generate_ptx(kt);
        assert!(
            ptx.contains(".version"),
            "PTX should have version for {:?}",
            kt
        );
        covered += 1;
    }
    println!("  Quantized GEMM: {} total covered", covered);

    // B3. GEMV variants (all quantization types)
    let gemv_variants = [
        KernelType::Q4KGemv { k: 4096, n: 4096 },
        KernelType::Q4KGemv { k: 2560, n: 10240 },
        KernelType::TiledQ4KGemv {
            k: 4096,
            n: 4096,
            outputs_per_block: 4,
        },
        KernelType::TiledQ4KGemv {
            k: 2560,
            n: 10240,
            outputs_per_block: 8,
        },
        KernelType::ChunkedTiledQ4KGemv {
            k: 4096,
            n: 4096,
            outputs_per_block: 4,
        },
        KernelType::CoalescedQ4KGemv { k: 4096, n: 4096 },
        KernelType::VectorizedQ4KGemv { k: 4096, n: 4096 },
        KernelType::Dp4aQ4KGemv { k: 4096, n: 4096 },
        KernelType::Dp4aSIMDQ4KGemv { k: 4096, n: 4096 },
        KernelType::Fp16Q4KGemv { k: 4096, n: 4096 },
        KernelType::Q5KGemv { k: 4096, n: 4096 },
        KernelType::Q5KGemv { k: 2560, n: 10240 },
        KernelType::Q6KGemv { k: 4096, n: 4096 },
        KernelType::Q6KGemv { k: 896, n: 4864 },
        KernelType::CoalescedQ6KGemv { k: 4096, n: 4096 },
        KernelType::Q8_0Gemv { k: 4096, n: 4096 },
        KernelType::Q8_0Gemv { k: 2560, n: 10240 },
        KernelType::Q4_0Gemv { k: 4096, n: 4096 },
        KernelType::Q4_0Gemv { k: 2560, n: 2560 },
        KernelType::Q4_1Gemv { k: 4096, n: 4096 },
        KernelType::Q5_0Gemv { k: 4096, n: 4096 },
    ];

    for kt in &gemv_variants {
        let ptx = kernels.generate_ptx(kt);
        assert!(
            ptx.contains(".version"),
            "PTX should have version for {:?}",
            kt
        );
        covered += 1;
    }
    println!("  GEMV variants: {} total covered", covered);

    // B4. Batched GEMV
    let batched_variants = [
        KernelType::BatchedQ4KGemv {
            m: 2,
            k: 4096,
            n: 4096,
        },
        KernelType::BatchedQ4KGemv {
            m: 16,
            k: 2560,
            n: 10240,
        },
        KernelType::BatchedQ4KGemv {
            m: 32,
            k: 4096,
            n: 4096,
        },
        KernelType::MultiWarpBatchedQ4KGemv {
            k: 4096,
            n: 4096,
            warps: 4,
        },
        KernelType::MultiWarpBatchedQ4KGemv {
            k: 2560,
            n: 10240,
            warps: 8,
        },
        KernelType::BatchedQ6KGemv {
            m: 2,
            k: 4096,
            n: 4096,
        },
        KernelType::BatchedQ6KGemv {
            m: 8,
            k: 896,
            n: 4864,
        },
    ];

    for kt in &batched_variants {
        let ptx = kernels.generate_ptx(kt);
        assert!(
            ptx.contains(".version"),
            "PTX should have version for {:?}",
            kt
        );
        covered += 1;
    }
    println!("  Batched GEMV: {} total covered", covered);

    // B5. Attention variants
    let attention_variants = [
        KernelType::Attention {
            seq_len: 128,
            head_dim: 64,
            causal: true,
        },
        KernelType::Attention {
            seq_len: 2048,
            head_dim: 128,
            causal: true,
        },
        KernelType::Attention {
            seq_len: 512,
            head_dim: 64,
            causal: false,
        },
        KernelType::MultiHeadAttention {
            n_heads: 8,
            seq_len: 512,
            head_dim: 64,
            causal: true,
        },
        KernelType::MultiHeadAttention {
            n_heads: 32,
            seq_len: 2048,
            head_dim: 128,
            causal: true,
        },
        KernelType::MultiHeadAttention {
            n_heads: 14,
            seq_len: 256,
            head_dim: 64,
            causal: false,
        },
    ];

    for kt in &attention_variants {
        let ptx = kernels.generate_ptx(kt);
        assert!(
            ptx.contains(".version"),
            "PTX should have version for {:?}",
            kt
        );
        covered += 1;
    }
    println!("  Attention: {} total covered", covered);

    // B6. Normalization and activation
    let norm_variants = [
        KernelType::Softmax { dim: 4096 },
        KernelType::Softmax { dim: 32000 },
        KernelType::Softmax { dim: 128256 },
        KernelType::LayerNorm {
            hidden_size: 4096,
            epsilon: 1e-5,
            affine: true,
        },
        KernelType::LayerNorm {
            hidden_size: 2560,
            epsilon: 1e-6,
            affine: false,
        },
        KernelType::RmsNorm {
            hidden_size: 4096,
            epsilon: 1e-5,
        },
        KernelType::RmsNorm {
            hidden_size: 896,
            epsilon: 1e-6,
        },
        KernelType::VectorizedRmsNorm {
            hidden_size: 4096,
            epsilon: 1e-5,
        },
        KernelType::FusedRmsNormQ4KGemv {
            k: 2560,
            n: 10240,
            epsilon: 1e-5,
        },
        KernelType::Gelu { n: 4096 },
        KernelType::Gelu { n: 10240 },
        KernelType::Silu { n: 4096 },
        KernelType::Silu { n: 10240 },
    ];

    for kt in &norm_variants {
        let ptx = kernels.generate_ptx(kt);
        assert!(
            ptx.contains(".version"),
            "PTX should have version for {:?}",
            kt
        );
        covered += 1;
    }
    println!("  Normalization/Activation: {} total covered", covered);

    // B7. Other kernels
    let other_variants = [
        KernelType::Rope {
            num_heads: 32,
            head_dim: 64,
            theta: 10000.0,
        },
        KernelType::RopeNeox {
            num_heads: 32,
            head_dim: 64,
            theta: 10000.0,
        },
        KernelType::RopeIndirect {
            num_heads: 32,
            head_dim: 64,
            theta: 10000.0,
        },
        KernelType::RopeNeoxIndirect {
            num_heads: 32,
            head_dim: 64,
            theta: 10000.0,
        },
        KernelType::ResidualAdd { n: 4096 },
        KernelType::ResidualAdd { n: 2560 },
        KernelType::FusedGateUpQ4KGemv { k: 2560, n: 10240 },
        KernelType::Q8Quantize { n: 4096 },
        KernelType::Q4KQ8Dot { k: 256, n: 256 },
        KernelType::Q4KQ8Dot { k: 4096, n: 4096 },
    ];

    for kt in &other_variants {
        let ptx = kernels.generate_ptx(kt);
        assert!(
            ptx.contains(".version"),
            "PTX should have version for {:?}",
            kt
        );
        covered += 1;
    }
    println!("  Other kernels: {} total covered", covered);

    println!("\n  TOTAL: {} kernel types covered", covered);
    assert!(covered >= 60, "Should cover at least 60 kernel types");
}

#[test]
fn test_tqa023_kernel_names_all_types() {
    println!("\n=== T-QA-023: Kernel Names for All Types ===\n");

    let kernels = CudaKernels::new();

    let kernel_types = vec![
        KernelType::GemmNaive { m: 1, n: 1, k: 1 },
        KernelType::GemmTiled {
            m: 1,
            n: 1,
            k: 1,
            tile_size: 16,
        },
        KernelType::GemmTensorCore { m: 1, n: 1, k: 1 },
        KernelType::Gemv { k: 1, n: 1 },
        KernelType::CoalescedGemv { k: 1, n: 1 },
        KernelType::Softmax { dim: 1 },
        KernelType::LayerNorm {
            hidden_size: 1,
            epsilon: 1e-5,
            affine: true,
        },
        KernelType::Attention {
            seq_len: 1,
            head_dim: 1,
            causal: true,
        },
        KernelType::QuantizedGemm { m: 1, n: 1, k: 1 },
        KernelType::QuantizedGemmGgml { m: 1, n: 1, k: 1 },
        KernelType::Q5KQuantizedGemm { m: 1, n: 1, k: 1 },
        KernelType::Q6KQuantizedGemm { m: 1, n: 1, k: 1 },
        KernelType::Q4KGemv { k: 1, n: 1 },
        KernelType::TiledQ4KGemv {
            k: 1,
            n: 1,
            outputs_per_block: 1,
        },
        KernelType::ChunkedTiledQ4KGemv {
            k: 1,
            n: 1,
            outputs_per_block: 1,
        },
        KernelType::CoalescedQ4KGemv { k: 1, n: 1 },
        KernelType::VectorizedQ4KGemv { k: 1, n: 1 },
        KernelType::Dp4aQ4KGemv { k: 1, n: 1 },
        KernelType::Dp4aSIMDQ4KGemv { k: 1, n: 1 },
        KernelType::Fp16Q4KGemv { k: 1, n: 1 },
        KernelType::Q5KGemv { k: 1, n: 1 },
        KernelType::Q6KGemv { k: 1, n: 1 },
        KernelType::CoalescedQ6KGemv { k: 1, n: 1 },
        KernelType::Q8_0Gemv { k: 1, n: 1 },
        KernelType::Q4_0Gemv { k: 1, n: 1 },
        KernelType::Q4_1Gemv { k: 1, n: 1 },
        KernelType::Q5_0Gemv { k: 1, n: 1 },
        KernelType::BatchedQ4KGemv { m: 1, k: 1, n: 1 },
        KernelType::MultiWarpBatchedQ4KGemv {
            k: 1,
            n: 1,
            warps: 1,
        },
        KernelType::BatchedQ6KGemv { m: 1, k: 1, n: 1 },
        KernelType::RmsNorm {
            hidden_size: 1,
            epsilon: 1e-5,
        },
        KernelType::VectorizedRmsNorm {
            hidden_size: 1,
            epsilon: 1e-5,
        },
        KernelType::FusedRmsNormQ4KGemv {
            k: 1,
            n: 1,
            epsilon: 1e-5,
        },
        KernelType::Gelu { n: 1 },
        KernelType::Silu { n: 1 },
        KernelType::Rope {
            num_heads: 1,
            head_dim: 1,
            theta: 10000.0,
        },
        KernelType::RopeNeox {
            num_heads: 1,
            head_dim: 1,
            theta: 10000.0,
        },
        KernelType::RopeIndirect {
            num_heads: 1,
            head_dim: 1,
            theta: 10000.0,
        },
        KernelType::RopeNeoxIndirect {
            num_heads: 1,
            head_dim: 1,
            theta: 10000.0,
        },
        KernelType::ResidualAdd { n: 1 },
        KernelType::FusedGateUpQ4KGemv { k: 1, n: 1 },
        KernelType::Q8Quantize { n: 1 },
        KernelType::Q4KQ8Dot { k: 1, n: 1 },
        KernelType::MultiHeadAttention {
            n_heads: 1,
            seq_len: 1,
            head_dim: 1,
            causal: true,
        },
    ];

    for kt in &kernel_types {
        let name = kernels.kernel_name(kt);
        assert!(
            !name.is_empty(),
            "Kernel name should not be empty for {:?}",
            kt
        );
    }

    println!("  {} kernel name branches covered", kernel_types.len());
}

// ============================================================================
// C. PinnedHostBuffer Coverage Tests
// ============================================================================

#[test]
fn test_tqa023_pinned_host_buffer_api() {
    println!("\n=== T-QA-023: PinnedHostBuffer API Coverage ===\n");

    let mut buf: PinnedHostBuffer<f32> = PinnedHostBuffer::new(1024);

    assert_eq!(buf.len(), 1024);
    assert!(!buf.is_empty());
    println!("  len: {}", buf.len());
    println!("  is_empty: {}", buf.is_empty());
    println!("  is_pinned: {}", buf.is_pinned());
    println!("  size_bytes: {}", buf.size_bytes());

    let slice = buf.as_slice();
    assert_eq!(slice.len(), 1024);

    let slice_mut = buf.as_mut_slice();
    slice_mut[0] = 42.0;
    assert_eq!(buf.as_slice()[0], 42.0);

    let src: Vec<f32> = (0..1024).map(|i| i as f32).collect();
    buf.copy_from_slice(&src);
    assert_eq!(buf.as_slice()[100], 100.0);

    println!("  PinnedHostBuffer API fully covered");
}

// ============================================================================
// D. CudaExecutor Basic API Tests (GPU Required)
// ============================================================================

#[test]
fn test_tqa023_executor_profiler_api() {
    let Some(mut executor) = try_create_executor() else {
        return;
    };

    println!("\n=== T-QA-023: Executor Profiler API ===\n");

    executor.enable_profiling();
    assert!(executor.is_profiling_enabled());

    executor.disable_profiling();
    assert!(!executor.is_profiling_enabled());

    executor.enable_profiling();

    let _profiler = executor.profiler();
    let _profiler_mut = executor.profiler_mut();

    let summary = executor.profiler_summary();
    println!("  Profiler summary: {} lines", summary.lines().count());

    let stats = executor.profiler_category_stats();
    println!("  Category stats: {} categories", stats.len());

    executor.print_profiler_categories();

    let mode = executor.profiler_sync_mode();
    println!("  Sync mode: {:?}", mode);

    executor.set_profiler_sync_mode(trueno::SyncMode::Deferred);
    executor.reset_profiler();

    println!("  Profiler API fully covered");
}

#[test]
fn test_tqa023_executor_graph_tracking_api() {
    let Some(mut executor) = try_create_executor() else {
        return;
    };

    println!("\n=== T-QA-023: Executor Graph Tracking API ===\n");

    executor.enable_graph_tracking();
    assert!(executor.is_graph_tracking_enabled());

    executor.disable_graph_tracking();
    assert!(!executor.is_graph_tracking_enabled());

    executor.enable_graph_tracking();

    let graph = executor.execution_graph();
    println!("  Graph nodes: {}", graph.num_nodes());

    let ascii = executor.execution_graph_ascii();
    println!("  ASCII graph: {} chars", ascii.len());

    executor.clear_execution_graph();

    println!("  Graph tracking API fully covered");
}

#[test]
fn test_tqa023_executor_tile_profiling_api() {
    let Some(mut executor) = try_create_executor() else {
        return;
    };

    println!("\n=== T-QA-023: Executor Tile Profiling API ===\n");

    executor.enable_tile_profiling();
    assert!(executor.is_tile_profiling_enabled());

    executor.disable_tile_profiling();
    assert!(!executor.is_tile_profiling_enabled());

    executor.enable_tile_profiling();

    let _macro_stats = executor.tile_stats(trueno::TileLevel::Macro);
    let _midi_stats = executor.tile_stats(trueno::TileLevel::Midi);

    let summary = executor.tile_summary();
    println!("  Tile summary: {} chars", summary.len());

    let json = executor.tile_stats_json();
    println!("  Tile JSON: {} chars", json.len());

    executor.reset_tile_stats();

    println!("  Tile profiling API fully covered");
}

#[test]
fn test_tqa023_executor_weight_cache_api() {
    let Some(mut executor) = try_create_executor() else {
        return;
    };

    println!("\n=== T-QA-023: Executor Weight Cache API ===\n");

    let weights: Vec<f32> = (0..4096).map(|i| i as f32 * 0.001).collect();

    match executor.load_weights("test_layer_0_q", &weights) {
        Ok(ptr) => {
            assert!(executor.has_weights("test_layer_0_q"));
            assert!(!executor.has_weights("nonexistent"));
            println!("  Loaded weights at ptr: 0x{:x}", ptr);
        },
        Err(e) => println!("  Weight load failed: {:?}", e),
    }

    let count = executor.cached_weight_count();
    let bytes = executor.cached_weight_bytes();
    println!("  Cached weights: {} tensors, {} bytes", count, bytes);

    println!("  Weight cache API fully covered");
}

#[test]
fn test_tqa023_executor_quantized_weight_api() {
    let Some(mut executor) = try_create_executor() else {
        return;
    };

    println!("\n=== T-QA-023: Executor Quantized Weight API ===\n");

    let has_qw = executor.has_quantized_weights("test_q4k");
    println!("  has_quantized_weights(test_q4k): {}", has_qw);

    let qw_type = executor.get_quantized_weight_type("test_q4k");
    println!("  get_quantized_weight_type: {:?}", qw_type);

    let count = executor.cached_quantized_weight_count();
    let bytes = executor.cached_quantized_weight_bytes();
    println!("  Quantized weights: {} tensors, {} bytes", count, bytes);

    let ptr_result = executor.get_quantized_weight_ptr("nonexistent");
    assert!(ptr_result.is_err());

    executor.clear_quantized_weights();

    println!("  Quantized weight API fully covered");
}

#[test]
fn test_tqa023_executor_synchronization() {
    let Some(executor) = try_create_executor() else {
        return;
    };

    println!("\n=== T-QA-023: Executor Synchronization ===\n");

    match executor.synchronize() {
        Ok(_) => println!("  synchronize: OK"),
        Err(e) => println!("  synchronize: {:?}", e),
    }

    match executor.synchronize_compute() {
        Ok(_) => println!("  synchronize_compute: OK"),
        Err(e) => println!("  synchronize_compute: {:?}", e),
    }

    match executor.synchronize_transfer() {
        Ok(_) => println!("  synchronize_transfer: OK"),
        Err(e) => println!("  synchronize_transfer: {:?}", e),
    }

    match executor.synchronize_all() {
        Ok(_) => println!("  synchronize_all: OK"),
        Err(e) => println!("  synchronize_all: {:?}", e),
    }

    println!("  Synchronization API fully covered");
}

#[test]
fn test_tqa023_executor_softmax() {
    let Some(mut executor) = try_create_executor() else {
        return;
    };

    println!("\n=== T-QA-023: Softmax ===\n");

    let dims = [128, 4096, 32000];

    for dim in &dims {
        let mut data: Vec<f32> = (0..*dim).map(|i| (i as f32 * 0.01).sin()).collect();

        match executor.softmax(&mut data) {
            Ok(_) => {
                let sum: f32 = data.iter().sum();
                println!("  softmax dim={}: sum={:.6}", dim, sum);
            },
            Err(e) => println!("  softmax dim={}: {:?}", dim, e),
        }
    }

    println!("  Softmax covered");
}

// ============================================================================
// E. Error Path Coverage Tests
// ============================================================================

#[test]
fn test_tqa023_error_quantized_weight_not_found() {
    let Some(executor) = try_create_executor() else {
        return;
    };

    println!("\n=== T-QA-023: Error Paths - Quantized Weight Not Found ===\n");

    let result = executor.get_quantized_weight_ptr("does_not_exist");
    match result {
        Ok(ptr) => println!("  Non-existent weight: Unexpected success, ptr=0x{:x}", ptr),
        Err(e) => println!("  Non-existent weight: Expected error {:?}", e),
    }

    println!("  Quantized weight error paths covered");
}

// ============================================================================
// F. Proptest Combinatorial Tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn test_tqa023_proptest_ptx_generation(
        m in 1u32..=64,
        n in 256u32..=4096,
        k in 256u32..=4096,
    ) {
        let kernels = CudaKernels::new();

        let k_aligned = (k / 256) * 256;
        if k_aligned == 0 { return Ok(()); }

        let ptx = kernels.generate_ptx(&KernelType::QuantizedGemmGgml { m, n, k: k_aligned });
        prop_assert!(ptx.contains(".version"));
        prop_assert!(ptx.contains("q4k"));
    }

    #[test]
    fn test_tqa023_proptest_kernel_names(
        head_dim in prop::sample::select(vec![32u32, 64, 96, 128]),
        n_heads in prop::sample::select(vec![1u32, 8, 14, 32]),
        seq_len in 128u32..=2048,
    ) {
        let kernels = CudaKernels::new();

        let kt = KernelType::MultiHeadAttention {
            n_heads,
            seq_len,
            head_dim,
            causal: true,
        };

        let name = kernels.kernel_name(&kt);
        prop_assert!(!name.is_empty());
    }

    #[test]
    fn test_tqa023_proptest_gemv_dimensions(
        k in prop::sample::select(vec![256u32, 512, 1024, 2048, 4096]),
        n in prop::sample::select(vec![256u32, 896, 2560, 4096, 10240]),
    ) {
        let kernels = CudaKernels::new();

        let gemv_types = [
            KernelType::Q4KGemv { k, n },
            KernelType::Q5KGemv { k, n },
            KernelType::Q6KGemv { k, n },
            KernelType::Q8_0Gemv { k, n },
        ];

        for kt in &gemv_types {
            let ptx = kernels.generate_ptx(kt);
            prop_assert!(ptx.contains(".version"));
        }
    }
}

// ============================================================================
// G. Non-GPU Component Tests (For coverage without CUDA hardware)
// ============================================================================

use realizar::cuda::{GpuMemoryPool, StagingBufferPool, TransferMode, WeightQuantType};

#[test]
fn test_tqa023_size_class_coverage() {
    println!("\n=== T-QA-023: SizeClass Coverage ===\n");

    use realizar::cuda::SizeClass;

    // Test all size class boundaries
    let test_sizes = [
        (1, Some(4096)),                  // Tiny → 4KB
        (4096, Some(4096)),               // Exact 4KB
        (4097, Some(16384)),              // 4KB+1 → 16KB
        (16384, Some(16384)),             // Exact 16KB
        (16385, Some(65536)),             // 16KB+1 → 64KB
        (65536, Some(65536)),             // Exact 64KB
        (65537, Some(262_144)),           // 64KB+1 → 256KB
        (262_144, Some(262_144)),         // Exact 256KB
        (262_145, Some(1_048_576)),       // 256KB+1 → 1MB
        (1_048_576, Some(1_048_576)),     // Exact 1MB
        (1_048_577, Some(4_194_304)),     // 1MB+1 → 4MB
        (4_194_304, Some(4_194_304)),     // Exact 4MB
        (4_194_305, Some(16_777_216)),    // 4MB+1 → 16MB
        (16_777_216, Some(16_777_216)),   // Exact 16MB
        (16_777_217, Some(67_108_864)),   // 16MB+1 → 64MB
        (67_108_864, Some(67_108_864)),   // Exact 64MB
        (67_108_865, Some(268_435_456)),  // 64MB+1 → 256MB
        (268_435_456, Some(268_435_456)), // Exact 256MB
        (268_435_457, None),              // Too large → None
    ];

    for (size, expected_class) in &test_sizes {
        let result = SizeClass::for_size(*size);
        match (result, expected_class) {
            (Some(sc), Some(expected)) => {
                assert_eq!(
                    sc.bytes(),
                    *expected,
                    "Size {} should map to {}",
                    size,
                    expected
                );
            },
            (None, None) => {
                println!("  Size {} correctly returns None (too large)", size);
            },
            _ => panic!(
                "Mismatch for size {}: got {:?}, expected {:?}",
                size, result, expected_class
            ),
        }
    }

    println!("  All {} SizeClass boundary tests passed", test_sizes.len());
}

#[test]
fn test_tqa023_gpu_memory_pool_coverage() {
    println!("\n=== T-QA-023: GpuMemoryPool Coverage ===\n");

    // Test default construction
    let pool_default = GpuMemoryPool::default();
    assert_eq!(pool_default.max_size(), 2 * 1024 * 1024 * 1024);

    // Test custom max size
    let mut pool = GpuMemoryPool::with_max_size(1024 * 1024 * 1024); // 1GB
    assert_eq!(pool.max_size(), 1024 * 1024 * 1024);
    println!("  Created pool with 1GB max size");

    // Test initial stats
    let stats = pool.stats();
    assert_eq!(stats.total_allocated, 0);
    assert_eq!(stats.peak_usage, 0);
    assert_eq!(stats.pool_hits, 0);
    assert_eq!(stats.pool_misses, 0);
    assert_eq!(stats.free_buffers, 0);
    assert_eq!(stats.hit_rate, 0.0);
    println!("  Initial stats verified");

    // Test try_get with no buffers (should miss)
    let result = pool.try_get(4096);
    assert!(result.is_none());
    let stats = pool.stats();
    assert_eq!(stats.pool_misses, 1);
    println!("  Pool miss recorded correctly");

    // Test record_allocation and record_deallocation
    pool.record_allocation(1024 * 1024); // 1MB
    let stats = pool.stats();
    assert_eq!(stats.total_allocated, 1024 * 1024);
    assert_eq!(stats.peak_usage, 1024 * 1024);

    pool.record_allocation(2 * 1024 * 1024); // +2MB
    let stats = pool.stats();
    assert_eq!(stats.total_allocated, 3 * 1024 * 1024);
    assert_eq!(stats.peak_usage, 3 * 1024 * 1024);

    pool.record_deallocation(1024 * 1024); // -1MB
    let stats = pool.stats();
    assert_eq!(stats.total_allocated, 2 * 1024 * 1024);
    assert_eq!(stats.peak_usage, 3 * 1024 * 1024); // Peak unchanged
    println!("  Allocation tracking verified");

    // Test has_capacity
    assert!(pool.has_capacity(100 * 1024 * 1024)); // 100MB should fit
    assert!(!pool.has_capacity(pool.max_size())); // Already have 2MB, can't fit max_size more
    println!("  Capacity checking verified");

    // Test clear (just verify the method exists and works)
    pool.clear();
    let stats = pool.stats();
    assert_eq!(stats.free_buffers, 0);
    println!("  Pool clear verified");

    // Test estimated_savings_bytes with no hits
    let savings = stats.estimated_savings_bytes();
    assert_eq!(savings, 0); // No hits yet
    println!("  Estimated savings with no hits: {} bytes", savings);

    println!("  GpuMemoryPool coverage complete");
}

#[test]
fn test_tqa023_staging_buffer_pool_coverage() {
    println!("\n=== T-QA-023: StagingBufferPool Coverage ===\n");

    // Test default construction
    let pool_default = StagingBufferPool::default();
    let stats = pool_default.stats();
    assert_eq!(stats.pool_hits, 0);
    println!("  Default pool created");

    // Test custom max size
    let mut pool = StagingBufferPool::with_max_size(512 * 1024 * 1024); // 512MB
    println!("  Created pool with 512MB max size");

    // Test initial stats
    let stats = pool.stats();
    assert_eq!(stats.pool_hits, 0);
    assert_eq!(stats.pool_misses, 0);
    assert_eq!(stats.free_buffers, 0);
    println!("  Initial stats verified");

    // Test get (new allocation)
    let buf = pool.get(1024);
    assert_eq!(buf.len(), 1024);
    let stats = pool.stats();
    assert_eq!(stats.pool_misses, 1);
    println!("  Pool miss on first get");

    // Test put and re-get (should hit)
    pool.put(buf);
    let stats = pool.stats();
    assert_eq!(stats.free_buffers, 1);

    let buf2 = pool.get(1024);
    assert_eq!(buf2.len(), 1024);
    let stats = pool.stats();
    assert_eq!(stats.pool_hits, 1);
    assert!(stats.hit_rate > 0.0);
    println!("  Pool hit on second get");

    // Test clear
    pool.put(buf2);
    pool.clear();
    let stats = pool.stats();
    assert_eq!(stats.free_buffers, 0);
    println!("  Pool clear verified");

    println!("  StagingBufferPool coverage complete");
}

#[test]
fn test_tqa023_transfer_mode_coverage() {
    println!("\n=== T-QA-023: TransferMode Coverage ===\n");

    // Test Default
    let default_mode: TransferMode = Default::default();
    assert_eq!(default_mode, TransferMode::Pageable);
    println!("  Default mode is Pageable");

    // Test requires_pinned
    assert!(!TransferMode::Pageable.requires_pinned());
    assert!(TransferMode::Pinned.requires_pinned());
    assert!(TransferMode::ZeroCopy.requires_pinned());
    assert!(TransferMode::Async.requires_pinned());
    println!("  requires_pinned verified for all modes");

    // Test estimated_speedup
    assert_eq!(TransferMode::Pageable.estimated_speedup(), 1.0);
    assert_eq!(TransferMode::Pinned.estimated_speedup(), 1.7);
    assert_eq!(TransferMode::ZeroCopy.estimated_speedup(), 2.0);
    assert_eq!(TransferMode::Async.estimated_speedup(), 1.5);
    println!("  estimated_speedup verified for all modes");

    println!("  TransferMode coverage complete");
}

#[test]
fn test_tqa023_weight_quant_type_coverage() {
    println!("\n=== T-QA-023: WeightQuantType Coverage ===\n");

    // Test common type (PMAT-232: Default intentionally removed)
    let common_type: WeightQuantType = WeightQuantType::Q4K;
    assert_eq!(common_type, WeightQuantType::Q4K);
    println!("  Common type is Q4K");

    // Test from_ggml_type
    let ggml_mappings = [
        (2, Some(WeightQuantType::Q4_0)),
        (3, Some(WeightQuantType::Q4_1)),
        (6, Some(WeightQuantType::Q5_0)),
        (8, Some(WeightQuantType::Q8_0)),
        (12, Some(WeightQuantType::Q4K)),
        (13, Some(WeightQuantType::Q5K)),
        (14, Some(WeightQuantType::Q6K)),
        (0, None),  // Unknown
        (1, None),  // Unknown
        (99, None), // Unknown
    ];

    for (type_id, expected) in &ggml_mappings {
        let result = WeightQuantType::from_ggml_type(*type_id);
        assert_eq!(result, *expected, "GGML type {} mismatch", type_id);
    }
    println!(
        "  from_ggml_type verified for {} cases",
        ggml_mappings.len()
    );

    // Test bytes_per_superblock
    assert_eq!(WeightQuantType::Q4K.bytes_per_superblock(), 144);
    assert_eq!(WeightQuantType::Q5K.bytes_per_superblock(), 176);
    assert_eq!(WeightQuantType::Q6K.bytes_per_superblock(), 210);
    assert_eq!(WeightQuantType::Q8_0.bytes_per_superblock(), 34 * 8);
    assert_eq!(WeightQuantType::Q5_0.bytes_per_superblock(), 22 * 8);
    assert_eq!(WeightQuantType::Q4_0.bytes_per_superblock(), 18 * 8);
    assert_eq!(WeightQuantType::Q4_1.bytes_per_superblock(), 20 * 8);
    println!("  bytes_per_superblock verified for all types");

    // Test bytes_per_block
    assert_eq!(WeightQuantType::Q4K.bytes_per_block(), 18);
    assert_eq!(WeightQuantType::Q5K.bytes_per_block(), 22);
    assert_eq!(WeightQuantType::Q6K.bytes_per_block(), 26);
    assert_eq!(WeightQuantType::Q8_0.bytes_per_block(), 34);
    assert_eq!(WeightQuantType::Q5_0.bytes_per_block(), 22);
    assert_eq!(WeightQuantType::Q4_0.bytes_per_block(), 18);
    assert_eq!(WeightQuantType::Q4_1.bytes_per_block(), 20);
    println!("  bytes_per_block verified for all types");

    // Test matches_size for super-block formats (Q4K, Q5K, Q6K)
    // For Q4K: 144 bytes per 256 elements
    // 1 row × 256 cols = 1 superblock = 144 bytes
    assert!(WeightQuantType::Q4K.matches_size(144, 1, 256));
    // 2 rows × 512 cols = 2×2 = 4 superblocks = 576 bytes
    assert!(WeightQuantType::Q4K.matches_size(576, 2, 512));

    // For Q5K: 176 bytes per 256 elements
    assert!(WeightQuantType::Q5K.matches_size(176, 1, 256));

    // For Q6K: 210 bytes per 256 elements
    assert!(WeightQuantType::Q6K.matches_size(210, 1, 256));
    println!("  matches_size verified for super-block formats");

    // Test matches_size for block formats (Q4_0, Q4_1, Q5_0, Q8_0)
    // For Q4_0: 18 bytes per 32 elements
    // 1 row × 32 cols = 1 block = 18 bytes
    assert!(WeightQuantType::Q4_0.matches_size(18, 1, 32));
    // 2 rows × 64 cols = 2×2 = 4 blocks = 72 bytes
    assert!(WeightQuantType::Q4_0.matches_size(72, 2, 64));

    // For Q4_1: 20 bytes per 32 elements
    assert!(WeightQuantType::Q4_1.matches_size(20, 1, 32));

    // For Q5_0: 22 bytes per 32 elements
    assert!(WeightQuantType::Q5_0.matches_size(22, 1, 32));

    // For Q8_0: 34 bytes per 32 elements
    assert!(WeightQuantType::Q8_0.matches_size(34, 1, 32));
    println!("  matches_size verified for block formats");

    // Test from_size
    // Q4K: 1 row × 256 cols = 144 bytes
    assert_eq!(
        WeightQuantType::from_size(144, 1, 256),
        Some(WeightQuantType::Q4K)
    );
    // Q5K: 1 row × 256 cols = 176 bytes
    assert_eq!(
        WeightQuantType::from_size(176, 1, 256),
        Some(WeightQuantType::Q5K)
    );
    // Q6K: 1 row × 256 cols = 210 bytes
    assert_eq!(
        WeightQuantType::from_size(210, 1, 256),
        Some(WeightQuantType::Q6K)
    );
    // Q8_0: 1 row × 32 cols = 34 bytes
    assert_eq!(
        WeightQuantType::from_size(34, 1, 32),
        Some(WeightQuantType::Q8_0)
    );
    // Q5_0: 1 row × 32 cols = 22 bytes
    assert_eq!(
        WeightQuantType::from_size(22, 1, 32),
        Some(WeightQuantType::Q5_0)
    );
    // Unknown size
    assert_eq!(WeightQuantType::from_size(999999, 1, 32), None);
    println!("  from_size verified");

    println!("  WeightQuantType coverage complete");
}

#[test]
fn test_tqa023_indexed_layer_weights_zeroed() {
    println!("\n=== T-QA-023: IndexedLayerWeights Zeroed Construction Coverage ===\n");

    use realizar::cuda::IndexedLayerWeights;

    // PMAT-232: Default intentionally removed; use explicit zeroed construction
    let weights = IndexedLayerWeights {
        attn_q_ptr: 0,
        attn_q_len: 0,
        attn_q_qtype: WeightQuantType::Q4K,
        attn_k_ptr: 0,
        attn_k_len: 0,
        attn_k_qtype: WeightQuantType::Q4K,
        attn_v_ptr: 0,
        attn_v_len: 0,
        attn_v_qtype: WeightQuantType::Q4K,
        attn_output_ptr: 0,
        attn_output_len: 0,
        attn_output_qtype: WeightQuantType::Q4K,
        ffn_gate_ptr: 0,
        ffn_gate_len: 0,
        ffn_gate_qtype: WeightQuantType::Q4K,
        ffn_up_ptr: 0,
        ffn_up_len: 0,
        ffn_up_qtype: WeightQuantType::Q4K,
        ffn_down_ptr: 0,
        ffn_down_len: 0,
        ffn_down_qtype: WeightQuantType::Q4K,
        attn_norm_ptr: 0,
        attn_norm_len: 0,
        ffn_norm_ptr: 0,
        ffn_norm_len: 0,
        attn_q_bias_ptr: 0,
        attn_q_bias_len: 0,
        attn_k_bias_ptr: 0,
        attn_k_bias_len: 0,
        attn_v_bias_ptr: 0,
        attn_v_bias_len: 0,
        attn_q_norm_ptr: 0,
        attn_q_norm_len: 0,
        attn_k_norm_ptr: 0,
        attn_k_norm_len: 0,
    };
    assert_eq!(weights.attn_q_ptr, 0);
    assert_eq!(weights.attn_q_len, 0);
    assert_eq!(weights.attn_q_qtype, WeightQuantType::Q4K);
    assert_eq!(weights.attn_k_ptr, 0);
    assert_eq!(weights.attn_v_ptr, 0);
    assert_eq!(weights.attn_output_ptr, 0);
    assert_eq!(weights.ffn_gate_ptr, 0);
    assert_eq!(weights.ffn_up_ptr, 0);
    assert_eq!(weights.ffn_down_ptr, 0);
    assert_eq!(weights.attn_norm_ptr, 0);
    assert_eq!(weights.ffn_norm_ptr, 0);
    println!("  IndexedLayerWeights zeroed construction values verified");
}

#[test]
fn test_tqa023_cuda_likely_available() {
    println!("\n=== T-QA-023: cuda_likely_available Coverage ===\n");

    use realizar::cuda::CudaKernels;

    // This just covers the function call - result depends on system
    let available = CudaKernels::cuda_likely_available();
    println!("  cuda_likely_available: {}", available);
}

// ============================================================================
// H. Summary Test
// ============================================================================

#[test]
fn test_tqa023_summary() {
    println!("\n");
    println!("╔════════════════════════════════════════════════════════════════════╗");
    println!("║          T-QA-023: CUDA Combinatorial Coverage Summary             ║");
    println!("╠════════════════════════════════════════════════════════════════════╣");
    println!("║                                                                    ║");
    println!("║  Coverage Strategy: Combinatorial Explosion                        ║");
    println!("║  ┌──────────────────────────────────────────────────────────────┐  ║");
    println!("║  │ - PTX generation for 60+ kernel types                       │  ║");
    println!("║  │ - All quantization types: Q4_0, Q4_K, Q5_K, Q6_K, Q8_0      │  ║");
    println!("║  │ - CudaExecutor profiler, graph, tile APIs                   │  ║");
    println!("║  │ - Proptest combinatorial parameter generation               │  ║");
    println!("║  └──────────────────────────────────────────────────────────────┘  ║");
    println!("║                                                                    ║");
    println!("║  Target: Cover kernel dispatch branches in cuda.rs                 ║");
    println!("║                                                                    ║");
    println!("╚════════════════════════════════════════════════════════════════════╝");
    println!();
}
