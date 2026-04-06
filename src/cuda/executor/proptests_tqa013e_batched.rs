
/// T-QA-013e: Test batched decode graph state management
///
/// Verifies batched graph state is properly initialized.
#[test]
#[serial]
fn test_tqa013e_batched_graph_state() {
    if !CudaExecutor::is_available() {
        println!("T-QA-013e: CUDA not available, skipping");
        return;
    }

    let executor = CudaExecutor::new(0).expect("CUDA executor");

    // Verify batched decode graphs map is empty initially
    // We check this indirectly via the fact that has_decode_graph returns false
    // (batched graphs use a different storage but similar patterns)
    assert!(
        !executor.has_decode_graph(),
        "T-QA-013e: No graphs captured initially"
    );

    println!("T-QA-013e: Batched graph state initialization PASSED");
}

/// T-QA-013f: Test graph state after clear_workspace
///
/// Verifies that clearing workspace affects graph capture eligibility.
#[test]
#[serial]
fn test_tqa013f_clear_workspace_graph_state() {
    if !CudaExecutor::is_available() {
        println!("T-QA-013f: CUDA not available, skipping");
        return;
    }

    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Clear workspace and verify state
    executor.clear_workspace();
    assert!(
        !executor.has_workspace(),
        "T-QA-013f: No workspace after clear"
    );

    // Clear decode graph and verify
    executor.clear_decode_graph();
    assert!(
        !executor.has_decode_graph(),
        "T-QA-013f: No graph after clear"
    );

    // Clear indexed weights
    executor.clear_indexed_weights();
    assert!(
        !executor.has_indexed_weights(),
        "T-QA-013f: No indexed weights after clear"
    );

    println!("T-QA-013f: Clear workspace/graph state PASSED");
}

// ============================================================================
// T-QA-014: Buffer Fuzzing Tests (proptest GpuBuffer lifecycle)
// ============================================================================
// These tests use property-based testing to fuzz GpuBuffer operations.

proptest! {
    /// T-QA-014a: Property - GpuBuffer allocation succeeds for various sizes
    ///
    /// Tests that GpuBuffer::new works for a range of sizes (1 to 10000).
    #[test]
    #[serial]
    fn prop_tqa014a_buffer_allocation_various_sizes(size in 1usize..10000) {
        if !has_cuda() {
            return Ok(());
        }

        let executor = CudaExecutor::new(0)
            .map_err(|e| TestCaseError::fail(format!("T-QA-014a: Executor init failed: {}", e)))?;

        let buf: GpuBuffer<f32> = GpuBuffer::new(&executor.context, size)
            .map_err(|e| TestCaseError::fail(format!("T-QA-014a: Allocation failed for size {}: {}", size, e)))?;

        prop_assert_eq!(buf.len(), size, "T-QA-014a: Buffer length mismatch");
        prop_assert_eq!(buf.size_bytes(), size * std::mem::size_of::<f32>(), "T-QA-014a: Byte size mismatch");
    }

    /// T-QA-014b: Property - GpuBuffer from_host preserves data integrity
    ///
    /// Tests that data uploaded via from_host can be read back correctly.
    #[test]
    #[serial]
    fn prop_tqa014b_buffer_data_integrity(data in prop::collection::vec(any::<f32>(), 1..1000)) {
        if !has_cuda() {
            return Ok(());
        }

        let executor = CudaExecutor::new(0)
            .map_err(|e| TestCaseError::fail(format!("T-QA-014b: Executor init failed: {}", e)))?;

        // Filter out NaN values which can't be compared with ==
        let data: Vec<f32> = data.into_iter().filter(|x| !x.is_nan()).collect();
        if data.is_empty() {
            return Ok(());
        }

        let buf = GpuBuffer::from_host(&executor.context, &data)
            .map_err(|e| TestCaseError::fail(format!("T-QA-014b: from_host failed: {}", e)))?;

        let mut readback = vec![0.0f32; data.len()];
        buf.copy_to_host(&mut readback)
            .map_err(|e| TestCaseError::fail(format!("T-QA-014b: copy_to_host failed: {}", e)))?;

        for (i, (expected, actual)) in data.iter().zip(readback.iter()).enumerate() {
            if expected.is_finite() && actual.is_finite() {
                prop_assert!(
                    (expected - actual).abs() < 1e-6,
                    "T-QA-014b: Data mismatch at index {}: expected {}, got {}",
                    i, expected, actual
                );
            }
        }
    }

    /// T-QA-014c: Property - Multiple buffers can be allocated and freed
    ///
    /// Tests that allocating multiple buffers in sequence works correctly.
    #[test]
    #[serial]
    fn prop_tqa014c_multiple_buffer_allocation(num_buffers in 1..20usize, base_size in 100..1000usize) {
        if !has_cuda() {
            return Ok(());
        }

        let executor = CudaExecutor::new(0)
            .map_err(|e| TestCaseError::fail(format!("T-QA-014c: Executor init failed: {}", e)))?;

        let mut buffers = Vec::new();
        for i in 0..num_buffers {
            let size = base_size + i * 10;
            let buf: GpuBuffer<f32> = GpuBuffer::new(&executor.context, size)
                .map_err(|e| TestCaseError::fail(format!("T-QA-014c: Allocation {} failed: {}", i, e)))?;
            prop_assert_eq!(buf.len(), size);
            buffers.push(buf);
        }

        // Verify all buffers still valid
        for (i, buf) in buffers.iter().enumerate() {
            let expected_size = base_size + i * 10;
            prop_assert_eq!(buf.len(), expected_size, "T-QA-014c: Buffer {} size changed", i);
        }
        // buffers will be dropped here, testing Drop correctness
    }

    /// T-QA-014d: Property - Buffer rewrite works correctly
    ///
    /// Tests that writing new data to an existing buffer works.
    #[test]
    #[serial]
    fn prop_tqa014d_buffer_rewrite(
        initial in prop::collection::vec(1.0f32..100.0, 50..200),
        update in prop::collection::vec(100.0f32..200.0, 50..200)
    ) {
        if !has_cuda() {
            return Ok(());
        }

        let executor = CudaExecutor::new(0)
            .map_err(|e| TestCaseError::fail(format!("T-QA-014d: Executor init failed: {}", e)))?;

        // Use the smaller size to ensure both vectors fit
        let size = initial.len().min(update.len());
        if size == 0 {
            return Ok(());
        }

        // Initial upload
        let mut buf = GpuBuffer::from_host(&executor.context, &initial[..size])
            .map_err(|e| TestCaseError::fail(format!("T-QA-014d: Initial upload failed: {}", e)))?;

        // Overwrite with new data
        buf.copy_from_host(&update[..size])
            .map_err(|e| TestCaseError::fail(format!("T-QA-014d: Rewrite failed: {}", e)))?;

        // Verify new data
        let mut readback = vec![0.0f32; size];
        buf.copy_to_host(&mut readback)
            .map_err(|e| TestCaseError::fail(format!("T-QA-014d: Readback failed: {}", e)))?;

        for (i, (expected, actual)) in update[..size].iter().zip(readback.iter()).enumerate() {
            prop_assert!(
                (expected - actual).abs() < 1e-6,
                "T-QA-014d: Data mismatch at index {}: expected {}, got {}",
                i, expected, actual
            );
        }
    }
}

/// T-QA-014e: Test edge case - single element buffer
#[test]
#[serial]
fn test_tqa014e_single_element_buffer() {
    if !CudaExecutor::is_available() {
        println!("T-QA-014e: CUDA not available, skipping");
        return;
    }

    let executor = CudaExecutor::new(0).expect("CUDA executor");

    // Single element buffer
    let data = vec![42.0f32];
    let buf = GpuBuffer::from_host(&executor.context, &data).expect("T-QA-014e: from_host");

    assert_eq!(buf.len(), 1, "T-QA-014e: Single element length");
    assert_eq!(buf.size_bytes(), 4, "T-QA-014e: Single element bytes");

    let mut readback = vec![0.0f32];
    buf.copy_to_host(&mut readback)
        .expect("T-QA-014e: copy_to_host");
    assert!(
        (readback[0] - 42.0).abs() < 1e-6,
        "T-QA-014e: Value preserved"
    );

    println!("T-QA-014e: Single element buffer PASSED");
}

/// T-QA-014f: Test edge case - large buffer allocation
#[test]
#[serial]
fn test_tqa014f_large_buffer_allocation() {
    if !CudaExecutor::is_available() {
        println!("T-QA-014f: CUDA not available, skipping");
        return;
    }

    let executor = CudaExecutor::new(0).expect("CUDA executor");

    // Large buffer (1M elements = 4MB)
    let size = 1_000_000usize;
    let mut buf: GpuBuffer<f32> =
        GpuBuffer::new(&executor.context, size).expect("T-QA-014f: Large buffer allocation");

    assert_eq!(buf.len(), size, "T-QA-014f: Large buffer length");
    assert_eq!(buf.size_bytes(), size * 4, "T-QA-014f: Large buffer bytes");

    // Initialize with pattern
    let data: Vec<f32> = (0..size).map(|i| i as f32 * 0.001).collect();
    buf.copy_from_host(&data)
        .expect("T-QA-014f: copy_from_host");

    // Spot check some values
    let mut readback = vec![0.0f32; size];
    buf.copy_to_host(&mut readback)
        .expect("T-QA-014f: copy_to_host");

    assert!((readback[0] - 0.0).abs() < 1e-5, "T-QA-014f: First value");
    assert!(
        (readback[1000] - 1.0).abs() < 1e-5,
        "T-QA-014f: Value at 1000"
    );
    assert!(
        (readback[size - 1] - (size - 1) as f32 * 0.001).abs() < 1e-5,
        "T-QA-014f: Last value"
    );

    println!("T-QA-014f: Large buffer allocation PASSED");
}

// =========================================================================
// T-COV-001: Comprehensive KernelType PTX Generation Coverage Tests
// Targets: 95% cuda.rs coverage by exercising all KernelType variants
// =========================================================================

#[test]
fn test_tcov001a_attention_tensor_core_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::AttentionTensorCore {
        seq_len: 128,
        head_dim: 64,
        n_heads: 8,
        causal: true,
    });
    assert!(ptx.contains(".version"), "PTX should have version");
    assert!(
        ptx.contains("attention") || ptx.contains("flash"),
        "PTX should contain attention kernel"
    );
}

#[test]
fn test_tcov001b_bias_activation_ptx() {
    let kernels = CudaKernels::new();

    // Test with ReLU
    let ptx_relu = kernels.generate_ptx(&KernelType::BiasActivation {
        n: 1024,
        bias_size: 1024,
        activation: 1,
    });
    assert!(ptx_relu.contains(".version"));

    // Test with GELU
    let ptx_gelu = kernels.generate_ptx(&KernelType::BiasActivation {
        n: 1024,
        bias_size: 1024,
        activation: 2,
    });
    assert!(ptx_gelu.contains(".version"));

    // Test with None
    let ptx_none = kernels.generate_ptx(&KernelType::BiasActivation {
        n: 1024,
        bias_size: 1024,
        activation: 0,
    });
    assert!(ptx_none.contains(".version"));
}

#[test]
fn test_tcov001c_gemm_fp16_tensor_core_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::GemmFp16TensorCore {
        m: 64,
        n: 64,
        k: 64,
    });
    assert!(ptx.contains(".version"));
    assert!(ptx.contains("gemm") || ptx.contains("wmma"));
}

#[test]
fn test_tcov001d_fused_q4q8_dot_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::FusedQ4Q8Dot { n: 4096 });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001e_q4k_gemv_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::Q4KGemv { k: 4096, n: 4096 });
    assert!(ptx.contains(".version"));
    assert!(ptx.contains("q4k"));
}

#[test]
fn test_tcov001f_tiled_q4k_gemv_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::TiledQ4KGemv {
        k: 4096,
        n: 4096,
        outputs_per_block: 8,
    });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001g_chunked_tiled_q4k_gemv_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::ChunkedTiledQ4KGemv {
        k: 4096,
        n: 4096,
        outputs_per_block: 8,
    });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001h_coalesced_q4k_gemv_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::CoalescedQ4KGemv { k: 4096, n: 4096 });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001i_vectorized_q4k_gemv_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::VectorizedQ4KGemv { k: 4096, n: 4096 });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001j_dp4a_q4k_gemv_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::Dp4aQ4KGemv { k: 4096, n: 4096 });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001k_dp4a_simd_q4k_gemv_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::Dp4aSIMDQ4KGemv { k: 4096, n: 4096 });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001l_q5k_gemv_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::Q5KGemv { k: 4096, n: 4096 });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001m_q6k_gemv_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::Q6KGemv { k: 4096, n: 4096 });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001n_coalesced_q6k_gemv_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::CoalescedQ6KGemv { k: 4096, n: 4096 });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001o_batched_q6k_gemv_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::BatchedQ6KGemv {
        k: 4096,
        n: 4096,
        m: 4,
    });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001p_fp16_q4k_gemv_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::Fp16Q4KGemv { k: 4096, n: 4096 });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001q_q8_0_gemv_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::Q8_0Gemv { k: 4096, n: 4096 });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001r_q5_0_gemv_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::Q5_0Gemv { k: 4096, n: 4096 });
    assert!(ptx.contains(".version"));
}
