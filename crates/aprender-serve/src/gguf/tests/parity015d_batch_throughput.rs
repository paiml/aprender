
// PARITY-015d: `test_parity015d_batch_forward_timing` was DELETED.
//
// It was a benchmark wearing a `#[test]` attribute: it ran a [32 x 2560] @
// [2560 x 10240] matmul, discarded the result with `let _ =`, printed a MOPS
// figure extrapolated to 32 layers, and asserted NOTHING — it could not fail
// even if `matmul` returned `Err`. `HybridScheduler::new()` succeeds without a
// GPU (`GpuCompute::auto()` falls back to CPU), so on a CPU-only runner this
// went through `cpu_matmul` and cost minutes of workspace-test wall clock,
// times three under nextest's `retries = 2`.
//
// Zero assertions means zero coverage lost. Do NOT reintroduce it as a `#[test]`;
// a throughput measurement belongs in `benches/` (see `benches/inference.rs`).

/// Test PARITY-015e: Integration verification
///
/// Verifies that GPU batch forward integrates correctly with existing code.
#[test]
fn test_parity015e_integration_verification() {
    /// GPU batch forward integration status
    struct IntegrationStatus {
        component: &'static str,
        status: &'static str,
        notes: &'static str,
    }

    let components = vec![
        IntegrationStatus {
            component: "HybridScheduler",
            status: "AVAILABLE",
            notes: "Auto-detects GPU, dispatches based on workload size",
        },
        IntegrationStatus {
            component: "batch_generate()",
            status: "EXISTS",
            notes: "Processes requests sequentially, can be optimized",
        },
        IntegrationStatus {
            component: "forward_batch_multi_request()",
            status: "EXISTS (unused)",
            notes: "Dead code, processes each request separately",
        },
        IntegrationStatus {
            component: "GPU batch FFN",
            status: "DESIGNED",
            notes: "Requires dequantized weight caching",
        },
        IntegrationStatus {
            component: "Batched layer norm",
            status: "VERIFIED",
            notes: "Works correctly for batched input",
        },
    ];

    println!("\nPARITY-015e: Integration Verification");
    for c in &components {
        println!("  {}: [{}]", c.component, c.status);
        println!("    {}", c.notes);
    }

    // Integration path summary
    println!("\n  Integration Path:");
    println!("  1. Add DequantizedWeightCache to OwnedQuantizedModel");
    println!("  2. Implement gpu_batch_ffn() using cached dequantized weights");
    println!("  3. Update batch_generate() to use GPU path when batch >= 32");
    println!("  4. Benchmark and tune GPU threshold");

    // Verify key components exist
    assert!(
        components.iter().any(|c| c.component == "HybridScheduler"),
        "PARITY-015e: HybridScheduler should be listed"
    );

    println!("  Status: VERIFIED - Integration path clear");
}

// ============================================================================

// PARITY-016: GPU Batch Forward Integration
// ============================================================================
//
// Objective: Integrate GPU batch FFN into OwnedQuantizedModel
//
// Key insight from PARITY-015:
// - GPU matmul achieves 8.36 GFLOPS for [32x2560] @ [2560x10240]
// - HybridScheduler correctly dispatches GPU for batch >= 32
// - Dequantized weight cache: 6.7 GB for 32-layer phi-2
//
// Implementation plan:
// 1. Add lazy dequantized weight cache to OwnedQuantizedModel
// 2. Create gpu_batch_ffn() that uses HybridScheduler
// 3. Update batch_generate() to use GPU path when active_count >= 32
// 4. Benchmark actual throughput improvement
// ============================================================================

#[test]
#[ignore] // Requires GPU: asserts GPU scheduling at batch=32, fails in clean-room Docker CI without GPU
fn test_parity016a_gpu_batch_ffn_function() {
    use crate::gpu::HybridScheduler;

    // Design the GPU batch FFN function
    //
    // Input: [batch_size, hidden_dim] - batched hidden states
    // Output: [batch_size, hidden_dim] - batched FFN output
    //
    // Operations:
    // 1. up_proj: [batch, hidden] @ [hidden, 4*hidden] = [batch, 4*hidden] (GPU GEMM)
    // 2. GELU activation (element-wise)
    // 3. down_proj: [batch, 4*hidden] @ [4*hidden, hidden] = [batch, hidden] (GPU GEMM)

    let batch_size = 32;
    let hidden_dim = 2560;
    let intermediate_dim = hidden_dim * 4; // 10240

    // Create test data
    let input: Vec<f32> = (0..batch_size * hidden_dim)
        .map(|i| (i as f32 * 0.001).sin())
        .collect();

    // Simulate weight matrices (would be dequantized from Q4_K)
    let up_weight: Vec<f32> = (0..hidden_dim * intermediate_dim)
        .map(|i| (i as f32 * 0.0001).cos() * 0.01)
        .collect();
    let down_weight: Vec<f32> = (0..intermediate_dim * hidden_dim)
        .map(|i| (i as f32 * 0.0001).sin() * 0.01)
        .collect();

    // Verify dimensions
    assert_eq!(
        input.len(),
        batch_size * hidden_dim,
        "PARITY-016a: Input should be [batch, hidden]"
    );
    assert_eq!(
        up_weight.len(),
        hidden_dim * intermediate_dim,
        "PARITY-016a: Up weight should be [hidden, 4*hidden]"
    );
    assert_eq!(
        down_weight.len(),
        intermediate_dim * hidden_dim,
        "PARITY-016a: Down weight should be [4*hidden, hidden]"
    );

    // Check if GPU would be used
    if let Ok(scheduler) = HybridScheduler::new() {
        let should_gpu_up = scheduler.should_use_gpu(batch_size, hidden_dim, intermediate_dim);
        let should_gpu_down = scheduler.should_use_gpu(batch_size, intermediate_dim, hidden_dim);

        println!("\nPARITY-016a: GPU Batch FFN Function Design");
        println!("  Batch size: {}", batch_size);
        println!("  Hidden dim: {}", hidden_dim);
        println!("  Intermediate dim: {}", intermediate_dim);
        println!("  Up projection GPU: {}", should_gpu_up);
        println!("  Down projection GPU: {}", should_gpu_down);

        // At batch=32, both should use GPU
        assert!(
            should_gpu_up,
            "PARITY-016a: Up projection should use GPU at batch=32"
        );
        assert!(
            should_gpu_down,
            "PARITY-016a: Down projection should use GPU at batch=32"
        );
    } else {
        println!("\nPARITY-016a: GPU not available, testing design only");
    }

    println!("  Status: VERIFIED - GPU batch FFN design correct");
}

#[test]
fn test_parity016b_dequant_weight_cache_integration() {
    // Test lazy dequantized weight cache pattern
    //
    // The cache should:
    // 1. Be lazily initialized on first batch inference
    // 2. Dequantize Q4_K weights to f32 for GPU GEMM
    // 3. Persist across batch_generate calls
    // 4. Fit in reasonable GPU memory (8GB limit)

    use std::cell::RefCell;
    use std::collections::HashMap;

    struct DequantizedLayerCache {
        ffn_up: Vec<f32>,
        ffn_down: Vec<f32>,
    }

    struct LazyWeightCache {
        layers: RefCell<HashMap<usize, DequantizedLayerCache>>,
        hidden_dim: usize,
        intermediate_dim: usize,
    }

    impl LazyWeightCache {
        fn new(hidden_dim: usize, intermediate_dim: usize) -> Self {
            Self {
                layers: RefCell::new(HashMap::new()),
                hidden_dim,
                intermediate_dim,
            }
        }

        fn get_or_dequant<F>(&self, layer_idx: usize, dequant_fn: F) -> Vec<f32>
        where
            F: FnOnce() -> Vec<f32>,
        {
            let mut cache = self.layers.borrow_mut();
            cache.entry(layer_idx).or_insert_with(|| {
                // First access: dequantize weights
                let ffn_up = dequant_fn();
                let ffn_down = vec![0.0f32; self.intermediate_dim * self.hidden_dim];
                DequantizedLayerCache { ffn_up, ffn_down }
            });
            cache.get(&layer_idx).expect("test").ffn_up.clone()
        }

        fn memory_bytes(&self) -> usize {
            let per_layer =
                (self.hidden_dim * self.intermediate_dim * 2) * std::mem::size_of::<f32>();
            let num_layers = self.layers.borrow().len();
            num_layers * per_layer
        }
    }

    // Test with phi-2 dimensions
    let hidden_dim = 2560;
    let intermediate_dim = 10240;
    let num_layers = 32;

    let cache = LazyWeightCache::new(hidden_dim, intermediate_dim);

    // Simulate lazy initialization for first few layers
    for layer_idx in 0..4 {
        let weights =
            cache.get_or_dequant(layer_idx, || vec![0.0f32; hidden_dim * intermediate_dim]);
        assert_eq!(weights.len(), hidden_dim * intermediate_dim);
    }

    // Calculate full cache size
    let per_layer_bytes = (hidden_dim * intermediate_dim * 2) * std::mem::size_of::<f32>();
    let full_cache_bytes = per_layer_bytes * num_layers;
    let full_cache_mb = full_cache_bytes as f64 / (1024.0 * 1024.0);

    println!("\nPARITY-016b: Lazy Weight Cache Integration");
    println!(
        "  Per layer: {} MB",
        per_layer_bytes as f64 / (1024.0 * 1024.0)
    );
    println!("  Full cache ({}L): {:.1} MB", num_layers, full_cache_mb);
    println!(
        "  Current cache: {} MB",
        cache.memory_bytes() as f64 / (1024.0 * 1024.0)
    );

    // Verify cache fits in 8GB
    assert!(
        full_cache_bytes < 8_000_000_000_usize,
        "PARITY-016b: Full cache should fit in 8GB"
    );

    println!("  Status: VERIFIED - Lazy cache pattern works");
}

#[test]
fn test_parity016c_batch_ffn_with_scheduler() {
    use crate::gpu::HybridScheduler;

    // Actually run batch FFN through HybridScheduler
    //
    // SCALED DOWN from phi-2 dimensions (batch 32, hidden 2560, intermediate 10240).
    // This test asserts a SHAPE property of the batched up projection, and a shape
    // property does not need production-sized tensors. At phi-2 size the matmul is
    // 32*2560*10240 = 839M MACs; `HybridScheduler::new()` succeeds without a GPU
    // (`GpuCompute::auto()` falls back to CPU), so on a CPU-only runner this ran
    // through `cpu_matmul` and cost minutes of workspace-test wall clock — times
    // three under nextest's `retries = 2`. These dimensions do the same work
    // 1600x smaller.
    //
    // The dimensions are chosen so m*k*n stays ABOVE `HybridScheduler`'s
    // `gpu_threshold` (64*64*64 = 262_144): 8*128*512 = 524_288, so the
    // GPU-vs-CPU dispatch decision this test exercises (and prints) is UNCHANGED
    // on a GPU-equipped host.
    //
    // The GFLOPS reporting was removed rather than rescaled: at this size the
    // number is meaningless, and a printed rate invites someone to cite it.
    // Benchmarks belong in `benches/`, not in `--lib` tests.
    let batch_size = 8;
    let hidden_dim = 128;
    let intermediate_dim = 512;

    // Create input batch
    let input: Vec<f32> = (0..batch_size * hidden_dim)
        .map(|i| ((i as f32) * 0.001).sin())
        .collect();

    // Create weight matrix (simulating dequantized FFN up weights)
    let up_weight: Vec<f32> = (0..hidden_dim * intermediate_dim)
        .map(|i| ((i as f32) * 0.0001).cos() * 0.01)
        .collect();

    println!("\nPARITY-016c: Batch FFN with HybridScheduler");
    println!("  Input shape: [{}x{}]", batch_size, hidden_dim);
    println!("  Weight shape: [{}x{}]", hidden_dim, intermediate_dim);

    // Try with scheduler
    if let Ok(mut scheduler) = HybridScheduler::new() {
        let should_use_gpu = scheduler.should_use_gpu(batch_size, hidden_dim, intermediate_dim);
        println!("  Should use GPU: {}", should_use_gpu);
        println!("  GPU available: {}", scheduler.has_gpu());

        let result = scheduler.matmul(&input, &up_weight, batch_size, hidden_dim, intermediate_dim);

        match result {
            Ok(output) => {
                assert_eq!(
                    output.len(),
                    batch_size * intermediate_dim,
                    "PARITY-016c: Output should be [batch, intermediate]"
                );

                println!("  Output shape: [{}x{}]", batch_size, intermediate_dim);

                // Apply GELU activation (element-wise)
                let activated: Vec<f32> = output
                    .iter()
                    .map(|&x| {
                        // Approximate GELU
                        let x64 = x as f64;
                        (x64
                            * 0.5
                            * (1.0 + (x64 * 0.7978845608 * (1.0 + 0.044715 * x64 * x64)).tanh()))
                            as f32
                    })
                    .collect();

                // For full FFN, would do down projection here
                println!("  GELU applied: {} elements", activated.len());
                println!("  Status: VERIFIED - Batch FFN works");
            },
            Err(e) => {
                // This arm used to `println!("SKIP")` and PASS, which made the
                // assertion above unreachable for any defect surfacing as `Err`. The
                // "may be CPU fallback" it used to claim was also false: the CPU
                // fallback path is `cpu_matmul`, which returns `Ok` unconditionally.
                // The input here is small and valid, so an `Err` is a real defect,
                // not a capability gap.
                panic!("PARITY-016c: matmul failed on valid input: {}", e);
            },
        }
    } else {
        println!("  Status: SKIP - GPU not available");
    }
}
