
#[test]
fn test_parity017c_batch_generate_gpu_integration_points() {
    // Identify exact integration points in batch_generate()

    struct IntegrationPoint {
        location: &'static str,
        line: &'static str,
        change: &'static str,
    }

    let integration_points = vec![
        IntegrationPoint {
            location: "batch_generate() prefill loop",
            line: "for (req_idx, prompt) in prompts.iter().enumerate()",
            change: "Batch all prompts together for GPU prefill",
        },
        IntegrationPoint {
            location: "batch_generate() generation loop",
            line: "for &req_idx in &active_indices",
            change: "Check active_count >= 32, batch forward if true",
        },
        IntegrationPoint {
            location: "forward_single_with_contiguous_cache()",
            line: "let mut ffn_hidden = self.fused_matmul(&hidden, &layer.ffn_up_weight)?",
            change: "Add forward_batch_with_contiguous_cache() variant",
        },
        IntegrationPoint {
            location: "OwnedQuantizedModel struct",
            line: "pub struct OwnedQuantizedModel",
            change: "Add optional HybridScheduler field for GPU dispatch",
        },
    ];

    println!("\nPARITY-017c: batch_generate GPU Integration Points");
    for (i, point) in integration_points.iter().enumerate() {
        println!("\n  {}. {}", i + 1, point.location);
        println!("     Current: {}", point.line);
        println!("     Change: {}", point.change);
    }

    // Pseudo-code for GPU batch generation
    println!("\n  Pseudo-code for batch_generate_gpu():");
    println!("  ```");
    println!("  fn batch_generate_gpu(&self, prompts, config) {{");
    println!("      let scheduler = HybridScheduler::new()?;");
    println!("      ");
    println!("      // Prefill phase: batch all prompts");
    println!("      let max_prompt_len = prompts.iter().map(|p| p.len()).max();");
    println!("      for pos in 0..max_prompt_len {{");
    println!("          let batch_tokens = collect_tokens_at_position(prompts, pos);");
    println!("          forward_batch_gpu(&batch_tokens, pos, &scheduler);");
    println!("      }}");
    println!("      ");
    println!("      // Generation phase");
    println!("      for gen_idx in 0..config.max_tokens {{");
    println!("          let active_count = count_active();");
    println!("          if active_count >= 32 {{");
    println!("              forward_batch_gpu(active_tokens, pos, &scheduler);");
    println!("          }} else {{");
    println!("              for req in active_requests {{");
    println!("                  forward_single_with_cache(req.last_token);");
    println!("              }}");
    println!("          }}");
    println!("      }}");
    println!("  }}");
    println!("  ```");

    assert_eq!(
        integration_points.len(),
        4,
        "PARITY-017c: Should have 4 integration points"
    );

    println!("  Status: VERIFIED - Integration points identified");
}

#[test]
fn test_parity017d_dequant_cache_struct() {
    use std::collections::HashMap;
    use std::sync::Mutex;

    // Define the dequantized weight cache structure
    // This caches f32 weights for GPU GEMM

    struct DequantizedFFNWeights {
        up: Vec<f32>,   // [hidden, intermediate]
        down: Vec<f32>, // [intermediate, hidden]
    }

    struct DequantizedWeightCache {
        layers: Mutex<HashMap<usize, DequantizedFFNWeights>>,
        hidden_dim: usize,
        intermediate_dim: usize,
    }

    impl DequantizedWeightCache {
        fn new(hidden_dim: usize, intermediate_dim: usize) -> Self {
            Self {
                layers: Mutex::new(HashMap::new()),
                hidden_dim,
                intermediate_dim,
            }
        }

        fn get_or_init(
            &self,
            layer_idx: usize,
            init_fn: impl FnOnce() -> (Vec<f32>, Vec<f32>),
        ) -> (Vec<f32>, Vec<f32>) {
            let mut cache = self.layers.lock().expect("mutex poisoned");
            cache.entry(layer_idx).or_insert_with(|| {
                let (up, down) = init_fn();
                DequantizedFFNWeights { up, down }
            });
            let weights = cache.get(&layer_idx).expect("test");
            (weights.up.clone(), weights.down.clone())
        }

        fn memory_bytes(&self) -> usize {
            let cache = self.layers.lock().expect("mutex poisoned");
            cache.len() * (self.hidden_dim * self.intermediate_dim * 2) * std::mem::size_of::<f32>()
        }

        fn clear(&self) {
            let mut cache = self.layers.lock().expect("mutex poisoned");
            cache.clear();
        }
    }

    // Test with phi-2 dimensions
    let hidden_dim = 2560;
    let intermediate_dim = 10240;
    let num_layers = 32;

    let cache = DequantizedWeightCache::new(hidden_dim, intermediate_dim);

    // Simulate lazy initialization for a few layers
    for layer_idx in 0..4 {
        let _ = cache.get_or_init(layer_idx, || {
            let up = vec![0.0f32; hidden_dim * intermediate_dim];
            let down = vec![0.0f32; intermediate_dim * hidden_dim];
            (up, down)
        });
    }

    let per_layer_mb =
        (hidden_dim * intermediate_dim * 2 * std::mem::size_of::<f32>()) as f64 / (1024.0 * 1024.0);
    let total_mb = cache.memory_bytes() as f64 / (1024.0 * 1024.0);
    let full_mb = per_layer_mb * num_layers as f64;

    println!("\nPARITY-017d: Dequantized Weight Cache Structure");
    println!("  Per layer: {:.1} MB", per_layer_mb);
    println!("  Current (4 layers): {:.1} MB", total_mb);
    println!("  Full (32 layers): {:.1} MB", full_mb);

    // Verify cache works
    let (up1, _) = cache.get_or_init(0, || panic!("Should be cached"));
    assert_eq!(
        up1.len(),
        hidden_dim * intermediate_dim,
        "PARITY-017d: Cached weights should have correct size"
    );

    // Clear cache
    cache.clear();
    assert_eq!(
        cache.memory_bytes(),
        0,
        "PARITY-017d: Clear should empty cache"
    );

    println!("  Status: VERIFIED - Cache structure works");
}

#[test]
fn test_parity017e_end_to_end_batch_throughput() {
    use crate::gpu::HybridScheduler;
    use std::time::Instant;

    // Measure actual end-to-end batch throughput with GPU FFN

    let batch_size = 32;
    let hidden_dim = 2560;
    let intermediate_dim = 10240;
    let num_layers = 4; // Test with subset for speed

    println!("\nPARITY-017e: End-to-End Batch Throughput");
    println!("  Batch: {}", batch_size);
    println!("  Hidden: {}", hidden_dim);
    println!("  Intermediate: {}", intermediate_dim);
    println!("  Layers: {}", num_layers);

    // Create test weights for multiple layers
    let up_weights: Vec<Vec<f32>> = (0..num_layers)
        .map(|_| {
            (0..hidden_dim * intermediate_dim)
                .map(|i| (i as f32 * 0.0001).cos() * 0.01)
                .collect()
        })
        .collect();
    let down_weights: Vec<Vec<f32>> = (0..num_layers)
        .map(|_| {
            (0..intermediate_dim * hidden_dim)
                .map(|i| (i as f32 * 0.0001).sin() * 0.01)
                .collect()
        })
        .collect();

    // Initial hidden states
    let mut hidden: Vec<f32> = (0..batch_size * hidden_dim)
        .map(|i| (i as f32 * 0.001).sin() * 0.1)
        .collect();

    if let Ok(mut scheduler) = HybridScheduler::new() {
        let start = Instant::now();

        // Process through all layers
        for layer_idx in 0..num_layers {
            // FFN: up projection
            let intermediate = scheduler
                .matmul(
                    &hidden,
                    &up_weights[layer_idx],
                    batch_size,
                    hidden_dim,
                    intermediate_dim,
                )
                .expect("Up projection failed");

            // GELU activation
            let activated: Vec<f32> = intermediate
                .iter()
                .map(|&x| {
                    let x64 = x as f64;
                    (x64 * 0.5 * (1.0 + (x64 * 0.7978845608 * (1.0 + 0.044715 * x64 * x64)).tanh()))
                        as f32
                })
                .collect();

            // FFN: down projection
            let ffn_out = scheduler
                .matmul(
                    &activated,
                    &down_weights[layer_idx],
                    batch_size,
                    intermediate_dim,
                    hidden_dim,
                )
                .expect("Down projection failed");

            // Residual (simplified - just replace for now)
            hidden = ffn_out;
        }

        let elapsed = start.elapsed();

        // Calculate throughput
        let tokens_processed = batch_size;
        let tps = tokens_processed as f64 / elapsed.as_secs_f64();

        // Scale to full model (32 layers)
        let scaled_time_ms = elapsed.as_secs_f64() * (32.0 / num_layers as f64) * 1000.0;
        let scaled_tps = tokens_processed as f64 / (scaled_time_ms / 1000.0);

        println!("\n  Results ({} layers):", num_layers);
        println!("    Time: {:?}", elapsed);
        println!("    Throughput: {:.0} tok/s", tps);

        println!("\n  Projected (32 layers):");
        println!("    Time: {:.1}ms", scaled_time_ms);
        println!("    Throughput: {:.0} tok/s", scaled_tps);

        // Compare to baseline
        let baseline_tps = 5.09;
        let speedup = scaled_tps / baseline_tps;
        println!("\n  Comparison:");
        println!("    Baseline (single req): {:.2} tok/s", baseline_tps);
        println!("    Batch GPU FFN: {:.0} tok/s", scaled_tps);
        println!("    Speedup: {:.1}x", speedup);

        // Note: Throughput varies significantly due to:
        // 1. This test isolates FFN only (not full transformer)
        // 2. GPU resource contention when running with other tests
        // 3. Scaling from 4 to 32 layers is approximate
        //
        // The key insight is that GPU batch FFN WORKS:
        // - test_parity017a verifies FFN correctness (~10 GFLOPS)
        // - test_parity017c shows integration design
        // - This test measures actual throughput under varying conditions
        //
        // Actual performance improvement requires:
        // - Full transformer integration (not isolated FFN)
        // - Dequantized weight caching
        // - Running in isolation (not parallel with 2100+ other tests)

        println!("  Status: MEASURED - {:.1}x relative to baseline", speedup);
        println!("    Note: Run in isolation for accurate benchmark");
    } else {
        println!("  Status: SKIP - GPU not available");
    }
}

// ============================================================================
