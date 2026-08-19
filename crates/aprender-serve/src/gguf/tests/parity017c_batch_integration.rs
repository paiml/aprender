
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

// PARITY-017e: `test_parity017e_end_to_end_batch_throughput` was DELETED.
//
// It was a benchmark wearing a `#[test]` attribute: it allocated 4 layers of
// [2560 x 10240] f32 FFN weights (~840 MB), pushed a batch of 32 through
// up/GELU/down for every layer, then printed a tok/s number and asserted
// NOTHING. `HybridScheduler::new()` succeeds without a GPU (`GpuCompute::auto()`
// falls back to CPU), so the "GPU not available" arm never ran on a CPU-only
// runner and the whole thing went through `cpu_matmul` — >19 min of workspace-test
// wall clock, times three under nextest's `retries = 2`.
//
// Zero assertions means zero coverage lost. Do NOT reintroduce it as a `#[test]`;
// a throughput measurement belongs in `benches/` (see `benches/inference.rs`),
// where it is not on the critical path of every PR. The FFN shape/dispatch
// properties it nominally exercised are still covered by test_parity017a and
// test_parity018b.

// ============================================================================
