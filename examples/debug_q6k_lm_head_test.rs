//! CORRECTNESS-002: Direct Q6K LM head test with synthetic input
//!
//! Tests that GPU Q6K kernel produces same output as CPU for identical input.
//!
//! Run with: CUDA_GRAPH_DISABLE=1 cargo run --release --features cuda --example debug_q6k_lm_head_test

use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, OwnedQuantizedModelCuda};
use realizar::quantize::fused_q6k_parallel_matvec;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path =
        "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";

    eprintln!("Loading model...");
    let mapped = MappedGGUFModel::from_path(model_path)?;
    let cpu_model = OwnedQuantizedModel::from_mapped(&mapped)?;

    let hidden_dim = cpu_model.config().hidden_dim;
    let vocab_size = cpu_model.config().vocab_size;

    eprintln!(
        "Model config: hidden_dim={}, vocab_size={}",
        hidden_dim, vocab_size
    );
    eprintln!(
        "LM head: qtype={}, data_len={}",
        cpu_model.lm_head_weight().qtype,
        cpu_model.lm_head_weight().data.len()
    );

    // Create synthetic input (simple pattern for easier debugging)
    // Use a pattern that has distinct values per position
    let input_test: Vec<f32> = (0..hidden_dim)
        .map(|i| {
            // Oscillating pattern with position-dependent offset
            let base = (i as f32) * 0.001;
            let osc = ((i as f32) * 0.1).sin() * 0.5;
            base + osc
        })
        .collect();

    let input_sum: f32 = input_test.iter().sum();
    let input_rms: f32 =
        (input_test.iter().map(|x| x * x).sum::<f32>() / input_test.len() as f32).sqrt();
    eprintln!(
        "\nTest input: sum={:.4}, rms={:.4}, first 5={:?}",
        input_sum,
        input_rms,
        &input_test[..5]
    );

    // CPU reference: compute LM head with test input
    eprintln!("\n=== CPU Reference (fused_q6k_parallel_matvec) ===");
    let cpu_logits = fused_q6k_parallel_matvec(
        &cpu_model.lm_head_weight().data,
        &input_test,
        hidden_dim,
        vocab_size,
    )?;

    let cpu_sum: f32 = cpu_logits.iter().sum();
    let cpu_max = cpu_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let cpu_argmax = cpu_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    eprintln!(
        "[CPU] Logits: sum={:.4}, max={:.4}, argmax={}",
        cpu_sum, cpu_max, cpu_argmax
    );
    eprintln!("[CPU] First 10: {:?}", &cpu_logits[..10]);
    eprintln!("[CPU] Last 10: {:?}", &cpu_logits[vocab_size - 10..]);

    // GPU: Initialize and run the same computation
    eprintln!("\n=== GPU Q6K GEMV ===");
    let mapped_gpu = MappedGGUFModel::from_path(model_path)?;
    let gpu_model = OwnedQuantizedModel::from_mapped(&mapped_gpu)?;
    let mut cuda_model = OwnedQuantizedModelCuda::new(gpu_model, 0)?;

    // Preload just LM head weights
    eprintln!("Preloading weights to GPU...");
    cuda_model.preload_weights_gpu()?;

    // To test the Q6K kernel, we need to call a forward pass that uses it
    // Let's use a workaround - create a tiny forward that just does the LM head

    // Actually, let's verify the data is uploaded correctly first
    // Check the raw weight bytes match
    let lm_data_cpu = &cpu_model.lm_head_weight().data;
    eprintln!("LM head raw data first 40 bytes: {:?}", &lm_data_cpu[..40]);
    eprintln!(
        "LM head raw data last 40 bytes: {:?}",
        &lm_data_cpu[lm_data_cpu.len() - 40..]
    );

    // Compute expected bytes per row for Q6K
    let super_blocks_per_row = hidden_dim.div_ceil(256);
    let bytes_per_row = super_blocks_per_row * 210;
    eprintln!(
        "Super-blocks per row: {}, bytes per row: {}",
        super_blocks_per_row, bytes_per_row
    );
    eprintln!(
        "Total rows: {}, expected size: {}",
        vocab_size,
        vocab_size * bytes_per_row
    );
    eprintln!("Actual size: {}", lm_data_cpu.len());

    // Check row 0 of LM head
    let row0 = &lm_data_cpu[0..bytes_per_row];
    // Q6_K layout: ql (128) + qh (64) + scales (16) + d (2) = 210 bytes per super-block
    eprintln!("\nRow 0, super-block 0:");
    eprintln!("  ql[0..16]: {:?}", &row0[0..16]);
    eprintln!("  qh[0..16]: {:?}", &row0[128..144]);
    eprintln!("  scales: {:?}", &row0[192..208]);
    let d_bytes = &row0[208..210];
    let d_f16 = half::f16::from_bits(u16::from_le_bytes([d_bytes[0], d_bytes[1]]));
    eprintln!("  d (f16): {} -> f32: {}", d_f16, d_f16.to_f32());

    // Now run a full forward pass to get GPU logits
    // We'll use the forward path and compare the output
    eprintln!("\n=== Running GPU forward with test token to verify Q6K kernel ===");

    // Since we can't directly call the LM head kernel, let's run a forward pass
    // and compare the logits output
    let kv_dim = cpu_model.config().num_kv_heads * (hidden_dim / cpu_model.config().num_heads);
    let mut gpu_cache =
        realizar::gguf::OwnedQuantizedKVCache::new(cpu_model.config().num_layers, kv_dim, 64);

    // Enable debug
    std::env::set_var("GPU_DEBUG", "1");

    let test_token: u32 = 791;
    let gpu_logits = cuda_model.forward_gpu_resident(test_token, &mut gpu_cache, 0)?;
    let gpu_sum: f32 = gpu_logits.iter().sum();
    let gpu_max = gpu_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let gpu_argmax = gpu_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    eprintln!(
        "[GPU] Logits: sum={:.4}, max={:.4}, argmax={}",
        gpu_sum, gpu_max, gpu_argmax
    );

    // Also run CPU forward for comparison
    let mut cpu_cache =
        realizar::gguf::OwnedQuantizedKVCache::new(cpu_model.config().num_layers, kv_dim, 64);
    let cpu_forward_logits = cpu_model.forward_single_with_cache(test_token, &mut cpu_cache, 0)?;
    let cpu_forward_sum: f32 = cpu_forward_logits.iter().sum();
    let cpu_forward_argmax = cpu_forward_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    eprintln!(
        "[CPU forward] Logits: sum={:.4}, argmax={}",
        cpu_forward_sum, cpu_forward_argmax
    );

    // Element-wise comparison
    eprintln!("\n=== Comparison ===");
    let mut dot = 0.0f64;
    let mut cpu_sq = 0.0f64;
    let mut gpu_sq = 0.0f64;
    let mut max_diff = 0.0f32;
    let mut max_diff_idx = 0;

    for i in 0..vocab_size {
        let c = cpu_forward_logits[i] as f64;
        let g = gpu_logits[i] as f64;
        dot += c * g;
        cpu_sq += c * c;
        gpu_sq += g * g;
        let diff = (c - g).abs() as f32;
        if diff > max_diff {
            max_diff = diff;
            max_diff_idx = i;
        }
    }

    let corr = dot / (cpu_sq.sqrt() * gpu_sq.sqrt());
    eprintln!("Correlation: {:.6}", corr);
    eprintln!(
        "Max diff: {:.6} at idx {} (CPU={:.4}, GPU={:.4})",
        max_diff, max_diff_idx, cpu_forward_logits[max_diff_idx], gpu_logits[max_diff_idx]
    );

    if corr > 0.99 {
        eprintln!("\n[OK] GPU Q6K kernel matches CPU reference");
    } else if corr < 0.0 {
        eprintln!("\n[FAIL] Q6K kernel has NEGATIVE correlation - CORRECTNESS-002 confirmed");

        // Additional debug: print samples of CPU and GPU logits
        eprintln!("\nSample logits comparison:");
        for i in [0, 100, 1000, 10000, 50000, vocab_size - 1] {
            if i < vocab_size {
                eprintln!(
                    "  idx {}: CPU={:.4}, GPU={:.4}, diff={:.4}",
                    i,
                    cpu_forward_logits[i],
                    gpu_logits[i],
                    cpu_forward_logits[i] - gpu_logits[i]
                );
            }
        }
    } else {
        eprintln!("\n[FAIL] Q6K kernel diverges from CPU (corr={:.4})", corr);
    }

    Ok(())
}
