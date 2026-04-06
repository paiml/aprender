//! CORRECTNESS-002: Compare normed_hidden between CPU and GPU directly
//!
//! Run with: CUDA_GRAPH_DISABLE=1 cargo run --release --features cuda --example debug_normed_hidden_compare

use realizar::gguf::{
    MappedGGUFModel, OwnedQuantizedKVCache, OwnedQuantizedModel, OwnedQuantizedModelCuda,
};
use realizar::quantize::fused_q6k_parallel_matvec;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path =
        "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";

    eprintln!("Loading model...");
    let mapped = MappedGGUFModel::from_path(model_path)?;
    let cpu_model = OwnedQuantizedModel::from_mapped(&mapped)?;

    let hidden_dim = cpu_model.config().hidden_dim;
    let num_layers = cpu_model.config().num_layers;
    let num_heads = cpu_model.config().num_heads;
    let num_kv_heads = cpu_model.config().num_kv_heads;
    let vocab_size = cpu_model.config().vocab_size;
    let head_dim = hidden_dim / num_heads;
    let kv_dim = num_kv_heads * head_dim;

    let test_token: u32 = 791;
    eprintln!("Testing token {}\n", test_token);

    // CPU forward pass - capture normed_hidden before LM head
    eprintln!("=== CPU Forward ===");
    let mut cpu_cache = OwnedQuantizedKVCache::new(num_layers, kv_dim, 64);

    // We need to get the normed_hidden from CPU. Since we can't directly access it,
    // let's compute it: run forward, but also manually compute normed_hidden.
    // Actually, the forward returns logits. Let's compute what normed_hidden should be
    // by reversing: given logits, if we knew normed_hidden...
    // This is complex. Let's just get the full logits from both and compare.

    let cpu_logits = cpu_model.forward_single_with_cache(test_token, &mut cpu_cache, 0)?;
    let cpu_sum: f32 = cpu_logits.iter().sum();
    let cpu_argmax = cpu_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    eprintln!(
        "[CPU] Logits sum={:.2}, argmax={}, first 10={:?}",
        cpu_sum,
        cpu_argmax,
        &cpu_logits[..10]
    );

    // GPU forward pass
    eprintln!("\n=== GPU Forward ===");
    let mapped_gpu = MappedGGUFModel::from_path(model_path)?;
    let gpu_model = OwnedQuantizedModel::from_mapped(&mapped_gpu)?;
    let mut cuda_model = OwnedQuantizedModelCuda::new(gpu_model, 0)?;
    cuda_model.preload_weights_gpu()?;

    std::env::set_var("GPU_DEBUG", "1");
    let mut gpu_cache = OwnedQuantizedKVCache::new(num_layers, kv_dim, 64);
    let gpu_logits = cuda_model.forward_gpu_resident(test_token, &mut gpu_cache, 0)?;

    let gpu_sum: f32 = gpu_logits.iter().sum();
    let gpu_argmax = gpu_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    eprintln!(
        "[GPU] Logits sum={:.2}, argmax={}, first 10={:?}",
        gpu_sum,
        gpu_argmax,
        &gpu_logits[..10]
    );

    // Direct comparison
    eprintln!("\n=== Direct Comparison ===");

    // Compute correlation
    let mut dot = 0.0f64;
    let mut cpu_sq = 0.0f64;
    let mut gpu_sq = 0.0f64;
    let mut max_diff = 0.0f32;
    let mut max_diff_idx = 0;

    for i in 0..vocab_size {
        let c = cpu_logits[i] as f64;
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
        "Max diff: {:.4} at idx {} (CPU={:.4}, GPU={:.4})",
        max_diff, max_diff_idx, cpu_logits[max_diff_idx], gpu_logits[max_diff_idx]
    );

    // Sample comparison
    eprintln!("\nSample logits:");
    for i in [0, 1, 2, 100, 1000, 10000, vocab_size / 2, vocab_size - 1] {
        if i < vocab_size {
            eprintln!(
                "  [{}]: CPU={:.4}, GPU={:.4}, diff={:.4}",
                i,
                cpu_logits[i],
                gpu_logits[i],
                cpu_logits[i] - gpu_logits[i]
            );
        }
    }

    // Test LM head directly: use all-ones input
    eprintln!("\n=== Direct LM head test with all-ones input ===");
    let ones_input: Vec<f32> = vec![1.0; hidden_dim];

    // CPU LM head
    let cpu_lm_ones = fused_q6k_parallel_matvec(
        &cpu_model.lm_head_weight().data,
        &ones_input,
        hidden_dim,
        vocab_size,
    )?;

    let cpu_ones_sum: f32 = cpu_lm_ones.iter().sum();
    let cpu_ones_argmax = cpu_lm_ones
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    eprintln!(
        "[CPU LM head, ones input] sum={:.2}, argmax={}, first 5={:?}",
        cpu_ones_sum,
        cpu_ones_argmax,
        &cpu_lm_ones[..5]
    );

    // GPU LM head with ones - we need to call the kernel directly
    // For now, let's compare the results we already have

    // Check if there's a pattern in the errors
    eprintln!("\n=== Error pattern analysis ===");
    let mut pos_errors = 0;
    let mut neg_errors = 0;
    let mut error_sum = 0.0f64;

    for i in 0..vocab_size {
        let err = (gpu_logits[i] - cpu_logits[i]) as f64;
        error_sum += err;
        if err > 0.1 {
            pos_errors += 1;
        } else if err < -0.1 {
            neg_errors += 1;
        }
    }

    let avg_error = error_sum / vocab_size as f64;
    eprintln!("Average error: {:.4}", avg_error);
    eprintln!("Positive errors (>0.1): {}", pos_errors);
    eprintln!("Negative errors (<-0.1): {}", neg_errors);

    if corr > 0.99 {
        eprintln!("\n[OK] GPU matches CPU");
    } else {
        eprintln!("\n[FAIL] GPU diverges from CPU (corr={:.4})", corr);
    }

    Ok(())
}
