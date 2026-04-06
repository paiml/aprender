//! CORRECTNESS-002: Debug LM head divergence specifically
//!
//! Compare CPU and GPU logits with detailed analysis
//!
//! Run with: CUDA_GRAPH_DISABLE=1 cargo run --release --features cuda --example debug_lm_head_divergence

use realizar::gguf::{
    MappedGGUFModel, OwnedQuantizedKVCache, OwnedQuantizedModel, OwnedQuantizedModelCuda,
};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path =
        "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";

    eprintln!("Loading model...");
    let start = Instant::now();
    let mapped = MappedGGUFModel::from_path(model_path)?;
    let cpu_model = OwnedQuantizedModel::from_mapped(&mapped)?;
    eprintln!("Model loaded in {:?}", start.elapsed());

    // Get config
    let hidden_dim = cpu_model.config().hidden_dim;
    let num_layers = cpu_model.config().num_layers;
    let num_heads = cpu_model.config().num_heads;
    let num_kv_heads = cpu_model.config().num_kv_heads;
    let vocab_size = cpu_model.config().vocab_size;
    let head_dim = hidden_dim / num_heads;
    let kv_dim = num_kv_heads * head_dim;

    eprintln!(
        "\nConfig: hidden={}, layers={}, vocab={}",
        hidden_dim, num_layers, vocab_size
    );

    let test_token: u32 = 791;
    eprintln!("\n=== Testing token {} ===", test_token);

    // ===== CPU PATH =====
    eprintln!("\n--- CPU Path ---");
    let mut cpu_cache = OwnedQuantizedKVCache::new(num_layers, kv_dim, 64);
    let cpu_logits = cpu_model.forward_single_with_cache(test_token, &mut cpu_cache, 0)?;

    // Analyze CPU logits
    let cpu_sum: f32 = cpu_logits.iter().sum();
    let cpu_mean: f32 = cpu_sum / cpu_logits.len() as f32;
    let cpu_max = cpu_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let cpu_min = cpu_logits.iter().cloned().fold(f32::INFINITY, f32::min);
    let cpu_argmax = cpu_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    let cpu_var: f32 = cpu_logits
        .iter()
        .map(|&x| (x - cpu_mean).powi(2))
        .sum::<f32>()
        / cpu_logits.len() as f32;

    eprintln!("[CPU] Logits stats:");
    eprintln!(
        "  sum={:.2}, mean={:.6}, var={:.6}",
        cpu_sum, cpu_mean, cpu_var
    );
    eprintln!(
        "  min={:.4}, max={:.4}, argmax={}",
        cpu_min, cpu_max, cpu_argmax
    );
    eprintln!("  first 10: {:?}", &cpu_logits[..10]);
    eprintln!("  middle 10 (at 75000): {:?}", &cpu_logits[75000..75010]);
    eprintln!("  last 10: {:?}", &cpu_logits[vocab_size - 10..]);

    // ===== GPU PATH =====
    eprintln!("\n--- GPU Path ---");
    let mapped_gpu = MappedGGUFModel::from_path(model_path)?;
    let gpu_model = OwnedQuantizedModel::from_mapped(&mapped_gpu)?;
    let mut cuda_model = OwnedQuantizedModelCuda::new(gpu_model, 0)?;

    // Preload weights (no debug output)
    cuda_model.preload_weights_gpu()?;

    let mut gpu_cache = OwnedQuantizedKVCache::new(num_layers, kv_dim, 64);

    // Run GPU forward
    let gpu_logits = cuda_model.forward_gpu_resident(test_token, &mut gpu_cache, 0)?;

    // Analyze GPU logits
    let gpu_sum: f32 = gpu_logits.iter().sum();
    let gpu_mean: f32 = gpu_sum / gpu_logits.len() as f32;
    let gpu_max = gpu_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let gpu_min = gpu_logits.iter().cloned().fold(f32::INFINITY, f32::min);
    let gpu_argmax = gpu_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    let gpu_var: f32 = gpu_logits
        .iter()
        .map(|&x| (x - gpu_mean).powi(2))
        .sum::<f32>()
        / gpu_logits.len() as f32;

    eprintln!("[GPU] Logits stats:");
    eprintln!(
        "  sum={:.2}, mean={:.6}, var={:.6}",
        gpu_sum, gpu_mean, gpu_var
    );
    eprintln!(
        "  min={:.4}, max={:.4}, argmax={}",
        gpu_min, gpu_max, gpu_argmax
    );
    eprintln!("  first 10: {:?}", &gpu_logits[..10]);
    eprintln!("  middle 10 (at 75000): {:?}", &gpu_logits[75000..75010]);
    eprintln!("  last 10: {:?}", &gpu_logits[vocab_size - 10..]);

    // ===== Detailed comparison =====
    eprintln!("\n=== Element-wise Comparison ===");

    // Check correlation
    let mut dot_product = 0.0f64;
    let mut cpu_norm = 0.0f64;
    let mut gpu_norm = 0.0f64;
    let mut diff_sum = 0.0f64;
    let mut max_diff = 0.0f32;
    let mut max_diff_idx = 0;

    for i in 0..vocab_size {
        let c = cpu_logits[i] as f64;
        let g = gpu_logits[i] as f64;
        dot_product += c * g;
        cpu_norm += c * c;
        gpu_norm += g * g;
        let diff = (c - g).abs() as f32;
        diff_sum += diff as f64;
        if diff > max_diff {
            max_diff = diff;
            max_diff_idx = i;
        }
    }

    let correlation = dot_product / (cpu_norm.sqrt() * gpu_norm.sqrt());
    eprintln!("Correlation coefficient: {:.6}", correlation);
    eprintln!("Mean absolute diff: {:.6}", diff_sum / vocab_size as f64);
    eprintln!(
        "Max diff: {:.4} at index {} (CPU={:.4}, GPU={:.4})",
        max_diff, max_diff_idx, cpu_logits[max_diff_idx], gpu_logits[max_diff_idx]
    );

    // Check for sign flip or scaling issues
    let sign_match = cpu_logits
        .iter()
        .zip(gpu_logits.iter())
        .filter(|(c, g)| (c.signum() == g.signum()) || (c.abs() < 0.01 || g.abs() < 0.01))
        .count();
    eprintln!(
        "Sign agreement: {}/{} ({:.1}%)",
        sign_match,
        vocab_size,
        100.0 * sign_match as f64 / vocab_size as f64
    );

    // Final verdict
    eprintln!("\n=== Verdict ===");
    if correlation > 0.99 && cpu_argmax == gpu_argmax {
        eprintln!(
            "[OK] GPU matches CPU (corr={:.4}, argmax match)",
            correlation
        );
    } else if correlation > 0.9 {
        eprintln!("[WARN] GPU partially matches CPU (corr={:.4})", correlation);
    } else if correlation < 0.0 {
        eprintln!("[FAIL] GPU has NEGATIVE correlation with CPU - sign flip or major bug!");
    } else {
        eprintln!(
            "[FAIL] CORRECTNESS-002: GPU diverges from CPU (corr={:.4})",
            correlation
        );
    }

    Ok(())
}
