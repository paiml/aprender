//! Test LM head Q6K GEMV by comparing CPU vs GPU output for the same hidden state
//!
//! This test runs the full model up to the hidden state before output norm,
//! then compares CPU vs GPU for just the output norm + LM head projection.
//!
//! Run: CUDA_GRAPH_DISABLE=1 cargo run --release --features cuda --example test_lm_head_only

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("This example requires the 'cuda' feature. Run with: cargo run --features cuda --example test_lm_head_only");
}

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    cuda_main()
}

#[cfg(feature = "cuda")]
fn cuda_main() -> Result<(), Box<dyn std::error::Error>> {
    use realizar::gguf::{
        MappedGGUFModel, OwnedQuantizedKVCache, OwnedQuantizedModel, OwnedQuantizedModelCuda,
    };
    use std::time::Instant;
    let model_path =
        "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";

    // Load model for CPU
    eprintln!("Loading model...");
    let start = Instant::now();
    let mapped = MappedGGUFModel::from_path(model_path)?;
    let cpu_model = OwnedQuantizedModel::from_mapped(&mapped)?;
    eprintln!("Model loaded in {:?}", start.elapsed());

    let hidden_dim = cpu_model.config().hidden_dim;
    let num_layers = cpu_model.config().num_layers;
    let num_heads = cpu_model.config().num_heads;
    let num_kv_heads = cpu_model.config().num_kv_heads;
    let vocab_size = cpu_model.config().vocab_size;
    let head_dim = hidden_dim / num_heads;
    let kv_dim = num_kv_heads * head_dim;

    eprintln!("\nModel config:");
    eprintln!("  hidden_dim: {}", hidden_dim);
    eprintln!("  num_layers: {}", num_layers);
    eprintln!("  vocab_size: {}", vocab_size);

    // Test token
    let test_token: u32 = 791;
    eprintln!("\n=== Testing with token {} ===", test_token);

    // Get embedding
    let embedding = cpu_model.embed(&[test_token]);
    eprintln!("Embedding sum: {:.4}", embedding.iter().sum::<f32>());

    // CPU forward pass to get hidden state before output norm
    let mut cpu_cache = OwnedQuantizedKVCache::new(num_layers, kv_dim, 64);
    let logits_cpu = cpu_model.forward_single_with_cache(test_token, &mut cpu_cache, 0)?;

    let cpu_sum: f32 = logits_cpu.iter().sum();
    let cpu_argmax = logits_cpu
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    eprintln!("[CPU] Logits sum={:.4}, argmax={}", cpu_sum, cpu_argmax);
    eprintln!("[CPU] first 10: {:?}", &logits_cpu[..10]);

    // Now test GPU with manual LM head
    eprintln!("\n=== GPU Path ===");

    // Load for GPU
    let mapped_gpu = MappedGGUFModel::from_path(model_path)?;
    let gpu_model = OwnedQuantizedModel::from_mapped(&mapped_gpu)?;
    let mut cuda_model = OwnedQuantizedModelCuda::new(gpu_model, 0)?;

    // Preload weights
    let bytes = cuda_model.preload_weights_gpu()?;
    eprintln!("Uploaded {} MB to GPU", bytes / (1024 * 1024));

    let mut gpu_cache = OwnedQuantizedKVCache::new(num_layers, kv_dim, 64);
    let logits_gpu = cuda_model.forward_gpu_resident(test_token, &mut gpu_cache, 0)?;

    let gpu_sum: f32 = logits_gpu.iter().sum();
    let gpu_argmax = logits_gpu
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    eprintln!("[GPU] Logits sum={:.4}, argmax={}", gpu_sum, gpu_argmax);
    eprintln!("[GPU] first 10: {:?}", &logits_gpu[..10]);

    // Detailed comparison
    eprintln!("\n=== Detailed Comparison ===");

    // Calculate stats
    let mut max_diff = 0.0f32;
    let mut max_diff_idx = 0;
    let mut total_abs_diff = 0.0f32;

    for i in 0..vocab_size {
        let diff = (logits_cpu[i] - logits_gpu[i]).abs();
        total_abs_diff += diff;
        if diff > max_diff {
            max_diff = diff;
            max_diff_idx = i;
        }
    }

    let avg_diff = total_abs_diff / vocab_size as f32;
    eprintln!("Max diff: {:.6} at token {}", max_diff, max_diff_idx);
    eprintln!("Avg diff: {:.6}", avg_diff);

    // Correlation
    let cpu_mean = cpu_sum / vocab_size as f32;
    let gpu_mean = gpu_sum / vocab_size as f32;
    let mut cov = 0.0f32;
    let mut cpu_var = 0.0f32;
    let mut gpu_var = 0.0f32;
    for i in 0..vocab_size {
        let cpu_d = logits_cpu[i] - cpu_mean;
        let gpu_d = logits_gpu[i] - gpu_mean;
        cov += cpu_d * gpu_d;
        cpu_var += cpu_d * cpu_d;
        gpu_var += gpu_d * gpu_d;
    }
    let corr = if cpu_var > 0.0 && gpu_var > 0.0 {
        cov / (cpu_var.sqrt() * gpu_var.sqrt())
    } else {
        0.0
    };
    eprintln!("Correlation: {:.4}", corr);

    // Show values at argmax positions
    eprintln!("\nAt CPU argmax (token {}):", cpu_argmax);
    eprintln!("  CPU: {:.6}", logits_cpu[cpu_argmax]);
    eprintln!("  GPU: {:.6}", logits_gpu[cpu_argmax]);

    eprintln!("\nAt GPU argmax (token {}):", gpu_argmax);
    eprintln!("  CPU: {:.6}", logits_cpu[gpu_argmax]);
    eprintln!("  GPU: {:.6}", logits_gpu[gpu_argmax]);

    // Check some specific tokens where they might differ
    eprintln!("\nSample tokens (0, 100, 1000, 10000, 50000):");
    for tok in [0, 100, 1000, 10000, 50000] {
        if tok < vocab_size {
            eprintln!(
                "  Token {:5}: CPU={:8.4}, GPU={:8.4}, diff={:8.4}",
                tok,
                logits_cpu[tok],
                logits_gpu[tok],
                logits_cpu[tok] - logits_gpu[tok]
            );
        }
    }

    // Find where they agree and disagree most
    let mut tokens_by_diff: Vec<(usize, f32)> = (0..vocab_size)
        .map(|i| (i, (logits_cpu[i] - logits_gpu[i]).abs()))
        .collect();
    tokens_by_diff.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    eprintln!("\nTop 10 tokens with largest diff:");
    for &(tok, diff) in tokens_by_diff.iter().take(10) {
        eprintln!(
            "  Token {:6}: CPU={:8.4}, GPU={:8.4}, diff={:8.4}",
            tok, logits_cpu[tok], logits_gpu[tok], diff
        );
    }

    // Final verdict
    if corr > 0.99 && max_diff < 0.5 {
        eprintln!("\n✅ CPU and GPU match closely!");
    } else if corr > 0.9 {
        eprintln!("\n⚠️ Mostly correlated but with errors (corr={:.4})", corr);
    } else {
        eprintln!("\n❌ CPU and GPU produce very different output!");
        eprintln!("   This indicates a bug in the GPU forward pass.");
    }

    Ok(())
}
