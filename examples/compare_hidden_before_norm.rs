//! CORRECTNESS-011: Compare hidden state BEFORE output_norm
//!
//! If hidden states match → bug is in output_norm or LM head
//! If hidden states differ → bug is in transformer layers (RoPE/Cache per spec)
//!
//! Run with: cargo run --example compare_hidden_before_norm --release --features cuda

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("This example requires the 'cuda' feature. Run with --features cuda");
}

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, OwnedQuantizedModelCuda};

    let path = std::env::var("MODEL_PATH").unwrap_or_else(|_| {
        "/home/noah/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf".to_string()
    });

    println!("CORRECTNESS-011: Hidden State Comparison Before Output Norm");
    println!("============================================================");
    println!("Model: {}", path);

    let mapped = MappedGGUFModel::from_path(&path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;

    let token_id = 791u32;
    let position: usize = 0;

    println!("\nToken ID: {}", token_id);
    println!("Position: {}", position);
    println!("Hidden dim: {}", model.config().hidden_dim);

    // ========================================================================
    // Get CPU hidden state before output_norm
    // We need to trace through the CPU forward path
    // ========================================================================
    println!("\n=== CPU Forward (tracing hidden state) ===");

    // Get embedding
    let embedding = model.embed(&[token_id]);
    println!("Embedding sum: {:.6}", embedding.iter().sum::<f32>());

    // Run CPU forward to get final logits
    let cpu_logits = model.forward(&[token_id])?;
    let cpu_argmax = cpu_logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, v)| (i, *v));
    println!("CPU argmax: {:?}", cpu_argmax);

    // ========================================================================
    // Get GPU hidden state
    // ========================================================================
    println!("\n=== GPU Forward ===");

    let mut cuda_model = OwnedQuantizedModelCuda::new(model.clone(), 0)?;
    cuda_model.preload_weights_gpu()?;
    cuda_model.clear_decode_graph();

    // Enable debug to get hidden state output
    std::env::set_var("GPU_DEBUG", "1");
    std::env::set_var("CUDA_GRAPH_DISABLE", "1");

    let mut dummy_cache = realizar::gguf::OwnedQuantizedKVCache::new(
        model.config().num_layers,
        model.config().num_kv_heads * (model.config().hidden_dim / model.config().num_heads),
        100,
    );

    let gpu_logits = cuda_model.forward_gpu_resident(token_id, &mut dummy_cache, position)?;
    let gpu_argmax = gpu_logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, v)| (i, *v));
    println!("GPU argmax: {:?}", gpu_argmax);

    // ========================================================================
    // Analysis: Check if the hidden states diverged or if it's the LM head
    // ========================================================================
    println!("\n=== Analysis ===");

    // The GPU_DEBUG output shows:
    // [CORRECTNESS-001] Hidden before output_norm: first 5 = [...], sum = ..., rms = ...
    // [CORRECTNESS-002] Normed hidden: first 5 = [...], sum = ..., rms = ...
    //
    // Compare these with CPU manually to determine if:
    // 1. Hidden before output_norm matches → bug is in norm or LM head
    // 2. Hidden before output_norm differs → bug is in transformer layers

    // Correlation analysis
    if cpu_logits.len() == gpu_logits.len() {
        let mean_cpu: f32 = cpu_logits.iter().sum::<f32>() / cpu_logits.len() as f32;
        let mean_gpu: f32 = gpu_logits.iter().sum::<f32>() / gpu_logits.len() as f32;

        let mut cov = 0.0f32;
        let mut var_cpu = 0.0f32;
        let mut var_gpu = 0.0f32;

        for (c, g) in cpu_logits.iter().zip(gpu_logits.iter()) {
            let dc = c - mean_cpu;
            let dg = g - mean_gpu;
            cov += dc * dg;
            var_cpu += dc * dc;
            var_gpu += dg * dg;
        }

        let corr = cov / (var_cpu.sqrt() * var_gpu.sqrt() + 1e-10);
        let slope = cov / (var_cpu + 1e-10);
        let intercept = mean_gpu - slope * mean_cpu;

        println!("Correlation: {:.6}", corr);
        println!("Linear fit: GPU ≈ {:.4}*CPU + {:.4}", slope, intercept);

        // Diagnosis based on slope
        if (slope - 1.0).abs() < 0.1 {
            println!("\nDIAGNOSIS: Slope ≈ 1.0, likely offset/bias issue");
        } else if slope < 1.0 {
            println!("\nDIAGNOSIS: Slope = {:.4} < 1.0", slope);
            println!("GPU output is scaled DOWN by {:.1}%", (1.0 - slope) * 100.0);
            println!("\nPossible causes:");
            println!("  1. Missing weight multiplication in RMSNorm");
            println!("  2. Wrong scaling in attention (1/sqrt(d_k))");
            println!("  3. Missing residual connection");
            println!("  4. Accumulated numerical error through layers");
        } else {
            println!("\nDIAGNOSIS: Slope = {:.4} > 1.0", slope);
            println!("GPU output is scaled UP by {:.1}%", (slope - 1.0) * 100.0);
        }

        // Check specific positions
        println!("\n=== Key Position Analysis ===");
        for pos in [16, 13, 15, 74403]
            .iter()
            .filter(|&&p| p < cpu_logits.len())
        {
            let cpu_val = cpu_logits[*pos];
            let gpu_val = gpu_logits[*pos];
            let predicted = slope * cpu_val + intercept;
            let residual = (gpu_val - predicted).abs();

            println!(
                "pos={}: CPU={:.4}, GPU={:.4}, predicted={:.4}, residual={:.4}",
                pos, cpu_val, gpu_val, predicted, residual
            );

            if residual > 5.0 {
                println!(
                    "  ^^ LARGE RESIDUAL - this position deviates significantly from linear fit!"
                );
            }
        }
    }

    // Result
    println!("\n=== Result ===");
    if cpu_argmax.map(|(i, _)| i) == gpu_argmax.map(|(i, _)| i) {
        println!("PASS: CPU and GPU argmax match");
    } else {
        println!("FAIL: Argmax mismatch");
        println!("\nNext steps:");
        println!("1. Check the [CORRECTNESS-001] debug output above");
        println!("2. Compare hidden state sum/rms with CPU");
        println!("3. If they differ, trace earlier layers");
        println!("4. Per spec ROOT CAUSE: 'Simplified trace omitted RoPE/Cache state management'");
    }

    Ok(())
}
