//! CORRECTNESS-011: Per-layer checksum trace to find divergence point
//!
//! Traces hidden state after each layer to find where GPU diverges from CPU.
//!
//! Run with: cargo run --example layer_by_layer_trace --release --features cuda

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("This example requires the 'cuda' feature. Run with --features cuda");
}

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use realizar::apr::MappedAprModel;
    use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, OwnedQuantizedModelCuda};

    let path = std::env::var("MODEL_PATH").unwrap_or_else(|_| {
        "/home/noah/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf".to_string()
    });

    println!("CORRECTNESS-011: Per-Layer Divergence Trace");
    println!("============================================");
    println!("Model: {}", path);

    // Detect format: APR, GGUF, or SafeTensors
    let data = std::fs::read(&path)?;
    let format = realizar::format::detect_format(&data[..8.min(data.len())])?;
    drop(data);
    println!("Format: {:?}", format);

    let model = match format {
        realizar::format::ModelFormat::Apr { .. } => {
            let mapped = MappedAprModel::from_path(std::path::Path::new(&path))?;
            OwnedQuantizedModel::from_apr(&mapped)?
        },
        _ => {
            let mapped = MappedGGUFModel::from_path(&path)?;
            OwnedQuantizedModel::from_mapped(&mapped)?
        },
    };

    let token_id = 791u32;
    let position: usize = 0;

    println!("\nToken ID: {}", token_id);
    println!("Position: {}", position);
    println!("Hidden dim: {}", model.config().hidden_dim);
    println!("Num layers: {}", model.config().num_layers);

    // ========================================================================
    // Phase 1: Get CPU embedding as baseline
    // ========================================================================
    println!("\n=== Phase 1: CPU Embedding ===");
    let cpu_embedding = model.embed(&[token_id]);

    // Compute checksum
    let cpu_embed_sum: f32 = cpu_embedding.iter().sum();
    let cpu_embed_sqsum: f32 = cpu_embedding.iter().map(|x| x * x).sum();
    let cpu_embed_rms = (cpu_embed_sqsum / cpu_embedding.len() as f32).sqrt();

    println!(
        "CPU embedding: first 5 = {:?}",
        &cpu_embedding[..5.min(cpu_embedding.len())]
    );
    println!(
        "CPU embedding: sum={:.6}, rms={:.6}",
        cpu_embed_sum, cpu_embed_rms
    );

    // ========================================================================
    // Phase 2: Get GPU embedding
    // ========================================================================
    println!("\n=== Phase 2: GPU Setup ===");

    let mut cuda_model = OwnedQuantizedModelCuda::new(model.clone(), 0)?;
    cuda_model.preload_weights_gpu()?;
    cuda_model.clear_decode_graph();
    cuda_model.enable_profiling();

    // Disable CUDA graphs for debugging
    std::env::set_var("CUDA_GRAPH_DISABLE", "1");
    // Enable per-layer debug output
    std::env::set_var("GPU_DEBUG_ALL_LAYERS", "1");

    println!("GPU executor ready");

    // ========================================================================
    // Phase 3: Run GPU forward with layer-by-layer trace
    // ========================================================================
    println!("\n=== Phase 3: GPU Forward (with layer trace) ===");

    let mut dummy_cache = realizar::gguf::OwnedQuantizedKVCache::new(
        model.config().num_layers,
        model.config().num_kv_heads * (model.config().hidden_dim / model.config().num_heads),
        100,
    );

    let gpu_logits = cuda_model.forward_gpu_resident(token_id, &mut dummy_cache, position)?;

    // ========================================================================
    // Phase 4: Compare final outputs
    // ========================================================================
    println!("\n=== Phase 4: Final Comparison ===");

    let cpu_logits = model.forward(&[token_id])?;

    let cpu_argmax = cpu_logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, v)| (i, *v));

    let gpu_argmax = gpu_logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, v)| (i, *v));

    println!("CPU argmax: {:?}", cpu_argmax);
    println!("GPU argmax: {:?}", gpu_argmax);

    // Compute correlation
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

        println!("\nStatistics:");
        println!("  Correlation: {:.6}", corr);
        println!("  Mean CPU: {:.6}, Mean GPU: {:.6}", mean_cpu, mean_gpu);
        println!("  Linear fit: GPU ≈ {:.4}*CPU + {:.4}", slope, intercept);

        // Check specific positions
        if let (Some((ci, cv)), Some((gi, gv))) = (cpu_argmax, gpu_argmax) {
            println!("\nArgmax analysis:");
            println!("  CPU[{}] = {:.6}", ci, cv);
            println!("  GPU[{}] = {:.6}", gi, gv);
            println!("  CPU[{}] = {:.6}", gi, cpu_logits.get(gi).unwrap_or(&0.0));
            println!("  GPU[{}] = {:.6}", ci, gpu_logits.get(ci).unwrap_or(&0.0));

            // What would CPU argmax value be under linear transform?
            let transformed_cpu_argmax = slope * cv + intercept;
            println!("\nLinear transform analysis:");
            println!(
                "  Expected GPU[{}] under linear transform: {:.6}",
                ci, transformed_cpu_argmax
            );
            println!(
                "  Actual GPU[{}]: {:.6}",
                ci,
                gpu_logits.get(ci).unwrap_or(&0.0)
            );
            println!(
                "  Residual: {:.6}",
                (gpu_logits.get(ci).unwrap_or(&0.0) - transformed_cpu_argmax).abs()
            );
        }

        // Find largest residuals (positions where GPU deviates most from linear transform)
        let mut residuals: Vec<(usize, f32)> = cpu_logits
            .iter()
            .zip(gpu_logits.iter())
            .enumerate()
            .map(|(i, (c, g))| {
                let predicted = slope * c + intercept;
                (i, (g - predicted).abs())
            })
            .collect();
        residuals.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        println!("\nTop 10 largest residuals (deviations from linear fit):");
        for (i, (idx, residual)) in residuals.iter().take(10).enumerate() {
            println!(
                "  {}: pos={}, residual={:.4}, CPU={:.4}, GPU={:.4}",
                i + 1,
                idx,
                residual,
                cpu_logits.get(*idx).unwrap_or(&0.0),
                gpu_logits.get(*idx).unwrap_or(&0.0)
            );
        }
    }

    // ========================================================================
    // Phase 5: Diagnosis
    // ========================================================================
    println!("\n=== Phase 5: Diagnosis ===");

    if cpu_argmax.map(|(i, _)| i) == gpu_argmax.map(|(i, _)| i) {
        println!("PASS: CPU and GPU argmax match");
    } else {
        println!("FAIL: Argmax mismatch");
        println!("\nLook at the layer-by-layer debug output above to find:");
        println!("1. Which layer first shows significant divergence?");
        println!("2. Is the divergence in RMSNorm, QKV, RoPE, Attention, or FFN?");
        println!("\nRoot cause per spec: 'Simplified trace omitted RoPE/Cache state management'");
        println!("Check if position encoding or KV cache handling differs between CPU and GPU.");
    }

    Ok(())
}
