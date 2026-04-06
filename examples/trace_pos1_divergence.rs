//! CORRECTNESS-013: Trace divergence at position 1 (after first token)
//!
//! At position 0, CPU and GPU match perfectly.
//! This test traces what happens at position 1 to identify where divergence starts.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("This example requires the 'cuda' feature. Run with --features cuda");
}

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use realizar::gguf::{
        MappedGGUFModel, OwnedQuantizedKVCache, OwnedQuantizedModel, OwnedQuantizedModelCuda,
    };

    let path = std::env::var("MODEL_PATH").unwrap_or_else(|_| {
        "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
            .to_string()
    });

    println!("CORRECTNESS-013: Position 1 Divergence Trace");
    println!("=============================================");
    println!("Model: {}", path);

    let mapped = MappedGGUFModel::from_path(&path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;

    let hidden_dim = model.config().hidden_dim;
    let num_layers = model.config().num_layers;
    let num_heads = model.config().num_heads;
    let num_kv_heads = model.config().num_kv_heads;
    let head_dim = hidden_dim / num_heads;
    let kv_dim = num_kv_heads * head_dim;

    // Use tokens [17, 10] which are "2" and "+"
    let token0 = 17u32; // "2"
    let token1 = 10u32; // "+"

    println!("\nTokens: {} (pos=0), {} (pos=1)", token0, token1);
    println!("Hidden dim: {}", hidden_dim);
    println!("Num layers: {}", num_layers);
    println!("KV dim: {}", kv_dim);

    // ========================================================================
    // CPU path: Process both tokens
    // ========================================================================
    println!("\n=== CPU Path ===");
    let mut cpu_cache = OwnedQuantizedKVCache::new(num_layers, kv_dim, 100);

    let cpu_logits_0 = model.forward_single_with_cache(token0, &mut cpu_cache, 0)?;
    let cpu_argmax_0 = cpu_logits_0
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, v)| (i, *v));
    println!("After pos 0: argmax = {:?}", cpu_argmax_0);

    let cpu_logits_1 = model.forward_single_with_cache(token1, &mut cpu_cache, 1)?;
    let cpu_argmax_1 = cpu_logits_1
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, v)| (i, *v));
    println!("After pos 1: argmax = {:?}", cpu_argmax_1);

    // ========================================================================
    // GPU path: Process both tokens
    // ========================================================================
    println!("\n=== GPU Path ===");

    // Disable CUDA graphs for debugging
    std::env::set_var("CUDA_GRAPH_DISABLE", "1");

    let mut cuda_model = OwnedQuantizedModelCuda::new(model.clone(), 0)?;
    cuda_model.preload_weights_gpu()?;
    cuda_model.clear_decode_graph();

    let mut gpu_cache = OwnedQuantizedKVCache::new(num_layers, kv_dim, 100);

    let gpu_logits_0 = cuda_model.forward_gpu_resident(token0, &mut gpu_cache, 0)?;
    let gpu_argmax_0 = gpu_logits_0
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, v)| (i, *v));
    println!("After pos 0: argmax = {:?}", gpu_argmax_0);

    let gpu_logits_1 = cuda_model.forward_gpu_resident(token1, &mut gpu_cache, 1)?;
    let gpu_argmax_1 = gpu_logits_1
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, v)| (i, *v));
    println!("After pos 1: argmax = {:?}", gpu_argmax_1);

    // ========================================================================
    // Comparison
    // ========================================================================
    println!("\n=== Comparison ===");

    // Position 0 comparison
    let corr_0 = correlation(&cpu_logits_0, &gpu_logits_0);
    println!(
        "Position 0: CPU={:?}, GPU={:?}, correlation={:.6}",
        cpu_argmax_0, gpu_argmax_0, corr_0
    );
    if cpu_argmax_0.map(|(i, _)| i) == gpu_argmax_0.map(|(i, _)| i) {
        println!("  ✅ Argmax MATCH at position 0");
    } else {
        println!("  ❌ Argmax DIFFER at position 0");
    }

    // Position 1 comparison
    let corr_1 = correlation(&cpu_logits_1, &gpu_logits_1);
    println!(
        "Position 1: CPU={:?}, GPU={:?}, correlation={:.6}",
        cpu_argmax_1, gpu_argmax_1, corr_1
    );
    if cpu_argmax_1.map(|(i, _)| i) == gpu_argmax_1.map(|(i, _)| i) {
        println!("  ✅ Argmax MATCH at position 1");
    } else {
        println!("  ❌ Argmax DIFFER at position 1");

        // Detailed comparison for position 1
        println!("\n  Detailed position 1 analysis:");
        println!(
            "  CPU logit at GPU choice: {:.6}",
            cpu_logits_1.get(gpu_argmax_1.unwrap().0).unwrap_or(&0.0)
        );
        println!(
            "  GPU logit at CPU choice: {:.6}",
            gpu_logits_1.get(cpu_argmax_1.unwrap().0).unwrap_or(&0.0)
        );

        // Compare distributions
        println!("\n  Logit statistics:");
        println!(
            "  CPU: min={:.4}, max={:.4}, mean={:.4}",
            cpu_logits_1.iter().cloned().fold(f32::INFINITY, f32::min),
            cpu_logits_1
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max),
            cpu_logits_1.iter().sum::<f32>() / cpu_logits_1.len() as f32
        );
        println!(
            "  GPU: min={:.4}, max={:.4}, mean={:.4}",
            gpu_logits_1.iter().cloned().fold(f32::INFINITY, f32::min),
            gpu_logits_1
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max),
            gpu_logits_1.iter().sum::<f32>() / gpu_logits_1.len() as f32
        );
    }

    Ok(())
}

fn correlation(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mean_a: f32 = a.iter().sum::<f32>() / a.len() as f32;
    let mean_b: f32 = b.iter().sum::<f32>() / b.len() as f32;
    let mut cov = 0.0f32;
    let mut var_a = 0.0f32;
    let mut var_b = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let da = x - mean_a;
        let db = y - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }
    cov / (var_a.sqrt() * var_b.sqrt() + 1e-10)
}
