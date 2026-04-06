//! Check RMSNorm parameters for the model
use realizar::gguf::MappedGGUFModel;

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        "../aprender/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf".to_string()
    });

    let mapped = MappedGGUFModel::from_path(&path).expect("load");

    println!("=== RMSNorm Parameters ===\n");

    // Get epsilon from metadata
    let arch = mapped.model.architecture().unwrap_or("unknown");
    println!("Architecture: {}", arch);

    let eps_key = format!("{}.attention.layer_norm_rms_epsilon", arch);
    println!("Looking for key: {}", eps_key);

    if let Some(val) = mapped.model.metadata.get(&eps_key) {
        println!("  Found: {:?}", val);
    } else {
        println!("  Not found, will use default 1e-5");
    }

    // Also check all metadata keys containing "epsilon" or "eps"
    println!("\n=== All epsilon-related metadata ===");
    for (key, val) in &mapped.model.metadata {
        if key.to_lowercase().contains("eps") || key.to_lowercase().contains("epsilon") {
            println!("  {}: {:?}", key, val);
        }
    }

    // Also check RMSNorm weight values
    println!("\n=== Attention Norm Weight (Layer 0) ===");
    for tensor in &mapped.model.tensors {
        if tensor.name.contains("blk.0") && tensor.name.contains("attn_norm") {
            println!(
                "  {}: dims={:?}, qtype={}",
                tensor.name, tensor.dims, tensor.qtype
            );

            // Load weight and show stats
            if let Ok(weight) = mapped.model.get_tensor_f32(&tensor.name, mapped.data()) {
                let min = weight.iter().copied().fold(f32::INFINITY, f32::min);
                let max = weight.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mean: f32 = weight.iter().sum::<f32>() / weight.len() as f32;
                println!(
                    "  Weight stats: min={:.6}, max={:.6}, mean={:.6}",
                    min, max, mean
                );
                println!("  First 5: {:?}", &weight[..5.min(weight.len())]);
            }
        }
    }

    // Compare RMSNorm computation
    println!("\n=== RMSNorm Manual Test ===");

    // Use the same input as compare_layer0 (token 791 embedding)
    let hidden_dim = mapped
        .model
        .metadata
        .iter()
        .find(|(k, _)| k.contains("embedding_length"))
        .and_then(|(_, v)| match v {
            realizar::gguf::GGUFValue::UInt32(n) => Some(*n as usize),
            _ => None,
        })
        .unwrap_or(1536);

    println!("Hidden dim: {}", hidden_dim);

    // Get embedding for token 791
    let embed_tensor = mapped
        .model
        .get_tensor_f32("token_embd.weight", mapped.data())
        .expect("embed");
    let token_id = 791usize;
    let embedding = &embed_tensor[token_id * hidden_dim..(token_id + 1) * hidden_dim];
    println!("Embedding[791] first 5: {:?}", &embedding[..5]);

    // Get norm weight
    let norm_weight = mapped
        .model
        .get_tensor_f32("blk.0.attn_norm.weight", mapped.data())
        .expect("norm");

    // Compute RMSNorm with different epsilon values
    for eps in [1e-5f32, 1e-6f32] {
        let sum_sq: f32 = embedding.iter().map(|x| x * x).sum();
        let mean_sq = sum_sq / hidden_dim as f32;
        let inv_rms = 1.0 / (mean_sq + eps).sqrt();

        let normed: Vec<f32> = embedding
            .iter()
            .zip(norm_weight.iter())
            .map(|(x, w)| (x * inv_rms) * w)
            .collect();

        println!(
            "\neps={:.0e}: normed first 3 = [{:.8}, {:.8}, {:.8}]",
            eps, normed[0], normed[1], normed[2]
        );
        println!("  inv_rms = {:.8}", inv_rms);
        println!("  mean_sq = {:.8}", mean_sq);
    }
}
