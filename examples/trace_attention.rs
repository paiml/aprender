//! Trace attention to find bug
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "/home/noah/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct-GGUF/snapshots/198f08841147e5196a6a69bd0053690fb1fd3857/qwen2-0_5b-instruct-q4_0.gguf";
    let mapped = MappedGGUFModel::from_path(path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;

    let hidden_dim = model.config().hidden_dim;
    let num_heads = model.config().num_heads;
    let num_kv_heads = model.config().num_kv_heads;
    let head_dim = hidden_dim / num_heads;
    let q_dim = num_heads * head_dim; // 896
    let k_dim = num_kv_heads * head_dim; // 128
    let v_dim = k_dim; // 128

    println!("=== Attention Trace ===\n");
    println!("Config:");
    println!("  hidden_dim: {}", hidden_dim);
    println!("  num_heads: {}, num_kv_heads: {}", num_heads, num_kv_heads);
    println!("  head_dim: {}", head_dim);
    println!("  q_dim: {}, k_dim: {}, v_dim: {}", q_dim, k_dim, v_dim);

    // Get embedding and apply attn norm
    let tok = 17u32;
    let emb_start = tok as usize * hidden_dim;
    let emb = &model.token_embedding()[emb_start..emb_start + hidden_dim];

    let sum_sq: f32 = emb.iter().map(|x| x * x).sum();
    let mean_sq = sum_sq / hidden_dim as f32;
    let inv_rms = 1.0 / (mean_sq + model.config().eps).sqrt();

    let layer = &model.layers()[0];
    let mut normed = vec![0.0f32; hidden_dim];
    for i in 0..hidden_dim {
        normed[i] = emb[i] * inv_rms * layer.attn_norm_weight[i];
    }

    // Simulate QKV computation
    // For single token, QKV output should be [Q..., K..., V...]
    // Q: 896 values, K: 128 values, V: 128 values = 1152 total

    // Just check the order and values after bias
    if let Some(ref bias) = layer.qkv_bias {
        println!("\nQKV bias structure:");
        println!("  Total bias len: {}", bias.len());

        // Q bias statistics
        let q_bias = &bias[0..q_dim];
        let k_bias = &bias[q_dim..q_dim + k_dim];
        let _v_bias = &bias[q_dim + k_dim..];

        println!("\n  Q bias ({} values):", q_bias.len());
        println!(
            "    max: {:.4}, min: {:.4}",
            q_bias.iter().copied().fold(f32::NEG_INFINITY, f32::max),
            q_bias.iter().copied().fold(f32::INFINITY, f32::min)
        );

        println!("\n  K bias ({} values):", k_bias.len());
        println!(
            "    max: {:.4}, min: {:.4}",
            k_bias.iter().copied().fold(f32::NEG_INFINITY, f32::max),
            k_bias.iter().copied().fold(f32::INFINITY, f32::min)
        );
        println!("    first 8: {:?}", &k_bias[..8]);

        // The scale for attention is 1/sqrt(head_dim) = 1/sqrt(64) = 0.125
        let scale = 1.0 / (head_dim as f32).sqrt();
        println!("\n  Attention scale: {:.4}", scale);

        // With K bias of ~100, after adding to K projections and computing Q·K/√d,
        // we'd get attention scores of ~100 * typical_q_value * scale
        // If typical_q_value is ~1-10, attention scores could be ~12-125
        // This would dominate softmax, causing attention to focus on wrong things

        // Check: What's the typical magnitude of K projections without bias?
        // If K without bias is ~1, then K with bias of 100 would be dominated by bias
    }

    // Let's check what the K projection looks like before and after bias
    // We need to compute the actual K projection

    // For now, just note that K biases up to 100+ are suspicious
    // Let me compare with llama.cpp to see how it handles this

    println!("\n\nNOTE: K bias has very large values (up to 152 in magnitude).");
    println!("This could be intentional for Qwen2, or could indicate a bug in loading.");
    println!("Need to compare with llama.cpp's handling.");

    Ok(())
}
