//! Trace single layer attention scores
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "/home/noah/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct-GGUF/snapshots/198f08841147e5196a6a69bd0053690fb1fd3857/qwen2-0_5b-instruct-q4_0.gguf";
    let mapped = MappedGGUFModel::from_path(path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;

    let hidden_dim = model.config().hidden_dim;
    let num_heads = model.config().num_heads;
    let num_kv_heads = model.config().num_kv_heads;
    let head_dim = hidden_dim / num_heads; // 64
    let q_dim = num_heads * head_dim; // 896
    let k_dim = num_kv_heads * head_dim; // 128
    let _v_dim = k_dim; // 128

    println!("=== Single Layer Trace ===\n");

    // Get embedding
    let tok = 17u32; // "2"
    let emb_start = tok as usize * hidden_dim;
    let hidden = model.token_embedding()[emb_start..emb_start + hidden_dim].to_vec();

    let hidden_norm: f32 = hidden.iter().map(|x| x * x).sum::<f32>().sqrt();
    println!("Initial hidden: norm={:.4}", hidden_norm);

    // Process through layer 0 manually
    let layer = &model.layers()[0];

    // 1. RMSNorm
    let sum_sq: f32 = hidden.iter().map(|x| x * x).sum();
    let mean_sq = sum_sq / hidden_dim as f32;
    let inv_rms = 1.0 / (mean_sq + model.config().eps).sqrt();

    let mut normed = vec![0.0f32; hidden_dim];
    for i in 0..hidden_dim {
        normed[i] = hidden[i] * inv_rms * layer.attn_norm_weight[i];
    }

    let normed_norm: f32 = normed.iter().map(|x| x * x).sum::<f32>().sqrt();
    println!("After RMSNorm: norm={:.4}", normed_norm);

    // 2. QKV projection - use the model's method
    let qkv = model.qkv_matmul(&normed, &layer.qkv_weight)?;

    let qkv_norm: f32 = qkv.iter().map(|x| x * x).sum::<f32>().sqrt();
    println!(
        "After QKV projection: len={}, norm={:.4}",
        qkv.len(),
        qkv_norm
    );

    // Check Q, K, V separately
    let q_proj = &qkv[0..q_dim];
    let k_proj = &qkv[q_dim..q_dim + k_dim];
    let v_proj = &qkv[q_dim + k_dim..];

    let q_norm: f32 = q_proj.iter().map(|x| x * x).sum::<f32>().sqrt();
    let k_norm: f32 = k_proj.iter().map(|x| x * x).sum::<f32>().sqrt();
    let v_norm: f32 = v_proj.iter().map(|x| x * x).sum::<f32>().sqrt();

    println!(
        "  Q norm: {:.4}, K norm: {:.4}, V norm: {:.4}",
        q_norm, k_norm, v_norm
    );
    println!("  Q first 8: {:?}", &q_proj[..8]);
    println!("  K first 8: {:?}", &k_proj[..8]);

    // 3. Add bias
    let mut qkv_biased = qkv.clone();
    if let Some(ref bias) = layer.qkv_bias {
        for i in 0..qkv_biased.len() {
            qkv_biased[i] += bias[i];
        }
    }

    let q_biased = &qkv_biased[0..q_dim];
    let k_biased = &qkv_biased[q_dim..q_dim + k_dim];

    let q_biased_norm: f32 = q_biased.iter().map(|x| x * x).sum::<f32>().sqrt();
    let k_biased_norm: f32 = k_biased.iter().map(|x| x * x).sum::<f32>().sqrt();

    println!("\nAfter bias:");
    println!(
        "  Q norm: {:.4}, K norm: {:.4}",
        q_biased_norm, k_biased_norm
    );
    println!("  K first 8: {:?}", &k_biased[..8]);

    // The K values after bias should be dominated by bias if bias >> K_proj
    // K_proj norm was ~20, bias can add up to 152 per element
    // So K after bias could have norm ~several hundred

    // Check if K is now dominated by bias
    let k_bias = layer
        .qkv_bias
        .as_ref()
        .map(|b| &b[q_dim..q_dim + k_dim])
        .unwrap();
    let k_bias_norm: f32 = k_bias.iter().map(|x| x * x).sum::<f32>().sqrt();
    println!("\n  K_bias norm: {:.4}", k_bias_norm);
    println!("  K_bias / K_proj = {:.2}", k_bias_norm / k_norm);

    Ok(())
}
