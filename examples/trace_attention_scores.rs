//! Trace attention scores directly
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "/home/noah/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct-GGUF/snapshots/198f08841147e5196a6a69bd0053690fb1fd3857/qwen2-0_5b-instruct-q4_0.gguf";
    let mapped = MappedGGUFModel::from_path(path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;

    let hidden_dim = model.config().hidden_dim;
    let num_heads = model.config().num_heads; // 14
    let num_kv_heads = model.config().num_kv_heads; // 2
    let head_dim = hidden_dim / num_heads; // 64
    let q_dim = num_heads * head_dim; // 896
    let k_dim = num_kv_heads * head_dim; // 128

    println!("=== Attention Score Trace ===\n");

    // Get embedding and process through layer 0
    let tok = 17u32;
    let emb_start = tok as usize * hidden_dim;
    let hidden = model.token_embedding()[emb_start..emb_start + hidden_dim].to_vec();

    let layer = &model.layers()[0];

    // RMSNorm
    let sum_sq: f32 = hidden.iter().map(|x| x * x).sum();
    let inv_rms = 1.0 / ((sum_sq / hidden_dim as f32) + model.config().eps).sqrt();
    let mut normed = vec![0.0f32; hidden_dim];
    for i in 0..hidden_dim {
        normed[i] = hidden[i] * inv_rms * layer.attn_norm_weight[i];
    }

    // QKV projection
    let mut qkv = model.qkv_matmul(&normed, &layer.qkv_weight)?;

    // Add bias
    if let Some(ref bias) = layer.qkv_bias {
        for i in 0..qkv.len() {
            qkv[i] += bias[i];
        }
    }

    // Extract Q, K (first position, seq_len=1)
    let q = &qkv[0..q_dim];
    let k = &qkv[q_dim..q_dim + k_dim];

    // Note: We should apply RoPE here, but for position 0, RoPE is identity
    // (cos=1, sin=0 for all frequencies at position 0)

    // Compute attention score for head 0
    // Q head 0: q[0..64]
    // K head 0 (with GQA): k[0..64] (K head 0 serves Q heads 0-6)

    let scale = 1.0 / (head_dim as f32).sqrt(); // 0.125

    // For each Q head, compute Q·K/√d
    println!("Attention scores (Q·K/√d) for each Q head:");
    println!(
        "(GQA: {} Q heads share each K head)\n",
        num_heads / num_kv_heads
    );

    for q_head in 0..num_heads {
        let kv_head = q_head / (num_heads / num_kv_heads); // 0-6 -> 0, 7-13 -> 1

        let q_start = q_head * head_dim;
        let k_start = kv_head * head_dim;

        let q_vec = &q[q_start..q_start + head_dim];
        let k_vec = &k[k_start..k_start + head_dim];

        // Q · K
        let dot: f32 = q_vec.iter().zip(k_vec.iter()).map(|(a, b)| a * b).sum();
        let score = dot * scale;

        let q_norm: f32 = q_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        let k_norm: f32 = k_vec.iter().map(|x| x * x).sum::<f32>().sqrt();

        println!(
            "  Head {}: score={:.2}, Q_norm={:.2}, K_norm={:.2}",
            q_head, score, q_norm, k_norm
        );
    }

    // For single position (seq_len=1), softmax([score]) = [1.0] always
    // So attention output = 1.0 * V

    println!("\nNote: For seq_len=1, softmax of single score = 1.0 always.");
    println!("So attention output = V directly.");

    let v = &qkv[q_dim + k_dim..];
    let v_norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    println!("\nV norm: {:.4}", v_norm);
    println!("V first 8: {:?}", &v[..8]);

    Ok(())
}
