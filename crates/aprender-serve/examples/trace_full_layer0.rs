//! Trace full layer 0 for Qwen2 to find the bug
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
    let v_dim = k_dim; // 128

    println!("=== Full Layer 0 Trace ===\n");
    println!("Config:");
    println!("  hidden_dim: {}", hidden_dim);
    println!("  num_heads: {}, num_kv_heads: {}", num_heads, num_kv_heads);
    println!(
        "  head_dim: {}, q_dim: {}, k_dim: {}, v_dim: {}",
        head_dim, q_dim, k_dim, v_dim
    );

    // Token "2" (token ID 17)
    let tok = 17u32;

    // Step 1: Embedding
    let emb_start = tok as usize * hidden_dim;
    let hidden: Vec<f32> = model.token_embedding()[emb_start..emb_start + hidden_dim].to_vec();
    let hidden_norm: f32 = hidden.iter().map(|x| x * x).sum::<f32>().sqrt();
    println!("\nStep 1: Embedding");
    println!("  norm: {:.4}", hidden_norm);
    println!("  first 8: {:?}", &hidden[..8]);

    // Step 2: RMSNorm
    let layer = &model.layers()[0];
    let sum_sq: f32 = hidden.iter().map(|x| x * x).sum();
    let inv_rms = 1.0 / ((sum_sq / hidden_dim as f32) + model.config().eps).sqrt();
    let mut normed = vec![0.0f32; hidden_dim];
    for i in 0..hidden_dim {
        normed[i] = hidden[i] * inv_rms * layer.attn_norm_weight[i];
    }
    let normed_norm: f32 = normed.iter().map(|x| x * x).sum::<f32>().sqrt();
    println!("\nStep 2: RMSNorm");
    println!("  inv_rms: {:.6}", inv_rms);
    println!("  norm: {:.4}", normed_norm);
    println!("  first 8: {:?}", &normed[..8]);

    // Step 3: QKV projection
    let mut qkv = model.qkv_matmul(&normed, &layer.qkv_weight)?;
    let qkv_norm_before_bias: f32 = qkv.iter().map(|x| x * x).sum::<f32>().sqrt();
    println!("\nStep 3a: QKV projection (before bias)");
    println!("  len: {}, norm: {:.4}", qkv.len(), qkv_norm_before_bias);

    // Add bias
    if let Some(ref bias) = layer.qkv_bias {
        for i in 0..qkv.len() {
            qkv[i] += bias[i];
        }
    }
    let qkv_norm_after_bias: f32 = qkv.iter().map(|x| x * x).sum::<f32>().sqrt();
    println!("\nStep 3b: QKV projection (after bias)");
    println!("  norm: {:.4}", qkv_norm_after_bias);

    // Extract Q, K, V
    let q = qkv[0..q_dim].to_vec();
    let k = qkv[q_dim..q_dim + k_dim].to_vec();
    let v = qkv[q_dim + k_dim..].to_vec();

    println!("\nStep 3c: Q, K, V extracted");
    println!(
        "  Q norm: {:.4}, len: {}",
        q.iter().map(|x| x * x).sum::<f32>().sqrt(),
        q.len()
    );
    println!(
        "  K norm: {:.4}, len: {}",
        k.iter().map(|x| x * x).sum::<f32>().sqrt(),
        k.len()
    );
    println!(
        "  V norm: {:.4}, len: {}",
        v.iter().map(|x| x * x).sum::<f32>().sqrt(),
        v.len()
    );
    println!("  V first 8: {:?}", &v[..8]);

    // Step 4: For position 0, RoPE doesn't change values (cos=1, sin=0)
    // So attention output = expanded V for GQA
    println!("\nStep 4: Attention (position 0, seq_len=1)");
    println!("  For single token, softmax of single score = 1.0");
    println!("  So attention output = expanded V");

    // Expand V for GQA: each KV head serves 7 Q heads (14 Q / 2 KV = 7)
    let group_size = num_heads / num_kv_heads; // 7
    let mut attn_out = vec![0.0f32; q_dim];
    for h in 0..num_heads {
        let kv_head = h / group_size;
        let v_start = kv_head * head_dim;
        let out_start = h * head_dim;
        attn_out[out_start..out_start + head_dim].copy_from_slice(&v[v_start..v_start + head_dim]);
    }
    let attn_out_norm: f32 = attn_out.iter().map(|x| x * x).sum::<f32>().sqrt();
    println!(
        "  Expanded V norm: {:.4}, len: {}",
        attn_out_norm,
        attn_out.len()
    );
    println!("  Expanded V first 8: {:?}", &attn_out[..8]);

    // Step 5: Attention output projection
    println!("\nStep 5: Attention output projection");
    println!(
        "  attn_output_weight: in={}, out={}, qtype={}",
        layer.attn_output_weight.in_dim,
        layer.attn_output_weight.out_dim,
        layer.attn_output_weight.qtype
    );

    // Just verify dimensions match
    println!(
        "  Expected: in={} (q_dim), out={} (hidden_dim)",
        q_dim, hidden_dim
    );

    // Compare with model.forward() output for single token
    println!("\n=== Verify against model.forward() ===");
    let forward_logits = model.forward(&[tok])?;
    let mut indexed: Vec<_> = forward_logits.iter().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    println!("Top 5 predictions:");
    let vocab = mapped.model.vocabulary().expect("vocab");
    for (tok_id, logit) in indexed.iter().take(5) {
        let tok_str = vocab.get(*tok_id).map_or("?", |s| s.as_str());
        println!("  Token {} ({:?}): logit={:.4}", tok_id, tok_str, logit);
    }

    // Check the token "4" specifically
    println!("\nSpecific tokens:");
    println!("  Token 19 (\"4\"): logit={:.4}", forward_logits[19]);
    println!("  Token 17 (\"2\"): logit={:.4}", forward_logits[17]);
    println!("  Token 0 (\"!\"): logit={:.4}", forward_logits[0]);

    Ok(())
}
