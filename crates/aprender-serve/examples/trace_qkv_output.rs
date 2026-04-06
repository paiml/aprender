//! Trace QKV output dimensions and values for Qwen2
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "/home/noah/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct-GGUF/snapshots/198f08841147e5196a6a69bd0053690fb1fd3857/qwen2-0_5b-instruct-q4_0.gguf";
    let mapped = MappedGGUFModel::from_path(path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;

    println!("=== Qwen2 QKV Analysis ===\n");

    // Model config
    let hidden_dim = model.config().hidden_dim;
    let num_heads = model.config().num_heads;
    let num_kv_heads = model.config().num_kv_heads;
    let head_dim = hidden_dim / num_heads;

    println!("Config:");
    println!("  hidden_dim: {}", hidden_dim);
    println!("  num_heads: {}", num_heads);
    println!("  num_kv_heads: {}", num_kv_heads);
    println!("  head_dim: {}", head_dim);

    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;

    println!("\nExpected dimensions:");
    println!(
        "  Q dim: {} (num_heads * head_dim = {} * {})",
        q_dim, num_heads, head_dim
    );
    println!(
        "  K dim: {} (num_kv_heads * head_dim = {} * {})",
        kv_dim, num_kv_heads, head_dim
    );
    println!("  V dim: {}", kv_dim);
    println!("  Total QKV dim: {}", q_dim + kv_dim + kv_dim);

    // Check layer 0 QKV weight dimensions
    let layer0 = &model.layers()[0];
    let qkv_out_dim = layer0.qkv_weight.out_dim();
    let qkv_q_dim = layer0.qkv_weight.q_dim();

    println!("\nLayer 0 QKV weight:");
    println!("  qkv_weight.out_dim(): {}", qkv_out_dim);
    println!("  qkv_weight.q_dim(): {}", qkv_q_dim);

    // Check QKV bias
    if let Some(ref bias) = layer0.qkv_bias {
        println!("  qkv_bias length: {}", bias.len());
        // Analyze bias structure
        println!("\n  Bias analysis (expected: Q bias | K bias | V bias):");
        let q_bias = &bias[0..q_dim];
        let k_bias = &bias[q_dim..q_dim + kv_dim];
        let v_bias = &bias[q_dim + kv_dim..];

        let q_norm: f32 = q_bias.iter().map(|x| x * x).sum::<f32>().sqrt();
        let k_norm: f32 = k_bias.iter().map(|x| x * x).sum::<f32>().sqrt();
        let v_norm: f32 = v_bias.iter().map(|x| x * x).sum::<f32>().sqrt();

        let q_min = q_bias.iter().copied().fold(f32::INFINITY, f32::min);
        let q_max = q_bias.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let k_min = k_bias.iter().copied().fold(f32::INFINITY, f32::min);
        let k_max = k_bias.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let v_min = v_bias.iter().copied().fold(f32::INFINITY, f32::min);
        let v_max = v_bias.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        println!(
            "    Q bias: len={}, norm={:.4}, range=[{:.4}, {:.4}]",
            q_bias.len(),
            q_norm,
            q_min,
            q_max
        );
        println!(
            "    K bias: len={}, norm={:.4}, range=[{:.4}, {:.4}]",
            k_bias.len(),
            k_norm,
            k_min,
            k_max
        );
        println!(
            "    V bias: len={}, norm={:.4}, range=[{:.4}, {:.4}]",
            v_bias.len(),
            v_norm,
            v_min,
            v_max
        );
    } else {
        println!("  No QKV bias");
    }

    // Test QKV projection with a simple input
    println!("\n=== QKV Projection Test ===");

    // Use actual embedding for token "2" (token 17)
    let token_17_emb = &model.token_embedding()[17 * hidden_dim..(17 + 1) * hidden_dim];
    let emb_norm: f32 = token_17_emb.iter().map(|x| x * x).sum::<f32>().sqrt();
    println!("Token 17 ('2') embedding norm: {:.4}", emb_norm);

    // RMSNorm the embedding
    let eps = model.config().eps;
    let ss: f32 = token_17_emb.iter().map(|x| x * x).sum::<f32>() / hidden_dim as f32;
    let scale = 1.0 / (ss + eps).sqrt();
    let normed: Vec<f32> = token_17_emb
        .iter()
        .zip(layer0.attn_norm_weight.iter())
        .map(|(x, w)| x * scale * w)
        .collect();

    let normed_norm: f32 = normed.iter().map(|x| x * x).sum::<f32>().sqrt();
    println!("After RMSNorm: norm={:.4}", normed_norm);

    // QKV projection
    let qkv = model.qkv_matmul(&normed, &layer0.qkv_weight)?;
    println!("QKV output length: {}", qkv.len());

    // Expected: [Q, K, V] concatenated
    let q_out = &qkv[0..q_dim];
    let k_out = &qkv[q_dim..q_dim + kv_dim];
    let v_out = &qkv[q_dim + kv_dim..];

    println!("\nQKV output (before bias):");
    println!("  Q: len={}, first 4: {:?}", q_out.len(), &q_out[..4]);
    println!("  K: len={}, first 4: {:?}", k_out.len(), &k_out[..4]);
    println!("  V: len={}, first 4: {:?}", v_out.len(), &v_out[..4]);

    // Add bias and check
    let mut qkv_with_bias = qkv.clone();
    if let Some(ref bias) = layer0.qkv_bias {
        for i in 0..qkv_with_bias.len().min(bias.len()) {
            qkv_with_bias[i] += bias[i];
        }
    }

    let q_biased = &qkv_with_bias[0..q_dim];
    let k_biased = &qkv_with_bias[q_dim..q_dim + kv_dim];
    let v_biased = &qkv_with_bias[q_dim + kv_dim..];

    let q_biased_norm: f32 = q_biased.iter().map(|x| x * x).sum::<f32>().sqrt();
    let k_biased_norm: f32 = k_biased.iter().map(|x| x * x).sum::<f32>().sqrt();
    let v_biased_norm: f32 = v_biased.iter().map(|x| x * x).sum::<f32>().sqrt();

    println!("\nQKV output (after bias):");
    println!("  Q norm: {:.4}", q_biased_norm);
    println!("  K norm: {:.4}", k_biased_norm);
    println!("  V norm: {:.4}", v_biased_norm);

    // Check for extreme values that could cause attention issues
    let q_max = q_biased.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let k_max = k_biased.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    println!("\n  Q max: {:.4}, K max: {:.4}", q_max, k_max);

    // Attention score analysis
    // score = Q · K / sqrt(head_dim)
    let scale = 1.0 / (head_dim as f32).sqrt();
    let potential_max_score = q_max.abs() * k_max.abs() * (head_dim as f32).sqrt() * scale;
    println!(
        "  Potential max attention score (rough): {:.4}",
        potential_max_score
    );

    // If scores are >~50, softmax will saturate
    if potential_max_score > 50.0 {
        println!("  ⚠️ WARNING: Attention scores may cause softmax saturation!");
    }

    Ok(())
}
