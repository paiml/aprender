//! Compare QKV structure between TinyLlama and Qwen2
use realizar::gguf::{MappedGGUFModel, OwnedQKVWeights, OwnedQuantizedModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== QKV Structure Comparison ===\n");

    // TinyLlama
    let tinyllama_path = "/tmp/tinyllama.gguf";
    if std::path::Path::new(tinyllama_path).exists() {
        let mapped = MappedGGUFModel::from_path(tinyllama_path)?;
        let model = OwnedQuantizedModel::from_mapped(&mapped)?;

        println!("TinyLlama:");
        println!("  hidden_dim: {}", model.config().hidden_dim);
        println!("  num_heads: {}", model.config().num_heads);
        println!("  num_kv_heads: {}", model.config().num_kv_heads);
        println!("  rope_type: {}", model.config().rope_type);

        let layer0 = &model.layers()[0];
        match &layer0.qkv_weight {
            OwnedQKVWeights::Fused(tensor) => {
                println!(
                    "  QKV: FUSED, out_dim={}, qtype={}",
                    tensor.out_dim, tensor.qtype
                );
            },
            OwnedQKVWeights::Separate { q, k, v } => {
                println!("  QKV: SEPARATE");
                println!("    Q: out_dim={}, qtype={}", q.out_dim, q.qtype);
                println!("    K: out_dim={}, qtype={}", k.out_dim, k.qtype);
                println!("    V: out_dim={}, qtype={}", v.out_dim, v.qtype);
            },
        }

        if let Some(ref bias) = layer0.qkv_bias {
            let norm: f32 = bias.iter().map(|x| x * x).sum::<f32>().sqrt();
            println!("  QKV bias: len={}, norm={:.4}", bias.len(), norm);
        } else {
            println!("  QKV bias: NONE");
        }
    } else {
        println!("TinyLlama not found at {}", tinyllama_path);
    }

    // Qwen2
    let qwen_path = "/home/noah/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct-GGUF/snapshots/198f08841147e5196a6a69bd0053690fb1fd3857/qwen2-0_5b-instruct-q4_0.gguf";
    println!("\nQwen2-0.5B:");
    let mapped = MappedGGUFModel::from_path(qwen_path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;

    println!("  hidden_dim: {}", model.config().hidden_dim);
    println!("  num_heads: {}", model.config().num_heads);
    println!("  num_kv_heads: {}", model.config().num_kv_heads);
    println!("  rope_type: {}", model.config().rope_type);

    let layer0 = &model.layers()[0];
    match &layer0.qkv_weight {
        OwnedQKVWeights::Fused(tensor) => {
            println!(
                "  QKV: FUSED, out_dim={}, qtype={}",
                tensor.out_dim, tensor.qtype
            );
        },
        OwnedQKVWeights::Separate { q, k, v } => {
            println!("  QKV: SEPARATE");
            println!(
                "    Q: out_dim={}, in_dim={}, qtype={}",
                q.out_dim, q.in_dim, q.qtype
            );
            println!(
                "    K: out_dim={}, in_dim={}, qtype={}",
                k.out_dim, k.in_dim, k.qtype
            );
            println!(
                "    V: out_dim={}, in_dim={}, qtype={}",
                v.out_dim, v.in_dim, v.qtype
            );
        },
    }

    if let Some(ref bias) = layer0.qkv_bias {
        let norm: f32 = bias.iter().map(|x| x * x).sum::<f32>().sqrt();
        let q_dim =
            model.config().num_heads * (model.config().hidden_dim / model.config().num_heads);
        let kv_dim =
            model.config().num_kv_heads * (model.config().hidden_dim / model.config().num_heads);

        println!("  QKV bias: len={}, norm={:.4}", bias.len(), norm);
        println!(
            "  Expected layout: Q[{}] + K[{}] + V[{}] = {}",
            q_dim,
            kv_dim,
            kv_dim,
            q_dim + 2 * kv_dim
        );

        // Check individual bias norms
        let q_bias = &bias[0..q_dim];
        let k_bias = &bias[q_dim..q_dim + kv_dim];
        let v_bias = &bias[q_dim + kv_dim..];

        let q_norm: f32 = q_bias.iter().map(|x| x * x).sum::<f32>().sqrt();
        let k_norm: f32 = k_bias.iter().map(|x| x * x).sum::<f32>().sqrt();
        let v_norm: f32 = v_bias.iter().map(|x| x * x).sum::<f32>().sqrt();

        println!("    Q bias norm: {:.4}", q_norm);
        println!("    K bias norm: {:.4}", k_norm);
        println!("    V bias norm: {:.4}", v_norm);
    } else {
        println!("  QKV bias: NONE");
    }

    Ok(())
}
