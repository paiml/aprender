//! Test QKV matmul with bias
use realizar::gguf::{MappedGGUFModel, OwnedQKVWeights, OwnedQuantizedModel};
use realizar::rms_norm;

fn stats(name: &str, v: &[f32]) {
    if v.is_empty() {
        return;
    }
    let sum: f32 = v.iter().sum();
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    let min = v.iter().copied().fold(f32::INFINITY, f32::min);
    let max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    eprintln!(
        "{}: len={}, sum={:.4}, norm={:.4}, min={:.4}, max={:.4}",
        name,
        v.len(),
        sum,
        norm,
        min,
        max
    );
}

fn main() {
    let path = "../aprender/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf";
    let mapped = MappedGGUFModel::from_path(path).expect("load");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("model");

    let token_id = 151644u32; // <|im_start|>
    let input = model.embed(&[token_id]);
    let layer = &model.layers()[0];
    let normed = rms_norm(&input, &layer.attn_norm_weight, model.config().eps);

    let mut qkv = model.qkv_matmul(&normed, &layer.qkv_weight).expect("qkv");

    eprintln!("=== Before bias ===");
    let (q_dim, k_dim, v_dim) = match &layer.qkv_weight {
        OwnedQKVWeights::Separate { q, k, v } => (q.out_dim, k.out_dim, v.out_dim),
        _ => panic!("Expected separate"),
    };
    stats("Q", &qkv[0..q_dim]);
    stats("K", &qkv[q_dim..q_dim + k_dim]);
    stats("V", &qkv[q_dim + k_dim..q_dim + k_dim + v_dim]);

    // Add bias
    if let Some(ref bias) = layer.qkv_bias {
        for (x, b) in qkv.iter_mut().zip(bias.iter()) {
            *x += *b;
        }
    }

    eprintln!("\n=== After bias ===");
    stats("Q", &qkv[0..q_dim]);
    stats("K", &qkv[q_dim..q_dim + k_dim]);
    stats("V", &qkv[q_dim + k_dim..q_dim + k_dim + v_dim]);

    // Print V bias separately
    if let Some(ref bias) = layer.qkv_bias {
        let v_bias = &bias[q_dim + k_dim..];
        stats("V bias", v_bias);
    }

    // Also compare TinyLlama if available
    eprintln!("\n=== TinyLlama comparison ===");
    let tinyllama_path = "/home/noah/models/TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf";
    if let Ok(mapped2) = MappedGGUFModel::from_path(tinyllama_path) {
        let model2 = OwnedQuantizedModel::from_mapped(&mapped2).expect("model");
        let input2 = model2.embed(&[1u32]); // BOS token
        let layer2 = &model2.layers()[0];
        let normed2 = rms_norm(&input2, &layer2.attn_norm_weight, model2.config().eps);
        let qkv2 = model2
            .qkv_matmul(&normed2, &layer2.qkv_weight)
            .expect("qkv");

        // TinyLlama has separate Q, K, V
        let (q_dim2, k_dim2, v_dim2) = match &layer2.qkv_weight {
            OwnedQKVWeights::Separate { q, k, v } => (q.out_dim, k.out_dim, v.out_dim),
            _ => panic!("Expected separate"),
        };
        stats("TinyLlama Q", &qkv2[0..q_dim2]);
        stats("TinyLlama K", &qkv2[q_dim2..q_dim2 + k_dim2]);
        stats(
            "TinyLlama V",
            &qkv2[q_dim2 + k_dim2..q_dim2 + k_dim2 + v_dim2],
        );
    }
}
