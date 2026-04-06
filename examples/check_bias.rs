//! Check Qwen2 bias values
use realizar::gguf::{MappedGGUFModel, OwnedQKVWeights, OwnedQuantizedModel};

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

    let layer = &model.layers()[0];

    let (q_dim, k_dim, v_dim) = match &layer.qkv_weight {
        OwnedQKVWeights::Separate { q, k, v } => (q.out_dim, k.out_dim, v.out_dim),
        _ => panic!("Expected separate"),
    };

    eprintln!("q_dim={}, k_dim={}, v_dim={}", q_dim, k_dim, v_dim);

    if let Some(ref bias) = layer.qkv_bias {
        eprintln!("\nFull bias len: {}", bias.len());
        let q_bias = &bias[0..q_dim];
        let k_bias = &bias[q_dim..q_dim + k_dim];
        let v_bias = &bias[q_dim + k_dim..];

        stats("Q bias", q_bias);
        stats("K bias", k_bias);
        stats("V bias", v_bias);

        // Print first few values
        eprintln!("\nQ bias first 10: {:?}", &q_bias[..10.min(q_bias.len())]);
        eprintln!("K bias first 10: {:?}", &k_bias[..10.min(k_bias.len())]);
        eprintln!("V bias first 10: {:?}", &v_bias[..10.min(v_bias.len())]);
    }

    // Check raw bias tensors from GGUF
    eprintln!("\n=== Checking raw GGUF bias tensors ===");
    for name in [
        "blk.0.attn_q.bias",
        "blk.0.attn_k.bias",
        "blk.0.attn_v.bias",
    ] {
        if let Some(tensor) = mapped.model.tensors.iter().find(|t| t.name == name) {
            eprintln!(
                "{}: dims={:?}, n_dims={}, qtype={}",
                name, tensor.dims, tensor.n_dims, tensor.qtype
            );
        }
    }
}
