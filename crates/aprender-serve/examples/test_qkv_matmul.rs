//! Test individual Q, K, V matmuls
use realizar::gguf::{MappedGGUFModel, OwnedQKVWeights, OwnedQuantizedModel};
use realizar::rms_norm;

fn stats(name: &str, v: &[f32]) {
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

    let _hidden_dim = model.config().hidden_dim;

    // Create test input: embedding for token 151644 (<|im_start|>)
    let token_id = 151644u32;
    let input = model.embed(&[token_id]);
    stats("Input (embedding)", &input);

    // Apply RMSNorm using the attention norm weight
    let layer = &model.layers()[0];
    let normed = rms_norm(&input, &layer.attn_norm_weight, model.config().eps);
    stats("After RMSNorm", &normed);

    // Compute QKV using the model's method
    let qkv = model.qkv_matmul(&normed, &layer.qkv_weight).expect("qkv");
    stats("QKV combined", &qkv);

    // Split QKV
    let (q, k, v) = match &layer.qkv_weight {
        OwnedQKVWeights::Separate {
            q: q_w,
            k: k_w,
            v: v_w,
        } => {
            let q_dim = q_w.out_dim;
            let k_dim = k_w.out_dim;
            let v_dim = v_w.out_dim;
            (
                &qkv[0..q_dim],
                &qkv[q_dim..q_dim + k_dim],
                &qkv[q_dim + k_dim..q_dim + k_dim + v_dim],
            )
        },
        _ => panic!("Expected separate QKV"),
    };

    stats("Q part", q);
    stats("K part", k);
    stats("V part", v);

    // Check if Q bias is applied
    if let Some(ref bias) = layer.qkv_bias {
        eprintln!("\nQKV bias len: {}", bias.len());
        stats("QKV bias first 10", &bias[..10.min(bias.len())]);
    }
}
