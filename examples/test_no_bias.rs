//! Test generation WITHOUT QKV bias
use realizar::gguf::{
    MappedGGUFModel, OwnedQKVWeights, OwnedQuantizedKVCache, OwnedQuantizedModel,
};
use realizar::rms_norm;

fn main() {
    let path = "../aprender/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf";
    let mapped = MappedGGUFModel::from_path(path).expect("load");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("model");
    let vocab = mapped.model.vocabulary().expect("vocab");

    let token_id = 151644u32; // <|im_start|>

    // Do forward pass WITHOUT bias
    let layer = &model.layers()[0];
    let input = model.embed(&[token_id]);
    let normed = rms_norm(&input, &layer.attn_norm_weight, model.config().eps);
    let qkv_no_bias = model.qkv_matmul(&normed, &layer.qkv_weight).expect("qkv");

    // Do forward pass WITH bias
    let mut qkv_with_bias = qkv_no_bias.clone();
    if let Some(ref bias) = layer.qkv_bias {
        for (x, b) in qkv_with_bias.iter_mut().zip(bias.iter()) {
            *x += *b;
        }
    }

    let (q_dim, _k_dim, _v_dim) = match &layer.qkv_weight {
        OwnedQKVWeights::Separate { q, k, v } => (q.out_dim, k.out_dim, v.out_dim),
        _ => panic!("Expected separate"),
    };

    eprintln!("=== Q comparison ===");
    eprintln!("No bias Q[0:8]: {:?}", &qkv_no_bias[0..8]);
    eprintln!("With bias Q[0:8]: {:?}", &qkv_with_bias[0..8]);

    eprintln!("\n=== K comparison ===");
    eprintln!("No bias K[0:8]: {:?}", &qkv_no_bias[q_dim..q_dim + 8]);
    eprintln!("With bias K[0:8]: {:?}", &qkv_with_bias[q_dim..q_dim + 8]);

    // Now test full forward pass with bias removed from model
    // We can't easily modify the model, so let's compare attention scores
    eprintln!("\n=== Testing with normal forward (with bias) ===");
    let head_dim = model.config().hidden_dim / model.config().num_heads;
    let kv_dim = model.config().num_kv_heads * head_dim;
    let mut cache = OwnedQuantizedKVCache::new(model.config().num_layers, kv_dim, 8);
    let logits = model
        .forward_single_with_cache(token_id, &mut cache, 0)
        .expect("forward");

    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    eprintln!("Top 5 predictions:");
    for (rank, (idx, score)) in indexed.iter().take(5).enumerate() {
        let tok_str = vocab
            .get(*idx)
            .map_or_else(|| "?".to_string(), |s| s.escape_debug().to_string());
        eprintln!("{}: {} score={:.4} '{}'", rank + 1, idx, score, tok_str);
    }

    // Check "system" token
    let system_id = vocab.iter().position(|t| t == "system").unwrap_or(0);
    let system_score = logits[system_id];
    let system_rank = indexed
        .iter()
        .position(|(i, _)| *i == system_id)
        .map_or(0, |r| r + 1);
    eprintln!(
        "\n'system' (id={}) score={:.4} rank={}",
        system_id, system_score, system_rank
    );
}
