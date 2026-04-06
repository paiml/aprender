//! Check weight tensor statistics to see if something is obviously wrong
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "/home/noah/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct-GGUF/snapshots/198f08841147e5196a6a69bd0053690fb1fd3857/qwen2-0_5b-instruct-q4_0.gguf";
    let mapped = MappedGGUFModel::from_path(path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;

    println!("=== Qwen2 Weight Statistics ===\n");

    // Check token embedding stats
    println!("Token embedding:");
    let emb_len = model.token_embedding().len();
    let hidden_dim = model.config().hidden_dim;
    let vocab_size = emb_len / hidden_dim;
    println!(
        "  Size: {} elements ({} tokens x {} hidden)",
        emb_len, vocab_size, hidden_dim
    );

    // Check overall embedding stats
    let emb_sum: f32 = model.token_embedding().iter().sum();
    let emb_mean = emb_sum / emb_len as f32;
    let emb_min = model
        .token_embedding()
        .iter()
        .copied()
        .fold(f32::INFINITY, f32::min);
    let emb_max = model
        .token_embedding()
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);

    println!(
        "  mean: {:.6}, range: [{:.6}, {:.6}]",
        emb_mean, emb_min, emb_max
    );

    // Check a few specific token embeddings
    println!("\nSpecific token embedding norms:");
    for tok in [0, 1, 10, 15, 16, 17, 18, 19, 20] {
        let start = tok * hidden_dim;
        let end = start + hidden_dim;
        let emb = &model.token_embedding()[start..end];
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        let sum: f32 = emb.iter().sum();
        println!("  Token {}: norm={:.4}, sum={:.4}", tok, norm, sum);
    }

    // Check LM head weight stats
    println!("\nLM head weight:");
    println!("  qtype: {}", model.lm_head_weight().qtype);
    println!(
        "  in_dim: {}, out_dim: {}",
        model.lm_head_weight().in_dim,
        model.lm_head_weight().out_dim
    );
    println!("  data len: {} bytes", model.lm_head_weight().data.len());

    // Check layer 0 stats
    println!("\nLayer 0:");
    let layer0 = &model.layers()[0];
    println!("  attn_norm_weight len: {}", layer0.attn_norm_weight.len());

    // QKV weight stats
    let qkv_out = layer0.qkv_weight.out_dim();
    println!("  QKV out_dim: {}", qkv_out);

    if let Some(ref bias) = layer0.qkv_bias {
        let bias_norm: f32 = bias.iter().map(|x| x * x).sum::<f32>().sqrt();
        let bias_max = bias.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let bias_min = bias.iter().copied().fold(f32::INFINITY, f32::min);
        println!(
            "  QKV bias: len={}, norm={:.4}, range=[{:.4}, {:.4}]",
            bias.len(),
            bias_norm,
            bias_min,
            bias_max
        );
    }

    Ok(())
}
