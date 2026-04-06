//! Trace LM head to find bug
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "/home/noah/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct-GGUF/snapshots/198f08841147e5196a6a69bd0053690fb1fd3857/qwen2-0_5b-instruct-q4_0.gguf";
    let mapped = MappedGGUFModel::from_path(path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;

    let hidden_dim = model.config().hidden_dim;
    let vocab_size = model.token_embedding().len() / hidden_dim;

    println!("Model info:");
    println!("  hidden_dim: {}", hidden_dim);
    println!("  vocab_size: {}", vocab_size);
    println!("  lm_head qtype: {}", model.lm_head_weight().qtype);
    println!(
        "  lm_head dims: in={}, out={}",
        model.lm_head_weight().in_dim,
        model.lm_head_weight().out_dim
    );

    // Run forward with just BOS token
    let bos = 151643u32;
    let logits = model.forward(&[bos])?;

    println!("\nLogits (first 20 tokens):");
    for (i, logit) in logits.iter().enumerate().take(20) {
        println!("  token {}: {:.4}", i, logit);
    }

    // Find max logit
    let (max_idx, max_val) = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    println!("\nArgmax: token {} with logit {:.4}", max_idx, max_val);

    // Get lm_head weights manually and compute expected logits
    // The lm_head is tied to token_embd.weight
    // So logits[i] = dot(final_hidden, token_embd[i, :])

    // Let's compute what the hidden state would need to be to produce these logits
    // If logits[0] = 15.0 and token_0 has norm 0.47, then:
    // dot(hidden, token_0_emb) = 15.0
    // ||hidden|| * ||token_0|| * cos(theta) = 15.0
    // ||hidden|| * 0.47 * cos(theta) = 15.0
    // If hidden is perfectly aligned: ||hidden|| = 15.0 / 0.47 ≈ 32

    let tok0_emb = &model.token_embedding()[0..hidden_dim];
    let tok0_norm: f32 = tok0_emb.iter().map(|x| x * x).sum::<f32>().sqrt();

    let bos_emb =
        &model.token_embedding()[bos as usize * hidden_dim..(bos as usize + 1) * hidden_dim];
    let bos_norm: f32 = bos_emb.iter().map(|x| x * x).sum::<f32>().sqrt();

    println!("\nEmbedding analysis:");
    println!("  token_0 norm: {:.4}", tok0_norm);
    println!("  BOS norm: {:.4}", bos_norm);

    // If logit = hidden · token_emb, then for logit[0] = 15:
    println!("\nRequired hidden state norm (if perfectly aligned with token 0):");
    println!(
        "  ||hidden|| = {:.4} / {:.4} = {:.4}",
        logits[0],
        tok0_norm,
        logits[0] / tok0_norm
    );

    // Check cosine similarity between token 0 and BOS embedding
    let dot: f32 = tok0_emb
        .iter()
        .zip(bos_emb.iter())
        .map(|(a, b)| a * b)
        .sum();
    let cos_sim = dot / (tok0_norm * bos_norm);
    println!("  cosine_sim(token_0, BOS) = {:.4}", cos_sim);

    // Check logit for BOS
    let bos_logit = logits[bos as usize];
    println!("  BOS logit: {:.4}", bos_logit);

    Ok(())
}
