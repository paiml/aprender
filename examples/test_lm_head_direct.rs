//! Test LM head directly with embeddings (bypass transformer)
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "/home/noah/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct-GGUF/snapshots/198f08841147e5196a6a69bd0053690fb1fd3857/qwen2-0_5b-instruct-q4_0.gguf";
    let mapped = MappedGGUFModel::from_path(path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;
    let vocab = mapped.model.vocabulary().expect("vocab");

    let hidden_dim = model.config().hidden_dim;

    println!("=== Direct LM Head Test ===\n");

    // Get embedding for token 17 ("2")
    let tok = 17;
    let emb_start = tok * hidden_dim;
    let emb = &model.token_embedding()[emb_start..emb_start + hidden_dim];

    let emb_norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
    println!(
        "Token {} ({:?}) embedding: norm={:.4}",
        tok,
        vocab.get(tok).unwrap(),
        emb_norm
    );

    // Apply RMSNorm with output_norm_weight
    let sum_sq: f32 = emb.iter().map(|x| x * x).sum();
    let mean_sq = sum_sq / hidden_dim as f32;
    let inv_rms = 1.0 / (mean_sq + model.config().eps).sqrt();

    let mut normed = vec![0.0f32; hidden_dim];
    for i in 0..hidden_dim {
        normed[i] = emb[i] * inv_rms * model.output_norm_weight()[i];
    }

    let normed_norm: f32 = normed.iter().map(|x| x * x).sum::<f32>().sqrt();
    println!(
        "After RMSNorm: norm={:.4}, inv_rms={:.4}",
        normed_norm, inv_rms
    );

    // Manually compute logits using token_embedding (tied weights)
    // logit[i] = normed · token_embd[i]

    println!("\nManual logit computation (tied weights, first 25 tokens):");

    let mut manual_logits = Vec::with_capacity(25);
    for i in 0..25 {
        let tok_emb = &model.token_embedding()[i * hidden_dim..(i + 1) * hidden_dim];
        let logit: f32 = normed.iter().zip(tok_emb.iter()).map(|(a, b)| a * b).sum();
        let tok_str = vocab.get(i).map_or("?", |s| s.as_str());
        manual_logits.push((i, logit, tok_str.to_string()));
    }

    // Sort by logit
    manual_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("Top 10 (manual):");
    for (i, logit, tok_str) in manual_logits.iter().take(10) {
        println!("  Token {} ({:?}): {:.4}", i, tok_str, logit);
    }

    // Now run actual forward pass
    println!("\n\nActual forward pass with token 17 (\"2\"):");
    let logits = model.forward(&[tok as u32])?;

    // Get top 10
    let mut indexed: Vec<_> = logits.iter().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

    println!("Top 10 (forward):");
    for (tok_id, logit) in indexed.iter().take(10) {
        let tok_str = vocab.get(*tok_id).map_or("?", |s| s.as_str());
        println!("  Token {} ({:?}): {:.4}", tok_id, tok_str, logit);
    }

    Ok(())
}
