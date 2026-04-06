//! Check if token 0 ("!") has unusual properties in embeddings
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "/home/noah/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct-GGUF/snapshots/198f08841147e5196a6a69bd0053690fb1fd3857/qwen2-0_5b-instruct-q4_0.gguf";
    let mapped = MappedGGUFModel::from_path(path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;
    let vocab = mapped.model.vocabulary().expect("vocab");

    let hidden_dim = model.config().hidden_dim;
    let vocab_size = model.config().vocab_size;

    println!("=== Token 0 Investigation ===\n");

    // Check embedding for token 0 vs other tokens
    println!("Token embeddings (norm comparison):");
    for tok in [0u32, 1, 10, 15, 17, 19, 28, 100, 1000] {
        let start = tok as usize * hidden_dim;
        let end = start + hidden_dim;
        let emb = &model.token_embedding()[start..end];
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        let sum: f32 = emb.iter().sum();
        println!(
            "  Token {} ({}): norm={:.4}, sum={:.4}",
            tok,
            vocab.get(tok as usize).map_or("?", |s| s.as_str()),
            norm,
            sum
        );
    }

    // Check if there's an LM head bias
    println!(
        "\nLM head bias: {:?}",
        model.lm_head_bias().as_ref().map(|b| b.len())
    );
    if let Some(bias) = model.lm_head_bias() {
        println!("  Token 0 bias: {:.4}", bias[0]);
        println!("  Token 19 (\"4\") bias: {:.4}", bias[19]);
    }

    // Check LM head weight tensor info
    println!("\nLM head weight tensor:");
    println!("  qtype: {}", model.lm_head_weight().qtype);
    println!("  in_dim: {}", model.lm_head_weight().in_dim);
    println!("  out_dim: {}", model.lm_head_weight().out_dim);

    // The logits are computed as: hidden @ lm_head_weight^T (with tied embeddings, lm_head = embeddings)
    // For tied embeddings, logit[i] = dot(final_hidden, embedding[i])

    // Test with a synthetic hidden state and see which tokens have highest dot products
    println!("\n=== Dot Product Analysis ===\n");

    // Create a test hidden state (normalized random-like values)
    let test_hidden: Vec<f32> = (0..hidden_dim)
        .map(|i| ((i * 7919) % 1000) as f32 / 1000.0 - 0.5)
        .collect();
    let test_norm: f32 = test_hidden.iter().map(|x| x * x).sum::<f32>().sqrt();
    println!("Test hidden state norm: {:.4}", test_norm);

    // Compute dot products with all embeddings
    let mut dot_products: Vec<(usize, f32)> = (0..vocab_size.min(151936))
        .map(|tok| {
            let emb_start = tok * hidden_dim;
            let emb_end = emb_start + hidden_dim;
            if emb_end <= model.token_embedding().len() {
                let dot: f32 = test_hidden
                    .iter()
                    .zip(&model.token_embedding()[emb_start..emb_end])
                    .map(|(a, b)| a * b)
                    .sum();
                (tok, dot)
            } else {
                (tok, 0.0)
            }
        })
        .collect();

    dot_products.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("Top 10 dot products with synthetic hidden:");
    for (tok, dot) in dot_products.iter().take(10) {
        let tok_str = vocab.get(*tok).map_or("?", |s| s.as_str());
        println!("  Token {} ({:?}): dot={:.4}", tok, tok_str, dot);
    }

    // Check token 0's position in ranking
    let tok0_rank = dot_products.iter().position(|(t, _)| *t == 0);
    let tok19_rank = dot_products.iter().position(|(t, _)| *t == 19);
    println!("\nRanking:");
    println!("  Token 0 ('!'): rank {}", tok0_rank.map_or(0, |r| r + 1));
    println!("  Token 19 ('4'): rank {}", tok19_rank.map_or(0, |r| r + 1));

    // Check if token 0 embedding has unusual properties
    println!("\n=== Token 0 Embedding Analysis ===\n");
    let emb0 = &model.token_embedding()[0..hidden_dim];
    let emb0_min = emb0.iter().copied().fold(f32::INFINITY, f32::min);
    let emb0_max = emb0.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let emb0_mean: f32 = emb0.iter().sum::<f32>() / hidden_dim as f32;
    let emb0_abs_sum: f32 = emb0.iter().map(|x| x.abs()).sum();
    println!("Token 0 ('!') embedding stats:");
    println!(
        "  min: {:.6}, max: {:.6}, mean: {:.6}",
        emb0_min, emb0_max, emb0_mean
    );
    println!("  abs_sum: {:.4}", emb0_abs_sum);
    println!("  first 8: {:?}", &emb0[..8]);

    // Compare with token 19 ("4")
    let emb19 = &model.token_embedding()[19 * hidden_dim..20 * hidden_dim];
    let emb19_min = emb19.iter().copied().fold(f32::INFINITY, f32::min);
    let emb19_max = emb19.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let emb19_mean: f32 = emb19.iter().sum::<f32>() / hidden_dim as f32;
    let emb19_abs_sum: f32 = emb19.iter().map(|x| x.abs()).sum();
    println!("\nToken 19 ('4') embedding stats:");
    println!(
        "  min: {:.6}, max: {:.6}, mean: {:.6}",
        emb19_min, emb19_max, emb19_mean
    );
    println!("  abs_sum: {:.4}", emb19_abs_sum);
    println!("  first 8: {:?}", &emb19[..8]);

    Ok(())
}
