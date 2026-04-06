//! Compare final hidden states between single and multi-token inference
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "/home/noah/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct-GGUF/snapshots/198f08841147e5196a6a69bd0053690fb1fd3857/qwen2-0_5b-instruct-q4_0.gguf";
    let mapped = MappedGGUFModel::from_path(path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;
    let vocab = mapped.model.vocabulary().expect("vocab");

    println!("=== Final Hidden State Analysis ===\n");

    let hidden_dim = model.config().hidden_dim;
    let num_layers = model.layers().len();
    let _eps = model.config().eps;

    println!("Model: {} layers, {} hidden_dim", num_layers, hidden_dim);

    // We need to manually trace through the model to get intermediate states
    // Let's use a simpler approach: use forward() and compare logits structure

    // Actually, let's compare the dot product of final hidden with specific token embeddings
    // to understand why "!" gets such high logits

    // First, let's verify that the LM head is correctly computing logits
    // logit[i] = final_hidden · lm_head_weight[i]

    // Get logits for both cases
    let single_logits = model.forward(&[17u32])?; // "2"
    let multi_logits = model.forward(&[17u32, 10, 17, 28])?; // "2+2="

    // Now let's check the LM head weight for token 0 ("!")
    // For tied embeddings, lm_head_weight[i] = token_embedding[i]
    // So logit[0] = hidden · embedding[0]

    // The question is: what's different about the hidden state that makes it
    // have high dot product with token 0 embedding in multi-token case?

    println!("\n=== LM Head Analysis ===");
    println!("LM head weight type: {}", model.lm_head_weight().qtype);
    println!("LM head in_dim: {}", model.lm_head_weight().in_dim);
    println!("LM head out_dim: {}", model.lm_head_weight().out_dim);

    // Check token 0 vs token 19 ("4") embeddings
    let emb_0 = &model.token_embedding()[0..hidden_dim];
    let emb_19 = &model.token_embedding()[19 * hidden_dim..20 * hidden_dim];

    let emb_0_norm: f32 = emb_0.iter().map(|x| x * x).sum::<f32>().sqrt();
    let emb_19_norm: f32 = emb_19.iter().map(|x| x * x).sum::<f32>().sqrt();

    println!("\nEmbedding norms:");
    println!("  Token 0 ('!'): {:.4}", emb_0_norm);
    println!("  Token 19 ('4'): {:.4}", emb_19_norm);

    // Check if there's a systematic pattern in token 0's embedding
    let emb_0_mean: f32 = emb_0.iter().sum::<f32>() / hidden_dim as f32;
    let emb_0_sum: f32 = emb_0.iter().sum();
    let emb_0_abs_sum: f32 = emb_0.iter().map(|x| x.abs()).sum();

    let emb_19_mean: f32 = emb_19.iter().sum::<f32>() / hidden_dim as f32;
    let emb_19_sum: f32 = emb_19.iter().sum();
    let emb_19_abs_sum: f32 = emb_19.iter().map(|x| x.abs()).sum();

    println!("\nEmbedding statistics:");
    println!(
        "  Token 0 ('!'): sum={:.4}, abs_sum={:.4}, mean={:.6}",
        emb_0_sum, emb_0_abs_sum, emb_0_mean
    );
    println!(
        "  Token 19 ('4'): sum={:.4}, abs_sum={:.4}, mean={:.6}",
        emb_19_sum, emb_19_abs_sum, emb_19_mean
    );

    // Hypothesis: If emb_0 has a large positive sum and the final hidden also has a large positive sum,
    // this would explain the high dot product

    // The logit formula is: logit[i] = hidden · lm_head_weight[i]
    // If lm_head_weight is Q8_0 quantized, it's dequantized then dot product computed

    // Let me check if there's a bias in the LM head
    if let Some(bias) = model.lm_head_bias() {
        println!("\nLM head has bias!");
        println!("  Bias[0] ('!'): {:.4}", bias[0]);
        println!("  Bias[19] ('4'): {:.4}", bias[19]);

        // Check if bias[0] is unusually high
        let bias_mean: f32 = bias.iter().sum::<f32>() / bias.len() as f32;
        let bias_max = bias.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let bias_min = bias.iter().copied().fold(f32::INFINITY, f32::min);

        println!(
            "  Bias stats: mean={:.4}, range=[{:.4}, {:.4}]",
            bias_mean, bias_min, bias_max
        );

        // Find rank of token 0's bias
        let mut bias_indexed: Vec<_> = bias.iter().enumerate().collect();
        bias_indexed.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
        let bias_0_rank = bias_indexed.iter().position(|(i, _)| *i == 0).unwrap_or(0) + 1;
        println!("  Token 0 bias rank: {} of {}", bias_0_rank, bias.len());
    } else {
        println!("\nNo LM head bias");
    }

    // Now let's see the difference in logits more carefully
    println!("\n=== Logit Analysis ===");

    // Key tokens
    let tokens_to_check = vec![(0, "!"), (19, "4"), (17, "2"), (11, ",")];

    println!(
        "\n{:>15} {:>12} {:>12} {:>12}",
        "Token", "Single", "Multi", "Diff"
    );
    for (tok, name) in &tokens_to_check {
        let single = single_logits[*tok];
        let multi = multi_logits[*tok];
        let diff = multi - single;
        println!(
            "{:>15} {:>12.4} {:>12.4} {:>12.4}",
            format!("{} ('{}')", tok, name),
            single,
            multi,
            diff
        );
    }

    // Calculate mean shift in logits
    let mean_shift: f32 = multi_logits
        .iter()
        .zip(single_logits.iter())
        .map(|(m, s)| m - s)
        .sum::<f32>()
        / multi_logits.len() as f32;

    let token_0_shift = multi_logits[0] - single_logits[0];

    println!("\nMean logit shift (multi - single): {:.4}", mean_shift);
    println!("Token 0 shift: {:.4}", token_0_shift);
    println!(
        "Token 0 relative shift: {:.4} (vs mean)",
        token_0_shift - mean_shift
    );

    // Check how many tokens got boosted more than token 0
    let shifts: Vec<f32> = multi_logits
        .iter()
        .zip(single_logits.iter())
        .map(|(m, s)| m - s)
        .collect();
    let tokens_boosted_more = shifts.iter().filter(|&&s| s > token_0_shift).count();

    println!("Tokens with larger boost than '!': {}", tokens_boosted_more);

    // Top tokens by boost
    let mut boost_indexed: Vec<_> = shifts.iter().enumerate().collect();
    boost_indexed.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

    println!("\nTop 10 tokens by boost (multi - single):");
    for (tok, boost) in boost_indexed.iter().take(10) {
        let s = vocab.get(*tok).map_or("?", |s| s.as_str());
        println!("  Token {} ({:?}): +{:.4}", tok, s, boost);
    }

    Ok(())
}
