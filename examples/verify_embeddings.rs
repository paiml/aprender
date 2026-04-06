//! Verify token embeddings
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "/home/noah/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct-GGUF/snapshots/198f08841147e5196a6a69bd0053690fb1fd3857/qwen2-0_5b-instruct-q4_0.gguf";
    let mapped = MappedGGUFModel::from_path(path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;

    let hidden_dim = model.config().hidden_dim;

    println!("Verifying embeddings (hidden_dim={}):\n", hidden_dim);

    // Check embeddings for tokens 0-5
    for tok in 0..5 {
        let start = tok * hidden_dim;
        let emb = &model.token_embedding()[start..start + hidden_dim];

        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        let sum: f32 = emb.iter().sum();
        let max: f32 = emb.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let min: f32 = emb.iter().copied().fold(f32::INFINITY, f32::min);

        println!(
            "Token {}: norm={:.4}, sum={:.4}, max={:.4}, min={:.4}",
            tok, norm, sum, max, min
        );
        println!("  first 4: {:?}", &emb[..4]);
    }

    // Check if all embeddings are similar (which would be wrong)
    println!("\nComparing cosine similarity between consecutive tokens:");
    for tok in 0..4 {
        let start1 = tok * hidden_dim;
        let start2 = (tok + 1) * hidden_dim;
        let emb1 = &model.token_embedding()[start1..start1 + hidden_dim];
        let emb2 = &model.token_embedding()[start2..start2 + hidden_dim];

        let dot: f32 = emb1.iter().zip(emb2.iter()).map(|(a, b)| a * b).sum();
        let norm1: f32 = emb1.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm2: f32 = emb2.iter().map(|x| x * x).sum::<f32>().sqrt();
        let cos_sim = dot / (norm1 * norm2);

        println!(
            "  cos_sim(token_{}, token_{}) = {:.4}",
            tok,
            tok + 1,
            cos_sim
        );
    }

    // Check token embedding vs LM head weight for token 0
    // Since weights are tied, they should be identical (or very similar after quantization)
    println!("\nComparing token_embd vs lm_head for token 0:");
    let _tok0_emb = &model.token_embedding()[0..hidden_dim];

    // Note: lm_head is quantized, so we can't easily compare
    // But we can verify by checking forward pass behavior

    // Manual matmul: logit[0] should equal token_embd[0] · token_embd[input_tok]
    // after all the transformer processing

    Ok(())
}
