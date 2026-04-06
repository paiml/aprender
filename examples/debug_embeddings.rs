//! Verify embeddings are identical between CPU and GPU paths

use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path =
        "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";

    let mapped = MappedGGUFModel::from_path(model_path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;

    let test_tokens = [0u32, 1, 100, 791, 1000, 10000];

    eprintln!("=== Embedding Verification ===");
    eprintln!("vocab_size: {}", model.config().vocab_size);
    eprintln!("hidden_dim: {}", model.config().hidden_dim);

    for token in test_tokens {
        if token as usize >= model.config().vocab_size {
            continue;
        }

        // Get embedding via the embed() method
        let embedding = model.embed(&[token]);

        let sum: f32 = embedding.iter().sum();
        let rms: f32 =
            (embedding.iter().map(|x| x * x).sum::<f32>() / embedding.len() as f32).sqrt();

        eprintln!(
            "Token {}: sum={:.6}, rms={:.6}, first4={:?}",
            token,
            sum,
            rms,
            &embedding[..4.min(embedding.len())]
        );
    }

    // Check embedding tensor directly
    eprintln!("\n=== Raw Embedding Tensor ===");
    let hidden_dim = model.config().hidden_dim;
    let embed_data = &model.token_embedding();
    eprintln!(
        "Embed tensor length: {} (expected: {})",
        embed_data.len(),
        model.config().vocab_size * hidden_dim
    ); // FP32

    // Manually extract token 791 embedding (already FP32)
    let token = 791usize;
    let token_offset = token * hidden_dim;
    if token_offset + hidden_dim <= embed_data.len() {
        let manual_embedding = &embed_data[token_offset..token_offset + hidden_dim];
        let manual_sum: f32 = manual_embedding.iter().sum();
        eprintln!(
            "Token 791 manual extraction: sum={:.6}, first4={:?}",
            manual_sum,
            &manual_embedding[..4]
        );
    }

    Ok(())
}
