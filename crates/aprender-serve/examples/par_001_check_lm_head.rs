//! PAR-001: Check LM head and output normalization
//!
//! Verify the final layers are producing reasonable outputs.

use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn l2_norm(v: &[f32]) -> f32 {
    (v.iter().map(|x| x * x).sum::<f32>()).sqrt()
}

fn main() {
    let path = "/tmp/parity-bench/tinyllama-1.1b-q4_k_m.gguf";

    println!("=== PAR-001: Check LM Head ===\n");

    let mapped = MappedGGUFModel::from_path(path).expect("Failed to load model");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("test");
    let vocab = mapped.model.vocabulary().expect("test");

    println!("Model config:");
    println!("  hidden_dim: {}", model.config().hidden_dim);
    println!("  vocab_size: {}", model.config().vocab_size);

    println!("\nLM head weight:");
    println!("  in_dim: {}", model.lm_head_weight().in_dim);
    println!("  out_dim: {}", model.lm_head_weight().out_dim);
    println!(
        "  qtype: {} ({})",
        model.lm_head_weight().qtype,
        if model.lm_head_weight().qtype == 12 {
            "Q4_K"
        } else if model.lm_head_weight().qtype == 14 {
            "Q6_K"
        } else {
            "other"
        }
    );
    println!("  data size: {} bytes", model.lm_head_weight().data.len());

    println!("\nOutput norm weight:");
    println!("  length: {}", model.output_norm_weight().len());
    println!("  L2 norm: {:.4}", l2_norm(model.output_norm_weight()));
    println!(
        "  first 5: {:?}",
        &model.output_norm_weight()[..5.min(model.output_norm_weight().len())]
    );

    // Check vocabulary
    println!("\nVocabulary:");
    println!("  size: {}", vocab.len());
    println!("  first 5 tokens: {:?}", &vocab[..5.min(vocab.len())]);

    // Check some specific token IDs
    let test_tokens = [0, 1, 26222, 931, 123, 18456, 23565];
    println!("\nSpecific tokens:");
    for &tid in &test_tokens {
        if (tid as usize) < vocab.len() {
            println!(
                "  {}: '{}'",
                tid,
                vocab[tid as usize]
                    .replace("▁", " ")
                    .replace('\u{0120}', " ")
            );
        }
    }

    // Try doing a simple forward pass manually to check intermediate values
    println!("\n=== Simple forward pass (layer 0 only) ===");

    let token_id: u32 = 26222; // "Once"
    println!(
        "Input token: {} ('{}')",
        token_id,
        vocab.get(token_id as usize).unwrap_or(&"?".to_string())
    );

    // Embed
    let hidden = model.embed(&[token_id]);
    println!("\nEmbedding:");
    println!("  L2 norm: {:.4}", l2_norm(&hidden));
    println!("  first 5: {:?}", &hidden[..5.min(hidden.len())]);

    // Check for NaN or unreasonable values
    let has_nan = hidden.iter().any(|x| x.is_nan());
    let has_inf = hidden.iter().any(|x| x.is_infinite());
    let max_abs = hidden.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    println!(
        "  has NaN: {}, has Inf: {}, max abs: {:.4}",
        has_nan, has_inf, max_abs
    );

    println!("\n=== Complete ===");
}
