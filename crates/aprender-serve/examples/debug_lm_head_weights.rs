//! Debug LM head weight loading to compare with llama.cpp
//!
//! This script checks:
//! 1. Q8_0 block structure and byte layout
//! 2. Dequantized values for first few rows
//! 3. Manual dot product calculation
//!
//! Run: cd /home/noah/src/realizar && cargo run --release --example debug_lm_head_weights

use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};
use realizar::quantize::dequantize_q8_0;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== LM Head Weight Debug ===\n");

    let path = "/home/noah/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct-GGUF/snapshots/198f08841147e5196a6a69bd0053690fb1fd3857/qwen2-0_5b-instruct-q4_0.gguf";
    let mapped = MappedGGUFModel::from_path(path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;
    let vocab = mapped.model.vocabulary().expect("vocab");

    println!("Model config:");
    println!("  hidden_dim: {}", model.config().hidden_dim);
    println!("  vocab_size: {}", model.config().vocab_size);

    println!("\nLM head weight:");
    println!("  qtype: {}", model.lm_head_weight().qtype);
    println!("  in_dim: {}", model.lm_head_weight().in_dim);
    println!("  out_dim: {}", model.lm_head_weight().out_dim);
    println!("  data len: {} bytes", model.lm_head_weight().data.len());

    // Q8_0 format: 34 bytes per 32 elements
    let hidden_dim = model.config().hidden_dim;
    let vocab_size = model.config().vocab_size;
    let blocks_per_row = hidden_dim.div_ceil(32);
    let bytes_per_row = blocks_per_row * 34;
    println!("  blocks_per_row: {}", blocks_per_row);
    println!("  bytes_per_row: {}", bytes_per_row);
    println!(
        "  expected total bytes: {} (actual: {})",
        vocab_size * bytes_per_row,
        model.lm_head_weight().data.len()
    );

    // Dequantize entire LM head (this is what llama.cpp would do)
    println!("\nDequantizing LM head (Q8_0 -> f32)...");
    let lm_head_f32 = dequantize_q8_0(&model.lm_head_weight().data)?;
    println!("  dequantized elements: {}", lm_head_f32.len());
    println!("  expected elements: {}", vocab_size * hidden_dim);

    // Check first few rows
    println!("\nFirst few rows (each row = token's LM head weights):");
    for token_id in [0, 1, 19, 20] {
        let row_start = token_id * hidden_dim;
        let row_end = row_start + hidden_dim;
        let row = &lm_head_f32[row_start..row_end];

        let sum: f32 = row.iter().sum();
        let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        let min = row.iter().copied().fold(f32::INFINITY, f32::min);
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        let token_str = vocab.get(token_id).map_or("?", |s| s.as_str());
        println!(
            "  Row {} (token {:?}): sum={:.4}, norm={:.4}, min={:.4}, max={:.4}",
            token_id, token_str, sum, norm, min, max
        );
        println!("    first 8: {:?}", &row[..8]);
    }

    // Check embedding for comparison
    println!("\nEmbedding check (token_embd.weight):");
    println!("  embedding len: {}", model.token_embedding().len());
    println!("  expected: {}", vocab_size * hidden_dim);

    for token_id in [0, 1, 19, 20] {
        let emb_start = token_id * hidden_dim;
        let emb_end = emb_start + hidden_dim;
        let emb = &model.token_embedding()[emb_start..emb_end];

        let sum: f32 = emb.iter().sum();
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();

        println!(
            "  Token {} embedding: sum={:.4}, norm={:.4}",
            token_id, sum, norm
        );
        println!("    first 8: {:?}", &emb[..8]);
    }

    // Test dot product with a random hidden state
    println!("\nDot product test:");
    let test_hidden: Vec<f32> = (0..hidden_dim).map(|i| (i as f32 * 0.01).sin()).collect();
    let hidden_norm: f32 = test_hidden.iter().map(|x| x * x).sum::<f32>().sqrt();
    println!(
        "  Test hidden state: {} elements, norm={:.4}",
        test_hidden.len(),
        hidden_norm
    );

    // Compute dot products manually
    for token_id in [0, 1, 19, 20] {
        let row_start = token_id * hidden_dim;
        let row = &lm_head_f32[row_start..row_start + hidden_dim];
        let dot: f32 = test_hidden.iter().zip(row.iter()).map(|(a, b)| a * b).sum();
        let token_str = vocab.get(token_id).map_or("?", |s| s.as_str());
        println!("  Token {} ({:?}): logit = {:.4}", token_id, token_str, dot);
    }

    // Now test with actual model forward
    println!("\nForward test with BOS token:");
    let logits = model.forward(&[151643])?;
    println!("  Logits len: {}", logits.len());
    println!("  Logit[0] (\"!\"): {:.4}", logits[0]);
    println!("  Logit[19] (\"4\"): {:.4}", logits[19]);

    // Find argmax
    let (argmax_idx, argmax_val) = logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();
    let argmax_str = vocab.get(argmax_idx).map_or("?", |s| s.as_str());
    println!(
        "  Argmax: token {} ({:?}) with logit {:.4}",
        argmax_idx, argmax_str, argmax_val
    );

    Ok(())
}
