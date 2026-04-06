//! Debug embedding layer

use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn l2_norm(v: &[f32]) -> f32 {
    (v.iter().map(|x| x * x).sum::<f32>()).sqrt()
}

fn main() {
    let path = "/tmp/parity-bench/tinyllama-1.1b-q4_k_m.gguf";
    let mapped = MappedGGUFModel::from_path(path).expect("Failed");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("test");

    let hidden_dim = model.config().hidden_dim;
    let vocab_size = model.config().vocab_size;

    println!("=== Embedding Debug ===\n");
    println!("vocab_size: {}", vocab_size);
    println!("hidden_dim: {}", hidden_dim);
    println!("token_embedding.len(): {}", model.token_embedding().len());
    println!("Expected: {} (vocab * hidden)", vocab_size * hidden_dim);

    // Token 450 = "▁The"
    let token_id = 450u32;
    let start = token_id as usize * hidden_dim;
    let embedding: Vec<f32> = model.token_embedding()[start..start + hidden_dim].to_vec();

    println!("\nToken {} embedding:", token_id);
    println!("  L2: {:.4}", l2_norm(&embedding));
    println!(
        "  First 20: {:?}",
        &embedding[0..20]
            .iter()
            .map(|x| format!("{:.8}", x))
            .collect::<Vec<_>>()
    );
    println!(
        "  Last 10: {:?}",
        &embedding[hidden_dim - 10..hidden_dim]
            .iter()
            .map(|x| format!("{:.8}", x))
            .collect::<Vec<_>>()
    );

    // Check token 0 for reference
    let token0_emb: Vec<f32> = model.token_embedding()[0..hidden_dim].to_vec();
    println!("\nToken 0 embedding:");
    println!("  L2: {:.4}", l2_norm(&token0_emb));
    println!(
        "  First 10: {:?}",
        &token0_emb[0..10]
            .iter()
            .map(|x| format!("{:.8}", x))
            .collect::<Vec<_>>()
    );

    // Check token 1 for reference
    let token1_emb: Vec<f32> = model.token_embedding()[hidden_dim..hidden_dim * 2].to_vec();
    println!("\nToken 1 embedding:");
    println!("  L2: {:.4}", l2_norm(&token1_emb));
    println!(
        "  First 10: {:?}",
        &token1_emb[0..10]
            .iter()
            .map(|x| format!("{:.8}", x))
            .collect::<Vec<_>>()
    );

    // Print expected HF values
    println!("\n=== HuggingFace Expected ===");
    println!("Token 450 embedding:");
    println!("  L2: 0.3906");
    println!("  First 20: [-0.00983, 0.00964, 0.02039, -0.00555, 0.00215, 0.00143, 0.00322, -0.00577, -0.01489, 0.00717, -0.00620, -0.00060, 0.00047, -0.00135, 0.00687, 0.00577, 0.00313, -0.00017, -0.00681, 0.00089]");

    // Check statistics
    println!("\n=== Embedding Statistics ===");
    let min = embedding.iter().copied().fold(f32::INFINITY, f32::min);
    let max = embedding.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mean = embedding.iter().sum::<f32>() / embedding.len() as f32;
    println!(
        "Token 450 - min: {:.6}, max: {:.6}, mean: {:.8}",
        min, max, mean
    );
}
