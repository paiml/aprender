//! Verify embedding values for token 450

use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn main() {
    let path = "/tmp/parity-bench/tinyllama-1.1b-q4_k_m.gguf";
    let mapped = MappedGGUFModel::from_path(path).expect("Failed");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("test");

    let hidden_dim = model.config().hidden_dim;
    let token_id = 450usize;

    // Get token 450's embedding
    let start = token_id * hidden_dim;
    let embedding: Vec<f32> = model.token_embedding()[start..start + hidden_dim].to_vec();

    println!("Token 450 embedding stats:");
    println!("  First 10 values: {:?}", &embedding[..10]);
    println!(
        "  L2 norm: {:.6}",
        embedding.iter().map(|x| x * x).sum::<f32>().sqrt()
    );
    println!(
        "  Mean: {:.6}",
        embedding.iter().sum::<f32>() / hidden_dim as f32
    );
    println!(
        "  Min: {:.6}",
        embedding.iter().copied().fold(f32::INFINITY, f32::min)
    );
    println!(
        "  Max: {:.6}",
        embedding.iter().copied().fold(f32::NEG_INFINITY, f32::max)
    );

    // Count zeros
    let zeros = embedding.iter().filter(|&&x| x == 0.0).count();
    println!(
        "  Zeros: {} ({:.1}%)",
        zeros,
        100.0 * zeros as f32 / hidden_dim as f32
    );

    // Also check BOS (token 1)
    let bos_start = hidden_dim;
    let bos_embedding: Vec<f32> =
        model.token_embedding()[bos_start..bos_start + hidden_dim].to_vec();
    println!("\nBOS (token 1) embedding stats:");
    println!(
        "  L2 norm: {:.6}",
        bos_embedding.iter().map(|x| x * x).sum::<f32>().sqrt()
    );

    // Check if embeddings are all the same (would indicate bug)
    let same_count = embedding.windows(2).filter(|w| w[0] == w[1]).count();
    println!("\nConsecutive equal values: {}", same_count);
}
