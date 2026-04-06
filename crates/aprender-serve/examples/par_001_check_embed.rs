//! PAR-001: Check embedding table values

use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn l2_norm(v: &[f32]) -> f32 {
    (v.iter().map(|x| x * x).sum::<f32>()).sqrt()
}

fn main() {
    let path = "/tmp/parity-bench/tinyllama-1.1b-q4_k_m.gguf";
    let mapped = MappedGGUFModel::from_path(path).expect("Failed");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("test");
    let vocab = mapped.model.vocabulary().expect("test");

    println!("=== PAR-001: Embedding Check ===\n");

    let hidden_dim = model.config().hidden_dim;
    let vocab_size = model.token_embedding().len() / hidden_dim;

    println!("hidden_dim: {}", hidden_dim);
    println!("vocab_size: {}", vocab_size);
    println!(
        "embedding table size: {} ({}x{})",
        model.token_embedding().len(),
        vocab_size,
        hidden_dim
    );

    // Check a few token embeddings
    let tokens_to_check = [
        1,     // BOS <s>
        2,     // EOS </s>
        12,    // <0x09> (TAB)
        29871, // ▁ (space)
        29906, // 2
        15043, // ▁Hello
    ];

    for &tok in &tokens_to_check {
        if tok < vocab_size {
            let start = tok * hidden_dim;
            let end = start + hidden_dim;
            let embedding = &model.token_embedding()[start..end];
            let l2 = l2_norm(embedding);
            let min = embedding.iter().copied().fold(f32::INFINITY, f32::min);
            let max = embedding.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mean = embedding.iter().sum::<f32>() / hidden_dim as f32;
            let name = vocab.get(tok).map_or("?", |s| s.as_str());

            println!("\nToken {} ('{}'):", tok, name);
            println!(
                "  L2: {:.4}, min: {:.4}, max: {:.4}, mean: {:.6}",
                l2, min, max, mean
            );
            println!("  First 10: {:?}", &embedding[..10.min(hidden_dim)]);
        }
    }

    // Check if TAB embedding looks unusual
    let tab_start = 12 * hidden_dim;
    let tab_embedding = &model.token_embedding()[tab_start..tab_start + hidden_dim];
    let bos_start = hidden_dim; // BOS token at index 1
    let bos_embedding = &model.token_embedding()[bos_start..bos_start + hidden_dim];

    // Compute similarity between TAB and BOS embeddings
    let dot: f32 = tab_embedding
        .iter()
        .zip(bos_embedding.iter())
        .map(|(a, b)| a * b)
        .sum();
    let sim = dot / (l2_norm(tab_embedding) * l2_norm(bos_embedding));
    println!("\n\nCosine similarity TAB vs BOS: {:.4}", sim);

    // Also check output norm
    println!("\n=== Output Norm ===");
    println!(
        "output_norm_weight len: {}",
        model.output_norm_weight().len()
    );
    println!(
        "output_norm_weight L2: {:.4}",
        l2_norm(model.output_norm_weight())
    );
    println!(
        "First 10: {:?}",
        &model.output_norm_weight()[..10.min(model.output_norm_weight().len())]
    );

    println!("\n=== Complete ===");
}
