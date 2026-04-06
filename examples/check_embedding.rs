//! Check if embedding lookup produces correct values
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn main() {
    let path = "../aprender/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf";
    let mapped = MappedGGUFModel::from_path(path).expect("load");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("model");

    eprintln!("hidden_dim: {}", model.config().hidden_dim);
    eprintln!("vocab_size: {}", model.config().vocab_size);
    eprintln!("token_embedding len: {}", model.token_embedding().len());
    eprintln!(
        "Expected embedding size: {} * {} = {}",
        model.config().vocab_size,
        model.config().hidden_dim,
        model.config().vocab_size * model.config().hidden_dim
    );

    // Check embedding for token 9707 ("Hello")
    let token_id = 9707u32;
    let hidden_dim = model.config().hidden_dim;
    let start = token_id as usize * hidden_dim;
    let end = start + hidden_dim;

    if start >= model.token_embedding().len() {
        eprintln!(
            "ERROR: Token {} embedding out of bounds! start={} but len={}",
            token_id,
            start,
            model.token_embedding().len()
        );
        return;
    }

    let emb = &model.token_embedding()[start..end];
    eprintln!("\nToken {} embedding (first 8): {:?}", token_id, &emb[..8]);

    let emb_sum: f32 = emb.iter().sum();
    let emb_norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
    eprintln!("Embedding sum: {:.4}, L2 norm: {:.4}", emb_sum, emb_norm);

    // Check several tokens
    for tok in [0u32, 1, 100, 1000, 9707, 151644] {
        let start = tok as usize * hidden_dim;
        if start + hidden_dim <= model.token_embedding().len() {
            let e = &model.token_embedding()[start..start + hidden_dim];
            let sum: f32 = e.iter().sum();
            let norm: f32 = e.iter().map(|x| x * x).sum::<f32>().sqrt();
            eprintln!(
                "Token {:6}: sum={:.4}, norm={:.4}, first2=[{:.4}, {:.4}]",
                tok, sum, norm, e[0], e[1]
            );
        }
    }

    // Now do a forward pass for single token and check output
    let embed = model.embed(&[token_id]);
    eprintln!("\nAfter embed() call:");
    eprintln!("embed len: {}", embed.len());
    eprintln!("embed first 8: {:?}", &embed[..8.min(embed.len())]);

    // These should match
    eprintln!(
        "\nDirect vs embed() match: {}",
        emb[..8]
            .iter()
            .zip(embed[..8].iter())
            .all(|(a, b)| (a - b).abs() < 1e-6)
    );
}
