//! Debug APR vs GGUF embeddings
use realizar::apr_transformer::AprTransformer;
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let apr_path = "/tmp/qwen2.5-coder-1.5b-q4k.apr";
    let gguf_path =
        "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";

    // Load GGUF
    println!("Loading GGUF...");
    let mapped = MappedGGUFModel::from_path(gguf_path)?;
    let gguf_model = OwnedQuantizedModel::from_mapped(&mapped)?;

    // Load APR
    println!("Loading APR...");
    let apr = AprTransformer::from_apr_file(apr_path)?;

    // Compare embedding dimensions
    println!("\n=== Config Comparison ===");
    println!("GGUF hidden_dim: {}", gguf_model.config().hidden_dim);
    println!("APR hidden_dim: {}", apr.config.hidden_dim);
    println!("GGUF vocab_size: {}", gguf_model.config().vocab_size);
    println!("APR vocab_size: {}", apr.config.vocab_size);

    // Look up token embedding for token 0 (BOS or first token)
    let test_token = 151643u32; // Qwen2 BOS

    // GGUF embedding
    let gguf_embed = &gguf_model.token_embedding();
    let hidden_dim = gguf_model.config().hidden_dim;
    let start = (test_token as usize) * hidden_dim;
    let gguf_vec: Vec<f32> = gguf_embed[start..start + hidden_dim].to_vec();

    // APR embedding
    let apr_hidden = apr.config.hidden_dim;
    let apr_start = (test_token as usize) * apr_hidden;
    let apr_vec: Vec<f32> = apr.token_embedding[apr_start..apr_start + apr_hidden].to_vec();

    println!("\n=== Token {} Embedding (first 10) ===", test_token);
    println!("GGUF: {:?}", &gguf_vec[..10]);
    println!("APR:  {:?}", &apr_vec[..10]);

    // Compute correlation
    let dot: f64 = gguf_vec
        .iter()
        .zip(&apr_vec)
        .map(|(a, b)| (*a as f64) * (*b as f64))
        .sum();
    let gguf_sq: f64 = gguf_vec.iter().map(|a| (*a as f64).powi(2)).sum();
    let apr_sq: f64 = apr_vec.iter().map(|a| (*a as f64).powi(2)).sum();
    let corr = dot / (gguf_sq.sqrt() * apr_sq.sqrt());
    println!("\nCorrelation: {:.6}", corr);

    // Check if embeddings are equal
    let max_diff = gguf_vec
        .iter()
        .zip(&apr_vec)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("Max difference: {:.6}", max_diff);

    Ok(())
}
