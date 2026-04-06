//! Test APR vs GGUF forward pass for Q4_0 quantized model
//!
//! This tests the problematic Q4_0 → F32 dequantization path.
use std::path::Path;

use realizar::apr_transformer::AprTransformer;
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let gguf_path =
        "/home/noah/src/HF-Advanced-Fine-Tuning/corpus/models/qwen2-0.5b-instruct-q4_0.gguf";
    let apr_path = "/tmp/qwen2-test6.apr";

    // Check if files exist
    if !Path::new(gguf_path).exists() {
        eprintln!("GGUF file not found: {}", gguf_path);
        return Ok(());
    }
    if !Path::new(apr_path).exists() {
        eprintln!("APR file not found: {}", apr_path);
        return Ok(());
    }

    println!("Loading GGUF model: {}", gguf_path);
    let mapped = MappedGGUFModel::from_path(gguf_path)?;
    let gguf_model = OwnedQuantizedModel::from_mapped(&mapped)?;
    println!(
        "GGUF config: hidden_dim={}, num_layers={}, vocab_size={}",
        gguf_model.config().hidden_dim,
        gguf_model.config().num_layers,
        gguf_model.config().vocab_size
    );

    println!("\nLoading APR model: {}", apr_path);
    let apr_model = AprTransformer::from_apr_file(apr_path)?;
    println!(
        "APR config: hidden_dim={}, num_layers={}, vocab_size={}",
        apr_model.config.hidden_dim, apr_model.config.num_layers, apr_model.config.vocab_size
    );

    // Compare embeddings
    println!("\n=== Comparing Embeddings ===");
    let gguf_embed = gguf_model.embed(&[0, 1, 2]);
    let apr_embed = apr_model.embed(&[0, 1, 2]);
    println!("GGUF embed(0..3) first 5: {:?}", &gguf_embed[..5]);
    println!("APR embed(0..3) first 5: {:?}", &apr_embed[..5]);

    // Test with BOS token
    let bos: u32 = 151643; // Qwen2 BOS token
    println!("\n=== Forward with BOS token [{}] ===", bos);

    // GGUF forward
    let gguf_logits = gguf_model.forward(&[bos])?;
    let gguf_argmax = gguf_logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    println!(
        "GGUF argmax: {} logit={:.4}",
        gguf_argmax, gguf_logits[gguf_argmax]
    );
    println!("GGUF first 10 logits: {:?}", &gguf_logits[..10]);

    // APR forward
    let apr_logits = apr_model.forward(&[bos])?;
    let apr_argmax = apr_logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    println!(
        "\nAPR argmax: {} logit={:.4}",
        apr_argmax, apr_logits[apr_argmax]
    );
    println!("APR first 10 logits: {:?}", &apr_logits[..10]);

    // Compare logits
    let correlation: f32 = {
        let mean_a: f32 = gguf_logits.iter().sum::<f32>() / gguf_logits.len() as f32;
        let mean_b: f32 = apr_logits.iter().sum::<f32>() / apr_logits.len() as f32;
        let mut cov = 0.0f32;
        let mut var_a = 0.0f32;
        let mut var_b = 0.0f32;
        for (a, b) in gguf_logits.iter().zip(apr_logits.iter()) {
            let da = a - mean_a;
            let db = b - mean_b;
            cov += da * db;
            var_a += da * da;
            var_b += db * db;
        }
        cov / (var_a.sqrt() * var_b.sqrt())
    };

    let mean_diff: f32 = gguf_logits
        .iter()
        .zip(apr_logits.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / gguf_logits.len() as f32;

    let max_diff: f32 = gguf_logits
        .iter()
        .zip(apr_logits.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    println!("\n=== Comparison ===");
    println!("Correlation: {:.6}", correlation);
    println!("Mean absolute diff: {:.4}", mean_diff);
    println!("Max absolute diff: {:.4}", max_diff);

    if correlation < 0.99 {
        println!("\n⚠ WARNING: Low correlation indicates significant divergence!");
    }

    Ok(())
}
