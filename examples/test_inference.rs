//! Quick inference test for fixed Q4_0 model
use realizar::apr_transformer::AprTransformer;
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, QuantizedGenerateConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let apr_path = "/tmp/qwen2-test6.apr";
    let gguf_path =
        "/home/noah/src/HF-Advanced-Fine-Tuning/corpus/models/qwen2-0.5b-instruct-q4_0.gguf";

    println!("Loading APR model...");
    let apr_model = AprTransformer::from_apr_file(apr_path)?;

    println!("Loading GGUF model...");
    let mapped = MappedGGUFModel::from_path(gguf_path)?;
    let gguf_model = OwnedQuantizedModel::from_mapped(&mapped)?;

    // Generate with both models
    let prompt_tokens: Vec<u32> = vec![151643]; // BOS token

    println!("\n=== GGUF Generation ===");
    let config = QuantizedGenerateConfig {
        max_tokens: 5,
        ..Default::default()
    };
    let gguf_output = gguf_model.generate(&prompt_tokens, &config)?;
    println!("GGUF tokens: {:?}", gguf_output);

    println!("\n=== APR Generation ===");
    let apr_output = apr_model.generate(&prompt_tokens, 5)?;
    println!("APR tokens: {:?}", apr_output);

    // Compare
    println!("\n=== Comparison ===");
    if gguf_output == apr_output {
        println!("✓ Generated tokens match exactly!");
    } else {
        println!("⚠ Generated tokens differ:");
        for (i, (g, a)) in gguf_output.iter().zip(apr_output.iter()).enumerate() {
            let marker = if g == a { "✓" } else { "✗" };
            println!("  {}: GGUF={}, APR={} {}", i, g, a, marker);
        }
    }

    Ok(())
}
