//! APR Format with JSON Metadata Example
//!
//! Demonstrates how to embed arbitrary metadata (vocab, config, tokenizer settings)
//! alongside tensors in a single WASM-deployable file.
//!
//! Run with: `cargo run --example apr_with_metadata`

use aprender::serialization::apr::{AprReader, AprWriter};
use serde_json::json;

fn main() -> Result<(), String> {
    println!("=== APR Format with JSON Metadata ===\n");

    // Create a writer
    let mut writer = AprWriter::new();

    // Add model configuration metadata
    writer.set_metadata("model_type", json!("whisper-tiny"));
    writer.set_metadata("n_vocab", json!(51865));
    writer.set_metadata("n_audio_ctx", json!(1500));
    writer.set_metadata("n_audio_state", json!(384));
    writer.set_metadata("n_layers", json!(4));

    // Add vocabulary as metadata (for BPE tokenization)
    let vocab_sample = json!({
        "tokens": ["<|endoftext|>", "<|startoftranscript|>", "the", "a", "is"],
        "merges": [["t", "h"], ["th", "e"], ["a", "n"]],
        "special_tokens": {
            "eot": 50256,
            "sot": 50257,
            "transcribe": 50358
        }
    });
    writer.set_metadata("tokenizer", vocab_sample);

    // Add model tensors
    println!("Adding tensors...");
    writer.add_tensor_f32(
        "encoder.conv1.weight",
        vec![384, 80, 3],
        &vec![0.01; 384 * 80 * 3],
    );
    writer.add_tensor_f32("encoder.conv1.bias", vec![384], &vec![0.0; 384]);
    writer.add_tensor_f32(
        "decoder.embed_tokens.weight",
        vec![51865, 384],
        &vec![0.001; 51865 * 384],
    );

    // Write to bytes (could also write to file)
    let bytes = writer.to_bytes()?;
    println!(
        "Total file size: {} bytes ({:.2} MB)",
        bytes.len(),
        bytes.len() as f64 / 1_000_000.0
    );

    // Read it back
    println!("\nReading APR file...");
    let reader = AprReader::from_bytes(bytes)?;

    // Access metadata
    println!("\n--- Metadata ---");
    if let Some(model_type) = reader.get_metadata("model_type") {
        println!("Model type: {}", model_type);
    }
    if let Some(n_vocab) = reader.get_metadata("n_vocab") {
        println!("Vocab size: {}", n_vocab);
    }
    if let Some(tokenizer) = reader.get_metadata("tokenizer") {
        println!(
            "Tokenizer config: {}",
            serde_json::to_string_pretty(tokenizer).unwrap_or_default()
        );
    }

    // Access tensors
    println!("\n--- Tensors ---");
    for tensor in &reader.tensors {
        println!(
            "  {} - shape: {:?}, dtype: {}, size: {} bytes",
            tensor.name, tensor.shape, tensor.dtype, tensor.size
        );
    }

    // Read specific tensor data
    let conv_weight = reader.read_tensor_f32("encoder.conv1.weight")?;
    println!(
        "\nFirst 5 values of encoder.conv1.weight: {:?}",
        &conv_weight[..5]
    );

    println!("\n=== Done ===");
    Ok(())
}
