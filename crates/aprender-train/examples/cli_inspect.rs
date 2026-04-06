//! Inspect Example
//!
//! Demonstrates programmatic model and data inspection.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example cli_inspect
//! ```
//!
//! Or use the CLI:
//! ```bash
//! entrenar inspect model.safetensors
//! entrenar inspect data/train.parquet --mode summary
//! ```

use safetensors::SafeTensors;
use std::path::Path;

fn main() {
    println!("Model Inspection Example");
    println!("========================\n");

    // Example: Inspect a SafeTensors file if available
    let model_path = Path::new("model.safetensors");

    if model_path.exists() {
        inspect_safetensors(model_path);
    } else {
        println!("No SafeTensors model found. Creating demo inspection...\n");
        demo_inspection();
    }

    // Example: Inspect data file
    let data_path = Path::new("data/train.parquet");
    if data_path.exists() {
        inspect_data(data_path);
    }
}

fn inspect_safetensors(path: &Path) {
    println!("Inspecting: {}\n", path.display());

    let data = match std::fs::read(path) {
        Ok(d) => d,
        Err(e) => {
            println!("Failed to read file: {e}");
            return;
        }
    };

    let tensors = match SafeTensors::deserialize(&data) {
        Ok(t) => t,
        Err(e) => {
            println!("Failed to parse SafeTensors: {e}");
            return;
        }
    };

    let tensor_names: Vec<String> =
        tensors.names().iter().map(std::string::ToString::to_string).collect();
    let mut total_params: u64 = 0;

    for name in &tensor_names {
        if let Ok(tensor) = tensors.tensor(name) {
            let params: u64 = tensor.shape().iter().product::<usize>() as u64;
            total_params += params;
        }
    }

    let file_size = data.len();

    println!("Model Information:");
    println!("  File size: {:.2} MB", file_size as f64 / 1_000_000.0);
    println!("  Parameters: {:.2}B", total_params as f64 / 1e9);
    println!("  Tensors: {}", tensor_names.len());

    println!("\nTensor Details (first 10):");
    for name in tensor_names.iter().take(10) {
        if let Ok(tensor) = tensors.tensor(name) {
            println!("  {}: {:?} ({:?})", name, tensor.shape(), tensor.dtype());
        }
    }

    if tensor_names.len() > 10 {
        println!("  ... and {} more tensors", tensor_names.len() - 10);
    }
}

fn inspect_data(path: &Path) {
    println!("\nInspecting data: {}\n", path.display());

    let metadata = match std::fs::metadata(path) {
        Ok(m) => m,
        Err(e) => {
            println!("Failed to read metadata: {e}");
            return;
        }
    };

    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");

    println!("Data Summary:");
    println!("  File size: {:.2} MB", metadata.len() as f64 / 1_000_000.0);
    println!("  Format: {ext}");
}

fn demo_inspection() {
    // Demonstrate what inspection output looks like
    println!("Demo Model Information:");
    println!("  File size: 125.42 MB");
    println!("  Parameters: 0.03B");
    println!("  Tensors: 48");

    println!("\nDemo Tensor Details:");
    println!("  model.embed_tokens.weight: [32000, 4096] (F32)");
    println!("  model.layers.0.self_attn.q_proj.weight: [4096, 4096] (F32)");
    println!("  model.layers.0.self_attn.k_proj.weight: [4096, 4096] (F32)");
    println!("  model.layers.0.self_attn.v_proj.weight: [4096, 4096] (F32)");
    println!("  model.layers.0.self_attn.o_proj.weight: [4096, 4096] (F32)");
    println!("  ... and 43 more tensors");
}
