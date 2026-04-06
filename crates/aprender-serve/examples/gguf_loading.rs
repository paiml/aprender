//! GGUF loading example for realizar
//!
//! Demonstrates how to:
//! - Load GGUF files (llama.cpp/Ollama format)
//! - Parse header, metadata, and tensor information
//! - Inspect model structure
//! - Extract and dequantize tensor data
//!
//! GGUF is the standard format for llama.cpp and Ollama models.

use std::io::{Cursor, Write};

use realizar::gguf::{GGUFModel, GGUFValue, GGUF_MAGIC, GGUF_TYPE_F32, GGUF_TYPE_Q4_0};

fn main() {
    println!("=== GGUF Loading Example ===\n");

    // Example 1: Create a simple GGUF file (simulating llama.cpp output)
    println!("--- Creating Example GGUF File ---");
    let gguf_data = create_example_gguf_model();
    println!("Created GGUF file: {} bytes", gguf_data.len());
    println!();

    // Example 2: Load and parse GGUF
    println!("--- Loading GGUF Model ---");
    let model = GGUFModel::from_bytes(&gguf_data).expect("Failed to load GGUF model");

    println!("Successfully loaded model");
    println!("  - Number of tensors: {}", model.tensors.len());
    println!("  - Metadata entries: {}", model.metadata.len());
    println!();

    // Example 3: Inspect header
    println!("--- GGUF Header ---");
    println!(
        "Magic: 0x{:08X} ({})",
        model.header.magic,
        if model.header.magic == GGUF_MAGIC {
            "valid ✓"
        } else {
            "invalid ✗"
        }
    );
    println!("Version: {}", model.header.version);
    println!("Tensor count: {}", model.header.tensor_count);
    println!("Metadata count: {}", model.header.metadata_count);
    println!();

    // Example 4: Inspect metadata
    println!("--- Metadata ---");
    for (key, value) in &model.metadata {
        match value {
            GGUFValue::String(s) => println!("{}: \"{}\"", key, s),
            GGUFValue::UInt32(v) => println!("{}: {}", key, v),
            GGUFValue::UInt64(v) => println!("{}: {}", key, v),
            GGUFValue::Float32(v) => println!("{}: {}", key, v),
            GGUFValue::Bool(v) => println!("{}: {}", key, v),
            GGUFValue::Array(arr) => println!("{}: [array with {} elements]", key, arr.len()),
            _ => println!("{}: {:?}", key, value),
        }
    }
    println!();

    // Example 5: Inspect tensor information
    println!("--- Tensor Information ---");
    for tensor in &model.tensors {
        println!("Tensor: {}", tensor.name);
        println!("  - Dimensions: {:?}", tensor.dims);
        println!(
            "  - Quantization type: {} ({})",
            tensor.qtype,
            match tensor.qtype {
                GGUF_TYPE_F32 => "F32 (unquantized)",
                GGUF_TYPE_Q4_0 => "Q4_0 (4-bit)",
                _ => "other",
            }
        );

        // Calculate total elements
        let total_elements: u64 = tensor.dims.iter().product();
        println!("  - Total elements: {}", total_elements);
        println!("  - Data offset: {}", tensor.offset);
        println!();
    }

    // Example 6: Extract tensor data
    println!("--- Extracting Tensor Data ---");

    // Extract embedding weights (F32 unquantized)
    match model.get_tensor_f32("embedding.weight", &gguf_data) {
        Ok(weights) => {
            println!("Embedding weights (F32):");
            println!("  - Length: {}", weights.len());
            println!("  - First 5 values: {:?}", &weights[..5.min(weights.len())]);
            println!(
                "  - Stats: min={:.4}, max={:.4}, mean={:.4}",
                weights.iter().copied().fold(f32::INFINITY, f32::min),
                weights.iter().copied().fold(f32::NEG_INFINITY, f32::max),
                weights.iter().sum::<f32>() / weights.len() as f32
            );
        },
        Err(e) => println!("  Failed to extract embedding.weight: {}", e),
    }
    println!();

    // Extract quantized tensor (Q4_0)
    match model.get_tensor_f32("layer.0.weight", &gguf_data) {
        Ok(dequantized) => {
            println!("Layer 0 weights (Q4_0 dequantized to F32):");
            println!("  - Length: {}", dequantized.len());
            println!(
                "  - First 5 values: {:?}",
                &dequantized[..5.min(dequantized.len())]
            );
            println!(
                "  - Stats: min={:.4}, max={:.4}, mean={:.4}",
                dequantized.iter().copied().fold(f32::INFINITY, f32::min),
                dequantized
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, f32::max),
                dequantized.iter().sum::<f32>() / dequantized.len() as f32
            );
        },
        Err(e) => println!("  Failed to extract layer.0.weight: {}", e),
    }
    println!();

    // Example 7: Summary
    println!("--- Summary ---");
    println!("GGUF format features demonstrated:");
    println!("  ✓ Binary format parsing (magic number, version)");
    println!("  ✓ Metadata extraction (key-value pairs)");
    println!("  ✓ Tensor information (shapes, types, offsets)");
    println!("  ✓ Unquantized tensor data (F32)");
    println!("  ✓ Quantized tensor data (Q4_0 with dequantization)");
    println!();
    println!("This demonstrates llama.cpp/Ollama model compatibility!");
    println!();

    println!("=== GGUF Loading Complete ===");
}

/// Create a minimal example GGUF file for demonstration
fn create_example_gguf_model() -> Vec<u8> {
    let mut buffer = Vec::new();
    let mut cursor = Cursor::new(&mut buffer);

    // Header
    cursor
        .write_all(&GGUF_MAGIC.to_le_bytes())
        .expect("Failed to write GGUF magic number");
    cursor
        .write_all(&3u32.to_le_bytes())
        .expect("Failed to write GGUF version");
    cursor
        .write_all(&2u64.to_le_bytes())
        .expect("Failed to write tensor count");
    cursor
        .write_all(&3u64.to_le_bytes())
        .expect("Failed to write metadata count");

    // Metadata 1: model name (string)
    write_string(&mut cursor, "model.name");
    cursor
        .write_all(&8u32.to_le_bytes())
        .expect("Failed to write metadata type for model.name");
    write_string(&mut cursor, "demo-model");

    // Metadata 2: vocab size (uint32)
    write_string(&mut cursor, "vocab.size");
    cursor
        .write_all(&4u32.to_le_bytes())
        .expect("Failed to write metadata type for vocab.size");
    cursor
        .write_all(&1000u32.to_le_bytes())
        .expect("Failed to write vocab size value");

    // Metadata 3: hidden size (uint32)
    write_string(&mut cursor, "hidden.size");
    cursor
        .write_all(&4u32.to_le_bytes())
        .expect("Failed to write metadata type for hidden.size");
    cursor
        .write_all(&256u32.to_le_bytes())
        .expect("Failed to write hidden size value");

    // Tensor info 1: embedding.weight (F32, unquantized)
    write_string(&mut cursor, "embedding.weight");
    cursor
        .write_all(&2u32.to_le_bytes())
        .expect("Failed to write n_dims for embedding.weight");
    cursor
        .write_all(&1000u64.to_le_bytes())
        .expect("Failed to write dim 0 for embedding.weight");
    cursor
        .write_all(&256u64.to_le_bytes())
        .expect("Failed to write dim 1 for embedding.weight");
    cursor
        .write_all(&GGUF_TYPE_F32.to_le_bytes())
        .expect("Failed to write tensor type for embedding.weight");
    cursor
        .write_all(&0u64.to_le_bytes())
        .expect("Failed to write offset for embedding.weight");

    // Tensor info 2: layer.0.weight (Q4_0, quantized)
    write_string(&mut cursor, "layer.0.weight");
    cursor
        .write_all(&2u32.to_le_bytes())
        .expect("Failed to write n_dims for layer.0.weight");
    cursor
        .write_all(&256u64.to_le_bytes())
        .expect("Failed to write dim 0 for layer.0.weight");
    cursor
        .write_all(&256u64.to_le_bytes())
        .expect("Failed to write dim 1 for layer.0.weight");
    cursor
        .write_all(&GGUF_TYPE_Q4_0.to_le_bytes())
        .expect("Failed to write tensor type for layer.0.weight");

    // Calculate offset for second tensor
    // First tensor: 1000 * 256 * 4 bytes (F32)
    let tensor1_size = 1000 * 256 * 4;
    cursor
        .write_all(&(tensor1_size as u64).to_le_bytes())
        .expect("Failed to write offset for layer.0.weight");

    // Alignment padding (GGUF requires 32-byte alignment for tensor data)
    let alignment = 32;
    let current_pos = buffer.len();
    let padding = (alignment - (current_pos % alignment)) % alignment;
    buffer.resize(current_pos + padding, 0);

    // Tensor data 1: embedding.weight (F32)
    for i in 0..1000 {
        for j in 0..256 {
            let value = ((i + j) as f32) * 0.01;
            buffer.extend_from_slice(&value.to_le_bytes());
        }
    }

    // Tensor data 2: layer.0.weight (Q4_0)
    // Q4_0 format: groups of 32 floats quantized to 4 bits each
    // Each block: 1 f16 scale + 16 bytes of 4-bit values (32 values packed)
    let num_elements = 256 * 256;
    let num_blocks = num_elements / 32;

    for _ in 0..num_blocks {
        // Scale factor (f16, we'll use 1.0)
        buffer.extend_from_slice(&1.0f32.to_le_bytes()[..2]); // simplified f16

        // 16 bytes of 4-bit quantized data (32 values)
        buffer.extend(std::iter::repeat_n(0x12, 16)); // Demo data: each byte contains two 4-bit values
    }

    buffer
}

/// Helper to write a length-prefixed string
fn write_string(cursor: &mut Cursor<&mut Vec<u8>>, s: &str) {
    let bytes = s.as_bytes();
    cursor
        .write_all(&(bytes.len() as u64).to_le_bytes())
        .expect("Failed to write string length");
    cursor
        .write_all(bytes)
        .expect("Failed to write string bytes");
}
