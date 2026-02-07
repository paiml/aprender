//! Rosetta Stone - Universal Model Format Converter Example
//!
//! Demonstrates the Rosetta Stone pattern for model format conversion:
//! - Format detection from magic bytes and extensions
//! - Direct conversion paths (A -> B)
//! - Multi-step conversion chains (A -> B -> C)
//! - Round-trip verification for lossless conversion
//! - **Tokenizer Preservation (PMAT-APR-TOK-001)**: Embedded tokenizers travel with the model
//!
//! ## Tokenizer Preservation
//!
//! APR format embeds tokenizers during conversion, making models truly portable:
//! - SafeTensors → APR: Reads sibling `tokenizer.json` (vocab, BOS/EOS tokens)
//! - GGUF → APR: Extracts vocabulary from GGUF metadata
//! - APR inference: Uses embedded tokenizer for automatic token decoding
//!
//! Verification: `strings model.apr | grep tokenizer.vocabulary`
//!
//! Toyota Way Alignment:
//! - **Genchi Genbutsu**: Inspect actual tensor data before/after conversion
//! - **Jidoka**: Stop on any conversion anomaly (dimension mismatch, NaN)
//! - **Kaizen**: Multi-step chains for iterative improvement
//!
//! Run with: `cargo run --example rosetta_stone`

use aprender::format::rosetta::{
    ConversionOptions, ConversionPath, FormatType, RosettaStone, TensorInfo,
};
use std::path::Path;

fn main() {
    println!("=== Rosetta Stone - Universal Model Format Converter ===\n");

    // Part 1: Format Types
    format_types_demo();

    // Part 2: Conversion Paths
    conversion_paths_demo();

    // Part 3: Options Configuration
    options_demo();

    // Part 4: Cycle Detection
    cycle_detection_demo();

    // Part 5: Tensor Information
    tensor_info_demo();

    // Part 6: RosettaStone API
    rosetta_api_demo();

    println!("\n=== Rosetta Stone Demo Complete! ===");
}

fn format_types_demo() {
    println!("--- Part 1: Format Types ---\n");

    // Format detection from file extension
    let files = [
        ("model.gguf", Path::new("model.gguf")),
        ("model.safetensors", Path::new("model.safetensors")),
        ("model.apr", Path::new("model.apr")),
        ("model.bin", Path::new("model.bin")),
    ];

    println!("Format Detection (by extension):");
    for (name, path) in &files {
        match FormatType::from_extension(path) {
            Ok(f) => println!("  {} -> {} (ext: .{})", name, f, f.extension()),
            Err(_) => println!("  {} -> Unknown format", name),
        }
    }

    // Magic bytes (normally read from file)
    println!("\nMagic Bytes:");
    println!("  GGUF: 0x47475546 ('GGUF')");
    println!("  SafeTensors: JSON header starting with '{{\"'");
    println!("  APR: 0x41505232 ('APR2')");

    // Conversion compatibility
    println!("\nConversion Matrix:");
    for source in [FormatType::Gguf, FormatType::SafeTensors, FormatType::Apr] {
        for target in [FormatType::Gguf, FormatType::SafeTensors, FormatType::Apr] {
            let status = if source.can_convert_to(target) {
                "OK"
            } else {
                "SAME"
            };
            println!("  {} -> {}: {}", source, target, status);
        }
    }
    println!();
}

fn conversion_paths_demo() {
    println!("--- Part 2: Conversion Paths ---\n");

    // Direct path: A -> B
    let direct = ConversionPath::direct(FormatType::Gguf, FormatType::SafeTensors);
    let is_direct = direct.intermediates.is_empty();
    println!("Direct Path:");
    println!("  Path: {}", direct);
    println!("  Steps: {}", direct.steps().len());
    println!("  Is Direct: {}", is_direct);
    println!("  Has Cycle: {}", direct.has_cycle());

    // Chain path: A -> B -> C
    let chain = ConversionPath::chain(
        FormatType::Gguf,
        vec![FormatType::SafeTensors],
        FormatType::Apr,
    );
    let chain_is_direct = chain.intermediates.is_empty();
    println!("\nChain Path:");
    println!("  Path: {}", chain);
    println!("  Steps: {}", chain.steps().len());
    println!("  Is Direct: {}", chain_is_direct);
    println!("  Has Cycle: {}", chain.has_cycle());

    // Roundtrip path: A -> B -> A
    let roundtrip = ConversionPath::chain(FormatType::Apr, vec![FormatType::Gguf], FormatType::Apr);
    println!("\nRoundtrip Path:");
    println!("  Path: {}", roundtrip);
    println!("  Steps: {}", roundtrip.steps().len());
    println!("  Is Roundtrip: {}", roundtrip.is_roundtrip());
    println!();
}

fn options_demo() {
    println!("--- Part 3: Options Configuration ---\n");

    // Default options
    let default_opts = ConversionOptions::default();
    println!("Default Options:");
    println!("  Verify: {}", default_opts.verify);
    println!("  Tolerance: {}", default_opts.tolerance);
    println!("  Quantization: {:?}", default_opts.quantization);
    println!("  Preserve Metadata: {}", default_opts.preserve_metadata);
    println!("  Add Provenance: {}", default_opts.add_provenance);
    println!("  Compute Stats: {}", default_opts.compute_stats);

    // Custom options for quantized conversion
    let quant_opts = ConversionOptions {
        quantization: Some("Q4_K_M".to_string()),
        verify: true,
        tolerance: 1e-4,
        preserve_metadata: true,
        add_provenance: true,
        compute_stats: true,
        tokenizer_path: None,
    };
    println!("\nQuantized Conversion Options:");
    println!("  Quantization: {:?}", quant_opts.quantization);
    println!("  Verify: {}", quant_opts.verify);
    println!("  Tolerance: {}", quant_opts.tolerance);
    println!("  Compute Stats: {}", quant_opts.compute_stats);

    // Strict verification options
    let strict_opts = ConversionOptions {
        verify: true,
        tolerance: 1e-7,
        ..Default::default()
    };
    println!("\nStrict Verification Options:");
    println!("  Tolerance: {} (very strict)", strict_opts.tolerance);
    println!();
}

fn cycle_detection_demo() {
    println!("--- Part 4: Cycle Detection ---\n");

    // Valid chain (no cycle)
    let valid_chain = ConversionPath::chain(
        FormatType::Gguf,
        vec![FormatType::SafeTensors],
        FormatType::Apr,
    );
    println!("Valid Chain: {}", valid_chain);
    println!("  Has Cycle: {} (should be false)", valid_chain.has_cycle());

    // Invalid chain (has cycle in middle)
    let cycle_chain = ConversionPath::chain(
        FormatType::Gguf,
        vec![FormatType::SafeTensors, FormatType::SafeTensors],
        FormatType::Apr,
    );
    println!("\nChain with Cycle: {}", cycle_chain);
    println!("  Has Cycle: {} (should be true)", cycle_chain.has_cycle());

    // Roundtrip is NOT a cycle (it's intentional)
    let roundtrip = ConversionPath::chain(FormatType::Apr, vec![FormatType::Gguf], FormatType::Apr);
    println!("\nRoundtrip (intentional): {}", roundtrip);
    println!(
        "  Has Cycle: {} (roundtrip start/end same is allowed)",
        roundtrip.has_cycle()
    );
    println!();
}

fn tensor_info_demo() {
    println!("--- Part 5: Tensor Information ---\n");

    // Create tensor information structures
    let tensors = vec![
        TensorInfo {
            name: "model.embed_tokens.weight".to_string(),
            shape: vec![32000, 4096],
            dtype: "F16".to_string(),
            size_bytes: 32000 * 4096 * 2,
            stats: None,
        },
        TensorInfo {
            name: "model.layers.0.self_attn.q_proj.weight".to_string(),
            shape: vec![4096, 4096],
            dtype: "Q4_K".to_string(),
            size_bytes: 4096 * 4096 / 2, // Q4 is ~0.5 bytes per element
            stats: None,
        },
        TensorInfo {
            name: "model.layers.0.self_attn.k_proj.weight".to_string(),
            shape: vec![1024, 4096],
            dtype: "Q4_K".to_string(),
            size_bytes: 1024 * 4096 / 2,
            stats: None,
        },
        TensorInfo {
            name: "lm_head.weight".to_string(),
            shape: vec![32000, 4096],
            dtype: "F16".to_string(),
            size_bytes: 32000 * 4096 * 2,
            stats: None,
        },
    ];

    println!("Sample Tensor Inventory:");
    let mut total_bytes = 0u64;
    for t in &tensors {
        let shape_str: String = t
            .shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join("x");
        println!(
            "  {} [{}] {} ({:.1} MB)",
            t.name,
            shape_str,
            t.dtype,
            t.size_bytes as f64 / (1024.0 * 1024.0)
        );
        total_bytes += t.size_bytes as u64;
    }
    println!(
        "\nTotal: {} tensors, {:.1} MB",
        tensors.len(),
        total_bytes as f64 / (1024.0 * 1024.0)
    );
    println!();
}

fn rosetta_api_demo() {
    println!("--- Part 6: RosettaStone API ---\n");

    // Create default RosettaStone
    let _rosetta = RosettaStone::new();
    println!("RosettaStone (default options):");
    println!("  Created successfully");

    // Create with custom options
    let custom_opts = ConversionOptions {
        verify: true,
        tolerance: 1e-5,
        quantization: Some("Q8_0".to_string()),
        preserve_metadata: true,
        add_provenance: true,
        compute_stats: true,
        tokenizer_path: None,
    };
    let _custom_rosetta = RosettaStone::with_options(custom_opts);
    println!("\nRosettaStone (custom options):");
    println!("  Quantization: Q8_0");
    println!("  Verify: enabled");
    println!("  Tolerance: 1e-5");
    println!("  Compute Stats: enabled");

    // API documentation
    println!("\nAPI Methods:");
    println!("  rosetta.inspect(path)              - Detect format and list tensors");
    println!("  rosetta.convert(src, dst, opts)    - Convert between formats");
    println!("  rosetta.chain(src, formats, dir)   - Multi-step conversion");
    println!("  rosetta.verify_roundtrip(src, fmt) - Verify lossless conversion");

    // Example CLI usage (informational)
    println!("\nCLI Usage Examples:");
    println!("  $ apr rosetta inspect model.gguf");
    println!("  $ apr rosetta convert model.gguf model.safetensors --verify");
    println!("  $ apr rosetta chain model.gguf safetensors apr --work-dir /tmp");
    println!("  $ apr rosetta verify model.apr --intermediate gguf --tolerance 1e-4");

    // Note about actual file operations
    println!("\nNote: Actual conversion requires model files. This demo shows the API.");
    println!("      For file operations, use the apr CLI or rosetta.inspect/convert methods.");
}
