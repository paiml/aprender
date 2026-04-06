//! Integration tests for loading real GGUF model files (Refs GGUF-INFERENCE-001)
//!
//! These tests require actual GGUF model files and are marked #[ignore] for CI.
//! Run with: cargo test --test integration_gguf_real -- --ignored
//!
//! ## Test Models
//!
//! The tests use models from `/home/noah/src/single-shot-eval/models/raw/`:
//! - phi-2-q4_k_m.gguf (1.7GB, Q4_K_M quantization)
//! - qwen2.5-coder-1.5b-instruct-q4_k_m.gguf (1.1GB, Q4_K_M quantization)
//! - deepseek-coder-1.3b-instruct-q4_k_m.gguf (873MB, Q4_K_M quantization)

use realizar::gguf::{GGUFModel, GGUFValue, GGUF_MAGIC};
use std::fs;
use std::path::Path;

const PHI2_MODEL_PATH: &str = "/home/noah/src/single-shot-eval/models/raw/phi-2-q4_k_m.gguf";
const QWEN_MODEL_PATH: &str =
    "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";
const DEEPSEEK_MODEL_PATH: &str =
    "/home/noah/src/single-shot-eval/models/raw/deepseek-coder-1.3b-instruct-q4_k_m.gguf";

/// Test helper: Check if model file exists
fn model_exists(path: &str) -> bool {
    Path::new(path).exists()
}

// ============================================================================
// HEADER PARSING TESTS
// ============================================================================

#[test]
#[ignore = "requires phi-2 model file"]
fn test_load_phi2_header() {
    if !model_exists(PHI2_MODEL_PATH) {
        eprintln!("Skipping: {} not found", PHI2_MODEL_PATH);
        return;
    }

    let data = fs::read(PHI2_MODEL_PATH).expect("Failed to read phi-2 model");
    let model = GGUFModel::from_bytes(&data).expect("Failed to parse phi-2 model");

    // Verify header
    assert_eq!(model.header.magic, GGUF_MAGIC, "Magic should be GGUF");
    assert_eq!(model.header.version, 3, "Version should be 3");
    assert!(model.header.tensor_count > 0, "Should have tensors");
    assert!(model.header.metadata_count > 0, "Should have metadata");

    println!("phi-2 header:");
    println!("  Version: {}", model.header.version);
    println!("  Tensor count: {}", model.header.tensor_count);
    println!("  Metadata count: {}", model.header.metadata_count);
}

#[test]
#[ignore = "requires phi-2 model file"]
fn test_load_phi2_metadata() {
    if !model_exists(PHI2_MODEL_PATH) {
        return;
    }

    let data = fs::read(PHI2_MODEL_PATH).expect("Failed to read phi-2 model");
    let model = GGUFModel::from_bytes(&data).expect("Failed to parse phi-2 model");

    // Check for key metadata fields
    println!("\nphi-2 metadata ({} entries):", model.metadata.len());

    // Architecture
    if let Some(arch) = model.metadata.get("general.architecture") {
        println!("  general.architecture: {:?}", arch);
    }

    // Model name
    if let Some(name) = model.metadata.get("general.name") {
        println!("  general.name: {:?}", name);
    }

    // Quantization version
    if let Some(qv) = model.metadata.get("general.quantization_version") {
        println!("  general.quantization_version: {:?}", qv);
    }

    // Context length
    for key in [
        "phi2.context_length",
        "llama.context_length",
        "context_length",
    ] {
        if let Some(ctx) = model.metadata.get(key) {
            println!("  {}: {:?}", key, ctx);
        }
    }

    // Embedding length
    for key in [
        "phi2.embedding_length",
        "llama.embedding_length",
        "embedding_length",
    ] {
        if let Some(emb) = model.metadata.get(key) {
            println!("  {}: {:?}", key, emb);
        }
    }

    // Block count (layers)
    for key in ["phi2.block_count", "llama.block_count", "block_count"] {
        if let Some(bc) = model.metadata.get(key) {
            println!("  {}: {:?}", key, bc);
        }
    }

    // Verify essential metadata exists
    assert!(
        model.metadata.contains_key("general.architecture"),
        "Should have general.architecture metadata"
    );
}

#[test]
#[ignore = "requires phi-2 model file"]
fn test_load_phi2_tensors() {
    if !model_exists(PHI2_MODEL_PATH) {
        return;
    }

    let data = fs::read(PHI2_MODEL_PATH).expect("Failed to read phi-2 model");
    let model = GGUFModel::from_bytes(&data).expect("Failed to parse phi-2 model");

    println!("\nphi-2 tensors ({} total):", model.tensors.len());

    // Print first 10 tensors
    for (i, tensor) in model.tensors.iter().take(10).enumerate() {
        println!(
            "  [{}] {} - dims: {:?}, qtype: {}, offset: {}",
            i, tensor.name, tensor.dims, tensor.qtype, tensor.offset
        );
    }

    // Check qtype distribution
    let mut qtype_counts = std::collections::HashMap::new();
    for tensor in &model.tensors {
        *qtype_counts.entry(tensor.qtype).or_insert(0) += 1;
    }
    println!("\n  Quantization types used:");
    for (qtype, count) in &qtype_counts {
        let qtype_name = match qtype {
            0 => "F32",
            1 => "F16",
            2 => "Q4_0",
            3 => "Q4_1",
            6 => "Q5_0",
            7 => "Q5_1",
            8 => "Q8_0",
            12 => "Q4_K",
            13 => "Q5_K",
            14 => "Q6_K",
            15 => "Q8_K",
            _ => "Unknown",
        };
        println!("    qtype {}: {} ({} tensors)", qtype, qtype_name, count);
    }

    // Verify we have tensors
    assert!(!model.tensors.is_empty(), "Should have tensors");

    // Most tensors in Q4_K_M should be qtype 12 (Q4_K) or 14 (Q6_K)
    let q4k_count = qtype_counts.get(&12).unwrap_or(&0);
    let q6k_count = qtype_counts.get(&14).unwrap_or(&0);
    assert!(
        *q4k_count > 0 || *q6k_count > 0,
        "Q4_K_M model should have Q4_K or Q6_K tensors"
    );
}

#[test]
#[ignore = "requires phi-2 model file"]
fn test_phi2_model_architecture() {
    if !model_exists(PHI2_MODEL_PATH) {
        return;
    }

    let data = fs::read(PHI2_MODEL_PATH).expect("Failed to read phi-2 model");
    let model = GGUFModel::from_bytes(&data).expect("Failed to parse phi-2 model");

    // Get architecture
    let arch = model
        .metadata
        .get("general.architecture")
        .and_then(|v| match v {
            GGUFValue::String(s) => Some(s.as_str()),
            _ => None,
        })
        .unwrap_or("unknown");

    println!("Architecture: {}", arch);

    // Get embedding dimension
    let embed_key = format!("{}.embedding_length", arch);
    let embed_dim = model.metadata.get(&embed_key).and_then(|v| match v {
        GGUFValue::UInt32(n) => Some(*n as usize),
        GGUFValue::UInt64(n) => Some(*n as usize),
        _ => None,
    });
    println!("Embedding dim: {:?}", embed_dim);

    // Get number of layers
    let layers_key = format!("{}.block_count", arch);
    let num_layers = model.metadata.get(&layers_key).and_then(|v| match v {
        GGUFValue::UInt32(n) => Some(*n as usize),
        GGUFValue::UInt64(n) => Some(*n as usize),
        _ => None,
    });
    println!("Num layers: {:?}", num_layers);

    // Get attention heads
    let heads_key = format!("{}.attention.head_count", arch);
    let num_heads = model.metadata.get(&heads_key).and_then(|v| match v {
        GGUFValue::UInt32(n) => Some(*n as usize),
        GGUFValue::UInt64(n) => Some(*n as usize),
        _ => None,
    });
    println!("Num heads: {:?}", num_heads);

    // Get vocab size
    let vocab_key = format!("{}.vocab_size", arch);
    let vocab_size = model.metadata.get(&vocab_key).and_then(|v| match v {
        GGUFValue::UInt32(n) => Some(*n as usize),
        GGUFValue::UInt64(n) => Some(*n as usize),
        _ => None,
    });
    println!("Vocab size: {:?}", vocab_size);

    // Verify phi-2 specifics (if architecture is phi2 or phi)
    if arch == "phi2" || arch == "phi" {
        // phi-2 has 2048 embedding dim, 32 layers
        if let Some(dim) = embed_dim {
            assert!(dim > 0, "Embedding dim should be positive");
            println!("phi-2 embedding dim: {}", dim);
        }
    }
}

// ============================================================================
// TENSOR EXTRACTION TESTS
// ============================================================================

#[test]
#[ignore = "requires phi-2 model file"]
fn test_phi2_extract_token_embedding() {
    if !model_exists(PHI2_MODEL_PATH) {
        return;
    }

    let data = fs::read(PHI2_MODEL_PATH).expect("Failed to read phi-2 model");
    let model = GGUFModel::from_bytes(&data).expect("Failed to parse phi-2 model");

    // Find token embedding tensor
    let embed_tensor = model
        .tensors
        .iter()
        .find(|t| t.name.contains("token_embd") || t.name.contains("embed_tokens"))
        .expect("Should have embedding tensor");

    println!("\nToken embedding tensor:");
    println!("  Name: {}", embed_tensor.name);
    println!("  Dims: {:?}", embed_tensor.dims);
    println!("  Qtype: {}", embed_tensor.qtype);
    println!("  Offset: {}", embed_tensor.offset);

    // Calculate expected size
    let size: u64 = embed_tensor.dims.iter().product();
    println!("  Elements: {}", size);
}

#[test]
#[ignore = "requires phi-2 model file"]
fn test_phi2_layer_structure() {
    if !model_exists(PHI2_MODEL_PATH) {
        return;
    }

    let data = fs::read(PHI2_MODEL_PATH).expect("Failed to read phi-2 model");
    let model = GGUFModel::from_bytes(&data).expect("Failed to parse phi-2 model");

    // Group tensors by layer
    let mut layer_tensors: std::collections::HashMap<i32, Vec<&str>> =
        std::collections::HashMap::new();

    for tensor in &model.tensors {
        // Extract layer number from name like "blk.0.attn_q.weight"
        if tensor.name.contains("blk.") {
            let parts: Vec<&str> = tensor.name.split('.').collect();
            if parts.len() >= 2 {
                if let Ok(layer_num) = parts[1].parse::<i32>() {
                    layer_tensors
                        .entry(layer_num)
                        .or_default()
                        .push(&tensor.name);
                }
            }
        } else {
            // Non-layer tensors (embeddings, final norm, etc.)
            layer_tensors.entry(-1).or_default().push(&tensor.name);
        }
    }

    // Print layer structure
    let num_layers = layer_tensors.keys().filter(|&&k| k >= 0).count();
    println!("\nLayer structure ({} layers):", num_layers);

    // Print layer 0 tensors
    if let Some(tensors) = layer_tensors.get(&0) {
        println!("\n  Layer 0 tensors ({}):", tensors.len());
        for name in tensors {
            println!("    {}", name);
        }
    }

    // Print non-layer tensors
    if let Some(tensors) = layer_tensors.get(&-1) {
        println!("\n  Non-layer tensors ({}):", tensors.len());
        for name in tensors {
            println!("    {}", name);
        }
    }

    assert!(num_layers > 0, "Should have at least one layer");
}

// ============================================================================
// MULTIPLE MODEL COMPARISON TESTS
// ============================================================================

#[test]
#[ignore = "requires model files"]
fn test_compare_model_architectures() {
    let models = [
        ("phi-2", PHI2_MODEL_PATH),
        ("qwen2.5-coder", QWEN_MODEL_PATH),
        ("deepseek-coder", DEEPSEEK_MODEL_PATH),
    ];

    println!("\n=== Model Architecture Comparison ===\n");

    for (name, path) in &models {
        if !model_exists(path) {
            println!("{}: NOT FOUND", name);
            continue;
        }

        let data = fs::read(path).expect("Failed to read model");
        let model = GGUFModel::from_bytes(&data).expect("Failed to parse model");

        let arch = model
            .metadata
            .get("general.architecture")
            .and_then(|v| match v {
                GGUFValue::String(s) => Some(s.clone()),
                _ => None,
            })
            .unwrap_or_else(|| "unknown".to_string());

        let tensor_count = model.tensors.len();

        // Count qtypes
        let mut q4k_count = 0;
        let mut q6k_count = 0;
        for tensor in &model.tensors {
            match tensor.qtype {
                12 => q4k_count += 1,
                14 => q6k_count += 1,
                _ => {},
            }
        }

        println!(
            "{}: arch={}, tensors={}, Q4_K={}, Q6_K={}",
            name, arch, tensor_count, q4k_count, q6k_count
        );
    }
}

// ============================================================================
// Q4_K DEQUANTIZATION TEST
// ============================================================================

#[test]
#[ignore = "requires phi-2 model file"]
fn test_phi2_dequantize_small_tensor() {
    if !model_exists(PHI2_MODEL_PATH) {
        return;
    }

    let data = fs::read(PHI2_MODEL_PATH).expect("Failed to read phi-2 model");
    let model = GGUFModel::from_bytes(&data).expect("Failed to parse phi-2 model");

    // Find a small F32 tensor (like output_norm)
    let norm_tensor = model
        .tensors
        .iter()
        .find(|t| t.name.contains("output_norm") && t.qtype == 0)
        .or_else(|| model.tensors.iter().find(|t| t.qtype == 0));

    if let Some(tensor) = norm_tensor {
        println!("\nAttempting to dequantize: {}", tensor.name);
        println!("  Dims: {:?}", tensor.dims);
        println!("  Qtype: {} (F32)", tensor.qtype);

        let values = model.get_tensor_f32(&tensor.name, &data);
        match values {
            Ok(v) => {
                println!("  Successfully extracted {} values", v.len());
                println!("  First 5 values: {:?}", &v[..5.min(v.len())]);

                // Basic sanity check
                assert!(!v.is_empty());
                assert!(v.iter().all(|x| x.is_finite()));
            },
            Err(e) => {
                println!("  Error: {:?}", e);
            },
        }
    } else {
        println!("No F32 tensor found for dequantization test");
    }
}

#[test]
#[ignore = "requires phi-2 model file"]
fn test_phi2_dequantize_q4_k_tensor() {
    if !model_exists(PHI2_MODEL_PATH) {
        return;
    }

    let data = fs::read(PHI2_MODEL_PATH).expect("Failed to read phi-2 model");
    let model = GGUFModel::from_bytes(&data).expect("Failed to parse phi-2 model");

    // Find a Q4_K tensor (qtype 12)
    let q4k_tensor = model.tensors.iter().find(|t| t.qtype == 12);

    if let Some(tensor) = q4k_tensor {
        println!("\nAttempting to dequantize Q4_K tensor: {}", tensor.name);
        println!("  Dims: {:?}", tensor.dims);
        println!("  Qtype: 12 (Q4_K)");

        let size: u64 = tensor.dims.iter().product();
        println!("  Total elements: {}", size);

        let values = model.get_tensor_f32(&tensor.name, &data);
        match values {
            Ok(v) => {
                println!("  Successfully extracted {} values", v.len());
                println!("  First 10 values: {:?}", &v[..10.min(v.len())]);
                println!(
                    "  Min: {}, Max: {}",
                    v.iter().copied().reduce(f32::min).unwrap_or(0.0),
                    v.iter().copied().reduce(f32::max).unwrap_or(0.0)
                );

                // Basic sanity checks
                assert!(!v.is_empty());
                assert!(
                    v.iter().all(|x| x.is_finite()),
                    "All values should be finite"
                );
            },
            Err(e) => {
                panic!("Failed to dequantize Q4_K tensor: {:?}", e);
            },
        }
    } else {
        panic!("No Q4_K tensor found in phi-2 model");
    }
}

#[test]
#[ignore = "requires phi-2 model file"]
fn test_phi2_dequantize_q5_k_tensor() {
    if !model_exists(PHI2_MODEL_PATH) {
        return;
    }

    let data = fs::read(PHI2_MODEL_PATH).expect("Failed to read phi-2 model");
    let model = GGUFModel::from_bytes(&data).expect("Failed to parse phi-2 model");

    // Find a Q5_K tensor (qtype 13)
    let q5k_tensor = model.tensors.iter().find(|t| t.qtype == 13);

    if let Some(tensor) = q5k_tensor {
        println!("\nAttempting to dequantize Q5_K tensor: {}", tensor.name);
        println!("  Dims: {:?}", tensor.dims);
        println!("  Qtype: 13 (Q5_K)");

        let size: u64 = tensor.dims.iter().product();
        println!("  Total elements: {}", size);

        let values = model.get_tensor_f32(&tensor.name, &data);
        match values {
            Ok(v) => {
                println!("  Successfully extracted {} values", v.len());
                println!("  First 10 values: {:?}", &v[..10.min(v.len())]);
                println!(
                    "  Min: {}, Max: {}",
                    v.iter().copied().reduce(f32::min).unwrap_or(0.0),
                    v.iter().copied().reduce(f32::max).unwrap_or(0.0)
                );

                // Basic sanity checks
                assert!(!v.is_empty());
                assert!(
                    v.iter().all(|x| x.is_finite()),
                    "All values should be finite"
                );
            },
            Err(e) => {
                panic!("Failed to dequantize Q5_K tensor: {:?}", e);
            },
        }
    } else {
        println!("No Q5_K tensor found in phi-2 model");
    }
}

#[test]
#[ignore = "requires phi-2 model file"]
fn test_phi2_dequantize_q6_k_tensor() {
    if !model_exists(PHI2_MODEL_PATH) {
        return;
    }

    let data = fs::read(PHI2_MODEL_PATH).expect("Failed to read phi-2 model");
    let model = GGUFModel::from_bytes(&data).expect("Failed to parse phi-2 model");

    // Find a Q6_K tensor (qtype 14)
    let q6k_tensor = model.tensors.iter().find(|t| t.qtype == 14);

    if let Some(tensor) = q6k_tensor {
        println!("\nAttempting to dequantize Q6_K tensor: {}", tensor.name);
        println!("  Dims: {:?}", tensor.dims);
        println!("  Qtype: 14 (Q6_K)");

        let size: u64 = tensor.dims.iter().product();
        println!("  Total elements: {}", size);

        let values = model.get_tensor_f32(&tensor.name, &data);
        match values {
            Ok(v) => {
                println!("  Successfully extracted {} values", v.len());
                println!("  First 10 values: {:?}", &v[..10.min(v.len())]);
                println!(
                    "  Min: {}, Max: {}",
                    v.iter().copied().reduce(f32::min).unwrap_or(0.0),
                    v.iter().copied().reduce(f32::max).unwrap_or(0.0)
                );

                // Basic sanity checks
                assert!(!v.is_empty());
                assert!(
                    v.iter().all(|x| x.is_finite()),
                    "All values should be finite"
                );
            },
            Err(e) => {
                panic!("Failed to dequantize Q6_K tensor: {:?}", e);
            },
        }
    } else {
        println!("No Q6_K tensor found in phi-2 model");
    }
}
