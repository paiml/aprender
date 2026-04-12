//! Model Integrity Checker (G0 Gateway)
//!
//! Pre-flight check that validates config.json matches tensor metadata in SafeTensors models.
//! This catches corrupted configs that would pass G1 (model loads) but cause silent inference failures.
//!
//! ## Background
//!
//! A corrupted config.json was found with:
//! - `num_hidden_layers: 14` (should be 24)
//! - `hidden_size: 4096` (should be 896)
//! - `vocab_size: 896` (should be 151_936)
//!
//! This passed G1 (model loads) but would cause silent inference failures.
//!
//! ## Checks
//!
//! - G0-INTEGRITY-CONFIG: config.json exists
//! - G0-INTEGRITY-LAYERS: layer count matches tensors
//! - G0-INTEGRITY-HIDDEN: hidden_size matches embedding shape
//! - G0-INTEGRITY-VOCAB: vocab_size matches embedding shape

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

/// Result of model integrity check
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(clippy::struct_excessive_bools)]
pub struct IntegrityResult {
    /// Whether all integrity checks passed
    pub passed: bool,
    /// Whether config.json was found
    pub config_found: bool,
    /// Whether layer count matches
    pub layer_count_match: bool,
    /// Whether hidden_size matches
    pub hidden_size_match: bool,
    /// Whether vocab_size matches
    pub vocab_size_match: bool,
    /// Detailed error messages
    pub errors: Vec<String>,
    /// Config values found (for diagnostics)
    pub config_values: Option<ConfigValues>,
    /// Tensor-derived values (for diagnostics)
    pub tensor_values: Option<TensorDerivedValues>,
}

/// Values parsed from config.json
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigValues {
    /// Number of hidden layers from config
    pub num_hidden_layers: Option<usize>,
    /// Hidden size from config
    pub hidden_size: Option<usize>,
    /// Vocabulary size from config
    pub vocab_size: Option<usize>,
    /// Number of attention heads from config
    pub num_attention_heads: Option<usize>,
}

/// Values derived from tensor metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorDerivedValues {
    /// Layer count from tensor names (max layer index + 1)
    pub layer_count: Option<usize>,
    /// Hidden size from embedding tensor shape[1]
    pub hidden_size: Option<usize>,
    /// Vocab size from embedding tensor shape[0]
    pub vocab_size: Option<usize>,
}

/// HuggingFace config.json structure (partial)
#[derive(Debug, Deserialize)]
struct HfConfig {
    num_hidden_layers: Option<usize>,
    hidden_size: Option<usize>,
    vocab_size: Option<usize>,
    num_attention_heads: Option<usize>,
}

/// Check integrity of a SafeTensors model directory
///
/// Validates that config.json metadata matches actual tensor shapes.
///
/// # Arguments
///
/// * `model_dir` - Path to the model directory containing config.json and .safetensors files
///
/// # Returns
///
/// `IntegrityResult` with pass/fail status and detailed error messages
#[must_use]
pub fn check_safetensors_integrity(model_dir: &Path) -> IntegrityResult {
    let mut result = IntegrityResult {
        passed: true,
        config_found: false,
        layer_count_match: true,
        hidden_size_match: true,
        vocab_size_match: true,
        errors: Vec::new(),
        config_values: None,
        tensor_values: None,
    };

    // Step 1: Check for config.json (supports pacha cache naming: <hash>.config.json)
    let config_path = model_dir.join("config.json");
    let config_path = if config_path.exists() {
        config_path
    } else {
        // Fallback: pacha cache uses <hash>.config.json naming
        find_config_json(model_dir).unwrap_or(config_path)
    };
    let config = match read_config(&config_path) {
        Ok(cfg) => {
            result.config_found = true;
            result.config_values = Some(ConfigValues {
                num_hidden_layers: cfg.num_hidden_layers,
                hidden_size: cfg.hidden_size,
                vocab_size: cfg.vocab_size,
                num_attention_heads: cfg.num_attention_heads,
            });
            cfg
        }
        Err(e) => {
            result.config_found = false;
            result.passed = false;
            result.errors.push(format!("G0-INTEGRITY-CONFIG: {e}"));
            return result;
        }
    };

    // Step 2: Find and parse SafeTensors files
    let safetensors_files = find_safetensors_files(model_dir);
    if safetensors_files.is_empty() {
        result.passed = false;
        result
            .errors
            .push("G0-INTEGRITY-CONFIG: No .safetensors files found".to_string());
        return result;
    }

    // Step 3: Extract tensor metadata from all files
    let mut all_tensors: HashMap<String, Vec<usize>> = HashMap::new();
    for st_path in &safetensors_files {
        match read_safetensors_metadata(st_path) {
            Ok(tensors) => {
                all_tensors.extend(tensors);
            }
            Err(e) => {
                result.passed = false;
                result.errors.push(format!(
                    "G0-INTEGRITY-CONFIG: Failed to read {}: {e}",
                    st_path.display()
                ));
                return result;
            }
        }
    }

    // Step 4: Derive values from tensors
    let tensor_values = derive_values_from_tensors(&all_tensors);
    result.tensor_values = Some(tensor_values.clone());

    // Step 5: Validate config vs tensor values
    validate_config_vs_tensors(&config, &tensor_values, &mut result);

    result
}

/// Validate config.json values against tensor-derived values
///
/// Checks layer count, hidden_size, and vocab_size, recording mismatches in the result.
fn validate_config_vs_tensors(
    config: &HfConfig,
    tensor_values: &TensorDerivedValues,
    result: &mut IntegrityResult,
) {
    if let (Some(config_layers), Some(tensor_layers)) =
        (config.num_hidden_layers, tensor_values.layer_count)
    {
        if config_layers != tensor_layers {
            result.layer_count_match = false;
            result.passed = false;
            result.errors.push(format!(
                "G0-INTEGRITY-LAYERS: config says {config_layers} layers but tensors have {tensor_layers}"
            ));
        }
    }

    if let (Some(config_hidden), Some(tensor_hidden)) =
        (config.hidden_size, tensor_values.hidden_size)
    {
        if config_hidden != tensor_hidden {
            result.hidden_size_match = false;
            result.passed = false;
            result.errors.push(format!(
                "G0-INTEGRITY-HIDDEN: config says hidden_size={config_hidden} but embedding has {tensor_hidden}"
            ));
        }
    }

    if let (Some(config_vocab), Some(tensor_vocab)) = (config.vocab_size, tensor_values.vocab_size)
    {
        if config_vocab != tensor_vocab {
            result.vocab_size_match = false;
            result.passed = false;
            result.errors.push(format!(
                "G0-INTEGRITY-VOCAB: config says vocab_size={config_vocab} but embedding has {tensor_vocab}"
            ));
        }
    }
}

/// Check integrity of a single SafeTensors model file
///
/// For use when model_path points to a specific file (e.g., from `apr pull`).
/// Finds the associated config.json using the file's hash prefix (pacha cache
/// naming: `<hash>.safetensors` + `<hash>.config.json`), and validates only
/// against this file's tensor metadata — not all files in the parent directory.
///
/// # Arguments
///
/// * `model_file` - Path to a specific `.safetensors` file
///
/// # Returns
///
/// `IntegrityResult` with pass/fail status and detailed error messages
#[must_use]
pub fn check_safetensors_file_integrity(model_file: &Path) -> IntegrityResult {
    let mut result = IntegrityResult {
        passed: true,
        config_found: false,
        layer_count_match: true,
        hidden_size_match: true,
        vocab_size_match: true,
        errors: Vec::new(),
        config_values: None,
        tensor_values: None,
    };

    // Step 1: Find the associated config.json via hash prefix
    let config_path = find_config_for_model_file(model_file);
    let Some(config_path) = config_path else {
        result.config_found = false;
        result.passed = false;
        result.errors.push(format!(
            "G0-INTEGRITY-CONFIG: No config.json found for {}",
            model_file.display()
        ));
        return result;
    };

    let config = match read_config(&config_path) {
        Ok(cfg) => {
            result.config_found = true;
            result.config_values = Some(ConfigValues {
                num_hidden_layers: cfg.num_hidden_layers,
                hidden_size: cfg.hidden_size,
                vocab_size: cfg.vocab_size,
                num_attention_heads: cfg.num_attention_heads,
            });
            cfg
        }
        Err(e) => {
            result.config_found = false;
            result.passed = false;
            result.errors.push(format!("G0-INTEGRITY-CONFIG: {e}"));
            return result;
        }
    };

    // Step 2: Read tensor metadata from THIS file only
    let all_tensors = match read_safetensors_metadata(model_file) {
        Ok(tensors) => tensors,
        Err(e) => {
            result.passed = false;
            result.errors.push(format!(
                "G0-INTEGRITY-CONFIG: Failed to read {}: {e}",
                model_file.display()
            ));
            return result;
        }
    };

    // Step 3: Derive and validate (same logic as directory version)
    let tensor_values = derive_values_from_tensors(&all_tensors);
    result.tensor_values = Some(tensor_values.clone());

    validate_config_vs_tensors(&config, &tensor_values, &mut result);

    result
}

/// Find the config.json associated with a specific model file
///
/// Pacha cache naming: `<hash>.safetensors` + `<hash>.config.json`
/// Falls back to `config.json` in the same directory.
fn find_config_for_model_file(model_file: &Path) -> Option<std::path::PathBuf> {
    let parent = model_file.parent()?;
    let stem = model_file.file_name()?.to_str()?;

    // Try hash-prefix match: foo.safetensors → foo.config.json
    if let Some(hash_prefix) = stem.strip_suffix(".safetensors") {
        let config_name = format!("{hash_prefix}.config.json");
        let config_path = parent.join(&config_name);
        if config_path.exists() {
            return Some(config_path);
        }
    }

    // Fallback: config.json in same directory
    let config_path = parent.join("config.json");
    if config_path.exists() {
        return Some(config_path);
    }

    None
}

/// Read and parse config.json
fn read_config(path: &Path) -> Result<HfConfig, String> {
    let file = File::open(path).map_err(|e| format!("config.json not found or unreadable: {e}"))?;
    let reader = BufReader::new(file);
    serde_json::from_reader(reader).map_err(|e| format!("config.json parse error: {e}"))
}

/// Find a `*.config.json` file in a directory (pacha cache naming convention)
///
/// Pacha cache uses `<hash>.config.json` instead of plain `config.json`.
fn find_config_json(dir: &Path) -> Option<std::path::PathBuf> {
    let entries = std::fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
            if name.ends_with(".config.json") {
                return Some(path);
            }
        }
    }
    None
}

/// Find all .safetensors files in a directory, sorted for consistent ordering
fn find_safetensors_files(dir: &Path) -> Vec<std::path::PathBuf> {
    let mut files = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "safetensors") {
                files.push(path);
            }
        }
    }
    files.sort(); // Ensure consistent ordering
    files
}

/// Maximum header size for SafeTensors files (100MB)
const MAX_HEADER_SIZE: usize = 100 * 1024 * 1024;

/// Read SafeTensors metadata header and extract tensor name -> shape mapping
fn read_safetensors_metadata(path: &Path) -> Result<HashMap<String, Vec<usize>>, String> {
    let mut file = File::open(path).map_err(|e| format!("Failed to open safetensors file: {e}"))?;

    // SafeTensors format: first 8 bytes are header length (little endian u64)
    let mut header_len_bytes = [0u8; 8];
    file.read_exact(&mut header_len_bytes)
        .map_err(|e| format!("Failed to read header length: {e}"))?;
    let header_len = u64::from_le_bytes(header_len_bytes) as usize;

    // Safety check: header shouldn't be unreasonably large
    if header_len > MAX_HEADER_SIZE {
        return Err(format!(
            "Header size {header_len} exceeds maximum {MAX_HEADER_SIZE}"
        ));
    }

    // Read the JSON header
    let mut header_bytes = vec![0u8; header_len];
    file.read_exact(&mut header_bytes)
        .map_err(|e| format!("Failed to read header: {e}"))?;

    let header_str = std::str::from_utf8(&header_bytes)
        .map_err(|e| format!("Header is not valid UTF-8: {e}"))?;

    // Parse as JSON object
    let header: serde_json::Value =
        serde_json::from_str(header_str).map_err(|e| format!("Header JSON parse error: {e}"))?;

    let obj = header.as_object().ok_or("Header is not a JSON object")?;

    let tensors = obj
        .iter()
        .filter(|(name, _)| *name != "__metadata__")
        .filter_map(|(name, value)| {
            let shape = value.as_object()?.get("shape")?.as_array()?;
            let dims: Vec<usize> = shape
                .iter()
                .filter_map(|v| v.as_u64().map(|n| n as usize))
                .collect();
            Some((name.clone(), dims))
        })
        .collect();

    Ok(tensors)
}

/// Derive model configuration values from tensor metadata
fn derive_values_from_tensors(tensors: &HashMap<String, Vec<usize>>) -> TensorDerivedValues {
    let layer_count = derive_layer_count(tensors);
    let (vocab_size, hidden_size) = find_embedding_shape(tensors);

    TensorDerivedValues {
        layer_count,
        hidden_size,
        vocab_size,
    }
}

/// Count layers by finding the max layer index in tensor names (0-based → +1)
fn derive_layer_count(tensors: &HashMap<String, Vec<usize>>) -> Option<usize> {
    tensors
        .keys()
        .filter_map(|name| extract_layer_number(name))
        .max()
        .map(|n| n + 1)
}

/// Find (vocab_size, hidden_size) from embedding or lm_head tensors
fn find_embedding_shape(tensors: &HashMap<String, Vec<usize>>) -> (Option<usize>, Option<usize>) {
    let candidates = [
        "model.embed_tokens.weight",
        "embed_tokens.weight",
        "transformer.wte.weight",
        "wte.weight",
        "lm_head.weight",
        "model.lm_head.weight",
    ];

    for name in candidates {
        if let Some(shape) = tensors.get(name) {
            if shape.len() >= 2 {
                return (Some(shape[0]), Some(shape[1]));
            }
        }
    }

    (None, None)
}

include!("integrity_helpers.rs");
