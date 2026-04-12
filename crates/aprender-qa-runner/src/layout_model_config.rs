
/// Validate all 2D tensors (F-LAYOUT-CONTRACT-001)
fn validate_2d_tensors(
    contract: &TensorLayoutContract,
    all_tensors: &HashMap<String, Vec<usize>>,
    config: &LayoutModelConfig,
    results: &mut Vec<TensorValidationResult>,
) {
    for (name, spec) in &contract.tensors {
        if !spec.transpose {
            continue;
        }

        if spec.apr_name.contains("{n}") {
            validate_layer_tensors(&spec.apr_name, all_tensors, config, spec, results);
        } else if let Some(actual_shape) = all_tensors.get(&spec.apr_name) {
            results.push(validate_2d_tensor_shape(name, actual_shape, spec, config));
        }
    }
}

/// Validate all 1D tensors (F-LAYOUT-CONTRACT-003)
fn validate_1d_tensors(
    contract: &TensorLayoutContract,
    all_tensors: &HashMap<String, Vec<usize>>,
    config: &LayoutModelConfig,
    results: &mut Vec<TensorValidationResult>,
) {
    for (name, spec) in &contract.tensors {
        if spec.transpose {
            continue;
        }

        if spec.apr_name.contains("{n}") {
            validate_1d_layer_tensors(&spec.apr_name, all_tensors, config, spec, results);
        } else if let Some(actual_shape) = all_tensors.get(&spec.apr_name) {
            results.push(validate_1d_tensor_shape(name, actual_shape, spec, config));
        }
    }
}

/// Model configuration values for validation
#[derive(Debug, Default)]
pub struct LayoutModelConfig {
    /// Vocabulary size from config.json
    pub vocab_size: Option<usize>,
    /// Hidden size from config.json
    pub hidden_size: Option<usize>,
    /// Intermediate/FFN size from config.json
    pub intermediate_size: Option<usize>,
    /// Number of attention heads from config.json
    pub num_attention_heads: Option<usize>,
    /// Number of key-value heads from config.json
    pub num_key_value_heads: Option<usize>,
    /// Number of hidden layers from config.json
    pub num_hidden_layers: Option<usize>,
    /// Head dimension from config.json (explicit, not derived)
    pub head_dim: Option<usize>,
}

/// Find SafeTensors files in a path
#[must_use]
pub fn find_safetensors_files(path: &Path) -> Vec<PathBuf> {
    if path.is_file() {
        if path.extension().is_some_and(|e| e == "safetensors") {
            return vec![path.to_path_buf()];
        }
        return Vec::new();
    }

    // Try safetensors subdirectory first
    let st_dir = path.join("safetensors");
    let search_dir = if st_dir.exists() { &st_dir } else { path };

    let Ok(entries) = search_dir.read_dir() else {
        return Vec::new();
    };

    entries
        .flatten()
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "safetensors"))
        .map(|e| e.path())
        .collect()
}

/// Read SafeTensors header to extract tensor shapes
///
/// # Errors
///
/// Returns an error string if the file cannot be opened, the header
/// is malformed, or the JSON cannot be parsed.
pub fn read_safetensors_metadata(
    path: &Path,
) -> std::result::Result<HashMap<String, Vec<usize>>, String> {
    let mut file = File::open(path).map_err(|e| format!("Failed to open: {e}"))?;

    // SafeTensors format: first 8 bytes are header length (little endian u64)
    let mut header_len_bytes = [0u8; 8];
    file.read_exact(&mut header_len_bytes)
        .map_err(|e| format!("Failed to read header length: {e}"))?;
    let header_len = u64::from_le_bytes(header_len_bytes) as usize;

    if header_len > MAX_HEADER_SIZE {
        return Err(format!("Header too large: {header_len}"));
    }

    let mut header_bytes = vec![0u8; header_len];
    file.read_exact(&mut header_bytes)
        .map_err(|e| format!("Failed to read header: {e}"))?;

    let header_str =
        std::str::from_utf8(&header_bytes).map_err(|e| format!("Invalid UTF-8: {e}"))?;

    let header: serde_json::Value =
        serde_json::from_str(header_str).map_err(|e| format!("JSON parse error: {e}"))?;

    let obj = header.as_object().ok_or("Header is not JSON object")?;

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

/// Tensor metadata including dtype (for G0-DTYPE checks)
#[derive(Debug, Clone)]
pub struct TensorDtypeInfo {
    /// Tensor shape dimensions
    pub shape: Vec<usize>,
    /// Dtype string from SafeTensors header (e.g., "F32", "BF16", "F16")
    pub dtype: String,
}

/// Read SafeTensors header to extract tensor shapes AND dtypes.
///
/// Like `read_safetensors_metadata` but also captures the dtype field
/// from each tensor entry, needed for G0-DTYPE validation.
///
/// # Errors
///
/// Returns an error string if the file cannot be opened, the header
/// is malformed, or the JSON cannot be parsed.
pub fn read_safetensors_metadata_with_dtypes(
    path: &Path,
) -> std::result::Result<HashMap<String, TensorDtypeInfo>, String> {
    let mut file = File::open(path).map_err(|e| format!("Failed to open: {e}"))?;

    let mut header_len_bytes = [0u8; 8];
    file.read_exact(&mut header_len_bytes)
        .map_err(|e| format!("Failed to read header length: {e}"))?;
    let header_len = u64::from_le_bytes(header_len_bytes) as usize;

    if header_len > MAX_HEADER_SIZE {
        return Err(format!("Header too large: {header_len}"));
    }

    let mut header_bytes = vec![0u8; header_len];
    file.read_exact(&mut header_bytes)
        .map_err(|e| format!("Failed to read header: {e}"))?;

    let header_str =
        std::str::from_utf8(&header_bytes).map_err(|e| format!("Invalid UTF-8: {e}"))?;

    let header: serde_json::Value =
        serde_json::from_str(header_str).map_err(|e| format!("JSON parse error: {e}"))?;

    let obj = header.as_object().ok_or("Header is not JSON object")?;

    let tensors = obj
        .iter()
        .filter(|(name, _)| *name != "__metadata__")
        .filter_map(|(name, value)| {
            let tensor_obj = value.as_object()?;
            let shape = tensor_obj.get("shape")?.as_array()?;
            let dims: Vec<usize> = shape
                .iter()
                .filter_map(|v| v.as_u64().map(|n| n as usize))
                .collect();
            let dtype = tensor_obj
                .get("dtype")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("UNKNOWN")
                .to_string();
            Some((name.clone(), TensorDtypeInfo { shape: dims, dtype }))
        })
        .collect();

    Ok(tensors)
}

/// Helper to extract usize from JSON
fn get_usize(json: &serde_json::Value, key: &str) -> Option<usize> {
    json.get(key)
        .and_then(serde_json::Value::as_u64)
        .map(|n| n as usize)
}

/// Helper to extract usize from JSON with fallback keys (e.g., GPT-2 uses `n_embd` for `hidden_size`)
fn get_usize_or(json: &serde_json::Value, keys: &[&str]) -> Option<usize> {
    keys.iter().find_map(|k| get_usize(json, k))
}

/// Find and load config.json
#[must_use]
pub fn find_and_load_config(model_path: &Path) -> LayoutModelConfig {
    let config_paths = if model_path.is_file() {
        // For file mode, check parent dir and look for hash-prefixed config
        let parent = model_path.parent().unwrap_or(model_path);
        let stem = model_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("");
        vec![
            parent.join(format!("{stem}.config.json")),
            parent.join("config.json"),
        ]
    } else {
        vec![
            model_path.join("config.json"),
            model_path.join("safetensors/config.json"),
        ]
    };

    for path in config_paths {
        if let Ok(content) = std::fs::read_to_string(&path) {
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) {
                return LayoutModelConfig {
                    vocab_size: get_usize(&json, "vocab_size"),
                    hidden_size: get_usize_or(
                        &json,
                        &["hidden_size", "n_embd", "n_embed", "d_model", "model_dim"],
                    ),
                    intermediate_size: get_usize_or(
                        &json,
                        &["intermediate_size", "n_inner", "ffn_dim"],
                    ),
                    num_attention_heads: get_usize_or(
                        &json,
                        &[
                            "num_attention_heads",
                            "n_head",
                            "num_heads",
                            "attention_heads",
                            "num_query_heads",
                        ],
                    ),
                    num_key_value_heads: get_usize_or(
                        &json,
                        &["num_key_value_heads", "num_kv_heads"],
                    ),
                    num_hidden_layers: get_usize_or(
                        &json,
                        &[
                            "num_hidden_layers",
                            "n_layer",
                            "num_layers",
                            "num_transformer_layers",
                        ],
                    ),
                    head_dim: get_usize_or(
                        &json,
                        &["head_dim", "kv_channels"],
                    ),
                };
            }
        }
    }

    LayoutModelConfig::default()
}

/// Validate lm_head shape (F-LAYOUT-CONTRACT-002) - CRITICAL
fn validate_lm_head_shape(
    actual_shape: &[usize],
    config: &LayoutModelConfig,
    _contract: &TensorLayoutContract,
) -> TensorValidationResult {
    // lm_head.weight should be [vocab_size, hidden_size] in row-major
    if actual_shape.len() != 2 {
        return TensorValidationResult {
            tensor_name: "lm_head.weight".to_string(),
            rule_id: "F-LAYOUT-CONTRACT-002".to_string(),
            passed: false,
            details: "lm_head.weight must be 2D tensor".to_string(),
            expected: Some("[vocab_size, hidden_size]".to_string()),
            actual: Some(format!("{actual_shape:?}")),
        };
    }

    let (expected_vocab, expected_hidden) = (config.vocab_size, config.hidden_size);

    // Check if shape matches [vocab, hidden]
    // Popper: (None, None) → false. Untested dimensions ≠ validated dimensions.
    let shape_valid = match (expected_vocab, expected_hidden) {
        (Some(vocab), Some(hidden)) => actual_shape[0] == vocab && actual_shape[1] == hidden,
        (Some(vocab), None) => actual_shape[0] == vocab,
        (None, Some(hidden)) => actual_shape[1] == hidden,
        (None, None) => false,
    };

    if shape_valid {
        TensorValidationResult {
            tensor_name: "lm_head.weight".to_string(),
            rule_id: "F-LAYOUT-CONTRACT-002".to_string(),
            passed: true,
            details: format!("lm_head.weight shape correct: {:?}", actual_shape),
            expected: Some(format!("[{:?}, {:?}]", expected_vocab, expected_hidden)),
            actual: Some(format!("{actual_shape:?}")),
        }
    } else if expected_vocab.is_none() && expected_hidden.is_none() {
        TensorValidationResult {
            tensor_name: "lm_head.weight".to_string(),
            rule_id: "F-LAYOUT-CONTRACT-002".to_string(),
            passed: false,
            details: format!(
                "lm_head.weight UNVALIDATED: config.json missing vocab_size and hidden_size, got {:?}",
                actual_shape
            ),
            expected: Some("[vocab_size, hidden_size]".to_string()),
            actual: Some(format!("{actual_shape:?}")),
        }
    } else {
        TensorValidationResult {
            tensor_name: "lm_head.weight".to_string(),
            rule_id: "F-LAYOUT-CONTRACT-002".to_string(),
            passed: false,
            details: format!(
                "lm_head.weight shape MISMATCH (GH-202 bug pattern): expected [{:?}, {:?}], got {:?}",
                expected_vocab, expected_hidden, actual_shape
            ),
            expected: Some(format!("[{:?}, {:?}]", expected_vocab, expected_hidden)),
            actual: Some(format!("{actual_shape:?}")),
        }
    }
}

/// Validate a 2D tensor shape (F-LAYOUT-CONTRACT-001)
fn validate_2d_tensor_shape(
    name: &str,
    actual_shape: &[usize],
    spec: &TensorSpec,
    config: &LayoutModelConfig,
) -> TensorValidationResult {
    if actual_shape.len() != 2 {
        return TensorValidationResult {
            tensor_name: spec.apr_name.clone(),
            rule_id: "F-LAYOUT-CONTRACT-001".to_string(),
            passed: false,
            details: format!("{name} must be 2D, got {}D", actual_shape.len()),
            expected: Some(spec.apr_shape.clone()),
            actual: Some(format!("{actual_shape:?}")),
        };
    }

    // Parse expected shape from contract
    // Popper: None → false. Cannot earn credit for dimensions never subjected to falsification.
    let expected = parse_expected_shape(&spec.apr_shape, config);

    let (shape_valid, detail_msg) = match expected {
        Some((dim0, dim1)) => {
            let valid = actual_shape[0] == dim0 && actual_shape[1] == dim1;
            let msg = if valid {
                format!("{name} shape correct: {actual_shape:?}")
            } else {
                format!("{name} shape mismatch: expected [{dim0}, {dim1}], got {actual_shape:?}")
            };
            (valid, msg)
        }
        None => (
            false,
            format!(
                "{name} UNVALIDATED: cannot resolve expected shape '{}' from config",
                spec.apr_shape
            ),
        ),
    };

    TensorValidationResult {
        tensor_name: spec.apr_name.clone(),
        rule_id: "F-LAYOUT-CONTRACT-001".to_string(),
        passed: shape_valid,
        details: detail_msg,
        expected: Some(spec.apr_shape.clone()),
        actual: Some(format!("{actual_shape:?}")),
    }
}

/// Validate layer tensors (for patterns like model.layers.{n}.*)
fn validate_layer_tensors(
    pattern: &str,
    all_tensors: &HashMap<String, Vec<usize>>,
    config: &LayoutModelConfig,
    spec: &TensorSpec,
    results: &mut Vec<TensorValidationResult>,
) {
    let Some(num_layers) = config.num_hidden_layers else {
        // Popper: unknown layer count → cannot validate layer tensors
        results.push(TensorValidationResult {
            tensor_name: pattern.to_string(),
            rule_id: "F-LAYOUT-CONTRACT-001".to_string(),
            passed: false,
            details: format!(
                "{pattern} UNVALIDATED: num_hidden_layers missing from config.json"
            ),
            expected: Some(spec.apr_shape.clone()),
            actual: None,
        });
        return;
    };
    for layer_idx in 0..num_layers {
        let tensor_name = pattern.replace("{n}", &layer_idx.to_string());
        if let Some(actual_shape) = all_tensors.get(&tensor_name) {
            let validation = validate_2d_tensor_shape(&tensor_name, actual_shape, spec, config);
            results.push(validation);
        }
    }
}

/// Validate 1D layer tensors (F-LAYOUT-CONTRACT-003)
fn validate_1d_layer_tensors(
    pattern: &str,
    all_tensors: &HashMap<String, Vec<usize>>,
    config: &LayoutModelConfig,
    spec: &TensorSpec,
    results: &mut Vec<TensorValidationResult>,
) {
    let Some(num_layers) = config.num_hidden_layers else {
        // Popper: unknown layer count → cannot validate 1D layer tensors
        results.push(TensorValidationResult {
            tensor_name: pattern.to_string(),
            rule_id: "F-LAYOUT-CONTRACT-003".to_string(),
            passed: false,
            details: format!(
                "{pattern} UNVALIDATED: num_hidden_layers missing from config.json"
            ),
            expected: Some(spec.apr_shape.clone()),
            actual: None,
        });
        return;
    };
    for layer_idx in 0..num_layers {
        let tensor_name = pattern.replace("{n}", &layer_idx.to_string());
        if let Some(actual_shape) = all_tensors.get(&tensor_name) {
            let validation = validate_1d_tensor_shape(&tensor_name, actual_shape, spec, config);
            results.push(validation);
        }
    }
}

/// Validate a 1D tensor shape (F-LAYOUT-CONTRACT-003)
fn validate_1d_tensor_shape(
    name: &str,
    actual_shape: &[usize],
    spec: &TensorSpec,
    config: &LayoutModelConfig,
) -> TensorValidationResult {
    if actual_shape.len() != 1 {
        return TensorValidationResult {
            tensor_name: name.to_string(),
            rule_id: "F-LAYOUT-CONTRACT-003".to_string(),
            passed: false,
            details: format!("{name} must be 1D, got {}D", actual_shape.len()),
            expected: Some(spec.apr_shape.clone()),
            actual: Some(format!("{actual_shape:?}")),
        };
    }

    // 1D tensors should match hidden_size
    // Popper: None → false. Untested dimension ≠ validated dimension.
    #[allow(clippy::option_if_let_else)]
    let (shape_valid, detail_msg) = match config.hidden_size {
        Some(h) => {
            let valid = actual_shape[0] == h;
            let msg = if valid {
                format!("{name} shape correct: {actual_shape:?}")
            } else {
                format!("{name} shape mismatch: expected [{h}], got {actual_shape:?}")
            };
            (valid, msg)
        }
        None => (
            false,
            format!("{name} UNVALIDATED: hidden_size missing from config.json"),
        ),
    };

    TensorValidationResult {
        tensor_name: name.to_string(),
        rule_id: "F-LAYOUT-CONTRACT-003".to_string(),
        passed: shape_valid,
        details: detail_msg,
        expected: Some(spec.apr_shape.clone()),
        actual: Some(format!("{actual_shape:?}")),
    }
}

/// Derive head_dim from hidden_size / num_attention_heads (fallback for models without explicit head_dim)
fn derived_head_dim(config: &LayoutModelConfig) -> Option<usize> {
    match (config.hidden_size, config.num_attention_heads) {
        (Some(h), Some(n)) if n > 0 => Some(h / n),
        _ => None,
    }
}

/// Parse expected shape from contract string like "[vocab, hidden]"
fn parse_expected_shape(shape_str: &str, config: &LayoutModelConfig) -> Option<(usize, usize)> {
    let shape_parts = parse_shape_dims(shape_str);
    if shape_parts.len() != 2 {
        return None;
    }

    let first_dim = resolve_dimension(&shape_parts[0], config)?;
    let second_dim = resolve_dimension(&shape_parts[1], config)?;
    Some((first_dim, second_dim))
}

/// Resolve a dimension name to its value from config
fn resolve_dimension(dim: &str, config: &LayoutModelConfig) -> Option<usize> {
    match dim {
        "vocab" | "vocab_size" => config.vocab_size,
        "hidden" | "hidden_dim" | "hidden_size" => config.hidden_size,
        "intermediate" | "intermediate_dim" | "intermediate_size" => config.intermediate_size,
        s if s.contains('*') => {
            // Handle expressions like "heads*head_dim" or "kv_heads*head_dim"
            let parts: Vec<&str> = s.split('*').map(str::trim).collect();
            if parts.len() == 2 {
                let left = resolve_dimension(parts[0], config)?;
                let right = resolve_dimension(parts[1], config)?;
                Some(left * right)
            } else {
                None
            }
        }
        "heads" | "num_heads" | "num_attention_heads" => config.num_attention_heads,
        "kv_heads" | "num_kv_heads" | "num_key_value_heads" => config.num_key_value_heads,
        "head_dim" => config.head_dim.or_else(|| derived_head_dim(config)),
        _ => dim.parse().ok(),
    }
}

/// Get all validation rules from the contract.
#[must_use]
pub fn get_validation_rules(contract: &TensorLayoutContract) -> &[ValidationRule] {
    &contract.validation_rules
}

/// Get critical tensors from the contract (those marked with critical=true).
#[must_use]
pub fn get_critical_tensors(contract: &TensorLayoutContract) -> Vec<&TensorSpec> {
    contract.tensors.values().filter(|t| t.critical).collect()
}

/// Check if a shape string represents a 2D tensor.
#[must_use]
pub fn is_2d_shape(shape: &str) -> bool {
    // Count commas - 2D has exactly one comma
    shape.matches(',').count() == 1
}

/// Parse shape string to dimensions (e.g., `"[vocab, hidden]"` -> `["vocab", "hidden"]`).
#[must_use]
pub fn parse_shape_dims(shape: &str) -> Vec<String> {
    shape
        .trim_matches(|c| c == '[' || c == ']')
        .split(',')
        .map(|s| s.trim().to_string())
        .collect()
}

#[cfg(test)]
#[path = "layout_contract_tests.rs"]
mod tests;
