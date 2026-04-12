//! Metadata-only dimensional verification for dim-smoke tier
//!
//! Verifies model dimensional correctness by parsing `config.json` and
//! SafeTensors headers without loading model weights into memory.
//! Target: complete in under 5 seconds per model.

use crate::layout_contract::{
    find_and_load_config, find_safetensors_files, read_safetensors_metadata,
    read_safetensors_metadata_with_dtypes, LayoutModelConfig,
};
use crate::playbook::Playbook;
use std::collections::HashSet;
use std::path::Path;
use std::time::Instant;

/// Result of a single dimensional check
#[derive(Debug, Clone)]
pub struct DimensionalCheck {
    /// Check name (e.g., "config_parse", "hidden_size", "num_layers")
    pub name: String,
    /// Expected value
    pub expected: String,
    /// Actual value found
    pub actual: String,
    /// Whether the check passed
    pub passed: bool,
}

/// Aggregated result of all dimensional checks for a model
#[derive(Debug, Clone)]
pub struct DimensionalCheckResult {
    /// Model identifier
    pub model_id: String,
    /// Whether all checks passed
    pub passed: bool,
    /// Individual check results
    pub checks: Vec<DimensionalCheck>,
    /// Total duration in milliseconds
    pub duration_ms: u64,
}

/// Run metadata-only dimensional verification against a model directory.
///
/// Checks:
/// 1. `config.json` exists and parses successfully
/// 2. Architecture dimensions match playbook expectations
/// 3. SafeTensors files exist and headers parse
/// 4. Key tensors have correct shapes
#[must_use]
pub fn run_dimensional_check(model_path: &Path, playbook: &Playbook) -> DimensionalCheckResult {
    let start = Instant::now();
    let model_id = playbook.model.hf_repo.clone();
    let mut checks = Vec::new();

    let config = find_and_load_config(model_path);
    let config_parsed = config.hidden_size.is_some() || config.num_hidden_layers.is_some();
    checks.push(DimensionalCheck {
        name: "config_parse".to_string(),
        expected: "config.json parseable".to_string(),
        actual: if config_parsed {
            "parsed successfully".to_string()
        } else {
            "no config.json or empty".to_string()
        },
        passed: config_parsed,
    });

    check_architecture_dims(&playbook.model, &config, &mut checks);
    check_safetensors(model_path, &config, &mut checks);
    check_tokenizer(model_path, &mut checks);
    check_dtypes(model_path, &config, &mut checks);

    let all_passed = checks.iter().all(|c| c.passed);
    DimensionalCheckResult {
        model_id,
        passed: all_passed,
        checks,
        duration_ms: start.elapsed().as_millis() as u64,
    }
}

/// Format an optional u32 value as a string, or "missing" if None.
fn fmt_opt(v: Option<u32>) -> String {
    v.map_or_else(|| "missing".to_string(), |v| v.to_string())
}

/// Check architecture dimensions from config.json against playbook expectations.
fn check_architecture_dims(
    model: &crate::playbook::ModelConfig,
    config: &LayoutModelConfig,
    checks: &mut Vec<DimensionalCheck>,
) {
    let dim_checks: &[(&str, Option<u32>, Option<usize>)] = &[
        ("hidden_size", model.expected_hidden_dim, config.hidden_size),
        (
            "num_layers",
            model.expected_num_layers,
            config.num_hidden_layers,
        ),
        (
            "num_heads",
            model.expected_num_heads,
            config.num_attention_heads,
        ),
        (
            "num_kv_heads",
            model.expected_num_kv_heads,
            config.num_key_value_heads,
        ),
        ("vocab_size", model.expected_vocab_size, config.vocab_size),
    ];

    for &(name, expected, actual_raw) in dim_checks {
        if let Some(expected_val) = expected {
            let actual = actual_raw.map(|v| v as u32);
            checks.push(DimensionalCheck {
                name: name.to_string(),
                expected: expected_val.to_string(),
                actual: fmt_opt(actual),
                passed: actual == Some(expected_val),
            });
        }
    }
}

/// Check SafeTensors file existence and header tensor shapes.
fn check_safetensors(
    model_path: &Path,
    config: &LayoutModelConfig,
    checks: &mut Vec<DimensionalCheck>,
) {
    let st_files = find_safetensors_files(model_path);
    checks.push(DimensionalCheck {
        name: "safetensors_found".to_string(),
        expected: ">= 1 file".to_string(),
        actual: format!("{} file(s)", st_files.len()),
        passed: !st_files.is_empty(),
    });

    let Some(first_file) = st_files.first() else {
        return;
    };

    if let Ok(tensors) = read_safetensors_metadata(first_file) {
        checks.push(DimensionalCheck {
            name: "safetensors_header".to_string(),
            expected: ">= 1 tensor".to_string(),
            actual: format!("{} tensor(s)", tensors.len()),
            passed: !tensors.is_empty(),
        });

        check_key_tensor(&tensors, "model.embed_tokens.weight", config, checks);
        check_key_tensor(&tensors, "lm_head.weight", config, checks);
    } else {
        checks.push(DimensionalCheck {
            name: "safetensors_header".to_string(),
            expected: "parseable header".to_string(),
            actual: "parse error".to_string(),
            passed: false,
        });
    }
}

/// Check that a key tensor has expected shape [dim0, dim1].
fn check_key_tensor(
    tensors: &std::collections::HashMap<String, Vec<usize>>,
    name: &str,
    config: &LayoutModelConfig,
    checks: &mut Vec<DimensionalCheck>,
) {
    let short_name = name.rsplit('.').nth(1).unwrap_or(name);
    let Some(shape) = tensors.get(name) else {
        // Tensor not found is not a failure — sharded models may split tensors across files
        return;
    };

    if shape.len() != 2 {
        checks.push(DimensionalCheck {
            name: format!("tensor_{short_name}"),
            expected: "2D tensor".to_string(),
            actual: format!("{}D tensor: {shape:?}", shape.len()),
            passed: false,
        });
        return;
    }

    let mut passed = true;
    let mut expected_parts = Vec::new();

    if let Some(d0) = config.vocab_size {
        expected_parts.push(format!("dim0={d0}"));
        if shape[0] != d0 {
            passed = false;
        }
    }
    if let Some(d1) = config.hidden_size {
        expected_parts.push(format!("dim1={d1}"));
        if shape[1] != d1 {
            passed = false;
        }
    }

    // Popperian: if no dimensions to validate, emit no evidence.
    // Untested hypotheses must not be marked as corroborated.
    if expected_parts.is_empty() {
        return;
    }

    checks.push(DimensionalCheck {
        name: format!("tensor_{short_name}"),
        expected: expected_parts.join(", "),
        actual: format!("{shape:?}"),
        passed,
    });
}

/// Known valid SafeTensors dtype strings.
const SUPPORTED_DTYPES: &[&str] = &[
    "F32", "F16", "BF16", "F64", "I8", "I16", "I32", "I64", "U8", "U16", "U32", "U64", "BOOL",
];

/// Torch dtype string → SafeTensors dtype mapping.
fn torch_dtype_to_safetensors(torch_dtype: &str) -> Option<&'static str> {
    match torch_dtype {
        "float32" | "torch.float32" => Some("F32"),
        "float16" | "torch.float16" => Some("F16"),
        "bfloat16" | "torch.bfloat16" => Some("BF16"),
        "float64" | "torch.float64" => Some("F64"),
        "int8" | "torch.int8" => Some("I8"),
        "int16" | "torch.int16" => Some("I16"),
        "int32" | "torch.int32" => Some("I32"),
        "int64" | "torch.int64" => Some("I64"),
        "uint8" | "torch.uint8" => Some("U8"),
        "bool" | "torch.bool" => Some("BOOL"),
        _ => None,
    }
}

/// Tokenizer file discovery result with content validation.
struct TokenizerFiles {
    /// tokenizer.json found and is valid JSON
    json_valid: bool,
    /// tokenizer.model found and is non-empty
    model_valid: bool,
    /// Detail string for diagnostics
    detail: String,
}

/// Validate a single tokenizer.json file: must exist and parse as JSON.
/// Returns `(is_valid, detail_message)`.
fn validate_tokenizer_json(path: &Path) -> Option<(bool, &'static str)> {
    let content = std::fs::read_to_string(path).ok()?;
    if serde_json::from_str::<serde_json::Value>(&content).is_ok() {
        Some((true, "tokenizer.json found (valid JSON)"))
    } else {
        Some((false, "tokenizer.json found but invalid JSON"))
    }
}

/// Validate a single tokenizer.model file: must exist and be non-empty.
/// Returns `(is_valid, detail_message)`.
fn validate_tokenizer_model(path: &Path) -> Option<(bool, &'static str)> {
    let meta = std::fs::metadata(path).ok()?;
    if meta.len() > 0 {
        Some((true, "tokenizer.model found (non-empty)"))
    } else {
        Some((false, "tokenizer.model found but empty"))
    }
}

/// Find and validate tokenizer files in the model directory.
///
/// Checks model_path and model_path/safetensors/ for:
/// - `tokenizer.json`: must exist AND be parseable JSON (corrupt file = fail)
/// - `tokenizer.model`: must exist AND be non-empty (SentencePiece protobuf)
fn find_and_validate_tokenizer_files(model_path: &Path) -> TokenizerFiles {
    let search_dirs = [model_path.to_path_buf(), model_path.join("safetensors")];
    let mut json_valid = false;
    let mut model_valid = false;
    let mut detail = String::new();

    for dir in &search_dirs {
        if !json_valid {
            if let Some((valid, msg)) = validate_tokenizer_json(&dir.join("tokenizer.json")) {
                json_valid = valid;
                detail = msg.to_string();
            }
        }
        if !model_valid {
            if let Some((valid, msg)) = validate_tokenizer_model(&dir.join("tokenizer.model")) {
                model_valid = valid;
                if detail.is_empty() {
                    detail = msg.to_string();
                }
            }
        }
    }

    if detail.is_empty() {
        detail = "no tokenizer file".to_string();
    }

    TokenizerFiles {
        json_valid,
        model_valid,
        detail,
    }
}

/// Extract token string from a tokenizer_config value that can be either a plain
/// string or an object with a "content" field (Qwen format).
fn extract_token_string(value: &serde_json::Value) -> Option<String> {
    if let Some(s) = value.as_str() {
        if s.is_empty() {
            return None;
        }
        return Some(s.to_string());
    }
    if let Some(obj) = value.as_object() {
        if let Some(content) = obj.get("content").and_then(serde_json::Value::as_str) {
            if !content.is_empty() {
                return Some(content.to_string());
            }
        }
    }
    None
}

/// G0-TOKENIZER: Check tokenizer file presence, content validity, and EOS token.
fn check_tokenizer(model_path: &Path, checks: &mut Vec<DimensionalCheck>) {
    let tok = find_and_validate_tokenizer_files(model_path);
    let tokenizer_found = tok.json_valid || tok.model_valid;

    checks.push(DimensionalCheck {
        name: "tokenizer_exists".to_string(),
        expected: "valid tokenizer.json or non-empty tokenizer.model".to_string(),
        actual: tok.detail,
        passed: tokenizer_found,
    });

    // Remaining checks require tokenizer_config.json — skip if absent
    let raw_content = [
        model_path.join("tokenizer_config.json"),
        model_path.join("safetensors/tokenizer_config.json"),
    ]
    .iter()
    .find_map(|p| std::fs::read_to_string(p).ok());

    let Some(raw) = raw_content else {
        // No tokenizer_config.json — check config.json for eos_token_id fallback
        check_eos_token_id_fallback(model_path, checks);
        return;
    };

    let config_parsed = serde_json::from_str::<serde_json::Value>(&raw);
    let config_valid = config_parsed.is_ok();

    checks.push(DimensionalCheck {
        name: "tokenizer_config_valid".to_string(),
        expected: "valid JSON".to_string(),
        actual: if config_valid {
            "parsed successfully".to_string()
        } else {
            "invalid JSON".to_string()
        },
        passed: config_valid,
    });

    let Some(config) = config_parsed.ok() else {
        return;
    };

    // eos_token: required — missing eos_token falls back to eos_token_id in config.json
    if let Some(eos_val) = config.get("eos_token") {
        let eos = extract_token_string(eos_val);
        checks.push(DimensionalCheck {
            name: "eos_token_valid".to_string(),
            expected: "non-empty eos_token".to_string(),
            actual: eos.as_deref().unwrap_or("empty/null").to_string(),
            passed: eos.is_some(),
        });
    } else {
        // No eos_token string — check for eos_token_id in config.json (GPT-2 pattern)
        check_eos_token_id_fallback(model_path, checks);
    }

    // bos_token: optional — skip if absent
    if let Some(bos_val) = config.get("bos_token") {
        let bos = extract_token_string(bos_val);
        checks.push(DimensionalCheck {
            name: "bos_token_valid".to_string(),
            expected: "non-empty bos_token".to_string(),
            actual: bos.as_deref().unwrap_or("empty/null").to_string(),
            passed: bos.is_some(),
        });
    }
}

/// Fallback: check config.json for eos_token_id (integer) when eos_token (string) is absent.
///
/// GPT-2 style models define `eos_token_id: 50256` in config.json without an explicit
/// eos_token string in tokenizer_config.json. Either form is sufficient for stop token handling.
fn check_eos_token_id_fallback(model_path: &Path, checks: &mut Vec<DimensionalCheck>) {
    let config_paths = [
        model_path.join("config.json"),
        model_path.join("safetensors/config.json"),
    ];
    let eos_id = config_paths.iter().find_map(|p| {
        let content = std::fs::read_to_string(p).ok()?;
        let json: serde_json::Value = serde_json::from_str(&content).ok()?;
        json.get("eos_token_id").and_then(serde_json::Value::as_u64)
    });

    checks.push(DimensionalCheck {
        name: "eos_token_valid".to_string(),
        expected: "eos_token or eos_token_id".to_string(),
        actual: eos_id.map_or_else(
            || "missing (no eos_token or eos_token_id)".to_string(),
            |id| format!("eos_token_id={id} (from config.json)"),
        ),
        passed: eos_id.is_some(),
    });
}

/// Check if a tensor name is an embedding or lm_head tensor.
///
/// These are allowed to have different dtype from interior weight tensors
/// (e.g., F32 embeddings + BF16 weights for numerical stability in Llama-70B).
fn is_embedding_tensor(name: &str) -> bool {
    let lower = name.to_lowercase();
    lower.contains("embed_tokens")
        || lower.contains("lm_head")
        || lower.contains("word_embeddings")
        || lower.contains("wte")
        || lower.contains("wpe")
        || lower.contains("token_embeddings")
}

/// G0-DTYPE: Check SafeTensors dtype validity and consistency.
fn check_dtypes(model_path: &Path, config: &LayoutModelConfig, checks: &mut Vec<DimensionalCheck>) {
    let st_files = find_safetensors_files(model_path);
    let Some(first_file) = st_files.first() else {
        checks.push(DimensionalCheck {
            name: "dtype_supported".to_string(),
            expected: "at least one SafeTensors file".to_string(),
            actual: "no SafeTensors files found".to_string(),
            passed: false,
        });
        return;
    };

    let Ok(tensors) = read_safetensors_metadata_with_dtypes(first_file) else {
        checks.push(DimensionalCheck {
            name: "dtype_supported".to_string(),
            expected: "readable SafeTensors metadata".to_string(),
            actual: "failed to read SafeTensors header".to_string(),
            passed: false,
        });
        return;
    };

    if tensors.is_empty() {
        checks.push(DimensionalCheck {
            name: "dtype_supported".to_string(),
            expected: "at least one tensor".to_string(),
            actual: "SafeTensors file contains no tensors".to_string(),
            passed: false,
        });
        return;
    }

    // dtype_supported: check that all dtypes are known SafeTensors types
    let all_dtypes: HashSet<&str> = tensors.values().map(|t| t.dtype.as_str()).collect();
    let unsupported: Vec<&&str> = all_dtypes
        .iter()
        .filter(|d| !SUPPORTED_DTYPES.contains(d))
        .collect();

    checks.push(DimensionalCheck {
        name: "dtype_supported".to_string(),
        expected: "all dtypes in supported set".to_string(),
        actual: if unsupported.is_empty() {
            format!(
                "all supported ({})",
                all_dtypes.iter().copied().collect::<Vec<_>>().join(", ")
            )
        } else {
            format!(
                "unsupported: {}",
                unsupported
                    .iter()
                    .map(|d| **d)
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        },
        passed: unsupported.is_empty(),
    });

    // dtype_consistent: interior 2D weight tensors (excluding embed_tokens and lm_head)
    // should use the same dtype. Embedding layers are allowed to differ — Llama-70B
    // uses F32 embeddings + BF16 weights for numerical stability.
    let interior_weight_dtypes: HashSet<&str> = tensors
        .iter()
        .filter(|(name, info)| info.shape.len() == 2 && !is_embedding_tensor(name))
        .map(|(_, info)| info.dtype.as_str())
        .collect();

    // Collect ALL 2D dtypes for dtype_config_match (including embeddings)
    let weight_dtypes: HashSet<&str> = tensors
        .values()
        .filter(|t| t.shape.len() == 2)
        .map(|t| t.dtype.as_str())
        .collect();

    checks.push(DimensionalCheck {
        name: "dtype_consistent".to_string(),
        expected: "single dtype across interior 2D weight tensors".to_string(),
        actual: if interior_weight_dtypes.is_empty() {
            "no interior weight tensors found (all embeddings)".to_string()
        } else if interior_weight_dtypes.len() == 1 {
            format!(
                "uniform: {}",
                interior_weight_dtypes.iter().next().unwrap_or(&"?")
            )
        } else {
            format!(
                "mixed: {}",
                interior_weight_dtypes
                    .iter()
                    .copied()
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        },
        // Empty set is OK (model may have only embedding tensors), single dtype is consistent
        passed: interior_weight_dtypes.len() <= 1,
    });

    // dtype_config_match: if config.json has torch_dtype, verify it matches tensor dtypes
    check_dtype_config_match(model_path, config, &weight_dtypes, checks);
}

/// Sub-check: verify torch_dtype from config.json matches actual tensor dtypes.
fn check_dtype_config_match(
    model_path: &Path,
    _config: &LayoutModelConfig,
    weight_dtypes: &HashSet<&str>,
    checks: &mut Vec<DimensionalCheck>,
) {
    // Re-read config.json to get torch_dtype (LayoutModelConfig doesn't store it)
    let config_paths = [
        model_path.join("config.json"),
        model_path.join("safetensors/config.json"),
    ];
    let torch_dtype = config_paths.iter().find_map(|p| {
        let content = std::fs::read_to_string(p).ok()?;
        let json: serde_json::Value = serde_json::from_str(&content).ok()?;
        json.get("torch_dtype")
            .and_then(serde_json::Value::as_str)
            .map(String::from)
    });

    let Some(torch_dtype_str) = torch_dtype else {
        return;
    };

    let Some(expected_st_dtype) = torch_dtype_to_safetensors(&torch_dtype_str) else {
        checks.push(DimensionalCheck {
            name: "dtype_config_match".to_string(),
            expected: format!("torch_dtype '{torch_dtype_str}' maps to known SafeTensors dtype"),
            actual: "unknown torch_dtype mapping".to_string(),
            passed: false,
        });
        return;
    };

    let matches = !weight_dtypes.is_empty() && weight_dtypes.contains(expected_st_dtype);
    checks.push(DimensionalCheck {
        name: "dtype_config_match".to_string(),
        expected: format!("{expected_st_dtype} (from torch_dtype='{torch_dtype_str}')"),
        actual: if weight_dtypes.is_empty() {
            "no 2D tensors found (cannot verify)".to_string()
        } else {
            weight_dtypes.iter().copied().collect::<Vec<_>>().join(", ")
        },
        passed: matches,
    });
}

#[cfg(test)]
#[path = "dimensional_check_tests.rs"]
mod dimensional_check_tests;

#[cfg(test)]
#[path = "dimensional_check_tests_tokenizer_dtype.rs"]
mod dimensional_check_tests_tokenizer_dtype;
