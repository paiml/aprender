
// ============================================================================
// GH-182: COMPANION FILE HELPERS
// ============================================================================

/// Infer hidden_size from embedding tensor (BUG-EXPORT-001: pick smaller dim)
fn infer_hidden_size(tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>) -> (usize, bool) {
    tensors
        .iter()
        .find(|(name, _)| name.contains("embed_tokens") || name.contains("token_embd"))
        .map(|(name, (_, shape))| {
            let dim = if shape.len() >= 2 {
                let inferred = shape[0].min(shape[1]);
                eprintln!(
                    "[GH-197] Inferred hidden_size={inferred} from tensor '{name}' \
                     (shape={shape:?}, picked smaller dim)"
                );
                inferred
            } else {
                // C-16 (Meyer DbC): 0 = unknown, no architecture-specific magic number.
                shape.last().copied().unwrap_or(0)
            };
            (dim, true)
        })
        .unwrap_or((0, false))
}

/// Count transformer layers from tensor name patterns
fn infer_num_layers(tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>) -> usize {
    let max_layer: Option<usize> = tensors
        .keys()
        .filter_map(|name| {
            if name.contains("layers.") || name.contains("blk.") {
                let parts: Vec<&str> = name.split(&['.', '_'][..]).collect();
                for (i, part) in parts.iter().enumerate() {
                    if (*part == "layers" || *part == "blk") && i + 1 < parts.len() {
                        return parts[i + 1].parse::<usize>().ok();
                    }
                }
            }
            None
        })
        .max();

    if let Some(max) = max_layer {
        let count = max + 1;
        eprintln!("[GH-197] Inferred num_layers={count} from layer indices 0..{max}");
        count
    } else {
        12
    }
}

/// Infer vocab_size from lm_head, output, or embedding tensor (BUG-EXPORT-001: pick larger dim)
fn infer_vocab_size(tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>) -> (usize, bool) {
    tensors
        .iter()
        .find(|(name, _)| name.contains("lm_head") || name.contains("output.weight"))
        .or_else(|| {
            tensors
                .iter()
                .find(|(name, _)| name.contains("embed_tokens") || name.contains("token_embd"))
        })
        .map(|(name, (_, shape))| {
            let dim = if shape.len() >= 2 {
                let inferred = shape[0].max(shape[1]);
                eprintln!(
                    "[GH-197] Inferred vocab_size={inferred} from tensor '{name}' \
                     (shape={shape:?}, picked larger dim)"
                );
                inferred
            } else {
                // C-16 (Meyer DbC): 0 = unknown, no architecture-specific magic number.
                shape.first().copied().unwrap_or(0)
            };
            (dim, true)
        })
        .unwrap_or((0, false))
}

/// Infer model config.json from tensor shapes (GH-182, GH-193)
fn infer_model_config(tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>) -> String {
    let (hidden_size, hidden_inferred) = infer_hidden_size(tensors);
    let num_layers = infer_num_layers(tensors);
    let (vocab_size, vocab_inferred) = infer_vocab_size(tensors);

    // GH-197 FIX: Sanity validation
    if vocab_inferred && hidden_inferred && vocab_size < hidden_size {
        eprintln!(
            "[GH-197] WARNING: vocab_size ({vocab_size}) < hidden_size ({hidden_size}). \
             This is unusual for LLMs - dimensions may be swapped!"
        );
    }

    // GH-193: Infer num_attention_heads from attention Q/K/V weights
    // Shape is typically [hidden_size, num_heads * head_dim]
    let num_attention_heads = tensors
        .iter()
        .find(|(name, _)| {
            name.contains("self_attn.q_proj")
                || name.contains("attn.q_proj")
                || name.contains("attention.wq")
        })
        .map(|(_, (_, _shape))| {
            // Common head dimensions: 64, 128 - infer num_heads from hidden_size
            // Most models use head_dim = hidden_size / num_heads
            // Common configs: 4096/32=128 head_dim, 2048/16=128, etc.
            let head_dim = if hidden_size >= 4096 { 128 } else { 64 };
            hidden_size / head_dim
        })
        .unwrap_or_else(|| {
            // Fallback: standard ratios
            match hidden_size {
                896 => 14,                       // Qwen2.5-0.5B
                1536 => 12,                      // Qwen2.5-1.5B
                2048 => 16,                      // Llama-7B style
                4096 => 32,                      // Llama-7B
                5120 => 40,                      // Llama-13B
                8192 => 64,                      // Llama-70B
                _ => (hidden_size / 128).max(1), // Default: head_dim=128
            }
        });

    // GH-193: Infer intermediate_size from MLP weights
    let intermediate_size = tensors
        .iter()
        .find(|(name, _)| {
            name.contains("mlp.gate_proj")
                || name.contains("mlp.up_proj")
                || name.contains("feed_forward.w1")
        })
        .map(|(_, (_, shape))| shape.first().copied().unwrap_or(hidden_size * 4))
        .unwrap_or(hidden_size * 4); // Default to 4x hidden_size (common in transformers)

    // GH-193: Infer head_dim (guard against division by zero)
    let head_dim = if num_attention_heads > 0 {
        hidden_size / num_attention_heads
    } else {
        64 // Default head dimension
    };

    // GH-193: Infer num_key_value_heads (GQA support)
    // Look for k_proj shape to detect if using GQA (grouped query attention)
    let num_key_value_heads = tensors
        .iter()
        .find(|(name, _)| {
            name.contains("self_attn.k_proj")
                || name.contains("attn.k_proj")
                || name.contains("attention.wk")
        })
        .map(|(_, (_, shape))| {
            // For GQA: k_proj shape is [hidden_size, num_kv_heads * head_dim]
            // If shape[0] < hidden_size, it's GQA
            let kv_dim = shape.first().copied().unwrap_or(hidden_size);
            // Guard against division by zero
            if head_dim > 0 {
                (kv_dim / head_dim).max(1)
            } else {
                1
            }
        })
        .unwrap_or(num_attention_heads); // Default: same as num_attention_heads (MHA)

    // Create HuggingFace-compatible config.json with all required fields (GH-193)
    format!(
        r#"{{
  "architectures": ["Qwen2ForCausalLM"],
  "bos_token_id": 151643,
  "eos_token_id": 151645,
  "hidden_act": "silu",
  "hidden_size": {hidden_size},
  "initializer_range": 0.02,
  "intermediate_size": {intermediate_size},
  "max_position_embeddings": 32768,
  "model_type": "qwen2",
  "num_attention_heads": {num_attention_heads},
  "num_hidden_layers": {num_layers},
  "num_key_value_heads": {num_key_value_heads},
  "rms_norm_eps": 1e-06,
  "rope_theta": 1000000.0,
  "sliding_window": 32768,
  "tie_word_embeddings": true,
  "torch_dtype": "bfloat16",
  "use_cache": true,
  "use_sliding_window": false,
  "vocab_size": {vocab_size}
}}"#
    )
}

/// Extract tokenizer.json from APR input file (GH-182)
///
/// If the input is APR format with embedded tokenizer, extract it.
/// Otherwise return empty string.
fn infer_tokenizer_json(input_path: &Path) -> String {
    if input_path.extension().and_then(|e| e.to_str()) != Some("apr") {
        return String::new();
    }
    extract_apr_tokenizer_hint(input_path).unwrap_or_default()
}

/// Try to extract tokenizer hint from APR metadata section.
fn extract_apr_tokenizer_hint(input_path: &Path) -> Option<String> {
    let data = fs::read(input_path).ok()?;
    if data.len() <= 44 {
        return None;
    }
    let metadata_start = 44;
    let metadata_end = data[metadata_start..]
        .windows(4)
        .position(|w| w == b"}\n\n\n" || w == b"}\r\n\r")
        .map(|p| metadata_start + p + 1)?;
    let metadata_str = std::str::from_utf8(&data[metadata_start..metadata_end]).ok()?;
    if metadata_str.contains("\"tokenizer\"") || metadata_str.contains("\"vocabulary\"") {
        Some(r#"{"version": "1.0", "model": {"type": "BPE"}}"#.to_string())
    } else {
        None
    }
}

/// ROSETTA-003: Read APR v2 metadata from file.
///
/// Returns `None` for non-APR files or on any read/parse failure.
fn read_apr_metadata(apr_path: &Path) -> Option<crate::format::v2::AprV2Metadata> {
    if apr_path.extension().and_then(|e| e.to_str()) != Some("apr") {
        return None;
    }
    let data = fs::read(apr_path).ok()?;
    let reader = crate::format::v2::AprV2Reader::from_bytes(&data).ok()?;
    Some(reader.metadata().clone())
}

/// ROSETTA-003: Unfuse legacy QKV tensors for lossless round-trip export.
///
/// Old APR files (pre-ROSETTA-003) stored fused `qkv_proj.weight` tensors.
/// This function splits them back into separate `q_proj`, `k_proj`, `v_proj`
/// for correct GGUF/SafeTensors export. New APR files with separate Q/K/V
/// pass through unchanged.
/// Split a fused QKV weight tensor into separate Q, K, V weight tensors.
fn split_qkv_weight(
    name: &str,
    data: &[f32],
    shape: &[usize],
    hidden_size: usize,
    kv_dim: usize,
    result: &mut BTreeMap<String, (Vec<f32>, Vec<usize>)>,
) -> bool {
    let hidden_dim = if shape.len() >= 2 { shape[1] } else { hidden_size };
    let q_elements = hidden_size * hidden_dim;
    let kv_elements = kv_dim * hidden_dim;

    if data.len() < q_elements + 2 * kv_elements {
        return false;
    }

    let prefix = name.strip_suffix("qkv_proj.weight").unwrap_or(name);
    result.insert(
        format!("{prefix}q_proj.weight"),
        (data[..q_elements].to_vec(), vec![hidden_size, hidden_dim]),
    );
    result.insert(
        format!("{prefix}k_proj.weight"),
        (
            data[q_elements..q_elements + kv_elements].to_vec(),
            vec![kv_dim, hidden_dim],
        ),
    );
    result.insert(
        format!("{prefix}v_proj.weight"),
        (
            data[q_elements + kv_elements..q_elements + 2 * kv_elements].to_vec(),
            vec![kv_dim, hidden_dim],
        ),
    );
    true
}

/// Split a fused QKV bias tensor into separate Q, K, V bias tensors.
fn split_qkv_bias(
    name: &str,
    data: &[f32],
    hidden_size: usize,
    kv_dim: usize,
    result: &mut BTreeMap<String, (Vec<f32>, Vec<usize>)>,
) -> bool {
    let qkv_dim = hidden_size + 2 * kv_dim;
    if data.len() != qkv_dim {
        return false;
    }

    let prefix = name.strip_suffix("qkv_proj.bias").unwrap_or(name);
    result.insert(
        format!("{prefix}q_proj.bias"),
        (data[..hidden_size].to_vec(), vec![hidden_size]),
    );
    result.insert(
        format!("{prefix}k_proj.bias"),
        (
            data[hidden_size..hidden_size + kv_dim].to_vec(),
            vec![kv_dim],
        ),
    );
    result.insert(
        format!("{prefix}v_proj.bias"),
        (data[hidden_size + kv_dim..].to_vec(), vec![kv_dim]),
    );
    true
}

fn unfuse_qkv_tensors(
    tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    apr_path: &Path,
) -> BTreeMap<String, (Vec<f32>, Vec<usize>)> {
    let has_fused = tensors.keys().any(|k| k.contains("qkv_proj."));
    if !has_fused {
        return tensors;
    }

    let metadata = read_apr_metadata(apr_path);
    let (hidden_size, num_heads, num_kv_heads) = match &metadata {
        Some(m) => {
            let hs = m.hidden_size.unwrap_or(0);
            let nh = m.num_heads.unwrap_or(0);
            let nkv = m.num_kv_heads.unwrap_or(nh);
            (hs, nh, nkv)
        }
        None => return tensors,
    };

    if hidden_size == 0 || num_heads == 0 {
        return tensors;
    }

    let head_dim = hidden_size / num_heads;
    let kv_dim = num_kv_heads * head_dim;

    let mut result = BTreeMap::new();

    for (name, (data, shape)) in tensors {
        if name.contains("qkv_proj.weight") {
            if !split_qkv_weight(&name, &data, &shape, hidden_size, kv_dim, &mut result) {
                result.insert(name, (data, shape));
            }
        } else if name.contains("qkv_proj.bias") {
            if !split_qkv_bias(&name, &data, hidden_size, kv_dim, &mut result) {
                result.insert(name, (data, shape));
            }
        } else {
            result.insert(name, (data, shape));
        }
    }

    result
}

/// ROSETTA-003: Remove synthesized `lm_head.weight` for SafeTensors export.
///
/// When the APR metadata has `tied_embeddings: true`, the lm_head was copied
/// from embed_tokens during import. For SafeTensors round-trip fidelity,
/// remove it so the exported file matches the original HuggingFace convention.
fn remove_tied_lm_head(
    mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    apr_path: &Path,
) -> BTreeMap<String, (Vec<f32>, Vec<usize>)> {
    let metadata = read_apr_metadata(apr_path);
    let is_tied = metadata
        .as_ref()
        .and_then(|m| m.custom.get("tied_embeddings"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    if is_tied {
        tensors.remove("lm_head.weight");
    }

    tensors
}

/// PMAT-223: Extract user metadata from APR file's custom field.
///
/// Reads the APR metadata JSON and looks for the `"source_metadata"` key
/// that was preserved during import from SafeTensors.
fn extract_user_metadata(apr_path: &Path) -> UserMetadata {
    let data = match fs::read(apr_path) {
        Ok(d) => d,
        Err(_) => return UserMetadata::new(),
    };

    // APR v2 format: magic(4) + version(4) + metadata_len(8) + metadata_json
    if data.len() < 16 {
        return UserMetadata::new();
    }

    let metadata_len = u64::from_le_bytes(data[8..16].try_into().unwrap_or([0u8; 8])) as usize;

    if data.len() < 16 + metadata_len {
        return UserMetadata::new();
    }

    let metadata_json = match std::str::from_utf8(&data[16..16 + metadata_len]) {
        Ok(s) => s,
        Err(_) => return UserMetadata::new(),
    };

    let parsed: serde_json::Value = match serde_json::from_str(metadata_json) {
        Ok(v) => v,
        Err(_) => return UserMetadata::new(),
    };

    // Look for custom.source_metadata
    if let Some(serde_json::Value::Object(map)) =
        parsed.get("custom").and_then(|c| c.get("source_metadata"))
    {
        let mut result = UserMetadata::new();
        for (k, v) in map {
            if let serde_json::Value::String(s) = v {
                result.insert(k.clone(), s.clone());
            }
        }
        return result;
    }

    UserMetadata::new()
}

/// Detect predominant quantization type from an APR file (PMAT-252).
///
/// Reads the tensor index and checks the dtype of 2D weight tensors.
/// Returns the `QuantizationType` if the majority of weights use a
/// quantized format (Q4K, Q6K), or `None` for F32/F16 files.
pub(crate) fn detect_apr_quantization(apr_path: &Path) -> Option<QuantizationType> {
    use crate::format::v2::{AprV2Reader, TensorDType};

    let data = fs::read(apr_path).ok()?;
    let reader = AprV2Reader::from_bytes(&data).ok()?;

    // Count dtypes across 2D weight tensors (skip 1D biases/norms)
    let mut q4k_count = 0usize;
    let mut q6k_count = 0usize;
    let mut other_count = 0usize;

    for name in reader.tensor_names() {
        if let Some(entry) = reader.get_tensor(name) {
            if entry.shape.len() >= 2 {
                match entry.dtype {
                    TensorDType::Q4K => q4k_count += 1,
                    TensorDType::Q6K => q6k_count += 1,
                    _ => other_count += 1,
                }
            }
        }
    }

    let total = q4k_count + q6k_count + other_count;
    if total == 0 {
        return None;
    }

    // If majority of 2D tensors are Q4K, default to Q4K export
    if q4k_count > q6k_count && q4k_count > other_count {
        return Some(QuantizationType::Q4K);
    }

    // Q6K not yet in QuantizationType — treat as no auto-detect
    None
}
