
/// Load tokenizer from sibling tokenizer.json file (PMAT-APR-TOK-001)
///
/// For SafeTensors models, the tokenizer is stored in a separate tokenizer.json file.
/// This function reads it and converts to GgufTokenizer format for APR embedding.
///
/// BUG-TOK-002 FIX: Support both HuggingFace layout (tokenizer.json) and
/// Pacha cache layout ({hash}.tokenizer.json).
pub(crate) fn load_tokenizer_from_json(model_path: &Path) -> Option<GgufTokenizer> {
    // Try standard HuggingFace layout: tokenizer.json in same directory
    let standard_path = model_path.with_file_name("tokenizer.json");

    // Try Pacha cache layout: {hash}.tokenizer.json (same stem as model)
    let stem = model_path.file_stem()?.to_str()?;
    // Strip any existing extensions like .converted from the stem
    let base_stem = stem.split('.').next().unwrap_or(stem);
    let pacha_path = model_path.with_file_name(format!("{}.tokenizer.json", base_stem));

    // GH-226: Also check parent directory and sibling subdirectories.
    // QA workspace layout: safetensors/, apr/, gguf/ are siblings under a model dir,
    // with tokenizer.json in the safetensors/ subdirectory. When exporting APR→GGUF,
    // the input APR is at apr/model.apr but tokenizer.json is at ../safetensors/tokenizer.json.
    let parent_path = model_path
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.join("tokenizer.json"));
    let safetensors_sibling_path = model_path
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.join("safetensors").join("tokenizer.json"));

    let tokenizer_path = if standard_path.exists() {
        standard_path
    } else if pacha_path.exists() {
        eprintln!(
            "[BUG-TOK-002] Found tokenizer at Pacha cache path: {}",
            pacha_path.display()
        );
        pacha_path
    } else if parent_path.as_ref().is_some_and(|p| p.exists()) {
        let p = parent_path.expect("checked above");
        eprintln!(
            "[GH-226] Found tokenizer in parent directory: {}",
            p.display()
        );
        p
    } else if safetensors_sibling_path
        .as_ref()
        .is_some_and(|p| p.exists())
    {
        let p = safetensors_sibling_path.expect("checked above");
        eprintln!(
            "[GH-226] Found tokenizer in sibling safetensors/: {}",
            p.display()
        );
        p
    } else {
        return None;
    };

    let content = fs::read_to_string(&tokenizer_path).ok()?;
    let json: serde_json::Value = serde_json::from_str(&content).ok()?;

    // Load config.json (try standard path, then Pacha cache path)
    let config_path = tokenizer_path.with_file_name("config.json");
    let pacha_config_path = tokenizer_path
        .file_stem()
        .and_then(|s| s.to_str())
        .map(|s| s.split('.').next().unwrap_or(s))
        .map(|stem| tokenizer_path.with_file_name(format!("{stem}.config.json")));

    let config_json = config_path
        .exists()
        .then(|| fs::read_to_string(&config_path).ok())
        .flatten()
        .or_else(|| {
            pacha_config_path
                .as_ref()
                .filter(|p| p.exists())
                .and_then(|p| fs::read_to_string(p).ok())
        })
        .and_then(|s| {
            let sanitized = sanitize_hf_json(&s);
            serde_json::from_str::<serde_json::Value>(&sanitized).ok()
        });

    parse_tokenizer_json(&json, config_json.as_ref())
}

/// Load sibling config.json from the same directory as a given file.
fn load_sibling_config(path: &Path) -> Option<serde_json::Value> {
    let config_path = path.with_file_name("config.json");
    config_path
        .exists()
        .then(|| fs::read_to_string(&config_path).ok())
        .flatten()
        .and_then(|s| {
            let sanitized = sanitize_hf_json(&s);
            serde_json::from_str::<serde_json::Value>(&sanitized).ok()
        })
}

/// Build vocabulary vector from token-to-id map, padded to expected vocab size.
fn build_vocab_vector(
    token_to_id: &std::collections::BTreeMap<u32, String>,
    expected_vocab_size: u32,
) -> Vec<String> {
    let max_id = token_to_id.keys().max().copied().unwrap_or(0);
    let final_size = (expected_vocab_size.max(max_id + 1)) as usize;
    let mut vocabulary: Vec<String> = vec!["<unk>".to_string(); final_size];
    for (id, token) in token_to_id {
        if (*id as usize) < vocabulary.len() {
            vocabulary[*id as usize] = token.clone();
        }
    }
    vocabulary
}

/// Infer BOS/EOS token IDs from added_tokens array by name heuristics.
fn infer_bos_eos_from_added_tokens(
    added_tokens: &[serde_json::Value],
    mut bos: Option<u32>,
    mut eos: Option<u32>,
) -> (Option<u32>, Option<u32>) {
    for token in added_tokens {
        let content = token.get("content").and_then(|v| v.as_str());
        let id = token.get("id").and_then(|v| v.as_u64()).map(|v| v as u32);
        let (Some(content), Some(id)) = (content, id) else {
            continue;
        };
        if bos.is_none() && is_bos_token(content) {
            bos = Some(id);
        }
        if eos.is_none() && is_eos_token(content) {
            eos = Some(id);
        }
    }
    (bos, eos)
}

fn is_bos_token(content: &str) -> bool {
    content.contains("bos") || content == "<s>" || content == "<|startoftext|>"
}

fn is_eos_token(content: &str) -> bool {
    content.contains("eos") || content == "</s>" || content == "<|eot_id|>"
}

/// PMAT-232: Load tokenizer from explicit path (for --tokenizer CLI option)
///
/// Unlike `load_tokenizer_from_json` which searches for tokenizer.json in standard locations,
/// this function loads directly from the provided path. Used for weights-only GGUF imports
/// where the tokenizer is provided externally.
pub(crate) fn load_tokenizer_from_explicit_path(tokenizer_path: &Path) -> Option<GgufTokenizer> {
    if !tokenizer_path.exists() {
        eprintln!(
            "[PMAT-232] External tokenizer not found: {}",
            tokenizer_path.display()
        );
        return None;
    }

    let content = fs::read_to_string(tokenizer_path).ok()?;
    let json: serde_json::Value = serde_json::from_str(&content).ok()?;

    let token_to_id = extract_vocab_with_added_tokens(&json)?;

    let sibling_config = load_sibling_config(tokenizer_path);
    let expected_vocab_size = get_config_u32(&sibling_config, "vocab_size");
    let vocabulary = build_vocab_vector(&token_to_id, expected_vocab_size);

    eprintln!(
        "[PMAT-232] External tokenizer loaded: {} vocab tokens from {}",
        vocabulary.len(),
        tokenizer_path.display()
    );

    if vocabulary.is_empty() {
        return None;
    }

    let (bos_token_id, eos_token_id) =
        resolve_bos_eos(&json, &sibling_config);

    Some(GgufTokenizer {
        vocabulary,
        merges: parse_merges(&json),
        model_type: extract_model_type(&json),
        bos_token_id,
        eos_token_id,
        architecture: None,
        model_name: None,
        ..Default::default()
    })
}

/// Extract base vocab + added_tokens into a single BTreeMap.
fn extract_vocab_with_added_tokens(
    json: &serde_json::Value,
) -> Option<std::collections::BTreeMap<u32, String>> {
    let vocab_obj = json.get("model")?.get("vocab")?;
    let vocab_map = vocab_obj.as_object()?;

    let mut token_to_id: std::collections::BTreeMap<u32, String> = vocab_map
        .iter()
        .filter_map(|(token, id)| Some((id.as_u64()? as u32, token.clone())))
        .collect();

    if let Some(added) = json.get("added_tokens").and_then(|v| v.as_array()) {
        for token in added {
            if let (Some(content), Some(id)) = (
                token.get("content").and_then(|v| v.as_str()),
                token.get("id").and_then(|v| v.as_u64()),
            ) {
                token_to_id.insert(id as u32, content.to_string());
            }
        }
    }

    Some(token_to_id)
}

/// Extract a u32 field from a config JSON, returning 0 if missing.
fn get_config_u32(config: &Option<serde_json::Value>, key: &str) -> u32 {
    config
        .as_ref()
        .and_then(|cfg| cfg.get(key).and_then(|v| v.as_u64()))
        .map(|v| v as u32)
        .unwrap_or(0)
}

/// Resolve BOS/EOS token IDs from sibling config.json, falling back to added_tokens.
fn resolve_bos_eos(
    json: &serde_json::Value,
    sibling_config: &Option<serde_json::Value>,
) -> (Option<u32>, Option<u32>) {
    let mut bos = sibling_config
        .as_ref()
        .and_then(|cfg| cfg.get("bos_token_id").and_then(|v| v.as_u64()))
        .map(|v| v as u32);
    let mut eos = sibling_config
        .as_ref()
        .and_then(|cfg| cfg.get("eos_token_id").and_then(|v| v.as_u64()))
        .map(|v| v as u32);

    if bos.is_none() || eos.is_none() {
        if let Some(added_tokens) = json.get("added_tokens").and_then(|v| v.as_array()) {
            (bos, eos) = infer_bos_eos_from_added_tokens(added_tokens, bos, eos);
        }
    }

    (bos, eos)
}

// extract_model_type() is defined in import_part_02.rs

/// Infer vocab_size and hidden_size from embedding tensor shape.
///
/// GH-165 FIX: Handles both shape orders (HF: `[vocab, hidden]`, GGUF: `[hidden, vocab]`).
fn infer_embedding_dims(
    tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>,
) -> Option<(usize, usize)> {
    tensors
        .iter()
        .find(|(name, _)| {
            name.contains("embed_tokens")
                || name.contains("wte")
                || name.contains("word_embeddings")
                || name.contains("token_embd") // GGUF naming
        })
        .and_then(|(_, (_, shape))| {
            if shape.len() == 2 {
                let (dim0, dim1) = (shape[0], shape[1]);
                if dim0 > dim1 {
                    Some((dim0, dim1)) // [vocab_size, hidden_size] - HuggingFace
                } else {
                    Some((dim1, dim0)) // [hidden_size, vocab_size] - GGUF
                }
            } else {
                None
            }
        })
}

/// Count transformer layers from tensor names (supports HF and GGUF naming).
fn count_transformer_layers(tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>) -> usize {
    tensors
        .keys()
        .filter_map(|name| {
            if let Some(start) = name.find("blk.") {
                let rest = &name[start + 4..];
                if let Some(end) = rest.find('.') {
                    if let Ok(n) = rest[..end].parse::<usize>() {
                        return Some(n);
                    }
                }
            }
            let patterns = [
                (name.find("layers."), 7),
                (name.find("h."), 2),
                (name.find("blocks."), 7),
            ];
            for (pos, skip_len) in patterns {
                if let Some(start) = pos {
                    let rest = &name[start + skip_len..];
                    if let Some(end) = rest.find('.') {
                        if let Ok(n) = rest[..end].parse::<usize>() {
                            return Some(n);
                        }
                    }
                }
            }
            None
        })
        .max()
        .map(|n| n + 1)
        .unwrap_or(0)
}

/// Find the smaller dimension of a 2D projection tensor matching any of the given name patterns.
fn find_projection_dim(
    tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    name_patterns: &[&str],
) -> Option<usize> {
    tensors
        .iter()
        .find(|(name, _)| name_patterns.iter().any(|p| name.contains(p)))
        .and_then(|(_, (_, shape))| {
            if shape.len() == 2 {
                Some(shape[0].min(shape[1]))
            } else {
                None
            }
        })
}

/// Infer num_heads and num_kv_heads from Q and KV projection dimensions.
///
/// BUG-EXPORT-004 FIX: Correctly handles GQA models where `kv_dim < q_dim`.
fn infer_head_counts(
    q_dim: Option<usize>,
    kv_dim: Option<usize>,
    hidden_size: usize,
) -> (Option<usize>, Option<usize>) {
    match (q_dim, kv_dim) {
        (Some(q), Some(kv)) if kv < q => infer_gqa_heads(q, kv),
        (Some(q), _) if q == hidden_size => infer_mha_heads(hidden_size),
        _ => (None, None),
    }
}

/// GQA model: derive head counts from common head dimensions.
fn infer_gqa_heads(q: usize, kv: usize) -> (Option<usize>, Option<usize>) {
    const HEAD_DIMS: [usize; 4] = [64, 128, 96, 80];
    for head_dim in HEAD_DIMS {
        if kv % head_dim == 0 && q % head_dim == 0 {
            let n_kv = kv / head_dim;
            let n_heads = q / head_dim;
            if n_heads >= n_kv && n_kv > 0 {
                return (Some(n_heads), Some(n_kv));
            }
        }
    }
    (None, None)
}

/// MHA model: q_dim == hidden_size, heads share same dimension.
fn infer_mha_heads(hidden_size: usize) -> (Option<usize>, Option<usize>) {
    const HEAD_DIMS: [usize; 4] = [64, 128, 96, 80];
    for head_dim in HEAD_DIMS {
        if hidden_size % head_dim == 0 {
            let n_heads = hidden_size / head_dim;
            return (Some(n_heads), Some(n_heads));
        }
    }
    (None, None)
}

/// Infer intermediate_size from gate/up projection tensor.
///
/// GH-165 FIX: Takes larger dimension since intermediate_size > hidden_size.
fn infer_intermediate_size_from_tensors(
    tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>,
) -> Option<usize> {
    tensors
        .iter()
        .find(|(name, _)| {
            name.contains("gate_proj")
                || name.contains("up_proj")
                || name.contains("fc1")
                || name.contains("ffn_gate")
                || name.contains("ffn_up")
        })
        .and_then(|(_, (_, shape))| {
            if shape.len() == 2 {
                Some(shape[0].max(shape[1]))
            } else {
                None
            }
        })
}

/// Infer architecture string from tensor naming conventions.
///
/// Bug 210 (GH-222): Previously assumed ALL `model.layers` models were "qwen2" — wrong.
/// LLaMA, Mistral, Phi all use `model.layers`. Now uses Qwen2-specific signals:
/// - Qwen2 has attention bias (`self_attn.q_proj.bias`) — LLaMA/Mistral do not.
/// - Qwen2 sometimes has fused `qkv_proj.weight`.
fn infer_architecture_from_names(
    tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>,
) -> Option<String> {
    let has_model_layers = tensors.keys().any(|k| k.contains("model.layers"));
    // GH-255: SafeTensors GPT-2 uses "h.N.*" without "transformer." prefix
    let has_transformer_h = tensors.keys().any(|k| k.contains("transformer.h"))
        || tensors
            .keys()
            .any(|k| k.starts_with("h.") && k.contains(".attn."));
    let has_blk = tensors.keys().any(|k| k.contains("blk."));

    if has_model_layers {
        // Distinguish Qwen2 from LLaMA/Mistral by attention bias presence
        let has_attn_bias = tensors.keys().any(|k| k.contains("self_attn.q_proj.bias"));
        let has_fused_qkv = tensors.keys().any(|k| k.contains("qkv_proj.weight"));
        if has_attn_bias || has_fused_qkv {
            Some("qwen2".to_string())
        } else {
            Some("llama".to_string())
        }
    } else if has_transformer_h {
        Some("gpt2".to_string())
    } else if has_blk {
        // GGUF naming — cannot reliably distinguish architectures
        Some("unknown".to_string())
    } else {
        Some("unknown".to_string())
    }
}
