
/// Resolve GGUF tokenizer: use embedded, external, or error.
fn resolve_gguf_tokenizer(
    embedded: &GgufTokenizer,
    gguf_path: &Path,
    external_path: Option<&Path>,
) -> Result<GgufTokenizer> {
    if !embedded.vocabulary.is_empty() {
        if embedded.merges.is_empty() {
            eprintln!(
                "[PMAT-232] WARNING: GGUF file has vocabulary but no BPE merges. \
                 Text encoding may fail for multi-character tokens."
            );
        } else {
            eprintln!(
                "[PMAT-232] Tokenizer validated: {} vocab tokens, {} merge rules",
                embedded.vocabulary.len(),
                embedded.merges.len()
            );
        }
        return Ok(embedded.clone());
    }
    // No embedded tokenizer — try external
    if let Some(tokenizer_path) = external_path {
        eprintln!(
            "[PMAT-232] GGUF has no embedded tokenizer, trying external: {}",
            tokenizer_path.display()
        );
        return load_tokenizer_from_explicit_path(tokenizer_path).ok_or_else(|| {
            AprenderError::FormatError {
                message: format!(
                    "Failed to load external tokenizer from '{}'. \
                     Ensure the file is a valid HuggingFace tokenizer.json.",
                    tokenizer_path.display()
                ),
            }
        });
    }
    Err(AprenderError::FormatError {
        message: format!(
            "GGUF file '{}' has no embedded tokenizer vocabulary. \
             Solutions: (1) Use a GGUF with embedded tokenizer, or \
             (2) Provide --tokenizer /path/to/tokenizer.json, or \
             (3) Use SafeTensors format with sibling tokenizer.json, or \
             (4) Import from HuggingFace source: apr import hf://ORG/REPO -o model.apr",
            gguf_path.display()
        ),
    })
}

/// Get XDG cache directory or fallback.
fn get_xdg_cache_dir() -> PathBuf {
    std::env::var("XDG_CACHE_HOME")
        .ok()
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::var("HOME")
                .map(|h| PathBuf::from(h).join(".cache"))
                .unwrap_or_else(|_| PathBuf::from(".cache"))
        })
}

/// Get `HuggingFace` cache directory.
fn get_hf_cache_dir() -> PathBuf {
    std::env::var("HF_HOME")
        .ok()
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::var("HOME")
                .map(|h| PathBuf::from(h).join(".cache").join("huggingface"))
                .unwrap_or_else(|_| PathBuf::from(".cache").join("huggingface"))
        })
}

/// Check aprender cache for a file.
fn find_in_aprender_cache(
    cache_base: &Path,
    org: &str,
    repo: &str,
    filename: &str,
) -> Option<PathBuf> {
    let apr_cache = cache_base
        .join("aprender")
        .join("hf")
        .join(org)
        .join(repo)
        .join(filename);
    apr_cache.exists().then_some(apr_cache)
}

/// Check `HuggingFace` hub cache for a file.
fn find_in_hf_hub_cache(
    cache_base: &Path,
    org: &str,
    repo: &str,
    filename: &str,
) -> Option<PathBuf> {
    let hf_cache = cache_base
        .join("hub")
        .join(format!("models--{org}--{repo}"));

    if !hf_cache.exists() {
        return None;
    }

    let snapshot_dir = hf_cache.join("snapshots");
    let entries = fs::read_dir(&snapshot_dir).ok()?;

    for entry in entries.flatten() {
        let file_path = entry.path().join(filename);
        if file_path.exists() {
            return Some(file_path);
        }
    }
    None
}

/// Find a model file in standard cache locations
fn find_in_cache(org: &str, repo: &str, filename: &str) -> Option<PathBuf> {
    let cache_paths = [get_xdg_cache_dir(), get_hf_cache_dir()];

    for cache_base in &cache_paths {
        if let Some(path) = find_in_aprender_cache(cache_base, org, repo, filename) {
            return Some(path);
        }
        if let Some(path) = find_in_hf_hub_cache(cache_base, org, repo, filename) {
            return Some(path);
        }
    }

    None
}

/// Download a file from HuggingFace Hub
#[cfg(feature = "hf-hub-integration")]
fn download_from_hf(repo_id: &str, filename: &str) -> Result<PathBuf> {
    use hf_hub::api::sync::ApiBuilder;

    // Build API client (uses HF_TOKEN if available)
    let token = std::env::var("HF_TOKEN").ok();
    let mut builder = ApiBuilder::new();
    if let Some(t) = token {
        builder = builder.with_token(Some(t));
    }

    let api = builder.build().map_err(|e| {
        let resource = format!("{repo_id}/{filename}");
        let err = parse_import_error(&e.to_string(), &resource);
        AprenderError::from(err)
    })?;

    // Get repo handle
    let repo = api.model(repo_id.to_string());

    // Download the file (GH-129: parse error for actionable messages)
    let path = repo.get(filename).map_err(|e| {
        let resource = format!("{repo_id}/{filename}");
        let err = parse_import_error(&e.to_string(), &resource);
        AprenderError::from(err)
    })?;

    Ok(path)
}

/// Result of loading source tensors (may include tokenizer data)
#[derive(Debug)]
pub(crate) struct SourceLoadResult {
    /// Tensor data (name -> (data, shape))
    pub(crate) tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    /// GH-205: Raw F16 tensor bytes for passthrough (name -> bytes)
    /// Tensors in this map should NOT be converted - write raw to APR
    pub(crate) f16_raw_tensors: BTreeMap<String, (Vec<u8>, Vec<usize>)>,
    /// Tokenizer data (only present for GGUF files)
    pub(crate) tokenizer: Option<GgufTokenizer>,
    /// Model config (CRITICAL for inference - from GGUF)
    pub(crate) model_config: Option<GgufModelConfig>,
    /// PMAT-223: User metadata from SafeTensors `__metadata__` section
    pub(crate) user_metadata: UserMetadata,
}

/// Load model config from config.json alongside the model file (PMAT-098)
///
/// This is the preferred way to get model config for SafeTensors models.
/// Falls back to shape inference if config.json is not found.
pub(crate) fn load_model_config_from_json(model_path: &Path) -> Option<GgufModelConfig> {
    // Look for config.json alongside the model file
    let config_path = model_path.with_file_name("config.json");
    if !config_path.exists() {
        return None;
    }

    let content = fs::read_to_string(&config_path).ok()?;
    let json: serde_json::Value = serde_json::from_str(&content).ok()?;

    // Parse HuggingFace config.json format
    // GH-235: GPT-2 uses different field names (n_embd, n_head, n_layer, n_inner, n_positions).
    // Try standard names first, fall back to GPT-2 aliases.
    let hidden_size = json
        .get("hidden_size")
        .or_else(|| json.get("n_embd")) // GPT-2
        .and_then(serde_json::Value::as_u64)
        .map(|v| v as usize);

    let num_layers = json
        .get("num_hidden_layers")
        .or_else(|| json.get("n_layer")) // GPT-2
        .and_then(serde_json::Value::as_u64)
        .map(|v| v as usize);

    let num_heads = json
        .get("num_attention_heads")
        .or_else(|| json.get("n_head")) // GPT-2
        .and_then(serde_json::Value::as_u64)
        .map(|v| v as usize);

    let num_kv_heads = json
        .get("num_key_value_heads")
        .and_then(serde_json::Value::as_u64)
        .map(|v| v as usize)
        .or(num_heads); // Default to num_heads if not specified (no GQA)

    let vocab_size = json
        .get("vocab_size")
        .and_then(serde_json::Value::as_u64)
        .map(|v| v as usize);

    let intermediate_size = json
        .get("intermediate_size")
        .or_else(|| json.get("n_inner")) // GPT-2
        .and_then(serde_json::Value::as_u64)
        .map(|v| v as usize)
        .or_else(|| hidden_size.map(|h| 4 * h)); // GPT-2 default: 4 * hidden_size

    let max_position_embeddings = json
        .get("max_position_embeddings")
        .or_else(|| json.get("n_positions")) // GPT-2
        .and_then(serde_json::Value::as_u64)
        .map(|v| v as usize);

    let rope_theta = json
        .get("rope_theta")
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(10000.0);

    let rms_norm_eps = json
        .get("rms_norm_eps")
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(1e-6);

    let architecture = json
        .get("model_type")
        .and_then(|v| v.as_str())
        .map(ToString::to_string);

    // PMAT-114: Infer rope_type from architecture
    // Qwen2/Qwen2.5/Qwen3 and Phi models use NEOX-style RoPE (type 2)
    let rope_type = match architecture.as_deref() {
        Some("qwen2" | "qwen2.5" | "qwen" | "qwen3") => Some(2),
        Some("phi" | "phi3" | "phi4") => Some(2),
        _ => Some(0),
    };

    Some(GgufModelConfig {
        architecture,
        hidden_size,
        num_layers,
        num_heads,
        num_kv_heads,
        vocab_size,
        intermediate_size,
        max_position_embeddings,
        rope_theta: Some(rope_theta as f32),
        rms_norm_eps: Some(rms_norm_eps as f32),
        rope_type,
    })
}

/// Parse tokenizer from already-loaded JSON content.
///
/// Extracted from `load_tokenizer_from_json` for testability. This is the pure
/// JSON-parsing core that receives parsed JSON values directly, with no filesystem I/O.
///
/// # Arguments
/// * `json` - Parsed tokenizer.json content
/// * `config_json` - Optional parsed config.json content (for `vocab_size`, BOS/EOS)
pub(crate) fn parse_tokenizer_json(
    json: &serde_json::Value,
    config_json: Option<&serde_json::Value>,
) -> Option<GgufTokenizer> {
    // Step 1: Build token-to-id map from model.vocab + added_tokens
    let (token_to_id, base_vocab_len) = parse_vocab_from_model(json)?;

    let added_count = json
        .get("added_tokens")
        .and_then(|v| v.as_array())
        .map(|a| a.len())
        .unwrap_or(0);

    // Step 2: Build padded vocabulary vector
    let expected_vocab_size = config_json
        .and_then(|cfg| cfg.get("vocab_size").and_then(|v| v.as_u64()))
        .map(|v| v as u32)
        .unwrap_or(0);

    let vocabulary = build_vocab_vector(&token_to_id, expected_vocab_size);

    eprintln!(
        "[BUG-EXPORT-004] Vocab: base={}, added={}, expected={}, final={}",
        base_vocab_len,
        added_count,
        expected_vocab_size,
        vocabulary.len()
    );

    if vocabulary.is_empty() {
        return None;
    }

    // Step 3: Extract BOS/EOS special token IDs
    let (bos_token_id, eos_token_id) = parse_special_tokens(json, config_json);

    // Step 4: Extract model type and merge rules
    let model_type = json
        .get("model")
        .and_then(|m| m.get("type"))
        .and_then(|t| t.as_str())
        .map(String::from);

    let merges = parse_merges(json);

    Some(GgufTokenizer {
        vocabulary,
        merges,
        model_type,
        bos_token_id,
        eos_token_id,
        architecture: None,
        model_name: None,
        ..Default::default()
    })
}

/// Build a token-to-id map from the base vocabulary in `model.vocab`, then overlay
/// any entries from the `added_tokens` array.
///
/// Returns `(token_to_id_map, base_vocab_len)` or `None` if `model.vocab` is missing.
fn parse_vocab_from_model(
    json: &serde_json::Value,
) -> Option<(std::collections::BTreeMap<u32, String>, usize)> {
    let vocab_obj = json.get("model")?.get("vocab")?;
    let vocab_map = vocab_obj.as_object()?;
    let base_vocab_len = vocab_map.len();

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

    Some((token_to_id, base_vocab_len))
}

/// Extract BOS/EOS token IDs.
///
/// BUG-EXPORT-004: Priority 1 is config.json (authoritative). Priority 2 is
/// inferring from `added_tokens` patterns in tokenizer.json (fallback via
/// `infer_bos_eos_from_added_tokens`).
fn parse_special_tokens(
    json: &serde_json::Value,
    config_json: Option<&serde_json::Value>,
) -> (Option<u32>, Option<u32>) {
    let mut bos_token_id = config_json
        .and_then(|cfg| cfg.get("bos_token_id"))
        .and_then(|v| v.as_u64())
        .map(|v| v as u32);
    let mut eos_token_id = config_json
        .and_then(|cfg| cfg.get("eos_token_id"))
        .and_then(|v| v.as_u64())
        .map(|v| v as u32);

    if config_json.is_some() && (bos_token_id.is_some() || eos_token_id.is_some()) {
        eprintln!(
            "[BUG-EXPORT-004] Read BOS/EOS from config.json: bos={:?}, eos={:?}",
            bos_token_id, eos_token_id
        );
    }

    // Fallback: infer from added_tokens (less reliable)
    if bos_token_id.is_none() || eos_token_id.is_none() {
        if let Some(added_tokens) = json.get("added_tokens").and_then(|v| v.as_array()) {
            (bos_token_id, eos_token_id) =
                infer_bos_eos_from_added_tokens(added_tokens, bos_token_id, eos_token_id);
        }
    }

    (bos_token_id, eos_token_id)
}

/// Extract BPE merge rules from `model.merges` (PMAT-171).
fn parse_merges(json: &serde_json::Value) -> Vec<String> {
    json.get("model")
        .and_then(|m| m.get("merges"))
        .and_then(|m| m.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default()
}
