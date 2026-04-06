/// Resolve a model path, downloading from HuggingFace Hub if it's a repo ID.
///
/// If `model_path` looks like a HF repo ID (e.g., "Qwen/Qwen2.5-Coder-0.5B"),
/// downloads the model to the HF cache and returns the resolved local path.
/// Otherwise, returns the original path unchanged.
#[cfg(feature = "hub-publish")]
fn resolve_model_path(model_path: &Path) -> Result<PathBuf> {
    use crate::config::schema::is_hf_repo_id;
    use crate::hf_pipeline::{FetchOptions, HfModelFetcher};

    let path_str = model_path.to_string_lossy();
    if !is_hf_repo_id(&path_str) {
        return Ok(model_path.to_path_buf());
    }

    println!("Downloading {path_str} from HuggingFace Hub...");
    let fetcher = HfModelFetcher::new()
        .map_err(|e| Error::ConfigError(format!("HF fetcher initialization: {e}")))?;

    let artifact = fetcher
        .download_model(&path_str, FetchOptions::new())
        .map_err(|e| Error::ConfigError(format!("Model download failed: {e}")))?;

    println!("  Cached at: {}", artifact.path.display());
    Ok(artifact.path)
}

#[cfg(not(feature = "hub-publish"))]
fn resolve_model_path(model_path: &Path) -> Result<PathBuf> {
    use crate::config::schema::is_hf_repo_id;

    let path_str = model_path.to_string_lossy();
    if is_hf_repo_id(&path_str) {
        return Err(Error::ConfigError(format!(
            "HF model ID '{path_str}' requires the 'hub-publish' feature. \
             Rebuild with: cargo install entrenar --features hub-publish"
        )));
    }
    Ok(model_path.to_path_buf())
}

/// ALB-096: Load transformer model from APR or SafeTensors weights.
///
/// Tries APR first (sovereign format), then falls back to SafeTensors.
/// Returns `(model, checkpoint_step)` where checkpoint_step is extracted
/// from APR metadata or parsed from checkpoint filename.
fn load_transformer_model(
    model_path: &Path,
    config: &TransformerConfig,
    output_dir: &Path,
) -> Result<(Option<Transformer>, usize)> {
    // ALB-097: Check output_dir first for checkpoint resume (APR, then SafeTensors)
    if output_dir.is_dir() {
        // ENT-282: Check if checkpoint is a delta (NF4+QLoRA — no frozen base weights).
        // If delta, load base model from model_path first, then overlay delta tensors.
        if let Some(result) = try_load_apr_delta(output_dir, config, model_path) {
            return Ok(result);
        }
        // Try full APR checkpoint
        if let Some(result) = try_load_apr(output_dir, config) {
            return Ok(result);
        }
        // Try SafeTensors from output_dir (backward compat with pre-APR checkpoints)
        if let Some(result) = try_load_safetensors_dir(output_dir, config) {
            return Ok(result);
        }
    }

    if !model_path.exists() {
        println!("  Model path not found, using random initialization");
        return Ok((None, 0));
    }

    println!("Loading model weights from {}...", model_path.display());

    // ALB-117: When loading from model_path (initial weights, NOT resume from output_dir),
    // always return step=0. The checkpoint step from the source model is irrelevant —
    // we're starting fresh training with pre-trained weights, not resuming a training run.
    // Without this, loading model-step-14500.apr as initial weights would set step=14500,
    // causing immediate exit when max_steps < 14500 (loss=0.0, no training executed).

    // ALB-096: Try APR format from model_path (direct .apr file or HF download)
    if let Some((model, _source_step)) = try_load_apr(model_path, config) {
        return Ok((model, 0));
    }

    // Fallback: SafeTensors from model_path
    if let Some((model, _source_step)) = try_load_safetensors_dir(model_path, config) {
        return Ok((model, 0));
    }

    eprintln!("Warning: No loadable checkpoint found, using random initialization");
    Ok((None, 0))
}

/// Try loading SafeTensors checkpoint from a directory.
/// Returns `Some((model, step))` if successful, `None` to fall back.
fn try_load_safetensors_dir(
    dir: &Path,
    config: &TransformerConfig,
) -> Option<(Option<Transformer>, usize)> {
    let checkpoint_step = detect_checkpoint_step(dir);

    match load_safetensors_weights(dir, Architecture::Auto) {
        Ok(weights) => {
            println!("  Found {} weight tensors (SafeTensors)", weights.len());
            if let Some(transformer) = Transformer::from_params(config, &weights) {
                let embed = &transformer.embed_tokens.weight;
                let embed_data = embed.data();
                let embed_slice = embed_data.as_slice().unwrap_or(&[]);
                let (emin, emax, emean) = if embed_slice.is_empty() {
                    (0.0, 0.0, 0.0)
                } else {
                    let min = embed_slice.iter().copied().fold(f32::INFINITY, f32::min);
                    let max = embed_slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let mean = embed_slice.iter().sum::<f32>() / embed_slice.len() as f32;
                    (min, max, mean)
                };
                println!("✓ Loaded pre-trained weights successfully (SafeTensors)");
                println!("  embed_tokens stats: min={emin:.4e} max={emax:.4e} mean={emean:.4e}");
                return Some((Some(transformer), checkpoint_step));
            }
            None
        }
        Err(_) => None,
    }
}

/// ALB-096: Try to load a model from APR format.
///
/// Looks for `.apr` files: direct file, `model-best.apr`, or latest `model-step-*.apr`.
/// Returns `Some((model, step))` if successful, `None` to fall back to SafeTensors.
///
/// Public variant for use by `CudaTransformerTrainer::for_inference` (ALB-089).
pub fn try_load_apr_for_inference(
    model_path: &Path,
    config: &TransformerConfig,
) -> Option<(Option<Transformer>, usize)> {
    try_load_apr(model_path, config)
}

/// ENT-282: Load a delta checkpoint (NF4+QLoRA lazy save).
///
/// Delta checkpoints only contain trainable/updated weights (norms, embed, lm_head, LoRA)
/// and skip frozen NF4 base weights (~15 GB). On resume, base weights are loaded from the
/// original model_path first, then delta tensors are overlaid.
///
/// Returns None if the checkpoint is not a delta (falls through to full-checkpoint load).
fn try_load_apr_delta(
    output_dir: &Path,
    config: &TransformerConfig,
    base_model_path: &Path,
) -> Option<(Option<Transformer>, usize)> {
    use aprender::serialization::apr::AprReader;

    let apr_path = find_latest_apr_checkpoint(output_dir)?;
    let reader = AprReader::open(&apr_path).ok()?;

    // Only handle delta checkpoints
    let format = reader.get_metadata("format").and_then(|v| v.as_str().map(String::from))?;
    if format != "entrenar-delta-checkpoint" {
        return None; // Not a delta — fall through to full load
    }

    let checkpoint_step = reader
        .get_metadata("checkpoint_step")
        .and_then(|v| v.as_str())
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(0);

    println!(
        "  Delta checkpoint at step {checkpoint_step} (loading base from {})",
        base_model_path.display()
    );

    // Step 1: Load full base model from original pretrained weights
    let (base_model, _) = try_load_apr(base_model_path, config)
        .or_else(|| try_load_safetensors_dir(base_model_path, config))?;
    let mut transformer = base_model?;

    // Step 2: Overlay delta tensors (norms, embed, lm_head)
    let mut overlaid = 0usize;
    for desc in &reader.tensors {
        let name = &desc.name;
        if name.starts_with("__training__") || name.starts_with("lora.") {
            continue; // Handled separately by restore_lora_from_apr / load_optimizer_state_apr
        }
        if let Ok(data) = reader.read_tensor_as_f32(name) {
            let tensor = crate::Tensor::from_vec(data, false);
            if transformer.set_named_parameter(name, tensor) {
                overlaid += 1;
            }
        }
    }
    println!("  ✓ Delta: {overlaid} tensors overlaid on base model");

    Some((Some(transformer), checkpoint_step))
}

fn try_load_apr(
    model_path: &Path,
    config: &TransformerConfig,
) -> Option<(Option<Transformer>, usize)> {
    use aprender::serialization::apr::AprReader;
    use std::collections::HashMap;

    // Determine which APR file to load
    let apr_path =
        if model_path.is_file() && model_path.extension().and_then(|e| e.to_str()) == Some("apr") {
            model_path.to_path_buf()
        } else if model_path.is_dir() {
            find_latest_apr_checkpoint(model_path)?
        } else {
            return None;
        };

    let reader = match AprReader::open(&apr_path) {
        Ok(r) => r,
        Err(_) => return None,
    };

    // Extract checkpoint step from APR metadata
    let checkpoint_step = reader
        .get_metadata("checkpoint_step")
        .and_then(|v| v.as_str())
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or_else(|| {
            apr_path
                .file_name()
                .and_then(|n| n.to_str())
                .and_then(parse_checkpoint_step)
                .unwrap_or(0)
        });

    // Load weight tensors (skip __training__.* namespace)
    // Detect GGUF tensor names and map to HF convention (PMAT-489)
    let mut weights = HashMap::new();
    let is_gguf_names = reader.tensors.iter().any(|t| t.name == "token_embd.weight");
    if is_gguf_names {
        eprintln!("[PMAT-489] Detected GGUF tensor names in APR file, mapping to HF convention");
    }
    for desc in &reader.tensors {
        let tensor_name = &desc.name;
        if tensor_name.starts_with("__training__") {
            continue;
        }
        match reader.read_tensor_as_f32(tensor_name) {
            Ok(data) => {
                let mapped_name = if is_gguf_names {
                    use crate::transformer::weights::{Architecture, mapping::map_weight_name};
                    map_weight_name(tensor_name, Architecture::Gguf)
                } else {
                    tensor_name.clone()
                };
                weights.insert(mapped_name, crate::Tensor::from_vec(data, false));
            }
            Err(e) => {
                eprintln!("Warning: Failed to read APR tensor '{tensor_name}': {e}");
                return None;
            }
        }
    }

    println!("  Found {} weight tensors (APR)", weights.len());

    let transformer = Transformer::from_params(config, &weights)?;

    let embed = &transformer.embed_tokens.weight;
    let embed_data = embed.data();
    let embed_slice = embed_data.as_slice().unwrap_or(&[]);
    let (emin, emax, emean) = if embed_slice.is_empty() {
        (0.0, 0.0, 0.0)
    } else {
        let min = embed_slice.iter().copied().fold(f32::INFINITY, f32::min);
        let max = embed_slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mean = embed_slice.iter().sum::<f32>() / embed_slice.len() as f32;
        (min, max, mean)
    };
    println!("✓ Loaded pre-trained weights successfully (APR)");
    // ALB-117: Don't print "Resuming from step" here — the caller decides whether
    // this is a genuine resume (output_dir) or fresh training (model_path).
    // The caller at set_initial_step prints the resume message when appropriate.
    println!("  embed_tokens stats: min={emin:.4e} max={emax:.4e} mean={emean:.4e}");

    Some((Some(transformer), checkpoint_step))
}

/// Find the latest APR checkpoint in a directory.
///
/// Priority: latest `model-step-N.apr` by step number. Falls back to `model-best.apr`.
fn find_latest_apr_checkpoint(dir: &Path) -> Option<std::path::PathBuf> {
    let mut best_step = 0usize;
    let mut best_path = None;

    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if let Some(step) = parse_checkpoint_step(name) {
                    if step >= best_step {
                        best_step = step;
                        best_path = Some(path);
                    }
                }
            }
        }
    }

    if best_path.is_some() {
        return best_path;
    }

    // Fallback: model-best.apr, then model.apr (final checkpoint)
    let best = dir.join("model-best.apr");
    if best.exists() {
        return Some(best);
    }
    let model = dir.join("model.apr");
    if model.exists() {
        return Some(model);
    }

    None
}

/// Detect the step number from the latest checkpoint in a directory.
///
/// Checks both APR (`.apr`) and legacy SafeTensors (`.safetensors`) checkpoint files.
fn detect_checkpoint_step(model_path: &Path) -> usize {
    use crate::transformer::weights::parse_checkpoint_step_from_path;

    if model_path.is_file() {
        if let Some(name) = model_path.file_name().and_then(|n| n.to_str()) {
            if let Some(step) = parse_checkpoint_step(name) {
                return step;
            }
        }
        return parse_checkpoint_step_from_path(model_path).unwrap_or(0);
    }
    if !model_path.is_dir() {
        return 0;
    }
    // Check for model-step-*.{apr,safetensors} files
    let Ok(entries) = std::fs::read_dir(model_path) else { return 0 };
    let mut max_step = 0usize;
    for entry in entries.flatten() {
        if let Some(name) = entry.file_name().to_str() {
            if let Some(step) = parse_checkpoint_step(name) {
                max_step = max_step.max(step);
            }
        }
        if let Some(step) = parse_checkpoint_step_from_path(&entry.path()) {
            max_step = max_step.max(step);
        }
    }
    max_step
}

/// Apply architecture overrides to a `TransformerConfig`.
///
/// Only `Some` fields in the overrides replace the corresponding base config field.
fn apply_architecture_overrides(
    config: &mut TransformerConfig,
    overrides: &crate::config::ArchitectureOverrides,
) {
    if let Some(v) = overrides.hidden_size {
        config.hidden_size = v;
    }
    if let Some(v) = overrides.num_hidden_layers {
        config.num_hidden_layers = v;
    }
    if let Some(v) = overrides.num_attention_heads {
        config.num_attention_heads = v;
    }
    if let Some(v) = overrides.num_kv_heads {
        config.num_kv_heads = v;
    }
    if let Some(v) = overrides.intermediate_size {
        config.intermediate_size = v;
    }
    if let Some(v) = overrides.vocab_size {
        config.vocab_size = v;
    }
    if let Some(v) = overrides.max_position_embeddings {
        config.max_position_embeddings = v;
    }
    if let Some(v) = overrides.rms_norm_eps {
        config.rms_norm_eps = v;
    }
    if let Some(v) = overrides.rope_theta {
        config.rope_theta = v;
    }
    if let Some(v) = overrides.use_bias {
        config.use_bias = v;
    }
    if let Some(v) = overrides.head_dim {
        config.head_dim_override = Some(v);
    }
}

/// Build TransformerConfig from TrainSpec
///
/// Uses config file if specified, otherwise defaults to a small model.
/// Architecture overrides from the YAML manifest are applied on top.
fn build_transformer_config_from_spec(spec: &TrainSpec) -> Result<TransformerConfig> {
    // Check if config file is specified (explicit path or auto-detect from model dir)
    let config_path_resolved = spec.model.config.clone().or_else(|| {
        // Auto-detect config.json in model directory
        let model_config = spec.model.path.join("config.json");
        if model_config.exists() {
            Some(model_config.to_string_lossy().into_owned())
        } else {
            None
        }
    });

    let mut config = if let Some(config_path) = &config_path_resolved {
        let config_file = std::path::Path::new(config_path);
        if config_file.exists() {
            let config_content = std::fs::read_to_string(config_file)
                .map_err(|e| Error::ConfigError(format!("Failed to read model config: {e}")))?;

            if let Ok(hf_config) = serde_json::from_str::<serde_json::Value>(&config_content) {
                parse_hf_config(&hf_config)?
            } else {
                fallback_demo_config()
            }
        } else {
            fallback_demo_config()
        }
    } else if let Some(ref overrides) = spec.model.architecture {
        // No config file specified — try to build entirely from architecture overrides
        if let Some(cfg) = config_from_overrides(overrides) {
            cfg
        } else {
            fallback_demo_config()
        }
    } else {
        fallback_demo_config()
    };

    // Apply architecture overrides from YAML manifest (handles partial overrides on top of base)
    if let Some(ref overrides) = spec.model.architecture {
        apply_architecture_overrides(&mut config, overrides);
    }

    Ok(config)
}

/// Build a TransformerConfig directly from architecture overrides if all required fields are present.
/// Required: hidden_size, num_attention_heads, num_hidden_layers, vocab_size, intermediate_size.
fn config_from_overrides(
    overrides: &crate::config::ArchitectureOverrides,
) -> Option<TransformerConfig> {
    let hidden_size = overrides.hidden_size?;
    let num_attention_heads = overrides.num_attention_heads?;
    let num_hidden_layers = overrides.num_hidden_layers?;
    let vocab_size = overrides.vocab_size?;
    let intermediate_size = overrides.intermediate_size?;

    Some(TransformerConfig {
        hidden_size,
        num_attention_heads,
        num_kv_heads: overrides.num_kv_heads.unwrap_or(num_attention_heads),
        intermediate_size,
        num_hidden_layers,
        vocab_size,
        max_position_embeddings: overrides.max_position_embeddings.unwrap_or(2048),
        rms_norm_eps: overrides.rms_norm_eps.unwrap_or(1e-5),
        rope_theta: overrides.rope_theta.unwrap_or(10000.0),
        use_bias: overrides.use_bias.unwrap_or(false),
        head_dim_override: overrides.head_dim,
        architecture: ModelArchitecture::Decoder,
        hf_architecture: None,
        hf_model_type: None,
        tie_word_embeddings: false,
    })
}

/// R-05 (Meyer DbC): Explicit Qwen2-0.5B demo config — NOT a generic default.
/// This path is ONLY for testing without a model. Production callers must provide config.json.
fn fallback_demo_config() -> TransformerConfig {
    eprintln!("WARNING: No model config found — using Qwen2-0.5B demo config (NOT suitable for production training)");
    TransformerConfig {
        hidden_size: QWEN_HIDDEN_SIZE,
        num_attention_heads: QWEN_NUM_ATTENTION_HEADS,
        num_kv_heads: QWEN_NUM_KV_HEADS,
        intermediate_size: QWEN_INTERMEDIATE_SIZE,
        num_hidden_layers: QWEN_NUM_HIDDEN_LAYERS,
        vocab_size: QWEN_VOCAB_SIZE,
        max_position_embeddings: QWEN_MAX_POSITION_EMBEDDINGS,
        rms_norm_eps: 1e-6,
        rope_theta: QWEN_ROPE_THETA as f32,
        use_bias: false,
        head_dim_override: None,
        architecture: ModelArchitecture::Decoder,
        hf_architecture: None,
        hf_model_type: None,
        tie_word_embeddings: false,
    }
}

/// Parse HuggingFace config.json into `TransformerConfig`.
///
/// C-10/C-11 (Meyer DbC): Required fields (hidden_size, num_attention_heads,
/// num_hidden_layers, vocab_size, intermediate_size) must be present — no silent defaults.
/// R-04: Optional fields use generic defaults with warnings for likely-wrong values.
fn parse_hf_config(hf_config: &serde_json::Value) -> Result<TransformerConfig> {
    let hidden_size = hf_config["hidden_size"].as_u64().ok_or_else(|| {
        Error::ConfigError(
            "C-11: config.json missing 'hidden_size' — cannot train without model dimensions"
                .into(),
        )
    })? as usize;
    let num_attention_heads = hf_config["num_attention_heads"].as_u64().ok_or_else(|| {
        Error::ConfigError("C-11: config.json missing 'num_attention_heads'".into())
    })? as usize;
    let num_hidden_layers = hf_config["num_hidden_layers"]
        .as_u64()
        .ok_or_else(|| Error::ConfigError("C-11: config.json missing 'num_hidden_layers'".into()))?
        as usize;
    let vocab_size = hf_config["vocab_size"]
        .as_u64()
        .ok_or_else(|| Error::ConfigError(
            "C-10: config.json missing 'vocab_size' — training with wrong vocab corrupts embeddings".into()
        ))? as usize;
    let intermediate_size = hf_config["intermediate_size"]
        .as_u64()
        .ok_or_else(|| Error::ConfigError("C-11: config.json missing 'intermediate_size'".into()))?
        as usize;

    // R-04 (Meyer DbC): Optional fields with generic defaults.
    // num_kv_heads → num_attention_heads is the correct GQA→MHA fallback.
    // max_position_embeddings → 2048 is a conservative safe minimum.
    // rms_norm_eps → 1e-6 is the most common default.
    // rope_theta → 10000 is the LLaMA/Mistral standard (WRONG for Qwen at 1M).
    // use_bias → false is correct for most modern architectures.
    let num_kv_heads =
        hf_config["num_key_value_heads"].as_u64().unwrap_or(num_attention_heads as u64) as usize;

    let max_position_embeddings = match hf_config["max_position_embeddings"].as_u64() {
        Some(v) => v as usize,
        None => {
            eprintln!("Warning: config.json missing 'max_position_embeddings', defaulting to 2048");
            2048
        }
    };

    let rope_theta = match hf_config["rope_theta"].as_f64() {
        Some(v) => v as f32,
        None => {
            eprintln!(
                "Warning: config.json missing 'rope_theta', defaulting to 10000.0 \
                (Qwen models use 1000000.0 — check your config)"
            );
            10_000.0
        }
    };

    let rms_norm_eps = hf_config["rms_norm_eps"].as_f64().unwrap_or_else(|| {
        eprintln!(
            "Warning: config.json missing 'rms_norm_eps', defaulting to 1e-6 \
            (some models use 1e-5 or 1e-12 — check your config)"
        );
        1e-6
    }) as f32;
    let use_bias = hf_config["attention_bias"].as_bool().unwrap_or(false);
    let head_dim_override = hf_config["head_dim"].as_u64().map(|v| v as usize);

    // Detect encoder architectures from HuggingFace model_type
    let architecture = match hf_config["model_type"].as_str() {
        Some("bert" | "roberta" | "distilbert" | "albert" | "electra" | "deberta") => {
            ModelArchitecture::Encoder
        }
        _ => ModelArchitecture::Decoder,
    };

    // Preserve HuggingFace architecture metadata for checkpoint config.json (#259)
    let hf_architecture = hf_config["architectures"]
        .as_array()
        .and_then(|a| a.first())
        .and_then(|v| v.as_str())
        .map(String::from);
    let hf_model_type = hf_config["model_type"].as_str().map(String::from);
    let tie_word_embeddings = hf_config["tie_word_embeddings"].as_bool().unwrap_or(false);

    Ok(TransformerConfig {
        hidden_size,
        num_attention_heads,
        num_kv_heads,
        intermediate_size,
        num_hidden_layers,
        vocab_size,
        max_position_embeddings,
        rms_norm_eps,
        rope_theta,
        use_bias,
        head_dim_override,
        architecture,
        hf_architecture,
        hf_model_type,
        tie_word_embeddings,
    })
}

/// Load training data as LMBatches for transformer training
///
/// Supports:
/// 1. Pre-tokenized JSON with `input_ids` arrays
/// 2. Text JSON/JSONL with `text` or `content` fields (requires tokenizer)
/// 3. Demo mode fallback for testing
fn load_lm_batches(spec: &TrainSpec) -> Result<Vec<LMBatch>> {
    let batch_size = spec.data.batch_size;
    let seq_len = spec.data.seq_len.unwrap_or_else(|| {
        eprintln!("Warning: seq_len not specified, defaulting to 512 for LM batch loading");
        512
    });
    let tokenizer = load_tokenizer(spec)?;

    if let Some(result) = try_load_lm_from_file(spec, tokenizer.as_ref(), batch_size, seq_len) {
        return result;
    }

    eprintln!(
        "Warning: Training data not found at '{}', using demo LM batches",
        spec.data.train.display()
    );
    create_demo_lm_batches(batch_size, seq_len)
}

/// Attempt to load LM batches from the training data file or directory
fn try_load_lm_from_file(
    spec: &TrainSpec,
    tokenizer: Option<&HfTokenizer>,
    batch_size: usize,
    seq_len: usize,
) -> Option<Result<Vec<LMBatch>>> {
    if !spec.data.train.exists() {
        return None;
    }

    // Handle directory of Parquet shards (ALB-007)
    if spec.data.train.is_dir() {
        let tokenizer = tokenizer?;
        return Some(load_lm_batches_from_parquet(
            &spec.data.train,
            tokenizer,
            batch_size,
            seq_len,
            spec.data.input_column.as_deref().unwrap_or("text"),
        ));
    }

    let ext = spec.data.train.extension()?;

    if ext == "json" || ext == "jsonl" {
        let content = std::fs::read_to_string(&spec.data.train).ok()?;
        return Some(load_lm_batches_from_json(
            &content,
            tokenizer,
            batch_size,
            seq_len,
            spec.data.input_column.as_deref(),
        ));
    }

    if ext == "parquet" {
        let tokenizer = tokenizer?;
        return Some(load_lm_batches_from_parquet(
            &spec.data.train,
            tokenizer,
            batch_size,
            seq_len,
            spec.data.input_column.as_deref().unwrap_or("text"),
        ));
    }

    None
}

/// Load HfTokenizer from spec if tokenizer path is specified
fn load_tokenizer(spec: &TrainSpec) -> Result<Option<HfTokenizer>> {
    if let Some(ref tokenizer_path) = spec.data.tokenizer {
        if tokenizer_path.exists() {
            println!("  Loading tokenizer from: {}", tokenizer_path.display());
            let tokenizer = HfTokenizer::from_file(tokenizer_path)
                .map_err(|e| Error::ConfigError(format!("Failed to load tokenizer: {e}")))?;
            println!("  Tokenizer vocab size: {}", tokenizer.vocab_size());
            return Ok(Some(tokenizer));
        }
        eprintln!(
            "Warning: Tokenizer not found at '{}', using default Qwen2 tokenizer",
            tokenizer_path.display()
        );
    }

    // No tokenizer specified - use default for transformer mode
    println!("  Using default Qwen2 tokenizer");
    Ok(Some(HfTokenizer::qwen2()))
}

/// Extract text strings from a JSON array using the given column names
fn extract_texts_from_array(array: &[serde_json::Value], text_col: &str) -> Vec<String> {
    array
        .iter()
        .filter_map(|e| {
            e.get(text_col).or_else(|| e.get("content")).and_then(|v| v.as_str()).map(String::from)
        })
        .collect()
}

/// Try loading from a JSON array (either pre-tokenized or text)
fn try_load_from_array(
    array: &[serde_json::Value],
    tokenizer: Option<&HfTokenizer>,
    batch_size: usize,
    seq_len: usize,
    text_col: &str,
    label: &str,
) -> Option<Result<Vec<LMBatch>>> {
    // Check for pre-tokenized
    if array.first().and_then(|e| e.get("input_ids")).is_some() {
        return Some(load_pretokenized_json(array, batch_size, seq_len));
    }

    // Extract text and tokenize
    let tokenizer = tokenizer?;
    let texts = extract_texts_from_array(array, text_col);
    if texts.is_empty() {
        return None;
    }

    println!("  Loaded {} text examples from {label}, tokenizing...", texts.len());
    Some(tokenize_texts_to_batches(&texts, tokenizer, batch_size, seq_len))
}

/// Try loading from JSONL (newline-delimited JSON)
fn try_load_from_jsonl(
    content: &str,
    tokenizer: Option<&HfTokenizer>,
    batch_size: usize,
    seq_len: usize,
    text_col: &str,
) -> Option<Result<Vec<LMBatch>>> {
    let tokenizer = tokenizer?;
    let texts: Vec<String> = content
        .lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|line| {
            serde_json::from_str::<serde_json::Value>(line).ok().and_then(|obj| {
                obj.get(text_col)
                    .or_else(|| obj.get("content"))
                    .and_then(|v| v.as_str())
                    .map(String::from)
            })
        })
        .collect();

    if texts.is_empty() {
        return None;
    }

    println!("  Loaded {} text examples from JSONL, tokenizing...", texts.len());
    Some(tokenize_texts_to_batches(&texts, tokenizer, batch_size, seq_len))
}

/// Try to load LM batches from a parsed JSON value (object or array)
fn try_load_from_json_value(
    data: &serde_json::Value,
    tokenizer: Option<&HfTokenizer>,
    batch_size: usize,
    seq_len: usize,
    text_col: &str,
) -> Option<Result<Vec<LMBatch>>> {
    // Try {"examples": [...]} format
    if let Some(examples) = data.get("examples").and_then(|e| e.as_array()) {
        if let Some(result) =
            try_load_from_array(examples, tokenizer, batch_size, seq_len, text_col, "JSON")
        {
            return Some(result);
        }
    }

    // Try top-level array format
    if let Some(array) = data.as_array() {
        if let Some(result) =
            try_load_from_array(array, tokenizer, batch_size, seq_len, text_col, "JSON array")
        {
            return Some(result);
        }
    }

    None
}

/// Load LM batches from JSON content
///
/// Supports formats:
/// - Pre-tokenized: `{"examples": [{"input_ids": [...]}]}`
/// - Text data: `{"examples": [{"text": "..."}]}` or `[{"text": "..."}]`
/// - JSONL: `{"text": "..."}\n{"text": "..."}`
fn load_lm_batches_from_json(
    content: &str,
    tokenizer: Option<&HfTokenizer>,
    batch_size: usize,
    seq_len: usize,
    input_column: Option<&str>,
) -> Result<Vec<LMBatch>> {
    let text_col = input_column.unwrap_or("text");

    // Try parsing as single JSON object or array
    if let Ok(data) = serde_json::from_str::<serde_json::Value>(content) {
        if let Some(result) =
            try_load_from_json_value(&data, tokenizer, batch_size, seq_len, text_col)
        {
            return result;
        }
    }

    // Try JSONL format
    if let Some(result) = try_load_from_jsonl(content, tokenizer, batch_size, seq_len, text_col) {
        return result;
    }

    // Fallback to demo batches
    eprintln!("Warning: Could not parse training data, using demo LM batches");
    create_demo_lm_batches(batch_size, seq_len)
}

/// Load pre-tokenized sequences from JSON
fn load_pretokenized_json(
    examples: &[serde_json::Value],
    batch_size: usize,
    seq_len: usize,
) -> Result<Vec<LMBatch>> {
    let mut all_sequences: Vec<Vec<u32>> = Vec::new();

    for example in examples {
        if let Some(tokens) = example.get("input_ids").and_then(|t| t.as_array()) {
            let seq: Vec<u32> =
                tokens.iter().filter_map(|t| t.as_u64().map(|v| v as u32)).collect();
            if !seq.is_empty() {
                all_sequences.push(seq);
            }
        }
    }

    if !all_sequences.is_empty() {
        println!("  Loaded {} pre-tokenized sequences from JSON", all_sequences.len());
        return create_lm_batches_from_sequences(&all_sequences, batch_size, seq_len);
    }

    eprintln!("Warning: No valid sequences found in JSON");
    create_demo_lm_batches(batch_size, seq_len)
}

/// Tokenize texts and create LM batches
fn tokenize_texts_to_batches(
    texts: &[String],
    tokenizer: &HfTokenizer,
    batch_size: usize,
    seq_len: usize,
) -> Result<Vec<LMBatch>> {
    let sequences: Vec<Vec<u32>> = texts
        .iter()
        .map(|text| {
            let mut tokens = tokenizer.encode_with_special(text);
            tokens.truncate(seq_len);
            tokens
        })
        .filter(|seq| seq.len() > 1) // Need at least 2 tokens for causal LM
        .collect();

    if sequences.is_empty() {
        eprintln!("Warning: No valid sequences after tokenization");
        return create_demo_lm_batches(batch_size, seq_len);
    }

    println!("  Tokenized {} sequences", sequences.len());
    create_lm_batches_from_sequences(&sequences, batch_size, seq_len)
}

/// Load LM batches from Parquet file with text or pre-tokenized columns (ALB-007)
///
/// Supports two modes:
/// 1. **Text column** (Utf8): reads text, tokenizes with HfTokenizer, creates LMBatch
/// 2. **Pre-tokenized column** (List<UInt32/Int32>): reads token ID lists directly
///
/// Also handles directory paths containing multiple .parquet shard files.
#[cfg(all(not(target_arch = "wasm32"), feature = "parquet"))]
fn load_lm_batches_from_parquet(
    path: &std::path::Path,
    tokenizer: &HfTokenizer,
    batch_size: usize,
    seq_len: usize,
    text_column: &str,
) -> Result<Vec<LMBatch>> {
    use alimentar::{ArrowDataset, Dataset};

    // Handle directory of parquet shards
    if path.is_dir() {
        return load_lm_batches_from_parquet_dir(path, tokenizer, batch_size, seq_len, text_column);
    }

    println!("  Loading Parquet LM data: {}", path.display());

    // ALB-099: Scope ArrowDataset so it drops before LMBatch construction,
    // avoiding triple materialization (Arrow + Vec<Vec<u32>> + LMBatch).
    let (sequences, texts) = {
        let dataset = ArrowDataset::from_parquet(path).map_err(|e| {
            Error::ConfigError(format!("Failed to load parquet {}: {e}", path.display()))
        })?;

        println!("  Loaded {} rows from Parquet", dataset.len());

        let schema = dataset.schema();
        let column_names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();

        // Try pre-tokenized first (input_ids column with integer list type)
        let seqs = try_extract_pretokenized(&dataset, &column_names);
        let txts = if seqs.is_none() {
            Some(extract_text_column(&dataset, text_column, &column_names)?)
        } else {
            None
        };
        (seqs, txts)
        // dataset dropped here — frees Arrow RecordBatch memory
    };

    if let Some(sequences) = sequences {
        println!("  Found pre-tokenized column, loaded {} sequences", sequences.len());
        return create_lm_batches_from_sequences(&sequences, batch_size, seq_len);
    }

    if let Some(texts) = texts {
        println!("  Extracted {} text rows, tokenizing...", texts.len());
        return tokenize_texts_to_batches(&texts, tokenizer, batch_size, seq_len);
    }

    Err(Error::ConfigError("No pre-tokenized or text column found".into()))
}

/// Load LM batches from a directory of Parquet shard files (ALB-007, ALB-101)
///
/// Uses `StreamingParquetLoader` to process shards one at a time, keeping only
/// one shard's worth of raw Arrow data in memory at any point. The resulting
/// `LMBatch`es are still accumulated into a single Vec (full streaming during
/// training is a future enhancement).
#[cfg(all(not(target_arch = "wasm32"), feature = "parquet"))]
fn load_lm_batches_from_parquet_dir(
    dir: &std::path::Path,
    _tokenizer: &HfTokenizer,
    batch_size: usize,
    seq_len: usize,
    _text_column: &str,
) -> Result<Vec<LMBatch>> {
    use crate::config::train::batches::streaming::{ShardConfig, StreamingParquetLoader};

    let mut loader = StreamingParquetLoader::new(dir, ShardConfig::single(), batch_size, seq_len)
        .map_err(|e| Error::ConfigError(e))?;

    let total_shards = loader.total_files();
    println!("  Streaming {} Parquet shard(s) from {} (ALB-101)", total_shards, dir.display());

    let mut all_batches = Vec::new();
    let mut shard_idx = 0usize;

    while let Some(shard_batches) = loader.next_batches().map_err(|e| Error::ConfigError(e))? {
        shard_idx += 1;
        let n = shard_batches.len();
        all_batches.extend(shard_batches);
        // shard Arrow data already dropped inside next_batches()
        println!(
            "    shard {}/{}: {} batches (cumulative: {})",
            shard_idx,
            total_shards,
            n,
            all_batches.len()
        );
    }

    // entrenar#315: Shuffle all batches across shards to prevent overfitting.
    // Without this, the model sees all sequences from shard 1 before shard 2,
    // causing memorization of shard 1 by step 2K (v22: val_ppl regressed
    // from 9.44 to 118 between step 2K and 3K).
    //
    // PyTorch DataLoader with shuffle=True provides this by default.
    // We shuffle deterministically using the config seed for reproducibility.
    {
        let seed = 123u64; // TODO: propagate config seed
        let mut rng_state = seed;
        // Fisher-Yates shuffle with simple LCG
        for i in (1..all_batches.len()).rev() {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let j = (rng_state >> 33) as usize % (i + 1);
            all_batches.swap(i, j);
        }
    }

    println!("  Total: {} batches from {} shards (shuffled)", all_batches.len(), total_shards);
    Ok(all_batches)
}

/// Try to extract pre-tokenized sequences from a Parquet dataset (ALB-007)
///
/// Looks for columns named `input_ids` or `token_ids` containing integer arrays.
#[cfg(all(not(target_arch = "wasm32"), feature = "parquet"))]
fn try_extract_pretokenized(
    dataset: &alimentar::ArrowDataset,
    column_names: &[&str],
) -> Option<Vec<Vec<u32>>> {
    use alimentar::Dataset;

    let token_col =
        column_names.iter().find(|&&n| n == "input_ids" || n == "token_ids").copied()?;

    let schema = dataset.schema();
    let col_idx = schema.index_of(token_col).ok()?;

    // ALB-099: Pre-allocate with known row count
    let mut all_sequences = Vec::with_capacity(dataset.len());

    for batch in dataset.iter() {
        let col = batch.column(col_idx);
        extract_sequences_from_column(col, &mut all_sequences);
    }

    if all_sequences.is_empty() {
        None
    } else {
        Some(all_sequences)
    }
}

/// Extract token sequences from a single Arrow column (List or flat integer types)
#[cfg(all(not(target_arch = "wasm32"), feature = "parquet"))]
fn extract_sequences_from_column(col: &arrow::array::ArrayRef, sequences: &mut Vec<Vec<u32>>) {
    use arrow::array::{Array, ListArray};

    if let Some(list_arr) = col.as_any().downcast_ref::<ListArray>() {
        for i in 0..list_arr.len() {
            if list_arr.is_null(i) {
                continue;
            }
            let values = list_arr.value(i);
            let seq = extract_u32_from_array(&values);
            if !seq.is_empty() {
                sequences.push(seq);
            }
        }
    } else {
        // Flat integer column: treat entire column as one sequence
        let seq = extract_u32_from_array(col.as_ref());
        if !seq.is_empty() {
            sequences.push(seq);
        }
    }
}

/// Extract u32 token IDs from an Arrow array (inner values of a ListArray)
#[cfg(all(not(target_arch = "wasm32"), feature = "parquet"))]
fn extract_u32_from_array(array: &dyn arrow::array::Array) -> Vec<u32> {
    use arrow::array::{Int32Array, Int64Array, UInt32Array};

    if let Some(arr) = array.as_any().downcast_ref::<UInt32Array>() {
        arr.values().to_vec()
    } else if let Some(arr) = array.as_any().downcast_ref::<Int32Array>() {
        arr.values().iter().map(|&v| v as u32).collect()
    } else if let Some(arr) = array.as_any().downcast_ref::<Int64Array>() {
        arr.values().iter().map(|&v| v as u32).collect()
    } else {
        Vec::new()
    }
}

/// Resolve text column name from available columns (ALB-007)
///
/// Tries the specified name first, then common alternatives: text, content, code.
#[cfg(all(not(target_arch = "wasm32"), feature = "parquet"))]
fn resolve_text_column_name(text_column: &str, column_names: &[&str]) -> Result<String> {
    if column_names.contains(&text_column) {
        return Ok(text_column.to_string());
    }
    for &fallback in &["text", "content", "code"] {
        if column_names.contains(&fallback) {
            return Ok(fallback.to_string());
        }
    }
    Err(Error::ConfigError(format!(
        "No text column found in Parquet (tried '{text_column}', 'text', 'content', 'code'). Available: {column_names:?}"
    )))
}

/// Extract text strings from a Parquet dataset column (ALB-007)
#[cfg(all(not(target_arch = "wasm32"), feature = "parquet"))]
fn extract_text_column(
    dataset: &alimentar::ArrowDataset,
    text_column: &str,
    column_names: &[&str],
) -> Result<Vec<String>> {
    use alimentar::Dataset;

    let col_name = resolve_text_column_name(text_column, column_names)?;

    let schema = dataset.schema();
    let col_idx = schema
        .index_of(&col_name)
        .map_err(|e| Error::ConfigError(format!("Column '{col_name}' not found: {e}")))?;

    let mut texts = Vec::new();
    for batch in dataset.iter() {
        let col = batch.column(col_idx);
        extract_strings_from_array(col, &col_name, &mut texts)?;
    }
    Ok(texts)
}

/// Extract non-empty strings from a StringArray column
#[cfg(all(not(target_arch = "wasm32"), feature = "parquet"))]
fn extract_strings_from_array(
    col: &arrow::array::ArrayRef,
    col_name: &str,
    texts: &mut Vec<String>,
) -> Result<()> {
    use arrow::array::{Array, StringArray};

    let str_arr = col.as_any().downcast_ref::<StringArray>().ok_or_else(|| {
        Error::ConfigError(format!(
            "Column '{col_name}' is not a string type (found {:?})",
            col.data_type()
        ))
    })?;

    for i in 0..str_arr.len() {
        if !str_arr.is_null(i) {
            let text = str_arr.value(i);
            if !text.is_empty() {
                texts.push(text.to_string());
            }
        }
    }
    Ok(())
}

/// Fallback: Parquet loading without alimentar feature
#[cfg(not(all(not(target_arch = "wasm32"), feature = "parquet")))]
fn load_lm_batches_from_parquet(
    path: &std::path::Path,
    _tokenizer: &HfTokenizer,
    batch_size: usize,
    seq_len: usize,
    text_column: &str,
) -> Result<Vec<LMBatch>> {
    if !path.exists() {
        return Err(Error::Io(format!("Parquet path does not exist: {}", path.display())));
    }
    eprintln!(
        "Warning: Parquet LM loading requires the 'parquet' feature. \
         Build with: cargo build --features parquet"
    );
    eprintln!(
        "  Alternatively, convert to JSONL: alimentar export {} -o train.jsonl --text-column {}",
        path.display(),
        text_column
    );
    create_demo_lm_batches(batch_size, seq_len)
}

/// Create LMBatches from tokenized sequences
fn create_lm_batches_from_sequences(
    sequences: &[Vec<u32>],
    batch_size: usize,
    _seq_len: usize,
) -> Result<Vec<LMBatch>> {
    // ALB-099: Pre-allocate with known batch count
    let num_batches = sequences.len().div_ceil(batch_size);
    let mut batches = Vec::with_capacity(num_batches);
    let pad_id = 0u32; // Standard pad token
    let eos_id = 2u32; // Standard EOS token

    for chunk in sequences.chunks(batch_size) {
        let batch = LMBatch::from_sequences(chunk, pad_id, eos_id);
        batches.push(batch);
    }

    Ok(batches)
}

/// Create demo LM batches for testing
fn create_demo_lm_batches(batch_size: usize, seq_len: usize) -> Result<Vec<LMBatch>> {
    let mut batches = Vec::new();

    // Create a few demo sequences with synthetic tokens
    // Simulating simple patterns like: [1, 2, 3, 4, ...] with slight variations
    for batch_idx in 0..4 {
        let mut sequences = Vec::new();
        for item in 0..batch_size {
            let offset = (batch_idx * batch_size + item) as u32;
            // Create a sequence with incrementing tokens
            let seq: Vec<u32> = (0..seq_len.min(64))
                .map(|i| (offset + i as u32) % 1000 + 1) // Keep in reasonable token range
                .collect();
            sequences.push(seq);
        }

        let batch = LMBatch::from_sequences(&sequences, 0, 2);
        batches.push(batch);
    }

    Ok(batches)
}

/// Detect whether YAML content is in the new manifest format.
///
/// Returns true if the content contains an `entrenar:` key at the start of a line,
/// which is the discriminating field in the manifest schema.
fn is_manifest_format(yaml: &str) -> bool {
    yaml.lines().any(|line| line.starts_with("entrenar:") || line.starts_with("entrenar :"))
}

/// Load training spec from YAML file (without running training)
///
/// Auto-detects format:
/// - If the YAML contains `entrenar:`, it's parsed as a `TrainingManifest` and
///   converted to `TrainSpec` via the bridge converter.
/// - Otherwise, it's parsed directly as `TrainSpec` (legacy format).
pub fn load_config<P: AsRef<Path>>(config_path: P) -> Result<TrainSpec> {
    let yaml_content = fs::read_to_string(config_path.as_ref()).map_err(|e| {
        Error::ConfigError(format!(
            "Failed to read config file {}: {}",
            config_path.as_ref().display(),
            e
        ))
    })?;

    if is_manifest_format(&yaml_content) {
        // New declarative manifest format
        let manifest: yaml_mode::TrainingManifest = serde_yaml::from_str(&yaml_content)
            .map_err(|e| Error::ConfigError(format!("Failed to parse manifest YAML: {e}")))?;

        yaml_mode::validate_manifest(&manifest)
            .map_err(|e| Error::ConfigError(format!("Invalid manifest: {e}")))?;

        let bridge_result = yaml_mode::manifest_to_spec(&manifest)
            .map_err(|e| Error::ConfigError(format!("Manifest conversion failed: {e}")))?;

        for warning in &bridge_result.warnings {
            eprintln!("Warning: {warning}");
        }

        validate_config(&bridge_result.spec)
            .map_err(|e| Error::ConfigError(format!("Invalid config after conversion: {e}")))?;

        Ok(bridge_result.spec)
    } else {
        // Legacy TrainSpec format
        let spec: TrainSpec = serde_yaml::from_str(&yaml_content)
            .map_err(|e| Error::ConfigError(format!("Failed to parse YAML config: {e}")))?;

        validate_config(&spec).map_err(|e| Error::ConfigError(format!("Invalid config: {e}")))?;

        Ok(spec)
    }
}

