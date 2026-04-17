
/// Result from inference
#[derive(Debug, Clone)]
pub struct InferenceResult {
    /// Generated text (decoded from tokens)
    pub text: String,
    /// All tokens (input + generated)
    pub tokens: Vec<u32>,
    /// Number of input tokens
    pub input_token_count: usize,
    /// Number of generated tokens
    pub generated_token_count: usize,
    /// Inference time in milliseconds
    pub inference_ms: f64,
    /// Tokens per second
    pub tok_per_sec: f64,
    /// Model load time in milliseconds
    pub load_ms: f64,
    /// Model format that was loaded
    pub format: String,
    /// Whether GPU was used
    pub used_gpu: bool,
}

// ============================================================================
// Security - Path Validation (F-SEC-222)
// ============================================================================

/// Valid model file extensions
const VALID_MODEL_EXTENSIONS: &[&str] = &["gguf", "safetensors", "apr", "bin", "json"];

/// Validate that a path is a valid model file path.
///
/// # Security (F-SEC-222)
///
/// This prevents path traversal attacks where an attacker could trick the
/// tool into reading arbitrary files (e.g., `/etc/passwd`, `~/.ssh/id_rsa`).
///
/// ## Validation Rules
///
/// 1. Path must have a valid model extension (.gguf, .safetensors, .apr, .bin)
/// 2. Path must not contain path traversal sequences (`../`)
/// 3. Path must be a regular file (not a directory, symlink to directory, etc.)
///
/// # Errors
///
/// Returns error if:
/// - Path has invalid or missing extension
/// - Path contains traversal sequences
/// - Path doesn't exist or isn't a file
pub(crate) fn validate_model_path(path: &std::path::Path) -> Result<()> {
    // Check for path traversal sequences
    let path_str = path.to_string_lossy();
    if path_str.contains("..") {
        return Err(RealizarError::SecurityError {
            reason: format!(
                "Path traversal detected: '{}'. Use absolute paths or paths without '..'",
                path_str
            ),
        });
    }

    // Check file extension
    let extension = path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_lowercase)
        .unwrap_or_default();

    if !VALID_MODEL_EXTENSIONS.contains(&extension.as_str()) {
        return Err(RealizarError::SecurityError {
            reason: format!(
                "Invalid model file extension: '.{}'. Expected one of: {}",
                extension,
                VALID_MODEL_EXTENSIONS.join(", ")
            ),
        });
    }

    // Check that path exists and is a file
    if !path.exists() {
        return Err(RealizarError::IoError {
            message: format!("File not found: {}", path.display()),
        });
    }

    if !path.is_file() {
        return Err(RealizarError::SecurityError {
            reason: format!("Path is not a regular file: {}", path.display()),
        });
    }

    Ok(())
}

/// Run inference on a model
///
/// This is the main entry point for inference. It handles:
/// - Model format detection (GGUF, APR, SafeTensors)
/// - Tokenization (using embedded tokenizer for GGUF)
/// - Generation with configurable sampling
/// - GPU acceleration when available
/// - Inference tracing (APR-TRACE-001)
///
/// # Errors
///
/// Returns error if:
/// - Model file cannot be read
/// - Model format is unsupported
/// - Generation fails
pub fn run_inference(config: &InferenceConfig) -> Result<InferenceResult> {
    // PMAT-COV-95: Mock backend for testing without disk I/O
    if config.use_mock_backend {
        return run_mock_inference(config);
    }

    // GH-213: Detect sharded SafeTensors index.json BEFORE reading the file.
    // The index.json is a small JSON file (~15KB) that maps tensor names to shard files.
    // We detect it by suffix to avoid reading it as binary model data.
    let path_str = config.model_path.to_string_lossy();
    if path_str.ends_with(".safetensors.index.json") {
        // Validate path (F-SEC-222) - json extension is now allowed
        validate_model_path(&config.model_path)?;

        let format = ModelFormat::SafeTensors;
        let prepared = prepare_tokens(config, &format)?;
        return run_sharded_safetensors_inference(config, &prepared);
    }

    // Validate path to prevent traversal attacks (F-SEC-222)
    validate_model_path(&config.model_path)?;

    // ALB-099: Read only 8 bytes for format detection (was reading entire file)
    let magic = {
        use std::io::Read;
        let mut file = std::fs::File::open(&config.model_path).map_err(|e| RealizarError::IoError {
            message: format!("Failed to read model: {}", e),
        })?;
        let mut buf = [0u8; 8];
        file.read_exact(&mut buf).map_err(|e| {
            if e.kind() == std::io::ErrorKind::UnexpectedEof {
                RealizarError::FormatError {
                    reason: "File too small for format detection".to_string(),
                }
            } else {
                RealizarError::IoError {
                    message: format!("Failed to read model header: {}", e),
                }
            }
        })?;
        buf
    };

    // Detect format
    let format = detect_format(&magic).map_err(|e| RealizarError::FormatError {
        reason: format!("Format detection failed: {}", e),
    })?;

    // PMAT-236: Prepare tokens with chat template BEFORE format dispatch.
    // This is compile-time enforced - format-specific functions accept
    // PreparedTokens (private inner data) which can ONLY be created here.
    let prepared = prepare_tokens(config, &format)?;

    match format {
        ModelFormat::Gguf => run_gguf_inference(config, &prepared),
        ModelFormat::Apr => run_apr_inference(config, &prepared),
        ModelFormat::SafeTensors => run_safetensors_inference(config, &prepared),
    }
}

/// Run GGUF model inference
///
/// PMAT-236: Accepts `PreparedTokens` (compile-time enforced chat template).
fn run_gguf_inference(
    config: &InferenceConfig,
    prepared: &PreparedTokens,
) -> Result<InferenceResult> {
    use crate::gguf::{MappedGGUFModel, OwnedQuantizedModel, QuantizedGenerateConfig};

    if config.verbose {
        eprintln!("Loading model: {}", config.model_path.display());
    }

    let load_start = Instant::now();
    let mapped = MappedGGUFModel::from_path(&config.model_path)?;
    prefault_mmap(mapped.data());
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;
    let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;

    // PMAT-109: Architecture from GGUF metadata (not filename)
    let gguf_arch = mapped.model.architecture().unwrap_or("transformer");

    if config.verbose {
        print_gguf_verbose_info(gguf_arch, &model, load_ms);
    }

    // PMAT-236: Use PreparedTokens (chat template already applied by prepare_tokens)
    let input_tokens = prepared.tokens().to_vec();
    let input_token_count = prepared.input_count();
    let model_config = model.config.clone();

    // GH-373: Merge model EOS + caller stop tokens
    let mut stop_tokens: Vec<u32> = model_config.eos_token_id.into_iter().collect();
    for &t in &config.stop_tokens {
        if !stop_tokens.contains(&t) {
            stop_tokens.push(t);
        }
    }

    let gen_config = QuantizedGenerateConfig {
        max_tokens: config.max_tokens,
        temperature: config.temperature,
        top_k: config.top_k,
        stop_tokens,
        trace: config.trace,
            ..Default::default()
    };

    let infer_start = Instant::now();
    let (tokens, used_gpu) = run_gguf_generate(model, &input_tokens, &gen_config, config)?;
    let inference_ms = infer_start.elapsed().as_secs_f64() * 1000.0;

    let generated_tokens = &tokens[input_token_count..];
    let raw_text = mapped.model.decode(generated_tokens);
    if config.verbose {
        eprintln!("[DEBUG] input_count={}, total_tokens={}, generated_count={}", input_token_count, tokens.len(), generated_tokens.len());
        eprintln!("[DEBUG] generated token ids: {:?}", &generated_tokens[..generated_tokens.len().min(20)]);
        eprintln!("[DEBUG] raw decoded: {:?}", &raw_text[..raw_text.len().min(200)]);
    }
    let text = clean_model_output(&raw_text);
    let generated_token_count = generated_tokens.len();
    let tps = tok_per_sec(generated_token_count, inference_ms);

    write_gguf_trace(
        config,
        &model_config,
        input_token_count,
        generated_token_count,
        load_ms,
        inference_ms,
        tps,
        used_gpu,
    );

    Ok(InferenceResult {
        text,
        tokens,
        input_token_count,
        generated_token_count,
        inference_ms,
        tok_per_sec: tps,
        load_ms,
        format: "GGUF".to_string(),
        used_gpu,
    })
}

/// Print verbose model info for GGUF inference
fn print_gguf_verbose_info(
    gguf_arch: &str,
    model: &crate::gguf::OwnedQuantizedModel,
    load_ms: f64,
) {
    let arch = match gguf_arch.to_lowercase().as_str() {
        "qwen2" | "qwen" => "Qwen2",
        "llama" => "LLaMA",
        "mistral" => "Mistral",
        "phi" | "phi3" => "Phi",
        _ => "Transformer",
    };
    let quant_type = qtype_to_dtype_str(model.lm_head_weight.qtype);
    let thread_count = rayon::current_num_threads();
    eprintln!(
        "Architecture: {} [GGUF: {}] ({} layers, vocab_size={})",
        arch, gguf_arch, model.config.num_layers, model.config.vocab_size
    );
    eprintln!(
        "Config: hidden_size={}, context_length={}, quant={}, threads={}",
        model.config.hidden_dim, model.config.context_length, quant_type, thread_count
    );
    eprintln!("Model loaded in {:.1}ms", load_ms);
}

/// Write GGUF trace output if requested (PMAT-SHOWCASE-METHODOLOGY-001)
fn write_gguf_trace(
    config: &InferenceConfig,
    model_config: &crate::gguf::GGUFConfig,
    input_token_count: usize,
    generated_token_count: usize,
    load_ms: f64,
    inference_ms: f64,
    tps: f64,
    used_gpu: bool,
) {
    let trace_path = match config.trace_output {
        Some(ref p) => p,
        None => return,
    };
    let trace_json = format!(
        r#"{{
  "version": "1.0",
  "timestamp": "{}",
  "model": {{
    "path": "{}",
    "format": "GGUF",
    "num_layers": {},
    "hidden_dim": {},
    "vocab_size": {},
    "num_heads": {}
  }},
  "inference": {{
    "input_tokens": {},
    "generated_tokens": {},
    "load_ms": {:.2},
    "inference_ms": {:.2},
    "tok_per_sec": {:.2},
    "used_gpu": {}
  }},
  "events": []
}}
"#,
        chrono::Utc::now().to_rfc3339(),
        config.model_path.display(),
        model_config.num_layers,
        model_config.hidden_dim,
        model_config.vocab_size,
        model_config.num_heads,
        input_token_count,
        generated_token_count,
        load_ms,
        inference_ms,
        tps,
        used_gpu
    );
    if let Err(e) = std::fs::write(trace_path, trace_json) {
        eprintln!(
            "Warning: Failed to write trace output to {}: {}",
            trace_path.display(),
            e
        );
    }
}

/// Check if a quantization type is legacy (Q4_0, Q4_1, Q5_0, Q5_1)
/// GPU only supports Q4_K/Q5_K/Q6_K; legacy types produce garbage on GPU.
#[inline]
fn is_legacy_gguf_quant(qtype: u32) -> bool {
    matches!(qtype, 2 | 3 | 6 | 7)
}

/// Check if model uses any legacy quantization types
fn model_has_legacy_quant(model: &crate::gguf::OwnedQuantizedModel) -> bool {
    is_legacy_gguf_quant(model.lm_head_weight.qtype)
        || model.layers.iter().any(|l| {
            is_legacy_gguf_quant(l.ffn_down_weight.qtype)
                || is_legacy_gguf_quant(l.ffn_up_weight.qtype)
                || is_legacy_gguf_quant(l.attn_output_weight.qtype)
        })
}

/// Log CPU backend selection reason
#[inline]
fn log_cpu_backend(verbose: bool, is_legacy: bool) {
    if !verbose {
        return;
    }
    if is_legacy {
        eprintln!("Backend: CPU (Q4_0 format - GPU Q4_K kernels incompatible)");
    } else {
        eprintln!("Backend: CPU (SIMD-accelerated)");
    }
}

/// F2-FIX: Validate GPU output by comparing first predicted token with CPU.
///
/// Uses a single BOS token (not the full prompt) to test kernel correctness.
/// The Q6K kernel bug is dimension-dependent, not prompt-dependent, so a
/// single-token probe is sufficient and avoids O(n) CPU prefill overhead.
///
/// Uses the CUDA model's inner model reference to avoid requiring a separate model clone.
/// Skip with SKIP_PARITY_GATE=1 (same env var as the cosine parity gate).
#[cfg(feature = "cuda")]
fn validate_gpu_first_token(
    cuda_model: &mut crate::gguf::OwnedQuantizedModelCuda,
    gen_config: &crate::gguf::QuantizedGenerateConfig,
) -> bool {
    use crate::gguf::OwnedQuantizedKVCache;

    // SKIP_PARITY_GATE=1 bypasses both this F2 check and the cosine parity gate.
    // Used for forward-compatible GPUs (e.g., Blackwell sm_121) where minor FP
    // differences cause argmax disagreement but inference quality is unaffected.
    if std::env::var("SKIP_PARITY_GATE")
        .map(|v| v == "1")
        .unwrap_or(false)
    {
        return true;
    }

    let model = cuda_model.model();

    // BOS token flows from GGUF metadata → GGUFConfig → here.
    // GGUFConfig::from_gguf() applies architecture-default fallback for weights-only GGUFs.
    // If BOS is STILL unknown (e.g., phi architecture), skip validation.
    let bos_id = match model.config.bos_token_id {
        Some(id) => id,
        None => {
            eprintln!("[F2-VALIDATION] BOS token unknown — skipping GPU validation");
            return true;
        },
    };
    let probe_token: &[u32] = &[bos_id];

    let kv_dim = model.config.num_kv_heads * (model.config.hidden_dim / model.config.num_heads);
    let num_layers = model.config.num_layers;

    // CPU forward pass for reference
    let mut cpu_cache = OwnedQuantizedKVCache::new(num_layers, kv_dim, 2);
    let cpu_logits = match model.forward_single_with_cache(probe_token[0], &mut cpu_cache, 0) {
        Ok(logits) => logits,
        Err(_) => return true, // CPU forward failed — can't validate, assume GPU is fine
    };

    let cpu_argmax = cpu_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0, |(i, _)| i as u32);

    // GPU: generate 1 token from same BOS probe
    let gpu_first_config = crate::gguf::QuantizedGenerateConfig {
        max_tokens: 1,
        temperature: 0.0,
        top_k: 1,
        ..gen_config.clone()
    };
    match cuda_model.generate_gpu_resident(probe_token, &gpu_first_config) {
        Ok(gpu_tokens) if gpu_tokens.len() > 1 => {
            let gpu_first = gpu_tokens[1];
            if gpu_first == cpu_argmax {
                true
            } else {
                eprintln!(
                    "[F2-VALIDATION] GPU token {} != CPU token {} for BOS probe — falling back to CPU",
                    gpu_first, cpu_argmax
                );
                false
            }
        },
        Ok(_) => true,
        Err(_) => false,
    }
}
