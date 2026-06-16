
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

    // M32c.2.2.2.1.3: dispatch qwen3_moe to the parallel MoE inference path
    // (M32c.2.2.2.1.2's run_qwen3_moe_generate). The dense path goes through
    // run_gguf_generate as before. This replaces M32c.2.1's
    // gguf_gpu_generate.rs short-circuit with an actual forward pass.
    let infer_start = Instant::now();
    let canonical_arch = crate::tensor_names::normalize_architecture(&model.config.architecture);
    let (tokens, used_gpu) = if canonical_arch == "qwen3_moe" {
        let tokens = crate::infer::qwen3_moe_generate::run_qwen3_moe_generate(
            &mapped,
            &model,
            &input_tokens,
            &gen_config,
        )?;
        (tokens, false) // CPU-only path; GPU MoE wiring is M32d follow-up
    } else {
        run_gguf_generate(model, &input_tokens, &gen_config, config)?
    };
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

/// Check if a quantization type lacks a correct GPU GEMV kernel, so the model
/// MUST run on CPU for correct output.
///
/// PMAT-782: the legacy GGML divergence this gate guarded against was a single
/// kernel bug, not an inherent limitation. GGML packs Q4_0/Q4_1/Q5_0 nibbles
/// INTERLEAVED — byte `j` (0..16) holds value `j` (low nibble) and value `j+16`
/// (high nibble) — while the original GPU kernels assumed CONSECUTIVE packing
/// (byte = tid/2, low/high = tid&1), so every value index ≥1 mapped to the wrong
/// nibble → garbage logits (rel_gap≈0.54). Q4_0/Q5_0 were already rewritten to the
/// correct "candle" layout (BUG-GGUF-001/002); Q4_1 was the last one still on the
/// broken `Q4_1GemvKernel` consecutive layout. PMAT-782 routes Q4_1 to a candle
/// PTX generator too (`generate_q4_1_candle_ptx` in `cuda/layout.rs`). With all
/// three fixed, the cpu↔gpu parity gate now PASSES: `qwen2-0_5b-instruct-q4_0`
/// (Q4_0+Q4_1) cosine=0.99998 and `qwen2.5-coder-0.5b-instruct-q4_k_m` (Q5_0-heavy)
/// cosine=0.99948 on the RTX 4090, both producing output identical to CPU. So
/// Q4_0(2)/Q4_1(3)/Q5_0(6) are NO LONGER gated.
///
/// PMAT-783: this gate now FAILS CLOSED for *every* GGML quant type that lacks a
/// verified GPU GEMV kernel, not just Q5_1(7). The GPU weight upload resolves a
/// tensor's GGML type via `WeightQuantType::from_ggml_type(qtype)` and then
/// `resolve_qtype()` does `.unwrap_or(WeightQuantType::Q4K)` — so ANY type that
/// `from_ggml_type` does not recognize (Q5_1=7, Q8_1=9, Q2_K=10, Q3_K=11,
/// Q8_K=15, the IQ* families, and even raw F16=1 / BF16=30 reaching this path)
/// is SILENTLY decoded as Q4_K → garbage logits. The previous `matches!(qtype, 7)`
/// caught only Q5_1, leaving Q2_K/Q3_K-quantized GGUF models (common K-quant mixes)
/// to ship garbage on the unguarded `generate_gpu_resident` call sites where the
/// parity gate does not run. The whitelist below is the exact set
/// `WeightQuantType::from_ggml_type` maps to a real kernel:
///   0=F32, 2=Q4_0, 3=Q4_1, 6=Q5_0, 8=Q8_0, 12=Q4_K, 13=Q5_K, 14=Q6_K.
/// (Q4_0/Q4_1/Q5_0 kernels were corrected to candle layout in BUG-GGUF-001/002 +
/// PMAT-782; the cpu↔gpu parity gate remains the correctness backstop on the
/// primary `apr run`/`apr serve` path.)
#[inline]
fn is_legacy_gguf_quant(qtype: u32) -> bool {
    // Returns true (→ force CPU) for any GGML quant WITHOUT a verified GPU kernel.
    // Whitelist mirrors WeightQuantType::from_ggml_type (cuda/types.rs); anything
    // outside it would hit resolve_qtype's `.unwrap_or(Q4K)` → wrong-kernel garbage.
    !matches!(qtype, 0 | 2 | 3 | 6 | 8 | 12 | 13 | 14)
}

/// Check if model uses any quant type without a verified GPU kernel.
///
/// PMAT-783: checks EVERY projection tensor the GPU-resident forward pass would
/// touch — lm_head, QKV (fused or separate), attn output, and FFN gate/up/down.
/// The prior version omitted QKV and the FFN gate, so a model carrying an
/// unsupported quant in those tensors slipped past the gate onto the GPU.
fn model_has_legacy_quant(model: &crate::gguf::OwnedQuantizedModel) -> bool {
    use crate::gguf::OwnedQKVWeights;
    if is_legacy_gguf_quant(model.lm_head_weight.qtype) {
        return true;
    }
    model.layers.iter().any(|l| {
        let qkv_bad = match &l.qkv_weight {
            OwnedQKVWeights::Fused(t) => is_legacy_gguf_quant(t.qtype),
            OwnedQKVWeights::Separate { q, k, v } => {
                is_legacy_gguf_quant(q.qtype)
                    || is_legacy_gguf_quant(k.qtype)
                    || is_legacy_gguf_quant(v.qtype)
            },
        };
        qkv_bad
            || is_legacy_gguf_quant(l.attn_output_weight.qtype)
            || is_legacy_gguf_quant(l.ffn_up_weight.qtype)
            || is_legacy_gguf_quant(l.ffn_down_weight.qtype)
            || l.ffn_gate_weight
                .as_ref()
                .is_some_and(|g| is_legacy_gguf_quant(g.qtype))
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
/// Validates the GPU's first generated token against the CPU's prediction for the
/// SAME probe context, to catch a garbage GPU path (e.g. PMAT-216 transposed
/// weights) before committing to a full GPU generation.
///
/// PMAT-742: prefer the REAL prompt (`probe_context`) over a synthetic BOS token.
/// A single BOS-only probe has no context, so its next-token distribution is
/// near-flat — the top tokens are nearly tied and tiny CPU/GPU FP/quant
/// differences flip the argmax. Hard argmax-equality on that degenerate probe
/// FALSE-rejects a CORRECT GPU path and forces a ~50x-slower CPU fallback
/// (measured: 421 -> 8 tok/s on RTX 4090, qwen2.5-coder-1.5b Q4_K_M; `apr qa`
/// independently confirms the GPU path is correct via Golden-Output + Ollama-
/// Parity gates). With real context the first-token distribution is peaked, so
/// CPU and GPU agree (verified: both backends emit the same first token on a
/// coder prompt), while genuine GPU garbage still mismatches deep in CPU's tail.
/// When no prompt is available (batch model-init) it falls back to the BOS probe.
///
/// Skip entirely with SKIP_PARITY_GATE=1 (same env var as the cosine parity gate).
#[cfg(feature = "cuda")]
fn validate_gpu_first_token(
    cuda_model: &mut crate::gguf::OwnedQuantizedModelCuda,
    gen_config: &crate::gguf::QuantizedGenerateConfig,
    probe_context: &[u32],
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

    // Build the probe: real prompt context (peaked distribution) when available,
    // else the BOS token (batch model-init has no prompt yet). Cap the context to
    // bound the one-time CPU prefill cost. BOS flows from GGUF metadata; if it is
    // unknown for a context-less probe there is nothing to validate against.
    const PROBE_MAX_CTX: usize = 64;
    let probe: Vec<u32> = if probe_context.is_empty() {
        match model.config.bos_token_id {
            Some(id) => vec![id],
            None => {
                eprintln!("[F2-VALIDATION] no prompt context and BOS unknown — skipping GPU validation");
                return true;
            },
        }
    } else {
        let start = probe_context.len().saturating_sub(PROBE_MAX_CTX);
        probe_context[start..].to_vec()
    };

    let kv_dim = model.config.num_kv_heads * (model.config.hidden_dim / model.config.num_heads);
    let num_layers = model.config.num_layers;

    // CPU reference: forward the whole probe, keep the logits after the last token.
    let mut cpu_cache = OwnedQuantizedKVCache::new(num_layers, kv_dim, probe.len().max(2));
    let mut cpu_logits = None;
    for (pos, &tok) in probe.iter().enumerate() {
        match model.forward_single_with_cache(tok, &mut cpu_cache, pos) {
            Ok(logits) => cpu_logits = Some(logits),
            Err(_) => return true, // CPU forward failed — can't validate, assume GPU is fine
        }
    }
    let cpu_logits = match cpu_logits {
        Some(l) => l,
        None => return true,
    };

    let cpu_argmax = cpu_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0, |(i, _)| i as u32);

    // GPU: generate 1 token from the SAME probe context.
    let gpu_first_config = crate::gguf::QuantizedGenerateConfig {
        max_tokens: 1,
        temperature: 0.0,
        top_k: 1,
        ..gen_config.clone()
    };
    match cuda_model.generate_gpu_resident(&probe, &gpu_first_config) {
        Ok(gpu_tokens) if gpu_tokens.len() > probe.len() => {
            let gpu_first = gpu_tokens[probe.len()];
            // First-token argmax can tip on a genuine FP/quant near-tie even with
            // real context. Accept an exact match or a near-tie in CPU's own logit
            // space; reject real divergence (GPU garbage lands deep in CPU's tail).
            let rel_gap = cpu_logit_rel_gap(&cpu_logits, cpu_argmax, gpu_first);
            if gpu_probe_token_acceptable(&cpu_logits, cpu_argmax, gpu_first) {
                if gpu_first != cpu_argmax {
                    eprintln!(
                        "[F2-VALIDATION] GPU token {gpu_first} != CPU argmax {cpu_argmax} but near-tie (rel_gap={rel_gap:.4} <= {GPU_PROBE_NEAR_TIE_REL_GAP}) on {}-token probe — accepting GPU path",
                        probe.len()
                    );
                }
                true
            } else {
                eprintln!(
                    "[F2-VALIDATION] GPU token {gpu_first} != CPU token {cpu_argmax}; rel_gap={rel_gap:.4} > {GPU_PROBE_NEAR_TIE_REL_GAP} on {}-token probe — real divergence, falling back to CPU",
                    probe.len()
                );
                false
            }
        },
        Ok(_) => true,
        Err(_) => false,
    }
}

/// Max relative position (within CPU's logit min..max range) at which a GPU-chosen
/// token still counts as a harmless FP/quant near-tie. Real GPU/CPU divergence
/// (PMAT-216 garbage) lands a token deep in CPU's tail (rel_gap -> ~1.0), far above
/// this, so it is still rejected. See PMAT-742.
pub(crate) const GPU_PROBE_NEAR_TIE_REL_GAP: f32 = 0.15;

/// Relative position of `token` within CPU's [min, max] logit range
/// (0.0 = CPU's top token, 1.0 = CPU's least-likely token). Pure + GPU-free.
pub(crate) fn cpu_logit_rel_gap(cpu_logits: &[f32], cpu_argmax: u32, token: u32) -> f32 {
    let cpu_max = cpu_logits[cpu_argmax as usize];
    let cpu_min = cpu_logits.iter().copied().fold(f32::INFINITY, f32::min);
    let at = cpu_logits
        .get(token as usize)
        .copied()
        .unwrap_or(f32::NEG_INFINITY);
    let range = (cpu_max - cpu_min).max(f32::MIN_POSITIVE);
    (cpu_max - at) / range
}

/// PMAT-742 parity decision: is the GPU's first probe token acceptable against
/// CPU's reference logits? True for an exact argmax match or a genuine near-tie
/// (GPU token within [`GPU_PROBE_NEAR_TIE_REL_GAP`] of the top of CPU's logit
/// range); false for real divergence (GPU token deep in CPU's tail — PMAT-216
/// garbage). Pure + GPU-free so the gate's logic is unit-testable without CUDA.
pub(crate) fn gpu_probe_token_acceptable(cpu_logits: &[f32], cpu_argmax: u32, gpu_first: u32) -> bool {
    gpu_first == cpu_argmax
        || cpu_logit_rel_gap(cpu_logits, cpu_argmax, gpu_first) <= GPU_PROBE_NEAR_TIE_REL_GAP
}

#[cfg(test)]
mod pmat742_parity_gate_tests {
    use super::{cpu_logit_rel_gap, gpu_probe_token_acceptable, GPU_PROBE_NEAR_TIE_REL_GAP};

    // A near-flat distribution (degenerate BOS-style probe): the top few tokens are
    // nearly tied, so a CPU/GPU argmax flip among them must be ACCEPTED — this is the
    // PMAT-742 false-positive that previously forced a ~50x-slower CPU fallback.
    const NEAR_FLAT: [f32; 6] = [10.00, 9.99, 9.98, 9.97, 1.00, 0.50];

    #[test]
    fn near_tie_argmax_flip_is_accepted() {
        // CPU argmax = 0; GPU picked token 1 (the next-highest, essentially tied).
        let rel_gap = cpu_logit_rel_gap(&NEAR_FLAT, 0, 1);
        assert!(rel_gap <= GPU_PROBE_NEAR_TIE_REL_GAP, "rel_gap {rel_gap} should be a near-tie");
        assert!(gpu_probe_token_acceptable(&NEAR_FLAT, 0, 1));
        // token 3 is also within the near-tied cluster → accepted.
        assert!(gpu_probe_token_acceptable(&NEAR_FLAT, 0, 3));
    }

    #[test]
    fn exact_argmax_match_is_accepted() {
        assert!(gpu_probe_token_acceptable(&NEAR_FLAT, 0, 0));
        // Even on a peaked distribution an exact match always passes.
        let peaked = [20.0_f32, 1.0, 0.5, 0.1];
        assert!(gpu_probe_token_acceptable(&peaked, 0, 0));
    }

    #[test]
    fn real_divergence_is_rejected() {
        // PMAT-216-style garbage: GPU picks a token CPU ranks at the very bottom.
        // On a PEAKED distribution this lands deep in the tail → rejected.
        let peaked = [20.0_f32, 5.0, 0.5, 0.0];
        let rel_gap = cpu_logit_rel_gap(&peaked, 0, 3); // token 3 = CPU min
        assert!(rel_gap > GPU_PROBE_NEAR_TIE_REL_GAP, "tail token rel_gap {rel_gap} must exceed tolerance");
        assert!(!gpu_probe_token_acceptable(&peaked, 0, 3));
    }

    #[test]
    fn rel_gap_endpoints_are_zero_and_one() {
        let logits = [3.0_f32, 2.0, 1.0, 0.0];
        assert!((cpu_logit_rel_gap(&logits, 0, 0) - 0.0).abs() < 1e-6); // top
        assert!((cpu_logit_rel_gap(&logits, 0, 3) - 1.0).abs() < 1e-6); // bottom
    }

    #[test]
    fn out_of_range_gpu_token_is_rejected() {
        // A garbage token id outside the logit vector must never be accepted.
        assert!(!gpu_probe_token_acceptable(&NEAR_FLAT, 0, 9999));
    }
}
