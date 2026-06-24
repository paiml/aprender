
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

    let mut gen_config = QuantizedGenerateConfig {
        max_tokens: config.max_tokens,
        stop_tokens,
        trace: config.trace,
        ..Default::default()
    };
    // PMAT-823: forward EVERY sampling param (temperature/top_k/top_p/seed/
    // repeat_penalty/repeat_last_n) — previously only temperature+top_k were
    // copied and the rest silently fell to greedy defaults, so the GGUF/GPU
    // decode path ran argmax regardless of `apr run` sampling flags.
    config.apply_sampling_to(&mut gen_config);

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
///
/// PMAT-785: delegates to `gguf::gpu_unsupported_quant_qtype` (the single source
/// of truth shared with the construction-time gate
/// `OwnedQuantizedModel::has_gpu_unsupported_quant`) so the whitelist can never
/// drift between the primary `apr run`/`apr serve` path and the
/// `generate_gpu_resident` construction entry points.
#[inline]
fn is_legacy_gguf_quant(qtype: u32) -> bool {
    // Returns true (→ force CPU) for any GGML quant WITHOUT a verified GPU kernel.
    // Whitelist mirrors WeightQuantType::from_ggml_type (cuda/types.rs); anything
    // outside it would hit resolve_qtype's `.unwrap_or(Q4K)` → wrong-kernel garbage.
    crate::gguf::gpu_unsupported_quant_qtype(qtype)
}

/// Check if model uses any quant type without a verified GPU kernel.
///
/// PMAT-783: checks EVERY projection tensor the GPU-resident forward pass would
/// touch — lm_head, QKV (fused or separate), attn output, and FFN gate/up/down.
/// The prior version omitted QKV and the FFN gate, so a model carrying an
/// unsupported quant in those tensors slipped past the gate onto the GPU.
///
/// PMAT-785: delegates to `OwnedQuantizedModel::has_gpu_unsupported_quant`, the
/// shared construction-time gate, so this `apr run`/`apr serve`-path check and
/// the `generate_gpu_resident` construction gate apply identical tensor coverage
/// and the same quant whitelist.
fn model_has_legacy_quant(model: &crate::gguf::OwnedQuantizedModel) -> bool {
    model.has_gpu_unsupported_quant()
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

/// F2-FIX: Validate the GPU path with a SHORT MULTI-TOKEN probe, asserting
/// per-position argmax agreement AND a per-position cosine floor for every REAL
/// (≥1) position, not just the last token.
///
/// RECONCILED GROUND TRUTH (gx10 Blackwell study, 2026-06-24): on Blackwell the
/// production default is fp32 **Mwv** Q4K (`auto_q4k(cc≥120) = Mwv`). It is
/// byte-identical to CPU-Q4K and to llama.cpp token-for-token; every real position
/// argmax-matches at cosine ≥ 0.9998. The **HwDp4a** path is the GENUINELY DEGRADED
/// one: its INT8 Q8_1 *activation* quantization mis-estimates massive-activation
/// channels, producing a real argmax MISMATCH at mid-context (measured: pos3 @ 0.9705
/// on qwen2.5-coder-1.5B, pos6 @ 0.9398 on the 7B). A last-token-only cosine gate
/// MISSES that — the divergence is mid-sequence and the *final* token can still be
/// fine — so the gate forwards the WHOLE probe on both backends and checks EVERY
/// position.
///
/// Decision (see [`f2_multi_position_acceptable`]):
///   ACCEPT ⟺ ∀ p ≥ 1: argmax(cpu[p]) == argmax(gpu[p])
///            ∧ min_{p≥1} cosine(cpu[p], gpu[p]) ≥ F2_GATE_COSINE_MIN (0.95)
///            ∧ no NaN / zero-norm (catastrophic floor)
///
/// Position 0 is EXCLUDED: a context-less BOS / first-token distribution is
/// near-flat, so an argmax flip there is a benign FP near-tie (PMAT-742 / #1864 —
/// the correct fp32-Mwv default itself flips pos0 at cosine ~0.945). Including pos0
/// would FALSE-REJECT the correct production path. Every REAL position (≥1) on the
/// correct fp32-Mwv path argmax-matches at ≥0.9998, while HwDp4a's mid-position
/// argmax mismatch is caught regardless of where its cosine lands.
///
/// When no prompt is available (batch model-init) only a single BOS token exists;
/// there are no positions ≥1 to validate, so the gate is a no-op (returns true) —
/// the load-time `parity_gate` is the primary defense in that case.
/// Skip entirely with SKIP_PARITY_GATE=1 (same env var as the cosine parity gate).
#[cfg(feature = "cuda")]
fn validate_gpu_first_token(
    cuda_model: &mut crate::gguf::OwnedQuantizedModelCuda,
    _gen_config: &crate::gguf::QuantizedGenerateConfig,
    probe_context: &[u32],
) -> bool {
    use crate::gguf::OwnedQuantizedKVCache;

    // SKIP_PARITY_GATE=1 bypasses both this F2 check and the cosine parity gate.
    if std::env::var("SKIP_PARITY_GATE")
        .map(|v| v == "1")
        .unwrap_or(false)
    {
        return true;
    }

    // Build the probe: real prompt context (peaked distribution) when available,
    // else the BOS token (batch model-init has no prompt yet). Cap the context to
    // bound the one-time CPU/GPU prefill cost. BOS flows from GGUF metadata; if it
    // is unknown for a context-less probe there is nothing to validate against.
    const PROBE_MAX_CTX: usize = 64;
    let (kv_dim, num_layers, probe): (usize, usize, Vec<u32>) = {
        let model = cuda_model.model();
        let kv_dim =
            model.config.num_kv_heads * (model.config.hidden_dim / model.config.num_heads);
        let num_layers = model.config.num_layers;
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
        (kv_dim, num_layers, probe)
    };

    // A single-token probe (context-less BOS) has NO real position (≥1) to validate
    // — the pos0 distribution is a benign near-tie. The load-time parity gate is the
    // primary defense there; skip the per-position F2 check.
    if probe.len() < 2 {
        return true;
    }

    // CPU reference: forward the whole probe, KEEP THE LOGITS AT EVERY POSITION.
    let mut cpu_logits_per_pos: Vec<Vec<f32>> = Vec::with_capacity(probe.len());
    {
        let model = cuda_model.model();
        let mut cpu_cache = OwnedQuantizedKVCache::new(num_layers, kv_dim, probe.len().max(2));
        for (pos, &tok) in probe.iter().enumerate() {
            match model.forward_single_with_cache(tok, &mut cpu_cache, pos) {
                Ok(logits) => cpu_logits_per_pos.push(logits),
                Err(_) => return true, // CPU forward failed — can't validate, assume GPU is fine
            }
        }
    }

    // GPU reference: forward the SAME probe on the GPU-resident path (the same
    // `forward_gpu_resident` the load-time `parity_gate` compares), keeping the full
    // logit vector at EVERY position so we can catch a mid-context divergence.
    cuda_model.executor.reset_kv_cache_gpu();
    let mut gpu_logits_per_pos: Vec<Vec<f32>> = Vec::with_capacity(probe.len());
    let mut gpu_cache = OwnedQuantizedKVCache::new(num_layers, kv_dim, probe.len().max(2));
    for (pos, &tok) in probe.iter().enumerate() {
        match cuda_model.forward_gpu_resident(tok, &mut gpu_cache, pos) {
            Ok(logits) => gpu_logits_per_pos.push(logits),
            Err(_) => {
                cuda_model.executor.reset_kv_cache_gpu();
                return false; // GPU forward failed — fail closed.
            },
        }
    }
    cuda_model.executor.reset_kv_cache_gpu();

    // Per-position decision over REAL positions (≥1). Excludes pos0 (BOS near-tie).
    let report = f2_multi_position_report(&cpu_logits_per_pos, &gpu_logits_per_pos);
    if report.accepted {
        if report.pos0_argmax_flip {
            eprintln!(
                "[F2-VALIDATION] pos0 argmax flip (benign BOS near-tie) ignored; all {} real positions match (min cosine {:.4} >= {F2_GATE_COSINE_MIN}) — accepting GPU path",
                probe.len() - 1,
                report.min_cosine_real,
            );
        }
        true
    } else {
        eprintln!(
            "[F2-VALIDATION] GPU diverges from CPU at real position {} (argmax {} != {}, cosine {:.4}); min real-position cosine {:.4} — HwDp4a-class mid-context degradation, falling back to CPU",
            report.first_bad_pos,
            report.first_bad_gpu_argmax,
            report.first_bad_cpu_argmax,
            report.first_bad_cosine,
            report.min_cosine_real,
        );
        false
    }
}

/// PMAT-919 F2 per-position cosine floor. For every REAL probe position (≥1) the
/// GPU's logits must score `cosine ≥ F2_GATE_COSINE_MIN` against CPU's. Set strictly
/// below the load-time `PARITY_GATE_COSINE_MIN` (0.98 in `cuda/mod.rs`) so this
/// subordinate gate never rejects a model the load-time gate accepted, yet far above
/// the cosine ≈ 0 produced by orthogonal garbage (PMAT-216 transposed weights). The
/// PRIMARY catch for the degraded HwDp4a path is the per-position argmax MISMATCH
/// (HwDp4a flips a mid-context argmax even at cosine ~0.97); this floor backstops the
/// 7B case (pos6 @ 0.9398) and any catastrophic kernel.
pub(crate) const F2_GATE_COSINE_MIN: f32 = 0.95;

/// Catastrophic cosine floor (orthogonal garbage / NaN). Anything below this is a
/// hard reject regardless of argmax, matching the `apr parity` catastrophic floor.
pub(crate) const F2_CATASTROPHIC_COSINE: f32 = 0.90;

/// Per-position F2 decision report. Pure + GPU-free → unit-testable without CUDA.
#[derive(Debug, Clone)]
pub(crate) struct F2PositionReport {
    /// Accept the GPU path?
    pub accepted: bool,
    /// Was position 0's argmax flipped (benign BOS near-tie, ignored)?
    pub pos0_argmax_flip: bool,
    /// Minimum cosine over REAL positions (≥1); 1.0 if there are none.
    pub min_cosine_real: f32,
    /// First REAL position that caused a reject (0 if accepted).
    pub first_bad_pos: usize,
    pub first_bad_cpu_argmax: u32,
    pub first_bad_gpu_argmax: u32,
    pub first_bad_cosine: f32,
}

/// PMAT-919 (reconciled) F2 decision over a multi-token probe: ACCEPT iff every REAL
/// position (index ≥ 1) has matching CPU/GPU argmax AND cosine ≥ `F2_GATE_COSINE_MIN`,
/// with a catastrophic floor (`F2_CATASTROPHIC_COSINE`) that rejects orthogonal
/// garbage on ANY position. Position 0 (the context-less BOS near-tie) is EXCLUDED
/// from the argmax/cosine assertions so the correct fp32-Mwv default (pos0 @ ~0.945,
/// argmax flip, all real positions ≥ 0.9998) is NOT false-rejected. The degraded
/// HwDp4a path (mid-context argmax mismatch — pos3 @ 0.9705 on 1.5B, pos6 @ 0.9398 on
/// 7B) is rejected. Pure + GPU-free.
pub(crate) fn f2_multi_position_report(
    cpu_per_pos: &[Vec<f32>],
    gpu_per_pos: &[Vec<f32>],
) -> F2PositionReport {
    let n = cpu_per_pos.len().min(gpu_per_pos.len());
    let mut min_cosine_real = 1.0_f32;
    let mut pos0_argmax_flip = false;

    // Detect the benign pos0 near-tie purely for diagnostics (it never causes reject).
    if n > 0 {
        pos0_argmax_flip = argmax_u32(&cpu_per_pos[0]) != argmax_u32(&gpu_per_pos[0]);
    }

    // No real position to validate → no-op accept (load-time gate is primary).
    if n < 2 {
        return F2PositionReport {
            accepted: true,
            pos0_argmax_flip,
            min_cosine_real: 1.0,
            first_bad_pos: 0,
            first_bad_cpu_argmax: 0,
            first_bad_gpu_argmax: 0,
            first_bad_cosine: 1.0,
        };
    }

    for pos in 1..n {
        let cpu = &cpu_per_pos[pos];
        let gpu = &gpu_per_pos[pos];
        let cosine = logits_cosine_similarity(cpu, gpu);
        if cosine < min_cosine_real {
            min_cosine_real = cosine;
        }
        let cpu_argmax = argmax_u32(cpu);
        let gpu_argmax = argmax_u32(gpu);
        // Reject on: orthogonal garbage (catastrophic floor), cosine below the
        // quant-aware floor, OR a real-position argmax mismatch (the HwDp4a symptom,
        // which can occur even at cosine ~0.97).
        let bad = cosine < F2_CATASTROPHIC_COSINE
            || cosine < F2_GATE_COSINE_MIN
            || cpu_argmax != gpu_argmax;
        if bad {
            return F2PositionReport {
                accepted: false,
                pos0_argmax_flip,
                min_cosine_real,
                first_bad_pos: pos,
                first_bad_cpu_argmax: cpu_argmax,
                first_bad_gpu_argmax: gpu_argmax,
                first_bad_cosine: cosine,
            };
        }
    }

    F2PositionReport {
        accepted: true,
        pos0_argmax_flip,
        min_cosine_real,
        first_bad_pos: 0,
        first_bad_cpu_argmax: 0,
        first_bad_gpu_argmax: 0,
        first_bad_cosine: 1.0,
    }
}

/// Cosine similarity between two logit vectors, f64-accumulated (matches the
/// load-time gate's `cosine_similarity` in `mod_parity_gate.rs`). Pure + GPU-free
/// so the F2 gate's decision logic is unit-testable without CUDA. Returns 0.0 when
/// either vector has ~zero norm (a degenerate/garbage logit vector → rejected).
pub(crate) fn logits_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot: f64 = 0.0;
    let mut norm_a: f64 = 0.0;
    let mut norm_b: f64 = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        let x = f64::from(*x);
        let y = f64::from(*y);
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < 1e-12 {
        0.0
    } else {
        (dot / denom) as f32
    }
}

/// argmax of a logit vector as a token id (0 on empty / all-NaN). Pure + GPU-free.
pub(crate) fn argmax_u32(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0, |(i, _)| i as u32)
}

/// Max relative position (within CPU's logit min..max range) at which a GPU-chosen
/// token still counts as a harmless FP/quant near-tie. Real GPU/CPU divergence
/// (PMAT-216 garbage) lands a token deep in CPU's tail (rel_gap -> ~1.0), far above
/// this, so it is still rejected. See PMAT-742. (Retained for the legacy
/// token-index helpers + their unit tests; the live F2 gate now uses cosine.)
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
mod pmat919_cosine_gate_tests {
    use super::{
        argmax_u32, f2_multi_position_report, logits_cosine_similarity, F2_CATASTROPHIC_COSINE,
        F2_GATE_COSINE_MIN,
    };

    /// A realistic vocab-head logit vector with a clear leader at `leader`.
    fn vocab_logits(leader: usize, seed: f32) -> Vec<f32> {
        (0..256)
            .map(|i| {
                if i == leader {
                    20.0
                } else {
                    6.0 + ((i as f32 + seed) * 0.013).sin() * 4.0
                }
            })
            .collect()
    }

    /// CRITICAL FALSIFIER (i) ACCEPT — the CORRECT fp32-Mwv production default.
    /// Multi-position probe where EVERY real position (≥1) argmax-matches at cosine
    /// ≥ 0.9998 (only quant noise). pos0 is a benign BOS near-tie (argmax flip) that
    /// must be IGNORED. The reconciled-truth correct path → ACCEPT.
    #[test]
    fn correct_fp32_mwv_multi_position_logits_are_accepted() {
        // 4-position probe. CPU = clean leaders 5, 12, 33, 71 at positions 0..4.
        let cpu: Vec<Vec<f32>> = vec![
            vocab_logits(5, 0.0),
            vocab_logits(12, 1.0),
            vocab_logits(33, 2.0),
            vocab_logits(71, 3.0),
        ];
        // GPU = CPU + ~0.5% quant noise on every real position (cosine ≥ 0.9998,
        // argmax preserved). pos0 gets a deliberate benign argmax flip (near-tie).
        let mut gpu = cpu.clone();
        for (pos, v) in gpu.iter_mut().enumerate() {
            for (i, g) in v.iter_mut().enumerate() {
                *g += ((i as f32 * 0.37 + pos as f32).cos()) * (g.abs() * 0.002);
            }
        }
        // pos0 benign BOS near-tie: flip the argmax to a near-tied token.
        gpu[0][5] = 19.99;
        gpu[0][6] = 20.01;

        let report = f2_multi_position_report(&cpu, &gpu);
        assert!(report.pos0_argmax_flip, "test must reproduce the benign pos0 flip");
        // All REAL positions match.
        for pos in 1..4 {
            assert_eq!(
                argmax_u32(&cpu[pos]),
                argmax_u32(&gpu[pos]),
                "real position {pos} argmax must match for the correct fp32-Mwv path"
            );
        }
        assert!(
            report.min_cosine_real >= 0.9998,
            "correct fp32-Mwv min real-position cosine {} must be >= 0.9998",
            report.min_cosine_real
        );
        assert!(
            report.accepted,
            "the F2 gate MUST accept the correct fp32-Mwv default despite the pos0 BOS flip"
        );
    }

    /// CRITICAL FALSIFIER (ii) REJECT at the REAL margin — the degraded HwDp4a path.
    /// A mid-position (pos2) argmax MISMATCH at cosine ~0.94 (the measured 7B pos6 @
    /// 0.9398 symptom). Last-token-only cosine would MISS this; the per-position gate
    /// catches it → REJECT. This is the headline fail-closed requirement.
    #[test]
    fn hwdp4a_mid_position_argmax_mismatch_is_rejected() {
        let cpu: Vec<Vec<f32>> = vec![
            vocab_logits(5, 0.0),
            vocab_logits(12, 1.0),
            vocab_logits(33, 2.0), // pos2: the mid-context divergence position
            vocab_logits(71, 3.0),
        ];
        let mut gpu = cpu.clone();
        // pos2: HwDp4a's INT8 activation-quant error flips the argmax to a different
        // token (40) and degrades cosine to ~0.94 — above the catastrophic floor but
        // a genuine mid-context divergence.
        gpu[2] = (0..256)
            .map(|i| match i {
                33 => 17.0,        // demote CPU's leader
                40 => 20.5,        // GPU now argmaxes a *different* token
                _ => 6.0 + ((i as f32 + 2.0) * 0.013).sin() * 4.0 + 3.5, // shift → cosine ~0.94
            })
            .collect();

        let report = f2_multi_position_report(&cpu, &gpu);
        assert_ne!(
            argmax_u32(&cpu[2]),
            argmax_u32(&gpu[2]),
            "test must reproduce the HwDp4a mid-position argmax mismatch"
        );
        let pos2_cos = logits_cosine_similarity(&cpu[2], &gpu[2]);
        assert!(
            pos2_cos > F2_CATASTROPHIC_COSINE,
            "this is a REAL-margin (not cosine~0) case: pos2 cosine {pos2_cos} must be > {F2_CATASTROPHIC_COSINE}"
        );
        assert!(
            !report.accepted,
            "the F2 gate MUST REJECT the degraded HwDp4a path (mid-position argmax mismatch)"
        );
        assert_eq!(report.first_bad_pos, 2, "the reject must be attributed to pos2");
    }

    /// CRITICAL FALSIFIER (iii) NO FALSE-REJECT — pos0-ONLY BOS near-tie.
    /// An argmax flip at pos0 (cosine ~0.945) with EVERY real position matching is the
    /// benign FP near-tie (PMAT-742/#1864). The gate must NOT false-reject it → ACCEPT.
    #[test]
    fn pos0_only_bos_near_tie_is_not_false_rejected() {
        // Near-flat pos0 (BOS): top tokens essentially tied → cosine ~0.945 + flip.
        let near_flat: Vec<f32> = (0..256)
            .map(|i| match i {
                5 => 10.00,
                6 => 9.99,
                _ => 9.0 + (i as f32 * 0.5).sin() * 0.4,
            })
            .collect();
        let mut gpu_pos0 = near_flat.clone();
        gpu_pos0[5] = 9.99;
        gpu_pos0[6] = 10.00; // flip argmax 5 -> 6 (benign near-tie)

        let cpu: Vec<Vec<f32>> = vec![near_flat, vocab_logits(12, 1.0), vocab_logits(33, 2.0)];
        let mut gpu = cpu.clone();
        gpu[0] = gpu_pos0;
        // Real positions: tiny quant noise only, argmax preserved.
        for pos in 1..3 {
            for (i, g) in gpu[pos].iter_mut().enumerate() {
                *g += ((i as f32 * 0.21).cos()) * (g.abs() * 0.002);
            }
        }

        let report = f2_multi_position_report(&cpu, &gpu);
        assert!(report.pos0_argmax_flip, "test must reproduce the pos0 argmax flip");
        assert!(
            report.accepted,
            "pos0-only BOS near-tie must NOT be false-rejected (all real positions match)"
        );
    }

    /// CRITICAL FALSIFIER (iv) REJECT orthogonal garbage (cosine ~0) — PMAT-216-class
    /// transposed-weight garbage on a real position → catastrophic floor → REJECT.
    #[test]
    fn orthogonal_garbage_logits_are_rejected() {
        let cpu: Vec<Vec<f32>> = vec![vocab_logits(5, 0.0), vocab_logits(12, 1.0)];
        let mut gpu = cpu.clone();
        // pos1: uncorrelated pseudo-random garbage (transposed-weight class).
        gpu[1] = (0..256)
            .map(|i| ((i as f32 * 12.9898).sin() * 43758.547).fract() * 20.0 - 10.0)
            .collect();
        let pos1_cos = logits_cosine_similarity(&cpu[1], &gpu[1]);
        assert!(
            pos1_cos < F2_CATASTROPHIC_COSINE,
            "garbage pos1 cosine {pos1_cos} must be below the catastrophic floor"
        );
        let report = f2_multi_position_report(&cpu, &gpu);
        assert!(
            !report.accepted,
            "the F2 gate MUST reject orthogonal garbage (cosine ~0) — fail-closed"
        );
    }

    /// Threshold hierarchy: τ_f2 (0.95) < load-time PARITY_GATE_COSINE_MIN (0.98),
    /// so the subordinate F2 gate can never reject what the load-time gate accepted.
    #[test]
    fn f2_threshold_below_load_time_gate() {
        const LOAD_TIME_PARITY_GATE_COSINE_MIN: f32 = 0.98;
        assert!(
            F2_GATE_COSINE_MIN < LOAD_TIME_PARITY_GATE_COSINE_MIN,
            "F2 tau {F2_GATE_COSINE_MIN} must be < load-time tau {LOAD_TIME_PARITY_GATE_COSINE_MIN}"
        );
        assert!(F2_CATASTROPHIC_COSINE < F2_GATE_COSINE_MIN);
    }

    /// Identical logits across all positions → accepted (sanity).
    #[test]
    fn identical_multi_position_logits_are_accepted() {
        let cpu: Vec<Vec<f32>> = vec![vocab_logits(5, 0.0), vocab_logits(12, 1.0), vocab_logits(33, 2.0)];
        let report = f2_multi_position_report(&cpu, &cpu.clone());
        assert!(report.accepted);
        assert!((report.min_cosine_real - 1.0).abs() < 1e-5);
    }

    /// Single-token probe (context-less BOS) has no real position → no-op accept
    /// (load-time gate is primary). Must NOT false-reject on a pos0-only near-tie.
    #[test]
    fn single_token_probe_is_noop_accept() {
        let cpu: Vec<Vec<f32>> = vec![vocab_logits(5, 0.0)];
        let mut gpu = cpu.clone();
        gpu[0][5] = 19.0;
        gpu[0][6] = 21.0; // pos0 flip
        let report = f2_multi_position_report(&cpu, &gpu);
        assert!(report.accepted, "single-token probe must be a no-op accept");
    }
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
