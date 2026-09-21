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
        let mut file =
            std::fs::File::open(&config.model_path).map_err(|e| RealizarError::IoError {
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
    // #3091: Qwen3.5/Qwen3.8 hybrids (Gated DeltaNet) have their own CPU forward. The dense
    // loader refuses them, so only the shared base (embeddings, final norm, lm_head) is built
    // here and generation dispatches to `run_qwen35_generate` below.
    let is_qwen35 = mapped.model.architecture() == Some("qwen35");
    let model = if is_qwen35 {
        crate::gguf::forward_qwen35::Qwen35Model::create_base_model(&mapped.model, mapped.data())?
    } else {
        OwnedQuantizedModel::from_mapped(&mapped)?
    };
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
    // #3714 R2: `moe_forward_handles` is the one dispatch predicate — `apr
    // parity` and `apr qa` ask the same function, so no tool can route this
    // architecture differently from `apr run`.
    let (tokens, used_gpu) = if crate::gguf::moe_forward_handles(&model.config.architecture) {
        // #3714: the CUDA forward serves unless --no-gpu; a GPU that cannot
        // serve prints its reason before the CPU chain runs. This site used to
        // hard-code `(tokens, false)` and never try CUDA at all.
        crate::infer::qwen3_moe_dispatch::run_qwen3_moe_generate_dispatch(
            &mapped,
            &model,
            &input_tokens,
            &gen_config,
            config.no_gpu,
        )?
    } else if is_qwen35 {
        // #3477: the hybrid now has a GPU forward (#3090), so `apr run --gpu`
        // routes to it and reports CUDA; the CPU forward (#3091) serves
        // `--no-gpu`, a build without cuda, and any GPU failure — the last of
        // which is printed, never silent.
        crate::gguf::forward_qwen35::run_qwen35_generate_dispatch(
            &mapped,
            &model,
            &input_tokens,
            &gen_config,
            config.no_gpu,
        )?
    } else {
        run_gguf_generate(model, &input_tokens, &gen_config, config)?
    };
    let inference_ms = infer_start.elapsed().as_secs_f64() * 1000.0;

    let generated_tokens = &tokens[input_token_count..];
    let raw_text = mapped.model.decode(generated_tokens);
    if config.verbose {
        eprintln!(
            "[DEBUG] input_count={}, total_tokens={}, generated_count={}",
            input_token_count,
            tokens.len(),
            generated_tokens.len()
        );
        eprintln!(
            "[DEBUG] generated token ids: {:?}",
            &generated_tokens[..generated_tokens.len().min(20)]
        );
        eprintln!(
            "[DEBUG] raw decoded: {:?}",
            &raw_text[..raw_text.len().min(200)]
        );
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

/// The measured facts a `--trace-output` document is built from.
///
/// Grouped into one struct so [`build_gguf_trace_json`] is a pure function of
/// plain values and can be asserted on directly in a unit test without
/// constructing a full `GGUFConfig`.
#[derive(Debug, Clone, Copy)]
struct GgufTraceFacts<'a> {
    model_path: &'a str,
    num_layers: usize,
    hidden_dim: usize,
    vocab_size: usize,
    num_heads: usize,
    input_token_count: usize,
    generated_token_count: usize,
    load_ms: f64,
    inference_ms: f64,
    tps: f64,
    used_gpu: bool,
}

/// Build the `--trace-output` JSON document for a GGUF run.
///
/// # Why this exists
///
/// The document used to be assembled with `format!` and ended in a hardcoded
/// `"events": []`, so **every** `--trace-output FILE` — at every
/// `--trace-level` — wrote a file whose event list was empty. A consumer that
/// scripted `--trace-output` got a well-formed file that said nothing had
/// happened. Two defects came out of that one string:
///
/// 1. `events` was a literal empty array, never populated from the run.
/// 2. The model path was interpolated raw into a JSON string literal, so a path
///    containing `"` or `\` emitted a file that is not parseable JSON.
///
/// Both are fixed by building a `serde_json::Value` and filling `events` from
/// the timings the run actually measured.
///
/// # What goes in `events`
///
/// Only measured spans. `model_load` and `generate` are wall-clock timings
/// taken around the real work (`load_ms` / `inference_ms`), expressed as
/// chrome-tracing-compatible complete events (`ph: "X"`, µs timestamps) so the
/// file can be read by the same tooling as `--trace-level chrome`. Per-brick
/// and per-layer spans are deliberately NOT synthesised here: the brick
/// profiler owns those and inventing a split from the total would be the same
/// class of defect this function is fixing.
fn build_gguf_trace_json(facts: &GgufTraceFacts<'_>) -> serde_json::Value {
    let GgufTraceFacts {
        model_path,
        num_layers,
        hidden_dim,
        vocab_size,
        num_heads,
        input_token_count,
        generated_token_count,
        load_ms,
        inference_ms,
        tps,
        used_gpu,
    } = *facts;
    let load_us = (load_ms * 1000.0).round() as u64;
    let infer_us = (inference_ms * 1000.0).round() as u64;
    let events = serde_json::json!([
        {
            "name": "model_load",
            "cat": "lifecycle",
            "ph": "X",
            "ts": 0,
            "dur": load_us,
            "pid": 1,
            "tid": 1,
            "args": { "path": model_path, "format": "GGUF" }
        },
        {
            "name": "generate",
            "cat": "inference",
            "ph": "X",
            "ts": load_us,
            "dur": infer_us,
            "pid": 1,
            "tid": 1,
            "args": {
                "input_tokens": input_token_count,
                "generated_tokens": generated_token_count,
                "tok_per_sec": tps,
                "used_gpu": used_gpu
            }
        }
    ]);

    serde_json::json!({
        "version": "1.0",
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "model": {
            "path": model_path,
            "format": "GGUF",
            "num_layers": num_layers,
            "hidden_dim": hidden_dim,
            "vocab_size": vocab_size,
            "num_heads": num_heads,
        },
        "inference": {
            "input_tokens": input_token_count,
            "generated_tokens": generated_token_count,
            "load_ms": load_ms,
            "inference_ms": inference_ms,
            "tok_per_sec": tps,
            "used_gpu": used_gpu,
        },
        "events": events,
    })
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
    let model_path = config.model_path.display().to_string();
    let trace_json = serde_json::to_string_pretty(&build_gguf_trace_json(&GgufTraceFacts {
        model_path: &model_path,
        num_layers: model_config.num_layers,
        hidden_dim: model_config.hidden_dim,
        vocab_size: model_config.vocab_size,
        num_heads: model_config.num_heads,
        input_token_count,
        generated_token_count,
        load_ms,
        inference_ms,
        tps,
        used_gpu,
    }))
    .unwrap_or_default();
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
/// Diagnostic for a GPU forward failure during F2 first-token validation.
///
/// Pure + string-returning so it can be unit-tested without a GPU (CI has none).
/// FALSIFY-CUDA-SILENT-FALLBACK-001 asserts this names the failing position and
/// carries the underlying error, so the user is never left inferring a silent 20x
/// backend downgrade from a throughput number alone.
pub(crate) fn gpu_forward_failure_msg(
    pos: usize,
    probe_len: usize,
    err: &dyn std::fmt::Display,
) -> String {
    format!(
        "warning: GPU forward failed at probe position {pos}/{probe_len}: {err} \
— rejecting the CUDA path and falling back (fail-closed). This is NOT a decode \
regression: throughput measured after this point reflects the fallback backend, \
not CUDA."
    )
}

/// The F2 probe: the real prompt context (a peaked distribution) when there is one, else the
/// BOS token (batch model-init has no prompt yet), capped to bound the one-time CPU/GPU
/// prefill cost. BOS flows from GGUF metadata. Returns `(kv_dim, num_layers, probe)`, or
/// `None` when there is neither context nor a known BOS, so nothing to validate against.
#[cfg(feature = "cuda")]
fn gpu_probe(
    model: &crate::gguf::OwnedQuantizedModel,
    probe_context: &[u32],
) -> Option<(usize, usize, Vec<u32>)> {
    const PROBE_MAX_CTX: usize = 64;
    let kv_dim = model.config.num_kv_heads * (model.config.hidden_dim / model.config.num_heads);
    let probe = if probe_context.is_empty() {
        let Some(id) = model.config.bos_token_id else {
            eprintln!("warning: no prompt context and no BOS token — skipping the GPU-vs-CPU output check");
            return None;
        };
        vec![id]
    } else {
        probe_context[probe_context.len().saturating_sub(PROBE_MAX_CTX)..].to_vec()
    };
    Some((kv_dim, model.config.num_layers, probe))
}

/// Which GPU probe the F2 gate runs — i.e. which prefill KERNELS it judges.
///
/// #3413 C. The gate used to ALWAYS replay the probe token-by-token through
/// `forward_gpu_resident`, which is the decode / single-token path. But when
/// `GpuProfile::prefill_path()` resolves to `Batched` — the default on every GPU
/// below the sm_12x line, including the RTX 4090 — generation prefills the
/// prompt through `prefill_all_layers_gpu` → `batched_qkv_rope_phase`, kernels
/// the probe never executed. Measured on a 4090 at 4a538ddef with
/// `SKIP_CUDA_GRAPH=1`: this guard PASSED and `apr run --gpu` then printed
/// `adies――dart,`, because batched prefill cached un-normed K (#3413 B, fixed by
/// e296147d8). A guard that never runs the kernels the run uses cannot fail
/// closed on them, so the probe now follows the resolved path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum F2ProbePath {
    /// One `forward_gpu_resident` per probe token (what the gate always did).
    Serial,
    /// One packed batched prefill over the whole probe, then one decode step —
    /// exactly what `run_prefill` + the first decode do for a real prompt.
    Batched,
}

impl F2ProbePath {
    /// How the path reads in a log line.
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Serial => "serial prefill",
            Self::Batched => "batched prefill",
        }
    }
}

/// Pure: which probe the F2 gate must run, from the prefill path the engine
/// resolved for this process and the probe length.
///
/// `Batched` requires BOTH: the engine will prefill batched (otherwise the
/// batched kernels are dead code for this run and judging them would reject a
/// model on a path it never takes), AND the probe has more than one token — a
/// 1-token prompt runs no prefill PHASE at all (`run_prefill` returns early),
/// so there is nothing batched to exercise.
///
/// Pure + GPU-free so the routing itself is falsifiable without a device.
pub(crate) fn f2_select_probe_path(
    engine_prefill_is_batched: bool,
    probe_len: usize,
) -> F2ProbePath {
    if engine_prefill_is_batched && probe_len > 1 {
        F2ProbePath::Batched
    } else {
        F2ProbePath::Serial
    }
}

/// The rejection line, naming the path that was actually judged.
///
/// Without the path, a log cannot distinguish "the batched prefill diverged"
/// from "the decode path diverged" — the exact ambiguity that let #3413 B ship
/// behind a green guard. Pure + string-returning so it is unit-testable.
pub(crate) fn f2_divergence_msg(report: &F2PositionReport, via: F2ProbePath) -> String {
    format!(
        "warning: GPU output diverges from CPU at position {} (argmax {} != {}, cosine {:.4}); \
min cosine {:.4}, validated via {} — falling back to CPU",
        report.first_bad_pos,
        report.first_bad_gpu_argmax,
        report.first_bad_cpu_argmax,
        report.first_bad_cosine,
        report.min_cosine_real,
        via.as_str(),
    )
}

/// CPU reference logits for every probe position, forwarded one token at a time
/// through `forward_single_with_cache` (unchanged from before #3413 C — the
/// reference side is the same on both paths). `None` when the CPU forward
/// itself failed, in which case nothing can be validated.
#[cfg(feature = "cuda")]
fn f2_cpu_reference_logits(
    model: &crate::gguf::OwnedQuantizedModel,
    probe: &[u32],
    kv_dim: usize,
    num_layers: usize,
) -> Option<Vec<Vec<f32>>> {
    use crate::gguf::OwnedQuantizedKVCache;
    let mut per_pos: Vec<Vec<f32>> = Vec::with_capacity(probe.len() + 1);
    let mut cpu_cache = OwnedQuantizedKVCache::new(num_layers, kv_dim, probe.len().max(2) + 1);
    for (pos, &tok) in probe.iter().enumerate() {
        match model.forward_single_with_cache(tok, &mut cpu_cache, pos) {
            Ok(logits) => per_pos.push(logits),
            Err(_) => return None,
        }
    }
    // One CPU decode step, so the batched probe's decode step has a reference.
    // Greedy: the token CPU itself would emit after the probe.
    if let Some(last) = per_pos.last() {
        let next = argmax_u32(last);
        match model.forward_single_with_cache(next, &mut cpu_cache, probe.len()) {
            Ok(logits) => per_pos.push(logits),
            Err(_) => return None,
        }
    }
    Some(per_pos)
}

/// The GPU probe as it has always run: `forward_gpu_resident` once per probe
/// token, plus the one decode step the CPU reference now also takes.
#[cfg(feature = "cuda")]
fn f2_gpu_serial_logits(
    cuda_model: &mut crate::gguf::OwnedQuantizedModelCuda,
    probe: &[u32],
    decode_token: u32,
    kv_dim: usize,
    num_layers: usize,
) -> std::result::Result<Vec<Vec<f32>>, String> {
    use crate::gguf::OwnedQuantizedKVCache;
    let mut per_pos: Vec<Vec<f32>> = Vec::with_capacity(probe.len() + 1);
    let mut gpu_cache = OwnedQuantizedKVCache::new(num_layers, kv_dim, probe.len().max(2) + 1);
    let steps = probe.len() + 1;
    for (pos, tok) in probe
        .iter()
        .copied()
        .chain(std::iter::once(decode_token))
        .enumerate()
    {
        match cuda_model.forward_gpu_resident(tok, &mut gpu_cache, pos) {
            Ok(logits) => per_pos.push(logits),
            Err(e) => return Err(gpu_forward_failure_msg(pos, steps, &e)),
        }
    }
    Ok(per_pos)
}

/// The GPU probe through the BATCHED prefill (#3413 C).
///
/// This is the sequence `run_prefill`'s batched branch runs for a real prompt —
/// `init_prefill_workspace` → `prefill_all_layers_gpu` (→ `batched_qkv_rope_phase`,
/// the packed KV scatter) → workspace restore — followed by the per-position LM
/// head (`hidden_to_logits`, as `perplexity_gpu_batched` extracts it) and ONE
/// decode step at position `probe.len()`. The decode step is what makes a
/// KV-cache defect visible: a prefill that computes correct hidden states but
/// scatters poisoned K/V shows up only when a later token attends to them.
///
/// `run_prefill` itself is private to `gguf/cuda/generate_2.rs`; this calls the
/// same executor entry point it calls, so the kernels under judgement are the
/// kernels that serve the prompt.
#[cfg(feature = "cuda")]
fn f2_gpu_batched_logits(
    cuda_model: &mut crate::gguf::OwnedQuantizedModelCuda,
    probe: &[u32],
    decode_token: u32,
    kv_dim: usize,
    num_layers: usize,
) -> std::result::Result<Vec<Vec<f32>>, String> {
    use crate::gguf::OwnedQuantizedKVCache;

    let hidden_dim = cuda_model.model.config.hidden_dim;
    let vocab_size = cuda_model.model.config.vocab_size;
    let eps = cuda_model.model.config.eps;
    let Some(layer0) = cuda_model.model.layers.first() else {
        return Err("F2 batched probe: model has no layers".to_string());
    };
    let intermediate_dim = layer0.ffn_up_weight.out_dim;
    let s = probe.len();

    let embeddings = cuda_model.model.embed(probe);
    let positions: Vec<u32> = (0..s as u32).collect();

    cuda_model.executor.reset_kv_cache_gpu();
    cuda_model
        .executor
        .init_prefill_workspace(s, hidden_dim, intermediate_dim)
        .map_err(|e| format!("F2 batched probe: init_prefill_workspace failed: {e}"))?;
    cuda_model
        .executor
        .prefill_all_layers_gpu(
            &embeddings,
            &positions,
            num_layers,
            hidden_dim as u32,
            intermediate_dim as u32,
            eps,
        )
        .map_err(|e| format!("F2 batched probe: prefill_all_layers_gpu failed: {e}"))?;
    cuda_model
        .executor
        .sync_stream()
        .map_err(|e| format!("F2 batched probe: stream sync failed: {e}"))?;

    let mut all_hidden = vec![0.0f32; s * hidden_dim];
    cuda_model
        .executor
        .download_hidden_buf2(&mut all_hidden)
        .map_err(|e| format!("F2 batched probe: hidden download failed: {e}"))?;

    let mut per_pos: Vec<Vec<f32>> = Vec::with_capacity(s + 1);
    for i in 0..s {
        let mut logits = vec![0.0f32; vocab_size];
        cuda_model
            .executor
            .hidden_to_logits(
                &all_hidden[i * hidden_dim..(i + 1) * hidden_dim],
                &mut logits,
                hidden_dim as u32,
                vocab_size as u32,
                eps,
            )
            .map_err(|e| format!("F2 batched probe: hidden_to_logits failed at {i}: {e}"))?;
        if let Some(bias) = cuda_model.model.lm_head_bias.as_ref() {
            crate::gguf::ops::add_bias(&mut logits, bias);
        }
        per_pos.push(logits);
    }

    // Restore the decode workspace exactly as `run_prefill` does before the
    // first decode step, then take that step against the prefilled KV cache.
    cuda_model
        .executor
        .init_workspace(hidden_dim, intermediate_dim)
        .map_err(|e| format!("F2 batched probe: workspace restore failed: {e}"))?;
    let mut gpu_cache = OwnedQuantizedKVCache::new(num_layers, kv_dim, s.max(2) + 1);
    match cuda_model.forward_gpu_resident(decode_token, &mut gpu_cache, s) {
        Ok(logits) => per_pos.push(logits),
        Err(e) => return Err(gpu_forward_failure_msg(s, s + 1, &e)),
    }
    Ok(per_pos)
}

/// The three ways the F2 check declines to judge at all, in one place. `None` means
/// "nothing to validate — assume the GPU is fine"; `Some((kv_dim, num_layers, probe))`
/// is a probe with at least one REAL position (≥1) in it.
///
/// `SKIP_PARITY_GATE=1` bypasses both this F2 check and the cosine parity gate. A
/// single-token probe (context-less BOS) has NO real position to validate — the pos0
/// distribution is a benign near-tie, and the load-time parity gate is the primary
/// defense there.
#[cfg(feature = "cuda")]
fn f2_probe_to_judge(
    cuda_model: &crate::gguf::OwnedQuantizedModelCuda,
    probe_context: &[u32],
) -> Option<(usize, usize, Vec<u32>)> {
    if std::env::var("SKIP_PARITY_GATE")
        .map(|v| v == "1")
        .unwrap_or(false)
    {
        return None;
    }
    let (kv_dim, num_layers, probe) = gpu_probe(cuda_model.model(), probe_context)?;
    if probe.len() < 2 {
        return None;
    }
    Some((kv_dim, num_layers, probe))
}

/// Run the probe through whichever prefill path the engine resolved (#3413 C).
#[cfg(feature = "cuda")]
fn f2_gpu_logits_via(
    cuda_model: &mut crate::gguf::OwnedQuantizedModelCuda,
    via: F2ProbePath,
    probe: &[u32],
    decode_token: u32,
    kv_dim: usize,
    num_layers: usize,
) -> std::result::Result<Vec<Vec<f32>>, String> {
    match via {
        F2ProbePath::Batched => {
            f2_gpu_batched_logits(cuda_model, probe, decode_token, kv_dim, num_layers)
        },
        F2ProbePath::Serial => {
            f2_gpu_serial_logits(cuda_model, probe, decode_token, kv_dim, num_layers)
        },
    }
}

/// FALSIFY-CUDA-SILENT-FALLBACK-001: never discard this error.
///
/// Every OTHER rejection in this validator explains itself (see the
/// F2 divergence branch, and the BOS-unknown skip), but this one used
/// `Err(_)` and returned a bare `false`. The caller
/// (gguf_gpu_generate.rs:303) can only turn that into
/// `Err(Box::new(model))` for CPU fallback — the reason has nowhere
/// to go and is lost.
///
/// Observed 2026-07-27 on an RTX 4090: CUDA initialises fine and the
/// run still ends on CPU, with the only clue being the backend
/// cascade —
///     Backend: GPU (NVIDIA GeForce RTX 4090, 24045 MB VRAM)
///     Backend: wgpu (Vulkan)
///     Backend: CPU (wgpu unavailable: cosine 0.884 < 0.99)
/// with no stated cause even under --verbose. A silent downgrade is
/// indistinguishable from a decode regression, and it makes any
/// throughput measured through it a fabrication (the Pillar-4 beat
/// raises BEAT-REGRESSION off exactly this state).
///
/// Unconditional, not verbose-gated: the user is about to silently
/// receive the fallback backend, which they need to know regardless
/// of verbosity.
///
/// #3413 C: the message now carries the path that failed, because a
/// batched-prefill failure and a decode failure are different faults.
#[cfg(feature = "cuda")]
fn f2_report_forward_failure(msg: &str, via: F2ProbePath) {
    eprintln!("{msg} [{}]", via.as_str());
}

/// #3413 C: re-measure the probe on the FP16 HGEMM prefill with FP8 turned OFF.
/// A precision fallback, not a backend fallback — printed, never silent. `None`
/// when the FP16 forward itself failed, in which case the FP8 report stands.
#[cfg(feature = "cuda")]
fn f2_remeasure_without_fp8(
    cuda_model: &mut crate::gguf::OwnedQuantizedModelCuda,
    probe: &[u32],
    decode_token: u32,
    kv_dim: usize,
    num_layers: usize,
    cpu_logits_per_pos: &[Vec<f32>],
    via: F2ProbePath,
) -> Option<F2PositionReport> {
    cuda_model.executor.gpu_profile.fp8_prefill = false;
    cuda_model.executor.gpu_profile.fp8_decode = false;
    cuda_model.executor.reset_kv_cache_gpu();
    let out = match f2_gpu_batched_logits(cuda_model, probe, decode_token, kv_dim, num_layers) {
        Ok(v) => {
            let report = f2_multi_position_report(cpu_logits_per_pos, &v);
            if report.accepted {
                eprintln!(
                    "note: FP16 prefill passes (min cosine {:.4}); FP8 stays OFF for this model",
                    report.min_cosine_real,
                );
            }
            Some(report)
        },
        Err(msg) => {
            eprintln!("{msg} [{} retry without FP8]", via.as_str());
            None
        },
    };
    cuda_model.executor.reset_kv_cache_gpu();
    out
}

/// The F2 verdict itself, once a report exists.
#[cfg(feature = "cuda")]
fn f2_accept_or_reject(
    report: &F2PositionReport,
    via: F2ProbePath,
    real_positions: usize,
) -> bool {
    if !report.accepted {
        eprintln!("{}", f2_divergence_msg(report, via));
        return false;
    }
    // #2405: the GPU was ACCEPTED — this line reports a benign near-tie on a
    // successful run and says nothing the user can act on, so it is a
    // developer trace, not user output.
    if report.pos0_argmax_flip && crate::dev_trace::dev_trace_enabled() {
        eprintln!(
            "pos0 argmax flip (benign BOS near-tie) ignored; all {real_positions} real positions match (min cosine {:.4} >= {F2_GATE_COSINE_MIN}) via {} — accepting GPU path",
            report.min_cosine_real,
            via.as_str(),
        );
    }
    true
}

#[cfg(feature = "cuda")]
fn validate_gpu_first_token(
    cuda_model: &mut crate::gguf::OwnedQuantizedModelCuda,
    _gen_config: &crate::gguf::QuantizedGenerateConfig,
    probe_context: &[u32],
) -> bool {
    let Some((kv_dim, num_layers, probe)) = f2_probe_to_judge(cuda_model, probe_context) else {
        return true;
    };

    // CPU reference: forward the whole probe (plus one greedy decode step),
    // KEEPING THE LOGITS AT EVERY POSITION.
    let Some(cpu_logits_per_pos) =
        f2_cpu_reference_logits(cuda_model.model(), &probe, kv_dim, num_layers)
    else {
        return true; // CPU forward failed — can't validate, assume GPU is fine
    };
    // The decode token both sides take after the probe: CPU's own greedy choice,
    // so the GPU is measured on the continuation a real run would generate.
    let decode_token = cpu_logits_per_pos
        .get(probe.len().saturating_sub(1))
        .map_or(0, |l| argmax_u32(l));

    // #3413 C: judge the prefill path the ENGINE resolved for this process. On a
    // batched-prefill GPU the prompt is served by `prefill_all_layers_gpu` /
    // `batched_qkv_rope_phase`, which the old token-by-token probe never ran.
    let via = f2_select_probe_path(
        cuda_model.executor.gpu_profile.prefill_path().path
            == crate::cuda::gpu_profile::PrefillPath::Batched,
        probe.len(),
    );

    cuda_model.executor.reset_kv_cache_gpu();
    let gpu_result =
        f2_gpu_logits_via(cuda_model, via, &probe, decode_token, kv_dim, num_layers);
    let gpu_logits_per_pos = match gpu_result {
        Ok(v) => v,
        Err(msg) => {
            cuda_model.executor.reset_kv_cache_gpu();
            f2_report_forward_failure(&msg, via);
            return false; // GPU forward failed — fail closed.
        },
    };
    cuda_model.executor.reset_kv_cache_gpu();

    // Per-position decision over REAL positions (≥1). Excludes pos0 (BOS near-tie).
    let mut report = f2_multi_position_report(&cpu_logits_per_pos, &gpu_logits_per_pos);

    // #3413 C / PMAT-798 pattern: the FP8 batched prefill's precision shows at the
    // decode step AFTER the prompt (the K/V it attends to were produced by FP8
    // GEMMs) and sits at the floor run-to-run: qwen2.5-1.5b "Write one sentence
    // about the ocean." → pos 27, cosine 0.9187 with the SAME argmax on one run,
    // ≥ 0.95 on another (RTX 4090, 2026-09-18). A miss inside the non-catastrophic
    // band on an FP8 path is re-measured on the FP16 HGEMM prefill before the model
    // is pushed off the GPU — a precision fallback, not a backend fallback, printed,
    // never silent; the floors themselves do not move. Anything below the
    // catastrophic band (Qwen3-8B under FP8: −0.0972) is never retried.
    if f2_should_retry_without_fp8(&report, via, cuda_model.executor.gpu_profile.fp8_prefill) {
        eprintln!(
            "note: FP8 batched prefill scored min cosine {:.4} vs CPU (floor {F2_GATE_COSINE_MIN}) — re-measuring on the FP16 prefill path",
            report.min_cosine_real,
        );
        if let Some(fp16) = f2_remeasure_without_fp8(
            cuda_model,
            &probe,
            decode_token,
            kv_dim,
            num_layers,
            &cpu_logits_per_pos,
            via,
        ) {
            report = fp16;
        }
    }

    f2_accept_or_reject(&report, via, cpu_logits_per_pos.len().saturating_sub(1))
}

/// #3413 C: retry the batched probe with FP8 prefill off iff the miss is inside the
/// non-catastrophic band (`≥ F2_CATASTROPHIC_COSINE`), the path judged was the
/// batched prefill, and FP8 was actually on — otherwise a retry could not change
/// the answer and would only hide a real divergence behind a second measurement.
/// Pure + GPU-free.
#[cfg(feature = "cuda")]
pub(crate) fn f2_should_retry_without_fp8(
    report: &F2PositionReport,
    via: F2ProbePath,
    fp8_prefill_on: bool,
) -> bool {
    !report.accepted
        && via == F2ProbePath::Batched
        && fp8_prefill_on
        && report.min_cosine_real >= F2_CATASTROPHIC_COSINE
        && report.first_bad_cosine >= F2_CATASTROPHIC_COSINE
}

#[cfg(all(test, feature = "cuda"))]
mod pmat3477_f2_fp8_retry_tests {
    use super::{f2_should_retry_without_fp8, F2PositionReport, F2ProbePath};

    fn miss(cos: f32) -> F2PositionReport {
        F2PositionReport {
            accepted: false,
            pos0_argmax_flip: false,
            min_cosine_real: cos,
            first_bad_pos: 27,
            first_bad_cpu_argmax: 29,
            first_bad_gpu_argmax: 29,
            first_bad_cosine: cos,
        }
    }

    // The measured case: FP8 batched prefill on the qwen2.5 control, pos 27,
    // cosine 0.9187, same argmax → re-measure on FP16.
    #[test]
    fn fp8_batched_miss_in_band_is_retried() {
        assert!(f2_should_retry_without_fp8(
            &miss(0.9187),
            F2ProbePath::Batched,
            true
        ));
    }

    // A catastrophic miss (Qwen3-8B under FP8: −0.0972) is never retried: a
    // second measurement cannot turn orthogonal logits into parity.
    #[test]
    fn catastrophic_miss_is_not_retried() {
        assert!(!f2_should_retry_without_fp8(
            &miss(-0.0972),
            F2ProbePath::Batched,
            true
        ));
        assert!(!f2_should_retry_without_fp8(
            &miss(0.40),
            F2ProbePath::Batched,
            true
        ));
    }

    // FP8 already off, or the serial path: nothing to switch, no retry.
    #[test]
    fn retry_needs_fp8_on_and_the_batched_path() {
        assert!(!f2_should_retry_without_fp8(
            &miss(0.9187),
            F2ProbePath::Batched,
            false
        ));
        assert!(!f2_should_retry_without_fp8(
            &miss(0.9187),
            F2ProbePath::Serial,
            true
        ));
    }

    #[test]
    fn an_accepted_report_is_not_retried() {
        let mut r = miss(0.99);
        r.accepted = true;
        assert!(!f2_should_retry_without_fp8(&r, F2ProbePath::Batched, true));
    }
}

/// PMAT-919 F2 per-position cosine floor. For every REAL probe position (≥1) the
/// GPU's logits must score `cosine ≥ F2_GATE_COSINE_MIN` against CPU's regardless of
/// argmax. Set strictly below the load-time `PARITY_GATE_COSINE_MIN` (0.98 in
/// `cuda/mod.rs`) so this subordinate gate never rejects a model the load-time gate
/// accepted. Backstops the degraded HwDp4a 7B case (pos6 @ 0.9398 → reject).
pub(crate) const F2_GATE_COSINE_MIN: f32 = 0.95;

/// PMAT-919 F2 argmax-mismatch cosine threshold. A per-position argmax MISMATCH is
/// only fatal when the cosine is ALSO degraded below this (a genuine divergence,
/// e.g. degraded HwDp4a 1.5B pos3 @ 0.9705). A high-cosine (≥ this) argmax flip is a
/// benign FP/quant NEAR-TIE — two logits within epsilon, the distributions are
/// essentially identical — and must be ACCEPTED (the CORRECT fp32-Mwv default flips a
/// late-position argmax at cosine 0.9995; rejecting that is the PMAT-742/#1864
/// false-positive at a non-zero position). Set to the load-time τ_load (0.98) so the
/// "argmax flip below 0.98" reject window is exactly the degraded-cosine band.
pub(crate) const F2_ARGMAX_MISMATCH_COSINE: f32 = 0.98;

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

/// PMAT-919 (reconciled) F2 decision over a multi-token probe. For every REAL
/// position (index ≥ 1) the GPU path is REJECTED iff:
///   • cosine < `F2_GATE_COSINE_MIN` (0.95) — quant/catastrophic floor (degraded
///     HwDp4a 7B pos6 @ 0.9398, orthogonal garbage cos≈0), OR
///   • argmax mismatch AND cosine < `F2_ARGMAX_MISMATCH_COSINE` (0.98) — a genuine
///     mid-context divergence (degraded HwDp4a 1.5B pos3, argmax flip @ 0.9705).
/// A high-cosine (≥0.98) argmax flip is a BENIGN near-tie and is ACCEPTED (the
/// correct fp32-Mwv default flips a late-position argmax at cosine 0.9995). Position 0
/// (the context-less BOS near-tie) is EXCLUDED entirely so the correct path is never
/// false-rejected there (PMAT-742/#1864). Pure + GPU-free.
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
        let argmax_mismatch = cpu_argmax != gpu_argmax;
        // Reject on: cosine below the quant-aware floor (degraded HwDp4a 7B /
        // orthogonal garbage), OR a real-position argmax mismatch AT a degraded
        // cosine (< 0.98 — degraded HwDp4a 1.5B). A high-cosine (≥0.98) argmax flip
        // is a benign FP/quant near-tie → NOT a reject (correct fp32-Mwv flips a
        // late argmax at cosine 0.9995).
        let bad =
            cosine < F2_GATE_COSINE_MIN || (argmax_mismatch && cosine < F2_ARGMAX_MISMATCH_COSINE);
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
pub(crate) fn gpu_probe_token_acceptable(
    cpu_logits: &[f32],
    cpu_argmax: u32,
    gpu_first: u32,
) -> bool {
    gpu_first == cpu_argmax
        || cpu_logit_rel_gap(cpu_logits, cpu_argmax, gpu_first) <= GPU_PROBE_NEAR_TIE_REL_GAP
}

#[cfg(test)]
mod pmat919_cosine_gate_tests {
    use super::{
        argmax_u32, f2_multi_position_report, logits_cosine_similarity, F2_ARGMAX_MISMATCH_COSINE,
        F2_CATASTROPHIC_COSINE, F2_GATE_COSINE_MIN,
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
        assert!(
            report.pos0_argmax_flip,
            "test must reproduce the benign pos0 flip"
        );
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
        // token (40) and degrades cosine into the [0.95, 0.98) band — the measured
        // 1.5B pos3 @ 0.9705 symptom: a genuine mid-context divergence, NOT a benign
        // near-tie (whose cosine would be ≥ 0.98). Built by adding ~25%-of-magnitude
        // structured noise (enough angular deviation to land ~0.96) plus the flip.
        gpu[2] = cpu[2]
            .iter()
            .enumerate()
            .map(|(i, &c)| match i {
                33 => 16.0,                                                  // demote CPU's leader
                40 => 21.0, // GPU now argmaxes a *different* token
                _ => c + ((i as f32 * 1.7).sin() * 0.30 * c.abs().max(2.0)), // ~cosine 0.96
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
            pos2_cos > F2_CATASTROPHIC_COSINE && pos2_cos < F2_ARGMAX_MISMATCH_COSINE,
            "this is a REAL-margin (degraded but not catastrophic) case: pos2 cosine {pos2_cos} \
             must be in (0.90, 0.98) — argmax mismatch at degraded cosine, the 1.5B symptom"
        );
        assert!(
            !report.accepted,
            "the F2 gate MUST REJECT the degraded HwDp4a path (mid-position argmax mismatch at degraded cosine)"
        );
        assert_eq!(
            report.first_bad_pos, 2,
            "the reject must be attributed to pos2"
        );
    }

    /// CRITICAL FALSIFIER (ii-b) ACCEPT — a HIGH-cosine late-position argmax flip is a
    /// BENIGN near-tie, NOT a degradation. The CORRECT fp32-Mwv default flips a
    /// late-position argmax at cosine 0.9995 (measured on lambda 4090 pos11: argmax
    /// 12669 != 71341 @ 0.9995). An argmax mismatch ABOVE F2_ARGMAX_MISMATCH_COSINE
    /// (0.98) must be ACCEPTED — else the correct path is false-rejected (the
    /// PMAT-742/#1864 false-positive, recurring at a non-zero position).
    #[test]
    fn high_cosine_late_argmax_flip_is_accepted() {
        let cpu: Vec<Vec<f32>> = vec![
            vocab_logits(5, 0.0),
            vocab_logits(12, 1.0),
            // pos2: two near-tied leaders (33 and 34) within FP noise.
            (0..256)
                .map(|i| match i {
                    33 => 20.00,
                    34 => 19.99,
                    _ => 6.0 + ((i as f32 + 2.0) * 0.013).sin() * 4.0,
                })
                .collect(),
        ];
        let mut gpu = cpu.clone();
        // pos2: the near-tie tips the OTHER way under quant noise (argmax 33 -> 34),
        // but the distributions are essentially identical → cosine ~0.9999.
        gpu[2][33] = 19.99;
        gpu[2][34] = 20.00;

        let report = f2_multi_position_report(&cpu, &gpu);
        assert_ne!(
            argmax_u32(&cpu[2]),
            argmax_u32(&gpu[2]),
            "test must reproduce a late-position argmax flip"
        );
        let pos2_cos = logits_cosine_similarity(&cpu[2], &gpu[2]);
        assert!(
            pos2_cos >= F2_ARGMAX_MISMATCH_COSINE,
            "benign near-tie pos2 cosine {pos2_cos} must be >= {F2_ARGMAX_MISMATCH_COSINE}"
        );
        assert!(
            report.accepted,
            "the F2 gate MUST ACCEPT a high-cosine ({pos2_cos}) argmax flip — it is a benign \
             near-tie (the correct fp32-Mwv path), NOT a degradation"
        );
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
        assert!(
            report.pos0_argmax_flip,
            "test must reproduce the pos0 argmax flip"
        );
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

    /// Threshold hierarchy: τ_cat (0.90) < τ_f2 (0.95) < τ_argmax (0.98) ≤ load-time
    /// PARITY_GATE_COSINE_MIN (0.98), so the subordinate F2 gate can never reject what
    /// the load-time gate accepted.
    #[test]
    fn f2_threshold_below_load_time_gate() {
        const LOAD_TIME_PARITY_GATE_COSINE_MIN: f32 = 0.98;
        assert!(
            F2_GATE_COSINE_MIN < LOAD_TIME_PARITY_GATE_COSINE_MIN,
            "F2 tau {F2_GATE_COSINE_MIN} must be < load-time tau {LOAD_TIME_PARITY_GATE_COSINE_MIN}"
        );
        assert!(F2_CATASTROPHIC_COSINE < F2_GATE_COSINE_MIN);
        assert!(F2_GATE_COSINE_MIN < F2_ARGMAX_MISMATCH_COSINE);
        assert!(F2_ARGMAX_MISMATCH_COSINE <= LOAD_TIME_PARITY_GATE_COSINE_MIN);
    }

    /// Identical logits across all positions → accepted (sanity).
    #[test]
    fn identical_multi_position_logits_are_accepted() {
        let cpu: Vec<Vec<f32>> = vec![
            vocab_logits(5, 0.0),
            vocab_logits(12, 1.0),
            vocab_logits(33, 2.0),
        ];
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
        assert!(
            rel_gap <= GPU_PROBE_NEAR_TIE_REL_GAP,
            "rel_gap {rel_gap} should be a near-tie"
        );
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
        assert!(
            rel_gap > GPU_PROBE_NEAR_TIE_REL_GAP,
            "tail token rel_gap {rel_gap} must exceed tolerance"
        );
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

#[cfg(test)]
mod cuda_silent_fallback_tests {
    use super::gpu_forward_failure_msg;

    /// FALSIFY-CUDA-SILENT-FALLBACK-001: a GPU forward failure during F2 validation
    /// must explain itself.
    ///
    /// The rejection used `Err(_)` and returned a bare `false`. Its caller
    /// (gguf_gpu_generate.rs:303) can only convert that into `Err(Box::new(model))`
    /// for CPU fallback, so the reason had nowhere to go. Observed on an RTX 4090:
    /// CUDA initialises, the run silently ends on CPU far below the GPU decode rate,
    /// and nothing says why even under --verbose.
    ///
    /// That matters beyond ergonomics: throughput measured after a silent fallback
    /// is a fabrication. The Pillar-4 decode beat reports `ratio_median=0.070x` and
    /// a BEAT-REGRESSION panic off exactly this state — a 14x "regression" with
    /// nothing wrong in the decode path.
    ///
    /// CI has no GPU, so the diagnostic is a pure function and this asserts its
    /// content directly.
    #[test]
    fn gpu_forward_failure_is_explained_not_swallowed() {
        let msg = gpu_forward_failure_msg(3, 17, &"CUDA_ERROR_ILLEGAL_ADDRESS");

        assert!(
            msg.contains("CUDA_ERROR_ILLEGAL_ADDRESS"),
            "FALSIFY-CUDA-SILENT-FALLBACK-001: the underlying error must survive into \
             the message — discarding it is the defect. Got: {msg}"
        );
        assert!(
            msg.contains('3') && msg.contains("17"),
            "must name the failing probe position and probe length. Got: {msg}"
        );
        assert!(
            msg.to_lowercase().contains("fallback") || msg.to_lowercase().contains("falling back"),
            "must state that a fallback is happening, so a 20x slower backend is never \
             silent. Got: {msg}"
        );
        assert!(
            msg.contains("NOT a decode regression"),
            "must inoculate against the fabricated-regression reading: any throughput \
             measured after this point reflects the fallback backend. Got: {msg}"
        );
    }

    /// #2405: the same message opened with `[F2-VALIDATION]`, an internal
    /// falsifier ID. It told the user which of our tickets to go read, which is
    /// not a thing a user of `apr run` can do. The diagnostic content above is
    /// unchanged; only the ticket number is gone.
    #[test]
    fn gpu_forward_failure_does_not_address_the_user_in_ticket_numbers() {
        let msg = gpu_forward_failure_msg(3, 17, &"CUDA_ERROR_ILLEGAL_ADDRESS");
        assert!(
            !msg.contains("F2-VALIDATION"),
            "internal falsifier ID leaked into user output: {msg}"
        );
        assert!(
            !msg.starts_with('['),
            "message opens with a bracketed internal tag: {msg}"
        );
        assert!(
            msg.starts_with("warning:"),
            "a fail-closed GPU rejection must announce itself as a warning: {msg}"
        );
    }
}

#[cfg(test)]
mod trace_output_events_tests {
    use super::{build_gguf_trace_json, GgufTraceFacts};

    fn facts(path: &str) -> GgufTraceFacts<'_> {
        GgufTraceFacts {
            model_path: path,
            num_layers: 28,
            hidden_dim: 1536,
            vocab_size: 151_936,
            num_heads: 12,
            input_token_count: 9,
            generated_token_count: 3,
            load_ms: 120.5,
            inference_ms: 640.25,
            tps: 4.7,
            used_gpu: false,
        }
    }

    /// FALSIFIER: `--trace-output FILE` must not write an empty event list.
    ///
    /// The document was assembled by `format!` and ended in a hardcoded
    /// `"events": []`, so every `--trace-output` at every `--trace-level`
    /// produced a file that said nothing had happened. Shape alone is not
    /// enough here — asserting that `events` is an array passes on the bug — so
    /// this asserts the events exist AND carry the run's measured durations.
    #[test]
    fn trace_output_events_are_populated_from_measured_timings() {
        let v = build_gguf_trace_json(&facts("/models/qwen.gguf"));
        let events = v["events"].as_array().expect("events must be an array");
        assert!(
            !events.is_empty(),
            "trace-output must carry events, got an empty list: {v}"
        );

        let load = events
            .iter()
            .find(|e| e["name"] == "model_load")
            .unwrap_or_else(|| panic!("no model_load event in {v}"));
        assert_eq!(
            load["dur"].as_u64(),
            Some(120_500),
            "model_load duration must be the MEASURED load_ms in microseconds"
        );

        let generate = events
            .iter()
            .find(|e| e["name"] == "generate")
            .unwrap_or_else(|| panic!("no generate event in {v}"));
        assert_eq!(
            generate["dur"].as_u64(),
            Some(640_250),
            "generate duration must be the MEASURED inference_ms in microseconds"
        );
        assert_eq!(generate["args"]["generated_tokens"], 3);
        assert_eq!(generate["args"]["input_tokens"], 9);
    }

    /// A model path containing a quote used to be interpolated raw into a JSON
    /// string literal, so the "trace" file was not parseable JSON at all.
    #[test]
    fn trace_output_is_valid_json_for_a_path_with_a_quote() {
        let v = build_gguf_trace_json(&facts(r#"/models/we"ird\path.gguf"#));
        let text = serde_json::to_string(&v).expect("serialise");
        let round: serde_json::Value = serde_json::from_str(&text)
            .unwrap_or_else(|e| panic!("trace output must be parseable JSON: {e}\n{text}"));
        assert_eq!(round["model"]["path"], r#"/models/we"ird\path.gguf"#);
    }
}

/// #3413 C — the F2 GPU guard must judge the prefill path generation will USE.
#[cfg(test)]
mod pmat3477_f2_prefill_path_tests {
    use super::{f2_divergence_msg, f2_select_probe_path, F2PositionReport, F2ProbePath};

    /// The probe follows the ENGINE's resolved prefill path, not a constant.
    ///
    /// Before #3413 C the gate always replayed the probe token-by-token through
    /// `forward_gpu_resident`, so on every GPU below the sm_12x line (where
    /// `select_prefill_path` returns `Batched`) it validated kernels the run
    /// does not use for the prompt, and was blind to `batched_qkv_rope_phase`.
    #[test]
    fn f2_probe_path_follows_the_engine_prefill_path() {
        let cases: [(bool, usize, F2ProbePath); 5] = [
            (true, 8, F2ProbePath::Batched),
            (true, 2, F2ProbePath::Batched),
            (false, 8, F2ProbePath::Serial),
            // A 1-token probe runs no prefill PHASE at all (`run_prefill`
            // returns early), so there are no batched kernels to judge.
            (true, 1, F2ProbePath::Serial),
            (false, 1, F2ProbePath::Serial),
        ];
        for (engine_batched, probe_len, expected) in cases {
            assert_eq!(
                f2_select_probe_path(engine_batched, probe_len),
                expected,
                "engine_batched={engine_batched} probe_len={probe_len}"
            );
        }
    }

    /// A log line that does not name the path it judged cannot tell a future
    /// reader whether the batched prefill was ever exercised.
    #[test]
    fn f2_divergence_message_names_the_path_that_was_judged() {
        let report = F2PositionReport {
            accepted: false,
            pos0_argmax_flip: false,
            min_cosine_real: 0.9186,
            first_bad_pos: 1,
            first_bad_cpu_argmax: 40,
            first_bad_gpu_argmax: 198,
            first_bad_cosine: 0.9186,
        };
        let batched = f2_divergence_msg(&report, F2ProbePath::Batched);
        assert!(
            batched.contains("via batched prefill"),
            "a batched-prefill rejection must say so: {batched}"
        );
        let serial = f2_divergence_msg(&report, F2ProbePath::Serial);
        assert!(
            serial.contains("via serial prefill"),
            "a serial rejection must say so: {serial}"
        );
        assert!(
            !serial.contains("batched"),
            "a serial rejection must not claim the batched path: {serial}"
        );
        // The measured facts survive into both.
        for msg in [&batched, &serial] {
            assert!(msg.contains("position 1"), "{msg}");
            assert!(msg.contains("198"), "{msg}");
            assert!(msg.contains("0.9186"), "{msg}");
        }
    }
}

/// #3413 C, on-device: the F2 guard must EXERCISE the batched prefill.
///
/// `cuda_executor_or_skip!` returns early on a GPU-less host, so this passes
/// vacuously there; it is a device test, run on the 4090 lane.
#[cfg(all(test, feature = "cuda"))]
mod pmat3477_f2_batched_probe_cuda_tests {
    use super::{
        argmax_u32, f2_cpu_reference_logits, f2_gpu_batched_logits, f2_multi_position_report,
        f2_select_probe_path, F2ProbePath, F2_GATE_COSINE_MIN,
    };

    fn model_path() -> std::path::PathBuf {
        std::path::Path::new(env!("HOME")).join(".apr/models/Qwen3-1.7B-Q4_K_M.gguf")
    }

    /// The batched probe's logits must agree with the per-position CPU
    /// reference, on a real Qwen3 model, with `BATCHED_PREFILL` forced on.
    ///
    /// This is the falsifier for #3413 C: before e296147d8 the batched prefill
    /// cached un-normed K, and the ONLY reason the guard stayed green was that
    /// it never ran these kernels. Reverting that fix must turn this RED.
    #[test]
    #[serial_test::serial]
    fn f2_batched_probe_agrees_with_the_cpu_reference() {
        let path = model_path();
        if !path.exists() {
            eprintln!("SKIP: no model at {}", path.display());
            return;
        }
        let _skip_probe = crate::cuda_executor_or_skip!(0);
        drop(_skip_probe);

        // Force the batched path so the routing under test is the one taken,
        // regardless of this box's compute capability.
        // SAFETY: `#[serial]` keeps this the only test mutating the environment.
        std::env::set_var("BATCHED_PREFILL", "1");

        let mapped = match crate::gguf::MappedGGUFModel::from_path(&path) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("SKIP: cannot map {}: {e}", path.display());
                return;
            },
        };
        let cpu_model = crate::gguf::OwnedQuantizedModel::from_mapped(&mapped)
            .expect("Qwen3-1.7B Q4_K_M must load on CPU");
        let mut cuda_model = match crate::gguf::OwnedQuantizedModelCuda::new(cpu_model, 0) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("SKIP: CUDA model init failed: {e}");
                return;
            },
        };

        // The engine must have resolved BATCHED — otherwise this test would be
        // measuring the serial path and claiming the batched one.
        let choice = cuda_model.executor.gpu_profile.prefill_path();
        assert_eq!(
            choice.path,
            crate::cuda::gpu_profile::PrefillPath::Batched,
            "BATCHED_PREFILL=1 must resolve to the batched prefill (reason {})",
            choice.reason
        );

        let probe: Vec<u32> = vec![9707, 11, 1246, 525, 498, 30, 358, 1079];
        assert_eq!(
            f2_select_probe_path(true, probe.len()),
            F2ProbePath::Batched,
            "the guard must route this probe through the batched prefill"
        );

        let kv_dim = cuda_model.model().config.num_kv_heads
            * (cuda_model.model().config.hidden_dim / cuda_model.model().config.num_heads);
        let num_layers = cuda_model.model().config.num_layers;

        let cpu = f2_cpu_reference_logits(cuda_model.model(), &probe, kv_dim, num_layers)
            .expect("CPU reference must forward");
        let decode_token = argmax_u32(&cpu[probe.len() - 1]);

        cuda_model.executor.reset_kv_cache_gpu();
        let gpu = f2_gpu_batched_logits(&mut cuda_model, &probe, decode_token, kv_dim, num_layers)
            .expect("the batched prefill probe must run on the device");
        cuda_model.executor.reset_kv_cache_gpu();

        assert_eq!(
            gpu.len(),
            probe.len() + 1,
            "the batched probe must cover every prompt position AND one decode step"
        );
        assert_eq!(cpu.len(), gpu.len(), "both sides must cover the same steps");

        let report = f2_multi_position_report(&cpu, &gpu);
        assert!(
            report.accepted,
            "batched prefill must agree with CPU: first bad pos {} (argmax {} != {}, cosine {:.4}), min cosine {:.4}",
            report.first_bad_pos,
            report.first_bad_gpu_argmax,
            report.first_bad_cpu_argmax,
            report.first_bad_cosine,
            report.min_cosine_real,
        );
        assert!(
            report.min_cosine_real >= F2_GATE_COSINE_MIN,
            "min real-position cosine {} must clear the F2 floor",
            report.min_cosine_real
        );
    }
}
