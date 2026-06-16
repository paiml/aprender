
/// FALSIFY-CPU-GPU-003 jidoka tag emitted on stderr when a GPU init/parity
/// rejection forces a fallback. Locked in by `tests::cuda_fallback_log_prefix_is_contract_tagged`
/// to prevent regression to the verbose-only behaviour that v6 fixed.
///
/// See `contracts/apr-cpu-vs-gpu-output-parity-v1.yaml` and
/// `evidence/ship-007-layer-0-oracle-bisection-2026-05-03/findings-v6-parity-gate-fires-but-fallback-is-silent.md`.
pub(crate) const CUDA_FALLBACK_LOG_PREFIX: &str =
    "[apr-cpu-vs-gpu-output-parity-v1] CUDA path rejected";

/// FALSIFY-CPU-GPU-005 jidoka tag emitted on stderr when wgpu init/forward
/// rejection forces a fallback. Locked in by
/// `tests::wgpu_fallback_log_prefix_is_contract_tagged` to prevent the same
/// silent-fallback regression class that #1428 closed for CUDA — the v1.2.0
/// contract predicts this tag at `gguf_gpu_generate.rs:317`-style rejection
/// points so users always see which backend was rejected without --verbose.
///
/// See `contracts/apr-cpu-vs-gpu-output-parity-v1.yaml` § FALSIFY-CPU-GPU-005.
pub(crate) const WGPU_FALLBACK_LOG_PREFIX: &str =
    "[apr-cpu-vs-gpu-output-parity-v1] wgpu path rejected";

/// f64-accumulated cosine similarity for FALSIFY-CPU-GPU-005 part b.
///
/// Numerically-stable companion to `cuda::mod_parity_gate::cosine_similarity` (which
/// lives behind `cfg(feature = "cuda")`). Lifted to this module so the future wgpu
/// cosine gate (predicted by contract `apr-cpu-vs-gpu-output-parity-v1` v1.2.0
/// FALSIFY-CPU-GPU-005 part b) can compare a wgpu single-step decode against a
/// CPU reference forward at init without taking a `--features cuda` build dependency.
///
/// Returns 0.0 when either input is zero-norm or the inputs differ in length —
/// this is the conservative "fail-closed" default that triggers fallback to CPU.
///
/// See `contracts/apr-cpu-vs-gpu-output-parity-v1.yaml` § FALSIFY-CPU-GPU-005
/// implementation_evidence line 201 for the gate algorithm.
pub(crate) fn cpu_vs_gpu_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
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

/// GH-559: Try wgpu (Vulkan) generation as fallback when CUDA JIT fails.
/// Uses trueno's WgslForwardPass with dequantized F32 weights.
/// Proven: cosine=0.999863 on Blackwell sm_121.
#[cfg(feature = "gpu")]
fn try_wgpu_generate(
    model: &crate::gguf::OwnedQuantizedModel,
    input_tokens: &[u32],
    gen_config: &crate::gguf::QuantizedGenerateConfig,
    verbose: bool,
) -> Result<(Vec<u32>, bool)> {
    use crate::gpu::adapters::wgpu_adapter;

    if !trueno::backends::gpu::GpuDevice::is_available() {
        return Err(RealizarError::InferenceError("wgpu not available".into()));
    }

    let gpu = trueno::backends::gpu::GpuDevice::new()
        .map_err(|e| RealizarError::InferenceError(format!("wgpu init: {e}")))?;

    // FALSIFY-CPU-GPU-005: wgpu lifecycle visible without --verbose so users
    // see which backend actually serves their tokens after CUDA fallback.
    let _ = verbose;
    eprintln!("Backend: wgpu (Vulkan)");

    let config = model.config();
    let hidden_dim = config.hidden_dim;
    let num_layers = config.num_layers;
    let num_heads = config.num_heads;
    let num_kv_heads = config.num_kv_heads;
    let head_dim = hidden_dim / num_heads;
    let intermediate_dim = config.intermediate_dim;
    let vocab_size = config.vocab_size;
    let eps = config.eps;
    let kv_dim = num_kv_heads * head_dim;

    // Create forward pass and upload dequantized weights
    let mut fwd = trueno::backends::gpu::WgslForwardPass::new(
        gpu.device, gpu.queue,
        hidden_dim, num_heads, num_kv_heads, head_dim, intermediate_dim,
    );

    // C-WGPU-Q4K-001: Upload raw Q4K bytes for projection weights.
    // encode_matmul() auto-selects Q4K GEMV when M=1 and Q4K weights exist.
    let raw_q4k = wgpu_adapter::raw_q4k_weights(model);
    let q4k_names: std::collections::HashSet<String> =
        raw_q4k.iter().map(|(n, _, _, _)| n.clone()).collect();
    for (name, data, _rows, _cols) in &raw_q4k {
        fwd.upload_q4k_weight(name, data);
    }

    // Upload F32 weights for norms, biases, and non-Q4K tensors.
    // Q4K projection weights are skipped (already uploaded as raw Q4K).
    let weights = wgpu_adapter::dequant_model_weights(model)?;
    for (name, data, _rows, _cols) in &weights {
        if !q4k_names.contains(name) {
            fwd.upload_weight(name, data);
        }
    }

    // Get output norm and LM head weights
    let output_norm = model.output_norm_weight();
    let lm_head_f32: Vec<f32> = weights.iter()
        .find(|(n, _, _, _)| n == "lm_head")
        .map(|(_, d, _, _)| d.clone())
        .unwrap_or_default();

    // KV caches
    let max_seq = gen_config.max_tokens + input_tokens.len() + 16;
    let mut kv_caches: Vec<(Vec<f32>, Vec<f32>)> = (0..num_layers)
        .map(|_| (vec![0.0f32; max_seq * kv_dim], vec![0.0f32; max_seq * kv_dim]))
        .collect();

    // FALSIFY-CPU-GPU-006 (#1864): multi-step CPU vs wgpu parity gate.
    //
    // The pre-#1864 GGUF wgpu path had NO parity gate at all — it loaded
    // weights, ran the autoregressive loop, and returned. Qwen2.5-7B Q4K
    // shipped "ampiezza"-style gibberish straight to the user with exit 0.
    // The .apr wgpu path had a single-step gate (FALSIFY-CPU-GPU-005) but
    // that's also insufficient: 7B's wgpu KV cache drifts each step even
    // when step 0 cosine ≥ 0.99.
    //
    // This gate runs CPU vs wgpu in lockstep for N steps (default 3, override
    // via APR_WGPU_PARITY_STEPS in [1, 16]). Both paths advance through the
    // same deterministic token sequence (CPU argmax), and we cosine-compare
    // the full vocab-size logit vectors at every step. ANY cosine < 0.99
    // aborts with the WGPU_FALLBACK_LOG_PREFIX tag so the caller falls back
    // to CPU rather than ship silent drift.
    //
    // Cost: N forward passes at init (~0.5-2s on 7B Q4K) — paid once per
    // `apr run`, not per token. See contracts/apr-cpu-vs-gpu-output-parity-v1.yaml.
    {
        const MULTI_STEP_PROBE_DEFAULT: usize = 3;
        let multi_step_probe: usize = std::env::var("APR_WGPU_PARITY_STEPS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&n| (1..=16).contains(&n))
            .unwrap_or(MULTI_STEP_PROBE_DEFAULT);

        let probe_max_seq = multi_step_probe + 1;
        let mut cpu_cache = crate::gguf::OwnedQuantizedKVCache::from_config(&config, probe_max_seq);
        let mut probe_kv_caches: Vec<(Vec<f32>, Vec<f32>)> = (0..num_layers)
            .map(|_| (vec![0.0f32; probe_max_seq * kv_dim], vec![0.0f32; probe_max_seq * kv_dim]))
            .collect();
        let mut probe_token = *input_tokens.first().unwrap_or(&0);

        for probe_step in 0..multi_step_probe {
            let cpu_logits = match model.forward_single_with_cache(probe_token, &mut cpu_cache, probe_step) {
                Ok(l) => l,
                Err(e) => {
                    eprintln!(
                        "{}, attempting fallback: CPU probe step {} forward failed: {}",
                        WGPU_FALLBACK_LOG_PREFIX, probe_step, e
                    );
                    return Err(RealizarError::InferenceError(format!("wgpu parity gate: CPU probe step {probe_step} failed: {e}")));
                }
            };

            let mut hidden = model.embed(&[probe_token]);
            for layer_idx in 0..num_layers {
                let prefix = format!("layer.{layer_idx}");
                let (ref mut kv_k, ref mut kv_v) = probe_kv_caches[layer_idx];
                if let Err(e) = fwd.forward_layer(&mut hidden, &prefix, probe_step, kv_k, kv_v) {
                    eprintln!(
                        "{}, attempting fallback: wgpu probe step {} layer {} failed: {}",
                        WGPU_FALLBACK_LOG_PREFIX, probe_step, layer_idx, e
                    );
                    return Err(RealizarError::InferenceError(format!("wgpu parity gate: step {probe_step} layer {layer_idx} failed: {e}")));
                }
            }
            let sq_sum: f32 = hidden.iter().map(|x| x * x).sum();
            let rms = (sq_sum / hidden.len() as f32 + eps).sqrt();
            let normed: Vec<f32> = hidden
                .iter()
                .zip(output_norm.iter())
                .map(|(x, g)| (x / rms) * g)
                .collect();
            let mut wgpu_logits = vec![0.0_f32; vocab_size];
            for i in 0..vocab_size {
                let row = &lm_head_f32[i * hidden_dim..(i + 1) * hidden_dim];
                wgpu_logits[i] = row.iter().zip(normed.iter()).map(|(w, x)| w * x).sum();
            }

            let cos = cpu_vs_gpu_cosine_similarity(&cpu_logits, &wgpu_logits);
            if !(cos.is_finite() && cos >= 0.99) {
                eprintln!(
                    "{}, attempting fallback: cosine vs CPU = {:.6} (< 0.99) at step {}/{}",
                    WGPU_FALLBACK_LOG_PREFIX, cos, probe_step + 1, multi_step_probe
                );
                return Err(RealizarError::InferenceError(format!(
                    "wgpu parity gate: cosine={cos:.6} < 0.99 at step {}/{}",
                    probe_step + 1, multi_step_probe
                )));
            }

            // Advance both paths via CPU argmax (deterministic).
            let mut best_idx: u32 = 0;
            let mut best_val = f32::NEG_INFINITY;
            for (i, &v) in cpu_logits.iter().enumerate() {
                if v > best_val {
                    best_val = v;
                    best_idx = i as u32;
                }
            }
            probe_token = best_idx;
        }
    }

    // Autoregressive generation
    let mut output_tokens = input_tokens.to_vec();
    let stop_tokens = &gen_config.stop_tokens;

    for step in 0..gen_config.max_tokens {
        let token_id = *output_tokens.last().unwrap();
        let position = output_tokens.len() - 1;
        let seq_len_before = if step == 0 { 0 } else { position };

        // Forward pass through all layers
        let mut hidden = model.embed(&[token_id]);
        for layer_idx in 0..num_layers {
            let prefix = format!("layer.{layer_idx}");
            let (ref mut kv_k, ref mut kv_v) = kv_caches[layer_idx];
            fwd.forward_layer(
                &mut hidden, &prefix, position, kv_k, kv_v,
            ).map_err(|e| RealizarError::InferenceError(format!("wgpu layer {layer_idx}: {e}")))?;
        }

        // Output norm + LM head (CPU — small cost)
        let sq_sum: f32 = hidden.iter().map(|x| x * x).sum();
        let rms = (sq_sum / hidden.len() as f32 + eps).sqrt();
        let normed: Vec<f32> = hidden.iter().zip(output_norm.iter())
            .map(|(x, g)| (x / rms) * g)
            .collect();

        // Argmax (greedy)
        let mut best_idx = 0u32;
        let mut best_val = f32::NEG_INFINITY;
        for i in 0..vocab_size {
            let row = &lm_head_f32[i * hidden_dim..(i + 1) * hidden_dim];
            let logit: f32 = row.iter().zip(normed.iter()).map(|(w, x)| w * x).sum();
            if logit > best_val {
                best_val = logit;
                best_idx = i as u32;
            }
        }

        output_tokens.push(best_idx);

        if stop_tokens.contains(&best_idx) {
            break;
        }
    }

    Ok((output_tokens, true)) // true = used GPU (wgpu)
}

/// Try GGUF GPU generation. Takes model by value to avoid expensive clone (~1GB).
/// Returns `Ok(result)` on GPU success, `Err(model)` to return model for CPU fallback.
#[cfg(feature = "cuda")]
fn try_gguf_gpu_generate(
    model: crate::gguf::OwnedQuantizedModel,
    input_tokens: &[u32],
    gen_config: &crate::gguf::QuantizedGenerateConfig,
    verbose: bool,
) -> std::result::Result<Result<(Vec<u32>, bool)>, Box<crate::gguf::OwnedQuantizedModel>> {
    use crate::gguf::OwnedQuantizedModelCuda;

    let mut cuda_model = match OwnedQuantizedModelCuda::with_max_seq_len(model, 0, 2048) {
        Ok(m) => m,
        Err(e) => {
            if verbose {
                eprintln!("Backend: CPU (GPU unavailable: {})", e);
            }
            // Model is preserved inside CudaInitError for CPU fallback.
            // Boxed to keep the `Err` variant small (clippy::result_large_err).
            return Err(Box::new(e.into_model()));
        },
    };

    if verbose {
        eprintln!(
            "Backend: GPU ({}, {} MB VRAM)",
            cuda_model.device_name(),
            cuda_model.vram_mb()
        );
    }

    if !validate_gpu_first_token(&mut cuda_model, gen_config, input_tokens) {
        // Validation failed — extract model back for CPU fallback
        return Err(Box::new(cuda_model.into_model()));
    }

    // Reuse existing CUDA model — generate_gpu_resident() creates fresh KV cache
    // and resets GPU KV positions internally, so validation doesn't "consume" it.
    let result = cuda_model
        .generate_gpu_resident(input_tokens, gen_config)
        .map(|tokens| (tokens, true))
        .map_err(|e| RealizarError::InferenceError(format!("GPU generation failed: {}", e)));
    Ok(result)
}

/// Run GGUF generation with GPU or CPU
#[allow(unused_variables)] // config used only in CUDA feature
fn run_gguf_generate(
    model: crate::gguf::OwnedQuantizedModel,
    input_tokens: &[u32],
    gen_config: &crate::gguf::QuantizedGenerateConfig,
    config: &InferenceConfig,
) -> Result<(Vec<u32>, bool)> {
    // M32c.2.1: short-circuit MoE forward attempts BEFORE any GPU/CPU
    // dispatch. M32c.2 made `from_gguf` succeed for qwen3_moe by routing
    // to `from_gguf_for_moe` (which leaves dense FFN tensor refs as
    // zero-byte placeholders). Without this guard, the wgpu/CUDA forward
    // path tries to bind those zero-byte buffers and panics deep in
    // `wgpu_core::create_bind_group` with `Buffer with 'layer.0.up_proj'
    // label binding size is zero`. M32c.2.2 will replace this guard with
    // an actual MoE forward via `moe_forward_token`. See
    // contracts/qwen3-moe-forward-v1.yaml § FALSIFY-QW3-MOE-FORWARD-003.
    let canonical_arch =
        crate::tensor_names::normalize_architecture(&model.config.architecture);
    if canonical_arch == "qwen3_moe" {
        return Err(RealizarError::UnsupportedOperation {
            operation: "moe_forward_dispatch".to_string(),
            reason: format!(
                "Architecture '{}' (canonical 'qwen3_moe') uses Mixture-of-Experts FFN. \
                 Load step succeeded via QuantizedGGUFTransformer::from_gguf_for_moe (M32c.2) \
                 with all 4 contract-declared MoE tensors per layer present, but the \
                 forward dispatch is not yet wired to moe_forward_token in \
                 gpu/scheduler/moe_dispatch.rs. Tracked under contract qwen3-moe-forward-v1 \
                 (M32 staged plan: M32a/b/c.1/c.2 SHIPPED; M32c.2.1 forward-refusal \
                 IN PROGRESS; M32c.2.2 forward-wiring + M32d numerical parity PENDING). \
                 See contracts/qwen3-moe-forward-v1.yaml.",
                model.config.architecture
            ),
        });
    }

    let has_legacy_quant = model_has_legacy_quant(&model);

    // GPU path: pass model by value (zero-clone) — model is returned on failure for CPU fallback
    #[cfg(feature = "cuda")]
    let model = if !config.no_gpu && !has_legacy_quant {
        match try_gguf_gpu_generate(model, input_tokens, gen_config, config.verbose) {
            Ok(result) => return result,
            Err(returned_model) => *returned_model, // GPU failed, use returned model for CPU
        }
    } else {
        model
    };

    // GH-559: wgpu fallback — try Vulkan compute before CPU.
    // Proven: wgpu cosine=0.999863 on Blackwell sm_121 where CUDA JIT fails.
    #[cfg(feature = "gpu")]
    if !config.no_gpu && !has_legacy_quant {
        match try_wgpu_generate(&model, input_tokens, gen_config, config.verbose) {
            Ok(result) => return Ok(result),
            Err(e) => {
                if config.verbose {
                    eprintln!("Backend: CPU (wgpu unavailable: {})", e);
                }
            }
        }
    }

    log_cpu_backend(config.verbose, has_legacy_quant);
    let tokens = model
        .generate_with_cache(input_tokens, gen_config)
        .map_err(|e| RealizarError::InferenceError(format!("CPU generation failed: {}", e)))?;
    Ok((tokens, false))
}

/// Run APR model inference (PAR-302, PMAT-APR-CUDA-001)
///
/// Uses AprV2ModelCuda for GPU acceleration when available, falls back to
/// AprTransformer (CPU with proper RoPE and SwiGLU) otherwise.
/// PMAT-237: APR inference now uses PreparedTokens (compile-time enforced chat template).
/// Previously bypassed PreparedTokens entirely via prepare_apr_input_tokens().
fn run_apr_inference(
    config: &InferenceConfig,
    prepared: &PreparedTokens,
) -> Result<InferenceResult> {
    if config.verbose {
        eprintln!("Loading APR model: {}", config.model_path.display());
    }

    let load_start = Instant::now();
    let input_tokens = prepared.tokens();
    let input_token_count = prepared.input_count();

    // Try GPU path first
    #[cfg(feature = "cuda")]
    if !config.no_gpu {
        if let Some(result) =
            try_apr_cuda_inference(config, input_tokens, input_token_count, load_start)
        {
            return result;
        }
    }

    // GH-559: wgpu fallback for APR models — try Vulkan before CPU.
    #[cfg(feature = "gpu")]
    if !config.no_gpu {
        match try_apr_wgpu_inference(config, input_tokens, input_token_count, load_start) {
            Some(Ok(result)) => return Ok(result),
            Some(Err(e)) => {
                if config.verbose {
                    eprintln!("Backend: CPU (wgpu failed: {})", e);
                }
            }
            None => {
                if config.verbose {
                    eprintln!("Backend: CPU (wgpu not available)");
                }
            }
        }
    }

    // CPU fallback: AprTransformer with RoPE and SwiGLU
    run_apr_cpu_inference(config, input_tokens, input_token_count, load_start)
}

/// GH-559: Try wgpu (Vulkan) inference for APR models.
/// Returns None if wgpu not available, Some(Result) if attempted.
#[cfg(feature = "gpu")]
fn try_apr_wgpu_inference(
    config: &InferenceConfig,
    input_tokens: &[u32],
    input_token_count: usize,
    load_start: Instant,
) -> Option<Result<InferenceResult>> {
    use crate::apr::MappedAprModel;
    use crate::gpu::adapters::wgpu_adapter;
    use trueno::backends::gpu::GpuDevice;

    if !GpuDevice::is_available() {
        return None;
    }

    let gpu = match GpuDevice::new() {
        Ok(g) => g,
        Err(e) => {
            // FALSIFY-CPU-GPU-005: wgpu init failure is a backend-fallback decision —
            // user must see why this backend was rejected without --verbose.
            // Emit BOTH the contract-tagged prefix (greppable, locked in by
            // `wgpu_fallback_log_prefix_is_contract_tagged` test) AND the
            // existing [GH-559] tag (preserved for runbook continuity).
            eprintln!("{}, attempting fallback: {}", WGPU_FALLBACK_LOG_PREFIX, e);
            eprintln!("[GH-559] wgpu init failed: {}", e);
            return None;
        }
    };

    // FALSIFY-CPU-GPU-005: wgpu lifecycle visible without --verbose. Symmetric to
    // FALSIFY-CPU-GPU-003's CUDA-fallback log so users always know which backend
    // actually serves their tokens.
    eprintln!("Backend: wgpu (Vulkan)");

    // Load model
    let mapped = match MappedAprModel::from_path(&config.model_path) {
        Ok(m) => m,
        Err(_) => return None,
    };
    let model = match crate::gguf::OwnedQuantizedModel::from_apr(&mapped) {
        Ok(m) => m,
        Err(_) => return None,
    };

    let cfg = model.config();
    let hidden_dim = cfg.hidden_dim;
    let num_layers = cfg.num_layers;
    let num_heads = cfg.num_heads;
    let num_kv_heads = cfg.num_kv_heads;
    let head_dim = hidden_dim / num_heads;
    let intermediate_dim = cfg.intermediate_dim;
    let vocab_size = cfg.vocab_size;
    let eps = cfg.eps;
    let kv_dim = num_kv_heads * head_dim;
    // Resolve stop tokens from model config + sibling tokenizer
    let mut stop_toks: Vec<u32> = cfg.eos_token_id.into_iter().collect();
    let extra = crate::infer::resolve_apr_stop_tokens(
        cfg.eos_token_id, &[], &config.model_path,
    );
    for t in &extra {
        if !stop_toks.contains(t) { stop_toks.push(*t); }
    }
    let gen_config = crate::gguf::QuantizedGenerateConfig {
        max_tokens: config.max_tokens,
        temperature: 0.0,
        top_k: 1,
        stop_tokens: stop_toks,
        trace: false,
            ..Default::default()
    };

    // Dequantize and upload weights
    let weights = match wgpu_adapter::dequant_model_weights(&model) {
        Ok(w) => w,
        Err(e) => return Some(Err(e)),
    };

    let mut fwd = trueno::backends::gpu::WgslForwardPass::new(
        gpu.device, gpu.queue,
        hidden_dim, num_heads, num_kv_heads, head_dim, intermediate_dim,
    );

    for (name, data, _rows, _cols) in &weights {
        fwd.upload_weight(name, data);
    }
    // KV cache initialized by caller (no init_kv_cache needed — API change)

    let output_norm = model.output_norm_weight();
    let lm_head_f32: Vec<f32> = weights.iter()
        .find(|(n, _, _, _)| n == "lm_head")
        .map(|(_, d, _, _)| d.clone())
        .unwrap_or_default();

    let max_seq = gen_config.max_tokens + input_tokens.len() + 16;
    let mut kv_caches: Vec<(Vec<f32>, Vec<f32>)> = (0..num_layers)
        .map(|_| (vec![0.0f32; max_seq * kv_dim], vec![0.0f32; max_seq * kv_dim]))
        .collect();

    // FALSIFY-CPU-GPU-005 part b: wgpu cosine parity gate.
    //
    // Symmetric to FALSIFY-CPU-GPU-003's CUDA parity_gate (cuda::mod_parity_gate).
    // Runs CPU vs wgpu side-by-side for MULTI_STEP_PROBE tokens, advancing both
    // KV caches in lockstep. Cosine-compares logits at every step. If any step
    // fails the 0.99 threshold, emit `WGPU_FALLBACK_LOG_PREFIX` and return None
    // so we fall back to CPU rather than ship silent wgpu gibberish.
    //
    // **Why multi-step?** Single-step (the pre-#1864 design) caught divergence
    // on the first forward but missed autoregressive drift in the KV cache.
    // Qwen2.5-7B Q4K shipped "ampiezza"-style gibberish via wgpu in the v0.34.0
    // window because the first-token cosine was ≥ 0.99 but every subsequent
    // step diverged as the KV cache accumulated error. The multi-step gate
    // catches that without paying for a full max-tokens probe.
    //
    // **Cost.** Each extra step ~ 5-50ms on a 1.5B Q4K, ~30-200ms on 7B Q4K.
    // MULTI_STEP_PROBE=3 keeps init overhead under 1s for the common case;
    // tunable via APR_WGPU_PARITY_STEPS env var (1..16) for diagnostic runs.
    //
    // See contracts/apr-cpu-vs-gpu-output-parity-v1.yaml § FALSIFY-CPU-GPU-006
    // (multi_step_parity_gate) for the formal invariant.
    {
        const MULTI_STEP_PROBE_DEFAULT: usize = 3;
        let multi_step_probe: usize = std::env::var("APR_WGPU_PARITY_STEPS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&n| (1..=16).contains(&n))
            .unwrap_or(MULTI_STEP_PROBE_DEFAULT);

        // Reuse a single CPU cache + wgpu KV cache across all probe steps so
        // both paths see identical autoregressive state. max_seq sized to fit
        // the probe.
        let probe_max_seq = multi_step_probe + 1;
        let mut cpu_cache = crate::gguf::OwnedQuantizedKVCache::from_config(cfg, probe_max_seq);
        let mut probe_kv_caches: Vec<(Vec<f32>, Vec<f32>)> = (0..num_layers)
            .map(|_| (vec![0.0f32; probe_max_seq * kv_dim], vec![0.0f32; probe_max_seq * kv_dim]))
            .collect();
        let mut probe_token = *input_tokens.first().unwrap_or(&0);

        for step in 0..multi_step_probe {
            // CPU reference logits at this step.
            let cpu_logits = match model.forward_single_with_cache(probe_token, &mut cpu_cache, step) {
                Ok(l) => l,
                Err(e) => {
                    eprintln!(
                        "{}, attempting fallback: CPU probe step {} forward failed: {}",
                        WGPU_FALLBACK_LOG_PREFIX, step, e
                    );
                    return None;
                }
            };

            // wgpu single-step replay at the same position.
            let mut hidden = model.embed(&[probe_token]);
            for layer_idx in 0..num_layers {
                let prefix = format!("layer.{layer_idx}");
                let (ref mut kv_k, ref mut kv_v) = probe_kv_caches[layer_idx];
                if let Err(e) = fwd.forward_layer(&mut hidden, &prefix, step, kv_k, kv_v) {
                    eprintln!(
                        "{}, attempting fallback: wgpu probe step {} layer {} failed: {}",
                        WGPU_FALLBACK_LOG_PREFIX, step, layer_idx, e
                    );
                    return None;
                }
            }
            // Output norm + LM head (mirrors the autoregressive loop body).
            let sq_sum: f32 = hidden.iter().map(|x| x * x).sum();
            let rms = (sq_sum / hidden.len() as f32 + eps).sqrt();
            let normed: Vec<f32> = hidden
                .iter()
                .zip(output_norm.iter())
                .map(|(x, g)| (x / rms) * g)
                .collect();
            let mut wgpu_logits = vec![0.0_f32; vocab_size];
            for i in 0..vocab_size {
                let row = &lm_head_f32[i * hidden_dim..(i + 1) * hidden_dim];
                wgpu_logits[i] = row.iter().zip(normed.iter()).map(|(w, x)| w * x).sum();
            }

            let cos = cpu_vs_gpu_cosine_similarity(&cpu_logits, &wgpu_logits);
            if !(cos.is_finite() && cos >= 0.99) {
                eprintln!(
                    "{}, attempting fallback: cosine vs CPU = {:.6} (< 0.99) at step {}/{}",
                    WGPU_FALLBACK_LOG_PREFIX, cos, step + 1, multi_step_probe
                );
                return None;
            }

            // Advance both paths deterministically via CPU argmax.
            // (Greedy choice; matches what the autoregressive loop will do for
            // step 0 in the common case. Probe is contract verification, not
            // user-visible generation.)
            let mut best_idx: u32 = 0;
            let mut best_val = f32::NEG_INFINITY;
            for (i, &v) in cpu_logits.iter().enumerate() {
                if v > best_val {
                    best_val = v;
                    best_idx = i as u32;
                }
            }
            probe_token = best_idx;
        }
    }

    let model_load_ms = load_start.elapsed().as_millis() as f64;

    // Autoregressive generation
    let infer_start = Instant::now();
    let mut output_tokens = input_tokens.to_vec();
    let stop_tokens = &gen_config.stop_tokens;

    for step in 0..gen_config.max_tokens {
        let token_id = *output_tokens.last().unwrap();
        let position = output_tokens.len() - 1;

        let mut hidden = model.embed(&[token_id]);
        for layer_idx in 0..num_layers {
            let prefix = format!("layer.{layer_idx}");
            let (ref mut kv_k, ref mut kv_v) = kv_caches[layer_idx];
            if let Err(e) = fwd.forward_layer(&mut hidden, &prefix, position, kv_k, kv_v) {
                return Some(Err(RealizarError::InferenceError(format!("wgpu layer {layer_idx}: {e}"))));
            }
        }

        // Output norm (apply RMSNorm with output_norm gamma)
        let sq_sum: f32 = hidden.iter().map(|x| x * x).sum();
        let rms = (sq_sum / hidden.len() as f32 + eps).sqrt();
        let normed: Vec<f32> = hidden.iter().zip(output_norm.iter())
            .map(|(x, g)| (x / rms) * g)
            .collect();

        // LM head argmax (CPU matmul)
        let mut best_idx = 0u32;
        let mut best_val = f32::NEG_INFINITY;
        for i in 0..vocab_size {
            let row = &lm_head_f32[i * hidden_dim..(i + 1) * hidden_dim];
            let logit: f32 = row.iter().zip(normed.iter()).map(|(w, x)| w * x).sum();
            if logit > best_val {
                best_val = logit;
                best_idx = i as u32;
            }
        }

        output_tokens.push(best_idx);
        if stop_tokens.contains(&best_idx) { break; }
    }

    let inference_ms = infer_start.elapsed().as_millis() as f64;
    let tokens_generated = output_tokens.len() - input_token_count;

    // Decode tokens
    let text = crate::infer::decode_apr_tokens(&config.model_path, &output_tokens[input_token_count..]);

    Some(Ok(InferenceResult {
        text,
        tokens: output_tokens,
        input_token_count,
        generated_token_count: tokens_generated,
        inference_ms,
        load_ms: model_load_ms,
        tok_per_sec: if inference_ms > 0.0 { tokens_generated as f64 / (inference_ms / 1000.0) } else { 0.0 },
        format: "APR".to_string(),
        used_gpu: true,
    }))
}

/// GH-318: Map APR architecture string to chat template hint using contract.
///
/// Uses `normalize_architecture()` from tensor-names-v1.yaml — no fallback.
/// Unknown architectures default to "llama" (safest default per contract).
fn apr_arch_to_template_hint(apr_arch: &str, _model_name: &str) -> &'static str {
    crate::tensor_names::normalize_architecture(apr_arch)
}

/// Metadata captured from the model config before it is moved into CUDA.
#[cfg(feature = "cuda")]
struct AprCudaModelInfo {
    arch: String,
    num_layers: usize,
    vocab_size: usize,
    hidden_dim: usize,
}

/// Load an APR model and initialize it on CUDA, returning None on any failure.
#[cfg(feature = "cuda")]
fn load_apr_cuda_model(
    model_path: &std::path::Path,
    verbose: bool,
) -> Option<(crate::gguf::OwnedQuantizedModelCuda, AprCudaModelInfo)> {
    use crate::apr::MappedAprModel;
    use crate::gguf::{OwnedQuantizedModel, OwnedQuantizedModelCuda};

    let mapped = MappedAprModel::from_path(model_path).map_err(|e| {
        if verbose { eprintln!("[APR-CUDA] MappedAprModel::from_path failed: {}", e); }
    }).ok()?;

    let model = OwnedQuantizedModel::from_apr(&mapped).map_err(|e| {
        if verbose { eprintln!("[APR-CUDA] OwnedQuantizedModel::from_apr failed: {}", e); }
    }).ok()?;

    if model_has_legacy_quant(&model) {
        return None;
    }

    let info = AprCudaModelInfo {
        arch: model.config.architecture.clone(),
        num_layers: model.config.num_layers,
        vocab_size: model.config.vocab_size,
        hidden_dim: model.config.hidden_dim,
    };

    // FALSIFY-CPU-GPU-003: CUDA init failure (e.g. parity_gate cosine < 0.99 on
    // a broken GPU build, or ILLEGAL_ADDRESS during the gate's GPU forward) MUST
    // be visible without --verbose. Silent fallback was the SHIP-007 jidoka gap:
    // user saw downstream wgpu gibberish without ever knowing CUDA was rejected.
    let cuda_model = OwnedQuantizedModelCuda::with_max_seq_len(model, 0, 2048).map_err(|e| {
        eprintln!("{}, attempting fallback: {}", CUDA_FALLBACK_LOG_PREFIX, e);
    }).ok()?;

    Some((cuda_model, info))
}

#[cfg(feature = "cuda")]
fn log_apr_cuda_info(
    info: &AprCudaModelInfo,
    cuda_model: &crate::gguf::OwnedQuantizedModelCuda,
    load_ms: f64,
) {
    eprintln!(
        "Architecture: {} ({} layers, vocab_size={})",
        info.arch, info.num_layers, info.vocab_size
    );
    eprintln!(
        "Config: hidden_size={}, quant=CUDA+KVCache, threads=1 (GPU)",
        info.hidden_dim
    );
    eprintln!("Model loaded in {:.1}ms", load_ms);
    eprintln!(
        "Backend: GPU ({}, {} MB VRAM)",
        cuda_model.device_name(),
        cuda_model.vram_mb()
    );
}

/// Try APR CUDA inference, returning None to fall through to CPU.
///
/// Converts APR Q4K model to `OwnedQuantizedModel` and uses the proven GGUF CUDA
/// pipeline (same path as `try_gguf_gpu_generate`). The previous wgpu path used
/// `AprF32ToGpuAdapter` which only reads F32 fields — empty for Q4K models → garbage.
#[cfg(feature = "cuda")]
fn try_apr_cuda_inference(
    config: &InferenceConfig,
    input_tokens: &[u32],
    input_token_count: usize,
    load_start: Instant,
) -> Option<Result<InferenceResult>> {
    use crate::gguf::QuantizedGenerateConfig;

    let (mut cuda_model, info) = load_apr_cuda_model(&config.model_path, config.verbose)?;

    let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;

    if config.verbose {
        log_apr_cuda_info(&info, &cuda_model, load_ms);
    }
    eprintln!("[GH-480-TRACE] try_apr_cuda_inference: model loaded OK, about to resolve stop tokens");

    // GH-373: EOS from model config + caller stop tokens + sibling tokenizer
    let stop_tokens = resolve_apr_stop_tokens(
        cuda_model.model().config.eos_token_id,
        &config.stop_tokens,
        &config.model_path,
    );
    let gen_config = QuantizedGenerateConfig {
        max_tokens: config.max_tokens,
        temperature: 0.0,
        top_k: 1,
        stop_tokens,
        trace: false,
            ..Default::default()
    };

    eprintln!("[GH-480] F2 validation starting...");
    if !validate_gpu_first_token(&mut cuda_model, &gen_config, input_tokens) {
        eprintln!("[GH-480] F2 validation FAILED — falling back to CPU");
        return None;
    }
    eprintln!("[GH-480] F2 validation PASSED — launching GPU generation");

    let infer_start = Instant::now();

    let tokens = match cuda_model.generate_gpu_resident(input_tokens, &gen_config) {
        Ok(t) => t,
        Err(e) => {
            let msg = e.to_string();
            eprintln!("[GH-480] generate_gpu_resident FAILED: {msg}");
            // GH-278: Fall back to CPU for unsupported architectures (GPT-2 has no SwiGLU/RMSNorm)
            if msg.contains("not supported") || msg.contains("architecture") {
                if config.verbose {
                    eprintln!("[APR-CUDA] GPU-resident not supported, falling back to CPU: {msg}");
                }
                return None;
            }
            return Some(Err(RealizarError::InferenceError(format!(
                "GPU generation failed: {}",
                e
            ))));
        },
    };

    let inference_ms = infer_start.elapsed().as_secs_f64() * 1000.0;
    let generated_tokens = &tokens[input_token_count..];
    let text = decode_apr_tokens(&config.model_path, generated_tokens);
    let generated_token_count = generated_tokens.len();

    Some(Ok(InferenceResult {
        text,
        tokens,
        input_token_count,
        generated_token_count,
        inference_ms,
        tok_per_sec: tok_per_sec(generated_token_count, inference_ms),
        load_ms,
        format: "APR".to_string(),
        used_gpu: true,
    }))
}

/// Run APR inference on CPU.
///
/// GH-479: Delegates unconditionally to `run_apr_quantized_cpu_inference`,
/// which uses `OwnedQuantizedModel` with per-tensor scratch dequant (GH-478).
/// The previous F32 `AprTransformer` path required eager dequant of the entire
/// model (peak memory ≈ file_size × 8) and has been removed.
fn run_apr_cpu_inference(
    config: &InferenceConfig,
    input_tokens: &[u32],
    input_token_count: usize,
    load_start: Instant,
) -> Result<InferenceResult> {
    run_apr_quantized_cpu_inference(config, input_tokens, input_token_count, load_start)
}

/// GH-278: CPU inference for APR models using OwnedQuantizedModel
///
/// Used for architectures not supported by AprTransformer (GPT-2, etc.).
/// AprTransformer only supports LLaMA-style (RoPE + SwiGLU).
fn run_apr_quantized_cpu_inference(
    config: &InferenceConfig,
    input_tokens: &[u32],
    input_token_count: usize,
    load_start: Instant,
) -> Result<InferenceResult> {
    use crate::apr::MappedAprModel;
    use crate::gguf::{OwnedQuantizedModel, QuantizedGenerateConfig};

    let mapped = MappedAprModel::from_path(&config.model_path)?;
    let model = OwnedQuantizedModel::from_apr(&mapped)?;
    let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;

    if config.verbose {
        eprintln!(
            "Architecture: {} ({} layers, vocab_size={})",
            model.config.architecture, model.config.num_layers, model.config.vocab_size
        );
        eprintln!(
            "Config: hidden_size={}, quant=Q4_K (OwnedQuantizedModel CPU), threads={}",
            model.config.hidden_dim,
            rayon::current_num_threads()
        );
        eprintln!("Model loaded in {:.1}ms", load_ms);
        eprintln!("Backend: CPU (OwnedQuantizedModel fallback for non-LLaMA arch)");
    }

    // GH-373: Resolve stop tokens for quantized path
    let stop_tokens = resolve_apr_stop_tokens(
        model.config.eos_token_id,
        &config.stop_tokens,
        &config.model_path,
    );

    let gen_config = QuantizedGenerateConfig {
        max_tokens: config.max_tokens,
        temperature: config.temperature,
        top_k: config.top_k,
        stop_tokens,
        trace: config.trace,
        // PMAT-818: forward repetition penalty on this GPU-path fallback too.
        repeat_penalty: config.repeat_penalty,
        repeat_last_n: config.repeat_last_n,
            ..Default::default()
    };

    let infer_start = Instant::now();
    let tokens = model.generate_with_cache(input_tokens, &gen_config)?;
    let inference_ms = infer_start.elapsed().as_secs_f64() * 1000.0;
    let generated_tokens = &tokens[input_token_count..];
    let text = decode_apr_tokens(&config.model_path, generated_tokens);
    let generated_token_count = generated_tokens.len();

    Ok(InferenceResult {
        text,
        tokens,
        input_token_count,
        generated_token_count,
        inference_ms,
        tok_per_sec: tok_per_sec(generated_token_count, inference_ms),
        load_ms,
        format: "APR".to_string(),
        used_gpu: false,
    })
}

/// GH-373: Resolve stop tokens from model config, caller, and sibling tokenizer.
///
/// Merges EOS tokens from three sources:
/// 1. Model config (`eos_token_id` from APR/GGUF metadata)
/// 2. Caller-provided stop tokens (`InferenceConfig.stop_tokens`)
/// 3. Sibling tokenizer (ChatML markers like `<|im_end|>`, `<|endoftext|>`)
fn resolve_apr_stop_tokens(
    model_eos: Option<u32>,
    caller_stop_tokens: &[u32],
    model_path: &std::path::Path,
) -> Vec<u32> {
    let mut tokens: Vec<u32> = model_eos.into_iter().collect();

    // Caller-provided stop tokens
    for &t in caller_stop_tokens {
        if !tokens.contains(&t) {
            tokens.push(t);
        }
    }

    // Sibling tokenizer fallback (GH-373)
    if tokens.is_empty() {
        tokens = resolve_stop_tokens_from_tokenizer(model_path);
    }

    tokens
}

/// Load stop tokens from sibling tokenizer.json (GH-373 helper)
fn resolve_stop_tokens_from_tokenizer(model_path: &std::path::Path) -> Vec<u32> {
    let tokenizer = match crate::apr::AprV2Model::load_tokenizer(model_path) {
        Some(t) => t,
        None => return Vec::new(),
    };

    let mut tokens: Vec<u32> = tokenizer.eos_id.into_iter().collect();

    // ChatML stop tokens for instruct models
    for marker in &["<|im_end|>", "<|endoftext|>"] {
        let id = tokenizer
            .special_tokens
            .get(*marker)
            .or_else(|| tokenizer.token_to_id.get(*marker));
        if let Some(&id) = id {
            if !tokens.contains(&id) {
                tokens.push(id);
            }
        }
    }

    tokens
}

/// Decode APR output tokens using available tokenizer (GH-156)
fn decode_apr_tokens(model_path: &std::path::Path, tokens: &[u32]) -> String {
    use crate::apr::AprV2Model;

    let text = if let Some(tokenizer) = AprV2Model::load_tokenizer(model_path) {
        tokenizer.decode(tokens)
    } else if let Some(tokenizer) = find_fallback_tokenizer(model_path) {
        tokenizer.decode(tokens)
    } else {
        format!("[{} tokens generated, tokenizer not found]", tokens.len())
    };
    clean_model_output(&text)
}

/// Compute tokens per second from count and elapsed milliseconds
fn tok_per_sec(count: usize, ms: f64) -> f64 {
    if ms > 0.0 {
        count as f64 / (ms / 1000.0)
    } else {
        0.0
    }
}

/// Run SafeTensors model inference (PAR-301, PMAT-129)
///
/// PMAT-236: Accepts `PreparedTokens` (compile-time enforced chat template).
/// Previously, this function raw-encoded prompts WITHOUT chat template,
/// producing garbage output for instruct models.
fn run_safetensors_inference(
    config: &InferenceConfig,
    prepared: &PreparedTokens,
) -> Result<InferenceResult> {
    if config.verbose {
        eprintln!("Loading SafeTensors model: {}", config.model_path.display());
    }

    // PMAT-236: Use PreparedTokens (chat template already applied by prepare_tokens)
    let input_tokens = prepared.tokens().to_vec();
    let input_token_count = prepared.input_count();

    // PMAT-129: Try GPU path first
    #[cfg(feature = "cuda")]
    if !config.no_gpu {
        if let Some(result) =
            try_safetensors_cuda_inference(config, &input_tokens, input_token_count)
        {
            return result;
        }
    }

    // CPU fallback: SafeTensors → AprTransformer conversion
    run_safetensors_cpu_inference(config, &input_tokens, input_token_count)
}

/// Try SafeTensors CUDA inference, returning None to fall through to CPU
#[cfg(feature = "cuda")]
fn try_safetensors_cuda_inference(
    config: &InferenceConfig,
    input_tokens: &[u32],
    input_token_count: usize,
) -> Option<Result<InferenceResult>> {
    use crate::safetensors_cuda::SafeTensorsCudaModel;

    let load_start = Instant::now();
    let mut cuda_model = match SafeTensorsCudaModel::load(&config.model_path, 0) {
        Ok(m) => m,
        Err(e) => {
            if config.verbose {
                eprintln!("Backend: CPU (GPU init failed: {})", e);
            }
            return None;
        },
    };

    let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;

    if config.verbose {
        eprintln!(
            "Architecture: SafeTensors ({} layers, vocab_size={})",
            cuda_model.config().num_layers,
            cuda_model.config().vocab_size
        );
        eprintln!(
            "Config: hidden_size={}, context_length={}, quant=F16/BF16, threads=1 (GPU)",
            cuda_model.config().hidden_dim,
            cuda_model.config().context_length
        );
        eprintln!("Model loaded in {:.1}ms", load_ms);
        eprintln!(
            "Backend: GPU ({}, {} MB VRAM)",
            cuda_model.device_name(),
            cuda_model.vram_mb()
        );
    }

    let infer_start = Instant::now();
    // GH-330: EOS from model config (Design by Contract)
    let eos_id = cuda_model.config().eos_token_id.unwrap_or(0);
    let tokens = match cuda_model.generate(input_tokens, config.max_tokens, eos_id) {
        Ok(t) => t,
        Err(e) => {
            return Some(Err(RealizarError::InferenceError(format!(
                "GPU generation failed: {}",
                e
            ))))
        },
    };

    let inference_ms = infer_start.elapsed().as_secs_f64() * 1000.0;
    let generated_tokens = &tokens[input_token_count..];
    let text = decode_apr_tokens(&config.model_path, generated_tokens);
    let generated_token_count = generated_tokens.len();

    Some(Ok(InferenceResult {
        text,
        tokens,
        input_token_count,
        generated_token_count,
        inference_ms,
        tok_per_sec: tok_per_sec(generated_token_count, inference_ms),
        load_ms,
        format: "SafeTensors".to_string(),
        used_gpu: true,
    }))
}

#[cfg(test)]
mod tests {
    use super::{CUDA_FALLBACK_LOG_PREFIX, WGPU_FALLBACK_LOG_PREFIX};

    /// Drift-prevention for FALSIFY-CPU-GPU-003 (PR #1428):
    /// the user-visible eprintln tag MUST start with the contract ID so that
    /// `apr run` users (without --verbose) see exactly which backend was rejected
    /// rather than silent gibberish from a downstream fallback.
    ///
    /// If this assertion ever fails, do NOT loosen it — re-read
    /// `evidence/ship-007-layer-0-oracle-bisection-2026-05-03/findings-v6-parity-gate-fires-but-fallback-is-silent.md`
    /// and either keep the tag stable or bump the parity contract before changing
    /// the wire format.
    #[test]
    fn cuda_fallback_log_prefix_is_contract_tagged() {
        assert!(
            CUDA_FALLBACK_LOG_PREFIX.starts_with("[apr-cpu-vs-gpu-output-parity-v1]"),
            "FALSIFY-CPU-GPU-003 jidoka tag was renamed; bump contract version first. \
             Got: {CUDA_FALLBACK_LOG_PREFIX}"
        );
        assert!(
            CUDA_FALLBACK_LOG_PREFIX.contains("CUDA path rejected"),
            "fallback message must say which backend was rejected; got: {CUDA_FALLBACK_LOG_PREFIX}"
        );
    }

    /// Drift-prevention for FALSIFY-CPU-GPU-005 (contract v1.2.0):
    /// symmetric to the CUDA tag above, the wgpu fallback log MUST also be
    /// contract-tagged so `apr run` users see WHICH backend was rejected when
    /// CUDA falls through to wgpu and wgpu itself is rejected (e.g. broken
    /// GPU build, no Vulkan ICD, etc.).
    ///
    /// Same regression class as #1428→#1429: a future refactor could rename
    /// the tag, drop the contract ID prefix, or revert to verbose-only —
    /// each silently re-introduces the silent-gibberish loophole the
    /// `apr-cpu-vs-gpu-output-parity-v1` chain was authored to close.
    #[test]
    fn wgpu_fallback_log_prefix_is_contract_tagged() {
        assert!(
            WGPU_FALLBACK_LOG_PREFIX.starts_with("[apr-cpu-vs-gpu-output-parity-v1]"),
            "FALSIFY-CPU-GPU-005 jidoka tag was renamed; bump contract version first. \
             Got: {WGPU_FALLBACK_LOG_PREFIX}"
        );
        assert!(
            WGPU_FALLBACK_LOG_PREFIX.contains("wgpu path rejected"),
            "fallback message must say which backend was rejected; got: {WGPU_FALLBACK_LOG_PREFIX}"
        );
    }

    /// Symmetry guard: CUDA and wgpu prefixes must share the same contract
    /// tag and structure (`[CONTRACT_ID] <backend> path rejected`). If they
    /// diverge, the user-facing log format becomes inconsistent across
    /// fallback hops and grep recipes break. Locks in the symmetry that
    /// PR #1428 (CUDA) and this PR (wgpu) explicitly established.
    #[test]
    fn cuda_and_wgpu_fallback_log_prefixes_share_contract_tag() {
        let contract_tag = "[apr-cpu-vs-gpu-output-parity-v1]";
        assert!(CUDA_FALLBACK_LOG_PREFIX.starts_with(contract_tag));
        assert!(WGPU_FALLBACK_LOG_PREFIX.starts_with(contract_tag));
        assert!(CUDA_FALLBACK_LOG_PREFIX.ends_with("path rejected"));
        assert!(WGPU_FALLBACK_LOG_PREFIX.ends_with("path rejected"));
    }

    /// FALSIFY-CPU-GPU-005 part b cosine helper — parallel vectors return 1.
    ///
    /// Locks in the gate's positive case: when wgpu produces logits identical
    /// to CPU, the gate must NOT trigger fallback (cosine = 1.0 ≥ 0.99 floor).
    #[test]
    fn cpu_vs_gpu_cosine_similarity_parallel_returns_one() {
        let a = vec![1.0_f32, 2.0, 3.0, 4.0];
        let b = a.clone();
        let cos = super::cpu_vs_gpu_cosine_similarity(&a, &b);
        assert!(
            (cos - 1.0).abs() < 1e-6,
            "parallel vectors must yield cosine 1.0, got {cos}"
        );
    }

    /// FALSIFY-CPU-GPU-005 part b cosine helper — orthogonal returns 0.
    ///
    /// Negative case: orthogonal vectors must yield cosine 0.0 which is well
    /// below the 0.99 gate floor → fallback triggers.
    #[test]
    fn cpu_vs_gpu_cosine_similarity_orthogonal_returns_zero() {
        let a = vec![1.0_f32, 0.0, 0.0, 0.0];
        let b = vec![0.0_f32, 1.0, 0.0, 0.0];
        let cos = super::cpu_vs_gpu_cosine_similarity(&a, &b);
        assert!(
            cos.abs() < 1e-6,
            "orthogonal vectors must yield cosine 0.0, got {cos}"
        );
    }

    /// FALSIFY-CPU-GPU-005 part b cosine helper — fail-closed on bad input.
    ///
    /// Zero-norm or mismatched-length inputs MUST return 0.0 so the future gate
    /// triggers fallback rather than dividing by zero or panicking. This is the
    /// "conservative default" that closes the silent-gibberish loophole even
    /// when the probe forward itself emits NaN/zeros.
    #[test]
    fn cpu_vs_gpu_cosine_similarity_fails_closed() {
        // Zero-norm input
        let zero = vec![0.0_f32; 4];
        let nonzero = vec![1.0_f32, 2.0, 3.0, 4.0];
        assert_eq!(
            super::cpu_vs_gpu_cosine_similarity(&zero, &nonzero),
            0.0,
            "zero-norm input must fail closed"
        );
        // Length mismatch
        let short = vec![1.0_f32, 2.0];
        let long = vec![1.0_f32, 2.0, 3.0, 4.0];
        assert_eq!(
            super::cpu_vs_gpu_cosine_similarity(&short, &long),
            0.0,
            "length mismatch must fail closed"
        );
        // Empty input
        let empty: Vec<f32> = Vec::new();
        assert_eq!(
            super::cpu_vs_gpu_cosine_similarity(&empty, &empty),
            0.0,
            "empty input must fail closed"
        );
    }
}
