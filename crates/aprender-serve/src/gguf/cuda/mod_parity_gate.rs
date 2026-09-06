
/// Run the load-time parity gate.
///
/// Processes BOS token (ID=1) through both CPU and GPU forward passes.
/// Compares the resulting logit vectors via cosine similarity.
///
/// # Errors
///
/// Returns `RealizarError` if GPU and CPU produce divergent logits.
/// Returns the cosine the gate measured on PASS (REG-15: the record travels with the model).
fn parity_gate(cuda_model: &mut OwnedQuantizedModelCuda) -> Result<f32> {
    // GH-280: The capability gate in with_max_seq_len() now prevents models
    // with unsupported ops (e.g., QkNorm) from reaching this point.
    // If parity gate runs, the model MUST be fully supported by GPU.

    // Extract config values before any mutable borrows
    let hidden_dim = cuda_model.model.config.hidden_dim;
    let num_heads = cuda_model.model.config.num_heads;
    let num_kv_heads = cuda_model.model.config.num_kv_heads;
    let head_dim = if num_heads > 0 {
        hidden_dim / num_heads
    } else {
        0
    };
    let kv_dim = num_kv_heads * head_dim;
    let num_layers = cuda_model.model.config.num_layers;

    // Use architecture-aware BOS token from GGUFConfig (which applies
    // default_bos_for_architecture fallback for weights-only GGUFs).
    // Falls back to 1 only for architectures with no known BOS.
    let token_id: u32 = cuda_model.model.config.bos_token_id.unwrap_or(1);
    let position: usize = 0;

    // Independent KV caches
    let mut cpu_cache = OwnedQuantizedKVCache::new(num_layers, kv_dim, 2);
    let mut gpu_cache = OwnedQuantizedKVCache::new(num_layers, kv_dim, 2);
    cuda_model.executor.reset_kv_cache_gpu();

    // CPU forward
    let cpu_logits = cuda_model
        .model
        .forward_single_with_cache(token_id, &mut cpu_cache, position)
        .map_err(|e| {
            RealizarError::InferenceError(format!("PARITY-GATE: CPU forward failed: {e}"))
        })?;

    // SHIP-007 PR-B: Dump CPU logits to <APR_GPU_STAGE_DUMP>/cpu/lm_head.bin
    // for direct CPU-vs-GPU comparison on the SAME single BOS token at
    // position 0. The GPU side dumps to <APR_GPU_STAGE_DUMP>/lm_head.bin
    // inside forward_gpu_resident.
    if let Some(cfg) =
        crate::inference_trace::gpu_stage_dump::GpuStageDumpConfig::from_env()
    {
        let cpu_dir = cfg.output_dir().join("cpu");
        let cpu_cfg =
            crate::inference_trace::gpu_stage_dump::GpuStageDumpConfig::with_output_dir(
                &cpu_dir,
            );
        if let Err(e) = crate::inference_trace::gpu_stage_dump::maybe_dump_host_buffer(
            Some(&cpu_cfg),
            crate::inference_trace::save_tensor_stage::SaveTensorStage::LmHead,
            0,
            &cpu_logits,
        ) {
            eprintln!("[SHIP-007-PR-B] CPU logits dump failed (non-fatal): {e}");
        }
    }

    // GPU forward
    let gpu_logits = cuda_model
        .forward_gpu_resident(token_id, &mut gpu_cache, position)
        .map_err(|e| {
            RealizarError::InferenceError(format!("PARITY-GATE: GPU forward failed: {e}"))
        })?;

    // Cosine similarity — the single metric that catches completely wrong computation
    let cosine = cosine_similarity(&cpu_logits, &gpu_logits);

    // Reset KV caches so the model starts fresh for actual inference
    cuda_model.executor.reset_kv_cache_gpu();

    // PMAT-798: If the default (fused HW DP4A) path narrowly misses parity,
    // retry once on the high-precision (float MWV, non-fused) FFN path before
    // failing closed. The fused gate+up+SwiGLU kernel quantizes activations to
    // Q8_1, which costs ~1.5% first-token cosine on LLaMA-NORM models with
    // massive activations (e.g. TinyLlama: 0.972 fused → 0.990 float). Without
    // this retry such models are wrongly pushed off the GPU. Only retry when we
    // are close (cosine already in [0.90, gate)) — a genuinely broken GPU forward
    // (cosine < 0.5) is NOT rescued by changing the FFN precision.
    let cosine = if cosine < PARITY_GATE_COSINE_MIN && cosine >= 0.90 {
        let changed = cuda_model.executor.force_high_precision_ffn();
        if changed {
            if verbose() {
                eprintln!(
                    "[PARITY-GATE] fused gate+up+SwiGLU FFN cosine={:.6} < {:.2}; retrying on unfused FFN path",
                    cosine, PARITY_GATE_COSINE_MIN,
                );
            }
            cuda_model.executor.reset_kv_cache_gpu();
            let mut cpu_cache2 = OwnedQuantizedKVCache::new(num_layers, kv_dim, 2);
            let mut gpu_cache2 = OwnedQuantizedKVCache::new(num_layers, kv_dim, 2);
            let cpu_logits2 = cuda_model
                .model
                .forward_single_with_cache(token_id, &mut cpu_cache2, position)
                .map_err(|e| {
                    RealizarError::InferenceError(format!("PARITY-GATE: CPU forward (retry) failed: {e}"))
                })?;
            let gpu_logits2 = cuda_model
                .forward_gpu_resident(token_id, &mut gpu_cache2, position)
                .map_err(|e| {
                    RealizarError::InferenceError(format!("PARITY-GATE: GPU forward (retry) failed: {e}"))
                })?;
            let cosine2 = cosine_similarity(&cpu_logits2, &gpu_logits2);
            cuda_model.executor.reset_kv_cache_gpu();
            cosine2
        } else {
            cosine
        }
    } else {
        cosine
    };

    if cosine >= PARITY_GATE_COSINE_MIN {
        if verbose() {
            eprintln!(
                "[PARITY-GATE] PASS: cosine={:.6} (threshold={:.2})",
                cosine, PARITY_GATE_COSINE_MIN,
            );
        }
        Ok(cosine)
    } else {
        // Compute additional diagnostics for the error message
        let cpu_argmax = cpu_logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(i, _)| i);
        let gpu_argmax = gpu_logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(i, _)| i);
        let max_diff = cpu_logits
            .iter()
            .zip(gpu_logits.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        Err(RealizarError::InferenceError(format!(
            "PARITY-GATE FAILED: GPU computes a DIFFERENT function than CPU.\n\
             \n\
             Cosine similarity: {cosine:.6} (required: ≥{PARITY_GATE_COSINE_MIN:.2})\n\
             CPU argmax: {cpu_argmax} | GPU argmax: {gpu_argmax}\n\
             Max absolute logit difference: {max_diff:.4}\n\
             \n\
             This model's dimensions (hidden={hidden_dim}, heads={num_heads}, kv_heads={num_kv_heads}) cause\n\
             GPU forward pass to diverge from CPU. The GPU CANNOT serve this model.\n\
             \n\
             Run `apr parity <model>` for full SPC diagnosis.\n\
             Set SKIP_PARITY_GATE=1 to bypass (for debugging only).",
        )))
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot: f64 = 0.0;
    let mut norm_a: f64 = 0.0;
    let mut norm_b: f64 = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        let x = *x as f64;
        let y = *y as f64;
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

impl OwnedQuantizedModelCuda {
    /// REG-15 (#2971): the one admission decision at load — the gate runs unless it is
    /// skipped (`SKIP_PARITY_GATE`, recorded as an override) or does not apply (MoE); a
    /// PASS records the cosine on the model, a failure refuses to construct it.
    fn admit_by_parity_gate(mut self, skip_gate: bool) -> std::result::Result<Self, CudaInitError> {
        if skip_gate {
            self.parity = ParityGateRecord::skipped();
            return Ok(self);
        }
        if self.model.config.constraints.is_moe {
            self.parity = ParityGateRecord::not_run(
                "MoE architecture: the dense load-time gate does not apply (qwen3_moe_gpu_parity.rs covers it)",
            );
            return Ok(self);
        }
        match parity_gate(&mut self) {
            Ok(cosine) => {
                self.parity = ParityGateRecord::passed(cosine);
                Ok(self)
            }
            Err(e) => Err(CudaInitError {
                error: e,
                model: Box::new(self.into_model()),
            }),
        }
    }
}
