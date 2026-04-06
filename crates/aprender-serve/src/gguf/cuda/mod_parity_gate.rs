
/// Run the load-time parity gate.
///
/// Processes BOS token (ID=1) through both CPU and GPU forward passes.
/// Compares the resulting logit vectors via cosine similarity.
///
/// # Errors
///
/// Returns `RealizarError` if GPU and CPU produce divergent logits.
fn parity_gate(cuda_model: &mut OwnedQuantizedModelCuda) -> Result<()> {
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

    if cosine >= PARITY_GATE_COSINE_MIN {
        if verbose() {
            eprintln!(
                "[PARITY-GATE] PASS: cosine={:.6} (threshold={:.2})",
                cosine, PARITY_GATE_COSINE_MIN,
            );
        }
        Ok(())
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
