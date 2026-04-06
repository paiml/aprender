/// GH-560: wgpu (Vulkan) batch inference state.
///
/// Toyota Way: lazy mmap for embedding (zero-copy), no model.clone().
/// Dequantized weights uploaded to GPU once, reused across all prompts.
/// Embedding lookup uses mmap'd model file — zero heap allocation.
///
/// Five-whys root cause (OOM, GH-560):
///   1. Why OOM? 56 GB peak during init
///   2. Why 56 GB? dequant_model_weights() called TWICE (28 GB each)
///   3. Why twice? First for upload loop, second for lm_head extraction
///   4. Why separate? into_iter() consumed the Vec, couldn't find lm_head after
///   5. Why not extract inline? Oversight — fixed by extracting during iteration
///
/// Five-whys root cause (garbage output, GH-560):
///   1. Why garbage? Attention produces near-uniform scores → random tokens
///   2. Why uniform? KV cache has max_seq zero-vectors before actual K/V data
///   3. Why zeros? generate() zeroed cache contents but didn't clear length
///   4. Why wrong length? Pre-allocated vec![0.0; max_seq * kv_dim] starts at full length
///   5. Why full length? forward_layer uses extend_from_slice + len()/kv_dim for seq_len
///   - Fix: Vec::with_capacity (empty, pre-alloc) + clear() (reset length to 0)
///
// Included from batch.rs via include!() — shares the same module scope.
//
/// Holds wgpu forward pass + CPU LM head for batch inference.
#[cfg(feature = "gpu")]
struct WgpuBatchState {
    fwd: trueno::backends::gpu::WgslForwardPass,
    model_path: std::path::PathBuf,
    hidden_dim: usize,
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
    vocab_size: usize,
    eps: f32,
    lm_head_f32: Vec<f32>,
    output_norm: Vec<f32>,
    kv_caches: Vec<(Vec<f32>, Vec<f32>)>,
}

#[cfg(feature = "gpu")]
impl WgpuBatchState {
    /// Generate tokens from input using wgpu forward pass.
    /// Contract: output must match CPU forward pass (cosine >= 0.98).
    #[provable_contracts_macros::contract("gpu-multi-backend-parity-v1", equation = "multi_backend_parity")]
    fn generate(
        &mut self,
        input_tokens: &[u32],
        config: &crate::gguf::QuantizedGenerateConfig,
    ) -> std::result::Result<Vec<u32>, RealizarError> {
        let hd = self.hidden_dim;
        let kv_dim = self.num_kv_heads * self.head_dim;

        // Lazy mmap for embedding lookup — zero-copy, no 7.5 GB clone
        let mapped = crate::apr::MappedAprModel::from_path(&self.model_path)?;
        let embed_model = crate::gguf::OwnedQuantizedModel::from_apr(&mapped)?;

        // Reset KV caches — clear() resets length to 0 (capacity retained).
        // forward_layer uses extend_from_slice + len()/kv_dim to derive seq_len,
        // so length MUST be 0 at start. Zeroing contents preserves wrong length.
        for (k, v) in &mut self.kv_caches {
            k.clear();
            v.clear();
        }

        let mut output_tokens = input_tokens.to_vec();

        // PREFILL: Process all input tokens EXCEPT the last to build KV cache.
        if input_tokens.len() > 1 {
            for (pos, &tok) in input_tokens[..input_tokens.len()-1].iter().enumerate() {
                let mut hidden = embed_model.embed(&[tok]);
                for layer_idx in 0..self.num_layers {
                    let prefix = format!("layer.{layer_idx}");
                    let (ref mut kv_k, ref mut kv_v) = self.kv_caches[layer_idx];
                    self.fwd.forward_layer(
                        &mut hidden, &prefix, pos, kv_k, kv_v,
                    ).map_err(|e| RealizarError::InferenceError(format!("wgpu prefill L{layer_idx}: {e}")))?;
                }
            }
        }

        // DECODE: Generate new tokens. First step processes the last input token.
        for _step in 0..config.max_tokens {
            let token_id = *output_tokens.last().unwrap();
            let position = output_tokens.len() - 1;

            let mut hidden = embed_model.embed(&[token_id]);
            for layer_idx in 0..self.num_layers {
                let prefix = format!("layer.{layer_idx}");
                let (ref mut kv_k, ref mut kv_v) = self.kv_caches[layer_idx];
                self.fwd.forward_layer(
                    &mut hidden, &prefix, position, kv_k, kv_v,
                ).map_err(|e| RealizarError::InferenceError(format!("wgpu L{layer_idx}: {e}")))?;
            }

            // Output norm + LM head argmax (CPU, 4x unrolled)
            let sq_sum: f32 = hidden.iter().map(|x| x * x).sum();
            let rms = (sq_sum / hd as f32 + self.eps).sqrt();
            let normed: Vec<f32> = hidden.iter().zip(self.output_norm.iter())
                .map(|(x, g)| (x / rms) * g).collect();

            let mut best_idx = 0u32;
            let mut best_val = f32::NEG_INFINITY;
            for i in 0..self.vocab_size {
                let row = &self.lm_head_f32[i * hd..(i + 1) * hd];
                let mut sum = 0.0f32;
                let chunks = hd / 4;
                for c in 0..chunks {
                    let j = c * 4;
                    sum += row[j]*normed[j] + row[j+1]*normed[j+1]
                         + row[j+2]*normed[j+2] + row[j+3]*normed[j+3];
                }
                for j in (chunks*4)..hd { sum += row[j] * normed[j]; }
                if sum > best_val { best_val = sum; best_idx = i as u32; }
            }

            output_tokens.push(best_idx);
            if config.stop_tokens.contains(&best_idx) { break; }
        }

        Ok(output_tokens)
    }
}

/// GH-560: Try initializing wgpu batch state.
///
/// Single-pass dequant: extracts lm_head inline during weight upload iteration.
/// Previous code called dequant_model_weights() twice (56 GB peak → 28 GB peak).
#[cfg(feature = "gpu")]
#[provable_contracts_macros::contract("gpu-weight-residency-v1", equation = "pcie_overhead")]
fn try_init_wgpu_batch(
    model: &crate::gguf::OwnedQuantizedModel,
    config: &BatchInferenceConfig,
) -> Option<WgpuBatchState> {
    use crate::gpu::adapters::wgpu_adapter;
    use trueno::backends::gpu::GpuDevice;

    if !GpuDevice::is_available() {
        return None;
    }

    let gpu = GpuDevice::new().ok()?;

    if config.verbose {
        eprintln!("[batch] Backend: wgpu (Vulkan)");
    }

    let cfg = model.config();
    let hidden_dim = cfg.hidden_dim;
    let num_heads = cfg.num_heads;
    let num_kv_heads = cfg.num_kv_heads;
    let head_dim = hidden_dim / num_heads;
    let intermediate_dim = cfg.intermediate_dim;
    let kv_dim = num_kv_heads * head_dim;
    let num_layers = cfg.num_layers;

    let mut fwd = trueno::backends::gpu::WgslForwardPass::new(
        gpu.device, gpu.queue,
        hidden_dim, num_heads, num_kv_heads, head_dim, intermediate_dim,
    );

    // C-WGPU-Q4K-001: Upload raw Q4K bytes for projection weights (7x less VRAM).
    let raw_q4k = wgpu_adapter::raw_q4k_weights(model);
    let q4k_names: std::collections::HashSet<String> =
        raw_q4k.iter().map(|(n, _, _, _)| n.clone()).collect();
    for (name, data, _rows, _cols) in &raw_q4k {
        fwd.upload_q4k_weight(name, data);
    }

    // GH-560 FIX: Dequantize non-Q4K weights in ONE pass. Extract lm_head during iteration.
    let weights = wgpu_adapter::dequant_model_weights(model).ok()?;
    let mut lm_head_f32 = Vec::new();
    for (name, data, _rows, _cols) in weights {
        if name == "lm_head" {
            lm_head_f32 = data;
            continue;
        }
        if !q4k_names.contains(&name) {
            fwd.upload_weight(&name, &data);
        }
    }

    // Upload biases (small, F32)
    for (i, layer) in model.layers().iter().enumerate() {
        let prefix = format!("layer.{i}");
        if let Some(ref bias) = layer.qkv_bias {
            let q_dim = num_heads * head_dim;
            let kv_d = num_kv_heads * head_dim;
            if bias.len() >= q_dim + 2 * kv_d {
                fwd.upload_weight(&format!("{prefix}.q_bias"), &bias[..q_dim]);
                fwd.upload_weight(&format!("{prefix}.k_bias"), &bias[q_dim..q_dim+kv_d]);
                fwd.upload_weight(&format!("{prefix}.v_bias"), &bias[q_dim+kv_d..q_dim+2*kv_d]);
            }
        }
    }

    // Output norm (small: hidden_dim floats)
    let output_norm = model.output_norm_weight().to_vec();

    // Initialize GPU KV caches
    fwd.init_kv_cache(num_layers);

    // KV caches: start empty (length 0), pre-allocate capacity to avoid reallocs.
    // forward_layer appends via extend_from_slice and derives seq_len from len().
    let max_seq = config.max_tokens + 128;
    let kv_caches: Vec<(Vec<f32>, Vec<f32>)> = (0..num_layers)
        .map(|_| (Vec::with_capacity(max_seq * kv_dim), Vec::with_capacity(max_seq * kv_dim)))
        .collect();

    Some(WgpuBatchState {
        fwd,
        model_path: config.model_path.clone(),
        hidden_dim, num_layers, num_kv_heads, head_dim,
        vocab_size: cfg.vocab_size, eps: cfg.eps,
        lm_head_f32, output_norm, kv_caches,
    })
}
