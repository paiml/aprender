impl ClassifyPipeline {
    /// Build LoRA layers for Q and V projections across all transformer layers.
    fn build_lora_layers(
        model: &Transformer,
        model_config: &TransformerConfig,
        classify_config: &ClassifyConfig,
    ) -> Vec<LoRALayer> {
        let lora_config = LoRAConfig::new(classify_config.lora_rank, classify_config.lora_alpha)
            .target_qv_projections();

        let mut lora_layers = Vec::new();
        let hidden = model_config.hidden_size;
        let head_dim = model_config.head_dim();

        for layer in &model.layers {
            let attn = &layer.self_attn;

            // Q projection LoRA
            if lora_config.should_apply("q_proj", None) {
                let q_dim = model_config.num_attention_heads * head_dim;
                let q_weight = Tensor::from_vec(
                    attn.w_q.data().as_slice().expect("contiguous w_q").to_vec(),
                    false,
                );
                lora_layers.push(LoRALayer::new(
                    q_weight,
                    q_dim,
                    hidden,
                    classify_config.lora_rank,
                    classify_config.lora_alpha,
                ));
            }

            // V projection LoRA
            if lora_config.should_apply("v_proj", None) {
                let v_dim = model_config.num_kv_heads * head_dim;
                let v_weight = Tensor::from_vec(
                    attn.w_v.data().as_slice().expect("contiguous w_v").to_vec(),
                    false,
                );
                lora_layers.push(LoRALayer::new(
                    v_weight,
                    v_dim,
                    hidden,
                    classify_config.lora_rank,
                    classify_config.lora_alpha,
                ));
            }
        }

        lora_layers
    }

    // ── CUDA GPU acceleration (F-CUDA-001..014) ────────────────────────

    /// C-PREWARM-001: JIT-compile all CUDA kernels before block upload.
    ///
    /// CUDA JIT needs free VRAM for PTX compilation. After uploading transformer
    /// layers, JIT fails with CUDA_ERROR_ILLEGAL_ADDRESS or OOM.
    #[cfg(feature = "cuda")]
    fn pre_warm_all_kernels(
        model_config: &TransformerConfig,
        classify_config: &ClassifyConfig,
    ) -> bool {
        let max_seq_len = classify_config.max_seq_len;
        let quantize_nf4 = classify_config.quantize_nf4;
        let head_dim = model_config.head_dim();

        if let Err(e) = pre_warm_forward_kernels(
            model_config.hidden_size,
            model_config.intermediate_size,
            model_config.num_attention_heads,
            model_config.num_kv_heads,
            head_dim,
            max_seq_len,
        ) {
            eprintln!("[CUDA] Failed to pre-warm forward kernels: {e} — using CPU");
            return false;
        }

        if quantize_nf4 {
            eprintln!(
                "[CUDA] NF4 quantization enabled — frozen weights will be 4-bit (~8x compression)"
            );
        }

        if let Err(e) = pre_warm_lora_backward_kernels(
            model_config.hidden_size,
            model_config.num_attention_heads * head_dim,
            model_config.num_kv_heads * head_dim,
            max_seq_len,
            classify_config.lora_rank,
        ) {
            eprintln!(
                "[CUDA] Failed to pre-warm LoRA forward-cache backward kernels: {e} — using CPU"
            );
            return false;
        }

        if let Err(e) = pre_warm_backward_cache_kernels(
            model_config.hidden_size,
            model_config.num_attention_heads * head_dim,
            model_config.num_kv_heads * head_dim,
            max_seq_len,
            classify_config.lora_rank,
            model_config.intermediate_size,
            model_config.num_attention_heads,
            quantize_nf4,
        ) {
            eprintln!("[CUDA] Failed to pre-warm backward cache kernels: {e} — using CPU");
            return false;
        }

        if let Err(e) = pre_warm_lora_adamw_kernels(
            model_config.hidden_size,
            model_config.num_attention_heads * head_dim,
            model_config.num_kv_heads * head_dim,
            classify_config.lora_rank,
            classify_config.num_classes,
            model_config.intermediate_size,
            quantize_nf4,
        ) {
            eprintln!("[CUDA] Failed to pre-warm AdamW kernels: {e} — using CPU");
            return false;
        }

        true
    }

    /// Estimate VRAM usage (MB) for GPU training based on model architecture.
    ///
    /// Used by GPU-SHARE-002 to reserve VRAM via the ledger before allocation.
    #[cfg(feature = "cuda")]
    fn estimate_vram_mb(model_config: &TransformerConfig, config: &ClassifyConfig) -> usize {
        if config.quantize_nf4 {
            let weight_elements =
                model_config.per_layer_weight_elements() * model_config.num_hidden_layers;
            let weight_mb = weight_elements / (2 * 1024 * 1024);
            let scratch_mb =
                (config.max_seq_len * model_config.hidden_size * 4 * 10) / (1024 * 1024);
            let overhead_mb = 512;
            weight_mb + scratch_mb + overhead_mb
        } else {
            model_config.total_training_vram_bytes_shared(config.max_seq_len) / (1024 * 1024) + 256
        }
    }

    /// GPU-SHARE-002: Acquire VRAM guard before GPU allocation.
    ///
    /// Returns `None` if VRAM is insufficient (C-VRAM-001 enforcement).
    #[cfg(feature = "cuda")]
    fn acquire_vram_guard(
        model_config: &TransformerConfig,
        classify_config: &ClassifyConfig,
    ) -> Option<VramGuard> {
        let budget_mb = Self::estimate_vram_mb(model_config, classify_config);
        let task_label =
            if classify_config.quantize_nf4 { "classify-qlora" } else { "classify-lora" };
        match VramGuard::acquire(budget_mb, task_label) {
            Ok(guard) => {
                eprintln!(
                    "[GPU-SHARE] VRAM reserved: {budget_mb} MB for {task_label} (gpu: {})",
                    guard.gpu_uuid()
                );
                Some(guard)
            }
            Err(e) => {
                eprintln!("[GPU-SHARE] VRAM guard denied: {e} — falling back to CPU");
                None
            }
        }
    }

    /// Attempt to initialize CUDA acceleration.
    ///
    /// Creates `CudaTrainer` and uploads all transformer layer weights to GPU as
    /// `CudaTransformerBlock`s. Returns `(None, None, None, None)` if CUDA is
    /// unavailable or any initialization step fails (F-CUDA-003: graceful fallback).
    ///
    /// GPU-SHARE-002: Acquires a VRAM guard from the ledger before allocating GPU
    /// memory. The guard is returned and must be stored in the pipeline struct for
    /// RAII release on Drop.
    #[cfg(feature = "cuda")]
    fn try_init_cuda(
        model: &Transformer,
        model_config: &TransformerConfig,
        classify_config: &ClassifyConfig,
        lora_layers: &[LoRALayer],
    ) -> (Option<CudaTrainer>, Option<Vec<CudaBlock>>, Option<CudaBlockScratch>, Option<VramGuard>)
    {
        if !cuda_training_available() {
            eprintln!("[CUDA] No CUDA runtime detected — using CPU");
            return (None, None, None, None);
        }

        // GPU-SHARE-002: Acquire VRAM reservation before allocating
        let mut vram_guard = Self::acquire_vram_guard(model_config, classify_config);
        if vram_guard.is_none() {
            return (None, None, None, None);
        }

        let trainer = match CudaTrainer::new() {
            Ok(t) => {
                eprintln!(
                    "[CUDA] Initialized: {} ({:.1} GB)",
                    t.device_name(),
                    t.total_memory() as f64 / 1e9
                );
                t
            }
            Err(e) => {
                eprintln!("[CUDA] Failed to create trainer: {e} — using CPU");
                return (None, None, None, None);
            }
        };

        let ctx = Arc::clone(trainer.context());
        let max_seq_len = classify_config.max_seq_len;
        let quantize_nf4 = classify_config.quantize_nf4;

        if !Self::pre_warm_all_kernels(model_config, classify_config) {
            return (None, None, None, None);
        }

        let mut blocks = Vec::with_capacity(model.config.num_hidden_layers);

        for (i, layer) in model.layers.iter().enumerate() {
            // Extract weight data from CPU tensors (F-CUDA-005)
            let input_norm = layer.input_norm.weight.data();
            let input_norm = input_norm.as_slice().expect("contiguous input_norm");
            let post_attn_norm = layer.post_attn_norm.weight.data();
            let post_attn_norm = post_attn_norm.as_slice().expect("contiguous post_attn_norm");
            let w_q = layer.self_attn.w_q.data();
            let w_q = w_q.as_slice().expect("contiguous w_q");
            let w_k = layer.self_attn.w_k.data();
            let w_k = w_k.as_slice().expect("contiguous w_k");
            let w_v = layer.self_attn.w_v.data();
            let w_v = w_v.as_slice().expect("contiguous w_v");
            let w_o = layer.self_attn.w_o.data();
            let w_o = w_o.as_slice().expect("contiguous w_o");
            let w_gate = layer.ffn.w_gate.data();
            let w_gate = w_gate.as_slice().expect("contiguous w_gate");
            let w_up = layer.ffn.w_up.data();
            let w_up = w_up.as_slice().expect("contiguous w_up");
            let w_down = layer.ffn.w_down.data();
            let w_down = w_down.as_slice().expect("contiguous w_down");

            let result = if quantize_nf4 {
                // Extract LoRA data for this layer's Q and V projections
                let lora_scale = classify_config.lora_alpha / classify_config.lora_rank as f32;
                let lora_rank = classify_config.lora_rank;
                let q_lora_idx = i * 2;
                let v_lora_idx = i * 2 + 1;

                // Q LoRA
                let q_a_data;
                let q_b_data;
                let q_lora = if q_lora_idx < lora_layers.len() {
                    q_a_data = lora_layers[q_lora_idx].lora_a().data();
                    q_b_data = lora_layers[q_lora_idx].lora_b().data();
                    Some((
                        q_a_data.as_slice().expect("contiguous lora_a_q"),
                        q_b_data.as_slice().expect("contiguous lora_b_q"),
                    ))
                } else {
                    None
                };

                // V LoRA
                let v_a_data;
                let v_b_data;
                let v_lora = if v_lora_idx < lora_layers.len() {
                    v_a_data = lora_layers[v_lora_idx].lora_a().data();
                    v_b_data = lora_layers[v_lora_idx].lora_b().data();
                    Some((
                        v_a_data.as_slice().expect("contiguous lora_a_v"),
                        v_b_data.as_slice().expect("contiguous lora_b_v"),
                    ))
                } else {
                    None
                };

                // ENT-270: Extract QK-norm weights if present
                let q_norm_data = layer
                    .self_attn
                    .q_norm
                    .as_ref()
                    .map(|t| t.data().as_slice().expect("contiguous q_norm").to_vec());
                let k_norm_data = layer
                    .self_attn
                    .k_norm
                    .as_ref()
                    .map(|t| t.data().as_slice().expect("contiguous k_norm").to_vec());

                crate::transformer::CudaNf4TransformerBlock::new(
                    model_config,
                    i,
                    Arc::clone(&ctx),
                    input_norm,
                    post_attn_norm,
                    w_q,
                    w_k,
                    w_v,
                    w_o,
                    w_gate,
                    w_up,
                    w_down,
                    max_seq_len,
                    q_lora,
                    v_lora,
                    lora_scale,
                    lora_rank,
                    q_norm_data.as_deref(),
                    k_norm_data.as_deref(),
                )
                .map(CudaBlock::Nf4)
            } else {
                CudaTransformerBlock::new(
                    model_config,
                    i,
                    Arc::clone(&ctx),
                    input_norm,
                    post_attn_norm,
                    w_q,
                    w_k,
                    w_v,
                    w_o,
                    w_gate,
                    w_up,
                    w_down,
                    max_seq_len,
                )
                .map(CudaBlock::Fp32)
            };

            match result {
                Ok(block) => blocks.push(block),
                Err(e) => {
                    eprintln!(
                        "[CUDA] Failed to upload layer {i} to GPU: {e} — falling back to CPU"
                    );
                    return (None, None, None, None);
                }
            }
        }

        eprintln!(
            "[CUDA] Uploaded {} transformer layers to GPU (max_seq_len={})",
            blocks.len(),
            max_seq_len
        );

        // F-CUDA-006: verify all layers uploaded
        assert_eq!(blocks.len(), model.config.num_hidden_layers);

        // C-SCRATCH-001: Allocate one shared scratch for NF4 (saves 7.5 GB for Qwen3-4B)
        let shared_scratch = if quantize_nf4 {
            match CudaBlockScratch::new(model_config, max_seq_len, &ctx, classify_config.lora_rank)
            {
                Ok(s) => Some(s),
                Err(e) => {
                    eprintln!("[CUDA] Failed to allocate shared scratch: {e} — using CPU");
                    return (None, None, None, None);
                }
            }
        } else {
            None // fp32 blocks own their scratch (needed for backward)
        };

        // GPU-SHARE-002: Update actual VRAM usage after all allocations
        if let Some(ref mut guard) = vram_guard {
            let _ = guard.update_actual(guard.budget_mb());
        }

        (Some(trainer), Some(blocks), shared_scratch, vram_guard)
    }

    /// Initialize GPU training state for full-finetune backward pass (F-CUDA-014).
    ///
    /// Allocates layer-input snapshot buffers, uploads final RMSNorm weight,
    /// and initializes per-block AdamW optimizer state. Returns `None` if CUDA
    /// is not active or any allocation fails.
    ///
    /// # Contract (C-GPUTRAINIT-001)
    ///
    /// - **Precondition**: CUDA trainer and blocks are initialized (`Some`)
    /// - **Postcondition**: All buffers allocated; optimizer states zero-initialized
    /// - **Invariant**: Returns `None` on any failure (graceful fallback to CPU training)
    #[cfg(feature = "cuda")]
    fn try_init_gpu_training(
        model: &Transformer,
        model_config: &TransformerConfig,
        max_seq_len: usize,
        cuda_trainer: Option<&CudaTrainer>,
        cuda_blocks: Option<&Vec<CudaBlock>>,
    ) -> Option<GpuTrainingState> {
        let trainer = cuda_trainer?;
        let blocks = cuda_blocks?;

        let hidden_size = model_config.hidden_size;
        let buf_size = max_seq_len * hidden_size;
        let num_layers = blocks.len();

        // ── VRAM budget guard (C-GPUTRAINIT-002) ────────────────────────
        // Pre-compute optimizer state size to avoid OOM that would poison
        // the CUDA context (CUDA_ERROR_ILLEGAL_ADDRESS after failed alloc).
        // NF4 blocks have no per-layer fp32 optimizer — only need layer inputs + grad scratch
        let is_nf4 = blocks.first().is_some_and(|b| matches!(b, CudaBlock::Nf4(_)));
        let per_layer_weights = model_config.per_layer_weight_elements();
        let optimizer_bytes = if is_nf4 { 0 } else { num_layers * per_layer_weights * 2 * 4 };
        let layer_input_bytes = num_layers * buf_size * 4;
        let grad_scratch_bytes = (3 * buf_size + hidden_size) * 4;
        let total_training_bytes = optimizer_bytes + layer_input_bytes + grad_scratch_bytes;

        // After block upload + grad workspace, remaining VRAM is approximately:
        //   device_vram - (weights + scratch + grad_workspace)
        // For safety, we estimate remaining as device_vram minus our VRAM budget formula.
        let model_vram = model_config.total_training_vram_bytes_shared(max_seq_len);
        // Use 24 GB as conservative device VRAM (RTX 4090 = 24564 MiB)
        let device_vram = 24_u64 * 1024 * 1024 * 1024;
        let remaining_vram = device_vram.saturating_sub(model_vram as u64);

        if total_training_bytes as u64 > remaining_vram {
            eprintln!(
                "[CUDA] Skipping GPU training state: needs {:.1} GB \
                 (optimizer: {:.1} GB, layer inputs: {:.1} GB), \
                 estimated remaining VRAM: {:.1} GB — will use GPU forward + CPU backward",
                total_training_bytes as f64 / 1e9,
                optimizer_bytes as f64 / 1e9,
                layer_input_bytes as f64 / 1e9,
                remaining_vram as f64 / 1e9,
            );
            return None;
        }

        // Allocate layer-input snapshot buffers
        let mut layer_inputs = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            match trainer.zeros(buf_size) {
                Ok(buf) => layer_inputs.push(buf),
                Err(e) => {
                    eprintln!("[CUDA] GPU training init failed (layer input alloc): {e}");
                    return None;
                }
            }
        }

        // Upload final RMSNorm weight
        let norm_data = model.norm.weight.data();
        let norm_slice = norm_data.as_slice().expect("contiguous final norm weight");
        let final_norm_weight = match trainer.upload(norm_slice) {
            Ok(buf) => buf,
            Err(e) => {
                eprintln!("[CUDA] GPU training init failed (final norm upload): {e}");
                return None;
            }
        };

        // Allocate gradient scratch buffers
        let blocks_output = trainer.zeros(buf_size).ok()?;
        let grad_buf_a = trainer.zeros(buf_size).ok()?;
        let grad_buf_b = trainer.zeros(buf_size).ok()?;
        let grad_final_norm_weight = trainer.zeros(hidden_size).ok()?;

        // Initialize per-block optimizer states (fp32 only — NF4 uses separate LoRA optimizer)
        let mut optimizer_states = Vec::with_capacity(num_layers);
        let is_nf4 = blocks.first().is_some_and(|b| matches!(b, CudaBlock::Nf4(_)));
        if !is_nf4 {
            for (i, block) in blocks.iter().enumerate() {
                match block.init_optimizer_state() {
                    Ok(state) => optimizer_states.push(state),
                    Err(e) => {
                        eprintln!(
                            "[CUDA] GPU training init failed (optimizer state layer {i}): {e}"
                        );
                        return None;
                    }
                }
            }
        }

        eprintln!(
            "[CUDA] GPU training state initialized: {num_layers} layers, \
             {buf_size} buf_size{}",
            if is_nf4 {
                " (NF4 QLoRA mode — LoRA optimizer separate)".to_string()
            } else {
                format!(
                    " ({:.1} MB optimizer state)",
                    (optimizer_states.len() * 18 * buf_size * 4) as f64 / 1e6
                )
            }
        );

        // KAIZEN-045: Pre-allocate backward scratch buffers to eliminate per-backward
        // cuMemAlloc/cuMemFree. Each cuMemAlloc costs ~10-100µs; over 14K samples this
        // saves 28K+ CUDA memory operations per epoch.
        let output_scratch = trainer.zeros(buf_size).ok()?;
        let grad_upload_buf = trainer.zeros(buf_size).ok()?;

        // KAIZEN-060: Pre-allocate forward ping-pong buffers to eliminate
        // 2 × cuMemAlloc/Free per forward pass (was trainer.upload + trainer.zeros per call).
        let fwd_scratch_a = trainer.zeros(buf_size).ok()?;
        let fwd_scratch_b = trainer.zeros(buf_size).ok()?;

        // KAIZEN-061: Pre-allocate CPU staging buffer for backward mean-pool gradient.
        // Eliminates vec![0.0; seq_len * hidden_size] (~1.25MB) per sample in both
        // backward_gpu_blocks and backward_nf4_gpu_blocks.
        let backward_cpu_staging = vec![0.0f32; buf_size];

        Some(GpuTrainingState {
            layer_inputs,
            final_norm_weight,
            blocks_output,
            grad_buf_a,
            grad_buf_b,
            grad_final_norm_weight,
            optimizer_states,
            step: 0,
            output_scratch,
            grad_upload_buf,
            fwd_scratch_a,
            fwd_scratch_b,
            backward_cpu_staging,
        })
    }

    /// Initialize NF4 LoRA training state: gradient workspace + per-layer optimizer states + accumulators.
    ///
    /// # Contract (C-NF4LORA-INIT-001)
    ///
    /// - **Precondition**: CUDA trainer and NF4 blocks initialized
    /// - **Postcondition**: LoRA grad workspace allocated, one optimizer state per NF4 block,
    ///   one gradient accumulator per NF4 block (KAIZEN-014)
    /// - **Invariant**: Returns `(None, None, None)` on any failure (graceful fallback to CPU LoRA)
    #[cfg(feature = "cuda")]
    fn try_init_nf4_lora_training(
        cuda_trainer: Option<&CudaTrainer>,
        cuda_blocks: Option<&Vec<CudaBlock>>,
        model_config: &TransformerConfig,
        classify_config: &ClassifyConfig,
    ) -> (
        Option<CudaLoraGradWorkspace>,
        Option<Vec<GpuLoraOptimizerState>>,
        Option<Vec<CudaLoraGradWorkspace>>,
    ) {
        let trainer = match cuda_trainer {
            Some(t) => t,
            None => return (None, None, None),
        };
        let blocks = match cuda_blocks {
            Some(b) => b,
            None => return (None, None, None),
        };

        // Allocate shared LoRA gradient workspace
        let grad_ws = match CudaLoraGradWorkspace::new(
            trainer.context(),
            model_config,
            classify_config.lora_rank,
        ) {
            Ok(ws) => ws,
            Err(e) => {
                eprintln!("[CUDA] NF4 LoRA grad workspace alloc failed: {e}");
                return (None, None, None);
            }
        };

        // Initialize per-block LoRA optimizer states
        let mut opt_states = Vec::with_capacity(blocks.len());
        for (i, block) in blocks.iter().enumerate() {
            match block.init_lora_optimizer_state() {
                Ok(state) => opt_states.push(state),
                Err(e) => {
                    eprintln!("[CUDA] NF4 LoRA optimizer init failed (layer {i}): {e}");
                    return (None, None, None);
                }
            }
        }

        // KAIZEN-014: Allocate per-layer gradient accumulators
        let mut grad_accum = Vec::with_capacity(blocks.len());
        for i in 0..blocks.len() {
            match CudaLoraGradWorkspace::new(
                trainer.context(),
                model_config,
                classify_config.lora_rank,
            ) {
                Ok(ws) => grad_accum.push(ws),
                Err(e) => {
                    eprintln!("[CUDA] NF4 LoRA grad accum alloc failed (layer {i}): {e}");
                    return (None, None, None);
                }
            }
        }

        let accum_vram_mb = {
            let h = model_config.hidden_size;
            let q_dim = model_config.q_dim();
            let kv = model_config.num_kv_heads * model_config.head_dim();
            let r = classify_config.lora_rank;
            let per_layer_elems = h * r + r * q_dim + h * r + r * kv + h + h;
            let total_bytes = per_layer_elems * 4 * blocks.len();
            total_bytes as f64 / (1024.0 * 1024.0)
        };

        eprintln!(
            "[CUDA] NF4 QLoRA training initialized: {} layers, rank={}, scale={:.2}, accum={:.1} MB",
            blocks.len(),
            classify_config.lora_rank,
            classify_config.lora_alpha / classify_config.lora_rank as f32,
            accum_vram_mb,
        );

        (Some(grad_ws), Some(opt_states), Some(grad_accum))
    }

    /// GPU-accelerated forward pass that saves layer inputs for backward (F-CUDA-014).
    ///
    /// Like `forward_hidden_cuda_impl` but additionally:
    /// 1. Saves each block's input for backward pass
    /// 2. Keeps blocks output on GPU (for RMSNorm backward)
    /// 3. Also downloads and applies final RMSNorm on CPU for the classifier
    ///
    /// # Contract (C-GPUFWD-001)
    ///
    /// - **Precondition**: CUDA trainer, blocks, and gpu_training are all `Some`;
    ///   `token_ids.len() <= max_seq_len`
    /// - **Postcondition**: `gpu_training.layer_inputs[i]` contains input to block `i`;
    ///   `gpu_training.blocks_output` contains final block output (pre-norm);
    ///   returned Tensor is the normed hidden states on CPU
    /// - **Invariant**: GPU blocks contain valid forward-pass scratch for backward
    #[cfg(feature = "cuda")]
    #[allow(unsafe_code)]
    fn forward_hidden_cuda_training(
        model: &Transformer,
        token_ids: &[u32],
        trainer: &CudaTrainer,
        cuda_blocks: &mut [CudaBlock],
        training_state: &mut GpuTrainingState,
        shared_scratch: &mut Option<CudaBlockScratch>,
    ) -> Option<Tensor> {
        let seq_len = token_ids.len();
        let hidden_size = model.config.hidden_size;

        // Step 1: Embed on CPU
        let hidden = model.embed_tokens.forward(token_ids);
        let hidden_data = hidden.data();
        let hidden_slice = hidden_data.as_slice().expect("contiguous hidden");

        // Step 2: Upload to GPU using pre-allocated fwd_scratch_a (KAIZEN-060)
        // Pad remaining buffer to match pre-allocated size (kernels use seq_len param).
        training_state.fwd_scratch_a.copy_from_host_at(hidden_slice, 0).ok()?;

        // Step 3: Run through CUDA transformer blocks, saving inputs
        // KAIZEN-060: Use pre-allocated ping-pong buffers instead of per-call alloc.
        let stream = trainer.stream();
        let scratch_a_ptr: *mut GpuBuffer<f32> = &raw mut training_state.fwd_scratch_a;
        let scratch_b_ptr: *mut GpuBuffer<f32> = &raw mut training_state.fwd_scratch_b;
        let mut input_is_a = true;

        for (i, block) in cuda_blocks.iter_mut().enumerate() {
            // SAFETY: scratch_a_ptr and scratch_b_ptr point to disjoint fields.
            let (input, output) = unsafe {
                if input_is_a {
                    (&*scratch_a_ptr, &mut *scratch_b_ptr)
                } else {
                    (&*scratch_b_ptr, &mut *scratch_a_ptr)
                }
            };

            // Save input to this block for backward pass
            // SAFETY: Both buffers are valid GPU allocations with matching sizes.
            // The copy completes before block.forward() reads from input.
            unsafe {
                training_state.layer_inputs[i].copy_from_buffer_async(input, stream).ok()?;
            }

            if let Err(e) = block.forward(input, output, seq_len, stream, shared_scratch.as_mut()) {
                eprintln!("[CUDA] Layer {i} forward failed: {e}");
                return None;
            }
            input_is_a = !input_is_a;
        }
        // After loop: the buffer indicated by input_is_a holds the final output
        let final_output = unsafe {
            if input_is_a {
                &*scratch_a_ptr
            } else {
                &*scratch_b_ptr
            }
        };

        // Save blocks output for final norm backward
        // SAFETY: Both buffers valid, copy completes before any read.
        unsafe {
            training_state.blocks_output.copy_from_buffer_async(final_output, stream).ok()?;
        }

        // Sync and download for CPU classifier path
        stream.synchronize().ok()?;
        let result_data = trainer.download(final_output).ok()?;

        // NaN guard
        if result_data.iter().any(|v| !v.is_finite()) {
            return None;
        }

        // Apply final RMSNorm on CPU
        let result_tensor = Tensor::from_vec(result_data, false);
        Some(model.norm.forward_batched(&result_tensor, seq_len, hidden_size))
    }

    /// Run backward pass through all GPU transformer blocks (F-CUDA-014).
    ///
    /// Computes gradients for all transformer weights by:
    /// 1. Computing grad through classifier and mean-pool on CPU
    /// 2. Uploading gradient to GPU
    /// 3. Running RMSNorm backward on GPU (final norm)
    /// 4. Running block.backward() for each layer in reverse
    ///
    /// # Contract (C-GPUBACK-001)
    ///
    /// - **Precondition**: Forward training pass completed (`forward_hidden_cuda_training`),
    ///   `grad_logits` has length `num_classes`, `hidden_pre_norm` is the raw block output
    /// - **Postcondition**: Each block's scratch contains weight gradients (grad_w_q/k/v/o,
    ///   grad_gate/up/down, grad_input_norm, grad_post_attn_norm)
    /// - **Invariant**: Zero GPU memory allocation during backward; all buffers preallocated
    #[cfg(feature = "cuda")]
    #[allow(unsafe_code)]
    fn backward_gpu_blocks(&mut self, grad_logits: &[f32], seq_len: usize) -> Option<()> {
        let grad_ws = self.cuda_grad_workspace.as_mut()?;
        let trainer = self.cuda_trainer.as_ref()?;
        let hidden_size = self.model.config.hidden_size;
        let num_classes = self.config.num_classes;

        // Step 1: Classifier backward on CPU (trivial: hidden_size * num_classes mults)
        // grad_pooled = W_classifier^T @ grad_logits
        let w_data = self.classifier.weight.data();
        let w_slice = w_data.as_slice().expect("contiguous classifier weight");
        let mut grad_pooled = vec![0.0f32; hidden_size];
        for j in 0..hidden_size {
            let mut sum = 0.0f32;
            for c in 0..num_classes {
                sum += grad_logits[c] * w_slice[j * num_classes + c];
            }
            grad_pooled[j] = sum;
        }

        // Step 2: Mean-pool backward into pre-allocated CPU staging buffer (KAIZEN-061)
        let scale = 1.0 / seq_len as f32;
        let training_state = self.gpu_training.as_mut()?;
        for i in 0..seq_len {
            for j in 0..hidden_size {
                training_state.backward_cpu_staging[i * hidden_size + j] = grad_pooled[j] * scale;
            }
        }

        // Step 3: Upload gradient to pre-allocated GPU buffer (KAIZEN-045)
        let stream = trainer.stream();
        training_state
            .grad_upload_buf
            .copy_from_host_at(&training_state.backward_cpu_staging[..seq_len * hidden_size], 0)
            .ok()?;

        let blocks = self.cuda_blocks.as_mut()?;

        // Step 4: RMSNorm backward on GPU (final norm)
        // input = blocks_output, gamma = final_norm_weight
        // grad_output = grad_upload_buf, grad_input = grad_buf_a, grad_gamma = grad_final_norm_weight
        crate::autograd::cuda_backward::rms_norm_backward(
            &training_state.blocks_output,
            &training_state.final_norm_weight,
            &training_state.grad_upload_buf,
            &mut training_state.grad_buf_a,
            &mut training_state.grad_final_norm_weight,
            seq_len as u32,
            hidden_size as u32,
            1e-5_f32,
            stream,
        )
        .ok()?;

        // Step 5: Backward through blocks in reverse
        // grad_buf_a has the gradient w.r.t. blocks_output (= grad for last block's output)
        // We alternate between grad_buf_a and grad_buf_b as we propagate backward
        let num_layers = blocks.len();

        // Alternate between grad_buf_a (output) and grad_buf_b (input) buffers.
        // After RMSNorm backward, grad_buf_a holds the gradient for the last block's output.
        // We use raw pointers to get disjoint mutable borrows of grad_buf_a/b.
        // SAFETY: grad_buf_a and grad_buf_b are separate heap-allocated GPU buffers
        // (disjoint fields of GpuTrainingState). We never alias them.
        let grad_a_ptr: *mut GpuBuffer<f32> = &raw mut training_state.grad_buf_a;
        let grad_b_ptr: *mut GpuBuffer<f32> = &raw mut training_state.grad_buf_b;
        let mut grad_output_is_a = true;

        for layer_idx in (0..num_layers).rev() {
            // SAFETY: grad_a_ptr and grad_b_ptr point to disjoint fields.
            let (grad_output, grad_input) = unsafe {
                if grad_output_is_a {
                    (&*grad_a_ptr, &mut *grad_b_ptr)
                } else {
                    (&*grad_b_ptr, &mut *grad_a_ptr)
                }
            };

            blocks[layer_idx]
                .backward(
                    &training_state.layer_inputs[layer_idx],
                    grad_output,
                    grad_input,
                    seq_len,
                    stream,
                    grad_ws,
                )
                .ok()?;

            grad_output_is_a = !grad_output_is_a;
        }

        // Sync to ensure all backward kernels completed
        stream.synchronize().ok()?;

        Some(())
    }

    /// Run GPU-resident AdamW optimizer step on all transformer block weights.
    ///
    /// # Contract (C-GPUOPT-001)
    ///
    /// - **Precondition**: `backward_gpu_blocks()` completed successfully
    /// - **Postcondition**: All block weights updated; optimizer step counter incremented
    /// - **Invariant**: Learning rate and hyperparameters applied uniformly across all blocks
    #[cfg(feature = "cuda")]
    fn gpu_optimizer_step(&mut self, lr: f32) -> Option<()> {
        let grad_ws = self.cuda_grad_workspace.as_ref()?;
        let trainer = self.cuda_trainer.as_ref()?;
        let stream = trainer.stream();
        let training_state = self.gpu_training.as_mut()?;
        let blocks = self.cuda_blocks.as_mut()?;

        training_state.step += 1;
        let step = training_state.step;

        for (block, opt_state) in blocks.iter_mut().zip(training_state.optimizer_states.iter_mut())
        {
            block
                .optimizer_step(
                    opt_state, step, lr, 0.9,   // beta1
                    0.999, // beta2
                    1e-8,  // eps
                    0.01,  // weight_decay
                    stream, grad_ws,
                )
                .ok()?;
        }

        // Sync to ensure all optimizer kernels completed
        stream.synchronize().ok()?;

        Some(())
    }

    /// NF4 QLoRA backward pass through all GPU transformer blocks (ENT-153).
    ///
    /// Like `backward_gpu_blocks` but uses NF4 transposed GEMM for gradient flow
    /// through frozen weights, and computes LoRA gradients for Q/V projections.
    /// After each block backward, immediately runs the LoRA optimizer step
    /// (grad workspace is shared across layers, so we must consume grads before
    /// the next layer overwrites them).
    ///
    /// # Contract (C-NF4BACK-001)
    ///
    /// - **Precondition**: Forward training pass completed, grad_logits has length num_classes
    /// - **Postcondition**: LoRA weights updated for all NF4 blocks
    /// - **Invariant**: Zero GPU memory allocation during backward
    #[cfg(feature = "cuda")]
    #[allow(unsafe_code)]
    fn backward_nf4_gpu_blocks(&mut self, grad_logits: &[f32], seq_len: usize) -> Option<()> {
        use crate::transformer::cuda_block::cuda_add_inplace;

        let trainer = self.cuda_trainer.as_ref()?;
        let hidden_size = self.model.config.hidden_size;
        let num_classes = self.config.num_classes;

        // Step 1: Classifier backward on CPU (trivial: hidden_size * num_classes mults)
        let w_data = self.classifier.weight.data();
        let w_slice = w_data.as_slice().expect("contiguous classifier weight");
        let mut grad_pooled = vec![0.0f32; hidden_size];
        for j in 0..hidden_size {
            let mut sum = 0.0f32;
            for c in 0..num_classes {
                sum += grad_logits[c] * w_slice[j * num_classes + c];
            }
            grad_pooled[j] = sum;
        }

        // Step 2: Mean-pool backward into pre-allocated CPU staging buffer (KAIZEN-061)
        let scale = 1.0 / seq_len as f32;
        let training_state = self.gpu_training.as_mut()?;
        for i in 0..seq_len {
            for j in 0..hidden_size {
                training_state.backward_cpu_staging[i * hidden_size + j] = grad_pooled[j] * scale;
            }
        }

        // Step 3: Upload gradient to pre-allocated GPU buffer (KAIZEN-045)
        let stream = trainer.stream();
        training_state
            .grad_upload_buf
            .copy_from_host_at(&training_state.backward_cpu_staging[..seq_len * hidden_size], 0)
            .ok()?;

        let blocks = self.cuda_blocks.as_mut()?;
        let shared_scratch = self.shared_scratch.as_mut()?;
        let grad_lora = self.cuda_lora_grad_workspace.as_mut()?;
        let grad_accum = self.cuda_lora_grad_accum.as_mut()?;

        // Step 4: RMSNorm backward on GPU (final norm)
        crate::autograd::cuda_backward::rms_norm_backward(
            &training_state.blocks_output,
            &training_state.final_norm_weight,
            &training_state.grad_upload_buf,
            &mut training_state.grad_buf_a,
            &mut training_state.grad_final_norm_weight,
            seq_len as u32,
            hidden_size as u32,
            1e-5_f32,
            stream,
        )
        .ok()?;

        // Step 5: Backward through blocks in reverse, accumulate gradients (KAIZEN-014)
        let num_layers = blocks.len();

        // Alternate gradient buffers using raw pointers for disjoint mutable borrows
        let grad_a_ptr: *mut GpuBuffer<f32> = &raw mut training_state.grad_buf_a;
        let grad_b_ptr: *mut GpuBuffer<f32> = &raw mut training_state.grad_buf_b;
        let mut grad_output_is_a = true;

        // KAIZEN-045: Use pre-allocated output_scratch from training state
        let output_scratch_ptr: *mut GpuBuffer<f32> = &raw mut training_state.output_scratch;

        for layer_idx in (0..num_layers).rev() {
            // SAFETY: grad_a_ptr and grad_b_ptr point to disjoint fields.
            let (grad_output, grad_input) = unsafe {
                if grad_output_is_a {
                    (&*grad_a_ptr, &mut *grad_b_ptr)
                } else {
                    (&*grad_b_ptr, &mut *grad_a_ptr)
                }
            };

            // NF4 backward: activation checkpointing + LoRA gradient computation
            // SAFETY: output_scratch_ptr points to a disjoint field of training_state.
            blocks[layer_idx]
                .backward_nf4(
                    &training_state.layer_inputs[layer_idx],
                    grad_output,
                    grad_input,
                    unsafe { &mut *output_scratch_ptr },
                    seq_len,
                    stream,
                    shared_scratch,
                    grad_lora,
                )
                .ok()?;

            // KAIZEN-014: Accumulate gradients into per-layer accumulators
            // (grad_lora workspace is shared — must consume before next layer overwrites)
            let accum = &mut grad_accum[layer_idx];
            cuda_add_inplace(
                &mut accum.grad_lora_a_q,
                &grad_lora.grad_lora_a_q,
                grad_lora.grad_lora_a_q.len(),
                stream,
            )
            .ok()?;
            cuda_add_inplace(
                &mut accum.grad_lora_b_q,
                &grad_lora.grad_lora_b_q,
                grad_lora.grad_lora_b_q.len(),
                stream,
            )
            .ok()?;
            cuda_add_inplace(
                &mut accum.grad_lora_a_v,
                &grad_lora.grad_lora_a_v,
                grad_lora.grad_lora_a_v.len(),
                stream,
            )
            .ok()?;
            cuda_add_inplace(
                &mut accum.grad_lora_b_v,
                &grad_lora.grad_lora_b_v,
                grad_lora.grad_lora_b_v.len(),
                stream,
            )
            .ok()?;
            cuda_add_inplace(
                &mut accum.grad_input_norm,
                &grad_lora.grad_input_norm,
                grad_lora.grad_input_norm.len(),
                stream,
            )
            .ok()?;
            cuda_add_inplace(
                &mut accum.grad_post_attn_norm,
                &grad_lora.grad_post_attn_norm,
                grad_lora.grad_post_attn_norm.len(),
                stream,
            )
            .ok()?;

            grad_output_is_a = !grad_output_is_a;
        }

        // Sync to ensure all backward + accumulation kernels completed
        stream.synchronize().ok()?;

        Some(())
    }

    /// KAIZEN-014: Batch-level LoRA optimizer step with averaged gradients.
    ///
    /// Divides accumulated gradients by `batch_size` (via reduced `lr`) and applies
    /// a single AdamW step per layer. Then zeros the accumulators for the next batch.
    ///
    /// # Contract (C-NF4BATCH-001)
    ///
    /// - **Precondition**: `backward_nf4_gpu_blocks` called `batch_size` times since last step
    /// - **Postcondition**: LoRA weights updated once, accumulators zeroed
    /// - **Invariant**: Effective LR = base_lr / batch_size (gradient averaging)
    #[cfg(feature = "cuda")]
    pub(crate) fn nf4_lora_batch_optimizer_step(&mut self, batch_size: usize) {
        let Some(trainer) = self.cuda_trainer.as_ref() else { return };
        let Some(blocks) = self.cuda_blocks.as_mut() else { return };
        let Some(opt_states) = self.cuda_lora_optimizer_states.as_mut() else { return };
        let Some(grad_accum) = self.cuda_lora_grad_accum.as_mut() else { return };

        let stream = trainer.stream();
        let lr = self.optimizer.lr() / batch_size as f32;

        self.nf4_lora_step += 1;
        let step = self.nf4_lora_step;

        // KAIZEN-043: Pre-allocate a single zero buffer sized to the largest
        // accumulator. Reuse via slicing instead of allocating vec![0.0; len]
        // per buffer per layer (was: 216 allocations/batch for 36 layers × 6 bufs).
        let max_accum_len = grad_accum
            .iter()
            .map(|g| {
                g.grad_lora_a_q
                    .len()
                    .max(g.grad_lora_b_q.len())
                    .max(g.grad_lora_a_v.len())
                    .max(g.grad_lora_b_v.len())
                    .max(g.grad_input_norm.len())
                    .max(g.grad_post_attn_norm.len())
            })
            .max()
            .unwrap_or(0);
        let zeros = vec![0.0f32; max_accum_len];

        // ENT-265: Clip accumulated NF4 LoRA gradients before optimizer step.
        // Without this, NF4 dequant amplifies gradients causing divergence.
        if let Some(max_norm) = self.config.gradient_clip_norm {
            for accum in grad_accum.iter_mut() {
                accum.clip_gradients(max_norm, stream);
            }
        }

        for layer_idx in 0..blocks.len() {
            let _ = blocks[layer_idx].lora_optimizer_step(
                &mut opt_states[layer_idx],
                step,
                lr,
                0.9,
                0.999,
                1e-8,
                0.01,
                stream,
                &grad_accum[layer_idx],
            );

            // Zero accumulators for next batch (reuse pre-allocated zero buffer)
            let zero_buf = |buf: &mut GpuBuffer<f32>| {
                let _ = buf.copy_from_host(&zeros[..buf.len()]);
            };
            zero_buf(&mut grad_accum[layer_idx].grad_lora_a_q);
            zero_buf(&mut grad_accum[layer_idx].grad_lora_b_q);
            zero_buf(&mut grad_accum[layer_idx].grad_lora_a_v);
            zero_buf(&mut grad_accum[layer_idx].grad_lora_b_v);
            zero_buf(&mut grad_accum[layer_idx].grad_input_norm);
            zero_buf(&mut grad_accum[layer_idx].grad_post_attn_norm);
        }

        let _ = stream.synchronize();
    }

    /// Synchronize GPU-updated weights back to CPU model tensors.
    ///
    /// Required for checkpointing and after training completes. Downloads all
    /// weight data from GPU and updates the CPU model's weight tensors in-place.
    ///
    /// # Contract (C-SYNCWT-001)
    ///
    /// - **Precondition**: GPU blocks have valid weights (after one or more optimizer steps)
    /// - **Postcondition**: CPU model weights match GPU weights exactly
    /// - **Invariant**: GPU weights are not modified
    #[cfg(feature = "cuda")]
    pub fn sync_weights_to_cpu(&mut self) {
        let blocks = match self.cuda_blocks.as_ref() {
            Some(b) => b,
            None => return,
        };

        for (layer_idx, block) in blocks.iter().enumerate() {
            if let Ok(weights) = block.download_weights() {
                let layer = &mut self.model.layers[layer_idx];

                // Update attention weights
                layer.self_attn.w_q = Tensor::from_vec(weights.w_q, false);
                layer.self_attn.w_k = Tensor::from_vec(weights.w_k, false);
                layer.self_attn.w_v = Tensor::from_vec(weights.w_v, false);
                layer.self_attn.w_o = Tensor::from_vec(weights.w_o, false);

                // Update FFN weights
                layer.ffn.w_gate = Tensor::from_vec(weights.w_gate, false);
                layer.ffn.w_up = Tensor::from_vec(weights.w_up, false);
                layer.ffn.w_down = Tensor::from_vec(weights.w_down, false);

                // Update norm weights
                layer.input_norm.weight = Tensor::from_vec(weights.input_norm_weight, false);
                layer.post_attn_norm.weight =
                    Tensor::from_vec(weights.post_attn_norm_weight, false);
            }
        }
    }

    /// Synchronize GPU LoRA weights back to CPU LoRA layers (NF4 QLoRA).
    ///
    /// Required for checkpointing after NF4 QLoRA training. Downloads A_q, B_q,
    /// A_v, B_v from each NF4 block and updates the corresponding CPU LoRA layers.
    ///
    /// B matrices are stored pre-scaled on GPU (includes lora_scale). This method
    /// un-scales them before writing back to CPU.
    ///
    /// # Contract (C-QLORA-CKPT-001)
    ///
    /// - **Precondition**: NF4 QLoRA training completed (optimizer steps applied)
    /// - **Postcondition**: CPU LoRA layers match GPU-trained LoRA weights
    #[cfg(feature = "cuda")]
    pub fn sync_lora_to_cpu(&mut self) {
        let blocks = match self.cuda_blocks.as_ref() {
            Some(b) => b,
            None => return,
        };

        let lora_scale = self.config.lora_alpha / self.config.lora_rank.max(1) as f32;
        let inv_scale = if lora_scale.abs() > 1e-10 { 1.0 / lora_scale } else { 1.0 };

        for (layer_idx, block) in blocks.iter().enumerate() {
            if let Ok((a_q, b_q, a_v, b_v)) = block.download_lora_weights() {
                let q_lora_idx = layer_idx * 2;
                let v_lora_idx = layer_idx * 2 + 1;

                // Un-scale B matrices (GPU stores B * lora_scale)
                let b_q_unscaled: Vec<f32> = b_q.iter().map(|&v| v * inv_scale).collect();
                let b_v_unscaled: Vec<f32> = b_v.iter().map(|&v| v * inv_scale).collect();

                if q_lora_idx < self.lora_layers.len() {
                    *self.lora_layers[q_lora_idx].lora_a_mut() = Tensor::from_vec(a_q, true);
                    *self.lora_layers[q_lora_idx].lora_b_mut() =
                        Tensor::from_vec(b_q_unscaled, true);
                }
                if v_lora_idx < self.lora_layers.len() {
                    *self.lora_layers[v_lora_idx].lora_a_mut() = Tensor::from_vec(a_v, true);
                    *self.lora_layers[v_lora_idx].lora_b_mut() =
                        Tensor::from_vec(b_v_unscaled, true);
                }
            }
        }
    }

    /// Check if GPU training (full finetune backward) is active.
    #[must_use]
    pub fn is_gpu_training(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            self.gpu_training.is_some()
        }
        #[cfg(not(feature = "cuda"))]
        {
            false
        }
    }

    /// Forward pass through transformer layers, dispatching to GPU when available.
    ///
    /// Priority: CUDA > wgpu > CPU
    ///
    /// - **CUDA path** (F-CUDA-007..009): Embed on CPU, upload to GPU, run CUDA layers, download
    /// - **wgpu path**: Batched FFN matmuls via `WgpuForwardPass`, attention on CPU
    /// - **CPU path**: Use `Transformer::forward_hidden()`
    fn forward_hidden_dispatch(&mut self, token_ids: &[u32]) -> Tensor {
        // Poka-yoke: clamp seq_len to max_seq_len at GPU kernel boundary.
        // GPU buffers are pre-allocated to max_seq_len * hidden_size; passing
        // longer sequences causes OOB writes and SIGSEGV (paiml/entrenar#280).
        let max_seq = self.config.max_seq_len;
        let token_ids = if token_ids.len() > max_seq { &token_ids[..max_seq] } else { token_ids };

        #[cfg(feature = "cuda")]
        if let Some(tensor) = self.try_forward_hidden_gpu(token_ids) {
            return tensor;
        }

        // wgpu fallback (batched FFN matmuls on GPU, attention on CPU)
        #[cfg(feature = "gpu")]
        if let Some(ref wgpu_pass) = self.wgpu_forward_pass {
            match wgpu_pass.forward_hidden(&self.model, token_ids) {
                Ok(tensor) => return tensor,
                Err(e) => {
                    eprintln!("[wgpu] Forward pass failed, falling back to CPU: {e}");
                }
            }
        }

        // CPU fallback — KAIZEN-011: use LoRA-aware forward when adapters exist
        if self.lora_layers.is_empty() {
            self.model.forward_hidden(token_ids)
        } else {
            self.model.forward_hidden_with_lora(token_ids, &self.lora_layers)
        }
    }

    /// Attempt GPU-accelerated forward pass (training or inference).
    ///
    /// Returns `Some(tensor)` on success, `None` to fall back to CPU.
    #[cfg(feature = "cuda")]
    fn try_forward_hidden_gpu(&mut self, token_ids: &[u32]) -> Option<Tensor> {
        if self.gpu_training.is_some() {
            // GPU training path: saves layer inputs for backward
            let (trainer, blocks) = match (&self.cuda_trainer, &mut self.cuda_blocks) {
                (Some(ref t), Some(ref mut b)) => (t, b),
                _ => return None,
            };
            let mut training = self.gpu_training.take();
            let result = Self::forward_hidden_cuda_training(
                &self.model,
                token_ids,
                trainer,
                blocks,
                training.as_mut().expect("gpu_training was Some"),
                &mut self.shared_scratch,
            );
            self.gpu_training = training;

            if result.is_none() {
                self.cuda_nan_count += 1;
            }
            return result;
        }

        // Inference-only GPU path (no layer input saving)
        let (trainer, blocks) = match (&self.cuda_trainer, &mut self.cuda_blocks) {
            (Some(ref t), Some(ref mut b)) => (t, b),
            _ => return None,
        };
        match Self::forward_hidden_cuda_impl(
            &self.model,
            token_ids,
            trainer,
            blocks,
            &mut self.shared_scratch,
        ) {
            Some(tensor) => Some(tensor),
            None => {
                self.cuda_nan_count += 1;
                if self.cuda_nan_count > 100 {
                    eprintln!(
                        "[CUDA] {} NaN fallbacks — disabling GPU acceleration",
                        self.cuda_nan_count
                    );
                    self.cuda_trainer = None;
                    self.cuda_blocks = None;
                }
                None
            }
        }
    }

    /// GPU-accelerated forward pass (F-CUDA-007).
    ///
    /// 1. Embed tokens on CPU (F-CUDA-008: small op)
    /// 2. Upload hidden states to GPU
    /// 3. Run through all CudaTransformerBlocks
    /// 4. Apply final RMSNorm on CPU
    /// 5. Return hidden states (F-CUDA-009)
    ///
    /// Returns `None` on any GPU error, signaling caller to use CPU fallback.
    #[cfg(feature = "cuda")]
    fn forward_hidden_cuda_impl(
        model: &Transformer,
        token_ids: &[u32],
        trainer: &CudaTrainer,
        cuda_blocks: &mut [CudaBlock],
        shared_scratch: &mut Option<CudaBlockScratch>,
    ) -> Option<Tensor> {
        let seq_len = token_ids.len();
        let hidden_size = model.config.hidden_size;

        // Step 1: Embed on CPU
        let hidden = model.embed_tokens.forward(token_ids);
        let hidden_data = hidden.data();
        let hidden_slice = hidden_data.as_slice().expect("contiguous hidden");

        // Step 2: Upload to GPU
        let mut gpu_input = trainer.upload(hidden_slice).ok()?;
        let mut gpu_output = trainer.zeros(seq_len * hidden_size).ok()?;

        // Step 3: Run through CUDA transformer blocks
        let stream = trainer.stream();
        for (i, block) in cuda_blocks.iter_mut().enumerate() {
            if let Err(e) =
                block.forward(&gpu_input, &mut gpu_output, seq_len, stream, shared_scratch.as_mut())
            {
                eprintln!("[CUDA] Layer {i} forward failed: {e}");
                return None;
            }
            // Swap: output becomes input for next layer
            std::mem::swap(&mut gpu_input, &mut gpu_output);
        }
        // After the loop, gpu_input holds the final output (due to swap)

        // Sync stream to ensure all CUDA kernels have completed before download
        if let Err(e) = stream.synchronize() {
            eprintln!("[CUDA] Stream sync failed: {e}");
            return None;
        }

        // Step 4: Download from GPU
        let result_data = trainer.download(&gpu_input).ok()?;

        // Step 4.5: NaN guard — GPU kernels can produce NaN with certain weight
        // distributions (e.g., random init). Fall back to CPU if detected.
        if result_data.iter().any(|v| !v.is_finite()) {
            return None;
        }

        // Step 5: Apply final RMSNorm on CPU
        let result_tensor = Tensor::from_vec(result_data, false);
        Some(model.norm.forward_batched(&result_tensor, seq_len, hidden_size))
    }

    /// Check if this pipeline is using CUDA acceleration.
    #[must_use]
    pub fn is_cuda(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            self.cuda_blocks.is_some()
        }
        #[cfg(not(feature = "cuda"))]
        {
            false
        }
    }

    /// Get GPU device name, or `None` if not using CUDA.
    #[must_use]
    pub fn gpu_name(&self) -> Option<String> {
        #[cfg(feature = "cuda")]
        {
            self.cuda_trainer.as_ref().map(crate::autograd::cuda_training::CudaTrainer::device_name)
        }
        #[cfg(not(feature = "cuda"))]
        {
            None
        }
    }

    /// Get total GPU memory in bytes, or `None` if not using CUDA.
    #[must_use]
    pub fn gpu_total_memory(&self) -> Option<usize> {
        #[cfg(feature = "cuda")]
        {
            self.cuda_trainer
                .as_ref()
                .map(crate::autograd::cuda_training::CudaTrainer::total_memory)
        }
        #[cfg(not(feature = "cuda"))]
        {
            None
        }
    }

    /// Get the base model directory path (if loaded from pretrained weights).
    #[must_use]
    pub fn model_dir(&self) -> Option<&Path> {
        self.model_dir.as_deref()
    }

    /// Set the base model path (used when the model was loaded from an APR file
    /// rather than a SafeTensors directory, so checkpoint saves can record the
    /// provenance in `adapter_config.json`).
    pub fn set_model_path(&mut self, path: impl Into<PathBuf>) {
        self.model_dir = Some(path.into());
    }
}
