#[cfg(feature = "cuda")]
use super::{
    CudaTrainer, InstructConfig, InstructGpuTrainingState, InstructPipeline, LoRALayer,
    Transformer, TransformerConfig, VramGuard,
};

#[cfg(feature = "cuda")]
use crate::autograd::cuda_backward::pre_warm_lora_backward_kernels as pre_warm_backward_cache_kernels;
#[cfg(feature = "cuda")]
use crate::autograd::cuda_forward::{pre_warm_forward_kernels, pre_warm_lora_backward_kernels};
#[cfg(feature = "cuda")]
use crate::autograd::cuda_optim::pre_warm_lora_adamw_kernels;
#[cfg(feature = "cuda")]
use crate::autograd::cuda_training::cuda_training_available;
#[cfg(feature = "cuda")]
use crate::transformer::{
    CudaBlock, CudaBlockScratch, CudaLoraGradWorkspace, CudaTransformerBlock, GpuLoraOptimizerState,
};
#[cfg(feature = "cuda")]
use std::sync::Arc;

#[cfg(feature = "cuda")]
impl InstructPipeline {
    /// Initialize CUDA acceleration: create trainer, upload blocks, init LoRA training.
    /// GPU-SHARE-002: Acquires VRAM guard; falls back to CPU if denied.
    pub(super) fn init_cuda(&mut self, model_config: &TransformerConfig) {
        // GPU-SHARE-002: Acquire VRAM reservation before allocating
        let budget_mb = Self::estimate_vram_mb(model_config, &self.config);
        let task_label = if self.config.quantize_nf4 { "instruct-qlora" } else { "instruct-lora" };
        match VramGuard::acquire(budget_mb, task_label) {
            Ok(guard) => {
                eprintln!(
                    "[GPU-SHARE] VRAM reserved: {budget_mb} MB for {task_label} (gpu: {})",
                    guard.gpu_uuid()
                );
                self.vram_guard = Some(guard);
            }
            Err(e) => {
                eprintln!("[GPU-SHARE] VRAM guard denied: {e} — falling back to CPU");
                return;
            }
        }

        let (trainer, blocks, scratch) =
            Self::try_init_cuda(&self.model, model_config, &self.config, &self.lora_layers);

        if trainer.is_none() {
            // CUDA init failed — release the guard
            self.vram_guard = None;
            return;
        }

        self.cuda_trainer = trainer;
        self.cuda_blocks = blocks;
        self.shared_scratch = scratch;

        // GPU training state (layer input snapshots for backward)
        self.gpu_training = Self::try_init_gpu_training(
            &self.model,
            model_config,
            self.config.max_seq_len,
            self.cuda_trainer.as_ref(),
            self.cuda_blocks.as_ref(),
        );

        if self.config.quantize_nf4 {
            let (grad_ws, opt_states) = Self::try_init_nf4_lora_training(
                self.cuda_trainer.as_ref(),
                self.cuda_blocks.as_ref(),
                model_config,
                &self.config,
            );
            if let (Some(ws), Some(t)) = (&grad_ws, &self.cuda_trainer) {
                self.lora_fused_clip =
                    super::super::fused_lora_clip::init_lora_fused_clip(ws, t.context());
            }
            self.cuda_lora_grad_workspace = grad_ws;
            self.cuda_lora_optimizer_states = opt_states;
        }

        // GPU-SHARE-002: Update actual VRAM usage after all allocations
        if let Some(ref mut guard) = self.vram_guard {
            let _ = guard.update_actual(budget_mb);
        }
    }

    /// Estimate VRAM usage (MB) for GPU training (GPU-SHARE-002 ledger reservation).
    fn estimate_vram_mb(model_config: &TransformerConfig, config: &InstructConfig) -> usize {
        if config.quantize_nf4 {
            let weight_elements =
                model_config.per_layer_weight_elements() * model_config.num_hidden_layers;
            let weight_mb = weight_elements / (2 * 1024 * 1024);
            let scratch_mb =
                (config.max_seq_len * model_config.hidden_size * 4 * 10) / (1024 * 1024);
            weight_mb + scratch_mb + 512
        } else {
            model_config.total_training_vram_bytes_shared(config.max_seq_len) / (1024 * 1024) + 256
        }
    }

    /// Create `CudaTrainer` and upload all transformer layer weights to GPU.
    /// Returns `(None, None, None)` if CUDA is unavailable or any step fails.
    fn try_init_cuda(
        model: &Transformer,
        model_config: &TransformerConfig,
        config: &InstructConfig,
        lora_layers: &[LoRALayer],
    ) -> (Option<CudaTrainer>, Option<Vec<CudaBlock>>, Option<CudaBlockScratch>) {
        if !cuda_training_available() {
            eprintln!("[CUDA] No CUDA runtime detected — using CPU");
            return (None, None, None);
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
                return (None, None, None);
            }
        };

        let ctx = Arc::clone(trainer.context());
        let max_seq_len = config.max_seq_len;

        // C-PREWARM-001: JIT-compile forward kernels before block upload
        if let Err(e) = pre_warm_forward_kernels(
            model_config.hidden_size,
            model_config.intermediate_size,
            model_config.num_attention_heads,
            model_config.num_kv_heads,
            model_config.head_dim(),
            max_seq_len,
        ) {
            eprintln!("[CUDA] Failed to pre-warm forward kernels: {e} — using CPU");
            return (None, None, None);
        }

        let quantize_nf4 = config.quantize_nf4;
        if quantize_nf4 {
            eprintln!(
                "[CUDA] NF4 quantization enabled — frozen weights will be 4-bit (~8x compression)"
            );
        }

        let head_dim = model_config.head_dim();
        if let Err(e) = pre_warm_lora_backward_kernels(
            model_config.hidden_size,
            model_config.num_attention_heads * head_dim,
            model_config.num_kv_heads * head_dim,
            max_seq_len,
            config.lora_rank,
        ) {
            eprintln!("[CUDA] Failed to pre-warm LoRA backward kernels: {e} — using CPU");
            return (None, None, None);
        }

        if let Err(e) = pre_warm_backward_cache_kernels(
            model_config.hidden_size,
            model_config.num_attention_heads * head_dim,
            model_config.num_kv_heads * head_dim,
            max_seq_len,
            config.lora_rank,
            model_config.intermediate_size,
            model_config.num_attention_heads,
            quantize_nf4,
        ) {
            eprintln!("[CUDA] Failed to pre-warm backward cache kernels: {e}");
            eprintln!("[CUDA] STOP THE LINE: backward kernel pre-warming failed.");
            eprintln!("[CUDA] This is a FATAL error — training will produce loss=0.0 if backward");
            eprintln!("[CUDA] kernels are compiled during active GPU work (trueno#200).");
            return (None, None, None);
        }
        eprintln!("[CUDA] Backward kernels pre-warmed successfully");
        if let Err(e) = pre_warm_lora_adamw_kernels(
            model_config.hidden_size,
            model_config.num_attention_heads * head_dim,
            model_config.num_kv_heads * head_dim,
            config.lora_rank,
            0, // instruct has no classifier head
            model_config.intermediate_size,
            quantize_nf4,
        ) {
            eprintln!("[CUDA] Failed to pre-warm AdamW kernels: {e} — using CPU");
            return (None, None, None);
        }

        let mut blocks = Vec::with_capacity(model.config.num_hidden_layers);
        for (i, layer) in model.layers.iter().enumerate() {
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
                let lora_scale = config.lora_alpha / config.lora_rank as f32;
                let lora_rank = config.lora_rank;
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
                    return (None, None, None);
                }
            }
        }

        eprintln!(
            "[CUDA] Uploaded {} transformer layers to GPU (max_seq_len={})",
            blocks.len(),
            max_seq_len
        );

        assert_eq!(blocks.len(), model.config.num_hidden_layers);
        // PMAT-470: FP16 weight cast for tensor core GEMM
        if std::env::var("FP16_GEMM").as_deref() == Ok("1") && quantize_nf4 {
            super::super::gpu_backward_fallback::init_fp16_weights(&mut blocks, trainer.stream());
        }

        // C-SCRATCH-001: Shared scratch for NF4
        let shared_scratch = if quantize_nf4 {
            match CudaBlockScratch::new(model_config, max_seq_len, &ctx, config.lora_rank) {
                Ok(s) => Some(s),
                Err(e) => {
                    eprintln!("[CUDA] Failed to allocate shared scratch: {e} — using CPU");
                    return (None, None, None);
                }
            }
        } else {
            None
        };

        (Some(trainer), Some(blocks), shared_scratch)
    }

    /// Initialize GPU training state for NF4 QLoRA backward pass.
    pub(super) fn try_init_gpu_training(
        model: &Transformer,
        model_config: &TransformerConfig,
        max_seq_len: usize,
        cuda_trainer: Option<&CudaTrainer>,
        cuda_blocks: Option<&Vec<CudaBlock>>,
    ) -> Option<InstructGpuTrainingState> {
        let trainer = cuda_trainer?;
        let blocks = cuda_blocks?;

        let hidden_size = model_config.hidden_size;
        let buf_size = max_seq_len * hidden_size;
        let num_layers = blocks.len();

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

        // Upload embeddings for GPU lm_head (KAIZEN-068). PMAT-420: skip on <=16GB.
        let vocab_size = model_config.vocab_size;
        let embed_data = model.embed_tokens.weight.data();
        let embed_slice = embed_data.as_slice().expect("contiguous embed");
        let embed_bytes = vocab_size * hidden_size * 4; // entrenar#317: single layout
        let vram_available_mb = trainer.free_memory_mb().unwrap_or(0);
        let embed_mb = embed_bytes / (1024 * 1024);
        let use_gpu_embed = vram_available_mb > (embed_mb + 256) as u64;

        let (embed_original, embed_transposed) = if use_gpu_embed {
            eprintln!(
                "[CUDA] GPU-resident embeddings: {embed_mb}MB (VRAM free: {vram_available_mb}MB)"
            );
            let orig = trainer
                .upload(embed_slice)
                .map_err(|e| eprintln!("[CUDA] embed_original upload failed: {e}"))
                .ok()?;
            let trans = trainer.zeros(1).ok()?;
            (orig, trans)
        } else {
            eprintln!("[CUDA] Skipping GPU embeddings ({embed_mb}MB > {vram_available_mb}MB free)");
            let orig = trainer.zeros(1).ok()?;
            let trans = trainer.zeros(1).ok()?;
            (orig, trans)
        };

        // Logits scratch: [max_seq_len, vocab_size]
        let logits_buf = trainer
            .zeros(max_seq_len * vocab_size)
            .map_err(|e| eprintln!("[CUDA] logits_buf alloc failed: {e}"))
            .ok()?;

        // Grad-hidden scratch: [max_seq_len, hidden_size]
        let grad_hidden_buf = trainer.zeros(buf_size).ok()?;

        eprintln!(
            "[CUDA] GPU training state initialized: {num_layers} layers, {buf_size} buf_size, \
             embed=[{vocab_size}x{hidden_size}] on GPU (NF4 QLoRA mode)"
        );

        // KAIZEN-045/062: Pre-allocate backward + forward scratch buffers
        let output_scratch = trainer.zeros(buf_size).ok()?;
        let grad_upload_buf = trainer.zeros(buf_size).ok()?;
        let fwd_scratch_a = trainer.zeros(buf_size).ok()?;
        let fwd_scratch_b = trainer.zeros(buf_size).ok()?;
        let lm_head_hidden_buf = trainer.zeros(buf_size).ok()?;

        let num_layers = layer_inputs.len();
        Some(InstructGpuTrainingState {
            layer_inputs,
            final_norm_weight,
            blocks_output,
            grad_buf_a,
            grad_buf_b,
            grad_final_norm_weight,
            embed_transposed,
            embed_original,
            logits_buf,
            grad_hidden_buf,
            output_scratch,
            grad_upload_buf,
            fwd_scratch_a,
            fwd_scratch_b,
            lm_head_hidden_buf,
            forward_graph_exec: None,
            graph_cached_seq_len: 0,
            backward_graph_state: None,
            cublas_workspace: None,
            profiler_layer_fwd_us: vec![0u64; num_layers],
            profiler_layer_bwd_us: vec![0u64; num_layers],
            profiler_layer_start: None,
            profiler_op_us: [0u64; 16],
            profiler_op_start: None,
        })
    }

    /// Initialize NF4 LoRA training state: gradient workspace + per-layer optimizer states.
    fn try_init_nf4_lora_training(
        cuda_trainer: Option<&CudaTrainer>,
        cuda_blocks: Option<&Vec<CudaBlock>>,
        model_config: &TransformerConfig,
        config: &InstructConfig,
    ) -> (Option<CudaLoraGradWorkspace>, Option<Vec<GpuLoraOptimizerState>>) {
        let trainer = match cuda_trainer {
            Some(t) => t,
            None => return (None, None),
        };
        let blocks = match cuda_blocks {
            Some(b) => b,
            None => return (None, None),
        };

        let grad_ws =
            match CudaLoraGradWorkspace::new(trainer.context(), model_config, config.lora_rank) {
                Ok(ws) => ws,
                Err(e) => {
                    eprintln!("[CUDA] NF4 LoRA grad workspace alloc failed: {e}");
                    return (None, None);
                }
            };

        let mut opt_states = Vec::with_capacity(blocks.len());
        for (i, block) in blocks.iter().enumerate() {
            match block.init_lora_optimizer_state() {
                Ok(state) => opt_states.push(state),
                Err(e) => {
                    eprintln!("[CUDA] NF4 LoRA optimizer init failed (layer {i}): {e}");
                    return (None, None);
                }
            }
        }

        eprintln!(
            "[CUDA] NF4 QLoRA training initialized: {} layers, rank={}, scale={:.2}",
            blocks.len(),
            config.lora_rank,
            config.lora_alpha / config.lora_rank as f32,
        );

        (Some(grad_ws), Some(opt_states))
    }
}
