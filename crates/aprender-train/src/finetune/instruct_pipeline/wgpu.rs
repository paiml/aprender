//! WGPU GPU acceleration: `try_init_wgpu`, `wgpu_train_step`.

#[cfg(feature = "gpu")]
use super::{
    clip_grad_norm_refs, InstructPipeline, InstructStepResult, Optimizer, Tensor,
    TransformerConfig, WgpuTrainingState,
};

#[cfg(feature = "gpu")]
impl InstructPipeline {
    /// wgpu GPU training step (§26 WgpuTrainingPipeline)
    ///
    /// Uses CPU forward (model.forward) + GPU fused cross-entropy loss + CPU backward.
    /// The GPU handles the loss computation (fused CE) and optimizer (AdamW).
    /// Forward and backward GEMM through transformer layers stay on CPU for now —
    /// full GPU forward/backward is Step 0d.2/0d.3 (WgslForwardPass/WgslBackwardPass).
    ///
    /// This is the integration point: proves the pipeline works end-to-end,
    /// then incrementally moves forward/backward to GPU.
    pub(super) fn wgpu_train_step(
        &mut self,
        full_ids: &[u32],
        prompt_len: usize,
        seq_len: usize,
        vocab_size: usize,
    ) -> InstructStepResult {
        let loss_start = prompt_len.saturating_sub(1);
        let loss_end = seq_len - 1;
        let num_loss_tokens = loss_end.saturating_sub(loss_start);

        if num_loss_tokens == 0 {
            return InstructStepResult { loss: 0.0, num_response_tokens: 0, perplexity: 1.0 };
        }

        // KAIZEN: instrument every phase to find bottleneck
        let t0 = std::time::Instant::now();

        // 1. Forward pass: CPU model.forward() (fast init, profiled per step)
        let hidden_dim = self.wgpu_training.as_ref().unwrap().hidden_dim;
        let _ = hidden_dim;

        let logits_tensor = self.model.forward(full_ids);
        let logits_data = logits_tensor.data().as_slice().expect("contiguous").to_vec();

        let t1 = std::time::Instant::now();
        eprintln!("[PROFILE] cpu_forward: {:.0}ms", t1.duration_since(t0).as_millis());

        let t2 = t1;
        let t3 = t1;

        // Upload logits to GPU for fused cross-entropy
        {
            let wgpu = self.wgpu_training.as_ref().unwrap();
            wgpu.trainer.queue_ref().write_buffer(
                &wgpu.logits_buf,
                0,
                bytemuck::cast_slice(&logits_data[..seq_len * vocab_size]),
            );
        }

        // 2. GPU fused cross-entropy loss
        let wgpu = self.wgpu_training.as_ref().unwrap();

        // Shifted labels: position i predicts token at i+1
        let labels: Vec<u32> = (0..seq_len)
            .map(|i| if i + 1 < full_ids.len() { full_ids[i + 1] } else { 0 })
            .collect();
        wgpu.trainer.queue_ref().write_buffer(&wgpu.labels_buf, 0, bytemuck::cast_slice(&labels));

        let avg_loss = wgpu.cross_entropy.forward(
            &wgpu.logits_buf,
            &wgpu.labels_buf,
            &wgpu.losses_buf,
            &wgpu.logsumexp_buf,
            seq_len as u32,
            vocab_size as u32,
            loss_start as u32,
            loss_end as u32,
        );

        if !avg_loss.is_finite() {
            eprintln!("[wgpu] NaN/Inf loss detected — skipping backward");
            return InstructStepResult {
                loss: 100.0,
                num_response_tokens: num_loss_tokens,
                perplexity: 1e6,
            };
        }

        // 3. GPU fused cross-entropy backward (in-place into logits_buf)
        wgpu.cross_entropy.backward(
            &wgpu.logits_buf,
            &wgpu.labels_buf,
            &wgpu.logsumexp_buf,
            seq_len as u32,
            vocab_size as u32,
            loss_start as u32,
            loss_end as u32,
        );

        let t4 = std::time::Instant::now();
        eprintln!("[PROFILE] fused_ce: {:.0}ms", t4.duration_since(t3).as_millis());

        // Backward: use CPU autograd (simple, correct, profiled)
        let wgpu = self.wgpu_training.as_ref().unwrap();
        let grad_logits_data = wgpu.trainer.download(&wgpu.logits_buf);
        logits_tensor
            .set_grad(ndarray::Array1::from(grad_logits_data[..seq_len * vocab_size].to_vec()));
        if let Some(op) = logits_tensor.backward_op() {
            op.backward();
        }

        // Optimizer step on LoRA parameters
        let mut params: Vec<&mut Tensor> = Vec::new();
        for lora in &mut self.lora_layers {
            params.extend(lora.trainable_params());
        }
        if let Some(max_norm) = self.config.gradient_clip_norm {
            clip_grad_norm_refs(&mut params, max_norm);
        }
        self.optimizer.step_refs(&mut params);

        let t5 = std::time::Instant::now();
        eprintln!("[PROFILE] lm_head_backward: {:.0}ms", t5.duration_since(t4).as_millis());

        let t6 = std::time::Instant::now();
        eprintln!(
            "[PROFILE] total_step: {:.0}ms (embed={:.0} fwd={:.0} lm={:.0} ce={:.0} bwd={:.0})",
            t6.duration_since(t0).as_millis(),
            t1.duration_since(t0).as_millis(),
            t2.duration_since(t1).as_millis(),
            t3.duration_since(t2).as_millis(),
            t4.duration_since(t3).as_millis(),
            t5.duration_since(t4).as_millis(),
        );

        InstructStepResult {
            loss: avg_loss,
            num_response_tokens: num_loss_tokens,
            perplexity: avg_loss.exp().min(1e6),
        }
    }

    // ── wgpu GPU acceleration (§26 WgpuTrainingPipeline) ────────────────

    pub(super) fn try_init_wgpu(&mut self, _model_config: &TransformerConfig) {
        use crate::autograd::wgpu_cross_entropy::WgslCrossEntropy;
        use crate::autograd::wgpu_training::WgpuTrainer;

        let trainer = match WgpuTrainer::new() {
            Ok(t) => t,
            Err(e) => {
                eprintln!("[wgpu] Failed to init: {e} — using CPU");
                return;
            }
        };

        let seq = self.config.max_seq_len as u32;
        let vocab = _model_config.vocab_size as u32;
        let hidden = _model_config.hidden_size as u32;
        let num_layers = _model_config.num_hidden_layers;
        let num_heads = _model_config.num_attention_heads as u32;
        let num_kv_heads = _model_config.num_kv_heads as u32;
        let head_dim = (hidden / num_heads);
        let inter = _model_config.intermediate_size as u32;

        // Create WgslForwardPass with persistent weight buffers + tiled GEMM
        let mut fwd = trueno::backends::gpu::WgslForwardPass::new(
            trainer.device_ref().clone(),
            trainer.queue_ref().clone(),
            hidden as usize,
            num_heads as usize,
            num_kv_heads as usize,
            head_dim as usize,
            inter as usize,
        );

        // KAIZEN: Only upload norm weights (tiny: 14 KB each, 28 layers = ~800 KB total).
        let mut uploaded = 0usize;
        for (name, tensor) in self.model.named_parameters() {
            let data = match tensor.data().as_slice() {
                Some(s) => s,
                None => continue,
            };

            let gpu_name = name
                .replace("model.layers.", "layer.")
                .replace(".input_layernorm.weight", ".attn_norm")
                .replace(".post_attention_layernorm.weight", ".ffn_norm")
                .replace(".self_attn.", ".")
                .replace(".mlp.", ".")
                .replace(".weight", "");

            if gpu_name.ends_with(".attn_norm") || gpu_name.ends_with(".ffn_norm") {
                fwd.upload_weight(&gpu_name, data);
                uploaded += 1;
            }
        }

        fwd.init_kv_cache(num_layers);

        eprintln!(
            "[wgpu] Uploaded {uploaded} norm weights ({num_layers} layers, projections on-demand)"
        );

        let make_buf = |size: u64, label: &str| -> trueno::backends::gpu::wgpu::Buffer {
            trainer.device_ref().create_buffer(&trueno::backends::gpu::wgpu::BufferDescriptor {
                label: Some(label),
                size: size * 4,
                usage: trueno::backends::gpu::wgpu::BufferUsages::STORAGE
                    | trueno::backends::gpu::wgpu::BufferUsages::COPY_SRC
                    | trueno::backends::gpu::wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };

        let ce = WgslCrossEntropy::new(trainer.device_ref().clone(), trainer.queue_ref().clone());

        // KAIZEN: precompute lm_head + transpose, upload to GPU ONCE
        let lm_head_raw = self.model.lm_head_weight_slice();
        let h = hidden as usize;
        let v = vocab as usize;
        let mut lm_head_transposed = vec![0.0f32; h * v];
        for vi in 0..v {
            for hi in 0..h {
                lm_head_transposed[hi * v + vi] = lm_head_raw[vi * h + hi];
            }
        }
        let lm_head_gpu = trainer.upload(lm_head_raw);
        let lm_head_t_gpu = trainer.upload(&lm_head_transposed);
        drop(lm_head_transposed);
        eprintln!(
            "[wgpu] Training initialized (seq={seq}, vocab={vocab}, layers={num_layers}, lm_head on GPU)"
        );

        self.wgpu_training = Some(WgpuTrainingState {
            fwd,
            logits_buf: make_buf(u64::from(seq) * u64::from(vocab), "logits"),
            labels_buf: make_buf(u64::from(seq), "labels"),
            losses_buf: make_buf(u64::from(seq), "losses"),
            logsumexp_buf: make_buf(u64::from(seq), "logsumexp"),
            cross_entropy: ce,
            trainer,
            lm_head_gpu,
            lm_head_t_gpu,
            num_layers,
            hidden_dim: hidden as usize,
            vocab_size: vocab as usize,
        });
    }
}
