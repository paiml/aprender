#[allow(clippy::wildcard_imports)]
use super::*;

#[cfg(feature = "cuda")]
use crate::autograd::cuda_forward::gemm_forward;
#[cfg(feature = "cuda")]
use crate::autograd::cuda_optim::fused_causal_cross_entropy_cuda;

impl InstructPipeline {
    /// Compute causal LM loss on a single instruction-response pair.
    ///
    /// # Contract (F-INST-002)
    /// Loss is computed only on response tokens. Prompt tokens are masked.
    ///
    /// When CUDA NF4 blocks are available, dispatches to GPU forward pass
    /// with CPU loss computation and GPU backward/optimizer.
    pub fn train_step(&mut self, prompt_ids: &[u32], response_ids: &[u32]) -> InstructStepResult {
        self.profiler.begin_step();
        let full_ids: Vec<u32> = prompt_ids.iter().chain(response_ids.iter()).copied().collect();

        let prompt_len = prompt_ids.len();
        let response_len = response_ids.len();

        if response_len == 0 || full_ids.len() < 2 {
            self.profiler.finish_step();
            return InstructStepResult { loss: 0.0, num_response_tokens: 0, perplexity: 1.0 };
        }

        let full_ids = if full_ids.len() > self.config.max_seq_len {
            full_ids[..self.config.max_seq_len].to_vec()
        } else {
            full_ids
        };
        let seq_len = full_ids.len();
        let vocab_size = self.model.config().vocab_size;

        // Cap prompt_len at truncated sequence length. If the prompt alone
        // exceeds max_seq_len, all response tokens were truncated away.
        let prompt_len = prompt_len.min(seq_len);

        // ── CUDA GPU path (NF4 QLoRA) ─────────────────────────────────
        // ── CUDA GPU path (NF4 QLoRA) ─────────────────────────────────
        // PMAT-420: Use CUDA path for ALL configs. On 8GB, the inference-style
        // forward (fresh buffers, saves inputs) replaces the NaN-prone training forward.
        #[cfg(feature = "cuda")]
        if self.cuda_blocks.is_some() {
            let result = self.cuda_train_step(&full_ids, prompt_len, seq_len, vocab_size);
            self.profiler.finish_step();
            return result;
        }

        // ── wgpu GPU path (§26 WgpuTrainingPipeline) ─────────────────
        #[cfg(feature = "gpu")]
        if self.wgpu_training.is_some() {
            return self.wgpu_train_step(&full_ids, prompt_len, seq_len, vocab_size);
        }

        // ── CPU path ──────────────────────────────────────────────────

        // 1. Zero gradients
        for lora in &mut self.lora_layers {
            for param in lora.trainable_params() {
                param.zero_grad();
            }
        }

        // 2. Forward pass → logits [seq_len, vocab_size]
        //
        // PMAT-805 (#2051-class): route through `forward_with_lora` when LoRA
        // adapters exist. `model.forward()` runs plain attention and never
        // touches `self.lora_layers`, so the adapter params stay orphaned from
        // the autograd graph → `op.backward()` produces zero LoRA gradients →
        // flat loss (the adapters never train). `forward_with_lora` wires the
        // [Q0,V0,Q1,V1,...] adapters into the Q/V projections (KAIZEN-011 path).
        let logits = if self.lora_layers.is_empty() {
            self.model.forward(&full_ids)
        } else {
            self.model.forward_with_lora(&full_ids, &self.lora_layers)
        };
        let logits_data = logits.data().as_slice().expect("contiguous logits").to_vec();

        // 3. Causal LM loss on response tokens only
        let loss_start = prompt_len.saturating_sub(1);
        let loss_end = seq_len - 1;
        let num_loss_tokens = loss_end.saturating_sub(loss_start);

        if num_loss_tokens == 0 {
            return InstructStepResult { loss: 0.0, num_response_tokens: 0, perplexity: 1.0 };
        }

        let (avg_loss, grad_logits) =
            Self::compute_causal_lm_loss(&logits_data, &full_ids, loss_start, loss_end, vocab_size);

        // 4. Backward through autograd
        logits.set_grad(ndarray::Array1::from(grad_logits));
        if let Some(op) = logits.backward_op() {
            op.backward();
        }

        // 5. Optimizer step on LoRA parameters
        let mut params: Vec<&mut Tensor> = Vec::new();
        for lora in &mut self.lora_layers {
            params.extend(lora.trainable_params());
        }

        if let Some(max_norm) = self.config.gradient_clip_norm {
            clip_grad_norm_refs(&mut params, max_norm);
        }

        self.optimizer.step_refs(&mut params);

        InstructStepResult {
            loss: avg_loss,
            num_response_tokens: num_loss_tokens,
            perplexity: avg_loss.exp().min(1e6),
        }
    }
    /// GPU-accelerated training step for NF4 QLoRA.
    ///
    /// 1. GPU forward through NF4 transformer blocks → normed hidden states
    /// 2. CPU lm_head matmul → logits
    /// 3. CPU causal LM loss on response tokens only
    /// 4. CPU gradient of loss w.r.t. hidden states (through lm_head)
    /// 5. GPU backward through NF4 blocks → LoRA gradient + optimizer step
    #[cfg(feature = "cuda")]
    fn cuda_train_step(
        &mut self,
        full_ids: &[u32],
        prompt_len: usize,
        seq_len: usize,
        vocab_size: usize,
    ) -> InstructStepResult {
        // FALSIFY-CUDA-LOSS-WINDOW-512-001: clamp the loss window to the SAME
        // capacity the forward uses (scratch-driven), not a hardcoded 512.
        // Pre-fix this was `max_position_embeddings.min(512)` regardless of
        // `--max-seq-len`, so any sample whose prompt exceeded 512 tokens had
        // its ENTIRE response clamped out of the window — silent loss=0.0,
        // zero gradient. On the apr-code SFT corpus (system prompt ≫ 512
        // tokens) that was ~all samples: an epoch "trained" with no learning.
        let hidden_size = self.model.config().hidden_size;
        let scratch_capacity = self.shared_scratch.as_ref().map(|s| s.max_seq_len(hidden_size));
        let (loss_start, loss_end, num_loss_tokens, seq_len, prompt_len) = cuda_loss_window(
            prompt_len,
            seq_len,
            scratch_capacity,
            self.model.config().max_position_embeddings,
        );

        if num_loss_tokens == 0 {
            // Loud skip: silent 0.0 made a no-op epoch look like training.
            // Token-weighted epoch aggregation ignores this step (0 tokens).
            eprintln!(
                "[CUDA] sample skipped: prompt ({prompt_len} tokens) fills the \
                 {seq_len}-token training window — 0 response tokens to train on \
                 (raise --max-seq-len)"
            );
            return InstructStepResult { loss: 0.0, num_response_tokens: 0, perplexity: 1.0 };
        }

        // PMAT-420: If GPU embeddings are minimal (VRAM-constrained), skip the GPU-resident
        // logits path entirely — go straight to CPU-loss path which uses GPU transformer + CPU lm_head.
        let has_gpu_embed = self.gpu_training.as_ref().is_some_and(|t| {
            t.embed_original.len() >= self.model.config().hidden_size * vocab_size
        });

        if !has_gpu_embed {
            return self.cuda_train_step_cpu_loss(
                full_ids,
                loss_start,
                loss_end,
                num_loss_tokens,
                seq_len,
                vocab_size,
            );
        }

        // PMAT-483: Enable per-op profiling on scratch if profiler is active
        if self.profiler.is_enabled() {
            if let Some(ref mut scratch) = self.shared_scratch {
                scratch.op_profiling_enabled = true;
                scratch.op_us = [0u64; 16];
            }
        }

        // 1. GPU forward → logits stay GPU-resident in training.logits_buf (KAIZEN-064)
        self.profiler.begin(StepProfiler::FORWARD);
        if !self.forward_logits_gpu_resident(full_ids) {
            self.profiler.end(StepProfiler::FORWARD);
            eprintln!("[CUDA] GPU forward failed, falling back to CPU for this step");
            return self.cuda_train_step_cpu_loss(
                full_ids,
                loss_start,
                loss_end,
                num_loss_tokens,
                seq_len,
                vocab_size,
            );
        }
        self.profiler.end(StepProfiler::FORWARD);

        // 2. Fused GPU causal cross-entropy loss + softmax backward (KAIZEN-064)
        let targets: Vec<u32> = (0..seq_len)
            .map(|pos| if pos + 1 < full_ids.len() { full_ids[pos + 1] } else { 0 })
            .collect();

        let scale = 1.0 / num_loss_tokens as f32;

        self.profiler.begin(StepProfiler::LOSS);
        let avg_loss = (|| -> Option<f32> {
            let trainer = self.cuda_trainer.as_ref()?;
            let stream = trainer.stream();
            let training = self.gpu_training.as_mut()?;
            fused_causal_cross_entropy_cuda(
                &mut training.logits_buf,
                &targets,
                seq_len as u32,
                vocab_size as u32,
                loss_start as u32,
                loss_end as u32,
                scale,
                stream,
            )
            .ok()
        })();
        self.profiler.end(StepProfiler::LOSS);

        let avg_loss = match avg_loss {
            Some(l) if l.is_finite() => {
                eprintln!("[CUDA] loss={l:.4} (finite, proceeding with backward)");
                l
            }
            Some(l) => {
                eprintln!("[CUDA] NaN/Inf loss detected (loss={l}) — skipping backward pass");
                return InstructStepResult {
                    loss: 100.0,
                    num_response_tokens: num_loss_tokens,
                    perplexity: 1e6,
                };
            }
            None => {
                eprintln!("[CUDA] fused causal cross-entropy failed — falling back to CPU");
                return self.cuda_train_step_cpu_loss(
                    full_ids,
                    loss_start,
                    loss_end,
                    num_loss_tokens,
                    seq_len,
                    vocab_size,
                );
            }
        };

        // 3. GPU GEMM backward: grad_hidden = grad_logits @ embed (KAIZEN-064/065/068)
        self.profiler.begin(StepProfiler::LM_BWD);
        let hidden_size = self.model.config().hidden_size;

        let gemm_ok = (|| -> Option<()> {
            let trainer = self.cuda_trainer.as_ref()?;
            let stream = trainer.stream();
            let training = self.gpu_training.as_mut()?;
            if training.embed_original.len() < vocab_size * hidden_size {
                return None;
            }
            gemm_forward(
                &training.logits_buf,
                &training.embed_original,
                &mut training.grad_hidden_buf,
                seq_len as u32,
                vocab_size as u32,
                hidden_size as u32,
                stream,
            )
            .map_err(|e| eprintln!("[CUDA] lm_head backward GEMM failed: {e}"))
            .ok()?;
            Some(())
        })();

        self.profiler.end(StepProfiler::LM_BWD);

        if gemm_ok.is_none() {
            // PMAT-471: CPU fallback when GPU embeddings don't fit
            let cpu_ok = (|| -> Option<()> {
                let trainer = self.cuda_trainer.as_ref()?;
                let training = self.gpu_training.as_mut()?;
                let embed = self.model.embed_tokens.weight.data();
                let embed = embed.as_slice().expect("contiguous embed");
                super::super::gpu_backward_fallback::cpu_lmhead_backward(
                    trainer,
                    &training.logits_buf,
                    &mut training.grad_hidden_buf,
                    embed,
                    seq_len,
                    vocab_size,
                    hidden_size,
                    trainer.stream(),
                )
            })();
            if cpu_ok.is_none() {
                return InstructStepResult {
                    loss: avg_loss,
                    num_response_tokens: num_loss_tokens,
                    perplexity: avg_loss.exp().min(1e6),
                };
            }
        }

        // 4. GPU backward through NF4 blocks (KAIZEN-065: GPU-resident)
        self.profiler.begin(StepProfiler::BLK_BWD);
        if self.config.quantize_nf4 {
            self.backward_nf4_gpu_blocks_gpu_resident(seq_len);
        }
        self.profiler.end(StepProfiler::BLK_BWD);

        // PMAT-483: Feed per-layer timing from training state to profiler
        if let Some(ref training) = self.gpu_training {
            self.profiler.record_layer_times(
                &training.profiler_layer_fwd_us,
                &training.profiler_layer_bwd_us,
            );
        }

        // PMAT-483/entrenar#328: Feed per-op timing from scratch to profiler
        if let Some(ref scratch) = self.shared_scratch {
            if scratch.op_profiling_enabled {
                for (i, &us) in scratch.op_us.iter().enumerate() {
                    if us > 0 {
                        self.profiler.end_op_raw(i, us);
                    }
                }
            }
        }

        InstructStepResult {
            loss: avg_loss,
            num_response_tokens: num_loss_tokens,
            perplexity: avg_loss.exp().min(1e6),
        }
    }
    /// CPU fallback for causal LM loss when GPU fused kernel is unavailable.
    /// Used when forward_logits_gpu_resident or fused_causal_cross_entropy_cuda fails.
    #[cfg(feature = "cuda")]
    fn cuda_train_step_cpu_loss(
        &mut self,
        full_ids: &[u32],
        loss_start: usize,
        loss_end: usize,
        num_loss_tokens: usize,
        seq_len: usize,
        vocab_size: usize,
    ) -> InstructStepResult {
        // PMAT-420: Check if GPU embeddings are available. If not (VRAM-constrained),
        // skip forward_logits_gpu entirely to avoid CUDA context poisoning.
        let has_gpu_embed = self.gpu_training.as_ref().is_some_and(|t| {
            t.embed_original.len() >= vocab_size * self.model.config().hidden_size
        });

        let logits_data = if has_gpu_embed {
            match self.forward_logits_gpu(full_ids) {
                Some(data) => data,
                None => {
                    let logits = self.model.forward(full_ids);
                    logits.data().as_slice().expect("contiguous logits").to_vec()
                }
            }
        } else {
            // PMAT-420: Inference-style forward + save inputs for backward
            match self.forward_inference_saving_inputs(full_ids) {
                Some(data) => data,
                None => {
                    let logits = self.model.forward(full_ids);
                    logits.data().as_slice().expect("contiguous logits").to_vec()
                }
            }
        };

        let (avg_loss, grad_logits) =
            Self::compute_causal_lm_loss(&logits_data, full_ids, loss_start, loss_end, vocab_size);

        if !avg_loss.is_finite() {
            return InstructStepResult {
                loss: 100.0,
                num_response_tokens: num_loss_tokens,
                perplexity: 1e6,
            };
        }

        let hidden_size = self.model.config().hidden_size;

        let grad_hidden = (|| -> Option<Vec<f32>> {
            let trainer = self.cuda_trainer.as_ref()?;
            let stream = trainer.stream();
            let training = self.gpu_training.as_mut()?;
            if training.logits_buf.len() < grad_logits.len() {
                return None;
            }
            training
                .logits_buf
                .copy_from_host_at(&grad_logits, 0)
                .map_err(|e| eprintln!("[CUDA] lm_head backward: grad_logits upload failed: {e}"))
                .ok()?;
            if training.embed_original.len() < vocab_size * hidden_size {
                return None;
            }
            gemm_forward(
                &training.logits_buf,
                &training.embed_original,
                &mut training.grad_hidden_buf,
                seq_len as u32,
                vocab_size as u32,
                hidden_size as u32,
                stream,
            )
            .map_err(|e| eprintln!("[CUDA] lm_head backward GEMM failed: {e}"))
            .ok()?;
            stream.synchronize().ok()?;
            let full_grad = trainer.download(&training.grad_hidden_buf).ok()?;
            Some(full_grad[..seq_len * hidden_size].to_vec())
        })();

        let grad_hidden = match grad_hidden {
            Some(g) => g,
            None => {
                let hidden_size = self.model.config().hidden_size;
                let lm_weight =
                    self.model.lm_head.as_ref().unwrap_or(&self.model.embed_tokens.weight);
                let lm_data = lm_weight.data();
                let lm_slice = lm_data.as_slice().expect("contiguous lm_head");
                crate::autograd::ops::matmul::matmul_compute(
                    &grad_logits[..seq_len * vocab_size],
                    lm_slice,
                    seq_len,
                    vocab_size,
                    hidden_size,
                )
            }
        };

        if self.config.quantize_nf4 {
            let grad_nz = grad_hidden.iter().filter(|&&x| x != 0.0).count();
            static BWD_LOG: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if BWD_LOG.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 3 {
                eprintln!(
                    "[PMAT-420] backward: grad_hidden len={} nonzero={grad_nz} first5={:?}",
                    grad_hidden.len(),
                    &grad_hidden[..5.min(grad_hidden.len())]
                );
            }
            self.backward_nf4_gpu_blocks(&grad_hidden, seq_len);
        }

        InstructStepResult {
            loss: avg_loss,
            num_response_tokens: num_loss_tokens,
            perplexity: avg_loss.exp().min(1e6),
        }
    }
    /// Evaluate loss and perplexity on a set of samples without updating weights.
    ///
    /// # Correctness (C-QLORA-EVAL-SYNC-001)
    ///
    /// On the NF4 QLoRA (`cuda`) path, `train_step` writes adapter deltas into
    /// the GPU-resident `cuda_blocks`; the CPU `self.lora_layers` this method
    /// forwards through are only refreshed by `sync_lora_to_cpu()`. We therefore
    /// sync BEFORE evaluating so `val_loss` reflects the current trained
    /// adapters rather than a stale never-synced copy (which is constant across
    /// epochs and would freeze best-epoch selection at epoch 0). On the
    /// CPU/WGPU paths the sync is a no-op — those adapters are already current.
    pub fn evaluate(
        &mut self,
        prompt_ids_batch: &[Vec<u32>],
        response_ids_batch: &[Vec<u32>],
    ) -> InstructBatchResult {
        // FALSIFY-CUDA-EVAL-ADAPTER-SYNC-001: pull GPU-trained adapters to CPU.
        self.sync_lora_to_cpu();

        let mut total_loss = 0.0f32;
        let mut total_response_tokens = 0usize;

        for (prompt_ids, response_ids) in prompt_ids_batch.iter().zip(response_ids_batch.iter()) {
            let full_ids: Vec<u32> =
                prompt_ids.iter().chain(response_ids.iter()).copied().collect();

            let prompt_len = prompt_ids.len();
            if response_ids.is_empty() || full_ids.len() < 2 {
                continue;
            }

            let full_ids = if full_ids.len() > self.config.max_seq_len {
                full_ids[..self.config.max_seq_len].to_vec()
            } else {
                full_ids
            };
            let seq_len = full_ids.len();
            let vocab_size = self.model.config().vocab_size;
            let prompt_len = prompt_len.min(seq_len);

            // PMAT-805: evaluate the LoRA-adapted model so val_loss reflects the
            // trained adapters (matches train_step's forward_with_lora path).
            let logits = if self.lora_layers.is_empty() {
                self.model.forward(&full_ids)
            } else {
                self.model.forward_with_lora(&full_ids, &self.lora_layers)
            };
            let logits_data = logits.data().as_slice().expect("contiguous logits").to_vec();

            let loss_start = prompt_len.saturating_sub(1);
            let loss_end = seq_len - 1;
            let num_loss_tokens = loss_end.saturating_sub(loss_start);

            let (sample_loss, _) = Self::compute_causal_lm_loss(
                &logits_data,
                &full_ids,
                loss_start,
                loss_end,
                vocab_size,
            );

            total_loss += sample_loss * num_loss_tokens as f32;
            total_response_tokens += num_loss_tokens;
        }

        let avg_loss =
            if total_response_tokens > 0 { total_loss / total_response_tokens as f32 } else { 0.0 };

        InstructBatchResult {
            avg_loss,
            total_response_tokens,
            perplexity: avg_loss.exp().min(1e6),
            grad_norm: 0.0,
        }
    }
    /// Compute causal LM loss and gradients for the given position range.
    ///
    /// Returns (average_loss, gradient_logits).
    pub(super) fn compute_causal_lm_loss(
        logits_data: &[f32],
        full_ids: &[u32],
        loss_start: usize,
        loss_end: usize,
        vocab_size: usize,
    ) -> (f32, Vec<f32>) {
        let seq_len = full_ids.len();
        let num_loss_tokens = loss_end.saturating_sub(loss_start);
        let mut total_loss = 0.0f32;
        let mut grad_logits = vec![0.0f32; seq_len * vocab_size];

        for pos in loss_start..loss_end {
            let target = full_ids[pos + 1] as usize;
            if target >= vocab_size {
                continue;
            }

            let logit_start = pos * vocab_size;
            let row = &logits_data[logit_start..logit_start + vocab_size];

            let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let grad_row = &mut grad_logits[logit_start..logit_start + vocab_size];
            let mut sum_exp = 0.0f32;
            for j in 0..vocab_size {
                let exp_v = (row[j] - max_val).exp();
                grad_row[j] = exp_v;
                sum_exp += exp_v;
            }

            let log_sum_exp = sum_exp.ln() + max_val;
            let loss_i = -(row[target] - log_sum_exp);
            total_loss += if loss_i.is_finite() { loss_i } else { 100.0 };

            let inv_n = 1.0 / num_loss_tokens as f32;
            let scale = inv_n / sum_exp;
            for j in 0..vocab_size {
                grad_row[j] *= scale;
            }
            grad_row[target] -= inv_n;
        }

        let avg_loss = if num_loss_tokens > 0 { total_loss / num_loss_tokens as f32 } else { 0.0 };

        (avg_loss, grad_logits)
    }
}

/// Compute the causal-LM loss window for a CUDA training step.
///
/// FALSIFY-CUDA-LOSS-WINDOW-512-001: the effective sequence capacity is the
/// GPU scratch's row capacity when CUDA scratch exists (sized from
/// `config.max_seq_len` at init — the value `--max-seq-len` threads through),
/// with `max_position_embeddings.min(512)` ONLY as the no-scratch fallback.
/// This mirrors `forward_cuda_training` (cuda_forward.rs) exactly — the loss
/// window and the forward MUST clamp to the same capacity or response tokens
/// silently fall outside the trained window.
///
/// Pre-fix `cuda_train_step` hardcoded `.min(512)` unconditionally: with
/// `--max-seq-len 2048` the buffers held 2048 rows but the loss window was
/// still cut at 512, so any prompt longer than 512 tokens yielded
/// `num_loss_tokens == 0` → silent `loss=0.0`, zero gradient, no learning.
///
/// Returns `(loss_start, loss_end, num_loss_tokens, clamped_seq_len,
/// clamped_prompt_len)`.
#[cfg(feature = "cuda")]
pub(crate) fn cuda_loss_window(
    prompt_len: usize,
    seq_len: usize,
    scratch_capacity: Option<usize>,
    max_position_embeddings: usize,
) -> (usize, usize, usize, usize, usize) {
    let max_pos = scratch_capacity.unwrap_or_else(|| max_position_embeddings.min(512));
    let seq_len = seq_len.min(max_pos);
    let prompt_len = prompt_len.min(seq_len);
    let loss_start = prompt_len.saturating_sub(1);
    let loss_end = seq_len.saturating_sub(1);
    let num_loss_tokens = loss_end.saturating_sub(loss_start);
    (loss_start, loss_end, num_loss_tokens, seq_len, prompt_len)
}

#[cfg(all(test, feature = "cuda"))]
mod loss_window_tests {
    use super::cuda_loss_window;

    /// FALSIFY-CUDA-LOSS-WINDOW-512-001: with a scratch sized for 2048 rows
    /// (`--max-seq-len 2048`), a 600-token prompt + 100-token response MUST
    /// produce a non-empty loss window. Pre-fix the hardcoded `.min(512)`
    /// clamped seq AND prompt to 512 → `num_loss_tokens == 0` → the step
    /// silently returned loss=0.0 and the epoch trained nothing (observed
    /// live on the apr-code SFT corpus: every step 0.0/NaN, avg_loss=93.87
    /// from NaN sentinels, 559 loss tokens across 160 samples).
    #[test]
    fn falsify_cuda_loss_window_honors_scratch_capacity() {
        let (loss_start, loss_end, num_loss_tokens, seq_len, prompt_len) =
            cuda_loss_window(600, 700, Some(2048), 32768);
        assert_eq!(seq_len, 700, "seq must not be clamped below scratch capacity");
        assert_eq!(prompt_len, 600);
        assert_eq!(loss_start, 599);
        assert_eq!(loss_end, 699);
        assert_eq!(
            num_loss_tokens, 100,
            "FALSIFY-CUDA-LOSS-WINDOW-512-001: a 600-token prompt with a \
             2048-capacity scratch must keep all 100 response tokens in the \
             loss window (pre-fix: hardcoded .min(512) clamp yielded 0)"
        );
    }

    /// No-scratch fallback preserves the conservative entrenar#318 clamp.
    #[test]
    fn cuda_loss_window_no_scratch_falls_back_to_512() {
        let (_, _, num_loss_tokens, seq_len, prompt_len) = cuda_loss_window(600, 700, None, 32768);
        assert_eq!(seq_len, 512);
        assert_eq!(prompt_len, 512);
        assert_eq!(num_loss_tokens, 0, "prompt fills the fallback window — nothing to train");
    }

    /// Prompt overflowing even the scratch capacity yields an empty window
    /// (the caller must skip LOUDLY, never train on it silently).
    #[test]
    fn cuda_loss_window_prompt_overflow_is_empty() {
        let (_, _, num_loss_tokens, seq_len, prompt_len) =
            cuda_loss_window(3000, 3050, Some(2048), 32768);
        assert_eq!(seq_len, 2048);
        assert_eq!(prompt_len, 2048);
        assert_eq!(num_loss_tokens, 0);
    }

    /// max_position_embeddings below 512 still bounds the fallback.
    #[test]
    fn cuda_loss_window_small_model_fallback() {
        let (_, _, num_loss_tokens, seq_len, _) = cuda_loss_window(10, 40, None, 128);
        assert_eq!(seq_len, 40);
        assert_eq!(num_loss_tokens, 30);
    }
}
