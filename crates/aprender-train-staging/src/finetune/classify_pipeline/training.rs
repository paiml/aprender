impl ClassifyPipeline {
    /// Single training step: forward + loss + backward + optimizer update.
    ///
    /// Performs the complete training cycle:
    /// 1. Forward pass through transformer + classification head
    /// 2. Cross-entropy loss computation
    /// 3. Gradient computation via autograd backward
    /// 4. AdamW optimizer step on trainable parameters
    ///
    /// # Arguments
    /// * `token_ids` - Tokenized input
    /// * `label` - Target class index
    ///
    /// # Returns
    /// Loss value as f32
    pub fn train_step(&mut self, token_ids: &[u32], label: usize) -> f32 {
        let seq_len = token_ids.len().min(self.config.max_seq_len);
        let token_ids = &token_ids[..seq_len];
        let num_classes = self.config.num_classes;

        // ── 1. Zero gradients ─────────────────────────────────────────
        self.classifier.weight.zero_grad();
        self.classifier.bias.zero_grad();
        for lora in &self.lora_layers {
            lora.lora_a().zero_grad();
            lora.lora_b().zero_grad();
        }

        // ── 2. Forward pass (GPU-dispatched if available) ─────────────
        let hidden = self.forward_hidden_dispatch(token_ids);
        let pooled = self.classifier.mean_pool(&hidden, seq_len);

        // matmul builds autograd backward ops (connects classifier.weight to loss)
        let logits =
            matmul(&pooled, &self.classifier.weight, 1, self.classifier.hidden_size(), num_classes);

        // Add bias (element-wise, preserving grad tracking)
        let logits_with_bias: Vec<f32> = logits
            .data()
            .as_slice()
            .expect("contiguous logits")
            .iter()
            .zip(self.classifier.bias.data().as_slice().expect("contiguous bias").iter())
            .map(|(&l, &b)| l + b)
            .collect();

        // ── 3. Cross-entropy loss + manual gradient ───────────────────
        // Compute softmax probabilities
        let max_val = logits_with_bias.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_vals: Vec<f32> = logits_with_bias.iter().map(|&v| (v - max_val).exp()).collect();
        let sum_exp: f32 = exp_vals.iter().sum();
        let probs: Vec<f32> = exp_vals.iter().map(|&e| e / sum_exp).collect();

        // Class weight for this sample's label (default 1.0 if no weights configured)
        let w = self.config.class_weights.as_ref().map_or(1.0, |weights| weights[label]);

        // Loss = -w[label] * log(prob[target])
        let loss_val = -w * (probs[label].max(1e-10).ln());
        let loss_val = if loss_val.is_finite() { loss_val } else { 100.0 };

        // ∂L/∂logits = w[label] * (probs - one_hot(target))
        // The weight multiplier applies to both loss and gradient
        let mut grad_logits: Vec<f32> = probs.iter().map(|&p| w * p).collect();
        grad_logits[label] -= w;

        // ── 4. Backward through matmul (autograd) ─────────────────────
        // Set loss gradient on the matmul output, then call backward
        let logits_tensor = logits;
        logits_tensor.set_grad(ndarray::Array1::from(grad_logits.clone()));
        if let Some(op) = logits_tensor.backward_op() {
            op.backward();
        }

        // Manually set bias gradient (∂L/∂bias = ∂L/∂logits)
        self.classifier.bias.set_grad(ndarray::Array1::from(grad_logits.clone()));

        // GPU backward through transformer blocks (F-CUDA-014 / ENT-153)
        #[cfg(feature = "cuda")]
        {
            if self.gpu_training.is_some() {
                if self.config.quantize_nf4 {
                    // NF4 QLoRA: backward accumulates gradients (KAIZEN-014)
                    self.backward_nf4_gpu_blocks(&grad_logits, seq_len);
                } else {
                    self.backward_gpu_blocks(&grad_logits, seq_len);
                }
            }
        }

        // KAIZEN-014: NF4 QLoRA batch step (batch_size=1 for single-sample train_step)
        #[cfg(feature = "cuda")]
        {
            if self.gpu_training.is_some() && self.config.quantize_nf4 {
                self.nf4_lora_batch_optimizer_step(1);
            }
        }

        // ── 5. Optimizer step ─────────────────────────────────────────
        // GPU optimizer step for fp32 transformer block weights (F-CUDA-014)
        #[cfg(feature = "cuda")]
        {
            if self.gpu_training.is_some() && !self.config.quantize_nf4 {
                let lr = self.optimizer.lr();
                self.gpu_optimizer_step(lr);
            }
        }

        // CPU optimizer step
        // KAIZEN-011: Include LoRA in CPU optimizer when NOT on CUDA
        let has_cuda_training = {
            #[cfg(feature = "cuda")]
            {
                self.gpu_training.is_some()
            }
            #[cfg(not(feature = "cuda"))]
            {
                false
            }
        };
        let mut params: Vec<&mut Tensor> = Vec::new();
        if !self.config.quantize_nf4 || !has_cuda_training {
            for lora in &mut self.lora_layers {
                params.extend(lora.trainable_params());
            }
        }
        params.extend(self.classifier.parameters_mut());
        self.optimizer.step_refs(&mut params);

        loss_val
    }

    // ── Mini-batch training (SSC-025) ───────────────────────────────────

    /// Train on a mini-batch of samples with gradient accumulation.
    ///
    /// Unlike [`train_step`] which processes one sample and immediately calls
    /// `optimizer.step()`, this method:
    ///
    /// 1. Zeros all gradients
    /// 2. Iterates over every sample in the batch, computing forward + loss + backward
    /// 3. Gradients accumulate naturally across samples (sum)
    /// 4. Normalizes accumulated gradients by batch size
    /// 5. Optionally clips gradient norm (if `config.gradient_clip_norm` is set)
    /// 6. Calls `optimizer.step()` **once** for the entire batch
    ///
    /// This reduces optimizer overhead from O(N) to O(1) per batch and produces
    /// smoother gradient estimates.
    ///
    /// # Arguments
    /// * `samples` - Slice of `SafetySample` (shell text + label). Text is
    ///   tokenized via byte-level encoding internally.
    ///
    /// # Returns
    /// [`BatchResult`] with average loss, correct predictions, and total count
    pub fn train_batch(&mut self, samples: &[SafetySample]) -> BatchResult {
        if samples.is_empty() {
            return BatchResult { avg_loss: 0.0, correct: 0, total: 0, grad_norm: 0.0 };
        }

        let batch_size = samples.len();

        // ── 1. Zero gradients ──────────────────────────────────────────
        self.zero_all_gradients();

        // ── 2. Accumulate gradients over all samples ───────────────────
        // KAIZEN-008: try batched wgpu forward (uploads FFN weights ONCE per layer)
        #[cfg(feature = "gpu")]
        let (total_loss, correct) = self
            .try_train_batch_wgpu(samples)
            .unwrap_or_else(|| self.train_batch_per_sample(samples));

        #[cfg(not(feature = "gpu"))]
        let (total_loss, correct) = self.train_batch_per_sample(samples);

        // ── 3. Normalize gradients by batch size ───────────────────────
        self.scale_all_gradients(1.0 / batch_size as f32);

        // ── 4. Gradient clipping (captures pre-clip norm) ────────────
        let grad_norm = if let Some(max_norm) = self.config.gradient_clip_norm {
            let mut params = self.trainable_parameters_mut();
            clip_grad_norm_refs(&mut params, max_norm)
        } else {
            self.compute_grad_norm()
        };

        // ── 5. Optimizer step (once for the whole batch) ───────────────
        self.apply_optimizer_step(batch_size);

        BatchResult {
            avg_loss: total_loss / batch_size as f32,
            correct,
            total: batch_size,
            grad_norm,
        }
    }

    /// Combined GPU + CPU optimizer step dispatch (extracted from train_batch).
    #[allow(unused_variables)]
    fn apply_optimizer_step(&mut self, batch_size: usize) {
        // GPU optimizer step for fp32 transformer block weights (F-CUDA-014)
        #[cfg(feature = "cuda")]
        {
            if self.gpu_training.is_some() && !self.config.quantize_nf4 {
                let lr = self.optimizer.lr();
                self.gpu_optimizer_step(lr);
            }
        }
        // NF4 QLoRA: batch optimizer step (KAIZEN-014)
        #[cfg(feature = "cuda")]
        {
            if self.gpu_training.is_some() && self.config.quantize_nf4 {
                self.nf4_lora_batch_optimizer_step(batch_size);
            }
        }
        // CPU optimizer step (KAIZEN-011)
        let has_cuda_training = {
            #[cfg(feature = "cuda")]
            {
                self.gpu_training.is_some()
            }
            #[cfg(not(feature = "cuda"))]
            {
                false
            }
        };
        let mut params: Vec<&mut Tensor> = Vec::new();
        if !self.config.quantize_nf4 || !has_cuda_training {
            for lora in &mut self.lora_layers {
                params.extend(lora.trainable_params());
            }
        }
        params.extend(self.classifier.parameters_mut());
        self.optimizer.step_refs(&mut params);
    }

    /// Per-sample forward + backward fallback for train_batch.
    fn train_batch_per_sample(&mut self, samples: &[SafetySample]) -> (f32, usize) {
        let mut total_loss = 0.0f32;
        let mut correct = 0usize;
        for sample in samples {
            let ids = self.tokenize(&sample.input);
            let (loss, predicted) = self.forward_backward_single(&ids, sample.label);
            total_loss += loss;
            if predicted == sample.label {
                correct += 1;
            }
        }
        (total_loss, correct)
    }

    /// Batched wgpu forward pass for train_batch (KAIZEN-008 + KAIZEN-010).
    ///
    /// Tokenizes all samples, runs a single batched forward through all transformer
    /// layers (uploading FFN weights ONCE per layer), then classifies each sample.
    ///
    /// KAIZEN-010: Passes LoRA layers to the batched forward so that LoRA
    /// adjusts are applied to Q/V projections. Without this, only the
    /// classifier head (5,122 params) trains on the wgpu path.
    ///
    /// Returns `Some((total_loss, correct))` on success, `None` to fall back.
    #[cfg(feature = "gpu")]
    fn try_train_batch_wgpu(&mut self, samples: &[SafetySample]) -> Option<(f32, usize)> {
        self.wgpu_forward_pass.as_ref()?;

        let batch_token_ids: Vec<Vec<u32>> =
            samples.iter().map(|s| self.tokenize(&s.input)).collect();

        // KAIZEN-010: Pass LoRA layers so gradients flow through Q/V adapters
        let lora_ref =
            if self.lora_layers.is_empty() { None } else { Some(self.lora_layers.as_slice()) };

        let hiddens = self
            .wgpu_forward_pass
            .as_ref()
            .expect("checked is_none above")
            .forward_hidden_batch(&self.model, &batch_token_ids, lora_ref)
            .map_err(|e| {
                eprintln!("[wgpu] Batched forward failed, falling back to per-sample: {e}");
            })
            .ok()?;

        let mut total_loss = 0.0f32;
        let mut correct = 0usize;
        for (i, hidden) in hiddens.iter().enumerate() {
            let (loss, predicted) = self.classify_backward_from_hidden(
                hidden,
                batch_token_ids[i].len(),
                samples[i].label,
            );
            total_loss += loss;
            if predicted == samples[i].label {
                correct += 1;
            }
        }
        Some((total_loss, correct))
    }

    /// Accumulate gradients for a micro-batch without calling optimizer.step().
    ///
    /// Use this with [`apply_accumulated_gradients`] for gradient accumulation
    /// across multiple micro-batches. This enables effective batch sizes larger
    /// than what fits in memory:
    ///
    /// ```text
    /// effective_batch_size = micro_batch_size * accumulation_steps
    /// ```
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Effective batch_size = 8 * 4 = 32
    /// for micro_batch in data.chunks(8) {
    ///     pipeline.accumulate_gradients(micro_batch);
    /// }
    /// pipeline.apply_accumulated_gradients(4);
    /// ```
    ///
    /// # Arguments
    /// * `micro_batch` - Slice of samples for one accumulation step
    ///
    /// # Returns
    /// [`BatchResult`] for this micro-batch (loss/accuracy before optimizer step)
    pub fn accumulate_gradients(&mut self, micro_batch: &[SafetySample]) -> BatchResult {
        if micro_batch.is_empty() {
            return BatchResult { avg_loss: 0.0, correct: 0, total: 0, grad_norm: 0.0 };
        }

        let mut total_loss = 0.0f32;
        let mut correct = 0usize;

        for sample in micro_batch {
            let ids = self.tokenize(&sample.input);
            let (loss, predicted) = self.forward_backward_single(&ids, sample.label);
            total_loss += loss;
            if predicted == sample.label {
                correct += 1;
            }
        }

        BatchResult {
            avg_loss: total_loss / micro_batch.len() as f32,
            correct,
            total: micro_batch.len(),
            grad_norm: 0.0, // Grad norm computed at apply time, not accumulate time
        }
    }

    /// Normalize accumulated gradients and apply optimizer step.
    ///
    /// Call this after one or more [`accumulate_gradients`] calls. It:
    /// 1. Divides all gradients by `num_accumulation_steps * micro_batch_size`
    ///    (the total sample count across all micro-batches)
    /// 2. Clips gradient norm if configured
    /// 3. Calls `optimizer.step()` once
    /// 4. Zeros all gradients for the next accumulation cycle
    ///
    /// # Arguments
    /// * `total_samples` - Total number of samples accumulated (sum of micro-batch sizes)
    pub fn apply_accumulated_gradients(&mut self, total_samples: usize) {
        if total_samples == 0 {
            return;
        }

        // ── 1. Normalize gradients ─────────────────────────────────────
        self.scale_all_gradients(1.0 / total_samples as f32);

        // ── 2. Gradient clipping ───────────────────────────────────────
        if let Some(max_norm) = self.config.gradient_clip_norm {
            let mut params = self.trainable_parameters_mut();
            clip_grad_norm_refs(&mut params, max_norm);
        }

        // ── 3. Optimizer step ──────────────────────────────────────────
        // NF4 mode: only classifier head (LoRA trained on GPU in backward)
        let mut params: Vec<&mut Tensor> = Vec::new();
        if !self.config.quantize_nf4 {
            for lora in &mut self.lora_layers {
                params.extend(lora.trainable_params());
            }
        }
        params.extend(self.classifier.parameters_mut());
        self.optimizer.step_refs(&mut params);

        // ── 4. Zero gradients for next cycle ───────────────────────────
        self.zero_all_gradients();
    }

    /// Forward pass + backward for a single sample (no optimizer step).
    ///
    /// Computes cross-entropy loss and accumulates gradients into the existing
    /// gradient buffers (does NOT zero them first). Returns the loss and
    /// the predicted class index (argmax of logits).
    fn forward_backward_single(&mut self, token_ids: &[u32], label: usize) -> (f32, usize) {
        let num_classes = self.config.num_classes;

        // ── Contract precondition (F-CLASS-002): label in bounds ─────────
        debug_assert!(
            label < num_classes,
            "F-CLASS-002: label index {label} >= num_classes {num_classes}"
        );

        // ── Pad to max_seq_len for deterministic GPU kernel shapes (C-PREWARM-001) ──
        // GPU backward kernels embed seq_len in cache keys. Pre-warming compiles for
        // max_seq_len only, so variable-length inputs would trigger JIT compilation
        // post-VRAM-fill. Pad to max_seq_len; mean_pool uses orig_seq_len for correctness.
        let orig_seq_len = token_ids.len().min(self.config.max_seq_len);
        let has_gpu_training = {
            #[cfg(feature = "cuda")]
            {
                self.gpu_training.is_some()
            }
            #[cfg(not(feature = "cuda"))]
            {
                false
            }
        };
        // KAIZEN-035: Avoid token_ids.to_vec() on CPU path — borrow directly.
        let hidden = if has_gpu_training {
            let mut padded = vec![0u32; self.config.max_seq_len];
            padded[..orig_seq_len].copy_from_slice(&token_ids[..orig_seq_len]);
            self.forward_hidden_dispatch(&padded)
        } else {
            self.forward_hidden_dispatch(token_ids)
        };
        let pooled = self.classifier.mean_pool(&hidden, orig_seq_len);

        let logits =
            matmul(&pooled, &self.classifier.weight, 1, self.classifier.hidden_size(), num_classes);

        let logits_with_bias: Vec<f32> = logits
            .data()
            .as_slice()
            .expect("contiguous logits")
            .iter()
            .zip(self.classifier.bias.data().as_slice().expect("contiguous bias").iter())
            .map(|(&l, &b)| l + b)
            .collect();

        // ── Contract postcondition (F-CLASS-001): logit shape ────────────
        debug_assert_eq!(
            logits_with_bias.len(),
            num_classes,
            "F-CLASS-001: logits.len()={} != num_classes={num_classes}",
            logits_with_bias.len()
        );
        // ── Contract postcondition: no NaN in logits ────────────────────
        debug_assert!(
            logits_with_bias.iter().all(|v| v.is_finite()),
            "F-CLASS-001: logits contain NaN or Inf"
        );

        // ── Predicted class (argmax) ────────────────────────────────────
        let predicted = logits_with_bias
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(idx, _)| idx);

        // ── Cross-entropy loss (weighted) ────────────────────────────────
        let max_val = logits_with_bias.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_vals: Vec<f32> = logits_with_bias.iter().map(|&v| (v - max_val).exp()).collect();
        let sum_exp: f32 = exp_vals.iter().sum();
        let probs: Vec<f32> = exp_vals.iter().map(|&e| e / sum_exp).collect();

        // Class weight for this sample's label (default 1.0 if no weights configured)
        let w = self.config.class_weights.as_ref().map_or(1.0, |weights| weights[label]);

        let loss_val = -w * (probs[label].max(1e-10).ln());
        let loss_val = if loss_val.is_finite() { loss_val } else { 100.0 };

        // ── Contract postcondition (F-CLASS-005): loss finite & non-negative
        debug_assert!(loss_val.is_finite(), "F-CLASS-005: loss is not finite");
        debug_assert!(loss_val >= 0.0, "F-CLASS-005: loss is negative: {loss_val}");

        // ── Backward ────────────────────────────────────────────────────
        // dL/d_logits = w[label] * (softmax(logits) - one_hot(label))
        let mut grad_logits: Vec<f32> = probs.iter().map(|&p| w * p).collect();
        grad_logits[label] -= w;

        // CPU autograd backward for classifier head (LoRA + classifier.weight)
        logits.set_grad(ndarray::Array1::from(grad_logits.clone()));
        if let Some(op) = logits.backward_op() {
            op.backward();
        }

        // Accumulate bias gradient (not set — accumulate)
        self.classifier.bias.accumulate_grad(ndarray::Array1::from(grad_logits.clone()));

        // GPU backward through all transformer blocks (F-CUDA-014 / ENT-153)
        #[cfg(feature = "cuda")]
        {
            if self.gpu_training.is_some() {
                if self.config.quantize_nf4 {
                    self.backward_nf4_gpu_blocks(&grad_logits, orig_seq_len);
                } else {
                    self.backward_gpu_blocks(&grad_logits, orig_seq_len);
                }
            }
        }

        (loss_val, predicted)
    }

    /// Classifier head + loss + backward from pre-computed hidden states (KAIZEN-008).
    ///
    /// Extracts the classify-and-backward logic from `forward_backward_single` for use
    /// with batched wgpu forward pass, where hidden states are computed in bulk.
    ///
    /// # Contract (C-WGPU-BATCH-001)
    ///
    /// - **Precondition**: hidden tensor has shape (seq_len * hidden_size), label < num_classes
    /// - **Postcondition**: gradients accumulated into classifier.weight, classifier.bias, LoRA params
    /// - **Invariant**: numerically identical to forward_backward_single classifier path
    #[cfg(feature = "gpu")]
    fn classify_backward_from_hidden(
        &mut self,
        hidden: &Tensor,
        orig_seq_len: usize,
        label: usize,
    ) -> (f32, usize) {
        let num_classes = self.config.num_classes;

        debug_assert!(
            label < num_classes,
            "F-CLASS-002: label index {label} >= num_classes {num_classes}"
        );

        // ── Classifier forward ────────────────────────────────────────
        let pooled = self.classifier.mean_pool(hidden, orig_seq_len);
        let logits =
            matmul(&pooled, &self.classifier.weight, 1, self.classifier.hidden_size(), num_classes);

        let logits_with_bias: Vec<f32> = logits
            .data()
            .as_slice()
            .expect("contiguous logits")
            .iter()
            .zip(self.classifier.bias.data().as_slice().expect("contiguous bias").iter())
            .map(|(&l, &b)| l + b)
            .collect();

        debug_assert_eq!(
            logits_with_bias.len(),
            num_classes,
            "F-CLASS-001: logits.len()={} != num_classes={num_classes}",
            logits_with_bias.len()
        );
        debug_assert!(
            logits_with_bias.iter().all(|v| v.is_finite()),
            "F-CLASS-001: logits contain NaN or Inf"
        );

        // ── Predicted class (argmax) ────────────────────────────────────
        let predicted = logits_with_bias
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(idx, _)| idx);

        // ── Cross-entropy loss (weighted) ────────────────────────────────
        let max_val = logits_with_bias.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_vals: Vec<f32> = logits_with_bias.iter().map(|&v| (v - max_val).exp()).collect();
        let sum_exp: f32 = exp_vals.iter().sum();
        let probs: Vec<f32> = exp_vals.iter().map(|&e| e / sum_exp).collect();

        let w = self.config.class_weights.as_ref().map_or(1.0, |weights| weights[label]);
        let loss_val = -w * (probs[label].max(1e-10).ln());
        let loss_val = if loss_val.is_finite() { loss_val } else { 100.0 };

        // ── Backward ────────────────────────────────────────────────────
        let mut grad_logits: Vec<f32> = probs.iter().map(|&p| w * p).collect();
        grad_logits[label] -= w;

        logits.set_grad(ndarray::Array1::from(grad_logits.clone()));
        if let Some(op) = logits.backward_op() {
            op.backward();
        }

        self.classifier.bias.accumulate_grad(ndarray::Array1::from(grad_logits));

        (loss_val, predicted)
    }

    /// Zero all trainable parameter gradients.
    fn zero_all_gradients(&self) {
        self.classifier.weight.zero_grad();
        self.classifier.bias.zero_grad();
        for lora in &self.lora_layers {
            lora.lora_a().zero_grad();
            lora.lora_b().zero_grad();
        }
    }

    /// Scale all trainable parameter gradients by a constant factor.
    ///
    /// Used to normalize accumulated gradients: `grad *= factor`.
    fn scale_all_gradients(&self, factor: f32) {
        // KAIZEN-037: scale in-place — zero allocation (was: clone + alloc per param)
        for lora in &self.lora_layers {
            lora.lora_a().scale_grad(factor);
            lora.lora_b().scale_grad(factor);
        }
        for param in self.classifier.parameters() {
            param.scale_grad(factor);
        }
    }

    /// Compute the global L2 norm of all trainable gradients.
    ///
    /// Used by the monitor when gradient clipping is not enabled.
    fn compute_grad_norm(&self) -> f32 {
        let mut total_norm_sq = 0.0f32;
        // KAIZEN-037: iterate directly — no intermediate Vec collection
        for lora in &self.lora_layers {
            for param in [lora.lora_a(), lora.lora_b()] {
                if let Some(grad) = param.grad() {
                    total_norm_sq += grad.iter().map(|&g| g * g).sum::<f32>();
                }
            }
        }
        for param in self.classifier.parameters() {
            if let Some(grad) = param.grad() {
                total_norm_sq += grad.iter().map(|&g| g * g).sum::<f32>();
            }
        }
        total_norm_sq.sqrt()
    }

    /// Forward-only pass for a single sample (no backward, no optimizer step).
    ///
    /// Computes cross-entropy loss and predicted class without accumulating
    /// gradients. Used for validation/evaluation.
    ///
    /// # Arguments
    /// * `token_ids` - Tokenized input
    /// * `label` - Target class index
    ///
    /// # Returns
    /// `(loss, predicted_class)` tuple
    pub fn forward_only(&mut self, token_ids: &[u32], label: usize) -> (f32, usize) {
        let num_classes = self.config.num_classes;

        // Pad to max_seq_len for deterministic GPU kernel shapes (matches forward_backward_single)
        let orig_seq_len = token_ids.len().min(self.config.max_seq_len);
        let has_gpu_training = {
            #[cfg(feature = "cuda")]
            {
                self.gpu_training.is_some()
            }
            #[cfg(not(feature = "cuda"))]
            {
                false
            }
        };
        // KAIZEN-035: Avoid token_ids.to_vec() on CPU path — borrow directly.
        let hidden = if has_gpu_training {
            let mut padded = vec![0u32; self.config.max_seq_len];
            padded[..orig_seq_len].copy_from_slice(&token_ids[..orig_seq_len]);
            self.forward_hidden_dispatch(&padded)
        } else {
            self.forward_hidden_dispatch(token_ids)
        };
        let pooled = self.classifier.mean_pool(&hidden, orig_seq_len);

        let logits =
            matmul(&pooled, &self.classifier.weight, 1, self.classifier.hidden_size(), num_classes);

        let logits_with_bias: Vec<f32> = logits
            .data()
            .as_slice()
            .expect("contiguous logits")
            .iter()
            .zip(self.classifier.bias.data().as_slice().expect("contiguous bias").iter())
            .map(|(&l, &b)| l + b)
            .collect();

        // Predicted class (argmax)
        let predicted = logits_with_bias
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(idx, _)| idx);

        // Cross-entropy loss (weighted)
        let max_val = logits_with_bias.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_vals: Vec<f32> = logits_with_bias.iter().map(|&v| (v - max_val).exp()).collect();
        let sum_exp: f32 = exp_vals.iter().sum();
        let probs: Vec<f32> = exp_vals.iter().map(|&e| e / sum_exp).collect();

        // Class weight for this sample's label (default 1.0 if no weights configured)
        let w = self.config.class_weights.as_ref().map_or(1.0, |weights| weights[label]);

        let loss_val = -w * (probs[label].max(1e-10).ln());
        let loss_val = if loss_val.is_finite() { loss_val } else { 100.0 };

        (loss_val, predicted)
    }

    /// Forward-only pass returning loss, predicted class, and softmax probabilities.
    ///
    /// Identical to [`forward_only`] but also returns the full probability distribution
    /// for confidence analysis, calibration, and per-sample diagnostics.
    pub fn forward_only_with_probs(
        &mut self,
        token_ids: &[u32],
        label: usize,
    ) -> (f32, usize, Vec<f32>) {
        let seq_len = token_ids.len().min(self.config.max_seq_len);
        let token_ids = &token_ids[..seq_len];
        let num_classes = self.config.num_classes;

        let hidden = self.forward_hidden_dispatch(token_ids);
        let pooled = self.classifier.mean_pool(&hidden, seq_len);

        let logits =
            matmul(&pooled, &self.classifier.weight, 1, self.classifier.hidden_size(), num_classes);

        let logits_with_bias: Vec<f32> = logits
            .data()
            .as_slice()
            .expect("contiguous logits")
            .iter()
            .zip(self.classifier.bias.data().as_slice().expect("contiguous bias").iter())
            .map(|(&l, &b)| l + b)
            .collect();

        let predicted = logits_with_bias
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(idx, _)| idx);

        let max_val = logits_with_bias.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_vals: Vec<f32> = logits_with_bias.iter().map(|&v| (v - max_val).exp()).collect();
        let sum_exp: f32 = exp_vals.iter().sum();
        let probs: Vec<f32> = exp_vals.iter().map(|&e| e / sum_exp).collect();

        let w = self.config.class_weights.as_ref().map_or(1.0, |weights| weights[label]);
        let loss_val = -w * (probs[label].max(1e-10).ln());
        let loss_val = if loss_val.is_finite() { loss_val } else { 100.0 };

        (loss_val, predicted, probs)
    }

    /// Multi-label training step using BCE with logits loss.
    ///
    /// Unlike `train_step` which uses cross-entropy (mutually exclusive classes),
    /// this uses BCE with logits (each class is independent binary decision).
    ///
    /// # Arguments
    /// * `token_ids` - Tokenized input
    /// * `targets` - Multi-hot target vector (length == num_classes)
    ///
    /// # Returns
    /// Loss value as f32
    pub fn multi_label_train_step(&mut self, token_ids: &[u32], targets: &[f32]) -> f32 {
        let seq_len = token_ids.len().min(self.config.max_seq_len);
        let token_ids = &token_ids[..seq_len];
        let num_classes = self.config.num_classes;
        assert_eq!(targets.len(), num_classes, "F-CLASS-001: target length must match num_classes");

        // ── 1. Zero gradients ─────────────────────────────────────────
        self.classifier.weight.zero_grad();
        self.classifier.bias.zero_grad();
        for lora in &self.lora_layers {
            lora.lora_a().zero_grad();
            lora.lora_b().zero_grad();
        }

        // ── 2. Forward pass (GPU-dispatched if available) ─────────────
        let hidden = self.forward_hidden_dispatch(token_ids);
        let pooled = self.classifier.mean_pool(&hidden, seq_len);

        let logits =
            matmul(&pooled, &self.classifier.weight, 1, self.classifier.hidden_size(), num_classes);

        // Add bias
        let logits_with_bias: Vec<f32> = logits
            .data()
            .as_slice()
            .expect("contiguous logits")
            .iter()
            .zip(self.classifier.bias.data().as_slice().expect("contiguous bias").iter())
            .map(|(&l, &b)| l + b)
            .collect();

        // ── 3. BCE with logits loss + manual gradient ───────────────
        // Per-element: L_i = max(x_i, 0) - x_i * t_i + log(1 + exp(-|x_i|))
        let loss_val: f32 = logits_with_bias
            .iter()
            .zip(targets.iter())
            .map(|(&x, &t)| {
                let relu = x.max(0.0);
                relu - x * t + (1.0 + (-x.abs()).exp()).ln()
            })
            .sum::<f32>()
            / num_classes as f32;

        let loss_val = if loss_val.is_finite() { loss_val } else { 100.0 };

        // ∂L/∂logits = (σ(x) - targets) / N
        let grad_logits: Vec<f32> = logits_with_bias
            .iter()
            .zip(targets.iter())
            .map(|(&x, &t)| {
                let sigma = if x >= 0.0 {
                    1.0 / (1.0 + (-x).exp())
                } else {
                    let e = x.exp();
                    e / (1.0 + e)
                };
                (sigma - t) / num_classes as f32
            })
            .collect();

        // ── 4. Backward through matmul (autograd) ─────────────────────
        let logits_tensor = logits;
        logits_tensor.set_grad(ndarray::Array1::from(grad_logits.clone()));
        if let Some(op) = logits_tensor.backward_op() {
            op.backward();
        }

        // Manually set bias gradient
        self.classifier.bias.set_grad(ndarray::Array1::from(grad_logits));

        // ── 5. Optimizer step ─────────────────────────────────────────
        let mut params: Vec<&mut Tensor> = Vec::new();
        for lora in &mut self.lora_layers {
            params.extend(lora.trainable_params());
        }
        params.extend(self.classifier.parameters_mut());
        self.optimizer.step_refs(&mut params);

        loss_val
    }

    /// Load multi-label corpus from JSONL file.
    ///
    /// Supports both single-label `{"input","label"}` and multi-label `{"input","labels"}`
    /// formats. Single-label entries are automatically converted to multi-hot encoding.
    ///
    /// # Errors
    /// Returns error if file is invalid or labels out of range.
    pub fn load_multi_label_corpus(
        &self,
        path: &Path,
    ) -> crate::Result<Vec<MultiLabelSafetySample>> {
        load_multi_label_corpus(path, self.config.num_classes)
    }

    /// Get all trainable parameters (LoRA A/B + classifier weight/bias).
    pub fn trainable_parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params: Vec<&mut Tensor> = Vec::new();
        for lora in &mut self.lora_layers {
            params.extend(lora.trainable_params());
        }
        params.extend(self.classifier.parameters_mut());
        params
    }

    /// Count total trainable parameters.
    #[must_use]
    pub fn num_trainable_parameters(&self) -> usize {
        let lora_params: usize =
            self.lora_layers.iter().map(|l: &LoRALayer| l.rank() * (l.d_in() + l.d_out())).sum();
        lora_params + self.classifier.num_parameters()
    }

    /// Load corpus from JSONL file.
    ///
    /// # Errors
    /// Returns error if file is invalid or labels out of range.
    pub fn load_corpus(&self, path: &Path) -> crate::Result<Vec<SafetySample>> {
        load_safety_corpus(path, self.config.num_classes)
    }

    /// Merge all LoRA adapters into base weights (for inference).
    pub fn merge_adapters(&mut self) {
        for lora in &mut self.lora_layers {
            lora.merge();
        }
    }

    /// Set the learning rate of the internal optimizer.
    ///
    /// Used by `ClassifyTrainer` to apply LR scheduling.
    pub fn set_optimizer_lr(&mut self, lr: f32) {
        self.optimizer.set_lr(lr);
    }

    /// Get the current learning rate of the internal optimizer.
    #[must_use]
    pub fn optimizer_lr(&self) -> f32 {
        self.optimizer.lr()
    }

    /// Get a reference to the internal AdamW optimizer (F-CKPT-004).
    #[must_use]
    pub fn optimizer(&self) -> &AdamW {
        &self.optimizer
    }

    /// Get a mutable reference to the internal AdamW optimizer (F-CKPT-004).
    pub fn optimizer_mut(&mut self) -> &mut AdamW {
        &mut self.optimizer
    }

    /// Summary of the pipeline configuration.
    #[must_use]
    pub fn summary(&self) -> String {
        let tokenizer_info = if let Some(ref tok) = self.tokenizer {
            format!("BPE (vocab={})", tok.vocab_size())
        } else {
            "byte-level (256)".to_string()
        };
        let device_info = if let Some(name) = self.gpu_name() {
            format!("CUDA ({name})")
        } else {
            "CPU".to_string()
        };
        format!(
            "ClassifyPipeline:\n  Model: {} hidden, {} layers\n  Device: {}\n  Tokenizer: {}\n  LoRA: rank={}, alpha={:.1}, {} adapters\n  Classifier: {}->{} ({} params)\n  Total trainable: {} params",
            self.model.config.hidden_size,
            self.model.config.num_hidden_layers,
            device_info,
            tokenizer_info,
            self.config.lora_rank,
            self.config.lora_alpha,
            self.lora_layers.len(),
            self.classifier.hidden_size(),
            self.classifier.num_classes(),
            self.classifier.num_parameters(),
            self.num_trainable_parameters(),
        )
    }

    /// Collect all LoRA + classifier gradients into a flat `Vec<f32>`.
    ///
    /// Used by distributed training workers to send gradients to the coordinator
    /// for AllReduce averaging (F-DP-001).
    ///
    /// Layout: `[lora_0_a_grad, lora_0_b_grad, ..., lora_N_a_grad, lora_N_b_grad,
    ///           classifier_weight_grad, classifier_bias_grad]`
    #[must_use]
    pub fn collect_lora_gradients(&self) -> Vec<f32> {
        let total = self.num_trainable_parameters();
        let mut grads = Vec::with_capacity(total);

        for lora in &self.lora_layers {
            if let Some(g) = lora.lora_a().grad() {
                grads.extend(g.iter());
            } else {
                grads.extend(std::iter::repeat_n(0.0f32, lora.lora_a().data().len()));
            }
            if let Some(g) = lora.lora_b().grad() {
                grads.extend(g.iter());
            } else {
                grads.extend(std::iter::repeat_n(0.0f32, lora.lora_b().data().len()));
            }
        }

        if let Some(g) = self.classifier.weight.grad() {
            grads.extend(g.iter());
        } else {
            grads.extend(std::iter::repeat_n(0.0f32, self.classifier.weight.data().len()));
        }
        if let Some(g) = self.classifier.bias.grad() {
            grads.extend(g.iter());
        } else {
            grads.extend(std::iter::repeat_n(0.0f32, self.classifier.bias.data().len()));
        }

        grads
    }

    /// Apply averaged gradients from AllReduce and run optimizer step.
    ///
    /// Used by distributed training workers after receiving averaged gradients
    /// from the coordinator (F-DP-001 weight consistency).
    ///
    /// The gradient vector layout must match `collect_lora_gradients()`.
    pub fn apply_lora_gradients(&mut self, averaged_grads: &[f32]) {
        let mut offset = 0usize;

        // Write averaged gradients into each parameter's grad slot
        for lora in &self.lora_layers {
            let a_len = lora.lora_a().data().len();
            if offset + a_len <= averaged_grads.len() {
                lora.lora_a().set_grad(ndarray::Array1::from_vec(
                    averaged_grads[offset..offset + a_len].to_vec(),
                ));
            }
            offset += a_len;

            let b_len = lora.lora_b().data().len();
            if offset + b_len <= averaged_grads.len() {
                lora.lora_b().set_grad(ndarray::Array1::from_vec(
                    averaged_grads[offset..offset + b_len].to_vec(),
                ));
            }
            offset += b_len;
        }

        let w_len = self.classifier.weight.data().len();
        if offset + w_len <= averaged_grads.len() {
            self.classifier.weight.set_grad(ndarray::Array1::from_vec(
                averaged_grads[offset..offset + w_len].to_vec(),
            ));
        }
        offset += w_len;

        let b_len = self.classifier.bias.data().len();
        if offset + b_len <= averaged_grads.len() {
            self.classifier.bias.set_grad(ndarray::Array1::from_vec(
                averaged_grads[offset..offset + b_len].to_vec(),
            ));
        }

        // Now run optimizer step with the averaged gradients
        let mut params: Vec<&mut Tensor> = Vec::new();
        for lora in &mut self.lora_layers {
            params.extend(lora.trainable_params());
        }
        params.extend(self.classifier.parameters_mut());
        self.optimizer.step_refs(&mut params);
    }
}
