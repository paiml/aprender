//! Transformer trainer implementation

use crate::autograd::{checkpoint, GradScaler};
use crate::io::{save_model, Model, ModelFormat, ModelMetadata, SaveConfig};
use crate::lora::LoRALayer;
use crate::optim::{AdamW, Optimizer};
use crate::train::{CausalLMLoss, LossFn, MetricsTracker};
use crate::transformer::Transformer;
use crate::Tensor;
use std::path::Path;

use super::batch::LMBatch;
use super::config::TransformerTrainConfig;

/// Transformer training state
pub struct TransformerTrainer {
    /// Model
    model: Transformer,
    /// Loss function
    loss_fn: CausalLMLoss,
    /// Optimizer
    optimizer: AdamW,
    /// Gradient scaler for mixed precision
    grad_scaler: GradScaler,
    /// Configuration
    config: TransformerTrainConfig,
    /// Metrics tracker
    pub metrics: MetricsTracker,
    /// Current step
    step: usize,
    /// Accumulated gradients (for gradient accumulation)
    accumulated_loss: f32,
    /// Number of accumulated batches
    accumulated_batches: usize,
    /// LoRA layers (ENT-LoRA-001): [Q_0, V_0, Q_1, V_1, ...] per transformer layer
    /// None = full fine-tuning, Some = LoRA fine-tuning
    lora_layers: Option<Vec<LoRALayer>>,
}

impl TransformerTrainer {
    /// Create a new transformer trainer
    pub fn new(config: TransformerTrainConfig) -> Self {
        let model = Transformer::new(&config.model_config);
        Self::build(model, config)
    }

    /// Create trainer from existing model
    pub fn with_model(model: Transformer, config: TransformerTrainConfig) -> Self {
        Self::build(model, config)
    }

    /// Internal builder: initializes LoRA layers when config has LoRA enabled
    fn build(model: Transformer, config: TransformerTrainConfig) -> Self {
        let loss_fn = CausalLMLoss::new(config.model_config.vocab_size);
        let optimizer = AdamW::default_params(config.lr);
        let grad_scaler = GradScaler::from_config(&config.precision_config);

        // ENT-LoRA-001: Create LoRA layers when config has LoRA rank
        let lora_layers = if let Some(rank) = config.lora_rank {
            let alpha = config.lora_alpha.unwrap_or(rank as f32 * 2.0);
            let default_targets = vec!["q_proj".to_string(), "v_proj".to_string()];
            // ENT-LoRA-005: Expand shorthand targets ("all_linear", "attention", etc.)
            let raw_targets = config.lora_target_modules.as_deref().unwrap_or(&default_targets);
            let expanded = crate::lora::LoRAConfig::expand_shorthand(raw_targets);
            let target_modules = expanded.as_slice();

            let mut layers = Vec::new();
            let hidden_size = config.model_config.hidden_size;
            let num_kv_heads = config.model_config.num_kv_heads;
            let head_dim = config.model_config.head_dim();
            let q_dim = config.model_config.q_dim();
            let kv_hidden_size = num_kv_heads * head_dim;

            let intermediate = config.model_config.intermediate_size;

            for block in &model.layers {
                // Attention projections (ENT-LoRA-005: flexible targets)
                if target_modules.iter().any(|m| m == "q_proj") {
                    layers.push(LoRALayer::new(
                        block.self_attn.w_q.clone(),
                        q_dim,
                        hidden_size,
                        rank,
                        alpha,
                    ));
                }
                if target_modules.iter().any(|m| m == "k_proj") {
                    layers.push(LoRALayer::new(
                        block.self_attn.w_k.clone(),
                        kv_hidden_size,
                        hidden_size,
                        rank,
                        alpha,
                    ));
                }
                if target_modules.iter().any(|m| m == "v_proj") {
                    layers.push(LoRALayer::new(
                        block.self_attn.w_v.clone(),
                        kv_hidden_size,
                        hidden_size,
                        rank,
                        alpha,
                    ));
                }
                if target_modules.iter().any(|m| m == "o_proj") {
                    layers.push(LoRALayer::new(
                        block.self_attn.w_o.clone(),
                        hidden_size,
                        q_dim,
                        rank,
                        alpha,
                    ));
                }
                // MLP projections (ENT-LoRA-005)
                if target_modules.iter().any(|m| m == "gate_proj") {
                    layers.push(LoRALayer::new(
                        block.ffn.w_gate.clone(),
                        intermediate,
                        hidden_size,
                        rank,
                        alpha,
                    ));
                }
                if target_modules.iter().any(|m| m == "up_proj") {
                    layers.push(LoRALayer::new(
                        block.ffn.w_up.clone(),
                        intermediate,
                        hidden_size,
                        rank,
                        alpha,
                    ));
                }
                if target_modules.iter().any(|m| m == "down_proj") {
                    layers.push(LoRALayer::new(
                        block.ffn.w_down.clone(),
                        hidden_size,
                        intermediate,
                        rank,
                        alpha,
                    ));
                }
            }

            let lora_param_count: usize =
                layers.iter().map(|l| l.rank() * (l.d_in() + l.d_out())).sum();
            let total_params: usize = model.parameters().iter().map(|p| p.len()).sum();
            println!(
                "  LoRA enabled: rank={rank}, alpha={alpha}, \
                 {lora_param_count} trainable params ({:.2}% of {total_params})",
                100.0 * lora_param_count as f64 / total_params as f64
            );

            Some(layers)
        } else {
            None
        };

        Self {
            model,
            loss_fn,
            optimizer,
            grad_scaler,
            config,
            metrics: MetricsTracker::new(),
            step: 0,
            accumulated_loss: 0.0,
            accumulated_batches: 0,
            lora_layers,
        }
    }

    /// Forward pass on a single batch item
    ///
    /// Returns (loss_value, loss_tensor, logits)
    /// When LoRA is active, routes through `forward_with_lora` so only
    /// LoRA adapter gradients are accumulated.
    pub fn forward_single(&self, input_ids: &[u32], target_ids: &[u32]) -> (f32, Tensor, Tensor) {
        // Forward through transformer (LoRA or full)
        let logits = if let Some(ref lora) = self.lora_layers {
            // ENT-LoRA-001: Use LoRA forward path
            self.model.forward_with_lora(input_ids, lora)
        } else if self.config.checkpoint_config.enabled {
            checkpoint(|_| self.model.forward(input_ids), &Tensor::zeros(1, false))
        } else {
            self.model.forward(input_ids)
        };

        // Compute loss
        let targets = Tensor::from_vec(target_ids.iter().map(|&id| id as f32).collect(), false);
        let loss = self.loss_fn.forward(&logits, &targets);
        let loss_val = loss.data()[0];

        (loss_val, loss, logits)
    }

    /// Compute forward + backward for all items in a batch, returning average loss.
    fn compute_batch_gradients(&self, batch: &LMBatch) -> f32 {
        let mut total_loss = 0.0;

        for i in 0..batch.batch_size {
            let Some(input_ids) = batch.get_input(i) else {
                continue;
            };
            let Some(target_ids) = batch.get_target(i) else {
                continue;
            };

            let (loss_val, loss, _logits) = self.forward_single(input_ids, target_ids);

            if let Some(backward_op) = loss.backward_op() {
                backward_op.backward();
            }

            total_loss += loss_val / self.config.accumulation_steps as f32;
        }

        total_loss / batch.batch_size as f32
    }

    /// Apply gradient clipping and run the optimizer step, then reset accumulation.
    fn clip_and_step(&mut self) {
        if let Some(max_norm) = self.config.base.max_grad_norm {
            let params = if let Some(ref lora) = self.lora_layers {
                lora.iter().flat_map(|l| vec![l.lora_a(), l.lora_b()]).collect::<Vec<_>>()
            } else {
                self.model.parameters()
            };
            let total_norm: f32 = params
                .iter()
                .filter_map(|p| p.grad())
                .map(|g| g.iter().map(|x| x * x).sum::<f32>())
                .sum::<f32>()
                .sqrt();

            if total_norm > max_norm {
                let scale = max_norm / (total_norm + 1e-6);
                let _ = scale;
            }
        }

        // ENT-LoRA-002: Only update trainable params (LoRA A/B + norms when active)
        if let Some(ref mut lora) = self.lora_layers {
            // ENT-LoRA-006: LoRA+ gradient scaling for B matrices
            let ratio = self.config.lora_plus_ratio;
            if ratio != 1.0 {
                for layer in lora.iter_mut() {
                    if let Some(grad) = layer.lora_b_mut().grad() {
                        let scaled = grad.mapv(|g| g * ratio);
                        layer.lora_b_mut().set_grad(scaled);
                    }
                }
            }

            let mut params: Vec<&mut Tensor> =
                lora.iter_mut().flat_map(|l| l.trainable_params()).collect();
            // Also include norm weights (small, critical for adaptation)
            for layer in &mut self.model.layers {
                params.push(&mut layer.input_norm.weight);
                params.push(&mut layer.post_attn_norm.weight);
            }
            params.push(&mut self.model.norm.weight);
            self.optimizer.step_refs(&mut params);
        } else {
            let mut params = self.model.parameters_mut();
            self.optimizer.step_refs(&mut params);
        }

        self.step += 1;
        self.metrics.losses.push(self.accumulated_loss);
        self.metrics.increment_step();

        self.accumulated_loss = 0.0;
        self.accumulated_batches = 0;
    }

    /// Process a batch (forward + backward + optimizer step)
    ///
    /// Returns average loss for the batch
    pub fn train_batch(&mut self, batch: &LMBatch) -> f32 {
        if batch.batch_size == 0 {
            return 0.0;
        }

        if self.accumulated_batches == 0 {
            // ENT-LoRA-002: Zero grad only on trainable params (LoRA A/B + norms)
            if let Some(ref mut lora) = self.lora_layers {
                let mut params: Vec<&mut Tensor> =
                    lora.iter_mut().flat_map(|l| l.trainable_params()).collect();
                for layer in &mut self.model.layers {
                    params.push(&mut layer.input_norm.weight);
                    params.push(&mut layer.post_attn_norm.weight);
                }
                params.push(&mut self.model.norm.weight);
                self.optimizer.zero_grad_refs(&mut params);
            } else {
                let mut params = self.model.parameters_mut();
                self.optimizer.zero_grad_refs(&mut params);
            }
        }

        let avg_loss = self.compute_batch_gradients(batch);

        self.accumulated_loss += avg_loss;
        self.accumulated_batches += 1;

        if self.accumulated_batches >= self.config.accumulation_steps {
            self.clip_and_step();
        }

        avg_loss
    }

    /// Train for one epoch over batches
    pub fn train_epoch(&mut self, batches: &[LMBatch]) -> f32 {
        self.train_epoch_with_callback(batches, |_, _, _| {})
    }

    /// Train for one epoch with a per-step callback.
    ///
    /// The callback receives (batch_index, batch_loss, &self) after each batch.
    /// Use this for progress logging, checkpointing, or early stopping.
    ///
    /// Stops early if `max_steps` is set and the step count reaches it.
    /// Returns `(avg_loss, reached_max_steps)`.
    pub fn train_epoch_with_callback<F>(&mut self, batches: &[LMBatch], mut on_batch: F) -> f32
    where
        F: FnMut(usize, f32, &Self),
    {
        if batches.is_empty() {
            return 0.0;
        }

        let mut total_loss = 0.0;
        let mut batches_processed = 0;

        for (i, batch) in batches.iter().enumerate() {
            // Check max_steps before processing
            if let Some(max) = self.config.max_steps {
                if self.step >= max {
                    break;
                }
            }

            let batch_loss = self.train_batch(batch);
            total_loss += batch_loss;
            batches_processed += 1;
            on_batch(i, batch_loss, self);
        }

        total_loss / batches_processed.max(1) as f32
    }

    /// Returns true if max_steps has been reached.
    pub fn reached_max_steps(&self) -> bool {
        self.config.max_steps.is_some_and(|max| self.step >= max)
    }

    /// Get current step count
    pub fn step(&self) -> usize {
        self.step
    }

    /// Get reference to model
    pub fn model(&self) -> &Transformer {
        &self.model
    }

    /// Get mutable reference to model
    pub fn model_mut(&mut self) -> &mut Transformer {
        &mut self.model
    }

    /// Get current learning rate (with warmup applied)
    pub fn current_lr(&self) -> f32 {
        let base_lr = self.config.lr;

        if self.step < self.config.warmup_steps {
            // Linear warmup
            base_lr * (self.step as f32 / self.config.warmup_steps as f32)
        } else {
            base_lr
        }
    }

    /// Get gradient scaler stats
    pub fn grad_scaler_stats(&self) -> (f32, usize, usize) {
        (
            self.grad_scaler.scale(),
            self.grad_scaler.overflow_count(),
            self.grad_scaler.successful_steps(),
        )
    }

    /// Check if using mixed precision
    pub fn is_mixed_precision(&self) -> bool {
        self.config.precision_config.is_mixed()
    }

    /// Check if using gradient checkpointing
    pub fn is_checkpointing(&self) -> bool {
        self.config.checkpoint_config.enabled
    }

    /// Check if LoRA training is active
    pub fn is_lora(&self) -> bool {
        self.lora_layers.is_some()
    }

    /// Get reference to LoRA layers (for checkpoint saving)
    pub fn lora_layers(&self) -> Option<&[LoRALayer]> {
        self.lora_layers.as_deref()
    }

    /// Get mutable reference to LoRA layers
    pub fn lora_layers_mut(&mut self) -> Option<&mut Vec<LoRALayer>> {
        self.lora_layers.as_mut()
    }

    /// Save LoRA adapter in PEFT-compatible format (ENT-LoRA-003)
    ///
    /// Saves only LoRA A/B weights as `adapter_model.safetensors` + `adapter_config.json`.
    /// Adapter checkpoint is typically <1% of full model size.
    ///
    /// # Arguments
    /// * `output_dir` - Directory to save adapter files
    /// * `base_model_name` - Optional HuggingFace model ID for adapter_config.json
    ///
    /// # Errors
    /// Returns error if not in LoRA mode or I/O fails.
    pub fn save_lora_adapter(
        &self,
        output_dir: impl AsRef<Path>,
        base_model_name: Option<&str>,
    ) -> crate::Result<()> {
        let lora = self.lora_layers.as_ref().ok_or_else(|| {
            crate::error::Error::ConfigError("Cannot save adapter: LoRA not enabled".into())
        })?;

        let rank = self.config.lora_rank.unwrap_or(8);
        let alpha = self.config.lora_alpha.unwrap_or(rank as f32 * 2.0);
        let target_modules = self
            .config
            .lora_target_modules
            .clone()
            .unwrap_or_else(|| vec!["q_proj".to_string(), "v_proj".to_string()]);

        // ENT-LoRA-005: Expand shorthand targets for correct naming
        let expanded = crate::lora::LoRAConfig::expand_shorthand(&target_modules);
        let lora_config = crate::lora::LoRAConfig::new(rank, alpha)
            .target_modules(&expanded.iter().map(String::as_str).collect::<Vec<_>>());

        // ENT-LoRA-007: Build named adapter pairs with correct PEFT naming
        // Layers are ordered per build(): [q, k, v, o, gate, up, down] per block
        let num_layers = self.model.layers.len();

        // Map target module names to their layer path prefix
        let module_paths: Vec<(&str, &str)> = [
            ("q_proj", "self_attn.q_proj"),
            ("k_proj", "self_attn.k_proj"),
            ("v_proj", "self_attn.v_proj"),
            ("o_proj", "self_attn.o_proj"),
            ("gate_proj", "mlp.gate_proj"),
            ("up_proj", "mlp.up_proj"),
            ("down_proj", "mlp.down_proj"),
        ]
        .iter()
        .filter(|(name, _)| expanded.iter().any(|t| t == *name))
        .copied()
        .collect();

        // Generate full path names for each (block, module) pair
        let all_names: Vec<String> = (0..num_layers)
            .flat_map(|i| {
                module_paths.iter().map(move |(_, path)| format!("model.layers.{i}.{path}"))
            })
            .collect();

        let mut adapters: Vec<(&str, &LoRALayer)> = Vec::new();
        for (idx, layer) in lora.iter().enumerate() {
            if idx < all_names.len() {
                adapters.push((&all_names[idx], layer));
            }
        }

        crate::lora::save_adapter_peft(&adapters, &lora_config, base_model_name, output_dir)
            .map_err(|e| crate::error::Error::Io(e.to_string()))
    }

    /// Save model weights to a SafeTensors file
    ///
    /// This persists the trained transformer weights to disk.
    /// Call this after training completes to preserve the learned parameters.
    ///
    /// # Arguments
    ///
    /// * `path` - Output file path (should end in .safetensors)
    /// * `name` - Model name for metadata
    /// * `architecture` - Model architecture description (e.g., "Qwen2ForCausalLM")
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written.
    pub fn save(
        &self,
        path: impl AsRef<Path>,
        name: &str,
        architecture: &str,
    ) -> crate::Result<()> {
        // Use named_parameters() for correct name mapping (handles attention biases etc.)
        let params: Vec<(String, Tensor)> = self
            .model
            .named_parameters()
            .into_iter()
            .map(|(name, tensor)| (name, tensor.clone()))
            .collect();

        let metadata = ModelMetadata::new(name, architecture);
        let model = Model::new(metadata, params);
        let config = SaveConfig::new(ModelFormat::SafeTensors);

        save_model(&model, path, &config)
    }
}
