//! Small accessor and utility methods on `InstructPipeline`:
//! `tokenize`, `has_tokenizer`, `num_trainable_parameters`, `set_learning_rate`,
//! `learning_rate`, `set_model_path`, `sync_lora_to_cpu`, `is_cuda`, `gpu_name`,
//! `gpu_total_memory`, `summary`, `tokenizer`.

#[allow(clippy::wildcard_imports)]
use super::*;

#[cfg(feature = "cuda")]
use crate::autograd::cuda_training::CudaTrainer;

impl InstructPipeline {
    /// Tokenize text without truncation.
    ///
    /// Returns the full token sequence. Callers (e.g., `train_step`) are
    /// responsible for budget allocation and truncation of the concatenated
    /// prompt+response sequence.
    ///
    /// Falls back to byte-level encoding (each UTF-8 byte as a u32 token ID)
    /// when no BPE tokenizer is loaded.
    pub fn tokenize(&self, text: &str) -> Vec<u32> {
        match self.tokenizer.as_ref() {
            Some(tok) => tok.encode(text),
            None => {
                // Byte-level fallback when no BPE tokenizer is loaded
                text.bytes().map(u32::from).collect()
            }
        }
    }

    /// Returns `true` if a BPE tokenizer is loaded.
    #[must_use]
    pub fn has_tokenizer(&self) -> bool {
        self.tokenizer.is_some()
    }

    /// Number of trainable LoRA parameters.
    #[must_use]
    pub fn num_trainable_parameters(&self) -> usize {
        // LoRA layers store weight + lora_a + lora_b; we count lora_a + lora_b
        self.lora_layers.len()
            * 2
            * self.config.lora_rank
            * (self.lora_layers.first().map_or(0, |_| {
                // Approximate: each LoRA pair has rank * (rows + cols) params
                // This is a rough estimate since layers may differ in size
                1
            }))
    }

    /// Update learning rate (for LR scheduling).
    pub fn set_learning_rate(&mut self, lr: f32) {
        self.optimizer.set_lr(lr);
    }

    /// Get current learning rate.
    #[must_use]
    pub fn learning_rate(&self) -> f32 {
        contract_pre_learning_rate_scaling!();
        self.optimizer.lr()
    }

    /// Set model path for checkpoint provenance.
    pub fn set_model_path(&mut self, path: &Path) {
        self.model_dir = Some(path.to_path_buf());
    }

    /// Synchronize GPU LoRA weights back to CPU LoRA layers (NF4 QLoRA).
    ///
    /// Required for checkpointing after NF4 QLoRA training. Downloads A_q, B_q,
    /// A_v, B_v from each NF4 block and updates the corresponding CPU LoRA layers.
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
                    *self.lora_layers[q_lora_idx].lora_a_mut() = crate::Tensor::from_vec(a_q, true);
                    *self.lora_layers[q_lora_idx].lora_b_mut() =
                        crate::Tensor::from_vec(b_q_unscaled, true);
                }
                if v_lora_idx < self.lora_layers.len() {
                    *self.lora_layers[v_lora_idx].lora_a_mut() = crate::Tensor::from_vec(a_v, true);
                    *self.lora_layers[v_lora_idx].lora_b_mut() =
                        crate::Tensor::from_vec(b_v_unscaled, true);
                }
            }
        }
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
            self.cuda_trainer.as_ref().map(CudaTrainer::device_name)
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
            self.cuda_trainer.as_ref().map(CudaTrainer::total_memory)
        }
        #[cfg(not(feature = "cuda"))]
        {
            None
        }
    }

    /// Summary of pipeline configuration.
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "InstructPipeline: {} LoRA layers, rank={}, alpha={:.1}{}",
            self.lora_layers.len(),
            self.config.lora_rank,
            self.config.lora_alpha,
            if self.config.quantize_nf4 { ", NF4 QLoRA" } else { "" },
        )
    }

    /// Get a reference to the tokenizer, if loaded.
    #[must_use]
    pub fn tokenizer(&self) -> Option<&HfTokenizer> {
        self.tokenizer.as_ref()
    }

    /// PMAT-483: Enable the per-step profiler with the given report interval.
    /// When enabled, profiler measures per-phase and per-layer timing.
    /// Call `profiler.print_report()` or `profiler.print_json_report()` to get results.
    pub fn enable_profiler(&mut self, report_interval: usize) {
        self.profiler = StepProfiler::new(true, report_interval);
    }
}
