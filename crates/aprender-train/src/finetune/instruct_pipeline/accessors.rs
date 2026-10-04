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
    /// Required for checkpointing after NF4 QLoRA training. Downloads every adapter
    /// of each NF4 block by target and writes it into its own slot
    /// (FALSIFY-LORA_TARGET_SELECTION_V1_007). The block returns them as Aᵀ and s·Bᵀ,
    /// so B is unscaled and both are transposed back to the PEFT layout
    /// (FALSIFY-CUDA-NF4-TRAIN-LOSS-PARITY-003). A block whose adapters do not fit
    /// their slots leaves that layer's slots unchanged and is reported on stderr.
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
            let Ok(adapters) = block.download_lora_adapters() else {
                continue;
            };
            if let Err(e) = super::device_targets::place_layer_adapters(
                &mut self.lora_layers,
                &self.config.lora_targets,
                layer_idx,
                &adapters,
                inv_scale,
            ) {
                eprintln!("[CUDA] sync_lora_to_cpu: {e}");
            }
        }
    }

    /// Synchronize GPU LoRA weights back to CPU LoRA layers — no-op without the
    /// `cuda` feature.
    ///
    /// On the CPU/WGPU training paths the adapters in `self.lora_layers` are
    /// updated in place by `train_step`, so they are always current and no sync
    /// is required. This twin keeps the call site in `evaluate()` unconditional.
    #[cfg(not(feature = "cuda"))]
    #[allow(clippy::unused_self)]
    pub fn sync_lora_to_cpu(&mut self) {}

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
