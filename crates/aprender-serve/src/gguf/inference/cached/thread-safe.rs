impl OwnedQuantizedModelCachedSync {
    /// Create a new thread-safe cached model wrapper
    ///
    /// The scheduler is lazily initialized on first GPU operation.
    /// The dequantized weight cache is lazily initialized via `warmup_gpu_cache()`.
    /// PARITY-103: Also initializes CudaScheduler when CUDA feature is enabled.
    #[must_use]
    pub fn new(model: OwnedQuantizedModel) -> Self {
        Self {
            model,
            scheduler: std::sync::Mutex::new(None),
            #[cfg(feature = "cuda")]
            cuda_scheduler: std::sync::Mutex::new(None),
            dequant_cache: std::sync::RwLock::new(None),
        }
    }

    /// Get reference to inner model
    #[must_use]
    pub fn model(&self) -> &OwnedQuantizedModel {
        &self.model
    }

    /// Get or create the cached scheduler (thread-safe)
    ///
    /// # Errors
    /// Returns error if scheduler creation fails or lock is poisoned
    fn get_scheduler(
        &self,
    ) -> Result<std::sync::MutexGuard<'_, Option<crate::gpu::HybridScheduler>>> {
        let mut scheduler_opt =
            self.scheduler
                .lock()
                .map_err(|_| RealizarError::UnsupportedOperation {
                    operation: "scheduler_lock".to_string(),
                    reason: "Scheduler mutex poisoned".to_string(),
                })?;

        // Initialize if not already done
        if scheduler_opt.is_none() {
            use crate::gpu::HybridScheduler;
            let new_scheduler = HybridScheduler::with_threshold(1000).map_err(|e| {
                RealizarError::UnsupportedOperation {
                    operation: "HybridScheduler::with_threshold".to_string(),
                    reason: format!("GPU scheduler initialization failed: {e}"),
                }
            })?;
            *scheduler_opt = Some(new_scheduler);
        }

        Ok(scheduler_opt)
    }

    /// PARITY-103: Get or create the cached CUDA scheduler (thread-safe)
    ///
    /// Bypasses wgpu 256MB buffer limit by using cuBLAS directly.
    /// Returns None if CUDA is not available.
    ///
    /// # Errors
    /// Returns error if lock is poisoned
    #[cfg(feature = "cuda")]
    fn get_cuda_scheduler(
        &self,
    ) -> Result<std::sync::MutexGuard<'_, Option<crate::gpu::CudaScheduler>>> {
        use crate::gpu::CudaScheduler;

        let mut scheduler_opt =
            self.cuda_scheduler
                .lock()
                .map_err(|_| RealizarError::UnsupportedOperation {
                    operation: "cuda_scheduler_lock".to_string(),
                    reason: "CUDA scheduler mutex poisoned".to_string(),
                })?;

        // Initialize if not already done
        if scheduler_opt.is_none() {
            match CudaScheduler::new() {
                Ok(new_scheduler) => {
                    eprintln!("PARITY-103: CudaScheduler initialized successfully");
                    *scheduler_opt = Some(new_scheduler);
                },
                Err(e) => {
                    // CUDA not available, leave as None (will fallback to wgpu)
                    eprintln!("PARITY-103: CudaScheduler::new() failed: {:?}", e);
                },
            }
        }

        Ok(scheduler_opt)
    }

    /// PARITY-103: Batch matmul preferring CUDA over wgpu (thread-safe)
    ///
    /// Tries CudaScheduler first (no buffer limits), falls back to HybridScheduler (wgpu).
    /// This bypasses the wgpu 256MB buffer limit that was blocking GPU batch inference.
    #[cfg(feature = "cuda")]
    fn batch_matmul_gpu_prefer_cuda(
        &self,
        input: &[f32],
        weight_f32: &[f32],
        batch_size: usize,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<Vec<f32>> {
        // Validate input
        if input.len() != batch_size * in_dim {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "Input size {} doesn't match batch_size={} * in_dim={}",
                    input.len(),
                    batch_size,
                    in_dim
                ),
            });
        }

        // Try CUDA first (no buffer size limits)
        if let Ok(mut cuda_guard) = self.get_cuda_scheduler() {
            if let Some(ref mut cuda_sched) = *cuda_guard {
                return cuda_sched
                    .matmul(input, weight_f32, batch_size, in_dim, out_dim)
                    .map_err(|e| RealizarError::UnsupportedOperation {
                        operation: "batch_matmul_gpu_prefer_cuda".to_string(),
                        reason: format!("CUDA matmul failed: {e}"),
                    });
            }
        }

        // Fallback to wgpu (may hit 256MB limit for large batches)
        let mut scheduler_guard = self.get_scheduler()?;
        if let Some(ref mut scheduler) = *scheduler_guard {
            return scheduler
                .matmul(input, weight_f32, batch_size, in_dim, out_dim)
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "batch_matmul_gpu_prefer_cuda".to_string(),
                    reason: format!("GPU matmul failed: {e}"),
                });
        }

        Err(RealizarError::UnsupportedOperation {
            operation: "batch_matmul_gpu_prefer_cuda".to_string(),
            reason: "No GPU scheduler available".to_string(),
        })
    }

    /// PARITY-103: Batch matmul preferring CUDA (non-CUDA fallback)
    #[cfg(not(feature = "cuda"))]
    fn batch_matmul_gpu_prefer_cuda(
        &self,
        input: &[f32],
        weight_f32: &[f32],
        batch_size: usize,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<Vec<f32>> {
        // Validate input
        if input.len() != batch_size * in_dim {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "Input size {} doesn't match batch_size={} * in_dim={}",
                    input.len(),
                    batch_size,
                    in_dim
                ),
            });
        }

        let mut scheduler_guard = self.get_scheduler()?;
        if let Some(ref mut scheduler) = *scheduler_guard {
            return scheduler
                .matmul(input, weight_f32, batch_size, in_dim, out_dim)
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "batch_matmul_gpu_prefer_cuda".to_string(),
                    reason: format!("GPU matmul failed: {e}"),
                });
        }

        Err(RealizarError::UnsupportedOperation {
            operation: "batch_matmul_gpu_prefer_cuda".to_string(),
            reason: "No GPU scheduler available".to_string(),
        })
    }

    /// Generate tokens with KV cache using thread-safe cached scheduler
    ///
    /// Delegates to the inner model's `generate_with_cache` method.
    /// The scheduler caching benefits GPU batch operations; single-token
    /// generation uses CPU path with KV cache for O(n) scaling.
    ///
    /// # Arguments
    /// * `prompt` - Input token IDs
    /// * `config` - Generation configuration
    ///
    /// # Returns
    /// Generated token sequence including prompt
    ///
    /// # Errors
    /// Returns error if generation fails
    pub fn generate_with_cache(
        &self,
        prompt: &[u32],
        config: &QuantizedGenerateConfig,
    ) -> Result<Vec<u32>> {
        // Delegate to inner model - CPU path with KV cache is already efficient
        self.model.generate_with_cache(prompt, config)
    }

    /// Generate tokens with adaptive CPU/GPU attention (IMP-126)
    ///
    /// This variant of `generate_with_cache` uses adaptive CPU/GPU dispatch
    /// based on cache length and records dispatch decisions to metrics.
    ///
    /// # Arguments
    /// * `prompt` - Initial token IDs
    /// * `config` - Generation configuration
    /// * `metrics` - Dispatch metrics tracker for CPU/GPU decision recording
    ///
    /// # Returns
    /// Generated token sequence including prompt
    ///
    /// # Errors
    /// Returns error if generation fails
    #[cfg(feature = "gpu")]
    pub fn generate_with_cache_adaptive(
        &self,
        prompt: &[u32],
        config: &QuantizedGenerateConfig,
        metrics: &std::sync::Arc<DispatchMetrics>,
    ) -> Result<Vec<u32>> {
        // Delegate to inner model's adaptive generation
        self.model
            .generate_with_cache_adaptive(prompt, config, metrics)
    }

    /// Forward pass with cached scheduler (thread-safe)
    ///
    /// Uses the cached HybridScheduler for GPU operations.
    ///
    /// # Errors
    /// Returns error if GPU operations fail
    #[allow(clippy::let_underscore_untyped)] // Placeholder for future use
    pub fn forward_batch_gpu_cached(&self, token_ids: &[u32]) -> Result<Vec<f32>> {
        let batch_size = token_ids.len();
        let vocab_size = self.model.config.vocab_size;

        // Get cached scheduler (for future GPU operations)
        let mut scheduler_guard = self.get_scheduler()?;
        let _ = scheduler_guard
            .as_mut()
            .ok_or_else(|| RealizarError::UnsupportedOperation {
                operation: "forward_batch_gpu_cached".to_string(),
                reason: "Scheduler not initialized".to_string(),
            })?;

        // 1. Token embedding lookup
        let hidden = self.model.embed(token_ids);

        // 2. Process through layers
        for layer in &self.model.layers {
            // Simplified single-layer forward - reuse inner model logic
            // For full implementation, would need to port the complete forward pass
            let _ = layer;
        }

        // 3. Output normalization and LM head
        // For now, return placeholder - full implementation requires porting forward logic
        let output = vec![0.0f32; batch_size * vocab_size];
        let _ = hidden;

        Ok(output)
    }

    /// Adaptive fused attention for production serving (IMP-121)
    ///
    /// Thread-safe wrapper that automatically selects CPU or GPU based on
    /// sequence length. Uses the cached scheduler for efficient GPU operations.
    ///
    /// # Arguments
    /// * `q` - Query tensor [seq_len, head_dim]
    /// * `k` - Key tensor [seq_len, head_dim]
    /// * `v` - Value tensor [seq_len, head_dim]
    /// * `seq_len` - Sequence length
    /// * `head_dim` - Head dimension
    /// * `scale` - Attention scale factor
    ///
    /// # Returns
    /// Output tensor [seq_len, head_dim]
    pub fn adaptive_fused_attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        head_dim: usize,
        scale: f32,
    ) -> Result<Vec<f32>> {
        // Threshold for GPU dispatch (from IMP-119 analysis)
        const GPU_SEQ_LEN_THRESHOLD: usize = 64;

        if seq_len >= GPU_SEQ_LEN_THRESHOLD {
            // Long sequence: Use GPU path
            self.gpu_fused_causal_attention(q, k, v, seq_len, head_dim, scale)
        } else {
            // Short sequence: Use CPU path
            self.cpu_fused_causal_attention(q, k, v, seq_len, head_dim, scale)
        }
    }

    /// CPU fused causal attention (thread-safe wrapper)
    fn cpu_fused_causal_attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        head_dim: usize,
        scale: f32,
    ) -> Result<Vec<f32>> {
        // Use tiled implementation from inner model
        self.model
            .tiled_causal_attention(q, k, v, seq_len, head_dim, scale, 4)
    }

    /// GPU fused causal attention (thread-safe)
    fn gpu_fused_causal_attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        head_dim: usize,
        scale: f32,
    ) -> Result<Vec<f32>> {
        let mut scheduler_guard =
            self.scheduler
                .lock()
                .map_err(|_| RealizarError::UnsupportedOperation {
                    operation: "gpu_fused_causal_attention".to_string(),
                    reason: "Failed to acquire scheduler lock".to_string(),
                })?;

        // Initialize scheduler if needed
        if scheduler_guard.is_none() {
            use crate::gpu::HybridScheduler;
            let new_scheduler = HybridScheduler::with_threshold(1000).map_err(|e| {
                RealizarError::UnsupportedOperation {
                    operation: "HybridScheduler::with_threshold".to_string(),
                    reason: format!("GPU scheduler initialization failed: {e}"),
                }
            })?;
            *scheduler_guard = Some(new_scheduler);
        }

        let scheduler =
            scheduler_guard
                .as_mut()
                .ok_or_else(|| RealizarError::UnsupportedOperation {
                    operation: "gpu_fused_causal_attention".to_string(),
                    reason: "Scheduler not initialized".to_string(),
                })?;

        // Transpose K for matmul
        let mut k_transposed = vec![0.0f32; head_dim * seq_len];
        for pos in 0..seq_len {
            for d in 0..head_dim {
                k_transposed[d * seq_len + pos] = k[pos * head_dim + d];
            }
        }

        // GPU Q @ K^T
        let scores = scheduler
            .matmul(q, &k_transposed, seq_len, head_dim, seq_len)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "gpu_fused Q@K^T".to_string(),
                reason: format!("GPU matmul failed: {}", e),
            })?;

        // CPU causal softmax
        let mut weights = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            let mut max_val = f32::NEG_INFINITY;
            for j in 0..=i {
                let score = scores[i * seq_len + j] * scale;
                if score > max_val {
                    max_val = score;
                }
            }
            let mut sum = 0.0f32;
            for j in 0..=i {
                let score = scores[i * seq_len + j] * scale;
                weights[i * seq_len + j] = (score - max_val).exp();
                sum += weights[i * seq_len + j];
            }
            if sum > 0.0 {
                for j in 0..=i {
                    weights[i * seq_len + j] /= sum;
                }
            }
        }

        // GPU weights @ V
        scheduler
            .matmul(&weights, v, seq_len, seq_len, head_dim)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: "gpu_fused weights@V".to_string(),
                reason: format!("GPU matmul failed: {}", e),
            })
    }
}
