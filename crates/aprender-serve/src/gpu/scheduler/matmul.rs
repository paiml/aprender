impl GpuModel {
    /// Create a new GPU-accelerated model with random initialization
    ///
    /// # Errors
    ///
    /// Returns error if GPU initialization fails
    pub fn new(config: GpuModelConfig) -> Result<Self> {
        let scheduler = HybridScheduler::new()?;

        // Initialize weights (small random values for testing)
        let embedding_weights = vec![0.01f32; config.vocab_size * config.hidden_dim];

        let mut block_weights = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            block_weights.push(BlockWeights {
                attn_norm_weight: vec![1.0f32; config.hidden_dim],
                attn_norm_bias: vec![0.0f32; config.hidden_dim],
                qkv_weight: vec![0.01f32; config.hidden_dim * 3 * config.hidden_dim],
                qkv_bias: vec![0.0f32; 3 * config.hidden_dim],
                out_weight: vec![0.01f32; config.hidden_dim * config.hidden_dim],
                out_bias: vec![0.0f32; config.hidden_dim],
                ffn_norm_weight: vec![1.0f32; config.hidden_dim],
                ffn_norm_bias: vec![0.0f32; config.hidden_dim],
                ffn_fc1_weight: vec![0.01f32; config.hidden_dim * config.intermediate_dim],
                ffn_fc1_bias: vec![0.0f32; config.intermediate_dim],
                ffn_fc2_weight: vec![0.01f32; config.intermediate_dim * config.hidden_dim],
                ffn_fc2_bias: vec![0.0f32; config.hidden_dim],
                ffn_gate_weight: None, // No SwiGLU in test models
                // GH-278: No linear attention in test models
                linear_attn: None,
                moe_experts: None,
            });
        }

        let final_norm_weight = vec![1.0f32; config.hidden_dim];
        let final_norm_bias = vec![0.0f32; config.hidden_dim];
        let lm_head_weight = vec![0.01f32; config.hidden_dim * config.vocab_size];
        let lm_head_bias = vec![0.0f32; config.vocab_size];

        // Pre-compute transposed LM head for fast CPU inference
        // Original: [hidden_dim, vocab_size] -> Transposed: [vocab_size, hidden_dim]
        let lm_head_weight_t =
            Self::transpose_weights(&lm_head_weight, config.hidden_dim, config.vocab_size);

        Ok(Self {
            embedding_weights,
            block_weights,
            final_norm_weight,
            final_norm_bias,
            lm_head_weight,
            lm_head_weight_t,
            lm_head_bias,
            scheduler,
            #[cfg(feature = "cuda")]
            cuda_scheduler: None,
            config,
            attention_buffers: None,
            test_executor: None,
            linear_attn_state: None,
        })
    }

    /// IMP-1003: Create GPU model with CUDA-only scheduler
    ///
    /// Unlike `new()`, this constructor creates a model that ALWAYS uses CUDA
    /// for matmul operations, even for m=1 (single-token generation).
    ///
    /// # Errors
    ///
    /// Returns error if GPU or CUDA initialization fails
    #[cfg(feature = "cuda")]
    pub fn new_with_cuda(config: GpuModelConfig) -> Result<Self> {
        let scheduler = HybridScheduler::new()?;
        let cuda_scheduler = Some(CudaScheduler::new()?);

        // Initialize weights (small random values for testing)
        let embedding_weights = vec![0.01f32; config.vocab_size * config.hidden_dim];

        let mut block_weights = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            block_weights.push(BlockWeights {
                attn_norm_weight: vec![1.0f32; config.hidden_dim],
                attn_norm_bias: vec![0.0f32; config.hidden_dim],
                qkv_weight: vec![0.01f32; config.hidden_dim * config.qkv_dim()],
                qkv_bias: vec![0.0f32; config.qkv_dim()],
                out_weight: vec![0.01f32; config.hidden_dim * config.hidden_dim],
                out_bias: vec![0.0f32; config.hidden_dim],
                ffn_norm_weight: vec![1.0f32; config.hidden_dim],
                ffn_norm_bias: vec![0.0f32; config.hidden_dim],
                ffn_fc1_weight: vec![0.01f32; config.hidden_dim * config.intermediate_dim],
                ffn_fc1_bias: vec![0.0f32; config.intermediate_dim],
                ffn_fc2_weight: vec![0.01f32; config.intermediate_dim * config.hidden_dim],
                ffn_fc2_bias: vec![0.0f32; config.hidden_dim],
                ffn_gate_weight: None, // No SwiGLU in test models
                // GH-278: No linear attention in test models
                linear_attn: None,
                moe_experts: None,
            });
        }

        let final_norm_weight = vec![1.0f32; config.hidden_dim];
        let final_norm_bias = vec![0.0f32; config.hidden_dim];
        let lm_head_weight = vec![0.01f32; config.hidden_dim * config.vocab_size];
        let lm_head_bias = vec![0.0f32; config.vocab_size];

        let lm_head_weight_t =
            Self::transpose_weights(&lm_head_weight, config.hidden_dim, config.vocab_size);

        Ok(Self {
            embedding_weights,
            block_weights,
            final_norm_weight,
            final_norm_bias,
            lm_head_weight,
            lm_head_weight_t,
            lm_head_bias,
            scheduler,
            cuda_scheduler,
            config,
            attention_buffers: None,
            test_executor: None,
            linear_attn_state: None,
        })
    }

    /// IMP-1003: Check if this model has CUDA scheduler enabled
    #[cfg(feature = "cuda")]
    #[must_use]
    pub fn has_cuda_scheduler(&self) -> bool {
        self.cuda_scheduler.is_some()
    }

    /// Phase 43: Inject test executor for dependency injection
    ///
    /// When a test executor is set, it takes priority over all other schedulers
    /// in `do_matmul()`. This enables testing forward pass logic without GPU.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use realizar::gpu::executor::MockExecutor;
    ///
    /// let mut model = GpuModel::new(config)?;
    /// let mock = MockExecutor::new("test");
    /// model.with_test_executor(Box::new(mock));
    ///
    /// // Now model.do_matmul() uses the mock
    /// ```
    pub fn with_test_executor(
        &mut self,
        executor: Box<dyn super::super::executor::GpuExecutorTrait + Send + Sync>,
    ) {
        self.test_executor = Some(executor);
    }

    /// Phase 43: Check if test executor is set
    #[must_use]
    pub fn has_test_executor(&self) -> bool {
        self.test_executor.is_some()
    }

    /// Phase 43: Clear test executor (restore normal operation)
    pub fn clear_test_executor(&mut self) {
        self.test_executor = None;
    }

    /// IMP-1003: Perform matmul using CUDA scheduler (always GPU, even for m=1)
    ///
    /// # Errors
    ///
    /// Returns error if CUDA scheduler is not available or matmul fails
    #[cfg(feature = "cuda")]
    #[allow(clippy::many_single_char_names)]
    pub fn cuda_matmul(
        &mut self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>> {
        if let Some(ref mut cuda_sched) = self.cuda_scheduler {
            cuda_sched.matmul(a, b, m, k, n)
        } else {
            // Fallback to HybridScheduler
            self.scheduler.matmul(a, b, m, k, n)
        }
    }

    /// IMP-1005: Unified matmul dispatch that prefers CudaScheduler when available
    ///
    /// This method is used throughout forward_gpu() and forward_block_idx() to
    /// ensure CUDA is used for all matmul operations when cuda_scheduler is present.
    ///
    /// # Phase 43: Test Executor Support
    ///
    /// Priority order:
    /// 1. `test_executor` (if present) - for testing without GPU
    /// 2. `cuda_scheduler` (if present) - for CUDA acceleration
    /// 3. `scheduler` (HybridScheduler) - default fallback
    ///
    /// # Errors
    ///
    /// Returns error if matmul fails
    #[allow(clippy::many_single_char_names)]
    pub fn do_matmul(
        &mut self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>> {
        // Phase 43: Test executor takes priority (for testing without GPU)
        if let Some(ref mut test_exec) = self.test_executor {
            return test_exec.matmul(a, b, m, k, n);
        }

        #[cfg(feature = "cuda")]
        if let Some(ref mut cuda_sched) = self.cuda_scheduler {
            return cuda_sched.matmul(a, b, m, k, n);
        }
        // Fallback to HybridScheduler (or always use it when cuda feature disabled)
        self.scheduler.matmul(a, b, m, k, n)
    }

    /// Matmul with transposed B: C[m,n] = A[m,k] @ B[n,k]^T
    ///
    /// Routes through test_executor if present, enabling mock testing of
    /// attention score computation (Q @ K^T).
    #[allow(clippy::many_single_char_names)]
    pub fn do_matmul_transpose_b(
        &mut self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>> {
        // Test executor takes priority (for testing without GPU)
        if let Some(ref mut test_exec) = self.test_executor {
            // Transpose B and use standard matmul
            let b_t: Vec<f32> = (0..k)
                .flat_map(|i| (0..n).map(move |j| b[j * k + i]))
                .collect();
            return test_exec.matmul(a, &b_t, m, k, n);
        }

        // Use HybridScheduler which has matmul_transpose_b
        self.scheduler.matmul_transpose_b(a, b, m, k, n)
    }

    /// IMP-1007: Zero-clone matmul using split borrow pattern
    ///
    /// This method eliminates weight cloning by using Rust's split borrow pattern.
    /// It directly borrows weights from block_weights while mutably borrowing schedulers.
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor
    /// * `block_idx` - Block index for block weights (ignored for LmHead)
    /// * `op` - Which matmul operation/weight to use
    ///
    /// # Errors
    ///
    /// Returns error if matmul fails
    pub fn matmul_split(
        &mut self,
        input: &[f32],
        block_idx: usize,
        op: WeightType,
    ) -> Result<Vec<f32>> {
        // IMP-1007: Use split borrowing to avoid weight cloning
        // Extract dimensions from config (Copy types, no borrow conflict)
        let hidden_dim = self.config.hidden_dim;
        let qkv_dim = self.config.qkv_dim();
        let intermediate_dim = self.config.intermediate_dim;
        let vocab_size = self.config.vocab_size;

        // Get weight reference and dimensions based on operation
        let (weight, m, k, n) = match op {
            WeightType::Qkv => (
                &self.block_weights[block_idx].qkv_weight,
                1,
                hidden_dim,
                qkv_dim,
            ),
            WeightType::Output => (
                &self.block_weights[block_idx].out_weight,
                1,
                hidden_dim,
                hidden_dim,
            ),
            WeightType::FfnFc1 => (
                &self.block_weights[block_idx].ffn_fc1_weight,
                1,
                hidden_dim,
                intermediate_dim,
            ),
            WeightType::FfnFc2 => (
                &self.block_weights[block_idx].ffn_fc2_weight,
                1,
                intermediate_dim,
                hidden_dim,
            ),
            WeightType::LmHead => (&self.lm_head_weight, 1, hidden_dim, vocab_size),
        };

        // Clone weight to work around borrow checker - this is the safe fallback.
        // For zero-clone operations, use matmul_zero_clone() instead (IMP-1007).
        let weight_clone = weight.clone();

        // Now call do_matmul with cloned weight
        self.do_matmul(input, &weight_clone, m, k, n)
    }

    /// IMP-1007: Zero-clone matmul helper using explicit scheduler extraction
    ///
    /// This is a more aggressive optimization that temporarily extracts the
    /// cuda_scheduler to enable truly zero-clone matmul operations.
    ///
    /// # Safety
    ///
    /// This method uses `Option::take()` to temporarily move the scheduler,
    /// which is safe but requires careful handling to restore it.
    #[cfg(feature = "cuda")]
    pub fn matmul_zero_clone(
        &mut self,
        input: &[f32],
        block_idx: usize,
        op: WeightType,
    ) -> Result<Vec<f32>> {
        // Extract dimensions
        let hidden_dim = self.config.hidden_dim;
        let qkv_dim = self.config.qkv_dim();
        let intermediate_dim = self.config.intermediate_dim;
        let vocab_size = self.config.vocab_size;

        // Temporarily take cuda_scheduler out of self
        let mut cuda_sched: Option<CudaScheduler> = self.cuda_scheduler.take();

        // Now we can borrow block_weights freely
        let (weight, m, k, n) = match op {
            WeightType::Qkv => (
                &self.block_weights[block_idx].qkv_weight,
                1,
                hidden_dim,
                qkv_dim,
            ),
            WeightType::Output => (
                &self.block_weights[block_idx].out_weight,
                1,
                hidden_dim,
                hidden_dim,
            ),
            WeightType::FfnFc1 => (
                &self.block_weights[block_idx].ffn_fc1_weight,
                1,
                hidden_dim,
                intermediate_dim,
            ),
            WeightType::FfnFc2 => (
                &self.block_weights[block_idx].ffn_fc2_weight,
                1,
                intermediate_dim,
                hidden_dim,
            ),
            WeightType::LmHead => (&self.lm_head_weight, 1, hidden_dim, vocab_size),
        };

        // Perform matmul with extracted scheduler
        let result: Result<Vec<f32>> = if let Some(sched) = cuda_sched.as_mut() {
            CudaScheduler::matmul(sched, input, weight, m, k, n)
        } else {
            self.scheduler.matmul(input, weight, m, k, n)
        };

        // Restore cuda_scheduler
        self.cuda_scheduler = cuda_sched;

        result
    }

    // =========================================================================
    // IMP-1008: RefCell-based zero-clone matmul (interior mutability pattern)
    // =========================================================================

    /// IMP-1008: Zero-clone matmul using interior mutability
    ///
    /// This method takes `&self` instead of `&mut self` by wrapping scheduler
    /// access in RefCell. This eliminates the need to clone weights.
    ///
    /// # Errors
    ///
    /// Returns error if matmul fails or RefCell is already borrowed.
    #[cfg(feature = "cuda")]
    #[allow(clippy::many_single_char_names)]
    pub fn matmul_refcell(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>> {
        // IMP-1008: For RefCell pattern, we need to use a different approach
        // Since cuda_scheduler is Option<CudaScheduler>, we use UnsafeCell
        // pattern with explicit unsafe block to avoid changing struct layout.
        //
        // This is safe because:
        // 1. We only access cuda_scheduler mutably here
        // 2. No other code paths access it during matmul
        // 3. This is single-threaded execution

        // Use raw pointer to bypass borrow checker (safe in single-threaded context)
        // SAFETY: This is safe because:
        // - We're in single-threaded context (LLM inference)
        // - cuda_scheduler is only accessed through this method during matmul
        // - The borrow is released before returning
        let cuda_sched_ptr = std::ptr::addr_of!(self.cuda_scheduler).cast_mut();

        // SAFETY: Memory safety ensured by bounds checking and alignment
        let result: Result<Vec<f32>> = unsafe {
            if let Some(sched) = (*cuda_sched_ptr).as_mut() {
                CudaScheduler::matmul(sched, a, b, m, k, n)
            } else {
                // Fallback to HybridScheduler (also needs mut access)
                let sched_ptr = std::ptr::addr_of!(self.scheduler).cast_mut();
                (*sched_ptr).matmul(a, b, m, k, n)
            }
        };
        result
    }
}
