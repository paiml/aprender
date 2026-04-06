impl AprV2ModelCuda {
    /// Create a new CUDA-accelerated APR model wrapper.
    ///
    /// # Arguments
    ///
    /// * `model` - The APR v2 model to wrap
    /// * `device_ordinal` - GPU device index (0 for first GPU)
    ///
    /// # Errors
    ///
    /// Returns error if CUDA is not available or device doesn't exist.
    pub fn new(model: AprV2Model, device_ordinal: i32) -> Result<Self> {
        Self::with_max_seq_len(model, device_ordinal, 2048)
    }

    /// Create a new CUDA-accelerated APR model wrapper with custom max sequence length.
    ///
    /// # Arguments
    ///
    /// * `model` - The APR v2 model to wrap
    /// * `device_ordinal` - GPU device index (0 for first GPU)
    /// * `max_seq_len` - Maximum sequence length for GPU KV cache
    ///
    /// # Errors
    ///
    /// Returns error if CUDA is not available or device doesn't exist.
    pub fn with_max_seq_len(
        model: AprV2Model,
        device_ordinal: i32,
        max_seq_len: usize,
    ) -> Result<Self> {
        use crate::cuda::{check_vram_sufficient, CudaExecutor, StreamingConfig, StreamingMode};

        // Extract metadata dimensions early for contract gate
        let num_layers = model.metadata.num_layers.unwrap_or(0);
        let num_heads = model.metadata.num_heads.unwrap_or(1);
        let num_kv_heads = model.metadata.num_kv_heads.unwrap_or(num_heads);
        let hidden_dim = model.metadata.hidden_size.unwrap_or(0);
        let vocab_size = model.metadata.vocab_size.unwrap_or(0);
        let intermediate_dim = model.metadata.intermediate_size.unwrap_or(hidden_dim * 4);

        // GH-279: Contract gate — validate architecture and dimensions before CUDA init
        let arch_name = model
            .metadata
            .architecture
            .as_deref()
            // R-01 (Meyer DbC): "unknown" — don't pretend unidentified model is Qwen2.
            .unwrap_or("unknown");
        if num_layers > 0 && hidden_dim > 0 && num_heads > 0 && vocab_size > 0 {
            let _proof = crate::contract_gate::validate_model_load_basic(
                arch_name,
                num_layers,
                hidden_dim,
                num_heads,
                num_kv_heads,
                intermediate_dim,
                vocab_size,
            )
            .map_err(crate::contract_gate::gate_error)?;
        }

        let mut executor =
            CudaExecutor::new(device_ordinal).map_err(|e| RealizarError::UnsupportedOperation {
                operation: "CudaExecutor::new".to_string(),
                reason: format!("CUDA initialization failed: {e}"),
            })?;

        let device_name = executor
            .device_name()
            .unwrap_or_else(|_| "Unknown GPU".to_string());
        let memory_info = executor.memory_info().unwrap_or((0, 0));
        let head_dim = if num_heads > 0 {
            hidden_dim / num_heads
        } else {
            0
        };

        // GH-201: Check VRAM and select streaming mode
        let streaming_config = StreamingConfig {
            hidden_dim,
            num_layers,
            num_heads,
            num_kv_heads,
            vocab_size,
            intermediate_dim,
            max_seq_len,
        };

        let (free_vram, total_vram) = memory_info;
        let streaming_mode = match check_vram_sufficient(free_vram, total_vram, &streaming_config) {
            Ok(StreamingMode::FullCache) => {
                eprintln!(
                    "[AprV2ModelCuda] VRAM sufficient ({} MB free), using full cache mode",
                    free_vram / (1024 * 1024)
                );
                false
            },
            Ok(StreamingMode::LayerStreaming) => {
                eprintln!(
                    "[AprV2ModelCuda] GH-201: Limited VRAM ({} MB free), using layer streaming mode",
                    free_vram / (1024 * 1024)
                );
                true
            },
            Err(e) => {
                return Err(RealizarError::UnsupportedOperation {
                    operation: "VRAM check".to_string(),
                    reason: e,
                });
            },
        };

        if num_layers > 0 && head_dim > 0 {
            executor
                .init_kv_cache_gpu(num_layers, num_heads, num_kv_heads, head_dim, max_seq_len)
                .map_err(|e| RealizarError::UnsupportedOperation {
                    operation: "init_kv_cache_gpu".to_string(),
                    reason: format!("GPU KV cache initialization failed: {e}"),
                })?;
        }

        // Set RoPE theta for position embeddings
        // R-02 (Meyer DbC): rope_theta from metadata, or architecture-specific default.
        let rope_theta = model.metadata.rope_theta.unwrap_or_else(|| {
            let arch = model.metadata.architecture.as_deref().unwrap_or("unknown");
            crate::gguf::default_rope_theta_for_architecture(arch)
        });
        executor.set_rope_theta(rope_theta);

        // CORRECTNESS-011: Set RoPE type (0=NORM adjacent pairs, 2=NEOX split halves)
        // GH-329: Use shared infer_rope_type() as single source of truth
        let rope_type = model.metadata.rope_type.unwrap_or_else(|| {
            let arch = model.metadata.architecture.as_deref()
                .or(model.metadata.model_type.as_deref())
                .unwrap_or("");
            crate::gguf::infer_rope_type(arch)
        });
        let rms_norm_eps = model.metadata.rms_norm_eps.unwrap_or(1e-6);

        // PMAT-114: Trace model configuration for precision debugging
        if std::env::var("APR_TRACE_CONFIG").is_ok() {
            eprintln!(
                "[APR CONFIG] rope_theta={} (raw={:?})",
                rope_theta, model.metadata.rope_theta
            );
            eprintln!(
                "[APR CONFIG] rope_type={} (raw={:?})",
                rope_type, model.metadata.rope_type
            );
            eprintln!(
                "[APR CONFIG] rms_norm_eps={} (raw={:?})",
                rms_norm_eps, model.metadata.rms_norm_eps
            );
        }
        executor.set_rope_type(rope_type);

        let hidden_dim = model.metadata.hidden_size.unwrap_or(0);

        let mut apr_cuda = Self {
            model,
            executor,
            device_name,
            memory_info,
            weight_cache: std::collections::HashMap::new(),
            embedding_cache: None, // Lazy-loaded on first forward
            hidden_dim,
            kv_position: 0,               // Start at position 0
            fallback_kv_used: false,      // PMAT-110: No fallback KV yet
            test_executor: None,          // Phase 45: No test executor by default
            streaming_mode,               // GH-201: Set based on VRAM check
            cached_streaming_layer: None, // GH-201: No layer cached yet
        };

        // GH-201: Choose weight caching strategy based on streaming mode
        if streaming_mode {
            // Layer streaming: only cache LM head and norms, not per-layer weights
            apr_cuda.pre_cache_weights_streaming()?;
        } else {
            // Full cache: pre-cache all transposed weights on GPU for 2x performance
            apr_cuda.pre_cache_weights()?;
        }

        // Pre-cache embedding table for fast token lookup
        apr_cuda.cache_embeddings()?;

        Ok(apr_cuda)
    }

    /// Check if CUDA is available.
    #[must_use]
    pub fn is_available() -> bool {
        crate::cuda::CudaExecutor::is_available()
    }

    /// Get number of CUDA devices.
    #[must_use]
    pub fn num_devices() -> usize {
        crate::cuda::CudaExecutor::num_devices()
    }

    /// Get reference to the inner APR model (PMAT-APR-CUDA-001)
    #[must_use]
    pub fn model(&self) -> &AprV2Model {
        &self.model
    }

    /// Phase 45: Inject a test executor for dependency injection.
    ///
    /// When present, GEMM operations are routed through the test executor
    /// instead of the CUDA executor, enabling testing without actual GPU hardware.
    ///
    /// # Arguments
    ///
    /// * `executor` - Test executor (typically `MockExecutor` or `CpuExecutor`)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use realizar::gpu::executor::{MockExecutor, CpuExecutor};
    ///
    /// let mut cuda_model = AprV2ModelCuda::new(model, 0)?;
    /// cuda_model.with_test_executor(Box::new(CpuExecutor::new()));
    /// ```
    pub fn with_test_executor(
        &mut self,
        executor: Box<dyn crate::gpu::executor::GpuExecutorTrait + Send + Sync>,
    ) {
        self.test_executor = Some(executor);
    }

    /// Check if a test executor is set.
    #[must_use]
    pub fn has_test_executor(&self) -> bool {
        self.test_executor.is_some()
    }

    /// Get GPU device name.
    #[must_use]
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// Get GPU memory info (free, total) in bytes.
    #[must_use]
    pub fn memory_info(&self) -> (usize, usize) {
        self.memory_info
    }

    /// Get VRAM usage in MB.
    #[must_use]
    pub fn vram_mb(&self) -> u64 {
        (self.memory_info.1 / (1024 * 1024)) as u64
    }

    /// Get reference to the inner APR model.
    #[must_use]
    pub fn inner(&self) -> &AprV2Model {
        &self.model
    }

    // ========================================================================
    // BrickProfiler API for per-brick timing
    // ========================================================================

    /// Enable per-brick profiling for real timing measurements.
    pub fn enable_profiling(&mut self) {
        self.executor.enable_profiling();
    }

    /// Disable per-brick profiling.
    pub fn disable_profiling(&mut self) {
        self.executor.disable_profiling();
    }

    /// Check if profiling is enabled.
    #[must_use]
    pub fn is_profiling_enabled(&self) -> bool {
        self.executor.is_profiling_enabled()
    }

    /// Get the brick profiler for reading statistics.
    #[must_use]
    pub fn profiler(&self) -> &trueno::BrickProfiler {
        self.executor.profiler()
    }

    /// Reset profiler statistics.
    pub fn reset_profiler(&mut self) {
        self.executor.reset_profiler();
    }

    /// Reset KV cache position for a new conversation.
    ///
    /// Call this before starting a new generation sequence to clear the
    /// KV cache state from the previous conversation.
    pub fn reset_kv_cache(&mut self) {
        self.kv_position = 0;
        self.fallback_kv_used = false; // PMAT-110: Reset fallback flag
        self.executor.reset_kv_cache_gpu();
    }
}
