
impl AppState {

    /// Create application state with thread-safe cached model (IMP-116)
    ///
    /// Uses Mutex-based scheduler caching for 10.6x GPU speedup.
    /// This is the recommended production configuration for HTTP serving.
    ///
    /// # Arguments
    ///
    /// * `cached_model` - Thread-safe cached model with scheduler
    ///
    /// # Errors
    ///
    /// Returns error if tokenizer creation fails
    #[cfg(feature = "gpu")]
    pub fn with_cached_model(
        cached_model: crate::gguf::OwnedQuantizedModelCachedSync,
    ) -> Result<Self, RealizarError> {
        // Create tokenizer with vocab size matching model
        let vocab_size = cached_model.model().config.vocab_size;
        let vocab: Vec<String> = (0..vocab_size)
            .map(|i| {
                if i == 0 {
                    "<unk>".to_string()
                } else {
                    format!("token{i}")
                }
            })
            .collect();
        let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>")?;

        let (audit_logger, audit_sink) = create_audit_state();
        Ok(Self {
            model: None,
            tokenizer: Some(Arc::new(tokenizer)),
            cache: None,
            cache_key: None,
            metrics: Arc::new(MetricsCollector::new()),
            registry: None,
            default_model_id: None,
            apr_model: None,
            audit_logger,
            audit_sink,
            gpu_model: None,
            quantized_model: None,
            cached_model: Some(Arc::new(cached_model)),
            // Initialize dispatch metrics for adaptive generation (IMP-126)
            dispatch_metrics: Some(Arc::new(crate::gguf::DispatchMetrics::new())),
            batch_request_tx: None,
            batch_config: None,
            #[cfg(feature = "cuda")]
            cuda_model: None,
            #[cfg(feature = "cuda")]
            safetensors_cuda_model: None,
            #[cfg(feature = "cuda")]
            cuda_batch_tx: None,
            #[cfg(feature = "cuda")]
            apr_q4k_tx: None,
            apr_transformer: None,
            cached_architecture: None,
            cached_eos_token_id: None,
            verbose: false,
            trace: false,
        })
    }

    /// Create application state with thread-safe cached model and real vocabulary (IMP-116)
    ///
    /// Uses Mutex-based scheduler caching for 10.6x GPU speedup with proper token decoding.
    ///
    /// # Arguments
    ///
    /// * `cached_model` - Thread-safe cached model with scheduler
    /// * `vocab` - Vocabulary tokens from GGUF metadata (tokenizer.ggml.tokens)
    ///
    /// # Errors
    ///
    /// Returns error if tokenizer creation fails
    #[cfg(feature = "gpu")]
    pub fn with_cached_model_and_vocab(
        cached_model: crate::gguf::OwnedQuantizedModelCachedSync,
        vocab: Vec<String>,
    ) -> Result<Self, RealizarError> {
        let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>")?;

        let (audit_logger, audit_sink) = create_audit_state();
        Ok(Self {
            model: None,
            tokenizer: Some(Arc::new(tokenizer)),
            cache: None,
            cache_key: None,
            metrics: Arc::new(MetricsCollector::new()),
            registry: None,
            default_model_id: None,
            apr_model: None,
            audit_logger,
            audit_sink,
            gpu_model: None,
            quantized_model: None,
            cached_model: Some(Arc::new(cached_model)),
            dispatch_metrics: Some(Arc::new(crate::gguf::DispatchMetrics::new())),
            batch_request_tx: None,
            batch_config: None,
            #[cfg(feature = "cuda")]
            cuda_model: None,
            #[cfg(feature = "cuda")]
            safetensors_cuda_model: None,
            #[cfg(feature = "cuda")]
            cuda_batch_tx: None,
            #[cfg(feature = "cuda")]
            apr_q4k_tx: None,
            apr_transformer: None,
            cached_architecture: None,
            cached_eos_token_id: None,
            verbose: false,
            trace: false,
        })
    }

    /// Create application state with quantized model and real vocabulary from GGUF
    ///
    /// This version uses the actual vocabulary from the GGUF file for proper decoding.
    ///
    /// # Arguments
    ///
    /// * `quantized_model` - Quantized model for fused Q4_K inference
    /// * `vocab` - Vocabulary tokens from GGUF metadata (tokenizer.ggml.tokens)
    ///
    /// # Errors
    ///
    /// Returns error if tokenizer creation fails
    pub fn with_quantized_model_and_vocab(
        quantized_model: crate::gguf::OwnedQuantizedModel,
        vocab: Vec<String>,
    ) -> Result<Self, RealizarError> {
        let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>")?;

        // PMAT-181: Cache architecture for chat template auto-detection.
        // Qwen3 models get Qwen3NoThinkTemplate (disables thinking mode).
        let arch = Some(quantized_model.config.architecture.clone());

        let (audit_logger, audit_sink) = create_audit_state();
        Ok(Self {
            model: None,
            tokenizer: Some(Arc::new(tokenizer)),
            cache: None,
            cache_key: None,
            metrics: Arc::new(MetricsCollector::new()),
            registry: None,
            default_model_id: None,
            apr_model: None,
            audit_logger,
            audit_sink,
            #[cfg(feature = "gpu")]
            gpu_model: None,
            quantized_model: Some(Arc::new(quantized_model)),
            #[cfg(feature = "gpu")]
            cached_model: None,
            #[cfg(feature = "gpu")]
            dispatch_metrics: None,
            #[cfg(feature = "gpu")]
            batch_request_tx: None,
            #[cfg(feature = "gpu")]
            batch_config: None,
            #[cfg(feature = "cuda")]
            cuda_model: None,
            #[cfg(feature = "cuda")]
            safetensors_cuda_model: None,
            #[cfg(feature = "cuda")]
            cuda_batch_tx: None,
            #[cfg(feature = "cuda")]
            apr_q4k_tx: None,
            apr_transformer: None,
            cached_architecture: arch,
            cached_eos_token_id: None,
            verbose: false,
            trace: false,
        })
    }

    /// Create application state with CUDA-optimized model for high-performance GPU inference (PAR-111)
    ///
    /// This uses the `OwnedQuantizedModelCuda` wrapper which achieves 755+ tok/s (2.6x Ollama) by:
    /// - Pre-uploading all weights to GPU via `preload_weights_gpu()`
    /// - Using batched workspaces for efficient inference
    /// - GPU-resident KV cache to avoid CPU→GPU transfers
    ///
    /// # Arguments
    ///
    /// * `cuda_model` - CUDA-optimized model wrapper (already initialized with GPU resources)
    /// * `vocab` - Vocabulary tokens from GGUF metadata (tokenizer.ggml.tokens)
    ///
    /// # Errors
    ///
    /// Returns error if tokenizer creation fails
    #[cfg(feature = "cuda")]
    pub fn with_cuda_model_and_vocab(
        cuda_model: crate::gguf::OwnedQuantizedModelCuda,
        vocab: Vec<String>,
    ) -> Result<Self, RealizarError> {
        let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>")?;
        // PMAT-073: Cache architecture at construction to avoid RwLock in hot path.
        // model_architecture() was blocking HTTP handlers for ~2s due to read lock
        // contention with the batch scheduler's write lock.
        let arch = Some(cuda_model.model().config.architecture.clone());
        let eos = cuda_model.model().config.eos_token_id;

        let (audit_logger, audit_sink) = create_audit_state();
        Ok(Self {
            model: None,
            tokenizer: Some(Arc::new(tokenizer)),
            cache: None,
            cache_key: None,
            metrics: Arc::new(MetricsCollector::new()),
            registry: None,
            default_model_id: None,
            apr_model: None,
            audit_logger,
            audit_sink,
            #[cfg(feature = "gpu")]
            gpu_model: None,
            quantized_model: None,
            #[cfg(feature = "gpu")]
            cached_model: None,
            #[cfg(feature = "gpu")]
            dispatch_metrics: None,
            #[cfg(feature = "gpu")]
            batch_request_tx: None,
            #[cfg(feature = "gpu")]
            batch_config: None,
            cuda_model: Some(Arc::new(std::sync::RwLock::new(cuda_model))),
            #[cfg(feature = "cuda")]
            safetensors_cuda_model: None,
            cuda_batch_tx: None,
            #[cfg(feature = "cuda")]
            apr_q4k_tx: None,
            apr_transformer: None,
            cached_architecture: arch,
            cached_eos_token_id: eos,
            verbose: false,
            trace: false,
        })
    }

    /// GH-88: Create CUDA model state with proper BPE tokenizer (merge rules + special tokens).
    ///
    /// HuggingFace models (SafeTensors/APR imports) require merge-based BPE encoding.
    /// GGUF models use greedy longest-match and should use `with_cuda_model_and_vocab`.
    #[cfg(feature = "cuda")]
    pub fn with_cuda_model_and_bpe(
        cuda_model: crate::gguf::OwnedQuantizedModelCuda,
        vocab: Vec<String>,
        merges: Vec<(String, String)>,
    ) -> Result<Self, RealizarError> {
        let tokenizer = BPETokenizer::with_merges(vocab, merges, "<unk>")?;
        let arch = Some(cuda_model.model().config.architecture.clone());
        let eos = cuda_model.model().config.eos_token_id;

        let (audit_logger, audit_sink) = create_audit_state();
        Ok(Self {
            model: None,
            tokenizer: Some(Arc::new(tokenizer)),
            cache: None,
            cache_key: None,
            metrics: Arc::new(MetricsCollector::new()),
            registry: None,
            default_model_id: None,
            apr_model: None,
            audit_logger,
            audit_sink,
            #[cfg(feature = "gpu")]
            gpu_model: None,
            quantized_model: None,
            #[cfg(feature = "gpu")]
            cached_model: None,
            #[cfg(feature = "gpu")]
            dispatch_metrics: None,
            #[cfg(feature = "gpu")]
            batch_request_tx: None,
            #[cfg(feature = "gpu")]
            batch_config: None,
            #[cfg(feature = "cuda")]
            cuda_model: Some(Arc::new(std::sync::RwLock::new(cuda_model))),
            #[cfg(feature = "cuda")]
            safetensors_cuda_model: None,
            #[cfg(feature = "cuda")]
            cuda_batch_tx: None,
            #[cfg(feature = "cuda")]
            apr_q4k_tx: None,
            apr_transformer: None,
            cached_architecture: arch,
            cached_eos_token_id: eos,
            verbose: false,
            trace: false,
        })
    }

    /// Create application state with APR Transformer for SafeTensors/APR inference (PMAT-SERVE-FIX-001)
    ///
    /// This enables the `/generate` and `/batch/generate` endpoints for SafeTensors and APR models.
    /// Uses F32 weights for inference, achieving ~1-10 tok/s on CPU.
    ///
    /// # Arguments
    ///
    /// * `transformer` - APR Transformer loaded from SafeTensors or APR file
    /// * `vocab` - Vocabulary tokens for tokenization
    ///
    /// # Errors
    ///
    /// Returns error if tokenizer creation fails
    pub fn with_apr_transformer_and_vocab(
        transformer: crate::apr_transformer::AprTransformer,
        vocab: Vec<String>,
    ) -> Result<Self, RealizarError> {
        let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>")?;

        let (audit_logger, audit_sink) = create_audit_state();
        Ok(Self {
            model: None,
            tokenizer: Some(Arc::new(tokenizer)),
            cache: None,
            cache_key: None,
            metrics: Arc::new(MetricsCollector::new()),
            registry: None,
            default_model_id: None,
            apr_model: None,
            audit_logger,
            audit_sink,
            #[cfg(feature = "gpu")]
            gpu_model: None,
            quantized_model: None,
            #[cfg(feature = "gpu")]
            cached_model: None,
            #[cfg(feature = "gpu")]
            dispatch_metrics: None,
            #[cfg(feature = "gpu")]
            batch_request_tx: None,
            #[cfg(feature = "gpu")]
            batch_config: None,
            #[cfg(feature = "cuda")]
            cuda_model: None,
            #[cfg(feature = "cuda")]
            safetensors_cuda_model: None,
            #[cfg(feature = "cuda")]
            cuda_batch_tx: None,
            #[cfg(feature = "cuda")]
            apr_q4k_tx: None,
            apr_transformer: Some(Arc::new(transformer)),
            cached_architecture: None,
            cached_eos_token_id: None,
            verbose: false,
            trace: false,
        })
    }

    /// Check if this AppState has a quantized model (IMP-100)
    #[must_use]
    pub fn has_quantized_model(&self) -> bool {
        self.quantized_model.is_some()
    }

    /// Get the quantized model for inference (IMP-100)
    pub fn quantized_model(&self) -> Option<&Arc<crate::gguf::OwnedQuantizedModel>> {
        self.quantized_model.as_ref()
    }

    /// Check if this AppState has an APR transformer (PMAT-SERVE-FIX-001)
    #[must_use]
    pub fn has_apr_transformer(&self) -> bool {
        self.apr_transformer.is_some()
    }

    /// Get the APR transformer for inference (PMAT-SERVE-FIX-001)
    pub fn apr_transformer(&self) -> Option<&Arc<crate::apr_transformer::AprTransformer>> {
        self.apr_transformer.as_ref()
    }

    /// Check if this AppState has a GPU model (M33: IMP-084)
    #[cfg(feature = "gpu")]
    #[must_use]
    pub fn has_gpu_model(&self) -> bool {
        self.gpu_model.is_some()
    }

    /// Get the GPU model for inference (M33: IMP-085)
    #[cfg(feature = "gpu")]
    pub fn gpu_model(&self) -> Option<&Arc<std::sync::RwLock<crate::gpu::GpuModel>>> {
        self.gpu_model.as_ref()
    }

    /// Check if this AppState has a cached model (IMP-116)
    #[cfg(feature = "gpu")]
    #[must_use]
    pub fn has_cached_model(&self) -> bool {
        self.cached_model.is_some()
    }

    /// Get the cached model for inference (IMP-116)
    #[cfg(feature = "gpu")]
    pub fn cached_model(&self) -> Option<&Arc<crate::gguf::OwnedQuantizedModelCachedSync>> {
        self.cached_model.as_ref()
    }

    /// Check if this AppState has a CUDA-optimized model (PAR-111)
    #[cfg(feature = "cuda")]
    #[must_use]
    pub fn has_cuda_model(&self) -> bool {
        self.cuda_model.is_some()
    }

    /// Get the CUDA-optimized model for high-performance GPU inference (PAR-111)
    ///
    /// Returns the model wrapper that achieves 755+ tok/s (2.6x Ollama) by using:
    /// - Pre-uploaded GPU weights
    /// - Batched workspaces
    /// - GPU-resident KV cache
    #[cfg(feature = "cuda")]
    pub fn cuda_model(
        &self,
    ) -> Option<&Arc<std::sync::RwLock<crate::gguf::OwnedQuantizedModelCuda>>> {
        self.cuda_model.as_ref()
    }

    /// #169: Get SafeTensors CUDA model for GPU-accelerated F16/F32 inference
    #[cfg(feature = "cuda")]
    #[must_use]
    pub fn safetensors_cuda_model(
        &self,
    ) -> Option<&Arc<std::sync::Mutex<crate::safetensors_cuda::SafeTensorsCudaModel>>> {
        self.safetensors_cuda_model.as_ref()
    }

    /// Get dispatch metrics for adaptive CPU/GPU tracking (IMP-126)
    #[cfg(feature = "gpu")]
    #[must_use]
    pub fn dispatch_metrics(&self) -> Option<&Arc<crate::gguf::DispatchMetrics>> {
        self.dispatch_metrics.as_ref()
    }

    /// Get batch request sender for continuous batching (PARITY-052)
    #[cfg(feature = "gpu")]
    #[must_use]
    pub fn batch_request_tx(&self) -> Option<&tokio::sync::mpsc::Sender<ContinuousBatchRequest>> {
        self.batch_request_tx.as_ref()
    }

    /// Get batch configuration (PARITY-052)
    #[cfg(feature = "gpu")]
    #[must_use]
    pub fn batch_config(&self) -> Option<&BatchConfig> {
        self.batch_config.as_ref()
    }

    /// Check if batch inference is enabled (PARITY-052)
    #[cfg(feature = "gpu")]
    #[must_use]
    pub fn batch_enabled(&self) -> bool {
        self.batch_request_tx.is_some() && self.batch_config.is_some()
    }

    /// Set batch request sender and config (PARITY-052)
    /// This enables continuous batch inference for the completions endpoint
    #[cfg(feature = "gpu")]
    #[must_use]
    pub fn with_batch_config(
        mut self,
        batch_request_tx: tokio::sync::mpsc::Sender<ContinuousBatchRequest>,
        batch_config: BatchConfig,
    ) -> Self {
        self.batch_request_tx = Some(batch_request_tx);
        self.batch_config = Some(batch_config);
        self
    }

    /// Get CUDA batch scheduler sender (PMAT-044)
    #[cfg(feature = "cuda")]
    #[must_use]
    pub fn cuda_batch_tx(
        &self,
    ) -> Option<&tokio::sync::mpsc::Sender<cuda_batch_scheduler::CudaBatchRequest>> {
        self.cuda_batch_tx.as_ref()
    }

    /// Set CUDA batch scheduler sender (PMAT-044)
    /// Enables continuous batching for /v1/chat/completions
    #[cfg(feature = "cuda")]
    #[must_use]
    pub fn with_cuda_batch_tx(
        mut self,
        tx: tokio::sync::mpsc::Sender<cuda_batch_scheduler::CudaBatchRequest>,
    ) -> Self {
        self.cuda_batch_tx = Some(tx);
        self
    }

    /// Create AppState with APR Q4K GPU inference channel and vocabulary (ALB-095)
    ///
    /// The Q4K inference runs on a dedicated thread; this state only holds
    /// the channel sender and tokenizer for HTTP request/response handling.
    #[cfg(feature = "cuda")]
    pub fn with_apr_q4k_and_vocab(
        q4k_tx: tokio::sync::mpsc::Sender<apr_q4k_scheduler::AprQ4kRequest>,
        vocab: Vec<String>,
    ) -> Result<Self, RealizarError> {
        Self::with_apr_q4k_and_vocab_eos(q4k_tx, vocab, None)
    }

    /// ALB-109: Create state for Q4K GPU inference with explicit EOS token ID.
    #[cfg(feature = "cuda")]
    pub fn with_apr_q4k_and_vocab_eos(
        q4k_tx: tokio::sync::mpsc::Sender<apr_q4k_scheduler::AprQ4kRequest>,
        vocab: Vec<String>,
        eos_id: Option<u32>,
    ) -> Result<Self, RealizarError> {
        let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>")?;
        let (audit_logger, audit_sink) = create_audit_state();
        Ok(Self {
            model: None,
            tokenizer: Some(Arc::new(tokenizer)),
            cache: None,
            cache_key: None,
            metrics: Arc::new(MetricsCollector::new()),
            registry: None,
            default_model_id: None,
            apr_model: None,
            audit_logger,
            audit_sink,
            #[cfg(feature = "gpu")]
            gpu_model: None,
            quantized_model: None,
            #[cfg(feature = "gpu")]
            cached_model: None,
            #[cfg(feature = "gpu")]
            dispatch_metrics: None,
            #[cfg(feature = "gpu")]
            batch_request_tx: None,
            #[cfg(feature = "gpu")]
            batch_config: None,
            cuda_model: None,
            #[cfg(feature = "cuda")]
            safetensors_cuda_model: None,
            cuda_batch_tx: None,
            apr_q4k_tx: Some(q4k_tx),
            apr_transformer: None,
            cached_architecture: None,
            cached_eos_token_id: eos_id,
            verbose: false,
            trace: false,
        })
    }

    /// #169: Create state with SafeTensors CUDA model for GPU-accelerated inference
    #[cfg(feature = "cuda")]
    pub fn with_safetensors_cuda_model_and_vocab(
        model: crate::safetensors_cuda::SafeTensorsCudaModel,
        vocab: Vec<String>,
    ) -> Result<Self, RealizarError> {
        let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>")?;
        let metrics = Arc::new(MetricsCollector::new());
        let (audit_logger, audit_sink) = create_audit_state();

        Ok(Self {
            model: None,
            tokenizer: Some(Arc::new(tokenizer)),
            cache: None,
            cache_key: None,
            metrics,
            registry: None,
            default_model_id: None,
            apr_model: None,
            audit_logger,
            audit_sink,
            #[cfg(feature = "gpu")]
            gpu_model: None,
            quantized_model: None,
            #[cfg(feature = "gpu")]
            cached_model: None,
            #[cfg(feature = "gpu")]
            dispatch_metrics: None,
            #[cfg(feature = "gpu")]
            batch_request_tx: None,
            #[cfg(feature = "gpu")]
            batch_config: None,
            cuda_model: None,
            #[cfg(feature = "cuda")]
            safetensors_cuda_model: Some(Arc::new(std::sync::Mutex::new(model))),
            cuda_batch_tx: None,
            apr_q4k_tx: None,
            apr_transformer: None,
            cached_architecture: None,
            cached_eos_token_id: None,
            verbose: false,
            trace: false,
        })
    }

    /// Get APR Q4K inference channel sender (ALB-095)
    #[cfg(feature = "cuda")]
    #[must_use]
    pub fn apr_q4k_tx(
        &self,
    ) -> Option<&tokio::sync::mpsc::Sender<apr_q4k_scheduler::AprQ4kRequest>> {
        self.apr_q4k_tx.as_ref()
    }

    /// Set APR Q4K inference channel sender (ALB-095)
    #[cfg(feature = "cuda")]
    #[must_use]
    pub fn with_apr_q4k_tx(
        mut self,
        tx: tokio::sync::mpsc::Sender<apr_q4k_scheduler::AprQ4kRequest>,
    ) -> Self {
        self.apr_q4k_tx = Some(tx);
        self
    }

    /// GH-319: Get model architecture from whichever backend is loaded.
    ///
    /// Used for chat template auto-detection instead of hardcoding "qwen".
    /// Returns the architecture string (e.g., "qwen2", "llama", "phi2").
    #[must_use]
    pub fn model_architecture(&self) -> Option<String> {
        // PMAT-073: Return cached architecture to avoid RwLock contention.
        // The CUDA model read lock blocks HTTP handlers for ~2s when the batch
        // scheduler holds the write lock during decode steps.
        if let Some(arch) = &self.cached_architecture {
            return Some(arch.clone());
        }

        // Check quantized model first (most common GGUF path)
        if let Some(qm) = &self.quantized_model {
            return Some(qm.config.architecture.clone());
        }

        // Check APR transformer (SafeTensors/APR path)
        if let Some(at) = &self.apr_transformer {
            return Some(at.config.architecture.clone());
        }

        // Check cached model
        #[cfg(feature = "gpu")]
        if let Some(cm) = &self.cached_model {
            return Some(cm.model().config.architecture.clone());
        }

        // GpuModel doesn't carry architecture info
        None
    }

    /// GH-330: Get EOS token ID from whichever model backend is loaded.
    ///
    /// **Design by Contract (Meyer 1992)**: The model config carries its own EOS
    /// token as a class invariant. Callers must NOT fall back to hardcoded values.
    #[must_use]
    pub fn model_eos_token_id(&self) -> Option<u32> {
        // PMAT-073: Return cached EOS to avoid RwLock contention
        if self.cached_eos_token_id.is_some() {
            return self.cached_eos_token_id;
        }
        if let Some(qm) = &self.quantized_model {
            return qm.config.eos_token_id;
        }
        if let Some(at) = &self.apr_transformer {
            return at.config.eos_token_id;
        }
        #[cfg(feature = "gpu")]
        if let Some(cm) = &self.cached_model {
            return cm.model().config.eos_token_id;
        }
        None
    }

    /// ALB-109: Get EOS token IDs for Q4K generation.
    /// Returns a list of token IDs that should stop generation.
    pub fn model_eos_ids(&self) -> Vec<u32> {
        if let Some(eos) = self.model_eos_token_id() {
            vec![eos]
        } else {
            // Fallback: common EOS tokens (0=padding, 2=</s> for LLaMA-style)
            vec![0, 2]
        }
    }

    /// GH-152: Enable verbose request/response logging
    #[must_use]
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// GH-152: Check if verbose logging is enabled
    #[must_use]
    pub fn is_verbose(&self) -> bool {
        self.verbose
    }

    /// GH-103: Enable inference tracing (propagates into QuantizedGenerateConfig.trace)
    #[must_use]
    pub fn with_trace(mut self, trace: bool) -> Self {
        self.trace = trace;
        self
    }

    /// GH-103: Check if inference tracing is enabled (server-wide default)
    #[must_use]
    pub fn is_trace_enabled(&self) -> bool {
        self.trace
    }

    /// GH-103: Resolve trace flag from server-wide default OR per-request X-Trace-Level header.
    /// Returns true if either the server was started with --trace or the request has a trace header.
    #[must_use]
    pub fn should_trace(&self, trace_level: Option<&str>) -> bool {
        self.trace || trace_level.is_some()
    }
}

include!("mod_app_state_new.rs");
