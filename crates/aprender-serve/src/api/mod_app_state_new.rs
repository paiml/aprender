impl AppState {
    /// Create new application state
    ///
    /// # Arguments
    ///
    /// * `model` - Model for inference
    /// * `tokenizer` - Tokenizer for text processing
    #[must_use]
    pub fn new(model: Model, tokenizer: BPETokenizer) -> Self {
        let (audit_logger, audit_sink) = create_audit_state();
        Self {
            model: Some(Arc::new(model)),
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
            apr_transformer: None,
            cached_architecture: None,
            cached_eos_token_id: None,
            verbose: false,
            trace: false,
        }
    }

    /// Create application state with model registry for multi-model serving
    ///
    /// # Arguments
    ///
    /// * `registry` - Model registry with pre-registered models
    /// * `default_model_id` - Default model to use when not specified
    ///
    /// # Errors
    ///
    /// Returns error if default model doesn't exist in registry
    pub fn with_registry(
        registry: ModelRegistry,
        default_model_id: &str,
    ) -> Result<Self, RealizarError> {
        // Verify default model exists
        if !registry.contains(default_model_id) {
            return Err(RealizarError::ModelNotFound(default_model_id.to_string()));
        }

        let (audit_logger, audit_sink) = create_audit_state();
        Ok(Self {
            model: None,
            tokenizer: None,
            cache: None,
            cache_key: None,
            metrics: Arc::new(MetricsCollector::new()),
            registry: Some(Arc::new(registry)),
            default_model_id: Some(default_model_id.to_string()),
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
            apr_transformer: None,
            cached_architecture: None,
            cached_eos_token_id: None,
            verbose: false,
            trace: false,
        })
    }

    /// Get model and tokenizer by ID (or default)
    #[allow(clippy::type_complexity)]
    fn get_model(
        &self,
        model_id: Option<&str>,
    ) -> Result<(Arc<Model>, Arc<BPETokenizer>), RealizarError> {
        // Multi-model mode
        if let Some(registry) = &self.registry {
            let id = model_id
                .or(self.default_model_id.as_deref())
                .ok_or_else(|| RealizarError::RegistryError("No model ID specified".to_string()))?;
            return registry.get(id);
        }

        // Single model mode
        let model = self
            .model
            .clone()
            .ok_or_else(|| RealizarError::RegistryError("No model available".to_string()))?;
        let tokenizer = self
            .tokenizer
            .clone()
            .ok_or_else(|| RealizarError::RegistryError("No tokenizer available".to_string()))?;

        Ok((model, tokenizer))
    }

    /// Create application state with model caching enabled
    ///
    /// # Arguments
    ///
    /// * `cache_capacity` - Maximum number of models to cache
    ///
    /// # Panics
    ///
    /// Panics if model or tokenizer creation fails (should not happen with valid config)
    #[must_use]
    pub fn with_cache(cache_capacity: usize) -> Self {
        // Create empty state with cache
        let config = ModelConfig {
            vocab_size: 100,
            hidden_dim: 32,
            num_heads: 1,
            num_layers: 1,
            intermediate_dim: 64,
            eps: 1e-5,
        };
        let model = Model::new(config).expect("Failed to create placeholder model");
        let vocab: Vec<String> = (0..100)
            .map(|i| {
                if i == 0 {
                    "<unk>".to_string()
                } else {
                    format!("token{i}")
                }
            })
            .collect();
        let tokenizer =
            BPETokenizer::new(vocab, vec![], "<unk>").expect("Failed to create tokenizer");

        let (audit_logger, audit_sink) = create_audit_state();
        Self {
            model: Some(Arc::new(model)),
            tokenizer: Some(Arc::new(tokenizer)),
            cache: Some(Arc::new(ModelCache::new(cache_capacity))),
            cache_key: Some(CacheKey::new("default".to_string())),
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
            apr_transformer: None,
            cached_architecture: None,
            cached_eos_token_id: None,
            verbose: false,
            trace: false,
        }
    }

    /// Create a demo state with small model for testing
    ///
    /// # Errors
    ///
    /// Returns error if model or tokenizer creation fails
    pub fn demo() -> Result<Self, RealizarError> {
        let config = ModelConfig {
            vocab_size: 100,
            hidden_dim: 32,
            num_heads: 1,
            num_layers: 1,
            intermediate_dim: 64,
            eps: 1e-5,
        };
        let model = Model::new(config)?;

        // Simple demo vocabulary
        let vocab: Vec<String> = (0..100)
            .map(|i| {
                if i == 0 {
                    "<unk>".to_string()
                } else {
                    format!("token{i}")
                }
            })
            .collect();
        let tokenizer = BPETokenizer::new(vocab, vec![], "<unk>")?;

        // Create demo APR model (real inference, not mock)
        // Simple model: sum of inputs with bias
        let apr_model = create_demo_apr_model(4)?; // 4 input features

        let (audit_logger, audit_sink) = create_audit_state();
        Ok(Self {
            model: Some(Arc::new(model)),
            tokenizer: Some(Arc::new(tokenizer)),
            cache: None,
            cache_key: None,
            metrics: Arc::new(MetricsCollector::new()),
            registry: None,
            default_model_id: None,
            apr_model: Some(Arc::new(apr_model)),
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
            apr_transformer: None,
            cached_architecture: None,
            cached_eos_token_id: None,
            verbose: false,
            trace: false,
        })
    }

    /// Create a MOCK demo state for fast HTTP handler testing (no inference)
    ///
    /// This creates an AppState with NO model loaded, so all inference endpoints
    /// return errors immediately. Used for testing HTTP handler code paths
    /// without the ~0.5s overhead of model creation per test.
    ///
    /// # Performance (Dr. Popper's "Tax of Setup" Fix)
    /// - `demo()`: ~0.5s (creates real model)
    /// - `demo_mock()`: ~0.001s (no model, instant errors)
    ///
    /// # When to use
    /// - Use `demo_mock()` for HTTP routing/parsing tests (95% of API tests)
    /// - Use `demo()` only when you need actual inference output
    pub fn demo_mock() -> Result<Self, RealizarError> {
        let (audit_logger, audit_sink) = create_audit_state();
        Ok(Self {
            model: None, // No model = instant "model not loaded" errors
            tokenizer: None,
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
            apr_transformer: None,
            cached_architecture: None,
            cached_eos_token_id: None,
            verbose: false,
            trace: false,
        })
    }

    /// Create application state with a GPU model for GGUF inference (M33: IMP-084)
    ///
    /// # Arguments
    ///
    /// * `gpu_model` - GPU model for inference
    ///
    /// # Errors
    ///
    /// Returns error if tokenizer creation fails
    #[cfg(feature = "gpu")]
    pub fn with_gpu_model(gpu_model: crate::gpu::GpuModel) -> Result<Self, RealizarError> {
        // Create tokenizer with vocab size matching GPU model
        let vocab_size = gpu_model.config().vocab_size;
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
            gpu_model: Some(Arc::new(std::sync::RwLock::new(gpu_model))),
            quantized_model: None,
            cached_model: None,
            dispatch_metrics: None,
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

    /// Create application state with GPU model and real vocabulary (IMP-152)
    ///
    /// This version uses the actual vocabulary from the GGUF file for proper text encoding/decoding.
    ///
    /// # Arguments
    ///
    /// * `gpu_model` - GPU model for inference
    /// * `vocab` - Vocabulary tokens from GGUF metadata (tokenizer.ggml.tokens)
    ///
    /// # Errors
    ///
    /// Returns error if tokenizer creation fails
    #[cfg(feature = "gpu")]
    pub fn with_gpu_model_and_vocab(
        gpu_model: crate::gpu::GpuModel,
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
            gpu_model: Some(Arc::new(std::sync::RwLock::new(gpu_model))),
            quantized_model: None,
            cached_model: None,
            dispatch_metrics: None,
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

    /// Create application state with a quantized model for fused Q4_K inference (IMP-100)
    ///
    /// This is 1.37x faster than dequantized GpuModel due to reduced memory bandwidth.
    ///
    /// # Arguments
    ///
    /// * `quantized_model` - Quantized model for fused Q4_K inference
    ///
    /// # Errors
    ///
    /// Returns error if tokenizer creation fails
    pub fn with_quantized_model(
        quantized_model: crate::gguf::OwnedQuantizedModel,
    ) -> Result<Self, RealizarError> {
        // Create tokenizer with vocab size matching model
        let vocab_size = quantized_model.config.vocab_size;
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
            cached_architecture: None,
            cached_eos_token_id: None,
            verbose: false,
            trace: false,
        })
    }
}
