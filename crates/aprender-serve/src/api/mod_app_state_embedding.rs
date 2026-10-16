// APR-EMBED-001 EG-3 (#4902): `apr serve` for an EmbeddingGemma 2 file.

/// A resident EmbeddingGemma 2 encoder and the tokenizer read from the same file.
///
/// The two come from one `MappedGGUFModel` and are the ones `apr embed` builds, so
/// the HTTP routes and the CLI hand the model the same tokens and get the same
/// vector (EG-3 `done_when`).
pub struct EmbeddingServed {
    /// The forward: shared blocks, mean pool, projection, L2 normalise.
    pub encoder: crate::gguf::forward_embedding_gemma2::EmbeddingGemma2,
    /// The file's SentencePiece vocabulary, with its BOS/EOS framing.
    pub tokenizer: crate::gguf::embedding_gemma2_input::EmbedTokenizer,
}

impl AppState {
    /// Application state that serves an EmbeddingGemma 2 file (EG-3, #4902).
    ///
    /// There is deliberately no `quantized_model` and no BPE tokenizer: an
    /// embedding model has no `lm_head`, so every generative route is refused by
    /// kind at the router, and the embedding routes answer from [`EmbeddingServed`].
    ///
    /// # Errors
    /// The file is not an EmbeddingGemma 2 model, or its tensors or vocabulary do
    /// not load.
    pub fn with_embedding_gemma2(
        mapped: &crate::gguf::MappedGGUFModel,
    ) -> Result<Self, RealizarError> {
        match crate::model_kind::gguf_model_kind(&mapped.model) {
            Ok(crate::model_kind::ModelKind::Embedding) => {},
            Ok(crate::model_kind::ModelKind::Generative) => {
                return Err(RealizarError::UnsupportedOperation {
                    operation: "with_embedding_gemma2".to_string(),
                    reason: format!(
                        "'{}' is a generative model (ModelKind::Generative); serve it on the \
                         generative path",
                        mapped.model.architecture().unwrap_or_default()
                    ),
                })
            },
            Err(reason) => {
                return Err(RealizarError::UnsupportedOperation {
                    operation: "with_embedding_gemma2".to_string(),
                    reason,
                })
            },
        }
        let encoder = crate::gguf::forward_embedding_gemma2::EmbeddingGemma2::from_mapped(mapped)?;
        let tokenizer =
            crate::gguf::embedding_gemma2_input::EmbedTokenizer::from_gguf(&mapped.model)?;
        let architecture = mapped.model.architecture().map(str::to_string);

        let (audit_logger, audit_sink) = create_audit_state();
        Ok(Self {
            model: None,
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
            cached_architecture: architecture,
            mapped_gguf_model: None,
            moe_no_gpu: true,
            qwen35_session: None,
            embedding_model: Some(Arc::new(EmbeddingServed { encoder, tokenizer })),
            cached_eos_token_id: None,
            cached_serving_context: None,
            verbose: false,
            trace: false,
            model_source: None,
            effective: EffectiveConfigState::new(),
        })
    }

    /// EG-3: the resident EmbeddingGemma 2 encoder, when this state serves one.
    #[must_use]
    pub fn embedding_model(&self) -> Option<Arc<EmbeddingServed>> {
        self.embedding_model.clone()
    }
}
