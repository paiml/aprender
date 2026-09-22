// #3571: `apr serve` for the Qwen3.5 hybrid (Gated `DeltaNet` + gated attention).

/// A resident Qwen3.5 session, and the one fact a request needs before it may
/// queue for it (#3571).
pub struct Qwen35Served {
    /// The GGUF's declared context length, readable while a generation holds
    /// the session — a request that cannot fit is refused without waiting.
    pub context_length: usize,
    /// One request at a time: the hybrid is a single-stream model.
    pub session: std::sync::Mutex<crate::gguf::qwen35_session::Qwen35Session>,
    /// Whether the session serves from the GPU — readable by `/health` while a
    /// generation holds the session, and refreshed after every generation, so a
    /// mid-run fallback to the CPU is reported, not remembered wrong.
    pub on_gpu: std::sync::atomic::AtomicBool,
}

impl AppState {
    /// Application state that serves a Qwen3.5 hybrid from a resident
    /// [`Qwen35Session`](crate::gguf::qwen35_session::Qwen35Session) (#3571).
    ///
    /// There is deliberately NO `quantized_model`. The hybrid's base is the
    /// embeddings, the final norm and `lm_head` with zero layers; every dense
    /// backend that found it would decode through it — measured on #3571 at
    /// batch-1: HTTP 200 with 1024 tokens of `"\n"` on the CPU route, HTTP 500
    /// on the GPU one. With no dense model those endpoints give their own "no
    /// model" answer, and `/v1/chat/completions` serves from the session.
    ///
    /// `vocab` builds the tokenizer the response path decodes with. Prompts are
    /// encoded with the GGUF's own tokenizer, the one `apr run` and `apr chat`
    /// use, so the three verbs hand the model the same tokens.
    ///
    /// # Errors
    /// The vocabulary does not make a tokenizer.
    pub fn with_qwen35_session(
        session: crate::gguf::qwen35_session::Qwen35Session,
        mapped: Arc<crate::gguf::MappedGGUFModel>,
        vocab: Vec<String>,
    ) -> Result<Self, RealizarError> {
        let unk = crate::tokenizer::vocabulary_unk_token(&vocab); // #3609: never a literal
        let tokenizer = BPETokenizer::new(vocab, vec![], unk)?;
        let architecture = mapped.model.architecture().map(str::to_string);
        let eos_token_id = mapped.model.eos_token_id();

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
            apr_transformer: None,
            cached_architecture: architecture,
            mapped_gguf_model: Some(mapped),
            qwen35_session: Some(Arc::new(Qwen35Served {
                context_length: session.context_length(),
                on_gpu: std::sync::atomic::AtomicBool::new(session.on_gpu()),
                session: std::sync::Mutex::new(session),
            })),
            cached_eos_token_id: eos_token_id,
            verbose: false,
            trace: false,
            model_source: None,
            effective: EffectiveConfigState::new(),
        })
    }

    /// #3571: the resident Qwen3.5 session, when this state serves one.
    #[must_use]
    pub fn qwen35_session(&self) -> Option<Arc<Qwen35Served>> {
        self.qwen35_session.clone()
    }
}
