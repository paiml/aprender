// #3571: `apr serve` for the Qwen3.5 hybrid (Gated `DeltaNet` + gated attention).

/// A resident Qwen3.5 session, and the one fact a request needs before it may
/// queue for it (#3571).
pub struct Qwen35Served {
    /// The GGUF's declared context length, readable while a generation holds
    /// the session — a request that cannot fit is refused without waiting.
    pub context_length: usize,
    /// The resident sessions, one per request in flight (#4234). With one slot
    /// (the default) requests take turns, as before; with more, their decode
    /// steps run together on the one shared device model.
    pub session: Qwen35Slots,
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
            moe_no_gpu: true,
            qwen35_session: Some(Arc::new(Qwen35Served {
                context_length: session.context_length(),
                on_gpu: std::sync::atomic::AtomicBool::new(session.on_gpu()),
                session: Qwen35Slots::new(session, qwen35_serve_slots()),
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

/// #4234: how many Qwen3.5 requests `apr serve` runs at once. Unset: 1 — one
/// request at a time, the 0.69.x behaviour.
pub const QWEN35_SERVE_SLOTS_ENV: &str = "APR_QWEN35_SERVE_SLOTS";

/// The largest slot count [`QWEN35_SERVE_SLOTS_ENV`] may ask for.
pub const QWEN35_SERVE_SLOTS_MAX: usize = 64;

/// [`QWEN35_SERVE_SLOTS_ENV`], read once at startup. A value that is not a count
/// in `1..=QWEN35_SERVE_SLOTS_MAX` is said loudly and replaced by 1, never
/// guessed at.
#[must_use]
pub fn qwen35_serve_slots() -> usize {
    match std::env::var(QWEN35_SERVE_SLOTS_ENV) {
        Err(_) => 1,
        Ok(raw) => match raw.trim().parse::<usize>() {
            Ok(n) if (1..=QWEN35_SERVE_SLOTS_MAX).contains(&n) => n,
            _ => {
                eprintln!(
                    "[qwen35] {QWEN35_SERVE_SLOTS_ENV}={raw:?} is not a slot count in \
                     1..={QWEN35_SERVE_SLOTS_MAX}; serving one request at a time"
                );
                1
            },
        },
    }
}

/// #4234: the resident Qwen3.5 sessions of one server — siblings over the same
/// host and device models ([`Qwen35Session::sibling`]), each holding its own
/// conversation. [`Self::lock`] waits for a free one.
///
/// [`Qwen35Session::sibling`]: crate::gguf::qwen35_session::Qwen35Session::sibling
pub struct Qwen35Slots {
    slots: Vec<std::sync::Mutex<crate::gguf::qwen35_session::Qwen35Session>>,
    /// Free slot indices; the most recently released is taken first, so a lone
    /// client's next turn lands on the session that holds its conversation.
    free: std::sync::Mutex<Vec<usize>>,
    released: std::sync::Condvar,
}

impl Qwen35Slots {
    /// `first` and `count - 1` siblings of it (at least one slot).
    #[must_use]
    pub fn new(first: crate::gguf::qwen35_session::Qwen35Session, count: usize) -> Self {
        let mut slots = Vec::with_capacity(count.max(1));
        for _ in 1..count.max(1) {
            slots.push(std::sync::Mutex::new(first.sibling()));
        }
        slots.insert(0, std::sync::Mutex::new(first));
        // Slot 0 on top: a lone client always gets the same session.
        let free = (0..slots.len()).rev().collect();
        Self {
            slots,
            free: std::sync::Mutex::new(free),
            released: std::sync::Condvar::new(),
        }
    }

    /// The number of sessions — requests that can be in flight at once.
    #[must_use]
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// Always false: there is at least one slot.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Wait for a free session and hold it until the guard drops.
    ///
    /// # Errors
    /// The session was poisoned by a panic mid-turn — the same answer a plain
    /// `Mutex` gives, with the guard inside it.
    pub fn lock(&self) -> std::sync::LockResult<Qwen35Slot<'_>> {
        let index = {
            let mut free = self
                .free
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            loop {
                if let Some(i) = free.pop() {
                    break i;
                }
                free = self
                    .released
                    .wait(free)
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
            }
        };
        match self.slots[index].lock() {
            Ok(guard) => Ok(Qwen35Slot {
                pool: self,
                index,
                guard: Some(guard),
            }),
            Err(poisoned) => Err(std::sync::PoisonError::new(Qwen35Slot {
                pool: self,
                index,
                guard: Some(poisoned.into_inner()),
            })),
        }
    }
}

/// A session held for one request; releasing it frees the slot.
pub struct Qwen35Slot<'a> {
    pool: &'a Qwen35Slots,
    index: usize,
    guard: Option<std::sync::MutexGuard<'a, crate::gguf::qwen35_session::Qwen35Session>>,
}

impl std::ops::Deref for Qwen35Slot<'_> {
    type Target = crate::gguf::qwen35_session::Qwen35Session;
    fn deref(&self) -> &Self::Target {
        self.guard.as_ref().expect("held until drop")
    }
}

impl std::ops::DerefMut for Qwen35Slot<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.guard.as_mut().expect("held until drop")
    }
}

impl Drop for Qwen35Slot<'_> {
    fn drop(&mut self) {
        // The session first, then the index: a waiter woken by the index must
        // find the session free.
        drop(self.guard.take());
        self.pool
            .free
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(self.index);
        self.pool.released.notify_one();
    }
}
