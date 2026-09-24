//! #4268: the dense GGUF forward (llama, qwen2, qwen3, mistral, ...) behind
//! the one engine, [`Session`](crate::session::Session).
//!
//! Before 0.69.3 each verb drove a dense model through its own loop:
//! `generate_with_cache` on the CPU and `generate_gpu_resident` on CUDA, each
//! with its own prefill, token choice and stop handling. [`DenseForward`] keeps
//! only what differs per backend — advance the KV cache over some tokens and
//! give back the logits (or, for a greedy turn on CUDA, the argmax chosen on the
//! device) — and the session owns the rest.
//!
//! A CUDA failure is the loud fallback the engine requires: the forward prints
//! the reason, records it in the notices, hands the host model to the CPU
//! backend and replays the whole span there. It never returns logits from a
//! state it did not finish.

use std::sync::Arc;

use crate::error::{RealizarError, Result};
use crate::gguf::{OwnedQuantizedKVCache, OwnedQuantizedModel};

/// The prefix of every line that reports a dense CUDA forward giving up for the
/// CPU. A log scan keys on it.
pub const DENSE_GPU_FALLBACK_PREFIX: &str = "[dense] GPU forward failed, falling back to CPU";

/// A dense GGUF model plus its decode state: the one engine over
/// [`DenseForward`].
pub type DenseSession = crate::session::Session<DenseForward>;

/// Where a dense forward runs.
enum Backend {
    /// The CPU forward. The model is shared, so `apr serve` can hold one host
    /// copy for every session it opens.
    Cpu {
        model: Arc<OwnedQuantizedModel>,
        cache: Option<OwnedQuantizedKVCache>,
    },
    /// The CUDA forward. Its KV cache lives on the device; the host cache is
    /// what `forward_gpu_resident` takes alongside it.
    #[cfg(feature = "cuda")]
    Cuda {
        model: Box<crate::gguf::OwnedQuantizedModelCuda>,
        cache: Option<OwnedQuantizedKVCache>,
    },
    /// Only while a fallback moves the model between backends.
    Moving,
}

/// Why a step could not complete: a GPU failure the forward recovers from by
/// moving to the CPU, or anything else, which the session gets.
enum Step {
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    Gpu(String),
    Fatal(RealizarError),
}

impl From<RealizarError> for Step {
    fn from(e: RealizarError) -> Self {
        Self::Fatal(e)
    }
}

/// A dense GGUF model's forward on one backend: the only dense code a verb
/// reaches, and only through a [`DenseSession`].
pub struct DenseForward {
    backend: Backend,
    arch: &'static str,
    context_length: usize,
    /// Positions the KV cache was sized for (0: not yet allocated).
    capacity: usize,
    /// Positions the current turn can reach, what a mid-turn move to the CPU
    /// must allocate.
    turn_positions: usize,
    /// Positions the KV cache holds. A fallback resets it to 0, so the next
    /// forward replays from the start whatever the session believes it holds.
    held: usize,
    notices: Vec<String>,
    batched_prefills: usize,
}

impl DenseForward {
    fn with_backend(backend: Backend) -> Self {
        let mut forward = Self {
            backend,
            arch: "",
            context_length: 1,
            capacity: 0,
            turn_positions: 0,
            held: 0,
            notices: Vec::new(),
            batched_prefills: 0,
        };
        let config = &forward.model().config;
        let (arch, context_length) = (
            crate::tensor_names::normalize_architecture(&config.architecture),
            config.context_length.max(1),
        );
        forward.arch = arch;
        forward.context_length = context_length;
        forward
    }

    /// A forward on the CPU.
    #[must_use]
    pub fn cpu(model: Arc<OwnedQuantizedModel>) -> Self {
        let mut forward = Self::with_backend(Backend::Cpu { model, cache: None });
        forward.notices.push("Backend: CPU".to_string());
        forward
    }

    /// A forward on a CUDA model the caller has already built (and, where it
    /// does, validated with the F2 guard).
    #[cfg(feature = "cuda")]
    #[must_use]
    pub fn cuda(model: crate::gguf::OwnedQuantizedModelCuda) -> Self {
        let line = format!(
            "Backend: GPU ({}, {} MB VRAM)",
            model.device_name(),
            model.vram_mb()
        );
        let mut forward = Self::with_backend(Backend::Cuda {
            model: Box::new(model),
            cache: None,
        });
        forward.notices.push(line);
        forward
    }

    /// The host model, whichever backend holds it.
    #[must_use]
    pub fn model(&self) -> &OwnedQuantizedModel {
        match &self.backend {
            Backend::Cpu { model, .. } => model,
            #[cfg(feature = "cuda")]
            Backend::Cuda { model, .. } => model.model(),
            Backend::Moving => {
                unreachable!("a dense forward is only Moving inside fall_back_to_cpu")
            },
        }
    }

    /// Size the KV cache for the turn: grow it when the turn reaches past it,
    /// never past the declared context.
    fn ensure_capacity(&mut self) -> std::result::Result<(), Step> {
        let positions = self.turn_positions.min(self.context_length).max(1);
        let context_length = self.context_length;
        let capacity = self.capacity;
        let (target, built) = match &mut self.backend {
            Backend::Cpu { model, cache } => {
                if positions <= capacity && cache.is_some() {
                    return Ok(());
                }
                let target = positions
                    .max(capacity.saturating_mul(2))
                    .min(context_length)
                    .max(positions);
                let config = &model.config;
                (
                    target,
                    grow_or_build(cache, target, |n| {
                        OwnedQuantizedKVCache::from_config(config, n)
                    }),
                )
            },
            #[cfg(feature = "cuda")]
            Backend::Cuda { model, cache } => {
                let device_max = model.executor().max_kv_len();
                if device_max > 0 && positions > device_max {
                    return Err(Step::Gpu(format!(
                        "the turn needs {positions} positions and the device KV cache holds {device_max}"
                    )));
                }
                if positions <= capacity && cache.is_some() {
                    return Ok(());
                }
                // The device cache is sized once, at build; only the host side grows.
                let config = &model.model().config;
                (
                    positions,
                    grow_or_build(cache, positions, |n| {
                        OwnedQuantizedKVCache::from_config(config, n)
                    }),
                )
            },
            Backend::Moving => {
                unreachable!("a dense forward is only Moving inside fall_back_to_cpu")
            },
        };
        self.capacity = target;
        if built {
            self.held = 0;
        }
        Ok(())
    }

    /// Move to the CPU forward, printing why, and size its cache for the turn.
    #[cfg(feature = "cuda")]
    fn fall_back_to_cpu(&mut self, reason: &str) -> Result<()> {
        let line = format!("{DENSE_GPU_FALLBACK_PREFIX}: {reason}");
        eprintln!("{line}");
        self.notices.push(line);
        if let Backend::Cuda { model, .. } = std::mem::replace(&mut self.backend, Backend::Moving) {
            self.backend = Backend::Cpu {
                model: Arc::new(model.into_model()),
                cache: None,
            };
        }
        self.capacity = 0;
        self.held = 0;
        match self.ensure_capacity() {
            Ok(()) => Ok(()),
            Err(Step::Fatal(e)) => Err(e),
            Err(Step::Gpu(reason)) => Err(RealizarError::UnsupportedOperation {
                operation: "dense_session".to_string(),
                reason: format!("the CPU backend reported a GPU failure: {reason}"),
            }),
        }
    }

    /// Recover a step: a GPU failure moves to the CPU and asks for a replay
    /// (`Ok(false)`); anything else is the session's.
    fn recover(&mut self, step: Step) -> Result<bool> {
        match step {
            #[cfg(feature = "cuda")]
            Step::Gpu(reason) => {
                self.fall_back_to_cpu(&reason)?;
                Ok(false)
            },
            #[cfg(not(feature = "cuda"))]
            Step::Gpu(reason) => Err(RealizarError::UnsupportedOperation {
                operation: "dense_session".to_string(),
                reason,
            }),
            Step::Fatal(e) => Err(e),
        }
    }

    /// Where a forward over `tokens` from the session's `start` really starts:
    /// never past what the cache holds (a fallback emptied it), and 0 resets it.
    fn resume_at(&mut self, start: usize) -> std::result::Result<usize, Step> {
        let start = start.min(self.held);
        if start == 0 {
            self.reset_cache()?;
        }
        Ok(start)
    }

    fn reset_cache(&mut self) -> std::result::Result<(), Step> {
        match &mut self.backend {
            Backend::Cpu { cache, .. } => {
                if let Some(cache) = cache {
                    cache.reset();
                }
            },
            #[cfg(feature = "cuda")]
            Backend::Cuda { model, cache } => {
                if let Some(cache) = cache {
                    cuda_reset(model, cache);
                }
            },
            Backend::Moving => {
                unreachable!("a dense forward is only Moving inside fall_back_to_cpu")
            },
        }
        self.held = 0;
        Ok(())
    }

    /// Advance the cache from `held` to all of `tokens`; return the logits
    /// after the last one.
    fn try_forward(&mut self, tokens: &[u32], start: usize) -> std::result::Result<Vec<f32>, Step> {
        let start = self.resume_at(start)?;
        let logits = match &mut self.backend {
            Backend::Cpu { model, cache } => {
                let cache = cache.as_mut().ok_or_else(|| never_reserved("CPU"))?;
                let mut logits = Vec::new();
                for (pos, &token) in tokens.iter().enumerate().skip(start) {
                    logits = model.forward_single_with_cache(token, cache, pos)?;
                }
                logits
            },
            #[cfg(feature = "cuda")]
            Backend::Cuda { model, cache } => {
                let cache = cache.as_mut().ok_or_else(|| never_reserved("CUDA"))?;
                cuda_forward(model, cache, tokens, start, &mut self.batched_prefills)
                    .map_err(Step::Gpu)?
            },
            Backend::Moving => {
                unreachable!("a dense forward is only Moving inside fall_back_to_cpu")
            },
        };
        self.held = tokens.len();
        Ok(logits)
    }

    /// The CUDA argmax path; `None` on the CPU, having done nothing.
    #[cfg_attr(
        not(feature = "cuda"),
        allow(clippy::unnecessary_wraps, clippy::unused_self)
    )]
    fn try_forward_greedy(
        &mut self,
        tokens: &[u32],
        start: usize,
    ) -> std::result::Result<Option<u32>, Step> {
        #[cfg(feature = "cuda")]
        if matches!(self.backend, Backend::Cuda { .. }) {
            let start = self.resume_at(start)?;
            let Backend::Cuda { model, cache } = &mut self.backend else {
                unreachable!("matched above");
            };
            let cache = cache.as_mut().ok_or_else(|| never_reserved("CUDA"))?;
            let Some(next) =
                cuda_forward_greedy(model, cache, tokens, start, &mut self.batched_prefills)
                    .map_err(Step::Gpu)?
            else {
                return Ok(None);
            };
            self.held = tokens.len();
            return Ok(Some(next));
        }
        let _ = (tokens, start);
        Ok(None)
    }
}

fn never_reserved(backend: &str) -> Step {
    Step::Fatal(RealizarError::InvalidShape {
        reason: format!("dense session: the {backend} KV cache was never reserved"),
    })
}

#[cfg(feature = "cuda")]
fn gpu_failed(what: &str, at: usize, e: RealizarError) -> String {
    format!("the GPU {what} failed at {at}: {e}")
}

/// Return a dense CUDA model's KV state (device and host) to position 0.
#[cfg(feature = "cuda")]
pub(crate) fn cuda_reset(
    model: &mut crate::gguf::OwnedQuantizedModelCuda,
    cache: &mut OwnedQuantizedKVCache,
) {
    cache.reset();
    model.executor_mut().reset_kv_cache_gpu();
}

/// The dense CUDA forward: the KV state holds `tokens[..start]` (`start == 0`:
/// the caller has reset it with [`cuda_reset`]); advance it over the rest and
/// return the logits after the last token. A whole prompt prefills every
/// position but the last in one batched call (it cannot start mid-sequence),
/// counted in `batched_prefills`, then the last one for its logits — the path
/// `generate_gpu_resident` takes on a sampled request.
///
/// # Errors
/// The reason the GPU failed. What the state holds after an error is unknown.
#[cfg(feature = "cuda")]
pub(crate) fn cuda_forward(
    model: &mut crate::gguf::OwnedQuantizedModelCuda,
    cache: &mut OwnedQuantizedKVCache,
    tokens: &[u32],
    start: usize,
    batched_prefills: &mut usize,
) -> std::result::Result<Vec<f32>, String> {
    let last = tokens.len() - 1;
    let mut from = start;
    if start == 0 && last > 1 {
        model
            .run_prefill(tokens, cache, last, false, false)
            .map_err(|e| gpu_failed("batched prefill", tokens.len(), e))?;
        *batched_prefills += 1;
        from = last;
    }
    let mut logits = Vec::new();
    for (pos, &token) in tokens.iter().enumerate().skip(from) {
        logits = model
            .forward_gpu_resident(token, cache, pos)
            .map_err(|e| gpu_failed("forward", pos, e))?;
    }
    Ok(logits)
}

/// [`cuda_forward`] for a greedy step: the argmax is chosen on the device, so
/// no logits cross to the host. A whole prompt goes through one batched
/// prefill that extracts the first token (PMAT-083). `None`: the prefill ran
/// but chose nothing, and the caller must [`cuda_reset`] and take the logits
/// path.
///
/// # Errors
/// As [`cuda_forward`].
#[cfg(feature = "cuda")]
pub(crate) fn cuda_forward_greedy(
    model: &mut crate::gguf::OwnedQuantizedModelCuda,
    cache: &mut OwnedQuantizedKVCache,
    tokens: &[u32],
    start: usize,
    batched_prefills: &mut usize,
) -> std::result::Result<Option<u32>, String> {
    let last = tokens.len() - 1;
    if start == 0 && last > 0 {
        let first = model
            .run_prefill(tokens, cache, tokens.len(), false, true)
            .map_err(|e| gpu_failed("batched prefill", tokens.len(), e))?;
        *batched_prefills += 1;
        return Ok(first);
    }
    for (pos, &token) in tokens.iter().enumerate().take(last).skip(start) {
        model
            .forward_gpu_resident(token, cache, pos)
            .map_err(|e| gpu_failed("forward", pos, e))?;
    }
    model
        .forward_gpu_resident_to_token_id(tokens[last], cache, last)
        .map(Some)
        .map_err(|e| gpu_failed("forward", last, e))
}

impl crate::session::ArchForward for DenseForward {
    fn arch(&self) -> &'static str {
        self.arch
    }

    fn on_gpu(&self) -> bool {
        match self.backend {
            #[cfg(feature = "cuda")]
            Backend::Cuda { .. } => true,
            _ => false,
        }
    }

    fn context_length(&self) -> usize {
        self.context_length
    }

    fn batched_prefills(&self) -> usize {
        self.batched_prefills
    }

    fn notices(&self) -> &[String] {
        &self.notices
    }

    fn reserve(&mut self, positions: usize) -> Result<bool> {
        self.turn_positions = positions;
        let held_before = self.held;
        #[cfg(feature = "cuda")]
        if let Backend::Cuda { model, .. } = &self.backend {
            if let Err(e) = model.executor().make_current() {
                self.fall_back_to_cpu(&format!(
                    "the CUDA context would not bind to this thread: {e}"
                ))?;
            }
        }
        if let Err(step) = self.ensure_capacity() {
            self.recover(step)?;
        }
        Ok(held_before > 0 && self.held == 0)
    }

    /// A GPU failure moves the forward to the CPU, loudly, and replays all of
    /// `tokens` there.
    fn forward(&mut self, tokens: &[u32], start: usize) -> Result<Vec<f32>> {
        loop {
            match self.try_forward(tokens, start) {
                Ok(logits) => return Ok(logits),
                Err(step) => {
                    self.held = 0;
                    self.recover(step)?;
                },
            }
        }
    }

    fn forward_greedy(&mut self, tokens: &[u32], start: usize) -> Result<Option<u32>> {
        match self.try_forward_greedy(tokens, start) {
            Ok(next) => Ok(next),
            Err(step) => {
                self.held = 0;
                // Now on the CPU: the session replays through `forward`.
                self.recover(step)?;
                Ok(None)
            },
        }
    }
}

/// One dense turn the way the verbs that gave up their own loops need it: the
/// prompt plus the reply WITHOUT the stop token that ended it (the old loops
/// never kept it; the engine does), and whether the GPU served it to the end.
///
/// A prompt longer than the model's context keeps the error the old loops
/// raised, [`RealizarError::ContextLimitExceeded`], so a caller matching on it
/// still sees it.
///
/// # Errors
/// The prompt does not fit the context, or a forward failure no fallback can
/// recover from.
pub fn dense_turn<F: crate::session::ArchForward>(
    session: &mut crate::session::Session<F>,
    prompt: &[u32],
    config: &crate::gguf::QuantizedGenerateConfig,
) -> Result<(Vec<u32>, bool)> {
    dense_stream(session, prompt, config, &mut |_| true)
}

/// [`dense_turn`] that hands each generated token to `on_token` as it is
/// chosen, the way `generate_with_cache_streaming` did: the stop token that
/// ends the turn is never handed on, and `on_token` returning `false` ends
/// the turn.
///
/// # Errors
/// As [`dense_turn`].
pub fn dense_stream<F: crate::session::ArchForward>(
    session: &mut crate::session::Session<F>,
    prompt: &[u32],
    config: &crate::gguf::QuantizedGenerateConfig,
    on_token: &mut dyn FnMut(u32) -> bool,
) -> Result<(Vec<u32>, bool)> {
    let maximum = session.context_length();
    if prompt.len() > maximum {
        return Err(RealizarError::ContextLimitExceeded {
            provided: prompt.len(),
            maximum,
        });
    }
    let turn = session.generate(prompt, config, &mut |t| {
        config.stop_tokens.contains(&t) || on_token(t)
    })?;
    let mut tokens = turn.tokens;
    if tokens.len() > prompt.len()
        && tokens
            .last()
            .is_some_and(|t| config.stop_tokens.contains(t))
    {
        tokens.pop();
    }
    Ok((tokens, turn.used_gpu))
}

/// Grow `cache` in place to `target` (what it holds stays held), or build a new one
/// at `target`. Returns `true` when a new cache was built, so it holds nothing yet.
fn grow_or_build(
    cache: &mut Option<OwnedQuantizedKVCache>,
    target: usize,
    build: impl FnOnce(usize) -> OwnedQuantizedKVCache,
) -> bool {
    if let Some(cache) = cache {
        cache.grow_to(target);
        return false;
    }
    *cache = Some(build(target));
    true
}

#[cfg(test)]
#[path = "dense_session_tests.rs"]
mod tests;
