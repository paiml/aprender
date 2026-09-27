//! PMAT-4269 (M2b): the APR Q4K forward behind the one engine,
//! [`Session`](crate::session::Session).
//!
//! Before this, the Q4K scheduler (`apr_q4k_scheduler::generate_q4k`) carried its
//! own prefill, first-token choice, decode loop, stop check and cancellation poll.
//! [`AprQ4kForward`] keeps only what is Q4K-specific: advance the per-layer KV
//! cache one token at a time and give back the logits. The session owns token
//! choice, stop tokens, cancellation and the context budget.
//!
//! The per-token step is a [`Q4kStep`], so the SAME `ArchForward` impl is what
//! ships (over [`CudaQ4kStep`]) and what the cancellation falsifiers drive (over
//! a counting step, with no GPU): FALSIFY-SERVE-CANCEL-009/010 assert on the
//! shipped control flow, the session's loop plus this forward.
//!
//! The session is built inside the scheduler's dedicated CUDA thread and never
//! leaves it, which is why `ArchForward` has no `Send` supertrait: the forward
//! borrows the thread's `!Send` `CudaExecutor`.

use crate::error::{RealizarError, Result};

/// A Q4K model plus its decode state: the one engine over [`AprQ4kForward`].
pub type AprQ4kSession<S> = crate::session::Session<AprQ4kForward<S>>;

/// One token of the Q4K forward: the state below the session's bookkeeping.
pub trait Q4kStep {
    /// Drop every position the KV cache holds.
    fn reset(&mut self);

    /// Advance the KV cache by `token` at `position` and return the logits
    /// after it.
    ///
    /// # Errors
    /// The forward failed.
    fn step(&mut self, token: u32, position: usize) -> Result<Vec<f32>>;
}

/// An APR Q4K model's forward: the only Q4K code a verb reaches, and only
/// through an [`AprQ4kSession`].
pub struct AprQ4kForward<S: Q4kStep> {
    step: S,
    on_gpu: bool,
    context_length: usize,
    /// Positions the KV cache holds.
    held: usize,
    notices: Vec<String>,
}

impl<S: Q4kStep> AprQ4kForward<S> {
    /// Wrap a step. `context_length` is the model's declared context
    /// (`max_position_embeddings`); `None` means the model declares none, and
    /// the session then caps nothing, as the scheduler never did.
    pub fn new(step: S, on_gpu: bool, context_length: Option<usize>) -> Self {
        let backend = if on_gpu { "GPU" } else { "CPU" };
        Self {
            step,
            on_gpu,
            context_length: context_length.unwrap_or(usize::MAX).max(1),
            held: 0,
            notices: vec![format!("Backend: {backend}")],
        }
    }
}

impl<S: Q4kStep> crate::session::ArchForward for AprQ4kForward<S> {
    fn arch(&self) -> &'static str {
        "apr-q4k"
    }

    fn on_gpu(&self) -> bool {
        self.on_gpu
    }

    fn context_length(&self) -> usize {
        self.context_length
    }

    fn batched_prefills(&self) -> usize {
        0
    }

    fn notices(&self) -> &[String] {
        &self.notices
    }

    /// The KV cache grows per token, so there is nothing to size.
    fn reserve(&mut self, _positions: usize) -> Result<bool> {
        Ok(false)
    }

    fn forward(&mut self, tokens: &[u32], start: usize) -> Result<Vec<f32>> {
        // The session resumes only at exactly what the cache holds; anything
        // else replays the whole span from a reset cache.
        let start = if start == self.held && start > 0 {
            start
        } else {
            self.step.reset();
            self.held = 0;
            0
        };
        let mut logits = Vec::new();
        for (position, &token) in tokens.iter().enumerate().skip(start) {
            // A failed step leaves the cache at an unknown length: the next
            // call must replay from 0.
            self.held = 0;
            logits = self.step.step(token, position)?;
            self.held = position + 1;
        }
        if logits.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "apr-q4k session: forward over {} tokens from {start} produced no logits",
                    tokens.len()
                ),
            });
        }
        Ok(logits)
    }
}

/// The shipped [`Q4kStep`]: `forward_token_apr_q4k` on the scheduler thread's
/// executor, over the uploaded weights, with a fresh host KV cache.
#[cfg(feature = "cuda")]
pub(crate) struct CudaQ4kStep<'a> {
    executor: &'a mut crate::cuda::CudaExecutor,
    config: &'a crate::gpu::adapters::apr_q4k::AprQ4KConfig,
    weights: &'a crate::api::apr_q4k_scheduler::Q4kHostWeights,
    kv_k: Vec<Vec<f32>>,
    kv_v: Vec<Vec<f32>>,
}

#[cfg(feature = "cuda")]
impl<'a> CudaQ4kStep<'a> {
    pub(crate) fn new(
        executor: &'a mut crate::cuda::CudaExecutor,
        config: &'a crate::gpu::adapters::apr_q4k::AprQ4KConfig,
        weights: &'a crate::api::apr_q4k_scheduler::Q4kHostWeights,
    ) -> Self {
        Self {
            executor,
            config,
            weights,
            kv_k: vec![Vec::new(); config.num_layers],
            kv_v: vec![Vec::new(); config.num_layers],
        }
    }
}

#[cfg(feature = "cuda")]
impl Q4kStep for CudaQ4kStep<'_> {
    fn reset(&mut self) {
        self.kv_k.iter_mut().for_each(Vec::clear);
        self.kv_v.iter_mut().for_each(Vec::clear);
    }

    fn step(&mut self, token: u32, position: usize) -> Result<Vec<f32>> {
        crate::gpu::adapters::apr_q4k::forward_token_apr_q4k(
            self.executor,
            self.config,
            &self.weights.embedding,
            &self.weights.output_norm,
            &self.weights.layer_norms,
            &self.weights.qkv_biases,
            &mut self.kv_k,
            &mut self.kv_v,
            token,
            position,
        )
    }
}
