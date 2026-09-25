//! #4280: the dense CUDA forward over a model the caller only BORROWS.
//!
//! `apr serve`'s CUDA scheduler shares its model through an
//! `Arc<RwLock<OwnedQuantizedModelCuda>>` (health, admission sizing and the
//! batched path read it too), so it cannot hand the model to
//! [`DenseForward::cuda`](crate::gguf::dense_session::DenseForward::cuda), which takes it by value.
//! [`BorrowedCudaForward`] runs the same forward ([`cuda_forward`],
//! [`cuda_forward_greedy`]) over a `&mut` for the length of one turn, so
//! the scheduler's single-request path goes through the one engine.
//!
//! It has no CPU to fall back to — the host model stays in the lock — so a GPU
//! failure is returned, loudly, as the pre-port `generate_gpu_resident_streaming`
//! returned it.

use crate::error::{RealizarError, Result};
use crate::gguf::{OwnedQuantizedKVCache, OwnedQuantizedModelCuda};

use crate::gguf::dense_session::{cuda_forward, cuda_forward_greedy, cuda_reset};

/// A dense CUDA forward over a borrowed model, for one scheduler turn.
pub struct BorrowedCudaForward<'a> {
    model: &'a mut OwnedQuantizedModelCuda,
    cache: Option<OwnedQuantizedKVCache>,
    arch: &'static str,
    context_length: usize,
    /// Positions the KV cache was sized for (0: not yet allocated).
    capacity: usize,
    /// Positions the KV cache holds.
    held: usize,
    notices: Vec<String>,
    batched_prefills: usize,
}

impl<'a> BorrowedCudaForward<'a> {
    /// A forward over `model` until it is dropped.
    #[must_use]
    pub fn new(model: &'a mut OwnedQuantizedModelCuda) -> Self {
        let config = &model.model().config;
        let arch = crate::tensor_names::normalize_architecture(&config.architecture);
        let context_length = config.context_length.max(1);
        let line = format!(
            "Backend: GPU ({}, {} MB VRAM)",
            model.device_name(),
            model.vram_mb()
        );
        Self {
            model,
            cache: None,
            capacity: 0,
            arch,
            context_length,
            held: 0,
            notices: vec![line],
            batched_prefills: 0,
        }
    }

    /// One forward from `start` (reset to position 0 when `start` is 0 or past
    /// what the cache holds). The model and the cache are borrowed apart, so
    /// the shared forward takes both.
    fn run<T>(
        &mut self,
        tokens: &[u32],
        start: usize,
        step: impl FnOnce(
            &mut OwnedQuantizedModelCuda,
            &mut OwnedQuantizedKVCache,
            &[u32],
            usize,
            &mut usize,
        ) -> std::result::Result<T, String>,
    ) -> Result<T> {
        let start = start.min(self.held);
        let cache = self
            .cache
            .as_mut()
            .ok_or_else(|| RealizarError::InvalidShape {
                reason: "dense session: the CUDA KV cache was never reserved".to_string(),
            })?;
        if start == 0 {
            cuda_reset(self.model, cache);
        }
        self.held = 0;
        step(self.model, cache, tokens, start, &mut self.batched_prefills)
            .map_err(|reason| loud(&reason))
    }
}

/// A GPU failure, printed and returned: serve has no CPU copy to move to.
fn loud(reason: &str) -> RealizarError {
    let line = format!("[dense] GPU forward failed (serve, no CPU fallback): {reason}");
    eprintln!("{line}");
    RealizarError::UnsupportedOperation {
        operation: "dense_session".to_string(),
        reason: line,
    }
}

impl crate::session::ArchForward for BorrowedCudaForward<'_> {
    fn arch(&self) -> &'static str {
        self.arch
    }

    fn on_gpu(&self) -> bool {
        true
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
        let positions = positions.min(self.context_length).max(1);
        if !self.model.supports_gpu_resident() {
            return Err(RealizarError::UnsupportedOperation {
                operation: "generate_gpu_resident_streaming".to_string(),
                reason: "Model architecture not supported for GPU-resident path".to_string(),
            });
        }
        self.model.executor().make_current().map_err(|e| {
            loud(&format!(
                "the CUDA context would not bind to this thread: {e}"
            ))
        })?;
        let device_max = self.model.executor().max_kv_len();
        if device_max > 0 && positions > device_max {
            return Err(loud(&format!(
                "the turn needs {positions} positions and the device KV cache holds {device_max}"
            )));
        }
        let held_before = self.held;
        if self.cache.is_none() || positions > self.capacity {
            self.cache = Some(OwnedQuantizedKVCache::from_config(
                &self.model.model().config,
                positions,
            ));
            self.capacity = positions;
            self.held = 0;
        }
        Ok(held_before > 0 && self.held == 0)
    }

    fn forward(&mut self, tokens: &[u32], start: usize) -> Result<Vec<f32>> {
        let logits = self.run(tokens, start, cuda_forward)?;
        self.held = tokens.len();
        Ok(logits)
    }

    fn forward_greedy(&mut self, tokens: &[u32], start: usize) -> Result<Option<u32>> {
        let next = self.run(tokens, start, cuda_forward_greedy)?;
        // `None`: the prefill ran and chose nothing; the session replays
        // through `forward`, from 0.
        if next.is_some() {
            self.held = tokens.len();
        }
        Ok(next)
    }
}
