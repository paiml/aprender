// #4269 (workstream M of #4263): the SafeTensors CPU forward, so `apr serve`
// and `apr chat` drive SafeTensors generation through the same
// `crate::session::Session` every other architecture uses, instead of
// `AprTransformer::generate_with_cache`'s own copy of the decode loop.
//
// `StCpuForward` wraps a borrowed `AprTransformer` and an `AprKVCache` sized
// once, up front, to the model's declared context length: `AprKVCache::new`
// already pre-allocates the whole context, so (unlike `Qwen35Forward`) this
// state never needs to double. It has no batched prefill —
// `AprTransformer::forward_with_cache` is one token at a time — so
// `ArchForward::batched_prefills` is always 0 here.
//
// This file is `include!`d into `safetensors_infer.rs`'s module scope
// (matching that file's own `include!` pattern), so it reuses that file's
// `use crate::error::{RealizarError, Result};` rather than re-importing.

use crate::apr_transformer::AprKVCache;
use crate::session::ArchForward;

/// The SafeTensors CPU forward: the only ST-CPU code a verb reaches, and
/// only through a [`Session`](crate::session::Session) (#4269).
pub struct StCpuForward<'a> {
    model: &'a AprTransformer,
    /// The decode state; `None` until the first turn sizes it (to the whole
    /// declared context, in one allocation — see the module docs).
    cache: Option<AprKVCache>,
    notices: Vec<String>,
}

/// A SafeTensors CPU session: the engine's
/// [`Session`](crate::session::Session) over [`StCpuForward`].
pub type StCpuSession<'a> = crate::session::Session<StCpuForward<'a>>;

impl<'a> StCpuForward<'a> {
    /// Wrap `model`'s CPU forward. The decode state starts empty.
    #[must_use]
    pub fn new(model: &'a AprTransformer) -> Self {
        Self {
            model,
            cache: None,
            notices: vec![
                "Backend: CPU (SafeTensors AprTransformer forward, #4269)".to_string(),
            ],
        }
    }
}

impl<'a> ArchForward for StCpuForward<'a> {
    fn arch(&self) -> &'static str {
        "safetensors"
    }

    fn on_gpu(&self) -> bool {
        false
    }

    fn context_length(&self) -> usize {
        // FIXME(#4269 RED): this must be the length `AprKVCache::new` builds
        // (config.context_length, or its own 2048 fallback when unset) —
        // returning 0 unconditionally is the RED commit's deliberate defect;
        // `Session::generate` refuses any prompt whole against it.
        0
    }

    fn batched_prefills(&self) -> usize {
        // AprTransformer::forward_with_cache is one token at a time; there
        // is no batched prefill call to count.
        0
    }

    fn notices(&self) -> &[String] {
        &self.notices
    }

    fn reserve(&mut self, _positions: usize) -> Result<bool> {
        if self.cache.is_some() {
            return Ok(false);
        }
        // AprKVCache::new pre-allocates the WHOLE declared context up front
        // (config.rs), so one allocation covers every turn this session will
        // ever see.
        self.cache = Some(AprKVCache::new(&self.model.config));
        Ok(true)
    }

    fn forward(&mut self, tokens: &[u32], start: usize) -> Result<Vec<f32>> {
        let cache = self.cache.as_mut().ok_or_else(|| RealizarError::InvalidShape {
            reason: "safetensors session: the CPU state was never allocated".to_string(),
        })?;
        if start == 0 {
            cache.clear();
        }
        let mut logits = Vec::new();
        for (pos, &token) in tokens.iter().enumerate().skip(start) {
            logits = self.model.forward_with_cache(token, cache, pos)?;
        }
        if logits.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "safetensors session: forward advanced zero positions (tokens[start..] \
                         was empty)"
                    .to_string(),
            });
        }
        Ok(logits)
    }
}

#[cfg(test)]
#[path = "safetensors_infer_session_tests.rs"]
mod safetensors_infer_session_tests;
