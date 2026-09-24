//! `AprCpuForward`: the APR transformer's [`ArchForward`], the seam that puts
//! `apr serve`'s APR CPU completion/chat path on [`crate::session::Session`]
//! (PMAT-4269, one engine, #4263).
//!
//! Unlike the Qwen3.5 GGUF forward
//! ([`crate::gguf::inference::forward::qwen35_session::Qwen35Forward`]), the
//! APR CPU backend has no batched-prefill path and no GPU fallback: it
//! forwards one token at a time
//! ([`AprTransformer::forward_with_cache`]), so [`ArchForward::on_gpu`] is
//! always `false` and [`ArchForward::batched_prefills`] is always `0`.

use super::{AprKVCache, AprTransformer};
use crate::error::{RealizarError, Result};
use crate::session::ArchForward;

/// The APR transformer's forward, wired to a [`crate::session::Session`].
///
/// The decode state ([`AprKVCache`]) is pre-allocated once, at construction, to
/// the model's declared context length (or [`AprKVCache`]'s own 2048-token
/// fallback when the model carries no `context_length`) — it is never grown or
/// dropped, so [`ArchForward::reserve`] never reports the state was dropped.
pub struct AprCpuForward {
    model: AprTransformer,
    cache: AprKVCache,
    /// [`AprKVCache::capacity`] at construction — the state's fixed size, and
    /// what [`ArchForward::context_length`] reports.
    context_length: usize,
}

impl AprCpuForward {
    /// Wrap `model` with a freshly-allocated decode state.
    #[must_use]
    pub fn new(model: AprTransformer) -> Self {
        let cache = AprKVCache::new(&model.config);
        let context_length = cache.capacity();
        Self {
            model,
            cache,
            context_length,
        }
    }

    /// The wrapped model, for callers that need its config or metadata outside
    /// the generate loop (tokenizer selection, architecture name, ...).
    #[must_use]
    pub fn model(&self) -> &AprTransformer {
        &self.model
    }
}

impl ArchForward for AprCpuForward {
    fn arch(&self) -> &'static str {
        "apr"
    }

    fn on_gpu(&self) -> bool {
        false
    }

    fn context_length(&self) -> usize {
        self.context_length
    }

    fn batched_prefills(&self) -> usize {
        0
    }

    fn notices(&self) -> &[String] {
        &[]
    }

    /// The cache is already sized to `context_length` at construction, so this
    /// only refuses a turn the fixed state cannot hold; it never (re)allocates
    /// and so never reports the state was dropped.
    ///
    /// # Errors
    /// `positions` exceeds the state's fixed capacity.
    fn reserve(&mut self, positions: usize) -> Result<bool> {
        if positions > self.cache.capacity() {
            return Err(RealizarError::ContextLimitExceeded {
                provided: positions,
                maximum: self.cache.capacity(),
            });
        }
        Ok(false)
    }

    fn forward(&mut self, tokens: &[u32], start: usize) -> Result<Vec<f32>> {
        if start == 0 {
            self.cache.clear();
        }
        let mut logits = Vec::new();
        for (pos, &token) in tokens.iter().enumerate().skip(start) {
            logits = self.model.forward_with_cache(token, &mut self.cache, pos)?;
        }
        Ok(logits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::apr_transformer::AprTransformerConfig;
    use crate::gguf::QuantizedGenerateConfig;
    use crate::session::{entries_for, EntryKind, Session};

    /// A tiny transformer with non-zero weights, so its logits are not all-zero
    /// ties (mirrors `apr_transformer/tests/create.rs::create_test_transformer`).
    fn tiny_model(hidden_dim: usize, vocab_size: usize) -> AprTransformer {
        let config = AprTransformerConfig {
            hidden_dim,
            num_layers: 1,
            num_heads: 2,
            num_kv_heads: 2,
            vocab_size,
            intermediate_dim: hidden_dim * 2,
            context_length: 64,
            rope_theta: 10000.0,
            eps: 1e-5,
            ..Default::default()
        };
        let mut t = AprTransformer::new(config);
        for tok in 0..vocab_size {
            for d in 0..hidden_dim {
                t.token_embedding[tok * hidden_dim + d] = ((tok + d + 1) as f32) * 0.01;
            }
        }
        for w in &mut t.output_norm_weight {
            *w = 1.0;
        }
        for v in 0..vocab_size {
            for d in 0..hidden_dim {
                t.lm_head_weight[v * hidden_dim + d] = ((v + d) as f32).sin() * 0.01;
            }
        }
        t
    }

    /// The APR CPU serve path (`Session<AprCpuForward>::generate`) leaves
    /// exactly one witness `Entry`, `arch == "apr"` — the guard
    /// `tests_engine_identity` (session.rs module docs) checks for every verb ×
    /// arch. A verb that decodes through its own loop or calls `AprCpuForward`
    /// directly leaves none.
    #[test]
    fn apr_cpu_forward_through_session_leaves_an_apr_witness_entry() {
        let model = tiny_model(16, 32);
        let mut session = Session::new(AprCpuForward::new(model));
        let prompt = [4269_u32, 4270, 4271];
        let config = QuantizedGenerateConfig::deterministic(2);
        session
            .generate(&prompt, &config, &mut |_| true)
            .expect("turn");

        let entries = entries_for(&prompt);
        assert_eq!(entries.len(), 1, "exactly one entry for this prompt");
        assert_eq!(entries[0].arch, "apr");
        assert_eq!(entries[0].kind, EntryKind::Generate);
        assert!(!session.on_gpu(), "the APR CPU forward never reports GPU");
    }

    /// A prompt that extends what the state holds reuses it — the same prefix
    /// reuse contract `session_tests.rs` pins for the scripted forward.
    #[test]
    fn an_extending_prompt_reuses_the_apr_cpu_state() {
        let model = tiny_model(16, 32);
        let mut session = Session::new(AprCpuForward::new(model));
        let config = QuantizedGenerateConfig::deterministic(1);
        let t1 = session
            .generate(&[4280, 4281], &config, &mut |_| true)
            .expect("t1");
        let mut p2 = t1.tokens.clone();
        p2.push(4282);
        let t2 = session
            .generate(&p2, &config, &mut |_| true)
            .expect("t2");
        assert_eq!(t2.reused, 2, "t1 held [4280, 4281] without forwarding its own token");
    }
}
