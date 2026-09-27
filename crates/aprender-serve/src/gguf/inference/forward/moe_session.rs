//! PMAT-4269 (M1): the Qwen3-MoE CPU forward behind the one engine,
//! [`Session`](crate::session::Session).
//!
//! Before this, `run_qwen3_moe_generate` and its streaming twin each carried
//! their own prefill, sampler and stop handling. [`Qwen3MoeForward`] keeps only
//! what is MoE-specific — advance the KV cache over some tokens through
//! `forward_single_qwen3_moe_with_cache` and give back the logits — and the
//! session owns token choice, stop tokens and the context budget.
//!
//! The expert tensors borrow from the mmapped GGUF, so the forward borrows the
//! mapped file and the model for the length of the session; it does not copy
//! either.

use crate::error::{RealizarError, Result};
use crate::gguf::qwen3_moe_load::{load_qwen3_moe_layer, Qwen3MoeQuantizedLayer};
use crate::gguf::{MappedGGUFModel, OwnedQuantizedKVCache, OwnedQuantizedModel};

/// A Qwen3-MoE model plus its decode state: the one engine over
/// [`Qwen3MoeForward`].
pub type Qwen3MoeSession<'a> = crate::session::Session<Qwen3MoeForward<'a>>;

/// A Qwen3-MoE GGUF's CPU forward: the only MoE code a verb reaches, and only
/// through a [`Qwen3MoeSession`].
pub struct Qwen3MoeForward<'a> {
    mapped: &'a MappedGGUFModel,
    model: &'a OwnedQuantizedModel,
    moe_layers: Vec<Qwen3MoeQuantizedLayer>,
    num_experts: usize,
    num_experts_per_tok: usize,
    moe_intermediate: usize,
    context_length: usize,
    cache: Option<OwnedQuantizedKVCache>,
    /// Positions the KV cache was sized for (0: not yet allocated).
    capacity: usize,
    /// Positions the KV cache holds.
    held: usize,
    notices: Vec<String>,
}

impl<'a> Qwen3MoeForward<'a> {
    /// Read the MoE metadata and load every layer's expert descriptors once.
    ///
    /// # Errors
    /// A model whose canonical architecture is not `qwen3_moe`; missing
    /// `expert_count` / `expert_used_count` / `expert_feed_forward_length`
    /// metadata; a layer whose expert tensors cannot be loaded.
    pub fn cpu(mapped: &'a MappedGGUFModel, model: &'a OwnedQuantizedModel) -> Result<Self> {
        let config = model.config();
        let arch = crate::tensor_names::normalize_architecture(&config.architecture);
        if arch != "qwen3_moe" {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "qwen3_moe session: arch '{}' (canonical '{arch}') is not qwen3_moe — \
                     the caller should open a DenseSession instead",
                    config.architecture
                ),
            });
        }
        let missing = |key: &str| RealizarError::InvalidShape {
            reason: format!(
                "qwen3_moe session: missing '{}.{key}' in GGUF metadata",
                config.architecture
            ),
        };
        let num_experts = mapped
            .model
            .expert_count()
            .ok_or_else(|| missing("expert_count"))?;
        let num_experts_per_tok = mapped
            .model
            .expert_used_count()
            .ok_or_else(|| missing("expert_used_count"))?;
        let moe_intermediate = mapped
            .model
            .expert_feed_forward_length()
            .ok_or_else(|| missing("expert_feed_forward_length"))?;
        let data = mapped.data();
        let moe_layers = (0..config.num_layers)
            .map(|layer_idx| load_qwen3_moe_layer(&mapped.model, data, layer_idx))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            mapped,
            model,
            moe_layers,
            num_experts,
            num_experts_per_tok,
            moe_intermediate,
            context_length: config.context_length.max(1),
            cache: None,
            capacity: 0,
            held: 0,
            notices: vec!["Backend: CPU".to_string()],
        })
    }
}

impl crate::session::ArchForward for Qwen3MoeForward<'_> {
    fn arch(&self) -> &'static str {
        "qwen3_moe"
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
        &self.notices
    }

    fn reserve(&mut self, positions: usize) -> Result<bool> {
        let positions = positions.min(self.context_length).max(1);
        if positions <= self.capacity && self.cache.is_some() {
            return Ok(false);
        }
        let capacity = positions
            .max(self.capacity.saturating_mul(2))
            .min(self.context_length)
            .max(positions);
        match &mut self.cache {
            // Grown in place: what it holds stays held.
            Some(cache) => cache.grow_to(capacity),
            None => {
                self.cache = Some(OwnedQuantizedKVCache::from_config(
                    self.model.config(),
                    capacity,
                ));
            },
        }
        self.capacity = capacity;
        Ok(false)
    }

    fn forward(&mut self, tokens: &[u32], start: usize) -> Result<Vec<f32>> {
        if self.cache.is_none() {
            self.reserve(tokens.len())?;
        }
        let cache = self
            .cache
            .as_mut()
            .ok_or_else(|| RealizarError::InvalidShape {
                reason: "qwen3_moe session: no KV cache after reserve".to_string(),
            })?;
        // The session resumes only at exactly what the cache holds; anything
        // else replays the whole span from a reset cache.
        let start = if start == self.held { start } else { 0 };
        if start == 0 {
            cache.reset();
            self.held = 0;
        }
        let data = self.mapped.data();
        let mut logits = Vec::new();
        for (pos, &token) in tokens.iter().enumerate().skip(start) {
            logits = self.model.forward_single_qwen3_moe_with_cache(
                token,
                cache,
                pos,
                &self.moe_layers,
                self.num_experts,
                self.num_experts_per_tok,
                self.moe_intermediate,
                data,
            )?;
            self.held = pos + 1;
        }
        if logits.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "qwen3_moe session: forward over {} tokens from {start} produced no logits",
                    tokens.len()
                ),
            });
        }
        Ok(logits)
    }
}
