//! ModernBERT embeddings: `LayerNorm(tok_embeddings(ids))` — no bias, no position
//! embedding (RoPE carries position inside attention).

use super::{layer_norm, ModernBertError};

/// Token embedding table `[vocab, d]` plus its bias-free LayerNorm weight `[d]`.
/// Constructed only by the loader, so the shapes agree with the config.
#[derive(Debug, Clone)]
pub struct ModernBertEmbeddings {
    tok_embeddings: Vec<f32>,
    norm: Vec<f32>,
    vocab_size: usize,
    hidden_size: usize,
}

impl ModernBertEmbeddings {
    pub(crate) fn from_parts(
        tok_embeddings: Vec<f32>,
        norm: Vec<f32>,
        vocab_size: usize,
        hidden_size: usize,
    ) -> Self {
        Self {
            tok_embeddings,
            norm,
            vocab_size,
            hidden_size,
        }
    }

    /// Gather `ids` and normalise; returns `[ids.len(), d]`.
    ///
    /// # Errors
    ///
    /// [`ModernBertError::OutOfVocab`] for an id `>= vocab_size` (never an index panic),
    /// or a shape error from [`layer_norm`].
    pub fn forward(&self, ids: &[u32], eps: f64) -> Result<Vec<f32>, ModernBertError> {
        let d = self.hidden_size;
        let mut x = Vec::with_capacity(ids.len().saturating_mul(d));
        for (position, &id) in ids.iter().enumerate() {
            let t = id as usize;
            let row = self
                .tok_embeddings
                .get(t.saturating_mul(d)..t.saturating_add(1).saturating_mul(d))
                .filter(|_| t < self.vocab_size)
                .ok_or(ModernBertError::OutOfVocab {
                    position,
                    id,
                    vocab_size: self.vocab_size,
                })?;
            x.extend_from_slice(row);
        }
        layer_norm(&x, d, &self.norm, None, eps)
    }
}
