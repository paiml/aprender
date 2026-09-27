//! T1-R3 step 4 (#4000, contract `qwen35-train-gdn-v1`, `FALSIFY-QTG-003`): the whole
//! Qwen3.5 language model — embedding, the hybrid layer stack, final norm, `lm_head`
//! (or the tied embedding) — as a generic forward to logits and a next-token
//! cross-entropy loss with its reverse-mode gradient over every weight.
//!
//! The backward keeps each layer's input (`layers + 1` hidden states) and lets each
//! block's backward recompute its own internals.

use super::gdn::{project, GdnFloat};
use super::gdn_backward::project_backward;
use super::qwen35_layer::{qwen35_block_forward, rms_norm_chunks, Qwen35Mixer, SwiGluWeights};
use super::qwen35_layer_backward::{
    qwen35_block_backward, rms_norm_chunks_backward, Qwen35BlockGrads,
};

/// One layer's weights, borrowed.
#[derive(Debug, Clone, Copy)]
pub struct Qwen35LayerRef<'a, T = f32> {
    /// `attn_norm`: `[hidden]`.
    pub attn_norm: &'a [T],
    /// The token mixer.
    pub mixer: Qwen35Mixer<'a, T>,
    /// `post_attention_norm`: `[hidden]`.
    pub post_norm: &'a [T],
    /// The `SwiGLU` FFN.
    pub ffn: SwiGluWeights<'a, T>,
}

/// A whole Qwen3.5 model's weights, borrowed.
#[derive(Debug, Clone)]
pub struct Qwen35LmRef<'a, T = f32> {
    /// `token_embd`: `[vocab × hidden]`.
    pub embed: &'a [T],
    /// The layer stack, bottom first.
    pub layers: Vec<Qwen35LayerRef<'a, T>>,
    /// `output_norm`: `[hidden]`.
    pub final_norm: &'a [T],
    /// `output`: `[vocab × hidden]`, or `None` when tied to `embed`.
    pub lm_head: Option<&'a [T]>,
    /// `RMSNorm` epsilon.
    pub eps: T,
}

/// `∂L/∂W` for a whole model, shaped like [`Qwen35LmRef`].
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen35LmGrads<T = f32> {
    /// `∂L/∂token_embd` (includes the tied `lm_head`'s share when tied).
    pub embed: Vec<T>,
    /// Per layer, bottom first.
    pub layers: Vec<Qwen35BlockGrads<T>>,
    /// `∂L/∂output_norm`.
    pub final_norm: Vec<T>,
    /// `∂L/∂output`, `None` when tied.
    pub lm_head: Option<Vec<T>>,
}

impl<T: GdnFloat> Qwen35LmRef<'_, T> {
    fn hidden_dim(&self) -> usize {
        self.final_norm.len()
    }

    fn vocab(&self) -> usize {
        self.embed.len() / self.hidden_dim()
    }

    fn head(&self) -> &[T] {
        self.lm_head.unwrap_or(self.embed)
    }

    fn embed_tokens(&self, tokens: &[u32]) -> Vec<T> {
        let (d, vocab) = (self.hidden_dim(), self.vocab());
        let mut h = Vec::with_capacity(tokens.len() * d);
        for &t in tokens {
            let t = t as usize;
            assert!(t < vocab, "token {t} outside vocab {vocab}");
            h.extend_from_slice(&self.embed[t * d..(t + 1) * d]);
        }
        h
    }

    fn block(&self, l: &Qwen35LayerRef<'_, T>, h: &[T]) -> Vec<T> {
        qwen35_block_forward(h, l.attn_norm, &l.mixer, l.post_norm, &l.ffn, self.eps)
    }

    /// Causal forward over `tokens` from an empty state; logits are `[len × vocab]`.
    ///
    /// # Panics
    /// If a token id is outside the vocabulary.
    #[must_use]
    pub fn logits(&self, tokens: &[u32]) -> Vec<T> {
        let mut h = self.embed_tokens(tokens);
        for l in &self.layers {
            h = self.block(l, &h);
        }
        rms_norm_chunks(&mut h, self.final_norm, self.eps);
        project(&h, self.head(), self.hidden_dim(), self.vocab())
    }

    /// Mean next-token cross-entropy: position `t` of `tokens` predicts `targets[t]`.
    ///
    /// # Panics
    /// If `targets` is not `tokens`' length, or an id is outside the vocabulary.
    #[must_use]
    pub fn loss(&self, tokens: &[u32], targets: &[u32]) -> T {
        cross_entropy(&self.logits(tokens), targets, self.vocab(), None)
    }

    /// [`Self::loss`] and its gradient with respect to every weight.
    ///
    /// # Panics
    /// As [`Self::loss`].
    #[must_use]
    pub fn loss_and_grads(&self, tokens: &[u32], targets: &[u32]) -> (T, Qwen35LmGrads<T>) {
        let (d, vocab) = (self.hidden_dim(), self.vocab());
        let mut hs = vec![self.embed_tokens(tokens)];
        for l in &self.layers {
            let next = self.block(l, &hs[hs.len() - 1]);
            hs.push(next);
        }
        let top = &hs[hs.len() - 1];
        let mut normed = top.clone();
        rms_norm_chunks(&mut normed, self.final_norm, self.eps);
        let logits = project(&normed, self.head(), d, vocab);
        let mut d_logits = vec![T::ZERO; logits.len()];
        let loss = cross_entropy(&logits, targets, vocab, Some(&mut d_logits));
        let mut d_normed = vec![T::ZERO; normed.len()];
        let d_head = project_backward(&normed, self.head(), &d_logits, d, vocab, &mut d_normed);
        let mut d_final = vec![T::ZERO; d];
        let mut dh =
            rms_norm_chunks_backward(top, self.final_norm, &d_normed, self.eps, &mut d_final);
        let mut layer_grads = Vec::with_capacity(self.layers.len());
        for (l, h_in) in self.layers.iter().zip(&hs).rev() {
            let (d_in, g) = qwen35_block_backward(
                h_in,
                l.attn_norm,
                &l.mixer,
                l.post_norm,
                &l.ffn,
                self.eps,
                &dh,
            );
            dh = d_in;
            layer_grads.push(g);
        }
        layer_grads.reverse();
        let (mut d_embed, lm_head) = if self.lm_head.is_some() {
            (vec![T::ZERO; self.embed.len()], Some(d_head))
        } else {
            (d_head, None)
        };
        for (&t, row) in tokens.iter().zip(dh.chunks_exact(d)) {
            let t = t as usize;
            for (e, &g) in d_embed[t * d..(t + 1) * d].iter_mut().zip(row) {
                *e += g;
            }
        }
        let grads =
            Qwen35LmGrads { embed: d_embed, layers: layer_grads, final_norm: d_final, lm_head };
        (loss, grads)
    }
}

/// Mean over rows of `−log softmax(row)[target]`; with `d_logits`, writes its gradient
/// `(softmax − onehot) / rows` there.
fn cross_entropy<T: GdnFloat>(
    logits: &[T],
    targets: &[u32],
    vocab: usize,
    mut d_logits: Option<&mut Vec<T>>,
) -> T {
    let rows = logits.len() / vocab;
    assert_eq!(targets.len(), rows, "one target per position");
    let n = T::from_usize(rows);
    let mut loss = T::ZERO;
    for (r, (row, &tgt)) in logits.chunks_exact(vocab).zip(targets).enumerate() {
        let tgt = tgt as usize;
        assert!(tgt < vocab, "target {tgt} outside vocab {vocab}");
        let max = row.iter().copied().fold(T::NEG_INFINITY, T::max);
        let sum = kahan_sum(row.iter().map(|&x| (x - max).exp()));
        loss += (max + sum.ln() - row[tgt]) / n;
        if let Some(d) = d_logits.as_deref_mut() {
            for (i, &x) in row.iter().enumerate() {
                let p = (x - max).exp() / sum;
                let onehot = if i == tgt { T::ONE } else { T::ZERO };
                d[r * vocab + i] = (p - onehot) / n;
            }
        }
    }
    loss
}

/// Compensated (Kahan) sum. The softmax denominator over a 248k-token vocabulary in
/// plain f32 drops every term below half an ulp of the running total; on the real 0.8B
/// that biased the loss 1.2e-4 low against an f64 reference.
fn kahan_sum<T: GdnFloat>(xs: impl Iterator<Item = T>) -> T {
    let (mut sum, mut c) = (T::ZERO, T::ZERO);
    for x in xs {
        let y = x - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    sum
}

#[cfg(test)]
#[path = "qwen35_lm_tests.rs"]
mod tests;
