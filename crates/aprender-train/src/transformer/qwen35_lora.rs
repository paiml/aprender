//! LoRA over the Qwen3.5 training model (R4 CPU prep).
//!
//! The adapted weight is `W' = W + s · B · A`, with `W` `[rows × cols]` row-major
//! (`y = W x`), `A` `[rank × cols]`, `B` `[rows × rank]`, and `s = alpha / rank`. `B`
//! starts at zero, so the adapted model starts equal to the base.
//!
//! The gradient reuses the full backward: merge the adapters into a working copy,
//! take `dW'` from [`Qwen35Model::loss_and_grads`], then project it by the chain rule:
//! `dA = s · Bᵀ · dW'` and `dB = s · dW' · Aᵀ`. That materialises `dW'` (R4 on CUDA
//! must not), but it makes the adapter gradient exact by construction of an already
//! gradchecked backward.

use rayon::prelude::*;

use super::super::qwen35_layer_backward::Qwen35MixerGrads;
use super::super::qwen35_lm::Qwen35LmGrads;
use super::{OwnedMixer, Qwen35Model};
use crate::Tensor;

/// A projection matrix LoRA can adapt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoraTarget {
    /// Attention `[q | gate]` projection.
    AttnQ,
    /// Attention key projection.
    AttnK,
    /// Attention value projection.
    AttnV,
    /// Attention output projection.
    AttnOut,
    /// Gated `DeltaNet` fused `qkv` projection.
    GdnQkv,
    /// Gated `DeltaNet` output-gate (`z`) projection.
    GdnGate,
    /// Gated `DeltaNet` output projection.
    GdnOut,
    /// `SwiGLU` gate projection.
    FfnGate,
    /// `SwiGLU` up projection.
    FfnUp,
    /// `SwiGLU` down projection.
    FfnDown,
}

impl LoraTarget {
    /// Every target: attention, Gated `DeltaNet` and MLP projections (R4's cell).
    pub const ALL: [Self; 10] = [
        Self::AttnQ,
        Self::AttnK,
        Self::AttnV,
        Self::AttnOut,
        Self::GdnQkv,
        Self::GdnGate,
        Self::GdnOut,
        Self::FfnGate,
        Self::FfnUp,
        Self::FfnDown,
    ];

    /// Whether the matrix writes the residual stream (`rows = hidden`) rather than
    /// reading it (`cols = hidden`).
    fn writes_hidden(self) -> bool {
        matches!(self, Self::AttnOut | Self::GdnOut | Self::FfnDown)
    }
}

/// One adapter's placement; its `A` and `B` are `params[2i]` and `params[2i + 1]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoraSlot {
    /// Layer index.
    pub layer: usize,
    /// Which matrix in that layer.
    pub target: LoraTarget,
    /// Rows of the adapted weight (its output width).
    pub rows: usize,
    /// Columns of the adapted weight (its input width).
    pub cols: usize,
}

/// LoRA adapters over a [`Qwen35Model`]. `params` is laid out `[A₀, B₀, A₁, B₁, …]` so
/// an [`crate::optim::Optimizer`] steps it directly.
#[derive(Debug)]
pub struct Qwen35Lora {
    /// Adapter rank.
    pub rank: usize,
    /// `alpha / rank`.
    pub scale: f32,
    /// Where each adapter sits.
    pub slots: Vec<LoraSlot>,
    /// `[A₀, B₀, A₁, B₁, …]`.
    pub params: Vec<Tensor>,
}

fn weight(m: &Qwen35Model, layer: usize, t: LoraTarget) -> Option<&Vec<f32>> {
    let l = &m.layers[layer];
    match (t, &l.mixer) {
        (LoraTarget::AttnQ, OwnedMixer::Attention(a)) => Some(&a.q),
        (LoraTarget::AttnK, OwnedMixer::Attention(a)) => Some(&a.k),
        (LoraTarget::AttnV, OwnedMixer::Attention(a)) => Some(&a.v),
        (LoraTarget::AttnOut, OwnedMixer::Attention(a)) => Some(&a.out),
        (LoraTarget::GdnQkv, OwnedMixer::Gdn(g)) => Some(&g.qkv),
        (LoraTarget::GdnGate, OwnedMixer::Gdn(g)) => Some(&g.gate),
        (LoraTarget::GdnOut, OwnedMixer::Gdn(g)) => Some(&g.out),
        (LoraTarget::FfnGate, _) => Some(&l.ffn_gate),
        (LoraTarget::FfnUp, _) => Some(&l.ffn_up),
        (LoraTarget::FfnDown, _) => Some(&l.ffn_down),
        _ => None,
    }
}

pub(super) fn weight_mut(
    m: &mut Qwen35Model,
    layer: usize,
    t: LoraTarget,
) -> Option<&mut Vec<f32>> {
    let l = &mut m.layers[layer];
    match (t, &mut l.mixer) {
        (LoraTarget::AttnQ, OwnedMixer::Attention(a)) => Some(&mut a.q),
        (LoraTarget::AttnK, OwnedMixer::Attention(a)) => Some(&mut a.k),
        (LoraTarget::AttnV, OwnedMixer::Attention(a)) => Some(&mut a.v),
        (LoraTarget::AttnOut, OwnedMixer::Attention(a)) => Some(&mut a.out),
        (LoraTarget::GdnQkv, OwnedMixer::Gdn(g)) => Some(&mut g.qkv),
        (LoraTarget::GdnGate, OwnedMixer::Gdn(g)) => Some(&mut g.gate),
        (LoraTarget::GdnOut, OwnedMixer::Gdn(g)) => Some(&mut g.out),
        (LoraTarget::FfnGate, _) => Some(&mut l.ffn_gate),
        (LoraTarget::FfnUp, _) => Some(&mut l.ffn_up),
        (LoraTarget::FfnDown, _) => Some(&mut l.ffn_down),
        _ => None,
    }
}

fn grad(g: &Qwen35LmGrads, layer: usize, t: LoraTarget) -> &[f32] {
    let l = &g.layers[layer];
    match (t, &l.mixer) {
        (LoraTarget::AttnQ, Qwen35MixerGrads::Attention(a)) => &a.q,
        (LoraTarget::AttnK, Qwen35MixerGrads::Attention(a)) => &a.k,
        (LoraTarget::AttnV, Qwen35MixerGrads::Attention(a)) => &a.v,
        (LoraTarget::AttnOut, Qwen35MixerGrads::Attention(a)) => &a.out,
        (LoraTarget::GdnQkv, Qwen35MixerGrads::Gdn(x)) => &x.qkv,
        (LoraTarget::GdnGate, Qwen35MixerGrads::Gdn(x)) => &x.gate,
        (LoraTarget::GdnOut, Qwen35MixerGrads::Gdn(x)) => &x.out,
        (LoraTarget::FfnGate, _) => &l.ffn_gate,
        (LoraTarget::FfnUp, _) => &l.ffn_up,
        (LoraTarget::FfnDown, _) => &l.ffn_down,
        _ => unreachable!("slot {t:?} was placed on a layer whose mixer has it"),
    }
}

/// Deterministic uniform draws in `(-bound, bound)`.
fn uniform(state: &mut u64, n: usize, bound: f32) -> Vec<f32> {
    (0..n)
        .map(|_| {
            *state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (((*state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0) * bound
        })
        .collect()
}

impl Qwen35Lora {
    /// Adapters of `rank` on every `targets` matrix each layer has. `A` is
    /// Kaiming-uniform (`±1/√cols`), `B` is zero.
    #[must_use]
    pub fn new(
        model: &Qwen35Model,
        targets: &[LoraTarget],
        rank: usize,
        alpha: f32,
        seed: u64,
    ) -> Self {
        let hidden = model.embed.len() / model.vocab_size;
        let mut state = seed;
        let (mut slots, mut params) = (Vec::new(), Vec::new());
        for layer in 0..model.layers.len() {
            for &target in targets {
                let Some(w) = weight(model, layer, target) else { continue };
                let other = w.len() / hidden;
                let (rows, cols) =
                    if target.writes_hidden() { (hidden, other) } else { (other, hidden) };
                let bound = 1.0 / (cols as f32).sqrt();
                params.push(Tensor::from_vec(uniform(&mut state, rank * cols, bound), true));
                params.push(Tensor::zeros(rows * rank, true));
                slots.push(LoraSlot { layer, target, rows, cols });
            }
        }
        Self { rank, scale: alpha / rank as f32, slots, params }
    }

    /// Trainable parameter count.
    #[must_use]
    pub fn num_params(&self) -> usize {
        self.params.iter().map(Tensor::len).sum()
    }

    /// Write `W + s · B · A` into `work`'s adapted matrices; `work` must be a copy of `base`
    /// (only the adapted matrices are rewritten).
    pub fn merge_into(&self, base: &Qwen35Model, work: &mut Qwen35Model) {
        let (r, s) = (self.rank, self.scale);
        for (i, slot) in self.slots.iter().enumerate() {
            let (a, b) = (self.params[2 * i].data(), self.params[2 * i + 1].data());
            let (a, b) = (a.as_slice().expect("contiguous"), b.as_slice().expect("contiguous"));
            let w0 = weight(base, slot.layer, slot.target).expect("slot placed on this matrix");
            let w = weight_mut(work, slot.layer, slot.target).expect("slot placed on this matrix");
            w.par_chunks_exact_mut(slot.cols)
                .zip(w0.par_chunks_exact(slot.cols))
                .enumerate()
                .for_each(|(o, (row, row0))| {
                    row.copy_from_slice(row0);
                    for (k, &bk) in b[o * r..(o + 1) * r].iter().enumerate() {
                        let c = s * bk;
                        if c != 0.0 {
                            for (x, &ak) in
                                row.iter_mut().zip(&a[k * slot.cols..(k + 1) * slot.cols])
                            {
                                *x += c * ak;
                            }
                        }
                    }
                });
        }
    }

    /// Set each `A`/`B` gradient from the merged model's full gradient:
    /// `dA = s · Bᵀ · dW'`, `dB = s · dW' · Aᵀ`.
    pub fn set_grads(&self, grads: &Qwen35LmGrads) {
        let (r, s) = (self.rank, self.scale);
        for (i, slot) in self.slots.iter().enumerate() {
            let dw = grad(grads, slot.layer, slot.target);
            let (ta, tb) = (&self.params[2 * i], &self.params[2 * i + 1]);
            let (a, b) = (ta.data(), tb.data());
            let (a, b) = (a.as_slice().expect("contiguous"), b.as_slice().expect("contiguous"));
            let c = slot.cols;
            let db: Vec<f32> = dw
                .par_chunks_exact(c)
                .flat_map_iter(|row| {
                    a.chunks_exact(c)
                        .map(move |ak| s * row.iter().zip(ak).map(|(x, y)| x * y).sum::<f32>())
                })
                .collect();
            let da: Vec<f32> = (0..r)
                .into_par_iter()
                .flat_map_iter(|k| {
                    let mut acc = vec![0.0_f32; c];
                    for (o, row) in dw.chunks_exact(c).enumerate() {
                        let bk = b[o * r + k];
                        if bk != 0.0 {
                            for (x, &g) in acc.iter_mut().zip(row) {
                                *x += bk * g;
                            }
                        }
                    }
                    acc.into_iter().map(move |x| s * x)
                })
                .collect();
            ta.set_grad(da.into());
            tb.set_grad(db.into());
        }
    }
}
