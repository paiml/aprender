//! Laya's per-marker scorer: `nn.Sequential(LayerNorm(d), Linear(d, d), GELU(),
//! Linear(d, 1))` (checkpoint names `scorer.0`, `scorer.1`, `scorer.3`), applied to the
//! head's hidden state at each `[MASK]` marker. GELU is torch's default exact (erf)
//! form, core's [`gelu_exact`].

use super::head::TORCH_LAYER_NORM_EPS;
use aprender::models::modernbert::{gelu_exact, layer_norm, Linear, ModernBertError};

/// The scorer MLP.
#[derive(Debug, Clone)]
pub struct Scorer {
    pub(crate) norm: (Vec<f32>, Vec<f32>),
    pub(crate) fc1: Linear,
    pub(crate) fc2: Linear,
}

impl Scorer {
    /// One logit per row of `m: [k, d]`.
    ///
    /// # Errors
    ///
    /// A shape or GEMM error from core's primitives.
    pub fn forward(&self, m: &[f32], k: usize) -> Result<Vec<f32>, ModernBertError> {
        let d = self.fc1.inp;
        let mn = layer_norm(m, d, &self.norm.0, Some(&self.norm.1), TORCH_LAYER_NORM_EPS)?;
        let mut s = self.fc1.forward(&mn, k)?;
        for v in &mut s {
            *v = gelu_exact(*v);
        }
        self.fc2.forward(&s, k)
    }
}
