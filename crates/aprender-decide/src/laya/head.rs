//! Laya's decision head: `head_layers` x torch `nn.TransformerEncoderLayer(d, nhead,
//! 4d, norm_first=True)`, ReLU FFN, every projection and norm with a bias.
//!
//! `nhead = max(1, d / 64)` and `hd = d / nhead` (Laya `common.py:165`) — NOT a fixed
//! 64-wide split: the tiny fixture's `d = 32` has one 32-wide head. A `d` the rule
//! does not divide is refused ([`LayaError::HeadDoesNotDivide`]), as torch's
//! `MultiheadAttention` refuses it, instead of being mis-sliced.
//!
//! Numerics are core's reusable primitives (OPS-03): [`Linear`], [`layer_norm`] with
//! torch's `nn.LayerNorm` default eps 1e-5, and [`attention`] (global, no window).

use super::LayaError;
use aprender::models::modernbert::{attention, layer_norm, Linear, ModernBertError};
use rayon::prelude::*;

/// torch `nn.LayerNorm` / `nn.TransformerEncoderLayer` default `eps`.
pub const TORCH_LAYER_NORM_EPS: f64 = 1e-5;

/// Laya's head geometry for hidden size `d`: `(nhead, hd)` with
/// `nhead = max(1, d / 64)`, `hd = d / nhead`.
///
/// # Errors
///
/// [`LayaError::HeadDoesNotDivide`] when `nhead` does not divide `d` (or `d == 0`).
pub fn head_geometry(d: usize) -> Result<(usize, usize), LayaError> {
    let nhead = (d / 64).max(1);
    if d == 0 || d % nhead != 0 {
        return Err(LayaError::HeadDoesNotDivide { d, nhead });
    }
    Ok((nhead, d / nhead))
}

/// One pre-norm `nn.TransformerEncoderLayer` of the decision head.
#[derive(Debug, Clone)]
pub struct HeadLayer {
    pub(crate) norm1: (Vec<f32>, Vec<f32>),
    pub(crate) in_proj: Linear,
    pub(crate) out_proj: Linear,
    pub(crate) norm2: (Vec<f32>, Vec<f32>),
    pub(crate) linear1: Linear,
    pub(crate) linear2: Linear,
    pub(crate) nhead: usize,
    pub(crate) hd: usize,
}

impl HeadLayer {
    /// `x <- x + out_proj(MHA(norm1(x)))`, then `x <- x + linear2(relu(linear1(norm2(x))))`
    /// over `x: [l, d]`, in place.
    ///
    /// # Errors
    ///
    /// A shape or GEMM error from core's primitives.
    pub fn forward(&self, x: &mut [f32], l: usize) -> Result<(), ModernBertError> {
        let d = self.nhead * self.hd;
        let xn = layer_norm(
            x,
            d,
            &self.norm1.0,
            Some(&self.norm1.1),
            TORCH_LAYER_NORM_EPS,
        )?;
        let qkv = self.in_proj.forward(&xn, l)?;
        let (mut q, mut k, mut v) = (
            vec![0.0f32; l * d],
            vec![0.0f32; l * d],
            vec![0.0f32; l * d],
        );
        for (i, r) in qkv.chunks_exact(3 * d).enumerate().take(l) {
            q[i * d..(i + 1) * d].copy_from_slice(&r[..d]);
            k[i * d..(i + 1) * d].copy_from_slice(&r[d..2 * d]);
            v[i * d..(i + 1) * d].copy_from_slice(&r[2 * d..]);
        }
        let a = attention(&q, &k, &v, l, self.nhead, self.hd, None)?;
        let a = self.out_proj.forward(&a, l)?;
        x.par_iter_mut()
            .zip(a.par_iter())
            .for_each(|(h, o)| *h += o);
        let xn = layer_norm(
            x,
            d,
            &self.norm2.0,
            Some(&self.norm2.1),
            TORCH_LAYER_NORM_EPS,
        )?;
        let mut f = self.linear1.forward(&xn, l)?;
        f.par_iter_mut().for_each(|v| *v = v.max(0.0));
        let f = self.linear2.forward(&f, l)?;
        x.par_iter_mut()
            .zip(f.par_iter())
            .for_each(|(h, o)| *h += o);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::head_geometry;
    use crate::LayaError;

    /// Laya `common.py:165`: `nhead = max(1, d // 64)`, `hd = d / nhead`.
    #[test]
    fn nhead_rule() {
        assert_eq!(head_geometry(32).expect("d=32"), (1, 32));
        assert_eq!(head_geometry(64).expect("d=64"), (1, 64));
        assert_eq!(head_geometry(768).expect("d=768"), (12, 64));
        assert_eq!(head_geometry(1024).expect("d=1024"), (16, 64));
    }

    /// A d the rule does not divide is refused, never mis-sliced.
    #[test]
    fn head_does_not_divide() {
        assert_eq!(
            head_geometry(129),
            Err(LayaError::HeadDoesNotDivide { d: 129, nhead: 2 })
        );
        assert_eq!(
            head_geometry(0),
            Err(LayaError::HeadDoesNotDivide { d: 0, nhead: 1 })
        );
    }
}
