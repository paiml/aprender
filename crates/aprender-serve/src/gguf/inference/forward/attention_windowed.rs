//! APR-EMBED-001 EG-2 (#4901): bidirectional windowed multi-head attention, the shared
//! block every non-causal forward calls (I-10).

use crate::error::{RealizarError, Result};
use crate::gguf::OwnedQuantizedModel;

impl OwnedQuantizedModel {
    /// Bidirectional multi-head attention over a whole sequence, with an optional symmetric
    /// window: the shared block every non-causal forward calls (APR-EMBED-001 EG-2, I-10).
    /// Associated, not a method: it reads no model field.
    ///
    /// `q` is `[seq_len, num_heads * head_dim]`; `k` and `v` are
    /// `[seq_len, num_kv_heads * head_dim]`. Query head `h` reads KV head
    /// `h / (num_heads / num_kv_heads)` (grouped-query). Position `i` attends to every `j`
    /// with `|i - j| <= half_window`, or to every position when `half_window` is `None`.
    /// A position outside the window is never scored, so its weight is exactly 0. There is
    /// no causal mask. Scores are `scale * q·k`; the caller passes the scale its model uses.
    ///
    /// # Errors
    /// A slice length that does not match the shape, a zero head count or `head_dim`, or
    /// `num_heads` not a multiple of `num_kv_heads`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn parallel_multihead_attention_windowed(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        scale: f32,
        half_window: Option<usize>,
    ) -> Result<Vec<f32>> {
        use rayon::prelude::*;

        let q_width = num_heads * head_dim;
        let kv_width = num_kv_heads * head_dim;
        if num_heads == 0 || num_kv_heads == 0 || head_dim == 0 || num_heads % num_kv_heads != 0 {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "windowed attention: {num_heads} query heads over {num_kv_heads} KV heads \
                     of width {head_dim}"
                ),
            });
        }
        if q.len() != seq_len * q_width || k.len() != seq_len * kv_width || k.len() != v.len() {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "windowed attention: q {} / k {} / v {} elements for {seq_len} positions \
                     of {q_width} / {kv_width}",
                    q.len(),
                    k.len(),
                    v.len()
                ),
            });
        }
        let group = num_heads / num_kv_heads;
        let mut out = vec![0.0f32; seq_len * q_width];
        out.par_chunks_mut(q_width.max(1))
            .enumerate()
            .for_each(|(i, row)| {
                let (lo, hi) = half_window.map_or((0, seq_len), |w| {
                    (
                        i.saturating_sub(w),
                        i.saturating_add(w).saturating_add(1).min(seq_len),
                    )
                });
                let mut weights = vec![0.0f32; hi - lo];
                for (h, out_h) in row.chunks_exact_mut(head_dim).enumerate() {
                    let kv = (h / group) * head_dim;
                    let q_h = &q[i * q_width + h * head_dim..][..head_dim];
                    for (w, j) in weights.iter_mut().zip(lo..hi) {
                        *w = scale * crate::apr::simd_dot(q_h, &k[j * kv_width + kv..][..head_dim]);
                    }
                    crate::gguf::ops::softmax(&mut weights);
                    for (w, j) in weights.iter().zip(lo..hi) {
                        let v_j = &v[j * kv_width + kv..][..head_dim];
                        for (o, x) in out_h.iter_mut().zip(v_j) {
                            *o += w * x;
                        }
                    }
                }
            });
        Ok(out)
    }
}
