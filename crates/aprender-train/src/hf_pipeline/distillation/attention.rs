//! Attention Transfer Loss
//!
//! Transfers attention maps from teacher to student.
//! Based on Zagoruyko & Komodakis (2017).

use ndarray::Array2;

use super::utils::l2_normalize;

/// Attention Transfer Loss
///
/// Transfers attention maps from teacher to student.
/// Based on Zagoruyko & Komodakis (2017).
#[derive(Debug, Clone)]
pub struct AttentionTransfer {
    /// Loss weight
    pub weight: f32,
}

impl Default for AttentionTransfer {
    fn default() -> Self {
        Self { weight: 0.1 }
    }
}

impl AttentionTransfer {
    /// Create new attention transfer config
    #[must_use]
    pub fn new(weight: f32) -> Self {
        Self { weight }
    }

    /// Compute attention transfer loss
    ///
    /// Uses L2 norm of normalized attention map differences.
    pub fn loss(
        &self,
        student_attention: &[Array2<f32>],
        teacher_attention: &[Array2<f32>],
    ) -> f32 {
        let mut total_loss = 0.0;
        let count = student_attention.len().min(teacher_attention.len());

        for (s_attn, t_attn) in student_attention.iter().zip(teacher_attention.iter()) {
            // L2 normalize attention maps
            let s_norm = l2_normalize(s_attn);
            let t_norm = l2_normalize(t_attn);

            // Frobenius norm of difference
            let diff = &s_norm - &t_norm;
            let frob = diff.mapv(|x| x * x).sum().sqrt();
            total_loss += frob * frob;
        }

        if count > 0 {
            self.weight * total_loss / count as f32
        } else {
            0.0
        }
    }
}
