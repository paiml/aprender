// ============================================================================
// E.11.5: QuantizationErrorTrace (MLT-04)
// ============================================================================

use super::quant_type::QuantType;
use crate::brick::exec_graph::BrickId;

/// Quantization error measurement for a single operation.
///
/// Compares quantized computation against FP32 reference using multiple metrics.
#[derive(Debug, Clone)]
pub struct QuantizationErrorTrace {
    /// Brick type being measured
    pub brick_id: BrickId,
    /// Layer index
    pub layer_idx: usize,
    /// Mean squared error vs FP32 reference
    pub mse: f32,
    /// Maximum absolute error
    pub max_abs_error: f32,
    /// Cosine similarity (1.0 = perfect match)
    pub cosine_similarity: f32,
    /// Signal-to-noise ratio in dB
    pub snr_db: f32,
    /// Quantization type used
    pub quant_type: QuantType,
}

impl QuantizationErrorTrace {
    /// Compute error metrics between quantized and reference outputs.
    pub fn compute(
        brick_id: BrickId,
        layer_idx: usize,
        quantized: &[f32],
        reference: &[f32],
        quant_type: QuantType,
    ) -> Self {
        assert_eq!(quantized.len(), reference.len(), "Length mismatch");
        let n = quantized.len();
        if n == 0 {
            return Self {
                brick_id,
                layer_idx,
                mse: 0.0,
                max_abs_error: 0.0,
                cosine_similarity: 1.0, // Perfect match when both empty
                snr_db: f32::INFINITY,
                quant_type,
            };
        }

        // MSE and max abs error
        let mut sum_sq_error = 0.0f64;
        let mut max_abs_error = 0.0f32;
        for (q, r) in quantized.iter().zip(reference.iter()) {
            let error = q - r;
            sum_sq_error += (error as f64) * (error as f64);
            max_abs_error = max_abs_error.max(error.abs());
        }
        let mse = (sum_sq_error / n as f64) as f32;

        // Cosine similarity
        let mut dot = 0.0f64;
        let mut norm_q = 0.0f64;
        let mut norm_r = 0.0f64;
        for (q, r) in quantized.iter().zip(reference.iter()) {
            dot += (*q as f64) * (*r as f64);
            norm_q += (*q as f64) * (*q as f64);
            norm_r += (*r as f64) * (*r as f64);
        }
        let cosine_similarity = if norm_q > 0.0 && norm_r > 0.0 {
            (dot / (norm_q.sqrt() * norm_r.sqrt())) as f32
        } else {
            0.0
        };

        // SNR in dB: 10 * log10(signal_power / noise_power)
        let signal_power = norm_r / n as f64;
        let noise_power = sum_sq_error / n as f64;
        let snr_db = if noise_power > 1e-10 {
            (10.0 * (signal_power / noise_power).max(f64::EPSILON).log10()) as f32
        } else {
            f32::INFINITY
        };

        Self { brick_id, layer_idx, mse, max_abs_error, cosine_similarity, snr_db, quant_type }
    }

    /// Check if error is acceptable (cosine > 0.995).
    pub fn is_acceptable(&self) -> bool {
        self.cosine_similarity > 0.995
    }

    /// Check if error is in warning zone (0.99 < cosine < 0.995).
    pub fn is_warning(&self) -> bool {
        self.cosine_similarity > 0.99 && self.cosine_similarity <= 0.995
    }

    /// Check if error is critical (cosine < 0.99).
    pub fn is_critical(&self) -> bool {
        self.cosine_similarity < 0.99
    }
}

/// Cumulative quantization error across an entire model.
#[derive(Debug, Clone, Default)]
pub struct ModelQuantizationError {
    /// Per-brick error traces
    pub brick_errors: Vec<QuantizationErrorTrace>,
    /// Overall cosine similarity of final logits
    pub logits_cosine: f32,
    /// KL divergence of output probability distributions
    pub output_kl_divergence: f32,
    /// Perplexity difference (PPL_quant - PPL_fp32)
    pub perplexity_delta: f32,
}

impl ModelQuantizationError {
    /// Add a brick error trace.
    pub fn add_error(&mut self, trace: QuantizationErrorTrace) {
        self.brick_errors.push(trace);
    }

    /// Get count of critical errors.
    pub fn critical_count(&self) -> usize {
        self.brick_errors.iter().filter(|e| e.is_critical()).count()
    }

    /// Get count of warning errors.
    pub fn warning_count(&self) -> usize {
        self.brick_errors.iter().filter(|e| e.is_warning()).count()
    }

    /// Get worst brick by cosine similarity.
    pub fn worst_brick(&self) -> Option<&QuantizationErrorTrace> {
        self.brick_errors.iter().min_by(|a, b| {
            a.cosine_similarity
                .partial_cmp(&b.cosine_similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quant_error_perfect_match() {
        let reference = vec![1.0, 2.0, 3.0];
        let quantized = vec![1.0, 2.0, 3.0];
        let trace = QuantizationErrorTrace::compute(
            BrickId::RmsNorm,
            0,
            &quantized,
            &reference,
            QuantType::Q4_K,
        );

        assert!((trace.mse - 0.0).abs() < 1e-6);
        assert!((trace.cosine_similarity - 1.0).abs() < 1e-6);
        assert!(trace.is_acceptable());
    }

    #[test]
    fn test_quant_error_significant_difference() {
        let reference = vec![1.0, 2.0, 3.0];
        // Non-proportional: adds different offsets, changing direction
        let quantized = vec![1.5, 2.1, 2.9];
        let trace = QuantizationErrorTrace::compute(
            BrickId::RmsNorm,
            0,
            &quantized,
            &reference,
            QuantType::Q4_K,
        );

        assert!(trace.mse > 0.0);
        assert!(trace.cosine_similarity < 1.0);
        // Cosine should still be high since vectors are close
        assert!(trace.cosine_similarity > 0.99);
    }

    /// FALSIFICATION TEST: Cosine similarity must be 1.0 for identical vectors
    #[test]
    fn test_falsify_cosine_identical_vectors() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let trace =
            QuantizationErrorTrace::compute(BrickId::RmsNorm, 0, &data, &data, QuantType::F32);

        assert!(
            (trace.cosine_similarity - 1.0).abs() < 1e-6,
            "FALSIFICATION FAILED: identical vectors have cosine {} != 1.0",
            trace.cosine_similarity
        );
    }

    /// FALSIFICATION TEST: Cosine similarity must be symmetric
    #[test]
    fn test_falsify_cosine_symmetry() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];

        let trace_ab = QuantizationErrorTrace::compute(BrickId::RmsNorm, 0, &a, &b, QuantType::F32);
        let trace_ba = QuantizationErrorTrace::compute(BrickId::RmsNorm, 0, &b, &a, QuantType::F32);

        assert!(
            (trace_ab.cosine_similarity - trace_ba.cosine_similarity).abs() < 1e-6,
            "FALSIFICATION FAILED: cosine(a,b) {} != cosine(b,a) {}",
            trace_ab.cosine_similarity,
            trace_ba.cosine_similarity
        );
    }
}
