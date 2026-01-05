//! Delta computation for comparing pipeline outputs against ground truth.
//!
//! Supports multiple metrics: mean/std difference, cosine similarity,
//! KL divergence, and Wasserstein distance.

use super::GroundTruth;

/// Statistical divergence metrics for comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    /// Mean squared error
    MeanSquaredError,
    /// Cosine similarity between vectors
    CosineSimilarity,
    /// Kullback-Leibler divergence for probability distributions
    KLDivergence,
    /// Wasserstein (Earth Mover's) distance
    WassersteinDistance,
    /// L2 norm of difference
    L2Norm,
}

/// Delta represents the divergence between computed output and ground truth.
#[derive(Debug, Clone)]
#[allow(clippy::struct_field_names)]
pub struct Delta {
    /// Absolute difference in mean
    mean_delta: f32,
    /// Absolute difference in standard deviation
    std_delta: f32,
    /// Percent divergence (combined metric)
    percent: f32,
    /// Whether the sign of the mean is flipped
    sign_flipped: bool,
    /// Cosine similarity (if computed)
    cosine: Option<f32>,
    /// KL divergence (if computed)
    kl_div: Option<f32>,
}

impl Delta {
    /// Compute delta between two ground truth statistics.
    #[must_use] 
    pub fn compute(our: &GroundTruth, gt: &GroundTruth) -> Self {
        let mean_delta = (our.mean() - gt.mean()).abs();
        let std_delta = (our.std() - gt.std()).abs();

        // Use std as reference since mean can be near zero
        let ref_val = gt.std().abs().max(0.001);
        let percent = ((mean_delta + std_delta) / ref_val) * 100.0;

        // Sign flip detection
        let sign_flipped = our.mean().signum() != gt.mean().signum()
            && our.mean().abs() > 0.01
            && gt.mean().abs() > 0.01;

        // Compute cosine similarity if both have data
        let cosine = match (our.data(), gt.data()) {
            (Some(a), Some(b)) if !a.is_empty() && !b.is_empty() => {
                Some(Self::cosine_similarity(a, b))
            }
            _ => None,
        };

        Self {
            mean_delta,
            std_delta,
            percent,
            sign_flipped,
            cosine,
            kl_div: None,
        }
    }

    /// Create a delta from a simple percent value (for testing).
    #[must_use] 
    pub fn from_percent(percent: f32) -> Self {
        Self {
            mean_delta: 0.0,
            std_delta: 0.0,
            percent,
            sign_flipped: false,
            cosine: None,
            kl_div: None,
        }
    }

    /// Create a delta from mean and std differences (for testing).
    #[must_use] 
    pub fn from_stats(mean_delta: f32, std_delta: f32) -> Self {
        let percent = (mean_delta + std_delta) * 100.0;
        Self {
            mean_delta,
            std_delta,
            percent,
            sign_flipped: false,
            cosine: None,
            kl_div: None,
        }
    }

    /// Get the absolute difference in mean.
    #[must_use] 
    pub fn mean_delta(&self) -> f32 {
        self.mean_delta
    }

    /// Get the absolute difference in standard deviation.
    #[must_use] 
    pub fn std_delta(&self) -> f32 {
        self.std_delta
    }

    /// Get the percent divergence.
    #[must_use] 
    pub fn percent(&self) -> f32 {
        self.percent
    }

    /// Check if there's a sign flip between our value and ground truth.
    #[must_use] 
    pub fn is_sign_flipped(&self) -> bool {
        self.sign_flipped
    }

    /// Get cosine similarity if computed.
    #[must_use] 
    pub fn cosine(&self) -> Option<f32> {
        self.cosine
    }

    /// Get KL divergence if computed.
    #[must_use] 
    pub fn kl_divergence_value(&self) -> Option<f32> {
        self.kl_div
    }

    /// Compute cosine similarity between two vectors.
    ///
    /// Returns a value in [-1, 1]:
    /// - 1.0: identical direction
    /// - 0.0: orthogonal
    /// - -1.0: opposite direction
    #[must_use] 
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }

        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a < 1e-10 || norm_b < 1e-10 {
            return 0.0;
        }

        dot / (norm_a * norm_b)
    }

    /// Compute KL divergence D(P || Q).
    ///
    /// P and Q must be probability distributions (sum to 1, non-negative).
    /// Returns bits of information lost when using Q to approximate P.
    #[must_use] 
    pub fn kl_divergence(p: &[f32], q: &[f32]) -> f32 {
        if p.len() != q.len() || p.is_empty() {
            return f32::INFINITY;
        }

        let epsilon = 1e-10;
        let mut kl = 0.0;

        for (pi, qi) in p.iter().zip(q.iter()) {
            if *pi > epsilon {
                let qi_safe = qi.max(epsilon);
                kl += pi * (pi / qi_safe).ln();
            }
        }

        kl
    }

    /// Compute 1-D Wasserstein distance (Earth Mover's Distance).
    ///
    /// For sorted 1-D distributions, this is the sum of absolute differences
    /// of the cumulative distribution functions.
    #[must_use] 
    pub fn wasserstein_1d(a: &[f32], b: &[f32]) -> f32 {
        if a.is_empty() || b.is_empty() {
            return 0.0;
        }

        // Sort both arrays
        let mut a_sorted: Vec<f32> = a.to_vec();
        let mut b_sorted: Vec<f32> = b.to_vec();
        a_sorted.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
        b_sorted.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));

        // Resample to same length if needed (simple linear interpolation)
        let n = a_sorted.len().max(b_sorted.len());
        let a_resampled = Self::resample(&a_sorted, n);
        let b_resampled = Self::resample(&b_sorted, n);

        // Compute EMD as sum of absolute differences
        a_resampled
            .iter()
            .zip(b_resampled.iter())
            .map(|(x, y)| (x - y).abs())
            .sum::<f32>()
            / n as f32
    }

    /// Resample an array to a target length.
    fn resample(data: &[f32], target_len: usize) -> Vec<f32> {
        if data.len() == target_len {
            return data.to_vec();
        }

        let mut result = Vec::with_capacity(target_len);
        for i in 0..target_len {
            let t = i as f32 / (target_len - 1) as f32;
            let idx = t * (data.len() - 1) as f32;
            let low = idx.floor() as usize;
            let high = (low + 1).min(data.len() - 1);
            let frac = idx - low as f32;
            result.push(data[low] * (1.0 - frac) + data[high] * frac);
        }
        result
    }

    /// Compute all metrics and return the delta with full information.
    #[must_use] 
    pub fn compute_all(our: &GroundTruth, gt: &GroundTruth) -> Self {
        let mut delta = Self::compute(our, gt);

        // Compute KL divergence if both have probability-like data
        if let (Some(our_data), Some(gt_data)) = (our.data(), gt.data()) {
            // Normalize to probability distributions
            let our_sum: f32 = our_data.iter().map(|x| x.abs()).sum();
            let gt_sum: f32 = gt_data.iter().map(|x| x.abs()).sum();

            if our_sum > 1e-10 && gt_sum > 1e-10 {
                let p: Vec<f32> = our_data.iter().map(|x| x.abs() / our_sum).collect();
                let q: Vec<f32> = gt_data.iter().map(|x| x.abs() / gt_sum).collect();
                delta.kl_div = Some(Self::kl_divergence(&p, &q));
            }
        }

        delta
    }
}

impl Default for Delta {
    fn default() -> Self {
        Self::from_percent(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let cos = Delta::cosine_similarity(&a, &b);
        assert!((cos - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_opposite() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![-1.0, -2.0, -3.0];
        let cos = Delta::cosine_similarity(&a, &b);
        assert!((cos - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let cos = Delta::cosine_similarity(&a, &b);
        assert!(cos.abs() < 1e-6);
    }

    #[test]
    fn test_kl_identical() {
        let p = vec![0.25, 0.25, 0.25, 0.25];
        let q = vec![0.25, 0.25, 0.25, 0.25];
        let kl = Delta::kl_divergence(&p, &q);
        assert!(kl.abs() < 1e-6);
    }

    #[test]
    fn test_kl_different() {
        let p = vec![0.9, 0.1];
        let q = vec![0.5, 0.5];
        let kl = Delta::kl_divergence(&p, &q);
        assert!(kl > 0.0);
    }

    #[test]
    fn test_wasserstein_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let w = Delta::wasserstein_1d(&a, &b);
        assert!(w < 1e-6);
    }

    #[test]
    fn test_wasserstein_shifted() {
        let a = vec![0.0, 1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        let w = Delta::wasserstein_1d(&a, &b);
        assert!((w - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_delta_compute_with_data() {
        let our = GroundTruth::from_slice(&[1.0, 2.0, 3.0]);
        let gt = GroundTruth::from_slice(&[1.0, 2.0, 3.0]);
        let delta = Delta::compute(&our, &gt);
        assert!(delta.cosine().is_some());
        assert!((delta.cosine().unwrap() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_delta_compute_sign_flip() {
        let our = GroundTruth::from_stats(0.5, 1.0);
        let gt = GroundTruth::from_stats(-0.5, 1.0);
        let delta = Delta::compute(&our, &gt);
        assert!(delta.is_sign_flipped());
    }

    #[test]
    fn test_delta_compute_no_sign_flip() {
        let our = GroundTruth::from_stats(0.5, 1.0);
        let gt = GroundTruth::from_stats(0.6, 1.0);
        let delta = Delta::compute(&our, &gt);
        assert!(!delta.is_sign_flipped());
    }

    #[test]
    fn test_delta_from_percent() {
        let delta = Delta::from_percent(42.5);
        assert!((delta.percent() - 42.5).abs() < 1e-6);
        assert!((delta.mean_delta() - 0.0).abs() < 1e-6);
        assert!((delta.std_delta() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_delta_from_stats() {
        let delta = Delta::from_stats(0.1, 0.2);
        assert!((delta.mean_delta() - 0.1).abs() < 1e-5);
        assert!((delta.std_delta() - 0.2).abs() < 1e-5);
        // percent = (0.1 + 0.2) * 100 = 30
        assert!((delta.percent() - 30.0).abs() < 1e-4);
    }

    #[test]
    fn test_delta_default() {
        let delta = Delta::default();
        assert!((delta.percent() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_delta_kl_divergence_value() {
        let delta = Delta::from_percent(5.0);
        assert!(delta.kl_divergence_value().is_none());
    }

    #[test]
    fn test_delta_compute_all_with_kl() {
        let our = GroundTruth::from_slice(&[0.1, 0.2, 0.3, 0.4]);
        let gt = GroundTruth::from_slice(&[0.1, 0.2, 0.3, 0.4]);
        let delta = Delta::compute_all(&our, &gt);
        assert!(delta.kl_divergence_value().is_some());
        // KL divergence of identical distributions should be ~0
        assert!(delta.kl_divergence_value().unwrap().abs() < 1e-6);
    }

    #[test]
    fn test_cosine_empty() {
        let cos = Delta::cosine_similarity(&[], &[]);
        assert!((cos - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_different_lengths() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        let cos = Delta::cosine_similarity(&a, &b);
        assert!((cos - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_zero_norm() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0];
        let cos = Delta::cosine_similarity(&a, &b);
        assert!((cos - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_kl_empty() {
        let kl = Delta::kl_divergence(&[], &[]);
        assert!(kl.is_infinite());
    }

    #[test]
    fn test_kl_different_lengths() {
        let p = vec![0.5, 0.5];
        let q = vec![0.33, 0.33, 0.34];
        let kl = Delta::kl_divergence(&p, &q);
        assert!(kl.is_infinite());
    }

    #[test]
    fn test_wasserstein_empty() {
        let w = Delta::wasserstein_1d(&[], &[1.0]);
        assert!((w - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_wasserstein_different_lengths() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let w = Delta::wasserstein_1d(&a, &b);
        assert!(w.is_finite());
    }
}
