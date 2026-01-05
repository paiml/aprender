//! Tolerance thresholds for pipeline verification.
//!
//! Defines acceptable divergence levels for stage comparisons.

use super::Delta;

/// Tolerance threshold for pipeline stage verification.
#[derive(Debug, Clone)]
pub enum Tolerance {
    /// Simple percent threshold
    Percent(f32),
    /// Separate thresholds for mean and std
    Stats { mean_delta: f32, std_delta: f32 },
    /// KL divergence threshold
    KLDivergence(f32),
    /// Cosine similarity minimum
    CosineSimilarity(f32),
    /// Custom tolerance with multiple criteria
    Custom {
        percent: Option<f32>,
        mean_delta: Option<f32>,
        std_delta: Option<f32>,
        kl_div: Option<f32>,
        cosine_min: Option<f32>,
    },
}

impl Tolerance {
    /// Create a simple percent tolerance.
    ///
    /// The stage passes if `delta.percent()` <= threshold.
    #[must_use]
    pub fn percent(threshold: f32) -> Self {
        Tolerance::Percent(threshold)
    }

    /// Create tolerance with separate mean and std thresholds.
    #[must_use]
    pub fn stats(mean_delta: f32, std_delta: f32) -> Self {
        Tolerance::Stats {
            mean_delta,
            std_delta,
        }
    }

    /// Create a KL divergence tolerance.
    #[must_use]
    pub fn kl_divergence(threshold: f32) -> Self {
        Tolerance::KLDivergence(threshold)
    }

    /// Create a cosine similarity minimum tolerance.
    ///
    /// The stage passes if cosine similarity >= threshold.
    #[must_use]
    pub fn cosine(min_similarity: f32) -> Self {
        Tolerance::CosineSimilarity(min_similarity)
    }

    /// Create a custom tolerance with multiple criteria.
    #[must_use]
    pub fn custom() -> ToleranceBuilder {
        ToleranceBuilder::new()
    }

    /// Check if a delta satisfies this tolerance.
    #[must_use]
    pub fn is_satisfied(&self, delta: &Delta) -> bool {
        match self {
            Tolerance::Percent(threshold) => delta.percent() <= *threshold,
            Tolerance::Stats {
                mean_delta,
                std_delta,
            } => delta.mean_delta() <= *mean_delta && delta.std_delta() <= *std_delta,
            Tolerance::KLDivergence(threshold) => delta
                .kl_divergence_value()
                .map_or(true, |kl| kl <= *threshold),
            Tolerance::CosineSimilarity(min) => delta.cosine().map_or(true, |c| c >= *min),
            Tolerance::Custom {
                percent,
                mean_delta,
                std_delta,
                kl_div,
                cosine_min,
            } => {
                let mut pass = true;
                if let Some(p) = percent {
                    pass &= delta.percent() <= *p;
                }
                if let Some(m) = mean_delta {
                    pass &= delta.mean_delta() <= *m;
                }
                if let Some(s) = std_delta {
                    pass &= delta.std_delta() <= *s;
                }
                if let Some(kl) = kl_div {
                    pass &= delta.kl_divergence_value().map_or(true, |k| k <= *kl);
                }
                if let Some(cos_min) = cosine_min {
                    pass &= delta.cosine().map_or(true, |c| c >= *cos_min);
                }
                pass
            }
        }
    }

    /// Get a human-readable description of this tolerance.
    #[must_use]
    pub fn description(&self) -> String {
        match self {
            Tolerance::Percent(p) => format!("≤{p:.1}%"),
            Tolerance::Stats {
                mean_delta,
                std_delta,
            } => format!("mean≤{mean_delta:.3}, std≤{std_delta:.3}"),
            Tolerance::KLDivergence(kl) => format!("KL≤{kl:.4}"),
            Tolerance::CosineSimilarity(cos) => format!("cos≥{cos:.3}"),
            Tolerance::Custom { .. } => "custom".to_string(),
        }
    }
}

impl Default for Tolerance {
    /// Default tolerance is 5% divergence.
    fn default() -> Self {
        Tolerance::Percent(5.0)
    }
}

/// Builder for custom tolerances.
#[derive(Debug)]
pub struct ToleranceBuilder {
    percent: Option<f32>,
    mean_delta: Option<f32>,
    std_delta: Option<f32>,
    kl_div: Option<f32>,
    cosine_min: Option<f32>,
}

impl ToleranceBuilder {
    fn new() -> Self {
        Self {
            percent: None,
            mean_delta: None,
            std_delta: None,
            kl_div: None,
            cosine_min: None,
        }
    }

    /// Set percent threshold.
    pub fn percent(mut self, threshold: f32) -> Self {
        self.percent = Some(threshold);
        self
    }

    /// Set mean delta threshold.
    pub fn mean_delta(mut self, threshold: f32) -> Self {
        self.mean_delta = Some(threshold);
        self
    }

    /// Set std delta threshold.
    pub fn std_delta(mut self, threshold: f32) -> Self {
        self.std_delta = Some(threshold);
        self
    }

    /// Set KL divergence threshold.
    pub fn kl_divergence(mut self, threshold: f32) -> Self {
        self.kl_div = Some(threshold);
        self
    }

    /// Set minimum cosine similarity.
    pub fn cosine_min(mut self, min: f32) -> Self {
        self.cosine_min = Some(min);
        self
    }

    /// Build the tolerance.
    pub fn build(self) -> Tolerance {
        Tolerance::Custom {
            percent: self.percent,
            mean_delta: self.mean_delta,
            std_delta: self.std_delta,
            kl_div: self.kl_div,
            cosine_min: self.cosine_min,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::verify::GroundTruth;

    #[test]
    fn test_percent_pass() {
        let tol = Tolerance::percent(5.0);
        let delta = Delta::from_percent(3.0);
        assert!(tol.is_satisfied(&delta));
    }

    #[test]
    fn test_percent_fail() {
        let tol = Tolerance::percent(5.0);
        let delta = Delta::from_percent(10.0);
        assert!(!tol.is_satisfied(&delta));
    }

    #[test]
    fn test_stats_pass() {
        let tol = Tolerance::stats(0.1, 0.1);
        let delta = Delta::from_stats(0.05, 0.05);
        assert!(tol.is_satisfied(&delta));
    }

    #[test]
    fn test_stats_fail_mean() {
        let tol = Tolerance::stats(0.1, 0.1);
        let delta = Delta::from_stats(0.2, 0.05);
        assert!(!tol.is_satisfied(&delta));
    }

    #[test]
    fn test_default_is_five_percent() {
        let tol = Tolerance::default();
        assert!(tol.is_satisfied(&Delta::from_percent(4.9)));
        assert!(!tol.is_satisfied(&Delta::from_percent(5.1)));
    }

    #[test]
    fn test_custom_builder() {
        let tol = Tolerance::custom().percent(50.0).mean_delta(0.5).build();
        // Delta with mean_delta=0.3 passes mean check (0.3 <= 0.5)
        // percent = 30% passes percent check (30 <= 50)
        let delta = Delta::from_stats(0.3, 0.0);
        assert!(tol.is_satisfied(&delta));
    }

    #[test]
    fn test_kl_divergence_tolerance() {
        let tol = Tolerance::kl_divergence(0.1);
        let delta = Delta::from_percent(50.0); // No KL value, passes vacuously
        assert!(tol.is_satisfied(&delta));
    }

    #[test]
    fn test_cosine_similarity_tolerance() {
        let tol = Tolerance::cosine(0.95);
        let delta = Delta::from_percent(5.0); // No cosine value, passes vacuously
        assert!(tol.is_satisfied(&delta));
    }

    #[test]
    fn test_cosine_tolerance_with_data() {
        let tol = Tolerance::cosine(0.99);
        let our = GroundTruth::from_slice(&[1.0, 2.0, 3.0]);
        let gt = GroundTruth::from_slice(&[1.0, 2.0, 3.0]);
        let delta = Delta::compute(&our, &gt);
        assert!(tol.is_satisfied(&delta));
    }

    #[test]
    fn test_cosine_tolerance_fail() {
        let tol = Tolerance::cosine(0.99);
        let our = GroundTruth::from_slice(&[1.0, 0.0, 0.0]);
        let gt = GroundTruth::from_slice(&[0.0, 1.0, 0.0]);
        let delta = Delta::compute(&our, &gt);
        assert!(!tol.is_satisfied(&delta));
    }

    #[test]
    fn test_stats_fail_std() {
        let tol = Tolerance::stats(0.1, 0.1);
        let delta = Delta::from_stats(0.05, 0.2);
        assert!(!tol.is_satisfied(&delta));
    }

    #[test]
    fn test_custom_with_std_delta() {
        let tol = Tolerance::custom().std_delta(0.1).build();
        let delta = Delta::from_stats(0.5, 0.05);
        assert!(tol.is_satisfied(&delta));
    }

    #[test]
    fn test_custom_with_kl_divergence() {
        let tol = Tolerance::custom().kl_divergence(1.0).build();
        let delta = Delta::from_percent(50.0);
        assert!(tol.is_satisfied(&delta));
    }

    #[test]
    fn test_custom_with_cosine_min() {
        let tol = Tolerance::custom().cosine_min(0.9).build();
        let delta = Delta::from_percent(5.0);
        assert!(tol.is_satisfied(&delta));
    }

    #[test]
    fn test_custom_all_criteria() {
        let tol = Tolerance::custom()
            .percent(10.0)
            .mean_delta(0.1)
            .std_delta(0.1)
            .kl_divergence(1.0)
            .cosine_min(0.9)
            .build();
        let delta = Delta::from_stats(0.05, 0.05);
        assert!(tol.is_satisfied(&delta));
    }

    #[test]
    fn test_custom_fail_percent() {
        let tol = Tolerance::custom().percent(5.0).build();
        let delta = Delta::from_percent(10.0);
        assert!(!tol.is_satisfied(&delta));
    }

    #[test]
    fn test_description_percent() {
        let tol = Tolerance::percent(5.0);
        let desc = tol.description();
        assert!(desc.contains("5.0%"));
    }

    #[test]
    fn test_description_stats() {
        let tol = Tolerance::stats(0.1, 0.2);
        let desc = tol.description();
        assert!(desc.contains("mean"));
        assert!(desc.contains("std"));
    }

    #[test]
    fn test_description_kl() {
        let tol = Tolerance::kl_divergence(0.5);
        let desc = tol.description();
        assert!(desc.contains("KL"));
    }

    #[test]
    fn test_description_cosine() {
        let tol = Tolerance::cosine(0.95);
        let desc = tol.description();
        assert!(desc.contains("cos"));
    }

    #[test]
    fn test_description_custom() {
        let tol = Tolerance::custom().percent(5.0).build();
        let desc = tol.description();
        assert_eq!(desc, "custom");
    }
}
