//! Retraining trigger policies.

/// Retraining trigger policy
#[derive(Clone, Debug, Default)]
pub enum RetrainPolicy {
    /// Retrain if >= N features show drift
    FeatureCount { count: usize },
    /// Retrain if any feature with these names drifts
    CriticalFeature { names: Vec<String> },
    /// Retrain if drift percentage exceeds threshold
    DriftPercentage { threshold: f64 },
    /// Retrain on any critical severity drift
    #[default]
    AnyCritical,
}
