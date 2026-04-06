//! Tail severity and distribution shape classification.

/// Tail latency severity classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TailSeverity {
    /// Excellent: P99/P50 < 2
    Excellent,
    /// Good: P99/P50 < 3
    Good,
    /// Warning: P99/P50 < 5
    Warning,
    /// Critical: P99/P50 >= 5
    Critical,
}

impl TailSeverity {
    /// Classify based on tail ratio
    pub fn from_ratio(ratio: f64) -> Self {
        if ratio < 2.0 {
            Self::Excellent
        } else if ratio < 3.0 {
            Self::Good
        } else if ratio < 5.0 {
            Self::Warning
        } else {
            Self::Critical
        }
    }

    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Excellent => "excellent",
            Self::Good => "good",
            Self::Warning => "warning",
            Self::Critical => "critical",
        }
    }
}

/// Distribution shape classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistributionShape {
    /// Unimodal (single peak)
    Unimodal,
    /// Bimodal (two peaks) - often indicates cache hit/miss
    Bimodal,
    /// Multimodal (multiple peaks)
    Multimodal,
    /// Uniform (flat)
    Uniform,
}

impl DistributionShape {
    /// Classify based on bimodality coefficient and entropy
    pub fn classify(bimodality_coeff: f64, entropy: f64) -> Self {
        if entropy > 0.95 {
            Self::Uniform
        } else if bimodality_coeff > 0.555 {
            Self::Bimodal
        } else if bimodality_coeff > 0.7 {
            Self::Multimodal
        } else {
            Self::Unimodal
        }
    }

    /// Get name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Unimodal => "unimodal",
            Self::Bimodal => "bimodal",
            Self::Multimodal => "multimodal",
            Self::Uniform => "uniform",
        }
    }
}
