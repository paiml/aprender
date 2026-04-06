//! Drift Detection Module
//!
//! Implements Jidoka (Automation with Human Touch) for detecting when the process
//! is out of control and signals for help (Retraining).
//!
//! Provides statistical tests for detecting data drift and concept drift:
//! - Kolmogorov-Smirnov test (continuous features)
//! - Chi-square test (categorical features)
//! - Population Stability Index (PSI)

mod detector;
mod statistical;
mod types;

#[cfg(test)]
mod tests;

// Re-export all public types
pub use detector::DriftDetector;
pub use types::{
    CategoricalBaseline, DriftCallback, DriftResult, DriftSummary, DriftTest, Severity,
};

// Re-export statistical functions for testing/advanced use
pub use statistical::{bin_counts, chi_square_p_value, erf, ks_p_value};
