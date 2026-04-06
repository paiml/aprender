//! Pareto analysis for failure diagnostics.

use std::collections::HashMap;

use super::types::{FailureCategory, FailureContext};

/// Pareto analysis result for failure categories
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParetoAnalysis {
    /// Category counts sorted by frequency (descending)
    pub categories: Vec<(FailureCategory, u32)>,

    /// Total failures analyzed
    pub total_failures: u32,
}

impl ParetoAnalysis {
    /// Perform Pareto analysis on a list of failure contexts
    pub fn from_failures(failures: &[FailureContext]) -> Self {
        let mut counts: HashMap<FailureCategory, u32> = HashMap::new();

        for failure in failures {
            *counts.entry(failure.category).or_insert(0) += 1;
        }

        let mut categories: Vec<(FailureCategory, u32)> = counts.into_iter().collect();
        categories.sort_by(|a, b| b.1.cmp(&a.1)); // Sort descending by count

        Self { categories, total_failures: failures.len() as u32 }
    }

    /// Get the top N failure categories
    pub fn top_categories(&self, n: usize) -> Vec<(FailureCategory, u32)> {
        self.categories.iter().take(n).copied().collect()
    }

    /// Get percentage of failures for each category
    pub fn percentages(&self) -> Vec<(FailureCategory, f64)> {
        if self.total_failures == 0 {
            return Vec::new();
        }

        let total = f64::from(self.total_failures);
        self.categories
            .iter()
            .map(|(cat, count)| (*cat, (f64::from(*count) / total) * 100.0))
            .collect()
    }

    /// Get cumulative percentages (for Pareto chart)
    pub fn cumulative_percentages(&self) -> Vec<(FailureCategory, f64)> {
        if self.total_failures == 0 {
            return Vec::new();
        }

        let total = f64::from(self.total_failures);
        let mut cumulative = 0.0;
        self.categories
            .iter()
            .map(|(cat, count)| {
                cumulative += (f64::from(*count) / total) * 100.0;
                (*cat, cumulative)
            })
            .collect()
    }

    /// Find categories that account for ~80% of failures (Pareto principle)
    pub fn vital_few(&self) -> Vec<(FailureCategory, u32)> {
        let mut cumulative = 0u32;
        let threshold = (f64::from(self.total_failures) * 0.8) as u32;

        self.categories
            .iter()
            .take_while(|(_, count)| {
                let result = cumulative < threshold;
                cumulative += count;
                result
            })
            .copied()
            .collect()
    }
}

/// Convenience function for Pareto analysis
pub fn top_failure_categories(failures: &[FailureContext]) -> Vec<(FailureCategory, u32)> {
    ParetoAnalysis::from_failures(failures).categories
}
