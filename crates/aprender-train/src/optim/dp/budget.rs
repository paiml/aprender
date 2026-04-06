//! Privacy budget tracking.

use serde::{Deserialize, Serialize};

/// Privacy budget (epsilon, delta)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PrivacyBudget {
    /// Privacy loss parameter epsilon (smaller = more private)
    pub epsilon: f64,
    /// Probability of privacy breach delta (smaller = more private)
    pub delta: f64,
}

impl PrivacyBudget {
    /// Create a new privacy budget
    pub fn new(epsilon: f64, delta: f64) -> Self {
        Self { epsilon, delta }
    }

    /// Check if the budget allows the given epsilon
    pub fn allows(&self, spent: f64) -> bool {
        spent <= self.epsilon
    }

    /// Get remaining budget
    pub fn remaining(&self, spent: f64) -> f64 {
        (self.epsilon - spent).max(0.0)
    }
}

impl Default for PrivacyBudget {
    fn default() -> Self {
        // Commonly used default: (8.0, 1e-5)
        Self { epsilon: 8.0, delta: 1e-5 }
    }
}
