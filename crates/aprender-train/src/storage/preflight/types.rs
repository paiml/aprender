//! Core types for preflight validation system.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

/// Preflight validation errors
#[derive(Debug, Error)]
pub enum PreflightError {
    #[error("Data integrity check failed: {0}")]
    DataIntegrity(String),

    #[error("Environment check failed: {0}")]
    Environment(String),

    #[error("Validation failed: {checks_failed} of {total_checks} checks failed")]
    ValidationFailed { checks_failed: usize, total_checks: usize },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Metadata for a preflight check (without the check function)
#[derive(Debug, Clone)]
pub struct CheckMetadata {
    /// Name of the check
    pub name: String,
    /// Type of check
    pub check_type: CheckType,
    /// Description of what this check validates
    pub description: String,
    /// Whether this check is required
    pub required: bool,
}

/// Context for preflight checks
#[derive(Debug, Clone, Default)]
pub struct PreflightContext {
    /// Minimum required samples
    pub min_samples: Option<usize>,
    /// Minimum required features
    pub min_features: Option<usize>,
    /// Minimum required disk space in MB
    pub min_disk_space_mb: Option<u64>,
    /// Minimum required memory in MB
    pub min_memory_mb: Option<u64>,
    /// Expected label range
    pub label_range: Option<(f64, f64)>,
    /// Custom parameters
    pub params: HashMap<String, String>,
}

impl PreflightContext {
    /// Create a new context
    pub fn new() -> Self {
        Self::default()
    }

    /// Set minimum samples
    pub fn with_min_samples(mut self, min: usize) -> Self {
        self.min_samples = Some(min);
        self
    }

    /// Set minimum features
    pub fn with_min_features(mut self, min: usize) -> Self {
        self.min_features = Some(min);
        self
    }

    /// Set minimum disk space
    pub fn with_min_disk_space_mb(mut self, mb: u64) -> Self {
        self.min_disk_space_mb = Some(mb);
        self
    }

    /// Set minimum memory
    pub fn with_min_memory_mb(mut self, mb: u64) -> Self {
        self.min_memory_mb = Some(mb);
        self
    }

    /// Set expected label range
    pub fn with_label_range(mut self, min: f64, max: f64) -> Self {
        self.label_range = Some((min, max));
        self
    }
}

/// Type of preflight check
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CheckType {
    /// Data integrity checks
    DataIntegrity,
    /// Environment checks
    Environment,
    /// Resource availability checks
    Resources,
    /// Configuration validation
    Configuration,
    /// Custom check
    Custom(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // PreflightContext Tests
    // =========================================================================

    #[test]
    fn test_context_default() {
        let ctx = PreflightContext::new();
        assert!(ctx.min_samples.is_none());
        assert!(ctx.min_features.is_none());
    }

    #[test]
    fn test_context_builder() {
        let ctx = PreflightContext::new()
            .with_min_samples(100)
            .with_min_features(10)
            .with_min_disk_space_mb(1024)
            .with_min_memory_mb(512)
            .with_label_range(0.0, 1.0);

        assert_eq!(ctx.min_samples, Some(100));
        assert_eq!(ctx.min_features, Some(10));
        assert_eq!(ctx.min_disk_space_mb, Some(1024));
        assert_eq!(ctx.min_memory_mb, Some(512));
        assert_eq!(ctx.label_range, Some((0.0, 1.0)));
    }
}
