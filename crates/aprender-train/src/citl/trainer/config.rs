//! CITL configuration types

/// Configuration for the CITL trainer
#[derive(Debug, Clone)]
pub struct CITLConfig {
    /// Maximum number of fix suggestions to return
    pub max_suggestions: usize,
    /// Minimum suspiciousness score to report
    pub min_suspiciousness: f32,
    /// Whether to build dependency graphs
    pub enable_dependency_graph: bool,
}

impl Default for CITLConfig {
    fn default() -> Self {
        Self { max_suggestions: 5, min_suspiciousness: 0.3, enable_dependency_graph: true }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_citl_config_default() {
        let config = CITLConfig::default();
        assert_eq!(config.max_suggestions, 5);
        assert!((config.min_suspiciousness - 0.3).abs() < 0.01);
        assert!(config.enable_dependency_graph);
    }
}
