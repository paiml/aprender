//! Sampling strategies for code generation
//!
//! Different strategies for exploring the space of valid programs.

use super::CoverageMap;

/// Sampling strategy for code generation
///
/// From spec Section 4.2: Different strategies for exploring the program space.
#[derive(Debug, Clone)]
pub enum SamplingStrategy {
    /// Exhaustive enumeration up to depth N
    ///
    /// Systematically enumerate all valid programs up to the specified AST depth.
    /// Best for small grammars or low depths.
    Exhaustive {
        /// Maximum AST depth to enumerate
        max_depth: usize,
    },

    /// Random sampling with grammar weights
    ///
    /// Generate random programs using production rule weights.
    Random {
        /// Random seed for reproducibility
        seed: u64,
        /// Number of samples to generate
        count: usize,
    },

    /// Coverage-guided generation (NAUTILUS-style)
    ///
    /// Prioritize unexplored AST paths based on coverage feedback.
    /// From Aschermann et al. (2019) NAUTILUS.
    CoverageGuided {
        /// Optional initial coverage map to guide generation
        coverage_map: Option<CoverageMap>,
        /// Maximum AST depth for generation
        max_depth: usize,
        /// Random seed for reproducibility
        seed: u64,
    },

    /// Swarm testing (random feature subsets per batch)
    ///
    /// Generate programs using random subsets of language features.
    /// Implements Heijunka (leveling) for balanced coverage.
    Swarm {
        /// Number of features to enable per batch
        features_per_batch: usize,
    },

    /// Boundary-focused sampling
    ///
    /// Emphasize edge values (0, -1, MAX_INT, empty collections, etc.).
    Boundary {
        /// Probability of using a boundary value
        boundary_probability: f64,
    },
}

impl Default for SamplingStrategy {
    fn default() -> Self {
        Self::CoverageGuided {
            coverage_map: None,
            max_depth: 3,
            seed: 42,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_strategy() {
        let strategy = SamplingStrategy::default();
        assert!(matches!(strategy, SamplingStrategy::CoverageGuided { .. }));
    }

    #[test]
    fn test_exhaustive_strategy() {
        let strategy = SamplingStrategy::Exhaustive { max_depth: 5 };
        if let SamplingStrategy::Exhaustive { max_depth } = strategy {
            assert_eq!(max_depth, 5);
        } else {
            panic!("Expected Exhaustive strategy");
        }
    }

    #[test]
    fn test_boundary_strategy() {
        let strategy = SamplingStrategy::Boundary {
            boundary_probability: 0.3,
        };
        if let SamplingStrategy::Boundary {
            boundary_probability,
        } = strategy
        {
            assert!((boundary_probability - 0.3).abs() < f64::EPSILON);
        } else {
            panic!("Expected Boundary strategy");
        }
    }

    #[test]
    fn test_random_strategy() {
        let strategy = SamplingStrategy::Random {
            seed: 42,
            count: 100,
        };
        if let SamplingStrategy::Random { seed, count } = strategy {
            assert_eq!(seed, 42);
            assert_eq!(count, 100);
        } else {
            panic!("Expected Random strategy");
        }
    }

    #[test]
    fn test_coverage_guided_strategy() {
        let strategy = SamplingStrategy::CoverageGuided {
            coverage_map: None,
            max_depth: 5,
            seed: 123,
        };
        if let SamplingStrategy::CoverageGuided {
            max_depth, seed, ..
        } = strategy
        {
            assert_eq!(max_depth, 5);
            assert_eq!(seed, 123);
        } else {
            panic!("Expected CoverageGuided strategy");
        }
    }

    #[test]
    fn test_swarm_strategy() {
        let strategy = SamplingStrategy::Swarm {
            features_per_batch: 10,
        };
        if let SamplingStrategy::Swarm { features_per_batch } = strategy {
            assert_eq!(features_per_batch, 10);
        } else {
            panic!("Expected Swarm strategy");
        }
    }

    #[test]
    fn test_strategy_debug() {
        let strategy = SamplingStrategy::Exhaustive { max_depth: 3 };
        let debug = format!("{:?}", strategy);
        assert!(debug.contains("Exhaustive"));
    }

    #[test]
    fn test_strategy_clone() {
        let strategy = SamplingStrategy::Random {
            seed: 42,
            count: 10,
        };
        let cloned = strategy.clone();
        if let SamplingStrategy::Random { seed, count } = cloned {
            assert_eq!(seed, 42);
            assert_eq!(count, 10);
        } else {
            panic!("Expected Random strategy clone");
        }
    }
}
