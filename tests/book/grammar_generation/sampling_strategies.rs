//! Sampling strategies book tests
//!
//! Tests validating examples from the sampling strategies chapter.

use verificar::generator::SamplingStrategy;

/// Example from Chapter 2: Basic exhaustive sampling
#[test]
fn test_exhaustive_sampling_example() {
    let strategy = SamplingStrategy::Exhaustive { max_depth: 3 };
    assert!(matches!(
        strategy,
        SamplingStrategy::Exhaustive { max_depth: 3 }
    ));
}

/// Example from Chapter 2: Coverage-guided sampling
#[test]
fn test_coverage_guided_example() {
    // Coverage-guided sampling prioritizes unexplored AST paths
    let strategy = SamplingStrategy::CoverageGuided {
        coverage_map: None,
        max_depth: 3,
        seed: 42,
    };
    assert!(matches!(strategy, SamplingStrategy::CoverageGuided { .. }));
}

/// Example from Chapter 2: Swarm testing
#[test]
fn test_swarm_testing_example() {
    // Swarm testing uses random feature subsets per batch
    let strategy = SamplingStrategy::Swarm {
        features_per_batch: 5,
    };
    assert!(matches!(
        strategy,
        SamplingStrategy::Swarm {
            features_per_batch: 5
        }
    ));
}

/// Example from Chapter 2: Boundary-focused sampling
#[test]
fn test_boundary_sampling_example() {
    // Boundary sampling emphasizes edge values
    let strategy = SamplingStrategy::Boundary {
        boundary_probability: 0.3,
    };
    assert!(matches!(
        strategy,
        SamplingStrategy::Boundary { boundary_probability } if (boundary_probability - 0.3).abs() < f64::EPSILON
    ));
}
