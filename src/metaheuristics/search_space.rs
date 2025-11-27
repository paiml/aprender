//! Search space abstraction for metaheuristic optimization.
//!
//! Provides type-safe representations of different optimization domains,
//! eliminating the "Muda" of forcing graph problems into hypercubes.

use serde::{Deserialize, Serialize};

/// Universal search space abstraction.
///
/// Each variant provides the natural representation for its problem class,
/// respecting the mathematical structure of the optimization domain.
///
/// # Design Rationale (Toyota Way Review v1.1)
///
/// The original `Bounds` struct forced all algorithms into a continuous
/// hypercube representation. This created architectural friction when
/// adapting graph-based problems (ACO, Tabu) to vector spaces.
///
/// This enum eliminates that waste by providing:
/// - `Continuous` for DE, PSO, CMA-ES
/// - `Mixed` for GA with integer hyperparameters
/// - `Binary` for feature selection (uses `BitVec` internally)
/// - `Permutation` for TSP, scheduling
/// - `Graph` for ACO, network optimization
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SearchSpace {
    /// Continuous bounded hypercube: x ∈ [lower, upper] ⊂ ℝⁿ
    ///
    /// Used by: DE, PSO, CMA-ES, continuous GA
    ///
    /// # Example
    /// ```
    /// use aprender::metaheuristics::SearchSpace;
    ///
    /// let space = SearchSpace::Continuous {
    ///     dim: 10,
    ///     lower: vec![-5.0; 10],
    ///     upper: vec![5.0; 10],
    /// };
    /// assert_eq!(space.dimension(), 10);
    /// ```
    Continuous {
        /// Dimensionality of the search space
        dim: usize,
        /// Lower bounds for each dimension
        lower: Vec<f64>,
        /// Upper bounds for each dimension
        upper: Vec<f64>,
    },

    /// Mixed continuous/discrete: some dimensions are integers.
    ///
    /// Used by: GA with mixed encoding, Harmony Search
    ///
    /// # Example
    /// ```
    /// use aprender::metaheuristics::SearchSpace;
    ///
    /// // HPO: [learning_rate, batch_size, num_layers]
    /// let space = SearchSpace::Mixed {
    ///     dim: 3,
    ///     lower: vec![1e-5, 16.0, 1.0],
    ///     upper: vec![1e-1, 256.0, 10.0],
    ///     discrete_dims: vec![1, 2], // batch_size and num_layers are integers
    /// };
    /// ```
    Mixed {
        /// Dimensionality of the search space
        dim: usize,
        /// Lower bounds for each dimension
        lower: Vec<f64>,
        /// Upper bounds for each dimension
        upper: Vec<f64>,
        /// Indices of dimensions that must be integers
        discrete_dims: Vec<usize>,
    },

    /// Binary search space: x ∈ {0,1}ⁿ
    ///
    /// Used by: Binary GA, feature selection
    ///
    /// Note: Implementations should use `BitVec` internally for efficiency,
    /// not `Vec<f64>` (Toyota Way: use specific containers).
    ///
    /// # Example
    /// ```
    /// use aprender::metaheuristics::SearchSpace;
    ///
    /// let space = SearchSpace::binary(100); // 100 features
    /// assert_eq!(space.dimension(), 100);
    /// ```
    Binary {
        /// Number of binary variables
        dim: usize,
    },

    /// Permutation space: x ∈ Sₙ (symmetric group)
    ///
    /// Used by: TSP, scheduling, assignment problems
    ///
    /// Solutions are permutations of [0, 1, ..., size-1].
    ///
    /// # Example
    /// ```
    /// use aprender::metaheuristics::SearchSpace;
    ///
    /// let space = SearchSpace::permutation(50); // 50-city TSP
    /// assert_eq!(space.dimension(), 50);
    /// ```
    Permutation {
        /// Number of elements to permute
        size: usize,
    },

    /// Graph-based search space: solutions constructed on G(V,E)
    ///
    /// Used by: ACO, graph-based Tabu Search
    ///
    /// This is for constructive metaheuristics that build solutions
    /// by traversing a graph structure.
    Graph {
        /// Number of nodes in the graph
        num_nodes: usize,
        /// Adjacency list: `adjacency[i]` contains `(neighbor, weight)` pairs
        adjacency: Vec<Vec<(usize, f64)>>,
        /// Optional heuristic matrix η for ACO (e.g., 1/distance)
        heuristic: Option<Vec<Vec<f64>>>,
    },
}

impl SearchSpace {
    /// Create a continuous search space with uniform bounds.
    ///
    /// # Arguments
    /// * `dim` - Dimensionality
    /// * `lower` - Lower bound for all dimensions
    /// * `upper` - Upper bound for all dimensions
    ///
    /// # Example
    /// ```
    /// use aprender::metaheuristics::SearchSpace;
    ///
    /// let space = SearchSpace::continuous(30, -5.0, 5.0);
    /// assert_eq!(space.dimension(), 30);
    /// ```
    #[must_use]
    pub fn continuous(dim: usize, lower: f64, upper: f64) -> Self {
        Self::Continuous {
            dim,
            lower: vec![lower; dim],
            upper: vec![upper; dim],
        }
    }

    /// Create a binary search space.
    ///
    /// # Arguments
    /// * `dim` - Number of binary variables
    #[must_use]
    pub fn binary(dim: usize) -> Self {
        Self::Binary { dim }
    }

    /// Create a permutation search space.
    ///
    /// # Arguments
    /// * `size` - Number of elements to permute
    #[must_use]
    pub fn permutation(size: usize) -> Self {
        Self::Permutation { size }
    }

    /// Create a graph search space from a distance matrix (for TSP).
    ///
    /// # Arguments
    /// * `distance_matrix` - Symmetric distance matrix
    ///
    /// # Example
    /// ```
    /// use aprender::metaheuristics::SearchSpace;
    ///
    /// let distances = vec![
    ///     vec![0.0, 10.0, 15.0],
    ///     vec![10.0, 0.0, 20.0],
    ///     vec![15.0, 20.0, 0.0],
    /// ];
    /// let space = SearchSpace::tsp(&distances);
    /// assert_eq!(space.dimension(), 3);
    /// ```
    #[must_use]
    pub fn tsp(distance_matrix: &[Vec<f64>]) -> Self {
        let n = distance_matrix.len();
        let adjacency = (0..n)
            .map(|i| {
                (0..n)
                    .filter(|&j| j != i)
                    .map(|j| (j, distance_matrix[i][j]))
                    .collect()
            })
            .collect();

        let heuristic = Some(
            distance_matrix
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|&d| if d > 0.0 { 1.0 / d } else { 0.0 })
                        .collect()
                })
                .collect(),
        );

        Self::Graph {
            num_nodes: n,
            adjacency,
            heuristic,
        }
    }

    /// Get the dimensionality of the search space.
    #[must_use]
    pub fn dimension(&self) -> usize {
        match self {
            Self::Continuous { dim, .. } | Self::Mixed { dim, .. } | Self::Binary { dim } => *dim,
            Self::Permutation { size }
            | Self::Graph {
                num_nodes: size, ..
            } => *size,
        }
    }

    /// Check if a continuous point is within bounds.
    ///
    /// Returns `None` for non-continuous spaces.
    #[must_use]
    pub fn contains(&self, point: &[f64]) -> Option<bool> {
        match self {
            Self::Continuous { dim, lower, upper } => {
                if point.len() != *dim {
                    return Some(false);
                }
                Some(
                    point
                        .iter()
                        .zip(lower.iter().zip(upper.iter()))
                        .all(|(&x, (&lo, &hi))| (lo..=hi).contains(&x)),
                )
            }
            Self::Mixed {
                dim, lower, upper, ..
            } => {
                if point.len() != *dim {
                    return Some(false);
                }
                Some(
                    point
                        .iter()
                        .zip(lower.iter().zip(upper.iter()))
                        .all(|(&x, (&lo, &hi))| (lo..=hi).contains(&x)),
                )
            }
            _ => None,
        }
    }

    /// Clip a point to be within bounds (for continuous/mixed spaces).
    ///
    /// Returns `None` for non-continuous spaces.
    #[must_use]
    pub fn clip(&self, point: &[f64]) -> Option<Vec<f64>> {
        match self {
            Self::Continuous { lower, upper, .. } | Self::Mixed { lower, upper, .. } => Some(
                point
                    .iter()
                    .zip(lower.iter().zip(upper.iter()))
                    .map(|(&x, (&lo, &hi))| x.clamp(lo, hi))
                    .collect(),
            ),
            _ => None,
        }
    }

    /// Get the range (upper - lower) for each dimension.
    ///
    /// Returns `None` for non-continuous spaces.
    #[must_use]
    pub fn ranges(&self) -> Option<Vec<f64>> {
        match self {
            Self::Continuous { lower, upper, .. } | Self::Mixed { lower, upper, .. } => Some(
                lower
                    .iter()
                    .zip(upper.iter())
                    .map(|(&lo, &hi)| hi - lo)
                    .collect(),
            ),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_continuous_space_creation() {
        let space = SearchSpace::continuous(10, -5.0, 5.0);
        assert_eq!(space.dimension(), 10);

        if let SearchSpace::Continuous { dim, lower, upper } = space {
            assert_eq!(dim, 10);
            assert_eq!(lower.len(), 10);
            assert_eq!(upper.len(), 10);
            assert!(lower.iter().all(|&x| (x - (-5.0)).abs() < 1e-10));
            assert!(upper.iter().all(|&x| (x - 5.0).abs() < 1e-10));
        } else {
            panic!("Expected Continuous variant");
        }
    }

    #[test]
    fn test_binary_space_creation() {
        let space = SearchSpace::binary(100);
        assert_eq!(space.dimension(), 100);

        if let SearchSpace::Binary { dim } = space {
            assert_eq!(dim, 100);
        } else {
            panic!("Expected Binary variant");
        }
    }

    #[test]
    fn test_permutation_space_creation() {
        let space = SearchSpace::permutation(50);
        assert_eq!(space.dimension(), 50);

        if let SearchSpace::Permutation { size } = space {
            assert_eq!(size, 50);
        } else {
            panic!("Expected Permutation variant");
        }
    }

    #[test]
    fn test_tsp_space_creation() {
        let distances = vec![
            vec![0.0, 10.0, 15.0, 20.0],
            vec![10.0, 0.0, 35.0, 25.0],
            vec![15.0, 35.0, 0.0, 30.0],
            vec![20.0, 25.0, 30.0, 0.0],
        ];
        let space = SearchSpace::tsp(&distances);
        assert_eq!(space.dimension(), 4);

        if let SearchSpace::Graph {
            num_nodes,
            adjacency,
            heuristic,
        } = space
        {
            assert_eq!(num_nodes, 4);
            assert_eq!(adjacency.len(), 4);
            // Node 0 should have edges to 1, 2, 3
            assert_eq!(adjacency[0].len(), 3);
            assert!(heuristic.is_some());
            let h = heuristic.expect("heuristic should exist");
            // Heuristic for distance 10 should be 0.1
            assert!((h[0][1] - 0.1).abs() < 1e-10);
        } else {
            panic!("Expected Graph variant");
        }
    }

    #[test]
    fn test_contains_continuous() {
        let space = SearchSpace::continuous(3, 0.0, 10.0);

        // Inside bounds
        assert_eq!(space.contains(&[5.0, 5.0, 5.0]), Some(true));
        assert_eq!(space.contains(&[0.0, 0.0, 0.0]), Some(true));
        assert_eq!(space.contains(&[10.0, 10.0, 10.0]), Some(true));

        // Outside bounds
        assert_eq!(space.contains(&[-1.0, 5.0, 5.0]), Some(false));
        assert_eq!(space.contains(&[11.0, 5.0, 5.0]), Some(false));

        // Wrong dimension
        assert_eq!(space.contains(&[5.0, 5.0]), Some(false));
    }

    #[test]
    fn test_clip() {
        let space = SearchSpace::continuous(3, 0.0, 10.0);

        let clipped = space.clip(&[-5.0, 5.0, 15.0]).expect("clip should succeed");
        assert!((clipped[0] - 0.0).abs() < 1e-10);
        assert!((clipped[1] - 5.0).abs() < 1e-10);
        assert!((clipped[2] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_ranges() {
        let space = SearchSpace::Continuous {
            dim: 3,
            lower: vec![0.0, -5.0, 10.0],
            upper: vec![10.0, 5.0, 20.0],
        };

        let ranges = space.ranges().expect("ranges should succeed");
        assert!((ranges[0] - 10.0).abs() < 1e-10);
        assert!((ranges[1] - 10.0).abs() < 1e-10);
        assert!((ranges[2] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_binary_contains_returns_none() {
        let space = SearchSpace::binary(10);
        assert_eq!(space.contains(&[0.0; 10]), None);
    }

    #[test]
    fn test_mixed_space() {
        let space = SearchSpace::Mixed {
            dim: 3,
            lower: vec![0.0, 1.0, 1.0],
            upper: vec![1.0, 100.0, 10.0],
            discrete_dims: vec![1, 2],
        };
        assert_eq!(space.dimension(), 3);
        assert_eq!(space.contains(&[0.5, 50.0, 5.0]), Some(true));
    }
}
