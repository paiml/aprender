//! Ant Colony Optimization for TSP - Adapter for core aprender::AntColony.
//!
//! This module wraps the core aprender metaheuristics library
//! to provide a TSP-specific interface.

use crate::error::TspResult;
use crate::instance::TspInstance;
use crate::solver::{Budget, TspSolution, TspSolver};
use aprender::metaheuristics::{
    AntColony, Budget as CoreBudget, ConstructiveMetaheuristic, SearchSpace,
};

/// Ant Colony Optimization solver for TSP.
///
/// This is a thin adapter around `aprender::metaheuristics::AntColony`.
#[derive(Debug, Clone)]
pub struct AcoSolver {
    /// Number of artificial ants
    pub num_ants: usize,
    /// Pheromone importance (α)
    pub alpha: f64,
    /// Heuristic importance (β)
    pub beta: f64,
    /// Evaporation rate (ρ)
    pub rho: f64,
    /// Exploitation probability (q₀) for ACS rule
    /// Note: Not used by core ACO (uses simpler Ant System)
    pub q0: f64,
    /// Random seed
    seed: Option<u64>,
    /// Core ACO instance (lazy initialized)
    inner: Option<AntColony>,
}

impl Default for AcoSolver {
    fn default() -> Self {
        Self {
            num_ants: 20,
            alpha: 1.0,
            beta: 2.5,
            rho: 0.1,
            q0: 0.9,
            seed: None,
            inner: None,
        }
    }
}

impl AcoSolver {
    /// Create a new ACO solver with default parameters
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set number of ants
    #[must_use]
    pub fn with_num_ants(mut self, num_ants: usize) -> Self {
        self.num_ants = num_ants;
        self
    }

    /// Set pheromone importance (α)
    #[must_use]
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set heuristic importance (β)
    #[must_use]
    pub fn with_beta(mut self, beta: f64) -> Self {
        self.beta = beta;
        self
    }

    /// Set evaporation rate (ρ)
    #[must_use]
    pub fn with_rho(mut self, rho: f64) -> Self {
        self.rho = rho;
        self
    }

    /// Set exploitation probability (q₀)
    /// Note: This parameter is stored for API compatibility but not used
    /// by the underlying core ACO (which uses simpler Ant System).
    #[must_use]
    pub fn with_q0(mut self, q0: f64) -> Self {
        self.q0 = q0;
        self
    }

    /// Set random seed
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Build the core AntColony instance
    fn build_inner(&self) -> AntColony {
        let mut aco = AntColony::new(self.num_ants)
            .with_alpha(self.alpha)
            .with_beta(self.beta)
            .with_rho(self.rho);

        if let Some(seed) = self.seed {
            aco = aco.with_seed(seed);
        }

        aco
    }

    /// Convert TSP budget to core budget
    fn to_core_budget(budget: Budget) -> CoreBudget {
        match budget {
            Budget::Iterations(n) => CoreBudget::Iterations(n),
            Budget::Evaluations(n) => CoreBudget::Evaluations(n),
        }
    }
}

impl TspSolver for AcoSolver {
    fn solve(&mut self, instance: &TspInstance, budget: Budget) -> TspResult<TspSolution> {
        // Build search space from TSP distance matrix
        let space = SearchSpace::tsp(&instance.distances);

        // Build core ACO solver
        let mut aco = self.build_inner();

        // Define objective function (tour length)
        let objective = |tour: &Vec<usize>| -> f64 { instance.tour_length(tour) };

        // Run optimization
        let result = aco.optimize(&objective, &space, Self::to_core_budget(budget));

        // Store inner for potential reuse
        self.inner = Some(aco);

        Ok(TspSolution {
            tour: result.solution,
            length: result.objective_value,
            evaluations: result.evaluations,
            history: result.history,
        })
    }

    fn name(&self) -> &'static str {
        "Ant Colony Optimization"
    }

    fn reset(&mut self) {
        self.inner = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square_instance() -> TspInstance {
        // 4 cities forming a unit square
        let coords = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        TspInstance::from_coords("square", coords).expect("should create")
    }

    fn triangle_instance() -> TspInstance {
        // 3-4-5 right triangle
        let coords = vec![(0.0, 0.0), (3.0, 0.0), (3.0, 4.0)];
        TspInstance::from_coords("triangle", coords).expect("should create")
    }

    #[test]
    fn test_aco_default_params() {
        let aco = AcoSolver::default();
        assert_eq!(aco.num_ants, 20);
        assert!((aco.alpha - 1.0).abs() < 1e-10);
        assert!((aco.beta - 2.5).abs() < 1e-10);
        assert!((aco.rho - 0.1).abs() < 1e-10);
        assert!((aco.q0 - 0.9).abs() < 1e-10);
    }

    #[test]
    fn test_aco_builder() {
        let aco = AcoSolver::new()
            .with_num_ants(50)
            .with_alpha(2.0)
            .with_beta(3.0)
            .with_rho(0.2)
            .with_seed(42);

        assert_eq!(aco.num_ants, 50);
        assert!((aco.alpha - 2.0).abs() < 1e-10);
        assert!((aco.beta - 3.0).abs() < 1e-10);
        assert!((aco.rho - 0.2).abs() < 1e-10);
        assert_eq!(aco.seed, Some(42));
    }

    #[test]
    fn test_aco_solves_square() {
        let instance = square_instance();
        let mut solver = AcoSolver::new().with_seed(42).with_num_ants(10);

        let solution = solver
            .solve(&instance, Budget::Iterations(50))
            .expect("should solve");

        // Optimal tour around square is 4.0
        assert!(solution.length <= 5.0, "Length {} > 5.0", solution.length);
        assert_eq!(solution.tour.len(), 4);
        assert!(instance.validate_tour(&solution.tour).is_ok());
    }

    #[test]
    fn test_aco_solves_triangle() {
        let instance = triangle_instance();
        let mut solver = AcoSolver::new().with_seed(42).with_num_ants(10);

        let solution = solver
            .solve(&instance, Budget::Iterations(50))
            .expect("should solve");

        // Triangle tour: 3 + 4 + 5 = 12
        assert!(solution.length <= 13.0, "Length {} > 13.0", solution.length);
        assert_eq!(solution.tour.len(), 3);
    }

    #[test]
    fn test_aco_deterministic_with_seed() {
        let instance = square_instance();

        let mut solver1 = AcoSolver::new().with_seed(42).with_num_ants(5);
        let mut solver2 = AcoSolver::new().with_seed(42).with_num_ants(5);

        let solution1 = solver1
            .solve(&instance, Budget::Iterations(10))
            .expect("should solve");
        let solution2 = solver2
            .solve(&instance, Budget::Iterations(10))
            .expect("should solve");

        assert!((solution1.length - solution2.length).abs() < 1e-10);
        assert_eq!(solution1.tour, solution2.tour);
    }

    #[test]
    fn test_aco_tracks_history() {
        let instance = square_instance();
        let mut solver = AcoSolver::new().with_seed(42).with_num_ants(5);

        let solution = solver
            .solve(&instance, Budget::Iterations(20))
            .expect("should solve");

        assert_eq!(solution.history.len(), 20);
        // History should be non-increasing (best-so-far)
        for window in solution.history.windows(2) {
            assert!(window[1] <= window[0] + 1e-10);
        }
    }

    #[test]
    fn test_aco_counts_evaluations() {
        let instance = square_instance();
        let mut solver = AcoSolver::new().with_seed(42).with_num_ants(10);

        let solution = solver
            .solve(&instance, Budget::Iterations(5))
            .expect("should solve");

        // 5 iterations * 10 ants = 50 evaluations
        assert_eq!(solution.evaluations, 50);
    }

    #[test]
    fn test_aco_reset() {
        let instance = square_instance();
        let mut solver = AcoSolver::new().with_seed(42);

        solver
            .solve(&instance, Budget::Iterations(5))
            .expect("should solve");
        assert!(solver.inner.is_some());

        solver.reset();
        assert!(solver.inner.is_none());
    }

    #[test]
    fn test_aco_name() {
        let solver = AcoSolver::new();
        assert_eq!(solver.name(), "Ant Colony Optimization");
    }

    #[test]
    fn test_aco_improves_over_iterations() {
        let instance = square_instance();
        let mut solver = AcoSolver::new().with_seed(42).with_num_ants(10);

        let solution = solver
            .solve(&instance, Budget::Iterations(100))
            .expect("should solve");

        // Final should be better than or equal to first
        let first = solution.history[0];
        let last = solution.history[solution.history.len() - 1];
        assert!(last <= first);
    }
}
