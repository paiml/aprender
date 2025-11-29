//! Hybrid solver combining GA, Tabu Search, and ACO.
//!
//! Toyota Way Principle: *Kaizen* - Combine strengths of multiple algorithms.
//!
//! References:
//! - Burke et al. (2013) "Hyper-heuristics: A Survey"
//! - Talbi (2002) "A Taxonomy of Hybrid Metaheuristics"

use crate::error::TspResult;
use crate::instance::TspInstance;
use crate::solver::{AcoSolver, Budget, GaSolver, TabuSolver, TspSolution, TspSolver};

/// Hybrid solver that combines GA, Tabu Search, and ACO
#[derive(Debug, Clone)]
pub struct HybridSolver {
    /// Fraction of budget for GA exploration (0.0-1.0)
    pub ga_fraction: f64,
    /// Fraction of budget for Tabu refinement (0.0-1.0)
    pub tabu_fraction: f64,
    /// Fraction of budget for ACO intensification (0.0-1.0)
    pub aco_fraction: f64,
    /// GA population size
    pub ga_population: usize,
    /// Number of top GA solutions to refine
    pub refine_top_k: usize,
    /// Random seed
    seed: Option<u64>,
}

impl Default for HybridSolver {
    fn default() -> Self {
        Self {
            ga_fraction: 0.4,
            tabu_fraction: 0.3,
            aco_fraction: 0.3,
            ga_population: 30,
            refine_top_k: 3,
            seed: None,
        }
    }
}

impl HybridSolver {
    /// Create a new hybrid solver
    pub fn new() -> Self {
        Self::default()
    }

    /// Set GA budget fraction
    pub fn with_ga_fraction(mut self, fraction: f64) -> Self {
        self.ga_fraction = fraction.clamp(0.0, 1.0);
        self
    }

    /// Set Tabu budget fraction
    pub fn with_tabu_fraction(mut self, fraction: f64) -> Self {
        self.tabu_fraction = fraction.clamp(0.0, 1.0);
        self
    }

    /// Set ACO budget fraction
    pub fn with_aco_fraction(mut self, fraction: f64) -> Self {
        self.aco_fraction = fraction.clamp(0.0, 1.0);
        self
    }

    /// Set GA population size
    pub fn with_ga_population(mut self, size: usize) -> Self {
        self.ga_population = size;
        self
    }

    /// Set number of top solutions to refine
    pub fn with_refine_top_k(mut self, k: usize) -> Self {
        self.refine_top_k = k;
        self
    }

    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Normalize fractions to sum to 1.0
    fn normalize_fractions(&self) -> (f64, f64, f64) {
        let total = self.ga_fraction + self.tabu_fraction + self.aco_fraction;
        if total <= 0.0 {
            return (0.34, 0.33, 0.33);
        }
        (
            self.ga_fraction / total,
            self.tabu_fraction / total,
            self.aco_fraction / total,
        )
    }
}

impl TspSolver for HybridSolver {
    fn solve(&mut self, instance: &TspInstance, budget: Budget) -> TspResult<TspSolution> {
        let total_iterations = budget.limit();
        let (ga_frac, tabu_frac, aco_frac) = self.normalize_fractions();

        let ga_iters = ((total_iterations as f64 * ga_frac) as usize).max(1);
        let tabu_iters = ((total_iterations as f64 * tabu_frac) as usize).max(1);
        let aco_iters = ((total_iterations as f64 * aco_frac) as usize).max(1);

        let mut total_evaluations = 0;
        let mut history = Vec::new();

        // Phase 1: Global exploration with GA
        let mut ga = GaSolver::new().with_population_size(self.ga_population);
        if let Some(seed) = self.seed {
            ga = ga.with_seed(seed);
        }

        let population = ga.evolve(instance, ga_iters)?;
        total_evaluations += self.ga_population * ga_iters;

        // Get top-k solutions for refinement
        let top_k: Vec<Vec<usize>> = population
            .iter()
            .take(self.refine_top_k)
            .map(|(tour, _)| tour.clone())
            .collect();

        let mut best_tour = top_k[0].clone();
        let mut best_length = instance.tour_length(&best_tour);
        history.push(best_length);

        // Phase 2: Local refinement with Tabu Search
        let iters_per_solution = tabu_iters / self.refine_top_k.max(1);

        for tour in &top_k {
            let mut tabu = TabuSolver::new().with_tenure(15);
            if let Some(seed) = self.seed {
                tabu = tabu.with_seed(seed);
            }

            let refined = tabu.refine(tour.clone(), instance, iters_per_solution)?;
            total_evaluations += refined.evaluations;

            if refined.length < best_length {
                best_length = refined.length;
                best_tour = refined.tour;
            }

            history.push(best_length);
        }

        // Phase 3: Intensification with ACO
        let mut aco = AcoSolver::new().with_num_ants(10);
        if let Some(seed) = self.seed {
            aco = aco.with_seed(seed);
        }

        // Seed pheromone from best tour found
        aco.seed_pheromone_from_tour(&best_tour, instance, 2.0);

        let aco_result = aco.solve(instance, Budget::Iterations(aco_iters))?;
        total_evaluations += aco_result.evaluations;

        if aco_result.length < best_length {
            best_length = aco_result.length;
            best_tour = aco_result.tour;
        }

        for &h in &aco_result.history {
            history.push(h.min(best_length));
        }

        Ok(TspSolution {
            tour: best_tour,
            length: best_length,
            evaluations: total_evaluations,
            history,
        })
    }

    fn name(&self) -> &'static str {
        "Hybrid (GA+Tabu+ACO)"
    }

    fn reset(&mut self) {
        // Hybrid is stateless
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square_instance() -> TspInstance {
        let coords = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        TspInstance::from_coords("square", coords).expect("should create")
    }

    fn pentagon_instance() -> TspInstance {
        let angle_step = 2.0 * std::f64::consts::PI / 5.0;
        let coords: Vec<(f64, f64)> = (0..5)
            .map(|i| {
                let angle = i as f64 * angle_step;
                (angle.cos(), angle.sin())
            })
            .collect();
        TspInstance::from_coords("pentagon", coords).expect("should create")
    }

    fn random_instance(n: usize, seed: u64) -> TspInstance {
        use rand::prelude::*;
        let mut rng = StdRng::seed_from_u64(seed);
        let coords: Vec<(f64, f64)> = (0..n).map(|_| (rng.gen(), rng.gen())).collect();
        TspInstance::from_coords("random", coords).expect("should create")
    }

    #[test]
    fn test_hybrid_default_params() {
        let hybrid = HybridSolver::default();
        assert!((hybrid.ga_fraction - 0.4).abs() < 1e-10);
        assert!((hybrid.tabu_fraction - 0.3).abs() < 1e-10);
        assert!((hybrid.aco_fraction - 0.3).abs() < 1e-10);
        assert_eq!(hybrid.ga_population, 30);
        assert_eq!(hybrid.refine_top_k, 3);
    }

    #[test]
    fn test_hybrid_builder() {
        let hybrid = HybridSolver::new()
            .with_ga_fraction(0.5)
            .with_tabu_fraction(0.25)
            .with_aco_fraction(0.25)
            .with_ga_population(50)
            .with_refine_top_k(5)
            .with_seed(42);

        assert!((hybrid.ga_fraction - 0.5).abs() < 1e-10);
        assert!((hybrid.tabu_fraction - 0.25).abs() < 1e-10);
        assert!((hybrid.aco_fraction - 0.25).abs() < 1e-10);
        assert_eq!(hybrid.ga_population, 50);
        assert_eq!(hybrid.refine_top_k, 5);
        assert_eq!(hybrid.seed, Some(42));
    }

    #[test]
    fn test_hybrid_solves_square() {
        let instance = square_instance();
        let mut solver = HybridSolver::new().with_seed(42).with_ga_population(15);

        let solution = solver
            .solve(&instance, Budget::Iterations(100))
            .expect("should solve");

        // Optimal tour around square is 4.0
        assert!(solution.length <= 5.0, "Length {} > 5.0", solution.length);
        assert_eq!(solution.tour.len(), 4);
        assert!(instance.validate_tour(&solution.tour).is_ok());
    }

    #[test]
    fn test_hybrid_solves_pentagon() {
        let instance = pentagon_instance();
        let mut solver = HybridSolver::new().with_seed(42).with_ga_population(20);

        let solution = solver
            .solve(&instance, Budget::Iterations(100))
            .expect("should solve");

        assert_eq!(solution.tour.len(), 5);
        assert!(instance.validate_tour(&solution.tour).is_ok());
    }

    #[test]
    fn test_hybrid_solves_larger_instance() {
        let instance = random_instance(20, 123);
        let mut solver = HybridSolver::new().with_seed(42).with_ga_population(25);

        let solution = solver
            .solve(&instance, Budget::Iterations(200))
            .expect("should solve");

        assert_eq!(solution.tour.len(), 20);
        assert!(instance.validate_tour(&solution.tour).is_ok());
        assert!(solution.length > 0.0);
    }

    #[test]
    fn test_hybrid_deterministic_with_seed() {
        let instance = pentagon_instance();

        let mut solver1 = HybridSolver::new().with_seed(42).with_ga_population(15);
        let mut solver2 = HybridSolver::new().with_seed(42).with_ga_population(15);

        let solution1 = solver1
            .solve(&instance, Budget::Iterations(50))
            .expect("should solve");
        let solution2 = solver2
            .solve(&instance, Budget::Iterations(50))
            .expect("should solve");

        assert!((solution1.length - solution2.length).abs() < 1e-10);
    }

    #[test]
    fn test_hybrid_tracks_history() {
        let instance = square_instance();
        let mut solver = HybridSolver::new().with_seed(42).with_ga_population(15);

        let solution = solver
            .solve(&instance, Budget::Iterations(50))
            .expect("should solve");

        assert!(!solution.history.is_empty());
    }

    #[test]
    fn test_hybrid_normalize_fractions() {
        let solver = HybridSolver::new()
            .with_ga_fraction(0.6)
            .with_tabu_fraction(0.2)
            .with_aco_fraction(0.2);

        let (ga, tabu, aco) = solver.normalize_fractions();
        assert!((ga + tabu + aco - 1.0).abs() < 1e-10);
        assert!((ga - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_hybrid_normalize_zero_fractions() {
        let solver = HybridSolver::new()
            .with_ga_fraction(0.0)
            .with_tabu_fraction(0.0)
            .with_aco_fraction(0.0);

        let (ga, tabu, aco) = solver.normalize_fractions();
        // Should default to roughly equal
        assert!((ga + tabu + aco - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_hybrid_name() {
        let solver = HybridSolver::new();
        assert_eq!(solver.name(), "Hybrid (GA+Tabu+ACO)");
    }

    #[test]
    fn test_hybrid_counts_evaluations() {
        let instance = square_instance();
        let mut solver = HybridSolver::new().with_seed(42).with_ga_population(10);

        let solution = solver
            .solve(&instance, Budget::Iterations(30))
            .expect("should solve");

        // Should have some evaluations
        assert!(solution.evaluations > 0);
    }

    #[test]
    fn test_hybrid_improves_solution() {
        // Use a slightly larger instance where improvement is more visible
        let instance = random_instance(15, 999);

        let mut solver = HybridSolver::new().with_seed(42).with_ga_population(20);

        let solution = solver
            .solve(&instance, Budget::Iterations(150))
            .expect("should solve");

        // First and last in history - should improve or stay same
        let first = solution.history[0];
        let last = solution.history[solution.history.len() - 1];
        assert!(last <= first);
    }
}
