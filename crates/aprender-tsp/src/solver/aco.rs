//! Ant Colony Optimization for TSP.
//!
//! Reference: Dorigo & Gambardella (1997) "Ant Colony System"

use crate::error::TspResult;
use crate::instance::TspInstance;
use crate::solver::{Budget, TspSolution, TspSolver};
use rand::prelude::*;

/// Ant Colony Optimization solver for TSP
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
    pub q0: f64,
    /// Initial pheromone level
    pub tau0: f64,
    /// Random seed
    seed: Option<u64>,
    /// Pheromone matrix
    pheromone: Vec<Vec<f64>>,
}

impl Default for AcoSolver {
    fn default() -> Self {
        Self {
            num_ants: 20,
            alpha: 1.0,
            beta: 2.5,
            rho: 0.1,
            q0: 0.9,
            tau0: 1.0,
            seed: None,
            pheromone: Vec::new(),
        }
    }
}

impl AcoSolver {
    /// Create a new ACO solver with default parameters
    pub fn new() -> Self {
        Self::default()
    }

    /// Set number of ants
    pub fn with_num_ants(mut self, num_ants: usize) -> Self {
        self.num_ants = num_ants;
        self
    }

    /// Set pheromone importance (α)
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set heuristic importance (β)
    pub fn with_beta(mut self, beta: f64) -> Self {
        self.beta = beta;
        self
    }

    /// Set evaporation rate (ρ)
    pub fn with_rho(mut self, rho: f64) -> Self {
        self.rho = rho;
        self
    }

    /// Set exploitation probability (q₀)
    pub fn with_q0(mut self, q0: f64) -> Self {
        self.q0 = q0;
        self
    }

    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Initialize pheromone matrix
    fn init_pheromone(&mut self, n: usize) {
        self.pheromone = vec![vec![self.tau0; n]; n];
    }

    /// Construct a tour for one ant using ACS transition rule
    fn construct_tour(&self, instance: &TspInstance, rng: &mut StdRng) -> Vec<usize> {
        let n = instance.num_cities();
        let mut tour = Vec::with_capacity(n);
        let mut visited = vec![false; n];

        // Start from random city
        let start = rng.gen_range(0..n);
        tour.push(start);
        visited[start] = true;

        while tour.len() < n {
            let current = tour[tour.len() - 1];
            let next = self.select_next_city(current, &visited, instance, rng);
            tour.push(next);
            visited[next] = true;
        }

        tour
    }

    /// Select next city using ACS transition rule
    fn select_next_city(
        &self,
        current: usize,
        visited: &[bool],
        instance: &TspInstance,
        rng: &mut StdRng,
    ) -> usize {
        // ACS: exploitation vs exploration
        if rng.gen::<f64>() < self.q0 {
            // Exploitation: choose best (greedy)
            self.argmax_attractiveness(current, visited, instance)
        } else {
            // Exploration: probabilistic selection
            self.roulette_selection(current, visited, instance, rng)
        }
    }

    /// Find city with maximum attractiveness (exploitation)
    fn argmax_attractiveness(
        &self,
        current: usize,
        visited: &[bool],
        instance: &TspInstance,
    ) -> usize {
        let mut best_city = 0;
        let mut best_value = f64::NEG_INFINITY;

        for (j, &is_visited) in visited.iter().enumerate() {
            if is_visited {
                continue;
            }

            let tau = self.pheromone[current][j];
            let eta = 1.0 / instance.distance(current, j).max(1e-10);
            let value = tau.powf(self.alpha) * eta.powf(self.beta);

            if value > best_value {
                best_value = value;
                best_city = j;
            }
        }

        best_city
    }

    /// Probabilistic city selection (exploration)
    fn roulette_selection(
        &self,
        current: usize,
        visited: &[bool],
        instance: &TspInstance,
        rng: &mut StdRng,
    ) -> usize {
        let mut probabilities = Vec::with_capacity(visited.len());
        let mut total = 0.0;

        for (j, &is_visited) in visited.iter().enumerate() {
            if is_visited {
                probabilities.push(0.0);
                continue;
            }

            let tau = self.pheromone[current][j];
            let eta = 1.0 / instance.distance(current, j).max(1e-10);
            let prob = tau.powf(self.alpha) * eta.powf(self.beta);
            probabilities.push(prob);
            total += prob;
        }

        // Normalize and select
        if total <= 0.0 {
            // Fallback: return first unvisited
            return visited.iter().position(|&v| !v).unwrap_or(0);
        }

        let r = rng.gen::<f64>() * total;
        let mut cumsum = 0.0;

        for (j, &prob) in probabilities.iter().enumerate() {
            cumsum += prob;
            if cumsum >= r {
                return j;
            }
        }

        // Fallback
        visited.iter().position(|&v| !v).unwrap_or(0)
    }

    /// Update pheromone using ACS rules
    fn update_pheromone(&mut self, best_tour: &[usize], best_length: f64, instance: &TspInstance) {
        let n = instance.num_cities();

        // Global evaporation
        for i in 0..n {
            for j in 0..n {
                self.pheromone[i][j] *= 1.0 - self.rho;
            }
        }

        // Deposit pheromone on best tour
        let deposit = 1.0 / best_length;
        for window in best_tour.windows(2) {
            let (i, j) = (window[0], window[1]);
            self.pheromone[i][j] += self.rho * deposit;
            self.pheromone[j][i] += self.rho * deposit;
        }
        // Close the tour
        let (last, first) = (best_tour[best_tour.len() - 1], best_tour[0]);
        self.pheromone[last][first] += self.rho * deposit;
        self.pheromone[first][last] += self.rho * deposit;
    }

    /// Local pheromone update (ACS)
    fn local_update(&mut self, i: usize, j: usize) {
        // ACS local update: reduce pheromone to encourage exploration
        self.pheromone[i][j] = (1.0 - self.rho) * self.pheromone[i][j] + self.rho * self.tau0;
        self.pheromone[j][i] = self.pheromone[i][j];
    }

    /// Seed pheromone from an existing tour (for hybrid use)
    pub fn seed_pheromone_from_tour(&mut self, tour: &[usize], instance: &TspInstance, boost: f64) {
        if self.pheromone.is_empty() {
            self.init_pheromone(instance.num_cities());
        }

        for window in tour.windows(2) {
            let (i, j) = (window[0], window[1]);
            self.pheromone[i][j] += boost;
            self.pheromone[j][i] += boost;
        }
        if !tour.is_empty() {
            let (last, first) = (tour[tour.len() - 1], tour[0]);
            self.pheromone[last][first] += boost;
            self.pheromone[first][last] += boost;
        }
    }
}

impl TspSolver for AcoSolver {
    fn solve(&mut self, instance: &TspInstance, budget: Budget) -> TspResult<TspSolution> {
        let n = instance.num_cities();
        let max_iterations = budget.limit();

        // Initialize
        self.init_pheromone(n);
        let mut rng = match self.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };

        let mut best_tour = Vec::new();
        let mut best_length = f64::INFINITY;
        let mut history = Vec::with_capacity(max_iterations);
        let mut evaluations = 0;

        for _ in 0..max_iterations {
            let mut iteration_best_tour = Vec::new();
            let mut iteration_best_length = f64::INFINITY;

            // Construct solutions for all ants
            for _ in 0..self.num_ants {
                let tour = self.construct_tour(instance, &mut rng);
                let length = instance.tour_length(&tour);
                evaluations += 1;

                // Local pheromone update
                for window in tour.windows(2) {
                    self.local_update(window[0], window[1]);
                }

                if length < iteration_best_length {
                    iteration_best_length = length;
                    iteration_best_tour = tour;
                }
            }

            // Update global best
            if iteration_best_length < best_length {
                best_length = iteration_best_length;
                best_tour.clone_from(&iteration_best_tour);
            }

            // Global pheromone update (best-so-far)
            self.update_pheromone(&best_tour, best_length, instance);
            history.push(best_length);
        }

        Ok(TspSolution {
            tour: best_tour,
            length: best_length,
            evaluations,
            history,
        })
    }

    fn name(&self) -> &'static str {
        "Ant Colony Optimization"
    }

    fn reset(&mut self) {
        self.pheromone.clear();
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
            .with_q0(0.8)
            .with_seed(42);

        assert_eq!(aco.num_ants, 50);
        assert!((aco.alpha - 2.0).abs() < 1e-10);
        assert!((aco.beta - 3.0).abs() < 1e-10);
        assert!((aco.rho - 0.2).abs() < 1e-10);
        assert!((aco.q0 - 0.8).abs() < 1e-10);
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
        assert!(!solver.pheromone.is_empty());

        solver.reset();
        assert!(solver.pheromone.is_empty());
    }

    #[test]
    fn test_aco_seed_pheromone() {
        let instance = square_instance();
        let mut solver = AcoSolver::new();

        // Seed with a known good tour
        let good_tour = vec![0, 1, 2, 3];
        solver.seed_pheromone_from_tour(&good_tour, &instance, 5.0);

        // Pheromone should be higher on seeded edges
        assert!(solver.pheromone[0][1] > solver.tau0);
        assert!(solver.pheromone[1][2] > solver.tau0);
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
