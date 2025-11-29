//! Tabu Search for TSP.
//!
//! Reference: Glover & Laguna (1997) "Tabu Search"

use crate::error::TspResult;
use crate::instance::TspInstance;
use crate::solver::{Budget, TspSolution, TspSolver};
use rand::prelude::*;
use std::collections::HashMap;

/// Tabu Search solver for TSP
#[derive(Debug, Clone)]
pub struct TabuSolver {
    /// Tabu tenure (moves stay forbidden for this many iterations)
    pub tenure: usize,
    /// Maximum neighbors to evaluate per iteration
    pub max_neighbors: usize,
    /// Use aspiration criterion (allow tabu move if it improves best known)
    pub use_aspiration: bool,
    /// Random seed
    seed: Option<u64>,
}

impl Default for TabuSolver {
    fn default() -> Self {
        Self {
            tenure: 20,
            max_neighbors: 100,
            use_aspiration: true,
            seed: None,
        }
    }
}

impl TabuSolver {
    /// Create a new Tabu Search solver
    pub fn new() -> Self {
        Self::default()
    }

    /// Set tabu tenure
    pub fn with_tenure(mut self, tenure: usize) -> Self {
        self.tenure = tenure;
        self
    }

    /// Set maximum neighbors to evaluate
    pub fn with_max_neighbors(mut self, max_neighbors: usize) -> Self {
        self.max_neighbors = max_neighbors;
        self
    }

    /// Enable/disable aspiration criterion
    pub fn with_aspiration(mut self, use_aspiration: bool) -> Self {
        self.use_aspiration = use_aspiration;
        self
    }

    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Generate initial tour using nearest neighbor heuristic
    fn nearest_neighbor_tour(instance: &TspInstance, start: usize) -> Vec<usize> {
        let n = instance.num_cities();
        let mut tour = Vec::with_capacity(n);
        let mut visited = vec![false; n];

        tour.push(start);
        visited[start] = true;

        while tour.len() < n {
            let current = tour[tour.len() - 1];
            let mut best_next = 0;
            let mut best_dist = f64::INFINITY;

            for (j, &is_visited) in visited.iter().enumerate() {
                if !is_visited && instance.distance(current, j) < best_dist {
                    best_dist = instance.distance(current, j);
                    best_next = j;
                }
            }

            tour.push(best_next);
            visited[best_next] = true;
        }

        tour
    }

    /// Calculate delta (change in tour length) for 2-opt move
    fn two_opt_delta(tour: &[usize], instance: &TspInstance, i: usize, j: usize) -> f64 {
        let n = tour.len();

        // Current edges being removed
        let i_next = (i + 1) % n;
        let j_next = (j + 1) % n;

        let d_removed =
            instance.distance(tour[i], tour[i_next]) + instance.distance(tour[j], tour[j_next]);

        // New edges being added
        let d_added =
            instance.distance(tour[i], tour[j]) + instance.distance(tour[i_next], tour[j_next]);

        d_added - d_removed
    }

    /// Apply 2-opt move: reverse segment between i+1 and j (inclusive)
    fn apply_two_opt(tour: &mut [usize], i: usize, j: usize) {
        let i_next = i + 1;
        tour[i_next..=j].reverse();
    }

    /// Find best non-tabu move (or tabu move that satisfies aspiration)
    #[allow(clippy::too_many_arguments)]
    fn find_best_move(
        &self,
        tour: &[usize],
        instance: &TspInstance,
        tabu_list: &HashMap<(usize, usize), usize>,
        iteration: usize,
        best_known: f64,
        current_length: f64,
        rng: &mut StdRng,
    ) -> Option<(usize, usize, f64)> {
        let n = tour.len();
        let mut best_move: Option<(usize, usize, f64)> = None;
        let mut best_delta = f64::INFINITY;

        // Generate candidate moves
        let mut candidates: Vec<(usize, usize)> = Vec::new();
        for i in 0..n - 2 {
            for j in i + 2..n {
                if i == 0 && j == n - 1 {
                    continue; // Skip: would just reverse entire tour
                }
                candidates.push((i, j));
            }
        }

        // Shuffle and limit
        candidates.shuffle(rng);
        candidates.truncate(self.max_neighbors);

        for (i, j) in candidates {
            let delta = Self::two_opt_delta(tour, instance, i, j);
            let new_length = current_length + delta;

            // Check tabu status
            let edge1 = (tour[i].min(tour[j]), tour[i].max(tour[j]));
            let edge2 = (
                tour[(i + 1) % n].min(tour[(j + 1) % n]),
                tour[(i + 1) % n].max(tour[(j + 1) % n]),
            );

            let is_tabu = tabu_list.get(&edge1).is_some_and(|&exp| exp > iteration)
                || tabu_list.get(&edge2).is_some_and(|&exp| exp > iteration);

            // Accept if not tabu, or if aspiration (improves best known)
            let accept = !is_tabu || (self.use_aspiration && new_length < best_known);

            if accept && delta < best_delta {
                best_delta = delta;
                best_move = Some((i, j, new_length));
            }
        }

        best_move
    }

    /// Refine an existing tour (for hybrid use)
    pub fn refine(
        &mut self,
        tour: Vec<usize>,
        instance: &TspInstance,
        iterations: usize,
    ) -> TspResult<TspSolution> {
        let mut current_tour = tour;
        let mut current_length = instance.tour_length(&current_tour);
        let mut best_tour = current_tour.clone();
        let mut best_length = current_length;

        let mut rng = match self.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };

        let mut tabu_list: HashMap<(usize, usize), usize> = HashMap::new();
        let mut history = Vec::with_capacity(iterations);
        let mut evaluations = 0;

        for iteration in 0..iterations {
            if let Some((i, j, new_length)) = self.find_best_move(
                &current_tour,
                instance,
                &tabu_list,
                iteration,
                best_length,
                current_length,
                &mut rng,
            ) {
                // Record edges being broken for tabu
                let n = current_tour.len();
                let edge1 = (
                    current_tour[i].min(current_tour[(i + 1) % n]),
                    current_tour[i].max(current_tour[(i + 1) % n]),
                );
                let edge2 = (
                    current_tour[j].min(current_tour[(j + 1) % n]),
                    current_tour[j].max(current_tour[(j + 1) % n]),
                );

                // Apply move
                Self::apply_two_opt(&mut current_tour, i, j);
                current_length = new_length;
                evaluations += 1;

                // Add to tabu list
                tabu_list.insert(edge1, iteration + self.tenure);
                tabu_list.insert(edge2, iteration + self.tenure);

                // Update best
                if current_length < best_length {
                    best_length = current_length;
                    best_tour.clone_from(&current_tour);
                }
            }

            history.push(best_length);
        }

        Ok(TspSolution {
            tour: best_tour,
            length: best_length,
            evaluations,
            history,
        })
    }
}

impl TspSolver for TabuSolver {
    fn solve(&mut self, instance: &TspInstance, budget: Budget) -> TspResult<TspSolution> {
        let max_iterations = budget.limit();

        let mut rng = match self.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };

        // Start with nearest neighbor tour
        let start_city = rng.gen_range(0..instance.num_cities());
        let initial_tour = Self::nearest_neighbor_tour(instance, start_city);

        self.refine(initial_tour, instance, max_iterations)
    }

    fn name(&self) -> &'static str {
        "Tabu Search"
    }

    fn reset(&mut self) {
        // Tabu Search is stateless between runs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square_instance() -> TspInstance {
        let coords = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        TspInstance::from_coords("square", coords).expect("should create")
    }

    fn triangle_instance() -> TspInstance {
        let coords = vec![(0.0, 0.0), (3.0, 0.0), (3.0, 4.0)];
        TspInstance::from_coords("triangle", coords).expect("should create")
    }

    fn line_instance() -> TspInstance {
        // 5 cities in a line: optimal tour is along the line
        let coords = vec![(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)];
        TspInstance::from_coords("line", coords).expect("should create")
    }

    #[test]
    fn test_tabu_default_params() {
        let tabu = TabuSolver::default();
        assert_eq!(tabu.tenure, 20);
        assert_eq!(tabu.max_neighbors, 100);
        assert!(tabu.use_aspiration);
    }

    #[test]
    fn test_tabu_builder() {
        let tabu = TabuSolver::new()
            .with_tenure(30)
            .with_max_neighbors(50)
            .with_aspiration(false)
            .with_seed(42);

        assert_eq!(tabu.tenure, 30);
        assert_eq!(tabu.max_neighbors, 50);
        assert!(!tabu.use_aspiration);
        assert_eq!(tabu.seed, Some(42));
    }

    #[test]
    fn test_tabu_solves_square() {
        let instance = square_instance();
        let mut solver = TabuSolver::new().with_seed(42);

        let solution = solver
            .solve(&instance, Budget::Iterations(50))
            .expect("should solve");

        // Optimal tour around square is 4.0
        assert!(solution.length <= 5.0, "Length {} > 5.0", solution.length);
        assert_eq!(solution.tour.len(), 4);
        assert!(instance.validate_tour(&solution.tour).is_ok());
    }

    #[test]
    fn test_tabu_solves_triangle() {
        let instance = triangle_instance();
        let mut solver = TabuSolver::new().with_seed(42);

        let solution = solver
            .solve(&instance, Budget::Iterations(50))
            .expect("should solve");

        // Triangle tour: 3 + 4 + 5 = 12
        assert!(solution.length <= 13.0, "Length {} > 13.0", solution.length);
        assert_eq!(solution.tour.len(), 3);
    }

    #[test]
    fn test_tabu_solves_line() {
        let instance = line_instance();
        let mut solver = TabuSolver::new().with_seed(42).with_tenure(10);

        let solution = solver
            .solve(&instance, Budget::Iterations(100))
            .expect("should solve");

        // Optimal tour: 0-1-2-3-4-0 = 4 + 4 = 8
        assert!(solution.length <= 9.0, "Length {} > 9.0", solution.length);
    }

    #[test]
    fn test_tabu_deterministic_with_seed() {
        let instance = square_instance();

        let mut solver1 = TabuSolver::new().with_seed(42);
        let mut solver2 = TabuSolver::new().with_seed(42);

        let solution1 = solver1
            .solve(&instance, Budget::Iterations(20))
            .expect("should solve");
        let solution2 = solver2
            .solve(&instance, Budget::Iterations(20))
            .expect("should solve");

        assert!((solution1.length - solution2.length).abs() < 1e-10);
    }

    #[test]
    fn test_tabu_tracks_history() {
        let instance = square_instance();
        let mut solver = TabuSolver::new().with_seed(42);

        let solution = solver
            .solve(&instance, Budget::Iterations(20))
            .expect("should solve");

        assert_eq!(solution.history.len(), 20);
    }

    #[test]
    fn test_tabu_refine() {
        let instance = square_instance();
        let mut solver = TabuSolver::new().with_seed(42);

        // Start with a suboptimal tour
        let initial_tour = vec![0, 2, 1, 3]; // Crossing tour
        let initial_length = instance.tour_length(&initial_tour);

        let solution = solver
            .refine(initial_tour, &instance, 50)
            .expect("should refine");

        // Should improve
        assert!(solution.length <= initial_length);
    }

    #[test]
    fn test_nearest_neighbor_heuristic() {
        let instance = line_instance();
        let tour = TabuSolver::nearest_neighbor_tour(&instance, 0);

        assert_eq!(tour.len(), 5);
        assert!(instance.validate_tour(&tour).is_ok());
        // Starting from 0, should go to nearest (1), then 2, etc.
        assert_eq!(tour[0], 0);
        assert_eq!(tour[1], 1);
    }

    #[test]
    fn test_two_opt_delta() {
        let instance = square_instance();
        let tour = vec![0, 1, 2, 3];
        let original_length = instance.tour_length(&tour);

        // Delta for a move
        let delta = TabuSolver::two_opt_delta(&tour, &instance, 0, 2);

        // Apply the move and verify
        let mut new_tour = tour.clone();
        TabuSolver::apply_two_opt(&mut new_tour, 0, 2);
        let new_length = instance.tour_length(&new_tour);

        assert!((new_length - original_length - delta).abs() < 1e-10);
    }

    #[test]
    fn test_tabu_name() {
        let solver = TabuSolver::new();
        assert_eq!(solver.name(), "Tabu Search");
    }

    #[test]
    fn test_tabu_aspiration_criterion() {
        let instance = square_instance();

        // Without aspiration
        let mut solver_no_asp = TabuSolver::new().with_aspiration(false).with_seed(42);

        // With aspiration (default)
        let mut solver_asp = TabuSolver::new().with_aspiration(true).with_seed(42);

        let sol1 = solver_no_asp
            .solve(&instance, Budget::Iterations(50))
            .expect("should solve");
        let sol2 = solver_asp
            .solve(&instance, Budget::Iterations(50))
            .expect("should solve");

        // Both should find valid solutions
        assert!(instance.validate_tour(&sol1.tour).is_ok());
        assert!(instance.validate_tour(&sol2.tour).is_ok());
    }
}
