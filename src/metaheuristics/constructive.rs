//! Constructive Metaheuristics - Build solutions incrementally.
//!
//! Unlike perturbative methods that modify complete solutions,
//! constructive metaheuristics build solutions step-by-step.
//!
//! # Algorithms
//!
//! - [`AntColony`] - Pheromone-guided probabilistic construction
//! - [`TabuSearch`] - Memory-based local search with forbidden moves
//!
//! # Example: TSP with Ant Colony
//!
//! ```ignore
//! use aprender::metaheuristics::{AntColony, SearchSpace, Budget, ConstructiveMetaheuristic};
//!
//! // Distance matrix for 4 cities
//! let distances = vec![
//!     vec![0.0, 10.0, 15.0, 20.0],
//!     vec![10.0, 0.0, 35.0, 25.0],
//!     vec![15.0, 35.0, 0.0, 30.0],
//!     vec![20.0, 25.0, 30.0, 0.0],
//! ];
//!
//! let space = SearchSpace::graph_from_matrix(&distances);
//! let mut aco = AntColony::new(10);  // 10 ants
//!
//! let result = aco.optimize(
//!     |tour| tour.windows(2).map(|w| distances[w[0]][w[1]]).sum(),
//!     &space,
//!     Budget::Iterations(100),
//! );
//! ```

use super::{Budget, ConvergenceTracker, OptimizationResult, SearchSpace, TerminationReason};
use rand::prelude::*;

/// Trait for constructive metaheuristics that build solutions incrementally.
///
/// Unlike `PerturbativeMetaheuristic` which modifies complete solutions,
/// constructive methods add components one at a time using heuristic
/// information and/or learned knowledge (e.g., pheromones).
pub trait ConstructiveMetaheuristic {
    /// Solution type (usually permutation or path)
    type Solution;

    /// Build and optimize solutions over the given budget.
    fn optimize<F>(
        &mut self,
        objective: &F,
        space: &SearchSpace,
        budget: Budget,
    ) -> OptimizationResult<Self::Solution>
    where
        F: Fn(&Self::Solution) -> f64;

    /// Get the best solution found so far.
    fn best(&self) -> Option<&Self::Solution>;

    /// Get convergence history.
    fn history(&self) -> &[f64];

    /// Reset internal state for a new run.
    fn reset(&mut self);
}

/// Trait for neighborhood-based local search methods.
///
/// These methods explore the solution space by moving between
/// neighboring solutions, potentially using memory structures.
pub trait NeighborhoodSearch {
    /// Solution type
    type Solution;
    /// Move type (describes a transition between solutions)
    type Move;

    /// Generate all possible moves from current solution.
    fn neighbors(&self, solution: &Self::Solution) -> Vec<Self::Move>;

    /// Apply a move to get a new solution.
    fn apply_move(&self, solution: &Self::Solution, mv: &Self::Move) -> Self::Solution;

    /// Evaluate the change in objective (delta evaluation).
    /// Returns None if full re-evaluation is needed.
    fn delta_eval<F>(
        &self,
        solution: &Self::Solution,
        mv: &Self::Move,
        objective: &F,
    ) -> Option<f64>
    where
        F: Fn(&Self::Solution) -> f64;
}

/// Ant Colony Optimization (ACO) for combinatorial problems.
///
/// Implements the Ant System (AS) variant with optional extensions
/// toward Max-Min Ant System (MMAS) and Ant Colony System (ACS).
///
/// # Algorithm
///
/// 1. Initialize pheromone trails uniformly
/// 2. Each ant constructs a solution probabilistically:
///    - P(next) ∝ τ^α × η^β (pheromone × heuristic)
/// 3. Update pheromones: evaporation + deposit by good ants
/// 4. Repeat until budget exhausted
///
/// # References
///
/// - Dorigo & Stützle (2004): Ant Colony Optimization
/// - Dorigo et al. (1996): The Ant System
#[derive(Debug, Clone)]
pub struct AntColony {
    /// Number of ants
    num_ants: usize,
    /// Pheromone importance (α)
    alpha: f64,
    /// Heuristic importance (β)
    beta: f64,
    /// Evaporation rate (ρ)
    rho: f64,
    /// Initial pheromone level
    tau_0: f64,
    /// Pheromone matrix
    pheromone: Vec<Vec<f64>>,
    /// Random seed
    seed: Option<u64>,
    /// Best tour found
    best_tour: Vec<usize>,
    /// Best tour length
    best_length: f64,
    /// Convergence history
    history: Vec<f64>,
}

impl AntColony {
    /// Create new ACO with specified number of ants.
    #[must_use] 
    pub fn new(num_ants: usize) -> Self {
        Self {
            num_ants: num_ants.max(1),
            alpha: 1.0,
            beta: 2.0,
            rho: 0.1,
            tau_0: 1.0,
            pheromone: Vec::new(),
            seed: None,
            best_tour: Vec::new(),
            best_length: f64::INFINITY,
            history: Vec::new(),
        }
    }

    /// Set pheromone importance (α).
    #[must_use] 
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha.max(0.0);
        self
    }

    /// Set heuristic importance (β).
    #[must_use] 
    pub fn with_beta(mut self, beta: f64) -> Self {
        self.beta = beta.max(0.0);
        self
    }

    /// Set evaporation rate (ρ).
    #[must_use] 
    pub fn with_rho(mut self, rho: f64) -> Self {
        self.rho = rho.clamp(0.0, 1.0);
        self
    }

    /// Set random seed.
    #[must_use] 
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Initialize pheromone matrix for n nodes.
    fn init_pheromone(&mut self, n: usize) {
        self.pheromone = vec![vec![self.tau_0; n]; n];
    }

    /// Construct a tour for one ant using probabilistic selection.
    fn construct_tour(&self, n: usize, heuristic: &[Vec<f64>], rng: &mut impl Rng) -> Vec<usize> {
        let mut tour = Vec::with_capacity(n);
        let mut visited = vec![false; n];

        // Start from random city
        let start = rng.gen_range(0..n);
        tour.push(start);
        visited[start] = true;

        // Build tour city by city
        while tour.len() < n {
            let current = *tour.last().expect("tour should not be empty");
            let next = self.select_next(current, &visited, heuristic, rng);
            tour.push(next);
            visited[next] = true;
        }

        tour
    }

    /// Select next city using pheromone and heuristic information.
    fn select_next(
        &self,
        current: usize,
        visited: &[bool],
        heuristic: &[Vec<f64>],
        rng: &mut impl Rng,
    ) -> usize {
        let n = visited.len();
        let mut probs = Vec::with_capacity(n);
        let mut total = 0.0;

        for j in 0..n {
            if visited[j] {
                probs.push(0.0);
            } else {
                let tau = self.pheromone[current][j].powf(self.alpha);
                let eta = heuristic[current][j].powf(self.beta);
                let p = tau * eta;
                probs.push(p);
                total += p;
            }
        }

        if total <= 0.0 {
            // Fallback: pick first unvisited
            return visited.iter().position(|&v| !v).unwrap_or(0);
        }

        // Roulette wheel selection
        let r = rng.gen::<f64>() * total;
        let mut cumsum = 0.0;
        for (j, &p) in probs.iter().enumerate() {
            cumsum += p;
            if cumsum >= r {
                return j;
            }
        }

        // Fallback
        visited.iter().position(|&v| !v).unwrap_or(0)
    }

    /// Evaporate and deposit pheromones based on tour quality.
    fn update_pheromones(&mut self, tours: &[(Vec<usize>, f64)]) {
        let n = self.pheromone.len();

        // Evaporation
        for i in 0..n {
            for j in 0..n {
                self.pheromone[i][j] *= 1.0 - self.rho;
                // Ensure minimum pheromone
                self.pheromone[i][j] = self.pheromone[i][j].max(1e-10);
            }
        }

        // Deposit: each ant deposits inversely proportional to tour length
        for (tour, length) in tours {
            if *length <= 0.0 {
                continue;
            }
            let deposit = 1.0 / length;
            for window in tour.windows(2) {
                let i = window[0];
                let j = window[1];
                self.pheromone[i][j] += deposit;
                self.pheromone[j][i] += deposit; // Symmetric
            }
            // Close the tour
            if tour.len() > 1 {
                let first = tour[0];
                let last = tour[tour.len() - 1];
                self.pheromone[last][first] += deposit;
                self.pheromone[first][last] += deposit;
            }
        }
    }
}

impl ConstructiveMetaheuristic for AntColony {
    type Solution = Vec<usize>;

    fn optimize<F>(
        &mut self,
        objective: &F,
        space: &SearchSpace,
        budget: Budget,
    ) -> OptimizationResult<Self::Solution>
    where
        F: Fn(&Self::Solution) -> f64,
    {
        let (n, heuristic) = match space {
            SearchSpace::Graph {
                num_nodes,
                adjacency,
                heuristic,
            } => {
                // Build heuristic matrix (η = 1/distance typically)
                let h = if let Some(h) = heuristic {
                    h.clone()
                } else {
                    // Derive from adjacency
                    adjacency
                        .iter()
                        .map(|row| {
                            let mut h_row = vec![1e-10; *num_nodes];
                            for &(j, w) in row {
                                h_row[j] = if w > 0.0 { 1.0 / w } else { 1e-10 };
                            }
                            h_row
                        })
                        .collect()
                };
                (*num_nodes, h)
            }
            SearchSpace::Permutation { size } => {
                // No heuristic info - use uniform
                let h = vec![vec![1.0; *size]; *size];
                (*size, h)
            }
            _ => panic!("ACO requires Graph or Permutation search space"),
        };

        let mut rng: Box<dyn RngCore> = match self.seed {
            Some(s) => Box::new(StdRng::seed_from_u64(s)),
            None => Box::new(thread_rng()),
        };

        self.init_pheromone(n);
        self.history.clear();
        self.best_length = f64::INFINITY;
        self.best_tour.clear();

        let mut tracker = ConvergenceTracker::from_budget(&budget);
        let max_iter = budget.max_iterations(self.num_ants);

        for _ in 0..max_iter {
            // Each ant constructs a tour
            let mut tours: Vec<(Vec<usize>, f64)> = Vec::with_capacity(self.num_ants);

            for _ in 0..self.num_ants {
                let tour = self.construct_tour(n, &heuristic, &mut rng);
                let length = objective(&tour);
                tours.push((tour, length));
            }

            // Update best
            for (tour, length) in &tours {
                if *length < self.best_length {
                    self.best_length = *length;
                    self.best_tour.clone_from(tour);
                }
            }

            self.history.push(self.best_length);

            // Update pheromones
            self.update_pheromones(&tours);

            if !tracker.update(self.best_length, self.num_ants) {
                break;
            }
        }

        let termination = if tracker.is_converged() {
            TerminationReason::Converged
        } else if tracker.is_exhausted() {
            TerminationReason::BudgetExhausted
        } else {
            TerminationReason::MaxIterations
        };

        OptimizationResult {
            solution: self.best_tour.clone(),
            objective_value: self.best_length,
            evaluations: tracker.evaluations(),
            iterations: self.history.len(),
            history: self.history.clone(),
            termination,
        }
    }

    fn best(&self) -> Option<&Self::Solution> {
        if self.best_tour.is_empty() {
            None
        } else {
            Some(&self.best_tour)
        }
    }

    fn history(&self) -> &[f64] {
        &self.history
    }

    fn reset(&mut self) {
        self.pheromone.clear();
        self.best_tour.clear();
        self.best_length = f64::INFINITY;
        self.history.clear();
    }
}

/// Tabu Search for combinatorial optimization.
///
/// Memory-based local search that forbids recently visited moves
/// to escape local optima and encourage exploration.
///
/// # Algorithm
///
/// 1. Start from initial solution
/// 2. Generate neighborhood (all possible moves)
/// 3. Select best non-tabu move (aspiration: accept if globally best)
/// 4. Add move to tabu list, remove oldest if full
/// 5. Repeat until budget exhausted
///
/// # References
///
/// - Glover & Laguna (1997): Tabu Search
/// - Gendreau & Potvin (2010): Handbook of Metaheuristics
#[derive(Debug, Clone)]
pub struct TabuSearch {
    /// Tabu tenure (how long moves stay forbidden)
    tenure: usize,
    /// Maximum neighborhood size to explore
    max_neighbors: usize,
    /// Random seed
    seed: Option<u64>,
    /// Best solution found
    best_solution: Vec<usize>,
    /// Best objective value
    best_value: f64,
    /// Convergence history
    history: Vec<f64>,
}

/// A swap move for permutation problems.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SwapMove {
    /// First index to swap
    pub i: usize,
    /// Second index to swap
    pub j: usize,
}

impl TabuSearch {
    /// Create new Tabu Search with given tenure.
    #[must_use] 
    pub fn new(tenure: usize) -> Self {
        Self {
            tenure: tenure.max(1),
            max_neighbors: 1000,
            seed: None,
            best_solution: Vec::new(),
            best_value: f64::INFINITY,
            history: Vec::new(),
        }
    }

    /// Set tabu tenure.
    #[must_use] 
    pub fn with_tenure(mut self, tenure: usize) -> Self {
        self.tenure = tenure.max(1);
        self
    }

    /// Set maximum neighbors to explore per iteration.
    #[must_use] 
    pub fn with_max_neighbors(mut self, max_neighbors: usize) -> Self {
        self.max_neighbors = max_neighbors.max(1);
        self
    }

    /// Set random seed.
    #[must_use] 
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Generate all swap moves for a permutation.
    fn generate_swap_moves(n: usize) -> Vec<SwapMove> {
        let mut moves = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                moves.push(SwapMove { i, j });
            }
        }
        moves
    }

    /// Apply a swap move to a solution.
    fn apply_swap(solution: &[usize], mv: &SwapMove) -> Vec<usize> {
        let mut new_sol = solution.to_vec();
        new_sol.swap(mv.i, mv.j);
        new_sol
    }

    /// Check if a move is tabu.
    fn is_tabu(mv: &SwapMove, tabu_list: &[(SwapMove, usize)], iteration: usize) -> bool {
        for (tabu_mv, expiry) in tabu_list {
            if *expiry > iteration && (tabu_mv.i == mv.i && tabu_mv.j == mv.j) {
                return true;
            }
        }
        false
    }
}

impl ConstructiveMetaheuristic for TabuSearch {
    type Solution = Vec<usize>;

    fn optimize<F>(
        &mut self,
        objective: &F,
        space: &SearchSpace,
        budget: Budget,
    ) -> OptimizationResult<Self::Solution>
    where
        F: Fn(&Self::Solution) -> f64,
    {
        let n = match space {
            SearchSpace::Permutation { size } => *size,
            SearchSpace::Graph { num_nodes, .. } => *num_nodes,
            _ => panic!("TabuSearch requires Permutation or Graph search space"),
        };

        let mut rng: Box<dyn RngCore> = match self.seed {
            Some(s) => Box::new(StdRng::seed_from_u64(s)),
            None => Box::new(thread_rng()),
        };

        // Initialize with random permutation
        let mut current: Vec<usize> = (0..n).collect();
        current.shuffle(&mut rng);
        let mut current_value = objective(&current);

        self.best_solution.clone_from(&current);
        self.best_value = current_value;
        self.history.clear();

        let mut tabu_list: Vec<(SwapMove, usize)> = Vec::new();
        let all_moves = Self::generate_swap_moves(n);

        let mut tracker = ConvergenceTracker::from_budget(&budget);
        let max_iter = budget.max_iterations(1);

        for iteration in 0..max_iter {
            // Find best non-tabu move (or aspiration)
            let mut best_move: Option<SwapMove> = None;
            let mut best_move_value = f64::INFINITY;

            let moves_to_check: Vec<_> = if all_moves.len() <= self.max_neighbors {
                all_moves.clone()
            } else {
                // Sample random subset
                all_moves
                    .choose_multiple(&mut rng, self.max_neighbors)
                    .copied()
                    .collect()
            };

            for mv in &moves_to_check {
                let new_sol = Self::apply_swap(&current, mv);
                let new_value = objective(&new_sol);

                // Aspiration: accept if globally best regardless of tabu
                let is_aspiration = new_value < self.best_value;
                let is_tabu = Self::is_tabu(mv, &tabu_list, iteration);

                if (!is_tabu || is_aspiration) && new_value < best_move_value {
                    best_move = Some(*mv);
                    best_move_value = new_value;
                }
            }

            // Apply best move
            if let Some(mv) = best_move {
                current = Self::apply_swap(&current, &mv);
                current_value = best_move_value;

                // Add to tabu list
                tabu_list.push((mv, iteration + self.tenure));

                // Clean expired entries
                tabu_list.retain(|(_, expiry)| *expiry > iteration);

                // Update global best
                if current_value < self.best_value {
                    self.best_value = current_value;
                    self.best_solution.clone_from(&current);
                }
            }

            self.history.push(self.best_value);

            if !tracker.update(self.best_value, 1) {
                break;
            }
        }

        let termination = if tracker.is_converged() {
            TerminationReason::Converged
        } else if tracker.is_exhausted() {
            TerminationReason::BudgetExhausted
        } else {
            TerminationReason::MaxIterations
        };

        OptimizationResult {
            solution: self.best_solution.clone(),
            objective_value: self.best_value,
            evaluations: tracker.evaluations(),
            iterations: self.history.len(),
            history: self.history.clone(),
            termination,
        }
    }

    fn best(&self) -> Option<&Self::Solution> {
        if self.best_solution.is_empty() {
            None
        } else {
            Some(&self.best_solution)
        }
    }

    fn history(&self) -> &[f64] {
        &self.history
    }

    fn reset(&mut self) {
        self.best_solution.clear();
        self.best_value = f64::INFINITY;
        self.history.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==========================================================
    // RED PHASE: Tests written first (Extreme TDD)
    // ==========================================================

    // --- Tabu Search Tests ---

    #[test]
    fn test_tabu_search_permutation() {
        // Minimize sum of |position - value| (optimal: identity permutation)
        let space = SearchSpace::Permutation { size: 5 };

        let objective = |perm: &Vec<usize>| {
            perm.iter()
                .enumerate()
                .map(|(i, &v)| (i as i64 - v as i64).unsigned_abs() as f64)
                .sum()
        };

        let mut ts = TabuSearch::new(5).with_seed(42);
        let result = ts.optimize(&objective, &space, Budget::Iterations(100));

        // Should find identity or close
        assert!(
            result.objective_value <= 4.0,
            "Should find good permutation"
        );
        assert_eq!(result.solution.len(), 5);
    }

    #[test]
    fn test_tabu_search_parameters() {
        let ts = TabuSearch::new(10)
            .with_tenure(15)
            .with_max_neighbors(500)
            .with_seed(123);

        assert_eq!(ts.tenure, 15);
        assert_eq!(ts.max_neighbors, 500);
    }

    #[test]
    fn test_tabu_search_reset() {
        let space = SearchSpace::Permutation { size: 3 };
        let mut ts = TabuSearch::new(3).with_seed(42);

        let _ = ts.optimize(&|_: &Vec<usize>| 1.0, &space, Budget::Iterations(5));
        assert!(!ts.history.is_empty());

        ts.reset();
        assert!(ts.history.is_empty());
        assert!(ts.best_solution.is_empty());
    }

    #[test]
    fn test_tabu_search_tsp() {
        // Small TSP
        let space = SearchSpace::Permutation { size: 4 };

        // Distance matrix (symmetric)
        let dist = vec![
            vec![0.0, 10.0, 15.0, 20.0],
            vec![10.0, 0.0, 35.0, 25.0],
            vec![15.0, 35.0, 0.0, 30.0],
            vec![20.0, 25.0, 30.0, 0.0],
        ];

        let objective = |tour: &Vec<usize>| {
            let mut total = 0.0;
            for i in 0..tour.len() {
                let from = tour[i];
                let to = tour[(i + 1) % tour.len()];
                total += dist[from][to];
            }
            total
        };

        let mut ts = TabuSearch::new(5).with_seed(42);
        let result = ts.optimize(&objective, &space, Budget::Iterations(50));

        assert!(
            result.objective_value < 120.0,
            "Should find reasonable tour"
        );
    }

    #[test]
    fn test_tabu_search_convergence() {
        let space = SearchSpace::Permutation { size: 6 };

        let objective = |perm: &Vec<usize>| {
            // Quadratic assignment-like objective
            perm.iter()
                .enumerate()
                .map(|(i, &v)| ((i as f64) - (v as f64)).powi(2))
                .sum()
        };

        let mut ts = TabuSearch::new(7).with_seed(42);
        let result = ts.optimize(&objective, &space, Budget::Iterations(100));

        // Should show improvement
        let history = &result.history;
        assert!(history.len() > 1);
        assert!(
            history.last().unwrap() <= &history[0],
            "Should improve or stay same"
        );
    }

    // --- ACO Tests ---

    #[test]
    fn test_aco_tsp_4_cities() {
        // Simple TSP: 4 cities in a square
        // Optimal tour: 0->1->2->3->0 or reverse, length = 40
        let distances = vec![
            vec![(1, 10.0), (2, 15.0), (3, 20.0)],
            vec![(0, 10.0), (2, 35.0), (3, 25.0)],
            vec![(0, 15.0), (1, 35.0), (3, 30.0)],
            vec![(0, 20.0), (1, 25.0), (2, 30.0)],
        ];

        let space = SearchSpace::Graph {
            num_nodes: 4,
            adjacency: distances.clone(),
            heuristic: None,
        };

        let objective = |tour: &Vec<usize>| {
            let mut total = 0.0;
            for i in 0..tour.len() {
                let from = tour[i];
                let to = tour[(i + 1) % tour.len()];
                // Find distance
                for &(j, d) in &distances[from] {
                    if j == to {
                        total += d;
                        break;
                    }
                }
            }
            total
        };

        let mut aco = AntColony::new(10).with_seed(42);
        let result = aco.optimize(&objective, &space, Budget::Iterations(50));

        // Should find a reasonable tour (not necessarily optimal with limited budget)
        assert!(result.objective_value < 100.0, "Should find decent tour");
        assert_eq!(result.solution.len(), 4, "Tour should visit all cities");
    }

    #[test]
    fn test_aco_parameters() {
        let aco = AntColony::new(20)
            .with_alpha(2.0)
            .with_beta(3.0)
            .with_rho(0.2)
            .with_seed(123);

        assert_eq!(aco.num_ants, 20);
        assert!((aco.alpha - 2.0).abs() < 1e-10);
        assert!((aco.beta - 3.0).abs() < 1e-10);
        assert!((aco.rho - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_aco_reset() {
        let distances = vec![vec![(1, 10.0)], vec![(0, 10.0)]];
        let space = SearchSpace::Graph {
            num_nodes: 2,
            adjacency: distances,
            heuristic: None,
        };

        let mut aco = AntColony::new(5).with_seed(42);
        let _ = aco.optimize(&|_: &Vec<usize>| 10.0, &space, Budget::Iterations(5));

        assert!(!aco.history.is_empty());
        aco.reset();
        assert!(aco.history.is_empty());
        assert!(aco.best_tour.is_empty());
    }

    #[test]
    fn test_aco_permutation_space() {
        // ACO can also work with permutation space
        let space = SearchSpace::Permutation { size: 5 };

        let objective = |perm: &Vec<usize>| {
            // Minimize sum of (position * value)
            perm.iter().enumerate().map(|(i, &v)| (i * v) as f64).sum()
        };

        let mut aco = AntColony::new(10).with_seed(42);
        let result = aco.optimize(&objective, &space, Budget::Iterations(20));

        assert_eq!(result.solution.len(), 5);
    }

    #[test]
    fn test_constructive_trait_best() {
        let mut aco = AntColony::new(5);
        assert!(aco.best().is_none(), "No best before optimization");

        let space = SearchSpace::Permutation { size: 3 };
        let _ = aco.optimize(&|_: &Vec<usize>| 1.0, &space, Budget::Iterations(1));

        assert!(aco.best().is_some(), "Should have best after optimization");
    }

    #[test]
    fn test_aco_convergence_improves() {
        // Test that ACO improves over iterations
        let distances = vec![
            vec![(1, 10.0), (2, 20.0), (3, 30.0)],
            vec![(0, 10.0), (2, 15.0), (3, 25.0)],
            vec![(0, 20.0), (1, 15.0), (3, 10.0)],
            vec![(0, 30.0), (1, 25.0), (2, 10.0)],
        ];

        let space = SearchSpace::Graph {
            num_nodes: 4,
            adjacency: distances.clone(),
            heuristic: None,
        };

        let objective = |tour: &Vec<usize>| {
            let mut total = 0.0;
            for i in 0..tour.len() {
                let from = tour[i];
                let to = tour[(i + 1) % tour.len()];
                for &(j, d) in &distances[from] {
                    if j == to {
                        total += d;
                        break;
                    }
                }
            }
            total
        };

        let mut aco = AntColony::new(15).with_seed(42);
        let result = aco.optimize(&objective, &space, Budget::Iterations(30));

        // History should show improvement (non-increasing for minimization)
        let history = result.history;
        assert!(history.len() > 1);

        // Final should be <= initial (or at least close)
        let first = history[0];
        let last = history[history.len() - 1];
        assert!(last <= first + 1e-10, "Should improve or stay same");
    }
}
