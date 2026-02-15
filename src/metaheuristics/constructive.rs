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

include!("constructive_part_02.rs");
include!("constructive_part_03.rs");
