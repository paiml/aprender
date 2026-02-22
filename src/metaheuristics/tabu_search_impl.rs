
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
            None => Box::new(rand::rng()),
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
