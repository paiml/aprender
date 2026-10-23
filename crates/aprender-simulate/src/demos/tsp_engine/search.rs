//! Greedy construction and 2-opt local search for [`TspEngine`] (#6025: split out of
//! `tsp_engine.rs` so each file stays under the pre-commit complexity limit).

use super::TspEngine;

impl TspEngine {
    /// Calculate tour length.
    pub(super) fn calculate_tour_length(&self, tour: &[usize]) -> u32 {
        if tour.is_empty() {
            return 0;
        }
        let mut total = 0u32;
        for i in 0..tour.len() {
            let from = tour[i];
            let to = tour[(i + 1) % tour.len()];
            total += self.distances[from][to];
        }
        total
    }

    /// Generate random usize in range [0, max).
    pub(super) fn gen_range(&mut self, max: usize) -> usize {
        if max == 0 {
            return 0;
        }
        (self.rng.gen_u64() as usize) % max
    }

    /// Construct initial tour using randomized greedy.
    pub(super) fn construct_greedy_tour(&mut self) -> Vec<usize> {
        let mut visited = vec![false; self.n];
        let mut tour = Vec::with_capacity(self.n);

        // Start from random city
        let start = self.gen_range(self.n);
        tour.push(start);
        visited[start] = true;

        while tour.len() < self.n {
            // Safe: tour is non-empty due to initial push above
            let current = tour.last().copied().unwrap_or(0);

            // Build RCL of nearest unvisited cities
            let mut candidates: Vec<(usize, u32)> = (0..self.n)
                .filter(|&c| !visited[c])
                .map(|c| (c, self.distances[current][c]))
                .collect();
            candidates.sort_by_key(|&(_, d)| d);

            // Select from RCL
            let rcl_size = self.rcl_size.min(candidates.len());
            let idx = self.gen_range(rcl_size);
            let next = candidates[idx].0;

            tour.push(next);
            visited[next] = true;
        }

        tour
    }

    /// Whether swapping edges (i,i+1) and (j,j+1) shortens the tour.
    fn two_opt_gains(&self, tour: &[usize], i: usize, j: usize) -> bool {
        let a = tour[i];
        let b = tour[i + 1];
        let c = tour[j];
        let d = tour[(j + 1) % self.n];

        let current = self.distances[a][b] + self.distances[c][d];
        let swapped = self.distances[a][c] + self.distances[b][d];
        swapped < current
    }

    /// One full sweep of 2-opt over all non-adjacent edge pairs.
    fn two_opt_pass(&self, tour: &mut [usize]) -> u64 {
        let mut improvements = 0u64;
        for i in 0..self.n - 1 {
            for j in i + 2..self.n {
                // Skip adjacent edges
                if j == i + 1 || (i == 0 && j == self.n - 1) {
                    continue;
                }
                if self.two_opt_gains(tour, i, j) {
                    // Reverse segment [i+1, j]
                    tour[i + 1..=j].reverse();
                    improvements += 1;
                }
            }
        }
        improvements
    }

    /// Apply 2-opt local search.
    pub(super) fn two_opt(&self, tour: &mut [usize]) -> u64 {
        let mut improvements = 0u64;
        loop {
            let pass = self.two_opt_pass(tour);
            if pass == 0 {
                return improvements;
            }
            improvements += pass;
        }
    }

    /// Perform one restart iteration.
    pub(super) fn do_restart(&mut self) -> bool {
        // Construct new tour
        let mut tour = self.construct_greedy_tour();

        // Apply 2-opt
        let improvements = self.two_opt(&mut tour);
        self.two_opt_count += improvements;

        // Calculate length
        let length = self.calculate_tour_length(&tour);

        // Update current
        self.tour = tour.clone();
        self.tour_length = length;
        self.restart_count += 1;

        // Check if improved best
        let improved = length < self.best_tour_length;
        if improved {
            self.best_tour = tour;
            self.best_tour_length = length;
            self.stagnation_count = 0;
        } else {
            self.stagnation_count += 1;
        }

        improved
    }
}
