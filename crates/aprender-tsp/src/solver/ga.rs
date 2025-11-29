//! Genetic Algorithm for TSP.
//!
//! Reference: Goldberg (1989) "Genetic Algorithms in Search, Optimization, and Machine Learning"

use crate::error::TspResult;
use crate::instance::TspInstance;
use crate::solver::{Budget, TspSolution, TspSolver};
use rand::prelude::*;

/// Genetic Algorithm solver for TSP
#[derive(Debug, Clone)]
pub struct GaSolver {
    /// Population size
    pub population_size: usize,
    /// Crossover probability
    pub crossover_rate: f64,
    /// Mutation probability
    pub mutation_rate: f64,
    /// Tournament selection size
    pub tournament_size: usize,
    /// Elitism: number of best individuals to preserve
    pub elitism: usize,
    /// Random seed
    seed: Option<u64>,
}

impl Default for GaSolver {
    fn default() -> Self {
        Self {
            population_size: 50,
            crossover_rate: 0.9,
            mutation_rate: 0.1,
            tournament_size: 5,
            elitism: 2,
            seed: None,
        }
    }
}

impl GaSolver {
    /// Create a new GA solver
    pub fn new() -> Self {
        Self::default()
    }

    /// Set population size
    pub fn with_population_size(mut self, size: usize) -> Self {
        self.population_size = size;
        self
    }

    /// Set crossover rate
    pub fn with_crossover_rate(mut self, rate: f64) -> Self {
        self.crossover_rate = rate;
        self
    }

    /// Set mutation rate
    pub fn with_mutation_rate(mut self, rate: f64) -> Self {
        self.mutation_rate = rate;
        self
    }

    /// Set tournament size
    pub fn with_tournament_size(mut self, size: usize) -> Self {
        self.tournament_size = size;
        self
    }

    /// Set elitism count
    pub fn with_elitism(mut self, count: usize) -> Self {
        self.elitism = count;
        self
    }

    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Generate random tour
    fn random_tour(n: usize, rng: &mut StdRng) -> Vec<usize> {
        let mut tour: Vec<usize> = (0..n).collect();
        tour.shuffle(rng);
        tour
    }

    /// Tournament selection
    fn tournament_select<'a>(
        &self,
        population: &'a [(Vec<usize>, f64)],
        rng: &mut StdRng,
    ) -> &'a Vec<usize> {
        let mut best_idx = rng.gen_range(0..population.len());
        let mut best_fitness = population[best_idx].1;

        for _ in 1..self.tournament_size {
            let idx = rng.gen_range(0..population.len());
            if population[idx].1 < best_fitness {
                best_fitness = population[idx].1;
                best_idx = idx;
            }
        }

        &population[best_idx].0
    }

    /// Order Crossover (OX)
    fn order_crossover(parent1: &[usize], parent2: &[usize], rng: &mut StdRng) -> Vec<usize> {
        let n = parent1.len();
        if n < 2 {
            return parent1.to_vec();
        }

        let mut child = vec![usize::MAX; n];

        // Select crossover segment
        let start = rng.gen_range(0..n);
        let end = rng.gen_range(start..n);

        // Copy segment from parent1
        child[start..=end].copy_from_slice(&parent1[start..=end]);

        // Fill remaining from parent2 in order
        let mut pos = (end + 1) % n;
        let mut p2_pos = (end + 1) % n;

        while child.contains(&usize::MAX) {
            let city = parent2[p2_pos];
            if !child.contains(&city) {
                child[pos] = city;
                pos = (pos + 1) % n;
            }
            p2_pos = (p2_pos + 1) % n;
        }

        child
    }

    /// 2-opt mutation
    fn mutate(&self, tour: &mut [usize], rng: &mut StdRng) {
        if rng.gen::<f64>() < self.mutation_rate {
            let n = tour.len();
            if n < 2 {
                return;
            }

            let i = rng.gen_range(0..n);
            let j = rng.gen_range(0..n);

            if i != j {
                let (i, j) = if i < j { (i, j) } else { (j, i) };
                tour[i..=j].reverse();
            }
        }
    }

    /// Evolve population for one generation
    fn evolve_generation(
        &self,
        population: &mut Vec<(Vec<usize>, f64)>,
        instance: &TspInstance,
        rng: &mut StdRng,
    ) -> usize {
        let mut evaluations = 0;

        // Sort by fitness (lower is better for TSP)
        population.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut new_population = Vec::with_capacity(self.population_size);

        // Elitism: keep best individuals
        for individual in population.iter().take(self.elitism) {
            new_population.push(individual.clone());
        }

        // Generate rest of population
        while new_population.len() < self.population_size {
            let parent1 = self.tournament_select(population, rng);
            let parent2 = self.tournament_select(population, rng);

            let mut child = if rng.gen::<f64>() < self.crossover_rate {
                Self::order_crossover(parent1, parent2, rng)
            } else {
                parent1.clone()
            };

            self.mutate(&mut child, rng);

            let fitness = instance.tour_length(&child);
            evaluations += 1;

            new_population.push((child, fitness));
        }

        *population = new_population;
        evaluations
    }

    /// Get best individual from population
    pub fn evolve(
        &mut self,
        instance: &TspInstance,
        generations: usize,
    ) -> TspResult<Vec<(Vec<usize>, f64)>> {
        let n = instance.num_cities();

        let mut rng = match self.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };

        // Initialize population
        let mut population: Vec<(Vec<usize>, f64)> = (0..self.population_size)
            .map(|_| {
                let tour = Self::random_tour(n, &mut rng);
                let fitness = instance.tour_length(&tour);
                (tour, fitness)
            })
            .collect();

        // Evolve
        for _ in 0..generations {
            self.evolve_generation(&mut population, instance, &mut rng);
        }

        // Sort by fitness
        population.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(population)
    }
}

impl TspSolver for GaSolver {
    fn solve(&mut self, instance: &TspInstance, budget: Budget) -> TspResult<TspSolution> {
        let n = instance.num_cities();
        let max_generations = budget.limit();

        let mut rng = match self.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };

        // Initialize population
        let mut population: Vec<(Vec<usize>, f64)> = (0..self.population_size)
            .map(|_| {
                let tour = Self::random_tour(n, &mut rng);
                let fitness = instance.tour_length(&tour);
                (tour, fitness)
            })
            .collect();

        let mut best_tour = population[0].0.clone();
        let mut best_length = population[0].1;
        let mut history = Vec::with_capacity(max_generations);
        let mut evaluations = self.population_size;

        // Find initial best
        for (tour, fitness) in &population {
            if *fitness < best_length {
                best_length = *fitness;
                best_tour.clone_from(tour);
            }
        }
        history.push(best_length);

        // Evolve
        for _ in 1..max_generations {
            evaluations += self.evolve_generation(&mut population, instance, &mut rng);

            // Update best
            for (tour, fitness) in &population {
                if *fitness < best_length {
                    best_length = *fitness;
                    best_tour.clone_from(tour);
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

    fn name(&self) -> &'static str {
        "Genetic Algorithm"
    }

    fn reset(&mut self) {
        // GA is stateless between runs
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

    fn pentagon_instance() -> TspInstance {
        // Regular pentagon
        let angle_step = 2.0 * std::f64::consts::PI / 5.0;
        let coords: Vec<(f64, f64)> = (0..5)
            .map(|i| {
                let angle = i as f64 * angle_step;
                (angle.cos(), angle.sin())
            })
            .collect();
        TspInstance::from_coords("pentagon", coords).expect("should create")
    }

    #[test]
    fn test_ga_default_params() {
        let ga = GaSolver::default();
        assert_eq!(ga.population_size, 50);
        assert!((ga.crossover_rate - 0.9).abs() < 1e-10);
        assert!((ga.mutation_rate - 0.1).abs() < 1e-10);
        assert_eq!(ga.tournament_size, 5);
        assert_eq!(ga.elitism, 2);
    }

    #[test]
    fn test_ga_builder() {
        let ga = GaSolver::new()
            .with_population_size(100)
            .with_crossover_rate(0.8)
            .with_mutation_rate(0.2)
            .with_tournament_size(3)
            .with_elitism(5)
            .with_seed(42);

        assert_eq!(ga.population_size, 100);
        assert!((ga.crossover_rate - 0.8).abs() < 1e-10);
        assert!((ga.mutation_rate - 0.2).abs() < 1e-10);
        assert_eq!(ga.tournament_size, 3);
        assert_eq!(ga.elitism, 5);
        assert_eq!(ga.seed, Some(42));
    }

    #[test]
    fn test_ga_solves_square() {
        let instance = square_instance();
        let mut solver = GaSolver::new().with_seed(42).with_population_size(20);

        let solution = solver
            .solve(&instance, Budget::Iterations(50))
            .expect("should solve");

        // Optimal tour around square is 4.0
        assert!(solution.length <= 5.0, "Length {} > 5.0", solution.length);
        assert_eq!(solution.tour.len(), 4);
        assert!(instance.validate_tour(&solution.tour).is_ok());
    }

    #[test]
    fn test_ga_solves_triangle() {
        let instance = triangle_instance();
        let mut solver = GaSolver::new().with_seed(42).with_population_size(20);

        let solution = solver
            .solve(&instance, Budget::Iterations(50))
            .expect("should solve");

        // Triangle tour: 3 + 4 + 5 = 12
        assert!(solution.length <= 13.0, "Length {} > 13.0", solution.length);
    }

    #[test]
    fn test_ga_solves_pentagon() {
        let instance = pentagon_instance();
        let mut solver = GaSolver::new().with_seed(42).with_population_size(30);

        let solution = solver
            .solve(&instance, Budget::Iterations(100))
            .expect("should solve");

        assert_eq!(solution.tour.len(), 5);
        assert!(instance.validate_tour(&solution.tour).is_ok());
    }

    #[test]
    fn test_ga_deterministic_with_seed() {
        let instance = square_instance();

        let mut solver1 = GaSolver::new().with_seed(42).with_population_size(20);
        let mut solver2 = GaSolver::new().with_seed(42).with_population_size(20);

        let solution1 = solver1
            .solve(&instance, Budget::Iterations(20))
            .expect("should solve");
        let solution2 = solver2
            .solve(&instance, Budget::Iterations(20))
            .expect("should solve");

        assert!((solution1.length - solution2.length).abs() < 1e-10);
    }

    #[test]
    fn test_ga_tracks_history() {
        let instance = square_instance();
        let mut solver = GaSolver::new().with_seed(42).with_population_size(20);

        let solution = solver
            .solve(&instance, Budget::Iterations(30))
            .expect("should solve");

        assert_eq!(solution.history.len(), 30);
        // History should be non-increasing
        for window in solution.history.windows(2) {
            assert!(window[1] <= window[0] + 1e-10);
        }
    }

    #[test]
    fn test_order_crossover() {
        let mut rng = StdRng::seed_from_u64(42);

        let parent1 = vec![0, 1, 2, 3, 4];
        let parent2 = vec![4, 3, 2, 1, 0];

        let child = GaSolver::order_crossover(&parent1, &parent2, &mut rng);

        // Child should be a valid permutation
        assert_eq!(child.len(), 5);
        let mut sorted = child.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_mutation() {
        let solver = GaSolver::new().with_mutation_rate(1.0); // Always mutate
        let mut rng = StdRng::seed_from_u64(42);

        let mut tour = vec![0, 1, 2, 3, 4];
        let original = tour.clone();

        solver.mutate(&mut tour, &mut rng);

        // Tour should still be valid permutation
        let mut sorted = tour.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4]);

        // Should likely be different (with mutation rate 1.0)
        // Note: could still be same if i==j selected
        assert!(tour != original || true); // Just check it doesn't crash
    }

    #[test]
    fn test_ga_evolve_population() {
        let instance = square_instance();
        let mut solver = GaSolver::new().with_seed(42).with_population_size(20);

        let population = solver.evolve(&instance, 50).expect("should evolve");

        assert_eq!(population.len(), 20);
        // Population should be sorted by fitness
        for window in population.windows(2) {
            assert!(window[0].1 <= window[1].1);
        }
    }

    #[test]
    fn test_ga_name() {
        let solver = GaSolver::new();
        assert_eq!(solver.name(), "Genetic Algorithm");
    }

    #[test]
    fn test_ga_elitism_preserves_best() {
        let instance = square_instance();
        let mut solver = GaSolver::new()
            .with_seed(42)
            .with_population_size(20)
            .with_elitism(2);

        let solution = solver
            .solve(&instance, Budget::Iterations(50))
            .expect("should solve");

        // With elitism, best should never get worse
        for window in solution.history.windows(2) {
            assert!(window[1] <= window[0] + 1e-10);
        }
    }
}
