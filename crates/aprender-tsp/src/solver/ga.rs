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
#[path = "ga_tests.rs"]
mod tests;
