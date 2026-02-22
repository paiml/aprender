//! Particle Swarm Optimization (PSO).
//!
//! A swarm intelligence algorithm inspired by bird flocking behavior.
//!
//! # Algorithm
//!
//! Each particle has position and velocity, updated by:
//! ```text
//! v = w·v + c₁·r₁·(pbest - x) + c₂·r₂·(gbest - x)
//! x = x + v
//! ```
//!
//! # References
//!
//! - Kennedy & Eberhart (1995): "Particle Swarm Optimization"
//! - Clerc & Kennedy (2002): "Constriction coefficient"

use rand::prelude::*;
use serde::{Deserialize, Serialize};

use super::{Budget, OptimizationResult, PerturbativeMetaheuristic, SearchSpace};
use crate::metaheuristics::budget::ConvergenceTracker;
use crate::metaheuristics::traits::TerminationReason;

/// Particle Swarm Optimization optimizer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParticleSwarm {
    /// Swarm size (default: 40)
    pub swarm_size: usize,
    /// Inertia weight w (default: 0.729)
    pub inertia: f64,
    /// Cognitive coefficient c₁ (default: 1.49445)
    pub cognitive: f64,
    /// Social coefficient c₂ (default: 1.49445)
    pub social: f64,
    /// Maximum velocity (fraction of range, default: 0.5)
    pub v_max_fraction: f64,
    /// Random seed
    #[serde(default)]
    seed: Option<u64>,

    // Internal state
    #[serde(skip)]
    positions: Vec<Vec<f64>>,
    #[serde(skip)]
    velocities: Vec<Vec<f64>>,
    #[serde(skip)]
    pbest_pos: Vec<Vec<f64>>,
    #[serde(skip)]
    pbest_val: Vec<f64>,
    #[serde(skip)]
    gbest_pos: Vec<f64>,
    #[serde(skip)]
    gbest_val: f64,
    #[serde(skip)]
    history: Vec<f64>,
}

impl Default for ParticleSwarm {
    fn default() -> Self {
        Self {
            swarm_size: 40,
            inertia: 0.729,     // Constriction coefficient
            cognitive: 1.49445, // c₁
            social: 1.49445,    // c₂
            v_max_fraction: 0.5,
            seed: None,
            positions: Vec::new(),
            velocities: Vec::new(),
            pbest_pos: Vec::new(),
            pbest_val: Vec::new(),
            gbest_pos: Vec::new(),
            gbest_val: f64::INFINITY,
            history: Vec::new(),
        }
    }
}

impl ParticleSwarm {
    /// Create with custom swarm size.
    #[must_use]
    pub fn with_swarm_size(mut self, size: usize) -> Self {
        self.swarm_size = size;
        self
    }

    /// Set inertia weight.
    #[must_use]
    pub fn with_inertia(mut self, w: f64) -> Self {
        self.inertia = w;
        self
    }

    /// Set random seed.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    fn initialize<F>(&mut self, objective: &F, space: &SearchSpace, rng: &mut impl Rng)
    where
        F: Fn(&[f64]) -> f64,
    {
        let (lower, upper, dim) = match space {
            SearchSpace::Continuous { dim, lower, upper } => (lower.clone(), upper.clone(), *dim),
            _ => panic!("PSO requires continuous search space"),
        };

        self.positions = Vec::with_capacity(self.swarm_size);
        self.velocities = Vec::with_capacity(self.swarm_size);
        self.pbest_pos = Vec::with_capacity(self.swarm_size);
        self.pbest_val = vec![f64::INFINITY; self.swarm_size];
        self.gbest_val = f64::INFINITY;

        let v_max: Vec<f64> = lower
            .iter()
            .zip(upper.iter())
            .map(|(l, u)| (u - l) * self.v_max_fraction)
            .collect();

        for i in 0..self.swarm_size {
            // Random position
            let pos: Vec<f64> = (0..dim)
                .map(|j| rng.random_range(lower[j]..=upper[j]))
                .collect();

            // Random velocity
            let vel: Vec<f64> = (0..dim)
                .map(|j| rng.random_range(-v_max[j]..=v_max[j]))
                .collect();

            let fit = objective(&pos);

            self.pbest_pos.push(pos.clone());
            self.pbest_val[i] = fit;

            if fit < self.gbest_val {
                self.gbest_val = fit;
                self.gbest_pos.clone_from(&pos);
            }

            self.positions.push(pos);
            self.velocities.push(vel);
        }

        self.history.clear();
        self.history.push(self.gbest_val);
    }
}

impl PerturbativeMetaheuristic for ParticleSwarm {
    type Solution = Vec<f64>;

    // Contract: metaheuristics-v1, equation = "pso_velocity"
    fn optimize<F>(
        &mut self,
        objective: &F,
        space: &SearchSpace,
        budget: Budget,
    ) -> OptimizationResult<Self::Solution>
    where
        F: Fn(&[f64]) -> f64,
    {
        let mut rng: Box<dyn RngCore> = match self.seed {
            Some(s) => Box::new(StdRng::seed_from_u64(s)),
            None => Box::new(rand::rng()),
        };

        self.initialize(objective, space, &mut rng);

        let (lower, upper, dim) = match space {
            SearchSpace::Continuous { dim, lower, upper } => (lower.clone(), upper.clone(), *dim),
            _ => panic!("PSO requires continuous search space"),
        };

        let v_max: Vec<f64> = lower
            .iter()
            .zip(upper.iter())
            .map(|(l, u)| (u - l) * self.v_max_fraction)
            .collect();

        let mut tracker = ConvergenceTracker::from_budget(&budget);
        tracker.update(self.gbest_val, self.swarm_size);

        let max_iter = budget.max_iterations(self.swarm_size);

        for _iter in 0..max_iter {
            let mut iter_evals = 0;

            for i in 0..self.swarm_size {
                // Update velocity - multiple arrays indexed by j
                #[allow(clippy::needless_range_loop)]
                for j in 0..dim {
                    let r1: f64 = rng.random();
                    let r2: f64 = rng.random();

                    self.velocities[i][j] = self.inertia * self.velocities[i][j]
                        + self.cognitive * r1 * (self.pbest_pos[i][j] - self.positions[i][j])
                        + self.social * r2 * (self.gbest_pos[j] - self.positions[i][j]);

                    // Clamp velocity
                    self.velocities[i][j] = self.velocities[i][j].clamp(-v_max[j], v_max[j]);
                }

                // Update position
                for j in 0..dim {
                    self.positions[i][j] += self.velocities[i][j];
                    self.positions[i][j] = self.positions[i][j].clamp(lower[j], upper[j]);
                }

                // Evaluate
                let fit = objective(&self.positions[i]);
                iter_evals += 1;

                // Update personal best
                if fit < self.pbest_val[i] {
                    self.pbest_val[i] = fit;
                    self.pbest_pos[i].clone_from(&self.positions[i]);

                    // Update global best
                    if fit < self.gbest_val {
                        self.gbest_val = fit;
                        self.gbest_pos.clone_from(&self.positions[i]);
                    }
                }
            }

            self.history.push(self.gbest_val);

            if !tracker.update(self.gbest_val, iter_evals) {
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

        OptimizationResult::new(
            self.gbest_pos.clone(),
            self.gbest_val,
            tracker.evaluations(),
            self.history.len(),
            self.history.clone(),
            termination,
        )
    }

    fn best(&self) -> Option<&Self::Solution> {
        if self.gbest_pos.is_empty() {
            None
        } else {
            Some(&self.gbest_pos)
        }
    }

    fn history(&self) -> &[f64] {
        &self.history
    }

    fn reset(&mut self) {
        self.positions.clear();
        self.velocities.clear();
        self.pbest_pos.clear();
        self.pbest_val.clear();
        self.gbest_pos.clear();
        self.gbest_val = f64::INFINITY;
        self.history.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pso_sphere() {
        let objective = |x: &[f64]| x.iter().map(|xi| xi * xi).sum();
        let mut pso = ParticleSwarm::default().with_seed(42);
        let space = SearchSpace::continuous(5, -5.0, 5.0);
        let result = pso.optimize(&objective, &space, Budget::Evaluations(5000));
        assert!(result.objective_value < 1.0);
    }

    #[test]
    fn test_pso_rosenbrock() {
        let rosenbrock = |x: &[f64]| {
            x.windows(2)
                .map(|w| 100.0 * (w[1] - w[0].powi(2)).powi(2) + (1.0 - w[0]).powi(2))
                .sum()
        };
        let mut pso = ParticleSwarm::default().with_seed(123).with_swarm_size(50);
        let space = SearchSpace::continuous(2, -5.0, 10.0);
        let result = pso.optimize(&rosenbrock, &space, Budget::Evaluations(10000));
        assert!(result.objective_value < 10.0);
    }

    #[test]
    fn test_pso_builder() {
        let pso = ParticleSwarm::default()
            .with_swarm_size(100)
            .with_inertia(0.5)
            .with_seed(999);
        assert_eq!(pso.swarm_size, 100);
        assert!((pso.inertia - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_pso_reset() {
        let objective = |x: &[f64]| x.iter().sum::<f64>();
        let mut pso = ParticleSwarm::default().with_seed(42);
        let space = SearchSpace::continuous(2, -1.0, 1.0);
        let _ = pso.optimize(&objective, &space, Budget::Evaluations(100));
        assert!(pso.best().is_some());
        pso.reset();
        assert!(pso.best().is_none());
    }
}
