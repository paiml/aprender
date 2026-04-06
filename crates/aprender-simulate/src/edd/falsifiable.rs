//! Falsifiable simulation traits for EDD.
//!
//! Every simulation in the EDD framework must be falsifiable. This module
//! provides the `FalsifiableSimulation` trait that defines how simulations
//! can be actively tested for falsification.
//!
//! # EDD-04: Falsification Criteria
//!
//! > **Claim:** Every model has explicit conditions for refutation.
//! > **Rejection Criteria:** Model without falsification tests.
//!
//! # References
//!
//! - [1] Popper, K. (1959). The Logic of Scientific Discovery
//! - [11] Donzé, A. & Maler, O. (2010). Robust Satisfaction of Temporal Logic (STL)

use super::experiment::FalsificationCriterion;
use std::collections::HashMap;

/// Result of a falsification search.
#[derive(Debug, Clone)]
pub struct FalsificationResult {
    /// Whether any falsifying condition was found
    pub falsified: bool,
    /// The parameter values that caused falsification (if any)
    pub falsifying_params: Option<HashMap<String, f64>>,
    /// Which criterion was violated (if any)
    pub violated_criterion: Option<String>,
    /// Robustness degree (positive = satisfied, negative = violated)
    pub robustness: f64,
    /// Number of parameter combinations tested
    pub tests_performed: usize,
    /// Human-readable summary
    pub summary: String,
}

impl FalsificationResult {
    /// Create a result indicating no falsification found.
    #[must_use]
    pub fn not_falsified(tests_performed: usize, robustness: f64) -> Self {
        Self {
            falsified: false,
            falsifying_params: None,
            violated_criterion: None,
            robustness,
            tests_performed,
            summary: format!(
                "Model not falsified after {tests_performed} tests (robustness: {robustness:.4})"
            ),
        }
    }

    /// Create a result indicating falsification was found.
    #[must_use]
    pub fn falsified(
        params: HashMap<String, f64>,
        criterion: &str,
        robustness: f64,
        tests_performed: usize,
    ) -> Self {
        Self {
            falsified: true,
            falsifying_params: Some(params),
            violated_criterion: Some(criterion.to_string()),
            robustness,
            tests_performed,
            summary: format!(
                "Model FALSIFIED at test {tests_performed}: criterion '{criterion}' violated (robustness: {robustness:.4})"
            ),
        }
    }
}

/// A trajectory of simulation states over time.
#[derive(Debug, Clone)]
pub struct Trajectory {
    /// Time points
    pub times: Vec<f64>,
    /// State values at each time point (column-major: `state_dim × n_times`)
    pub states: Vec<Vec<f64>>,
    /// State variable names
    pub state_names: Vec<String>,
}

impl Trajectory {
    /// Create a new empty trajectory.
    #[must_use]
    pub fn new(state_names: Vec<String>) -> Self {
        Self {
            times: Vec::new(),
            states: vec![Vec::new(); state_names.len()],
            state_names,
        }
    }

    /// Add a time point and state.
    pub fn push(&mut self, time: f64, state: &[f64]) {
        self.times.push(time);
        for (i, &val) in state.iter().enumerate() {
            if i < self.states.len() {
                self.states[i].push(val);
            }
        }
    }

    /// Get the number of time points.
    #[must_use]
    pub fn len(&self) -> usize {
        self.times.len()
    }

    /// Check if the trajectory is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.times.is_empty()
    }

    /// Get a state variable by name at a specific time index.
    #[must_use]
    pub fn get_state(&self, name: &str, time_idx: usize) -> Option<f64> {
        let state_idx = self.state_names.iter().position(|n| n == name)?;
        self.states.get(state_idx)?.get(time_idx).copied()
    }
}

/// Parameter space for falsification search.
#[derive(Debug, Clone)]
pub struct ParamSpace {
    /// Parameter bounds: name -> (min, max)
    pub bounds: HashMap<String, (f64, f64)>,
    /// Number of samples per dimension
    pub samples_per_dim: usize,
    /// Whether to use Latin Hypercube Sampling
    pub use_lhs: bool,
}

impl ParamSpace {
    /// Create a new parameter space.
    #[must_use]
    pub fn new() -> Self {
        Self {
            bounds: HashMap::new(),
            samples_per_dim: 10,
            use_lhs: true,
        }
    }

    /// Add a parameter with bounds.
    #[must_use]
    pub fn with_param(mut self, name: &str, min: f64, max: f64) -> Self {
        self.bounds.insert(name.to_string(), (min, max));
        self
    }

    /// Set samples per dimension.
    #[must_use]
    pub fn with_samples(mut self, n: usize) -> Self {
        self.samples_per_dim = n;
        self
    }

    /// Generate grid points for exhaustive search.
    #[must_use]
    pub fn grid_points(&self) -> Vec<HashMap<String, f64>> {
        let params: Vec<(&String, &(f64, f64))> = self.bounds.iter().collect();
        if params.is_empty() {
            return vec![HashMap::new()];
        }

        let mut points = Vec::new();
        let n = self.samples_per_dim;

        // Generate all combinations (simple grid)
        let total_points = n.pow(params.len() as u32);
        for i in 0..total_points {
            let mut point = HashMap::new();
            let mut idx = i;
            for (name, (min, max)) in &params {
                let dim_idx = idx % n;
                idx /= n;
                let t = if n > 1 {
                    dim_idx as f64 / (n - 1) as f64
                } else {
                    0.5
                };
                let value = min + t * (max - min);
                point.insert((*name).clone(), value);
            }
            points.push(point);
        }

        points
    }
}

impl Default for ParamSpace {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for simulations that can be falsified.
///
/// Every simulation in the EDD framework must implement this trait to
/// provide explicit falsification criteria and support active search
/// for falsifying conditions.
///
/// # Example
///
/// ```ignore
/// impl FalsifiableSimulation for MySimulation {
///     fn falsification_criteria(&self) -> Vec<FalsificationCriterion> {
///         vec![
///             FalsificationCriterion::new(
///                 "energy_conservation",
///                 "|E(t) - E(0)| / E(0) > 1e-6",
///                 FalsificationAction::RejectModel,
///             ),
///         ]
///     }
///
///     fn evaluate(&self, params: &HashMap<String, f64>) -> Trajectory {
///         // Run simulation with given parameters
///     }
///
///     fn check_criterion(&self, criterion: &str, trajectory: &Trajectory) -> f64 {
///         // Return robustness: positive = satisfied, negative = violated
///     }
/// }
/// ```
pub trait FalsifiableSimulation {
    /// Define conditions that would falsify this simulation.
    fn falsification_criteria(&self) -> Vec<FalsificationCriterion>;

    /// Evaluate the simulation with given parameters.
    fn evaluate(&self, params: &HashMap<String, f64>) -> Trajectory;

    /// Check a specific criterion against a trajectory.
    ///
    /// Returns robustness degree:
    /// - ρ > 0 → satisfies criterion with margin ρ
    /// - ρ < 0 → violates criterion by margin |ρ|
    fn check_criterion(&self, criterion: &str, trajectory: &Trajectory) -> f64;

    /// Actively search for falsifying conditions.
    ///
    /// Explores the parameter space looking for conditions that would
    /// falsify the model (violate any falsification criterion).
    fn seek_falsification(&self, params: &ParamSpace) -> FalsificationResult {
        let criteria = self.falsification_criteria();
        let points = params.grid_points();
        let mut min_robustness = f64::INFINITY;

        for (test_idx, point) in points.iter().enumerate() {
            let trajectory = self.evaluate(point);

            for criterion in &criteria {
                let robustness = self.check_criterion(&criterion.name, &trajectory);
                min_robustness = min_robustness.min(robustness);

                if robustness < 0.0 {
                    // Found falsifying condition
                    return FalsificationResult::falsified(
                        point.clone(),
                        &criterion.name,
                        robustness,
                        test_idx + 1,
                    );
                }
            }
        }

        FalsificationResult::not_falsified(points.len(), min_robustness)
    }

    /// Compute overall robustness for a trajectory.
    ///
    /// Returns the minimum robustness across all criteria.
    fn robustness(&self, trajectory: &Trajectory) -> f64 {
        let criteria = self.falsification_criteria();
        criteria
            .iter()
            .map(|c| self.check_criterion(&c.name, trajectory))
            .fold(f64::INFINITY, f64::min)
    }
}

/// Seed management for reproducible experiments.
///
/// Implements EDD-03: Deterministic Reproducibility.
///
/// ```ignore
/// let seed = ExperimentSeed::new(42);
/// let arrival_rng = seed.derive_rng("arrivals");
/// let service_rng = seed.derive_rng("service");
/// ```
#[derive(Debug, Clone)]
pub struct ExperimentSeed {
    /// Master seed for all RNG operations
    pub master_seed: u64,
    /// Per-component seeds derived from master
    pub component_seeds: HashMap<String, u64>,
    /// IEEE 754 strict mode for floating-point reproducibility
    pub ieee_strict: bool,
}

impl ExperimentSeed {
    /// Create a new experiment seed.
    #[must_use]
    pub fn new(master_seed: u64) -> Self {
        Self {
            master_seed,
            component_seeds: HashMap::new(),
            ieee_strict: true,
        }
    }

    /// Disable IEEE strict mode (for performance).
    #[must_use]
    pub fn relaxed(mut self) -> Self {
        self.ieee_strict = false;
        self
    }

    /// Pre-register a component seed.
    #[must_use]
    pub fn with_component(mut self, component: &str, seed: u64) -> Self {
        self.component_seeds.insert(component.to_string(), seed);
        self
    }

    /// Derive a seed for a specific component.
    ///
    /// Uses deterministic derivation from master seed if not pre-registered.
    #[must_use]
    pub fn derive_seed(&self, component: &str) -> u64 {
        if let Some(&seed) = self.component_seeds.get(component) {
            return seed;
        }

        // Deterministic derivation using simple hash mixing
        // This is a simplified version - production would use BLAKE3
        let mut hash = self.master_seed;
        for byte in component.as_bytes() {
            hash = hash.wrapping_mul(0x517c_c1b7_2722_0a95);
            hash = hash.wrapping_add(u64::from(*byte));
            hash ^= hash >> 33;
        }
        hash
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::edd::experiment::FalsificationAction;

    #[test]
    fn test_falsification_result_not_falsified() {
        let result = FalsificationResult::not_falsified(100, 0.5);
        assert!(!result.falsified);
        assert_eq!(result.tests_performed, 100);
        assert!((result.robustness - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_falsification_result_falsified() {
        let mut params = HashMap::new();
        params.insert("x".to_string(), 1.0);
        let result = FalsificationResult::falsified(params, "energy_drift", -0.1, 42);
        assert!(result.falsified);
        assert!(result.falsifying_params.is_some());
        assert_eq!(result.violated_criterion, Some("energy_drift".to_string()));
    }

    #[test]
    fn test_trajectory() {
        let mut traj = Trajectory::new(vec!["x".to_string(), "v".to_string()]);
        traj.push(0.0, &[1.0, 0.0]);
        traj.push(1.0, &[0.0, -1.0]);

        assert_eq!(traj.len(), 2);
        assert!(!traj.is_empty());
        assert!((traj.get_state("x", 0).unwrap() - 1.0).abs() < f64::EPSILON);
        assert!((traj.get_state("v", 1).unwrap() - (-1.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_param_space_grid() {
        let space = ParamSpace::new().with_param("x", 0.0, 1.0).with_samples(3);

        let points = space.grid_points();
        assert_eq!(points.len(), 3);
        assert!((points[0]["x"] - 0.0).abs() < f64::EPSILON);
        assert!((points[1]["x"] - 0.5).abs() < f64::EPSILON);
        assert!((points[2]["x"] - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_param_space_2d_grid() {
        let space = ParamSpace::new()
            .with_param("x", 0.0, 1.0)
            .with_param("y", 0.0, 1.0)
            .with_samples(2);

        let points = space.grid_points();
        assert_eq!(points.len(), 4); // 2^2
    }

    #[test]
    fn test_experiment_seed_derivation() {
        let seed = ExperimentSeed::new(42);

        let seed1 = seed.derive_seed("arrivals");
        let seed2 = seed.derive_seed("service");
        let seed3 = seed.derive_seed("arrivals");

        // Same component should give same seed
        assert_eq!(seed1, seed3);
        // Different components should give different seeds
        assert_ne!(seed1, seed2);
    }

    #[test]
    fn test_experiment_seed_preregistered() {
        let seed = ExperimentSeed::new(42).with_component("arrivals", 12345);

        assert_eq!(seed.derive_seed("arrivals"), 12345);
    }

    // Mock simulation for testing FalsifiableSimulation trait
    struct MockSimulation {
        fail_on: Option<String>,
    }

    impl FalsifiableSimulation for MockSimulation {
        fn falsification_criteria(&self) -> Vec<FalsificationCriterion> {
            vec![FalsificationCriterion::new(
                "bounds_check",
                "x < 10",
                FalsificationAction::RejectModel,
            )]
        }

        fn evaluate(&self, params: &HashMap<String, f64>) -> Trajectory {
            let mut traj = Trajectory::new(vec!["x".to_string()]);
            let x = params.get("x").copied().unwrap_or(0.0);
            traj.push(0.0, &[x]);
            traj
        }

        fn check_criterion(&self, criterion: &str, trajectory: &Trajectory) -> f64 {
            if Some(criterion.to_string()) == self.fail_on {
                return -1.0;
            }
            let x = trajectory.get_state("x", 0).unwrap_or(0.0);
            10.0 - x // robustness: positive if x < 10
        }
    }

    #[test]
    fn test_falsifiable_simulation_not_falsified() {
        let sim = MockSimulation { fail_on: None };
        let params = ParamSpace::new().with_param("x", 0.0, 5.0).with_samples(3);

        let result = sim.seek_falsification(&params);
        assert!(!result.falsified);
    }

    #[test]
    fn test_falsifiable_simulation_robustness() {
        let sim = MockSimulation { fail_on: None };
        let mut traj = Trajectory::new(vec!["x".to_string()]);
        traj.push(0.0, &[3.0]);

        let robustness = sim.robustness(&traj);
        assert!((robustness - 7.0).abs() < f64::EPSILON); // 10 - 3 = 7
    }
}
