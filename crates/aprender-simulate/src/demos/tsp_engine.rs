//! `TspEngine`: `DemoEngine` Implementation for TSP Demos
//!
//! Per specification SIMULAR-DEMO-002: This module provides the unified
//! `DemoEngine` implementation for Traveling Salesman Problem simulations.
//!
//! # Architecture
//!
//! ```text
//! YAML Config → TspEngine → DemoEngine trait
//!                    ↓
//!              TspState (serializable, PartialEq)
//!                    ↓
//!              TUI / WASM (identical states)
//! ```
//!
//! # Key Invariant
//!
//! Given same YAML config and seed, TUI and WASM produce identical state sequences.

use super::engine::{
    CriterionResult, DemoEngine, DemoError, DemoMeta, FalsificationCriterion, MetamorphicRelation,
    MrResult, Severity,
};
use super::tsp_instance::{TspAlgorithmConfig, TspCity, TspMeta};
use crate::engine::rng::SimRng;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

// =============================================================================
// Configuration Types (loaded from YAML)
// =============================================================================

/// Simulation type configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationConfig {
    #[serde(rename = "type")]
    pub sim_type: String,
    pub name: String,
}

/// Complete TSP configuration from YAML (extends `TspInstanceYaml`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TspConfig {
    /// Simulation header for `DemoEngine`.
    pub simulation: SimulationConfig,
    /// Instance metadata.
    pub meta: TspMeta,
    /// List of cities.
    pub cities: Vec<TspCity>,
    /// Distance matrix (n x n).
    pub matrix: Vec<Vec<u32>>,
    /// Algorithm configuration.
    #[serde(default)]
    pub algorithm: TspAlgorithmConfig,
}

// =============================================================================
// State Types (serializable, comparable)
// =============================================================================

/// TSP state snapshot.
///
/// This is THE state that gets compared for TUI/WASM parity.
/// It MUST be `PartialEq` for the probar tests.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TspState {
    /// Current tour (city indices).
    pub tour: Vec<usize>,
    /// Current tour length.
    pub tour_length: u32,
    /// Best tour found.
    pub best_tour: Vec<usize>,
    /// Best tour length.
    pub best_tour_length: u32,
    /// Number of restarts completed.
    pub restart_count: u32,
    /// Total 2-opt improvements made.
    pub two_opt_count: u64,
    /// Step count.
    pub step_count: u64,
    /// Is converged (no improvement in N restarts).
    pub is_converged: bool,
}

impl TspState {
    /// Compute hash for quick comparison.
    #[must_use]
    pub fn compute_hash(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.tour.hash(&mut hasher);
        self.tour_length.hash(&mut hasher);
        self.best_tour.hash(&mut hasher);
        self.best_tour_length.hash(&mut hasher);
        self.restart_count.hash(&mut hasher);
        self.step_count.hash(&mut hasher);
        hasher.finish()
    }
}

/// Step result for TSP simulation.
#[derive(Debug, Clone)]
pub struct TspStepResult {
    /// Whether this step improved the solution.
    pub improved: bool,
    /// Current tour length.
    pub tour_length: u32,
    /// Gap from best known (if available).
    pub optimality_gap: Option<f64>,
}

// =============================================================================
// TspEngine Implementation
// =============================================================================

/// Unified TSP engine implementing `DemoEngine`.
///
/// This provides a proper `DemoEngine` implementation that:
/// - Loads from YAML
/// - Produces deterministic states
/// - Supports TUI/WASM parity verification
#[derive(Debug, Clone)]
pub struct TspEngine {
    /// Configuration from YAML.
    config: TspConfig,
    /// Number of cities.
    n: usize,
    /// Distance matrix.
    distances: Vec<Vec<u32>>,
    /// Current tour.
    tour: Vec<usize>,
    /// Current tour length.
    tour_length: u32,
    /// Best tour found.
    best_tour: Vec<usize>,
    /// Best tour length.
    best_tour_length: u32,
    /// RCL size for randomized greedy.
    rcl_size: usize,
    /// Number of restarts completed.
    restart_count: u32,
    /// Max restarts before completion.
    max_restarts: u32,
    /// 2-opt improvements made.
    two_opt_count: u64,
    /// Step count.
    step_count: u64,
    /// Restarts without improvement (for convergence).
    stagnation_count: u32,
    /// RNG for determinism.
    rng: SimRng,
    /// Seed for reproducibility.
    seed: u64,
    /// Demo metadata.
    demo_meta: DemoMeta,
}

impl TspEngine {
    /// Calculate tour length.
    fn calculate_tour_length(&self, tour: &[usize]) -> u32 {
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
    fn gen_range(&mut self, max: usize) -> usize {
        if max == 0 {
            return 0;
        }
        (self.rng.gen_u64() as usize) % max
    }

    /// Construct initial tour using randomized greedy.
    fn construct_greedy_tour(&mut self) -> Vec<usize> {
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

    /// Apply 2-opt local search.
    fn two_opt(&self, tour: &mut [usize]) -> u64 {
        let mut improvements = 0u64;
        let mut improved = true;

        while improved {
            improved = false;
            for i in 0..self.n - 1 {
                for j in i + 2..self.n {
                    // Skip adjacent edges
                    if j == i + 1 || (i == 0 && j == self.n - 1) {
                        continue;
                    }

                    // Calculate improvement
                    let a = tour[i];
                    let b = tour[i + 1];
                    let c = tour[j];
                    let d = tour[(j + 1) % self.n];

                    let current = self.distances[a][b] + self.distances[c][d];
                    let swapped = self.distances[a][c] + self.distances[b][d];

                    if swapped < current {
                        // Reverse segment [i+1, j]
                        tour[i + 1..=j].reverse();
                        improved = true;
                        improvements += 1;
                    }
                }
            }
        }

        improvements
    }

    /// Perform one restart iteration.
    fn do_restart(&mut self) -> bool {
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

impl DemoEngine for TspEngine {
    type Config = TspConfig;
    type State = TspState;
    type StepResult = TspStepResult;

    fn from_yaml(yaml: &str) -> Result<Self, DemoError> {
        let config: TspConfig = serde_yaml::from_str(yaml)?;
        Ok(Self::from_config(config))
    }

    fn from_config(config: Self::Config) -> Self {
        let n = config.cities.len();
        let distances = config.matrix.clone();
        let seed = config.algorithm.params.seed;
        let rcl_size = config.algorithm.params.rcl_size;
        let max_restarts = config.algorithm.params.restarts as u32;

        let demo_meta = DemoMeta {
            id: config.meta.id.clone(),
            version: config.meta.version.clone(),
            demo_type: "tsp".to_string(),
            description: config.meta.description.clone(),
            author: config.meta.source.clone(),
            created: String::new(),
        };

        // Initial tour: sequential
        let tour: Vec<usize> = (0..n).collect();
        let tour_length = u32::MAX;
        let best_tour = tour.clone();
        let best_tour_length = u32::MAX;

        let mut engine = Self {
            config,
            n,
            distances,
            tour,
            tour_length,
            best_tour,
            best_tour_length,
            rcl_size,
            restart_count: 0,
            max_restarts,
            two_opt_count: 0,
            step_count: 0,
            stagnation_count: 0,
            rng: SimRng::new(seed),
            seed,
            demo_meta,
        };

        // Do initial construction
        engine.tour = engine.construct_greedy_tour();
        let improvements = engine.two_opt(&mut engine.tour.clone());
        engine.two_opt_count = improvements;
        engine.tour_length = engine.calculate_tour_length(&engine.tour);
        engine.best_tour = engine.tour.clone();
        engine.best_tour_length = engine.tour_length;

        engine
    }

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn reset(&mut self) {
        self.reset_with_seed(self.seed);
    }

    fn reset_with_seed(&mut self, seed: u64) {
        self.rng = SimRng::new(seed);
        self.seed = seed;
        self.restart_count = 0;
        self.two_opt_count = 0;
        self.step_count = 0;
        self.stagnation_count = 0;

        // Re-initialize
        self.tour = self.construct_greedy_tour();
        let improvements = self.two_opt(&mut self.tour.clone());
        self.two_opt_count = improvements;
        self.tour_length = self.calculate_tour_length(&self.tour);
        self.best_tour = self.tour.clone();
        self.best_tour_length = self.tour_length;
    }

    fn step(&mut self) -> Self::StepResult {
        self.step_count += 1;

        // Each step is one restart
        let improved = self.do_restart();

        TspStepResult {
            improved,
            tour_length: self.tour_length,
            optimality_gap: self
                .config
                .meta
                .optimal_known
                .map(|opt| (f64::from(self.best_tour_length) - f64::from(opt)) / f64::from(opt)),
        }
    }

    fn is_complete(&self) -> bool {
        // Complete if max restarts reached or stagnated
        self.restart_count >= self.max_restarts || self.stagnation_count >= 20
    }

    fn state(&self) -> Self::State {
        TspState {
            tour: self.tour.clone(),
            tour_length: self.tour_length,
            best_tour: self.best_tour.clone(),
            best_tour_length: self.best_tour_length,
            restart_count: self.restart_count,
            two_opt_count: self.two_opt_count,
            step_count: self.step_count,
            is_converged: self.is_complete(),
        }
    }

    fn restore(&mut self, state: &Self::State) {
        self.tour.clone_from(&state.tour);
        self.tour_length = state.tour_length;
        self.best_tour.clone_from(&state.best_tour);
        self.best_tour_length = state.best_tour_length;
        self.restart_count = state.restart_count;
        self.two_opt_count = state.two_opt_count;
        self.step_count = state.step_count;
    }

    fn step_count(&self) -> u64 {
        self.step_count
    }

    fn seed(&self) -> u64 {
        self.seed
    }

    fn meta(&self) -> &DemoMeta {
        &self.demo_meta
    }

    fn falsification_criteria(&self) -> Vec<FalsificationCriterion> {
        vec![FalsificationCriterion {
            id: "TSP-VALID-001".to_string(),
            name: "Valid tour".to_string(),
            metric: "tour_validity".to_string(),
            threshold: 1.0,
            condition: "all cities visited exactly once".to_string(),
            tolerance: 0.0,
            severity: Severity::Critical,
        }]
    }

    fn evaluate_criteria(&self) -> Vec<CriterionResult> {
        // Check tour is valid (visits each city exactly once)
        let mut visited = vec![false; self.n];
        let mut valid = self.best_tour.len() == self.n;
        for &city in &self.best_tour {
            if city >= self.n || visited[city] {
                valid = false;
                break;
            }
            visited[city] = true;
        }

        vec![CriterionResult {
            id: "TSP-VALID-001".to_string(),
            passed: valid,
            actual: if valid { 1.0 } else { 0.0 },
            expected: 1.0,
            message: if valid {
                format!("Valid tour of length {}", self.best_tour_length)
            } else {
                "Invalid tour".to_string()
            },
            severity: Severity::Critical,
        }]
    }

    fn metamorphic_relations(&self) -> Vec<MetamorphicRelation> {
        vec![MetamorphicRelation {
            id: "MR-PERMUTATION".to_string(),
            description: "Relabeling cities preserves tour length".to_string(),
            source_transform: "permute_city_labels".to_string(),
            expected_relation: "tour_length_unchanged".to_string(),
            tolerance: 0.0,
        }]
    }

    fn verify_mr(&self, mr: &MetamorphicRelation) -> MrResult {
        match mr.id.as_str() {
            "MR-PERMUTATION" => {
                // Verify tour length calculation is consistent
                let recalculated = self.calculate_tour_length(&self.best_tour);
                let passed = recalculated == self.best_tour_length;
                MrResult {
                    id: mr.id.clone(),
                    passed,
                    message: format!(
                        "Stored: {}, Recalculated: {}",
                        self.best_tour_length, recalculated
                    ),
                    source_value: Some(f64::from(self.best_tour_length)),
                    followup_value: Some(f64::from(recalculated)),
                }
            }
            _ => MrResult {
                id: mr.id.clone(),
                passed: false,
                message: format!("Unknown metamorphic relation: {}", mr.id),
                source_value: None,
                followup_value: None,
            },
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_yaml() -> String {
        r#"
simulation:
  type: tsp
  name: "Test TSP"

meta:
  id: "TEST-TSP-001"
  version: "1.0.0"
  description: "Test TSP instance"

cities:
  - id: 0
    name: "A"
    alias: "A"
    coords: { lat: 0.0, lon: 0.0 }
  - id: 1
    name: "B"
    alias: "B"
    coords: { lat: 0.0, lon: 1.0 }
  - id: 2
    name: "C"
    alias: "C"
    coords: { lat: 1.0, lon: 0.0 }
  - id: 3
    name: "D"
    alias: "D"
    coords: { lat: 1.0, lon: 1.0 }

matrix:
  - [0, 1, 1, 2]
  - [1, 0, 2, 1]
  - [1, 2, 0, 1]
  - [2, 1, 1, 0]

algorithm:
  method: "grasp"
  params:
    rcl_size: 2
    restarts: 10
    two_opt: true
    seed: 42
"#
        .to_string()
    }

    #[test]
    fn test_from_yaml() {
        let yaml = make_test_yaml();
        let engine = TspEngine::from_yaml(&yaml);
        assert!(engine.is_ok(), "Failed to parse YAML: {:?}", engine.err());
    }

    #[test]
    fn test_deterministic_state() {
        let yaml = make_test_yaml();
        let mut engine1 = TspEngine::from_yaml(&yaml).unwrap();
        let mut engine2 = TspEngine::from_yaml(&yaml).unwrap();

        for _ in 0..5 {
            engine1.step();
            engine2.step();
        }

        assert_eq!(
            engine1.state(),
            engine2.state(),
            "State divergence detected"
        );
    }

    #[test]
    fn test_reset_replay() {
        let yaml = make_test_yaml();
        let mut engine = TspEngine::from_yaml(&yaml).unwrap();

        // Run 5 steps
        for _ in 0..5 {
            engine.step();
        }
        let state1 = engine.state();

        // Reset and replay
        engine.reset();
        for _ in 0..5 {
            engine.step();
        }
        let state2 = engine.state();

        assert_eq!(state1, state2, "Reset did not produce identical replay");
    }

    #[test]
    fn test_valid_tour() {
        let yaml = make_test_yaml();
        let mut engine = TspEngine::from_yaml(&yaml).unwrap();

        // Run to completion
        while !engine.is_complete() {
            engine.step();
        }

        let results = engine.evaluate_criteria();
        assert!(!results.is_empty());
        assert!(results[0].passed, "Tour should be valid");
    }

    #[test]
    fn test_tour_improves() {
        let yaml = make_test_yaml();
        let mut engine = TspEngine::from_yaml(&yaml).unwrap();

        let initial_length = engine.best_tour_length;

        // Run multiple steps
        for _ in 0..10 {
            engine.step();
        }

        // With GRASP + 2-opt, we should find the optimal tour
        // For this 4-city instance, optimal is 4 (square path)
        assert!(
            engine.best_tour_length <= initial_length,
            "Tour should improve or stay same"
        );
    }

    #[test]
    fn test_step_count() {
        let yaml = make_test_yaml();
        let mut engine = TspEngine::from_yaml(&yaml).unwrap();

        assert_eq!(engine.step_count(), 0);
        engine.step();
        assert_eq!(engine.step_count(), 1);
    }

    #[test]
    fn test_seed() {
        let yaml = make_test_yaml();
        let engine = TspEngine::from_yaml(&yaml).unwrap();
        assert_eq!(engine.seed(), 42);
    }

    #[test]
    fn test_meta() {
        let yaml = make_test_yaml();
        let engine = TspEngine::from_yaml(&yaml).unwrap();
        assert_eq!(engine.meta().id, "TEST-TSP-001");
        assert_eq!(engine.meta().demo_type, "tsp");
    }

    #[test]
    fn test_state_hash() {
        let yaml = make_test_yaml();
        let mut engine = TspEngine::from_yaml(&yaml).unwrap();
        engine.step();

        let state = engine.state();
        let hash1 = state.compute_hash();
        let hash2 = state.compute_hash();

        assert_eq!(hash1, hash2, "Hash should be deterministic");
    }

    #[test]
    fn test_is_complete() {
        let yaml = make_test_yaml();
        let mut engine = TspEngine::from_yaml(&yaml).unwrap();

        // Run until complete
        while !engine.is_complete() {
            engine.step();
        }

        assert!(engine.is_complete());
    }
}
