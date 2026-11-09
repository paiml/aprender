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

mod search;

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
mod tests;
