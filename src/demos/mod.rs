//! EDD Showcase Demos - Interactive demonstrations of the EDD methodology.
//!
//! This module provides interactive demonstrations that embody the
//! Equation-Driven Development methodology. Each demo follows the complete cycle:
//!
//! 1. **Equation** - Define the governing equation
//! 2. **Failing Test** - Write a test that fails without implementation
//! 3. **Implementation** - Implement the simulation
//! 4. **Verification** - Test passes, equation verified
//! 5. **Falsification** - Demonstrate conditions that break the equation
//!
//! # Architecture (SIMULAR-DEMO-001)
//!
//! All demos implement the [`DemoEngine`] trait which ensures:
//! - **YAML-first**: Single source of truth for configuration
//! - **Deterministic replay**: Same seed → same output
//! - **Renderer independence**: TUI/WASM produce identical state sequences
//! - **Metamorphic testing**: Per Chen et al. (2018)
//! - **Falsification criteria**: Per Popperian methodology
//!
//! # Demos
//!
//! 1. [`harmonic_oscillator`] - Energy conservation with symplectic integrators
//! 2. [`littles_law_factory`] - WIP = Throughput × Cycle Time
//! 3. [`monte_carlo_pi`] - O(n^{-1/2}) convergence rate
//! 4. [`kingmans_hockey`] - Queue wait times at high utilization
//! 5. [`kepler_orbit`] - Orbital mechanics conservation laws
//! 6. [`tsp_grasp`] - TSP with randomized greedy + 2-opt (GRASP)
//!
//! # WASM Compatibility
//!
//! All demos compile to `wasm32-unknown-unknown` and export a standard interface
//! for integration with web frontends.

pub mod engine;
pub mod harmonic_oscillator;
pub mod kepler_orbit;
pub mod kingmans_hockey;
pub mod littles_law_factory;
pub mod monte_carlo_pi;
pub mod orbit_engine;
pub mod tsp_engine;
pub mod tsp_grasp;
pub mod tsp_instance;

// WASM modules: DOM operations require unwrap/expect - if elements don't exist, app can't run.
#[cfg(feature = "wasm")]
#[allow(clippy::unwrap_used)]
#[allow(clippy::expect_used)]
#[allow(clippy::missing_panics_doc)]
#[allow(clippy::missing_errors_doc)]
#[allow(clippy::cast_lossless)]
#[allow(clippy::redundant_closure_for_method_calls)]
#[allow(clippy::manual_let_else)]
pub mod tsp_wasm_app;

#[cfg(feature = "wasm")]
#[allow(clippy::unwrap_used)]
#[allow(clippy::expect_used)]
#[allow(clippy::missing_panics_doc)]
#[allow(clippy::missing_errors_doc)]
#[allow(clippy::cast_lossless)]
#[allow(clippy::redundant_closure_for_method_calls)]
#[allow(clippy::manual_let_else)]
pub mod orbit_wasm_app;

// Re-exports for convenience
pub use engine::{
    CriterionResult, DemoEngine, DemoError, DemoMeta, DeterministicReplay, FalsificationCriterion,
    MetamorphicRelation, MrResult, RendererIndependent, Severity,
};
pub use harmonic_oscillator::HarmonicOscillatorDemo;
pub use kepler_orbit::KeplerOrbitDemo;
pub use kingmans_hockey::KingmanHockeyDemo;
pub use littles_law_factory::LittlesLawFactoryDemo;
pub use monte_carlo_pi::MonteCarloPlDemo;
pub use orbit_engine::{OrbitConfig, OrbitalEngine, OrbitalState, OrbitalStepResult};
pub use tsp_engine::{TspConfig, TspEngine, TspState, TspStepResult};
pub use tsp_grasp::TspGraspDemo;
pub use tsp_instance::{
    Coords, TspAlgorithmConfig, TspCity, TspInstanceError, TspInstanceYaml, TspMeta, TspParams,
};

use serde::{Deserialize, Serialize};

/// Common trait for all EDD demos.
pub trait EddDemo {
    /// Demo name for display.
    fn name(&self) -> &'static str;

    /// EMC reference path.
    fn emc_ref(&self) -> &'static str;

    /// Advance the simulation by one timestep.
    fn step(&mut self, dt: f64);

    /// Check if the governing equation is currently verified.
    fn verify_equation(&self) -> bool;

    /// Get the current falsification status.
    fn get_falsification_status(&self) -> FalsificationStatus;

    /// Reset the demo to initial conditions.
    fn reset(&mut self);
}

/// Falsification status for a demo.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FalsificationStatus {
    /// Whether the equation is currently verified.
    pub verified: bool,
    /// List of falsification criteria and their status.
    pub criteria: Vec<CriterionStatus>,
    /// Overall message.
    pub message: String,
}

/// Status of a single falsification criterion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriterionStatus {
    /// Criterion ID (e.g., "HO-ENERGY").
    pub id: String,
    /// Criterion name.
    pub name: String,
    /// Whether it passed.
    pub passed: bool,
    /// Current value.
    pub value: f64,
    /// Threshold for passing.
    pub threshold: f64,
}

/// Integrator type for physics simulations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum IntegratorType {
    /// Störmer-Verlet symplectic integrator (energy-conserving).
    #[default]
    StormerVerlet,
    /// Runge-Kutta 4th order (non-symplectic, energy drifts).
    RK4,
    /// Euler method (1st order, for demonstration of failure).
    Euler,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integrator_type_default() {
        assert_eq!(IntegratorType::default(), IntegratorType::StormerVerlet);
    }

    #[test]
    fn test_falsification_status_serialization() {
        let status = FalsificationStatus {
            verified: true,
            criteria: vec![CriterionStatus {
                id: "TEST-001".to_string(),
                name: "Test criterion".to_string(),
                passed: true,
                value: 0.0,
                threshold: 1e-10,
            }],
            message: "All criteria passed".to_string(),
        };

        let json = serde_json::to_string(&status).expect("serialize");
        assert!(json.contains("TEST-001"));
    }
}
