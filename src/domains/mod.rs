//! Domain-specific simulation engines.
//!
//! Each domain implements a specific type of simulation:
//! - Physics: Rigid body, orbital, fluid dynamics
//! - Monte Carlo: Stochastic sampling with variance reduction
//! - Optimization: Bayesian, evolutionary, gradient-based
//! - ML: Training simulation, prediction, multi-turn evaluation

pub mod ml;
pub mod monte_carlo;
pub mod optimization;
pub mod physics;

pub use ml::{
    AnomalyDetector, AnomalyPattern, AnomalyType, InferenceConfig, JidokaMLFeedback,
    MultiTurnEvaluation, MultiTurnSimulation, ParetoAnalysis, ParetoPoint, PredictionSimulation,
    PredictionState, RollingStats, RulePatch, RuleType, TrainingAnomaly, TrainingConfig,
    TrainingMetrics, TrainingSimulation, TrainingState, TrainingTrajectory, Turn, TurnMetrics,
};
pub use monte_carlo::{MonteCarloEngine, MonteCarloResult, VarianceReduction};
pub use optimization::{AcquisitionFunction, BayesianOptimizer, GaussianProcess, OptimizerConfig};
pub use physics::{EulerIntegrator, Integrator, PhysicsEngine, RK4Integrator, VerletIntegrator};
