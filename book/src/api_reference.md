# API Reference

Quick reference for Simular's main types and functions.

## Prelude

```rust,ignore
use simular::prelude::*;

// Includes:
// - SimConfig, SimConfigBuilder
// - SimEngine, SimState, SimTime
// - SimRng
// - JidokaGuard, JidokaViolation
// - SimError, SimResult
// - FalsifiableHypothesis, NHSTResult
```

## Core Types

### SimState

```rust,ignore
use simular::engine::state::{SimState, Vec3};

let mut state = SimState::new();

// Bodies
state.add_body(mass, position, velocity);
state.num_bodies() -> usize
state.masses() -> &[f64]
state.positions() -> &[Vec3]
state.velocities() -> &[Vec3]
state.set_position(index, position)
state.set_velocity(index, velocity)

// Energy
state.kinetic_energy() -> f64
state.potential_energy() -> f64
state.total_energy() -> f64
state.set_potential_energy(energy)

// Validation
state.all_finite() -> bool
```

### Vec3

```rust,ignore
use simular::engine::state::Vec3;

let v = Vec3::new(x, y, z);
let zero = Vec3::zero();

// Operations
v.magnitude() -> f64
v.magnitude_squared() -> f64
v.normalize() -> Vec3
v.dot(&other) -> f64
v.cross(&other) -> Vec3
v.scale(s) -> Vec3
v.is_finite() -> bool

// Operators
v1 + v2, v1 - v2, v * scalar, -v
```

### SimRng

```rust,ignore
use simular::engine::rng::SimRng;

let mut rng = SimRng::new(seed);

rng.gen_f64() -> f64              // [0, 1)
rng.gen_range_f64(min, max) -> f64
rng.gen_u64() -> u64
rng.gen_standard_normal() -> f64  // N(0, 1)
rng.gen_normal(mean, std) -> f64
rng.sample_n(n) -> Vec<f64>
rng.partition(n) -> Vec<SimRng>

rng.master_seed() -> u64
rng.stream() -> u64
rng.save_state() -> RngState
rng.restore_state(&state) -> Result<()>
```

## Physics Domain

```rust,ignore
use simular::domains::physics::{
    PhysicsEngine, ForceField,
    GravityField, CentralForceField,
    Integrator, VerletIntegrator, RK4Integrator, EulerIntegrator,
};

// Engine
let engine = PhysicsEngine::new(force_field, integrator);
engine.step(&mut state, dt) -> SimResult<()>

// Integrator trait
trait Integrator {
    fn step(&self, state: &mut SimState, force: &dyn ForceField, dt: f64) -> SimResult<()>;
    fn error_order(&self) -> u32;
    fn is_symplectic(&self) -> bool;
}

// ForceField trait
trait ForceField {
    fn acceleration(&self, position: &Vec3, mass: f64) -> Vec3;
    fn potential(&self, position: &Vec3, mass: f64) -> f64;
}
```

## Monte Carlo Domain

```rust,ignore
use simular::domains::monte_carlo::{
    MonteCarloEngine, MonteCarloResult, VarianceReduction,
    WorkStealingMonteCarlo, SimulationTask,
};

// Engine
let engine = MonteCarloEngine::new(n_samples, variance_reduction);
let engine = MonteCarloEngine::with_samples(n_samples);

engine.run(|x| f64, &mut rng) -> MonteCarloResult
engine.run_nd(dim, |&[f64]| f64, &mut rng) -> MonteCarloResult

// Result
struct MonteCarloResult {
    estimate: f64,
    std_error: f64,
    samples: usize,
    confidence_interval: (f64, f64),
    variance_reduction_factor: Option<f64>,
}

result.contains(value) -> bool
result.relative_error() -> f64

// Variance reduction
enum VarianceReduction {
    None,
    Antithetic,
    ControlVariate { control_fn, expectation },
    ImportanceSampling { sample_fn, likelihood_ratio },
    SelfNormalizingIS { sample_fn, weight_fn },
    Stratified { num_strata },
}
```

## Optimization Domain

```rust,ignore
use simular::domains::optimization::{
    BayesianOptimizer, OptimizerConfig, AcquisitionFunction,
    GaussianProcess,
};

// Optimizer
let config = OptimizerConfig {
    bounds: Vec<(f64, f64)>,
    acquisition: AcquisitionFunction,
    length_scale: f64,
    signal_variance: f64,
    noise_variance: f64,
    n_candidates: usize,
    seed: u64,
};

let mut optimizer = BayesianOptimizer::new(config);
optimizer.suggest() -> Vec<f64>
optimizer.observe(x, y) -> SimResult<()>
optimizer.best() -> Option<(&[f64], f64)>

// Acquisition functions
enum AcquisitionFunction {
    ExpectedImprovement,
    UCB { kappa: f64 },
    ProbabilityOfImprovement,
}

// Gaussian Process
let mut gp = GaussianProcess::new(length_scale, signal_variance, noise_variance);
gp.add_observation(x, y)
gp.fit() -> SimResult<()>
gp.predict(&x) -> (mean, variance)
```

## Jidoka

```rust,ignore
use simular::engine::jidoka::{
    JidokaConfig, JidokaGuard, JidokaViolation, JidokaWarning,
    ViolationSeverity, SeverityClassifier,
};

// Config
let config = JidokaConfig {
    check_finite: bool,
    check_energy: bool,
    energy_tolerance: f64,
    constraint_tolerance: f64,
    severity_classifier: SeverityClassifier,
};

// Guard
let mut guard = JidokaGuard::new(config);
guard.check(&state) -> Result<(), JidokaViolation>
guard.set_initial_energy(energy)

// Severity
enum ViolationSeverity {
    Acceptable,
    Warning,
    Critical,
    Fatal,
}
```

## Scenarios

```rust,ignore
use simular::scenarios::{
    // Rocket
    RocketScenario, RocketConfig, StageSeparation,

    // Satellite
    SatelliteScenario, OrbitalElements,

    // Pendulum
    PendulumScenario, PendulumConfig,

    // Climate
    ClimateScenario, ClimateConfig,

    // Portfolio
    PortfolioScenario, PortfolioConfig, VaRResult,

    // Epidemic
    SIRScenario, SIRConfig,
    SEIRScenario, SEIRConfig,
};
```

## Error Handling

```rust,ignore
use simular::error::{SimError, SimResult};

type SimResult<T> = Result<T, SimError>;

enum SimError {
    InvalidConfig(String),
    SimulationFailed(String),
    JidokaViolation(JidokaViolation),
    IoError(std::io::Error),
    // ...
}
```

## Feature Flags

```toml
[dependencies]
simular = { version = "0.1", features = ["tui", "web"] }

# Available features:
# - tui: Terminal UI visualization
# - web: Web server visualization
# - full: All features
```

## Module Structure

```text
simular
├── prelude           # Common imports
├── config            # Configuration
├── engine            # Core engine
│   ├── state         # SimState, Vec3
│   ├── rng           # SimRng
│   ├── jidoka        # Anomaly detection
│   ├── scheduler     # Work-stealing
│   └── clock         # Simulation time
├── domains           # Simulation domains
│   ├── physics       # Integrators, forces
│   ├── monte_carlo   # Stochastic sampling
│   ├── optimization  # Bayesian optimization
│   └── ml            # ML training
├── scenarios         # Pre-built scenarios
│   ├── rocket
│   ├── satellite
│   ├── pendulum
│   ├── climate
│   ├── portfolio
│   └── epidemic
├── replay            # Record/replay
├── falsification     # Hypothesis testing
└── visualization     # TUI/Web
```
