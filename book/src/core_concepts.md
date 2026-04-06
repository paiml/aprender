# Core Concepts

Simular is built on three pillars: **determinism**, **anomaly detection**, and **domain separation**.

## Deterministic Simulation

Every simulation in Simular is fully deterministic. Given the same seed, you get bit-identical results:

```rust,ignore
use simular::prelude::*;

// Run 1
let mut rng1 = SimRng::new(42);
let values1: Vec<f64> = (0..100).map(|_| rng1.gen_f64()).collect();

// Run 2
let mut rng2 = SimRng::new(42);
let values2: Vec<f64> = (0..100).map(|_| rng2.gen_f64()).collect();

// Bit-identical
assert_eq!(values1, values2);
```

### Why Determinism Matters

- **Reproducibility**: Same seed = same results, always
- **Debugging**: Reproduce exact failure conditions
- **Testing**: Property-based tests with specific seeds
- **Science**: Required for reproducible research

## SimState: The World State

`SimState` holds all simulation variables in Structure-of-Arrays (SoA) layout:

```rust,ignore
use simular::engine::state::{SimState, Vec3};

let mut state = SimState::new();

// Add bodies
state.add_body(1.0, Vec3::new(0.0, 0.0, 0.0), Vec3::zero());  // Body 0
state.add_body(2.0, Vec3::new(1.0, 0.0, 0.0), Vec3::zero());  // Body 1

// Access state
let positions = state.positions();      // &[Vec3]
let velocities = state.velocities();    // &[Vec3]
let masses = state.masses();            // &[f64]

// Modify state
state.set_position(0, Vec3::new(0.0, 0.0, 10.0));
state.set_velocity(0, Vec3::new(1.0, 0.0, 0.0));

// Energy
let ke = state.kinetic_energy();        // 0.5 * m * v²
let pe = state.potential_energy();      // Set by domain engine
let total = state.total_energy();       // KE + PE
```

### SoA Layout Benefits

```text
// Array of Structures (AoS) - cache inefficient
bodies: [Body { pos, vel, mass }, Body { pos, vel, mass }, ...]

// Structure of Arrays (SoA) - cache efficient
positions:  [pos0, pos1, pos2, ...]   <- contiguous
velocities: [vel0, vel1, vel2, ...]   <- contiguous
masses:     [m0, m1, m2, ...]         <- contiguous
```

## Vec3: 3D Vectors

```rust,ignore
use simular::engine::state::Vec3;

let a = Vec3::new(1.0, 2.0, 3.0);
let b = Vec3::new(4.0, 5.0, 6.0);

// Operations
let sum = a + b;                    // Vector addition
let diff = a - b;                   // Vector subtraction
let scaled = a * 2.0;               // Scalar multiplication
let neg = -a;                       // Negation

// Vector operations
let dot = a.dot(&b);                // Dot product: 32.0
let cross = a.cross(&b);            // Cross product
let mag = a.magnitude();            // Length: √14
let unit = a.normalize();           // Unit vector

// Utilities
let zero = Vec3::zero();            // (0, 0, 0)
let finite = a.is_finite();         // Check for NaN/Inf
```

## SimRng: Deterministic RNG

Based on PCG (Permuted Congruential Generator):

```rust,ignore
use simular::prelude::*;

let mut rng = SimRng::new(42);  // Master seed

// Generate random values
let u: f64 = rng.gen_f64();              // [0, 1)
let x: f64 = rng.gen_range_f64(-1.0, 1.0); // [-1, 1)
let n: f64 = rng.gen_standard_normal();  // N(0, 1)
let g: f64 = rng.gen_normal(10.0, 2.0);  // N(10, 2)

// Batch sampling
let samples: Vec<f64> = rng.sample_n(1000);

// Parallel partitions (deterministic!)
let partitions = rng.partition(4);
// Each partition produces different but reproducible streams
```

### Partitioning for Parallelism

```rust,ignore
let mut rng = SimRng::new(42);
let mut partitions = rng.partition(4);

// Each thread gets its own deterministic stream
std::thread::scope(|s| {
    for (i, partition) in partitions.iter_mut().enumerate() {
        s.spawn(move || {
            let value = partition.gen_f64();
            println!("Thread {}: {}", i, value);
        });
    }
});

// Running twice gives identical output!
```

## Domain Engines

Simular separates simulation domains:

| Domain | Module | Purpose |
|--------|--------|---------|
| Physics | `domains::physics` | Rigid body dynamics |
| Monte Carlo | `domains::monte_carlo` | Stochastic sampling |
| Optimization | `domains::optimization` | Bayesian optimization |
| ML | `domains::ml` | Training simulation |

Each domain has specialized engines:

```rust,ignore
// Physics
use simular::domains::physics::{PhysicsEngine, GravityField, VerletIntegrator};
let engine = PhysicsEngine::new(GravityField::default(), VerletIntegrator::new());

// Monte Carlo
use simular::domains::monte_carlo::MonteCarloEngine;
let engine = MonteCarloEngine::with_samples(10_000);

// Optimization
use simular::domains::optimization::{BayesianOptimizer, OptimizerConfig};
let optimizer = BayesianOptimizer::new(config);
```

## Jidoka Guards: Anomaly Detection

Automatic detection of simulation failures:

```rust,ignore
use simular::engine::jidoka::{JidokaConfig, JidokaGuard};

let config = JidokaConfig {
    check_finite: true,       // Detect NaN/Inf
    check_energy: true,       // Monitor energy drift
    energy_tolerance: 1e-6,   // Max allowed drift
    ..Default::default()
};

let mut guard = JidokaGuard::new(config);

// Check state after each step
match guard.check(&state) {
    Ok(()) => { /* Continue */ }
    Err(violation) => {
        eprintln!("Jidoka violation: {:?}", violation);
        // Handle error
    }
}
```

## Pre-Built Scenarios

Ready-to-use simulation templates:

```rust,ignore
use simular::scenarios::{
    RocketScenario,      // Launch and staging
    SatelliteScenario,   // Orbital mechanics
    PendulumScenario,    // Classical mechanics
    ClimateScenario,     // Energy balance models
    PortfolioScenario,   // VaR calculations
    SIRScenario,         // Epidemic modeling
};
```

## Architecture Overview

```text
┌─────────────────────────────────────────────────────────┐
│                     User Code                            │
├─────────────────────────────────────────────────────────┤
│  Scenarios    │  Domains      │  Visualization          │
│  - Rocket     │  - Physics    │  - TUI                  │
│  - Satellite  │  - Monte Carlo│  - Web                  │
│  - Pendulum   │  - Optimization                         │
│  - Climate    │  - ML                                   │
├─────────────────────────────────────────────────────────┤
│                    Core Engine                           │
│  SimState  │  SimRng  │  JidokaGuard  │  Scheduler      │
└─────────────────────────────────────────────────────────┘
```

## Next Steps

- [Physics Simulations](./domain_physics.md) - Integrators and force fields
- [Monte Carlo Methods](./domain_monte_carlo.md) - Variance reduction
- [Bayesian Optimization](./domain_optimization.md) - GP-based search
