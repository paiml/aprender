# Quick Start

Get running with Simular in 5 minutes.

## Installation

```toml
[dependencies]
simular = "0.2"
```

## Your First Simulation: Pi Estimation

```rust,ignore
use simular::prelude::*;
use simular::domains::monte_carlo::MonteCarloEngine;

fn main() {
    // Deterministic RNG - same seed = same results
    let mut rng = SimRng::new(42);

    // Monte Carlo engine with 100,000 samples
    let engine = MonteCarloEngine::with_samples(100_000);

    // Estimate pi using quarter circle method
    let result = engine.run_nd(2, |x| {
        if x[0] * x[0] + x[1] * x[1] <= 1.0 {
            4.0  // Inside quarter circle
        } else {
            0.0  // Outside
        }
    }, &mut rng);

    println!("π ≈ {:.6} ± {:.6}", result.estimate, result.std_error);
    println!("95% CI: ({:.6}, {:.6})",
        result.confidence_interval.0,
        result.confidence_interval.1);
}
```

Output:
```text
π ≈ 3.141592 ± 0.001634
95% CI: (3.138389, 3.144795)
```

## Physics Simulation: Projectile Motion

```rust,ignore
use simular::prelude::*;
use simular::domains::physics::{PhysicsEngine, GravityField, VerletIntegrator};
use simular::engine::state::{SimState, Vec3};

fn main() {
    // Create simulation state
    let mut state = SimState::new();

    // Add a projectile: mass=1kg, position=(0,0,100m), velocity=(10,0,20)m/s
    state.add_body(
        1.0,                           // mass
        Vec3::new(0.0, 0.0, 100.0),    // initial position
        Vec3::new(10.0, 0.0, 20.0),    // initial velocity
    );

    // Physics engine with gravity and Verlet integrator
    let engine = PhysicsEngine::new(
        GravityField::default(),  // -9.81 m/s² in z
        VerletIntegrator::new(),  // Symplectic, energy-conserving
    );

    // Simulate until projectile hits ground
    let dt = 0.01;  // 10ms timestep
    while state.positions()[0].z > 0.0 {
        engine.step(&mut state, dt).unwrap();
    }

    let final_pos = state.positions()[0];
    println!("Landing position: ({:.2}, {:.2}, {:.2})",
        final_pos.x, final_pos.y, final_pos.z);
}
```

## Bayesian Optimization

```rust,ignore
use simular::domains::optimization::{
    BayesianOptimizer, OptimizerConfig, AcquisitionFunction,
};

fn main() {
    // Rosenbrock function: minimum at (1, 1)
    let rosenbrock = |x: &[f64]| -> f64 {
        (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2)
    };

    let config = OptimizerConfig {
        bounds: vec![(-2.0, 2.0), (-2.0, 2.0)],
        acquisition: AcquisitionFunction::ExpectedImprovement,
        seed: 42,
        ..Default::default()
    };

    let mut optimizer = BayesianOptimizer::new(config);

    // Run 20 iterations
    for _ in 0..20 {
        let x = optimizer.suggest();
        let y = rosenbrock(&x);
        optimizer.observe(x, y).unwrap();
    }

    let (best_x, best_y) = optimizer.best().unwrap();
    println!("Best: ({:.4}, {:.4}) with f(x) = {:.6}",
        best_x[0], best_x[1], best_y);
}
```

## Running Examples

```bash
# Monte Carlo pi estimation
cargo run --example monte_carlo

# Physics simulation
cargo run --example physics_simulation

# Bayesian optimization
cargo run --example optimization

# Reproducibility demo
cargo run --example reproducibility

# Jidoka guards
cargo run --example jidoka_guards

# TSP GRASP demo
cargo run --example tsp_grasp_demo

# Orbit demo
cargo run --example orbit_demo

# EDD demos
cargo run --example edd_yaml_loader
cargo run --example edd_operations
cargo run --example edd_falsification
```

## Interactive TUI Demos

```bash
# TSP GRASP TUI
cargo run --features tui --bin tsp-tui

# Orbit TUI
cargo run --features tui --bin orbit-tui
```

## WASM Demos

```bash
# Serve TSP WASM demo (auto-opens browser)
make serve-tsp

# Serve Orbit WASM demo (auto-opens browser)
make serve-orbit
```

## Key Concepts

| Concept | Description |
|---------|-------------|
| `SimRng` | Deterministic random number generator |
| `SimState` | Simulation state (positions, velocities, masses) |
| `PhysicsEngine` | Physics simulation with integrators |
| `MonteCarloEngine` | Stochastic sampling with variance reduction |
| `BayesianOptimizer` | Gaussian process optimization |
| `JidokaGuard` | Automatic anomaly detection |

## Next Steps

- [Core Concepts](./core_concepts.md) - Understand the architecture
- [Physics Simulations](./domain_physics.md) - Deep dive into physics
- [Monte Carlo Methods](./domain_monte_carlo.md) - Variance reduction techniques
