# Introduction

**Simular** is a unified simulation engine for the Sovereign AI Stack. It provides deterministic, reproducible simulations across multiple domains:

- **Physics**: Rigid body dynamics, orbital mechanics
- **Monte Carlo**: Stochastic sampling with variance reduction
- **Optimization**: Bayesian optimization with Gaussian processes
- **ML**: Training simulation and multi-turn evaluation

## Key Features

| Feature | Description |
|---------|-------------|
| **Deterministic** | Same seed = identical results, bit-for-bit |
| **Jidoka Guards** | Automatic anomaly detection (NaN, energy drift) |
| **Pre-built Scenarios** | Rocket, satellite, pendulum, climate, portfolio, epidemic |
| **Replay System** | Record and replay any simulation |

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
simular = "0.2"
```

## Quick Example

```rust,ignore
use simular::prelude::*;
use simular::domains::monte_carlo::{MonteCarloEngine, VarianceReduction};

fn main() {
    // Deterministic RNG
    let mut rng = SimRng::seed_from_u64(42);

    // Monte Carlo pi estimation
    let engine = MonteCarloEngine::with_samples(100_000);
    let result = engine.run_nd(2, |x| {
        if x[0]*x[0] + x[1]*x[1] <= 1.0 { 1.0 } else { 0.0 }
    }, &mut rng);

    let pi = result.estimate * 4.0;
    println!("π ≈ {:.6}", pi);
}
```

## Architecture

```text
simular/
├── engine/          # Core simulation engine
│   ├── state.rs     # SimState, Vec3
│   ├── rng.rs       # Deterministic RNG
│   ├── jidoka.rs    # Anomaly detection
│   └── scheduler.rs # Work-stealing scheduler
├── domains/         # Simulation types
│   ├── physics.rs   # Integrators, force fields
│   ├── monte_carlo.rs
│   ├── optimization.rs
│   └── ml.rs
└── scenarios/       # Ready-to-use templates
    ├── rocket.rs
    ├── satellite.rs
    ├── pendulum.rs
    ├── climate.rs
    ├── portfolio.rs
    └── epidemic.rs
```

## Next Steps

- [Quick Start](./quick_start.md) - Run your first simulation
- [Core Concepts](./core_concepts.md) - Understand the architecture
