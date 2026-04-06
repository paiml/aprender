# Jidoka Guards

Jidoka (自働化) means "automation with a human touch"—machines that detect problems and stop automatically.

## Overview

Jidoka guards detect anomalies during simulation:

1. **Non-finite values**: NaN or Infinity
2. **Energy drift**: Conservation law violations
3. **Constraint violations**: Physical limits exceeded

## JidokaConfig

```rust,ignore
use simular::engine::jidoka::{JidokaConfig, SeverityClassifier};

let config = JidokaConfig {
    // Detect NaN and Infinity
    check_finite: true,

    // Monitor energy conservation
    check_energy: true,
    energy_tolerance: 1e-6,

    // Check constraint violations
    constraint_tolerance: 0.01,

    // Graduated severity response
    severity_classifier: SeverityClassifier::default(),
};
```

## JidokaGuard

```rust,ignore
use simular::engine::jidoka::{JidokaConfig, JidokaGuard};
use simular::engine::state::SimState;

let config = JidokaConfig::default();
let mut guard = JidokaGuard::new(config);

// Check state after simulation step
match guard.check(&state) {
    Ok(()) => {
        // Continue simulation
    }
    Err(violation) => {
        eprintln!("Jidoka violation: {:?}", violation);
        // Handle error
    }
}
```

## Violation Types

### NonFiniteValue

Detects NaN or Infinity:

```rust,ignore
JidokaViolation::NonFiniteValue {
    location: String,  // e.g., "position.x", "velocity[0].z"
    value: f64,        // The non-finite value
}
```

### EnergyDrift

Detects energy conservation violations:

```rust,ignore
JidokaViolation::EnergyDrift {
    current: f64,     // Current total energy
    initial: f64,     // Initial total energy
    drift: f64,       // Relative drift |E - E₀| / |E₀|
    tolerance: f64,   // Configured tolerance
}
```

### ConstraintViolation

Detects physical constraint violations:

```rust,ignore
JidokaViolation::ConstraintViolation {
    name: String,     // Constraint name
    violation: f64,   // Violation amount
    tolerance: f64,   // Configured tolerance
}
```

## Severity Levels

Graduated response to prevent false positives:

```rust,ignore
pub enum ViolationSeverity {
    Acceptable,  // Within tolerance, continue
    Warning,     // Approaching tolerance, log and continue
    Critical,    // Exceeded tolerance, stop
    Fatal,       // Unrecoverable, halt immediately
}
```

### SeverityClassifier

```rust,ignore
use simular::engine::jidoka::SeverityClassifier;

let classifier = SeverityClassifier::new(0.8);  // Warn at 80% of tolerance

// Classify energy drift
let severity = classifier.classify_energy_drift(
    0.0009,  // Current drift
    0.001,   // Tolerance
);
// Returns Warning (90% of tolerance)

// Classify constraint
let severity = classifier.classify_constraint(
    0.015,  // Violation
    0.01,   // Tolerance
);
// Returns Critical (150% of tolerance)
```

## Warnings

Non-critical issues that don't stop simulation:

```rust,ignore
pub enum JidokaWarning {
    EnergyDriftApproaching {
        drift: f64,
        tolerance: f64,
    },
    ConstraintApproaching {
        name: String,
        violation: f64,
        tolerance: f64,
    },
}
```

## Example: Physics with Jidoka

```rust,ignore
use simular::engine::jidoka::{JidokaConfig, JidokaGuard};
use simular::domains::physics::{PhysicsEngine, GravityField, VerletIntegrator};
use simular::engine::state::{SimState, Vec3};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configure Jidoka
    let jidoka_config = JidokaConfig {
        check_finite: true,
        check_energy: true,
        energy_tolerance: 1e-6,
        ..Default::default()
    };
    let mut guard = JidokaGuard::new(jidoka_config);

    // Setup simulation
    let engine = PhysicsEngine::new(
        GravityField::default(),
        VerletIntegrator::new(),
    );

    let mut state = SimState::new();
    state.add_body(1.0, Vec3::new(0.0, 0.0, 100.0), Vec3::new(10.0, 0.0, 0.0));

    // Initialize guard with initial energy
    guard.set_initial_energy(state.total_energy());

    // Simulation loop with Jidoka checks
    let dt = 0.001;
    for step in 0..10000 {
        engine.step(&mut state, dt)?;

        // Jidoka check after every step
        match guard.check(&state) {
            Ok(()) => {}
            Err(violation) => {
                eprintln!("Step {}: {:?}", step, violation);
                return Err(violation.into());
            }
        }
    }

    println!("Simulation completed successfully");
    Ok(())
}
```

## Example: Detecting NaN

```rust,ignore
let mut state = SimState::new();
state.add_body(1.0, Vec3::new(0.0, 0.0, 0.0), Vec3::zero());

// Introduce NaN
state.set_position(0, Vec3::new(f64::NAN, 0.0, 0.0));

let guard = JidokaGuard::new(JidokaConfig::default());
match guard.check(&state) {
    Ok(()) => println!("State OK"),
    Err(JidokaViolation::NonFiniteValue { location, value }) => {
        println!("NaN detected at {}: {}", location, value);
    }
    Err(e) => println!("Other violation: {:?}", e),
}
```

## Example: Energy Monitoring

```rust,ignore
let config = JidokaConfig {
    check_energy: true,
    energy_tolerance: 0.01,  // 1% drift allowed
    ..Default::default()
};

let mut guard = JidokaGuard::new(config);
guard.set_initial_energy(100.0);  // E₀ = 100

// After simulation...
state.set_potential_energy(-50.0);  // Simulate energy drift

match guard.check(&state) {
    Ok(()) => println!("Energy OK"),
    Err(JidokaViolation::EnergyDrift { drift, .. }) => {
        println!("Energy drifted by {:.2}%", drift * 100.0);
    }
    _ => {}
}
```

## Strict vs Lenient Configs

```rust,ignore
// Strict (for scientific simulations)
let strict = JidokaConfig {
    check_finite: true,
    check_energy: true,
    energy_tolerance: 1e-9,
    constraint_tolerance: 1e-6,
    severity_classifier: SeverityClassifier::new(0.5),  // Warn at 50%
};

// Lenient (for games/visualizations)
let lenient = JidokaConfig {
    check_finite: true,
    check_energy: false,  // Don't check energy
    energy_tolerance: 0.1,
    constraint_tolerance: 0.1,
    severity_classifier: SeverityClassifier::new(0.95),  // Warn at 95%
};
```

## Best Practices

1. **Check Every Step**: Run guards after every simulation step
2. **Fail Fast**: Don't let bad state propagate
3. **Log Warnings**: Track approaching violations
4. **Tune Tolerances**: Start strict, relax based on domain

## Integration Pattern

```rust,ignore
fn run_simulation_with_guards(
    engine: &PhysicsEngine,
    state: &mut SimState,
    dt: f64,
    steps: usize,
) -> Result<(), JidokaViolation> {
    let config = JidokaConfig::strict();
    let mut guard = JidokaGuard::new(config);
    guard.set_initial_energy(state.total_energy());

    for _ in 0..steps {
        engine.step(state, dt)?;
        guard.check(state)?;
    }

    Ok(())
}
```

## Next Steps

- [Deterministic RNG](./engine_rng.md) - Reproducible randomness
- [Replay System](./engine_replay.md) - Record and replay simulations
