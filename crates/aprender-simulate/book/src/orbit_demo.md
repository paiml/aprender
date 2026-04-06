# Orbit Demo

The `simular::orbit` module provides a canonical visual demonstration of orbital mechanics simulation, following Toyota Way and NASA/JPL principles.

## Features

- **Type-safe units (Poka-Yoke)**: Prevents dimensional errors at compile time using the `uom` crate
- **Yoshida 4th-order symplectic integration**: Long-term energy conservation (<1e-9 drift over 100 orbits)
- **Jidoka guards**: Graceful degradation on physics violations (pause, don't crash)
- **Heijunka scheduling**: Time-budget management for consistent frame delivery
- **Multiple scenarios**: Kepler orbits, N-body, Hohmann transfers, Lagrange points

## Quick Start

```rust,ignore
use simular::orbit::prelude::*;
use simular::orbit::physics::YoshidaIntegrator;

// Create Earth-Sun system
let config = KeplerConfig::earth_sun();
let mut state = config.build(1e6); // 1000km softening

// Initialize Jidoka guard
let mut jidoka = OrbitJidokaGuard::new(OrbitJidokaConfig::default());
jidoka.initialize(&state);

// Create integrator
let yoshida = YoshidaIntegrator::new();
let dt = OrbitTime::from_seconds(3600.0); // 1 hour steps

// Simulate one year
for _ in 0..(365 * 24) {
    yoshida.step(&mut state, dt).expect("integration failed");

    let response = jidoka.check(&state);
    if !response.can_continue() {
        break; // Jidoka triggered - graceful pause
    }
}

println!("Energy error: {:.2e}",
    (state.total_energy() - initial_energy).abs() / initial_energy.abs());
```

## Type-Safe Units (Poka-Yoke)

The module uses type-safe units to prevent dimensional errors:

```rust,ignore
use simular::orbit::prelude::*;

// Positions in meters
let earth_pos = Position3D::from_au(1.0, 0.0, 0.0);  // 1 AU from Sun
let _meters = earth_pos.as_meters();  // (1.496e11, 0, 0)

// Velocities in m/s
let earth_vel = Velocity3D::from_mps(0.0, 29780.0, 0.0);  // 29.78 km/s

// Masses
let sun_mass = OrbitMass::from_solar_masses(1.0);
let earth_mass = OrbitMass::from_kg(5.972e24);

// Time
let one_day = OrbitTime::from_days(1.0);
let one_hour = OrbitTime::from_seconds(3600.0);

// Physical constants
assert!((AU - 1.496e11).abs() < 1e8);  // Astronomical Unit
assert!((G - 6.674e-11).abs() < 1e-14);  // Gravitational constant
```

## Scenarios

### Kepler Two-Body Orbit

```rust,ignore
let config = KeplerConfig::earth_sun();
println!("Orbital period: {:.2} days", config.period() / 86400.0);
println!("Circular velocity: {:.2} km/s", config.circular_velocity() / 1000.0);

let state = config.build(1e6);
```

### N-Body Solar System

```rust,ignore
let config = NBodyConfig::inner_solar_system();
// Contains: Sun, Mercury, Venus, Earth, Mars

let state = config.build(1e9);  // Larger softening for N-body
```

### Hohmann Transfer

```rust,ignore
let config = HohmannConfig::earth_to_mars();

println!("Delta-v1: {:.2} km/s", config.delta_v1() / 1000.0);
println!("Delta-v2: {:.2} km/s", config.delta_v2() / 1000.0);
println!("Transfer time: {:.0} days", config.transfer_time() / 86400.0);
```

### Lagrange Points

```rust,ignore
let config = LagrangeConfig::sun_earth_l2();
let (lx, ly, lz) = config.lagrange_position();

println!("L2 distance from Sun: {:.4} AU",
    (lx*lx + ly*ly + lz*lz).sqrt() / AU);
```

## Jidoka Guards

Jidoka provides graceful degradation when physics violations occur:

```rust,ignore
let mut jidoka = OrbitJidokaGuard::new(OrbitJidokaConfig {
    energy_tolerance: 1e-9,
    angular_momentum_tolerance: 1e-12,
    ..OrbitJidokaConfig::default()
});

jidoka.initialize(&state);

// After each step
let response = jidoka.check(&state);
match response {
    JidokaResponse::Continue => { /* All good */ }
    JidokaResponse::Warning { message, .. } => {
        println!("Warning: {}", message);
    }
    JidokaResponse::Pause { reason, .. } => {
        println!("Pausing: {:?}", reason);
        // Save state, notify user, allow recovery
    }
    JidokaResponse::Halt { reason, .. } => {
        println!("Critical: {:?}", reason);
        // Fatal error, must restart
    }
}
```

## Heijunka Scheduling

Heijunka provides time-budget management for consistent frame delivery:

```rust,ignore
let config = HeijunkaConfig {
    frame_budget_ms: 16.67,  // 60 FPS target
    physics_budget_fraction: 0.8,
    base_dt: 3600.0,  // 1 hour base timestep
    max_substeps: 24,
    ..HeijunkaConfig::default()
};

let mut scheduler = HeijunkaScheduler::new(config);

// Each frame
if let Ok(result) = scheduler.execute_frame(&mut state) {
    println!("Simulated {:.2}s", result.sim_time_advanced);

    let status = scheduler.status();
    if status.utilization > 1.0 {
        println!("Budget exceeded!");
    }
}
```

## TUI Visualization

Run the interactive terminal UI:

```bash
cargo run --features tui --bin orbit-tui
```

Controls:
- **Space**: Pause/Resume
- **R**: Reset simulation
- **+/-**: Adjust time scale
- **Q**: Quit

## WASM Demo

Run the interactive web demo:

```bash
make serve-orbit
```

This serves the demo at `http://localhost:8080/index.html` and auto-opens your browser.

### Building WASM from Source

```bash
cargo build --lib --target wasm32-unknown-unknown --features wasm --release
```

Bundle size: ~97KB gzipped (well under 500KB limit).

## Metamorphic Testing

The module includes metamorphic tests that verify physics invariants:

1. **Rotation Invariance**: System rotation preserves dynamics
2. **Time-Reversal Symmetry**: Symplectic integrators are reversible
3. **Energy Conservation**: Bounded energy oscillation
4. **Angular Momentum Conservation**: Exact preservation
5. **Deterministic Replay**: Bit-identical results

```rust,ignore
use simular::orbit::metamorphic::run_all_metamorphic_tests;

let state = KeplerConfig::earth_sun().build(1e6);
let results = run_all_metamorphic_tests(&state, 100, 3600.0);

for result in results {
    assert!(result.passed, "{} failed: {}", result.relation, result.details);
}
```

## Example

Run the complete orbit demo:

```bash
cargo run --example orbit_demo
```

Output:
```text
=== Simular Orbit Demo ===

1. Earth-Sun Keplerian Orbit:
   Semi-major axis: 1.50e11 m (1 AU)
   Eccentricity: 0.0167
   Orbital period: 365.21 days

2. Yoshida 4th-Order Symplectic Integration:
   Simulated 1 year (8760 steps)
   Energy drift: 3.70e-14 (target: <1e-9)
   Angular momentum drift: 1.78e-14 (target: <1e-12)

...
```
