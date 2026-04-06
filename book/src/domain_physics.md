# Physics Simulations

The physics domain provides numerical integration for classical mechanics simulations.

## Integrators

Simular provides three integrators with different trade-offs:

| Integrator | Order | Symplectic | Use Case |
|------------|-------|------------|----------|
| `VerletIntegrator` | 2 | Yes | Long-term simulations |
| `RK4Integrator` | 4 | No | Short-term accuracy |
| `EulerIntegrator` | 1 | No | Debugging, comparisons |

### Verlet Integrator (Recommended)

Störmer-Verlet is symplectic—it preserves phase space volume, leading to bounded energy error:

```rust,ignore
use simular::domains::physics::VerletIntegrator;

let integrator = VerletIntegrator::new();
assert!(integrator.is_symplectic());  // true
assert_eq!(integrator.error_order(), 2);
```

Algorithm:
```text
q_{n+1/2} = q_n + (h/2) * v_n
v_{n+1}   = v_n + h * a(q_{n+1/2})
q_{n+1}   = q_{n+1/2} + (h/2) * v_{n+1}
```

### RK4 Integrator

Fourth-order Runge-Kutta is more accurate per step but energy may drift:

```rust,ignore
use simular::domains::physics::RK4Integrator;

let integrator = RK4Integrator::new();
assert!(!integrator.is_symplectic());  // false
assert_eq!(integrator.error_order(), 4);
```

Use RK4 when:
- Short simulation duration
- High accuracy per step matters
- Non-conservative systems

### Euler Integrator

First-order, for comparisons and debugging:

```rust,ignore
use simular::domains::physics::EulerIntegrator;

let integrator = EulerIntegrator::new();
assert_eq!(integrator.error_order(), 1);
```

## Force Fields

### Gravity Field

Uniform gravitational acceleration:

```rust,ignore
use simular::domains::physics::GravityField;
use simular::engine::state::Vec3;

// Default: -9.81 m/s² in z direction
let gravity = GravityField::default();

// Custom gravity
let moon_gravity = GravityField {
    g: Vec3::new(0.0, 0.0, -1.62),  // Moon's surface gravity
};
```

### Central Force Field

Inverse-square law (orbital mechanics):

```rust,ignore
use simular::domains::physics::CentralForceField;
use simular::engine::state::Vec3;

// Earth-centered gravity
let earth = CentralForceField::new(
    3.986e14,        // μ = G*M for Earth (m³/s²)
    Vec3::zero(),    // Center at origin
);

// Acceleration: a = -μ/r² * r̂
// Potential: PE = -μ*m/r
```

### Custom Force Fields

Implement the `ForceField` trait:

```rust,ignore
use simular::domains::physics::ForceField;
use simular::engine::state::Vec3;

struct SpringField {
    k: f64,  // Spring constant
}

impl ForceField for SpringField {
    fn acceleration(&self, position: &Vec3, mass: f64) -> Vec3 {
        // F = -kx, a = F/m = -kx/m
        Vec3::new(
            -self.k * position.x / mass,
            -self.k * position.y / mass,
            -self.k * position.z / mass,
        )
    }

    fn potential(&self, position: &Vec3, _mass: f64) -> f64 {
        // PE = 0.5 * k * r²
        0.5 * self.k * position.magnitude_squared()
    }
}
```

## Physics Engine

Combine force field and integrator:

```rust,ignore
use simular::domains::physics::{PhysicsEngine, GravityField, VerletIntegrator};
use simular::engine::state::{SimState, Vec3};

// Create engine
let engine = PhysicsEngine::new(
    GravityField::default(),
    VerletIntegrator::new(),
);

// Create state
let mut state = SimState::new();
state.add_body(1.0, Vec3::new(0.0, 0.0, 100.0), Vec3::new(10.0, 0.0, 0.0));

// Simulate
let dt = 0.001;  // 1ms timestep
for _ in 0..10000 {
    engine.step(&mut state, dt).unwrap();
}
```

## Energy Conservation

Verlet preserves energy better than RK4/Euler over long simulations:

```rust,ignore
use simular::domains::physics::{PhysicsEngine, VerletIntegrator, EulerIntegrator};

fn measure_energy_drift<I: Integrator>(integrator: I, steps: usize) -> f64 {
    let engine = PhysicsEngine::new(spring_field, integrator);
    let mut state = create_harmonic_oscillator();

    engine.step(&mut state, dt).unwrap();
    let initial_energy = state.total_energy();

    for _ in 0..steps {
        engine.step(&mut state, dt).unwrap();
    }

    (state.total_energy() - initial_energy).abs() / initial_energy
}

// After 10,000 steps:
let verlet_drift = measure_energy_drift(VerletIntegrator::new(), 10000);
let euler_drift = measure_energy_drift(EulerIntegrator::new(), 10000);

assert!(verlet_drift < euler_drift);  // Verlet is much better
```

## Example: Harmonic Oscillator

```rust,ignore
use simular::domains::physics::{PhysicsEngine, ForceField, VerletIntegrator};
use simular::engine::state::{SimState, Vec3};

// Spring force: F = -kx
struct HarmonicOscillator { k: f64 }

impl ForceField for HarmonicOscillator {
    fn acceleration(&self, pos: &Vec3, _mass: f64) -> Vec3 {
        Vec3::new(-self.k * pos.x, 0.0, 0.0)
    }

    fn potential(&self, pos: &Vec3, _mass: f64) -> f64 {
        0.5 * self.k * pos.x * pos.x
    }
}

fn main() {
    let engine = PhysicsEngine::new(
        HarmonicOscillator { k: 1.0 },
        VerletIntegrator::new(),
    );

    let mut state = SimState::new();
    state.add_body(1.0, Vec3::new(1.0, 0.0, 0.0), Vec3::zero());

    let dt = 0.01;
    let period = 2.0 * std::f64::consts::PI;  // ω = √(k/m) = 1

    // Simulate one period
    let steps = (period / dt) as usize;
    for _ in 0..steps {
        engine.step(&mut state, dt).unwrap();
    }

    // Should return near initial position
    let pos = state.positions()[0];
    println!("Position after one period: ({:.4}, {:.4}, {:.4})",
        pos.x, pos.y, pos.z);
    // Expected: close to (1.0, 0.0, 0.0)
}
```

## Example: Projectile Motion

```rust,ignore
use simular::domains::physics::{PhysicsEngine, GravityField, VerletIntegrator};
use simular::engine::state::{SimState, Vec3};

fn main() {
    let engine = PhysicsEngine::new(
        GravityField::default(),
        VerletIntegrator::new(),
    );

    let mut state = SimState::new();

    // Launch at 45° with speed 100 m/s
    let angle = 45.0_f64.to_radians();
    let speed = 100.0;
    state.add_body(
        1.0,
        Vec3::zero(),
        Vec3::new(speed * angle.cos(), 0.0, speed * angle.sin()),
    );

    let dt = 0.01;
    let mut max_height = 0.0;

    while state.positions()[0].z >= 0.0 {
        engine.step(&mut state, dt).unwrap();
        max_height = max_height.max(state.positions()[0].z);
    }

    let range = state.positions()[0].x;
    println!("Max height: {:.2} m", max_height);
    println!("Range: {:.2} m", range);

    // Analytical: R = v²sin(2θ)/g ≈ 1020m, H = v²sin²(θ)/(2g) ≈ 255m
}
```

## Choosing an Integrator

| Scenario | Recommended | Reason |
|----------|-------------|--------|
| Orbital mechanics | Verlet | Energy conservation critical |
| Molecular dynamics | Verlet | Long simulation times |
| Game physics | Euler/Verlet | Speed vs accuracy trade-off |
| Trajectory optimization | RK4 | Short-term accuracy |
| Educational | Euler | Simple to understand |

## Next Steps

- [Monte Carlo Methods](./domain_monte_carlo.md) - Stochastic simulation
- [Rocket Launch Scenario](./scenario_rocket.md) - Multi-stage rockets
- [Satellite Orbits](./scenario_satellite.md) - Orbital mechanics
