# Pendulum Systems Scenario

The pendulum scenario provides classical mechanics simulation for pendulum systems.

## Simple Pendulum

```rust,ignore
use simular::scenarios::{PendulumScenario, PendulumConfig};

let config = PendulumConfig {
    length: 1.0,           // m
    mass: 1.0,             // kg
    gravity: 9.81,         // m/s²
    initial_angle: 0.5,    // radians (~29°)
    initial_angular_velocity: 0.0,
    damping: 0.0,          // No friction
};

let mut scenario = PendulumScenario::new(config);
```

## Running the Simulation

```rust,ignore
let dt = 0.01;  // 10ms timestep

for _ in 0..1000 {  // 10 seconds
    scenario.step(dt)?;

    let state = scenario.state();
    println!("t={:.2}s: θ={:.4} rad, ω={:.4} rad/s",
        state.time,
        state.angle,
        state.angular_velocity);
}
```

## Small Angle Approximation

For small angles, the pendulum period is:

T = 2π√(L/g)

```rust,ignore
let config = PendulumConfig {
    length: 1.0,
    gravity: 9.81,
    initial_angle: 0.1,  // ~6° - small angle
    ..Default::default()
};

let scenario = PendulumScenario::new(config);
let period = scenario.period();

// Compare to analytical
let analytical_period = 2.0 * std::f64::consts::PI * (1.0 / 9.81_f64).sqrt();
println!("Simulated: {:.4}s, Analytical: {:.4}s", period, analytical_period);
```

## Large Angle Pendulum

For large angles, the period depends on amplitude:

```rust,ignore
let config = PendulumConfig {
    initial_angle: std::f64::consts::PI / 2.0,  // 90°
    ..Default::default()
};

let scenario = PendulumScenario::new(config);
let period = scenario.period();

// Period is longer than small-angle approximation
println!("Large angle period: {:.4}s", period);
```

## Damped Pendulum

```rust,ignore
let config = PendulumConfig {
    damping: 0.1,  // Damping coefficient
    ..Default::default()
};

let mut scenario = PendulumScenario::new(config);

// Energy decays over time
let initial_energy = scenario.state().total_energy;

for _ in 0..10000 {
    scenario.step(0.01)?;
}

let final_energy = scenario.state().total_energy;
println!("Energy decay: {:.2}%",
    (1.0 - final_energy / initial_energy) * 100.0);
```

## Driven Pendulum

```rust,ignore
let config = PendulumConfig {
    driving: DrivingConfig {
        amplitude: 0.5,     // Driving amplitude
        frequency: 1.0,     // Driving frequency (Hz)
        phase: 0.0,
    },
    damping: 0.1,
    ..Default::default()
};

let scenario = PendulumScenario::new(config);
// Can exhibit chaotic behavior for certain parameters
```

## Double Pendulum

```rust,ignore
use simular::scenarios::DoublePendulumScenario;

let config = DoublePendulumConfig {
    length1: 1.0,
    length2: 1.0,
    mass1: 1.0,
    mass2: 1.0,
    initial_angle1: std::f64::consts::PI / 2.0,
    initial_angle2: std::f64::consts::PI / 2.0,
    ..Default::default()
};

let mut scenario = DoublePendulumScenario::new(config);

// Double pendulum is chaotic
for _ in 0..10000 {
    scenario.step(0.001)?;

    let state = scenario.state();
    let (x1, y1) = state.position1();
    let (x2, y2) = state.position2();
    println!("{:.4},{:.4},{:.4},{:.4}", x1, y1, x2, y2);
}
```

## Energy Analysis

```rust,ignore
let mut scenario = PendulumScenario::new(config);

for _ in 0..1000 {
    scenario.step(0.01)?;

    let state = scenario.state();
    println!("KE={:.4}, PE={:.4}, Total={:.4}",
        state.kinetic_energy,
        state.potential_energy,
        state.total_energy);
}

// For undamped pendulum, total energy should be conserved
```

## Phase Space

```rust,ignore
let mut scenario = PendulumScenario::new(config);

// Generate phase space trajectory
let mut phase_space = Vec::new();

for _ in 0..10000 {
    scenario.step(0.01)?;

    let state = scenario.state();
    phase_space.push((state.angle, state.angular_velocity));
}

// Export for plotting
for (theta, omega) in phase_space {
    println!("{:.4},{:.4}", theta, omega);
}
```

## Example: Period vs Amplitude

```rust,ignore
use simular::scenarios::{PendulumScenario, PendulumConfig};

fn main() {
    println!("Amplitude (°) | Period (s) | Deviation (%)");
    println!("--------------|------------|---------------");

    let small_angle_period = 2.0 * std::f64::consts::PI;  // L=g=1

    for angle_deg in [5, 15, 30, 45, 60, 75, 90] {
        let angle_rad = (angle_deg as f64).to_radians();

        let config = PendulumConfig {
            length: 1.0,
            gravity: 1.0,  // Normalized
            initial_angle: angle_rad,
            ..Default::default()
        };

        let scenario = PendulumScenario::new(config);
        let period = scenario.period();
        let deviation = (period / small_angle_period - 1.0) * 100.0;

        println!("{:>13} | {:>10.4} | {:>13.2}",
            angle_deg, period, deviation);
    }
}
```

Output:
```text
Amplitude (°) | Period (s) | Deviation (%)
--------------|------------|---------------
            5 |     6.2832 |          0.05
           15 |     6.3051 |          0.40
           30 |     6.3706 |          1.39
           45 |     6.4804 |          3.14
           60 |     6.6372 |          5.64
           75 |     6.8502 |          9.03
           90 |     7.1344 |         13.55
```

## Next Steps

- [Physics Simulations](./domain_physics.md) - Underlying physics
- [Climate Models](./scenario_climate.md) - Energy balance systems
