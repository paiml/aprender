# Rocket Launch Scenario

The rocket scenario provides multi-stage launch simulation.

## Basic Usage

```rust,ignore
use simular::scenarios::{RocketScenario, RocketConfig, StageSeparation};

let config = RocketConfig {
    stages: vec![
        // First stage
        StageConfig {
            dry_mass: 20000.0,      // kg
            fuel_mass: 400000.0,    // kg
            thrust: 7_000_000.0,    // N
            isp: 280.0,             // s (sea level)
            burn_time: 180.0,       // s
        },
        // Second stage
        StageConfig {
            dry_mass: 4000.0,
            fuel_mass: 100000.0,
            thrust: 1_000_000.0,
            isp: 350.0,             // Vacuum Isp
            burn_time: 300.0,
        },
    ],
    payload_mass: 10000.0,
    launch_site: LaunchSite::CapeCanaveral,
    target_orbit: OrbitTarget::LEO { altitude: 400_000.0 },
};

let mut scenario = RocketScenario::new(config);
```

## Running the Simulation

```rust,ignore
let dt = 0.1;  // 100ms timestep

while !scenario.is_complete() {
    scenario.step(dt)?;

    let state = scenario.state();
    println!("T+{:.1}s: altitude={:.1}km, velocity={:.1}m/s",
        state.time,
        state.altitude / 1000.0,
        state.velocity.magnitude());
}

let result = scenario.result();
println!("Final orbit: {}km x {}km",
    result.periapsis / 1000.0,
    result.apoapsis / 1000.0);
```

## Stage Separation

```rust,ignore
// Automatic separation based on fuel depletion
let config = RocketConfig {
    separation: StageSeparation::Automatic,
    ..Default::default()
};

// Or manual separation at specific times
let config = RocketConfig {
    separation: StageSeparation::Manual {
        times: vec![180.0, 480.0],  // Separate at T+180s and T+480s
    },
    ..Default::default()
};
```

## Gravity Turn

```rust,ignore
let config = RocketConfig {
    gravity_turn: GravityTurnConfig {
        start_altitude: 1000.0,     // Start turn at 1km
        end_altitude: 100_000.0,    // End turn at 100km
        initial_pitch: 90.0,        // Start vertical
        final_pitch: 0.0,           // End horizontal
    },
    ..Default::default()
};
```

## Atmospheric Model

```rust,ignore
let config = RocketConfig {
    atmosphere: AtmosphereModel::Standard1976,
    drag_coefficient: 0.3,
    reference_area: 10.0,  // m²
    ..Default::default()
};
```

## Example: Falcon 9-like Vehicle

```rust,ignore
use simular::scenarios::{RocketScenario, RocketConfig};

fn main() {
    let config = RocketConfig {
        stages: vec![
            StageConfig {
                dry_mass: 22_200.0,
                fuel_mass: 395_700.0,
                thrust: 7_607_000.0,  // 9 Merlin engines
                isp: 282.0,
                burn_time: 162.0,
            },
            StageConfig {
                dry_mass: 4_000.0,
                fuel_mass: 107_500.0,
                thrust: 934_000.0,  // 1 Merlin Vacuum
                isp: 348.0,
                burn_time: 397.0,
            },
        ],
        payload_mass: 22_800.0,  // Max to LEO
        ..Default::default()
    };

    let mut scenario = RocketScenario::new(config);

    let mut max_q = 0.0;
    let mut max_q_time = 0.0;

    while !scenario.is_complete() {
        scenario.step(0.1)?;

        let state = scenario.state();
        if state.dynamic_pressure > max_q {
            max_q = state.dynamic_pressure;
            max_q_time = state.time;
        }
    }

    println!("Max Q: {:.0} Pa at T+{:.1}s", max_q, max_q_time);

    let result = scenario.result();
    println!("Orbit achieved: {:.0}km x {:.0}km",
        result.periapsis / 1000.0,
        result.apoapsis / 1000.0);
}
```

## Telemetry

```rust,ignore
let scenario = RocketScenario::new(config);

// Get telemetry recorder
let telemetry = scenario.telemetry();

// After simulation
for point in telemetry.iter() {
    println!("{:.1},{:.1},{:.1},{:.1}",
        point.time,
        point.altitude,
        point.velocity,
        point.acceleration);
}

// Export to CSV
telemetry.export_csv("launch_telemetry.csv")?;
```

## Abort Conditions

```rust,ignore
let config = RocketConfig {
    abort_conditions: AbortConditions {
        max_g_load: 6.0,           // Abort if > 6g
        max_dynamic_pressure: 50_000.0,  // Pa
        min_throttle: 0.5,         // Abort if < 50% thrust
    },
    ..Default::default()
};
```

## Next Steps

- [Satellite Orbits](./scenario_satellite.md) - Orbital mechanics
- [Physics Simulations](./domain_physics.md) - Underlying physics
