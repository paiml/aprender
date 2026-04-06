# Satellite Orbits Scenario

The satellite scenario provides orbital mechanics simulation using Keplerian elements.

## Basic Usage

```rust,ignore
use simular::scenarios::{SatelliteScenario, OrbitalElements};

// Define orbit using Keplerian elements
let elements = OrbitalElements {
    semi_major_axis: 6_778_000.0,  // m (400km altitude)
    eccentricity: 0.001,           // Nearly circular
    inclination: 51.6_f64.to_radians(),  // ISS inclination
    raan: 0.0,                     // Right ascension of ascending node
    argument_of_periapsis: 0.0,
    true_anomaly: 0.0,
};

let scenario = SatelliteScenario::new(elements);
```

## Running the Simulation

```rust,ignore
let dt = 60.0;  // 1 minute timestep

for _ in 0..90 {  // ~1 orbit for ISS
    scenario.step(dt)?;

    let state = scenario.state();
    println!("Position: ({:.0}, {:.0}, {:.0}) km",
        state.position.x / 1000.0,
        state.position.y / 1000.0,
        state.position.z / 1000.0);
    println!("Altitude: {:.1} km", state.altitude / 1000.0);
    println!("Velocity: {:.1} m/s", state.velocity.magnitude());
}
```

## Orbital Elements

```rust,ignore
pub struct OrbitalElements {
    /// Semi-major axis (m)
    pub semi_major_axis: f64,

    /// Eccentricity (0 = circular, 0-1 = elliptical)
    pub eccentricity: f64,

    /// Inclination (radians)
    pub inclination: f64,

    /// Right Ascension of Ascending Node (radians)
    pub raan: f64,

    /// Argument of periapsis (radians)
    pub argument_of_periapsis: f64,

    /// True anomaly at epoch (radians)
    pub true_anomaly: f64,
}
```

## Common Orbits

```rust,ignore
// Low Earth Orbit (ISS)
let leo = OrbitalElements::leo(400_000.0);  // 400km altitude

// Geostationary
let geo = OrbitalElements::geostationary();

// Sun-synchronous
let sso = OrbitalElements::sun_synchronous(700_000.0);  // 700km

// Molniya
let molniya = OrbitalElements::molniya();
```

## Orbital Maneuvers

### Hohmann Transfer

```rust,ignore
use simular::scenarios::OrbitalManeuver;

let initial = OrbitalElements::leo(400_000.0);
let target = OrbitalElements::leo(800_000.0);

let transfer = OrbitalManeuver::hohmann_transfer(&initial, &target);
println!("Delta-v: {:.1} m/s", transfer.total_delta_v);
println!("Transfer time: {:.1} hours", transfer.duration / 3600.0);
```

### Plane Change

```rust,ignore
let maneuver = OrbitalManeuver::plane_change(
    &current_orbit,
    52.0_f64.to_radians(),  // New inclination
);
println!("Delta-v for plane change: {:.1} m/s", maneuver.delta_v);
```

## Perturbations

```rust,ignore
let config = SatelliteConfig {
    perturbations: Perturbations {
        j2: true,           // Earth oblateness
        atmospheric_drag: true,
        solar_radiation: true,
        third_body_moon: true,
        third_body_sun: true,
    },
    spacecraft: SpacecraftConfig {
        mass: 1000.0,       // kg
        drag_area: 10.0,    // m²
        drag_coefficient: 2.2,
        srp_area: 15.0,     // m²
        reflectivity: 0.8,
    },
    ..Default::default()
};
```

## Ground Track

```rust,ignore
let scenario = SatelliteScenario::new(elements);

// Generate ground track
let ground_track = scenario.ground_track(
    start_time,
    duration: 24.0 * 3600.0,  // 24 hours
    step: 60.0,               // 1 minute intervals
);

for point in ground_track {
    println!("{:.4}, {:.4}",  // lat, lon
        point.latitude.to_degrees(),
        point.longitude.to_degrees());
}
```

## Eclipse Prediction

```rust,ignore
let scenario = SatelliteScenario::new(elements);

let eclipses = scenario.predict_eclipses(
    start_time,
    duration: 24.0 * 3600.0,
);

for eclipse in eclipses {
    println!("Eclipse: {} to {} ({:.1} min)",
        eclipse.start,
        eclipse.end,
        eclipse.duration / 60.0);
}
```

## Example: ISS Orbit

```rust,ignore
use simular::scenarios::{SatelliteScenario, OrbitalElements};

fn main() {
    // ISS orbital elements (approximate)
    let iss = OrbitalElements {
        semi_major_axis: 6_778_000.0,  // ~400km altitude
        eccentricity: 0.0007,
        inclination: 51.6_f64.to_radians(),
        raan: 0.0,
        argument_of_periapsis: 0.0,
        true_anomaly: 0.0,
    };

    let mut scenario = SatelliteScenario::new(iss);

    // Simulate one orbit
    let orbital_period = scenario.orbital_period();
    println!("Orbital period: {:.1} minutes", orbital_period / 60.0);

    let steps = (orbital_period / 60.0) as usize;  // 1 minute steps
    for _ in 0..steps {
        scenario.step(60.0)?;
    }

    // Check if orbit is closed
    let initial_pos = scenario.initial_state().position;
    let final_pos = scenario.state().position;
    let closure_error = (final_pos - initial_pos).magnitude();

    println!("Orbit closure error: {:.1} m", closure_error);
}
```

## Orbital Mechanics Utilities

```rust,ignore
use simular::scenarios::orbital;

// Vis-viva equation
let velocity = orbital::vis_viva(
    398600.4418e9,  // μ for Earth
    6_778_000.0,    // r (distance from center)
    6_778_000.0,    // a (semi-major axis)
);

// Orbital period
let period = orbital::period(398600.4418e9, 6_778_000.0);

// Escape velocity
let v_escape = orbital::escape_velocity(398600.4418e9, 6_378_000.0);
```

## Next Steps

- [Physics Simulations](./domain_physics.md) - Underlying integrators
- [Rocket Launch](./scenario_rocket.md) - Getting to orbit
