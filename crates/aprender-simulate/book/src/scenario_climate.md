# Climate Models Scenario

The climate scenario provides energy balance models for climate simulation.

## Basic Energy Balance Model

```rust,ignore
use simular::scenarios::{ClimateScenario, ClimateConfig};

let config = ClimateConfig {
    solar_constant: 1361.0,     // W/m² at Earth
    albedo: 0.30,               // Earth's average reflectivity
    emissivity: 0.97,           // Infrared emissivity
    heat_capacity: 5e8,         // J/(m²·K) - ocean mixed layer
    initial_temperature: 288.0, // K (~15°C)
};

let mut scenario = ClimateScenario::new(config);
```

## Running the Simulation

```rust,ignore
let dt = 86400.0;  // 1 day in seconds

for year in 0..100 {
    for _ in 0..365 {
        scenario.step(dt)?;
    }

    let state = scenario.state();
    println!("Year {}: T = {:.2} K ({:.2}°C)",
        year,
        state.temperature,
        state.temperature - 273.15);
}
```

## Energy Balance Equation

The model solves:

C × dT/dt = S(1-α)/4 - εσT⁴

Where:
- C = heat capacity
- T = temperature
- S = solar constant
- α = albedo
- ε = emissivity
- σ = Stefan-Boltzmann constant

## Equilibrium Temperature

```rust,ignore
let scenario = ClimateScenario::new(config);
let equilibrium = scenario.equilibrium_temperature();

println!("Equilibrium temperature: {:.2} K", equilibrium);
// For Earth: ~255 K without greenhouse effect, ~288 K with
```

## Greenhouse Effect

```rust,ignore
let config = ClimateConfig {
    greenhouse_factor: 1.0,  // No greenhouse effect
    ..Default::default()
};

let scenario = ClimateScenario::new(config);
let t_no_greenhouse = scenario.equilibrium_temperature();

let config = ClimateConfig {
    greenhouse_factor: 0.61,  // Earth's current greenhouse effect
    ..Default::default()
};

let scenario = ClimateScenario::new(config);
let t_with_greenhouse = scenario.equilibrium_temperature();

println!("Without greenhouse: {:.1} K", t_no_greenhouse);   // ~255 K
println!("With greenhouse: {:.1} K", t_with_greenhouse);     // ~288 K
```

## CO2 Forcing

```rust,ignore
let config = ClimateConfig {
    co2_ppm: 280.0,  // Pre-industrial
    ..Default::default()
};

let pre_industrial = ClimateScenario::new(config);

let config = ClimateConfig {
    co2_ppm: 420.0,  // Current (2024)
    ..Default::default()
};

let current = ClimateScenario::new(config);

let forcing = current.radiative_forcing() - pre_industrial.radiative_forcing();
println!("CO2 forcing: {:.2} W/m²", forcing);
```

## Climate Sensitivity

```rust,ignore
// Equilibrium climate sensitivity (ECS)
// Temperature change for CO2 doubling

let baseline_co2 = 280.0;
let doubled_co2 = 560.0;

let baseline_scenario = ClimateScenario::new(ClimateConfig {
    co2_ppm: baseline_co2,
    ..Default::default()
});

let doubled_scenario = ClimateScenario::new(ClimateConfig {
    co2_ppm: doubled_co2,
    ..Default::default()
});

// Run to equilibrium
baseline_scenario.run_to_equilibrium()?;
doubled_scenario.run_to_equilibrium()?;

let ecs = doubled_scenario.state().temperature
    - baseline_scenario.state().temperature;

println!("Climate sensitivity: {:.2} K per CO2 doubling", ecs);
```

## Ice-Albedo Feedback

```rust,ignore
let config = ClimateConfig {
    ice_albedo_feedback: true,
    ice_line_temperature: 263.0,  // K (-10°C)
    ice_albedo: 0.6,
    ocean_albedo: 0.06,
    ..Default::default()
};

let mut scenario = ClimateScenario::new(config);

// With feedback, small perturbations can trigger ice ages
scenario.perturb_temperature(-5.0);  // Cool by 5K

for _ in 0..1000 {
    scenario.step(86400.0 * 365.0)?;  // 1 year steps
}

println!("Final temperature: {:.2} K", scenario.state().temperature);
```

## Multi-Box Model

```rust,ignore
use simular::scenarios::MultiBoxClimateScenario;

let config = MultiBoxClimateConfig {
    boxes: vec![
        Box { name: "atmosphere", heat_capacity: 1e7, ..Default::default() },
        Box { name: "surface", heat_capacity: 5e8, ..Default::default() },
        Box { name: "deep_ocean", heat_capacity: 1e10, ..Default::default() },
    ],
    exchange_rates: vec![
        (0, 1, 10.0),  // atmosphere <-> surface
        (1, 2, 1.0),   // surface <-> deep ocean
    ],
    ..Default::default()
};

let scenario = MultiBoxClimateScenario::new(config);
```

## Seasonal Cycle

```rust,ignore
let config = ClimateConfig {
    seasonal_cycle: true,
    latitude: 45.0,  // degrees
    eccentricity: 0.017,
    obliquity: 23.4,
    ..Default::default()
};

let mut scenario = ClimateScenario::new(config);

// Simulate one year
for day in 0..365 {
    scenario.step(86400.0)?;

    if day % 30 == 0 {
        let state = scenario.state();
        println!("Day {}: T = {:.1}°C",
            day, state.temperature - 273.15);
    }
}
```

## Example: Warming Scenarios

```rust,ignore
use simular::scenarios::{ClimateScenario, ClimateConfig};

fn main() {
    let baseline = ClimateConfig::earth_default();

    let scenarios = vec![
        ("Pre-industrial", 280.0),
        ("Current (2024)", 420.0),
        ("2050 Low", 480.0),
        ("2050 High", 560.0),
        ("2100 Low", 500.0),
        ("2100 High", 800.0),
    ];

    println!("Scenario          | CO2 (ppm) | ΔT (°C)");
    println!("------------------|-----------|--------");

    let baseline_t = ClimateScenario::new(ClimateConfig {
        co2_ppm: 280.0,
        ..baseline.clone()
    }).equilibrium_temperature();

    for (name, co2) in scenarios {
        let config = ClimateConfig {
            co2_ppm: co2,
            ..baseline.clone()
        };

        let scenario = ClimateScenario::new(config);
        let delta_t = scenario.equilibrium_temperature() - baseline_t;

        println!("{:17} | {:>9.0} | {:>+6.2}",
            name, co2, delta_t);
    }
}
```

## Next Steps

- [Portfolio Risk](./scenario_portfolio.md) - Financial modeling
- [Epidemic Models](./scenario_epidemic.md) - SIR/SEIR models
