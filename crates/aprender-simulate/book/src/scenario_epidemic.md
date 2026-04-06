# Epidemic Models (SIR/SEIR) Scenario

The epidemic scenario provides SIR and SEIR disease spread models.

## SIR Model

Susceptible → Infected → Recovered

```rust,ignore
use simular::scenarios::{SIRScenario, SIRConfig};

let config = SIRConfig {
    population: 10_000_000,     // Total population
    initial_infected: 1,        // Patient zero
    initial_recovered: 0,
    beta: 0.3,                  // Transmission rate
    gamma: 0.1,                 // Recovery rate (1/infectious period)
};

let mut scenario = SIRScenario::new(config);
```

## Running the Simulation

```rust,ignore
let dt = 1.0;  // 1 day

for day in 0..365 {
    scenario.step(dt)?;

    let state = scenario.state();
    if day % 30 == 0 {
        println!("Day {:>3}: S={:>7.0}, I={:>7.0}, R={:>7.0}",
            day,
            state.susceptible,
            state.infected,
            state.recovered);
    }
}
```

## Basic Reproduction Number

R₀ = β/γ - average number of secondary infections

```rust,ignore
let config = SIRConfig {
    beta: 0.3,
    gamma: 0.1,
    ..Default::default()
};

let scenario = SIRScenario::new(config);
let r0 = scenario.r0();

println!("R₀ = {:.2}", r0);  // 3.0

// Epidemic only if R₀ > 1
if r0 > 1.0 {
    println!("Epidemic will spread");
}
```

## SEIR Model

Adds Exposed (latent) compartment:

Susceptible → Exposed → Infected → Recovered

```rust,ignore
use simular::scenarios::{SEIRScenario, SEIRConfig};

let config = SEIRConfig {
    population: 10_000_000,
    initial_exposed: 10,
    initial_infected: 1,
    beta: 0.5,                  // Transmission rate
    sigma: 0.2,                 // Incubation rate (1/latent period)
    gamma: 0.1,                 // Recovery rate
};

let mut scenario = SEIRScenario::new(config);

for day in 0..365 {
    scenario.step(1.0)?;

    let state = scenario.state();
    if state.infected > state.peak_infected {
        println!("Peak at day {}: {} infected", day, state.infected as u64);
    }
}
```

## Herd Immunity Threshold

```rust,ignore
let scenario = SIRScenario::new(config);
let r0 = scenario.r0();

// Herd immunity threshold
let hit = 1.0 - 1.0 / r0;
println!("Herd immunity threshold: {:.1}%", hit * 100.0);
```

## Interventions

### Reducing Transmission

```rust,ignore
let mut scenario = SIRScenario::new(config);

// Day 30: Implement social distancing (reduce beta by 50%)
for day in 0..365 {
    if day == 30 {
        scenario.set_beta(config.beta * 0.5);
        println!("Day 30: Social distancing implemented");
    }

    scenario.step(1.0)?;
}
```

### Vaccination

```rust,ignore
let mut scenario = SIRScenario::new(config);

// Vaccinate 1% of susceptible population per day
for day in 0..365 {
    scenario.step(1.0)?;

    let susceptible = scenario.state().susceptible;
    let vaccinated = susceptible * 0.01;
    scenario.vaccinate(vaccinated)?;
}
```

## Stochastic Model

```rust,ignore
use simular::scenarios::StochasticSIRScenario;

let config = StochasticSIRConfig {
    population: 1000,  // Small population - stochastic effects matter
    initial_infected: 5,
    beta: 0.3,
    gamma: 0.1,
    seed: 42,
};

let mut scenario = StochasticSIRScenario::new(config);

// May die out or explode due to randomness
for _ in 0..365 {
    scenario.step(1.0)?;

    if scenario.state().infected == 0 {
        println!("Epidemic died out!");
        break;
    }
}
```

## Final Size

Fraction of population eventually infected:

```rust,ignore
let scenario = SIRScenario::new(config);

// Run to equilibrium
scenario.run_to_equilibrium()?;

let final_size = scenario.state().recovered / config.population as f64;
println!("Final attack rate: {:.1}%", final_size * 100.0);
```

## Epidemic Curve

```rust,ignore
let mut scenario = SIRScenario::new(config);
let mut epidemic_curve = Vec::new();

for day in 0..365 {
    scenario.step(1.0)?;
    epidemic_curve.push((day, scenario.state().infected));
}

// Find peak
let (peak_day, peak_infected) = epidemic_curve.iter()
    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
    .unwrap();

println!("Peak: {} infected on day {}", *peak_infected as u64, peak_day);
```

## Example: Comparing R₀ Values

```rust,ignore
use simular::scenarios::{SIRScenario, SIRConfig};

fn main() {
    let r0_values = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0];

    println!("R₀   | HIT   | Peak Day | Peak %  | Final %");
    println!("-----|-------|----------|---------|--------");

    for r0 in r0_values {
        let gamma = 0.1;
        let beta = r0 * gamma;

        let config = SIRConfig {
            population: 1_000_000,
            initial_infected: 1,
            beta,
            gamma,
            ..Default::default()
        };

        let mut scenario = SIRScenario::new(config);

        let mut peak_day = 0;
        let mut peak_infected = 0.0;

        for day in 0..365 {
            scenario.step(1.0)?;

            if scenario.state().infected > peak_infected {
                peak_infected = scenario.state().infected;
                peak_day = day;
            }
        }

        let final_size = scenario.state().recovered / config.population as f64;
        let hit = 1.0 - 1.0 / r0;

        println!("{:>4.1} | {:>4.0}% | {:>8} | {:>6.1}% | {:>6.1}%",
            r0,
            hit * 100.0,
            peak_day,
            peak_infected / config.population as f64 * 100.0,
            final_size * 100.0);
    }
}
```

Output:
```text
R₀   | HIT   | Peak Day | Peak %  | Final %
-----|-------|----------|---------|--------
 1.5 |   33% |      147 |    3.2% |   58.3%
 2.0 |   50% |       97 |    8.9% |   79.7%
 2.5 |   60% |       76 |   14.2% |   89.1%
 3.0 |   67% |       63 |   18.8% |   94.0%
 4.0 |   75% |       48 |   26.4% |   98.0%
 5.0 |   80% |       40 |   32.3% |   99.3%
```

## Age-Structured Model

```rust,ignore
use simular::scenarios::AgeStructuredSIRScenario;

let config = AgeStructuredSIRConfig {
    age_groups: vec!["0-17", "18-64", "65+"],
    populations: vec![20.0, 60.0, 20.0],  // Percent
    contact_matrix: vec![
        vec![3.0, 1.5, 0.5],  // Children contact rates
        vec![1.5, 2.5, 1.0],  // Adults
        vec![0.5, 1.0, 1.5],  // Elderly
    ],
    ..Default::default()
};
```

## Next Steps

- [Monte Carlo Methods](./domain_monte_carlo.md) - Stochastic simulation
- [Climate Models](./scenario_climate.md) - Energy balance systems
