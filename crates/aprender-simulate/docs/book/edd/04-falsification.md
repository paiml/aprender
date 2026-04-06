# Popperian Falsification

## Actively Searching for Model-Breaking Conditions

Karl Popper argued that scientific theories cannot be proven true - they can only be falsified. EDD embraces this principle: **we actively search for conditions that would disprove our models**.

## The Falsification Mindset

Traditional testing asks: "Does my model work?"

Falsification asks: "Under what conditions would my model fail?"

This shift is crucial. A model that passes 1000 tests proves nothing if we haven't searched for its breaking points.

## The FalsifiableSimulation Trait

Every simulation in EDD must implement `FalsifiableSimulation`:

```rust
pub trait FalsifiableSimulation {
    /// Define criteria that would falsify the model
    fn falsification_criteria(&self) -> Vec<FalsificationCriterion>;

    /// Evaluate the simulation at given parameters
    fn evaluate(&self, params: &HashMap<String, f64>) -> Trajectory;

    /// Check if a criterion is satisfied (positive = OK, negative = violated)
    fn check_criterion(&self, criterion: &str, trajectory: &Trajectory) -> f64;

    /// Actively search for falsifying conditions (default implementation)
    fn seek_falsification(&self, param_space: &ParamSpace) -> FalsificationResult;
}
```

## Falsification Criteria

Each criterion specifies:
- What condition would invalidate the model
- How to detect that condition
- What action to take if violated

```rust
use simular::edd::{FalsificationCriterion, FalsificationAction};

let criteria = vec![
    FalsificationCriterion::new(
        "energy_conservation",
        "|E(t) - E(0)| / E(0) < 1e-6",
        FalsificationAction::RejectModel,
    ),
    FalsificationCriterion::new(
        "bounded_motion",
        "|x(t)| <= amplitude for all t",
        FalsificationAction::Stop,
    ),
];
```

### Falsification Actions

| Action | Description |
|--------|-------------|
| `Warn` | Log warning, continue simulation |
| `Stop` | Halt the current test |
| `RejectModel` | Model is falsified, reject it |
| `FlagReview` | Flag for human review |

## Parameter Space Search

Falsification requires searching the parameter space systematically:

```rust
use simular::edd::ParamSpace;

// Define parameter bounds
let space = ParamSpace::new()
    .with_param("omega", 0.5, 10.0)    // Angular frequency range
    .with_param("amplitude", 0.1, 2.0) // Amplitude range
    .with_samples(10);                  // 10 samples per dimension

// Generate grid points (10 x 10 = 100 combinations)
let points = space.grid_points();
```

### Grid Search

For exhaustive testing:

```rust
for params in space.grid_points() {
    let trajectory = sim.evaluate(&params);
    for criterion in sim.falsification_criteria() {
        let robustness = sim.check_criterion(&criterion.name, &trajectory);
        if robustness < 0.0 {
            println!("FALSIFIED at {:?}", params);
        }
    }
}
```

### Robustness Metric

The `check_criterion` method returns a **robustness value**:
- **Positive**: Criterion satisfied (larger = more margin)
- **Zero**: At the boundary
- **Negative**: Criterion violated (more negative = worse violation)

This continuous metric enables optimization-based falsification search.

## Trajectory Analysis

The `Trajectory` struct captures simulation output for analysis:

```rust
use simular::edd::Trajectory;

let mut traj = Trajectory::new(vec![
    "position".to_string(),
    "velocity".to_string(),
    "energy".to_string(),
]);

// Record state at each time step
for i in 0..100 {
    let t = i as f64 * 0.1;
    let x = (omega * t).cos();
    let v = -(omega * t).sin();
    let e = 0.5 * (x * x + v * v);
    traj.push(t, &[x, v, e]);
}

// Extract state by name
let final_energy = traj.get_state("energy", traj.len() - 1).unwrap();
```

## Active Falsification Search

The `seek_falsification` method performs systematic search:

```rust
let result = sim.seek_falsification(&param_space);

println!("Tests performed: {}", result.tests_performed);
println!("Falsified: {}", result.falsified);
println!("Min robustness: {}", result.robustness);

if result.falsified {
    println!("Violated: {}", result.violated_criterion.unwrap());
    println!("At params: {:?}", result.falsifying_params.unwrap());
}
```

### Falsification Result

```rust
pub struct FalsificationResult {
    pub falsified: bool,                           // Was model falsified?
    pub violated_criterion: Option<String>,        // Which criterion failed
    pub falsifying_params: Option<HashMap<String, f64>>, // At what parameters
    pub robustness: f64,                          // Minimum robustness found
    pub tests_performed: usize,                   // How many tests run
    pub summary: String,                          // Human-readable summary
}
```

## Example: Harmonic Oscillator

A complete falsification example:

```rust
use simular::edd::{FalsifiableSimulation, FalsificationCriterion,
                   FalsificationAction, Trajectory, ParamSpace};
use std::collections::HashMap;

struct HarmonicOscillator {
    fail_on_high_omega: bool,  // Simulate numerical instability
}

impl FalsifiableSimulation for HarmonicOscillator {
    fn falsification_criteria(&self) -> Vec<FalsificationCriterion> {
        vec![
            FalsificationCriterion::new(
                "energy_conservation",
                "|E(t) - E(0)| / E(0) < 1e-6",
                FalsificationAction::RejectModel,
            ),
        ]
    }

    fn evaluate(&self, params: &HashMap<String, f64>) -> Trajectory {
        let omega = params.get("omega").copied().unwrap_or(1.0);
        let mut traj = Trajectory::new(vec!["x".to_string(), "energy".to_string()]);

        for i in 0..10 {
            let t = i as f64 * 0.1;
            let x = (omega * t).cos();

            // Introduce energy drift at high omega (simulating numerical issues)
            let energy = if self.fail_on_high_omega && omega > 5.0 {
                0.5 * (1.0 + 0.1 * t)  // Energy drifts up
            } else {
                0.5  // Energy conserved
            };

            traj.push(t, &[x, energy]);
        }
        traj
    }

    fn check_criterion(&self, criterion: &str, trajectory: &Trajectory) -> f64 {
        match criterion {
            "energy_conservation" => {
                let e0 = trajectory.get_state("energy", 0).unwrap();
                let e_final = trajectory.get_state("energy", trajectory.len() - 1).unwrap();
                let drift = (e_final - e0).abs() / e0;
                1e-6 - drift  // Positive if satisfied, negative if violated
            }
            _ => f64::INFINITY,
        }
    }
}

// Test the stable oscillator
let stable = HarmonicOscillator { fail_on_high_omega: false };
let params = ParamSpace::new().with_param("omega", 0.5, 10.0).with_samples(5);
let result = stable.seek_falsification(&params);
assert!(!result.falsified);

// Test the unstable oscillator
let unstable = HarmonicOscillator { fail_on_high_omega: true };
let result = unstable.seek_falsification(&params);
assert!(result.falsified);  // Should fail at high omega
```

## Robustness Analysis

Beyond binary pass/fail, EDD tracks **robustness** - how close the model came to failing:

```rust
// Even if not falsified, low robustness is a warning sign
if !result.falsified && result.robustness < 0.1 {
    println!("WARNING: Model barely passes (robustness: {})", result.robustness);
    println!("Consider investigating parameters near the boundary");
}
```

## Running the Example

```bash
cargo run --example edd_falsification
```

## Next Chapter

Learn about TPS Validation - the ten canonical test cases that validate operations science equations.
