# Fuzzing

Probar includes fuzzing support for finding edge cases in game logic.

## Fuzzing Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                       FUZZING PIPELINE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │                    Input Generation                        │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │   │
│  │  │ RandomWalk  │  │   Chaos     │  │ Adversarial │        │   │
│  │  │   Agent     │  │   Agent     │  │    Agent    │        │   │
│  │  │  (biased)   │  │ (uniform)   │  │ (targeted)  │        │   │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │   │
│  │         └────────────────┼────────────────┘               │   │
│  │                          ▼                                │   │
│  │                  ┌──────────────┐                         │   │
│  │                  │   Inputs     │                         │   │
│  │                  │ Vec<Event>   │                         │   │
│  │                  └──────────────┘                         │   │
│  └───────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │                    Simulation                              │   │
│  │           [seed] ─► [frames] ─► [state_hash]               │   │
│  └───────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │                 Invariant Checking                         │   │
│  │  ┌─────────────────────────────────────────────────────┐  │   │
│  │  │ ✓ ball_in_bounds    ✓ score_valid    ✗ no_crash    │  │   │
│  │  └─────────────────────────────────────────────────────┘  │   │
│  └───────────────────────────────────────────────────────────┘   │
│                              │                                   │
│              ┌───────────────┼───────────────┐                  │
│              ▼               ▼               ▼                  │
│      ┌────────────┐  ┌────────────┐  ┌────────────┐             │
│      │   Pass     │  │   Fail     │  │  Shrink    │             │
│      │  (next)    │  │  (report)  │  │  (minimize)│             │
│      └────────────┘  └────────────┘  └────────────┘             │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

![Coverage Gap Detection](../assets/coverage_pattern.png)

*Fuzzing helps identify coverage gaps (shown as darker regions in the heatmap above)*

## Random Walk Agent

```rust
use jugar_probar::{RandomWalkAgent, Seed};

let seed = Seed::from_u64(12345);
let mut agent = RandomWalkAgent::new(seed);

// Generate random inputs for each frame
for frame in 0..1000 {
    let inputs = agent.next_inputs();
    platform.process_inputs(&inputs);
    platform.advance_frame(1.0 / 60.0);
}
```

## Fuzzing with Invariants

```rust
use jugar_probar::{fuzz_with_invariants, FuzzConfig, Invariant};

let invariants = vec![
    Invariant::new("no_crashes", |state| state.is_valid()),
    Invariant::new("ball_visible", |state| {
        state.ball.x.is_finite() && state.ball.y.is_finite()
    }),
    Invariant::new("score_bounded", |state| {
        state.score_left <= 100 && state.score_right <= 100
    }),
];

let config = FuzzConfig {
    iterations: 1000,
    frames_per_iteration: 500,
    seed: 42,
};

let result = fuzz_with_invariants(config, invariants);

if !result.all_passed {
    for failure in &result.failures {
        println!("Invariant '{}' failed at iteration {} frame {}",
            failure.invariant_name,
            failure.iteration,
            failure.frame);
        println!("Reproducing seed: {}", failure.seed);
    }
}
```

## Input Generation Strategies

```rust
// Random inputs
let mut agent = RandomWalkAgent::new(seed);

// Biased toward movement
let mut agent = RandomWalkAgent::new(seed)
    .with_key_probability("KeyW", 0.3)
    .with_key_probability("KeyS", 0.3)
    .with_key_probability("Space", 0.1);

// Chaos monkey (random everything)
let mut agent = ChaosAgent::new(seed);

// Adversarial (try to break the game)
let mut agent = AdversarialAgent::new(seed)
    .target_invariant(|state| state.ball.y >= 0.0);
```

## Property-Based Testing

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn ball_stays_in_bounds(seed in 0u64..10000) {
        let config = SimulationConfig::new(seed, 1000);
        let result = run_simulation(config, |_| vec![]);

        prop_assert!(result.final_state.ball.x >= 0.0);
        prop_assert!(result.final_state.ball.x <= 800.0);
        prop_assert!(result.final_state.ball.y >= 0.0);
        prop_assert!(result.final_state.ball.y <= 600.0);
    }

    #[test]
    fn score_is_valid(
        seed in 0u64..10000,
        frames in 100u32..5000
    ) {
        let config = SimulationConfig::new(seed, frames);
        let result = run_simulation(config, |_| vec![]);

        prop_assert!(result.final_state.score_left <= 10);
        prop_assert!(result.final_state.score_right <= 10);
    }
}
```

## Seed Management

```rust
use jugar_probar::Seed;

// From u64
let seed = Seed::from_u64(42);

// From bytes
let seed = Seed::from_bytes(&[1, 2, 3, 4, 5, 6, 7, 8]);

// Random
let seed = Seed::random();

// Get value for reproduction
println!("Failing seed: {}", seed.as_u64());
```

## Reproducing Failures

When fuzzing finds a failure, reproduce it:

```rust
#[test]
fn reproduce_bug_12345() {
    // Seed from fuzzing failure
    let seed = Seed::from_u64(12345);

    let config = SimulationConfig::new(seed.as_u64(), 500);
    let result = run_simulation(config, |_| vec![]);

    // This should fail with the original bug
    assert!(result.final_state.ball.y >= 0.0);
}
```

## Fuzzing Configuration

```rust
pub struct FuzzConfig {
    pub iterations: u32,           // Number of random runs
    pub frames_per_iteration: u32, // Frames per run
    pub seed: u64,                 // Base seed
    pub timeout_seconds: u32,      // Max time per iteration
    pub parallel: bool,            // Run in parallel
    pub save_failures: bool,       // Save failing cases
}

let config = FuzzConfig {
    iterations: 10000,
    frames_per_iteration: 1000,
    seed: 42,
    timeout_seconds: 10,
    parallel: true,
    save_failures: true,
};
```

## Shrinking

When a failure is found, Probar automatically shrinks the input:

```rust
let result = fuzz_with_shrinking(config, invariants);

if let Some(failure) = result.first_failure {
    println!("Original failure at frame {}", failure.original_frame);
    println!("Shrunk to frame {}", failure.shrunk_frame);
    println!("Minimal inputs: {:?}", failure.minimal_inputs);
}
```

## Continuous Fuzzing

Run fuzzing in CI:

```bash
# Run fuzzing for 10 minutes
FUZZ_DURATION=600 cargo test fuzz_ -- --ignored

# Or via make
make fuzz-ci
```
