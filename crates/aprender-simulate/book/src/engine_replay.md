# Replay System

The replay system records and replays simulations for debugging, analysis, and reproducibility.

## Recording Simulations

```rust,ignore
use simular::replay::{ReplayRecorder, SimEvent};
use simular::engine::state::{SimState, Vec3};

let mut recorder = ReplayRecorder::new();

// Record initial state
let mut state = SimState::new();
state.add_body(1.0, Vec3::new(0.0, 0.0, 100.0), Vec3::new(10.0, 0.0, 0.0));
recorder.record_state(&state);

// Record events during simulation
for step in 0..1000 {
    engine.step(&mut state, dt)?;
    recorder.record_state(&state);
}

// Save replay
recorder.save("simulation.replay")?;
```

## Replaying Simulations

```rust,ignore
use simular::replay::ReplayPlayer;

let player = ReplayPlayer::load("simulation.replay")?;

// Step through replay
for (step, state) in player.iter() {
    println!("Step {}: position = {:?}", step, state.positions()[0]);
}

// Jump to specific step
let state_at_500 = player.get_state(500)?;
```

## Replay Format

Replays are stored in a compact binary format:

```rust,ignore
pub struct Replay {
    /// Simulation metadata
    pub metadata: ReplayMetadata,

    /// Recorded states
    pub states: Vec<SimState>,

    /// Events between states
    pub events: Vec<Vec<SimEvent>>,

    /// Checksum for integrity
    pub checksum: [u8; 32],
}
```

## Compression

Replays are compressed with Zstd:

```rust,ignore
// Save with compression (default)
recorder.save_compressed("simulation.replay.zst")?;

// Load compressed
let player = ReplayPlayer::load_compressed("simulation.replay.zst")?;
```

## Checkpointing

Save and restore from checkpoints:

```rust,ignore
use simular::replay::Checkpoint;

// Create checkpoint
let checkpoint = Checkpoint::new(&state, &rng);
checkpoint.save("checkpoint.bin")?;

// Later: restore from checkpoint
let checkpoint = Checkpoint::load("checkpoint.bin")?;
let (state, rng) = checkpoint.restore()?;

// Continue simulation from checkpoint
```

## Selective Recording

Record only specific data:

```rust,ignore
let mut recorder = ReplayRecorder::new()
    .record_positions(true)
    .record_velocities(true)
    .record_energies(true)
    .record_forces(false);  // Skip forces to save space
```

## Analysis

Analyze replays:

```rust,ignore
let player = ReplayPlayer::load("simulation.replay")?;

// Energy analysis
let energies: Vec<f64> = player.iter()
    .map(|(_, state)| state.total_energy())
    .collect();

let max_drift = energies.windows(2)
    .map(|w| (w[1] - w[0]).abs())
    .fold(0.0, f64::max);

println!("Max energy drift: {:.2e}", max_drift);

// Position analysis
let trajectory: Vec<Vec3> = player.iter()
    .map(|(_, state)| state.positions()[0])
    .collect();
```

## Diff Replays

Compare two simulation runs:

```rust,ignore
use simular::replay::ReplayDiff;

let replay1 = ReplayPlayer::load("run1.replay")?;
let replay2 = ReplayPlayer::load("run2.replay")?;

let diff = ReplayDiff::compare(&replay1, &replay2);

if diff.is_identical() {
    println!("Simulations are bit-identical");
} else {
    println!("First difference at step {}", diff.first_difference_step());
    println!("Max position error: {:.2e}", diff.max_position_error());
}
```

## Use Cases

### Debugging

```rust,ignore
// Record a failing simulation
let mut recorder = ReplayRecorder::new();
// ... simulation with bug ...

// Later: replay to investigate
let player = ReplayPlayer::load("buggy.replay")?;
for (step, state) in player.iter() {
    if !state.all_finite() {
        println!("NaN first appears at step {}", step);
        let prev_state = player.get_state(step - 1)?;
        // Analyze what went wrong
        break;
    }
}
```

### Validation

```rust,ignore
// Run simulation twice
let replay1 = run_and_record(seed: 42);
let replay2 = run_and_record(seed: 42);

// Must be identical
let diff = ReplayDiff::compare(&replay1, &replay2);
assert!(diff.is_identical(), "Non-determinism detected!");
```

### Visualization

```rust,ignore
let player = ReplayPlayer::load("simulation.replay")?;

// Export for visualization
for (step, state) in player.iter().step_by(10) {
    export_frame(step, &state);
}
```

## Memory Management

For large simulations, stream from disk:

```rust,ignore
// Streaming player - doesn't load all states into memory
let player = ReplayPlayer::stream("large_simulation.replay")?;

for (step, state) in player.iter() {
    // States loaded on-demand
    process(&state);
}
```

## Best Practices

1. **Always record** during development/debugging
2. **Compress** for storage efficiency
3. **Checkpoint** for long simulations
4. **Verify determinism** by diffing replays

## Next Steps

- [Jidoka Guards](./engine_jidoka.md) - Anomaly detection
- [Deterministic RNG](./engine_rng.md) - Reproducible randomness
