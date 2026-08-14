# Deterministic Replay

Probar enables frame-perfect replay of game sessions using **trueno's SimRng** (PCG-based deterministic RNG) to guarantee reproducibility across platforms and runs.

![Replay Coverage Analysis](../assets/coverage_magma.png)

*Replay sessions build comprehensive coverage maps over time*

## Why Deterministic Replay?

- **Bug Reproduction**: Replay exact sequence that caused a bug
- **Regression Testing**: Verify behavior matches after changes
- **Test Generation**: Record gameplay, convert to tests
- **Demo Playback**: Record and replay gameplay sequences

## Recording a Session

```rust
use jugar_probar::{Recorder, Recording};

let mut recorder = Recorder::new(seed);
let mut platform = WebPlatform::new_for_test(config);

// Play game and record
for frame in 0..1000 {
    let inputs = get_user_inputs();
    recorder.record_frame(frame, &inputs);

    platform.process_inputs(&inputs);
    platform.advance_frame(1.0 / 60.0);
}

// Get recording
let recording = recorder.finish();

// Save to file
recording.save("gameplay.replay")?;
```

## Replaying a Session

```rust
use jugar_probar::{Recording, Replayer};

// Load recording
let recording = Recording::load("gameplay.replay")?;

let mut replayer = Replayer::new(&recording);
let mut platform = WebPlatform::new_for_test(config);

// Replay exactly
while let Some(inputs) = replayer.next_frame() {
    platform.process_inputs(&inputs);
    platform.advance_frame(1.0 / 60.0);
}

// Verify final state matches
assert_eq!(
    replayer.expected_final_hash(),
    platform.state_hash()
);
```

## Recording Format

```rust
pub struct Recording {
    pub version: u32,
    pub seed: u64,
    pub config: GameConfig,
    pub frames: Vec<FrameData>,
    pub final_state_hash: u64,
}

pub struct FrameData {
    pub frame_number: u32,
    pub inputs: Vec<InputEvent>,
    pub state_hash: Option<u64>,  // Optional checkpoints
}
```

## State Hashing

```rust
// Hash game state for comparison
let hash = platform.state_hash();

// Or hash specific components
let ball_hash = hash_state(&platform.get_ball_state());
let score_hash = hash_state(&platform.get_score());
```

## Verification

```rust
use jugar_probar::{verify_replay, ReplayVerification};

let result = verify_replay(&recording);

match result {
    ReplayVerification::Perfect => {
        println!("Replay is deterministic!");
    }
    ReplayVerification::Diverged { frame, expected, actual } => {
        println!("Diverged at frame {}", frame);
        println!("Expected hash: {}", expected);
        println!("Actual hash: {}", actual);
    }
    ReplayVerification::Failed(error) => {
        println!("Replay failed: {}", error);
    }
}
```

## Checkpoints

Add checkpoints for faster debugging:

```rust
let mut recorder = Recorder::new(seed)
    .with_checkpoint_interval(60);  // Every 60 frames

// Or manual checkpoints
recorder.add_checkpoint(platform.snapshot());
```

## Binary Replay Format

```rust
// Compact binary format for storage
let bytes = recording.to_bytes();
let recording = Recording::from_bytes(&bytes)?;

// Compressed
let compressed = recording.to_compressed_bytes();
let recording = Recording::from_compressed_bytes(&compressed)?;
```

## Replay Speed Control

```rust
let mut replayer = Replayer::new(&recording);

// Normal speed
replayer.set_speed(1.0);

// Fast forward
replayer.set_speed(4.0);

// Slow motion
replayer.set_speed(0.25);

// Step by step
replayer.step();  // Advance one frame
```

## Example: Test from Replay

```rust
#[test]
fn test_from_recorded_gameplay() {
    let recording = Recording::load("tests/fixtures/win_game.replay").unwrap();

    let mut replayer = Replayer::new(&recording);
    let mut platform = WebPlatform::new_for_test(recording.config.clone());

    // Replay all frames
    while let Some(inputs) = replayer.next_frame() {
        platform.process_inputs(&inputs);
        platform.advance_frame(1.0 / 60.0);
    }

    // Verify end state
    let state = platform.get_game_state();
    assert_eq!(state.winner, Some(Player::Left));
    assert_eq!(state.score_left, 10);
}
```

## CI Integration

```bash
# Verify all replay files are still deterministic
cargo test replay_verification -- --include-ignored

# Or via make
make verify-replays
```

## Debugging with Replays

```rust
// Find frame where bug occurs
let bug_frame = binary_search_replay(&recording, |state| {
    state.ball.y < 0.0  // Bug condition
});

println!("Bug first occurs at frame {}", bug_frame);

// Get inputs leading up to bug
let inputs = recording.frames[..bug_frame].to_vec();
println!("Inputs: {:?}", inputs);
```

## Determinism Guarantees via Trueno

Probar's deterministic replay is powered by trueno's simulation testing framework (v0.8.5+):

### SimRng: PCG-Based Determinism

All randomness in simulations uses `trueno::simulation::SimRng`:

```rust
use trueno::simulation::SimRng;

// PCG algorithm guarantees identical sequences across:
// - Different operating systems (Linux, macOS, Windows)
// - Different CPU architectures (x86_64, ARM64, WASM)
// - Different compiler versions
let mut rng = SimRng::new(recording.seed);

// Every call produces identical results given same seed
let ball_angle = rng.range(0.0, std::f32::consts::TAU);
let spawn_delay = rng.range(30, 120);  // frames
```

### Cross-Backend Consistency

Trueno ensures consistent results even when switching compute backends:

```rust
use trueno::simulation::BackendTolerance;

// Simulation results are identical whether running on:
// - CPU Scalar backend
// - SIMD (SSE2/AVX2/AVX-512/NEON)
// - GPU (via wgpu)
let tolerance = BackendTolerance::strict();
assert!(verify_cross_backend_determinism(&recording, tolerance));
```

### JidokaGuard: Replay Validation

Automatic quality checks during replay:

```rust
use trueno::simulation::JidokaGuard;

let guard = JidokaGuard::new();

// Automatically detects non-determinism sources
guard.check_finite(&state)?;  // NaN/Inf corrupts determinism
guard.assert_invariant(
    || state.frame == expected_frame,
    "Frame count mismatch - possible non-determinism"
)?;
```

### Why SimRng over std::rand?

| Feature | SimRng (trueno) | std::rand |
|---------|-----------------|-----------|
| Cross-platform identical | Yes | No (implementation-defined) |
| WASM compatible | Yes | Requires `getrandom` |
| Fork for parallelism | Yes (deterministic) | No |
| Serializable state | Yes | No |
| Performance | ~2ns/call | ~3ns/call |
