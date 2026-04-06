# Deterministic RNG

SimRng provides deterministic, reproducible random number generation based on PCG (Permuted Congruential Generator).

## Basic Usage

```rust,ignore
use simular::engine::rng::SimRng;

// Create RNG with seed
let mut rng = SimRng::new(42);

// Generate random values
let uniform: f64 = rng.gen_f64();           // [0, 1)
let range: f64 = rng.gen_range_f64(-1.0, 1.0);  // [-1, 1)
let integer: u64 = rng.gen_u64();           // Full u64 range
let normal: f64 = rng.gen_standard_normal(); // N(0, 1)
let gaussian: f64 = rng.gen_normal(10.0, 2.0); // N(10, 2)
```

## Reproducibility Guarantee

Same seed = identical sequence, always:

```rust,ignore
let mut rng1 = SimRng::new(42);
let mut rng2 = SimRng::new(42);

let seq1: Vec<f64> = (0..100).map(|_| rng1.gen_f64()).collect();
let seq2: Vec<f64> = (0..100).map(|_| rng2.gen_f64()).collect();

assert_eq!(seq1, seq2);  // Bit-identical!
```

This holds across:
- Different runs
- Different platforms
- Different thread counts

## Batch Sampling

```rust,ignore
let mut rng = SimRng::new(42);

// Generate multiple samples efficiently
let samples: Vec<f64> = rng.sample_n(1000);
assert_eq!(samples.len(), 1000);
```

## Parallel Partitioning

For deterministic parallelism, partition the RNG:

```rust,ignore
let mut rng = SimRng::new(42);
let mut partitions = rng.partition(4);  // 4 independent streams

// Each partition is deterministic
std::thread::scope(|s| {
    for (i, partition) in partitions.iter_mut().enumerate() {
        s.spawn(move || {
            for _ in 0..100 {
                let _ = partition.gen_f64();
            }
            println!("Thread {} done", i);
        });
    }
});

// Running twice gives identical results!
```

### How Partitioning Works

Each partition gets a different stream derived from the master seed:

```text
Master seed: 42
├── Partition 0: seed = f(42, 0)
├── Partition 1: seed = f(42, 1)
├── Partition 2: seed = f(42, 2)
└── Partition 3: seed = f(42, 3)
```

The streams are:
- **Independent**: No correlation between partitions
- **Deterministic**: Same master seed = same partitions
- **Non-overlapping**: Guaranteed different sequences

## Normal Distribution

Box-Muller transform for normal samples:

```rust,ignore
let mut rng = SimRng::new(42);

// Standard normal N(0, 1)
let z = rng.gen_standard_normal();

// General normal N(μ, σ)
let x = rng.gen_normal(100.0, 15.0);  // Mean=100, Std=15

// Verify statistics
let samples: Vec<f64> = (0..10000)
    .map(|_| rng.gen_standard_normal())
    .collect();

let mean = samples.iter().sum::<f64>() / samples.len() as f64;
let variance = samples.iter()
    .map(|x| (x - mean).powi(2))
    .sum::<f64>() / samples.len() as f64;

println!("Mean: {:.4} (expected 0)", mean);
println!("Variance: {:.4} (expected 1)", variance);
```

## State Save/Restore

Checkpoint RNG state:

```rust,ignore
let mut rng = SimRng::new(42);

// Generate some values
for _ in 0..100 {
    rng.gen_f64();
}

// Save state
let state = rng.save_state();

// Continue generating
let next_value = rng.gen_f64();

// Restore state
let mut rng2 = SimRng::new(42);
rng2.restore_state(&state).unwrap();

// Same value as before
assert_eq!(rng2.gen_f64(), next_value);
```

### RngState

```rust,ignore
pub struct RngState {
    pub master_seed: u64,
    pub stream: u64,
    pub verification_values: Option<Vec<u64>>,
}
```

## Monte Carlo Example

```rust,ignore
use simular::engine::rng::SimRng;

fn estimate_pi(seed: u64, samples: usize) -> f64 {
    let mut rng = SimRng::new(seed);
    let mut inside = 0;

    for _ in 0..samples {
        let x = rng.gen_f64();
        let y = rng.gen_f64();
        if x * x + y * y <= 1.0 {
            inside += 1;
        }
    }

    4.0 * inside as f64 / samples as f64
}

fn main() {
    // Same seed = same result
    let pi1 = estimate_pi(42, 100_000);
    let pi2 = estimate_pi(42, 100_000);
    assert_eq!(pi1, pi2);

    println!("π ≈ {:.6}", pi1);
}
```

## Parallel Monte Carlo

```rust,ignore
use simular::engine::rng::SimRng;

fn parallel_estimate_pi(master_seed: u64, total_samples: usize, threads: usize) -> f64 {
    let mut rng = SimRng::new(master_seed);
    let mut partitions = rng.partition(threads);
    let samples_per_thread = total_samples / threads;

    let counts: Vec<usize> = std::thread::scope(|s| {
        partitions.iter_mut()
            .map(|partition| {
                s.spawn(move || {
                    let mut inside = 0;
                    for _ in 0..samples_per_thread {
                        let x = partition.gen_f64();
                        let y = partition.gen_f64();
                        if x * x + y * y <= 1.0 {
                            inside += 1;
                        }
                    }
                    inside
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|h| h.join().unwrap())
            .collect()
    });

    4.0 * counts.iter().sum::<usize>() as f64 / total_samples as f64
}

fn main() {
    // Deterministic despite parallel execution
    let pi1 = parallel_estimate_pi(42, 1_000_000, 4);
    let pi2 = parallel_estimate_pi(42, 1_000_000, 4);
    assert_eq!(pi1, pi2);

    println!("π ≈ {:.6}", pi1);
}
```

## PCG Algorithm

SimRng uses PCG64 (Permuted Congruential Generator):

- **Period**: 2^128
- **State**: 128 bits
- **Output**: 64 bits
- **Properties**: Excellent statistical quality, fast, predictable

## API Summary

| Method | Returns | Description |
|--------|---------|-------------|
| `new(seed)` | `SimRng` | Create RNG with seed |
| `gen_f64()` | `f64` | Uniform [0, 1) |
| `gen_range_f64(min, max)` | `f64` | Uniform [min, max) |
| `gen_u64()` | `u64` | Full u64 range |
| `gen_standard_normal()` | `f64` | N(0, 1) |
| `gen_normal(mean, std)` | `f64` | N(mean, std) |
| `sample_n(n)` | `Vec<f64>` | n uniform samples |
| `partition(n)` | `Vec<SimRng>` | n independent streams |
| `master_seed()` | `u64` | Get master seed |
| `stream()` | `u64` | Get current stream |
| `save_state()` | `RngState` | Save state |
| `restore_state(state)` | `Result<()>` | Restore state |

## Next Steps

- [Replay System](./engine_replay.md) - Record and replay simulations
- [Jidoka Guards](./engine_jidoka.md) - Anomaly detection
