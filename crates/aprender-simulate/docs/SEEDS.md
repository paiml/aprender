# Random Seed Documentation

This document describes all random seeds used in simular simulations.

## Seed Catalog

### TSP GRASP Algorithm

| Seed Variable | Default | Purpose |
|---------------|---------|---------|
| `grasp_seed` | 42 | Construction phase randomness |
| `perturbation_seed` | 123 | 2-opt perturbation |

### Monte Carlo Simulations

| Seed Variable | Default | Purpose |
|---------------|---------|---------|
| `mc_seed` | 42 | Point sampling |
| `antithetic_seed` | derived | Antithetic variates |

### Orbital Mechanics

| Seed Variable | Default | Purpose |
|---------------|---------|---------|
| `orbit_seed` | 12345 | Initial perturbations |
| `noise_seed` | 54321 | Measurement noise |

### Queueing Simulations

| Seed Variable | Default | Purpose |
|---------------|---------|---------|
| `arrival_seed` | 42 | Inter-arrival times |
| `service_seed` | 43 | Service times |

## RNG Algorithm

simular uses PCG64 (Permuted Congruential Generator):

- State size: 128 bits
- Period: 2^128
- Statistical quality: Passes TestU01, PractRand

## Seed Best Practices

### Explicit Seeding

Always pass seeds explicitly:

```rust
// Good: explicit seed
let sim = TspGraspDemo::new(42, n_cities);

// Bad: implicit seed (non-deterministic)
let sim = TspGraspDemo::default();  // ⚠️ Not reproducible
```

### Seed Recording

Log seeds for reproduction:

```rust
let seed = 42;
log::info!("Running with seed: {}", seed);
let result = run_simulation(seed);
```

### Cross-Platform Verification

Same seed must produce same results:

```bash
# Linux
cargo run -- --seed 42 > linux_output.txt

# macOS
cargo run -- --seed 42 > macos_output.txt

# Compare
diff linux_output.txt macos_output.txt  # Should be empty
```

## Magic Numbers Avoided

All seeds have documented purposes. No unexplained magic numbers.

| Seed | Rationale |
|------|-----------|
| 42 | Standard test seed (Hitchhiker's Guide) |
| 12345 | Sequential digits for clarity |
| 54321 | Reverse of 12345 for independence |
