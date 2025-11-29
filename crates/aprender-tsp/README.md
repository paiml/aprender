# aprender-tsp

Local TSP (Traveling Salesman Problem) optimization with personalized `.apr` models.

## Features

- **Multiple Solvers**: ACO (Ant Colony), Tabu Search, Genetic Algorithm, Hybrid
- **TSPLIB Support**: Standard benchmark instance format (berlin52, att48, eil51, etc.)
- **Model Persistence**: Train once, solve fast with `.apr` binary format
- **CLI & Library**: Use as command-line tool or Rust library
- **Deterministic**: Seed-based reproducibility for benchmarking

## Quick Start

```bash
# Install
cargo install aprender-tsp

# Train a model
aprender-tsp train instances/berlin52.tsp -o berlin52.apr --algorithm aco

# Solve new instances
aprender-tsp solve -m berlin52.apr instances/new-instance.tsp

# View model info
aprender-tsp info berlin52.apr

# Benchmark against known optimum
aprender-tsp benchmark berlin52.apr --instances instances/berlin52.tsp
```

## Algorithms

| Algorithm | Description | Best For |
|-----------|-------------|----------|
| ACO | Ant Colony Optimization | General TSP, exploration |
| Tabu | Tabu Search with 2-opt | Local refinement |
| GA | Genetic Algorithm | Large instances |
| Hybrid | GA + Tabu + ACO pipeline | Best quality |

## Library Usage

```rust
use aprender_tsp::{TspInstance, AcoSolver, TspSolver, Budget};

// Load instance
let instance = TspInstance::from_tsplib("berlin52.tsp")?;

// Solve with ACO
let mut solver = AcoSolver::new()
    .with_num_ants(20)
    .with_alpha(1.0)
    .with_beta(2.5);

let solution = solver.solve(&instance, Budget::iterations(1000))?;

println!("Tour length: {}", solution.length);
println!("Gap from optimal: {:.2}%", solution.optimality_gap(&instance)?);
```

## Pre-trained Models

POC models available on Hugging Face: [paiml/aprender-tsp-poc](https://huggingface.co/paiml/aprender-tsp-poc)

| Model | Instance | Gap from Optimal |
|-------|----------|------------------|
| berlin52-aco.apr | berlin52 | 1.92% |
| att48-aco.apr | att48 | 4.30% |
| eil51-aco.apr | eil51 | 4.07% |

## Instance Format

Supports TSPLIB format (`.tsp`):

```
NAME: example
TYPE: TSP
DIMENSION: 4
EDGE_WEIGHT_TYPE: EUC_2D
NODE_COORD_SECTION
1 0.0 0.0
2 1.0 0.0
3 1.0 1.0
4 0.0 1.0
EOF
```

Also supports CSV format with optional header.

## Model Format

`.apr` files are compact binary models (~77 bytes):
- Magic bytes: `APR\0`
- Version: 1
- CRC32 checksum for integrity
- Algorithm-specific parameters

## Part of Aprender

This crate uses `aprender::metaheuristics` for core optimization algorithms. See [aprender](https://github.com/paiml/aprender) for the full ML library.

## License

MIT
