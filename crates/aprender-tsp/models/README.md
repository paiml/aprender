---
license: mit
tags:
  - tsp
  - traveling-salesman-problem
  - combinatorial-optimization
  - ant-colony-optimization
  - metaheuristics
  - rust
library_name: aprender-tsp
pipeline_tag: other
---

# aprender-tsp POC Models

Pre-trained TSP (Traveling Salesman Problem) optimization models using Ant Colony Optimization, built with [aprender-tsp](https://github.com/paiml/aprender).

## Models Included

| Model | Instance | Cities | Optimal | Achieved | Gap | Tier |
|-------|----------|--------|---------|----------|-----|------|
| berlin52-aco.apr | berlin52 | 52 | 7,542 | 7,687 | 1.92% | Good |
| att48-aco.apr | att48 | 48 | 10,628 | 11,085 | 4.30% | Acceptable |
| eil51-aco.apr | eil51 | 51 | 426 | 443 | 4.07% | Acceptable |

All models achieve < 5% gap from TSPLIB optimal solutions.

## Quick Start

```bash
# Install aprender-tsp
cargo install aprender-tsp

# Download a model
huggingface-cli download paiml/aprender-tsp-poc berlin52-aco.apr

# Solve a new instance using the model
aprender-tsp solve -m berlin52-aco.apr your-instance.tsp

# View model info
aprender-tsp info berlin52-aco.apr

# Benchmark against known optimal
aprender-tsp benchmark berlin52-aco.apr --instances berlin52.tsp
```

## Training Parameters

All models trained with identical ACO parameters for reproducibility:

| Parameter | Value | Description |
|-----------|-------|-------------|
| Algorithm | ACO (Ant Colony Optimization) | Uses core aprender::AntColony |
| Iterations | 2000 | Number of optimization iterations |
| Ants | 20 | Number of artificial ants |
| Alpha (α) | 1.0 | Pheromone importance |
| Beta (β) | 2.5 | Heuristic importance |
| Rho (ρ) | 0.1 | Evaporation rate |
| Seed | 42 | Random seed for reproducibility |

## Instance Sources

Models are trained on standard TSPLIB benchmark instances:

- **berlin52**: 52 locations in Berlin, Germany (Groetschel)
- **att48**: 48 state capitals of the contiguous USA (Padberg/Rinaldi)
- **eil51**: 51-city problem (Christofides/Eilon)

Reference: [TSPLIB](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/)

## File Format

Models use the `.apr` binary format:
- Magic bytes: `APR\0`
- Version: 1
- CRC32 checksum for integrity
- Compact size: ~77 bytes per model

## Solution Quality Tiers

| Tier | Gap from Optimal |
|------|------------------|
| Optimal | < 0.1% |
| Excellent | < 1% |
| Good | < 2% |
| Acceptable | < 5% |
| Poor | >= 5% |

## Train Your Own

```bash
# Train on your instance
aprender-tsp train your-instance.tsp -o your-model.apr --algorithm aco --iterations 2000 --seed 42

# Or use other algorithms
aprender-tsp train your-instance.tsp -o model.apr --algorithm tabu   # Tabu Search (2-opt)
aprender-tsp train your-instance.tsp -o model.apr --algorithm ga     # Genetic Algorithm
aprender-tsp train your-instance.tsp -o model.apr --algorithm hybrid # GA + Tabu + ACO
```

## Citation

```bibtex
@software{aprender,
  title = {Aprender: Machine Learning in Pure Rust},
  author = {PAIML},
  url = {https://github.com/paiml/aprender},
  year = {2025}
}
```

## License

MIT License - see LICENSE file.
