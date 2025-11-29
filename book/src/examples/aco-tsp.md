# Ant Colony Optimization for TSP

This example demonstrates Ant Colony Optimization (ACO) solving the Traveling Salesman Problem (TSP), a classic combinatorial optimization problem.

## Problem Description

The Traveling Salesman Problem asks: given a list of cities and distances between them, what is the shortest route that visits each city exactly once and returns to the starting city?

**Why it's hard:**
- For n cities, there are (n-1)!/2 possible tours
- 10 cities → 181,440 tours
- 20 cities → 60+ quintillion tours
- Exact algorithms become intractable for large n

## Ant Colony Optimization

ACO is a swarm intelligence algorithm inspired by how real ants find shortest paths to food sources using pheromone trails.

### Key Concepts

1. **Pheromone Trails (τ)**: Ants deposit pheromones on edges they traverse
2. **Heuristic Information (η)**: Typically η = 1/distance (prefer shorter edges)
3. **Probabilistic Selection**: Next city chosen with probability proportional to τ^α × η^β
4. **Evaporation**: Old pheromones decay, preventing convergence to suboptimal solutions

### Algorithm Flow

```text
┌─────────────────────────────────────────────────────────┐
│  1. Initialize pheromone trails uniformly               │
│                      ↓                                   │
│  2. Each ant constructs a complete tour                 │
│     - Start from random city                            │
│     - Select next city: P(j) ∝ τᵢⱼ^α × ηᵢⱼ^β           │
│     - Repeat until all cities visited                   │
│                      ↓                                   │
│  3. Evaluate tour quality (total distance)              │
│                      ↓                                   │
│  4. Update pheromones                                   │
│     - Evaporation: τ = (1-ρ)τ                           │
│     - Deposit: τᵢⱼ += 1/tour_length for good tours      │
│                      ↓                                   │
│  5. Repeat until budget exhausted                       │
└─────────────────────────────────────────────────────────┘
```

## Running the Example

```bash
cargo run --example aco_tsp
```

## Code Walkthrough

### Setup

```rust,ignore
use aprender::metaheuristics::{AntColony, Budget, ConstructiveMetaheuristic, SearchSpace};

// Distance matrix for 10 US cities (miles)
let distances: Vec<Vec<f64>> = vec![
    vec![0.0, 1100.0, 720.0, ...],  // Atlanta
    vec![1100.0, 0.0, 980.0, ...],  // Boston
    // ... etc
];

// Build adjacency list for graph search space
let adjacency: Vec<Vec<(usize, f64)>> = distances
    .iter()
    .enumerate()
    .map(|(i, row)| {
        row.iter()
            .enumerate()
            .filter(|&(j, _)| i != j)
            .map(|(j, &d)| (j, d))
            .collect()
    })
    .collect();

let space = SearchSpace::Graph {
    num_nodes: 10,
    adjacency,
    heuristic: None,  // ACO computes 1/distance automatically
};
```

### Objective Function

```rust,ignore
let objective = |tour: &Vec<usize>| -> f64 {
    let mut total = 0.0;
    for i in 0..tour.len() {
        let from = tour[i];
        let to = tour[(i + 1) % tour.len()];  // Wrap to start
        total += distances[from][to];
    }
    total
};
```

### ACO Configuration

```rust,ignore
let mut aco = AntColony::new(20)  // 20 ants per iteration
    .with_alpha(1.0)              // Pheromone importance
    .with_beta(2.5)               // Heuristic importance (distance)
    .with_rho(0.1)                // 10% evaporation rate
    .with_seed(42);

let result = aco.optimize(&objective, &space, Budget::Iterations(100));
```

### Parameter Tuning Guide

| Parameter | Typical Range | Effect |
|-----------|---------------|--------|
| `num_ants` | 10-50 | More ants → better exploration, more compute |
| `alpha` | 0.5-2.0 | Higher → more influence from pheromones |
| `beta` | 2.0-5.0 | Higher → greedier (prefer short edges) |
| `rho` | 0.02-0.2 | Higher → faster forgetting, more exploration |

## Sample Output

```text
=== Ant Colony Optimization: Traveling Salesman Problem ===

Best tour found:
  Chicago -> Green Bay -> Indianapolis -> Boston -> Jacksonville
  -> Atlanta -> Houston -> El Paso -> Fresno -> Denver -> Chicago

Total distance: 7550 miles
Iterations: 100

Convergence:
  Iter   0: 8370 miles
  Iter  10: 7630 miles
  Iter  20: 7550 miles  (optimal found)

Comparison with Greedy:
  Greedy: 9320 miles
  ACO:    7550 miles
  Improvement: 19.0% (1770 miles saved)
```

## When to Use ACO

**Good for:**
- TSP and routing problems
- Scheduling and sequencing
- Network routing
- Any problem with graph structure

**Consider alternatives when:**
- Continuous optimization (use DE or PSO)
- Very large problems (>1000 nodes) without good heuristics
- Real-time requirements (ACO needs many iterations)

## Variants

Aprender implements the classic **Ant System (AS)**. More advanced variants include:

| Variant | Key Feature |
|---------|-------------|
| **MMAS** (Max-Min AS) | Bounds on pheromone levels |
| **ACS** (Ant Colony System) | Local pheromone update + q₀ exploitation |
| **Rank-Based AS** | Only best k ants deposit pheromone |

## References

1. Dorigo, M. & Stützle, T. (2004). *Ant Colony Optimization*. MIT Press.
2. Dorigo, M. et al. (1996). "The Ant System: Optimization by a Colony of Cooperating Agents." *IEEE Transactions on Systems, Man, and Cybernetics*, 26(1), 29-41.
