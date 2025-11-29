# Tabu Search for TSP

This example demonstrates Tabu Search solving the Traveling Salesman Problem using memory-based local search with swap moves.

## Problem Description

Given 8 European capital cities, find the shortest tour visiting each exactly once and returning to the start.

## Tabu Search Algorithm

Tabu Search is a memory-based local search that prevents cycling by maintaining a "tabu list" of recently visited moves.

### Key Concepts

1. **Neighborhood**: All solutions reachable by a single move (e.g., swap two cities)
2. **Tabu List**: Recent moves that are forbidden for `tenure` iterations
3. **Aspiration Criteria**: Override tabu status if move leads to global best
4. **Intensification/Diversification**: Balance exploitation and exploration

### Algorithm Flow

```text
┌─────────────────────────────────────────────────────────┐
│  1. Start with random initial solution                  │
│                      ↓                                   │
│  2. Generate neighborhood (all swap moves)              │
│                      ↓                                   │
│  3. Select best non-tabu move                           │
│     - Unless aspiration: move gives new global best     │
│                      ↓                                   │
│  4. Apply move, add to tabu list                        │
│                      ↓                                   │
│  5. Remove expired entries from tabu list               │
│                      ↓                                   │
│  6. Update global best if improved                      │
│                      ↓                                   │
│  7. Repeat until budget exhausted                       │
└─────────────────────────────────────────────────────────┘
```

## Running the Example

```bash
cargo run --example tabu_tsp
```

## Code Walkthrough

### Setup

```rust,ignore
use aprender::metaheuristics::{Budget, ConstructiveMetaheuristic, SearchSpace, TabuSearch};

// 8 European capitals with distances (km)
let city_names = ["Paris", "Berlin", "Rome", "Madrid",
                  "Vienna", "Amsterdam", "Prague", "Brussels"];

let distances: Vec<Vec<f64>> = vec![
    vec![0.0, 878.0, 1106.0, 1054.0, 1034.0, 430.0, 885.0, 265.0],  // Paris
    // ... etc
];

let space = SearchSpace::Permutation { size: 8 };
```

### Objective Function

```rust,ignore
let objective = |tour: &Vec<usize>| -> f64 {
    let mut total = 0.0;
    for i in 0..tour.len() {
        let from = tour[i];
        let to = tour[(i + 1) % tour.len()];
        total += distances[from][to];
    }
    total
};
```

### Tabu Search Configuration

```rust,ignore
let tenure = 7;  // Moves stay tabu for 7 iterations
let mut ts = TabuSearch::new(tenure)
    .with_max_neighbors(500)  // Evaluate up to 500 swaps
    .with_seed(42);

let result = ts.optimize(&objective, &space, Budget::Iterations(200));
```

### Parameter Tuning Guide

| Parameter | Typical Range | Effect |
|-----------|---------------|--------|
| `tenure` | n/4 to n | Higher → more exploration, slower convergence |
| `max_neighbors` | 100-1000 | Higher → better moves, more compute |

**Tenure selection heuristics:**
- Small problems (n < 20): tenure ≈ 5-10
- Medium (20-100): tenure ≈ n/3
- Large (>100): tenure ≈ √n

## Sample Output

```text
=== Tabu Search: Traveling Salesman Problem ===

Best tour found:
  Vienna -> Rome -> Madrid -> Paris -> Brussels
  -> Amsterdam -> Berlin -> Prague -> Vienna

Total distance: 4731 km
Iterations: 200

Leg-by-Leg Breakdown:
  Vienna -> Rome: 765 km
  Rome -> Madrid: 1365 km
  Madrid -> Paris: 1054 km
  Paris -> Brussels: 265 km
  Brussels -> Amsterdam: 173 km
  Amsterdam -> Berlin: 577 km
  Berlin -> Prague: 280 km
  Prague -> Vienna: 252 km

Sensitivity Analysis (Tabu Tenure):
  Tenure  3: 4731 km
  Tenure  5: 4731 km
  Tenure 10: 4731 km
  Tenure 15: 4731 km
```

## Swap Move Neighborhood

For a permutation of n elements, there are n(n-1)/2 possible swap moves:

```text
Tour: [A, B, C, D, E]

Swap(0,1) → [B, A, C, D, E]
Swap(0,2) → [C, B, A, D, E]
Swap(0,3) → [D, B, C, A, E]
...
Swap(3,4) → [A, B, C, E, D]

Total: 5×4/2 = 10 possible swaps
```

## Comparison: Tabu Search vs ACO

| Aspect | Tabu Search | ACO |
|--------|-------------|-----|
| **Type** | Single-solution local search | Population-based construction |
| **Memory** | Explicit tabu list | Implicit via pheromones |
| **Exploration** | Via diversification | Via randomization |
| **Best for** | Refining good solutions | Broad exploration |
| **Parallelism** | Limited | High (many ants) |

**Hybrid approach**: Use ACO to find initial solution, refine with Tabu Search.

## When to Use Tabu Search

**Good for:**
- Combinatorial optimization (scheduling, assignment)
- Refining solutions from other methods
- Problems with good neighborhood structure
- When solution quality matters more than speed

**Consider alternatives when:**
- Need highly parallel execution (use ACO or GA)
- Continuous optimization (use DE or PSO)
- Very large neighborhoods (sampling may miss good moves)

## Advanced Features

### Aspiration Criteria

The basic aspiration criterion accepts a tabu move if it produces a new global best:

```rust,ignore
let is_aspiration = new_value < self.best_value;
let is_tabu = Self::is_tabu(mv, &tabu_list, iteration);

if (!is_tabu || is_aspiration) && new_value < best_move_value {
    best_move = Some(*mv);
}
```

### Strategic Oscillation

Alternate between intensification (short tenure, exploit good regions) and diversification (long tenure, explore broadly).

## References

1. Glover, F. & Laguna, M. (1997). *Tabu Search*. Kluwer Academic.
2. Gendreau, M. & Potvin, J.Y. (2010). *Handbook of Metaheuristics*. Springer.
