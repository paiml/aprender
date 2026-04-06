# TSP GRASP Demo

Demo 6 in the EDD (Equation-Driven Development) showcase series demonstrates the **GRASP** methodology (Greedy Randomized Adaptive Search Procedure) applied to the classic **Traveling Salesman Problem (TSP)**.

## Overview

The TSP asks: given a set of cities, find the shortest tour that visits each city exactly once and returns to the starting city. This NP-hard problem is a cornerstone of combinatorial optimization.

GRASP combines:
1. **Randomized Greedy Construction** - Build initial solutions using a restricted candidate list (RCL)
2. **Local Search (2-opt)** - Improve solutions by swapping edges
3. **Multiple Restarts** - Run many iterations to escape local optima

## Governing Equations

### Tour Length

```text
L(π) = Σᵢ d(π(i), π(i+1)) + d(π(n), π(1))
```

The total distance of a tour π visiting n cities.

### 2-Opt Improvement

```text
Δ = d(i,i+1) + d(j,j+1) - d(i,j) - d(i+1,j+1)
```

When Δ > 0, reversing the segment between cities i+1 and j improves the tour.

### Beardwood-Halton-Hammersley Constant

```text
E[L_greedy] ≈ 0.7124 · √(n·A)
```

For n random points uniformly distributed in area A, the expected optimal tour length approaches this formula asymptotically.

### MST Lower Bound

```text
L* ≥ MST(G)
```

The optimal tour length is at least as long as the minimum spanning tree of the distance graph.

## EDD 5-Phase Cycle

### Phase 1: Equation

The governing equations define what we're measuring:
- Tour length calculation
- 2-opt improvement detection
- Lower bound estimation via MST

### Phase 2: Failing Test

Tests that demonstrate the problem before GRASP:
- Random construction yields tours 30-50% longer than greedy
- Single restart has high variance
- No 2-opt leaves obvious crossings

### Phase 3: Implementation

The `TspGraspDemo` struct implements:
- Three construction methods: `RandomizedGreedy`, `NearestNeighbor`, `Random`
- Precomputed distance matrix for O(1) lookups
- RCL-based greedy construction
- Full 2-opt local search

### Phase 4: Verification

Tests confirm:
- GRASP achieves reasonable optimality gap (<40% vs MST bound)
- Multiple restarts converge (CV < 10%)
- 2-opt removes edge crossings

### Phase 5: Falsification

Conditions that break the model:
- Random construction is provably worse than greedy
- Verified across multiple random seeds

## Usage

```rust,ignore
use simular::demos::tsp_grasp::{TspGraspDemo, ConstructionMethod};
use simular::demos::EddDemo;

// Create demo with 25 random cities
let mut demo = TspGraspDemo::new(42, 25);

// Configure GRASP parameters
demo.set_rcl_size(5);  // Top 5 candidates in RCL
demo.set_construction_method(ConstructionMethod::RandomizedGreedy);

// Run 20 GRASP iterations
demo.run_grasp(20);

// Check results
println!("Best tour length: {:.4}", demo.best_tour_length);
println!("Optimality gap: {:.1}%", demo.optimality_gap() * 100.0);
println!("Verified: {}", demo.verify_equation());
```

## Construction Methods

### Randomized Greedy (Default)

At each step:
1. Find the k nearest unvisited cities (RCL)
2. Randomly select one from the RCL
3. Add to tour

This balances exploitation (greedy) with exploration (randomization).

### Nearest Neighbor

Pure greedy: always select the nearest unvisited city. Deterministic but can get stuck in poor local optima.

### Random

Shuffle cities randomly. Poor initial quality but useful for falsification tests.

## 2-Opt Local Search

The 2-opt algorithm:
1. Consider all pairs of non-adjacent edges
2. If reversing the segment between them shortens the tour, apply the swap
3. Repeat until no improving swap exists

This runs to local optimum after each construction.

## Running the Example

```bash
cargo run --example tsp_grasp_demo
```

Sample output:

```text
╔══════════════════════════════════════════════════════════════╗
║           TSP GRASP Demo - EDD Showcase Demo 6              ║
╠══════════════════════════════════════════════════════════════╣
║  Greedy Randomized Adaptive Search Procedure for TSP        ║
╚══════════════════════════════════════════════════════════════╝

═══ Phase 1: Governing Equations ═══

Tour Length:           L(π) = Σᵢ d(π(i), π(i+1)) + d(π(n), π(1))
2-Opt Improvement:     Δ = d(i,i+1) + d(j,j+1) - d(i,j) - d(i+1,j+1)
Expected Greedy Tour:  E[L] ≈ 0.7124·√(n·A)  [Beardwood-Halton-Hammersley]
Lower Bound:           L* ≥ MST(G)

═══ Phase 2: Failing Test - Random vs Greedy Construction ═══

Problem: 25 random cities in unit square
Random construction tour length:  6.8234
Greedy construction tour length:  4.2156
Greedy improvement: 38.2%

✗ Random construction fails: tour is 61.9% longer than greedy
```

## References

1. Feo & Resende (1995) "Greedy Randomized Adaptive Search Procedures" - Journal of Global Optimization
2. Lin & Kernighan (1973) "An Effective Heuristic Algorithm for the TSP" - Operations Research
3. Johnson & McGeoch (1997) "The TSP: A Case Study in Local Optimization" - Local Search in Combinatorial Optimization
4. Christofides (1976) "Worst-Case Analysis of a New Heuristic for the TSP"
5. Beardwood, Halton & Hammersley (1959) "The Shortest Path Through Many Points" - Proc. Cambridge Phil. Soc.

## WASM Integration

The demo exports WebAssembly bindings for interactive visualization in web browsers.

### Running the WASM Demo

```bash
make serve-tsp
```

This serves the demo at `http://localhost:8080/tsp.html` and auto-opens your browser.

#### Building from Source

```bash
# Build WASM package
wasm-pack build --target web --no-default-features --features wasm

# Copy to web directory
cp pkg/* web/pkg/
```

### Features

- **▶ Play** - Real-time convergence animation (~20 iterations/sec)
- **Step** - Single GRASP iteration
- **Run 10/100** - Batch iterations
- **Reset** - New random seed

### TUI/WASM Parity

Both the TUI (`cargo run --bin tsp_tui`) and WASM demos share:
- Same YAML configuration (`bay_area_tsp.yaml`)
- Same governing equations
- Same color scheme (via `edd::style` module)
- 100% GUI coverage (22 elements, 4 screens, 5 journeys)

The shared style constants ensure visual consistency:

```rust,ignore
use simular::edd::style;

// Both TUI and WASM use these colors
assert_eq!(style::ACCENT, "#4ecdc4");     // Teal tour lines
assert_eq!(style::CITY_NODE, "#ffd93d");  // Yellow city dots
assert_eq!(style::BG_PRIMARY, "#0a0a1a"); // Dark background
```

### Probar Integration

The WASM demo includes embedded probar tests:

```javascript
// Run in-browser tests via the Probar Tests tab
// Tests include:
// - Valid permutation (visits all cities once)
// - Closed tour (forms loop)
// - Monotonic improvement
// - Deterministic replay
```

All 54 probar tests pass, ensuring TUI/WASM behavioral parity.
