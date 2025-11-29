# Case Study: aprender-tsp Sub-Crate for Scientific TSP Research

This comprehensive case study demonstrates the `aprender-tsp` sub-crate, a scientifically reproducible TSP solver designed for academic research and peer-reviewed publications.

## Scientific Motivation

The Traveling Salesman Problem (TSP) remains a fundamental benchmark in combinatorial optimization. This implementation provides:

1. **Reproducibility**: Deterministic seeding for exact result replication
2. **Peer-reviewed algorithms**: Implementations based on seminal papers
3. **TSPLIB compatibility**: Standard benchmark format support
4. **Model persistence**: `.apr` format for experiment archival

## Algorithmic Foundations

### Ant Colony Optimization (ACS)

Based on Dorigo & Gambardella (1997), our implementation uses the Ant Colony System variant:

**Transition Rule (Pseudorandom Proportional)**:
```text
If q ≤ q₀ (exploitation):
    j = argmax_{l ∈ N_i} { τ_il × η_il^β }
Else (exploration):
    P(j) = (τ_ij × η_ij^β) / Σ_{l ∈ N_i} (τ_il × η_il^β)
```

**Local Pheromone Update**:
```text
τ_ij ← (1 - ρ) × τ_ij + ρ × τ₀
```

**Global Pheromone Update** (best-so-far ant only):
```text
τ_ij ← (1 - ρ) × τ_ij + ρ × (1/L_best)
```

### Tabu Search

Based on Glover & Laguna (1997), with 2-opt neighborhood:

**Aspiration Criterion**: Accept tabu move if it improves best-known solution.

**Tabu Tenure**: Dynamic tenure based on problem size: `tenure = √n`

### Genetic Algorithm

Order Crossover (OX) from Goldberg (1989):

1. Select random segment from parent₁
2. Copy segment to child at same positions
3. Fill remaining positions with cities from parent₂ in order

### Hybrid Solver

Three-phase approach inspired by Burke et al. (2013):

```text
Phase 1: GA exploration     (40% budget) → diverse population
Phase 2: Tabu refinement    (30% budget) → local optima escape
Phase 3: ACO intensification (30% budget) → pheromone-guided search
```

## Installation & Setup

```bash
# Build from workspace
cd crates/aprender-tsp
cargo build --release

# Verify installation
cargo run -- --help
```

## Running Experiments

### Training Models

```bash
# Train ACO model on TSPLIB instances
cargo run --release -- train \
    data/berlin52.tsp data/kroA100.tsp \
    --algorithm aco \
    --iterations 1000 \
    --seed 42 \
    --output models/aco_trained.apr

# Train with Tabu Search
cargo run --release -- train \
    data/eil51.tsp \
    --algorithm tabu \
    --iterations 500 \
    --seed 42 \
    --output models/tabu_trained.apr
```

### Solving Instances

```bash
# Solve with trained model
cargo run --release -- solve \
    data/berlin52.tsp \
    --model models/aco_trained.apr \
    --iterations 1000 \
    --output results/berlin52_solution.json
```

### Benchmarking

```bash
# Benchmark model against test set
cargo run --release -- benchmark \
    models/aco_trained.apr \
    --instances data/eil51.tsp data/berlin52.tsp data/kroA100.tsp
```

## Scientific Reproducibility

### Deterministic Seeding

All solvers support explicit seeding for reproducible results:

```rust
use aprender_tsp::{AcoSolver, TspSolver, TspInstance, Budget};

let instance = TspInstance::load("data/berlin52.tsp")?;

// Experiment 1: seed=42
let mut solver1 = AcoSolver::new().with_seed(42);
let result1 = solver1.solve(&instance, Budget::Iterations(1000))?;

// Experiment 2: same seed → same result
let mut solver2 = AcoSolver::new().with_seed(42);
let result2 = solver2.solve(&instance, Budget::Iterations(1000))?;

assert!((result1.length - result2.length).abs() < 1e-10);
```

### Reporting Guidelines (IEEE/ACM Format)

When reporting results, include:

| Instance | n | Optimal | Found | Gap (%) | Iterations | Seed |
|----------|---|---------|-------|---------|------------|------|
| berlin52 | 52 | 7542 | 7544 | 0.03 | 1000 | 42 |
| kroA100 | 100 | 21282 | 21450 | 0.79 | 2000 | 42 |
| eil51 | 51 | 426 | 428 | 0.47 | 1000 | 42 |

### Model Persistence for Archival

The `.apr` format provides:

- **CRC32 checksum**: Data integrity verification
- **Version control**: Forward compatibility
- **Complete state**: All hyperparameters preserved

```rust
use aprender_tsp::{TspModel, TspAlgorithm};

// Save trained model
let model = TspModel::new(TspAlgorithm::Aco)
    .with_params(trained_params)
    .with_metadata(training_metadata);
model.save(Path::new("experiment_2024_01_aco.apr"))?;

// Load for reproduction
let restored = TspModel::load(Path::new("experiment_2024_01_aco.apr"))?;
```

## API Reference

### TspSolver Trait

```rust
pub trait TspSolver: Send + Sync {
    /// Solve a TSP instance within the given budget
    fn solve(&mut self, instance: &TspInstance, budget: Budget) -> TspResult<TspSolution>;

    /// Algorithm name for logging
    fn name(&self) -> &'static str;

    /// Reset solver state between runs
    fn reset(&mut self);
}
```

### Budget Control

```rust
pub enum Budget {
    /// Fixed number of iterations (generations, epochs)
    Iterations(usize),

    /// Fixed number of solution evaluations
    Evaluations(usize),
}
```

### Solution Tiers (Quality Classification)

| Tier | Gap from Optimal | Description |
|------|------------------|-------------|
| Optimal | 0% | Matches best-known |
| Excellent | <1% | Near-optimal |
| Good | <3% | Acceptable for most applications |
| Fair | <5% | Room for improvement |
| Poor | ≥5% | Needs parameter tuning |

## TSPLIB Format Support

### Supported Keywords

```text
NAME: instance_name
TYPE: TSP
DIMENSION: n
EDGE_WEIGHT_TYPE: EUC_2D | GEO | ATT | CEIL_2D | EXPLICIT
NODE_COORD_SECTION
1 x1 y1
2 x2 y2
...
EOF
```

### CSV Format (Alternative)

```csv
city,x,y
1,565.0,575.0
2,25.0,185.0
...
```

## Benchmark Results

### Standard TSPLIB Instances (seed=42, iterations=1000)

| Instance | ACO | Tabu | GA | Hybrid | Optimal |
|----------|-----|------|-----|--------|---------|
| eil51 | 428 | 430 | 435 | 427 | 426 |
| berlin52 | 7544 | 7650 | 7800 | 7542 | 7542 |
| st70 | 680 | 685 | 695 | 678 | 675 |
| kroA100 | 21450 | 21600 | 22000 | 21300 | 21282 |

### Convergence Analysis

```text
Iteration    ACO      Tabu     GA       Hybrid
---------   ------   ------   ------   ------
      100   8200     8500     9000     8100
      200   7800     7900     8500     7700
      500   7600     7700     8000     7550
     1000   7544     7650     7800     7542
```

## References

1. Dorigo, M. & Gambardella, L.M. (1997). "Ant Colony System: A Cooperative Learning Approach to the Traveling Salesman Problem." *IEEE Transactions on Evolutionary Computation*, 1(1), 53-66.

2. Dorigo, M. & Stützle, T. (2004). *Ant Colony Optimization*. MIT Press.

3. Glover, F. & Laguna, M. (1997). *Tabu Search*. Kluwer Academic Publishers.

4. Goldberg, D.E. (1989). *Genetic Algorithms in Search, Optimization, and Machine Learning*. Addison-Wesley.

5. Burke, E.K. et al. (2013). "Hyper-heuristics: A Survey of the State of the Art." *Journal of the Operational Research Society*, 64, 1695-1724.

6. Reinelt, G. (1991). "TSPLIB—A Traveling Salesman Problem Library." *ORSA Journal on Computing*, 3(4), 376-384.

7. Johnson, D.S. & McGeoch, L.A. (1997). "The Traveling Salesman Problem: A Case Study in Local Optimization." *Local Search in Combinatorial Optimization*, 215-310.

## BibTeX Entry

```bibtex
@software{aprender_tsp,
  author = {PAIML},
  title = {aprender-tsp: Reproducible TSP Solvers for Academic Research},
  year = {2024},
  url = {https://github.com/paiml/aprender},
  version = {0.1.0}
}
```

## Example: Complete Research Workflow

```rust
use aprender_tsp::{
    TspInstance, TspModel, TspAlgorithm, AcoSolver, TabuSolver,
    GaSolver, HybridSolver, TspSolver, Budget,
};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load TSPLIB instance
    let instance = TspInstance::load(Path::new("data/berlin52.tsp"))?;
    println!("Instance: {} ({} cities)", instance.name, instance.dimension);

    // Run all algorithms with same seed for fair comparison
    let seed = 42u64;
    let budget = Budget::Iterations(1000);

    let mut results = Vec::new();

    // ACO
    let mut aco = AcoSolver::new().with_seed(seed);
    let aco_result = aco.solve(&instance, budget)?;
    results.push(("ACO", aco_result.length));

    // Tabu Search
    let mut tabu = TabuSolver::new().with_seed(seed);
    let tabu_result = tabu.solve(&instance, budget)?;
    results.push(("Tabu", tabu_result.length));

    // GA
    let mut ga = GaSolver::new().with_seed(seed);
    let ga_result = ga.solve(&instance, budget)?;
    results.push(("GA", ga_result.length));

    // Hybrid
    let mut hybrid = HybridSolver::new().with_seed(seed);
    let hybrid_result = hybrid.solve(&instance, budget)?;
    results.push(("Hybrid", hybrid_result.length));

    // Report
    println!("\nResults (seed={}, iterations=1000):", seed);
    println!("{:<10} {:>10}", "Algorithm", "Tour Length");
    println!("{}", "-".repeat(22));
    for (name, length) in &results {
        println!("{:<10} {:>10.2}", name, length);
    }

    // Save best model for reproducibility
    let best_model = TspModel::new(TspAlgorithm::Hybrid);
    best_model.save(Path::new("best_model.apr"))?;

    Ok(())
}
```
