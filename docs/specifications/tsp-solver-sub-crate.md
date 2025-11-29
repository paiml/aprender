# aprender-tsp: Local TSP Optimization with .apr Models

**Version:** 1.0
**Date:** 2025-11-29
**Status:** Specification
**Target Release:** v0.13.0
**Methodology:** EXTREME TDD + Toyota Way (Genchi Genbutsu, Kaizen, Jidoka)

---

## Executive Summary

`aprender-tsp` is a command-line tool enabling users to train, optimize, and deploy personalized Traveling Salesman Problem (TSP) solvers using local `.apr` model files. Following the successful pattern established by `aprender-shell`, this sub-crate empowers users to:

1. **Train personalized TSP models** from their own problem instances
2. **Optimize routes** using state-of-the-art metaheuristics (ACO, Tabu Search, Genetic Algorithm)
3. **Export solutions** in standard formats (JSON, CSV, GeoJSON)
4. **Deploy offline** with zero cloud dependency

**Toyota Way Principle:** *Genchi Genbutsu* (Go and see) - Users understand their logistics problems best; we provide tools, not prescriptive solutions.

---

## Table of Contents

1. [Design Philosophy](#1-design-philosophy)
2. [The .apr Model Format for TSP](#2-the-apr-model-format-for-tsp)
3. [CLI Interface](#3-cli-interface)
4. [Metaheuristic Backends](#4-metaheuristic-backends)
5. [Problem Instance Formats](#5-problem-instance-formats)
6. [Training Pipeline](#6-training-pipeline)
7. [Solution Quality Metrics](#7-solution-quality-metrics)
8. [Implementation Architecture](#8-implementation-architecture)
9. [Quality Standards](#9-quality-standards)
10. [Academic References](#10-academic-references)

---

## 1. Design Philosophy

### 1.1 Local-First, User-Owned Models

**Toyota Way Principle:** *Respect for People* - Users own their data and models.

```
┌─────────────────────────────────────────────────────────────────┐
│                     aprender-tsp Architecture                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐   │
│   │   Problem    │────>│   Training   │────>│  .apr Model  │   │
│   │  Instances   │     │   Pipeline   │     │   (Local)    │   │
│   │  (TSPLIB)    │     │              │     │              │   │
│   └──────────────┘     └──────────────┘     └──────────────┘   │
│                                                    │            │
│                                                    ▼            │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐   │
│   │   Solution   │<────│  Optimizer   │<────│   Solver     │   │
│   │   Output     │     │   Engine     │     │   CLI        │   │
│   │  (JSON/CSV)  │     │              │     │              │   │
│   └──────────────┘     └──────────────┘     └──────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Why Local .apr Models?

| Aspect | Cloud-Based Solvers | aprender-tsp (.apr) |
|--------|---------------------|---------------------|
| **Privacy** | Problem data uploaded | Data never leaves device |
| **Latency** | Network round-trip | Sub-second local inference |
| **Cost** | Per-query pricing | Free after training |
| **Customization** | Generic models | Trained on your problem distribution |
| **Offline** | Requires internet | Works anywhere |

**Toyota Way Principle:** *Jidoka* (Build quality in) - Quality comes from understanding the specific problem domain, not generic cloud APIs.

### 1.3 Kaizen: Continuous Model Improvement

Users can incrementally improve their TSP models:

```bash
# Initial training on historical routes
aprender-tsp train routes-2024-q1.tsp --output delivery.apr

# Incremental update with new data
aprender-tsp update delivery.apr routes-2024-q2.tsp

# Evaluate model quality
aprender-tsp benchmark delivery.apr --instances test-set/
```

---

## 2. The .apr Model Format for TSP

### 2.1 Model Structure

The `.apr` file stores learned parameters for TSP optimization:

```
┌────────────────────────────────────────────────────┐
│                .apr TSP Model Layout               │
├────────────────────────────────────────────────────┤
│  Magic Number: "APR\x00"           (4 bytes)      │
│  Version: 1                        (4 bytes)      │
│  Model Type: TSP_SOLVER            (4 bytes)      │
│  Checksum: CRC32                   (4 bytes)      │
├────────────────────────────────────────────────────┤
│  HEADER SECTION                                    │
│  ├─ algorithm: "aco" | "tabu" | "ga" | "hybrid"   │
│  ├─ trained_instances: u32                         │
│  ├─ avg_instance_size: u32                         │
│  ├─ best_known_gap: f64                            │
│  └─ training_time_secs: f64                        │
├────────────────────────────────────────────────────┤
│  ALGORITHM PARAMETERS                              │
│  ├─ For ACO:                                       │
│  │   ├─ alpha: f64 (pheromone importance)         │
│  │   ├─ beta: f64 (heuristic importance)          │
│  │   ├─ rho: f64 (evaporation rate)               │
│  │   └─ q0: f64 (exploitation vs exploration)     │
│  ├─ For Tabu:                                      │
│  │   ├─ tenure: u32                                │
│  │   ├─ aspiration_threshold: f64                  │
│  │   └─ diversification_freq: u32                  │
│  └─ For GA:                                        │
│      ├─ population_size: u32                       │
│      ├─ crossover_rate: f64                        │
│      └─ mutation_rate: f64                         │
├────────────────────────────────────────────────────┤
│  LEARNED HEURISTICS                                │
│  ├─ distance_matrix_features: Vec<f64>            │
│  ├─ cluster_centroids: Vec<(f64, f64)>            │
│  └─ edge_scoring_weights: Vec<f64>                │
└────────────────────────────────────────────────────┘
```

### 2.2 Model Serialization

```rust
/// TSP model persisted in .apr format
#[derive(Debug, Clone)]
pub struct TspModel {
    /// Solver algorithm
    pub algorithm: TspAlgorithm,
    /// Learned parameters (algorithm-specific)
    pub params: TspParams,
    /// Training metadata
    pub metadata: TspModelMetadata,
    /// Optional: pre-computed heuristics for common patterns
    pub learned_heuristics: Option<LearnedHeuristics>,
}

impl TspModel {
    /// Save model to .apr file
    pub fn save(&self, path: &Path) -> Result<(), TspError> {
        let mut file = File::create(path)?;

        // Write magic + version + type
        file.write_all(b"APR\x00")?;
        file.write_all(&1u32.to_le_bytes())?;
        file.write_all(&MODEL_TYPE_TSP.to_le_bytes())?;

        // Serialize with bincode (no serde in dependencies)
        let payload = self.serialize_payload()?;

        // Write checksum
        let checksum = crc32fast::hash(&payload);
        file.write_all(&checksum.to_le_bytes())?;

        // Write payload
        file.write_all(&payload)?;

        Ok(())
    }

    /// Load model from .apr file with validation
    pub fn load(path: &Path) -> Result<Self, TspError> {
        let data = std::fs::read(path)?;

        // Verify magic
        if &data[0..4] != b"APR\x00" {
            return Err(TspError::InvalidFormat("Not an .apr file".into()));
        }

        // Verify checksum
        let stored_checksum = u32::from_le_bytes(data[12..16].try_into()?);
        let payload = &data[16..];
        let computed_checksum = crc32fast::hash(payload);

        if stored_checksum != computed_checksum {
            return Err(TspError::ChecksumMismatch {
                expected: stored_checksum,
                computed: computed_checksum,
            });
        }

        Self::deserialize_payload(payload)
    }
}
```

**Toyota Way Principle:** *Standardized Work* - Consistent .apr format enables reproducible results across environments.

---

## 3. CLI Interface

### 3.1 Command Structure

```bash
aprender-tsp <COMMAND>

Commands:
  train      Train a TSP model from problem instances
  solve      Solve a TSP instance using a trained model
  benchmark  Evaluate model against known optimal solutions
  update     Incrementally update a model with new instances
  export     Export solutions to various formats
  info       Display model information and statistics
```

### 3.2 Training Command

```bash
# Train from TSPLIB format
aprender-tsp train instances/*.tsp \
    --algorithm aco \
    --output delivery-routes.apr \
    --iterations 1000 \
    --seed 42

# Train with hyperparameter tuning
aprender-tsp train instances/*.tsp \
    --algorithm auto \           # Auto-select best algorithm
    --tune-budget 10000 \        # Evaluation budget for tuning
    --output optimized.apr
```

### 3.3 Solve Command

```bash
# Solve a single instance
aprender-tsp solve today-deliveries.tsp \
    --model delivery-routes.apr \
    --output solution.json \
    --timeout 30s

# Batch solve with parallel execution
aprender-tsp solve batch/*.tsp \
    --model delivery-routes.apr \
    --output-dir solutions/ \
    --parallel 4
```

### 3.4 Example Session

```bash
$ aprender-tsp train data/berlin52.tsp --algorithm aco --output berlin.apr

Training TSP Model
==================
Algorithm:    Ant Colony Optimization
Instances:    1 (berlin52.tsp, 52 cities)
Iterations:   1000

Progress: [========================================] 100%

Training Complete
─────────────────
Best tour length:  7,544.37 (optimal: 7,542)
Gap from optimal:  0.03%
Training time:     2.34s
Model saved:       berlin.apr (12.4 KB)

$ aprender-tsp solve new-instance.tsp --model berlin.apr

Solving TSP Instance
====================
Instance:     new-instance.tsp (48 cities)
Model:        berlin.apr (ACO, trained on berlin52)

Solution Found
──────────────
Tour length:      5,892.45
Computation time: 0.23s
Tour: 0 -> 12 -> 5 -> 33 -> ... -> 0

Output: solution.json
```

---

## 4. Metaheuristic Backends

### 4.1 Ant Colony Optimization (ACO)

**Reference:** Dorigo & Gambardella (1997) [1]

ACO excels at TSP due to its natural graph-based construction:

```rust
/// Ant Colony Optimization for TSP
pub struct AcoSolver {
    /// Number of artificial ants
    num_ants: usize,
    /// Pheromone importance (α)
    alpha: f64,
    /// Heuristic importance (β)
    beta: f64,
    /// Evaporation rate (ρ)
    rho: f64,
    /// Exploitation probability (q₀)
    q0: f64,
    /// Pheromone matrix
    pheromone: Vec<Vec<f64>>,
}

impl AcoSolver {
    /// Construct tour using probabilistic rule
    fn construct_tour(&self, distances: &[Vec<f64>], rng: &mut impl Rng) -> Vec<usize> {
        let n = distances.len();
        let mut tour = Vec::with_capacity(n);
        let mut visited = vec![false; n];

        // Start from random city
        let start = rng.gen_range(0..n);
        tour.push(start);
        visited[start] = true;

        while tour.len() < n {
            let current = *tour.last().unwrap();
            let next = self.select_next_city(current, &visited, distances, rng);
            tour.push(next);
            visited[next] = true;
        }

        tour
    }

    /// Select next city using ACS transition rule
    fn select_next_city(
        &self,
        current: usize,
        visited: &[bool],
        distances: &[Vec<f64>],
        rng: &mut impl Rng,
    ) -> usize {
        // Exploitation vs exploration (ACS rule)
        if rng.gen::<f64>() < self.q0 {
            // Exploitation: choose best
            self.argmax_attractiveness(current, visited, distances)
        } else {
            // Exploration: probabilistic selection
            self.roulette_selection(current, visited, distances, rng)
        }
    }
}
```

### 4.2 Tabu Search

**Reference:** Glover & Laguna (1997) [2]

Best for local refinement with memory-guided exploration:

```rust
/// Tabu Search for TSP
pub struct TabuSolver {
    /// Tabu tenure (moves stay forbidden for this many iterations)
    tenure: usize,
    /// Maximum neighbors to evaluate per iteration
    max_neighbors: usize,
    /// Tabu list: (city_i, city_j) -> forbidden_until_iteration
    tabu_list: HashMap<(usize, usize), usize>,
}

impl TabuSolver {
    /// Apply 2-opt move: reverse segment between i and j
    fn two_opt_move(tour: &mut [usize], i: usize, j: usize) {
        tour[i..=j].reverse();
    }

    /// Evaluate neighborhood using 2-opt moves
    fn get_best_neighbor(
        &self,
        tour: &[usize],
        distances: &[Vec<f64>],
        iteration: usize,
        best_known: f64,
    ) -> Option<(usize, usize, f64)> {
        let n = tour.len();
        let mut best_move = None;
        let mut best_delta = 0.0;

        for i in 0..n-2 {
            for j in i+2..n {
                // Calculate improvement from 2-opt
                let delta = self.two_opt_delta(tour, distances, i, j);

                // Check tabu status (with aspiration)
                let is_tabu = self.is_tabu(tour[i], tour[j], iteration);
                let current_cost = self.tour_length(tour, distances);
                let new_cost = current_cost + delta;

                // Accept if not tabu OR improves best known (aspiration)
                if (!is_tabu || new_cost < best_known) && delta < best_delta {
                    best_delta = delta;
                    best_move = Some((i, j, new_cost));
                }
            }
        }

        best_move
    }
}
```

### 4.3 Genetic Algorithm (GA)

**Reference:** Goldberg (1989) [3]

Effective for large-scale exploration with crossover operators:

```rust
/// Genetic Algorithm for TSP with Order Crossover (OX)
pub struct GaSolver {
    /// Population size
    population_size: usize,
    /// Crossover probability
    crossover_rate: f64,
    /// Mutation probability
    mutation_rate: f64,
    /// Tournament selection size
    tournament_size: usize,
}

impl GaSolver {
    /// Order Crossover (OX) preserving city sequences
    fn order_crossover(
        parent1: &[usize],
        parent2: &[usize],
        rng: &mut impl Rng,
    ) -> Vec<usize> {
        let n = parent1.len();
        let mut child = vec![usize::MAX; n];

        // Select crossover segment
        let start = rng.gen_range(0..n);
        let end = rng.gen_range(start..n);

        // Copy segment from parent1
        for i in start..=end {
            child[i] = parent1[i];
        }

        // Fill remaining from parent2 in order
        let mut pos = (end + 1) % n;
        for &city in parent2.iter().cycle().skip(end + 1).take(n) {
            if !child.contains(&city) {
                child[pos] = city;
                pos = (pos + 1) % n;
                if pos == start {
                    break;
                }
            }
        }

        child
    }

    /// 2-opt mutation
    fn mutate(&self, tour: &mut [usize], rng: &mut impl Rng) {
        if rng.gen::<f64>() < self.mutation_rate {
            let n = tour.len();
            let i = rng.gen_range(0..n);
            let j = rng.gen_range(0..n);
            let (i, j) = if i < j { (i, j) } else { (j, i) };
            tour[i..=j].reverse();
        }
    }
}
```

### 4.4 Hybrid Solver (Auto-Select)

**Toyota Way Principle:** *Kaizen* - Combine strengths of multiple algorithms.

```rust
/// Hybrid solver that combines algorithms
pub struct HybridSolver {
    /// Use GA for global exploration
    ga_iterations: usize,
    /// Refine with Tabu Search
    tabu_iterations: usize,
    /// Polish with ACO intensification
    aco_iterations: usize,
}

impl HybridSolver {
    pub fn solve(&mut self, distances: &[Vec<f64>], budget: Budget) -> Tour {
        // Phase 1: Global exploration with GA
        let ga = GaSolver::default();
        let population = ga.evolve(distances, self.ga_iterations);
        let best_from_ga = population.best();

        // Phase 2: Local refinement with Tabu Search
        let mut tabu = TabuSolver::default();
        let refined = tabu.refine(best_from_ga, distances, self.tabu_iterations);

        // Phase 3: Intensification with ACO (using GA solution as pheromone bias)
        let mut aco = AcoSolver::default();
        aco.seed_pheromone_from_tour(&refined);
        let polished = aco.optimize(distances, self.aco_iterations);

        // Return best overall
        if tour_length(&polished, distances) < tour_length(&refined, distances) {
            polished
        } else {
            refined
        }
    }
}
```

---

## 5. Problem Instance Formats

### 5.1 TSPLIB Format (Standard)

**Reference:** Reinelt (1991) [4]

```
NAME: berlin52
TYPE: TSP
COMMENT: 52 locations in Berlin
DIMENSION: 52
EDGE_WEIGHT_TYPE: EUC_2D
NODE_COORD_SECTION
1 565.0 575.0
2 25.0 185.0
3 345.0 750.0
...
52 1340.0 725.0
EOF
```

### 5.2 Simple CSV Format

```csv
# city_id,x,y[,demand][,time_window_start,time_window_end]
1,565.0,575.0
2,25.0,185.0
3,345.0,750.0
```

### 5.3 Distance Matrix Format

```json
{
  "type": "distance_matrix",
  "dimension": 5,
  "matrix": [
    [0, 10, 15, 20, 25],
    [10, 0, 35, 25, 30],
    [15, 35, 0, 30, 20],
    [20, 25, 30, 0, 15],
    [25, 30, 20, 15, 0]
  ]
}
```

### 5.4 GeoJSON Format (Real-World)

```json
{
  "type": "FeatureCollection",
  "properties": {
    "name": "delivery-route-2024-01-15",
    "depot": 0
  },
  "features": [
    {
      "type": "Feature",
      "properties": {"id": 0, "name": "Warehouse"},
      "geometry": {"type": "Point", "coordinates": [-122.4194, 37.7749]}
    },
    {
      "type": "Feature",
      "properties": {"id": 1, "name": "Customer A"},
      "geometry": {"type": "Point", "coordinates": [-122.4089, 37.7835]}
    }
  ]
}
```

---

## 6. Training Pipeline

### 6.1 Parameter Tuning via Metaheuristics

**Toyota Way Principle:** *Genchi Genbutsu* - Tune parameters on your specific problem distribution.

```rust
/// Auto-tune TSP solver parameters using differential evolution
pub fn tune_parameters(
    instances: &[TspInstance],
    algorithm: TspAlgorithm,
    budget: Budget,
) -> TspParams {
    let space = match algorithm {
        TspAlgorithm::Aco => SearchSpace::Continuous {
            dim: 4, // alpha, beta, rho, q0
            lower: vec![0.1, 0.5, 0.01, 0.0],
            upper: vec![5.0, 10.0, 0.5, 1.0],
        },
        TspAlgorithm::Tabu => SearchSpace::Mixed {
            dim: 3, // tenure, max_neighbors, diversification_freq
            lower: vec![5.0, 10.0, 10.0],
            upper: vec![50.0, 1000.0, 100.0],
            discrete_dims: vec![0, 1, 2],
        },
        TspAlgorithm::Ga => SearchSpace::Continuous {
            dim: 3, // population_size, crossover_rate, mutation_rate
            lower: vec![20.0, 0.5, 0.01],
            upper: vec![200.0, 1.0, 0.3],
        },
    };

    // Objective: average gap from best-known on training instances
    let objective = |params: &[f64]| -> f64 {
        let tsp_params = decode_params(algorithm, params);
        let mut total_gap = 0.0;

        for instance in instances {
            let solver = create_solver(algorithm, &tsp_params);
            let solution = solver.solve(&instance.distances, Budget::Iterations(100));
            let gap = (solution.length - instance.best_known) / instance.best_known;
            total_gap += gap;
        }

        total_gap / instances.len() as f64
    };

    // Use Differential Evolution for tuning
    let mut de = DifferentialEvolution::default().with_seed(42);
    let result = de.optimize(&objective, &space, budget);

    decode_params(algorithm, &result.solution)
}
```

### 6.2 Learned Heuristics

Train model to learn problem-specific patterns:

```rust
/// Learn edge-scoring heuristics from solved instances
pub fn learn_heuristics(
    instances: &[(TspInstance, Tour)], // Instance + optimal tour
) -> LearnedHeuristics {
    // Extract features from optimal tours
    let mut edge_features = Vec::new();

    for (instance, optimal_tour) in instances {
        for window in optimal_tour.cities.windows(2) {
            let (i, j) = (window[0], window[1]);

            // Feature vector for edge (i, j)
            let features = vec![
                instance.distances[i][j],                    // Raw distance
                instance.nearest_neighbor_rank(i, j) as f64, // NN rank
                instance.cluster_distance(i, j),             // Inter-cluster distance
                instance.angle_deviation(i, j),              // Angle from centroid
            ];

            edge_features.push((features, 1.0)); // 1.0 = in optimal tour
        }

        // Negative examples: edges NOT in optimal tour
        for i in 0..instance.num_cities() {
            for j in i+1..instance.num_cities() {
                if !optimal_tour.contains_edge(i, j) {
                    let features = vec![
                        instance.distances[i][j],
                        instance.nearest_neighbor_rank(i, j) as f64,
                        instance.cluster_distance(i, j),
                        instance.angle_deviation(i, j),
                    ];
                    edge_features.push((features, 0.0)); // 0.0 = not in optimal
                }
            }
        }
    }

    // Train simple linear model for edge scoring
    let weights = train_edge_scorer(&edge_features);

    LearnedHeuristics {
        edge_weights: weights,
        feature_names: vec![
            "distance".into(),
            "nn_rank".into(),
            "cluster_dist".into(),
            "angle_dev".into(),
        ],
    }
}
```

---

## 7. Solution Quality Metrics

### 7.1 Gap from Optimal

**Reference:** Johnson & McGeoch (1997) [5]

```rust
/// Calculate gap from optimal (or best-known)
pub fn optimality_gap(solution_length: f64, optimal_length: f64) -> f64 {
    ((solution_length - optimal_length) / optimal_length) * 100.0
}

/// Quality tier classification
pub fn solution_tier(gap: f64) -> SolutionTier {
    match gap {
        g if g < 0.1 => SolutionTier::Optimal,      // Within 0.1%
        g if g < 1.0 => SolutionTier::Excellent,    // Within 1%
        g if g < 2.0 => SolutionTier::Good,         // Within 2%
        g if g < 5.0 => SolutionTier::Acceptable,   // Within 5%
        _ => SolutionTier::Poor,
    }
}
```

### 7.2 Benchmark Results Format

```bash
$ aprender-tsp benchmark delivery.apr --instances tsplib/*.tsp

Benchmark Results
=================
Model: delivery.apr (ACO, trained on 10 instances)

Instance         Size    Optimal   Found      Gap     Time     Tier
─────────────────────────────────────────────────────────────────────
berlin52           52      7,542    7,544    0.03%   0.23s   Optimal
eil51              51        426      428    0.47%   0.19s   Excellent
eil76              76        538      542    0.74%   0.31s   Excellent
kroA100           100     21,282   21,389    0.50%   0.45s   Excellent
pr152             152     73,682   74,291    0.83%   0.89s   Excellent
d198              198     15,780   15,982    1.28%   1.12s   Good
lin318            318     42,029   43,105    2.56%   2.34s   Good
─────────────────────────────────────────────────────────────────────
Average Gap: 0.92%
Median Gap:  0.74%
Worst Gap:   2.56%
```

---

## 8. Implementation Architecture

### 8.1 Crate Structure

```
crates/aprender-tsp/
├── Cargo.toml
├── src/
│   ├── lib.rs           # Library entry point
│   ├── main.rs          # CLI entry point
│   ├── model.rs         # TspModel, .apr serialization
│   ├── solver/
│   │   ├── mod.rs       # TspSolver trait
│   │   ├── aco.rs       # Ant Colony Optimization
│   │   ├── tabu.rs      # Tabu Search
│   │   ├── ga.rs        # Genetic Algorithm
│   │   └── hybrid.rs    # Hybrid solver
│   ├── instance/
│   │   ├── mod.rs       # TspInstance struct
│   │   ├── tsplib.rs    # TSPLIB parser
│   │   ├── csv.rs       # CSV parser
│   │   └── geojson.rs   # GeoJSON parser
│   ├── training.rs      # Parameter tuning pipeline
│   ├── benchmark.rs     # Quality evaluation
│   ├── export.rs        # Solution exporters
│   └── error.rs         # TspError enum
└── tests/
    ├── integration.rs   # CLI integration tests
    ├── solvers.rs       # Solver correctness tests
    └── fixtures/        # Test instances (TSPLIB)
```

### 8.2 Core Traits

```rust
/// Trait for TSP solvers
pub trait TspSolver: Send + Sync {
    /// Solve a TSP instance
    fn solve(
        &mut self,
        distances: &[Vec<f64>],
        budget: Budget,
    ) -> TspSolution;

    /// Get algorithm name
    fn name(&self) -> &str;

    /// Get current best solution (for progress tracking)
    fn best(&self) -> Option<&TspSolution>;

    /// Reset solver state
    fn reset(&mut self);
}

/// TSP solution
#[derive(Debug, Clone)]
pub struct TspSolution {
    /// Tour as city indices (starting and ending at 0)
    pub tour: Vec<usize>,
    /// Total tour length
    pub length: f64,
    /// Number of evaluations used
    pub evaluations: usize,
    /// Convergence history
    pub history: Vec<f64>,
}
```

### 8.3 Error Handling

**Toyota Way Principle:** *Jidoka* - Stop immediately on errors, provide clear diagnostics.

```rust
/// TSP-specific errors with actionable hints
#[derive(Debug)]
pub enum TspError {
    /// Invalid .apr file format
    InvalidFormat {
        message: String,
        hint: String,
    },
    /// Checksum verification failed
    ChecksumMismatch {
        expected: u32,
        computed: u32,
    },
    /// Instance parsing failed
    ParseError {
        file: PathBuf,
        line: Option<usize>,
        cause: String,
    },
    /// Invalid instance data
    InvalidInstance {
        message: String,
    },
    /// Solver failed to find solution
    SolverFailed {
        algorithm: String,
        reason: String,
    },
    /// Budget exhausted without convergence
    BudgetExhausted {
        evaluations: usize,
        best_found: f64,
    },
}

impl std::fmt::Display for TspError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidFormat { message, hint } => {
                write!(f, "Invalid .apr format: {}\nHint: {}", message, hint)
            }
            Self::ChecksumMismatch { expected, computed } => {
                write!(
                    f,
                    "Model file corrupted: checksum mismatch\n\
                     Expected: 0x{:08X}, Computed: 0x{:08X}\n\
                     Hint: Re-train the model or restore from backup",
                    expected, computed
                )
            }
            Self::ParseError { file, line, cause } => {
                if let Some(line_num) = line {
                    write!(f, "Parse error in {} at line {}: {}", file.display(), line_num, cause)
                } else {
                    write!(f, "Parse error in {}: {}", file.display(), cause)
                }
            }
            Self::InvalidInstance { message } => {
                write!(f, "Invalid TSP instance: {}", message)
            }
            Self::SolverFailed { algorithm, reason } => {
                write!(f, "{} solver failed: {}", algorithm, reason)
            }
            Self::BudgetExhausted { evaluations, best_found } => {
                write!(
                    f,
                    "Budget exhausted after {} evaluations\n\
                     Best solution found: {:.2}\n\
                     Hint: Increase --iterations or --timeout",
                    evaluations, best_found
                )
            }
        }
    }
}
```

---

## 9. Quality Standards

### 9.1 Testing Requirements

| Category | Target | Methodology |
|----------|--------|-------------|
| Unit Test Coverage | ≥95% | `cargo llvm-cov` |
| Property Tests | 50+ | `proptest` |
| Integration Tests | 20+ | CLI e2e tests |
| Benchmark Instances | 30+ | TSPLIB standard |
| Mutation Score | ≥80% | `cargo mutants` |

### 9.2 Performance Targets

**Reference:** Applegate et al. (2006) [6]

| Instance Size | P50 Latency | P99 Latency | Gap Target |
|---------------|-------------|-------------|------------|
| n ≤ 50 | <500ms | <2s | <1% |
| 50 < n ≤ 200 | <5s | <30s | <2% |
| 200 < n ≤ 500 | <60s | <5min | <3% |
| n > 500 | <5min | <30min | <5% |

### 9.3 TSPLIB Benchmark Suite

**Reference:** Reinelt (1991) [4]

Required passing benchmarks:

```rust
#[test]
fn test_berlin52_within_1_percent() {
    let instance = TspInstance::load("fixtures/berlin52.tsp").unwrap();
    let mut solver = AcoSolver::default();
    let solution = solver.solve(&instance.distances, Budget::Iterations(1000));

    let optimal = 7542.0;
    let gap = (solution.length - optimal) / optimal * 100.0;
    assert!(gap < 1.0, "Gap {:.2}% exceeds 1% target", gap);
}

#[test]
fn test_eil51_within_1_percent() { /* ... */ }

#[test]
fn test_kroA100_within_2_percent() { /* ... */ }

#[test]
fn test_pr152_within_2_percent() { /* ... */ }
```

---

## 10. Academic References

### 10.1 Foundational Algorithms

1. **Dorigo, M., & Gambardella, L. M. (1997).** "Ant Colony System: A Cooperative Learning Approach to the Traveling Salesman Problem." *IEEE Transactions on Evolutionary Computation*, 1(1), 53-66. DOI: 10.1109/4235.585892
   - *Relevance:* Foundation for ACO implementation; defines α, β, ρ parameters and ACS transition rule.

2. **Glover, F., & Laguna, M. (1997).** *Tabu Search*. Kluwer Academic Publishers. ISBN: 978-0-7923-8187-2
   - *Relevance:* Canonical reference for Tabu Search; defines tenure, aspiration criteria, diversification.

3. **Goldberg, D. E. (1989).** *Genetic Algorithms in Search, Optimization, and Machine Learning*. Addison-Wesley. ISBN: 978-0-201-15767-3
   - *Relevance:* GA fundamentals; basis for order crossover and tournament selection.

### 10.2 TSP-Specific Research

4. **Reinelt, G. (1991).** "TSPLIB—A Traveling Salesman Problem Library." *ORSA Journal on Computing*, 3(4), 376-384. DOI: 10.1287/ijoc.3.4.376
   - *Relevance:* Standard benchmark instances; file format specification for TSPLIB parser.

5. **Johnson, D. S., & McGeoch, L. A. (1997).** "The Traveling Salesman Problem: A Case Study in Local Optimization." *Local Search in Combinatorial Optimization*, 215-310. Wiley.
   - *Relevance:* Comprehensive survey of TSP heuristics; defines quality gap metrics.

6. **Applegate, D. L., Bixby, R. E., Chvátal, V., & Cook, W. J. (2006).** *The Traveling Salesman Problem: A Computational Study*. Princeton University Press. ISBN: 978-0-691-12993-8
   - *Relevance:* State-of-the-art analysis; performance baselines for large instances.

### 10.3 Hybrid and Adaptive Methods

7. **Stützle, T., & Hoos, H. H. (2000).** "MAX–MIN Ant System." *Future Generation Computer Systems*, 16(8), 889-914. DOI: 10.1016/S0167-739X(00)00043-1
   - *Relevance:* Improved ACO with pheromone bounds; reduces premature convergence.

8. **Burke, E. K., Gendreau, M., Hyde, M., Kendall, G., Ochoa, G., Özcan, E., & Qu, R. (2013).** "Hyper-heuristics: A Survey of the State of the Art." *Journal of the Operational Research Society*, 64(12), 1695-1724. DOI: 10.1057/jors.2013.71
   - *Relevance:* Framework for hybrid solver design; algorithm selection strategies.

### 10.4 Machine Learning for TSP

9. **Vinyals, O., Fortunato, M., & Jaitly, N. (2015).** "Pointer Networks." *Advances in Neural Information Processing Systems (NeurIPS)*, 28, 2692-2700.
   - *Relevance:* Neural approach to TSP; informs learned heuristics design.

10. **Kool, W., van Hoof, H., & Welling, M. (2019).** "Attention, Learn to Solve Routing Problems!" *International Conference on Learning Representations (ICLR)*.
    - *Relevance:* State-of-the-art neural TSP solver; attention mechanism for edge scoring.

---

## Appendix A: Toyota Way Principles Applied

| Principle | Application in aprender-tsp |
|-----------|----------------------------|
| **Genchi Genbutsu** (Go and See) | Users train on their own problem instances |
| **Kaizen** (Continuous Improvement) | Incremental model updates with new data |
| **Jidoka** (Build Quality In) | Checksum validation, graceful error handling |
| **Heijunka** (Level Loading) | Budget-based resource management |
| **Standardized Work** | Consistent .apr format across tools |
| **Respect for People** | Local-first, user-owned models |
| **Muda Elimination** | Efficient algorithms, no cloud overhead |
| **Poka-yoke** (Error-Proofing) | Input validation, format verification |

---

## Appendix B: Comparison with Existing TSP Solvers

| Feature | OR-Tools | Concorde | aprender-tsp |
|---------|----------|----------|--------------|
| **License** | Apache 2.0 | Academic | MIT |
| **Language** | C++/Python | C | Rust |
| **Model Persistence** | No | No | Yes (.apr) |
| **Offline** | Yes | Yes | Yes |
| **Custom Training** | Limited | No | Yes |
| **Cloud Required** | No | No | No |
| **Metaheuristics** | Limited | No (exact) | Full suite |
| **Instance Learning** | No | No | Yes |

---

## Changelog

- **v1.0 (2025-11-29):** Initial specification with ACO, Tabu, GA solvers and .apr model format.
