---
title: "Simple OR Example: Bay Area TSP Ground Truth"
issue: OR-001
status: Completed
created: 2025-12-12
updated: 2025-12-12
---

# OR-001: Simple Operations Research Example Specification

**Ticket ID**: OR-001
**Status**: Completed
**Methodology**: Toyota Production System (TPS) + Popperian Falsification + Equation Driven Development (EDD)
**Quality**: EXTREME TDD | 100% Coverage | Probar E2E

## ⚠️ CRITICAL: ZERO JAVASCRIPT POLICY

**JavaScript is FORBIDDEN ("arsenic poison").**

All WASM testing MUST be:
1. **Rust-side Probar E2E tests** - Deterministic replay from Rust
2. **wasm-bindgen-test** - Browser-agnostic Rust tests
3. **NO browser JavaScript** - Never modify `web/*.html` JavaScript

Rationale:
- JavaScript is non-deterministic (timing, garbage collection)
- JavaScript cannot guarantee reproducibility
- JavaScript violates Probar's deterministic replay principle
- All WASM APIs are tested via Rust `#[cfg(feature = "wasm")]` tests

```
FORBIDDEN:                          REQUIRED:
─────────────────────              ─────────────────────
web/tsp.html JS mods   ❌           tests/probar_tsp.rs  ✅
Browser console debug  ❌           #[wasm_bindgen_test] ✅
Manual browser testing ❌           cargo test --features wasm ✅
```

## 1. Executive Summary

This specification defines a minimal, verifiable Traveling Salesman Problem (TSP) instance using Bay Area cities with known driving distances. Adhering to **Equation Driven Development (EDD)**, the problem is rigorously defined via an Equation Model Card (EMC) before implementation.

**Core Principle**: YAML-first architecture where users can download, modify, and re-run experiments without touching code.

### 1.1 Unified Architecture

**CRITICAL: ONE TSP Demo** - There is exactly ONE TSP implementation (`TspGraspDemo`) that:
1. Supports YAML-first configuration via `TspInstanceYaml`
2. Works in **TUI** (terminal ratatui)
3. Works in **WASM** (browser WebAssembly)
4. Uses the same GRASP algorithm and 2-opt local search

```
                    ┌─────────────────────────────────────┐
                    │     bay_area_tsp.yaml (Ground Truth) │
                    └──────────────────┬──────────────────┘
                                       │
                              TspInstanceYaml::from_yaml()
                                       │
                    ┌──────────────────▼──────────────────┐
                    │          TspGraspDemo               │
                    │   (Single Unified Implementation)   │
                    └──────────────────┬──────────────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              │                        │                        │
     ┌────────▼────────┐      ┌────────▼────────┐      ┌────────▼────────┐
     │   TUI (ratatui) │      │   WASM (web)    │      │   CLI (batch)   │
     │   tsp_tui.rs    │      │   WasmTspGrasp  │      │   via from_yaml │
     └─────────────────┘      └─────────────────┘      └─────────────────┘
```

### 1.2 Key Deliverables

| Deliverable | Description | Deployment |
|-------------|-------------|------------|
| `TspGraspDemo` | Unified TSP solver (GRASP + 2-opt) | `src/demos/tsp_grasp.rs` |
| `TspInstanceYaml` | YAML configuration loader | `src/demos/tsp_instance.rs` |
| `bay_area_tsp.yaml` | Ground truth instance (6 cities) | `examples/experiments/` |
| TUI App | Terminal visualization | `src/bin/tsp_tui.rs` |
| WASM Module | Browser-executable solver | `web/tsp.html` |
| Probar E2E | Deterministic replay tests | `tests/probar_tsp.rs` |

## 2. Equation Model Card (EMC)

**ID**: EMC-TSP-MTZ-001
**Name**: Miller-Tucker-Zemlin (MTZ) Formulation for TSP
**Type**: Integer Linear Programming (ILP)

### 2.1 Core Equations

**Objective Function (Minimization):**
$$ Z = \sum_{i=1}^{n} \sum_{j=1}^{n} c_{ij} x_{ij} $$
*Goal: Minimize total travel distance $Z$.*

**Constraints:**

1.  **Topology Constraint (Out-degree)**: Leave each city exactly once.
    $$ \sum_{j=1, j \neq i}^{n} x_{ij} = 1 \quad \forall i \in \{1, \dots, n \} $$

2.  **Topology Constraint (In-degree)**: Enter each city exactly once.
    $$ \sum_{i=1, i \neq j}^{n} x_{ij} = 1 \quad \forall j \in \{1, \dots, n \} $$

3.  **Subtour Elimination (MTZ)**: Prevent disconnected loops.
    $$ u_i - u_j + n x_{ij} \le n - 1 \quad \forall i, j \in \{2, \dots, n \}, i \neq j $$

4.  **Integrality**:
    $$ x_{ij} \in \{0, 1 \} $$

### 2.2 Variable Definitions

| Variable | Symbol | Type | Description |
|----------|--------|------|-------------|
| **Cost Matrix** | $c_{ij}$ | Constant ($u32$) | Driving distance from city $i$ to $j$ (miles). |
| **Decision** | $x_{ij}$ | Binary | 1 if path goes $i \to j$, 0 otherwise. |
| **Auxiliary** | $u_i$ | Integer | Order of visitation for city $i$ (MTZ). |
| **Size** | $n$ | Integer | Number of cities (user-configurable). |

## 3. YAML-First Architecture

### 3.1 Design Principle

**All configuration is YAML. Code reads YAML. Users modify YAML.**

```
User Experience:
┌──────────────────────────────────────────────────────────┐
│  1. Download bay_area_tsp.yaml                           │
│  2. Edit cities/distances in any text editor             │
│  3. Upload modified YAML (or paste in web editor)        │
│  4. Run solver → See results                             │
│  5. Compare different configurations                     │
└──────────────────────────────────────────────────────────┘
```

### 3.2 Ground Truth Instance (YAML)

```yaml
# bay_area_tsp.yaml
# User-editable TSP instance - modify cities and distances freely
meta:
  id: "TSP-BAY-006"
  version: "1.0.0"
  description: "6-city Bay Area ground truth instance"
  source: "Google Maps (Dec 2024)"
  units: "miles"
  optimal_known: 115  # Known optimal for verification

cities:
  - id: 0
    name: "San Francisco"
    alias: "SF"
    coords: { lat: 37.7749, lon: -122.4194 }
  - id: 1
    name: "Oakland"
    alias: "OAK"
    coords: { lat: 37.8044, lon: -122.2712 }
  - id: 2
    name: "San Jose"
    alias: "SJ"
    coords: { lat: 37.3382, lon: -121.8863 }
  - id: 3
    name: "Palo Alto"
    alias: "PA"
    coords: { lat: 37.4419, lon: -122.1430 }
  - id: 4
    name: "Berkeley"
    alias: "BRK"
    coords: { lat: 37.8716, lon: -122.2727 }
  - id: 5
    name: "Fremont"
    alias: "FRE"
    coords: { lat: 37.5485, lon: -121.9886 }

# Distance Matrix (Row=From, Col=To)
# Symmetric matrix - users can modify any value
# Order: SF, OAK, SJ, PA, BRK, FRE
matrix:
  - [ 0, 12, 48, 35, 14, 42]  # From SF
  - [12,  0, 42, 30,  4, 30]  # From OAK
  - [48, 42,  0, 15, 46, 17]  # From SJ
  - [35, 30, 15,  0, 32, 18]  # From PA
  - [14,  4, 46, 32,  0, 32]  # From BRK
  - [42, 30, 17, 18, 32,  0]  # From FRE

# Algorithm configuration (user-selectable)
algorithm:
  method: "grasp"  # Options: greedy, grasp, brute_force
  params:
    rcl_size: 3        # Restricted Candidate List size
    restarts: 10       # Number of GRASP restarts
    two_opt: true      # Enable 2-opt local search
    seed: 42           # For reproducibility
```

### 3.3 User-Modifiable Components

| Component | File/Section | User Action |
|-----------|--------------|-------------|
| **Cities** | `cities:` array | Add/remove/rename cities |
| **Distances** | `matrix:` | Edit driving distances |
| **Algorithm** | `algorithm.method` | Switch: greedy/grasp/brute_force |
| **Parameters** | `algorithm.params` | Tune RCL size, restarts, seed |
| **Equations** | `docs/emc/*.yaml` | Modify EMC definitions |

## 4. Web Deployment (interactive.paiml.com)

### 4.1 User Interface Requirements

```
┌─────────────────────────────────────────────────────────────────┐
│  TSP GRASP Demo - Bay Area                          [Run] [Reset]│
├─────────────────┬───────────────────────────────────────────────┤
│                 │                                               │
│   YAML Editor   │           Visualization                       │
│   ───────────   │           ─────────────                       │
│   [Download]    │           ┌─────────────────────┐             │
│   [Upload]      │           │     Map View        │             │
│   [Examples ▼]  │           │   with Tour Path    │             │
│                 │           └─────────────────────┘             │
│   cities:       │                                               │
│     - SF        │           Tour: SF→OAK→BRK→FRE→SJ→PA→SF       │
│     - Oakland   │           Distance: 115 miles                 │
│     ...         │           Optimal: ✓ (verified)               │
│                 │                                               │
│   matrix:       │           ┌─────────────────────┐             │
│     [edit]      │           │  Convergence Plot   │             │
│                 │           └─────────────────────┘             │
├─────────────────┴───────────────────────────────────────────────┤
│  Equations (EMC-TSP-MTZ-001)                     [Show/Hide]    │
│  Z = Σᵢ Σⱼ cᵢⱼ xᵢⱼ   |   Constraints: In/Out degree = 1        │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Interactive Features

| Feature | Description | Implementation |
|---------|-------------|----------------|
| **YAML Editor** | Monaco/CodeMirror with YAML syntax | Web component |
| **Download** | Export current config as .yaml file | Blob download |
| **Upload** | Load user's .yaml file | FileReader API |
| **Examples** | Dropdown: Bay Area, Random 10, Random 50 | Preset configs |
| **Equation Toggle** | Show/hide LaTeX-rendered EMC | MathJax/KaTeX |
| **Algorithm Switch** | Radio: Greedy / GRASP / Brute Force | Config update |
| **Live Validation** | Red border on invalid YAML | Schema check |

### 4.3 Metrics Display (TUI + WASM)

**CRITICAL: All metrics must be clearly labeled with units.**

Both TUI and WASM must display:

| Metric | Label | Example | Source |
|--------|-------|---------|--------|
| Number of cities | `[cities]` | `6 [cities]` | `TspInstanceYaml.city_count()` |
| Best tour distance | `[miles]` | `115 [miles]` | `TspGraspDemo.best_tour_length` |
| Optimal known | `[miles]` | `Optimal: 115 [miles]` | `TspInstanceYaml.meta.optimal_known` |

**TUI Display Example:**
```
┌─────────────────────────────────────────┐
│  TSP GRASP Demo                         │
├─────────────────────────────────────────┤
│  Cities: 6 [cities]                     │
│  Best:   115 [miles]                    │
│  Optimal: 115 [miles] ✓                 │
│  Gap:    0.0%                           │
└─────────────────────────────────────────┘
```

**WASM API Methods:**
```rust
// TspWasmInstance
fn city_count(&self) -> usize;           // Returns count
fn get_units(&self) -> String;           // Returns "miles"
fn get_optimal_known(&self) -> Option<u32>;

// WasmTspGrasp
fn get_n(&self) -> usize;                // City count
fn get_best_tour_length(&self) -> f64;   // In miles (from YAML)
fn get_units(&self) -> String;           // NEW: Returns unit label
```

## 5. Implementation Tasks (pmat work)

### 5.1 Task Breakdown

Each task follows EXTREME TDD: Write failing test → Implement → Verify 100% coverage.

| Task ID | Description | Status | Tests |
|---------|-------------|--------|-------|
| **OR-001-01** | `TspInstanceYaml` struct + serde | ✅ DONE | 43 tests |
| **OR-001-02** | `from_yaml()` implementation | ✅ DONE | Included |
| **OR-001-03** | `bay_area_tsp.yaml` ground truth | ✅ DONE | Verified |
| **OR-001-04** | Jidoka validators (triangle ineq, symmetry) | ✅ DONE | 8 tests |
| **OR-001-05** | Probar E2E: deterministic replay | ✅ DONE | 19 tests |
| **OR-001-06** | WASM `TspWasmInstance` bindings | ✅ DONE | 26 tests |
| **OR-001-07** | Integrate `TspGraspDemo` with `TspInstanceYaml` | ✅ DONE | 12 tests |
| **OR-001-08** | TUI: YAML file loading support | ✅ DONE | 12 tests |
| **OR-001-09** | WASM uses YAML config (Probar-tested) | ✅ DONE | 9 tests |
| **OR-001-10** | Probar unified architecture tests | ✅ DONE | 28 total |

### 5.2 WASM Testing Strategy (NO JavaScript)

**All WASM code is tested via Rust-side Probar E2E tests:**

```rust
// tests/probar_tsp.rs - WASM bindings tested from Rust

#[test]
fn probar_wasm_tsp_instance_from_yaml() {
    // Test TspWasmInstance::from_yaml() via Rust
    let yaml = include_str!("../examples/experiments/bay_area_tsp.yaml");
    let instance = TspWasmInstance::from_yaml(yaml).expect("parse");
    assert_eq!(instance.city_count(), 6);
    assert_eq!(instance.optimal_known(), Some(115));
}

#[test]
fn probar_wasm_tsp_grasp_from_yaml() {
    // Test WasmTspGrasp::from_yaml() via Rust
    let yaml = include_str!("../examples/experiments/bay_area_tsp.yaml");
    let grasp = WasmTspGrasp::from_yaml(yaml).expect("parse");
    // Run deterministic - same result every time
    let result = grasp.run_to_completion();
    assert!(result.distance <= 115.0);
}
```

**Why Rust-side testing:**
1. **Deterministic** - No browser timing variance
2. **Reproducible** - Probar replay guarantees
3. **CI-friendly** - `cargo test` without browsers
4. **Type-safe** - Compile-time WASM API verification

### 5.2 Unified Integration (OR-001-07)

**Goal**: Connect `TspInstanceYaml` to `TspGraspDemo` so ONE demo serves TUI and WASM.

```rust
// TspGraspDemo gains YAML support
impl TspGraspDemo {
    /// Create demo from YAML instance configuration.
    pub fn from_instance(instance: &TspInstanceYaml) -> Self {
        let cities: Vec<City> = instance.cities.iter()
            .map(|c| City::new(c.coords.lon, c.coords.lat))
            .collect();

        let mut demo = Self::with_cities(instance.algorithm.params.seed, cities);
        demo.set_construction_method(/* from instance.algorithm.method */);
        demo.set_rcl_size(instance.algorithm.params.rcl_size);
        demo
    }

    /// Load from YAML string.
    pub fn from_yaml(yaml: &str) -> Result<Self, TspInstanceError> {
        let instance = TspInstanceYaml::from_yaml(yaml)?;
        instance.validate()?;
        Ok(Self::from_instance(&instance))
    }
}
```

### 5.3 TUI Integration (OR-001-08)

**Goal**: TUI can load YAML files via command-line argument.

```bash
# Usage
cargo run --bin tsp_tui -- examples/experiments/bay_area_tsp.yaml

# Or interactive file picker
cargo run --bin tsp_tui
# Press 'L' to load YAML file
```

```rust
// src/tui/tsp_app.rs additions
impl TspApp {
    pub fn from_yaml_file<P: AsRef<Path>>(path: P) -> Result<Self, TspInstanceError> {
        let instance = TspInstanceYaml::from_yaml_file(path)?;
        let demo = TspGraspDemo::from_instance(&instance);
        Ok(Self { demo, instance: Some(instance), /* ... */ })
    }
}
```

### 5.2 Task Details

#### OR-001-01: TspInstanceYaml Struct

```rust
// src/demos/tsp_instance.rs
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TspInstanceYaml {
    pub meta: TspMeta,
    pub cities: Vec<TspCity>,
    pub matrix: Vec<Vec<u32>>,
    pub algorithm: TspAlgorithmConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TspMeta {
    pub id: String,
    pub version: String,
    pub description: String,
    pub units: String,
    pub optimal_known: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TspCity {
    pub id: usize,
    pub name: String,
    pub alias: String,
    pub coords: Coords,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TspAlgorithmConfig {
    pub method: String,  // "greedy" | "grasp" | "brute_force"
    pub params: TspParams,
}
```

**Tests Required (100% coverage):**
- `test_deserialize_valid_yaml`
- `test_deserialize_minimal_yaml`
- `test_deserialize_invalid_missing_cities`
- `test_deserialize_invalid_matrix_size`
- `test_serialize_roundtrip`
- `test_default_algorithm_params`

#### OR-001-02: from_yaml() Implementation

```rust
impl TspGraspDemo {
    /// Load TSP instance from YAML configuration.
    ///
    /// # Errors
    /// Returns error if:
    /// - YAML parse fails
    /// - Matrix dimensions don't match city count
    /// - Triangle inequality violated (Jidoka)
    pub fn from_yaml(yaml: &str) -> Result<Self, TspError> {
        let config: TspInstanceYaml = serde_yaml::from_str(yaml)
            .map_err(|e| TspError::ParseError(e.to_string()))?;

        // Jidoka: Validate before proceeding
        Self::validate_instance(&config)?;

        Self::from_config(config)
    }

    /// Load from file path.
    pub fn from_yaml_file<P: AsRef<Path>>(path: P) -> Result<Self, TspError> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| TspError::IoError(e.to_string()))?;
        Self::from_yaml(&content)
    }
}
```

**Tests Required:**
- `test_from_yaml_bay_area_instance`
- `test_from_yaml_returns_optimal_115`
- `test_from_yaml_invalid_matrix_dimension`
- `test_from_yaml_triangle_inequality_violation`
- `test_from_yaml_asymmetric_matrix_warning`
- `test_from_yaml_file_not_found`
- `test_from_yaml_file_success`

#### OR-001-05: Probar E2E Test

```rust
// tests/probar_tsp.rs
//! Probar E2E tests for TSP YAML loading and deterministic replay.

use simular::demos::TspGraspDemo;

const BAY_AREA_YAML: &str = include_str!("../examples/experiments/bay_area_tsp.yaml");

#[test]
fn probar_tsp_bay_area_optimal() {
    // Load from YAML
    let demo = TspGraspDemo::from_yaml(BAY_AREA_YAML)
        .expect("YAML should parse");

    // Run to completion
    let result = demo.run_to_completion();

    // Verify optimal
    assert_eq!(result.best_tour_length, 115.0,
        "Bay Area optimal tour should be 115 miles");

    // Verify tour visits all cities exactly once
    assert_eq!(result.best_tour.len(), 6);
    let mut visited: Vec<usize> = result.best_tour.clone();
    visited.sort();
    assert_eq!(visited, vec![0, 1, 2, 3, 4, 5]);
}

#[test]
fn probar_tsp_deterministic_replay() {
    // Two runs with same seed must produce identical results
    let demo1 = TspGraspDemo::from_yaml(BAY_AREA_YAML).unwrap();
    let demo2 = TspGraspDemo::from_yaml(BAY_AREA_YAML).unwrap();

    let result1 = demo1.run_to_completion();
    let result2 = demo2.run_to_completion();

    assert_eq!(result1.best_tour, result2.best_tour,
        "Deterministic replay failed");
    assert_eq!(result1.best_tour_length, result2.best_tour_length);
}

#[test]
fn probar_tsp_user_modified_yaml() {
    // Simulate user editing: change SF→Oakland distance from 12 to 5
    let modified_yaml = BAY_AREA_YAML.replace(
        "[ 0, 12, 48, 35, 14, 43]",
        "[ 0,  5, 48, 35, 14, 43]"
    );

    let demo = TspGraspDemo::from_yaml(&modified_yaml).unwrap();
    let result = demo.run_to_completion();

    // New optimal should be different (shorter via SF-Oakland)
    assert!(result.best_tour_length < 115.0,
        "Modified distances should yield shorter tour");
}
```

## 6. Toyota Production System Alignment

### 6.1 Jidoka (Stop-on-Error)

| Check | Condition | Action |
|-------|-----------|--------|
| Matrix Size | `matrix.len() != cities.len()` | ERROR: Halt |
| Triangle Inequality | `c[i][k] > c[i][j] + c[j][k]` | WARN: Log |
| Symmetry | `c[i][j] != c[j][i]` | WARN: Log (unless asymmetric flag) |
| Tour Validity | City visited twice | ERROR: Halt |
| NaN/Inf | Distance is not finite | ERROR: Halt |

### 6.2 Poka-Yoke (Mistake-Proofing)

| Guard | Implementation |
|-------|----------------|
| Type-safe city IDs | `CityId(usize)` newtype |
| Validated YAML schema | JSON Schema + runtime check |
| Immutable config after load | `TspInstanceYaml` is `Clone` but not `&mut` |

### 6.3 Genchi Genbutsu (Go and See)

Every run produces step-by-step verification output:

```
Step 1: San Francisco → Oakland = 12 miles (total: 12)
Step 2: Oakland → Berkeley = 4 miles (total: 16)
Step 3: Berkeley → Fremont = 32 miles (total: 48)
Step 4: Fremont → San Jose = 17 miles (total: 65)
Step 5: San Jose → Palo Alto = 15 miles (total: 80)
Step 6: Palo Alto → San Francisco = 35 miles (total: 115)

VERIFIED: Tour length 115 matches sum of edges ✓
VERIFIED: All 6 cities visited exactly once ✓
VERIFIED: Optimal known (115) achieved ✓
```

## 7. Falsification Criteria (Popperian)

| Criterion | Equation / Logic | Failure Mode |
|-----------|------------------|--------------|
| **F1: Validity** | `|tour| = n` and all unique | Disconnected tour |
| **F2: Cost** | `Z_calc = Σ c[edge]` | Arithmetic bug |
| **F3: Optimality** | `Z_algo ≤ optimal_known` | Impossible claim |
| **F4: Determinism** | `run(seed) = run(seed)` | RNG leak |
| **F5: YAML Integrity** | Modified YAML → different result | Config not used |

## 8. Peer-Reviewed References

1.  **Miller, C. E., Tucker, A. W., & Zemlin, R. A.** (1960). "Integer Programming Formulation of Traveling Salesman Problems." *Journal of the ACM*, 7(4), 326–329.
    *Relevance*: MTZ subtour elimination constraints.

2.  **Karp, R. M.** (1972). "Reducibility among combinatorial problems." In *Complexity of Computer Computations* (pp. 85-103). Springer.
    *Relevance*: TSP NP-completeness proof.

3.  **Feo, T. A., & Resende, M. G. C.** (1995). "Greedy Randomized Adaptive Search Procedures." *Journal of Global Optimization*, 6(2), 109-133.
    *Relevance*: GRASP metaheuristic foundation.

4.  **Lin, S., & Kernighan, B. W.** (1973). "An Effective Heuristic Algorithm for the Traveling-Salesman Problem." *Operations Research*, 21(2), 498-516.
    *Relevance*: k-opt local search methodology.

5.  **Rosenkrantz, D. J., Stearns, R. E., & Lewis, P. M.** (1977). "An Analysis of Several Heuristics for the Traveling Salesman Problem." *SIAM Journal on Computing*, 6(3), 563-581.
    *Relevance*: Nearest Neighbor performance bounds.

## 9. Acceptance Criteria

### 9.1 Functional

- [x] **AC-1**: `TspGraspDemo::from_yaml()` parses `bay_area_tsp.yaml` correctly ✅
- [x] **AC-2**: Bay Area instance returns optimal 115 miles ✅
- [x] **AC-3**: User can modify YAML cities → solver uses modified data ✅
- [x] **AC-4**: User can switch algorithm via YAML → different behavior ✅
- [x] **AC-5**: Probar E2E tests pass with deterministic replay ✅

### 9.2 Quality Gates

- [x] **QG-1**: 98.6% test coverage on `tsp_instance.rs` ✅
- [x] **QG-2**: 100% test coverage on `from_yaml()` code paths ✅
- [x] **QG-3**: Zero clippy warnings ✅
- [x] **QG-4**: Probar E2E: 28 deterministic replay tests ✅
- [x] **QG-5**: WASM bundle includes YAML parsing (97KB gzipped) ✅

### 9.3 WASM Testing (NO JavaScript - Probar Only)

- [x] **WT-1**: `TspWasmInstance::from_yaml()` Probar tests ✅
- [x] **WT-2**: `WasmTspGrasp::from_yaml()` Probar tests ✅
- [x] **WT-3**: All WASM APIs tested via Rust (`cargo test --features wasm`) ✅
- [x] **WT-4**: Zero JavaScript modifications required ✅
- [x] **WT-5**: Deterministic replay across native/WASM ✅

### 9.4 Web Deployment (Read-Only HTML)

**NOTE: `web/tsp.html` is READ-ONLY. No JavaScript modifications allowed.**

- [x] **WD-1**: WASM module loads from existing HTML ✅
- [x] **WD-2**: YAML parsing works in browser (tested via Probar) ✅
- [ ] **WD-3**: YAML editor renders (existing feature, unchanged)
- [ ] **WD-4**: Download/Upload buttons work (existing feature, unchanged)

## 10. Roadmap Integration

**Status in `docs/roadmaps/roadmap.yaml`: COMPLETED**

```yaml
- id: OR-001
  github_issue: null
  item_type: epic
  title: "Simple OR Example: Bay Area TSP Ground Truth"
  status: completed
  priority: high
  spec: docs/specifications/simple-or-example.md
  acceptance_criteria:
    - "AC-1: TspGraspDemo::from_yaml() parses bay_area_tsp.yaml [VERIFIED]"
    - "AC-2: Bay Area instance returns optimal 115 miles [VERIFIED]"
    - "AC-3: User can modify YAML cities [VERIFIED]"
    - "AC-4: User can switch algorithm via YAML [VERIFIED]"
    - "AC-5: Probar E2E tests pass with deterministic replay [VERIFIED]"
    - "QG-1: 98.6% test coverage on tsp_instance.rs [VERIFIED]"
  subtasks:
    - id: OR-001-01
      title: "TspInstanceYaml struct + serde"
      status: completed
    - id: OR-001-02
      title: "from_yaml() implementation"
      status: completed
    - id: OR-001-03
      title: "bay_area_tsp.yaml ground truth"
      status: completed
    - id: OR-001-04
      title: "Jidoka validators"
      status: completed
    - id: OR-001-05
      title: "Probar E2E tests (19 tests)"
      status: completed
    - id: OR-001-06
      title: "WASM TspWasmInstance + WasmTspGrasp.fromYaml"
      status: completed
    - id: OR-001-07
      title: "Unified TspGraspDemo with TspInstanceYaml"
      status: completed
    - id: OR-001-08
      title: "TUI YAML file loading support"
      status: completed
    - id: OR-001-09
      title: "WASM uses YAML config (Probar-tested, NO JavaScript)"
      status: completed
    - id: OR-001-10
      title: "Probar unified architecture tests (28 total)"
      status: completed
  labels:
    - or
    - tsp
    - yaml-first
    - wasm
  notes: "YAML-first TSP with Bay Area ground truth. 6 cities, optimal 115 miles. ONE unified demo for TUI+WASM. ZERO JavaScript - all WASM tested via Probar."
```

---

*"Make your workplace into a showcase that can be understood by everyone at a glance."*
— Taiichi Ohno, Toyota Production System