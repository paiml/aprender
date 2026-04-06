# Equation Model Cards (EMC)

## Mandatory Documentation for Every Simulation

An Equation Model Card (EMC) is a structured document that bridges mathematical theory and simulation code. **No simulation can run without a complete EMC**.

## Why EMCs Are Required

Traditional simulations often suffer from:

- **Undocumented assumptions** buried in code
- **Missing citations** - which paper is this based on?
- **No verification tests** - how do we know it's correct?
- **Knowledge loss** when developers leave

EMCs solve these problems by making documentation mandatory and machine-checkable.

## EMC Structure (9 Required Sections)

### 1. Identity

```yaml
identity:
  name: "Little's Law"
  version: "1.0.0"
  uuid: "EMC-OPS-001"
  authors:
    - name: "PAIML Engineering"
      affiliation: "Sovereign AI Stack"
  status: "production"  # draft | review | production | deprecated
```

### 2. Governing Equation

```yaml
governing_equation:
  latex: "L = \\lambda W"
  plain_text: "WIP = Throughput × Cycle Time"
  description: |
    Little's Law is the fundamental theorem of queueing theory.
    It states that the average number of items in a stable system
    equals the arrival rate multiplied by the average time in system.
  equation_type: "queueing"
```

### 3. Variables

```yaml
variables:
  - symbol: "L"
    description: "Average number of items in system (WIP)"
    units: "items"
    constraints:
      min: 0
      type: "real"
  - symbol: "λ"
    description: "Average arrival rate (Throughput)"
    units: "items/time"
    constraints:
      min: 0
      positive: true
  - symbol: "W"
    description: "Average time in system (Cycle Time)"
    units: "time"
    constraints:
      min: 0
```

### 4. Analytical Derivation (Citation Required)

```yaml
analytical_derivation:
  primary_citation:
    authors: ["Little, J.D.C."]
    title: "A Proof for the Queuing Formula: L = λW"
    journal: "Operations Research"
    year: 1961
    volume: 9
    pages: "383-387"
    doi: "10.1287/opre.9.3.383"
  derivation_notes: |
    The proof relies on a simple counting argument:
    Let A(t) be arrivals by time t, D(t) be departures by time t.
    Then L(t) = A(t) - D(t) is the queue length at time t.
```

### 5. Verification Tests (TDD from Equations)

```yaml
verification_tests:
  tests:
    - id: "LL-001"
      name: "Basic identity"
      description: "λ=5, W=2 should give L=10"
      parameters:
        lambda: 5.0
        W: 2.0
      expected:
        value: 10.0
      tolerance: 0.001
    - id: "LL-002"
      name: "High throughput"
      parameters:
        lambda: 100.0
        W: 0.5
      expected:
        value: 50.0
      tolerance: 0.001
```

### 6. Domain Constraints

```yaml
domain_constraints:
  - name: "Stability condition"
    expression: "ρ < 1"
    description: "System utilization must be less than 100%"
  - name: "Non-negative WIP"
    expression: "L ≥ 0"
    description: "WIP cannot be negative"
```

### 7. Falsification Criteria

```yaml
falsification_criteria:
  criteria:
    - id: "LL-FC-001"
      name: "Linear relationship"
      condition: "R² < 0.95"
      description: "L vs λW should have R² > 0.95"
      severity: "critical"
    - id: "LL-FC-002"
      name: "Outlier detection"
      condition: "|L - λW| / λW > 0.1"
      description: "Relative error should be < 10%"
      severity: "major"
```

### 8. Implementation Notes

```yaml
implementation_notes:
  numerical_stability: |
    Little's Law is numerically stable for most practical values.
    Care needed when λ or W approach zero.
  edge_cases:
    - "λ=0: Empty system, L should be 0"
    - "W=0: Instantaneous service, L=0"
  performance: "O(1) evaluation complexity"
```

### 9. Lineage

```yaml
lineage:
  parent_equations: []
  derived_equations:
    - "CONWIP formula"
    - "Kanban sizing"
  domain_applications:
    - "Manufacturing flow analysis"
    - "Call center staffing"
    - "Web server capacity planning"
```

## Using EMCs in Code

### Creating an EMC with the Builder

```rust
use simular::edd::{EmcBuilder, Citation, EquationClass};

let emc = EmcBuilder::new()
    .name("Little's Law")
    .version("1.0.0")
    .equation("L = \\lambda W")
    .description("The fundamental theorem of queueing theory...")
    .class(EquationClass::Queueing)
    .citation(Citation::new(
        &["Little, J.D.C."],
        "Operations Research",
        1961,
    ).with_title("A Proof for the Queuing Formula: L = λW"))
    .add_variable("L", "Average number of items in system", "items")
    .add_variable("λ", "Average arrival rate", "items/time")
    .add_variable("W", "Average time in system", "time")
    .add_verification_test("λ=5, W=2 => L=10", 10.0, 1e-10)
    .build()?;
```

### Loading from YAML

```rust
use simular::edd::EmcYaml;

let yaml = std::fs::read_to_string("docs/emc/operations/littles_law.emc.yaml")?;
let emc_yaml = EmcYaml::from_yaml(&yaml)?;

// Validate schema completeness
emc_yaml.validate_schema()?;

// Convert to runtime EMC
let emc = emc_yaml.to_model_card()?;
```

### Validating EMC Compliance

```rust
use simular::edd::EddValidator;

let validator = EddValidator::new();

// EDD-01: Simulation must have EMC
validator.validate_simulation_has_emc(Some(&emc))?;

// Run verification tests
let results = emc.run_verification_tests(|params| {
    let lambda = params.get("lambda").copied().unwrap_or(0.0);
    let w = params.get("W").copied().unwrap_or(0.0);
    lambda * w
});

for (name, passed, msg) in &results {
    println!("{}: {} - {}", name, if *passed { "PASS" } else { "FAIL" }, msg);
}
```

## EMC YAML Files

EMC files are stored in `docs/emc/` organized by domain:

```
docs/emc/
├── operations/
│   ├── littles_law.emc.yaml
│   ├── kingmans_formula.emc.yaml
│   ├── square_root_law.emc.yaml
│   └── bullwhip_effect.emc.yaml
├── physics/
│   └── newtonian_gravity.emc.yaml
└── schemas/
    └── emc-v1.0.json
```

## EDD Violations

The validator checks for these violations:

| Code | Description | Severity |
|------|-------------|----------|
| EDD-01 | Missing Equation Model Card | Critical |
| EDD-02 | Missing governing equation | Critical |
| EDD-03 | Missing analytical derivation | Major |
| EDD-04 | Missing verification tests | Major |
| EDD-05 | Missing explicit seed | Critical |
| EDD-06 | Missing falsification criteria | Major |
| EDD-07 | Verification test failed | Major |
| EDD-08 | Conservation law violated | Critical |
| EDD-09 | Cross-platform reproducibility failed | Critical |
| EDD-10 | TDD compliance not met | Major |

## Running the Example

```bash
cargo run --example edd_model_card
```

## Next Chapter

Learn about Experiment Specifications and YAML-driven declarative experiments.
