# Experiment Specification

## YAML-Driven Declarative Experiments

EDD experiments are defined declaratively in YAML, not imperatively in code. This ensures reproducibility, documentation, and scientific rigor.

## Experiment Structure

```yaml
experiment_version: "1.0"
experiment_id: "TPS-TC-001"

metadata:
  name: "Push vs Pull Effectiveness"
  description: |
    Validate CONWIP superiority via Little's Law.
    This is TPS Test Case 1 from the EDD specification.
  tags: ["tps", "conwip", "littles-law"]

equation_model_card:
  emc_ref: "operations/littles_law"

hypothesis:
  null_hypothesis: |
    H0: There is no statistically significant difference in Throughput (TH)
    or Cycle Time (CT) between Push and Pull systems when resource capacity
    and average demand are identical.
  alternative_hypothesis: |
    H1: Pull systems achieve equivalent throughput with significantly lower
    WIP and cycle time due to explicit WIP control.
  expected_outcome: "reject"

reproducibility:
  seed: 42
  ieee_strict: true

simulation:
  duration:
    warmup: 100.0
    simulation: 1000.0
    replications: 30
  parameters:
    stations: 5
    arrival_rate: 4.5
    cv_arrivals: 1.5
    cv_service: 1.2

falsification:
  import_from_emc: true
  criteria:
    - id: "TC1-CT"
      name: "Cycle time reduction"
      condition: "pull_ct < push_ct * 0.6"
      severity: "critical"
```

## Required Sections

### 1. Reproducibility (Pillar 3: Seed It)

**Every experiment must specify an explicit seed.** No exceptions.

```yaml
reproducibility:
  seed: 42                    # Master seed (REQUIRED)
  ieee_strict: true           # IEEE 754 strict floating-point
  component_seeds:            # Optional: per-component seeds
    arrivals: 12345
    service: 67890
```

### 2. Hypothesis (Scientific Method)

State what you're testing in falsifiable terms:

```yaml
hypothesis:
  null_hypothesis: "H0: System A and B have equal performance"
  alternative_hypothesis: "H1: System A outperforms System B"
  expected_outcome: "reject"  # What we expect to happen
  alpha: 0.05                 # Significance level
```

### 3. Falsification Criteria (Pillar 4: Falsify It)

Define what would disprove your model:

```yaml
falsification:
  import_from_emc: true       # Import criteria from EMC
  criteria:
    - id: "FC-001"
      name: "Energy conservation"
      condition: "|E(t) - E(0)| / E(0) < 1e-6"
      severity: "critical"    # critical | major | minor
    - id: "FC-002"
      name: "Little's Law holds"
      condition: "R² > 0.95"
      severity: "major"
```

### 4. Simulation Parameters

```yaml
simulation:
  duration:
    warmup: 100.0            # Warmup period (discarded)
    simulation: 1000.0       # Main simulation time
    replications: 30         # Number of independent runs
  parameters:
    # Domain-specific parameters
    stations: 5
    arrival_rate: 4.5
```

## Loading Experiments in Code

```rust
use simular::edd::ExperimentYaml;

let yaml = std::fs::read_to_string("experiments/tps-tc-001.yaml")?;
let exp_yaml = ExperimentYaml::from_yaml(&yaml)?;

// Validate schema
exp_yaml.validate_schema()?;

// Convert to runtime spec
let spec = exp_yaml.to_experiment_spec()?;

println!("Experiment: {}", spec.name());
println!("Seed: {}", spec.seed());
```

## Building Experiments Programmatically

```rust
use simular::edd::{ExperimentSpec, ExperimentHypothesis, FalsificationCriterion};

let spec = ExperimentSpec::builder()
    .name("Little's Law Validation")
    .seed(42)
    .hypothesis(ExperimentHypothesis::new(
        "H0: L ≠ λW under stochastic conditions",
        "H1: L = λW holds even with variability",
    ))
    .add_falsification_criterion(FalsificationCriterion::new(
        "linear_relationship",
        "R² < 0.95",
        FalsificationAction::RejectModel,
    ))
    .build()?;
```

## Deterministic Seed Management

The `ExperimentSeed` struct manages reproducibility:

```rust
use simular::edd::ExperimentSeed;

// Create with master seed
let seed = ExperimentSeed::new(42)
    .with_component("arrivals", 12345)   // Override for specific component
    .with_component("service", 67890);

// Derive seeds for other components (deterministic from master)
let routing_seed = seed.derive_seed("routing");
let demand_seed = seed.derive_seed("demand");

// Same component name always produces same seed
assert_eq!(seed.derive_seed("routing"), routing_seed);
```

### Cross-Run Reproducibility

```rust
// Different ExperimentSeed instances with same master seed
let seed_a = ExperimentSeed::new(42);
let seed_b = ExperimentSeed::new(42);

// Produce identical derived seeds
assert_eq!(seed_a.derive_seed("test"), seed_b.derive_seed("test"));
```

## Validation Checks

The experiment spec is validated for:

| Check | Description |
|-------|-------------|
| Seed present | Explicit seed must be specified |
| EMC reference valid | If specified, EMC must exist |
| Hypothesis stated | Null and alternative hypotheses required |
| Falsification criteria | At least one criterion required |
| Parameter completeness | All required parameters present |

## EDD Validator Integration

```rust
use simular::edd::EddValidator;

let validator = EddValidator::new();

// EDD-05: Experiment must have explicit seed
validator.validate_seed_specified(Some(42))?;

// EDD-06: Must have falsification criteria
validator.validate_falsification_criteria(&spec)?;
```

## Experiment Output

When an experiment completes, it produces:

```yaml
experiment_result:
  experiment_id: "TPS-TC-001"
  status: "completed"
  seed_used: 42

  hypothesis_test:
    h0_rejected: true
    p_value: 0.0001
    effect_size: 0.59
    confidence_interval: [0.55, 0.63]

  falsification:
    criteria_checked: 2
    criteria_violated: 0
    robustness: 0.12

  metrics:
    push_system:
      wip: 24.5
      throughput: 4.45
      cycle_time: 5.4
    pull_system:
      wip: 10.0
      throughput: 4.42
      cycle_time: 2.2
```

## Running the Example

```bash
cargo run --example edd_yaml_loader
```

## Next Chapter

Learn about Popperian Falsification - how to actively search for conditions that disprove your model.
