//! EDD YAML Loader Example
//!
//! This example demonstrates loading Equation Model Cards and Experiment
//! specifications from YAML files.
//!
//! Run with: cargo run --example edd_yaml_loader

use simular::edd::{EmcYaml, ExperimentYaml};

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║     EDD YAML Loader: Declarative Configuration                ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    demonstrate_emc_loading();
    demonstrate_experiment_loading();
    demonstrate_schema_validation();

    println!("\n✓ YAML loading demonstration completed!");
}

fn demonstrate_emc_loading() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Loading Equation Model Card from YAML");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let emc_yaml = r#"
emc_version: "1.0"
emc_id: "EMC-OPS-001"

identity:
  name: "Little's Law"
  version: "1.0.0"
  authors:
    - name: "PAIML Engineering"
      affiliation: "Sovereign AI Stack"
  status: "production"

governing_equation:
  latex: "L = \\lambda W"
  plain_text: "WIP = Throughput × Cycle Time"
  description: |
    Little's Law is the fundamental theorem of queueing theory.
    It states that the average number of items in a stable system
    equals the arrival rate multiplied by the average time in system.
  equation_type: "queueing"
  variables:
    - symbol: "L"
      description: "Average number of items in system (WIP)"
      units: "items"
    - symbol: "λ"
      description: "Average arrival rate (Throughput)"
      units: "items/time"
    - symbol: "W"
      description: "Average time in system (Cycle Time)"
      units: "time"

analytical_derivation:
  primary_citation:
    authors: ["Little, J.D.C."]
    title: "A Proof for the Queuing Formula: L = λW"
    journal: "Operations Research"
    year: 1961
    volume: 9
    pages: "383-387"

verification_tests:
  tests:
    - id: "LL-001"
      name: "Basic identity"
      parameters:
        lambda: 5.0
        W: 2.0
      expected:
        value: 10.0
      tolerance: 0.001

falsification_criteria:
  criteria:
    - id: "LL-FC-001"
      name: "Linear relationship"
      condition: "R² < 0.95"
      severity: "critical"
"#;

    match EmcYaml::from_yaml(emc_yaml) {
        Ok(emc) => {
            println!("✓ EMC parsed successfully!\n");
            println!("Identity:");
            println!("  Name: {}", emc.identity.name);
            println!("  Version: {}", emc.identity.version);
            println!("  Status: {}", emc.identity.status);

            println!("\nGoverning Equation:");
            println!("  LaTeX: {}", emc.governing_equation.latex);
            println!("  Type: {}", emc.governing_equation.equation_type);
            println!("  Variables: {}", emc.governing_equation.variables.len());

            if let Some(ref deriv) = emc.analytical_derivation {
                if let Some(ref cite) = deriv.primary_citation {
                    println!("\nCitation:");
                    println!("  Authors: {}", cite.authors.join(", "));
                    println!("  Title: {}", cite.title);
                    println!("  Journal: {} ({})", cite.journal, cite.year);
                }
            }

            if let Some(ref tests) = emc.verification_tests {
                println!("\nVerification Tests: {}", tests.tests.len());
                for test in &tests.tests {
                    println!("  - {}: {}", test.id, test.name);
                }
            }

            // Convert to EquationModelCard
            println!("\nConverting to EquationModelCard...");
            match emc.to_model_card() {
                Ok(mc) => {
                    println!("  ✓ Conversion successful");
                    println!("  Model: {} v{}", mc.name, mc.version);
                }
                Err(e) => println!("  ✗ Conversion failed: {e}"),
            }
        }
        Err(e) => {
            println!("✗ Failed to parse EMC: {e}");
        }
    }

    println!();
}

fn demonstrate_experiment_loading() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Loading Experiment Specification from YAML");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let experiment_yaml = r#"
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
    H₀: There is no statistically significant difference in Throughput (TH)
    or Cycle Time (CT) between Push and Pull systems when resource capacity
    and average demand are identical.
  alternative_hypothesis: |
    H₁: Pull systems achieve equivalent throughput with significantly lower
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
"#;

    match ExperimentYaml::from_yaml(experiment_yaml) {
        Ok(exp) => {
            println!("✓ Experiment parsed successfully!\n");

            println!("Metadata:");
            println!("  ID: {}", exp.experiment_id);
            println!("  Name: {}", exp.metadata.name);
            println!("  Tags: {:?}", exp.metadata.tags);

            if let Some(ref emc_ref) = exp.equation_model_card {
                println!("\nEMC Reference: {}", emc_ref.emc_ref);
            }

            if let Some(ref hyp) = exp.hypothesis {
                println!("\nHypothesis:");
                println!("  Expected outcome: {}", hyp.expected_outcome);
                println!(
                    "  H₀: {}...",
                    &hyp.null_hypothesis[..hyp.null_hypothesis.len().min(60)]
                );
            }

            println!("\nReproducibility:");
            println!("  Seed: {}", exp.reproducibility.seed);
            println!("  IEEE Strict: {}", exp.reproducibility.ieee_strict);

            if let Some(ref sim) = exp.simulation {
                if let Some(ref dur) = sim.duration {
                    println!("\nSimulation Duration:");
                    println!("  Warmup: {} time units", dur.warmup);
                    println!("  Simulation: {} time units", dur.simulation);
                    println!("  Replications: {}", dur.replications);
                }
            }

            if let Some(ref fals) = exp.falsification {
                println!("\nFalsification Criteria: {}", fals.criteria.len());
                for crit in &fals.criteria {
                    println!("  - {}: {}", crit.id, crit.name);
                    println!("    Condition: {}", crit.condition);
                }
            }

            // Convert to ExperimentSpec
            println!("\nConverting to ExperimentSpec...");
            match exp.to_experiment_spec() {
                Ok(spec) => {
                    println!("  ✓ Conversion successful");
                    println!("  Experiment: {}", spec.name());
                    println!("  Seed: {}", spec.seed());
                }
                Err(e) => println!("  ✗ Conversion failed: {e}"),
            }
        }
        Err(e) => {
            println!("✗ Failed to parse experiment: {e}");
        }
    }

    println!();
}

fn demonstrate_schema_validation() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Schema Validation");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Valid EMC
    let valid_emc = r#"
emc_version: "1.0"
emc_id: "EMC-TEST-001"
identity:
  name: "Test Equation"
  version: "1.0.0"
  status: "production"
governing_equation:
  latex: "y = mx + b"
  description: "Linear equation for testing schema validation thoroughly"
  variables:
    - symbol: "y"
      description: "Output"
      units: "units"
analytical_derivation:
  primary_citation:
    authors: ["Test Author"]
    title: "Test Paper"
    journal: "Test Journal"
    year: 2024
verification_tests:
  tests:
    - id: "T-001"
      name: "Basic test"
      parameters: {}
      expected:
        value: 1.0
      tolerance: 0.01
"#;

    println!("Test 1: Valid EMC");
    match EmcYaml::from_yaml(valid_emc) {
        Ok(emc) => match emc.validate_schema() {
            Ok(()) => println!("  ✓ Schema validation passed"),
            Err(errors) => {
                println!("  ✗ Schema validation failed:");
                for err in errors {
                    println!("    - {err}");
                }
            }
        },
        Err(e) => println!("  ✗ Parse failed: {e}"),
    }

    // Invalid EMC (missing citation)
    let invalid_emc = r#"
emc_version: "1.0"
emc_id: "EMC-TEST-002"
identity:
  name: "Incomplete EMC"
  version: "1.0.0"
governing_equation:
  latex: "x = y"
  description: "Short"
"#;

    println!("\nTest 2: Invalid EMC (missing required fields)");
    match EmcYaml::from_yaml(invalid_emc) {
        Ok(emc) => match emc.validate_schema() {
            Ok(()) => println!("  ✓ Schema validation passed (unexpected)"),
            Err(errors) => {
                println!("  ✗ Schema validation failed (expected):");
                for err in &errors[..errors.len().min(5)] {
                    println!("    - {err}");
                }
                if errors.len() > 5 {
                    println!("    ... and {} more errors", errors.len() - 5);
                }
            }
        },
        Err(e) => println!("  Parse failed: {e}"),
    }

    // Valid Experiment
    let valid_exp = r#"
experiment_version: "1.0"
experiment_id: "EXP-001"
metadata:
  name: "Test Experiment"
  description: "Testing schema validation"
equation_model_card:
  emc_ref: "test/equation"
hypothesis:
  null_hypothesis: "This is a test null hypothesis for validation purposes"
  expected_outcome: "reject"
reproducibility:
  seed: 42
falsification:
  criteria:
    - id: "F-001"
      name: "Test criterion"
      condition: "x > 0"
      severity: "major"
"#;

    println!("\nTest 3: Valid Experiment");
    match ExperimentYaml::from_yaml(valid_exp) {
        Ok(exp) => match exp.validate_schema() {
            Ok(()) => println!("  ✓ Schema validation passed"),
            Err(errors) => {
                println!("  ✗ Schema validation failed:");
                for err in errors {
                    println!("    - {err}");
                }
            }
        },
        Err(e) => println!("  ✗ Parse failed: {e}"),
    }
}
