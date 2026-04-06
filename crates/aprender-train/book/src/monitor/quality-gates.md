# Quality Gates (Jidoka)

Entrenar implements quality gates following Jidoka (自働化) principles -
automation with a human touch. The quality module provides structured metrics,
supply chain auditing, and failure diagnostics to ensure training runs
meet quality standards before deployment.

## Overview

The quality gates system consists of three components:

1. **CodeQualityMetrics** - PMAT-style code quality tracking
2. **DependencyAudit** - Supply chain security via cargo-deny
3. **FailureContext** - Structured failure diagnostics with Pareto analysis

## Code Quality Metrics (PMAT)

Track code quality metrics following the PMAT methodology:

```rust
use entrenar::quality::{CodeQualityMetrics, PmatGrade};

// Create metrics manually
let metrics = CodeQualityMetrics::new(
    92.5,  // coverage_percent
    85.0,  // mutation_score
    0,     // clippy_warnings
);

// Check quality thresholds
assert!(metrics.meets_threshold(90.0, 80.0));
assert_eq!(metrics.pmat_grade, PmatGrade::B);
assert!(metrics.is_clippy_clean());
```

### Parsing CI Output

Parse metrics directly from cargo tool output:

```rust
use entrenar::quality::CodeQualityMetrics;

// From cargo llvm-cov --json
let coverage_json = r#"{"data":[{"totals":{"lines":{"percent":95.5}}}]}"#;

// From cargo mutants --json
let mutants_json = r#"{"total_mutants":100,"caught":88,"missed":10,"timeout":2}"#;

let metrics = CodeQualityMetrics::from_cargo_output(
    coverage_json,
    mutants_json,
    0,  // clippy warnings count
).unwrap();

println!("Coverage: {:.1}%", metrics.coverage_percent);
println!("Mutation: {:.1}%", metrics.mutation_score);
println!("Grade: {}", metrics.pmat_grade);
```

### PMAT Grade Thresholds

| Grade | Coverage | Mutation Score |
|-------|----------|----------------|
| A     | >= 95%   | >= 85%         |
| B     | >= 85%   | >= 75%         |
| C     | >= 75%   | >= 65%         |
| D     | >= 60%   | >= 50%         |
| F     | < 60%    | < 50%          |

```rust
use entrenar::quality::PmatGrade;

// Calculate grade from scores
let grade = PmatGrade::from_scores(92.0, 80.0);
assert_eq!(grade, PmatGrade::B);

// Check if grade meets target
assert!(PmatGrade::A.meets_target(PmatGrade::B));
assert!(!PmatGrade::C.meets_target(PmatGrade::A));
```

## Supply Chain Auditing

Integrate with cargo-deny for dependency vulnerability scanning:

```rust
use entrenar::quality::{DependencyAudit, Advisory, Severity, AuditStatus};

// Create a clean audit
let audit = DependencyAudit::clean("serde", "1.0.200", "MIT OR Apache-2.0");
assert!(!audit.is_vulnerable());

// Create a vulnerable audit
let advisory = Advisory::new(
    "RUSTSEC-2024-0001",
    Severity::Critical,
    "Remote code execution vulnerability",
);
let audit = DependencyAudit::vulnerable(
    "unsafe-crate",
    "0.1.0",
    "MIT",
    vec![advisory],
);
assert!(audit.is_vulnerable());
assert_eq!(audit.max_severity(), Severity::Critical);
```

### Parsing cargo-deny Output

```rust
use entrenar::quality::DependencyAudit;

// Parse cargo deny check --format json output
let cargo_deny_output = r#"{"type":"diagnostic","fields":{"severity":"error","code":"A001","message":"Vulnerability found","labels":[{"span":{"crate":{"name":"vuln-crate","version":"1.0.0"}}}]}}"#;

let audits = DependencyAudit::from_cargo_deny_output(cargo_deny_output).unwrap();

for audit in &audits {
    if audit.is_vulnerable() {
        println!("VULNERABLE: {} v{}", audit.crate_name, audit.version);
        for advisory in &audit.advisories {
            println!("  - {} ({}): {}", advisory.id, advisory.severity, advisory.title);
        }
    }
}
```

### Audit Summary

Aggregate results for reporting:

```rust
use entrenar::quality::supply_chain::AuditSummary;

let summary = AuditSummary::from_audits(audits);

println!("Total dependencies: {}", summary.total_dependencies);
println!("Clean: {}", summary.clean_count);
println!("Warnings: {}", summary.warning_count);
println!("Vulnerable: {}", summary.vulnerable_count);

if summary.has_vulnerabilities() {
    println!("FAILED: Security vulnerabilities found!");
    for dep in summary.vulnerable_deps() {
        println!("  - {} v{}", dep.crate_name, dep.version);
    }
}
```

### Severity Levels

| Level    | Description                    |
|----------|--------------------------------|
| Critical | Immediate action required      |
| High     | Should be fixed soon           |
| Medium   | Fix when convenient            |
| Low      | Minor issues                   |
| None     | Informational                  |

```rust
use entrenar::quality::Severity;

// Severity is ordered for comparison
assert!(Severity::Critical > Severity::High);
assert!(Severity::High > Severity::Medium);

// Parse from string
let severity = Severity::parse("critical");
assert_eq!(severity, Severity::Critical);
```

## Failure Diagnostics

Structured failure context with automatic categorization:

```rust
use entrenar::quality::{FailureContext, FailureCategory};

// Auto-categorization from error message
let ctx = FailureContext::new("E001", "Training failed: loss is NaN at step 500");
assert_eq!(ctx.category, FailureCategory::ModelConvergence);

// With explicit category
let ctx = FailureContext::with_category(
    "OOM_001",
    "CUDA out of memory",
    FailureCategory::ResourceExhaustion,
);
```

### Failure Categories

| Category           | Patterns                                    |
|--------------------|---------------------------------------------|
| ModelConvergence   | NaN, Inf, exploding gradient, diverge       |
| ResourceExhaustion | OOM, out of memory, timeout, disk full      |
| DataQuality        | corrupt, invalid data, missing feature      |
| DependencyFailure  | compile, crate, version conflict            |
| ConfigurationError | config, parameter, missing field            |
| Unknown            | Default for unrecognized patterns           |

### Enriching Failure Context

```rust
use entrenar::quality::FailureContext;

let ctx = FailureContext::new("NAN_LOSS", "Loss became NaN at step 1000")
    .with_stack_trace("at training_loop:125\nat step:50")
    .with_suggested_fix("Try reducing learning rate to 1e-5")
    .with_related_runs(vec!["run-001".to_string(), "run-002".to_string()]);

// Auto-generate suggested fix based on category
let auto_fix = ctx.generate_suggested_fix();
println!("Suggested fix: {}", auto_fix);
```

### From Standard Errors

```rust
use entrenar::quality::FailureContext;
use std::io;

let error = io::Error::new(io::ErrorKind::OutOfMemory, "System out of memory");
let ctx = FailureContext::from(&error);

assert_eq!(ctx.category, entrenar::quality::FailureCategory::ResourceExhaustion);
```

## Pareto Analysis

Identify the vital few failure categories (80/20 rule):

```rust
use entrenar::quality::{FailureContext, FailureCategory};
use entrenar::quality::failure::ParetoAnalysis;

// Collect failures from multiple runs
let failures: Vec<FailureContext> = collect_failures_from_runs();

let analysis = ParetoAnalysis::from_failures(&failures);

// Get top categories
println!("Top failure categories:");
for (category, count) in analysis.top_categories(3) {
    println!("  {:?}: {} failures", category, count);
}

// Get percentages
for (category, percent) in analysis.percentages() {
    println!("  {:?}: {:.1}%", category, percent);
}

// Find vital few (categories causing ~80% of failures)
let vital = analysis.vital_few();
println!("Focus on these {} categories to address 80% of failures:", vital.len());
```

### Convenience Function

```rust
use entrenar::quality::failure::top_failure_categories;

let categories = top_failure_categories(&failures);
// Returns Vec<(FailureCategory, u32)> sorted by count descending
```

## Quality Gate Workflow

Complete workflow integrating all components:

```rust
use entrenar::quality::{CodeQualityMetrics, DependencyAudit, FailureContext, PmatGrade};

fn run_quality_gates() -> Result<(), String> {
    // Step 1: Check code quality
    let coverage_json = run_coverage_tool();
    let mutants_json = run_mutation_testing();
    let clippy_warnings = run_clippy();

    let metrics = CodeQualityMetrics::from_cargo_output(
        &coverage_json,
        &mutants_json,
        clippy_warnings,
    ).map_err(|e| e.to_string())?;

    if !metrics.meets_threshold(90.0, 80.0) {
        return Err(format!(
            "Quality gate failed: coverage {:.1}%, mutation {:.1}%, grade {}",
            metrics.coverage_percent,
            metrics.mutation_score,
            metrics.pmat_grade
        ));
    }

    if !metrics.is_clippy_clean() {
        return Err(format!("{} clippy warnings found", metrics.clippy_warnings));
    }

    // Step 2: Check supply chain security
    let deny_output = run_cargo_deny();
    let audits = DependencyAudit::from_cargo_deny_output(&deny_output)
        .map_err(|e| e.to_string())?;

    let vulnerable: Vec<_> = audits.iter().filter(|a| a.is_vulnerable()).collect();
    if !vulnerable.is_empty() {
        return Err(format!(
            "Security vulnerabilities found in {} dependencies",
            vulnerable.len()
        ));
    }

    println!("All quality gates passed!");
    println!("  Coverage: {:.1}%", metrics.coverage_percent);
    println!("  Mutation: {:.1}%", metrics.mutation_score);
    println!("  Grade: {}", metrics.pmat_grade);

    Ok(())
}
```

## Integration with Training Runs

Log quality metrics as part of experiment tracking:

```rust
use std::sync::{Arc, Mutex};
use entrenar::storage::{InMemoryStorage, ExperimentStorage};
use entrenar::run::{Run, TracingConfig};
use entrenar::quality::CodeQualityMetrics;

// Setup experiment
let mut storage = InMemoryStorage::new();
let exp_id = storage.create_experiment("quality-tracked-training", None).unwrap();
let storage = Arc::new(Mutex::new(storage));

let mut run = Run::new(&exp_id, storage.clone(), TracingConfig::default()).unwrap();

// ... training loop ...

// Log quality metrics at the end
let metrics = CodeQualityMetrics::new(95.0, 88.0, 0);
run.log_metric("code_coverage", metrics.coverage_percent).unwrap();
run.log_metric("mutation_score", metrics.mutation_score).unwrap();

// Complete run based on quality gate
let status = if metrics.meets_grade(entrenar::quality::PmatGrade::A) {
    entrenar::storage::RunStatus::Success
} else {
    entrenar::storage::RunStatus::Failed
};

run.finish(status).unwrap();
```

## Configuration

Quality thresholds can be configured per project:

```yaml
# entrenar.yaml
quality:
  coverage:
    minimum: 90.0
    target: 95.0
  mutation:
    minimum: 80.0
    target: 85.0
  clippy:
    allow_warnings: false
  supply_chain:
    fail_on_vulnerability: true
    allowed_licenses:
      - MIT
      - Apache-2.0
      - BSD-3-Clause
```
