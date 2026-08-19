use std::path::Path;

use provable_contracts::pipeline::{self, IssueSeverity};
use serde_json::Value;

use crate::json_obj::obj;

/// Run the `pv pipeline` command: validate a pipeline contract.
pub fn run(path: &Path, format: &str) -> Result<(), Box<dyn std::error::Error>> {
    let pipeline = pipeline::parse_pipeline(path)?;
    let issues = pipeline::validate_pipeline(&pipeline);

    let errors: Vec<_> = issues
        .iter()
        .filter(|i| i.severity == IssueSeverity::Error)
        .collect();
    let warnings: Vec<_> = issues
        .iter()
        .filter(|i| i.severity == IssueSeverity::Warning)
        .collect();

    match format {
        "json" => print_json(&pipeline, &issues),
        _ => print_text(&pipeline, &errors, &warnings),
    }

    if !errors.is_empty() {
        return Err(format!("{} error(s) in pipeline", errors.len()).into());
    }

    Ok(())
}

fn print_text(
    pipeline: &pipeline::PipelineContract,
    errors: &[&pipeline::PipelineIssue],
    warnings: &[&pipeline::PipelineIssue],
) {
    println!(
        "Pipeline: {} (v{})",
        pipeline.metadata.description, pipeline.metadata.version
    );
    println!();

    // Stages
    let stage_names = collect_stage_display(&pipeline.stages, 0);
    println!("Stages ({}):", stage_names.len());
    for (indent, name, repo, contract) in &stage_names {
        let prefix = "  ".repeat(*indent + 1);
        let repo_str = repo.as_deref().unwrap_or("—");
        let contract_str = contract.as_deref().unwrap_or("—");
        println!("{prefix}{name} [{repo_str}] → {contract_str}");
    }
    println!();

    // Cross-boundary obligations
    println!(
        "Cross-boundary obligations ({}):",
        pipeline.cross_boundary_obligations.len()
    );
    for ob in &pipeline.cross_boundary_obligations {
        println!(
            "  {} {} → {}: {}",
            ob.id, ob.from_stage, ob.to_stage, ob.property
        );
    }
    println!();

    // Performance contract
    if let Some(ref perf) = pipeline.performance_contract {
        println!("Performance:");
        if let Some(ref r) = perf.roofline {
            println!("  Roofline: {r}");
        }
        if let Some(ref p) = perf.prefill_bound {
            println!("  Prefill: {p}");
        }
        if let Some(ref d) = perf.decode_bound {
            println!("  Decode: {d}");
        }
        println!();
    }

    // Issues
    if errors.is_empty() && warnings.is_empty() {
        println!("Validation: PASS (0 errors, 0 warnings)");
    } else {
        for e in errors {
            println!("  [ERROR] {}", e.message);
        }
        for w in warnings {
            println!("  [WARN]  {}", w.message);
        }
        println!(
            "\nValidation: {} ({} error(s), {} warning(s))",
            if errors.is_empty() { "PASS" } else { "FAIL" },
            errors.len(),
            warnings.len()
        );
    }
}

fn print_json(pipeline: &pipeline::PipelineContract, issues: &[pipeline::PipelineIssue]) {
    let error_count = issues
        .iter()
        .filter(|i| i.severity == IssueSeverity::Error)
        .count();
    let warning_count = issues
        .iter()
        .filter(|i| i.severity == IssueSeverity::Warning)
        .count();
    let issue_arr: Vec<Value> = issues
        .iter()
        .map(|i| {
            obj([
                (
                    "severity",
                    Value::from(match i.severity {
                        IssueSeverity::Error => "error",
                        IssueSeverity::Warning => "warning",
                    }),
                ),
                ("message", Value::from(i.message.clone())),
            ])
        })
        .collect();

    let output = obj([
        (
            "description",
            Value::from(pipeline.metadata.description.clone()),
        ),
        ("version", Value::from(pipeline.metadata.version.clone())),
        ("stage_count", Value::from(pipeline.stages.len())),
        (
            "obligation_count",
            Value::from(pipeline.cross_boundary_obligations.len()),
        ),
        ("error_count", Value::from(error_count)),
        ("warning_count", Value::from(warning_count)),
        ("issues", Value::Array(issue_arr)),
    ]);

    println!(
        "{}",
        serde_json::to_string_pretty(&output).unwrap_or_default()
    );
}

fn collect_stage_display(
    stages: &[pipeline::PipelineStage],
    depth: usize,
) -> Vec<(usize, String, Option<String>, Option<String>)> {
    let mut result = Vec::new();
    for s in stages {
        result.push((depth, s.name.clone(), s.repo.clone(), s.contract.clone()));
        if !s.substages.is_empty() {
            result.extend(collect_stage_display(&s.substages, depth + 1));
        }
    }
    result
}
