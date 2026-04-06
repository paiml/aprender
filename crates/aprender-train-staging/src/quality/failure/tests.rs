//! Tests for failure context and analysis.

use super::*;

#[test]
fn test_failure_category_from_error_message_convergence() {
    assert_eq!(
        FailureCategory::from_error_message("Loss is NaN at step 100"),
        FailureCategory::ModelConvergence
    );
    assert_eq!(
        FailureCategory::from_error_message("Gradient exploding detected"),
        FailureCategory::ModelConvergence
    );
    assert_eq!(
        FailureCategory::from_error_message("Training diverged"),
        FailureCategory::ModelConvergence
    );
}

#[test]
fn test_failure_category_from_error_message_resource() {
    assert_eq!(
        FailureCategory::from_error_message("Out of memory"),
        FailureCategory::ResourceExhaustion
    );
    assert_eq!(
        FailureCategory::from_error_message("CUDA OOM error"),
        FailureCategory::ResourceExhaustion
    );
    assert_eq!(
        FailureCategory::from_error_message("Operation timeout"),
        FailureCategory::ResourceExhaustion
    );
}

#[test]
fn test_failure_category_from_error_message_data() {
    assert_eq!(
        FailureCategory::from_error_message("Corrupt data file"),
        FailureCategory::DataQuality
    );
    assert_eq!(
        FailureCategory::from_error_message("Invalid shape: expected [32, 512]"),
        FailureCategory::DataQuality
    );
}

#[test]
fn test_failure_category_from_error_message_dependency() {
    assert_eq!(
        FailureCategory::from_error_message("Failed to compile dependency"),
        FailureCategory::DependencyFailure
    );
    assert_eq!(
        FailureCategory::from_error_message("Version conflict in crate foo"),
        FailureCategory::DependencyFailure
    );
}

#[test]
fn test_failure_category_from_error_message_config() {
    assert_eq!(
        FailureCategory::from_error_message("Missing required field 'lr'"),
        FailureCategory::ConfigurationError
    );
    assert_eq!(
        FailureCategory::from_error_message("Invalid parameter value"),
        FailureCategory::ConfigurationError
    );
}

#[test]
fn test_failure_category_from_error_message_unknown() {
    assert_eq!(
        FailureCategory::from_error_message("Something went wrong"),
        FailureCategory::Unknown
    );
}

#[test]
fn test_failure_category_description() {
    assert_eq!(FailureCategory::ModelConvergence.description(), "Model convergence failure");
    assert_eq!(FailureCategory::Unknown.description(), "Unknown failure");
}

#[test]
fn test_failure_context_new() {
    let ctx = FailureContext::new("E001", "Loss is NaN at step 100");

    assert_eq!(ctx.error_code, "E001");
    assert_eq!(ctx.message, "Loss is NaN at step 100");
    assert_eq!(ctx.category, FailureCategory::ModelConvergence);
    assert!(ctx.stack_trace.is_none());
    assert!(ctx.suggested_fix.is_none());
    assert!(ctx.related_runs.is_empty());
}

#[test]
fn test_failure_context_with_category() {
    let ctx =
        FailureContext::with_category("E002", "Generic error", FailureCategory::ResourceExhaustion);

    assert_eq!(ctx.category, FailureCategory::ResourceExhaustion);
}

#[test]
fn test_failure_context_builders() {
    let ctx = FailureContext::new("E001", "Test error")
        .with_stack_trace("at line 100\nat line 200")
        .with_suggested_fix("Try this fix")
        .with_related_runs(vec!["run-1".to_string(), "run-2".to_string()]);

    assert_eq!(ctx.stack_trace, Some("at line 100\nat line 200".to_string()));
    assert_eq!(ctx.suggested_fix, Some("Try this fix".to_string()));
    assert_eq!(ctx.related_runs, vec!["run-1", "run-2"]);
}

#[test]
fn test_failure_context_generate_suggested_fix() {
    let ctx = FailureContext::with_category("E001", "NaN loss", FailureCategory::ModelConvergence);
    let fix = ctx.generate_suggested_fix();
    assert!(fix.contains("learning rate"));

    let ctx = FailureContext::with_category("E002", "OOM", FailureCategory::ResourceExhaustion);
    let fix = ctx.generate_suggested_fix();
    assert!(fix.contains("batch size"));
}

#[test]
fn test_failure_context_from_error() {
    use std::io;

    let error = io::Error::new(io::ErrorKind::OutOfMemory, "Out of memory");
    let ctx = FailureContext::from(&error);

    assert_eq!(ctx.category, FailureCategory::ResourceExhaustion);
    assert!(ctx.message.contains("Out of memory"));
}

#[test]
fn test_pareto_analysis_from_failures() {
    let failures = vec![
        FailureContext::with_category("E001", "NaN", FailureCategory::ModelConvergence),
        FailureContext::with_category("E002", "NaN", FailureCategory::ModelConvergence),
        FailureContext::with_category("E003", "NaN", FailureCategory::ModelConvergence),
        FailureContext::with_category("E004", "OOM", FailureCategory::ResourceExhaustion),
        FailureContext::with_category("E005", "Config", FailureCategory::ConfigurationError),
    ];

    let analysis = ParetoAnalysis::from_failures(&failures);

    assert_eq!(analysis.total_failures, 5);
    assert_eq!(analysis.categories[0].0, FailureCategory::ModelConvergence);
    assert_eq!(analysis.categories[0].1, 3);
}

#[test]
fn test_pareto_analysis_top_categories() {
    let failures = vec![
        FailureContext::with_category("E001", "NaN", FailureCategory::ModelConvergence),
        FailureContext::with_category("E002", "NaN", FailureCategory::ModelConvergence),
        FailureContext::with_category("E003", "OOM", FailureCategory::ResourceExhaustion),
        FailureContext::with_category("E004", "Config", FailureCategory::ConfigurationError),
        FailureContext::with_category("E005", "Data", FailureCategory::DataQuality),
    ];

    let analysis = ParetoAnalysis::from_failures(&failures);
    let top2 = analysis.top_categories(2);

    assert_eq!(top2.len(), 2);
    assert_eq!(top2[0].0, FailureCategory::ModelConvergence);
    assert_eq!(top2[0].1, 2);
}

#[test]
fn test_pareto_analysis_percentages() {
    let failures = vec![
        FailureContext::with_category("E001", "NaN", FailureCategory::ModelConvergence),
        FailureContext::with_category("E002", "NaN", FailureCategory::ModelConvergence),
        FailureContext::with_category("E003", "OOM", FailureCategory::ResourceExhaustion),
        FailureContext::with_category("E004", "OOM", FailureCategory::ResourceExhaustion),
    ];

    let analysis = ParetoAnalysis::from_failures(&failures);
    let percentages = analysis.percentages();

    // Both categories should be 50%
    assert!((percentages[0].1 - 50.0).abs() < f64::EPSILON);
    assert!((percentages[1].1 - 50.0).abs() < f64::EPSILON);
}

#[test]
fn test_pareto_analysis_cumulative_percentages() {
    let failures = vec![
        FailureContext::with_category("E001", "NaN", FailureCategory::ModelConvergence),
        FailureContext::with_category("E002", "NaN", FailureCategory::ModelConvergence),
        FailureContext::with_category("E003", "NaN", FailureCategory::ModelConvergence),
        FailureContext::with_category("E004", "OOM", FailureCategory::ResourceExhaustion),
    ];

    let analysis = ParetoAnalysis::from_failures(&failures);
    let cumulative = analysis.cumulative_percentages();

    // ModelConvergence is 75%, cumulative should be 75%
    assert!((cumulative[0].1 - 75.0).abs() < f64::EPSILON);
    // Adding ResourceExhaustion (25%), cumulative should be 100%
    assert!((cumulative[1].1 - 100.0).abs() < f64::EPSILON);
}

#[test]
fn test_pareto_analysis_vital_few() {
    // Create 10 failures: 6 convergence, 2 resource, 1 config, 1 data
    let mut failures = Vec::new();
    for i in 0..6 {
        failures.push(FailureContext::with_category(
            format!("E{i:03}"),
            "NaN",
            FailureCategory::ModelConvergence,
        ));
    }
    for i in 6..8 {
        failures.push(FailureContext::with_category(
            format!("E{i:03}"),
            "OOM",
            FailureCategory::ResourceExhaustion,
        ));
    }
    failures.push(FailureContext::with_category(
        "E008",
        "Config",
        FailureCategory::ConfigurationError,
    ));
    failures.push(FailureContext::with_category("E009", "Data", FailureCategory::DataQuality));

    let analysis = ParetoAnalysis::from_failures(&failures);
    let vital = analysis.vital_few();

    // ModelConvergence (60%) + ResourceExhaustion (20%) = 80%
    // So vital_few should include at least ModelConvergence
    assert!(!vital.is_empty());
    assert_eq!(vital[0].0, FailureCategory::ModelConvergence);
}

#[test]
fn test_pareto_analysis_empty() {
    let analysis = ParetoAnalysis::from_failures(&[]);

    assert_eq!(analysis.total_failures, 0);
    assert!(analysis.categories.is_empty());
    assert!(analysis.percentages().is_empty());
    assert!(analysis.cumulative_percentages().is_empty());
}

#[test]
fn test_top_failure_categories_function() {
    let failures = vec![
        FailureContext::with_category("E001", "NaN", FailureCategory::ModelConvergence),
        FailureContext::with_category("E002", "NaN", FailureCategory::ModelConvergence),
        FailureContext::with_category("E003", "OOM", FailureCategory::ResourceExhaustion),
    ];

    let categories = top_failure_categories(&failures);

    assert_eq!(categories.len(), 2);
    assert_eq!(categories[0].0, FailureCategory::ModelConvergence);
    assert_eq!(categories[0].1, 2);
}

#[test]
fn test_failure_context_serialization() {
    let ctx = FailureContext::new("E001", "Test error")
        .with_suggested_fix("Try this")
        .with_related_runs(vec!["run-1".to_string()]);

    let json = serde_json::to_string(&ctx).expect("JSON serialization should succeed");
    let parsed: FailureContext =
        serde_json::from_str(&json).expect("JSON deserialization should succeed");

    assert_eq!(parsed.error_code, ctx.error_code);
    assert_eq!(parsed.category, ctx.category);
    assert_eq!(parsed.suggested_fix, ctx.suggested_fix);
}
