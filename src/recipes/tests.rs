//! Tests for recipe modules.

use super::*;
use crate::experiment::{CpuArchitecture, ExperimentStorage, InMemoryExperimentStorage};
use std::collections::HashMap;

fn point(id: &str, performance: f64, cost: f64, energy: f64) -> CostPerformancePoint {
    CostPerformancePoint {
        id: id.to_string(),
        performance,
        cost,
        energy_joules: energy,
        latency_ms: None,
        metadata: HashMap::new(),
    }
}

// -------------------------------------------------------------------------
// RecipeResult Tests
// -------------------------------------------------------------------------

#[test]
fn test_recipe_result_success() {
    let result = RecipeResult::success("test-recipe");
    assert!(result.success);
    assert_eq!(result.recipe_name, "test-recipe");
    assert!(result.error.is_none());
}

#[test]
fn test_recipe_result_failure() {
    let result = RecipeResult::failure("test-recipe", "Something went wrong");
    assert!(!result.success);
    assert_eq!(result.error, Some("Something went wrong".to_string()));
}

#[test]
fn test_recipe_result_with_artifact() {
    let result =
        RecipeResult::success("test").with_artifact("artifact1").with_artifact("artifact2");
    assert_eq!(result.artifacts.len(), 2);
}

#[test]
fn test_recipe_result_with_metric() {
    let result =
        RecipeResult::success("test").with_metric("accuracy", 0.95).with_metric("loss", 0.05);
    assert_eq!(result.metrics.get("accuracy"), Some(&0.95));
    assert_eq!(result.metrics.get("loss"), Some(&0.05));
}

// -------------------------------------------------------------------------
// ExperimentTrackingRecipe Tests
// -------------------------------------------------------------------------

#[test]
fn test_experiment_tracking_recipe_creation() {
    let config = ExperimentTrackingConfig::default();
    let recipe = ExperimentTrackingRecipe::new(config);
    assert!(recipe.current_run().is_none());
}

#[test]
fn test_experiment_tracking_start_run() {
    let config = ExperimentTrackingConfig::default();
    let mut recipe = ExperimentTrackingRecipe::new(config);

    let run = recipe.start_run("run-001");
    assert_eq!(run.run_id, "run-001");
    assert!(recipe.current_run().is_some());
}

#[test]
fn test_experiment_tracking_log_metric() {
    let config = ExperimentTrackingConfig::default();
    let mut recipe = ExperimentTrackingRecipe::new(config);

    recipe.start_run("run-001");
    recipe.log_metric("accuracy", 0.95).unwrap();

    let run = recipe.current_run().unwrap();
    assert_eq!(run.metrics.get("accuracy"), Some(&0.95));
}

#[test]
fn test_experiment_tracking_log_metric_no_run() {
    let config = ExperimentTrackingConfig::default();
    let mut recipe = ExperimentTrackingRecipe::new(config);

    let result = recipe.log_metric("accuracy", 0.95);
    assert!(result.is_err());
}

#[test]
fn test_experiment_tracking_end_run() {
    let config = ExperimentTrackingConfig::default();
    let mut recipe = ExperimentTrackingRecipe::new(config);

    recipe.start_run("run-001");
    recipe.log_metric("accuracy", 0.95).unwrap();

    let result = recipe.end_run(true).unwrap();
    assert!(result.success);
    assert!(result.metrics.contains_key("duration_seconds"));
}

#[test]
fn test_experiment_tracking_store_run() {
    let config = ExperimentTrackingConfig::default();
    let mut recipe = ExperimentTrackingRecipe::new(config);
    let storage = InMemoryExperimentStorage::new();

    recipe.start_run("run-001");
    recipe.store_run(&storage).unwrap();

    let retrieved = storage.get_run("run-001").unwrap();
    assert!(retrieved.is_some());
}

// -------------------------------------------------------------------------
// CostPerformanceBenchmarkRecipe Tests
// -------------------------------------------------------------------------

#[test]
fn test_benchmark_recipe_creation() {
    let recipe = CostPerformanceBenchmarkRecipe::new("test-benchmark");
    assert_eq!(recipe.benchmark().name, "test-benchmark");
}

#[test]
fn test_benchmark_recipe_with_budget() {
    let recipe = CostPerformanceBenchmarkRecipe::new("test").with_budget(100.0);
    assert_eq!(recipe.budget_constraint, Some(100.0));
}

#[test]
fn test_benchmark_recipe_analyze() {
    let mut recipe = CostPerformanceBenchmarkRecipe::new("test")
        .with_budget(150.0)
        .with_performance_target(0.90);

    // Add some points directly
    recipe.benchmark_mut().add_point(point("config1", 0.95, 100.0, 1000.0));

    recipe.benchmark_mut().add_point(point("config2", 0.85, 50.0, 500.0));

    let result = recipe.analyze();
    assert!(result.success);
    assert!(result.metrics.contains_key("pareto_optimal_count"));
    assert!(result.metrics.contains_key("meets_target"));
}

// -------------------------------------------------------------------------
// SovereignDeploymentRecipe Tests
// -------------------------------------------------------------------------

#[test]
fn test_sovereign_deployment_creation() {
    let config = SovereignDeploymentConfig::default();
    let recipe = SovereignDeploymentRecipe::new(config);
    assert_eq!(recipe.distribution().name, "sovereign-model");
}

#[test]
fn test_sovereign_deployment_add_artifacts() {
    let config = SovereignDeploymentConfig { require_signatures: false, ..Default::default() };
    let mut recipe = SovereignDeploymentRecipe::new(config);

    recipe.add_model("model.onnx", "sha256_hash", 1000000);
    recipe.add_binary("inference", "sha256_hash2", 500000);

    assert_eq!(recipe.distribution().artifacts.len(), 2);
}

#[test]
fn test_sovereign_deployment_build_without_signatures() {
    let config = SovereignDeploymentConfig { require_signatures: false, ..Default::default() };
    let mut recipe = SovereignDeploymentRecipe::new(config);
    recipe.add_model("model.onnx", "sha256_hash", 1000000);

    let result = recipe.build().unwrap();
    assert!(result.success);
    assert_eq!(result.metrics.get("artifact_count"), Some(&1.0));
}

#[test]
fn test_sovereign_deployment_build_with_signatures() {
    let config = SovereignDeploymentConfig { require_signatures: true, ..Default::default() };
    let mut recipe = SovereignDeploymentRecipe::new(config);
    recipe.add_model("model.onnx", "sha256_hash", 1000000);
    recipe.sign_artifact("model.onnx", "key-001");

    let result = recipe.build().unwrap();
    assert!(result.success);
}

#[test]
fn test_sovereign_deployment_missing_signature() {
    let config = SovereignDeploymentConfig { require_signatures: true, ..Default::default() };
    let mut recipe = SovereignDeploymentRecipe::new(config);
    recipe.add_model("model.onnx", "sha256_hash", 1000000);

    let result = recipe.build();
    assert!(result.is_err());
}

#[test]
fn test_sovereign_deployment_export_manifest() {
    let config = SovereignDeploymentConfig { require_signatures: false, ..Default::default() };
    let mut recipe = SovereignDeploymentRecipe::new(config);
    recipe.add_model("model.onnx", "sha256_hash", 1000000);

    let manifest = recipe.export_manifest().unwrap();
    assert!(manifest.contains("sovereign-model"));
    assert!(manifest.contains("model.onnx"));
}

// -------------------------------------------------------------------------
// ResearchArtifactRecipe Tests
// -------------------------------------------------------------------------

#[test]
fn test_research_artifact_creation() {
    let recipe = ResearchArtifactRecipe::new("Test Paper", "This is the abstract.");
    assert_eq!(recipe.artifact().title, "Test Paper");
}

#[test]
fn test_research_artifact_add_contributor() {
    let mut recipe = ResearchArtifactRecipe::new("Test", "Abstract");
    recipe.add_contributor(
        "Alice Smith",
        "MIT",
        vec![CreditRole::Conceptualization, CreditRole::Software],
    );

    assert_eq!(recipe.artifact().contributors.len(), 1);
    assert_eq!(recipe.artifact().contributors[0].name, "Alice Smith");
}

#[test]
fn test_research_artifact_generate_citation() {
    let mut recipe = ResearchArtifactRecipe::new("Test Paper", "Abstract");
    recipe.add_contributor("Alice Smith", "MIT", vec![CreditRole::Software]);
    recipe.set_doi("10.1234/test");

    let citation = recipe.generate_citation();
    assert_eq!(citation.title, "Test Paper");
    assert_eq!(citation.authors, vec!["Alice Smith"]);
    assert_eq!(citation.doi, Some("10.1234/test".to_string()));
}

#[test]
fn test_research_artifact_build() {
    let mut recipe = ResearchArtifactRecipe::new("Test Paper", "Abstract");
    recipe.add_contributor("Alice Smith", "MIT", vec![CreditRole::Software]);
    recipe.add_keywords(vec!["ML".to_string(), "Rust".to_string()]);
    recipe.set_doi("10.1234/test");

    let result = recipe.build();
    assert!(result.success);
    assert_eq!(result.metrics.get("contributor_count"), Some(&1.0));
    assert_eq!(result.metrics.get("keyword_count"), Some(&2.0));
}

// -------------------------------------------------------------------------
// CiCdBenchmarkRecipe Tests
// -------------------------------------------------------------------------

#[test]
fn test_cicd_recipe_creation() {
    let recipe = CiCdBenchmarkRecipe::new("ci-benchmark");
    assert_eq!(recipe.name, "ci-benchmark");
}

#[test]
fn test_cicd_recipe_thresholds() {
    let mut recipe = CiCdBenchmarkRecipe::new("ci-benchmark");
    recipe.add_min_performance_threshold("performance", 0.90);
    recipe.add_max_cost_threshold(100.0);

    assert_eq!(recipe.thresholds.get("min_performance"), Some(&0.90));
    assert_eq!(recipe.thresholds.get("max_cost"), Some(&100.0));
}

#[test]
fn test_cicd_recipe_check_pass() {
    let mut recipe = CiCdBenchmarkRecipe::new("ci-benchmark");
    recipe.add_min_performance_threshold("performance", 0.90);
    recipe.add_max_cost_threshold(100.0);

    recipe.add_result(point("test", 0.95, 80.0, 1000.0));

    let result = recipe.check();
    assert!(result.success);
    assert_eq!(result.metrics.get("all_checks_passed"), Some(&1.0));
}

#[test]
fn test_cicd_recipe_check_fail_cost() {
    let mut recipe = CiCdBenchmarkRecipe::new("ci-benchmark");
    recipe.add_max_cost_threshold(100.0);

    recipe.add_result(point("test", 0.95, 150.0, 1000.0));

    let result = recipe.check();
    assert!(!result.success);
    assert_eq!(result.metrics.get("all_checks_passed"), Some(&0.0));
}

#[test]
fn test_cicd_recipe_check_fail_performance() {
    let mut recipe = CiCdBenchmarkRecipe::new("ci-benchmark");
    recipe.add_min_performance_threshold("performance", 0.90);

    recipe.add_result(point("test", 0.85, 80.0, 1000.0));

    let result = recipe.check();
    assert!(!result.success);
}

// -------------------------------------------------------------------------
// Integration Tests
// -------------------------------------------------------------------------

#[test]
fn test_full_experiment_workflow() {
    // 1. Create experiment tracking
    let config = ExperimentTrackingConfig {
        experiment_name: "integration-test".to_string(),
        paradigm: ModelParadigm::DeepLearning,
        device: ComputeDevice::Cpu {
            cores: 8,
            threads_per_core: 2,
            architecture: CpuArchitecture::X86_64,
        },
        platform: PlatformEfficiency::Server,
        track_energy: true,
        track_cost: false,
        carbon_intensity: Some(400.0),
        tags: vec!["test".to_string()],
    };
    let mut tracking = ExperimentTrackingRecipe::new(config);

    // 2. Run experiment
    tracking.start_run("run-001");
    tracking.log_metric("accuracy", 0.95).unwrap();
    tracking.log_metric("loss", 0.05).unwrap();
    tracking.log_param("learning_rate", serde_json::json!(0.001)).unwrap();

    let result = tracking.end_run(true).unwrap();
    assert!(result.success);

    // 3. Store in backend
    let storage = InMemoryExperimentStorage::new();
    tracking.store_run(&storage).unwrap();

    // 4. Verify stored
    let runs = storage.list_runs("integration-test").unwrap();
    assert_eq!(runs.len(), 1);
}

#[test]
fn test_benchmark_to_cicd_workflow() {
    // 1. Create benchmark
    let mut benchmark = CostPerformanceBenchmarkRecipe::new("perf-test")
        .with_budget(200.0)
        .with_performance_target(0.85);

    // 2. Add configurations
    benchmark.benchmark_mut().add_point(CostPerformancePoint {
        id: "small-model".to_string(),
        performance: 0.88,
        cost: 50.0,
        energy_joules: 500.0,
        latency_ms: Some(10.0),
        metadata: HashMap::new(),
    });

    benchmark.benchmark_mut().add_point(CostPerformancePoint {
        id: "large-model".to_string(),
        performance: 0.95,
        cost: 150.0,
        energy_joules: 1500.0,
        latency_ms: Some(50.0),
        metadata: HashMap::new(),
    });

    // 3. Analyze
    let analysis = benchmark.analyze();
    assert!(analysis.success);

    // 4. Feed into CI/CD
    let mut cicd = CiCdBenchmarkRecipe::new("ci-gate");
    cicd.add_min_performance_threshold("performance", 0.85);
    cicd.add_max_cost_threshold(200.0);

    for point in benchmark.benchmark().points.iter() {
        cicd.add_result(point.clone());
    }

    let cicd_result = cicd.check();
    assert!(cicd_result.success);
}

// -------------------------------------------------------------------------
// Additional Coverage Tests
// -------------------------------------------------------------------------

#[test]
fn test_experiment_tracking_end_run_failed() {
    let config = ExperimentTrackingConfig::default();
    let mut recipe = ExperimentTrackingRecipe::new(config);

    recipe.start_run("run-fail");
    let result = recipe.end_run(false).unwrap();
    assert!(result.success); // Recipe succeeds even if run fails
}

#[test]
fn test_experiment_tracking_end_run_no_start() {
    let config = ExperimentTrackingConfig::default();
    let mut recipe = ExperimentTrackingRecipe::new(config);

    let result = recipe.end_run(true);
    assert!(result.is_err());
}

#[test]
fn test_experiment_tracking_log_param_no_run() {
    let config = ExperimentTrackingConfig::default();
    let mut recipe = ExperimentTrackingRecipe::new(config);

    let result = recipe.log_param("lr", serde_json::json!(0.01));
    assert!(result.is_err());
}

#[test]
fn test_experiment_tracking_store_run_no_run() {
    let config = ExperimentTrackingConfig::default();
    let recipe = ExperimentTrackingRecipe::new(config);
    let storage = InMemoryExperimentStorage::new();

    // store_run silently succeeds when no run exists (does nothing)
    let result = recipe.store_run(&storage);
    assert!(result.is_ok());
}

#[test]
fn test_experiment_tracking_without_energy() {
    let config = ExperimentTrackingConfig { track_energy: false, ..Default::default() };
    let mut recipe = ExperimentTrackingRecipe::new(config);

    recipe.start_run("run-no-energy");
    let result = recipe.end_run(true).unwrap();
    assert!(result.success);
}

#[test]
fn test_experiment_tracking_without_carbon() {
    let config = ExperimentTrackingConfig {
        track_energy: true,
        carbon_intensity: None,
        ..Default::default()
    };
    let mut recipe = ExperimentTrackingRecipe::new(config);

    recipe.start_run("run-no-carbon");
    let result = recipe.end_run(true).unwrap();
    assert!(result.success);
}

#[test]
fn test_benchmark_recipe_without_budget() {
    let mut recipe = CostPerformanceBenchmarkRecipe::new("no-budget");

    recipe.benchmark_mut().add_point(point("test", 0.9, 100.0, 1000.0));

    let result = recipe.analyze();
    assert!(result.success);
    // Without budget, meets_budget is not in metrics
}

#[test]
fn test_benchmark_recipe_without_target() {
    let mut recipe = CostPerformanceBenchmarkRecipe::new("no-target");

    recipe.benchmark_mut().add_point(point("test", 0.9, 100.0, 1000.0));

    let result = recipe.analyze();
    assert!(result.success);
}

#[test]
fn test_benchmark_recipe_empty() {
    let mut recipe = CostPerformanceBenchmarkRecipe::new("empty");
    let result = recipe.analyze();
    assert!(result.success);
    assert_eq!(result.metrics.get("pareto_optimal_count"), Some(&0.0));
}

#[test]
fn test_sovereign_deployment_sign_nonexistent() {
    let config = SovereignDeploymentConfig::default();
    let mut recipe = SovereignDeploymentRecipe::new(config);
    recipe.sign_artifact("nonexistent.onnx", "key-001");
    // Should not crash, just does nothing
}

#[test]
fn test_research_artifact_add_multiple_contributors() {
    let mut recipe = ResearchArtifactRecipe::new("Test", "Abstract");
    recipe.add_contributor("Alice", "MIT", vec![CreditRole::Conceptualization]);
    recipe.add_contributor("Bob", "Stanford", vec![CreditRole::Software]);
    recipe.add_contributor("Carol", "CMU", vec![CreditRole::DataCuration]);

    assert_eq!(recipe.artifact().contributors.len(), 3);

    let citation = recipe.generate_citation();
    assert_eq!(citation.authors.len(), 3);
}

#[test]
fn test_research_artifact_multiple_roles() {
    let mut recipe = ResearchArtifactRecipe::new("Test", "Abstract");
    recipe.add_contributor(
        "Alice",
        "MIT",
        vec![CreditRole::Conceptualization, CreditRole::Software, CreditRole::WritingOriginalDraft],
    );

    let contributor = &recipe.artifact().contributors[0];
    assert_eq!(contributor.roles.len(), 3);
}

#[test]
fn test_cicd_recipe_empty_results() {
    let mut recipe = CiCdBenchmarkRecipe::new("empty-check");
    recipe.add_min_performance_threshold("performance", 0.9);

    let result = recipe.check();
    assert!(result.success); // No results means all pass
}

#[test]
fn test_cicd_recipe_multiple_results() {
    let mut recipe = CiCdBenchmarkRecipe::new("multi-check");
    recipe.add_min_performance_threshold("performance", 0.8);
    recipe.add_max_cost_threshold(200.0);

    // Add multiple results, all passing
    for i in 0..5 {
        recipe.add_result(point(
            &format!("test-{}", i),
            0.85 + (i as f64 * 0.02),
            100.0 + (i as f64 * 10.0),
            1000.0,
        ));
    }

    let result = recipe.check();
    assert!(result.success);
    assert_eq!(result.metrics.get("all_checks_passed"), Some(&1.0));
}

#[test]
fn test_cicd_recipe_one_fail() {
    let mut recipe = CiCdBenchmarkRecipe::new("one-fail");
    recipe.add_min_performance_threshold("performance", 0.9);

    // First passes
    recipe.add_result(point("pass", 0.95, 100.0, 1000.0));

    // Second fails
    recipe.add_result(point("fail", 0.85, 100.0, 1000.0));

    let result = recipe.check();
    assert!(!result.success);
}

#[test]
fn test_recipe_result_debug() {
    let result = RecipeResult::success("test").with_metric("acc", 0.9);
    let debug = format!("{:?}", result);
    assert!(debug.contains("RecipeResult"));
}

#[test]
fn test_experiment_tracking_config_with_tpu() {
    use crate::experiment::TpuVersion;
    let config = ExperimentTrackingConfig {
        device: ComputeDevice::Tpu { version: TpuVersion::V4, cores: 8 },
        ..Default::default()
    };
    let mut recipe = ExperimentTrackingRecipe::new(config);

    recipe.start_run("tpu-run");
    let result = recipe.end_run(true).unwrap();
    assert!(result.success);
}

// -------------------------------------------------------------------------
// Additional Coverage Tests
// -------------------------------------------------------------------------

#[test]
fn test_recipe_result_clone() {
    let result = RecipeResult::success("clone-test").with_artifact("art1").with_metric("m1", 1.0);
    let cloned = result.clone();
    assert_eq!(result.recipe_name, cloned.recipe_name);
    assert_eq!(result.artifacts, cloned.artifacts);
}

#[test]
fn test_experiment_tracking_config_serialize() {
    let config = ExperimentTrackingConfig::default();
    let json = serde_json::to_string(&config).unwrap();
    assert!(json.contains("experiment_name"));
    assert!(json.contains("default-experiment"));
}

#[test]
fn test_experiment_tracking_config_deserialize() {
    let json = r#"{"experiment_name":"test","paradigm":"TraditionalML","device":{"Cpu":{"cores":4,"threads_per_core":1,"architecture":"X86_64"}},"platform":"Server","track_energy":false,"track_cost":false,"carbon_intensity":null,"tags":[]}"#;
    let config: ExperimentTrackingConfig = serde_json::from_str(json).unwrap();
    assert_eq!(config.experiment_name, "test");
    assert!(!config.track_energy);
}

#[test]
fn test_experiment_tracking_config_clone() {
    let config = ExperimentTrackingConfig::default();
    let cloned = config.clone();
    assert_eq!(config.experiment_name, cloned.experiment_name);
}

#[test]
fn test_sovereign_deployment_config_serialize() {
    let config = SovereignDeploymentConfig::default();
    let json = serde_json::to_string(&config).unwrap();
    assert!(json.contains("sovereign-model"));
    assert!(json.contains("require_signatures"));
}

#[test]
fn test_sovereign_deployment_config_deserialize() {
    let json = r#"{"name":"test-model","version":"1.0.0","platforms":["linux"],"require_signatures":false,"enable_nix_flake":false,"offline_registry_path":null}"#;
    let config: SovereignDeploymentConfig = serde_json::from_str(json).unwrap();
    assert_eq!(config.name, "test-model");
    assert!(!config.require_signatures);
}

#[test]
fn test_sovereign_deployment_config_clone() {
    let config = SovereignDeploymentConfig::default();
    let cloned = config.clone();
    assert_eq!(config.name, cloned.name);
}

#[test]
fn test_experiment_tracking_recipe_debug() {
    let config = ExperimentTrackingConfig::default();
    let recipe = ExperimentTrackingRecipe::new(config);
    let debug_str = format!("{:?}", recipe);
    assert!(debug_str.contains("ExperimentTrackingRecipe"));
}

#[test]
fn test_sovereign_deployment_add_dataset() {
    let config = SovereignDeploymentConfig { require_signatures: false, ..Default::default() };
    let mut recipe = SovereignDeploymentRecipe::new(config);
    recipe.add_dataset("dataset.parquet", "sha256_hash", 5000000);

    assert_eq!(recipe.distribution().artifacts.len(), 1);
}

#[test]
fn test_sovereign_deployment_recipe_debug() {
    let config = SovereignDeploymentConfig::default();
    let recipe = SovereignDeploymentRecipe::new(config);
    let debug_str = format!("{:?}", recipe);
    assert!(debug_str.contains("SovereignDeploymentRecipe"));
}

#[test]
fn test_research_artifact_add_dataset() {
    let mut recipe = ResearchArtifactRecipe::new("Test", "Abstract");
    recipe.add_dataset("dataset1");
    recipe.add_dataset("dataset2");

    assert_eq!(recipe.artifact().datasets.len(), 2);
}

#[test]
fn test_research_artifact_add_repository() {
    let mut recipe = ResearchArtifactRecipe::new("Test", "Abstract");
    recipe.add_repository("https://github.com/user/repo");

    assert_eq!(recipe.artifact().code_repositories.len(), 1);
}

#[test]
fn test_research_artifact_recipe_debug() {
    let recipe = ResearchArtifactRecipe::new("Test", "Abstract");
    let debug_str = format!("{:?}", recipe);
    assert!(debug_str.contains("ResearchArtifactRecipe"));
}

#[test]
fn test_cost_performance_benchmark_recipe_debug() {
    let recipe = CostPerformanceBenchmarkRecipe::new("test");
    let debug_str = format!("{:?}", recipe);
    assert!(debug_str.contains("CostPerformanceBenchmarkRecipe"));
}

#[test]
fn test_cicd_benchmark_recipe_debug() {
    let recipe = CiCdBenchmarkRecipe::new("test");
    let debug_str = format!("{:?}", recipe);
    assert!(debug_str.contains("CiCdBenchmarkRecipe"));
}

#[test]
fn test_benchmark_recipe_add_run() {
    let config = ExperimentTrackingConfig::default();
    let mut tracking = ExperimentTrackingRecipe::new(config);
    tracking.start_run("run-for-benchmark");
    tracking.log_metric("accuracy", 0.92).unwrap();
    tracking.end_run(true).unwrap();

    let mut benchmark = CostPerformanceBenchmarkRecipe::new("bench");
    let run = tracking.current_run().unwrap();
    benchmark.add_run(run, "accuracy");

    assert_eq!(benchmark.benchmark().points.len(), 1);
}

#[test]
fn test_benchmark_best_within_budget() {
    let mut recipe = CostPerformanceBenchmarkRecipe::new("test").with_budget(100.0);

    recipe.benchmark_mut().add_point(point("cheap", 0.8, 50.0, 500.0));
    recipe.benchmark_mut().add_point(point("expensive", 0.95, 150.0, 1500.0));

    let result = recipe.analyze();
    // Should find cheap as best within budget
    assert!(result.metrics.contains_key("best_in_budget_performance"));
}

#[test]
fn test_benchmark_cheapest_meeting_target() {
    let mut recipe = CostPerformanceBenchmarkRecipe::new("test").with_performance_target(0.85);

    recipe.benchmark_mut().add_point(point("cheap", 0.9, 50.0, 500.0));
    recipe.benchmark_mut().add_point(point("cheaper", 0.88, 30.0, 300.0));
    recipe.benchmark_mut().add_point(point("expensive", 0.95, 150.0, 1500.0));

    let result = recipe.analyze();
    // Should find cheaper as cheapest meeting target
    assert!(result.metrics.contains_key("cheapest_meeting_target_cost"));
}

#[test]
fn test_sovereign_deployment_multiple_platforms() {
    let config = SovereignDeploymentConfig {
        platforms: vec![
            "linux-x86_64".to_string(),
            "darwin-aarch64".to_string(),
            "windows-x86_64".to_string(),
        ],
        require_signatures: false,
        ..Default::default()
    };
    let recipe = SovereignDeploymentRecipe::new(config);

    assert_eq!(recipe.distribution().platforms.len(), 3);
}

#[test]
fn test_recipe_result_failure_with_artifacts() {
    let result = RecipeResult::failure("test", "error occurred")
        .with_artifact("partial-output")
        .with_metric("progress", 0.5);

    assert!(!result.success);
    assert_eq!(result.artifacts.len(), 1);
    assert_eq!(result.metrics.get("progress"), Some(&0.5));
}

#[test]
fn test_research_artifact_citation_without_doi() {
    let mut recipe = ResearchArtifactRecipe::new("Test Paper", "Abstract");
    recipe.add_contributor("Alice", "MIT", vec![CreditRole::Software]);
    // No DOI set

    let citation = recipe.generate_citation();
    assert!(citation.doi.is_none());
}

#[test]
fn test_research_artifact_citation_with_repository() {
    let mut recipe = ResearchArtifactRecipe::new("Test Paper", "Abstract");
    recipe.add_contributor("Alice", "MIT", vec![CreditRole::Software]);
    recipe.add_repository("https://github.com/test/repo");

    let citation = recipe.generate_citation();
    assert_eq!(citation.url, Some("https://github.com/test/repo".to_string()));
}

#[test]
fn test_experiment_tracking_with_gpu() {
    use crate::experiment::GpuVendor;
    let config = ExperimentTrackingConfig {
        device: ComputeDevice::Gpu {
            name: "A100".to_string(),
            memory_gb: 80.0,
            compute_capability: Some("8.0".to_string()),
            vendor: GpuVendor::Nvidia,
        },
        ..Default::default()
    };
    let mut recipe = ExperimentTrackingRecipe::new(config);

    recipe.start_run("gpu-run");
    let result = recipe.end_run(true).unwrap();
    assert!(result.success);
    // Should have energy metrics since track_energy is true by default
    assert!(result.metrics.contains_key("energy_joules"));
}
