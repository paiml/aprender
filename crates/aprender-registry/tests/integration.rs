//! Integration tests for Pacha registry.

use pacha::prelude::*;
use tempfile::TempDir;

fn setup() -> (TempDir, Registry) {
    let dir = TempDir::new().expect("temp dir");
    let config = RegistryConfig::new(dir.path());
    let registry = Registry::open(config).expect("registry");
    (dir, registry)
}

#[test]
fn test_full_model_workflow() {
    let (_dir, registry) = setup();

    // Register model
    let card = ModelCard::builder().description("Test model").metrics([("accuracy", 0.95)]).build();

    let id = registry
        .register_model("test-model", &ModelVersion::new(1, 0, 0), b"weights", card)
        .expect("register");

    assert!(!id.to_string().is_empty());

    // Get model
    let model = registry.get_model("test-model", &ModelVersion::new(1, 0, 0)).expect("get");
    assert_eq!(model.name, "test-model");
    assert_eq!(model.stage, ModelStage::Development);

    // Transition stages
    registry
        .transition_model_stage("test-model", &ModelVersion::new(1, 0, 0), ModelStage::Staging)
        .expect("stage");

    let model = registry.get_model("test-model", &ModelVersion::new(1, 0, 0)).expect("get");
    assert_eq!(model.stage, ModelStage::Staging);

    // Download artifact
    let artifact =
        registry.get_model_artifact("test-model", &ModelVersion::new(1, 0, 0)).expect("artifact");
    assert_eq!(artifact, b"weights");
}

#[test]
fn test_full_dataset_workflow() {
    let (_dir, registry) = setup();

    let datasheet = Datasheet::builder().purpose("Test data").instance_count(1000).build();

    let id = registry
        .register_dataset("test-data", &DatasetVersion::new(1, 0, 0), b"csv,data", datasheet)
        .expect("register");

    assert!(!id.to_string().is_empty());

    let dataset = registry.get_dataset("test-data", &DatasetVersion::new(1, 0, 0)).expect("get");
    assert_eq!(dataset.name, "test-data");

    let data = registry.get_dataset_data("test-data", &DatasetVersion::new(1, 0, 0)).expect("data");
    assert_eq!(data, b"csv,data");
}

#[test]
fn test_full_recipe_workflow() {
    let (_dir, registry) = setup();

    let recipe = TrainingRecipe::builder()
        .name("test-recipe")
        .version(RecipeVersion::new(1, 0, 0))
        .description("Test training recipe")
        .hyperparameters(
            Hyperparameters::builder().learning_rate(0.001).batch_size(32).epochs(10).build(),
        )
        .build();

    let id = registry.register_recipe(&recipe).expect("register");
    assert!(!id.to_string().is_empty());

    let retrieved = registry.get_recipe("test-recipe", &RecipeVersion::new(1, 0, 0)).expect("get");
    assert_eq!(retrieved.name, "test-recipe");
    assert!((retrieved.hyperparameters.learning_rate - 0.001).abs() < 1e-9);
}

#[test]
fn test_experiment_run_workflow() {
    let (_dir, registry) = setup();

    // Register recipe first
    let recipe = TrainingRecipe::builder()
        .name("exp-recipe")
        .version(RecipeVersion::new(1, 0, 0))
        .description("Experiment recipe")
        .hyperparameters(Hyperparameters::default())
        .build();

    registry.register_recipe(&recipe).expect("register recipe");

    // Create run
    let mut run = ExperimentRun::from_recipe(recipe.reference(), Hyperparameters::default());
    run.log_metric("loss", 0.5, 0);
    run.log_metric("loss", 0.3, 100);
    run.log_metric("accuracy", 0.9, 100);

    let run_id = registry.start_run(run).expect("start run");

    // Get and update run
    let mut run = registry.get_run(&run_id).expect("get run");
    assert_eq!(run.status, RunStatus::Running);
    run.complete();
    registry.update_run(&run).expect("update run");

    let retrieved = registry.get_run(&run_id).expect("get run");
    assert_eq!(retrieved.status, RunStatus::Completed);
    assert_eq!(retrieved.get_metric("loss"), Some(0.3));
    assert_eq!(retrieved.get_metric("accuracy"), Some(0.9));
}

#[test]
fn test_multiple_versions() {
    let (_dir, registry) = setup();

    for i in 0..3 {
        let card = ModelCard::new(format!("Version {i}"));
        registry
            .register_model(
                "versioned-model",
                &ModelVersion::new(1, i, 0),
                format!("v1.{i}.0").as_bytes(),
                card,
            )
            .expect("register");
    }

    let versions = registry.list_model_versions("versioned-model").expect("list");
    assert_eq!(versions.len(), 3);
}

#[test]
fn test_content_deduplication() {
    let (_dir, registry) = setup();

    let data = b"same content for both";

    let card1 = ModelCard::new("Model 1");
    registry
        .register_model("model-1", &ModelVersion::new(1, 0, 0), data, card1)
        .expect("register 1");

    let card2 = ModelCard::new("Model 2");
    registry
        .register_model("model-2", &ModelVersion::new(1, 0, 0), data, card2)
        .expect("register 2");

    let stats = registry.storage_stats().expect("stats");
    // Same content should be deduplicated - only 1 object
    assert_eq!(stats.object_count, 1);
}
