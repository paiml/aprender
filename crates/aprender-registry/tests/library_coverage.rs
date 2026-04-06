//! Additional tests to boost library coverage.

use pacha::prelude::*;

// Model module tests
#[test]
fn test_model_id_display_and_parse() {
    let id = ModelId::new();
    let s = id.to_string();
    let parsed: ModelId = s.parse().expect("parse");
    assert_eq!(id, parsed);
}

#[test]
fn test_model_reference_components() {
    let r = ModelReference::new("test", ModelVersion::new(1, 2, 3));
    assert_eq!(r.name, "test");
    assert_eq!(r.version, ModelVersion::new(1, 2, 3));
}

#[test]
fn test_model_version_prerelease() {
    let v = ModelVersion::new(1, 0, 0).with_prerelease("alpha.1");
    assert!(v.is_prerelease());
    assert!(!v.is_stable());
    assert!(v.to_string().contains("alpha.1"));
}

#[test]
fn test_model_version_build() {
    let v = ModelVersion::new(1, 0, 0).with_build("build.123");
    assert!(v.to_string().contains("+build.123"));
}

#[test]
fn test_model_stage_all_values() {
    let stages = [
        ModelStage::Development,
        ModelStage::Staging,
        ModelStage::Production,
        ModelStage::Archived,
    ];
    for stage in stages {
        let s = stage.to_string();
        let parsed: ModelStage = s.parse().expect("parse");
        assert_eq!(stage, parsed);
    }
}

// Dataset module tests
#[test]
fn test_dataset_id_display_and_parse() {
    let id = DatasetId::new();
    let s = id.to_string();
    let parsed: DatasetId = s.parse().expect("parse");
    assert_eq!(id, parsed);
}

#[test]
fn test_dataset_reference_components() {
    let r = DatasetReference::new("data", DatasetVersion::new(2, 1, 0));
    assert_eq!(r.name, "data");
    assert_eq!(r.version, DatasetVersion::new(2, 1, 0));
}

#[test]
fn test_dataset_version_bumps() {
    let v = DatasetVersion::new(1, 2, 3);
    assert_eq!(v.bump_major(), DatasetVersion::new(2, 0, 0));
    assert_eq!(v.bump_minor(), DatasetVersion::new(1, 3, 0));
    assert_eq!(v.bump_patch(), DatasetVersion::new(1, 2, 4));
}

// Recipe module tests
#[test]
fn test_recipe_id_display_and_parse() {
    let id = RecipeId::new();
    let s = id.to_string();
    let parsed: RecipeId = s.parse().expect("parse");
    assert_eq!(id, parsed);
}

#[test]
fn test_recipe_reference_display() {
    let r = RecipeReference::new("train", RecipeVersion::new(1, 0, 0));
    assert!(r.to_string().contains("train"));
    assert!(r.to_string().contains("1.0.0"));
}

#[test]
fn test_hyperparameters_custom() {
    let mut h = Hyperparameters::default();
    h.set_custom("dropout", HyperparamValue::Float(0.5));
    h.set_custom("layers", HyperparamValue::Int(4));
    h.set_custom("use_bn", HyperparamValue::Bool(true));

    assert_eq!(h.get_custom("dropout").and_then(|v| v.as_float()), Some(0.5));
    assert_eq!(h.get_custom("layers").and_then(|v| v.as_int()), Some(4));
    assert_eq!(h.get_custom("use_bn").and_then(|v| v.as_bool()), Some(true));
}

#[test]
fn test_hyperparam_value_list() {
    let list = HyperparamValue::List(vec![
        HyperparamValue::Int(1),
        HyperparamValue::Int(2),
        HyperparamValue::Int(3),
    ]);
    assert_eq!(list.as_list().map(|l| l.len()), Some(3));
}

#[test]
fn test_hyperparam_value_string() {
    let s = HyperparamValue::String("adam".to_string());
    assert_eq!(s.as_string(), Some("adam"));
}

// Experiment module tests
#[test]
fn test_run_id_from_uuid() {
    let uuid = uuid::Uuid::new_v4();
    let id = RunId::from_uuid(uuid);
    assert_eq!(id.as_uuid(), &uuid);
}

#[test]
fn test_experiment_run_with_git() {
    let mut run = ExperimentRun::new(Hyperparameters::default());
    run.git_commit = Some("abc123".to_string());
    run.git_dirty = true;

    assert_eq!(run.git_commit, Some("abc123".to_string()));
    assert!(run.git_dirty);
}

#[test]
fn test_experiment_run_with_hardware() {
    let mut run = ExperimentRun::new(Hyperparameters::default());
    run.hardware.cpu_model = Some("Intel Xeon".to_string());
    run.hardware.gpu_count = Some(4);

    assert!(run.hardware.cpu_model.is_some());
    assert_eq!(run.hardware.gpu_count, Some(4));
}

// Lineage module tests
#[test]
fn test_lineage_distilled_edge() {
    let edge = ModelLineageEdge::Distilled { teacher: ModelId::new(), temperature: 2.0 };
    let json = serde_json::to_string(&edge).expect("json");
    assert!(json.contains("distilled"));
}

#[test]
fn test_lineage_pruned_edge() {
    let edge = ModelLineageEdge::Pruned { source: ModelId::new(), sparsity: 0.5 };
    let json = serde_json::to_string(&edge).expect("json");
    assert!(json.contains("pruned"));
}

// Storage tests
#[test]
fn test_content_address_compression() {
    let addr = ContentAddress::from_bytes(b"test");
    assert_eq!(addr.compression(), Compression::None);

    let with_zstd = addr.with_compression(Compression::Zstd);
    assert_eq!(with_zstd.compression(), Compression::Zstd);
}

// Model card tests
#[test]
fn test_model_card_full() {
    let card = ModelCard::builder()
        .description("Test model")
        .metrics([("accuracy", 0.95)])
        .primary_uses(["Classification"])
        .limitations(["Not for production"])
        .ethical_considerations(["Bias testing needed"])
        .build();

    assert!(!card.primary_uses.is_empty());
    assert!(!card.limitations.is_empty());
    assert!(!card.ethical_considerations.is_empty());
}

// Datasheet tests
#[test]
fn test_datasheet_full() {
    let sheet = Datasheet::builder()
        .purpose("Training data")
        .creators(["Team A"])
        .collection_method("Web scraping")
        .instance_count(10000)
        .license("MIT")
        .build();

    assert_eq!(sheet.purpose, "Training data");
    assert!(!sheet.creators.is_empty());
    assert_eq!(sheet.instance_count, Some(10000));
}

#[test]
fn test_datasheet_builder_full() {
    let sheet = Datasheet::builder()
        .purpose("ML Training")
        .creators(["Alice", "Bob"])
        .collection_method("Automated collection")
        .instance_count(50000)
        .license("Apache-2.0")
        .build();

    assert_eq!(sheet.purpose, "ML Training");
    assert_eq!(sheet.creators.len(), 2);
    assert!(sheet.collection_method.is_some());
    assert_eq!(sheet.instance_count, Some(50000));
    assert!(sheet.license.is_some());
}

// Additional coverage tests
#[test]
fn test_model_card_all_builders() {
    let card = ModelCard::builder()
        .description("Test model")
        .metrics([("auc", 0.9)])
        .primary_uses(["Classification", "Detection"])
        .limitations(["Small datasets only"])
        .ethical_considerations(["Bias evaluation needed"])
        .build();

    assert!(!card.primary_uses.is_empty());
    assert!(!card.limitations.is_empty());
    assert!(!card.ethical_considerations.is_empty());
}

#[test]
fn test_experiment_run_fail() {
    let mut run = ExperimentRun::new(Hyperparameters::default());
    run.start();
    run.fail("Out of memory");
    assert_eq!(run.status, RunStatus::Failed);
    assert!(run.error_message.is_some());
}

#[test]
fn test_experiment_run_cancel() {
    let mut run = ExperimentRun::new(Hyperparameters::default());
    run.start();
    run.cancel();
    assert_eq!(run.status, RunStatus::Cancelled);
}

#[test]
fn test_run_status_display() {
    assert_eq!(RunStatus::Pending.to_string(), "pending");
    assert_eq!(RunStatus::Running.to_string(), "running");
    assert_eq!(RunStatus::Completed.to_string(), "completed");
    assert_eq!(RunStatus::Failed.to_string(), "failed");
    assert_eq!(RunStatus::Cancelled.to_string(), "cancelled");
}

#[test]
fn test_model_stage_invalid() {
    let result: std::result::Result<ModelStage, _> = "invalid".parse();
    assert!(result.is_err());
}

#[test]
fn test_version_parse_invalid() {
    let result: std::result::Result<ModelVersion, _> = "not-a-version".parse();
    assert!(result.is_err());

    let result: std::result::Result<DatasetVersion, _> = "bad".parse();
    assert!(result.is_err());

    let result: std::result::Result<RecipeVersion, _> = "x.y.z".parse();
    assert!(result.is_err());
}

#[test]
fn test_hyperparameters_with_options() {
    let h = Hyperparameters::builder().learning_rate(0.01).batch_size(64).epochs(20).build();

    assert!((h.learning_rate - 0.01).abs() < 1e-10);
    assert_eq!(h.batch_size, 64);
    assert_eq!(h.epochs, 20);
}

// More coverage for model/mod.rs
#[test]
fn test_model_reference_new_and_display() {
    let r = ModelReference::new("test-model", ModelVersion::new(2, 3, 4));
    let display = r.to_string();
    assert!(display.contains("test-model"));
    assert!(display.contains("2.3.4"));
}

// More coverage for data/mod.rs
#[test]
fn test_dataset_reference_new_and_display() {
    let r = DatasetReference::new("test-data", DatasetVersion::new(1, 2, 3));
    let display = r.to_string();
    assert!(display.contains("test-data"));
    assert!(display.contains("1.2.3"));
}

// More coverage for recipe/mod.rs
#[test]
fn test_recipe_reference_new_and_display() {
    let r = RecipeReference::new("test-recipe", RecipeVersion::new(3, 2, 1));
    let display = r.to_string();
    assert!(display.contains("test-recipe"));
    assert!(display.contains("3.2.1"));
}

#[test]
fn test_recipe_full_builder() {
    let recipe = TrainingRecipe::builder()
        .name("full-recipe")
        .version(RecipeVersion::new(1, 0, 0))
        .description("Full recipe test")
        .architecture("transformer")
        .hyperparameters(
            Hyperparameters::builder().learning_rate(1e-4).batch_size(16).epochs(100).build(),
        )
        .random_seed(42)
        .deterministic(true)
        .build();

    assert_eq!(recipe.name, "full-recipe");
    assert!(recipe.architecture.is_some());
    assert!(recipe.deterministic);
}

#[test]
fn test_model_id_new_and_string() {
    let id1 = ModelId::new();
    let id2 = ModelId::new();
    assert_ne!(id1, id2);

    let s = id1.to_string();
    assert!(!s.is_empty());
}

#[test]
fn test_dataset_id_new_and_string() {
    let id1 = DatasetId::new();
    let id2 = DatasetId::new();
    assert_ne!(id1, id2);

    let s = id1.to_string();
    assert!(!s.is_empty());
}

#[test]
fn test_recipe_id_new_and_string() {
    let id1 = RecipeId::new();
    let id2 = RecipeId::new();
    assert_ne!(id1, id2);

    let s = id1.to_string();
    assert!(!s.is_empty());
}

// More data/mod.rs coverage
#[test]
fn test_dataset_id_default() {
    let id: pacha::data::DatasetId = Default::default();
    assert!(!id.to_string().is_empty());
}

#[test]
fn test_dataset_id_from_uuid() {
    let uuid = uuid::Uuid::new_v4();
    let id = pacha::data::DatasetId::from_uuid(uuid);
    assert_eq!(id.as_uuid(), &uuid);
}

// More model/mod.rs coverage
#[test]
fn test_model_id_default() {
    let id: ModelId = Default::default();
    assert!(!id.to_string().is_empty());
}

#[test]
fn test_model_id_from_uuid() {
    let uuid = uuid::Uuid::new_v4();
    let id = pacha::model::ModelId::from_uuid(uuid);
    assert_eq!(id.as_uuid(), &uuid);
}

// More recipe/mod.rs coverage
#[test]
fn test_recipe_id_default() {
    let id: RecipeId = Default::default();
    assert!(!id.to_string().is_empty());
}

#[test]
fn test_recipe_id_from_uuid() {
    let uuid = uuid::Uuid::new_v4();
    let id = pacha::recipe::RecipeId::from_uuid(uuid);
    assert_eq!(id.as_uuid(), &uuid);
}

#[test]
fn test_training_recipe_reference() {
    let recipe = TrainingRecipe::builder()
        .name("ref-recipe")
        .version(RecipeVersion::new(1, 0, 0))
        .description("Test")
        .hyperparameters(Hyperparameters::default())
        .build();

    let reference = recipe.reference();
    assert_eq!(reference.name, "ref-recipe");
    assert_eq!(reference.version, RecipeVersion::new(1, 0, 0));
}

// More experiment/mod.rs coverage
#[test]
fn test_experiment_run_duration() {
    let mut run = ExperimentRun::new(Hyperparameters::default());
    run.start();
    std::thread::sleep(std::time::Duration::from_millis(10));
    run.complete();

    let duration = run.duration_secs();
    assert!(duration.is_some());
}

// More version coverage
#[test]
fn test_model_version_with_both() {
    let v = ModelVersion::new(1, 0, 0).with_prerelease("rc.1").with_build("build.456");

    let s = v.to_string();
    assert!(s.contains("rc.1"));
    assert!(s.contains("build.456"));
}

#[test]
fn test_dataset_version_ordering() {
    let v1 = DatasetVersion::new(1, 0, 0);
    let v2 = DatasetVersion::new(1, 1, 0);
    let v3 = DatasetVersion::new(2, 0, 0);

    assert!(v1 < v2);
    assert!(v2 < v3);
}

#[test]
fn test_recipe_version_ordering() {
    let v1 = RecipeVersion::new(1, 0, 0);
    let v2 = RecipeVersion::new(1, 1, 0);
    let v3 = RecipeVersion::new(2, 0, 0);

    assert!(v1 < v2);
    assert!(v2 < v3);
}

// Registry coverage
use pacha::{Registry, RegistryConfig};
use tempfile::TempDir;

#[test]
fn test_registry_list_model_versions() {
    let dir = TempDir::new().unwrap();
    let registry = Registry::open(RegistryConfig::new(dir.path())).unwrap();

    let card = ModelCard::new("Test");
    for i in 0..3 {
        registry
            .register_model("ver-model", &ModelVersion::new(1, i, 0), &[i as u8], card.clone())
            .unwrap();
    }

    let versions = registry.list_model_versions("ver-model").unwrap();
    assert_eq!(versions.len(), 3);
}

#[test]
fn test_registry_model_not_found() {
    let dir = TempDir::new().unwrap();
    let registry = Registry::open(RegistryConfig::new(dir.path())).unwrap();

    let result = registry.get_model("nonexistent", &ModelVersion::new(1, 0, 0));
    assert!(result.is_err());
}

#[test]
fn test_registry_dataset_not_found() {
    let dir = TempDir::new().unwrap();
    let registry = Registry::open(RegistryConfig::new(dir.path())).unwrap();

    let result = registry.get_dataset("nonexistent", &DatasetVersion::new(1, 0, 0));
    assert!(result.is_err());
}

#[test]
fn test_registry_recipe_not_found() {
    let dir = TempDir::new().unwrap();
    let registry = Registry::open(RegistryConfig::new(dir.path())).unwrap();

    let result = registry.get_recipe("nonexistent", &RecipeVersion::new(1, 0, 0));
    assert!(result.is_err());
}

#[test]
fn test_registry_run_not_found() {
    let dir = TempDir::new().unwrap();
    let registry = Registry::open(RegistryConfig::new(dir.path())).unwrap();

    let run_id = pacha::experiment::RunId::new();
    let result = registry.get_run(&run_id);
    assert!(result.is_err());
}

#[test]
fn test_registry_duplicate_model() {
    let dir = TempDir::new().unwrap();
    let registry = Registry::open(RegistryConfig::new(dir.path())).unwrap();

    let card = ModelCard::new("Test");
    registry.register_model("dup", &ModelVersion::new(1, 0, 0), b"data", card.clone()).unwrap();

    let result = registry.register_model("dup", &ModelVersion::new(1, 0, 0), b"data2", card);
    assert!(result.is_err());
}

#[test]
fn test_content_address_verify_fail() {
    let addr = ContentAddress::from_bytes(b"original");
    assert!(!addr.verify(b"different"));
}

#[test]
fn test_lineage_graph_operations() {
    let mut graph = LineageGraph::new();

    let node1 = graph.add_node(pacha::lineage::LineageNode {
        model_id: ModelId::new(),
        model_name: "base".to_string(),
        model_version: "1.0.0".to_string(),
    });

    let node2 = graph.add_node(pacha::lineage::LineageNode {
        model_id: ModelId::new(),
        model_name: "derived".to_string(),
        model_version: "1.0.0".to_string(),
    });

    graph.add_edge(
        node1,
        node2,
        ModelLineageEdge::FineTuned { parent: ModelId::new(), recipe: RecipeId::new() },
    );

    assert_eq!(graph.node_count(), 2);
    assert_eq!(graph.edge_count(), 1);

    let ancestors = graph.ancestors(node2);
    assert!(!ancestors.is_empty());

    let descendants = graph.descendants(node1);
    assert!(!descendants.is_empty());
}

#[test]
fn test_error_display() {
    use pacha::PachaError;

    let err = PachaError::NotFound {
        kind: "model".to_string(),
        name: "test".to_string(),
        version: "1.0.0".to_string(),
    };
    assert!(err.to_string().contains("model"));

    let err = PachaError::Validation("bad input".into());
    assert!(err.to_string().contains("bad input"));
}

#[test]
fn test_storage_stats() {
    let dir = TempDir::new().unwrap();
    let registry = Registry::open(RegistryConfig::new(dir.path())).unwrap();

    let stats = registry.storage_stats().unwrap();
    assert_eq!(stats.model_count, 0);
    assert_eq!(stats.dataset_count, 0);
    assert_eq!(stats.recipe_count, 0);
}

#[test]
fn test_model_card_add_metric() {
    let mut card = ModelCard::new("Test");
    card.add_metric("accuracy", 0.95);
    card.add_metric("f1", 0.88);

    assert_eq!(card.metrics.len(), 2);
    assert_eq!(card.metrics.get("accuracy"), Some(&0.95));
}

#[test]
fn test_datasheet_new_and_purpose() {
    let sheet = Datasheet::new("Training data for ML");
    assert_eq!(sheet.purpose, "Training data for ML");
}

#[test]
fn test_hyperparameters_default() {
    let h = Hyperparameters::default();
    assert!((h.learning_rate - 0.001).abs() < 1e-9);
    assert_eq!(h.batch_size, 32);
    assert_eq!(h.epochs, 10);
}

#[test]
fn test_run_status_values() {
    // RunStatus doesn't implement FromStr, just test the enum values
    assert_eq!(RunStatus::Pending.to_string(), "pending");
    assert_eq!(RunStatus::Running.to_string(), "running");
}

#[test]
fn test_experiment_run_get_metric_none() {
    let run = ExperimentRun::new(Hyperparameters::default());
    assert!(run.get_metric("nonexistent").is_none());
}

#[test]
fn test_registry_open_default() {
    // Just test that default config works
    let config = RegistryConfig::default();
    assert!(config.base_path.to_str().unwrap().contains(".pacha"));
}

#[test]
fn test_provenance_variants() {
    use chrono::Utc;

    let derived = pacha::data::ProvenanceRecord::WasDerivedFrom {
        derived: pacha::data::DatasetId::new(),
        source: pacha::data::DatasetId::new(),
        transformation: "normalize".to_string(),
    };
    let json = serde_json::to_string(&derived).unwrap();
    assert!(json.contains("was_derived_from"));

    let generated = pacha::data::ProvenanceRecord::WasGeneratedBy {
        data: pacha::data::DatasetId::new(),
        activity: "training".to_string(),
        timestamp: Utc::now(),
    };
    let json = serde_json::to_string(&generated).unwrap();
    assert!(json.contains("was_generated_by"));

    let used = pacha::data::ProvenanceRecord::Used {
        activity: "inference".to_string(),
        data: pacha::data::DatasetId::new(),
    };
    let json = serde_json::to_string(&used).unwrap();
    assert!(json.contains("used"));

    let attributed = pacha::data::ProvenanceRecord::WasAttributedTo {
        entity: pacha::data::DatasetId::new(),
        agent: "data-team".to_string(),
    };
    let json = serde_json::to_string(&attributed).unwrap();
    assert!(json.contains("was_attributed_to"));
}

#[test]
fn test_quantization_types() {
    let types = [
        QuantizationType::Int8,
        QuantizationType::Int4,
        QuantizationType::Fp16,
        QuantizationType::Bf16,
        QuantizationType::Dynamic,
    ];
    for qt in types {
        let s = qt.to_string();
        assert!(!s.is_empty());
    }
}

#[test]
fn test_model_card_builder_with_add() {
    let mut card = ModelCard::builder().description("Test").build();

    card.add_primary_use("classification");
    card.add_limitation("small data only");

    assert!(!card.primary_uses.is_empty());
    assert!(!card.limitations.is_empty());
}

#[test]
fn test_object_store_operations() {
    let dir = TempDir::new().unwrap();
    let store = pacha::storage::ObjectStore::new(dir.path().join("objects")).unwrap();

    let addr = store.put(b"test data").unwrap();
    assert!(store.exists(&addr));

    let data = store.get(&addr).unwrap();
    assert_eq!(data, b"test data");

    let size = store.total_size().unwrap();
    assert!(size > 0);
}

#[test]
fn test_model_version_bump_methods() {
    let v = ModelVersion::new(1, 2, 3);
    assert_eq!(v.bump_major(), ModelVersion::new(2, 0, 0));
    assert_eq!(v.bump_minor(), ModelVersion::new(1, 3, 0));
    assert_eq!(v.bump_patch(), ModelVersion::new(1, 2, 4));
}

#[test]
fn test_recipe_version_bump_methods() {
    let v = RecipeVersion::new(2, 3, 4);
    assert_eq!(v.bump_major(), RecipeVersion::new(3, 0, 0));
    assert_eq!(v.bump_minor(), RecipeVersion::new(2, 4, 0));
    assert_eq!(v.bump_patch(), RecipeVersion::new(2, 3, 5));
}

#[test]
fn test_model_stage_transitions() {
    // Valid transitions
    assert!(ModelStage::Development.can_transition_to(ModelStage::Staging));
    assert!(ModelStage::Staging.can_transition_to(ModelStage::Production));
    assert!(ModelStage::Production.can_transition_to(ModelStage::Archived));

    // Invalid transitions
    assert!(!ModelStage::Development.can_transition_to(ModelStage::Production));
}

#[test]
fn test_registry_list_all() {
    let dir = TempDir::new().unwrap();
    let registry = Registry::open(RegistryConfig::new(dir.path())).unwrap();

    // Add some models
    let card = ModelCard::new("Test");
    registry.register_model("m1", &ModelVersion::new(1, 0, 0), b"a", card.clone()).unwrap();
    registry.register_model("m2", &ModelVersion::new(1, 0, 0), b"b", card.clone()).unwrap();

    let models = registry.list_models().unwrap();
    assert_eq!(models.len(), 2);

    // Add datasets
    let sheet = Datasheet::new("Test");
    registry.register_dataset("d1", &DatasetVersion::new(1, 0, 0), b"x", sheet.clone()).unwrap();

    let datasets = registry.list_datasets().unwrap();
    assert_eq!(datasets.len(), 1);

    // Add recipes
    let recipe = TrainingRecipe::builder()
        .name("r1")
        .version(RecipeVersion::new(1, 0, 0))
        .description("Test")
        .hyperparameters(Hyperparameters::default())
        .build();
    registry.register_recipe(&recipe).unwrap();

    let recipes = registry.list_recipes().unwrap();
    assert_eq!(recipes.len(), 1);
}

#[test]
fn test_registry_list_dataset_versions() {
    let dir = TempDir::new().unwrap();
    let registry = Registry::open(RegistryConfig::new(dir.path())).unwrap();

    let sheet = Datasheet::new("Test");
    registry
        .register_dataset("test-ds", &DatasetVersion::new(1, 0, 0), b"v1", sheet.clone())
        .unwrap();
    registry
        .register_dataset("test-ds", &DatasetVersion::new(1, 1, 0), b"v2", sheet.clone())
        .unwrap();
    registry
        .register_dataset("test-ds", &DatasetVersion::new(2, 0, 0), b"v3", sheet.clone())
        .unwrap();

    let versions = registry.list_dataset_versions("test-ds").unwrap();
    assert_eq!(versions.len(), 3);
    assert_eq!(versions[0], DatasetVersion::new(1, 0, 0));
    assert_eq!(versions[1], DatasetVersion::new(1, 1, 0));
    assert_eq!(versions[2], DatasetVersion::new(2, 0, 0));

    // Non-existent dataset
    let empty = registry.list_dataset_versions("nonexistent").unwrap();
    assert!(empty.is_empty());
}

#[test]
fn test_registry_list_recipe_versions() {
    let dir = TempDir::new().unwrap();
    let registry = Registry::open(RegistryConfig::new(dir.path())).unwrap();

    let recipe1 = TrainingRecipe::builder()
        .name("test-recipe")
        .version(RecipeVersion::new(1, 0, 0))
        .description("v1")
        .hyperparameters(Hyperparameters::default())
        .build();
    registry.register_recipe(&recipe1).unwrap();

    let recipe2 = TrainingRecipe::builder()
        .name("test-recipe")
        .version(RecipeVersion::new(1, 1, 0))
        .description("v2")
        .hyperparameters(Hyperparameters::default())
        .build();
    registry.register_recipe(&recipe2).unwrap();

    let versions = registry.list_recipe_versions("test-recipe").unwrap();
    assert_eq!(versions.len(), 2);
    assert_eq!(versions[0], RecipeVersion::new(1, 0, 0));
    assert_eq!(versions[1], RecipeVersion::new(1, 1, 0));

    // Non-existent recipe
    let empty = registry.list_recipe_versions("nonexistent").unwrap();
    assert!(empty.is_empty());
}

#[test]
fn test_content_address_from_reader() {
    let data = b"test content";
    let cursor = std::io::Cursor::new(data);
    let addr = ContentAddress::from_reader(cursor).unwrap();

    assert_eq!(addr.size(), data.len() as u64);
    assert!(addr.verify(data));
}

#[test]
fn test_lineage_find_node() {
    let mut graph = LineageGraph::new();
    let model_id = ModelId::new();

    let idx = graph.add_node(pacha::lineage::LineageNode {
        model_id: model_id.clone(),
        model_name: "test".to_string(),
        model_version: "1.0.0".to_string(),
    });

    let found = graph.find_node(&model_id);
    assert_eq!(found, Some(idx));

    let not_found = graph.find_node(&ModelId::new());
    assert!(not_found.is_none());
}

#[test]
fn test_registry_update_run() {
    let dir = TempDir::new().unwrap();
    let registry = Registry::open(RegistryConfig::new(dir.path())).unwrap();

    let recipe = TrainingRecipe::builder()
        .name("upd-recipe")
        .version(RecipeVersion::new(1, 0, 0))
        .description("Test")
        .hyperparameters(Hyperparameters::default())
        .build();
    registry.register_recipe(&recipe).unwrap();

    let run = ExperimentRun::from_recipe(recipe.reference(), Hyperparameters::default());
    let run_id = registry.start_run(run).unwrap();

    let mut run = registry.get_run(&run_id).unwrap();
    run.log_metric("loss", 0.5, 0);
    run.complete();
    registry.update_run(&run).unwrap();

    let updated = registry.get_run(&run_id).unwrap();
    assert_eq!(updated.status, RunStatus::Completed);
}

#[test]
fn test_registry_list_runs() {
    let dir = TempDir::new().unwrap();
    let registry = Registry::open(RegistryConfig::new(dir.path())).unwrap();

    let recipe = TrainingRecipe::builder()
        .name("runs-recipe")
        .version(RecipeVersion::new(1, 0, 0))
        .description("Test")
        .hyperparameters(Hyperparameters::default())
        .build();
    registry.register_recipe(&recipe).unwrap();

    for _ in 0..3 {
        let run = ExperimentRun::from_recipe(recipe.reference(), Hyperparameters::default());
        registry.start_run(run).unwrap();
    }

    let runs = registry.list_runs(&recipe.reference()).unwrap();
    assert_eq!(runs.len(), 3);
}

#[test]
fn test_registry_get_model_lineage() {
    let dir = TempDir::new().unwrap();
    let registry = Registry::open(RegistryConfig::new(dir.path())).unwrap();

    let card = ModelCard::new("Test");
    registry.register_model("lin-model", &ModelVersion::new(1, 0, 0), b"data", card).unwrap();

    let model = registry.get_model("lin-model", &ModelVersion::new(1, 0, 0)).unwrap();
    let lineage = registry.get_model_lineage(&model.id).unwrap();

    assert_eq!(lineage.node_count(), 0);
}

#[test]
fn test_model_stage_all_transitions() {
    // Test all valid stage transitions
    let stages = [
        ModelStage::Development,
        ModelStage::Staging,
        ModelStage::Production,
        ModelStage::Archived,
    ];

    for from in &stages {
        for to in &stages {
            let can = from.can_transition_to(*to);
            // Just make sure the method doesn't panic
            let _ = can;
        }
    }
}

#[test]
fn test_model_stage_is_active_and_mutable() {
    // Only Development is mutable
    assert!(ModelStage::Development.is_mutable());
    assert!(!ModelStage::Staging.is_mutable());
    assert!(!ModelStage::Production.is_mutable());
    assert!(!ModelStage::Archived.is_mutable());

    // All except Archived are active
    assert!(ModelStage::Development.is_active());
    assert!(ModelStage::Staging.is_active());
    assert!(ModelStage::Production.is_active());
    assert!(!ModelStage::Archived.is_active());
}

// Version initial() and Default coverage
#[test]
fn test_dataset_version_initial_and_default() {
    let initial = DatasetVersion::initial();
    assert_eq!(initial, DatasetVersion::new(1, 0, 0));

    let default: DatasetVersion = Default::default();
    assert_eq!(default, DatasetVersion::new(1, 0, 0));
}

#[test]
fn test_dataset_version_parse() {
    let v = DatasetVersion::parse("2.3.4").expect("parse");
    assert_eq!(v, DatasetVersion::new(2, 3, 4));

    let err = DatasetVersion::parse("invalid");
    assert!(err.is_err());
}

#[test]
fn test_recipe_version_initial_and_default() {
    let initial = RecipeVersion::initial();
    assert_eq!(initial, RecipeVersion::new(1, 0, 0));

    let default: RecipeVersion = Default::default();
    assert_eq!(default, RecipeVersion::new(1, 0, 0));
}

#[test]
fn test_model_version_initial_and_default() {
    let initial = ModelVersion::initial();
    assert_eq!(initial, ModelVersion::new(1, 0, 0));

    let default: ModelVersion = Default::default();
    assert_eq!(default, ModelVersion::new(1, 0, 0));
}

#[test]
fn test_model_version_parse() {
    let v = ModelVersion::parse("4.5.6").expect("parse");
    assert_eq!(v, ModelVersion::new(4, 5, 6));

    let err = ModelVersion::parse("x.y.z");
    assert!(err.is_err());
}

// ModelCard builder coverage
#[test]
fn test_model_card_builder_training_date() {
    use chrono::Utc;
    let now = Utc::now();
    let card = ModelCard::builder().description("Test").training_date(now).build();
    assert!(card.training_date.is_some());
}

#[test]
fn test_model_card_builder_training_duration() {
    use chrono::Duration;
    let card =
        ModelCard::builder().description("Test").training_duration(Duration::hours(2)).build();
    assert_eq!(card.training_duration_secs, Some(7200));
}

#[test]
fn test_model_card_builder_evaluation_data() {
    let eval_ref = DatasetReference::new("eval-data", DatasetVersion::new(1, 0, 0));
    let card = ModelCard::builder().description("Test").evaluation_data(eval_ref).build();
    assert!(card.evaluation_data.is_some());
}

#[test]
fn test_model_card_builder_out_of_scope() {
    let card = ModelCard::builder()
        .description("Test")
        .out_of_scope_uses(["Medical diagnosis", "Legal decisions"])
        .build();
    assert_eq!(card.out_of_scope_uses.len(), 2);
}

// ExperimentRun coverage
#[test]
fn test_experiment_run_new_status() {
    let run = ExperimentRun::new(Hyperparameters::default());
    assert_eq!(run.status, RunStatus::Pending);
}

// Datasheet builder additional methods
#[test]
fn test_datasheet_builder_additional() {
    let sheet = Datasheet::builder()
        .purpose("Test")
        .creators(["Team A", "Team B"])
        .collection_method("manual")
        .build();
    assert_eq!(sheet.creators.len(), 2);
}

// Hyperparameters type conversions
#[test]
fn test_hyperparam_value_type_conversions() {
    // Float converts to int by casting
    let float_val = HyperparamValue::Float(1.5);
    assert_eq!(float_val.as_int(), Some(1)); // truncates
    assert!(float_val.as_bool().is_none());
    assert!(float_val.as_string().is_none());
    assert!(float_val.as_list().is_none());

    // Int converts to float
    let int_val = HyperparamValue::Int(42);
    assert_eq!(int_val.as_float(), Some(42.0));
    assert!(int_val.as_bool().is_none());

    // Bool doesn't convert to numeric
    let bool_val = HyperparamValue::Bool(true);
    assert!(bool_val.as_float().is_none());
    assert!(bool_val.as_int().is_none());

    // String doesn't convert
    let str_val = HyperparamValue::String("test".to_string());
    assert!(str_val.as_float().is_none());
    assert!(str_val.as_int().is_none());

    // List doesn't convert
    let list_val = HyperparamValue::List(vec![]);
    assert!(list_val.as_float().is_none());
    assert!(list_val.as_int().is_none());
}

// Additional hyperparameters builder methods
#[test]
fn test_hyperparameters_builder_additional() {
    let h = Hyperparameters::builder()
        .learning_rate(0.01)
        .batch_size(32)
        .epochs(10)
        .weight_decay(0.001)
        .max_grad_norm(1.0)
        .warmup_steps(100)
        .custom("dropout", HyperparamValue::Float(0.5))
        .build();

    assert!((h.weight_decay - 0.001).abs() < 1e-9);
    assert_eq!(h.max_grad_norm, Some(1.0));
    assert_eq!(h.warmup_steps, Some(100));
    assert!(h.custom.contains_key("dropout"));
}

// Recipe builder additional methods
#[test]
fn test_recipe_builder_train_and_validation() {
    let train_ref = DatasetReference::new("train-data", DatasetVersion::new(1, 0, 0));
    let validation_ref = DatasetReference::new("val-data", DatasetVersion::new(1, 0, 0));

    let recipe = TrainingRecipe::builder()
        .name("full-recipe")
        .version(RecipeVersion::new(1, 0, 0))
        .description("Test")
        .hyperparameters(Hyperparameters::default())
        .train_data(train_ref)
        .validation_data(validation_ref)
        .build();

    assert!(recipe.train_data.is_some());
    assert!(recipe.validation_data.is_some());
}

// Lineage merge edge
#[test]
fn test_lineage_merged_edge() {
    let edge = ModelLineageEdge::Merged {
        sources: vec![ModelId::new(), ModelId::new()],
        weights: vec![0.5_f32, 0.5_f32],
    };
    let json = serde_json::to_string(&edge).expect("json");
    assert!(json.contains("merged"));
}

// Lineage quantized edge
#[test]
fn test_lineage_quantized_edge() {
    let edge = ModelLineageEdge::Quantized {
        source: ModelId::new(),
        quantization: QuantizationType::Int8,
    };
    let json = serde_json::to_string(&edge).expect("json");
    assert!(json.contains("quantized"));
}
