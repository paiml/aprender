#[cfg(test)]
mod tests {
    use super::super::*;
    use crate::data::DatasetVersion;
    use crate::recipe::{Hyperparameters, RecipeVersion};
    use tempfile::TempDir;

    fn setup() -> (TempDir, Registry) {
        let dir = TempDir::new().unwrap();
        let config = RegistryConfig::new(dir.path());
        let registry = Registry::open(config).unwrap();
        (dir, registry)
    }

    #[test]
    fn test_registry_open() {
        let (_dir, registry) = setup();
        assert!(registry.config.base_path.exists());
    }

    #[test]
    fn test_register_and_get_model() {
        let (_dir, registry) = setup();

        let name = "test-model";
        let version = ModelVersion::new(1, 0, 0);
        let artifact = b"model data";
        let card = ModelCard::new("Test model");

        let id = registry.register_model(name, &version, artifact, card.clone()).unwrap();

        let model = registry.get_model(name, &version).unwrap();
        assert_eq!(model.id, id);
        assert_eq!(model.name, name);
        assert_eq!(model.version, version);
        assert_eq!(model.card.description, card.description);
    }

    #[test]
    fn test_register_duplicate_model_fails() {
        let (_dir, registry) = setup();

        let name = "test-model";
        let version = ModelVersion::new(1, 0, 0);
        let artifact = b"model data";
        let card = ModelCard::new("Test model");

        registry.register_model(name, &version, artifact, card.clone()).unwrap();

        let result = registry.register_model(name, &version, artifact, card);
        assert!(matches!(result, Err(crate::error::PachaError::AlreadyExists { .. })));
    }

    #[test]
    fn test_model_artifact_roundtrip() {
        let (_dir, registry) = setup();

        let name = "test-model";
        let version = ModelVersion::new(1, 0, 0);
        let artifact = b"model binary data here";
        let card = ModelCard::new("Test");

        registry.register_model(name, &version, artifact, card).unwrap();

        let retrieved = registry.get_model_artifact(name, &version).unwrap();
        assert_eq!(retrieved, artifact);
    }

    #[test]
    fn test_model_stage_transition() {
        let (_dir, registry) = setup();

        let name = "test-model";
        let version = ModelVersion::new(1, 0, 0);
        registry.register_model(name, &version, b"data", ModelCard::new("Test")).unwrap();

        // Development -> Staging is valid
        registry.transition_model_stage(name, &version, crate::model::ModelStage::Staging).unwrap();

        let model = registry.get_model(name, &version).unwrap();
        assert_eq!(model.stage, crate::model::ModelStage::Staging);
    }

    #[test]
    fn test_register_and_get_dataset() {
        let (_dir, registry) = setup();

        let name = "test-dataset";
        let version = DatasetVersion::new(1, 0, 0);
        let data = b"csv,data,here";
        let datasheet = crate::data::Datasheet::new("Test dataset");

        let id = registry.register_dataset(name, &version, data, datasheet.clone()).unwrap();

        let dataset = registry.get_dataset(name, &version).unwrap();
        assert_eq!(dataset.id, id);
        assert_eq!(dataset.datasheet.purpose, datasheet.purpose);
    }

    #[test]
    fn test_dataset_data_roundtrip() {
        let (_dir, registry) = setup();

        let name = "test-dataset";
        let version = DatasetVersion::new(1, 0, 0);
        let data = b"raw dataset bytes";
        let datasheet = crate::data::Datasheet::new("Test");

        registry.register_dataset(name, &version, data, datasheet).unwrap();

        let retrieved = registry.get_dataset_data(name, &version).unwrap();
        assert_eq!(retrieved, data);
    }

    #[test]
    fn test_register_and_get_recipe() {
        let (_dir, registry) = setup();

        let recipe = crate::recipe::TrainingRecipe::builder()
            .name("test-recipe")
            .version(RecipeVersion::new(1, 0, 0))
            .description("Test recipe")
            .hyperparameters(Hyperparameters::default())
            .build();

        let id = registry.register_recipe(&recipe).unwrap();

        let retrieved = registry.get_recipe("test-recipe", &RecipeVersion::new(1, 0, 0)).unwrap();
        assert_eq!(retrieved.id, id);
        assert_eq!(retrieved.description, "Test recipe");
    }

    #[test]
    fn test_experiment_run() {
        let (_dir, registry) = setup();

        let mut run = crate::experiment::ExperimentRun::new(Hyperparameters::default());
        run.log_metric("loss", 0.5, 100);

        let run_id = registry.start_run(run).unwrap();

        let retrieved = registry.get_run(&run_id).unwrap();
        assert_eq!(retrieved.run_id, run_id);
        assert_eq!(retrieved.metrics.len(), 1);
    }

    #[test]
    fn test_storage_stats() {
        let (_dir, registry) = setup();

        registry
            .register_model("model1", &ModelVersion::new(1, 0, 0), b"data1", ModelCard::new("M1"))
            .unwrap();

        registry
            .register_dataset(
                "dataset1",
                &DatasetVersion::new(1, 0, 0),
                b"data2",
                crate::data::Datasheet::new("D1"),
            )
            .unwrap();

        let stats = registry.storage_stats().unwrap();
        assert_eq!(stats.model_count, 1);
        assert_eq!(stats.dataset_count, 1);
        assert_eq!(stats.object_count, 2);
    }

    #[test]
    fn test_list_operations() {
        let (_dir, registry) = setup();

        registry
            .register_model("model-a", &ModelVersion::new(1, 0, 0), b"data", ModelCard::new("A"))
            .unwrap();
        registry
            .register_model(
                "model-a",
                &ModelVersion::new(1, 1, 0),
                b"data2",
                ModelCard::new("A v1.1"),
            )
            .unwrap();
        registry
            .register_model("model-b", &ModelVersion::new(1, 0, 0), b"data3", ModelCard::new("B"))
            .unwrap();

        let models = registry.list_models().unwrap();
        assert_eq!(models.len(), 2);
        assert!(models.contains(&"model-a".to_string()));
        assert!(models.contains(&"model-b".to_string()));

        let versions = registry.list_model_versions("model-a").unwrap();
        assert_eq!(versions.len(), 2);
    }
}
