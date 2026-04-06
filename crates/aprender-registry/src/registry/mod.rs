//! Registry implementation with `SQLite` storage.

mod database;

pub use database::RegistryDb;

use crate::data::{Dataset, DatasetId, Datasheet};
use crate::error::{PachaError, Result};
use crate::experiment::{ExperimentRun, RunId};
use crate::lineage::LineageGraph;
use crate::model::{Model, ModelCard, ModelId, ModelStage, ModelVersion};
use crate::recipe::{RecipeId, RecipeReference, TrainingRecipe};
use crate::storage::ObjectStore;
use chrono::Utc;
use std::fs;
use std::path::{Path, PathBuf};

/// Configuration for the Pacha registry.
#[derive(Debug, Clone)]
pub struct RegistryConfig {
    /// Base path for the registry.
    pub base_path: PathBuf,
}

impl RegistryConfig {
    /// Create a new config with the given base path.
    #[must_use]
    pub fn new<P: AsRef<Path>>(base_path: P) -> Self {
        Self { base_path: base_path.as_ref().to_path_buf() }
    }

    /// Get the database path.
    #[must_use]
    pub fn db_path(&self) -> PathBuf {
        self.base_path.join("registry.db")
    }

    /// Get the objects path.
    #[must_use]
    pub fn objects_path(&self) -> PathBuf {
        self.base_path.join("objects")
    }

    /// Get the config file path.
    #[must_use]
    pub fn config_path(&self) -> PathBuf {
        self.base_path.join("config.toml")
    }
}

impl Default for RegistryConfig {
    fn default() -> Self {
        let home = dirs_path();
        Self::new(home.join(".pacha"))
    }
}

fn dirs_path() -> PathBuf {
    std::env::var("HOME").map_or_else(|_| PathBuf::from("."), PathBuf::from)
}

/// The main Pacha registry.
pub struct Registry {
    config: RegistryConfig,
    db: RegistryDb,
    objects: ObjectStore,
}

impl Registry {
    /// Create or open a registry at the default location (~/.pacha).
    ///
    /// # Errors
    ///
    /// Returns an error if initialization fails.
    pub fn open_default() -> Result<Self> {
        Self::open(RegistryConfig::default())
    }

    /// Create or open a registry with the given configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if initialization fails.
    pub fn open(config: RegistryConfig) -> Result<Self> {
        // Create base directory
        fs::create_dir_all(&config.base_path)?;

        // Initialize database
        let db = RegistryDb::open(config.db_path())?;

        // Initialize object store
        let objects = ObjectStore::new(config.objects_path())?;

        Ok(Self { config, db, objects })
    }

    /// Get the registry configuration.
    #[must_use]
    pub fn config(&self) -> &RegistryConfig {
        &self.config
    }

    // ==================== Model Registry ====================

    /// Register a new model.
    ///
    /// # Errors
    ///
    /// Returns an error if registration fails.
    pub fn register_model(
        &self,
        name: &str,
        version: &ModelVersion,
        artifact: &[u8],
        card: ModelCard,
    ) -> Result<ModelId> {
        // Check if already exists
        if self.db.model_exists(name, version)? {
            return Err(PachaError::AlreadyExists {
                kind: "model".to_string(),
                name: name.to_string(),
                version: version.to_string(),
            });
        }

        // Store artifact
        let content_address = self.objects.put(artifact)?;

        // Create model
        let model = Model {
            id: ModelId::new(),
            name: name.to_string(),
            version: version.clone(),
            content_address,
            card,
            stage: ModelStage::Development,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };

        // Save to database
        self.db.insert_model(&model)?;

        Ok(model.id)
    }

    /// Get a model by name and version.
    ///
    /// # Errors
    ///
    /// Returns an error if the model is not found.
    pub fn get_model(&self, name: &str, version: &ModelVersion) -> Result<Model> {
        // Contract: configuration-v1.yaml precondition (pv codegen)
        contract_pre_configuration!(name.as_bytes());
        self.db.get_model(name, version)
    }

    /// Get a model by ID.
    ///
    /// # Errors
    ///
    /// Returns an error if the model is not found.
    pub fn get_model_by_id(&self, id: &ModelId) -> Result<Model> {
        self.db.get_model_by_id(id)
    }

    /// List all versions of a model.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub fn list_model_versions(&self, name: &str) -> Result<Vec<ModelVersion>> {
        contract_pre_ols_fit!();
        let result = self.db.list_model_versions(name);
        if let Ok(ref val) = result { contract_post_configuration!(val); }
        result
    }

    /// List all model names.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub fn list_models(&self) -> Result<Vec<String>> {
        contract_pre_ols_fit!();
        let result = self.db.list_model_names();
        if let Ok(ref val) = result { contract_post_configuration!(val); }
        result
    }

    /// Transition a model to a new stage.
    ///
    /// # Errors
    ///
    /// Returns an error if the transition is invalid.
    pub fn transition_model_stage(
        &self,
        name: &str,
        version: &ModelVersion,
        target_stage: ModelStage,
    ) -> Result<()> {
        let model = self.get_model(name, version)?;
        let _new_stage = model.stage.transition_to(target_stage)?;
        self.db.update_model_stage(&model.id, target_stage)
    }

    /// Get the artifact data for a model.
    ///
    /// # Errors
    ///
    /// Returns an error if the artifact cannot be retrieved.
    pub fn get_model_artifact(&self, name: &str, version: &ModelVersion) -> Result<Vec<u8>> {
        let model = self.get_model(name, version)?;
        self.objects.get(&model.content_address)
    }

    /// Get model lineage graph.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub fn get_model_lineage(&self, _model_id: &ModelId) -> Result<LineageGraph> {
        // Returns empty graph until lineage data is populated
        Ok(LineageGraph::new())
    }

    // ==================== Dataset Registry ====================

    /// Register a new dataset.
    ///
    /// # Errors
    ///
    /// Returns an error if registration fails.
    pub fn register_dataset(
        &self,
        name: &str,
        version: &crate::data::DatasetVersion,
        data: &[u8],
        datasheet: Datasheet,
    ) -> Result<DatasetId> {
        // Check if already exists
        if self.db.dataset_exists(name, version)? {
            return Err(PachaError::AlreadyExists {
                kind: "dataset".to_string(),
                name: name.to_string(),
                version: version.to_string(),
            });
        }

        // Store data
        let content_address = self.objects.put(data)?;

        // Create dataset
        let dataset = Dataset {
            id: DatasetId::new(),
            name: name.to_string(),
            version: version.clone(),
            content_address,
            datasheet,
            created_at: Utc::now(),
        };

        // Save to database
        self.db.insert_dataset(&dataset)?;

        Ok(dataset.id)
    }

    /// Get a dataset by name and version.
    ///
    /// # Errors
    ///
    /// Returns an error if the dataset is not found.
    pub fn get_dataset(
        &self,
        name: &str,
        version: &crate::data::DatasetVersion,
    ) -> Result<Dataset> {
        self.db.get_dataset(name, version)
    }

    /// List all dataset names.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub fn list_datasets(&self) -> Result<Vec<String>> {
        contract_pre_configuration!();
        let result = self.db.list_dataset_names();
        if let Ok(ref val) = result { contract_post_configuration!(val); }
        result
    }

    /// List all versions of a dataset.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub fn list_dataset_versions(&self, name: &str) -> Result<Vec<crate::data::DatasetVersion>> {
        contract_pre_configuration!(name);
        let result = self.db.list_dataset_versions(name);
        if let Ok(ref val) = result { contract_post_configuration!(val); }
        result
    }

    /// Get the data for a dataset.
    ///
    /// # Errors
    ///
    /// Returns an error if the data cannot be retrieved.
    pub fn get_dataset_data(
        &self,
        name: &str,
        version: &crate::data::DatasetVersion,
    ) -> Result<Vec<u8>> {
        let dataset = self.get_dataset(name, version)?;
        self.objects.get(&dataset.content_address)
    }

    // ==================== Recipe Registry ====================

    /// Register a new recipe.
    ///
    /// # Errors
    ///
    /// Returns an error if registration fails.
    pub fn register_recipe(&self, recipe: &TrainingRecipe) -> Result<RecipeId> {
        // Check if already exists
        if self.db.recipe_exists(&recipe.name, &recipe.version)? {
            return Err(PachaError::AlreadyExists {
                kind: "recipe".to_string(),
                name: recipe.name.clone(),
                version: recipe.version.to_string(),
            });
        }

        let id = recipe.id.clone();
        self.db.insert_recipe(recipe)?;
        Ok(id)
    }

    /// Get a recipe by name and version.
    ///
    /// # Errors
    ///
    /// Returns an error if the recipe is not found.
    pub fn get_recipe(
        &self,
        name: &str,
        version: &crate::recipe::RecipeVersion,
    ) -> Result<TrainingRecipe> {
        self.db.get_recipe(name, version)
    }

    /// List all recipe names.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub fn list_recipes(&self) -> Result<Vec<String>> {
        contract_pre_configuration!();
        let result = self.db.list_recipe_names();
        if let Ok(ref val) = result { contract_post_configuration!(val); }
        result
    }

    /// List all versions of a recipe.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub fn list_recipe_versions(&self, name: &str) -> Result<Vec<crate::recipe::RecipeVersion>> {
        contract_pre_expand_recipe!(name);
        self.db.list_recipe_versions(name)
    }

    // ==================== Experiment Tracking ====================

    /// Start a new experiment run.
    ///
    /// # Errors
    ///
    /// Returns an error if starting fails.
    pub fn start_run(&self, mut run: ExperimentRun) -> Result<RunId> {
        contract_pre_configuration!();
        run.start();
        let id = run.run_id.clone();
        self.db.insert_run(&run)?;
        Ok(id)
    }

    /// Update an experiment run.
    ///
    /// # Errors
    ///
    /// Returns an error if the update fails.
    pub fn update_run(&self, run: &ExperimentRun) -> Result<()> {
        contract_pre_configuration!();
        self.db.update_run(run)
    }

    /// Get an experiment run by ID.
    ///
    /// # Errors
    ///
    /// Returns an error if the run is not found.
    pub fn get_run(&self, run_id: &RunId) -> Result<ExperimentRun> {
        contract_pre_configuration!();
        self.db.get_run(run_id)
    }

    /// List runs for a recipe.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub fn list_runs(&self, recipe_ref: &RecipeReference) -> Result<Vec<ExperimentRun>> {
        contract_pre_configuration!();
        self.db.list_runs_for_recipe(recipe_ref)
    }

    // ==================== Utility ====================

    /// Get storage statistics.
    ///
    /// # Errors
    ///
    /// Returns an error if querying fails.
    pub fn storage_stats(&self) -> Result<StorageStats> {
        let total_size = self.objects.total_size()?;
        let object_count = self.objects.list()?.len();
        let model_count = self.db.count_models()?;
        let dataset_count = self.db.count_datasets()?;
        let recipe_count = self.db.count_recipes()?;

        Ok(StorageStats {
            total_size_bytes: total_size,
            object_count,
            model_count,
            dataset_count,
            recipe_count,
        })
    }
}

/// Storage statistics.
#[derive(Debug, Clone)]
pub struct StorageStats {
    /// Total size of all objects in bytes.
    pub total_size_bytes: u64,
    /// Number of content-addressed objects.
    pub object_count: usize,
    /// Number of registered models.
    pub model_count: usize,
    /// Number of registered datasets.
    pub dataset_count: usize,
    /// Number of registered recipes.
    pub recipe_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
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
        assert!(matches!(result, Err(PachaError::AlreadyExists { .. })));
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
        registry.transition_model_stage(name, &version, ModelStage::Staging).unwrap();

        let model = registry.get_model(name, &version).unwrap();
        assert_eq!(model.stage, ModelStage::Staging);
    }

    #[test]
    fn test_register_and_get_dataset() {
        let (_dir, registry) = setup();

        let name = "test-dataset";
        let version = DatasetVersion::new(1, 0, 0);
        let data = b"csv,data,here";
        let datasheet = Datasheet::new("Test dataset");

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
        let datasheet = Datasheet::new("Test");

        registry.register_dataset(name, &version, data, datasheet).unwrap();

        let retrieved = registry.get_dataset_data(name, &version).unwrap();
        assert_eq!(retrieved, data);
    }

    #[test]
    fn test_register_and_get_recipe() {
        let (_dir, registry) = setup();

        let recipe = TrainingRecipe::builder()
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

        let mut run = ExperimentRun::new(Hyperparameters::default());
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
                Datasheet::new("D1"),
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
