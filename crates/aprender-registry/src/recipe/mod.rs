//! Recipe registry types and operations.
//!
//! Provides training recipes for reproducible ML workflows.

mod hyperparams;
mod version;

pub use hyperparams::{HyperparamValue, Hyperparameters};
pub use version::RecipeVersion;

use crate::data::DatasetReference;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Unique identifier for a registered recipe.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RecipeId(Uuid);

impl RecipeId {
    /// Create a new random recipe ID.
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Create from a UUID.
    #[must_use]
    pub fn from_uuid(uuid: Uuid) -> Self {
        Self(uuid)
    }

    /// Get the underlying UUID.
    #[must_use]
    pub fn as_uuid(&self) -> &Uuid {
        &self.0
    }
}

impl Default for RecipeId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for RecipeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::str::FromStr for RecipeId {
    type Err = uuid::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Self(Uuid::parse_str(s)?))
    }
}

/// Reference to a recipe (name + version).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecipeReference {
    /// Recipe name.
    pub name: String,
    /// Recipe version.
    pub version: RecipeVersion,
}

impl RecipeReference {
    /// Create a new recipe reference.
    #[must_use]
    pub fn new(name: impl Into<String>, version: RecipeVersion) -> Self {
        Self { name: name.into(), version }
    }
}

impl std::fmt::Display for RecipeReference {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.name, self.version)
    }
}

/// Training recipe for reproducible ML workflows.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingRecipe {
    /// Unique identifier.
    pub id: RecipeId,
    /// Recipe name.
    pub name: String,
    /// Recipe version.
    pub version: RecipeVersion,
    /// Description.
    pub description: String,

    /// Model architecture specification.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub architecture: Option<String>,

    /// Training hyperparameters.
    pub hyperparameters: Hyperparameters,

    /// Optimizer configuration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub optimizer: Option<OptimizerSpec>,

    /// Learning rate scheduler.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scheduler: Option<SchedulerSpec>,

    /// Loss function.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub loss: Option<LossSpec>,

    /// Training data reference.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub train_data: Option<DatasetReference>,

    /// Validation data reference.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub validation_data: Option<DatasetReference>,

    /// Preprocessing steps.
    #[serde(default)]
    pub preprocessing: Vec<String>,

    /// Data augmentation steps.
    #[serde(default)]
    pub augmentation: Vec<String>,

    /// Environment dependencies.
    pub dependencies: Dependencies,

    /// Hardware requirements.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hardware: Option<HardwareSpec>,

    /// Random seed for reproducibility.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub random_seed: Option<u64>,

    /// Whether the recipe produces deterministic results.
    #[serde(default)]
    pub deterministic: bool,

    /// Registration timestamp.
    pub created_at: DateTime<Utc>,

    /// Additional metadata.
    #[serde(default)]
    pub extra: HashMap<String, serde_json::Value>,
}

impl TrainingRecipe {
    /// Create a new recipe builder.
    #[must_use]
    pub fn builder() -> TrainingRecipeBuilder {
        TrainingRecipeBuilder::new()
    }

    /// Create a reference to this recipe.
    #[must_use]
    pub fn reference(&self) -> RecipeReference {
        RecipeReference::new(&self.name, self.version.clone())
    }
}

/// Optimizer specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerSpec {
    /// Optimizer type (e.g., "adam", "sgd").
    pub optimizer_type: String,
    /// Optimizer-specific parameters.
    #[serde(default)]
    pub params: HashMap<String, HyperparamValue>,
}

impl OptimizerSpec {
    /// Create a new optimizer spec.
    #[must_use]
    pub fn new(optimizer_type: impl Into<String>) -> Self {
        Self { optimizer_type: optimizer_type.into(), params: HashMap::new() }
    }

    /// Add a parameter.
    #[must_use]
    pub fn with_param(mut self, name: impl Into<String>, value: HyperparamValue) -> Self {
        self.params.insert(name.into(), value);
        self
    }
}

/// Learning rate scheduler specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerSpec {
    /// Scheduler type (e.g., "cosine", "step").
    pub scheduler_type: String,
    /// Scheduler-specific parameters.
    #[serde(default)]
    pub params: HashMap<String, HyperparamValue>,
}

impl SchedulerSpec {
    /// Create a new scheduler spec.
    #[must_use]
    pub fn new(scheduler_type: impl Into<String>) -> Self {
        Self { scheduler_type: scheduler_type.into(), params: HashMap::new() }
    }
}

/// Loss function specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LossSpec {
    /// Loss function type (e.g., "`cross_entropy`", "mse").
    pub loss_type: String,
    /// Loss-specific parameters.
    #[serde(default)]
    pub params: HashMap<String, HyperparamValue>,
}

impl LossSpec {
    /// Create a new loss spec.
    #[must_use]
    pub fn new(loss_type: impl Into<String>) -> Self {
        Self { loss_type: loss_type.into(), params: HashMap::new() }
    }
}

/// Environment dependencies.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Dependencies {
    /// Rust toolchain version.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rust_version: Option<String>,
    /// Cargo.lock hash for exact reproducibility.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cargo_lock_hash: Option<String>,
    /// System dependencies.
    #[serde(default)]
    pub system_deps: Vec<String>,
    /// Environment variables (non-sensitive).
    #[serde(default)]
    pub env_vars: HashMap<String, String>,
}

/// Hardware requirements.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareSpec {
    /// Minimum CPU cores.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_cpu_cores: Option<usize>,
    /// Minimum RAM in GB.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_ram_gb: Option<usize>,
    /// GPU requirements.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu: Option<GpuRequirement>,
    /// Estimated training time in seconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub estimated_duration_secs: Option<u64>,
}

/// GPU requirement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuRequirement {
    /// Minimum GPU count.
    pub count: usize,
    /// Minimum VRAM per GPU in GB.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_vram_gb: Option<usize>,
    /// Required compute capability.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compute_capability: Option<String>,
}

/// Builder for creating training recipes.
#[derive(Debug)]
pub struct TrainingRecipeBuilder {
    name: String,
    version: RecipeVersion,
    description: String,
    architecture: Option<String>,
    hyperparameters: Hyperparameters,
    optimizer: Option<OptimizerSpec>,
    scheduler: Option<SchedulerSpec>,
    loss: Option<LossSpec>,
    train_data: Option<DatasetReference>,
    validation_data: Option<DatasetReference>,
    preprocessing: Vec<String>,
    augmentation: Vec<String>,
    dependencies: Dependencies,
    hardware: Option<HardwareSpec>,
    random_seed: Option<u64>,
    deterministic: bool,
}

impl TrainingRecipeBuilder {
    /// Create a new builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            name: String::new(),
            version: RecipeVersion::initial(),
            description: String::new(),
            architecture: None,
            hyperparameters: Hyperparameters::default(),
            optimizer: None,
            scheduler: None,
            loss: None,
            train_data: None,
            validation_data: None,
            preprocessing: Vec::new(),
            augmentation: Vec::new(),
            dependencies: Dependencies::default(),
            hardware: None,
            random_seed: None,
            deterministic: false,
        }
    }

    /// Set the name.
    #[must_use]
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Set the version.
    #[must_use]
    pub fn version(mut self, version: RecipeVersion) -> Self {
        self.version = version;
        self
    }

    /// Set the description.
    #[must_use]
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// Set the architecture.
    #[must_use]
    pub fn architecture(mut self, architecture: impl Into<String>) -> Self {
        self.architecture = Some(architecture.into());
        self
    }

    /// Set hyperparameters.
    #[must_use]
    pub fn hyperparameters(mut self, hyperparameters: Hyperparameters) -> Self {
        self.hyperparameters = hyperparameters;
        self
    }

    /// Set optimizer.
    #[must_use]
    pub fn optimizer(mut self, optimizer: OptimizerSpec) -> Self {
        self.optimizer = Some(optimizer);
        self
    }

    /// Set scheduler.
    #[must_use]
    pub fn scheduler(mut self, scheduler: SchedulerSpec) -> Self {
        self.scheduler = Some(scheduler);
        self
    }

    /// Set loss.
    #[must_use]
    pub fn loss(mut self, loss: LossSpec) -> Self {
        self.loss = Some(loss);
        self
    }

    /// Set training data.
    #[must_use]
    pub fn train_data(mut self, data: DatasetReference) -> Self {
        self.train_data = Some(data);
        self
    }

    /// Set validation data.
    #[must_use]
    pub fn validation_data(mut self, data: DatasetReference) -> Self {
        self.validation_data = Some(data);
        self
    }

    /// Set random seed.
    #[must_use]
    pub fn random_seed(mut self, seed: u64) -> Self {
        self.random_seed = Some(seed);
        self
    }

    /// Set deterministic flag.
    #[must_use]
    pub fn deterministic(mut self, deterministic: bool) -> Self {
        self.deterministic = deterministic;
        self
    }

    /// Build the recipe.
    #[must_use]
    pub fn build(self) -> TrainingRecipe {
        TrainingRecipe {
            id: RecipeId::new(),
            name: self.name,
            version: self.version,
            description: self.description,
            architecture: self.architecture,
            hyperparameters: self.hyperparameters,
            optimizer: self.optimizer,
            scheduler: self.scheduler,
            loss: self.loss,
            train_data: self.train_data,
            validation_data: self.validation_data,
            preprocessing: self.preprocessing,
            augmentation: self.augmentation,
            dependencies: self.dependencies,
            hardware: self.hardware,
            random_seed: self.random_seed,
            deterministic: self.deterministic,
            created_at: Utc::now(),
            extra: HashMap::new(),
        }
    }
}

impl Default for TrainingRecipeBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recipe_id_generation() {
        let id1 = RecipeId::new();
        let id2 = RecipeId::new();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_recipe_reference_display() {
        let reference = RecipeReference::new("bert-finetune", RecipeVersion::new(1, 2, 3));
        assert_eq!(reference.to_string(), "bert-finetune:1.2.3");
    }

    #[test]
    fn test_recipe_builder() {
        let hyperparams = Hyperparameters {
            learning_rate: 2e-5,
            batch_size: 32,
            epochs: 3,
            ..Default::default()
        };

        let recipe = TrainingRecipe::builder()
            .name("bert-finetune")
            .version(RecipeVersion::new(1, 0, 0))
            .description("Fine-tune BERT for sentiment analysis")
            .hyperparameters(hyperparams)
            .optimizer(OptimizerSpec::new("adam"))
            .loss(LossSpec::new("cross_entropy"))
            .random_seed(42)
            .deterministic(true)
            .build();

        assert_eq!(recipe.name, "bert-finetune");
        assert_eq!(recipe.hyperparameters.learning_rate, 2e-5);
        assert_eq!(recipe.hyperparameters.batch_size, 32);
        assert_eq!(recipe.random_seed, Some(42));
        assert!(recipe.deterministic);
    }

    #[test]
    fn test_optimizer_spec() {
        let optimizer = OptimizerSpec::new("adam")
            .with_param("beta1", HyperparamValue::Float(0.9))
            .with_param("beta2", HyperparamValue::Float(0.999));

        assert_eq!(optimizer.optimizer_type, "adam");
        assert_eq!(optimizer.params.len(), 2);
    }

    #[test]
    fn test_recipe_serialization() {
        let recipe = TrainingRecipe::builder().name("test-recipe").description("Test").build();

        let json = serde_json::to_string(&recipe).unwrap();
        let deserialized: TrainingRecipe = serde_json::from_str(&json).unwrap();

        assert_eq!(recipe.name, deserialized.name);
    }
}
