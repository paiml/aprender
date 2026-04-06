# Pacha: Model, Data and Recipe Registry Specification

**Version:** 1.0.0
**Status:** Draft
**Authors:** Pragmatic AI Labs
**References:** PACHA-REGISTRY-001

## Abstract

Pacha provides a unified registry for machine learning artifacts—models, datasets, and training recipes—with full lineage tracking, semantic versioning, and cryptographic integrity. This specification defines the architecture, data models, and APIs for sovereign MLOps workflows within the Pragmatic AI Labs ecosystem.

## 1. Introduction

### 1.1 Motivation

Modern ML systems suffer from reproducibility crises [1] and lack systematic artifact management. Studies show that only 15-30% of ML experiments are reproducible due to missing metadata, undocumented preprocessing, version mismatches [2], and deployment failures [19]. Pacha addresses these challenges through:

1. **Model Registry** - Semantic versioned model artifacts with lineage
2. **Data Registry** - Dataset versioning with provenance tracking
3. **Recipe Registry** - Training configurations for exact reproduction

### 1.2 Design Principles

Following the Toyota Way methodology applied throughout the Sovereign AI Stack [11]:

- **Muda (Waste Elimination)** - No redundant artifact storage; content-addressed deduplication
- **Jidoka (Built-in Quality)** - Cryptographic integrity verification at every stage
- **Kaizen (Continuous Improvement)** - Incremental lineage tracking for iterative development

### 1.3 Scope

Pacha integrates with:
- **alimentar** - Data loading with `.ald` encrypted format
- **aprender** - Model training with `.apr` encrypted format
- **entrenar** - Training pipelines and hyperparameter optimization
- **realizar** - Model serving and inference

## 2. Architecture

### 2.1 Component Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         Pacha Registry                          │
├─────────────────┬─────────────────┬─────────────────────────────┤
│  Model Store    │   Data Store    │      Recipe Store           │
│  ─────────────  │   ──────────    │      ────────────           │
│  • .apr files   │   • .ald files  │      • .toml configs        │
│  • Metadata     │   • Schema      │      • Hyperparameters      │
│  • Metrics      │   • Statistics  │      • Dependencies         │
│  • Lineage      │   • Provenance  │      • Environment          │
├─────────────────┴─────────────────┴─────────────────────────────┤
│                    Lineage Graph (trueno-graph)                 │
├─────────────────────────────────────────────────────────────────┤
│                    Content-Addressed Storage                     │
│                    (BLAKE3 hashing + deduplication)             │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Storage Backend

Pacha uses content-addressed storage with BLAKE3 hashing [3] to minimize redundancy [12] for:
- Deduplication across versions
- Tamper detection
- Efficient delta storage

```rust
pub struct ContentAddress {
    /// BLAKE3 hash of content
    pub hash: [u8; 32],
    /// Content size in bytes
    pub size: u64,
    /// Compression algorithm used
    pub compression: Compression,
}
```

## 3. Model Registry

### 3.1 Model Versioning

Models follow Semantic Versioning 2.0.0 with ML-specific semantics [4] to manage technical debt [13]:

| Version Component | ML Semantics |
|-------------------|--------------|
| MAJOR | Architecture change (incompatible inputs/outputs) |
| MINOR | Retraining with new data (backward compatible) |
| PATCH | Bug fixes, quantization, optimization |

```rust
pub struct ModelVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
    /// Optional pre-release identifier (e.g., "beta.1")
    pub prerelease: Option<String>,
    /// Build metadata (e.g., training run ID)
    pub build: Option<String>,
}
```

### 3.2 Model Card

Every registered model includes a Model Card [5] with standardized documentation [20]:

```rust
pub struct ModelCard {
    /// Model identification
    pub name: String,
    pub version: ModelVersion,
    pub description: String,

    /// Training details
    pub training_data: DatasetReference,
    pub training_recipe: RecipeReference,
    pub training_date: DateTime<Utc>,
    pub training_duration: Duration,

    /// Performance metrics
    pub metrics: HashMap<String, f64>,
    pub evaluation_data: DatasetReference,

    /// Intended use
    pub primary_uses: Vec<String>,
    pub out_of_scope_uses: Vec<String>,

    /// Limitations and biases
    pub limitations: Vec<String>,
    pub ethical_considerations: Vec<String>,

    /// Lineage
    pub parent_model: Option<ModelReference>,
    pub derived_from: Vec<ModelReference>,
}
```

### 3.3 Model Lineage

Pacha tracks full model lineage using a directed acyclic graph (DAG) stored in trueno-graph [14]:

```rust
pub enum ModelLineageEdge {
    /// Model was fine-tuned from parent
    FineTuned { parent: ModelId, recipe: RecipeId },
    /// Model was distilled from teacher
    Distilled { teacher: ModelId, temperature: f32 },
    /// Model was merged from multiple sources
    Merged { sources: Vec<ModelId>, weights: Vec<f32> },
    /// Model was quantized from source
    Quantized { source: ModelId, quantization: QuantType },
    /// Model was pruned from source
    Pruned { source: ModelId, sparsity: f32 },
}
```

This enables "time travel" queries: *"What was the exact state that produced this prediction?"* [6]

## 4. Data Registry

### 4.1 Dataset Versioning

Datasets are versioned using content-based hashing, similar to Git but optimized for large binary files [7]:

```rust
pub struct DatasetVersion {
    /// Content hash of the dataset
    pub content_hash: ContentAddress,
    /// Schema version (for compatibility checking)
    pub schema_version: SchemaVersion,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Parent version (if incremental)
    pub parent: Option<DatasetVersionId>,
    /// Delta from parent (if incremental)
    pub delta: Option<DeltaSpec>,
}
```

### 4.2 Datasheet

Following "Datasheets for Datasets" [8] and prioritizing data quality [18], every dataset includes:

```rust
pub struct Datasheet {
    /// Motivation
    pub purpose: String,
    pub creators: Vec<String>,
    pub funding: Option<String>,

    /// Composition
    pub instance_count: u64,
    pub feature_schema: Schema,
    pub label_schema: Option<Schema>,
    pub sensitive_features: Vec<String>,

    /// Collection process
    pub collection_method: String,
    pub collection_date: DateRange,
    pub preprocessing: Vec<PreprocessingStep>,

    /// Distribution
    pub license: String,
    pub access_restrictions: Vec<String>,

    /// Maintenance
    pub maintainer: String,
    pub update_frequency: Option<String>,
    pub deprecation_policy: Option<String>,
}
```

### 4.3 Data Provenance

Pacha implements W3C PROV-DM [9] for data provenance:

```rust
pub enum ProvenanceRecord {
    /// Data was derived from source
    WasDerivedFrom {
        derived: DatasetId,
        source: DatasetId,
        transformation: TransformationId,
    },
    /// Data was generated by activity
    WasGeneratedBy {
        data: DatasetId,
        activity: ActivityId,
        timestamp: DateTime<Utc>,
    },
    /// Activity used data
    Used {
        activity: ActivityId,
        data: DatasetId,
    },
    /// Entity was attributed to agent
    WasAttributedTo {
        entity: EntityId,
        agent: AgentId,
    },
}
```

## 5. Recipe Registry

### 5.1 Training Recipe

A Recipe captures the complete specification for reproducing a training run:

```rust
pub struct TrainingRecipe {
    /// Recipe identification
    pub name: String,
    pub version: RecipeVersion,
    pub description: String,

    /// Model architecture
    pub architecture: ArchitectureSpec,

    /// Training configuration
    pub hyperparameters: Hyperparameters,
    pub optimizer: OptimizerSpec,
    pub scheduler: Option<SchedulerSpec>,
    pub loss: LossSpec,

    /// Data configuration
    pub train_data: DatasetReference,
    pub validation_data: Option<DatasetReference>,
    pub preprocessing: Vec<PreprocessingStep>,
    pub augmentation: Vec<AugmentationStep>,

    /// Environment
    pub dependencies: Dependencies,
    pub hardware_requirements: HardwareSpec,
    pub random_seed: Option<u64>,

    /// Reproducibility metadata
    pub deterministic: bool,
    pub expected_metrics: Option<ExpectedMetrics>,
}
```

### 5.2 Hyperparameter Schema

Hyperparameters are strongly typed with validation [15]:

```rust
pub struct Hyperparameters {
    /// Learning rate (with schedule reference)
    pub learning_rate: f64,
    /// Batch size
    pub batch_size: usize,
    /// Number of epochs
    pub epochs: usize,
    /// Weight decay (L2 regularization)
    pub weight_decay: f64,
    /// Gradient clipping norm
    pub max_grad_norm: Option<f64>,
    /// Warmup steps
    pub warmup_steps: Option<usize>,
    /// Custom parameters
    pub custom: HashMap<String, HyperparamValue>,
}

pub enum HyperparamValue {
    Float(f64),
    Int(i64),
    Bool(bool),
    String(String),
    List(Vec<HyperparamValue>),
}
```

### 5.3 Environment Specification

For exact reproduction and elimination of configuration waste [15], recipes capture the full environment [10] and lifecycle dependencies [16]:

```rust
pub struct Dependencies {
    /// Rust toolchain version
    pub rust_version: String,
    /// Cargo dependencies with exact versions
    pub cargo_lock_hash: ContentAddress,
    /// System dependencies
    pub system_deps: Vec<SystemDep>,
    /// Environment variables (non-sensitive)
    pub env_vars: HashMap<String, String>,
}

pub struct HardwareSpec {
    /// Minimum CPU cores
    pub min_cpu_cores: usize,
    /// Minimum RAM in GB
    pub min_ram_gb: usize,
    /// GPU requirements
    pub gpu: Option<GpuRequirement>,
    /// Estimated training time
    pub estimated_duration: Duration,
}
```

## 6. Experiment Tracking

### 6.1 Experiment Run

Each training execution is tracked as an ExperimentRun:

```rust
pub struct ExperimentRun {
    /// Unique run identifier
    pub run_id: RunId,
    /// Recipe used
    pub recipe: RecipeReference,
    /// Actual hyperparameters (may override recipe)
    pub hyperparameters: Hyperparameters,

    /// Execution details
    pub started_at: DateTime<Utc>,
    pub finished_at: Option<DateTime<Utc>>,
    pub status: RunStatus,
    pub hardware_used: HardwareInfo,

    /// Metrics (time series)
    pub metrics: Vec<MetricRecord>,
    /// Artifacts produced
    pub artifacts: Vec<ArtifactReference>,
    /// Logs
    pub log_uri: Option<String>,

    /// Git integration
    pub git_commit: Option<String>,
    pub git_dirty: bool,
}

pub struct MetricRecord {
    pub name: String,
    pub value: f64,
    pub step: u64,
    pub timestamp: DateTime<Utc>,
}
```

### 6.2 Experiment Comparison

Pacha enables systematic experiment comparison for continuous parameter optimization [17]:

```rust
impl ExperimentStore {
    /// Compare multiple runs
    pub fn compare_runs(&self, run_ids: &[RunId]) -> ComparisonReport {
        // Extract final metrics
        // Compute statistical significance
        // Generate comparison table
    }

    /// Find best run by metric
    pub fn best_run(&self,
        experiment: &str,
        metric: &str,
        minimize: bool
    ) -> Option<ExperimentRun> {
        // Query runs, sort by metric
    }

    /// Hyperparameter importance analysis
    pub fn hyperparameter_importance(&self,
        experiment: &str,
        metric: &str
    ) -> Vec<(String, f64)> {
        // Compute correlation between hyperparams and metric
    }
}
```

## 7. API Design

### 7.1 Model Registry API

```rust
pub trait ModelRegistry {
    /// Register a new model
    fn register_model(&self,
        name: &str,
        artifact: &Path,
        card: ModelCard
    ) -> Result<ModelId>;

    /// Get model by name and version
    fn get_model(&self,
        name: &str,
        version: &ModelVersion
    ) -> Result<Model>;

    /// List all versions of a model
    fn list_versions(&self, name: &str) -> Result<Vec<ModelVersion>>;

    /// Get model lineage
    fn get_lineage(&self, model_id: ModelId) -> Result<LineageGraph>;

    /// Transition model stage
    fn transition_stage(&self,
        model_id: ModelId,
        stage: ModelStage
    ) -> Result<()>;
}

pub enum ModelStage {
    Development,
    Staging,
    Production,
    Archived,
}
```

### 7.2 Data Registry API

```rust
pub trait DataRegistry {
    /// Register a dataset
    fn register_dataset(&self,
        name: &str,
        source: DataSource,
        datasheet: Datasheet
    ) -> Result<DatasetId>;

    /// Get dataset by name and version
    fn get_dataset(&self,
        name: &str,
        version: Option<&DatasetVersion>
    ) -> Result<Dataset>;

    /// Create incremental version
    fn create_version(&self,
        dataset_id: DatasetId,
        delta: DeltaSpec
    ) -> Result<DatasetVersion>;

    /// Get provenance chain
    fn get_provenance(&self,
        dataset_id: DatasetId
    ) -> Result<Vec<ProvenanceRecord>>;
}
```

### 7.3 Recipe Registry API

```rust
pub trait RecipeRegistry {
    /// Register a training recipe
    fn register_recipe(&self,
        recipe: TrainingRecipe
    ) -> Result<RecipeId>;

    /// Get recipe by name and version
    fn get_recipe(&self,
        name: &str,
        version: &RecipeVersion
    ) -> Result<TrainingRecipe>;

    /// Validate recipe reproducibility
    fn validate_recipe(&self,
        recipe_id: RecipeId
    ) -> Result<ValidationReport>;

    /// Start experiment run from recipe
    fn start_run(&self,
        recipe_id: RecipeId,
        overrides: Option<Hyperparameters>
    ) -> Result<RunId>;
}
```

## 8. Storage Format

### 8.1 Registry Database

Pacha stores metadata in a local SQLite database with the following schema:

```sql
-- Models
CREATE TABLE models (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    version TEXT NOT NULL,
    content_hash BLOB NOT NULL,
    card_json TEXT NOT NULL,
    stage TEXT DEFAULT 'development',
    created_at TEXT NOT NULL,
    UNIQUE(name, version)
);

-- Datasets
CREATE TABLE datasets (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    version TEXT NOT NULL,
    content_hash BLOB NOT NULL,
    datasheet_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(name, version)
);

-- Recipes
CREATE TABLE recipes (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    version TEXT NOT NULL,
    recipe_toml TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(name, version)
);

-- Experiment Runs
CREATE TABLE runs (
    id TEXT PRIMARY KEY,
    recipe_id TEXT REFERENCES recipes(id),
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    metrics_json TEXT
);

-- Lineage edges
CREATE TABLE lineage (
    id INTEGER PRIMARY KEY,
    from_id TEXT NOT NULL,
    to_id TEXT NOT NULL,
    edge_type TEXT NOT NULL,
    metadata_json TEXT
);
```

### 8.2 Artifact Storage

Large artifacts are stored in a content-addressed filesystem:

```
~/.pacha/
├── config.toml           # Registry configuration
├── registry.db           # SQLite metadata
└── objects/              # Content-addressed storage
    ├── ab/
    │   └── cdef1234...   # BLAKE3 hash prefix sharding
    ├── cd/
    │   └── ef5678...
    └── ...
```

## 9. Integration

### 9.1 Aprender Integration

```rust
use aprender::format::{save, load, ModelType};
use pacha::{ModelRegistry, ModelCard};

// Train model
let model = train_random_forest(&data)?;

// Save to .apr format
let artifact_path = "model.apr";
save(&model, ModelType::RandomForest, artifact_path, Default::default())?;

// Register with Pacha
let registry = PachaRegistry::open("~/.pacha")?;
let card = ModelCard::builder()
    .name("fraud-detector")
    .version("1.0.0")
    .training_data(dataset_ref)
    .metrics([("auc", 0.95), ("f1", 0.88)])
    .build()?;

registry.register_model("fraud-detector", artifact_path, card)?;
```

### 9.2 Entrenar Integration

```rust
use entrenar::Pipeline;
use pacha::{RecipeRegistry, TrainingRecipe};

// Define recipe
let recipe = TrainingRecipe::builder()
    .name("bert-finetune")
    .architecture(BertConfig::base())
    .hyperparameters(Hyperparameters {
        learning_rate: 2e-5,
        batch_size: 32,
        epochs: 3,
        ..Default::default()
    })
    .train_data(dataset_ref)
    .build()?;

// Register recipe
let registry = PachaRegistry::open("~/.pacha")?;
let recipe_id = registry.register_recipe(recipe)?;

// Start tracked run
let run_id = registry.start_run(recipe_id, None)?;
let pipeline = Pipeline::new().with_tracking(run_id);
pipeline.train(&model, &data)?;
```

### 9.3 CLI Interface

```bash
# Model operations
pacha model register fraud-detector ./model.apr --card card.toml
pacha model list fraud-detector
pacha model get fraud-detector:1.0.0 -o ./downloaded.apr
pacha model lineage fraud-detector:1.0.0

# Dataset operations
pacha data register customer-transactions ./data.ald --datasheet sheet.toml
pacha data list
pacha data provenance customer-transactions:latest

# Recipe operations
pacha recipe register ./recipe.toml
pacha recipe validate bert-finetune:1.0.0
pacha recipe run bert-finetune:1.0.0 --override learning_rate=1e-5

# Experiment tracking
pacha run list bert-finetune
pacha run compare run-001 run-002 run-003
pacha run best bert-finetune --metric val_loss --minimize
```

## 10. Security

### 10.1 Encryption

All artifacts support encryption via the Sovereign AI Stack cryptographic primitives:

- **Models (.apr)** - AES-256-GCM with Argon2id KDF
- **Datasets (.ald)** - AES-256-GCM with Argon2id KDF
- **Recipes (.toml)** - Optional GPG signing

### 10.2 Access Control

```rust
pub struct AccessPolicy {
    /// Who can read artifacts
    pub readers: Vec<Principal>,
    /// Who can write/modify
    pub writers: Vec<Principal>,
    /// Who can delete
    pub admins: Vec<Principal>,
}

pub enum Principal {
    User(String),
    Group(String),
    ServiceAccount(String),
    Public,
}
```

## 11. References

[1] Hutson, M. (2018). "Artificial intelligence faces reproducibility crisis." *Science*, 359(6377), 725-726. DOI: 10.1126/science.359.6377.725

[2] Gundersen, O. E., & Kjensmo, S. (2018). "State of the art: Reproducibility in artificial intelligence." *Proceedings of the AAAI Conference on Artificial Intelligence*, 32(1). DOI: 10.1609/aaai.v32i1.11503

[3] O'Connor, J., et al. (2020). "BLAKE3: One function, fast everywhere." *IACR Cryptology ePrint Archive*. https://github.com/BLAKE3-team/BLAKE3-specs

[4] Preston-Werner, T. (2013). "Semantic Versioning 2.0.0." https://semver.org/

[5] Mitchell, M., et al. (2019). "Model Cards for Model Reporting." *Proceedings of the Conference on Fairness, Accountability, and Transparency (FAT*)*, 220-229. DOI: 10.1145/3287560.3287596

[6] Schelter, S., et al. (2017). "Automatically tracking metadata and provenance of machine learning experiments." *Machine Learning Systems Workshop at NIPS*.

[7] Miao, H., et al. (2017). "Towards unified data and lifecycle management for deep learning." *IEEE 33rd International Conference on Data Engineering (ICDE)*, 571-582. DOI: 10.1109/ICDE.2017.112

[8] Gebru, T., et al. (2021). "Datasheets for Datasets." *Communications of the ACM*, 64(12), 86-92. DOI: 10.1145/3458723

[9] Moreau, L., & Missier, P. (2013). "PROV-DM: The PROV Data Model." *W3C Recommendation*. https://www.w3.org/TR/prov-dm/

[10] Pineau, J., et al. (2021). "Improving Reproducibility in Machine Learning Research (A Report from the NeurIPS 2019 Reproducibility Program)." *Journal of Machine Learning Research*, 22(164), 1-20.

[11] Poppendieck, M., & Poppendieck, T. (2003). "Lean Software Development: An Agile Toolkit." *Addison-Wesley Professional*.

[12] Hoefler, T., et al. (2020). "Data Distribution in Deep Learning." *arXiv preprint arXiv:2011.14417*.

[13] Sculley, D., et al. (2015). "Hidden technical debt in machine learning systems." *Advances in neural information processing systems*, 28.

[14] Vartak, M., et al. (2016). "ModelDB: a system for machine learning model management." *Proceedings of the Workshop on Human-In-the-Loop Data Analytics*, 1-3.

[15] Baylor, D., et al. (2017). "Tfx: A tensorflow-based production-scale machine learning platform." *Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 1387-1395.

[16] Zaharia, M., et al. (2018). "Accelerating the Machine Learning Lifecycle with MLflow." *IEEE Data Eng. Bull.*, 41(4), 39-45.

[17] Akiba, T., et al. (2019). "Optuna: A next-generation hyperparameter optimization framework." *Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery & data mining*, 2623-2631.

[18] Sambasivan, N., et al. (2021). "Everyone wants to do the model work, not the data work: Data cascades in high-stakes AI." *Proceedings of the 2021 CHI Conference on Human Factors in Computing Systems*, 1-15.

[19] Paleyes, A., et al. (2022). "Challenges in deploying machine learning: a survey of case studies." *ACM Computing Surveys*, 55(6), 1-29.

[20] Crisan, A., et al. (2022). "Interactive model cards: A human-centered approach to model documentation." *2022 ACM Conference on Fairness, Accountability, and Transparency*, 427-439.

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| **Artifact** | Any file produced by ML workflows (models, data, configs) |
| **Lineage** | The provenance chain showing how artifacts were derived |
| **Recipe** | Complete specification for reproducing a training run |
| **Stage** | Lifecycle state of a model (development, staging, production) |
| **Content Address** | Hash-based identifier for deduplication |

## Appendix B: Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2024-11-28 | Initial specification |