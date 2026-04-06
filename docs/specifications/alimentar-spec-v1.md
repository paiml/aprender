# alimentar Specification v1.0

> **Toyota Way Review Summary**
> *   **Reviewer:** Gemini CLI (Acting as Chief Engineer)
> *   **Date:** 2025-11-25
> *   **Philosophy:** The Toyota Way & Lean Product Development
>
> **Executive Summary:**
> This specification exhibits a strong adherence to **Jidoka** (Built-in Quality) through its "EXTREME TDD" section and **Standardized Work** via the `pmat` scaffolding. The **Sovereign-first** principle is a robust example of **Principle 1: Base your management decisions on a long-term philosophy, even at the expense of short-term financial goals**.
>
> However, the heavy reliance on specific cloud-provider compatibility lists requires rigorous **Genchi Genbutsu** (Go and see) to avoid integration friction. The architecture promotes **Flow** (Principle 2) through streaming data types, reducing "Muda" (waste) in memory usage.

**Sovereign Data Loading & Distribution for the Rust ML Stack**

## Overview

alimentar (Spanish: "to feed") is a pure Rust data loading, transformation, and distribution library for the paiml sovereign AI stack. It provides HuggingFace-compatible functionality without US cloud dependency.

## Design Principles

> **Review Note (Principle 1: Long-Term Philosophy):**
> The "Sovereign-first" principle prioritizes long-term resilience and independence over the short-term ease of using managed services. This aligns with Liker's observation that successful lean organizations prioritize long-term purpose.
> *   *Reference: Liker, J. K. (2004). The Toyota Way: 14 Management Principles from the World's Greatest Manufacturer. McGraw-Hill.*

1. **Sovereign-first** - Local storage default, no mandatory cloud dependency
2. **Pure Rust** - No Python, no FFI (WASM-compatible)
3. **Zero-copy** - Arrow RecordBatch throughout
4. **Ecosystem aligned** - Arrow 53, Parquet 53 (matches trueno-db, trueno-graph)

## Architecture

> **Review Note (Principle 7: Visual Control):**
> This diagram serves as excellent "Visual Control," allowing any team member to instantly understand the system boundaries and data flow. It reduces the cognitive load (Muri) on developers.
> *   *Reference: Poppendieck, M., & Poppendieck, T. (2003). Lean Software Development: An Agile Toolkit. Addison-Wesley.*

```
┌─────────────────────────────────────────────────────────────┐
│                        alimentar                            │
├─────────────────────────────────────────────────────────────┤
│  Importers          │  Core            │  Exporters         │
│  ─────────          │  ────            │  ─────────         │
│  • HuggingFace Hub  │  • Dataset       │  • Local FS        │
│  • Local files      │  • DataLoader    │  • S3-compatible   │
│  • S3-compatible    │  • Transforms    │  • Registry API    │
│  • HTTP/HTTPS       │  • Streaming     │                    │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
   trueno-db             aprender              trueno-viz
   (storage)             (ML/DL)               (WASM/browser)
```

## Core Types

### Dataset

```rust
pub trait Dataset: Send + Sync {
    fn len(&self) -> usize;
    fn get(&self, index: usize) -> Option<RecordBatch>;
    fn schema(&self) -> SchemaRef;
    fn iter(&self) -> impl Iterator<Item = RecordBatch>;
}

pub struct ArrowDataset {
    path: PathBuf,
    mmap: Option<Mmap>,           // Memory-mapped for large datasets
    batches: Vec<RecordBatch>,    // In-memory for small datasets
    schema: SchemaRef,
}

pub struct StreamingDataset {
    source: Box<dyn DataSource>,
    buffer_size: usize,
    prefetch: usize,
}
```

### DataLoader

```rust
pub struct DataLoader<D: Dataset> {
    dataset: D,
    batch_size: usize,
    shuffle: bool,
    drop_last: bool,
    num_workers: usize,          // 0 = main thread only (WASM-safe)
}

impl<D: Dataset> Iterator for DataLoader<D> {
    type Item = RecordBatch;
}
```

### Transforms

> **Review Note (Principle 2: Create Continuous Process Flow):**
> The `Transform` trait facilitates a continuous flow of data transformation, minimizing intermediate states. This is akin to a single-piece flow in manufacturing, reducing "inventory" (data sitting in memory).
> *   *Reference: Womack, J. P., & Jones, D. T. (1996). Lean Thinking. Simon & Schuster.*

```rust
pub trait Transform: Send + Sync {
    fn apply(&self, batch: RecordBatch) -> Result<RecordBatch>;
}

// Built-in transforms
pub struct Map<F>(F);
pub struct Filter<F>(F);
pub struct Shuffle { seed: Option<u64> }
pub struct Select { columns: Vec<String> }
pub struct Rename { mapping: HashMap<String, String> }
pub struct Cast { schema: SchemaRef }
pub struct Tokenize { tokenizer: TokenizerType }  // Uses aprender::text
```

## Dataset Splits (Optional)

> **Review Note (Principle 5: Build a Culture of Stopping to Fix Problems):**
> Explicit split validation prevents silent data leakage between train/test sets—a defect that would otherwise propagate through the entire ML pipeline undetected.

Splits are **optional and user-controlled**. alimentar stores one dataset per `.ald` file; splits are separate files by convention.

### Split Types

```rust
/// Optional split for ML workflows
#[derive(Debug, Clone)]
pub struct DatasetSplit {
    /// Training dataset (required)
    pub train: ArrowDataset,
    /// Test/holdout dataset (required)
    pub test: ArrowDataset,
    /// Validation dataset (optional, often carved from train during training)
    pub validation: Option<ArrowDataset>,
}

impl DatasetSplit {
    /// Create train/test split (validation carved by aprender during training)
    pub fn new(train: ArrowDataset, test: ArrowDataset) -> Self;

    /// Create train/test/validation split (user-provided validation set)
    pub fn with_validation(
        train: ArrowDataset,
        test: ArrowDataset,
        validation: ArrowDataset,
    ) -> Self;

    /// Split dataset by ratio (e.g., 0.8 train, 0.1 test, 0.1 val)
    pub fn from_ratios(
        dataset: &ArrowDataset,
        train_ratio: f64,
        test_ratio: f64,
        val_ratio: Option<f64>,
        seed: Option<u64>,
    ) -> Result<Self>;

    /// Stratified split preserving label distribution
    pub fn stratified(
        dataset: &ArrowDataset,
        label_column: &str,
        train_ratio: f64,
        test_ratio: f64,
        val_ratio: Option<f64>,
        seed: Option<u64>,
    ) -> Result<Self>;
}
```

### File Convention

```
my_dataset/
├── train.ald        # Training split
├── test.ald         # Test/holdout split
└── validation.ald   # Optional validation split
```

### Distributed/Federated Splits

> **Federated ML Use Case:** Data stays local on each node (sovereignty). Only metadata/sketches cross boundaries.

```
# Federated topology - each node owns local .ald files
Node A (EU):     data_eu.ald      → local train_eu.ald, test_eu.ald
Node B (US):     data_us.ald      → local train_us.ald, test_us.ald
Node C (APAC):   data_apac.ald    → local train_apac.ald, test_apac.ald
```

```rust
/// Federated split coordination (no raw data leaves nodes)
pub struct FederatedSplitCoordinator {
    /// Strategy for distributed splitting
    strategy: FederatedSplitStrategy,
}

#[derive(Debug, Clone)]
pub enum FederatedSplitStrategy {
    /// Each node splits locally with same seed (simple, no coordination)
    LocalWithSeed { seed: u64, train_ratio: f64 },

    /// Stratified across nodes - coordinator sees only label distributions
    GlobalStratified {
        label_column: String,
        /// Target distribution (coordinator computes from sketches)
        target_distribution: HashMap<String, f64>,
    },

    /// IID sampling - each node contributes proportionally
    ProportionalIID { train_ratio: f64 },
}

/// Per-node split manifest (shared with coordinator, no raw data)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeSplitManifest {
    pub node_id: String,
    pub total_rows: u64,
    pub train_rows: u64,
    pub test_rows: u64,
    pub validation_rows: Option<u64>,
    /// Label distribution (for stratification verification)
    pub label_distribution: Option<HashMap<String, u64>>,
    /// Hash of split indices (for reproducibility verification)
    pub split_hash: [u8; 32],
}

impl FederatedSplitCoordinator {
    /// Compute split instructions for each node (runs on coordinator)
    pub fn compute_split_plan(
        manifests: &[NodeSplitManifest],
        strategy: &FederatedSplitStrategy,
    ) -> Result<Vec<NodeSplitInstruction>>;

    /// Execute split locally (runs on each node)
    pub fn execute_local_split(
        dataset: &ArrowDataset,
        instruction: &NodeSplitInstruction,
    ) -> Result<DatasetSplit>;

    /// Verify global split quality (runs on coordinator)
    pub fn verify_global_split(
        manifests: &[NodeSplitManifest],
    ) -> Result<GlobalSplitReport>;
}
```

## Data Drift Detection

> **Review Note (Principle 5: Build a Culture of Stopping to Fix Problems):**
> Data drift is a silent killer of ML models. Detecting distribution shift at the data layer (before training) embodies Jidoka—building quality in at the source rather than discovering degraded model performance in production.

### Single Dataset Drift

Detect distribution changes between dataset versions or time periods.

```rust
/// Statistical drift detection between two datasets
pub struct DriftDetector {
    /// Reference dataset (baseline distribution)
    reference: ArrowDataset,
    /// Statistical tests to apply
    tests: Vec<DriftTest>,
    /// Significance threshold (default: 0.05)
    alpha: f64,
}

#[derive(Debug, Clone)]
pub enum DriftTest {
    /// Kolmogorov-Smirnov test for continuous features
    KolmogorovSmirnov,
    /// Chi-squared test for categorical features
    ChiSquared,
    /// Population Stability Index (PSI)
    PSI { buckets: usize },
    /// Jensen-Shannon divergence
    JensenShannon,
    /// Wasserstein distance (Earth Mover's Distance)
    Wasserstein,
}

#[derive(Debug, Clone)]
pub struct DriftReport {
    /// Per-column drift scores
    pub column_scores: HashMap<String, ColumnDrift>,
    /// Overall drift detected
    pub drift_detected: bool,
    /// Timestamp of analysis
    pub timestamp: u64,
}

#[derive(Debug, Clone)]
pub struct ColumnDrift {
    pub column: String,
    pub test: DriftTest,
    pub statistic: f64,
    pub p_value: f64,
    pub drift_detected: bool,
    /// Human-readable severity
    pub severity: DriftSeverity,
}

#[derive(Debug, Clone, Copy)]
pub enum DriftSeverity {
    None,
    Low,      // p > 0.01
    Medium,   // 0.001 < p <= 0.01
    High,     // p <= 0.001
    Critical, // Distribution fundamentally changed
}

impl DriftDetector {
    pub fn new(reference: ArrowDataset) -> Self;

    /// Compare current dataset against reference
    pub fn detect(&self, current: &ArrowDataset) -> Result<DriftReport>;

    /// Monitor streaming data for drift (sliding window)
    pub fn monitor_stream<I: Iterator<Item = RecordBatch>>(
        &self,
        stream: I,
        window_size: usize,
    ) -> impl Iterator<Item = DriftReport>;
}
```

### Distributed Data Drift

> **Review Note (Principle 11: Respect Your Extended Network):**
> In federated learning scenarios, drift detection must work across distributed nodes without centralizing sensitive data—respecting data sovereignty while maintaining quality control.

Detect drift across federated/distributed datasets without centralizing raw data.

```rust
/// Federated drift detection using sketch-based statistics
pub struct DistributedDriftDetector {
    /// Coordinator node (aggregates sketches, not raw data)
    coordinator: CoordinatorConfig,
    /// Sketch algorithm for privacy-preserving statistics
    sketch_type: SketchType,
}

#[derive(Debug, Clone)]
pub enum SketchType {
    /// Count-Min Sketch for frequency estimation
    CountMin { width: usize, depth: usize },
    /// HyperLogLog for cardinality
    HyperLogLog { precision: u8 },
    /// T-Digest for quantile estimation
    TDigest { compression: f64 },
    /// DDSketch for quantiles with error bounds
    DDSketch { relative_accuracy: f64 },
}

/// Statistics computed locally, shared with coordinator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSketch {
    /// Node identifier
    pub node_id: String,
    /// Timestamp of sketch creation
    pub timestamp: u64,
    /// Per-column sketches (no raw data)
    pub column_sketches: HashMap<String, ColumnSketch>,
    /// Row count (for weighting)
    pub row_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnSketch {
    /// Sketch bytes (algorithm-specific)
    pub sketch: Vec<u8>,
    /// Basic statistics (min, max, null_count)
    pub stats: ColumnStats,
}

impl DistributedDriftDetector {
    /// Create sketch from local dataset (runs on each node)
    pub fn create_sketch(
        dataset: &ArrowDataset,
        sketch_type: &SketchType,
    ) -> Result<DataSketch>;

    /// Merge sketches from multiple nodes (runs on coordinator)
    pub fn merge_sketches(sketches: &[DataSketch]) -> Result<DataSketch>;

    /// Compare merged sketch against reference
    pub fn detect_drift(
        reference_sketch: &DataSketch,
        current_sketch: &DataSketch,
    ) -> Result<DistributedDriftReport>;
}

#[derive(Debug, Clone)]
pub struct DistributedDriftReport {
    /// Per-column drift (estimated from sketches)
    pub column_drift: HashMap<String, SketchDrift>,
    /// Nodes contributing to current sketch
    pub participating_nodes: Vec<String>,
    /// Confidence in drift estimates (based on sketch accuracy)
    pub confidence: f64,
    /// Recommendations
    pub recommendations: Vec<DriftRecommendation>,
}

#[derive(Debug, Clone)]
pub enum DriftRecommendation {
    /// No action needed
    NoAction,
    /// Retrain model with new data
    RetrainRecommended { urgency: DriftSeverity },
    /// Investigate specific columns
    InvestigateColumns { columns: Vec<String> },
    /// Data quality issue (not drift)
    DataQualityIssue { description: String },
}
```

### CLI Commands

```bash
# === Local Splits ===
alimentar split --input data.ald --train 0.8 --test 0.2 --output ./splits/
alimentar split --input data.ald --train 0.7 --test 0.15 --val 0.15 --stratify label
alimentar split --input data.ald --train 0.8 --test 0.2 --seed 42

# === Federated Splits (run on each node) ===
# Step 1: Generate manifest (runs locally, shares only metadata)
alimentar fed manifest --input local/data.ald --output manifest.json

# Step 2: Coordinator computes split plan (sees only manifests)
alimentar fed plan --manifests node_a.json node_b.json node_c.json \
    --strategy stratified --label-column target --output plan.json

# Step 3: Each node executes its portion of the plan
alimentar fed split --input local/data.ald --plan plan.json --node-id node_a

# Step 4: Coordinator verifies global split quality
alimentar fed verify --manifests post_split_a.json post_split_b.json post_split_c.json

# === Single Dataset Drift ===
alimentar drift detect --reference v1/train.ald --current v2/train.ald
alimentar drift report --output drift-report.json

# === Distributed Drift (federated) ===
alimentar drift sketch --input local/train.ald --output sketch.json
alimentar drift merge --sketches node1.json node2.json node3.json --output merged.json
alimentar drift compare --reference baseline.json --current merged.json
```

## Storage Backends

### Backend Trait

```rust
#[async_trait]
pub trait StorageBackend: Send + Sync {
    async fn list(&self, prefix: &str) -> Result<Vec<String>>;
    async fn get(&self, key: &str) -> Result<Bytes>;
    async fn put(&self, key: &str, data: Bytes) -> Result<()>;
    async fn delete(&self, key: &str) -> Result<()>;
    async fn exists(&self, key: &str) -> Result<bool>;
}
```

### Implementations

> **Review Note (Principle 12: Genchi Genbutsu - Go and See):**
> Listing providers like "MinIO, Ceph, R2, Wasabi, OVH" implies compatibility. In the spirit of Genchi Genbutsu, we must *verify* these integrations with integration tests, not just assume the S3 API is perfectly standard across them.
> *   *Reference: Fagan, M. E. (1976). Design and Code Inspections to Reduce Errors in Program Development. IBM Systems Journal.*

| Backend | Crate | Use Case |
|---------|-------|----------|
| `LocalBackend` | std::fs | Air-gapped, on-prem, development |
| `S3Backend` | aws-sdk-s3 / rusoto | AWS, MinIO, Ceph, R2, Wasabi, OVH |
| `HttpBackend` | reqwest | Read-only HTTP/HTTPS sources |
| `MemoryBackend` | - | Testing, WASM browser cache |

### Configuration

```rust
pub enum BackendConfig {
    Local { root: PathBuf },
    S3 {
        endpoint: Option<String>,    // None = AWS, Some = MinIO/Ceph/etc
        bucket: String,
        region: String,
        credentials: CredentialSource,
    },
    Http { base_url: String },
}

pub enum CredentialSource {
    Environment,                     // AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
    File(PathBuf),                   // credentials file
    Static { key: String, secret: String },
    None,                            // Anonymous/public
}
```

## HuggingFace Hub Integration

### Import from HF Hub

> **Review Note (Principle 3: Use "Pull" Systems):**
> The `stream` functionality allows downstream processes (like `aprender` training loops) to "pull" data only as needed, preventing overproduction (downloading unnecessary data).
> *   *Reference: Rother, M., & Shook, J. (1999). Learning to See. Lean Enterprise Institute.*

```rust
pub struct HuggingFaceImporter {
    token: Option<String>,           // HF_TOKEN for private datasets
    cache_dir: PathBuf,
}

impl HuggingFaceImporter {
    /// Download and convert HF dataset to local Arrow format
    pub async fn import(
        &self,
        repo_id: &str,               // e.g., "squad", "glue/mrpc"
        split: Option<&str>,
        revision: Option<&str>,
    ) -> Result<ArrowDataset>;

    /// Stream without full download
    pub async fn stream(
        &self,
        repo_id: &str,
        split: Option<&str>,
    ) -> Result<StreamingDataset>;
}
```

### Supported HF Formats

- Parquet (native, zero conversion)
- JSON/JSONL → Arrow
- CSV → Arrow
- Arrow/IPC (native)
- Safetensors (tensor data)
- Images (PNG/JPEG → binary column)
- Audio (WAV/MP3 → binary column)

## Dataset Registry (Sharing)

### Registry API

```rust
pub struct Registry {
    backend: Box<dyn StorageBackend>,
    index_path: String,              // registry-index.json
}

impl Registry {
    /// Publish dataset to registry
    pub async fn publish(
        &self,
        name: &str,
        version: &str,
        dataset: &ArrowDataset,
        metadata: DatasetMetadata,
    ) -> Result<()>;

    /// List available datasets
    pub async fn list(&self) -> Result<Vec<DatasetInfo>>;

    /// Download dataset from registry
    pub async fn pull(
        &self,
        name: &str,
        version: Option<&str>,
    ) -> Result<ArrowDataset>;
}

pub struct DatasetMetadata {
    pub description: String,
    pub license: String,
    pub tags: Vec<String>,
    pub source: Option<String>,      // Original source URL
    pub citation: Option<String>,
}
```

### Registry Index Format

```json
{
  "version": "1.0",
  "datasets": [
    {
      "name": "mnist",
      "versions": ["1.0.0", "1.0.1"],
      "latest": "1.0.1",
      "description": "Handwritten digit recognition",
      "license": "CC-BY-SA-3.0",
      "size_bytes": 11490000,
      "num_rows": 70000,
      "schema": { ... }
    }
  ]
}
```

### Sovereign Deployment Examples

```bash
# Local registry (air-gapped)
alimentar registry init ./my-datasets

# MinIO (on-prem S3)
alimentar registry init s3://datasets --endpoint http://minio.local:9000

# Scaleway (EU)
alimentar registry init s3://datasets --endpoint s3.fr-par.scw.cloud

# OVH (EU)
alimentar registry init s3://datasets --endpoint s3.gra.cloud.ovh.net
```

## WASM Support

> **Review Note (Principle 8: Use Only Reliable Technology):**
> While WASM is essential for the `trueno-viz` goal, the constraints (no threads, no FS) introduce significant complexity. Ensure we have specific, isolated tests for these constraints to prevent "defects" from escaping to the browser environment.
> *   *Reference: Brooks, F. P. (1975). The Mythical Man-Month. Addison-Wesley.*

### Feature Flags

```toml
[features]
default = ["local", "tokio-runtime"]
local = []                           # Local filesystem
s3 = ["aws-sdk-s3"]                  # S3-compatible backends
http = ["reqwest"]                   # HTTP sources
hf-hub = ["http"]                    # HuggingFace Hub import
tokio-runtime = ["tokio"]            # Async runtime (non-WASM)
wasm = ["wasm-bindgen", "js-sys"]    # Browser/WASM target
```

### WASM Constraints

- No filesystem access → use `MemoryBackend` or `HttpBackend`
- No multi-threading → `num_workers = 0`
- No tokio → use `wasm-bindgen-futures`
- IndexedDB for caching (optional)

```rust
#[cfg(target_arch = "wasm32")]
pub struct WasmDataLoader {
    // Single-threaded, memory-based
}
```

## Search

### Search Architecture

alimentar owns **registry metadata search**. Content and semantic search delegate to trueno-db.

> **Review Note (Separation of Concerns):**
> This delegation strategy is excellent. It avoids "Muda" (re-inventing the wheel) by leveraging `trueno-db`'s existing capabilities for heavy lifting.
> *   *Reference: Bass, L., Clements, P., & Kazman, R. (2012). Software Architecture in Practice. Addison-Wesley.*

| Search Type | Owner | Implementation |
|-------------|-------|----------------|
| Registry metadata | alimentar | Text/tag matching on index |
| SQL/filter | trueno-db | Arrow predicates, SQL engine |
| Vector/semantic | trueno-db | HNSW index, embeddings |

### Registry Search (alimentar)

```rust
impl Registry {
    /// Full-text search on dataset metadata
    pub async fn search(&self, query: &str) -> Result<Vec<DatasetInfo>>;

    /// Tag-based filtering
    pub async fn search_tags(&self, tags: &[&str]) -> Result<Vec<DatasetInfo>>;

    /// Combined search with filters
    pub async fn search_filtered(
        &self,
        query: Option<&str>,
        tags: Option<&[&str]>,
        license: Option<&str>,
        min_rows: Option<usize>,
    ) -> Result<Vec<DatasetInfo>>;
}
```

### Content Search (delegate to trueno-db)

```rust
use alimentar::ArrowDataset;
use trueno_db::Database;

let dataset = ArrowDataset::open("./data.parquet")?;
let db = Database::new()?;
db.insert_batch("data", dataset.to_batch()?)?;

// SQL search
let results = db.query("SELECT * FROM data WHERE category = 'rust'")?;

// Predicate pushdown (returns filtered Arrow batches)
let filtered = db.scan("data")
    .filter(col("score").gt(0.9))
    .select(["id", "text"])
    .collect()?;
```

### Semantic Search (delegate to trueno-db)

```rust
use alimentar::ArrowDataset;
use aprender::text::SentenceEmbedder;
use trueno_db::{Database, HnswIndex};

// 1. Load dataset
let dataset = ArrowDataset::open("./documents.parquet")?;

// 2. Compute embeddings (aprender)
let embedder = SentenceEmbedder::load("all-MiniLM-L6-v2")?;
let embeddings = embedder.encode_batch(dataset.column("text")?)?;

// 3. Build index (trueno-db)
let db = Database::new()?;
db.create_vector_index("docs", embeddings, HnswIndex::default())?;

// 4. Search
let query_vec = embedder.encode("machine learning in rust")?;
let results = db.vector_search("docs", query_vec, top_k=10)?;
```

### Convenience API

alimentar provides a thin wrapper for common patterns:

```rust
impl ArrowDataset {
    /// Load into trueno-db for querying
    pub fn to_database(&self) -> Result<trueno_db::Database>;

    /// Create searchable index with embeddings
    pub fn with_embeddings<E: Embedder>(
        &self,
        embedder: E,
        column: &str,
    ) -> Result<SearchableDataset>;
}

pub struct SearchableDataset {
    dataset: ArrowDataset,
    db: trueno_db::Database,
    index_name: String,
}

impl SearchableDataset {
    pub fn search(&self, query: &str, top_k: usize) -> Result<Vec<SearchResult>>;
    pub fn sql(&self, query: &str) -> Result<RecordBatch>;
}
```

## Integration with Sovereign Stack

### trueno-db Integration

```rust
use alimentar::{ArrowDataset, DataLoader};
use trueno_db::Database;

let dataset = ArrowDataset::open("./data/train.parquet")?;
let db = Database::new()?;

// Zero-copy insert
for batch in dataset.iter() {
    db.insert_batch("training_data", batch)?;
}
```

### aprender Integration

```rust
use alimentar::{DataLoader, transforms::Shuffle};
use aprender::nn::Module;

let dataset = ArrowDataset::open("./mnist")?
    .with_transform(Shuffle::new(42));

let loader = DataLoader::new(dataset)
    .batch_size(32)
    .shuffle(true);

for batch in loader {
    let x = batch.column("pixels").to_tensor()?;  // → trueno::Tensor
    let y = batch.column("label").to_tensor()?;
    model.train_step(x, y)?;
}
```

### trueno-viz Integration (WASM)

```rust
// Browser-based visualization with live data
use alimentar::wasm::WasmDataLoader;
use trueno_viz::Chart;

let loader = WasmDataLoader::from_url("https://data.example.com/dataset.parquet").await?;
let batch = loader.get(0)?;
Chart::scatter(batch.column("x"), batch.column("y")).render()?;
```

## CLI

```bash
# Import from HuggingFace
alimentar import hf squad --output ./data/squad

# Convert formats
alimentar convert data.csv data.parquet
alimentar convert data.json data.arrow

# Registry operations
alimentar registry list
alimentar registry push my-dataset ./data --version 1.0.0
alimentar registry pull my-dataset --version 1.0.0

# Inspect dataset
alimentar info ./data/train.parquet
alimentar head ./data/train.parquet --rows 10
alimentar schema ./data/train.parquet
```

## Dependencies

> **Review Note (Reduce Waste/Muda):**
> Careful management of feature flags here is critical. Unnecessary dependencies increase build times and binary sizes (waste). The modular feature gating seen here is a good countermeasure.
> *   *Reference: Martin, R. C. (2008). Clean Code: A Handbook of Agile Software Craftsmanship. Prentice Hall.*

```toml
[dependencies]
# Core (always)
arrow = "53"
parquet = { version = "53", default-features = false, features = ["arrow"] }
thiserror = "2"
bytes = "1"

# Async (feature-gated)
tokio = { version = "1", features = ["rt", "fs", "sync"], optional = true }

# S3 (feature-gated)
aws-sdk-s3 = { version = "1", optional = true }
aws-config = { version = "1", optional = true }

# HTTP (feature-gated)
reqwest = { version = "0.12", default-features = false, features = ["rustls-tls"], optional = true }

# WASM (feature-gated)
wasm-bindgen = { version = "0.2", optional = true }
wasm-bindgen-futures = { version = "0.4", optional = true }
js-sys = { version = "0.3", optional = true }
```

## Quality Standards (EXTREME TDD)

> **Review Note (Principle 5: Jidoka - Build Quality In):**
> The use of strict gates like "Zero unwrap()" and "Mutation score >= 85%" is an extreme but effective form of Jidoka. It automatically stops the line (the build) when quality drops, preventing defects from moving downstream.
> *   *Reference: Beck, K. (2002). Test Driven Development: By Example. Addison-Wesley.*

Aligned with trueno and trueno-db quality gates.

### Metrics Targets

| Metric | Target | Enforcement |
|--------|--------|-------------|
| Test coverage | ≥90% | `make coverage` (blocks CI) |
| Mutation score | ≥85% | `cargo mutants` |
| Cyclomatic complexity | ≤15 | pmat analyze |
| SATD comments | 0 | pmat (zero tolerance) |
| unwrap() calls | 0 | clippy disallowed-methods |
| TDG grade | ≥B+ | pmat tdg |
| WASM binary | <500KB | CI check |

### Quality Gate Commands

```bash
# Tier 1: On-Save (<1s)
cargo fmt --check && cargo clippy -- -D warnings && cargo check

# Tier 2: Pre-Commit (<5s)
make check  # lint + test

# Tier 3: Pre-Push (1-5 min)
make quality-gate  # lint + test + coverage (blocks if <90%)

# Tier 4: CI/CD
make mutants       # Mutation testing
pmat tdg           # Technical debt grading
pmat rust-project-score
```

### PMAT Integration

```bash
# Quality gates (pre-commit)
pmat quality-gate

# Full analysis
pmat analyze complexity
pmat analyze satd
pmat rust-project-score

# Scaffold enforcement
pmat enforce
```

### Configuration Files

| File | Purpose |
|------|---------|
| `.pmat-gates.toml` | Quality gate thresholds |
| `.cargo-mutants.toml` | Mutation testing config |
| `deny.toml` | Dependency policy |
| `renacer.toml` | Deep inspection config |

## Documentation (mdBook)

alimentar includes a comprehensive mdBook at `book/`:

```
book/
├── book.toml
└── src/
    ├── SUMMARY.md
    ├── introduction.md
    ├── getting-started/
    │   ├── installation.md
    │   ├── quickstart.md
    │   └── configuration.md
    ├── core-concepts/
    │   ├── datasets.md
    │   ├── dataloaders.md
    │   ├── transforms.md
    │   └── streaming.md
    ├── storage/
    │   ├── local.md
    │   ├── s3.md
    │   └── registry.md
    ├── integrations/
    │   ├── huggingface.md
    │   ├── trueno-db.md
    │   ├── aprender.md
    │   └── trueno-viz.md
    ├── sovereign/
    │   ├── air-gapped.md
    │   ├── eu-providers.md
    │   └── self-hosted.md
    ├── wasm/
    │   ├── browser.md
    │   └── api-reference/
    │       └── ...
    └── appendix/
        ├── changelog.md
        └── migration.md
```

### Build Book

```bash
cd book && mdbook build
mdbook serve  # Local preview at localhost:3000
```

## Project Structure

> **Review Note (Principle 6: Standardized Work):**
> The use of `pmat scaffold` implies standardized work. Standardized work forms the baseline for Kaizen (continuous improvement). Without a standard, you cannot improve.
> *   *Reference: Humble, J., & Farley, D. (2010). Continuous Delivery: Reliable Software Releases through Build, Test, and Deployment Automation. Addison-Wesley.*

```
alimentar/
├── Cargo.toml
├── Cargo.lock
├── LICENSE
├── README.md
├── CLAUDE.md                    # Claude Code guidance
├── CHANGELOG.md
├── Makefile                     # Quality gate commands
│
├── .pmat-gates.toml             # PMAT quality thresholds
├── .cargo-mutants.toml          # Mutation testing config
├── deny.toml                    # Dependency policy
├── renacer.toml                 # Deep inspection
│
├── .github/
│   └── workflows/
│       ├── ci.yml               # Main CI (test, lint, coverage)
│       ├── security.yml         # cargo-audit, cargo-deny
│       └── release.yml          # crates.io publish
│
├── book/                        # mdBook documentation
│   ├── book.toml
│   └── src/
│       └── ...
│
├── docs/
│   └── specifications/
│       └── alimentar-spec-v1.md # This file
│
├── src/
│   ├── lib.rs                   # Library entry
│   ├── dataset.rs               # Dataset trait + ArrowDataset
│   ├── dataloader.rs            # DataLoader iterator
│   ├── transform.rs             # Transform trait + builtins
│   ├── streaming.rs             # StreamingDataset
│   ├── backend/
│   │   ├── mod.rs
│   │   ├── local.rs             # LocalBackend
│   │   ├── s3.rs                # S3Backend
│   │   ├── http.rs              # HttpBackend
│   │   └── memory.rs            # MemoryBackend (WASM)
│   ├── registry/
│   │   ├── mod.rs
│   │   ├── index.rs             # Registry index
│   │   └── publish.rs           # Push/pull operations
│   ├── import/
│   │   ├── mod.rs
│   │   ├── huggingface.rs       # HF Hub importer
│   │   ├── csv.rs
│   │   ├── json.rs
│   │   └── safetensors.rs
│   ├── search.rs                # SearchableDataset wrapper
│   ├── wasm.rs                  # WASM-specific types
│   └── error.rs                 # Error types
│
├── examples/
│   ├── basic_usage.rs
│   ├── s3_registry.rs
│   ├── huggingface_import.rs
│   └── wasm_demo.rs
│
├── benches/
│   ├── loading.rs               # Load performance
│   ├── transform.rs             # Transform throughput
│   └── vs_polars.rs             # Competitive benchmarks
│
└── tests/
    ├── integration.rs
    ├── backend_tests.rs
    └── property_tests.rs
```

## Scaffold Commands

```bash
# Initialize project with pmat
cd ~/src/alimentar
pmat scaffold project rust -t makefile,readme,gitignore

# Generate quality config
pmat generate .pmat-gates.toml
pmat generate deny.toml

# Verify scaffold
pmat diagnose
```

## Roadmap

> **Review Note (Principle 4: Heijunka - Level Out the Workload):**
> The roadmap is well-structured, but ensure that "Quality Gate" checks don't become bottlenecks at the end of versions. Integrate these checks continuously (CI) to avoid a "crunch" phase, smoothing the workload.
> *   *Reference: Reinertsen, D. G. (2009). The Principles of Product Development Flow: Second Generation Lean Product Development. Celeritas Publishing.*

### v0.1.0 - Core (Quality Gate: 90% coverage)
- [ ] ArrowDataset (memory-mapped)
- [ ] DataLoader (batching, shuffling)
- [ ] Local storage backend
- [ ] Basic transforms (map, filter, select)
- [ ] CLI: convert, info, head
- [ ] Makefile + quality gates
- [ ] CLAUDE.md
- [ ] 90% test coverage

### v0.2.0 - Storage (Quality Gate: +registry tests)
- [ ] S3-compatible backend
- [ ] HTTP backend
- [ ] Registry publish/pull
- [x] Streaming datasets
- [ ] Integration tests
- [ ] **S3 Provider Test Matrix (Genchi Genbutsu)**:
  - [ ] MinIO (on-prem reference)
  - [ ] Ceph/RadosGW
  - [ ] Cloudflare R2
  - [ ] Scaleway (EU)
  - [ ] OVH (EU)
  - [ ] Wasabi
  - [ ] Backblaze B2

### v0.3.0 - HuggingFace (Quality Gate: +import tests)
- [ ] HF Hub importer
- [ ] Safetensors support
- [ ] Image/audio loading
- [ ] Property-based tests

### v0.4.0 - WASM (Quality Gate: binary size <500KB)
- [ ] WASM target
- [ ] MemoryBackend
- [ ] trueno-viz integration
- [ ] Browser examples

### v0.5.0 - Book
- [ ] mdBook complete
- [ ] API reference
- [ ] Sovereign deployment guide

### v0.6.0 - Data Quality (Quality Gate: drift detection tests)
- [x] Dataset splits (train/test/validation optional)
- [x] Stratified splitting
- [x] Single dataset drift detection (KS, Chi-squared, PSI, Jensen-Shannon)
- [x] Imbalance detection (metrics, severity, recommendations)
- [x] Data quality checking (nulls, outliers, duplicates, constants)
- [x] Distributed drift (sketch-based) - TDigest, DDSketch, DistributedDriftDetector
- [x] Federated split coordination (FederatedSplitCoordinator, NodeSplitManifest)
- [x] CLI: `alimentar drift detect`
- [x] CLI: `alimentar quality check`
- [x] CLI: `alimentar drift sketch/merge/compare`
- [x] CLI: `alimentar fed manifest/plan/split/verify`

### v1.0.0 - Production (Quality Gate: all metrics green)
- [ ] ≥90% test coverage
- [ ] ≥85% mutation score
- [ ] Zero unwrap()
- [ ] TDG grade A
- [ ] Benchmarks vs polars/HF datasets
- [ ] crates.io publish

## References

- [HuggingFace Datasets Architecture](https://huggingface.co/docs/datasets/about_arrow)
- [Apache Arrow Rust](https://github.com/apache/arrow-rs)
- [trueno-db](https://github.com/paiml/trueno-db)
- [Sovereign AI Stack Book](https://github.com/paiml/sovereign-ai-stack-book)