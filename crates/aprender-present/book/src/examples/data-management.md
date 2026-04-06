# Data Management

Tools for model versioning, data lineage tracking, and batch data operations.

## Data Management Examples

| Type | Use Case | Example |
|------|----------|---------|
| Version History | Model versioning | `apr_version_history` |
| Lineage | Data provenance | `ald_lineage` |
| Batch Upload | File validation | `ald_batch_upload` |

## Model Version History (APR-009)

Track and compare ML model versions:

```rust
// From apr_version_history.rs
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VersionId {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

pub struct ModelVersion {
    pub version: VersionId,
    pub status: VersionStatus,
    pub metrics: HashMap<String, f64>,
    pub parent_version: Option<VersionId>,
}

pub struct VersionHistory {
    model_name: String,
    versions: Vec<ModelVersion>,
}

impl VersionHistory {
    pub fn compare(&self, v1: &VersionId, v2: &VersionId) -> Option<VersionComparison> {
        // Compare metrics between versions
    }

    pub fn lineage(&self, id: &VersionId) -> Vec<&ModelVersion> {
        // Get ancestry chain for a version
    }

    pub fn production_version(&self) -> Option<&ModelVersion> {
        self.versions.iter()
            .find(|v| v.status == VersionStatus::Production)
    }
}
```

### Version Status Flow

```
Development → Staging → Production → Deprecated → Archived
```

### Version Comparison

```rust
pub struct MetricChange {
    pub name: String,
    pub old_value: Option<f64>,
    pub new_value: Option<f64>,
    pub change_percent: Option<f64>,
}

impl MetricChange {
    pub fn is_improvement(&self, higher_is_better: bool) -> bool {
        match (self.old_value, self.new_value) {
            (Some(old), Some(new)) => {
                if higher_is_better { new > old }
                else { new < old }
            }
            _ => false,
        }
    }
}
```

Run: `cargo run --example apr_version_history`

## Dataset Lineage (ALD-007)

Track data provenance and transformations:

```rust
// From ald_lineage.rs
pub enum TransformationType {
    Source,      // Original data source
    Filter,      // Row filtering
    Map,         // Column transformation
    Join,        // Merge datasets
    Aggregate,   // Group and aggregate
    Split,       // Train/test split
    Normalize,   // Normalization
}

pub struct LineageNode {
    pub id: String,
    pub name: String,
    pub transformation: TransformationType,
    pub input_ids: Vec<String>,
    pub output_count: Option<usize>,
}

pub struct LineageGraph {
    nodes: HashMap<String, LineageNode>,
}

impl LineageGraph {
    pub fn upstream(&self, id: &str) -> Vec<&LineageNode> {
        // Get all upstream dependencies recursively
    }

    pub fn downstream(&self, id: &str) -> Vec<&LineageNode> {
        // Get all downstream dependents recursively
    }

    pub fn path(&self, from: &str, to: &str) -> Option<Vec<&LineageNode>> {
        // Find transformation path between nodes
    }
}
```

### Lineage Graph Example

```
raw-tweets ──► filtered ──► cleaned ──┐
                                      ├──► combined ──► normalized ──┬──► train
raw-reviews ──────────────────────────┘                              └──► test
```

### Transformation Types

| Type | Icon | Description |
|------|------|-------------|
| Source | ◆ | Original data source |
| Filter | ▽ | Row filtering |
| Map | ◇ | Column transformation |
| Join | ⊕ | Merge datasets |
| Split | ⊘ | Train/test split |
| Normalize | ≡ | Normalization |

Run: `cargo run --example ald_lineage`

## Batch Upload Preview (ALD-009)

File upload validation and preview:

```rust
// From ald_batch_upload.rs
pub enum UploadStatus {
    Pending,
    Validating,
    Valid,
    Invalid,
    Uploading,
    Complete,
    Failed,
}

pub struct UploadFile {
    pub name: String,
    pub size_bytes: usize,
    pub mime_type: String,
    pub status: UploadStatus,
    pub row_count: Option<usize>,
    pub errors: Vec<ValidationError>,
}

pub struct BatchUpload {
    files: Vec<UploadFile>,
    schema: Vec<SchemaColumn>,
    max_file_size: usize,
    allowed_types: Vec<String>,
}

impl BatchUpload {
    pub fn validate_file(&self, file_idx: usize) -> Vec<ValidationError> {
        // Validates file size, type, and schema compliance
    }

    pub fn can_upload(&self) -> bool {
        !self.files.is_empty()
            && self.files.iter().all(|f| f.is_valid())
    }
}
```

### Type Inference

```rust
pub struct UploadPreview {
    pub columns: Vec<String>,
    pub sample_rows: Vec<Vec<String>>,
    pub type_inference: HashMap<String, DataType>,
}

fn infer_types(columns: &[String], rows: &[Vec<String>]) -> HashMap<String, DataType> {
    // Automatically infers Integer, Float, Boolean, String types
}
```

### Validation Errors

| Severity | Icon | Action |
|----------|------|--------|
| Warning | ⚠ | Proceed with caution |
| Error | ✗ | Must fix before upload |

Run: `cargo run --example ald_batch_upload`

## YAML Configuration

### Model Card

```yaml
app:
  name: "Model Card Viewer"

data:
  model:
    source: "model.apr"

widgets:
  - type: ModelCard
    model: "{{ model }}"
    show_metrics: true
    show_lineage: true
```

### Dataset Card

```yaml
data:
  dataset:
    source: "dataset.ald"

widgets:
  - type: DataCard
    dataset: "{{ dataset }}"
    show_schema: true
    show_statistics: true
```

## Test Coverage

| Example | Tests | Coverage |
|---------|-------|----------|
| apr_version_history | 10 | Versions, comparison, lineage |
| ald_lineage | 8 | Graph, upstream/downstream, paths |
| ald_batch_upload | 9 | Validation, preview, type inference |

## Verified Test

```rust
#[test]
fn test_version_lineage() {
    let mut history = VersionHistory::new("test");
    history.add_version(ModelVersion::new(VersionId::new(1, 0, 0), "a", "v1"));
    history.add_version(
        ModelVersion::new(VersionId::new(2, 0, 0), "b", "v2")
            .with_parent(VersionId::new(1, 0, 0)),
    );
    history.add_version(
        ModelVersion::new(VersionId::new(3, 0, 0), "c", "v3")
            .with_parent(VersionId::new(2, 0, 0)),
    );

    let lineage = history.lineage(&VersionId::new(3, 0, 0));
    assert_eq!(lineage.len(), 3);
    assert_eq!(lineage[0].version, VersionId::new(3, 0, 0));
    assert_eq!(lineage[2].version, VersionId::new(1, 0, 0));
}
```
