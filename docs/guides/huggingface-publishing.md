# Publishing Datasets to HuggingFace Hub

This guide covers how to publish datasets to HuggingFace Hub using alimentar's `HfPublisher` API.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Authentication](#authentication)
4. [Basic Publishing](#basic-publishing)
5. [Publishing Arrow Data](#publishing-arrow-data)
6. [Publishing from Registry](#publishing-from-registry)
7. [CLI Usage](#cli-usage)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)
10. [API Reference](#api-reference)

---

## Overview

HuggingFace Hub is the leading platform for sharing ML datasets and models. alimentar provides native Rust support for:

- **Downloading** datasets from HuggingFace (via `HfDataset`)
- **Publishing** datasets to HuggingFace (via `HfPublisher`)

The publishing workflow:

```
Local Data → Arrow/Parquet → HfPublisher → HuggingFace Hub
```

### Why Publish to HuggingFace?

- **Discoverability**: Datasets are searchable and browsable
- **Versioning**: Automatic git-based version control
- **Streaming**: Consumers can stream without full download
- **Dataset Viewer**: Automatic preview in the browser
- **Community**: Comments, discussions, model links

---

## Prerequisites

### Cargo Features

Enable the required features in your `Cargo.toml`:

```toml
[dependencies]
alimentar = { version = "0.2", features = ["http", "tokio-runtime", "hf-hub"] }
```

Feature breakdown:
- `http`: HTTP client for API calls (reqwest)
- `tokio-runtime`: Async runtime for sync wrappers
- `hf-hub`: HuggingFace Hub integration

### HuggingFace Account

1. Create account at [huggingface.co](https://huggingface.co)
2. Generate API token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Choose "Write" access for publishing

---

## Authentication

### Environment Variable (Recommended)

```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

The `HfPublisher` automatically reads this:

```rust
use alimentar::hf_hub::HfPublisher;

// Automatically uses HF_TOKEN from environment
let publisher = HfPublisher::new("username/my-dataset");
```

### Explicit Token

```rust
let publisher = HfPublisher::new("username/my-dataset")
    .with_token("hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx");
```

### CLI Login

```bash
huggingface-cli login
# Stores token in ~/.cache/huggingface/token
```

---

## Basic Publishing

### Step 1: Create the Publisher

```rust
use alimentar::hf_hub::HfPublisher;

let publisher = HfPublisher::new("paiml/my-dataset")
    .with_private(false)  // Public dataset
    .with_commit_message("Initial upload");
```

### Step 2: Create the Repository

```rust
// Creates the dataset repo if it doesn't exist
publisher.create_repo_sync()?;
```

### Step 3: Upload Files

```rust
// Upload raw bytes
let data = std::fs::read("data/train.parquet")?;
publisher.upload_file_sync("data/train.parquet", &data)?;

// Or upload a local file directly
use std::path::Path;
publisher.upload_parquet_file_sync(
    Path::new("local/train.parquet"),
    "data/train.parquet"  // Path in repo
)?;
```

### Complete Example

```rust
use alimentar::hf_hub::HfPublisher;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create publisher
    let publisher = HfPublisher::new("myorg/citl-corpus")
        .with_private(false)
        .with_commit_message("Upload CITL training corpus");

    // Create repo (idempotent - safe to call if exists)
    publisher.create_repo_sync()?;

    // Upload parquet file
    publisher.upload_parquet_file_sync(
        Path::new("./data/corpus.parquet"),
        "data/train.parquet"
    )?;

    println!("Published to https://huggingface.co/datasets/myorg/citl-corpus");
    Ok(())
}
```

---

## Publishing Arrow Data

Convert Arrow RecordBatches directly to parquet and upload:

```rust
use alimentar::hf_hub::HfPublisher;
use arrow::array::{StringArray, Int32Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create schema
    let schema = Arc::new(Schema::new(vec![
        Field::new("text", DataType::Utf8, false),
        Field::new("label", DataType::Int32, false),
    ]));

    // Create data
    let texts = StringArray::from(vec!["hello", "world", "test"]);
    let labels = Int32Array::from(vec![0, 1, 0]);

    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(texts), Arc::new(labels)]
    )?;

    // Publish directly from RecordBatch
    let publisher = HfPublisher::new("myorg/text-classification");
    publisher.create_repo().await?;
    publisher.upload_batch("data/train.parquet", &batch).await?;

    Ok(())
}
```

---

## Publishing from Registry

If you're using alimentar's local registry, publish directly:

```rust
use alimentar::registry::Registry;
use alimentar::hf_hub::HfPublisher;

fn publish_from_registry() -> Result<(), Box<dyn std::error::Error>> {
    // Open local registry
    let registry = Registry::open("./my-registry")?;

    // Get dataset
    let dataset = registry.get("my-dataset", "1.0.0")?;

    // Export to parquet
    let temp_file = tempfile::NamedTempFile::new()?;
    dataset.to_parquet(temp_file.path())?;

    // Publish to HuggingFace
    let publisher = HfPublisher::new("myorg/my-dataset");
    publisher.create_repo_sync()?;
    publisher.upload_parquet_file_sync(
        temp_file.path(),
        "data/train.parquet"
    )?;

    Ok(())
}
```

---

## CLI Usage

### Using huggingface-cli

```bash
# Login (one-time)
huggingface-cli login

# Upload file
huggingface-cli upload myorg/my-dataset ./data.parquet data/train.parquet \
    --repo-type dataset \
    --commit-message "Add training data"

# Upload directory
huggingface-cli upload myorg/my-dataset ./data/ data/ \
    --repo-type dataset
```

### Using alimentar CLI (planned)

```bash
# Future CLI integration
alimentar hf-push myorg/my-dataset ./data/train.parquet
alimentar hf-push myorg/my-dataset --from-registry my-local-dataset
```

---

## Best Practices

### 1. Use Descriptive Repository Names

```
myorg/task-domain-version

Examples:
- paiml/depyler-citl
- acme/sentiment-analysis-v2
- research/imagenet-subset-10k
```

### 2. Include a Dataset Card

Create `README.md` in your repository:

```markdown
---
license: mit
task_categories:
  - text-classification
language:
  - en
tags:
  - custom-tag
size_categories:
  - 1K<n<10K
---

# My Dataset

Description of your dataset...

## Schema

- `text`: Input text
- `label`: Classification label

## Usage

\`\`\`python
from datasets import load_dataset
ds = load_dataset("myorg/my-dataset")
\`\`\`
```

### 3. Use Standard Splits

```
data/
  train.parquet
  validation.parquet
  test.parquet
```

### 4. Compress with Zstd

alimentar uses zstd compression by default for parquet:

```rust
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;

let props = WriterProperties::builder()
    .set_compression(Compression::ZSTD(Default::default()))
    .build();
```

### 5. Version Your Datasets

Use branches or tags for versions:

```bash
# Create version branch
huggingface-cli upload myorg/my-dataset ./data data/ \
    --repo-type dataset \
    --revision v2.0.0
```

---

## Troubleshooting

### Error: 401 Unauthorized

```
Solution: Check HF_TOKEN is set correctly
- Verify token has "Write" permissions
- Try: huggingface-cli whoami
```

### Error: 409 Conflict (repo already exists)

```
This is normal - create_repo() is idempotent.
The error is suppressed internally.
```

### Error: 413 Payload Too Large

```
Solution: HuggingFace has file size limits
- Split large files into chunks
- Use LFS for files >10MB (automatic for parquet)
```

### Error: Rate Limited

```
Solution: Add delays between uploads
- HuggingFace allows ~100 requests/minute
- Use exponential backoff for retries
```

### Async vs Sync

```rust
// Async (in async context)
publisher.upload_file("path", &data).await?;

// Sync (blocking, creates runtime)
publisher.upload_file_sync("path", &data)?;
```

---

## API Reference

### HfPublisher

```rust
pub struct HfPublisher {
    repo_id: String,
    token: Option<String>,
    private: bool,
    commit_message: String,
}

impl HfPublisher {
    /// Create new publisher for repository
    pub fn new(repo_id: impl Into<String>) -> Self;

    /// Set API token (default: HF_TOKEN env var)
    pub fn with_token(self, token: impl Into<String>) -> Self;

    /// Set private visibility (default: false)
    pub fn with_private(self, private: bool) -> Self;

    /// Set commit message (default: "Upload via alimentar")
    pub fn with_commit_message(self, message: impl Into<String>) -> Self;

    /// Get repository ID
    pub fn repo_id(&self) -> &str;

    // Async methods (require `http` feature)

    /// Create repository on HuggingFace Hub
    pub async fn create_repo(&self) -> Result<()>;

    /// Upload raw bytes to path in repository
    pub async fn upload_file(&self, path_in_repo: &str, data: &[u8]) -> Result<()>;

    /// Upload Arrow RecordBatch as parquet
    pub async fn upload_batch(&self, path_in_repo: &str, batch: &RecordBatch) -> Result<()>;

    /// Upload local parquet file
    pub async fn upload_parquet_file(&self, local_path: &Path, path_in_repo: &str) -> Result<()>;

    // Sync methods (require `http` + `tokio-runtime` features)

    pub fn create_repo_sync(&self) -> Result<()>;
    pub fn upload_file_sync(&self, path_in_repo: &str, data: &[u8]) -> Result<()>;
    pub fn upload_parquet_file_sync(&self, local_path: &Path, path_in_repo: &str) -> Result<()>;
}
```

### HfPublisherBuilder

```rust
pub struct HfPublisherBuilder {
    // ...
}

impl HfPublisherBuilder {
    pub fn new(repo_id: impl Into<String>) -> Self;
    pub fn token(self, token: impl Into<String>) -> Self;
    pub fn private(self, private: bool) -> Self;
    pub fn commit_message(self, message: impl Into<String>) -> Self;
    pub fn build(self) -> HfPublisher;
}
```

---

## Examples

### CITL Corpus Publishing

```rust
//! Publish CITL training corpus to HuggingFace
use alimentar::hf_hub::HfPublisher;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let publisher = HfPublisher::new("paiml/depyler-citl")
        .with_commit_message("CITL corpus v1.0: 606 Python→Rust pairs");

    publisher.create_repo_sync()?;

    // Upload train split
    publisher.upload_parquet_file_sync(
        Path::new("data/train.parquet"),
        "data/train.parquet"
    )?;

    // Upload dataset card
    let readme = std::fs::read("data/README.md")?;
    publisher.upload_file_sync("README.md", &readme)?;

    println!("Published: https://huggingface.co/datasets/paiml/depyler-citl");
    Ok(())
}
```

### Multi-Split Dataset

```rust
use alimentar::hf_hub::HfPublisher;
use std::path::Path;

async fn publish_splits() -> Result<(), Box<dyn std::error::Error>> {
    let publisher = HfPublisher::new("myorg/nlp-dataset");
    publisher.create_repo().await?;

    // Upload all splits
    for split in ["train", "validation", "test"] {
        let local = format!("data/{}.parquet", split);
        let remote = format!("data/{}.parquet", split);
        publisher.upload_parquet_file(Path::new(&local), &remote).await?;
    }

    Ok(())
}
```

---

## See Also

- [HuggingFace Datasets Documentation](https://huggingface.co/docs/datasets)
- [Dataset Card Specification](https://huggingface.co/docs/hub/datasets-cards)
- [alimentar HfDataset (downloading)](./huggingface-importing.md)
- [Local Registry Guide](./registry.md)
