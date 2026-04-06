# HuggingFace Hub (Examples 76-85)

This section covers HuggingFace Hub integration.

## Examples 76-77: Dataset Download and Card Validation

```rust
use alimentar::hf_hub::HfDataset;

// Download dataset
let dataset = HfDataset::builder("username/dataset")
    .revision("main")
    .split("train")
    .build()?
    .download()?;

// With card validation
let hf = HfDataset::builder("username/dataset")
    .validate_card(true)
    .build()?;
let validation = hf.validate_dataset_card()?;
```

## Examples 78-79: Quality Score and README Generation

```rust
use alimentar::hf_hub::{HfDataset, DatasetCardValidator};

let hf = HfDataset::builder("username/dataset").build()?;

// Get quality score
let quality = hf.compute_quality_score()?;
println!("Hub quality: {:.2}", quality);

// Generate README
let validator = DatasetCardValidator::new();
let readme = validator.generate_readme(&dataset)?;
```

## Examples 80-81: Native Upload and Revision

```rust
use alimentar::hf_hub::HfUploader;

// Upload dataset
let uploader = HfUploader::new("HF_TOKEN")
    .repo_id("username/my-dataset")
    .private(false);

uploader.upload(&dataset)?;

// Specific revision/branch
let uploader = HfUploader::new("HF_TOKEN")
    .repo_id("username/my-dataset")
    .revision("v1.0");
```

## Examples 82-83: Private Upload and Cache

```rust
use alimentar::hf_hub::{HfUploader, HfCache};

// Private repository
let uploader = HfUploader::new("HF_TOKEN")
    .repo_id("username/private-data")
    .private(true);

// Cache management
let cache = HfCache::default();
println!("Cache size: {} bytes", cache.size()?);
cache.clear()?;
```

## Examples 84-85: Offline Mode and Token Auth

```rust
use alimentar::hf_hub::HfDataset;

// Offline mode (use cache only)
let dataset = HfDataset::builder("username/dataset")
    .offline(true)
    .build()?
    .download()?;

// Explicit token authentication
let dataset = HfDataset::builder("username/private-dataset")
    .token("hf_xxxxx")
    .build()?
    .download()?;
```

## CLI Usage

```bash
# Download from Hub
alimentar hf download username/dataset --split train

# Upload to Hub
alimentar hf upload data.parquet username/my-dataset

# With authentication
HF_TOKEN=hf_xxx alimentar hf upload data.parquet username/my-dataset --private

# Cache management
alimentar hf cache --list
alimentar hf cache --clear
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | HuggingFace API token |
| `HF_HOME` | Cache directory location |
| `HF_OFFLINE` | Force offline mode (0/1) |

## Key Concepts

- **Dataset cards**: Metadata and documentation
- **Revisions**: Git-like versioning
- **Cache**: Local storage of downloaded datasets
- **Privacy**: Public vs private repositories
