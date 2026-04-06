# Introduction

**Alimentar** ("to feed" in Spanish) is a pure Rust data loading, transformation, and distribution library for the paiml sovereign AI stack. It provides HuggingFace-compatible functionality with sovereignty-first design.

## Why Alimentar?

The modern ML ecosystem often requires cloud connectivity, Python dependencies, and complex FFI bridges. Alimentar takes a different approach:

- **Sovereign-first** - Local storage by default, no mandatory cloud dependency
- **Pure Rust** - No Python, no FFI (fully WASM-compatible)
- **Zero-copy** - Arrow RecordBatch throughout for maximum efficiency
- **Ecosystem aligned** - Arrow 53, Parquet 53 (matches trueno, aprender)

## Key Features

### Data Loading
Load data from multiple sources with a unified API:

```rust
use alimentar::{ArrowDataset, DataLoader};

// Load from various formats
let csv_data = ArrowDataset::from_csv("data.csv", None)?;
let json_data = ArrowDataset::from_json("data.json", None)?;
let parquet_data = ArrowDataset::from_parquet("data.parquet")?;
```

### Transformations
Apply chainable transformations to your data:

```rust
use alimentar::{Dataset, Select, Filter, Normalize, Chain};

let dataset = ArrowDataset::from_parquet("train.parquet")?
    .with_transform(Chain::new(vec![
        Box::new(Select::new(vec!["feature1", "feature2", "label"])),
        Box::new(Normalize::zscore(vec!["feature1", "feature2"])),
    ]));
```

### DataLoader
Iterate over batches with shuffling support:

```rust
let loader = DataLoader::new(dataset)
    .batch_size(32)
    .shuffle(true);

for batch in loader {
    // Process batch
    println!("Batch with {} rows", batch.num_rows());
}
```

### Storage Backends
Store and retrieve datasets from multiple backends:

```rust
use alimentar::backend::{LocalBackend, S3Backend, MemoryBackend};

// Local filesystem
let local = LocalBackend::new("/data/datasets")?;

// S3-compatible storage
let s3 = S3Backend::builder()
    .bucket("my-datasets")
    .region("us-west-2")
    .build()
    .await?;

// In-memory (for WASM/testing)
let memory = MemoryBackend::new();
```

### Registry
Publish and discover datasets:

```rust
use alimentar::Registry;

let registry = Registry::new("/data/registry")?;

// Publish a dataset
registry.publish("my-dataset", dataset, metadata)?;

// Pull a dataset
let dataset = registry.pull("my-dataset", None)?;

// Search datasets
let results = registry.search("classification")?;
```

## Architecture

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
   trueno                aprender              assetgen
   (SIMD/GPU)            (ML/DL)              (Content)
```

## Quick Example

Here's a complete example of a typical ML data pipeline:

```rust
use alimentar::{
    ArrowDataset, DataLoader, Dataset,
    Select, FillNull, FillStrategy, Normalize, Chain,
};

fn main() -> alimentar::Result<()> {
    // Load training data
    let dataset = ArrowDataset::from_parquet("train.parquet")?;

    // Apply preprocessing transforms
    let processed = dataset.with_transform(Chain::new(vec![
        // Select relevant columns
        Box::new(Select::new(vec!["age", "income", "score", "label"])),
        // Handle missing values
        Box::new(FillNull::new("age", FillStrategy::Mean)),
        Box::new(FillNull::new("income", FillStrategy::Median)),
        // Normalize features
        Box::new(Normalize::zscore(vec!["age", "income", "score"])),
    ]));

    // Create data loader with batching and shuffling
    let loader = DataLoader::new(processed)
        .batch_size(64)
        .shuffle(true);

    // Iterate over batches for training
    for batch in loader {
        println!("Training on batch with {} rows", batch.num_rows());
        // Train your model here
    }

    Ok(())
}
```

## Next Steps

- [Installation](./getting-started/installation.md) - Get alimentar set up
- [Quick Start](./getting-started/quick-start.md) - Your first data pipeline
- [Core Concepts](./getting-started/core-concepts.md) - Understand the fundamentals
