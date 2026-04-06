# Quick Start

This guide walks you through your first alimentar data pipeline in under 5 minutes.

## Loading Data

### From CSV

```rust
use alimentar::{ArrowDataset, CsvOptions};

// Basic loading
let dataset = ArrowDataset::from_csv("data.csv", None)?;

// With options
let options = CsvOptions::new()
    .with_header(true)
    .with_delimiter(b',');
let dataset = ArrowDataset::from_csv("data.csv", Some(options))?;
```

### From JSON

```rust
use alimentar::{ArrowDataset, JsonOptions};

// Basic loading
let dataset = ArrowDataset::from_json("data.json", None)?;

// JSONL (newline-delimited JSON)
let dataset = ArrowDataset::from_json("data.jsonl", None)?;
```

### From Parquet

```rust
use alimentar::ArrowDataset;

let dataset = ArrowDataset::from_parquet("data.parquet")?;
```

## Inspecting Data

```rust
use alimentar::Dataset;

// Get basic info
println!("Rows: {}", dataset.len());
println!("Schema: {:?}", dataset.schema());

// Access specific rows
if let Some(batch) = dataset.get(0) {
    println!("First batch has {} rows", batch.num_rows());
}

// Iterate over all batches
for batch in dataset.iter() {
    println!("Batch: {} rows", batch.num_rows());
}
```

## Applying Transforms

```rust
use alimentar::{Dataset, Select, Filter, Normalize};

// Select specific columns
let selected = dataset.with_transform(Select::new(vec!["name", "age"]));

// Filter rows
let filtered = dataset.with_transform(Filter::new(|batch| {
    // Return indices of rows to keep
    // This is a simplified example
    Ok(batch.clone())
}));

// Normalize numeric columns
let normalized = dataset.with_transform(
    Normalize::zscore(vec!["age", "score"])
);
```

## Using DataLoader

```rust
use alimentar::{ArrowDataset, DataLoader};

let dataset = ArrowDataset::from_parquet("train.parquet")?;

let loader = DataLoader::new(dataset)
    .batch_size(32)
    .shuffle(true);

for batch in loader {
    println!("Processing batch with {} rows", batch.num_rows());
    // Your training code here
}
```

## Saving Data

```rust
use alimentar::ArrowDataset;

// Save to different formats
dataset.to_csv("output.csv")?;
dataset.to_json("output.json")?;
dataset.to_parquet("output.parquet")?;
```

## Complete Example

Here's a complete data pipeline:

```rust
use alimentar::{
    ArrowDataset, DataLoader, Dataset,
    Select, FillNull, FillStrategy, Normalize, Chain,
};

fn main() -> alimentar::Result<()> {
    // 1. Load data
    let dataset = ArrowDataset::from_csv("train.csv", None)?;
    println!("Loaded {} rows", dataset.len());

    // 2. Apply transforms
    let processed = dataset.with_transform(Chain::new(vec![
        Box::new(Select::new(vec!["feature1", "feature2", "label"])),
        Box::new(FillNull::new("feature1", FillStrategy::Mean)),
        Box::new(Normalize::minmax(vec!["feature1", "feature2"])),
    ]));

    // 3. Create data loader
    let loader = DataLoader::new(processed)
        .batch_size(64)
        .shuffle(true);

    // 4. Iterate over batches
    let mut total_rows = 0;
    for batch in loader {
        total_rows += batch.num_rows();
    }
    println!("Processed {} total rows", total_rows);

    Ok(())
}
```

## Using the CLI

The alimentar CLI provides quick data inspection:

```bash
# View schema
alimentar schema data.parquet

# View first rows
alimentar head data.parquet -n 10

# Get info
alimentar info data.parquet

# Convert formats
alimentar convert data.csv data.parquet
```

## Next Steps

- [Core Concepts](./core-concepts.md) - Understand Dataset, Transform, Backend
- [Transforms](../transforms/built-in.md) - All available transforms
- [DataLoader](../dataloader/overview.md) - Advanced batching options
