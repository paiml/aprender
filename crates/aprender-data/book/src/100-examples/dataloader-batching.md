# DataLoader & Batching (Examples 11-20)

This section covers the DataLoader for ML training workflows.

## Example 11: Basic Batching

```rust
use alimentar::{ArrowDataset, DataLoader};

let dataset = ArrowDataset::from_parquet("data.parquet")?;
let loader = DataLoader::new(dataset).batch_size(100);

for batch in loader {
    println!("Batch rows: {}", batch.num_rows());
}
```

## Example 12: Shuffle with Determinism

```rust
use alimentar::{ArrowDataset, DataLoader};

let dataset = ArrowDataset::from_parquet("data.parquet")?;
let loader = DataLoader::new(dataset)
    .batch_size(100)
    .shuffle(true)
    .seed(42); // Reproducible

let batches: Vec<_> = loader.into_iter().collect();
```

## Example 13: Drop Last

```rust
use alimentar::{ArrowDataset, DataLoader};

let dataset = ArrowDataset::from_parquet("data.parquet")?;
// 1000 rows, batch_size 300 = 3 full batches + 1 partial
let loader = DataLoader::new(dataset)
    .batch_size(300)
    .drop_last(true); // Drop incomplete last batch

let batches: Vec<_> = loader.into_iter().collect();
assert_eq!(batches.len(), 3);
```

## Examples 14-15: Parallel and Prefetch

```rust
use alimentar::{ArrowDataset, DataLoader};

let dataset = ArrowDataset::from_parquet("data.parquet")?;
let loader = DataLoader::new(dataset)
    .batch_size(100)
    .num_workers(4)      // Parallel loading
    .prefetch_factor(2); // 2x batch prefetch
```

## Examples 16-17: Weighted and Stratified Sampling

```rust
use alimentar::{ArrowDataset, DataLoader, WeightedSampler};

// Weighted sampling by column
let sampler = WeightedSampler::from_column("weight");
let loader = DataLoader::new(dataset)
    .batch_size(100)
    .sampler(sampler);

// Stratified by label
let loader = DataLoader::new(dataset)
    .batch_size(100)
    .stratify_by("label");
```

## Examples 18-19: Infinite Iterator and Collate

```rust
use alimentar::{ArrowDataset, DataLoader};

// Infinite iteration for training
let loader = DataLoader::new(dataset)
    .batch_size(100)
    .infinite(true);

// Custom collate function
let loader = DataLoader::new(dataset)
    .batch_size(100)
    .collate_fn(|batches| {
        // Custom batch merging logic
        Ok(concat_batches(batches)?)
    });
```

## Example 20: Batch Size Benchmark

```rust
use alimentar::{ArrowDataset, DataLoader};
use std::time::Instant;

let dataset = ArrowDataset::from_parquet("large.parquet")?;

for batch_size in [32, 64, 128, 256, 512] {
    let start = Instant::now();
    let loader = DataLoader::new(dataset.clone()).batch_size(batch_size);
    let _: Vec<_> = loader.into_iter().collect();
    println!("batch_size={}: {:?}", batch_size, start.elapsed());
}
```

## Key Concepts

- **Batch size**: Controls memory/compute tradeoff
- **Shuffling**: Seed for reproducibility in training
- **Drop last**: Ensures uniform batch sizes
- **Prefetch**: Overlaps data loading with compute
