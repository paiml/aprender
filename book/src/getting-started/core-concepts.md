# Core Concepts

Understanding these core concepts will help you use alimentar effectively.

## Arrow RecordBatch

At the heart of alimentar is Apache Arrow's `RecordBatch`. All data flows through the library as RecordBatches, providing:

- **Zero-copy** data access
- **Columnar** memory layout for efficient processing
- **Interoperability** with other Arrow-based tools
- **Type safety** through strongly-typed arrays

```rust
use arrow::record_batch::RecordBatch;

// RecordBatch is the fundamental data unit
let batch: RecordBatch = dataset.get(0).unwrap();

println!("Columns: {:?}", batch.schema().fields());
println!("Rows: {}", batch.num_rows());
```

## Dataset Trait

The `Dataset` trait defines the interface for all dataset types:

```rust
pub trait Dataset: Send + Sync {
    /// Number of rows in the dataset
    fn len(&self) -> usize;

    /// Check if dataset is empty
    fn is_empty(&self) -> bool;

    /// Get a specific batch by index
    fn get(&self, index: usize) -> Option<RecordBatch>;

    /// Get the Arrow schema
    fn schema(&self) -> SchemaRef;

    /// Iterate over all batches
    fn iter(&self) -> Box<dyn Iterator<Item = RecordBatch> + '_>;
}
```

### ArrowDataset

The primary dataset implementation:

```rust
use alimentar::ArrowDataset;

// ArrowDataset holds data as Vec<RecordBatch>
let dataset = ArrowDataset::from_parquet("data.parquet")?;

// Or create from existing batches
let dataset = ArrowDataset::new(batches)?;
```

### StreamingDataset

For large datasets that don't fit in memory:

```rust
use alimentar::streaming::{StreamingDataset, ParquetSource};

let source = ParquetSource::new("large_file.parquet")?;
let dataset = StreamingDataset::new(Box::new(source))
    .buffer_size(10)
    .prefetch(4);

for batch in dataset {
    // Process without loading everything
}
```

## Transform Trait

Transforms modify data as it flows through:

```rust
pub trait Transform: Send + Sync {
    /// Apply transformation to a batch
    fn apply(&self, batch: &RecordBatch) -> Result<RecordBatch>;
}
```

### Built-in Transforms

| Transform | Purpose |
|-----------|---------|
| `Select` | Keep specific columns |
| `Drop` | Remove columns |
| `Rename` | Rename columns |
| `Filter` | Filter rows |
| `Map` | Transform rows |
| `Cast` | Change column types |
| `FillNull` | Handle missing values |
| `Normalize` | Scale numeric values |
| `Sample` | Random sampling |
| `Shuffle` | Randomize order |
| `Sort` | Order by columns |
| `Unique` | Remove duplicates |
| `Take` / `Skip` | Slice data |

### Chaining Transforms

```rust
use alimentar::{Chain, Select, Normalize, FillNull, FillStrategy};

let pipeline = Chain::new(vec![
    Box::new(Select::new(vec!["a", "b", "c"])),
    Box::new(FillNull::new("a", FillStrategy::Zero)),
    Box::new(Normalize::zscore(vec!["a", "b"])),
]);

let processed = dataset.with_transform(pipeline);
```

## StorageBackend Trait

Backends handle data persistence:

```rust
#[async_trait]
pub trait StorageBackend: Send + Sync {
    async fn list(&self, prefix: &str) -> Result<Vec<String>>;
    async fn get(&self, key: &str) -> Result<Vec<u8>>;
    async fn put(&self, key: &str, data: &[u8]) -> Result<()>;
    async fn delete(&self, key: &str) -> Result<()>;
    async fn exists(&self, key: &str) -> Result<bool>;
    async fn size(&self, key: &str) -> Result<usize>;
}
```

### Available Backends

| Backend | Use Case |
|---------|----------|
| `LocalBackend` | Local filesystem |
| `MemoryBackend` | Testing, WASM |
| `S3Backend` | Cloud storage |
| `HttpBackend` | Read-only HTTP sources |

## DataLoader

DataLoader wraps a dataset for ML training:

```rust
use alimentar::DataLoader;

let loader = DataLoader::new(dataset)
    .batch_size(32)      // Rows per batch
    .shuffle(true)       // Randomize order
    .drop_last(false);   // Keep partial final batch

// Implements Iterator
for batch in loader {
    train_step(&batch);
}
```

### Key Properties

- **batch_size**: Number of rows per iteration
- **shuffle**: Whether to randomize order (default: false)
- **drop_last**: Whether to drop incomplete final batch (default: false)
- **seed**: Random seed for reproducibility

## Registry

Registry manages dataset storage and discovery:

```rust
use alimentar::Registry;

let registry = Registry::new("/data/registry")?;

// Publish
registry.publish("dataset-name", dataset, metadata)?;

// Pull
let dataset = registry.pull("dataset-name", None)?;

// Search
let results = registry.search("keyword")?;
```

## Error Handling

Alimentar uses a custom `Result` type:

```rust
use alimentar::{Error, Result};

fn process_data() -> Result<()> {
    let dataset = ArrowDataset::from_csv("data.csv", None)?;
    // ... processing
    Ok(())
}
```

### Error Types

| Error | Cause |
|-------|-------|
| `Error::Io` | File/network I/O failure |
| `Error::Schema` | Schema mismatch |
| `Error::ColumnNotFound` | Missing column |
| `Error::IndexOutOfBounds` | Invalid index |
| `Error::EmptyDataset` | Empty data |
| `Error::Transform` | Transform failure |
| `Error::Storage` | Backend failure |
| `Error::Config` | Invalid configuration |
| `Error::UnsupportedFormat` | Unknown format |

## WASM Considerations

When targeting WASM:

- Use `MemoryBackend` or `HttpBackend` (no filesystem)
- Set `num_workers = 0` in DataLoader (no threads)
- Use `wasm-bindgen-futures` for async operations

```rust
#[cfg(target_arch = "wasm32")]
fn wasm_setup() {
    let backend = MemoryBackend::new();
    let loader = DataLoader::new(dataset)
        .batch_size(32);  // num_workers defaults to 0 on WASM
}
```
