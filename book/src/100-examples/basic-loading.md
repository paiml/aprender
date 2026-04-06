# Basic Loading (Examples 1-10)

This section covers fundamental data loading operations.

## Example 1: CSV Loading

```rust
use alimentar::ArrowDataset;
let dataset = ArrowDataset::from_csv("test_fixtures/input.csv")?;
assert!(dataset.len() > 0);
```

## Example 2: JSON Loading

```rust
use alimentar::ArrowDataset;
let dataset = ArrowDataset::from_json("test_fixtures/data.json")?;
assert!(dataset.len() > 0);
```

## Example 3: Parquet Loading

```rust
use alimentar::ArrowDataset;
let dataset = ArrowDataset::from_parquet("test_fixtures/data.parquet")?;
assert_eq!(dataset.len(), 1000);
```

## Example 4: Schema Inference

```rust
use alimentar::ArrowDataset;
let dataset = ArrowDataset::from_csv("test_fixtures/input.csv")?;
let schema = dataset.schema();
assert!(schema.field_with_name("id").is_ok());
```

## Example 5: Explicit Schema

```rust
use alimentar::{ArrowDataset, CsvOptions};
use arrow::datatypes::{DataType, Field, Schema};

let schema = Schema::new(vec![
    Field::new("id", DataType::Int64, false),
    Field::new("name", DataType::Utf8, false),
    Field::new("value", DataType::Float64, false),
]);

let options = CsvOptions::default().with_schema(schema);
let dataset = ArrowDataset::from_csv_with_options("data.csv", options)?;
```

## Examples 6-7: Glob and Memory-Mapped Loading

```rust
// Glob loading multiple files
use alimentar::ArrowDataset;
let dataset = ArrowDataset::from_parquet_glob("data/*.parquet")?;

// Memory-mapped for large files
let dataset = ArrowDataset::from_parquet_mmap("large.parquet")?;
```

## Examples 8-9: Compressed Input

```rust
// ZSTD compressed
let dataset = ArrowDataset::from_parquet("data.parquet.zst")?;

// LZ4 compressed
let dataset = ArrowDataset::from_parquet("data.parquet.lz4")?;
```

## Example 10: Large File Handling

```rust
use alimentar::ArrowDataset;
let dataset = ArrowDataset::from_parquet("test_fixtures/large.parquet")?;
assert_eq!(dataset.len(), 1_000_000);
```

## Key Concepts

- **Zero-copy**: Arrow RecordBatch throughout
- **Format detection**: Automatic based on file extension
- **Schema inference**: Optional explicit schema override
- **Memory efficiency**: Memory-mapped for large files
