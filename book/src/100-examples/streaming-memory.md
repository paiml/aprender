# Streaming & Memory (Examples 21-30)

This section covers constant-memory streaming for large datasets.

## Example 21: Streaming Constant Memory

```rust
use alimentar::streaming::{StreamingDataset, MemorySource};

let batches = /* source batches */;
let source = MemorySource::new(batches)?;
let streaming = StreamingDataset::new(Box::new(source), 16);

for batch in streaming {
    process_batch(batch);
}
```

## Examples 22-23: Chained Sources and Memory Source

```rust
use alimentar::streaming::{StreamingDataset, ChainedSource, MemorySource};

// Chain multiple sources
let source1 = MemorySource::new(batches1)?;
let source2 = MemorySource::new(batches2)?;
let chained = ChainedSource::new(vec![
    Box::new(source1),
    Box::new(source2),
]);
let streaming = StreamingDataset::new(Box::new(chained), 16);
```

## Examples 24-25: Parquet Streaming and Buffer Tuning

```rust
use alimentar::streaming::{StreamingDataset, ParquetSource};

// Stream parquet row groups
let source = ParquetSource::new("large.parquet")?
    .row_group_size(1024);
let streaming = StreamingDataset::new(Box::new(source), 8);

// Buffer size tuning
let streaming = StreamingDataset::builder()
    .source(source)
    .buffer_size(32)
    .build()?;
```

## Examples 26-27: Async Prefetch and Backpressure

```rust
use alimentar::async_prefetch::AsyncPrefetchBuilder;

let prefetch = AsyncPrefetchBuilder::new(batches)
    .prefetch_size(4)
    .build()?;

// With backpressure
let streaming = StreamingDataset::new(Box::new(source), 16)
    .with_backpressure(8);
```

## Examples 28-29: Iterator Reset and Memory Profile

```rust
use alimentar::streaming::StreamingDataset;

// Deterministic reset
let mut streaming = StreamingDataset::new(source, 16);
let first_pass: Vec<_> = streaming.by_ref().take(10).collect();
streaming.reset();
let second_pass: Vec<_> = streaming.by_ref().take(10).collect();

// Memory profiling
let streaming = StreamingDataset::new(source, 16)
    .with_memory_tracking(true);
println!("Peak memory: {} bytes", streaming.peak_memory());
```

## Example 30: 10GB Dataset Test

```rust
use alimentar::streaming::{StreamingDataset, ParquetSource};

// Stream without loading entire dataset
let source = ParquetSource::new("10gb.parquet")?;
let streaming = StreamingDataset::new(Box::new(source), 16);

let mut total_rows = 0;
for batch in streaming {
    total_rows += batch.num_rows();
}
println!("Processed {} rows with constant memory", total_rows);
```

## Key Concepts

- **Constant memory**: Never loads full dataset
- **Buffer size**: Controls memory/throughput tradeoff
- **Backpressure**: Prevents producer outrunning consumer
- **Row groups**: Parquet-native streaming unit
