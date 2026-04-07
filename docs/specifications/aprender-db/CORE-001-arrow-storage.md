# CORE-001: Arrow Storage Backend with Morsel-Based Paging

**Ticket ID**: CORE-001
**Priority**: High
**Phase**: 1
**Status**: In Progress
**Toyota Way Principle**: Poka-Yoke (Mistake Proofing)

## Objective

Implement Arrow/Parquet storage backend with 128MB morsel-based paging to prevent VRAM out-of-memory crashes.

## Background

**Problem**: Loading large datasets (>VRAM capacity) directly to GPU causes driver crashes or panics. Traditional databases assume unlimited RAM, but GPUs have limited VRAM (typically 8-16GB).

**Solution**: Implement morsel-based paging (Funke et al. 2018) to stream data through GPU in fixed-size chunks.

**Toyota Way**: Poka-Yoke (mistake proofing) - system should never crash due to VRAM exhaustion.

## Academic Foundation

- **Funke et al. (2018)**: "Paging-based virtual memory for GPU databases" (VLDB)
  - URL: http://www.vldb.org/pvldb/vol11/p998-funke.pdf
  - Describes mechanisms to handle datasets larger than VRAM by swapping pages

- **Leis et al. (2014)**: "Morsel-driven parallelism" (SIGMOD)
  - URL: https://db.in.tum.de/~leis/papers/morsels.pdf
  - Fixed-size work chunks prevent long-running queries from blocking

## Technical Design

### Morsel Size

```rust
/// Morsel size for out-of-core execution (128MB chunks)
/// Based on: Leis et al. (2014) morsel-driven parallelism
pub const MORSEL_SIZE_BYTES: usize = 128 * 1024 * 1024;  // 128MB
```

**Rationale**:
- Large enough to amortize kernel launch overhead
- Small enough to fit in GPU shared memory for processing
- Aligns with typical PCIe transfer block sizes

### Architecture

```
┌─────────────────────────────────────────────┐
│           StorageEngine                     │
│                                             │
│  ┌───────────────────────────────────────┐ │
│  │   ParquetReader                       │ │
│  │   - Arrow RecordBatch streaming       │ │
│  │   - Column projection (late mat.)     │ │
│  └───────────────────────────────────────┘ │
│                    │                        │
│                    ▼                        │
│  ┌───────────────────────────────────────┐ │
│  │   MorselIterator                      │ │
│  │   - Split RecordBatch into 128MB     │ │
│  │   - Slice Arrow arrays                │ │
│  │   - Track offset/remaining            │ │
│  └───────────────────────────────────────┘ │
│                    │                        │
│                    ▼                        │
│  ┌───────────────────────────────────────┐ │
│  │   GPU Transfer Queue                  │ │
│  │   - Async PCIe transfer               │ │
│  │   - Bounded queue (max 2 in-flight)  │ │
│  └───────────────────────────────────────┘ │
└─────────────────────────────────────────────┘
```

### Implementation Plan

#### 1. Parquet Reader with Arrow Integration

```rust
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

pub struct StorageEngine {
    cache: Option<Vec<RecordBatch>>,
}

impl StorageEngine {
    pub fn load_parquet(path: &str) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;

        // Get schema for column projection (late materialization)
        let schema = builder.schema();

        // Create reader with default batch size
        let reader = builder.build()?;

        Ok(Self { cache: None })
    }
}
```

#### 2. Morsel Iterator

```rust
pub struct MorselIterator {
    record_batch: RecordBatch,
    current_offset: usize,
    morsel_rows: usize,  // Calculated to fit ~128MB
}

impl MorselIterator {
    pub fn new(record_batch: RecordBatch) -> Self {
        // Calculate rows per morsel based on schema
        let row_size_bytes = Self::estimate_row_size(&record_batch);
        let morsel_rows = MORSEL_SIZE_BYTES / row_size_bytes;

        Self {
            record_batch,
            current_offset: 0,
            morsel_rows,
        }
    }

    fn estimate_row_size(batch: &RecordBatch) -> usize {
        batch.get_array_memory_size() / batch.num_rows()
    }
}

impl Iterator for MorselIterator {
    type Item = RecordBatch;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_offset >= self.record_batch.num_rows() {
            return None;
        }

        let length = std::cmp::min(
            self.morsel_rows,
            self.record_batch.num_rows() - self.current_offset
        );

        let morsel = self.record_batch.slice(self.current_offset, length);
        self.current_offset += length;

        Some(morsel)
    }
}
```

#### 3. GPU Transfer Queue (Async, Bounded)

```rust
use tokio::sync::mpsc;

pub struct GpuTransferQueue {
    sender: mpsc::Sender<RecordBatch>,
    receiver: mpsc::Receiver<RecordBatch>,
}

impl GpuTransferQueue {
    pub fn new() -> Self {
        // Bounded queue: max 2 batches in-flight
        // Prevents memory explosion while keeping PCIe bus busy
        let (sender, receiver) = mpsc::channel(2);
        Self { sender, receiver }
    }

    pub async fn enqueue(&self, batch: RecordBatch) -> Result<()> {
        self.sender.send(batch).await
            .map_err(|_| Error::Other("Transfer queue closed".into()))
    }

    pub async fn dequeue(&mut self) -> Option<RecordBatch> {
        self.receiver.recv().await
    }
}
```

## Acceptance Criteria

### Functional Requirements

1. ✅ **Parquet Reader**
   - Load Parquet files using Arrow
   - Support all Arrow data types (Int32, Float32, String, etc.)
   - Column projection (late materialization)

2. ✅ **Morsel Paging**
   - Split data into 128MB chunks
   - Calculate morsel size based on schema
   - Iterator-based streaming API

3. ✅ **Out-of-Core Execution**
   - Process 10GB file with 8GB VRAM (no OOM)
   - Bounded transfer queue (max 2 in-flight)
   - Async/await integration

### Non-Functional Requirements

4. ✅ **Performance**
   - PCIe transfer overhead <10% of total query time
   - No blocking on Tokio reactor

5. ✅ **Memory Safety**
   - No panics on VRAM exhaustion
   - Graceful degradation to SIMD if GPU unavailable

## Test Plan (EXTREME TDD)

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_morsel_iterator_splits_correctly() {
        // Create test RecordBatch with known size
        let batch = create_test_batch(1000 rows);

        let mut iter = MorselIterator::new(batch);
        let morsels: Vec<_> = iter.collect();

        // Verify all rows accounted for
        let total_rows: usize = morsels.iter().map(|m| m.num_rows()).sum();
        assert_eq!(total_rows, 1000);

        // Verify morsel sizes ~128MB
        for morsel in &morsels {
            assert!(morsel.get_array_memory_size() <= MORSEL_SIZE_BYTES);
        }
    }

    #[test]
    fn test_parquet_reader_loads_file() {
        let engine = StorageEngine::load_parquet("test_data/events.parquet").unwrap();
        // Verify schema parsed correctly
    }

    #[tokio::test]
    async fn test_transfer_queue_bounded() {
        let queue = GpuTransferQueue::new();

        // Enqueue 3 batches (capacity = 2)
        // Third should block until first dequeued
        let batch = create_test_batch(100);
        queue.enqueue(batch.clone()).await.unwrap();
        queue.enqueue(batch.clone()).await.unwrap();

        // This would block if not for timeout
        let result = tokio::time::timeout(
            Duration::from_millis(100),
            queue.enqueue(batch)
        ).await;

        assert!(result.is_err());  // Timeout = queue is full
    }
}
```

### Integration Tests

```rust
#[tokio::test]
async fn test_load_10gb_file_with_8gb_vram() {
    // This test proves Poka-Yoke: no OOM on large files
    let engine = StorageEngine::load_parquet("test_data/10gb_file.parquet").unwrap();

    // Process in morsels
    for morsel in engine.morsels() {
        // Simulate GPU operation
        process_on_gpu(morsel).await.unwrap();
    }

    // No panic = success
}
```

### Property-Based Tests

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn prop_morsel_iterator_preserves_all_rows(
        num_rows in 1usize..1_000_000
    ) {
        let batch = create_test_batch(num_rows);
        let original_rows = batch.num_rows();

        let iter = MorselIterator::new(batch);
        let total_morsel_rows: usize = iter.map(|m| m.num_rows()).sum();

        assert_eq!(original_rows, total_morsel_rows);
    }
}
```

## Performance Targets

- **Load 1GB Parquet file**: <500ms (SSD)
- **Morsel iteration overhead**: <1% of total time
- **PCIe transfer (per 128MB morsel)**: ~4ms (32 GB/s)
- **Total query time**: Transfer time + compute time (no extra overhead)

## Quality Gates

Before marking CORE-001 complete:

- [ ] All unit tests pass (`cargo test storage`)
- [ ] Integration test: 10GB file with 8GB VRAM (no OOM)
- [ ] Property tests pass (morsel preservation)
- [ ] Benchmarks: PCIe transfer <5ms per 128MB
- [ ] Code coverage: >90% for storage module
- [ ] TDG score: ≥B+ (85/100)
- [ ] Mutation testing: ≥80% kill rate
- [ ] Zero clippy warnings

## References

1. Funke et al. (2018): "Paging-based virtual memory for GPU databases"
2. Leis et al. (2014): "Morsel-driven parallelism"
3. Apache Arrow documentation: https://arrow.apache.org/rust/
4. Parquet format spec: https://parquet.apache.org/docs/

## Next Steps

After CORE-001 completion:
- **CORE-002**: Cost-based backend dispatcher (arithmetic intensity model)
- **CORE-006**: Backend equivalence tests (safety net before GPU work)
