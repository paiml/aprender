# Case Study: CORE-001 Arrow Storage Backend

This case study demonstrates the complete EXTREME TDD workflow for implementing the Arrow storage backend (CORE-001).

## Overview

**Ticket**: CORE-001
**Component**: Arrow Storage Backend
**Lines of Code**: 404
**Tests**: 14 (6 unit + 4 property-based + 3 integration + 1 doctest)
**Coverage**: 100%
**Status**: ✅ Complete

## Requirements

From `docs/specifications/db-spec-v1.md`:

1. **Parquet Reader**: Streaming RecordBatch reading using Apache Arrow
2. **MorselIterator**: 128MB chunks for out-of-core execution (Poka-Yoke)
3. **GpuTransferQueue**: Bounded async queue with max 2 in-flight transfers (Heijunka)

## RED Phase: Write Failing Tests First

### Step 1: Unit Tests for MorselIterator

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Int32Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use std::sync::Arc;

    #[test]
    fn test_morsel_iterator_splits_correctly() {
        // Create test batch with known size
        let schema = Schema::new(vec![Field::new("value", DataType::Int32, false)]);
        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5]))],
        )
        .unwrap();

        let storage = StorageEngine { batches: vec![batch] };
        let morsels: Vec<_> = storage.morsels().collect();

        // Should create at least one morsel
        assert!(!morsels.is_empty());
    }
}
```

**Result**: ❌ Test fails (MorselIterator not implemented)

```
error[E0425]: cannot find value `MorselIterator` in this scope
```

### Step 2: Property-Based Tests

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn prop_morsel_iterator_preserves_all_rows(
        num_rows in 1usize..100_000
    ) {
        // Create batch with num_rows
        let batch = create_test_batch(num_rows);
        let storage = StorageEngine { batches: vec![batch] };

        // Collect all morsels
        let morsels: Vec<_> = storage.morsels().collect();

        // Sum morsel row counts
        let morsel_row_count: usize = morsels.iter().map(|m| m.num_rows()).sum();

        // Property: No rows lost
        prop_assert_eq!(morsel_row_count, num_rows);
    }
}
```

**Result**: ❌ Test fails (compile error)

### Step 3: Integration Test with Real Parquet

```rust
#[test]
fn test_storage_engine_loads_parquet() {
    let test_file = "/tmp/trueno_test_data.parquet";

    // Create test Parquet file (10,000 rows)
    create_test_parquet(test_file).expect("Failed to create Parquet");

    // Load with StorageEngine
    let storage = StorageEngine::load_parquet(test_file)
        .expect("Failed to load Parquet");

    // Verify batches loaded
    let batches = storage.batches();
    assert!(!batches.is_empty());

    // Verify total row count
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 10_000);
}
```

**Result**: ❌ Test fails (load_parquet not implemented)

## GREEN Phase: Minimal Implementation

### Step 1: Parquet Reader

```rust
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::fs::File;
use std::path::Path;

pub struct StorageEngine {
    batches: Vec<RecordBatch>,
}

impl StorageEngine {
    pub fn load_parquet<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path.as_ref())?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let reader = builder.build()?;

        let mut batches = Vec::new();
        for batch_result in reader {
            batches.push(batch_result?);
        }

        Ok(Self { batches })
    }

    pub fn batches(&self) -> &[RecordBatch] {
        &self.batches
    }
}
```

**Result**: ✅ Integration test passes!

```
test test_storage_engine_loads_parquet ... ok
```

### Step 2: MorselIterator (Naive)

```rust
pub struct MorselIterator<'a> {
    batches: &'a [RecordBatch],
    current_batch_idx: usize,
    current_offset: usize,
    morsel_rows: usize,
}

impl<'a> MorselIterator<'a> {
    fn new(batches: &'a [RecordBatch]) -> Self {
        // Calculate morsel rows dynamically
        let morsel_rows = if batches.is_empty() {
            0
        } else {
            let schema = batches[0].schema();
            let bytes_per_row = estimate_row_size(&schema);
            MORSEL_SIZE_BYTES / bytes_per_row
        };

        Self {
            batches,
            current_batch_idx: 0,
            current_offset: 0,
            morsel_rows,
        }
    }
}

impl<'a> Iterator for MorselIterator<'a> {
    type Item = RecordBatch;

    fn next(&mut self) -> Option<Self::Item> {
        // Simplified: just return full batches
        if self.current_batch_idx >= self.batches.len() {
            return None;
        }

        let batch = self.batches[self.current_batch_idx].clone();
        self.current_batch_idx += 1;
        Some(batch)
    }
}
```

**Result**: ✅ Basic unit test passes!

### Step 3: Implement Chunking

```rust
impl<'a> Iterator for MorselIterator<'a> {
    type Item = RecordBatch;

    fn next(&mut self) -> Option<Self::Item> {
        while self.current_batch_idx < self.batches.len() {
            let batch = &self.batches[self.current_batch_idx];
            let remaining_rows = batch.num_rows() - self.current_offset;

            if remaining_rows == 0 {
                // Move to next batch
                self.current_batch_idx += 1;
                self.current_offset = 0;
                continue;
            }

            // Calculate chunk size
            let chunk_size = remaining_rows.min(self.morsel_rows);

            // Slice batch
            let morsel = batch.slice(self.current_offset, chunk_size);
            self.current_offset += chunk_size;

            return Some(morsel);
        }

        None
    }
}
```

**Result**: ✅ All MorselIterator tests pass!

```
test test_morsel_iterator_splits_correctly ... ok
test test_morsel_iterator_empty_batch ... ok
test test_morsel_iterator_multiple_batches ... ok
```

### Step 4: GpuTransferQueue

```rust
use tokio::sync::mpsc;

const MAX_IN_FLIGHT_TRANSFERS: usize = 2;

pub struct GpuTransferQueue {
    sender: mpsc::Sender<RecordBatch>,
    receiver: mpsc::Receiver<RecordBatch>,
}

impl GpuTransferQueue {
    pub fn new() -> Self {
        let (sender, receiver) = mpsc::channel(MAX_IN_FLIGHT_TRANSFERS);
        Self { sender, receiver }
    }

    pub async fn enqueue(&self, batch: RecordBatch) -> Result<()> {
        self.sender
            .send(batch)
            .await
            .map_err(|_| Error::QueueClosed)?;
        Ok(())
    }

    pub async fn dequeue(&mut self) -> Result<RecordBatch> {
        self.receiver
            .recv()
            .await
            .ok_or(Error::QueueClosed)
    }
}
```

**Result**: ✅ Queue tests pass!

```
test test_gpu_transfer_queue_basic ... ok
test test_gpu_transfer_queue_bounded ... ok
```

## REFACTOR Phase: Improve Quality

### Issue 1: Hanging Async Test

**Problem**: `test_gpu_transfer_queue_concurrent_enqueue_dequeue` hangs

```rust
#[tokio::test]
async fn test_gpu_transfer_queue_concurrent_enqueue_dequeue() {
    let mut queue = GpuTransferQueue::new();

    // Enqueue 5 batches (but capacity is 2!)
    for i in 0..5 {
        queue.enqueue(batch.clone()).await.unwrap(); // HANGS HERE
    }
}
```

**Fix**: Spawn concurrent tasks

```rust
#[tokio::test]
async fn test_gpu_transfer_queue_concurrent_enqueue_dequeue() {
    let mut queue = GpuTransferQueue::new();
    let batch = create_test_batch(100);

    // Clone sender for concurrent enqueue
    let sender = queue.sender.clone();
    let enqueue_handle = task::spawn(async move {
        for i in 0..5 {
            sender.send(batch.clone()).await.unwrap();
        }
    });

    // Concurrently dequeue
    for i in 0..5 {
        let batch = queue.dequeue().await.unwrap();
        assert_eq!(batch.num_rows(), 100);
    }

    enqueue_handle.await.unwrap();
}
```

**Result**: ✅ Test passes in <100ms

### Issue 2: Clippy Warnings

```
warning: redundant closure
  --> src/storage/mod.rs:120:50
   |
120|     let morsel_row_count: usize = morsels.iter().map(|m| m.num_rows()).sum();
   |                                                      ^^^ help: use: `RecordBatch::num_rows`
```

**Fix**: Apply clippy suggestions

```rust
let morsel_row_count: usize = morsels.iter().map(RecordBatch::num_rows).sum();
```

**Result**: ✅ Zero clippy warnings

### Issue 3: Property-Based Tests Need More Cases

Add more property-based tests:

```rust
proptest! {
    #[test]
    fn prop_morsel_size_within_limit(num_rows in 1usize..1_000_000) {
        let batch = create_test_batch(num_rows);
        let storage = StorageEngine { batches: vec![batch] };

        for morsel in storage.morsels() {
            let size = morsel.get_array_memory_size();
            prop_assert!(size <= MORSEL_SIZE_BYTES);
        }
    }

    #[test]
    fn prop_multiple_batches_preserve_rows(
        num_batches in 1usize..10,
        rows_per_batch in 1usize..10_000
    ) {
        let batches: Vec<_> = (0..num_batches)
            .map(|_| create_test_batch(rows_per_batch))
            .collect();

        let total_rows = num_batches * rows_per_batch;
        let storage = StorageEngine { batches };

        let morsel_row_count: usize = storage.morsels()
            .map(RecordBatch::num_rows)
            .sum();

        prop_assert_eq!(morsel_row_count, total_rows);
    }
}
```

**Result**: ✅ 100 test cases generated, all pass

## Final Quality Metrics

```bash
$ cargo test --lib storage
running 10 tests
test storage::tests::test_morsel_iterator_splits_correctly ... ok
test storage::tests::test_morsel_iterator_empty_batch ... ok
test storage::tests::test_morsel_iterator_multiple_batches ... ok
test storage::tests::test_gpu_transfer_queue_basic ... ok
test storage::tests::test_gpu_transfer_queue_bounded ... ok
test storage::tests::test_gpu_transfer_queue_concurrent_enqueue_dequeue ... ok
test storage::tests::prop_morsel_iterator_preserves_all_rows ... ok (100 cases)
test storage::tests::prop_morsel_size_within_limit ... ok (100 cases)
test storage::tests::prop_multiple_batches_preserve_rows ... ok (100 cases)
test storage::tests::prop_empty_batches_handled ... ok (100 cases)

test result: ok. 10 passed; 0 failed; 0 ignored; 0 measured
```

```bash
$ cargo test --test integration_test
running 3 tests
test test_storage_engine_loads_parquet ... ok
test test_morsel_iterator_with_real_data ... ok
test test_full_pipeline_with_gpu_queue ... ok

test result: ok. 3 passed; 0 failed; 0 ignored
```

```bash
$ cargo clippy --all-targets
    Finished dev [unoptimized + debuginfo] target(s) in 0.12s
     Checking trueno-db v0.1.0
    Finished checking
```

```bash
$ make coverage
📊 Generating coverage report...
✅ Coverage report: target/coverage/html/index.html

TOTAL   77.71%   (storage module: 100%)
```

## Key Learnings

### 1. Property-Based Testing Catches Edge Cases

Property tests found:
- Empty batches handling
- Multiple batch iteration
- Morsel size boundary conditions

### 2. Async Testing Requires Concurrency

Initial test hung because:
- Channel capacity = 2
- Tried to enqueue 5 items sequentially
- No dequeue to make space

**Solution**: Spawn concurrent tasks

### 3. Integration Tests Build Confidence

Real Parquet files (10,000 rows) verified:
- Arrow integration works
- Morsels handle realistic data
- GPU queue works end-to-end

### 4. Toyota Way Principles

- **Poka-Yoke**: 128MB morsel limit prevents VRAM OOM
- **Heijunka**: Bounded queue (max 2 in-flight) prevents memory explosion
- **Jidoka**: Property-based tests ensure built-in quality

## Commits

```
c21c22a feat(CORE-001): Implement Arrow storage backend (Refs CORE-001)
992ee62 test(CORE-001): Add property-based tests (Refs CORE-001)
e148520 feat(CORE-001): Implement GPU transfer queue (Refs CORE-001)
b2bc8ec test(CORE-001): Add integration tests for storage backend (Refs CORE-001)
473134c docs(CORE-001): Mark CORE-001 complete in STATUS.md (Refs CORE-001)
```

## Next Steps

- [CORE-002: Cost-Based Backend Dispatcher](./core-002.md)
- [Property-Based Testing Morsels](./proptest-morsels.md)
- [Integration Testing Pipeline](./integration-pipeline.md)
