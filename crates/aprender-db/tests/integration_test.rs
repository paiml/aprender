//! Integration test for CORE-001: Arrow Storage Backend
//!
//! Tests the complete storage pipeline:
//! 1. Load Parquet file
//! 2. Iterate over morsels
//! 3. Enqueue batches to GPU transfer queue
//!
//! Toyota Way: Jidoka (Built-in Quality)

use arrow::array::{Float32Array, Int32Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use trueno_db::storage::StorageEngine;

/// Create a test Parquet file with 10,000 rows
#[allow(clippy::cast_precision_loss)]
fn create_test_parquet<P: AsRef<Path>>(path: P) -> Result<(), Box<dyn std::error::Error>> {
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("value", DataType::Float32, false),
        Field::new("category", DataType::Utf8, false),
    ]);

    // Create test data (10,000 rows)
    let num_rows: i32 = 10_000;
    let id_array = Int32Array::from_iter_values(0..num_rows);
    let value_array = Float32Array::from_iter_values((0..num_rows).map(|i| (i as f32) * 1.5));
    let category_array =
        StringArray::from_iter_values((0..num_rows).map(|i| format!("category_{}", i % 10)));

    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(id_array), Arc::new(value_array), Arc::new(category_array)],
    )?;

    // Write to Parquet file
    let file = File::create(path.as_ref())?;
    let props = WriterProperties::builder()
        .set_max_row_group_size(5000) // 2 row groups
        .build();
    let mut writer = ArrowWriter::try_new(file, Arc::new(schema), Some(props))?;
    writer.write(&batch)?;
    writer.close()?;

    Ok(())
}

#[test]
fn test_storage_engine_loads_parquet() {
    let test_file = "/tmp/trueno_test_data.parquet";

    // Create test Parquet file
    create_test_parquet(test_file).expect("Failed to create test Parquet file");

    // Load with StorageEngine
    let storage = StorageEngine::load_parquet(test_file).expect("Failed to load Parquet file");

    // Verify batches loaded
    let batches = storage.batches();
    assert!(!batches.is_empty(), "No batches loaded");

    // Verify total row count
    let total_rows: usize = batches.iter().map(arrow::array::RecordBatch::num_rows).sum();
    assert_eq!(total_rows, 10_000, "Expected 10,000 rows");

    // Verify schema
    let first_batch = &batches[0];
    assert_eq!(first_batch.num_columns(), 3, "Expected 3 columns");

    // Clean up
    std::fs::remove_file(test_file).ok();
}

#[test]
fn test_morsel_iterator_with_real_data() {
    let test_file = "/tmp/trueno_test_morsels.parquet";

    // Create test Parquet file
    create_test_parquet(test_file).expect("Failed to create test Parquet file");

    // Load with StorageEngine
    let storage = StorageEngine::load_parquet(test_file).expect("Failed to load Parquet file");

    // Iterate over morsels
    let morsels: Vec<_> = storage.morsels().collect();

    // Verify morsels created
    assert!(!morsels.is_empty(), "No morsels created");

    // Verify all rows accounted for
    let morsel_row_count: usize = morsels.iter().map(arrow::array::RecordBatch::num_rows).sum();
    assert_eq!(morsel_row_count, 10_000, "Morsel iteration lost rows");

    // Verify each morsel is within size limit (128MB)
    for (i, morsel) in morsels.iter().enumerate() {
        let size = morsel.get_array_memory_size();
        assert!(size <= 128 * 1024 * 1024, "Morsel {i} exceeds 128MB: {size} bytes");
    }

    // Clean up
    std::fs::remove_file(test_file).ok();
}

#[tokio::test]
async fn test_full_pipeline_with_gpu_queue() {
    let test_file = "/tmp/trueno_test_pipeline.parquet";

    // Create test Parquet file
    create_test_parquet(test_file).expect("Failed to create test Parquet file");

    // Load with StorageEngine
    let storage = StorageEngine::load_parquet(test_file).expect("Failed to load Parquet file");

    // Create GPU transfer queue
    let queue = trueno_db::storage::GpuTransferQueue::new();

    // Enqueue 2 morsels (queue capacity is 2, so this won't block)
    let mut count = 0;
    for morsel in storage.morsels().take(2) {
        queue.enqueue(morsel).await.expect("Failed to enqueue morsel");
        count += 1;
    }

    assert_eq!(count, 2, "Should have enqueued 2 morsels");

    // Clean up
    std::fs::remove_file(test_file).ok();
}
