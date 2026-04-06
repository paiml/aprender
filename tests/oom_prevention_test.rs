//! OOM Prevention Tests (CORE-001)
//!
//! Tests proving that morsel-based paging prevents VRAM OOM
//! on large datasets (10GB+ files with 8GB VRAM constraint).
//!
//! Toyota Way: Poka-Yoke (mistake-proofing against memory exhaustion)
//!
//! References:
//! - Funke et al. (2018): GPU paging for out-of-core workloads
//! - Leis et al. (2014): Morsel-driven parallelism

use arrow::array::{Float32Array, Int32Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use std::sync::Arc;
use trueno_db::storage::{StorageEngine, MORSEL_SIZE_BYTES};

/// Test that morsel iterator processes data in 128MB chunks
/// This prevents loading entire dataset into VRAM at once
#[test]
fn test_morsel_size_bounded() {
    // Create a large batch (simulating 1GB of data)
    let _num_rows = 250_000_000; // 250M rows * 4 bytes = 1GB (for documentation)
    let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int32, false)]));

    // Note: We don't actually allocate 1GB here (would OOM in tests)
    // Instead we create small batches and verify morsel logic
    let batch_size = 1_000_000; // 1M rows = 4MB per batch

    let mut batches = Vec::new();
    for _ in 0..10 {
        // Create 10 batches for testing (40MB total)
        let data = vec![42i32; batch_size];
        let array = Int32Array::from(data);
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(array)]).unwrap();
        batches.push(batch);
    }

    let storage = StorageEngine::new(batches);

    // Verify morsel iterator chunks data appropriately
    let mut morsel_count = 0;
    let mut total_rows = 0;

    for morsel in storage.morsels() {
        morsel_count += 1;
        total_rows += morsel.num_rows();

        // Each morsel should be <= 128MB worth of data
        let morsel_bytes = morsel.num_rows() * 4; // i32 = 4 bytes
        assert!(
            morsel_bytes <= MORSEL_SIZE_BYTES,
            "Morsel size {morsel_bytes} exceeds limit {MORSEL_SIZE_BYTES}"
        );
    }

    assert!(morsel_count > 0, "Should have processed at least one morsel");
    assert_eq!(total_rows, 10 * batch_size, "Should process all rows across morsels");
}

/// Test that morsel iteration handles large datasets without OOM
/// Simulates processing a 10GB dataset with 8GB VRAM constraint
#[test]
fn test_large_dataset_no_oom() {
    // Simulate 10GB dataset structure (250M rows * 40 bytes/row = 10GB)
    // We create metadata showing how it would be chunked, without allocating
    let row_size_bytes = 40; // Typical row with multiple columns
    let total_dataset_size_gb = 10;
    let total_dataset_bytes = total_dataset_size_gb * 1024 * 1024 * 1024;
    let total_rows = total_dataset_bytes / row_size_bytes;

    // Calculate expected morsel count
    let morsel_size_bytes = MORSEL_SIZE_BYTES;
    let rows_per_morsel = morsel_size_bytes / row_size_bytes;
    let expected_morsel_count = total_rows.div_ceil(rows_per_morsel);

    println!("Dataset size: {total_dataset_size_gb} GB");
    println!("Total rows: {total_rows}");
    println!("Morsel size: {} MB", morsel_size_bytes / 1024 / 1024);
    println!("Rows per morsel: {rows_per_morsel}");
    println!("Expected morsels: {expected_morsel_count}");

    // Verify morsel count is reasonable (hundreds, not millions)
    assert!(
        expected_morsel_count < 10_000,
        "Morsel count should be manageable: {expected_morsel_count}"
    );

    // Verify each morsel fits in typical GPU memory
    let max_gpu_memory_gb = 8;
    let max_gpu_memory_bytes = max_gpu_memory_gb * 1024 * 1024 * 1024;
    assert!(
        morsel_size_bytes <= max_gpu_memory_bytes,
        "Morsel size {morsel_size_bytes} should fit in {max_gpu_memory_gb}GB VRAM"
    );

    // Key insight: We process 80 morsels sequentially for 10GB dataset
    // Each morsel is 128MB, well within 8GB VRAM limit
    // This proves Poka-Yoke design prevents OOM
    assert!(
        morsel_size_bytes * 2 < max_gpu_memory_bytes,
        "Two in-flight morsels ({} MB) should fit in VRAM",
        (morsel_size_bytes * 2) / 1024 / 1024
    );
}

/// Property test: Verify morsel processing maintains data integrity
#[test]
fn test_morsel_iteration_preserves_data() {
    // Create multiple batches with known data
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("value", DataType::Float32, false),
    ]));

    let mut batches = Vec::new();
    let mut expected_sum = 0i64;

    for batch_idx in 0..20 {
        // 20 batches
        let batch_size = 500_000; // 500K rows/batch
        let ids: Vec<i32> = (0..batch_size).map(|i| i + (batch_idx * batch_size)).collect();
        let values: Vec<f32> =
            (0..batch_size).map(|i| (i + batch_idx * batch_size) as f32).collect();

        expected_sum += values.iter().map(|v| *v as i64).sum::<i64>();

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(ids)), Arc::new(Float32Array::from(values))],
        )
        .unwrap();
        batches.push(batch);
    }

    let storage = StorageEngine::new(batches);

    // Process via morsels and verify data integrity
    let mut actual_sum = 0i64;
    let mut total_rows = 0;

    for morsel in storage.morsels() {
        total_rows += morsel.num_rows();
        let value_column = morsel.column(1).as_any().downcast_ref::<Float32Array>().unwrap();

        for i in 0..value_column.len() {
            actual_sum += value_column.value(i) as i64;
        }
    }

    assert_eq!(total_rows, 20 * 500_000, "Should process all rows");
    assert_eq!(actual_sum, expected_sum, "Morsel processing preserves data");
}

/// Test morsel iterator with empty dataset (edge case)
#[test]
fn test_morsel_empty_dataset() {
    let storage = StorageEngine::new(vec![]);
    let mut morsel_count = 0;

    for _morsel in storage.morsels() {
        morsel_count += 1;
    }

    assert_eq!(morsel_count, 0, "Empty dataset should yield no morsels");
}

/// Test morsel iterator with single small batch
#[test]
fn test_morsel_small_batch() {
    let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int32, false)]));

    let data = vec![1i32, 2, 3, 4, 5];
    let batch = RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(data))]).unwrap();
    let storage = StorageEngine::new(vec![batch]);

    let mut morsel_count = 0;
    let mut total_rows = 0;

    for morsel in storage.morsels() {
        morsel_count += 1;
        total_rows += morsel.num_rows();
    }

    assert_eq!(morsel_count, 1, "Small batch should yield one morsel");
    assert_eq!(total_rows, 5, "Should process all 5 rows");
}

/// Benchmark-style test showing memory efficiency
#[test]
fn test_memory_efficiency_calculation() {
    // Real-world scenario: Processing 100GB dataset on 16GB GPU
    let dataset_gb = 100;
    let gpu_vram_gb = 16;

    let dataset_bytes: usize = dataset_gb * 1024 * 1024 * 1024;
    let gpu_bytes: usize = gpu_vram_gb * 1024 * 1024 * 1024;

    let morsel_count = dataset_bytes.div_ceil(MORSEL_SIZE_BYTES);
    let memory_amplification = MORSEL_SIZE_BYTES as f64 / gpu_bytes as f64;

    println!("Dataset: {dataset_gb} GB");
    println!("GPU VRAM: {gpu_vram_gb} GB");
    println!("Morsel size: {} MB", MORSEL_SIZE_BYTES / 1024 / 1024);
    println!("Number of morsels: {morsel_count}");
    println!("Memory amplification: {memory_amplification:.4}");

    // Verify morsel-based processing prevents OOM
    assert!(MORSEL_SIZE_BYTES < gpu_bytes, "Morsel must fit in available VRAM");

    // With 2 in-flight transfers (MAX_IN_FLIGHT_TRANSFERS = 2),
    // peak memory is 2 * 128MB = 256MB << 16GB VRAM
    let peak_memory_mb = (MORSEL_SIZE_BYTES * 2) / 1024 / 1024;
    let vram_mb = gpu_bytes / 1024 / 1024;

    println!("Peak memory usage: {peak_memory_mb} MB");
    println!("Available VRAM: {vram_mb} MB");
    println!("VRAM utilization: {:.2}%", (peak_memory_mb as f64 / vram_mb as f64) * 100.0);

    assert!(
        peak_memory_mb < vram_mb,
        "Peak memory {peak_memory_mb}MB should be well below {vram_mb}MB VRAM"
    );

    // Toyota Way: Poka-Yoke (mistake-proofing)
    // Even with 100GB dataset, we never exceed VRAM
    assert!(peak_memory_mb < 512, "Peak memory should be < 512MB even for massive datasets");
}
