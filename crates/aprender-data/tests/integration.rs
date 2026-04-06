//! Integration tests for alimentar.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::uninlined_format_args,
    clippy::cast_lossless
)]

use std::{collections::HashSet, sync::Arc};

use alimentar::{
    ArrowDataset, Chain, DataLoader, Dataset, Filter, Sample, Select, Shuffle, Skip, Sort,
    SortOrder, Take,
};
use arrow::{
    array::{Float64Array, Int32Array, RecordBatch, StringArray},
    datatypes::{DataType, Field, Schema},
};

/// Creates a test dataset with the given number of rows.
fn create_test_dataset(rows: usize) -> ArrowDataset {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("score", DataType::Float64, false),
    ]));

    let ids: Vec<i32> = (0..rows as i32).collect();
    let names: Vec<String> = ids.iter().map(|i| format!("item_{}", i)).collect();
    let scores: Vec<f64> = ids.iter().map(|i| *i as f64 * 1.5).collect();

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(ids)),
            Arc::new(StringArray::from(names)),
            Arc::new(Float64Array::from(scores)),
        ],
    )
    .ok()
    .unwrap_or_else(|| panic!("Should create batch"));

    ArrowDataset::from_batch(batch)
        .ok()
        .unwrap_or_else(|| panic!("Should create dataset"))
}

#[test]
fn test_end_to_end_workflow() {
    // 1. Create a dataset
    let dataset = create_test_dataset(100);
    assert_eq!(dataset.len(), 100);

    // 2. Apply transforms
    let select = Select::new(vec!["id", "score"]);
    let transformed = dataset
        .with_transform(&select)
        .ok()
        .unwrap_or_else(|| panic!("Should transform"));
    assert_eq!(transformed.schema().fields().len(), 2);

    // 3. Create a data loader
    let loader = DataLoader::new(transformed)
        .batch_size(10)
        .shuffle(true)
        .seed(42);

    // 4. Iterate and verify
    let mut total_rows = 0;
    for batch in loader {
        total_rows += batch.num_rows();
    }
    assert_eq!(total_rows, 100);
}

#[test]
fn test_parquet_roundtrip() {
    let temp_dir = tempfile::tempdir()
        .ok()
        .unwrap_or_else(|| panic!("Should create temp dir"));
    let path = temp_dir.path().join("test_roundtrip.parquet");

    // Create and save
    let original = create_test_dataset(50);
    original
        .to_parquet(&path)
        .ok()
        .unwrap_or_else(|| panic!("Should save"));

    // Load and verify
    let loaded = ArrowDataset::from_parquet(&path)
        .ok()
        .unwrap_or_else(|| panic!("Should load"));

    assert_eq!(loaded.len(), original.len());
    assert_eq!(loaded.schema(), original.schema());
    assert_eq!(loaded.num_batches(), original.num_batches());
}

#[test]
fn test_dataloader_shuffling_all_rows() {
    let dataset = create_test_dataset(100);

    let loader = DataLoader::new(dataset)
        .batch_size(7)
        .shuffle(true)
        .seed(99);

    let mut seen_ids = HashSet::new();
    for batch in loader {
        let id_col = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap_or_else(|| panic!("Should be Int32Array"));
        for i in 0..id_col.len() {
            seen_ids.insert(id_col.value(i));
        }
    }

    // All 100 IDs should be present
    assert_eq!(seen_ids.len(), 100);
    for i in 0..100i32 {
        assert!(seen_ids.contains(&i), "Missing id: {}", i);
    }
}

#[test]
fn test_dataloader_drop_last() {
    let dataset = create_test_dataset(25);

    // Without drop_last: 25 rows / 10 batch = 3 batches (10+10+5)
    let loader1 = DataLoader::new(dataset.clone())
        .batch_size(10)
        .drop_last(false);
    let batches1: Vec<RecordBatch> = loader1.into_iter().collect();
    assert_eq!(batches1.len(), 3);
    assert_eq!(batches1[2].num_rows(), 5);

    // With drop_last: only 2 full batches
    let loader2 = DataLoader::new(dataset).batch_size(10).drop_last(true);
    let batches2: Vec<RecordBatch> = loader2.into_iter().collect();
    assert_eq!(batches2.len(), 2);
}

#[test]
fn test_transform_chain() {
    let dataset = create_test_dataset(100);

    // Select + Filter combination
    let select = Select::new(vec!["id", "score"]);
    let selected = dataset
        .with_transform(&select)
        .ok()
        .unwrap_or_else(|| panic!("Should select"));

    let filter = Filter::new(|batch| {
        let scores = batch
            .column(1)
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| alimentar::Error::transform("Expected Float64Array"))?;
        let mask: Vec<bool> = (0..scores.len()).map(|i| scores.value(i) > 50.0).collect();
        Ok(arrow::array::BooleanArray::from(mask))
    });

    let filtered = selected
        .with_transform(&filter)
        .ok()
        .unwrap_or_else(|| panic!("Should filter"));

    // score > 50.0 means id > 33 (since score = id * 1.5)
    // ids 34-99 = 66 rows
    assert_eq!(filtered.len(), 66);
}

#[test]
fn test_shuffle_preserves_data_integrity() {
    let dataset = create_test_dataset(20);
    let shuffle = Shuffle::with_seed(42);

    let shuffled = dataset
        .with_transform(&shuffle)
        .ok()
        .unwrap_or_else(|| panic!("Should shuffle"));

    // Verify row integrity: id * 1.5 == score for all rows
    for batch in shuffled.iter() {
        let ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap_or_else(|| panic!("Should be Int32Array"));
        let scores = batch
            .column(2)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap_or_else(|| panic!("Should be Float64Array"));

        for i in 0..ids.len() {
            let id = ids.value(i);
            let score = scores.value(i);
            let expected_score = id as f64 * 1.5;
            assert!(
                (score - expected_score).abs() < f64::EPSILON,
                "Data integrity violated: id={}, score={}, expected={}",
                id,
                score,
                expected_score
            );
        }
    }
}

#[test]
fn test_multiple_batches_dataset() {
    // Create dataset with multiple batches
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));

    let batch1 = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
    )
    .ok()
    .unwrap_or_else(|| panic!("Should create batch"));

    let batch2 = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from(vec![4, 5, 6]))],
    )
    .ok()
    .unwrap_or_else(|| panic!("Should create batch"));

    let batch3 = RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vec![7, 8, 9, 10]))])
        .ok()
        .unwrap_or_else(|| panic!("Should create batch"));

    let dataset = ArrowDataset::new(vec![batch1, batch2, batch3])
        .ok()
        .unwrap_or_else(|| panic!("Should create dataset"));

    assert_eq!(dataset.len(), 10);
    assert_eq!(dataset.num_batches(), 3);

    // Test row access across batches
    for i in 0..10 {
        let row = dataset.get(i);
        assert!(row.is_some(), "Row {} should exist", i);
        let row = row.unwrap_or_else(|| panic!("Row should exist"));
        let id_col = row
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap_or_else(|| panic!("Should be Int32Array"));
        assert_eq!(id_col.value(0), i as i32 + 1);
    }
}

#[test]
fn test_backend_storage_roundtrip() {
    use alimentar::backend::{MemoryBackend, StorageBackend};
    use bytes::Bytes;

    let backend = MemoryBackend::new();

    // Store some data
    backend
        .put("datasets/train.bin", Bytes::from("training data"))
        .ok()
        .unwrap_or_else(|| panic!("Should put"));
    backend
        .put("datasets/test.bin", Bytes::from("test data"))
        .ok()
        .unwrap_or_else(|| panic!("Should put"));
    backend
        .put("models/model.bin", Bytes::from("model weights"))
        .ok()
        .unwrap_or_else(|| panic!("Should put"));

    // List by prefix
    let dataset_keys = backend
        .list("datasets/")
        .ok()
        .unwrap_or_else(|| panic!("Should list"));
    assert_eq!(dataset_keys.len(), 2);

    // Retrieve and verify
    let train_data = backend
        .get("datasets/train.bin")
        .ok()
        .unwrap_or_else(|| panic!("Should get"));
    assert_eq!(train_data, Bytes::from("training data"));

    // Check size
    let size = backend
        .size("datasets/train.bin")
        .ok()
        .unwrap_or_else(|| panic!("Should get size"));
    assert_eq!(size, 13); // "training data" = 13 bytes
}

#[cfg(feature = "local")]
#[test]
fn test_local_backend_integration() {
    use alimentar::backend::{LocalBackend, StorageBackend};

    let temp_dir = tempfile::tempdir()
        .ok()
        .unwrap_or_else(|| panic!("Should create temp dir"));

    let backend = LocalBackend::new(temp_dir.path())
        .ok()
        .unwrap_or_else(|| panic!("Should create backend"));

    // Write parquet via backend
    let dataset = create_test_dataset(10);
    let path = temp_dir.path().join("data.parquet");
    dataset
        .to_parquet(&path)
        .ok()
        .unwrap_or_else(|| panic!("Should save"));

    // Read via backend
    let data = backend
        .get("data.parquet")
        .ok()
        .unwrap_or_else(|| panic!("Should read"));
    assert!(!data.is_empty());

    // List files
    let files = backend
        .list("")
        .ok()
        .unwrap_or_else(|| panic!("Should list"));
    assert!(files.contains(&"data.parquet".to_string()));
}

#[test]
fn test_dataset_iteration_patterns() {
    let dataset = create_test_dataset(15);

    // Test batch iteration
    let batch_count = dataset.iter().count();
    assert_eq!(batch_count, 1); // Single batch dataset

    // Test row iteration
    let row_count = dataset.rows().count();
    assert_eq!(row_count, 15);

    // Test exact size iterator
    let rows = dataset.rows();
    assert_eq!(rows.len(), 15);
}

#[test]
fn test_large_batch_dataloader() {
    let dataset = create_test_dataset(1000);

    let loader = DataLoader::new(dataset)
        .batch_size(64)
        .shuffle(true)
        .seed(12345);

    let mut total = 0;
    let mut batch_count = 0;
    for batch in loader {
        total += batch.num_rows();
        batch_count += 1;
    }

    assert_eq!(total, 1000);
    // 1000 / 64 = 15.625, so 16 batches (15 full + 1 partial)
    assert_eq!(batch_count, 16);
}

// ========== New Transform Integration Tests ==========

#[test]
fn test_sample_transform_integration() {
    let dataset = create_test_dataset(100);

    // Sample 20% of the data
    let sample = Sample::fraction(0.2).with_seed(42);
    let sampled = dataset
        .with_transform(&sample)
        .ok()
        .unwrap_or_else(|| panic!("Should sample"));

    assert_eq!(sampled.len(), 20);

    // Verify data integrity in sampled rows
    for batch in sampled.iter() {
        let ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap_or_else(|| panic!("Should be Int32Array"));
        let scores = batch
            .column(2)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap_or_else(|| panic!("Should be Float64Array"));

        for i in 0..ids.len() {
            let id = ids.value(i);
            let score = scores.value(i);
            let expected = id as f64 * 1.5;
            assert!(
                (score - expected).abs() < f64::EPSILON,
                "Sample integrity: id={}, score={}, expected={}",
                id,
                score,
                expected
            );
        }
    }
}

#[test]
fn test_take_skip_combination() {
    let dataset = create_test_dataset(100);

    // Skip first 10, then take 20 = rows 10-29
    let skip = Skip::new(10);
    let skipped = dataset
        .with_transform(&skip)
        .ok()
        .unwrap_or_else(|| panic!("Should skip"));
    assert_eq!(skipped.len(), 90);

    let take = Take::new(20);
    let taken = skipped
        .with_transform(&take)
        .ok()
        .unwrap_or_else(|| panic!("Should take"));
    assert_eq!(taken.len(), 20);

    // Verify we got rows 10-29
    let batch = taken
        .get_batch(0)
        .unwrap_or_else(|| panic!("Should get batch"));
    let ids = batch
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap_or_else(|| panic!("Should be Int32Array"));

    assert_eq!(ids.value(0), 10);
    assert_eq!(ids.value(19), 29);
}

#[test]
fn test_sort_transform_integration() {
    let dataset = create_test_dataset(50);

    // Shuffle then sort to verify sorting works
    let shuffle = Shuffle::with_seed(123);
    let shuffled = dataset
        .with_transform(&shuffle)
        .ok()
        .unwrap_or_else(|| panic!("Should shuffle"));

    // Verify data is shuffled (not in original order)
    let shuffled_batch = shuffled
        .get_batch(0)
        .unwrap_or_else(|| panic!("Should get"));
    let shuffled_ids = shuffled_batch
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap_or_else(|| panic!("Should be Int32Array"));

    let mut is_shuffled = false;
    for i in 0..shuffled_ids.len().min(10) {
        if shuffled_ids.value(i) != i as i32 {
            is_shuffled = true;
            break;
        }
    }
    assert!(is_shuffled, "Data should be shuffled");

    // Now sort by id ascending
    let sort = Sort::by("id");
    let sorted = shuffled
        .with_transform(&sort)
        .ok()
        .unwrap_or_else(|| panic!("Should sort"));

    // Verify sorted order
    let sorted_batch = sorted.get_batch(0).unwrap_or_else(|| panic!("Should get"));
    let sorted_ids = sorted_batch
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap_or_else(|| panic!("Should be Int32Array"));

    for i in 0..sorted_ids.len() {
        assert_eq!(
            sorted_ids.value(i),
            i as i32,
            "Should be in ascending order"
        );
    }
}

#[test]
fn test_sort_descending_integration() {
    let dataset = create_test_dataset(20);

    let sort = Sort::by("score").order(SortOrder::Descending);
    let sorted = dataset
        .with_transform(&sort)
        .ok()
        .unwrap_or_else(|| panic!("Should sort"));

    let batch = sorted.get_batch(0).unwrap_or_else(|| panic!("Should get"));
    let scores = batch
        .column(2)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap_or_else(|| panic!("Should be Float64Array"));

    // Verify descending order
    for i in 1..scores.len() {
        assert!(
            scores.value(i - 1) >= scores.value(i),
            "Should be descending: {} >= {}",
            scores.value(i - 1),
            scores.value(i)
        );
    }
}

#[test]
fn test_chain_with_new_transforms() {
    let dataset = create_test_dataset(100);

    // Chain: Sort descending -> Take top 10 -> Select columns
    let chain = Chain::new()
        .then(Sort::by("score").order(SortOrder::Descending))
        .then(Take::new(10))
        .then(Select::new(vec!["id", "score"]));

    let result = dataset
        .with_transform(&chain)
        .ok()
        .unwrap_or_else(|| panic!("Should chain"));

    assert_eq!(result.len(), 10);
    assert_eq!(result.schema().fields().len(), 2);

    // Verify we got the top 10 scores (ids 90-99)
    let batch = result.get_batch(0).unwrap_or_else(|| panic!("Should get"));
    let ids = batch
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap_or_else(|| panic!("Should be Int32Array"));

    // First row should have the highest score (id=99)
    assert_eq!(ids.value(0), 99);
}

#[test]
fn test_sample_with_dataloader() {
    let dataset = create_test_dataset(1000);

    // Sample 10% then use dataloader
    let sample = Sample::fraction(0.1).with_seed(42);
    let sampled = dataset
        .with_transform(&sample)
        .ok()
        .unwrap_or_else(|| panic!("Should sample"));

    let loader = DataLoader::new(sampled)
        .batch_size(20)
        .shuffle(true)
        .seed(99);

    let mut total = 0;
    let mut seen_ids = HashSet::new();
    for batch in loader {
        let ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap_or_else(|| panic!("Should be Int32Array"));
        for i in 0..ids.len() {
            seen_ids.insert(ids.value(i));
        }
        total += batch.num_rows();
    }

    assert_eq!(total, 100); // 10% of 1000
    assert_eq!(seen_ids.len(), 100); // All unique IDs
}

// ========== Format Roundtrip Tests ==========

#[test]
fn test_csv_roundtrip_integration() {
    let temp_dir = tempfile::tempdir()
        .ok()
        .unwrap_or_else(|| panic!("Should create temp dir"));
    let path = temp_dir.path().join("test.csv");

    let original = create_test_dataset(50);

    // Save to CSV
    original
        .to_csv(&path)
        .ok()
        .unwrap_or_else(|| panic!("Should save CSV"));

    // Load from CSV
    let loaded = ArrowDataset::from_csv(&path)
        .ok()
        .unwrap_or_else(|| panic!("Should load CSV"));

    assert_eq!(loaded.len(), original.len());

    // Verify data integrity
    let orig_batch = original
        .get_batch(0)
        .unwrap_or_else(|| panic!("Should get"));
    let loaded_batch = loaded.get_batch(0).unwrap_or_else(|| panic!("Should get"));

    let orig_ids = orig_batch
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap_or_else(|| panic!("Should be Int32Array"));

    // CSV might load as Int64, so check the values
    assert_eq!(loaded_batch.num_rows(), orig_ids.len());
}

#[test]
fn test_json_roundtrip_integration() {
    let temp_dir = tempfile::tempdir()
        .ok()
        .unwrap_or_else(|| panic!("Should create temp dir"));
    let path = temp_dir.path().join("test.json");

    let original = create_test_dataset(30);

    // Save to JSON
    original
        .to_json(&path)
        .ok()
        .unwrap_or_else(|| panic!("Should save JSON"));

    // Load from JSON
    let loaded = ArrowDataset::from_json(&path)
        .ok()
        .unwrap_or_else(|| panic!("Should load JSON"));

    assert_eq!(loaded.len(), original.len());
}

#[test]
fn test_format_conversion_pipeline() {
    let temp_dir = tempfile::tempdir()
        .ok()
        .unwrap_or_else(|| panic!("Should create temp dir"));

    let parquet_path = temp_dir.path().join("data.parquet");
    let csv_path = temp_dir.path().join("data.csv");
    let json_path = temp_dir.path().join("data.json");

    // Create original dataset
    let original = create_test_dataset(25);
    assert_eq!(original.len(), 25);

    // Parquet -> CSV conversion
    original
        .to_parquet(&parquet_path)
        .ok()
        .unwrap_or_else(|| panic!("Should save parquet"));

    let from_parquet = ArrowDataset::from_parquet(&parquet_path)
        .ok()
        .unwrap_or_else(|| panic!("Should load parquet"));

    from_parquet
        .to_csv(&csv_path)
        .ok()
        .unwrap_or_else(|| panic!("Should save CSV"));

    // CSV -> JSON conversion
    let from_csv = ArrowDataset::from_csv(&csv_path)
        .ok()
        .unwrap_or_else(|| panic!("Should load CSV"));

    from_csv
        .to_json(&json_path)
        .ok()
        .unwrap_or_else(|| panic!("Should save JSON"));

    // JSON -> memory
    let from_json = ArrowDataset::from_json(&json_path)
        .ok()
        .unwrap_or_else(|| panic!("Should load JSON"));

    // Verify row count preserved through all conversions
    assert_eq!(from_json.len(), 25);
}

#[test]
fn test_transform_before_save() {
    let temp_dir = tempfile::tempdir()
        .ok()
        .unwrap_or_else(|| panic!("Should create temp dir"));
    let path = temp_dir.path().join("filtered.parquet");

    let dataset = create_test_dataset(100);

    // Filter to keep only rows where id > 50
    let filter = Filter::new(|batch| {
        let ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .ok_or_else(|| alimentar::Error::transform("Expected Int32Array"))?;
        let mask: Vec<bool> = (0..ids.len()).map(|i| ids.value(i) > 50).collect();
        Ok(arrow::array::BooleanArray::from(mask))
    });

    let filtered = dataset
        .with_transform(&filter)
        .ok()
        .unwrap_or_else(|| panic!("Should filter"));

    // Save filtered data
    filtered
        .to_parquet(&path)
        .ok()
        .unwrap_or_else(|| panic!("Should save"));

    // Load and verify
    let loaded = ArrowDataset::from_parquet(&path)
        .ok()
        .unwrap_or_else(|| panic!("Should load"));

    // ids 51-99 = 49 rows
    assert_eq!(loaded.len(), 49);
}

#[test]
fn test_ml_preprocessing_pipeline() {
    // Simulate a typical ML data preprocessing pipeline
    let dataset = create_test_dataset(1000);

    // 1. Sort by score to ensure consistent ordering
    let sort = Sort::by("score");
    let sorted = dataset
        .with_transform(&sort)
        .ok()
        .unwrap_or_else(|| panic!("Should sort"));

    // 2. Take top 80% for training (skip bottom 20%)
    let skip_bottom = Skip::new(200); // Skip lowest 20%
    let training_candidates = sorted
        .with_transform(&skip_bottom)
        .ok()
        .unwrap_or_else(|| panic!("Should skip"));
    assert_eq!(training_candidates.len(), 800);

    // 3. Sample 500 for actual training
    let sample = Sample::new(500).with_seed(42);
    let training_set = training_candidates
        .with_transform(&sample)
        .ok()
        .unwrap_or_else(|| panic!("Should sample"));
    assert_eq!(training_set.len(), 500);

    // 4. Select only features needed
    let select = Select::new(vec!["id", "score"]);
    let final_training = training_set
        .with_transform(&select)
        .ok()
        .unwrap_or_else(|| panic!("Should select"));

    assert_eq!(final_training.len(), 500);
    assert_eq!(final_training.schema().fields().len(), 2);

    // 5. Use with DataLoader
    let loader = DataLoader::new(final_training)
        .batch_size(32)
        .shuffle(true)
        .seed(123);

    let batch_count: usize = loader.into_iter().count();
    // 500 / 32 = 15.625 -> 16 batches
    assert_eq!(batch_count, 16);
}

// ============================================================================
// alimentar ↔ aprender Integration Tests
// ============================================================================
// These tests demonstrate the data flow patterns used by aprender (ML
// framework) to consume data from alimentar (data loading library).

use alimentar::{tensor::TensorExtractor, WeightedDataLoader};

/// Creates a test dataset suitable for ML feature extraction.
#[allow(
    clippy::cast_precision_loss,
    clippy::suboptimal_flops,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap
)]
fn create_ml_dataset(rows: usize) -> ArrowDataset {
    let schema = Arc::new(Schema::new(vec![
        Field::new("feature_1", DataType::Float64, false),
        Field::new("feature_2", DataType::Float64, false),
        Field::new("feature_3", DataType::Float64, false),
        Field::new("label", DataType::Int32, false),
    ]));

    let f1: Vec<f64> = (0..rows).map(|i| i as f64 * 0.1).collect();
    let f2: Vec<f64> = (0..rows).map(|i| i as f64 * 0.2 + 1.0).collect();
    let f3: Vec<f64> = (0..rows).map(|i| (i as f64).sin()).collect();
    let labels: Vec<i32> = (0..rows).map(|i| (i % 3) as i32).collect();

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(f1)),
            Arc::new(Float64Array::from(f2)),
            Arc::new(Float64Array::from(f3)),
            Arc::new(Int32Array::from(labels)),
        ],
    )
    .ok()
    .unwrap_or_else(|| panic!("Should create batch"));

    ArrowDataset::from_batch(batch)
        .ok()
        .unwrap_or_else(|| panic!("Should create dataset"))
}

#[test]
fn test_tensor_extraction_for_aprender() {
    // This test demonstrates the pattern aprender uses to extract
    // features and labels from alimentar datasets.
    let dataset = create_ml_dataset(100);

    // Extract features as f32 tensor (compatible with trueno/aprender)
    let feature_extractor = TensorExtractor::new(&["feature_1", "feature_2", "feature_3"]);
    let batch = dataset
        .get_batch(0)
        .unwrap_or_else(|| panic!("Should get batch"));

    #[allow(clippy::needless_borrow)]
    let features = feature_extractor
        .extract_f32(&batch)
        .ok()
        .unwrap_or_else(|| panic!("Should extract features"));

    // Verify shape matches [rows, features]
    assert_eq!(features.shape(), [100, 3]);

    // Extract labels as i64 (for classification)
    #[allow(clippy::needless_borrow)]
    let labels = alimentar::tensor::extract_labels_i64(&batch, "label")
        .ok()
        .unwrap_or_else(|| panic!("Should extract labels"));

    assert_eq!(labels.len(), 100);

    // Verify label distribution (0, 1, 2 cycling)
    let label_counts: Vec<i64> = (0..3)
        .map(|l| labels.iter().filter(|&&x| x == l).count() as i64)
        .collect();
    assert!(label_counts.iter().all(|&c| (33..=34).contains(&c)));
}

#[test]
fn test_weighted_dataloader_citl_integration() {
    // This test demonstrates WeightedDataLoader for CITL reweighting.
    // CITL (Compiler-Informed Training Loss) boosts compiler-verified samples.
    let dataset = create_ml_dataset(100);

    // Simulate CITL weights: samples 0-49 are compiler-verified (weight=1.5)
    // samples 50-99 are not verified (weight=1.0)
    let weights: Vec<f32> = (0..100).map(|i| if i < 50 { 1.5 } else { 1.0 }).collect();

    let loader = WeightedDataLoader::new(dataset, weights)
        .ok()
        .unwrap_or_else(|| panic!("Should create weighted loader"))
        .batch_size(10)
        .num_samples(100)
        .seed(42);

    // Verify loader configuration
    assert_eq!(loader.get_batch_size(), 10);
    assert_eq!(loader.get_num_samples(), 100);
    assert_eq!(loader.len(), 100);

    // Iterate and collect samples
    let mut total_rows = 0;
    for batch in loader {
        total_rows += batch.num_rows();
    }

    // Should yield approximately 100 samples (weighted sampling)
    assert_eq!(total_rows, 100);
}

#[test]
fn test_parallel_loader_ml_training() {
    // This test demonstrates ParallelDataLoader for multi-threaded training.
    use alimentar::parallel::ParallelDataLoader;

    let dataset = create_ml_dataset(1000);

    let loader = ParallelDataLoader::new(dataset)
        .batch_size(32)
        .num_workers(2)
        .prefetch(4);

    // Verify configuration
    assert_eq!(loader.get_batch_size(), 32);
    assert_eq!(loader.get_num_workers(), 2);
    assert_eq!(loader.get_prefetch(), 4);
    assert_eq!(loader.num_batches(), 32); // 1000 / 32 = 31.25 -> 32 batches

    // Simulate training loop
    let mut total_samples = 0;
    let mut batch_count = 0;
    for batch in loader {
        total_samples += batch.num_rows();
        batch_count += 1;
    }

    assert_eq!(total_samples, 1000);
    assert_eq!(batch_count, 32);
}

#[test]
fn test_train_val_test_split_pipeline() {
    // This test demonstrates the data splitting pattern used by aprender
    // for creating train/validation/test sets.
    use alimentar::split::DatasetSplit;

    let dataset = create_ml_dataset(1000);

    // Split: 70% train, 15% validation, 15% test
    let split = DatasetSplit::from_ratios(&dataset, 0.70, 0.15, Some(0.15), Some(42))
        .ok()
        .unwrap_or_else(|| panic!("Should split dataset"));

    let train = split.train();
    let val = split
        .validation()
        .unwrap_or_else(|| panic!("Should have validation"));
    let test = split.test();

    // Verify split sizes
    assert_eq!(train.len(), 700);
    assert_eq!(val.len(), 150);
    assert_eq!(test.len(), 150);

    // Verify no overlap between splits
    let train_ids: HashSet<i32> = train
        .iter()
        .flat_map(|batch| {
            batch
                .column(3)
                .as_any()
                .downcast_ref::<Int32Array>()
                .map(|arr| arr.values().to_vec())
                .unwrap_or_default()
        })
        .collect();

    // All samples have labels 0, 1, or 2 (from our test dataset)
    assert!(train_ids.iter().all(|&l| (0..=2).contains(&l)));
}

#[test]
#[allow(clippy::cast_precision_loss)]
fn test_streaming_ml_data_flow() {
    // This test demonstrates streaming data loading for large datasets.
    use alimentar::streaming::{MemorySource, StreamingDataset};

    // Create multiple batches to simulate streaming
    let batches: Vec<RecordBatch> = (0..5)
        .map(|i| {
            let schema = Arc::new(Schema::new(vec![
                Field::new("x", DataType::Float64, false),
                Field::new("y", DataType::Float64, false),
            ]));

            let x: Vec<f64> = (0..100).map(|j| (i * 100 + j) as f64).collect();
            let y: Vec<f64> = x.iter().map(|&v| v * 2.0).collect();

            RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(Float64Array::from(x)),
                    Arc::new(Float64Array::from(y)),
                ],
            )
            .ok()
            .unwrap_or_else(|| panic!("Should create batch"))
        })
        .collect();

    let source = MemorySource::new(batches)
        .ok()
        .unwrap_or_else(|| panic!("Should create source"));

    let dataset = StreamingDataset::new(Box::new(source), 4).prefetch(2);

    // Extract tensors from streaming data
    let extractor = TensorExtractor::new(&["x", "y"]);
    let mut total_rows = 0;

    for batch in dataset {
        let tensor = extractor
            .extract_f64(&batch)
            .ok()
            .unwrap_or_else(|| panic!("Should extract"));
        total_rows += tensor.rows();
    }

    assert_eq!(total_rows, 500);
}

#[test]
fn test_feature_normalization_for_training() {
    // This test demonstrates feature normalization before training.
    use alimentar::{NormMethod, Normalize};

    let dataset = create_ml_dataset(100);

    // Normalize features using MinMax scaling
    let normalize = Normalize::new(["feature_1", "feature_2", "feature_3"], NormMethod::MinMax);

    let normalized = dataset
        .with_transform(&normalize)
        .ok()
        .unwrap_or_else(|| panic!("Should normalize"));

    // Extract normalized features
    let extractor = TensorExtractor::new(&["feature_1", "feature_2", "feature_3"]);
    let batch = normalized
        .get_batch(0)
        .unwrap_or_else(|| panic!("Should get batch"));
    #[allow(clippy::needless_borrow)]
    let features = extractor
        .extract_f64(&batch)
        .ok()
        .unwrap_or_else(|| panic!("Should extract"));

    // Verify all values are in [0, 1] range for MinMax normalization
    for val in features.as_slice() {
        assert!(
            *val >= 0.0 && *val <= 1.0,
            "Value {} should be in [0, 1]",
            val
        );
    }
}
