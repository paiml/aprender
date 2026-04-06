//! Tests for dataset module

use super::*;
use ndarray::Array2;

// =========================================================================
// Split Tests
// =========================================================================

#[test]
fn test_split_display() {
    assert_eq!(format!("{}", Split::Train), "train");
    assert_eq!(format!("{}", Split::Validation), "validation");
    assert_eq!(format!("{}", Split::Test), "test");
}

// =========================================================================
// DatasetOptions Tests
// =========================================================================

#[test]
fn test_dataset_options_default() {
    let opts = DatasetOptions::default();
    assert_eq!(opts.split, Split::Train);
    assert!(opts.shuffle);
    assert!(!opts.streaming);
}

#[test]
fn test_dataset_options_builders() {
    let train = DatasetOptions::train();
    assert_eq!(train.split, Split::Train);
    assert!(train.shuffle);

    let val = DatasetOptions::validation();
    assert_eq!(val.split, Split::Validation);
    assert!(!val.shuffle);

    let test = DatasetOptions::test();
    assert_eq!(test.split, Split::Test);
    assert!(!test.shuffle);
}

#[test]
fn test_dataset_options_chaining() {
    let opts = DatasetOptions::train().max_examples(100).streaming(true).shuffle(false).seed(123);

    assert_eq!(opts.max_examples, Some(100));
    assert!(opts.streaming);
    assert!(!opts.shuffle);
    assert_eq!(opts.seed, Some(123));
}

// =========================================================================
// Example Tests
// =========================================================================

#[test]
fn test_example_creation() {
    let example = Example::from_tokens(vec![1, 2, 3, 4, 5]);
    assert_eq!(example.len(), 5);
    assert!(!example.is_empty());
    assert_eq!(example.attention_mask, vec![1, 1, 1, 1, 1]);
    assert!(example.labels.is_none());
}

#[test]
fn test_example_with_labels() {
    let example = Example::from_tokens(vec![1, 2, 3]).with_labels(vec![2, 3, 4]);
    assert_eq!(example.labels, Some(vec![2, 3, 4]));
}

#[test]
fn test_example_with_text() {
    let example = Example::from_tokens(vec![1, 2]).with_text("hello world");
    assert_eq!(example.text, Some("hello world".to_string()));
}

// =========================================================================
// Dataset Tests
// =========================================================================

#[test]
fn test_dataset_mock() {
    let dataset = Dataset::mock(10, 32);
    assert_eq!(dataset.len(), 10);
    assert_eq!(dataset.name(), "mock_dataset");

    let first = dataset.get(0).expect("key should exist");
    assert_eq!(first.len(), 32);
    assert!(first.labels.is_some());
}

#[test]
fn test_dataset_iteration() {
    let dataset = Dataset::mock(3, 16);
    let collected: Vec<_> = dataset.collect();
    assert_eq!(collected.len(), 3);
}

#[test]
fn test_dataset_shuffle() {
    let mut dataset1 = Dataset::mock(100, 16);
    let mut dataset2 = Dataset::mock(100, 16);

    dataset1.shuffle(42);
    dataset2.shuffle(42);

    // Same seed should produce same order
    for (e1, e2) in dataset1.examples().iter().zip(dataset2.examples().iter()) {
        assert_eq!(e1.input_ids, e2.input_ids);
    }
}

#[test]
fn test_dataset_take() {
    let dataset = Dataset::mock(100, 16).take(10);
    assert_eq!(dataset.len(), 10);
}

// =========================================================================
// HfDatasetFetcher Tests
// =========================================================================

#[test]
fn test_fetcher_creation() {
    let fetcher = HfDatasetFetcher::new();
    assert!(fetcher.is_ok());
}

#[test]
fn test_fetcher_fetch_mock() {
    let fetcher = HfDatasetFetcher::default();
    let dataset = fetcher.fetch("wikitext", DatasetOptions::train().max_examples(50));
    assert!(dataset.is_ok());
    assert_eq!(dataset.expect("operation should succeed").len(), 50);
}

#[test]
fn test_fetcher_invalid_id() {
    let fetcher = HfDatasetFetcher::default();
    let result = fetcher.fetch("", DatasetOptions::default());
    assert!(result.is_err());
}

#[test]
fn test_fetcher_load_nonexistent_parquet() {
    let fetcher = HfDatasetFetcher::default();
    let result = fetcher.load_parquet(std::path::Path::new("/nonexistent/file.parquet"));
    assert!(result.is_err());
}

// =========================================================================
// Batch Tests
// =========================================================================

#[test]
fn test_batch_dimensions() {
    let examples = vec![Example::from_tokens(vec![1, 2, 3]), Example::from_tokens(vec![4, 5])];
    let collator = DistillationCollator::default();
    let batch = collator.collate(&examples);

    assert_eq!(batch.batch_size(), 2);
    assert_eq!(batch.max_seq_len(), 3);
}

// =========================================================================
// DistillationCollator Tests
// =========================================================================

#[test]
fn test_collator_default() {
    let collator = DistillationCollator::default();
    assert_eq!(collator.pad_token_id, 0);
    assert_eq!(collator.max_length, 512);
    assert!(!collator.pad_left);
}

#[test]
fn test_collator_builder() {
    let collator = DistillationCollator::new(1).max_length(256).pad_left(true);
    assert_eq!(collator.pad_token_id, 1);
    assert_eq!(collator.max_length, 256);
    assert!(collator.pad_left);
}

#[test]
fn test_collator_empty_batch() {
    let collator = DistillationCollator::default();
    let batch = collator.collate(&[]);
    assert_eq!(batch.batch_size(), 0);
}

#[test]
fn test_collator_right_padding() {
    let examples = vec![Example::from_tokens(vec![1, 2, 3]), Example::from_tokens(vec![4, 5])];
    let collator = DistillationCollator::new(0);
    let batch = collator.collate(&examples);

    // First example: [1, 2, 3]
    assert_eq!(batch.input_ids[[0, 0]], 1);
    assert_eq!(batch.input_ids[[0, 1]], 2);
    assert_eq!(batch.input_ids[[0, 2]], 3);

    // Second example: [4, 5, PAD]
    assert_eq!(batch.input_ids[[1, 0]], 4);
    assert_eq!(batch.input_ids[[1, 1]], 5);
    assert_eq!(batch.input_ids[[1, 2]], 0); // Padded

    // Attention mask
    assert_eq!(batch.attention_mask[[0, 2]], 1); // Not padded
    assert_eq!(batch.attention_mask[[1, 2]], 0); // Padded
}

#[test]
fn test_collator_left_padding() {
    let examples = vec![Example::from_tokens(vec![1, 2, 3]), Example::from_tokens(vec![4, 5])];
    let collator = DistillationCollator::new(0).pad_left(true);
    let batch = collator.collate(&examples);

    // Second example: [PAD, 4, 5]
    assert_eq!(batch.input_ids[[1, 0]], 0); // Padded
    assert_eq!(batch.input_ids[[1, 1]], 4);
    assert_eq!(batch.input_ids[[1, 2]], 5);

    // Attention mask
    assert_eq!(batch.attention_mask[[1, 0]], 0); // Padded
    assert_eq!(batch.attention_mask[[1, 1]], 1);
}

#[test]
fn test_collator_truncation() {
    let examples = vec![Example::from_tokens(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10])];
    let collator = DistillationCollator::new(0).max_length(5);
    let batch = collator.collate(&examples);

    assert_eq!(batch.max_seq_len(), 5);
    assert_eq!(batch.input_ids[[0, 0]], 1);
    assert_eq!(batch.input_ids[[0, 4]], 5);
}

#[test]
fn test_collator_with_labels() {
    let examples = vec![
        Example::from_tokens(vec![1, 2]).with_labels(vec![2, 3]),
        Example::from_tokens(vec![4, 5, 6]).with_labels(vec![5, 6, 7]),
    ];
    let collator = DistillationCollator::new(0);
    let batch = collator.collate(&examples);

    assert!(batch.labels.is_some());
    let labels = batch.labels.expect("operation should succeed");
    assert_eq!(labels[[0, 0]], 2);
    assert_eq!(labels[[1, 2]], 7);
}

#[test]
fn test_collator_lengths() {
    let examples = vec![
        Example::from_tokens(vec![1, 2, 3]),
        Example::from_tokens(vec![4, 5]),
        Example::from_tokens(vec![6]),
    ];
    let collator = DistillationCollator::default();
    let batch = collator.collate(&examples);

    assert_eq!(batch.lengths, vec![3, 2, 1]);
}

#[test]
fn test_collator_batch_dataset() {
    let dataset = Dataset::mock(10, 16);
    let collator = DistillationCollator::default();
    let batches = collator.batch_dataset(&dataset, 3);

    assert_eq!(batches.len(), 4); // 10 / 3 = 3 full + 1 partial
    assert_eq!(batches[0].batch_size(), 3);
    assert_eq!(batches[3].batch_size(), 1); // Last batch is partial
}

// =========================================================================
// TeacherCache Tests
// =========================================================================

#[test]
fn test_cache_new() {
    let cache = TeacherCache::new();
    let stats = cache.stats();
    assert_eq!(stats.hits, 0);
    assert_eq!(stats.misses, 0);
}

#[test]
fn test_cache_logits() {
    let mut cache = TeacherCache::new();
    let logits = Array2::<f32>::zeros((4, 100));

    // Miss first
    assert!(cache.get_logits(0).is_none());

    // Cache and hit
    cache.cache_logits(0, logits.clone());
    assert!(cache.get_logits(0).is_some());

    let stats = cache.stats();
    assert_eq!(stats.hits, 1);
    assert_eq!(stats.misses, 1);
}

#[test]
fn test_cache_hidden_states() {
    let mut cache = TeacherCache::new();
    let states = vec![Array2::<f32>::zeros((4, 768)); 12];

    cache.cache_hidden_states(0, states);
    assert!(cache.get_hidden_states(0).is_some());
}

#[test]
fn test_cache_clear() {
    let mut cache = TeacherCache::new();
    cache.cache_logits(0, Array2::<f32>::zeros((4, 100)));
    cache.clear();

    let stats = cache.stats();
    assert_eq!(stats.logits_cached, 0);
}

#[test]
fn test_cache_hit_rate() {
    let mut cache = TeacherCache::new();
    let logits = Array2::<f32>::zeros((4, 100));

    // 1 miss, then 3 hits
    let _ = cache.get_logits(0);
    cache.cache_logits(0, logits);
    let _ = cache.get_logits(0);
    let _ = cache.get_logits(0);
    let _ = cache.get_logits(0);

    let stats = cache.stats();
    assert_eq!(stats.hit_rate(), 0.75); // 3 hits / 4 total
}
