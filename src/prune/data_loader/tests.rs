//! Tests for calibration data loader.

#![allow(clippy::module_inception)]
#[cfg(test)]
mod tests {
    use crate::prune::data_loader::{CalibrationDataConfig, CalibrationDataLoader};

    // =========================================================================
    // CalibrationDataConfig Tests
    // =========================================================================

    #[test]
    fn test_config_default() {
        // TEST_ID: DL-001
        let config = CalibrationDataConfig::default();
        assert_eq!(config.num_samples(), 128);
        assert_eq!(config.batch_size(), 1);
        assert_eq!(config.sequence_length(), 2048);
        assert_eq!(config.dataset(), "c4");
        assert_eq!(config.seed(), 42);
    }

    #[test]
    fn test_config_builder() {
        // TEST_ID: DL-002
        let config = CalibrationDataConfig::new()
            .with_num_samples(256)
            .with_batch_size(4)
            .with_sequence_length(1024)
            .with_dataset("wikitext")
            .with_seed(123);

        assert_eq!(config.num_samples(), 256);
        assert_eq!(config.batch_size(), 4);
        assert_eq!(config.sequence_length(), 1024);
        assert_eq!(config.dataset(), "wikitext");
        assert_eq!(config.seed(), 123);
    }

    #[test]
    fn test_config_batch_size_minimum() {
        // TEST_ID: DL-003
        let config = CalibrationDataConfig::new().with_batch_size(0);
        assert_eq!(config.batch_size(), 1, "DL-003 FALSIFIED: Batch size should be minimum 1");
    }

    #[test]
    fn test_config_num_batches() {
        // TEST_ID: DL-004
        let config = CalibrationDataConfig::new().with_num_samples(10).with_batch_size(3);
        // 10 samples / 3 per batch = ceil(10/3) = 4 batches
        assert_eq!(
            config.num_batches(),
            4,
            "DL-004 FALSIFIED: 10 samples with batch_size 3 should be 4 batches"
        );
    }

    #[test]
    fn test_config_cache_dir() {
        // TEST_ID: DL-005
        let config = CalibrationDataConfig::new().with_cache_dir("/tmp/cache");
        assert_eq!(
            config.cache_dir().map(|p| p.to_str().expect("config should be valid")),
            Some("/tmp/cache")
        );
    }

    #[test]
    fn test_config_serialize_json() {
        // TEST_ID: DL-006
        let config = CalibrationDataConfig::new().with_num_samples(64);
        let json = serde_json::to_string(&config).expect("JSON serialization should succeed");
        let deserialized: CalibrationDataConfig =
            serde_json::from_str(&json).expect("JSON deserialization should succeed");
        assert_eq!(config.num_samples(), deserialized.num_samples());
    }

    // =========================================================================
    // CalibrationDataLoader Tests
    // =========================================================================

    #[test]
    fn test_loader_new() {
        // TEST_ID: DL-010
        let config = CalibrationDataConfig::new();
        let loader = CalibrationDataLoader::new(config);
        assert!(!loader.is_loaded());
        assert_eq!(loader.num_batches(), 0);
    }

    #[test]
    fn test_loader_with_synthetic_data() {
        // TEST_ID: DL-011
        let config = CalibrationDataConfig::new().with_num_samples(10).with_batch_size(3);
        let loader = CalibrationDataLoader::with_synthetic_data(config);

        assert!(loader.is_loaded());
        assert_eq!(loader.num_batches(), 4, "DL-011 FALSIFIED: Should have 4 batches");
    }

    #[test]
    fn test_loader_load() {
        // TEST_ID: DL-012
        let config = CalibrationDataConfig::new().with_num_samples(5);
        let mut loader = CalibrationDataLoader::new(config);

        assert!(!loader.is_loaded());
        loader.load().expect("load should succeed");
        assert!(loader.is_loaded());
    }

    #[test]
    fn test_loader_get_batch() {
        // TEST_ID: DL-013
        let config = CalibrationDataConfig::new().with_num_samples(10).with_batch_size(5);
        let loader = CalibrationDataLoader::with_synthetic_data(config);

        assert!(loader.get_batch(0).is_some());
        assert!(loader.get_batch(1).is_some());
        assert!(loader.get_batch(2).is_none()); // Only 2 batches
    }

    #[test]
    fn test_loader_iter() {
        // TEST_ID: DL-014
        let config = CalibrationDataConfig::new().with_num_samples(9).with_batch_size(3);
        let loader = CalibrationDataLoader::with_synthetic_data(config);

        let batches: Vec<_> = loader.iter().collect();
        assert_eq!(batches.len(), 3, "DL-014 FALSIFIED: Should iterate over 3 batches");
    }

    #[test]
    fn test_loader_iter_size_hint() {
        // TEST_ID: DL-015
        let config = CalibrationDataConfig::new().with_num_samples(6).with_batch_size(2);
        let loader = CalibrationDataLoader::with_synthetic_data(config);

        let iter = loader.iter();
        assert_eq!(iter.len(), 3);
    }

    #[test]
    fn test_loader_reset() {
        // TEST_ID: DL-016
        let config = CalibrationDataConfig::new().with_num_samples(5);
        let mut loader = CalibrationDataLoader::with_synthetic_data(config);
        loader.position = 3;
        loader.reset();
        assert_eq!(loader.position, 0, "DL-016 FALSIFIED: Reset should set position to 0");
    }

    #[test]
    fn test_loader_deterministic_with_seed() {
        // TEST_ID: DL-017
        let config = CalibrationDataConfig::new().with_num_samples(5).with_seed(12345);

        let loader1 = CalibrationDataLoader::with_synthetic_data(config.clone());
        let loader2 = CalibrationDataLoader::with_synthetic_data(config);

        let batch1 = loader1.get_batch(0).expect("load should succeed");
        let batch2 = loader2.get_batch(0).expect("load should succeed");

        // Same seed should produce same data
        let data1: Vec<f32> = batch1.inputs.data().to_vec();
        let data2: Vec<f32> = batch2.inputs.data().to_vec();
        assert_eq!(data1, data2, "DL-017 FALSIFIED: Same seed should produce same data");
    }

    #[test]
    fn test_loader_different_seeds_different_data() {
        // TEST_ID: DL-018
        let config1 = CalibrationDataConfig::new().with_num_samples(5).with_seed(111);
        let config2 = CalibrationDataConfig::new().with_num_samples(5).with_seed(222);

        let loader1 = CalibrationDataLoader::with_synthetic_data(config1);
        let loader2 = CalibrationDataLoader::with_synthetic_data(config2);

        let batch1 = loader1.get_batch(0).expect("load should succeed");
        let batch2 = loader2.get_batch(0).expect("load should succeed");

        let data1: Vec<f32> = batch1.inputs.data().to_vec();
        let data2: Vec<f32> = batch2.inputs.data().to_vec();
        assert_ne!(data1, data2, "DL-018 FALSIFIED: Different seeds should produce different data");
    }

    #[test]
    fn test_loader_batch_sizes_correct() {
        // TEST_ID: DL-019
        let config = CalibrationDataConfig::new()
            .with_num_samples(10)
            .with_batch_size(4)
            .with_sequence_length(128);
        let loader = CalibrationDataLoader::with_synthetic_data(config);

        // First two batches should have 4 samples each (4 * 128 = 512 elements)
        // Last batch should have 2 samples (2 * 128 = 256 elements)
        let batch0 = loader.get_batch(0).expect("load should succeed");
        let batch1 = loader.get_batch(1).expect("load should succeed");
        let batch2 = loader.get_batch(2).expect("load should succeed");

        assert_eq!(batch0.inputs.len(), 4 * 128);
        assert_eq!(batch1.inputs.len(), 4 * 128);
        assert_eq!(batch2.inputs.len(), 2 * 128);
    }

    #[test]
    fn test_loader_config_access() {
        // TEST_ID: DL-020
        let config = CalibrationDataConfig::new().with_num_samples(50);
        let loader = CalibrationDataLoader::new(config);
        assert_eq!(loader.config().num_samples(), 50);
    }

    // =========================================================================
    // Edge Case Tests
    // =========================================================================

    #[test]
    fn test_loader_single_sample() {
        // TEST_ID: DL-030
        let config = CalibrationDataConfig::new().with_num_samples(1).with_batch_size(1);
        let loader = CalibrationDataLoader::with_synthetic_data(config);

        assert_eq!(loader.num_batches(), 1);
        assert!(loader.get_batch(0).is_some());
    }

    #[test]
    fn test_loader_batch_size_larger_than_samples() {
        // TEST_ID: DL-031
        let config = CalibrationDataConfig::new().with_num_samples(3).with_batch_size(10);
        let loader = CalibrationDataLoader::with_synthetic_data(config);

        assert_eq!(
            loader.num_batches(),
            1,
            "DL-031 FALSIFIED: Should have 1 batch when batch_size > num_samples"
        );
        let batch = loader.get_batch(0).expect("load should succeed");
        // Should have only 3 samples worth of data
        assert_eq!(batch.inputs.len(), 3 * 2048); // 3 samples * default seq len
    }

    #[test]
    fn test_loader_empty_iter() {
        // TEST_ID: DL-032
        let config = CalibrationDataConfig::new();
        let loader = CalibrationDataLoader::new(config);

        let count = loader.iter().count();
        assert_eq!(count, 0, "DL-032 FALSIFIED: Unloaded loader should have empty iterator");
    }

    #[test]
    fn test_loader_clone() {
        // TEST_ID: DL-033
        let config = CalibrationDataConfig::new().with_num_samples(5);
        let loader = CalibrationDataLoader::with_synthetic_data(config);
        let cloned = loader.clone();

        assert_eq!(loader.num_batches(), cloned.num_batches());
        assert!(cloned.is_loaded());
    }
}
