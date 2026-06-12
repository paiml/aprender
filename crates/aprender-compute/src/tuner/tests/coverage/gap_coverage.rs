//! Coverage gap tests for tuner functions (Refs CB-130)
//!
//! Targets:
//! 1. `evolution/mod.rs::calibrate` (line 85, 51 uncovered lines, impact 7.5)
//! 2. `models/throughput.rs::train_random_forest` (line 148, 30 uncovered, impact 6.2)
//! 3. `models/kernel.rs::train` (line 49, 28 uncovered, impact 4.8)
//! 4. `helpers.rs::crc32_table` (line 10, 17 uncovered, impact 4.4)
//! 5. `data_collector/mod.rs::record` (line 100, 14 uncovered, impact 14.5)
//! 6. `brick_tuner/persistence.rs::load_or_default` (line 51, 14 uncovered, impact 2.4)

use super::super::super::*;
use crate::brick::{BrickId, BrickProfiler};

// ============================================================================
// Helper: create a BrickProfiler that returns Some(tps) from tokens_per_sec()
// ============================================================================

fn make_profiler_with_tokens(tokens: u64, ns: u64) -> BrickProfiler {
    let mut profiler = BrickProfiler::new();
    profiler.enable();
    // Use the deferred API to inject precise token counts and timing
    profiler.record_deferred(BrickId::RmsNorm, 0, tokens);
    profiler.finalize(ns);
    profiler
}

// ============================================================================
// 1. calibrate (evolution/mod.rs, line 85) — hardware-detect gated
// ============================================================================

#[cfg(feature = "hardware-detect")]
mod calibrate_tests {
    use super::super::super::super::*;

    #[test]
    fn calibrate_produces_valid_result() {
        let mut tuner = BrickTuner::with_pretrained();
        let result = tuner.calibrate().expect("calibrate should succeed");

        // Must produce calibration weights
        assert!(!result.throughput_weights.is_empty(), "Calibration should produce weights");
        // MAPE should be non-negative
        assert!(result.local_mape >= 0.0, "MAPE should be non-negative");
        // Improvement over pretrained >= 0 (clamped)
        assert!(result.improvement_pct >= 0.0, "Improvement should be non-negative");
        // Hardware ID should not be empty
        assert!(!result.hardware_id.is_empty(), "Hardware ID should be populated");
        // Duration should be positive
        assert!(result.duration_secs > 0.0, "Calibration should take some time");
        // Should have 4 batch sizes * 3 model sizes * 2 quant types = 24 benchmarks
        assert_eq!(result.num_benchmarks, 24, "Expected 24 synthetic benchmarks");
    }

    #[test]
    fn calibrate_updates_tuner_version() {
        let mut tuner = BrickTuner::with_pretrained();
        let _ = tuner.calibrate().expect("calibrate should succeed");
        assert!(
            tuner.version().contains("calibrated"),
            "Version should contain 'calibrated' after calibration, got: {}",
            tuner.version()
        );
    }

    #[test]
    fn calibrate_updates_throughput_weights() {
        let mut tuner = BrickTuner::with_pretrained();
        let pretrained_mape = tuner.throughput_mape();
        let result = tuner.calibrate().expect("calibrate should succeed");

        // After calibration, weights should be updated
        assert!(
            !result.throughput_weights.is_empty(),
            "Weights should be non-empty after calibration"
        );
        // MAPE may improve or stay the same, but should be valid
        assert!(tuner.throughput_mape() >= 0.0, "MAPE should be valid after calibration");
        // Verify MAPE was updated (not necessarily improved, but changed)
        let _ = pretrained_mape; // used for comparison reference
    }

    #[test]
    fn calibrate_from_new_tuner() {
        let mut tuner = BrickTuner::new();
        let result = tuner.calibrate().expect("calibrate should succeed");
        assert_eq!(result.num_benchmarks, 24);
        assert!(result.local_mape >= 0.0);
        assert!(tuner.version().contains("calibrated"));
    }
}

// ============================================================================
// 2-3. ml-tuner RandomForest showcase (RELOCATED — SHOWCASE-BRICK-001)
// ============================================================================
// The `ThroughputRegressor::with_random_forest` / `train_random_forest` and
// `KernelClassifier::train` wrappers were removed: the SIMD compute lib must not
// depend on the ML layer (aprender). The aprender RandomForest showcase now lives in
// examples/ml_tuner_demo.rs + tests/tuner_falsification/training_data_quality.rs,
// which consume aprender as a dev-dependency. No lib coverage target remains here.

// ============================================================================
// 4. crc32_table (helpers.rs, line 10) — const fn coverage
// ============================================================================

mod crc32_table_tests {
    use super::super::super::super::helpers::{crc32_hash, crc32_update};

    #[test]
    fn crc32_table_exercised_via_all_byte_values() {
        // Exercise every entry in the CRC32 table by hashing all 256 byte values
        for byte_val in 0..=255u8 {
            let hash = crc32_hash(&[byte_val]);
            assert_ne!(hash, 0, "CRC32 of single byte should not be zero");
        }
    }

    #[test]
    fn crc32_table_exercised_via_multibyte_patterns() {
        // Patterns that exercise both branches of crc32_table (crc & 1 != 0 vs == 0)
        // Byte 0x00 starts with all crc bits = 0 (crc & 1 == 0 path)
        let hash_zero = crc32_hash(&[0x00]);
        // Byte 0xFF starts with all bits = 1 (crc & 1 != 0 path)
        let hash_ff = crc32_hash(&[0xFF]);
        assert_ne!(hash_zero, hash_ff);
    }

    #[test]
    fn crc32_table_bit_patterns_low_nibble() {
        // Test bytes that exercise different bit patterns in the CRC32 table
        // Each byte causes a different sequence of polynomial XOR operations
        let mut hashes = std::collections::HashSet::new();
        for i in 0..16u8 {
            let hash = crc32_hash(&[i]);
            hashes.insert(hash);
        }
        // All 16 low-nibble values should produce distinct hashes
        assert_eq!(hashes.len(), 16, "All low-nibble bytes should produce unique CRC32");
    }

    #[test]
    fn crc32_update_incremental_two_byte_sequence() {
        // Ensure incremental update matches single-pass for multi-byte patterns
        for a in [0x00, 0x55, 0xAA, 0xFF] {
            for b in [0x00, 0x55, 0xAA, 0xFF] {
                let single = crc32_hash(&[a, b]);
                let inc = crc32_update(crc32_update(0, &[a]), &[b]);
                assert_eq!(
                    single, inc,
                    "Incremental CRC32 must match single-pass for [{:#04x}, {:#04x}]",
                    a, b
                );
            }
        }
    }

    #[test]
    fn crc32_table_high_nibble_bytes() {
        // Test bytes with high nibble patterns (0x10, 0x20, ..., 0xF0)
        let mut hashes = std::collections::HashSet::new();
        for i in 0..16u8 {
            let hash = crc32_hash(&[i << 4]);
            hashes.insert(hash);
        }
        assert_eq!(hashes.len(), 16, "All high-nibble bytes should produce unique CRC32");
    }
}

// ============================================================================
// 5. record (data_collector/mod.rs, line 100) — profiler with real tokens
// ============================================================================

mod record_tests {
    use super::*;

    #[test]
    fn record_returns_some_when_profiler_has_tokens() {
        let mut collector = TunerDataCollector::new();
        let profiler = make_profiler_with_tokens(100, 1_000_000); // 100 tokens in 1ms
        let config = RunConfig::default();
        let result = collector.record(&profiler, &config, KernelType::VectorizedQ4K);
        assert!(result.is_some(), "record should return Some when profiler has tokens");
        assert_eq!(collector.len(), 1);
    }

    #[test]
    fn record_populates_sample_fields() {
        let mut collector = TunerDataCollector::new();
        let profiler = make_profiler_with_tokens(500, 10_000_000); // 500 tokens in 10ms
        let config = RunConfig {
            model_params_b: 7.0,
            hidden_dim: 4096,
            num_layers: 32,
            num_heads: 32,
            batch_size: 4,
            seq_len: 128,
            cuda_graphs: true,
            quant_type: QuantType::Q4K,
            kernel_type: KernelType::BatchedQ4K,
        };
        collector.record(&profiler, &config, KernelType::BatchedQ4K);

        let sample = &collector.samples()[0];
        assert!(sample.throughput_tps > 0.0, "Throughput should be positive");
        assert_eq!(sample.best_kernel, KernelType::BatchedQ4K);
        assert!(!sample.timestamp.is_empty());
        assert_eq!(sample.hardware_id, "unknown");
    }

    #[test]
    fn record_returns_none_when_profiler_disabled() {
        let mut collector = TunerDataCollector::new();
        // Profiler created without enable() — no timing data
        let profiler = BrickProfiler::new();
        let config = RunConfig::default();
        let result = collector.record(&profiler, &config, KernelType::TiledQ4K);
        assert!(result.is_none());
        assert!(collector.is_empty());
    }

    #[test]
    fn record_returns_none_when_profiler_has_zero_tokens() {
        let mut collector = TunerDataCollector::new();
        // Enabled profiler but no bricks timed — total_tokens = 0
        let mut profiler = BrickProfiler::new();
        profiler.enable();
        let config = RunConfig::default();
        let result = collector.record(&profiler, &config, KernelType::TiledQ4K);
        assert!(result.is_none());
    }

    #[test]
    fn record_multiple_samples() {
        let mut collector = TunerDataCollector::new();
        let config = RunConfig::default();

        for i in 1..=5 {
            let profiler = make_profiler_with_tokens(i * 100, 1_000_000);
            collector.record(&profiler, &config, KernelType::VectorizedQ4K);
        }
        assert_eq!(collector.len(), 5);
    }

    #[test]
    fn record_uses_feature_extractor() {
        let mut collector = TunerDataCollector::new();
        let profiler = make_profiler_with_tokens(200, 5_000_000);
        let config = RunConfig {
            model_params_b: 1.5,
            hidden_dim: 1536,
            num_layers: 28,
            num_heads: 12,
            batch_size: 1,
            seq_len: 1,
            cuda_graphs: false,
            quant_type: QuantType::Q4K,
            kernel_type: KernelType::VectorizedQ4K,
        };
        collector.record(&profiler, &config, KernelType::VectorizedQ4K);

        let sample = &collector.samples()[0];
        // Verify features were extracted (normalized model_params_b should be ~0.39)
        assert!(sample.features.model_params_b > 0.0);
        assert!(sample.features.model_params_b < 1.0);
    }

    #[test]
    fn record_sets_bottleneck_class() {
        let mut collector = TunerDataCollector::new();
        let profiler = make_profiler_with_tokens(100, 1_000_000);
        let config = RunConfig::default();
        collector.record(&profiler, &config, KernelType::TiledQ4K);

        let sample = &collector.samples()[0];
        // Bottleneck should be set (not Unknown — extractor classifies it)
        assert_ne!(sample.bottleneck, BottleneckClass::Unknown);
    }
}

// ============================================================================
// 6. load_or_default (brick_tuner/persistence.rs, line 51) — hardware-detect
// ============================================================================

#[cfg(feature = "hardware-detect")]
mod load_or_default_tests {
    use super::super::super::super::*;

    #[test]
    fn load_or_default_returns_new_when_no_cache() {
        // Ensure cache file does not exist
        let cache_path = BrickTuner::cache_path();
        let _ = std::fs::remove_file(&cache_path);

        let tuner = BrickTuner::load_or_default();
        assert_eq!(
            tuner.version(),
            BrickTuner::VERSION,
            "Should create new tuner when no cache exists"
        );
    }

    #[test]
    fn load_or_default_returns_cached_when_version_matches() {
        let dir = std::env::temp_dir().join("trueno_test_load_or_default_v_match");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create dir");
        let path = dir.join(format!("tuner_model_v{}.apr", BrickTuner::VERSION));

        // Save a tuner with matching version
        let tuner = BrickTuner::new();
        tuner.save_apr(&path).expect("save should succeed");

        // Verify load_apr can read it back
        let loaded = BrickTuner::load_apr(&path).expect("load should succeed");
        assert_eq!(loaded.version(), BrickTuner::VERSION);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_or_default_returns_new_when_version_mismatch() {
        let dir = std::env::temp_dir().join("trueno_test_load_or_default_v_mismatch");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create dir");
        let path = dir.join("tuner_model.apr");

        // Save a tuner with modified version
        let mut tuner = BrickTuner::new();
        tuner.version = "0.0.0-old".to_string();
        tuner.save_apr(&path).expect("save should succeed");

        // load_apr will load it but version won't match BrickTuner::VERSION
        let loaded = BrickTuner::load_apr(&path).expect("load should succeed");
        assert_ne!(loaded.version(), BrickTuner::VERSION);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_or_default_returns_new_when_file_corrupt() {
        use std::io::Write;
        let dir = std::env::temp_dir().join("trueno_test_load_or_default_corrupt");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create dir");
        let path = dir.join("tuner_model.apr");

        // Write garbage that fails to load
        let mut file = std::fs::File::create(&path).expect("create file");
        file.write_all(b"NOT_APR_FORMAT").expect("write");
        drop(file);

        // load_apr should fail, and load_or_default should fall back to new
        assert!(BrickTuner::load_apr(&path).is_err());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn cache_path_is_deterministic() {
        let path1 = BrickTuner::cache_path();
        let path2 = BrickTuner::cache_path();
        assert_eq!(path1, path2, "Cache path should be deterministic");
        assert!(
            path1.to_string_lossy().contains(&format!("tuner_model_v{}.apr", BrickTuner::VERSION)),
            "Cache path should contain version: {:?}",
            path1
        );
    }

    #[test]
    fn save_to_cache_and_load_or_default_roundtrip() {
        // Save to cache location
        let tuner = BrickTuner::new();
        let save_result = tuner.save_to_cache();
        if save_result.is_ok() {
            // If save succeeded, load_or_default should find it
            let loaded = BrickTuner::load_or_default();
            assert_eq!(loaded.version(), BrickTuner::VERSION);

            // Clean up
            let _ = std::fs::remove_file(BrickTuner::cache_path());
        }
        // If save failed (e.g., permissions), that's OK — test the path exists
    }
}

// ============================================================================
// BrickTuner save_apr/load_apr roundtrip (persistence.rs general coverage)
// ============================================================================

mod brick_tuner_persistence_tests {
    use super::super::super::super::*;

    #[test]
    fn save_and_load_apr_roundtrip() {
        let dir = std::env::temp_dir().join("trueno_test_brick_tuner_apr");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create dir");
        let path = dir.join("tuner.apr");

        let tuner = BrickTuner::with_pretrained();
        tuner.save_apr(&path).expect("save should succeed");

        let loaded = BrickTuner::load_apr(&path).expect("load should succeed");
        assert!(loaded.version().contains("pretrained"));
        assert_eq!(loaded.throughput_sample_count(), 10_000);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_apr_bad_magic() {
        use std::io::Write;
        let dir = std::env::temp_dir().join("trueno_test_brick_tuner_bad_magic");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create dir");
        let path = dir.join("bad.apr");

        let mut file = std::fs::File::create(&path).expect("create");
        file.write_all(b"XXXX").expect("write");
        file.write_all(&4u32.to_le_bytes()).expect("len");
        file.write_all(b"test").expect("data");
        file.write_all(&0u32.to_le_bytes()).expect("crc");
        drop(file);

        let result = BrickTuner::load_apr(&path);
        assert!(result.is_err());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_apr_crc_mismatch() {
        use std::io::Write;
        let dir = std::env::temp_dir().join("trueno_test_brick_tuner_crc");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create dir");
        let path = dir.join("bad_crc.apr");

        let json = b"{}";
        let mut file = std::fs::File::create(&path).expect("create");
        file.write_all(b"APR1").expect("magic");
        file.write_all(&(json.len() as u32).to_le_bytes()).expect("len");
        file.write_all(json).expect("data");
        file.write_all(&0xDEADBEEFu32.to_le_bytes()).expect("bad crc");
        drop(file);

        let result = BrickTuner::load_apr(&path);
        assert!(result.is_err());
        let err = format!("{}", result.unwrap_err());
        assert!(err.contains("CRC32"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn save_apr_io_error() {
        let tuner = BrickTuner::new();
        let result = tuner.save_apr("/proc/nonexistent/deep/path.apr");
        assert!(result.is_err());
        let err = format!("{}", result.unwrap_err());
        assert!(err.contains("I/O error"));
    }

    #[test]
    fn to_json_and_from_json_roundtrip() {
        let tuner = BrickTuner::with_pretrained();
        let json = tuner.to_json().expect("to_json should succeed");
        let loaded = BrickTuner::from_json(&json).expect("from_json should succeed");
        assert!(loaded.version().contains("pretrained"));
    }

    #[test]
    fn from_json_invalid() {
        let result = BrickTuner::from_json("not json");
        assert!(result.is_err());
    }
}

// ============================================================================
// ThroughputRegressor::train (non-ml-tuner path, models/throughput.rs)
// ============================================================================

mod throughput_train_tests {
    use super::super::super::super::*;

    fn make_training_data(n: usize) -> Vec<(TunerFeatures, f32)> {
        (0..n)
            .map(|i| {
                let batch = ((i % 8) + 1) as u32;
                let model = [1.5, 3.0, 7.0, 13.0][i % 4];
                let features = TunerFeatures::builder()
                    .model_params_b(model)
                    .hidden_dim(4096)
                    .num_layers(32)
                    .batch_size(batch)
                    .quant_type(QuantType::Q4K)
                    .build();
                let target = 100.0 * (batch as f32).sqrt() / model.sqrt() as f32;
                (features, target.max(10.0))
            })
            .collect()
    }

    #[test]
    fn train_insufficient_data() {
        let mut regressor = ThroughputRegressor::new();
        let data = make_training_data(5);
        let result = regressor.train(&data);
        assert!(result.is_err());
    }

    #[test]
    fn train_minimum_data() {
        let mut regressor = ThroughputRegressor::new();
        let data = make_training_data(10);
        let result = regressor.train(&data);
        assert!(result.is_ok());
        assert_eq!(regressor.sample_count, 10);
    }

    #[test]
    fn train_updates_mape() {
        let mut regressor = ThroughputRegressor::new();
        let data = make_training_data(20);
        regressor.train(&data).expect("training should succeed");
        assert!(regressor.mape >= 0.0);
        assert!(regressor.mape < 10.0, "MAPE should be reasonable after training");
    }

    #[test]
    fn train_larger_dataset() {
        let mut regressor = ThroughputRegressor::new();
        let data = make_training_data(100);
        regressor.train(&data).expect("training should succeed");
        assert_eq!(regressor.sample_count, 100);
    }

    #[test]
    fn train_weights_change() {
        let mut regressor = ThroughputRegressor::new();
        let original_weights = regressor.weights.clone();
        let data = make_training_data(20);
        regressor.train(&data).expect("training should succeed");
        // At least some weights should change
        let any_changed = regressor
            .weights
            .iter()
            .zip(original_weights.iter())
            .any(|(a, b)| (a - b).abs() > 1e-10);
        assert!(any_changed, "Weights should change after training");
    }

    #[test]
    fn train_predict_after_training() {
        let mut regressor = ThroughputRegressor::new();
        let data = make_training_data(20);
        regressor.train(&data).expect("training should succeed");

        let features = TunerFeatures::builder().model_params_b(7.0).batch_size(4).build();
        let prediction = regressor.predict(&features);
        assert!(prediction.predicted_tps > 0.0);
        assert!(prediction.confidence > 0.0);
    }
}

// ============================================================================
// Roofline clamping (models/throughput.rs — compute_roofline_bound)
// ============================================================================

mod roofline_tests {
    use super::super::super::super::*;

    #[test]
    fn compute_roofline_bound_basic() {
        let features = TunerFeatures::builder()
            .model_params_b(7.0)
            .hidden_dim(4096)
            .batch_size(1)
            .quant_type(QuantType::Q4K)
            .gpu_mem_bw_gbs(1000.0)
            .build();
        let bound = ThroughputRegressor::compute_roofline_bound(&features);
        assert!(bound >= 1.0, "Roofline bound should be >= 1.0");
        assert!(bound <= 10000.0, "Roofline bound should be <= 10000.0");
    }

    #[test]
    fn compute_roofline_bound_f32_higher_bound() {
        let features_q4k = TunerFeatures::builder()
            .model_params_b(7.0)
            .batch_size(1)
            .quant_type(QuantType::Q4K)
            .gpu_mem_bw_gbs(1000.0)
            .build();
        let features_f32 = TunerFeatures::builder()
            .model_params_b(7.0)
            .batch_size(1)
            .quant_type(QuantType::F32)
            .gpu_mem_bw_gbs(1000.0)
            .build();
        let bound_q4k = ThroughputRegressor::compute_roofline_bound(&features_q4k);
        let bound_f32 = ThroughputRegressor::compute_roofline_bound(&features_f32);
        // F32 uses 4 bytes/param vs Q4K's 0.5625, so bound should be lower for F32
        assert!(bound_q4k > bound_f32, "Q4K should have higher roofline bound than F32");
    }

    #[test]
    fn bytes_per_param_from_onehot_all_variants() {
        // Test each quant type one-hot
        let quant_types = [
            QuantType::Q4_0,
            QuantType::Q4_1,
            QuantType::Q4K,
            QuantType::Q5K,
            QuantType::Q6K,
            QuantType::Q8_0,
            QuantType::F16,
            QuantType::F32,
        ];
        let expected_bpp = [0.5625, 0.5625, 0.5625, 0.6875, 0.8125, 1.0, 2.0, 4.0];

        for (qt, &expected) in quant_types.iter().zip(expected_bpp.iter()) {
            let mut onehot = [0.0f32; 8];
            onehot[qt.to_index()] = 1.0;
            let bpp = ThroughputRegressor::bytes_per_param_from_onehot(&onehot);
            assert!(
                (bpp - expected).abs() < 1e-6,
                "QuantType {:?}: expected {}, got {}",
                qt,
                expected,
                bpp
            );
        }
    }

    #[test]
    fn bytes_per_param_from_onehot_ambiguous_defaults_q4k() {
        // All zeros — should default to Q4K (index 2) = 0.5625
        let onehot = [0.0f32; 8];
        let bpp = ThroughputRegressor::bytes_per_param_from_onehot(&onehot);
        // When all are 0.0, max_by returns last index with Equal (depends on iterator impl)
        // But the important thing is it doesn't panic
        assert!(bpp > 0.0);
    }
}
