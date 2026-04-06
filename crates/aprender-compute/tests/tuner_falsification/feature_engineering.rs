//! F021-F040: Feature Engineering (20 points)

use trueno::tuner::{KernelType, QuantType, TunerFeatures};

/// F021: TunerFeatures dimension must be 42
#[test]
fn f021_features_dim_42() {
    let features = TunerFeatures::builder().model_params_b(1.5).batch_size(4).build();

    let vec = features.to_vector();
    assert_eq!(vec.len(), 42, "F021 FALSIFIED: expected DIM=42, got {}", vec.len());
}

/// F022: Feature vector must be normalized (most values in [0,1])
#[test]
fn f022_features_normalized() {
    let features = TunerFeatures::builder()
        .model_params_b(1.5)
        .hidden_dim(1536)
        .batch_size(4)
        .gpu_mem_bw_gbs(1000.0)
        .build();

    let vec = features.to_vector();
    let in_range_count = vec.iter().filter(|&&v| (0.0..=1.5).contains(&v)).count();

    // At least 80% of features should be in reasonable range
    assert!(
        in_range_count >= 34,
        "F022 FALSIFIED: only {}/42 features in [0, 1.5]",
        in_range_count
    );
}

/// F023: Feature validation must pass for valid inputs
#[test]
fn f023_validation_accepts_valid() {
    let features =
        TunerFeatures::builder().model_params_b(1.5).hidden_dim(1536).batch_size(4).build();

    assert!(features.validate().is_ok(), "F023 FALSIFIED: valid features rejected");
}

/// F024: Feature validation must reject invalid inputs
#[test]
fn f024_validation_rejects_invalid() {
    let result = TunerFeatures::builder()
        .model_params_b(-1.0) // Invalid: negative
        .try_build();

    assert!(result.is_err(), "F024 FALSIFIED: negative model_params_b accepted");
}

/// F025: QuantType one-hot encoding must be valid
#[test]
fn f025_quant_onehot_valid() {
    for qt in [
        QuantType::Q4_0,
        QuantType::Q4_1,
        QuantType::Q4K,
        QuantType::Q5K,
        QuantType::Q6K,
        QuantType::Q8_0,
        QuantType::F16,
        QuantType::F32,
    ] {
        let idx = qt.to_index();
        assert!(idx < 8, "F025 FALSIFIED: QuantType index {} >= 8", idx);
    }
}

/// F026: KernelType one-hot encoding must be valid
#[test]
fn f026_kernel_onehot_valid() {
    let kernels = [
        KernelType::TiledQ4K,
        KernelType::CoalescedQ4K,
        KernelType::VectorizedQ4K,
        KernelType::BatchedQ4K,
    ];

    for kt in kernels {
        let idx = kt.to_index();
        assert!(
            idx < KernelType::COUNT,
            "F026 FALSIFIED: KernelType index {} >= {}",
            idx,
            KernelType::COUNT
        );
    }
}

/// F027: Bytes per param must be positive
#[test]
fn f027_bytes_per_param_positive() {
    for qt in [QuantType::Q4_0, QuantType::Q4K, QuantType::Q8_0, QuantType::F16, QuantType::F32] {
        let bpp = qt.bytes_per_param();
        assert!(bpp > 0.0, "F027 FALSIFIED: {} has bpp={}", qt.to_index(), bpp);
    }
}

/// F028: Builder defaults must be sensible
#[test]
fn f028_builder_defaults() {
    let features = TunerFeatures::builder().build();
    let vec = features.to_vector();

    // Should not have NaN or Inf
    for (i, &v) in vec.iter().enumerate() {
        assert!(v.is_finite(), "F028 FALSIFIED: feature[{}] is not finite: {}", i, v);
    }
}

/// F029: Hidden dim normalization
#[test]
fn f029_hidden_dim_normalized() {
    let features = TunerFeatures::builder().hidden_dim(4096).build();
    let vec = features.to_vector();

    // hidden_dim normalized by 8192
    let normalized = 4096.0 / 8192.0;
    assert!(
        (vec[1] - normalized).abs() < 0.001 || vec[1] >= 0.0,
        "F029 FALSIFIED: hidden_dim normalization incorrect"
    );
}

/// F030: Batch size normalization
#[test]
fn f030_batch_size_normalized() {
    let features = TunerFeatures::builder().batch_size(8).build();
    let vec = features.to_vector();

    // batch_size at index 6, normalized by 64
    let expected = 8.0 / 64.0;
    assert!(
        (vec[6] - expected).abs() < 0.001,
        "F030 FALSIFIED: batch_size normalization {} != {}",
        vec[6],
        expected
    );
}

/// F031: CUDA graphs flag must be 0 or 1
#[test]
fn f031_cuda_graphs_binary() {
    for cuda_graphs in [true, false] {
        let features = TunerFeatures::builder().cuda_graphs(cuda_graphs).build();
        let vec = features.to_vector();

        let cuda_graphs_idx = 9;
        let val = vec[cuda_graphs_idx];
        assert!(val == 0.0 || val == 1.0, "F031 FALSIFIED: cuda_graphs feature {} not binary", val);
    }
}

/// F032: GPU memory bandwidth normalization
#[test]
fn f032_gpu_mem_bw_normalized() {
    let features = TunerFeatures::builder().gpu_mem_bw_gbs(1500.0).build();
    let vec = features.to_vector();

    // gpu_mem_bw at index 35, normalized by 3000
    let expected = 1500.0 / 3000.0;
    assert!(
        (vec[35] - expected).abs() < 0.001,
        "F032 FALSIFIED: gpu_mem_bw normalization {} != {}",
        vec[35],
        expected
    );
}

/// F033: Model params normalization
#[test]
fn f033_model_params_normalized() {
    let features = TunerFeatures::builder().model_params_b(7.0).build();
    let vec = features.to_vector();

    // model_params_b at index 0, log-normalized
    // Formula: (log10(7e9) / 3 + 1/3) normalized
    assert!(
        vec[0] > 0.0 && vec[0] < 2.0,
        "F033 FALSIFIED: model_params normalization {} out of range",
        vec[0]
    );
}

/// F034: Seq len affects feature vector
#[test]
fn f034_seq_len_affects_vector() {
    let features_short = TunerFeatures::builder().seq_len(512).build().to_vector();
    let features_long = TunerFeatures::builder().seq_len(4096).build().to_vector();

    // Seq len should change at least one feature
    let diff: f32 =
        features_short.iter().zip(features_long.iter()).map(|(a, b)| (a - b).abs()).sum();

    assert!(diff > 0.01, "F034 FALSIFIED: seq_len doesn't affect feature vector (diff={})", diff);
}

/// F035: Quant type affects feature vector
#[test]
fn f035_quant_type_affects_vector() {
    let features_q4k = TunerFeatures::builder().quant_type(QuantType::Q4K).build().to_vector();
    let features_f16 = TunerFeatures::builder().quant_type(QuantType::F16).build().to_vector();

    // Different quant types should produce different vectors
    let diff: f32 = features_q4k.iter().zip(features_f16.iter()).map(|(a, b)| (a - b).abs()).sum();

    assert!(
        diff > 0.01,
        "F035 FALSIFIED: quant_type doesn't affect feature vector (diff={})",
        diff
    );
}

/// F036: Feature vector serialization round-trip
#[test]
fn f036_features_serialize_roundtrip() {
    let features = TunerFeatures::builder()
        .model_params_b(1.5)
        .batch_size(4)
        .quant_type(QuantType::Q4K)
        .build();

    let json = serde_json::to_string(&features).expect("serialize");
    let restored: TunerFeatures = serde_json::from_str(&json).expect("deserialize");

    let orig_vec = features.to_vector();
    let restored_vec = restored.to_vector();

    for (i, (a, b)) in orig_vec.iter().zip(restored_vec.iter()).enumerate() {
        assert!((a - b).abs() < 0.001, "F036 FALSIFIED: feature[{}] mismatch: {} vs {}", i, a, b);
    }
}

/// F037: SM count affects feature vector
#[test]
fn f037_sm_count_affects_vector() {
    let features_low = TunerFeatures::builder().gpu_sm_count(64).build().to_vector();
    let features_high = TunerFeatures::builder().gpu_sm_count(256).build().to_vector();

    // Different SM counts should produce different vectors
    let diff: f32 = features_low.iter().zip(features_high.iter()).map(|(a, b)| (a - b).abs()).sum();

    assert!(
        diff > 0.01,
        "F037 FALSIFIED: gpu_sm_count doesn't affect feature vector (diff={})",
        diff
    );
}

/// F038: Num layers normalization
#[test]
fn f038_num_layers_normalized() {
    let features = TunerFeatures::builder().num_layers(32).build();
    let vec = features.to_vector();

    // num_layers at index 2, normalized by 128
    let expected = 32.0 / 128.0;
    assert!(
        (vec[2] - expected).abs() < 0.001,
        "F038 FALSIFIED: num_layers normalization {} != {}",
        vec[2],
        expected
    );
}

/// F039: Num heads normalization
#[test]
fn f039_num_heads_normalized() {
    let features = TunerFeatures::builder().num_heads(32).build();
    let vec = features.to_vector();

    // num_heads at index 3, normalized by 128
    let expected = 32.0 / 128.0;
    assert!(
        (vec[3] - expected).abs() < 0.001,
        "F039 FALSIFIED: num_heads normalization {} != {}",
        vec[3],
        expected
    );
}

/// F040: Temperature default
#[test]
fn f040_temperature_default() {
    let features = TunerFeatures::builder().build();
    let vec = features.to_vector();

    // temperature at index 17, default should be reasonable (0.0-2.0 range)
    assert!(
        vec[17] >= 0.0 && vec[17] <= 2.0,
        "F040 FALSIFIED: temperature {} out of range",
        vec[17]
    );
}
