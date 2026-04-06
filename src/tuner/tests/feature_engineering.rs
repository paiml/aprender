//! Feature engineering tests (F021-F040).

use super::super::*;

#[test]
fn f021_no_nan_features() {
    let features = TunerFeatures::builder()
        .model_params_b(1.5)
        .hidden_dim(1536)
        .batch_size(4)
        .quant_type(QuantType::Q4K)
        .build();

    let v = features.to_vector();
    assert!(!v.iter().any(|x| x.is_nan()), "Features contain NaN");
}

#[test]
fn f022_no_infinite_features() {
    let features =
        TunerFeatures::builder().model_params_b(1.5).hidden_dim(1536).batch_size(4).build();

    let v = features.to_vector();
    assert!(!v.iter().any(|x| x.is_infinite()), "Features contain infinity");
}

#[test]
fn f023_features_in_0_1_range() {
    let features = TunerFeatures::builder()
        .model_params_b(100.0) // Very large
        .hidden_dim(16384) // Max
        .batch_size(64) // Max
        .seq_len(32768) // Max
        .build();

    let v = features.to_vector();
    for (i, x) in v.iter().enumerate() {
        assert!(*x >= -0.001 && *x <= 1.001, "Feature {} = {} is outside [0, 1]", i, x);
    }
}

/// f026: Roofline bound - predicted TPS must never exceed theoretical maximum
/// This is the crucible that ensures ML predictions do not violate hardware limits.
/// Roofline model: max_tps = memory_bw_bytes_per_sec / bytes_per_token
/// For decode: bytes_per_token = model_params * bytes_per_param
#[test]
fn f026_roofline_bound() {
    // Test configuration: 7B Q4_K model on RTX 4090 (1008 GB/s)
    let model_params_b: f32 = 7.0;
    let bytes_per_param: f32 = QuantType::Q4K.bytes_per_param();
    let gpu_mem_bw_gbs: f32 = 1008.0; // RTX 4090

    // Theoretical maximum tokens/sec for decode phase (batch=1)
    // bytes_per_token = model_params * bytes_per_param * 1e9
    // max_tps = mem_bw_bytes_per_sec / bytes_per_token
    let model_bytes: f32 = model_params_b * bytes_per_param * 1e9;
    let theoretical_max_tps: f32 = (gpu_mem_bw_gbs * 1e9) / model_bytes;

    // Build features for this configuration
    let features = TunerFeatures::builder()
        .model_params_b(model_params_b)
        .batch_size(1)
        .quant_type(QuantType::Q4K)
        .gpu_mem_bw_gbs(gpu_mem_bw_gbs)
        .gpu_compute_tflops(82.6) // RTX 4090 FP32
        .is_prefill(false) // Decode phase
        .build();

    let tuner = BrickTuner::new();
    let rec = tuner.recommend(&features);

    // CRITICAL ASSERTION: predicted_tps <= theoretical_max_tps
    // Allowing 10% margin for numerical precision
    let margin = 1.10;
    assert!(
        rec.throughput.predicted_tps <= theoretical_max_tps * margin,
        "Roofline violation: predicted {} tok/s exceeds theoretical max {} tok/s \
         (model: {}B, quant: Q4K, mem_bw: {} GB/s)",
        rec.throughput.predicted_tps,
        theoretical_max_tps,
        model_params_b,
        gpu_mem_bw_gbs
    );

    // Also verify theoretical max is reasonable (sanity check)
    // 7B Q4_K on 1008 GB/s should yield ~200-300 tok/s theoretical max
    assert!(
        theoretical_max_tps > 100.0 && theoretical_max_tps < 500.0,
        "Theoretical max {} tok/s is outside expected range for 7B Q4_K on RTX 4090",
        theoretical_max_tps
    );
}

#[test]
fn f029_onehot_sums_to_one() {
    let features = TunerFeatures::builder()
        .quant_type(QuantType::Q4K)
        .kernel_type(KernelType::VectorizedQ4K)
        .build();

    let quant_sum: f32 = features.quant_type_onehot.iter().sum();
    let kernel_sum: f32 = features.kernel_type_onehot.iter().sum();

    assert!((quant_sum - 1.0).abs() < 0.001, "Quant one-hot sum = {}", quant_sum);
    assert!((kernel_sum - 1.0).abs() < 0.001, "Kernel one-hot sum = {}", kernel_sum);
}

/// f040: Feature dimension must be 42 per spec v1.1.0
/// 11 static + 8 quant + 16 kernel + 5 hardware + 2 derived = 42
#[test]
fn f040_feature_dimension_is_42() {
    assert_eq!(TunerFeatures::DIM, 42, "DIM must be 42 per spec v1.1.0");

    let features = TunerFeatures::builder().build();
    assert_eq!(features.to_vector().len(), TunerFeatures::DIM);
}
