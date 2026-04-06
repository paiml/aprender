//! Classifier and kernel type tests.

use super::super::*;

// Bottleneck classification tests
#[test]
fn test_bottleneck_recommended_action() {
    assert!(BottleneckClass::MemoryBound.recommended_action().contains("batch size"));
    assert!(BottleneckClass::LaunchBound.recommended_action().contains("CUDA graphs"));
    assert!(BottleneckClass::AttentionBound.recommended_action().contains("Flash Decoding"));
}

// Kernel classifier tests
#[test]
fn test_kernel_classifier_batched_for_high_m() {
    let classifier = KernelClassifier::new();
    let features = TunerFeatures::builder().batch_size(8).build();

    let rec = classifier.predict(&features);
    assert_eq!(rec.top_kernel, KernelType::BatchedQ4K);
}

// Feature builder tests
#[test]
fn test_feature_builder_normalization() {
    let features = TunerFeatures::builder()
        .model_params_b(1.0) // log10(1.0) = 0, normalized = (0+1)/3 = 0.33
        .hidden_dim(1536) // 1536/16384 approx 0.094
        .batch_size(4) // 4/64 = 0.0625
        .build();

    assert!(features.model_params_b > 0.0 && features.model_params_b < 1.0);
    assert!(features.hidden_dim_norm > 0.0 && features.hidden_dim_norm < 1.0);
    assert!(features.batch_size_norm > 0.0 && features.batch_size_norm < 1.0);
}

#[test]
fn test_all_builder_methods() {
    let features = TunerFeatures::builder()
        .model_params_b(1.5)
        .hidden_dim(2048)
        .num_layers(32)
        .num_heads(16)
        .head_dim(128)
        .vocab_size(32000)
        .batch_size(4)
        .seq_len(512)
        .cuda_graphs(true)
        .kv_caches(4)
        .is_prefill(false)
        .quant_type(QuantType::Q4K)
        .kernel_type(KernelType::VectorizedQ4K)
        .gpu_mem_bw_gbs(1000.0)
        .gpu_compute_tflops(150.0)
        .gpu_sm_count(128)
        .measured_tps(100.0)
        .build();

    assert!(features.model_params_b > 0.0);
    assert!(features.cuda_graphs == 1.0);
    assert!(features.is_prefill == 0.0);
}

#[test]
fn test_quant_type_bytes_per_param() {
    assert_eq!(QuantType::Q4_0.bytes_per_param(), 0.5625);
    assert_eq!(QuantType::Q4_1.bytes_per_param(), 0.5625);
    assert_eq!(QuantType::Q5K.bytes_per_param(), 0.6875);
    assert_eq!(QuantType::Q6K.bytes_per_param(), 0.8125);
    assert_eq!(QuantType::Q8_0.bytes_per_param(), 1.0);
    assert_eq!(QuantType::F16.bytes_per_param(), 2.0);
    assert_eq!(QuantType::F32.bytes_per_param(), 4.0);
}

#[test]
fn test_kernel_type_to_index() {
    assert_eq!(KernelType::TiledQ4K.to_index(), 0);
    assert_eq!(KernelType::CoalescedQ4K.to_index(), 1);
    assert_eq!(KernelType::VectorizedQ4K.to_index(), 2);
    assert_eq!(KernelType::BatchedQ4K.to_index(), 3);
    assert_eq!(KernelType::Dp4aQ4K.to_index(), 4);
    assert_eq!(KernelType::FusedRmsNormQ4K.to_index(), 5);
    assert_eq!(KernelType::CoalescedQ6K.to_index(), 6);
    assert_eq!(KernelType::IncrementalAttention.to_index(), 7);
    assert_eq!(KernelType::MultiWarpAttention.to_index(), 8);
    assert_eq!(KernelType::BatchedAttention.to_index(), 9);
    assert_eq!(KernelType::RmsNorm.to_index(), 10);
    assert_eq!(KernelType::VectorizedRmsNorm.to_index(), 11);
    assert_eq!(KernelType::BatchedRmsNorm.to_index(), 12);
    assert_eq!(KernelType::FusedQKVHwDp4aQ4KGemv.to_index(), 13);
    assert_eq!(KernelType::Generic.to_index(), 14);
    assert_eq!(KernelType::Unknown.to_index(), 15);
}

#[test]
fn test_bottleneck_class_to_index() {
    assert_eq!(BottleneckClass::Unknown.to_index(), 0);
    assert_eq!(BottleneckClass::MemoryBound.to_index(), 1);
    assert_eq!(BottleneckClass::ComputeBound.to_index(), 2);
    assert_eq!(BottleneckClass::LaunchBound.to_index(), 3);
    assert_eq!(BottleneckClass::AttentionBound.to_index(), 4);
}

#[test]
fn test_bottleneck_display() {
    assert_eq!(format!("{}", BottleneckClass::Unknown), "Unknown");
    assert_eq!(format!("{}", BottleneckClass::MemoryBound), "MemoryBound");
    assert_eq!(format!("{}", BottleneckClass::ComputeBound), "ComputeBound");
    assert_eq!(format!("{}", BottleneckClass::LaunchBound), "LaunchBound");
    assert_eq!(format!("{}", BottleneckClass::AttentionBound), "AttentionBound");
}

#[test]
fn test_from_brick_bottleneck() {
    use crate::brick::BrickBottleneck;
    assert_eq!(
        BottleneckClass::from_brick_bottleneck(BrickBottleneck::Memory),
        BottleneckClass::MemoryBound
    );
    assert_eq!(
        BottleneckClass::from_brick_bottleneck(BrickBottleneck::Compute),
        BottleneckClass::ComputeBound
    );
    assert_eq!(
        BottleneckClass::from_brick_bottleneck(BrickBottleneck::Unknown),
        BottleneckClass::Unknown
    );
}

#[test]
fn test_run_config_default() {
    let config = RunConfig::default();
    assert_eq!(config.model_params_b, 1.5);
    assert_eq!(config.batch_size, 1);
    assert_eq!(config.quant_type, QuantType::Q4K);
}

#[test]
fn test_bottleneck_classifier() {
    let classifier = BottleneckClassifier::new();
    let features = TunerFeatures::builder().batch_size(4).build();
    let pred = classifier.predict(&features);
    // Default prediction should be MemoryBound for inference
    assert!(matches!(pred.class, BottleneckClass::MemoryBound | BottleneckClass::Unknown));
}

#[test]
fn test_quant_type_to_index_all_variants() {
    // Cover all QuantType::to_index branches
    assert_eq!(QuantType::Q4_0.to_index(), 0);
    assert_eq!(QuantType::Q4_1.to_index(), 1);
    assert_eq!(QuantType::Q4K.to_index(), 2);
    assert_eq!(QuantType::Q5K.to_index(), 3);
    assert_eq!(QuantType::Q6K.to_index(), 4);
    assert_eq!(QuantType::Q8_0.to_index(), 5);
    assert_eq!(QuantType::F16.to_index(), 6);
    assert_eq!(QuantType::F32.to_index(), 7);
}
