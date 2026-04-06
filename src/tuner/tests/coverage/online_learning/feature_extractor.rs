//! Coverage: BrickTuner, TunerFeaturesBuilder::hardware(), FeatureExtractor, classify_bottleneck tests.

use crate::tuner::*;

// =========================================================================
// Coverage: BrickTuner::default() (lines 109-111)
// =========================================================================

#[test]
fn test_brick_tuner_default_trait() {
    let tuner: BrickTuner = Default::default();
    assert_eq!(tuner.version(), BrickTuner::VERSION);
    assert_eq!(tuner.sample_count, 0);
}

// =========================================================================
// Coverage: render_comparison "Good" and "Fair" accuracy indicators
// =========================================================================

#[test]
fn test_brick_tuner_render_comparison_good() {
    let tuner = BrickTuner::new();
    let features = TunerFeatures::builder().batch_size(2).model_params_b(1.5).build();
    let rec = tuner.recommend(&features);
    // ~7% error -> "Good" branch (5% <= error < 10%)
    let actual_tps = rec.throughput.predicted_tps * 0.93;
    let comparison = tuner.render_comparison(&rec, actual_tps);
    assert_eq!(comparison.len(), 2);
    assert!(comparison[1].contains("Good"), "Expected 'Good' indicator, got: {}", comparison[1]);
}

#[test]
fn test_brick_tuner_render_comparison_fair() {
    let tuner = BrickTuner::new();
    let features = TunerFeatures::builder().batch_size(2).model_params_b(1.5).build();
    let rec = tuner.recommend(&features);
    // ~15% error -> "Fair" branch (10% <= error < 20%)
    let actual_tps = rec.throughput.predicted_tps * 0.85;
    let comparison = tuner.render_comparison(&rec, actual_tps);
    assert_eq!(comparison.len(), 2);
    assert!(comparison[1].contains("Fair"), "Expected 'Fair' indicator, got: {}", comparison[1]);
}

// =========================================================================
// Coverage: TunerFeaturesBuilder::hardware() with GPU (lines 351-359)
// =========================================================================

#[test]
fn test_builder_hardware_with_gpu() {
    use crate::hardware::{
        CpuCapability, GpuBackend, GpuCapability, HardwareCapability, RooflineParams, SimdWidth,
    };

    let hw = HardwareCapability {
        timestamp: "test".to_string(),
        hostname: "test-host".to_string(),
        cpu: CpuCapability {
            vendor: "Intel".to_string(),
            model: "Test CPU".to_string(),
            cores: 8,
            threads: 16,
            simd: SimdWidth::Avx2,
            base_freq_ghz: 3.5,
            peak_gflops: 100.0,
            memory_bw_gbps: 50.0,
        },
        gpu: Some(GpuCapability {
            vendor: "NVIDIA".to_string(),
            model: "RTX 4090".to_string(),
            backend: GpuBackend::Cuda,
            compute_capability: Some("8.9".to_string()),
            peak_tflops_fp32: 82.6,
            peak_tflops_tensor: Some(330.0),
            memory_bw_gbps: 1008.0,
            vram_gb: 24.0,
        }),
        roofline: RooflineParams {
            cpu_arithmetic_intensity: 10.0,
            gpu_arithmetic_intensity: Some(50.0),
        },
        byte_budget: None,
    };

    let features = TunerFeatures::builder().hardware(&hw).build();

    // Memory BW: 1008 / 3000 ~ 0.336
    assert!((features.gpu_mem_bw_norm - (1008.0 / 3000.0)).abs() < 0.01);
    // Compute: 82.6 / 500 ~ 0.1652
    assert!((features.gpu_compute_norm - (82.6 / 500.0)).abs() < 0.01);
}

#[test]
fn test_builder_hardware_without_gpu() {
    use crate::hardware::{CpuCapability, HardwareCapability, RooflineParams, SimdWidth};

    let hw = HardwareCapability {
        timestamp: "test".to_string(),
        hostname: "test-host".to_string(),
        cpu: CpuCapability {
            vendor: "Intel".to_string(),
            model: "Test CPU".to_string(),
            cores: 8,
            threads: 16,
            simd: SimdWidth::Avx2,
            base_freq_ghz: 3.5,
            peak_gflops: 100.0,
            memory_bw_gbps: 50.0,
        },
        gpu: None,
        roofline: RooflineParams { cpu_arithmetic_intensity: 10.0, gpu_arithmetic_intensity: None },
        byte_budget: None,
    };

    let features = TunerFeatures::builder().hardware(&hw).build();

    // No GPU: should use defaults
    // Default gpu_mem_bw_gbs = 1000.0 / 3000.0
    assert!((features.gpu_mem_bw_norm - (1000.0 / 3000.0)).abs() < 0.01);
}

// =========================================================================
// Coverage: FeatureExtractor::with_hardware() + extract() + calculate_efficiency()
// =========================================================================

#[test]
fn test_feature_extractor_with_hardware_and_extract() {
    use crate::brick::BrickProfiler;
    use crate::hardware::{
        CpuCapability, GpuBackend, GpuCapability, HardwareCapability, RooflineParams, SimdWidth,
    };

    let hw = HardwareCapability {
        timestamp: "test".to_string(),
        hostname: "test-host".to_string(),
        cpu: CpuCapability {
            vendor: "Intel".to_string(),
            model: "Test CPU".to_string(),
            cores: 8,
            threads: 16,
            simd: SimdWidth::Avx2,
            base_freq_ghz: 3.5,
            peak_gflops: 100.0,
            memory_bw_gbps: 50.0,
        },
        gpu: Some(GpuCapability {
            vendor: "NVIDIA".to_string(),
            model: "RTX 4090".to_string(),
            backend: GpuBackend::Cuda,
            compute_capability: Some("8.9".to_string()),
            peak_tflops_fp32: 82.6,
            peak_tflops_tensor: Some(330.0),
            memory_bw_gbps: 1008.0,
            vram_gb: 24.0,
        }),
        roofline: RooflineParams {
            cpu_arithmetic_intensity: 10.0,
            gpu_arithmetic_intensity: Some(50.0),
        },
        byte_budget: None,
    };

    let extractor = FeatureExtractor::with_hardware(hw);
    assert!(extractor.hardware.is_some());

    // Create a profiler with data so tokens_per_sec returns Some
    let mut profiler = BrickProfiler::new();
    profiler.enable();
    let elapsed = std::time::Duration::from_millis(10);
    profiler.record_elapsed("RmsNorm", elapsed, 1000);

    let config = RunConfig::default();
    let features = extractor.extract(&profiler, &config);

    // Should have measured_tps set
    assert!(features.measured_tps.is_some());
    // Should have theoretical_efficiency set
    assert!(features.theoretical_efficiency >= 0.0);
    assert!(features.theoretical_efficiency <= 1.0);
    // Should have bottleneck_class set
    assert!(features.bottleneck_class.is_some());
}

#[test]
fn test_calculate_efficiency_with_hardware() {
    use crate::brick::BrickProfiler;
    use crate::hardware::{
        CpuCapability, GpuBackend, GpuCapability, HardwareCapability, RooflineParams, SimdWidth,
    };

    let hw = HardwareCapability {
        timestamp: "test".to_string(),
        hostname: "test-host".to_string(),
        cpu: CpuCapability {
            vendor: "Intel".to_string(),
            model: "Test CPU".to_string(),
            cores: 8,
            threads: 16,
            simd: SimdWidth::Avx2,
            base_freq_ghz: 3.5,
            peak_gflops: 100.0,
            memory_bw_gbps: 50.0,
        },
        gpu: Some(GpuCapability {
            vendor: "NVIDIA".to_string(),
            model: "RTX 4090".to_string(),
            backend: GpuBackend::Cuda,
            compute_capability: Some("8.9".to_string()),
            peak_tflops_fp32: 82.6,
            peak_tflops_tensor: Some(330.0),
            memory_bw_gbps: 1008.0,
            vram_gb: 24.0,
        }),
        roofline: RooflineParams {
            cpu_arithmetic_intensity: 10.0,
            gpu_arithmetic_intensity: Some(50.0),
        },
        byte_budget: None,
    };

    let extractor = FeatureExtractor::with_hardware(hw);

    let mut profiler = BrickProfiler::new();
    profiler.enable();
    let elapsed = std::time::Duration::from_millis(10);
    profiler.record_elapsed("RmsNorm", elapsed, 1000);

    let config = RunConfig::default();
    let efficiency = extractor.calculate_efficiency(&profiler, &config);
    assert!(efficiency.is_some());
    let eff = efficiency.unwrap();
    assert!((0.0..=1.0).contains(&eff));
}

#[test]
fn test_calculate_efficiency_no_hardware() {
    use crate::brick::BrickProfiler;

    let extractor = FeatureExtractor::new();
    let mut profiler = BrickProfiler::new();
    profiler.enable();
    let elapsed = std::time::Duration::from_millis(10);
    profiler.record_elapsed("RmsNorm", elapsed, 1000);

    let config = RunConfig::default();
    let efficiency = extractor.calculate_efficiency(&profiler, &config);
    assert!(efficiency.is_none(), "No hardware -> no efficiency calculation");
}

// =========================================================================
// Coverage: classify_bottleneck with profiler data (lines 527-553)
// =========================================================================

#[test]
fn test_classify_bottleneck_attention_dominant() {
    use crate::brick::BrickProfiler;

    let extractor = FeatureExtractor::new();
    let mut profiler = BrickProfiler::new();
    profiler.enable();

    // Record attention bricks with 50% of time
    let attn_elapsed = std::time::Duration::from_millis(50);
    profiler.record_elapsed("QkvProjection", attn_elapsed, 100);
    profiler.record_elapsed("AttentionScore", attn_elapsed, 100);

    // Record FFN with 20% of time
    let ffn_elapsed = std::time::Duration::from_millis(20);
    profiler.record_elapsed("GateProjection", ffn_elapsed, 100);

    // Record norm with 5% of time
    let norm_elapsed = std::time::Duration::from_millis(5);
    profiler.record_elapsed("RmsNorm", norm_elapsed, 100);

    let bottleneck = extractor.classify_bottleneck(&profiler);
    assert_eq!(bottleneck, BottleneckClass::AttentionBound);
}

#[test]
fn test_classify_bottleneck_ffn_dominant() {
    use crate::brick::BrickProfiler;

    let extractor = FeatureExtractor::new();
    let mut profiler = BrickProfiler::new();
    profiler.enable();

    // Record FFN bricks with 60% of time
    let ffn_elapsed = std::time::Duration::from_millis(60);
    profiler.record_elapsed("GateProjection", ffn_elapsed, 100);
    profiler.record_elapsed("UpProjection", ffn_elapsed, 100);
    profiler.record_elapsed("DownProjection", ffn_elapsed, 100);

    // Record attention with 10% of time
    let attn_elapsed = std::time::Duration::from_millis(10);
    profiler.record_elapsed("QkvProjection", attn_elapsed, 100);

    // Record norm with 5% of time
    let norm_elapsed = std::time::Duration::from_millis(5);
    profiler.record_elapsed("RmsNorm", norm_elapsed, 100);

    let bottleneck = extractor.classify_bottleneck(&profiler);
    assert_eq!(bottleneck, BottleneckClass::MemoryBound);
}

#[test]
fn test_classify_bottleneck_norm_dominant() {
    use crate::brick::BrickProfiler;

    let extractor = FeatureExtractor::new();
    let mut profiler = BrickProfiler::new();
    profiler.enable();

    // Record norm bricks with 30% of time
    let norm_elapsed = std::time::Duration::from_millis(30);
    profiler.record_elapsed("RmsNorm", norm_elapsed, 100);

    // Record attention with 25% of time
    let attn_elapsed = std::time::Duration::from_millis(25);
    profiler.record_elapsed("QkvProjection", attn_elapsed, 100);

    // Record FFN with 30% of time
    let ffn_elapsed = std::time::Duration::from_millis(15);
    profiler.record_elapsed("GateProjection", ffn_elapsed, 100);
    profiler.record_elapsed("DownProjection", ffn_elapsed, 100);

    let bottleneck = extractor.classify_bottleneck(&profiler);
    assert_eq!(bottleneck, BottleneckClass::LaunchBound);
}

#[test]
fn test_classify_bottleneck_default_memory_bound() {
    use crate::brick::BrickProfiler;

    let extractor = FeatureExtractor::new();
    let mut profiler = BrickProfiler::new();
    profiler.enable();

    // Record mixed low-percentage bricks (no dominant category)
    // Attention < 35%, FFN < 50%, Norm < 20%
    let elapsed = std::time::Duration::from_millis(10);
    profiler.record_elapsed("QkvProjection", elapsed, 100); // ~25% attention
    profiler.record_elapsed("GateProjection", elapsed, 100); // ~25% FFN
    profiler.record_elapsed("RmsNorm", elapsed, 100); // ~25% norm... need to adjust

    // Actually: 3 equal parts means each is ~33%. Attention=33% < 35%, FFN=33% < 50%, Norm=33% > 20%
    // So this would hit LaunchBound. Let me adjust.
    // Need: attn<35%, ffn<50%, norm<20%
    // Use a dynamic "other" brick to dilute
    profiler.record_elapsed("Embedding", std::time::Duration::from_millis(30), 100);

    let bottleneck = extractor.classify_bottleneck(&profiler);
    // Embedding is "Other" category, so attn=10/60=16.7%, ffn=10/60=16.7%, norm=10/60=16.7%
    // All below thresholds -> default MemoryBound
    assert_eq!(bottleneck, BottleneckClass::MemoryBound);
}

#[test]
fn test_classify_bottleneck_empty_profiler() {
    use crate::brick::BrickProfiler;

    let extractor = FeatureExtractor::new();
    let profiler = BrickProfiler::new();

    let bottleneck = extractor.classify_bottleneck(&profiler);
    assert_eq!(bottleneck, BottleneckClass::Unknown);
}
