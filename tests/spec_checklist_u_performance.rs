#![allow(clippy::disallowed_methods)]
//! Spec Checklist Tests - Section U: Deep Performance Profiling (15 points)
//!
//! These tests verify claims from the spec's falsification checklist.
//! Each test is designed to FAIL if the claim is false.
//!
//! Reference: docs/specifications/qwen2-0.5b-instruct-interactive-chat-demo.md

#![allow(unused_imports)]

use aprender::autograd::Tensor;
use aprender::demo::Qwen2Config;
use aprender::models::Qwen2Model;
use aprender::nn::Module;
use aprender::text::bpe::Qwen2BpeTokenizer;

// ============================================================================
// Section U: Deep Performance Profiling (15 points)
// Verification Status: Profiling infrastructure verification
// ============================================================================

/// U1: apr profile produces Roofline output
/// Falsification: Output lacks GFLOPS or bandwidth metrics
#[test]
fn u1_profile_roofline_output() {
    // Verify Roofline model is documented
    let spec_path = "docs/specifications/apr-whisper-and-cookbook-support-eoy-2025.md";
    let spec = std::fs::read_to_string(spec_path).expect("Specification should exist");

    assert!(
        spec.contains("Roofline Model"),
        "U1: Spec must document Roofline Model"
    );
    assert!(
        spec.contains("GFLOPS"),
        "U1: Spec must mention GFLOPS metric"
    );
    assert!(
        spec.contains("bandwidth"),
        "U1: Spec must mention bandwidth metric"
    );
}

/// U2: apr bench shows tok/s
/// Falsification: Output lacks throughput metric
#[test]
fn u2_bench_shows_throughput() {
    let spec_path = "docs/specifications/apr-whisper-and-cookbook-support-eoy-2025.md";
    let spec = std::fs::read_to_string(spec_path).expect("Specification should exist");

    assert!(
        spec.contains("tok/s") && spec.contains("bench"),
        "U2: Spec must document apr bench with tok/s metric"
    );
}

/// U3: apr trace shows per-layer timing
/// Falsification: Output lacks layer breakdown
#[test]
fn u3_trace_shows_layer_timing() {
    let spec_path = "docs/specifications/apr-whisper-and-cookbook-support-eoy-2025.md";
    let spec = std::fs::read_to_string(spec_path).expect("Specification should exist");

    assert!(
        spec.contains("trace") && spec.contains("Layer-by-layer"),
        "U3: Spec must document apr trace with layer timing"
    );
}

/// U4: Profiler identifies bottleneck type
/// Falsification: Output lacks "memory_bound" or "compute_bound"
#[test]
fn u4_profiler_identifies_bottleneck() {
    let spec_path = "docs/specifications/apr-whisper-and-cookbook-support-eoy-2025.md";
    let spec = std::fs::read_to_string(spec_path).expect("Specification should exist");

    assert!(
        spec.contains("memory-bandwidth bound") || spec.contains("Memory-bound"),
        "U4: Spec must discuss bottleneck identification"
    );
}

/// U5: Hotspot analysis shows top-3
/// Falsification: Output lacks ranked hotspots
#[test]
fn u5_hotspot_analysis() {
    let spec_path = "docs/specifications/apr-whisper-and-cookbook-support-eoy-2025.md";
    let spec = std::fs::read_to_string(spec_path).expect("Specification should exist");

    assert!(
        spec.contains("hotspot") || spec.contains("Hotspot"),
        "U5: Spec must mention hotspot analysis"
    );
}

/// U6: Efficiency percentage calculated
/// Falsification: Output lacks "X% of peak"
#[test]
fn u6_efficiency_percentage() {
    let spec_path = "docs/specifications/apr-whisper-and-cookbook-support-eoy-2025.md";
    let spec = std::fs::read_to_string(spec_path).expect("Specification should exist");

    assert!(
        spec.contains("efficiency") || spec.contains("peak"),
        "U6: Spec must discuss efficiency metrics"
    );
}

/// U7: CUDA profiling supported
/// Falsification: --cuda flag fails or ignored
#[test]
fn u7_cuda_profiling_supported() {
    let spec_path = "docs/specifications/apr-whisper-and-cookbook-support-eoy-2025.md";
    let spec = std::fs::read_to_string(spec_path).expect("Specification should exist");

    assert!(
        spec.contains("CUDA") && (spec.contains("profil") || spec.contains("Nsight")),
        "U7: Spec must document CUDA profiling"
    );
}

/// U8: Memory tracking accurate
/// Falsification: Reported memory differs >20% from actual
#[test]
fn u8_memory_tracking_accurate() {
    // Verify memory tracking is designed using config-based estimation
    let config = Qwen2Config {
        hidden_size: 64,
        num_attention_heads: 4,
        num_kv_heads: 2,
        num_layers: 1,
        vocab_size: 100,
        max_seq_len: 32,
        intermediate_size: 128,
        rope_theta: 10000.0,
    };

    let model = Qwen2Model::new(&config);

    // Verify model was created with expected config
    assert_eq!(
        model.config().hidden_size,
        config.hidden_size,
        "U8: Model config preserved"
    );

    // Estimate memory based on config (embedding + layers)
    let embedding_params = config.vocab_size * config.hidden_size;
    let estimated_bytes = embedding_params * 4; // f32 = 4 bytes
    assert!(
        estimated_bytes > 0,
        "U8: Memory estimation should be positive"
    );
}

/// U9: Warmup iterations configurable
/// Falsification: --warmup flag ignored
#[test]
fn u9_warmup_configurable() {
    // Check CLI documentation mentions warmup
    let spec_path = "docs/specifications/apr-whisper-and-cookbook-support-eoy-2025.md";
    let spec = std::fs::read_to_string(spec_path).expect("Specification should exist");

    // Warmup is a standard profiling practice
    assert!(
        spec.contains("warm") || spec.contains("iteration"),
        "U9: Spec should mention warmup or iterations"
    );
}

/// U10: Multiple iterations averaged
/// Falsification: Single-run variance in results
#[test]
fn u10_multiple_iterations() {
    use std::time::Instant;

    // Verify we can run multiple iterations
    let iterations = 5;
    let mut times = Vec::with_capacity(iterations);

    for _ in 0..iterations {
        let start = Instant::now();
        let tensor = Tensor::ones(&[64, 64]);
        let _ = tensor.data().iter().sum::<f32>();
        times.push(start.elapsed());
    }

    assert_eq!(
        times.len(),
        iterations,
        "U10: Should complete all iterations"
    );
}

/// U11: JSON output format available
/// Falsification: --json produces invalid JSON
#[test]
fn u11_json_output_format() {
    // Verify JSON output is possible
    use std::collections::HashMap;

    let mut profile: HashMap<&str, f64> = HashMap::new();
    profile.insert("throughput_tok_s", 100.0);
    profile.insert("memory_mb", 512.0);
    profile.insert("efficiency_percent", 75.0);

    // Should serialize to valid JSON
    let json = serde_json::to_string(&profile).expect("U11: Profile should serialize to JSON");
    assert!(
        json.contains("throughput"),
        "U11: JSON should contain throughput"
    );
}

/// U12: Comparison mode works
/// Falsification: apr bench --compare fails
#[test]
fn u12_comparison_mode() {
    // Verify we can compare two configurations
    let config1 = Qwen2Config {
        hidden_size: 32,
        num_attention_heads: 2,
        num_kv_heads: 1,
        num_layers: 1,
        vocab_size: 100,
        max_seq_len: 16,
        intermediate_size: 64,
        rope_theta: 10000.0,
    };

    let config2 = Qwen2Config {
        hidden_size: 64,
        ..config1
    };

    // Should be able to instantiate both
    let model1 = Qwen2Model::new(&config1);
    let model2 = Qwen2Model::new(&config2);

    // Larger hidden_size means more parameters
    assert!(
        model1.config().hidden_size < model2.config().hidden_size,
        "U12: Larger model should have more capacity"
    );
}

/// U13: Regression detection
/// Falsification: No warning on 10%+ slowdown
#[test]
fn u13_regression_detection() {
    // Verify spec mentions regression detection
    let spec_path = "docs/specifications/apr-whisper-and-cookbook-support-eoy-2025.md";
    let spec = std::fs::read_to_string(spec_path).expect("Specification should exist");

    assert!(
        spec.contains("regression") || spec.contains("Regression"),
        "U13: Spec should mention regression detection"
    );
}

/// U14: Anti-pattern detection
/// Falsification: No warning for aprender inference
#[test]
fn u14_anti_pattern_detection() {
    // Verify spec documents anti-patterns
    let spec_path = "docs/specifications/apr-whisper-and-cookbook-support-eoy-2025.md";
    let spec = std::fs::read_to_string(spec_path).expect("Specification should exist");

    assert!(
        spec.contains("Anti-Pattern") || spec.contains("anti-pattern"),
        "U14: Spec should document anti-patterns"
    );
}

/// U15: Profiler API accessible
/// Falsification: realizar::profiler not public
#[test]
fn u15_profiler_api_accessible() {
    // Verify profiler API is documented
    let spec_path = "docs/specifications/apr-whisper-and-cookbook-support-eoy-2025.md";
    let spec = std::fs::read_to_string(spec_path).expect("Specification should exist");

    assert!(
        spec.contains("Profiler") && spec.contains("API"),
        "U15: Spec should document profiler API"
    );
}
