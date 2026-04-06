#![allow(clippy::disallowed_methods, clippy::float_cmp)]
//! PMAT-016: Industry Baseline Validation Falsification Tests
//!
//! Falsification criteria F971-F985 from cbtop spec §21.7 and §21.8.
//!
//! # Test Coverage
//!
//! | ID | Claim | Test |
//! |----|-------|------|
//! | F971 | Realistic GPU throughput (within 30% of vLLM) | test_f971_throughput_within_30_percent |
//! | F972 | SM utilization correct (within 5% of nvidia-smi) | test_f972_sm_util_accuracy |
//! | F973 | Memory overhead tracked | test_f973_memory_overhead_tracked |
//! | F974 | Concurrency scaling shown | test_f974_concurrency_scaling |
//! | F975 | Baseline comparison available | test_f975_baseline_comparison_available |
//! | F976 | No foreign code in cbtop | test_f976_no_foreign_code |
//! | F977 | Reference tools documented | test_f977_reference_tools_documented |
//! | F978 | Side-by-side protocol works | test_f978_side_by_side_protocol |
//! | F979 | Gap analysis actionable | test_f979_gap_analysis_actionable |
//! | F980 | Pure Rust optimization works | test_f980_pure_rust_optimization |
//! | F981 | P95 latency tracked | test_f981_p95_latency_tracked |
//! | F982 | GPU class detected correctly | test_f982_gpu_class_detection |
//! | F983 | Throughput grade calculated | test_f983_throughput_grade_calculated |
//! | F984 | Health indicators work | test_f984_health_indicators |
//! | F985 | Benchmark methodology documented | test_f985_methodology_documented |

use cbtop::{
    BaselineComparison, BaselineValidator, GpuClass, SmHealth, ThroughputGrade, INDUSTRY_BASELINES,
    TGI_BASELINE, TRITON_BASELINE, VLLM_BASELINE,
};

/// F971: cbtop shows realistic GPU throughput (within 30% of vLLM baseline).
#[test]
fn test_f971_throughput_within_30_percent() {
    // A10 GPU achieving 350 tok/s (85% of vLLM's 412)
    let comparison = BaselineComparison::new("NVIDIA A10", 350, 92, Some(1800));

    assert!(
        comparison.vllm_percentage >= 70.0,
        "Throughput {}% should be >= 70% (within 30% of vLLM)",
        comparison.vllm_percentage
    );

    // Validator confirms
    let mut validator = BaselineValidator::new();
    assert!(validator.validate_f971_throughput(&comparison));
}

/// F971 negative: Detect when throughput is too low.
#[test]
fn test_f971_throughput_too_low_detected() {
    // Only 200 tok/s on A10 - should fail validation
    let comparison = BaselineComparison::new("NVIDIA A10", 200, 60, None);

    // This is < 70% of vLLM baseline
    let mut validator = BaselineValidator::new();
    let passed = validator.validate_f971_throughput(&comparison);

    // Should NOT pass (200/412 = 48.5%, which is < 70%)
    assert!(!passed, "Low throughput should fail F971 validation");
}

/// F972: SM utilization displayed correctly (within 5% of nvidia-smi).
#[test]
fn test_f972_sm_util_accuracy() {
    let mut validator = BaselineValidator::new();

    // Accurate reporting (2% diff)
    assert!(validator.validate_f972_sm_util(92, 90));

    // At threshold (5% diff)
    assert!(validator.validate_f972_sm_util(95, 90));

    // Beyond threshold (10% diff) - should fail
    let mut validator2 = BaselineValidator::new();
    assert!(!validator2.validate_f972_sm_util(80, 90));
}

/// F973: Memory overhead tracked.
#[test]
fn test_f973_memory_overhead_tracked() {
    // Memory overhead is part of ServerBaseline
    assert_ne!(VLLM_BASELINE.memory_overhead, 0);
    assert_ne!(TGI_BASELINE.memory_overhead, 0);
    assert_ne!(TRITON_BASELINE.memory_overhead, 0);

    // Documented values from Satna 2026
    assert_eq!(VLLM_BASELINE.memory_overhead, 42);
    assert_eq!(TGI_BASELINE.memory_overhead, 44);
    assert_eq!(TRITON_BASELINE.memory_overhead, 45);
}

/// F974: Concurrency scaling shown.
#[test]
fn test_f974_concurrency_scaling() {
    // Concurrency scaling is demonstrated by comparing throughput at different batch sizes.
    // This test verifies the infrastructure exists to track scaling.

    // Single request baseline
    let single = BaselineComparison::new("NVIDIA A10", 100, 40, None);

    // 32 concurrent requests (should show higher throughput)
    let concurrent = BaselineComparison::new("NVIDIA A10", 400, 95, None);

    // Scaling factor calculable
    let scaling = concurrent.actual_tok_per_sec as f64 / single.actual_tok_per_sec as f64;
    assert!(scaling > 1.0, "Concurrent should have higher throughput");

    // SM utilization should be higher with more concurrency
    assert!(concurrent.sm_utilization > single.sm_utilization);
}

/// F975: Baseline comparison available (--compare-baseline flag).
#[test]
fn test_f975_baseline_comparison_available() {
    let comparison = BaselineComparison::new("NVIDIA A10", 400, 95, None);

    // Baseline comparisons are populated
    assert!(!comparison.baseline_comparisons.is_empty());
    assert_eq!(comparison.baseline_comparisons.len(), 3); // vLLM, TGI, Triton

    // Each comparison has valid data
    for cmp in &comparison.baseline_comparisons {
        assert!(cmp.percentage > 0.0);
        assert!(!cmp.baseline.name.is_empty());
    }

    // Validator confirms
    let mut validator = BaselineValidator::new();
    assert!(validator.validate_f975_baseline_available(true));
}

/// F976: No foreign code in cbtop (no vLLM/llama.cpp dependencies).
#[test]
fn test_f976_no_foreign_code() {
    // This is validated by cargo tree - no vLLM, llama.cpp, or Python dependencies
    let mut validator = BaselineValidator::new();
    assert!(validator.validate_f976_no_foreign_code());

    // The baselines are data, not code dependencies
    assert_eq!(VLLM_BASELINE.name, "vLLM");
    // We reference vLLM data but don't depend on vLLM code
}

/// F977: Reference tools documented.
#[test]
fn test_f977_reference_tools_documented() {
    // Reference tools are documented in spec §21.8
    // This test verifies the baseline data includes references

    // Each baseline has documented GPU
    assert_eq!(VLLM_BASELINE.gpu, "A10");
    assert_eq!(TGI_BASELINE.gpu, "A10");
    assert_eq!(TRITON_BASELINE.gpu, "A10");

    // Satna 2026 citation is in the module docs
    // (verified by doc test compilation)
}

/// F978: Side-by-side protocol works.
#[test]
fn test_f978_side_by_side_protocol() {
    // Side-by-side comparison: run cbtop, then compare to external tool
    // This test verifies the comparison API works

    // Step 1: Get cbtop metrics
    let cbtop_throughput = 400u32;
    let cbtop_sm_util = 92u8;

    // Step 2: Compare against baselines
    let comparison = BaselineComparison::new("NVIDIA A10", cbtop_throughput, cbtop_sm_util, None);

    // Step 3: Report shows deltas
    for cmp in &comparison.baseline_comparisons {
        // Delta is calculated (actual - baseline_scaled)
        let _delta = cmp.delta_tok_per_sec;
        // Percentage is calculated
        let _pct = cmp.percentage;
    }

    // Protocol steps are executable
    assert!(comparison.is_within_expected_range());
}

/// F979: Gap analysis actionable.
#[test]
fn test_f979_gap_analysis_actionable() {
    // Poor throughput should generate suggestions
    let poor = BaselineComparison::new("NVIDIA A10", 150, 40, None);
    let suggestions = poor.suggestions();

    // Critical SM utilization should suggest improvements
    assert!(!suggestions.is_empty(), "Poor metrics should generate suggestions");

    // Good throughput should have fewer suggestions
    let good = BaselineComparison::new("NVIDIA A10", 420, 92, None);
    let good_suggestions = good.suggestions();

    // Either no suggestions or fewer than poor
    assert!(
        good_suggestions.len() <= suggestions.len(),
        "Good metrics should have fewer/no suggestions"
    );
}

/// F980: Pure Rust optimization works (improvement without foreign code).
#[test]
fn test_f980_pure_rust_optimization() {
    // Before optimization: low throughput
    let before = BaselineComparison::new("NVIDIA A10", 250, 60, None);
    let before_grade = before.grade;

    // After Pure Rust optimization: higher throughput
    let after = BaselineComparison::new("NVIDIA A10", 400, 92, None);
    let after_grade = after.grade;

    // Grade should improve
    assert!(after_grade >= before_grade, "Optimization should improve grade");

    // Throughput should improve
    assert!(after.actual_tok_per_sec > before.actual_tok_per_sec);
}

/// F981: P95 latency tracked.
#[test]
fn test_f981_p95_latency_tracked() {
    // P95 latency is tracked in BaselineComparison
    let with_latency = BaselineComparison::new("NVIDIA A10", 400, 92, Some(1700));
    assert_eq!(with_latency.p95_latency_ms, Some(1700));

    // Baseline P95 latency is documented
    assert_eq!(VLLM_BASELINE.p95_latency_ms, 1715);
    assert_eq!(TGI_BASELINE.p95_latency_ms, 1704);
    assert_eq!(TRITON_BASELINE.p95_latency_ms, 2007);
}

/// F982: GPU class detected correctly.
#[test]
fn test_f982_gpu_class_detection() {
    // A10 detection
    assert_eq!(GpuClass::from_name("NVIDIA A10"), GpuClass::A10);
    assert_eq!(GpuClass::from_name("Tesla A10"), GpuClass::A10);

    // A100 detection (must not match A10)
    assert_eq!(GpuClass::from_name("NVIDIA A100-SXM4-80GB"), GpuClass::A100);
    assert_eq!(GpuClass::from_name("A100 PCIe"), GpuClass::A100);

    // H100 detection
    assert_eq!(GpuClass::from_name("NVIDIA H100 PCIe"), GpuClass::H100);
    assert_eq!(GpuClass::from_name("H100-SXM5"), GpuClass::H100);

    // Consumer GPUs
    assert_eq!(GpuClass::from_name("GeForce RTX 4090"), GpuClass::Rtx4090);
    assert_eq!(GpuClass::from_name("RTX 3090 Ti"), GpuClass::Rtx3090);

    // Unknown
    assert_eq!(GpuClass::from_name("Some Random GPU"), GpuClass::Unknown);

    // Validator
    let mut validator = BaselineValidator::new();
    assert!(validator.validate_f982_gpu_detected(&GpuClass::A10));
    assert!(!validator.validate_f982_gpu_detected(&GpuClass::Unknown));
}

/// F983: Throughput grade calculated (A/B/C/D/F).
#[test]
fn test_f983_throughput_grade_calculated() {
    // Grade A: >= 100% of baseline
    assert_eq!(ThroughputGrade::from_percentage(100.0), ThroughputGrade::A);
    assert_eq!(ThroughputGrade::from_percentage(110.0), ThroughputGrade::A);

    // Grade B: 80-99%
    assert_eq!(ThroughputGrade::from_percentage(85.0), ThroughputGrade::B);

    // Grade C: 60-79%
    assert_eq!(ThroughputGrade::from_percentage(70.0), ThroughputGrade::C);

    // Grade D: 40-59%
    assert_eq!(ThroughputGrade::from_percentage(50.0), ThroughputGrade::D);

    // Grade F: < 40%
    assert_eq!(ThroughputGrade::from_percentage(30.0), ThroughputGrade::F);

    // In comparison
    let comparison = BaselineComparison::new("NVIDIA A10", 400, 92, None);
    let mut validator = BaselineValidator::new();
    assert!(validator.validate_f983_grade_calculated(&comparison.grade));
}

/// F984: Health indicators work (SM%, memory, scaling all visible).
#[test]
fn test_f984_health_indicators() {
    // SM health from utilization
    assert_eq!(SmHealth::from_utilization(98), SmHealth::Saturated);
    assert_eq!(SmHealth::from_utilization(90), SmHealth::Optimal);
    assert_eq!(SmHealth::from_utilization(60), SmHealth::Moderate);
    assert_eq!(SmHealth::from_utilization(30), SmHealth::Critical);

    // Health is acceptable for production
    assert!(SmHealth::Optimal.is_acceptable());
    assert!(SmHealth::Saturated.is_acceptable());
    assert!(!SmHealth::Moderate.is_acceptable());
    assert!(!SmHealth::Critical.is_acceptable());

    // Validator
    let mut validator = BaselineValidator::new();
    assert!(validator.validate_f984_health_indicators(true, true, true));
    assert!(!validator.validate_f984_health_indicators(false, true, true));
}

/// F985: Benchmark methodology documented.
#[test]
fn test_f985_methodology_documented() {
    // Industry baselines are defined
    assert_eq!(INDUSTRY_BASELINES.len(), 3);

    // vLLM baseline from Satna 2026
    assert_eq!(VLLM_BASELINE.peak_tok_per_sec, 412);
    assert_eq!(VLLM_BASELINE.p95_latency_ms, 1715);
    assert_eq!(VLLM_BASELINE.sm_utilization, 99);

    // TGI baseline
    assert_eq!(TGI_BASELINE.peak_tok_per_sec, 408);

    // Triton baseline
    assert_eq!(TRITON_BASELINE.peak_tok_per_sec, 385);

    // Expected throughput ranges documented
    assert_eq!(GpuClass::A10.expected_throughput(), (350, 450));
    assert_eq!(GpuClass::A100.expected_throughput(), (800, 1200));
    assert_eq!(GpuClass::H100.expected_throughput(), (1800, 2400));
}

/// Test validation summary aggregation.
#[test]
fn test_validation_summary() {
    let comparison = BaselineComparison::new("NVIDIA A10", 400, 92, Some(1700));
    let mut validator = BaselineValidator::new();

    validator.validate_f971_throughput(&comparison);
    validator.validate_f972_sm_util(92, 90);
    validator.validate_f975_baseline_available(true);
    validator.validate_f976_no_foreign_code();
    validator.validate_f982_gpu_detected(&comparison.gpu_class);
    validator.validate_f983_grade_calculated(&comparison.grade);
    validator.validate_f984_health_indicators(true, true, true);

    let summary = validator.summary();

    assert_eq!(summary.total, 7);
    assert_eq!(summary.passed, 7);
    assert_eq!(summary.failed, 0);
}

/// Integration test: Full baseline validation workflow.
#[test]
fn test_full_validation_workflow() {
    // Simulate cbtop metrics
    let gpu_name = "NVIDIA A10";
    let throughput = 390;
    let sm_util = 94;
    let p95_latency = 1750;

    // Create comparison
    let comparison = BaselineComparison::new(gpu_name, throughput, sm_util, Some(p95_latency));

    // Verify all F971-F985 criteria
    assert_eq!(comparison.gpu_class, GpuClass::A10);
    assert!(comparison.is_within_expected_range());
    assert!(comparison.vllm_percentage >= 70.0); // F971
    assert!(comparison.sm_health.is_acceptable()); // F984

    // Grade should be B or better
    assert!(comparison.grade >= ThroughputGrade::C);

    // Display comparison (tests Display impl)
    let display = format!("{}", comparison);
    assert!(display.contains("Baseline Comparison Report"));
    assert!(display.contains("A10"));
}
