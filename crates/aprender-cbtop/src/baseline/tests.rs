use super::*;

#[test]
fn test_gpu_class_detection() {
    assert_eq!(GpuClass::from_name("NVIDIA A10"), GpuClass::A10);
    assert_eq!(GpuClass::from_name("NVIDIA A100-SXM4-80GB"), GpuClass::A100);
    assert_eq!(GpuClass::from_name("NVIDIA H100 PCIe"), GpuClass::H100);
    assert_eq!(
        GpuClass::from_name("NVIDIA GeForce RTX 4090"),
        GpuClass::Rtx4090
    );
    assert_eq!(
        GpuClass::from_name("NVIDIA GeForce RTX 3090"),
        GpuClass::Rtx3090
    );
    assert_eq!(GpuClass::from_name("Unknown GPU"), GpuClass::Unknown);
}

#[test]
fn test_throughput_grade() {
    assert_eq!(ThroughputGrade::from_percentage(105.0), ThroughputGrade::A);
    assert_eq!(ThroughputGrade::from_percentage(100.0), ThroughputGrade::A);
    assert_eq!(ThroughputGrade::from_percentage(85.0), ThroughputGrade::B);
    assert_eq!(ThroughputGrade::from_percentage(65.0), ThroughputGrade::C);
    assert_eq!(ThroughputGrade::from_percentage(45.0), ThroughputGrade::D);
    assert_eq!(ThroughputGrade::from_percentage(35.0), ThroughputGrade::F);
}

#[test]
fn test_sm_health() {
    assert_eq!(SmHealth::from_utilization(98), SmHealth::Saturated);
    assert_eq!(SmHealth::from_utilization(85), SmHealth::Optimal);
    assert_eq!(SmHealth::from_utilization(60), SmHealth::Moderate);
    assert_eq!(SmHealth::from_utilization(40), SmHealth::Critical);
}

#[test]
fn test_baseline_comparison_a10() {
    let comparison = BaselineComparison::new("NVIDIA A10", 400, 95, Some(1700));

    assert_eq!(comparison.gpu_class, GpuClass::A10);
    assert!(comparison.vllm_percentage > 90.0); // Should be close to 100%
    assert!(comparison.is_within_expected_range());
    assert!(comparison.grade >= ThroughputGrade::B);
}

#[test]
fn test_baseline_comparison_h100() {
    let comparison = BaselineComparison::new("NVIDIA H100 PCIe", 2000, 92, None);

    assert_eq!(comparison.gpu_class, GpuClass::H100);
    // H100 baseline is scaled up from A10
    assert!(comparison.is_within_expected_range());
}

#[test]
fn test_validator_f971() {
    let comparison = BaselineComparison::new("NVIDIA A10", 350, 90, None);
    let mut validator = BaselineValidator::new();

    let passed = validator.validate_f971_throughput(&comparison);
    assert!(passed); // 350/412 ~= 85% > 70%
}

#[test]
fn test_validator_f972() {
    let mut validator = BaselineValidator::new();

    assert!(validator.validate_f972_sm_util(92, 90)); // 2% diff
    assert!(!validator.validate_f972_sm_util(92, 80)); // 12% diff
}

#[test]
fn test_validator_f976_no_foreign() {
    let mut validator = BaselineValidator::new();
    assert!(validator.validate_f976_no_foreign_code());
}

#[test]
fn test_industry_baselines_defined() {
    // F985: Benchmark methodology documented
    assert_eq!(VLLM_BASELINE.peak_tok_per_sec, 412);
    assert_eq!(TGI_BASELINE.peak_tok_per_sec, 408);
    assert_eq!(TRITON_BASELINE.peak_tok_per_sec, 385);
}

#[test]
fn test_expected_throughput_ranges() {
    assert_eq!(GpuClass::A10.expected_throughput(), (350, 450));
    assert_eq!(GpuClass::A100.expected_throughput(), (800, 1200));
    assert_eq!(GpuClass::H100.expected_throughput(), (1800, 2400));
}

#[test]
fn test_grade_thresholds() {
    assert_eq!(ThroughputGrade::A.threshold(), 100.0);
    assert_eq!(ThroughputGrade::B.threshold(), 80.0);
    assert_eq!(ThroughputGrade::C.threshold(), 60.0);
    assert_eq!(ThroughputGrade::D.threshold(), 40.0);
    assert_eq!(ThroughputGrade::F.threshold(), 0.0);
}

#[test]
fn test_validation_summary() {
    let mut validator = BaselineValidator::new();
    validator.validate_f976_no_foreign_code();
    validator.validate_f975_baseline_available(true);

    let summary = validator.summary();
    assert_eq!(summary.total, 2);
    assert_eq!(summary.passed, 2);
    assert_eq!(summary.failed, 0);
}
