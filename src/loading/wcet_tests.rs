use super::*;

#[test]
fn test_platform_specs_effective_throughput() {
    let specs = PlatformSpecs::new(100.0, 200.0, 1000.0, 100.0, 1_000_000.0);
    assert!((specs.effective_throughput_mbps() - 100.0).abs() < 0.001);
}

#[test]
fn test_header_info_compression_ratio() {
    let header = HeaderInfo::new(100, 400, false);
    assert!((header.compression_ratio() - 4.0).abs() < 0.001);
}

#[test]
fn test_header_info_compression_ratio_zero() {
    let header = HeaderInfo::new(0, 400, false);
    assert!((header.compression_ratio() - 1.0).abs() < 0.001);
}

#[test]
fn test_calculate_wcet_small_model() {
    let header = HeaderInfo::new(1024 * 1024, 2 * 1024 * 1024, false);
    let wcet = calculate_wcet(&header, &platforms::DESKTOP_X86);

    // Should be in millisecond range for small model on desktop
    assert!(wcet.as_millis() < 100);
}

#[test]
fn test_calculate_wcet_signed_model() {
    let header = HeaderInfo::new(1024 * 1024, 2 * 1024 * 1024, true);
    let wcet_signed = calculate_wcet(&header, &platforms::DESKTOP_X86);

    let header_unsigned = HeaderInfo::new(1024 * 1024, 2 * 1024 * 1024, false);
    let wcet_unsigned = calculate_wcet(&header_unsigned, &platforms::DESKTOP_X86);

    // Signed should take longer due to Ed25519 verification
    assert!(wcet_signed >= wcet_unsigned);
}

#[test]
fn test_calculate_wcet_platform_comparison() {
    let header = HeaderInfo::new(10 * 1024 * 1024, 20 * 1024 * 1024, false);

    let wcet_desktop = calculate_wcet(&header, &platforms::DESKTOP_X86);
    let wcet_aerospace = calculate_wcet(&header, &platforms::AEROSPACE_RAD750);

    // Aerospace should be much slower due to rad-hardened components
    assert!(wcet_aerospace > wcet_desktop * 5);
}

#[test]
fn test_calculate_wcet_breakdown() {
    let header = HeaderInfo::new(1024 * 1024, 2 * 1024 * 1024, false);
    let breakdown = calculate_wcet_breakdown(&header, &platforms::DESKTOP_X86);

    assert!(breakdown.total > Duration::ZERO);
    assert!(!breakdown.dominant_component().is_empty());

    let percentages = breakdown.percentages();
    let total_pct = percentages.header
        + percentages.read
        + percentages.decompress
        + percentages.verify
        + percentages.deserialize;
    assert!((total_pct - 100.0).abs() < 1.0);
}

#[test]
fn test_estimate_max_size_for_budget() {
    let budget = Duration::from_millis(100);
    let max_size = estimate_max_size_for_budget(&platforms::DESKTOP_X86, budget);

    // Desktop should support several MB in 100ms
    assert!(max_size > 1024 * 1024);
}

#[test]
fn test_min_ring_buffer_size() {
    // Consumer faster than producer - minimal buffer
    let size = min_ring_buffer_size(100.0, 200.0, 10.0);
    assert_eq!(size, 64 * 1024);

    // Producer faster - need larger buffer
    let size = min_ring_buffer_size(200.0, 100.0, 10.0);
    assert!(size > 64 * 1024);

    // Should be page-aligned
    assert_eq!(size % 4096, 0);
}

#[test]
fn test_assert_time_budget_success() {
    let header = HeaderInfo::new(1024, 2048, false);
    let result = assert_time_budget(&header, &platforms::DESKTOP_X86, Duration::from_secs(1));
    assert!(result.is_ok());
}

#[test]
fn test_assert_time_budget_failure() {
    let header = HeaderInfo::new(1024 * 1024 * 1024, 2 * 1024 * 1024 * 1024, false);
    let result = assert_time_budget(
        &header,
        &platforms::AEROSPACE_RAD750,
        Duration::from_micros(1),
    );
    assert!(result.is_err());

    if let Err(SafetyError::TimeBudgetExceeded { recommendation, .. }) = result {
        assert!(!recommendation.is_empty());
    }
}

#[test]
fn test_assert_memory_budget() {
    assert!(assert_memory_budget(100, 200).is_ok());
    assert!(assert_memory_budget(300, 200).is_err());
}

#[test]
fn test_safety_error_display() {
    let err = SafetyError::TimeBudgetExceeded {
        budget: Duration::from_millis(10),
        worst_case: Duration::from_millis(100),
        model_type: 1,
        compressed_size: 1024,
        recommendation: "Test recommendation".to_string(),
    };
    let display = format!("{}", err);
    assert!(display.contains("Time budget exceeded"));

    let err = SafetyError::IntegrityCheckFailed {
        expected: 0x12345678,
        computed: 0xABCDEF01,
    };
    let display = format!("{}", err);
    assert!(display.contains("Integrity check failed"));
}

#[test]
fn test_pre_characterized_platforms() {
    // Ensure all platforms have sensible values
    assert!(platforms::AUTOMOTIVE_S32G.min_read_speed_mbps > 0.0);
    assert!(platforms::AEROSPACE_RAD750.min_read_speed_mbps > 0.0);
    assert!(platforms::EDGE_RPI4.min_read_speed_mbps > 0.0);
    assert!(platforms::DESKTOP_X86.min_read_speed_mbps > 0.0);
    assert!(platforms::WASM_BROWSER.min_read_speed_mbps > 0.0);
    assert!(platforms::INDUSTRIAL_PLC.min_read_speed_mbps > 0.0);
}
