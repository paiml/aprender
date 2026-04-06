use super::*;

#[test]
fn test_governor_names() {
    assert_eq!(CpuGovernor::Performance.name(), "performance");
    assert_eq!(CpuGovernor::Powersave.name(), "powersave");
    assert_eq!(CpuGovernor::Ondemand.name(), "ondemand");
}

#[test]
fn test_governor_from_str() {
    assert_eq!(CpuGovernor::parse("performance"), CpuGovernor::Performance);
    assert_eq!(CpuGovernor::parse("POWERSAVE"), CpuGovernor::Powersave);
    assert_eq!(CpuGovernor::parse("unknown_gov"), CpuGovernor::Unknown);
}

#[test]
fn test_governor_deterministic() {
    assert!(CpuGovernor::Performance.is_deterministic());
    assert!(!CpuGovernor::Ondemand.is_deterministic());
}

#[test]
fn test_mock_controller() {
    let controller = FrequencyController::new().with_mock(3_500_000, CpuGovernor::Performance);

    let info = controller.read_cpu_frequency(0).unwrap();
    assert_eq!(info.current_khz, 3_500_000);
    assert_eq!(info.governor, CpuGovernor::Performance);
}

#[test]
fn test_frequency_info_conversions() {
    let info = CpuFrequencyInfo {
        cpu_id: 0,
        current_khz: 3_500_000,
        min_khz: 800_000,
        max_khz: 4_000_000,
        governor: CpuGovernor::Performance,
        available_governors: vec![],
    };

    assert_eq!(info.current_mhz(), 3500.0);
    assert!((info.current_ghz() - 3.5).abs() < 0.001);
    assert!((info.utilization() - 0.875).abs() < 0.001);
}

#[test]
fn test_frequency_reading() {
    let controller = FrequencyController::new().with_mock(3_000_000, CpuGovernor::Performance);
    let reading = controller.read_all_frequencies();

    assert!(!reading.cpus.is_empty());
    assert!(reading.average_mhz() > 0.0);
}

#[test]
fn test_frequency_variance() {
    let variance = FrequencyVariance {
        mean_mhz: 3000.0,
        std_dev_mhz: 50.0,
        cv_percent: 1.67,
        min_mhz: 2900.0,
        max_mhz: 3100.0,
        sample_count: 10,
    };

    assert!(variance.is_stable());
    assert_eq!(variance.range_mhz(), 200.0);
}
