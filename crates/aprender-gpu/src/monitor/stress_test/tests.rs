use super::*;

// =========================================================================
// H052: Stress Config Tests
// =========================================================================

#[test]
fn h052_stress_config_default() {
    let config = StressTestConfig::default();
    assert_eq!(config.target, StressTarget::All);
    assert_eq!(config.duration, Duration::from_secs(60));
    assert!((config.intensity - 1.0).abs() < 0.001);
}

#[test]
fn h052_stress_config_builder() {
    let config = StressTestConfig::new()
        .with_target(StressTarget::Cpu)
        .with_duration(Duration::from_secs(30))
        .with_intensity(0.8)
        .with_ramp_up(Duration::from_secs(10))
        .with_chaos(ChaosPreset::Gentle);

    assert_eq!(config.target, StressTarget::Cpu);
    assert_eq!(config.duration, Duration::from_secs(30));
    assert!((config.intensity - 0.8).abs() < 0.001);
    assert_eq!(config.chaos_preset, Some(ChaosPreset::Gentle));
}

#[test]
fn h052_stress_config_intensity_clamp() {
    let config = StressTestConfig::new().with_intensity(1.5);
    assert_eq!(config.intensity, 1.0);

    let config2 = StressTestConfig::new().with_intensity(-0.5);
    assert_eq!(config2.intensity, 0.0);
}

#[test]
fn h052_parse_duration() {
    assert_eq!(StressTestConfig::parse_duration("60s"), Some(Duration::from_secs(60)));
    assert_eq!(StressTestConfig::parse_duration("5m"), Some(Duration::from_secs(300)));
    assert_eq!(StressTestConfig::parse_duration("1h"), Some(Duration::from_secs(3600)));
    assert_eq!(StressTestConfig::parse_duration(""), None);
    assert_eq!(StressTestConfig::parse_duration("invalid"), None);
}

// =========================================================================
// H053: Stress Target Tests
// =========================================================================

#[test]
fn h053_stress_target_parse() {
    assert_eq!(StressTarget::parse("all"), Some(StressTarget::All));
    assert_eq!(StressTarget::parse("cpu"), Some(StressTarget::Cpu));
    assert_eq!(StressTarget::parse("GPU"), Some(StressTarget::Gpu(None)));
    assert_eq!(StressTarget::parse("memory"), Some(StressTarget::Memory));
    assert_eq!(StressTarget::parse("pcie"), Some(StressTarget::Pcie));
    assert!(matches!(StressTarget::parse("gpu:0"), Some(StressTarget::Gpu(Some(_)))));
    assert_eq!(StressTarget::parse("invalid"), None);
}

#[test]
fn h053_stress_target_includes() {
    let all = StressTarget::All;
    assert!(all.includes_cpu());
    assert!(all.includes_gpu());
    assert!(all.includes_memory());
    assert!(all.includes_pcie());

    let cpu = StressTarget::Cpu;
    assert!(cpu.includes_cpu());
    assert!(!cpu.includes_gpu());
    assert!(!cpu.includes_memory());
    assert!(!cpu.includes_pcie());
}

#[test]
fn h053_stress_target_custom() {
    let custom = StressTarget::Custom(vec![StressTarget::Cpu, StressTarget::Memory]);
    assert!(custom.includes_cpu());
    assert!(!custom.includes_gpu());
    assert!(custom.includes_memory());
    assert!(!custom.includes_pcie());
}

// =========================================================================
// H054: Chaos Preset Tests
// =========================================================================

#[test]
fn h054_chaos_preset_parse() {
    assert_eq!(ChaosPreset::parse("gentle"), Some(ChaosPreset::Gentle));
    assert_eq!(ChaosPreset::parse("MODERATE"), Some(ChaosPreset::Moderate));
    assert_eq!(ChaosPreset::parse("aggressive"), Some(ChaosPreset::Aggressive));
    assert_eq!(ChaosPreset::parse("extreme"), Some(ChaosPreset::Extreme));
    assert_eq!(ChaosPreset::parse("invalid"), None);
}

#[test]
fn h054_chaos_preset_factors() {
    let gentle = ChaosPreset::Gentle;
    assert!((gentle.memory_limit_factor() - 0.9).abs() < 0.001);
    assert!((gentle.cpu_throttle_factor() - 1.0).abs() < 0.001);
    assert_eq!(gentle.network_latency_ms(), 0);
    assert!((gentle.failure_rate() - 0.0).abs() < 0.001);

    let extreme = ChaosPreset::Extreme;
    assert!((extreme.memory_limit_factor() - 0.25).abs() < 0.001);
    assert!((extreme.cpu_throttle_factor() - 0.5).abs() < 0.001);
    assert_eq!(extreme.network_latency_ms(), 200);
    assert!((extreme.failure_rate() - 0.10).abs() < 0.001);
}

// =========================================================================
// H055: Stress Metrics Tests
// =========================================================================

#[test]
fn h055_stress_metrics_default() {
    let metrics = StressMetrics::new();
    assert_eq!(metrics.peak_cpu_utilization, 0.0);
    assert_eq!(metrics.thermal_throttle_count, 0);
    assert!(!metrics.has_errors());
}

#[test]
fn h055_stress_metrics_update_peaks() {
    let mut metrics = StressMetrics::new();

    metrics.update_peaks(50.0, 60.0, 70.0, 75.0, 300.0, 15.0);
    assert!((metrics.peak_cpu_utilization - 50.0).abs() < 0.01);
    assert_eq!(metrics.sample_count, 1);

    metrics.update_peaks(80.0, 40.0, 60.0, 70.0, 200.0, 10.0);
    assert!((metrics.peak_cpu_utilization - 80.0).abs() < 0.01);
    assert!((metrics.peak_gpu_utilization - 60.0).abs() < 0.01); // Previous was higher
    assert_eq!(metrics.sample_count, 2);
}

#[test]
fn h055_stress_metrics_events() {
    let mut metrics = StressMetrics::new();

    metrics.record_thermal_throttle();
    metrics.record_thermal_throttle();
    metrics.record_power_throttle();
    metrics.record_memory_pressure();

    assert_eq!(metrics.thermal_throttle_count, 2);
    assert_eq!(metrics.power_throttle_count, 1);
    assert_eq!(metrics.memory_pressure_events, 1);
}

#[test]
fn h055_stress_metrics_errors() {
    let mut metrics = StressMetrics::new();
    assert!(!metrics.has_errors());
    assert_eq!(metrics.total_errors(), 0);

    metrics.add_gpu_error("GPU timeout");
    metrics.add_memory_error("OOM");
    metrics.add_transfer_error("Transfer failed");

    assert!(metrics.has_errors());
    assert_eq!(metrics.total_errors(), 3);
}

#[test]
fn h055_stress_metrics_degradation() {
    let mut metrics = StressMetrics::new();
    metrics.baseline_flops = 1000.0;
    metrics.achieved_flops = 750.0;
    metrics.calculate_degradation();

    assert!((metrics.performance_degradation_pct - 25.0).abs() < 0.01);
}

// =========================================================================
// H056: Stress Report Tests
// =========================================================================

#[test]
fn h056_stress_report_pass() {
    let config = StressTestConfig::default();
    let metrics = StressMetrics::new();
    let report = StressTestReport::new(config, metrics, Duration::from_secs(60));

    assert_eq!(report.verdict, StressTestVerdict::Pass);
    assert!(!report.recommendations.is_empty());
}

#[test]
fn h056_stress_report_pass_with_notes() {
    let config = StressTestConfig::default();
    let mut metrics = StressMetrics::new();
    metrics.thermal_throttle_count = 2;

    let report = StressTestReport::new(config, metrics, Duration::from_secs(60));

    assert_eq!(report.verdict, StressTestVerdict::PassWithNotes);
}

#[test]
fn h056_stress_report_fail_errors() {
    let config = StressTestConfig::default();
    let mut metrics = StressMetrics::new();
    metrics.add_gpu_error("Critical error");

    let report = StressTestReport::new(config, metrics, Duration::from_secs(60));

    assert_eq!(report.verdict, StressTestVerdict::Fail);
}

#[test]
fn h056_stress_report_fail_thermal() {
    let config = StressTestConfig::default();
    let mut metrics = StressMetrics::new();
    metrics.peak_temperature_c = 100.0;

    let report = StressTestReport::new(config, metrics, Duration::from_secs(60));

    assert_eq!(report.verdict, StressTestVerdict::Fail);
}

#[test]
fn h056_stress_report_to_json() {
    let config = StressTestConfig::default();
    let metrics = StressMetrics::new();
    let report = StressTestReport::new(config, metrics, Duration::from_secs(60));

    let json = report.to_json();
    assert!(json.contains("\"verdict\": \"PASS\""));
    assert!(json.contains("\"duration_seconds\""));
}

// =========================================================================
// H057: Stress Test State Tests
// =========================================================================

#[test]
fn h057_stress_state_display() {
    assert_eq!(format!("{}", StressTestState::Idle), "Idle");
    assert_eq!(format!("{}", StressTestState::RampUp), "Ramp-Up");
    assert_eq!(format!("{}", StressTestState::Running), "Running");
    assert_eq!(format!("{}", StressTestState::CoolDown), "Cool-Down");
    assert_eq!(format!("{}", StressTestState::Completed), "Completed");
    assert_eq!(format!("{}", StressTestState::Aborted), "Aborted");
}

// =========================================================================
// H058: Verdict Display Tests
// =========================================================================

#[test]
fn h058_verdict_display() {
    assert_eq!(format!("{}", StressTestVerdict::Pass), "PASS");
    assert_eq!(format!("{}", StressTestVerdict::PassWithNotes), "PASS_WITH_NOTES");
    assert_eq!(format!("{}", StressTestVerdict::Fail), "FAIL");
}
