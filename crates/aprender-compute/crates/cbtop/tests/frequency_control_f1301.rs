#![allow(clippy::disallowed_methods, clippy::float_cmp)]
//! Falsification Tests for PMAT-038: CPU Frequency Control Backend
//!
//! F1301-F1310: CPU frequency control falsification tests

use cbtop::{
    CpuFrequencyInfo, CpuGovernor, FrequencyController, FrequencyLock, FrequencyReading,
    FrequencyVariance,
};

// =============================================================================
// F1301: Frequency Reading Tests
// =============================================================================

/// F1301.1: Can read current frequency
#[test]
fn f1301_read_frequency() {
    let controller = FrequencyController::new().with_mock(3_500_000, CpuGovernor::Performance);

    let info = controller.read_cpu_frequency(0).unwrap();
    assert_eq!(info.current_khz, 3_500_000);
    assert!(info.current_mhz() > 0.0);
}

/// F1301.2: Read all CPU frequencies
#[test]
fn f1301_read_all_frequencies() {
    let controller = FrequencyController::new().with_mock(3_000_000, CpuGovernor::Performance);

    let reading = controller.read_all_frequencies();
    assert!(!reading.cpus.is_empty());
    assert!(reading.timestamp_ns > 0);
}

/// F1301.3: Frequency conversions work
#[test]
fn f1301_frequency_conversions() {
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
}

// =============================================================================
// F1302: Governor Detection Tests
// =============================================================================

/// F1302.1: Detect current governor
#[test]
fn f1302_detect_governor() {
    let controller = FrequencyController::new().with_mock(3_000_000, CpuGovernor::Performance);

    let governor = controller.detect_governor();
    assert_eq!(governor, CpuGovernor::Performance);
}

/// F1302.2: Governor name parsing
#[test]
fn f1302_governor_names() {
    assert_eq!(CpuGovernor::parse("performance"), CpuGovernor::Performance);
    assert_eq!(CpuGovernor::parse("powersave"), CpuGovernor::Powersave);
    assert_eq!(CpuGovernor::parse("ondemand"), CpuGovernor::Ondemand);
    assert_eq!(CpuGovernor::parse("conservative"), CpuGovernor::Conservative);
    assert_eq!(CpuGovernor::parse("schedutil"), CpuGovernor::Schedutil);
    assert_eq!(CpuGovernor::parse("userspace"), CpuGovernor::Userspace);
    assert_eq!(CpuGovernor::parse("unknown_gov"), CpuGovernor::Unknown);
}

/// F1302.3: Case insensitive parsing
#[test]
fn f1302_case_insensitive() {
    assert_eq!(CpuGovernor::parse("PERFORMANCE"), CpuGovernor::Performance);
    assert_eq!(CpuGovernor::parse("PoWeRsAvE"), CpuGovernor::Powersave);
}

// =============================================================================
// F1303: Deterministic Governor Tests
// =============================================================================

/// F1303.1: Deterministic governors identified
#[test]
fn f1303_deterministic_governors() {
    assert!(CpuGovernor::Performance.is_deterministic());
    assert!(CpuGovernor::Powersave.is_deterministic());
    assert!(CpuGovernor::Userspace.is_deterministic());
}

/// F1303.2: Non-deterministic governors identified
#[test]
fn f1303_non_deterministic_governors() {
    assert!(!CpuGovernor::Ondemand.is_deterministic());
    assert!(!CpuGovernor::Conservative.is_deterministic());
    assert!(!CpuGovernor::Schedutil.is_deterministic());
}

// =============================================================================
// F1304: Utilization Calculation Tests
// =============================================================================

/// F1304.1: Utilization percentage calculated
#[test]
fn f1304_utilization() {
    let info = CpuFrequencyInfo {
        cpu_id: 0,
        current_khz: 3_500_000,
        min_khz: 800_000,
        max_khz: 4_000_000,
        governor: CpuGovernor::Performance,
        available_governors: vec![],
    };

    let util = info.utilization();
    assert!((util - 0.875).abs() < 0.001);
}

/// F1304.2: Zero max handled
#[test]
fn f1304_zero_max() {
    let info = CpuFrequencyInfo {
        cpu_id: 0,
        current_khz: 3_500_000,
        min_khz: 0,
        max_khz: 0,
        governor: CpuGovernor::Unknown,
        available_governors: vec![],
    };

    assert_eq!(info.utilization(), 1.0);
}

// =============================================================================
// F1305: Reading Statistics Tests
// =============================================================================

/// F1305.1: Average frequency calculated
#[test]
fn f1305_average_frequency() {
    let controller = FrequencyController::new().with_mock(3_000_000, CpuGovernor::Performance);
    let reading = controller.read_all_frequencies();

    assert!(reading.average_mhz() > 0.0);
}

/// F1305.2: Min/max frequency
#[test]
fn f1305_min_max_frequency() {
    let controller = FrequencyController::new().with_mock(3_000_000, CpuGovernor::Performance);
    let reading = controller.read_all_frequencies();

    assert!(reading.min_mhz() > 0.0);
    assert!(reading.max_mhz() >= reading.min_mhz());
}

/// F1305.3: Variance calculation
#[test]
fn f1305_variance() {
    let controller = FrequencyController::new().with_mock(3_000_000, CpuGovernor::Performance);
    let reading = controller.read_all_frequencies();

    // With mock mode, all CPUs have same frequency
    assert!(reading.variance_mhz() >= 0.0);
}

// =============================================================================
// F1306: Uniform Governor Tests
// =============================================================================

/// F1306.1: Uniform governor detected
#[test]
fn f1306_uniform_governor() {
    let controller = FrequencyController::new().with_mock(3_000_000, CpuGovernor::Performance);
    let reading = controller.read_all_frequencies();

    assert!(reading.uniform_governor());
}

/// F1306.2: Common governor returned
#[test]
fn f1306_common_governor() {
    let controller = FrequencyController::new().with_mock(3_000_000, CpuGovernor::Powersave);
    let reading = controller.read_all_frequencies();

    assert_eq!(reading.common_governor(), CpuGovernor::Powersave);
}

// =============================================================================
// F1307: Frequency Lock Tests
// =============================================================================

/// F1307.1: Frequency lock in mock mode
#[test]
fn f1307_frequency_lock_mock() {
    let controller = FrequencyController::new().with_mock(3_000_000, CpuGovernor::Performance);

    let lock = FrequencyLock::try_lock(&controller);
    assert!(lock.is_locked());
}

/// F1307.2: Lock state preserved
#[test]
fn f1307_lock_state() {
    let controller = FrequencyController::new().with_mock(3_000_000, CpuGovernor::Performance);

    {
        let lock = FrequencyLock::try_lock(&controller);
        assert!(lock.is_locked());
    }
    // Lock dropped, governor should be restored (in real mode)
}

// =============================================================================
// F1308: Variance Measurement Tests
// =============================================================================

/// F1308.1: Variance measurement works
#[test]
fn f1308_measure_variance() {
    let controller = FrequencyController::new().with_mock(3_000_000, CpuGovernor::Performance);

    // Only 2 samples to keep test fast
    let variance = controller.measure_variance(2, 1);

    assert!(variance.mean_mhz > 0.0);
    assert_eq!(variance.sample_count, 2);
}

/// F1308.2: Stable variance detection
#[test]
fn f1308_stable_variance() {
    let variance = FrequencyVariance {
        mean_mhz: 3000.0,
        std_dev_mhz: 50.0,
        cv_percent: 1.67,
        min_mhz: 2900.0,
        max_mhz: 3100.0,
        sample_count: 10,
    };

    assert!(variance.is_stable()); // CV < 3%
}

/// F1308.3: Unstable variance detection
#[test]
fn f1308_unstable_variance() {
    let variance = FrequencyVariance {
        mean_mhz: 3000.0,
        std_dev_mhz: 200.0,
        cv_percent: 6.67,
        min_mhz: 2500.0,
        max_mhz: 3500.0,
        sample_count: 10,
    };

    assert!(!variance.is_stable()); // CV > 3%
}

// =============================================================================
// F1309: CPU Count Tests
// =============================================================================

/// F1309.1: CPU count detected
#[test]
fn f1309_cpu_count() {
    let controller = FrequencyController::new();

    assert!(controller.cpu_count() > 0);
}

/// F1309.2: Mock mode CPU count
#[test]
fn f1309_mock_cpu_count() {
    let controller = FrequencyController::new().with_mock(3_000_000, CpuGovernor::Performance);

    assert!(controller.cpu_count() > 0);
}

// =============================================================================
// F1310: Control Capability Tests
// =============================================================================

/// F1310.1: Can control in mock mode
#[test]
fn f1310_can_control_mock() {
    let controller = FrequencyController::new().with_mock(3_000_000, CpuGovernor::Performance);

    assert!(controller.can_control());
}

/// F1310.2: Range calculation
#[test]
fn f1310_variance_range() {
    let variance = FrequencyVariance {
        mean_mhz: 3000.0,
        std_dev_mhz: 50.0,
        cv_percent: 1.67,
        min_mhz: 2900.0,
        max_mhz: 3100.0,
        sample_count: 10,
    };

    assert_eq!(variance.range_mhz(), 200.0);
}

// =============================================================================
// Additional Tests
// =============================================================================

/// Test governor names match
#[test]
fn test_governor_names() {
    assert_eq!(CpuGovernor::Performance.name(), "performance");
    assert_eq!(CpuGovernor::Powersave.name(), "powersave");
    assert_eq!(CpuGovernor::Ondemand.name(), "ondemand");
    assert_eq!(CpuGovernor::Conservative.name(), "conservative");
    assert_eq!(CpuGovernor::Schedutil.name(), "schedutil");
    assert_eq!(CpuGovernor::Userspace.name(), "userspace");
    assert_eq!(CpuGovernor::Unknown.name(), "unknown");
}

/// Test default variance
#[test]
fn test_default_variance() {
    let variance = FrequencyVariance::default();
    assert_eq!(variance.mean_mhz, 0.0);
    assert_eq!(variance.sample_count, 0);
}

/// Test empty reading stats
#[test]
fn test_empty_reading() {
    let reading = FrequencyReading { cpus: vec![], timestamp_ns: 0 };

    assert_eq!(reading.average_mhz(), 0.0);
    assert_eq!(reading.min_mhz(), 0.0);
    assert_eq!(reading.max_mhz(), 0.0);
    assert!(reading.uniform_governor());
    assert_eq!(reading.common_governor(), CpuGovernor::Unknown);
}

/// Test default controller
#[test]
fn test_default_controller() {
    let controller = FrequencyController::default();
    assert!(controller.cpu_count() > 0);
}
