//! H0-MON-70 through H0-MON-80: MonitorConfig and MonitorError tests

use std::time::Duration;

use crate::monitor::*;

// =========================================================================
// H0-MON-70: MonitorConfig
// =========================================================================

#[test]
fn h0_mon_70_config_default() {
    let config = MonitorConfig::default();

    assert_eq!(config.poll_interval, Duration::from_millis(100));
    assert_eq!(config.history_size, 600);
    assert!(!config.background_collection);
}

#[test]
fn h0_mon_71_config_high_frequency() {
    let config = MonitorConfig::high_frequency();

    assert_eq!(config.poll_interval, Duration::from_millis(50));
    assert_eq!(config.history_size, 1200);
    assert!(config.background_collection);
}

#[test]
fn h0_mon_72_config_low_overhead() {
    let config = MonitorConfig::low_overhead();

    assert_eq!(config.poll_interval, Duration::from_millis(500));
    assert_eq!(config.history_size, 120);
    assert!(!config.background_collection);
}

// =========================================================================
// H0-MON-80: MonitorError
// =========================================================================

#[test]
fn h0_mon_80_error_display() {
    assert_eq!(format!("{}", MonitorError::NoDevice), "No GPU device available");
    assert_eq!(format!("{}", MonitorError::InvalidDevice(5)), "Invalid device index: 5");
    assert_eq!(
        format!("{}", MonitorError::BackendInit("test".to_string())),
        "Backend initialization failed: test"
    );
}
