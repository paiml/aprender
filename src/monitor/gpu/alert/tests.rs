//! Tests for GPU alert module.

use super::*;
use crate::monitor::gpu::GpuMetrics;

#[test]
fn test_gpu_alert_thermal_severity() {
    let alert = GpuAlert::ThermalThrottling { device: 0, temp: 90, threshold: 80 };
    assert_eq!(alert.severity(), 60); // 50 + min(10, 50)
}

#[test]
fn test_gpu_alert_memory_critical_severity() {
    let alert = GpuAlert::MemoryPressure { device: 0, used_percent: 99, threshold: 90 };
    assert_eq!(alert.severity(), 100);
}

#[test]
fn test_gpu_alert_message() {
    let alert = GpuAlert::ThermalThrottling { device: 0, temp: 85, threshold: 80 };
    let msg = alert.message();
    assert!(msg.contains("GPU 0"));
    assert!(msg.contains("85°C"));
    assert!(msg.contains("80°C"));
}

#[test]
fn test_gpu_alert_idle_message() {
    let alert = GpuAlert::GpuIdle { device: 1, duration_secs: 45 };
    let msg = alert.message();
    assert!(msg.contains("GPU 1"));
    assert!(msg.contains("45 seconds"));
}

#[test]
fn test_andon_thresholds_default() {
    let t = AndonThresholds::default();
    assert_eq!(t.thermal_warning, 80);
    assert_eq!(t.memory_warning, 90);
    assert_eq!(t.power_warning, 95);
}

#[test]
fn test_andon_system_no_alerts() {
    let mut andon = GpuAndonSystem::new();
    let metrics = vec![GpuMetrics {
        device_id: 0,
        temperature_celsius: 60,
        utilization_percent: 80,
        memory_used_mb: 4000,
        memory_total_mb: 16000,
        power_watts: 200.0,
        power_limit_watts: 450.0,
        ..Default::default()
    }];

    let alerts = andon.check(&metrics);
    assert!(alerts.is_empty());
}

#[test]
fn test_andon_system_thermal_alert() {
    let mut andon = GpuAndonSystem::new();
    let metrics = vec![GpuMetrics { device_id: 0, temperature_celsius: 85, ..Default::default() }];

    let alerts = andon.check(&metrics);
    assert_eq!(alerts.len(), 1);
    assert!(matches!(alerts[0], GpuAlert::ThermalThrottling { .. }));
}

#[test]
fn test_andon_system_memory_alert() {
    let mut andon = GpuAndonSystem::new();
    let metrics = vec![GpuMetrics {
        device_id: 0,
        memory_used_mb: 15000,
        memory_total_mb: 16000, // 93.75%
        ..Default::default()
    }];

    let alerts = andon.check(&metrics);
    assert!(alerts.iter().any(|a| matches!(a, GpuAlert::MemoryPressure { .. })));
}

#[test]
fn test_andon_system_power_alert() {
    let mut andon = GpuAndonSystem::new();
    let metrics = vec![GpuMetrics {
        device_id: 0,
        power_watts: 440.0,
        power_limit_watts: 450.0, // 97.8%
        ..Default::default()
    }];

    let alerts = andon.check(&metrics);
    assert!(alerts.iter().any(|a| matches!(a, GpuAlert::PowerLimit { .. })));
}

#[test]
fn test_andon_system_idle_alert() {
    let mut andon = GpuAndonSystem::new();
    andon.set_sample_interval(10);

    let metrics = vec![GpuMetrics { device_id: 0, utilization_percent: 0, ..Default::default() }];

    // Need 3 samples (30 seconds) to trigger idle alert
    andon.check(&metrics);
    andon.check(&metrics);
    andon.check(&metrics);
    let alerts = andon.check(&metrics);

    assert!(alerts.iter().any(|a| matches!(a, GpuAlert::GpuIdle { .. })));
}

#[test]
fn test_andon_system_idle_reset() {
    let mut andon = GpuAndonSystem::new();
    andon.set_sample_interval(10);

    let idle_metrics =
        vec![GpuMetrics { device_id: 0, utilization_percent: 0, ..Default::default() }];

    let active_metrics =
        vec![GpuMetrics { device_id: 0, utilization_percent: 50, ..Default::default() }];

    // Build up idle counter
    andon.check(&idle_metrics);
    andon.check(&idle_metrics);

    // Reset with activity
    andon.check(&active_metrics);

    // Check again - should not have idle alert yet
    let alerts = andon.check(&idle_metrics);
    assert!(!alerts.iter().any(|a| matches!(a, GpuAlert::GpuIdle { .. })));
}

#[test]
fn test_andon_system_has_critical_alerts() {
    let mut andon = GpuAndonSystem::new();
    let metrics = vec![GpuMetrics {
        device_id: 0,
        memory_used_mb: 15800,
        memory_total_mb: 16000, // 98.75% - critical
        ..Default::default()
    }];

    andon.check(&metrics);
    assert!(andon.has_critical_alerts());
}

#[test]
fn test_gpu_alert_memory_severity_thresholds() {
    // 95-98%
    let alert_95 = GpuAlert::MemoryPressure { device: 0, used_percent: 96, threshold: 90 };
    assert_eq!(alert_95.severity(), 80);

    // Below 95%
    let alert_91 = GpuAlert::MemoryPressure { device: 0, used_percent: 91, threshold: 90 };
    assert_eq!(alert_91.severity(), 60);
}

#[test]
fn test_gpu_alert_power_severity_thresholds() {
    // At 100%
    let alert_100 = GpuAlert::PowerLimit { device: 0, power_percent: 100, threshold: 95 };
    assert_eq!(alert_100.severity(), 70);

    // Below 100%
    let alert_99 = GpuAlert::PowerLimit { device: 0, power_percent: 99, threshold: 95 };
    assert_eq!(alert_99.severity(), 50);
}

#[test]
fn test_gpu_alert_idle_severity_thresholds() {
    // > 60 seconds
    let alert_long = GpuAlert::GpuIdle { device: 0, duration_secs: 65 };
    assert_eq!(alert_long.severity(), 90);

    // 31-60 seconds
    let alert_medium = GpuAlert::GpuIdle { device: 0, duration_secs: 45 };
    assert_eq!(alert_medium.severity(), 70);

    // <= 30 seconds
    let alert_short = GpuAlert::GpuIdle { device: 0, duration_secs: 20 };
    assert_eq!(alert_short.severity(), 40);
}

#[test]
fn test_gpu_alert_messages() {
    let memory_alert = GpuAlert::MemoryPressure { device: 2, used_percent: 95, threshold: 90 };
    let msg = memory_alert.message();
    assert!(msg.contains("GPU 2"));
    assert!(msg.contains("95%"));
    assert!(msg.contains("90%"));

    let power_alert = GpuAlert::PowerLimit { device: 3, power_percent: 98, threshold: 95 };
    let msg = power_alert.message();
    assert!(msg.contains("GPU 3"));
    assert!(msg.contains("98%"));
    assert!(msg.contains("95%"));
}

#[test]
fn test_gpu_alert_serde() {
    let alert = GpuAlert::ThermalThrottling { device: 0, temp: 85, threshold: 80 };
    let json = serde_json::to_string(&alert).expect("JSON serialization should succeed");
    let parsed: GpuAlert =
        serde_json::from_str(&json).expect("JSON deserialization should succeed");
    assert_eq!(alert, parsed);
}

#[test]
fn test_andon_thresholds_serde() {
    let thresholds = AndonThresholds::default();
    let json = serde_json::to_string(&thresholds).expect("JSON serialization should succeed");
    let parsed: AndonThresholds =
        serde_json::from_str(&json).expect("JSON deserialization should succeed");
    assert_eq!(thresholds.thermal_warning, parsed.thermal_warning);
}

#[test]
fn test_gpu_alert_clone() {
    let alert = GpuAlert::MemoryPressure { device: 1, used_percent: 95, threshold: 90 };
    let cloned = alert.clone();
    assert_eq!(alert, cloned);
}

#[test]
fn test_andon_thresholds_clone() {
    let thresholds = AndonThresholds::default();
    let cloned = thresholds.clone();
    assert_eq!(thresholds.thermal_warning, cloned.thermal_warning);
}

#[test]
fn test_andon_system_custom_thresholds() {
    let thresholds = AndonThresholds {
        thermal_warning: 70, // Lower threshold
        thermal_critical: 85,
        memory_warning: 80,
        memory_critical: 90,
        power_warning: 90,
        idle_threshold_secs: 20,
    };
    let mut andon = GpuAndonSystem::with_thresholds(thresholds);

    let metrics = vec![GpuMetrics {
        device_id: 0,
        temperature_celsius: 75, // Would not trigger default 80 but triggers 70
        ..Default::default()
    }];

    let alerts = andon.check(&metrics);
    assert!(alerts.iter().any(|a| matches!(a, GpuAlert::ThermalThrottling { .. })));
}

#[test]
fn test_andon_system_alerts_cleared_on_check() {
    let mut andon = GpuAndonSystem::new();
    // Use critical memory (99%) which has severity 100
    let critical_metrics = vec![GpuMetrics {
        device_id: 0,
        memory_used_mb: 15840, // 99% of 16000
        memory_total_mb: 16000,
        ..Default::default()
    }];

    andon.check(&critical_metrics);
    assert!(andon.has_critical_alerts());

    // Check with normal metrics - alerts should be cleared
    let normal_metrics = vec![GpuMetrics {
        device_id: 0,
        memory_used_mb: 8000,
        memory_total_mb: 16000,
        ..Default::default()
    }];
    andon.check(&normal_metrics);
    assert!(!andon.has_critical_alerts());
}

#[test]
fn test_thermal_severity_max_excess() {
    // Test thermal severity with large excess (should cap at 100)
    let alert = GpuAlert::ThermalThrottling { device: 0, temp: 200, threshold: 80 };
    // 50 + min(120, 50) = 50 + 50 = 100
    assert_eq!(alert.severity(), 100);
}

mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        #[test]
        fn prop_alert_severity_bounds(
            temp in 0u32..200,
            threshold in 1u32..100
        ) {
            let alert = GpuAlert::ThermalThrottling {
                device: 0,
                temp,
                threshold,
            };
            let severity = alert.severity();
            prop_assert!(severity >= 50);
            prop_assert!(severity <= 100);
        }
    }
}
