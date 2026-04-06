//! SIMD-accelerated Battery and Sensor metrics collector.
//!
//! This module provides SIMD-optimized collection of battery and temperature
//! sensor metrics using the SoA layouts defined in the simd module.
//!
//! ## Performance Targets (Falsifiable)
//!
//! - Battery collection: < 100μs
//! - Temperature sensors (≤16): < 200μs
//! - Combined collection: < 300μs
//!
//! ## Design
//!
//! Uses SIMD-accelerated parsing for sysfs files and vectorized statistics
//! computation for temperature data.

use crate::monitor::error::Result;
use crate::monitor::simd::ring_buffer::SimdRingBuffer;
use crate::monitor::simd::soa::{BatteryMetrics, BatteryStatus, SensorMetricsSoA, TempReading};
use crate::monitor::simd::{kernels, SimdStats};
use crate::monitor::types::{Collector, MetricValue, Metrics};
use std::path::PathBuf;
use std::time::Duration;

/// SIMD-accelerated battery and sensors collector.
#[derive(Debug)]
pub struct SimdBatterySensorsCollector {
    /// Battery metrics.
    battery: BatteryMetrics,
    /// Sensor metrics in SoA layout.
    sensors: SensorMetricsSoA,
    /// Battery capacity history (normalized 0-1).
    battery_history: SimdRingBuffer,
    /// Average temperature history (normalized 0-1, where 1 = 100°C).
    temp_history: SimdRingBuffer,
    /// Max temperature history.
    max_temp_history: SimdRingBuffer,
    /// Discovered battery path.
    battery_path: Option<PathBuf>,
    /// Discovered hwmon paths.
    hwmon_paths: Vec<PathBuf>,
    /// Whether battery is available.
    has_battery: bool,
}

impl SimdBatterySensorsCollector {
    /// Creates a new SIMD battery and sensors collector.
    #[must_use]
    pub fn new() -> Self {
        let battery_path = Self::find_battery_path();
        let hwmon_paths = Self::discover_hwmon();
        let has_battery = battery_path.is_some();

        Self {
            battery: BatteryMetrics::new(),
            sensors: SensorMetricsSoA::new(),
            battery_history: SimdRingBuffer::new(300),
            temp_history: SimdRingBuffer::new(300),
            max_temp_history: SimdRingBuffer::new(300),
            battery_path,
            hwmon_paths,
            has_battery,
        }
    }

    /// Finds the battery path in /sys/class/power_supply.
    #[cfg(target_os = "linux")]
    fn find_battery_path() -> Option<PathBuf> {
        let power_supply = PathBuf::from("/sys/class/power_supply");
        if !power_supply.exists() {
            return None;
        }

        std::fs::read_dir(&power_supply)
            .ok()?
            .flatten()
            .find(|entry| {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                name_str.starts_with("BAT") || name_str.contains("battery")
            })
            .map(|e| e.path())
    }

    #[cfg(not(target_os = "linux"))]
    fn find_battery_path() -> Option<PathBuf> {
        None
    }

    /// Discovers hwmon paths.
    #[cfg(target_os = "linux")]
    fn discover_hwmon() -> Vec<PathBuf> {
        let hwmon_base = PathBuf::from("/sys/class/hwmon");
        if !hwmon_base.exists() {
            return Vec::new();
        }

        std::fs::read_dir(&hwmon_base)
            .ok()
            .map(|entries| entries.flatten().map(|e| e.path()).collect())
            .unwrap_or_default()
    }

    #[cfg(not(target_os = "linux"))]
    fn discover_hwmon() -> Vec<PathBuf> {
        Vec::new()
    }

    /// Collects battery metrics using SIMD-optimized parsing.
    #[cfg(target_os = "linux")]
    fn collect_battery(&mut self) -> Result<()> {
        let path = match &self.battery_path {
            Some(p) => p,
            None => return Ok(()),
        };

        // Read capacity
        self.battery.capacity =
            Self::read_sysfs_u64(&path.join("capacity")).unwrap_or(0).min(100) as u8;

        // Read status
        let status_str = std::fs::read_to_string(path.join("status")).unwrap_or_default();
        self.battery.status = match status_str.trim().to_lowercase().as_str() {
            "charging" => BatteryStatus::Charging,
            "discharging" => BatteryStatus::Discharging,
            "full" => BatteryStatus::Full,
            "not charging" => BatteryStatus::NotCharging,
            _ => BatteryStatus::Unknown,
        };

        // Read energy values using SIMD integer parsing
        self.battery.energy_now = Self::read_sysfs_u64(&path.join("energy_now")).unwrap_or(0);
        self.battery.energy_full = Self::read_sysfs_u64(&path.join("energy_full")).unwrap_or(0);
        self.battery.energy_full_design =
            Self::read_sysfs_u64(&path.join("energy_full_design")).unwrap_or(0);
        self.battery.power_now = Self::read_sysfs_u64(&path.join("power_now")).unwrap_or(0);
        self.battery.voltage_now = Self::read_sysfs_u64(&path.join("voltage_now")).unwrap_or(0);

        // Calculate health and time remaining
        self.battery.calculate_health();
        self.battery.calculate_time_remaining();

        // Update history
        self.battery_history.push(f64::from(self.battery.capacity) / 100.0);

        Ok(())
    }

    #[cfg(not(target_os = "linux"))]
    fn collect_battery(&mut self) -> Result<()> {
        // Non-Linux: generate dummy data
        self.battery.capacity = 75;
        self.battery.status = BatteryStatus::Discharging;
        self.battery_history.push(0.75);
        Ok(())
    }

    /// Reads a u64 value from a sysfs file using SIMD parsing.
    #[cfg(target_os = "linux")]
    fn read_sysfs_u64(path: &PathBuf) -> Option<u64> {
        let content = std::fs::read_to_string(path).ok()?;
        let values = kernels::simd_parse_integers(content.as_bytes());
        values.first().copied()
    }

    /// Collects sensor metrics using SIMD-optimized parsing.
    #[cfg(target_os = "linux")]
    fn collect_sensors(&mut self) -> Result<()> {
        self.sensors.clear();

        for hwmon_path in &self.hwmon_paths {
            // Get hwmon name
            let name = std::fs::read_to_string(hwmon_path.join("name"))
                .unwrap_or_else(|_| "unknown".to_string())
                .trim()
                .to_string();

            // Find all temp inputs
            let entries = match std::fs::read_dir(hwmon_path) {
                Ok(e) => e,
                Err(_) => continue,
            };

            for entry in entries.flatten() {
                let file_name = entry.file_name();
                let file_str = file_name.to_string_lossy();

                // Look for temp*_input files
                if file_str.starts_with("temp") && file_str.ends_with("_input") {
                    let prefix = &file_str[..file_str.len() - 6]; // Remove "_input"

                    // Read temperature using SIMD parsing
                    let temp_raw = Self::read_sysfs_u64(&entry.path()).unwrap_or(0);
                    let current = temp_raw as f64 / 1000.0; // Convert from millidegrees

                    // Read label if available
                    let label = std::fs::read_to_string(hwmon_path.join(format!("{prefix}_label")))
                        .unwrap_or_else(|_| format!("{name} {prefix}"))
                        .trim()
                        .to_string();

                    // Read thresholds
                    let high = Self::read_sysfs_u64(&hwmon_path.join(format!("{prefix}_max")))
                        .map(|v| v as f64 / 1000.0);
                    let critical = Self::read_sysfs_u64(&hwmon_path.join(format!("{prefix}_crit")))
                        .map(|v| v as f64 / 1000.0);

                    self.sensors.add_temp(&label, current, high, critical);
                }
            }
        }

        // Update temperature history using SIMD statistics
        let avg_temp = self.sensors.avg_temp();
        let max_temp = self.sensors.max_temp();

        self.temp_history.push(avg_temp / 100.0); // Normalized to 0-1
        self.max_temp_history.push(max_temp / 100.0);

        Ok(())
    }

    #[cfg(not(target_os = "linux"))]
    fn collect_sensors(&mut self) -> Result<()> {
        // Non-Linux: generate dummy data
        self.sensors.clear();
        self.sensors.add_temp("CPU", 45.0, Some(80.0), Some(100.0));
        self.sensors.add_temp("GPU", 55.0, Some(85.0), Some(105.0));
        self.temp_history.push(0.50);
        self.max_temp_history.push(0.55);
        Ok(())
    }

    /// Returns the battery metrics.
    #[must_use]
    pub fn battery(&self) -> &BatteryMetrics {
        &self.battery
    }

    /// Returns the sensor metrics.
    #[must_use]
    pub fn sensors(&self) -> &SensorMetricsSoA {
        &self.sensors
    }

    /// Returns whether a battery is present.
    #[must_use]
    pub fn has_battery(&self) -> bool {
        self.has_battery
    }

    /// Returns battery capacity history.
    #[must_use]
    pub fn battery_history(&self) -> &SimdRingBuffer {
        &self.battery_history
    }

    /// Returns average temperature history.
    #[must_use]
    pub fn temp_history(&self) -> &SimdRingBuffer {
        &self.temp_history
    }

    /// Returns max temperature history.
    #[must_use]
    pub fn max_temp_history(&self) -> &SimdRingBuffer {
        &self.max_temp_history
    }

    /// Returns battery statistics.
    #[must_use]
    pub fn battery_stats(&self) -> &SimdStats {
        self.battery_history.statistics()
    }

    /// Returns temperature statistics.
    #[must_use]
    pub fn temp_stats(&self) -> &SimdStats {
        self.temp_history.statistics()
    }

    /// Returns all temperature readings.
    #[must_use]
    pub fn temp_readings(&self) -> &[TempReading] {
        &self.sensors.temps
    }
}

impl Default for SimdBatterySensorsCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl Collector for SimdBatterySensorsCollector {
    fn id(&self) -> &'static str {
        "battery_sensors_simd"
    }

    fn collect(&mut self) -> Result<Metrics> {
        // Collect battery and sensor data
        self.collect_battery()?;
        self.collect_sensors()?;

        let mut metrics = Metrics::new();

        // Battery metrics
        if self.has_battery {
            metrics
                .insert("battery.capacity", MetricValue::Counter(u64::from(self.battery.capacity)));
            metrics.insert("battery.health", MetricValue::Gauge(self.battery.health_pct));

            if self.battery.power_now > 0 {
                let power_w = self.battery.power_now as f64 / 1_000_000.0;
                metrics.insert("battery.power_watts", MetricValue::Gauge(power_w));
            }

            if let Some(time) = self.battery.time_to_empty {
                metrics.insert("battery.time_to_empty", MetricValue::Counter(time));
            }
            if let Some(time) = self.battery.time_to_full {
                metrics.insert("battery.time_to_full", MetricValue::Counter(time));
            }
        }

        // Sensor metrics
        metrics.insert("sensors.count", MetricValue::Counter(self.sensors.temps.len() as u64));

        if !self.sensors.temps.is_empty() {
            metrics.insert("sensors.avg_temp", MetricValue::Gauge(self.sensors.avg_temp()));
            metrics.insert("sensors.max_temp", MetricValue::Gauge(self.sensors.max_temp()));
        }

        Ok(metrics)
    }

    fn is_available(&self) -> bool {
        #[cfg(target_os = "linux")]
        {
            self.has_battery || !self.hwmon_paths.is_empty()
        }
        #[cfg(not(target_os = "linux"))]
        {
            false
        }
    }

    fn interval_hint(&self) -> Duration {
        Duration::from_millis(2000)
    }

    fn display_name(&self) -> &'static str {
        "Battery & Sensors (SIMD)"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_battery_sensors_new() {
        let collector = SimdBatterySensorsCollector::new();
        assert!(collector.battery_history.is_empty());
    }

    #[test]
    fn test_simd_battery_sensors_id() {
        let collector = SimdBatterySensorsCollector::new();
        assert_eq!(collector.id(), "battery_sensors_simd");
    }

    #[test]
    fn test_simd_battery_sensors_display_name() {
        let collector = SimdBatterySensorsCollector::new();
        assert_eq!(collector.display_name(), "Battery & Sensors (SIMD)");
    }

    #[test]
    fn test_simd_battery_sensors_interval() {
        let collector = SimdBatterySensorsCollector::new();
        assert_eq!(collector.interval_hint(), Duration::from_millis(2000));
    }

    #[test]
    fn test_battery_metrics() {
        let mut battery = BatteryMetrics::new();
        battery.capacity = 80;
        battery.status = BatteryStatus::Discharging;
        battery.energy_now = 40_000_000; // 40 Wh
        battery.energy_full = 50_000_000; // 50 Wh
        battery.energy_full_design = 60_000_000; // 60 Wh
        battery.power_now = 10_000_000; // 10 W

        battery.calculate_health();
        battery.calculate_time_remaining();

        assert!((battery.health_pct - 83.33).abs() < 0.1);
        assert!(battery.time_to_empty.is_some());
    }

    #[test]
    fn test_sensor_metrics() {
        let mut sensors = SensorMetricsSoA::new();
        sensors.add_temp("CPU", 45.0, Some(80.0), Some(100.0));
        sensors.add_temp("GPU", 55.0, Some(85.0), Some(105.0));

        assert_eq!(sensors.temps.len(), 2);
        assert!((sensors.avg_temp() - 50.0).abs() < 0.1);
        assert!((sensors.max_temp() - 55.0).abs() < 0.1);
    }

    #[test]
    fn test_battery_history() {
        let mut collector = SimdBatterySensorsCollector::new();
        collector.battery_history.push(0.75);
        collector.battery_history.push(0.70);
        collector.battery_history.push(0.65);

        assert_eq!(collector.battery_history().len(), 3);
        assert_eq!(collector.battery_history().latest(), Some(0.65));
    }

    #[test]
    fn test_temp_history() {
        let mut collector = SimdBatterySensorsCollector::new();
        collector.temp_history.push(0.45);
        collector.temp_history.push(0.50);
        collector.temp_history.push(0.55);

        assert_eq!(collector.temp_history().len(), 3);
        assert_eq!(collector.temp_history().latest(), Some(0.55));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_simd_battery_sensors_collect() {
        let mut collector = SimdBatterySensorsCollector::new();
        let result = collector.collect();
        assert!(result.is_ok());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_simd_battery_sensors_available() {
        let collector = SimdBatterySensorsCollector::new();
        // May or may not be available depending on hardware
        let _ = collector.is_available();
    }
}
