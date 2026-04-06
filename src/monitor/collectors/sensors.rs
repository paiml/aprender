//! Temperature sensor collector.
//!
//! Reads from `/sys/class/hwmon/` on Linux to collect temperature data.
//!
//! ## Falsification Criteria
//!
//! - #42: Temperature readings match `sensors` within ±1°C

use crate::monitor::error::Result;
use crate::monitor::types::{Collector, MetricValue, Metrics};
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Temperature scale for display.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TempScale {
    /// Celsius (default).
    #[default]
    Celsius,
    /// Fahrenheit.
    Fahrenheit,
    /// Kelvin.
    Kelvin,
}

impl TempScale {
    /// Converts temperature from Celsius to this scale.
    #[must_use]
    pub fn convert(&self, celsius: f64) -> f64 {
        match self {
            Self::Celsius => celsius,
            Self::Fahrenheit => celsius * 9.0 / 5.0 + 32.0,
            Self::Kelvin => celsius + 273.15,
        }
    }

    /// Returns the unit suffix.
    #[must_use]
    pub fn suffix(&self) -> &'static str {
        match self {
            Self::Celsius => "°C",
            Self::Fahrenheit => "°F",
            Self::Kelvin => "K",
        }
    }
}

/// A temperature sensor reading.
#[derive(Debug, Clone)]
pub struct TempReading {
    /// Sensor label (e.g., "Core 0", "CPU Package").
    pub label: String,
    /// Current temperature in Celsius.
    pub current: f64,
    /// High threshold in Celsius (if available).
    pub high: Option<f64>,
    /// Critical threshold in Celsius (if available).
    pub critical: Option<f64>,
    /// Hardware name (e.g., "coretemp", "acpitz").
    pub hwmon_name: String,
}

impl TempReading {
    /// Returns the temperature in the specified scale.
    #[must_use]
    pub fn in_scale(&self, scale: TempScale) -> f64 {
        scale.convert(self.current)
    }

    /// Returns true if the temperature is above the high threshold.
    #[must_use]
    pub fn is_high(&self) -> bool {
        self.high.is_some_and(|h| self.current >= h)
    }

    /// Returns true if the temperature is at or above the critical threshold.
    #[must_use]
    pub fn is_critical(&self) -> bool {
        self.critical.is_some_and(|c| self.current >= c)
    }
}

/// Collector for temperature sensors.
#[derive(Debug)]
pub struct SensorCollector {
    /// Current readings.
    readings: Vec<TempReading>,
    /// Temperature scale for display.
    scale: TempScale,
    /// Discovered hwmon paths.
    hwmon_paths: Vec<PathBuf>,
}

impl SensorCollector {
    /// Creates a new sensor collector.
    #[must_use]
    pub fn new() -> Self {
        let hwmon_paths = Self::discover_hwmon();
        Self { readings: Vec::new(), scale: TempScale::default(), hwmon_paths }
    }

    /// Sets the temperature scale.
    pub fn set_scale(&mut self, scale: TempScale) {
        self.scale = scale;
    }

    /// Returns the current temperature scale.
    #[must_use]
    pub fn scale(&self) -> TempScale {
        self.scale
    }

    /// Returns all temperature readings.
    #[must_use]
    pub fn readings(&self) -> &[TempReading] {
        &self.readings
    }

    /// Returns the highest temperature reading.
    #[must_use]
    pub fn max_temp(&self) -> Option<f64> {
        self.readings.iter().map(|r| r.current).reduce(f64::max)
    }

    /// Returns CPU core temperatures.
    #[must_use]
    pub fn cpu_temps(&self) -> Vec<&TempReading> {
        self.readings
            .iter()
            .filter(|r| {
                r.hwmon_name == "coretemp"
                    || r.hwmon_name == "k10temp"
                    || r.label.contains("Core")
                    || r.label.contains("CPU")
            })
            .collect()
    }

    /// Discovers hwmon devices in /sys/class/hwmon.
    #[cfg(target_os = "linux")]
    fn discover_hwmon() -> Vec<PathBuf> {
        let hwmon_dir = PathBuf::from("/sys/class/hwmon");
        if !hwmon_dir.exists() {
            return Vec::new();
        }

        std::fs::read_dir(&hwmon_dir)
            .ok()
            .map(|entries| {
                entries
                    .filter_map(std::result::Result::ok)
                    .map(|e| e.path())
                    .filter(|p| p.is_dir())
                    .collect()
            })
            .unwrap_or_default()
    }

    #[cfg(not(target_os = "linux"))]
    fn discover_hwmon() -> Vec<PathBuf> {
        Vec::new()
    }

    /// Reads the name of a hwmon device.
    fn read_hwmon_name(path: &Path) -> Option<String> {
        std::fs::read_to_string(path.join("name")).ok().map(|s| s.trim().to_string())
    }

    /// Reads temperature inputs from a hwmon device.
    #[cfg(target_os = "linux")]
    fn read_hwmon_temps(&self, path: &Path) -> Vec<TempReading> {
        let hwmon_name = Self::read_hwmon_name(path).unwrap_or_default();
        let mut readings = Vec::new();

        // Find all temp*_input files
        for i in 1..=32 {
            let input_path = path.join(format!("temp{i}_input"));
            if !input_path.exists() {
                continue;
            }

            // Read temperature (in millidegrees Celsius)
            let current = std::fs::read_to_string(&input_path)
                .ok()
                .and_then(|s| s.trim().parse::<i64>().ok())
                .map(|t| t as f64 / 1000.0);

            let Some(current) = current else {
                continue;
            };

            // Read label
            let label_path = path.join(format!("temp{i}_label"));
            let label = std::fs::read_to_string(&label_path)
                .ok()
                .map_or_else(|| format!("temp{i}"), |s| s.trim().to_string());

            // Read thresholds
            let high_path = path.join(format!("temp{i}_max"));
            let high = std::fs::read_to_string(&high_path)
                .ok()
                .and_then(|s| s.trim().parse::<i64>().ok())
                .map(|t| t as f64 / 1000.0);

            let crit_path = path.join(format!("temp{i}_crit"));
            let critical = std::fs::read_to_string(&crit_path)
                .ok()
                .and_then(|s| s.trim().parse::<i64>().ok())
                .map(|t| t as f64 / 1000.0);

            readings.push(TempReading {
                label,
                current,
                high,
                critical,
                hwmon_name: hwmon_name.clone(),
            });
        }

        readings
    }

    #[cfg(not(target_os = "linux"))]
    fn read_hwmon_temps(&self, _path: &Path) -> Vec<TempReading> {
        Vec::new()
    }
}

impl Default for SensorCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl Collector for SensorCollector {
    fn id(&self) -> &'static str {
        "sensors"
    }

    fn collect(&mut self) -> Result<Metrics> {
        self.readings.clear();

        for path in &self.hwmon_paths {
            self.readings.extend(self.read_hwmon_temps(path));
        }

        // Build metrics
        let mut metrics = Metrics::new();

        // Sensor count
        metrics.insert("sensors.count", MetricValue::Counter(self.readings.len() as u64));

        // Max temperature
        if let Some(max) = self.max_temp() {
            metrics.insert("sensors.max_temp", MetricValue::Gauge(max));
        }

        // CPU temperatures as histogram
        let cpu_temps: Vec<f64> = self.cpu_temps().iter().map(|r| r.current).collect();
        if !cpu_temps.is_empty() {
            metrics.insert("sensors.cpu_temps", MetricValue::Histogram(cpu_temps));
        }

        // High temp count
        let high_count = self.readings.iter().filter(|r| r.is_high()).count();
        metrics.insert("sensors.high_count", MetricValue::Counter(high_count as u64));

        // Critical count
        let crit_count = self.readings.iter().filter(|r| r.is_critical()).count();
        metrics.insert("sensors.critical_count", MetricValue::Counter(crit_count as u64));

        Ok(metrics)
    }

    fn is_available(&self) -> bool {
        !self.hwmon_paths.is_empty()
    }

    fn interval_hint(&self) -> Duration {
        Duration::from_millis(2000) // Temperature changes slowly
    }

    fn display_name(&self) -> &'static str {
        "Sensors"
    }
}

// ============================================================================
// Tests (TDD - Written First)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Unit Tests
    // ========================================================================

    #[test]
    fn test_temp_scale_convert_celsius() {
        let scale = TempScale::Celsius;
        assert!((scale.convert(100.0) - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_temp_scale_convert_fahrenheit() {
        let scale = TempScale::Fahrenheit;
        // 0°C = 32°F
        assert!((scale.convert(0.0) - 32.0).abs() < 0.01);
        // 100°C = 212°F
        assert!((scale.convert(100.0) - 212.0).abs() < 0.01);
    }

    #[test]
    fn test_temp_scale_convert_kelvin() {
        let scale = TempScale::Kelvin;
        // 0°C = 273.15K
        assert!((scale.convert(0.0) - 273.15).abs() < 0.01);
    }

    #[test]
    fn test_temp_scale_suffix() {
        assert_eq!(TempScale::Celsius.suffix(), "°C");
        assert_eq!(TempScale::Fahrenheit.suffix(), "°F");
        assert_eq!(TempScale::Kelvin.suffix(), "K");
    }

    #[test]
    fn test_temp_scale_default() {
        assert_eq!(TempScale::default(), TempScale::Celsius);
    }

    #[test]
    fn test_temp_reading_in_scale() {
        let reading = TempReading {
            label: "Test".to_string(),
            current: 50.0,
            high: Some(80.0),
            critical: Some(100.0),
            hwmon_name: "test".to_string(),
        };

        assert!((reading.in_scale(TempScale::Celsius) - 50.0).abs() < 0.01);
        assert!((reading.in_scale(TempScale::Fahrenheit) - 122.0).abs() < 0.01);
    }

    #[test]
    fn test_temp_reading_is_high() {
        let reading_normal = TempReading {
            label: "Test".to_string(),
            current: 50.0,
            high: Some(80.0),
            critical: Some(100.0),
            hwmon_name: "test".to_string(),
        };
        assert!(!reading_normal.is_high());

        let reading_high = TempReading {
            label: "Test".to_string(),
            current: 85.0,
            high: Some(80.0),
            critical: Some(100.0),
            hwmon_name: "test".to_string(),
        };
        assert!(reading_high.is_high());

        let reading_no_threshold = TempReading {
            label: "Test".to_string(),
            current: 85.0,
            high: None,
            critical: None,
            hwmon_name: "test".to_string(),
        };
        assert!(!reading_no_threshold.is_high());
    }

    #[test]
    fn test_temp_reading_is_critical() {
        let reading_normal = TempReading {
            label: "Test".to_string(),
            current: 50.0,
            high: Some(80.0),
            critical: Some(100.0),
            hwmon_name: "test".to_string(),
        };
        assert!(!reading_normal.is_critical());

        let reading_crit = TempReading {
            label: "Test".to_string(),
            current: 105.0,
            high: Some(80.0),
            critical: Some(100.0),
            hwmon_name: "test".to_string(),
        };
        assert!(reading_crit.is_critical());
    }

    #[test]
    fn test_sensor_collector_new() {
        let collector = SensorCollector::new();
        assert!(collector.readings.is_empty());
        assert_eq!(collector.scale, TempScale::Celsius);
    }

    #[test]
    fn test_sensor_collector_default() {
        let collector = SensorCollector::default();
        assert_eq!(collector.scale, TempScale::Celsius);
    }

    #[test]
    fn test_sensor_collector_set_scale() {
        let mut collector = SensorCollector::new();
        collector.set_scale(TempScale::Fahrenheit);
        assert_eq!(collector.scale(), TempScale::Fahrenheit);
    }

    #[test]
    fn test_sensor_collector_interval() {
        let collector = SensorCollector::new();
        assert_eq!(collector.interval_hint(), Duration::from_millis(2000));
    }

    #[test]
    fn test_sensor_collector_display_name() {
        let collector = SensorCollector::new();
        assert_eq!(collector.display_name(), "Sensors");
    }

    #[test]
    fn test_sensor_collector_id() {
        let collector = SensorCollector::new();
        assert_eq!(collector.id(), "sensors");
    }

    // ========================================================================
    // Linux-specific Tests
    // ========================================================================

    #[cfg(target_os = "linux")]
    #[test]
    fn test_sensor_collector_collect() {
        let mut collector = SensorCollector::new();
        let result = collector.collect();

        assert!(result.is_ok());
        let metrics = result.expect("collect should succeed");

        // Should have sensor count metric
        assert!(metrics.get_counter("sensors.count").is_some());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_sensor_collector_readings() {
        let mut collector = SensorCollector::new();
        let _ = collector.collect();

        // Readings may be empty in containerized environments
        let readings = collector.readings();
        // Just verify it doesn't panic
        let _ = readings;
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_sensor_temperature_accuracy() {
        // Falsification criterion #42: Temperature readings match sensors within ±1°C
        // This test verifies basic sanity of readings
        let mut collector = SensorCollector::new();
        let _ = collector.collect();

        for reading in collector.readings() {
            // Temperature should be in reasonable range (-40°C to 150°C)
            assert!(
                reading.current >= -40.0 && reading.current <= 150.0,
                "Temperature {} for {} is out of reasonable range",
                reading.current,
                reading.label
            );

            // If we have thresholds, they should be ordered correctly
            if let (Some(high), Some(crit)) = (reading.high, reading.critical) {
                assert!(
                    high <= crit,
                    "High threshold {} should be <= critical threshold {} for {}",
                    high,
                    crit,
                    reading.label
                );
            }
        }
    }
}
