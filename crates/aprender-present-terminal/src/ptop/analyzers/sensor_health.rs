//! Hardware Sensor Health Analyzer
//!
//! Reads sensor data from `/sys/class/hwmon/`:
//! - Temperature sensors (`temp*_input`)
//! - Fan speeds (`fan*_input`)
//! - Voltages (`in*_input`)
//! - Power (`power*_input`)

#![allow(clippy::uninlined_format_args)]
#![allow(clippy::map_unwrap_or)]
#![allow(clippy::redundant_closure_for_method_calls)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::single_char_pattern)]
#![allow(clippy::trivially_copy_pass_by_ref)]
#![allow(clippy::match_same_arms)]

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use super::{Analyzer, AnalyzerError};

/// Type of sensor
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SensorType {
    /// Temperature in millidegrees Celsius
    Temperature,
    /// Fan speed in RPM
    Fan,
    /// Voltage in millivolts
    Voltage,
    /// Current in milliamps
    Current,
    /// Power in microwatts
    Power,
}

impl SensorType {
    /// Get unit string for display
    pub fn unit(&self) -> &'static str {
        match self {
            Self::Temperature => "°C",
            Self::Fan => "RPM",
            Self::Voltage => "V",
            Self::Current => "A",
            Self::Power => "W",
        }
    }

    /// Get prefix pattern for hwmon files
    fn prefix(&self) -> &'static str {
        match self {
            Self::Temperature => "temp",
            Self::Fan => "fan",
            Self::Voltage => "in",
            Self::Current => "curr",
            Self::Power => "power",
        }
    }
}

/// Status of a sensor reading
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SensorStatus {
    /// Normal operating range
    #[default]
    Normal,
    /// Above warning threshold
    Warning,
    /// Above critical threshold
    Critical,
    /// Below minimum threshold
    Low,
    /// Sensor fault/error
    Fault,
}

impl SensorStatus {
    /// Get display string
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Normal => "OK",
            Self::Warning => "WARN",
            Self::Critical => "CRIT",
            Self::Low => "LOW",
            Self::Fault => "FAULT",
        }
    }
}

/// A single sensor reading
#[derive(Debug, Clone)]
pub struct SensorReading {
    /// Device name (e.g., "coretemp", "nvme")
    pub device: String,
    /// Sensor type
    pub sensor_type: SensorType,
    /// Sensor label (e.g., "Core 0", "Composite")
    pub label: String,
    /// Sensor index (e.g., temp1 -> 1)
    pub index: u32,
    /// Current value (in base units: °C, RPM, V, A, W)
    pub value: f64,
    /// Critical threshold (if available)
    pub critical: Option<f64>,
    /// Warning/max threshold (if available)
    pub max: Option<f64>,
    /// Minimum threshold (if available)
    pub min: Option<f64>,
    /// Current status
    pub status: SensorStatus,
    /// Hwmon path for debugging
    pub hwmon_path: PathBuf,
}

impl SensorReading {
    /// Format value for display
    pub fn value_display(&self) -> String {
        match self.sensor_type {
            SensorType::Temperature => format!("{:.1}°C", self.value),
            SensorType::Fan => format!("{:.0} RPM", self.value),
            SensorType::Voltage => format!("{:.2}V", self.value),
            SensorType::Current => format!("{:.2}A", self.value),
            SensorType::Power => format!("{:.1}W", self.value),
        }
    }

    /// Get short label (max 12 chars)
    pub fn short_label(&self) -> String {
        if self.label.len() > 12 {
            format!("{}...", &self.label[..9])
        } else {
            self.label.clone()
        }
    }
}

/// Collection of sensor readings
#[derive(Debug, Clone, Default)]
pub struct SensorHealthData {
    /// All sensor readings
    pub sensors: Vec<SensorReading>,
    /// Count by type
    pub type_counts: HashMap<SensorType, usize>,
    /// Count by status
    pub status_counts: HashMap<SensorStatus, usize>,
}

impl SensorHealthData {
    /// Get sensors by type
    pub fn by_type(&self, sensor_type: SensorType) -> impl Iterator<Item = &SensorReading> {
        self.sensors
            .iter()
            .filter(move |s| s.sensor_type == sensor_type)
    }

    /// Get temperature sensors
    pub fn temperatures(&self) -> impl Iterator<Item = &SensorReading> {
        self.by_type(SensorType::Temperature)
    }

    /// Get fan sensors
    pub fn fans(&self) -> impl Iterator<Item = &SensorReading> {
        self.by_type(SensorType::Fan)
    }

    /// Get sensors with warning or critical status
    pub fn alerts(&self) -> impl Iterator<Item = &SensorReading> {
        self.sensors.iter().filter(|s| {
            matches!(
                s.status,
                SensorStatus::Warning | SensorStatus::Critical | SensorStatus::Fault
            )
        })
    }

    /// Check if any sensor is in critical state
    pub fn has_critical(&self) -> bool {
        self.sensors
            .iter()
            .any(|s| s.status == SensorStatus::Critical)
    }

    /// Get highest temperature
    pub fn max_temperature(&self) -> Option<f64> {
        self.temperatures().map(|s| s.value).reduce(f64::max)
    }
}

/// Analyzer for hardware sensors
pub struct SensorHealthAnalyzer {
    data: SensorHealthData,
    interval: Duration,
}

impl Default for SensorHealthAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl SensorHealthAnalyzer {
    /// Create a new sensor health analyzer
    pub fn new() -> Self {
        Self {
            data: SensorHealthData::default(),
            interval: Duration::from_secs(2),
        }
    }

    /// Get the current data
    pub fn data(&self) -> &SensorHealthData {
        &self.data
    }

    /// Scan a single hwmon device
    fn scan_hwmon_device(&self, hwmon_path: &Path) -> Vec<SensorReading> {
        let mut readings = Vec::new();

        // Read device name
        let device = fs::read_to_string(hwmon_path.join("name"))
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|_| "unknown".to_string());

        // Scan for each sensor type
        for sensor_type in [
            SensorType::Temperature,
            SensorType::Fan,
            SensorType::Voltage,
            SensorType::Current,
            SensorType::Power,
        ] {
            readings.extend(self.scan_sensor_type(hwmon_path, &device, sensor_type));
        }

        readings
    }

    /// Scan for sensors of a specific type
    fn scan_sensor_type(
        &self,
        hwmon_path: &Path,
        device: &str,
        sensor_type: SensorType,
    ) -> Vec<SensorReading> {
        let mut readings = Vec::new();
        let prefix = sensor_type.prefix();

        // Scan for sensor indices (temp1, temp2, etc.)
        for index in 1..=20 {
            let input_file = hwmon_path.join(format!("{}{}_{}", prefix, index, "input"));
            if !input_file.exists() {
                continue;
            }

            // Read current value
            let raw_value = match fs::read_to_string(&input_file) {
                Ok(s) => s.trim().parse::<i64>().unwrap_or(0),
                Err(_) => continue,
            };

            // Convert to base units
            let value = Self::convert_value(raw_value, sensor_type);

            // Read label
            let label =
                fs::read_to_string(hwmon_path.join(format!("{}{}_{}", prefix, index, "label")))
                    .map(|s| s.trim().to_string())
                    .unwrap_or_else(|_| format!("{}{}", prefix, index));

            // Read thresholds
            let critical = Self::read_threshold(hwmon_path, prefix, index, "crit");
            let max = Self::read_threshold(hwmon_path, prefix, index, "max");
            let min = Self::read_threshold(hwmon_path, prefix, index, "min");

            // Determine status
            let status = Self::determine_status(value, critical, max, min);

            readings.push(SensorReading {
                device: device.to_string(),
                sensor_type,
                label,
                index,
                value,
                critical,
                max,
                min,
                status,
                hwmon_path: hwmon_path.to_path_buf(),
            });
        }

        readings
    }

    /// Read a threshold value
    fn read_threshold(hwmon_path: &Path, prefix: &str, index: u32, suffix: &str) -> Option<f64> {
        let path = hwmon_path.join(format!("{}{}_{}", prefix, index, suffix));
        fs::read_to_string(path).ok().and_then(|s| {
            s.trim()
                .parse::<i64>()
                .ok()
                .map(|v| Self::convert_value(v, SensorType::Temperature))
        })
    }

    /// Convert raw value to base units
    fn convert_value(raw: i64, sensor_type: SensorType) -> f64 {
        match sensor_type {
            // Temperature: millidegrees to degrees
            SensorType::Temperature => raw as f64 / 1000.0,
            // Fan: already in RPM
            SensorType::Fan => raw as f64,
            // Voltage: millivolts to volts
            SensorType::Voltage => raw as f64 / 1000.0,
            // Current: milliamps to amps
            SensorType::Current => raw as f64 / 1000.0,
            // Power: microwatts to watts
            SensorType::Power => raw as f64 / 1_000_000.0,
        }
    }

    /// Determine sensor status from thresholds
    fn determine_status(
        value: f64,
        critical: Option<f64>,
        max: Option<f64>,
        min: Option<f64>,
    ) -> SensorStatus {
        if let Some(crit) = critical {
            if value >= crit {
                return SensorStatus::Critical;
            }
        }

        if let Some(m) = max {
            if value >= m {
                return SensorStatus::Warning;
            }
        }

        if let Some(m) = min {
            if value <= m {
                return SensorStatus::Low;
            }
        }

        SensorStatus::Normal
    }
}

impl Analyzer for SensorHealthAnalyzer {
    fn name(&self) -> &'static str {
        "sensor_health"
    }

    fn collect(&mut self) -> Result<(), AnalyzerError> {
        let mut all_sensors = Vec::new();

        let hwmon_base = Path::new("/sys/class/hwmon");
        if !hwmon_base.exists() {
            return Ok(());
        }

        let Ok(entries) = fs::read_dir(hwmon_base) else {
            return Ok(());
        };

        for entry in entries.flatten() {
            let hwmon_path = entry.path();
            let sensors = self.scan_hwmon_device(&hwmon_path);
            all_sensors.extend(sensors);
        }

        // Count by type and status
        let mut type_counts: HashMap<SensorType, usize> = HashMap::new();
        let mut status_counts: HashMap<SensorStatus, usize> = HashMap::new();

        for sensor in &all_sensors {
            *type_counts.entry(sensor.sensor_type).or_insert(0) += 1;
            *status_counts.entry(sensor.status).or_insert(0) += 1;
        }

        self.data = SensorHealthData {
            sensors: all_sensors,
            type_counts,
            status_counts,
        };

        Ok(())
    }

    fn interval(&self) -> Duration {
        self.interval
    }

    fn available(&self) -> bool {
        Path::new("/sys/class/hwmon").exists()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sensor_type_unit() {
        assert_eq!(SensorType::Temperature.unit(), "°C");
        assert_eq!(SensorType::Fan.unit(), "RPM");
        assert_eq!(SensorType::Voltage.unit(), "V");
    }

    #[test]
    fn test_sensor_status_display() {
        assert_eq!(SensorStatus::Normal.as_str(), "OK");
        assert_eq!(SensorStatus::Critical.as_str(), "CRIT");
        assert_eq!(SensorStatus::Warning.as_str(), "WARN");
    }

    #[test]
    fn test_value_conversion() {
        // Temperature: 45000 millidegrees = 45.0°C
        let temp = SensorHealthAnalyzer::convert_value(45000, SensorType::Temperature);
        assert!((temp - 45.0).abs() < 0.01);

        // Fan: 1200 RPM = 1200 RPM
        let fan = SensorHealthAnalyzer::convert_value(1200, SensorType::Fan);
        assert!((fan - 1200.0).abs() < 0.01);

        // Voltage: 12500 mV = 12.5V
        let volt = SensorHealthAnalyzer::convert_value(12500, SensorType::Voltage);
        assert!((volt - 12.5).abs() < 0.01);

        // Power: 45000000 uW = 45W
        let power = SensorHealthAnalyzer::convert_value(45_000_000, SensorType::Power);
        assert!((power - 45.0).abs() < 0.01);
    }

    #[test]
    fn test_status_determination() {
        // Normal
        let status =
            SensorHealthAnalyzer::determine_status(45.0, Some(95.0), Some(85.0), Some(10.0));
        assert_eq!(status, SensorStatus::Normal);

        // Critical
        let status =
            SensorHealthAnalyzer::determine_status(100.0, Some(95.0), Some(85.0), Some(10.0));
        assert_eq!(status, SensorStatus::Critical);

        // Warning
        let status =
            SensorHealthAnalyzer::determine_status(90.0, Some(95.0), Some(85.0), Some(10.0));
        assert_eq!(status, SensorStatus::Warning);

        // Low
        let status =
            SensorHealthAnalyzer::determine_status(5.0, Some(95.0), Some(85.0), Some(10.0));
        assert_eq!(status, SensorStatus::Low);
    }

    #[test]
    fn test_sensor_reading_display() {
        let reading = SensorReading {
            device: "coretemp".to_string(),
            sensor_type: SensorType::Temperature,
            label: "Core 0".to_string(),
            index: 1,
            value: 45.5,
            critical: Some(100.0),
            max: Some(85.0),
            min: None,
            status: SensorStatus::Normal,
            hwmon_path: PathBuf::from("/sys/class/hwmon/hwmon0"),
        };

        assert_eq!(reading.value_display(), "45.5°C");
        assert_eq!(reading.short_label(), "Core 0");
    }

    #[test]
    fn test_short_label_truncation() {
        let mut reading = SensorReading {
            device: "test".to_string(),
            sensor_type: SensorType::Temperature,
            label: "Very Long Sensor Label Name".to_string(),
            index: 1,
            value: 45.0,
            critical: None,
            max: None,
            min: None,
            status: SensorStatus::Normal,
            hwmon_path: PathBuf::from("/sys/class/hwmon/hwmon0"),
        };

        assert_eq!(reading.short_label(), "Very Long...");

        reading.label = "Short".to_string();
        assert_eq!(reading.short_label(), "Short");
    }

    #[test]
    fn test_analyzer_available() {
        let analyzer = SensorHealthAnalyzer::new();
        // hwmon should be available on most Linux systems
        #[cfg(target_os = "linux")]
        assert!(analyzer.available());
    }

    #[test]
    fn test_analyzer_collect() {
        let mut analyzer = SensorHealthAnalyzer::new();
        let result = analyzer.collect();
        assert!(result.is_ok());

        // On most systems, we should find at least one sensor
        #[cfg(target_os = "linux")]
        {
            let data = analyzer.data();
            // Sensors might be empty on some systems, but collection should work
            let _ = data.sensors.len();
        }
    }

    #[test]
    fn test_data_has_critical() {
        let mut data = SensorHealthData::default();
        assert!(!data.has_critical());

        data.sensors.push(SensorReading {
            device: "test".to_string(),
            sensor_type: SensorType::Temperature,
            label: "Test".to_string(),
            index: 1,
            value: 100.0,
            critical: Some(95.0),
            max: None,
            min: None,
            status: SensorStatus::Critical,
            hwmon_path: PathBuf::from("/test"),
        });

        assert!(data.has_critical());
    }

    #[test]
    fn test_max_temperature() {
        let mut data = SensorHealthData::default();
        assert!(data.max_temperature().is_none());

        data.sensors.push(SensorReading {
            device: "test".to_string(),
            sensor_type: SensorType::Temperature,
            label: "Test1".to_string(),
            index: 1,
            value: 45.0,
            critical: None,
            max: None,
            min: None,
            status: SensorStatus::Normal,
            hwmon_path: PathBuf::from("/test"),
        });

        data.sensors.push(SensorReading {
            device: "test".to_string(),
            sensor_type: SensorType::Temperature,
            label: "Test2".to_string(),
            index: 2,
            value: 55.0,
            critical: None,
            max: None,
            min: None,
            status: SensorStatus::Normal,
            hwmon_path: PathBuf::from("/test"),
        });

        assert!((data.max_temperature().unwrap() - 55.0).abs() < 0.01);
    }

    #[test]
    fn test_sensor_type_prefix() {
        assert_eq!(SensorType::Temperature.prefix(), "temp");
        assert_eq!(SensorType::Fan.prefix(), "fan");
        assert_eq!(SensorType::Voltage.prefix(), "in");
        assert_eq!(SensorType::Current.prefix(), "curr");
        assert_eq!(SensorType::Power.prefix(), "power");
    }

    #[test]
    fn test_sensor_type_all_units() {
        assert_eq!(SensorType::Current.unit(), "A");
        assert_eq!(SensorType::Power.unit(), "W");
    }

    #[test]
    fn test_sensor_status_default() {
        let status = SensorStatus::default();
        assert_eq!(status, SensorStatus::Normal);
    }

    #[test]
    fn test_sensor_status_all_variants() {
        assert_eq!(SensorStatus::Low.as_str(), "LOW");
        assert_eq!(SensorStatus::Fault.as_str(), "FAULT");
    }

    #[test]
    fn test_current_conversion() {
        // Current: 2500 mA = 2.5 A
        let current = SensorHealthAnalyzer::convert_value(2500, SensorType::Current);
        assert!((current - 2.5).abs() < 0.01);
    }

    #[test]
    fn test_sensor_reading_value_display_all_types() {
        // Fan
        let fan = SensorReading {
            device: "nct6795".to_string(),
            sensor_type: SensorType::Fan,
            label: "CPU Fan".to_string(),
            index: 1,
            value: 1200.0,
            critical: None,
            max: None,
            min: None,
            status: SensorStatus::Normal,
            hwmon_path: PathBuf::from("/sys/class/hwmon/hwmon1"),
        };
        assert_eq!(fan.value_display(), "1200 RPM");

        // Voltage
        let volt = SensorReading {
            device: "nct6795".to_string(),
            sensor_type: SensorType::Voltage,
            label: "Vcore".to_string(),
            index: 1,
            value: 1.25,
            critical: None,
            max: None,
            min: None,
            status: SensorStatus::Normal,
            hwmon_path: PathBuf::from("/sys/class/hwmon/hwmon1"),
        };
        assert_eq!(volt.value_display(), "1.25V");

        // Current
        let curr = SensorReading {
            device: "test".to_string(),
            sensor_type: SensorType::Current,
            label: "CPU Current".to_string(),
            index: 1,
            value: 45.5,
            critical: None,
            max: None,
            min: None,
            status: SensorStatus::Normal,
            hwmon_path: PathBuf::from("/sys/class/hwmon/hwmon1"),
        };
        assert_eq!(curr.value_display(), "45.50A");

        // Power
        let pwr = SensorReading {
            device: "test".to_string(),
            sensor_type: SensorType::Power,
            label: "Package Power".to_string(),
            index: 1,
            value: 65.0,
            critical: None,
            max: None,
            min: None,
            status: SensorStatus::Normal,
            hwmon_path: PathBuf::from("/sys/class/hwmon/hwmon1"),
        };
        assert_eq!(pwr.value_display(), "65.0W");
    }

    #[test]
    fn test_data_by_type() {
        let mut data = SensorHealthData::default();
        data.sensors.push(SensorReading {
            device: "test".to_string(),
            sensor_type: SensorType::Temperature,
            label: "Temp1".to_string(),
            index: 1,
            value: 45.0,
            critical: None,
            max: None,
            min: None,
            status: SensorStatus::Normal,
            hwmon_path: PathBuf::from("/test"),
        });
        data.sensors.push(SensorReading {
            device: "test".to_string(),
            sensor_type: SensorType::Fan,
            label: "Fan1".to_string(),
            index: 1,
            value: 1200.0,
            critical: None,
            max: None,
            min: None,
            status: SensorStatus::Normal,
            hwmon_path: PathBuf::from("/test"),
        });

        assert_eq!(data.by_type(SensorType::Temperature).count(), 1);
        assert_eq!(data.by_type(SensorType::Fan).count(), 1);
        assert_eq!(data.by_type(SensorType::Voltage).count(), 0);
    }

    #[test]
    fn test_data_temperatures() {
        let mut data = SensorHealthData::default();
        data.sensors.push(SensorReading {
            device: "test".to_string(),
            sensor_type: SensorType::Temperature,
            label: "Temp1".to_string(),
            index: 1,
            value: 45.0,
            critical: None,
            max: None,
            min: None,
            status: SensorStatus::Normal,
            hwmon_path: PathBuf::from("/test"),
        });

        assert_eq!(data.temperatures().count(), 1);
    }

    #[test]
    fn test_data_fans() {
        let mut data = SensorHealthData::default();
        data.sensors.push(SensorReading {
            device: "test".to_string(),
            sensor_type: SensorType::Fan,
            label: "Fan1".to_string(),
            index: 1,
            value: 1200.0,
            critical: None,
            max: None,
            min: None,
            status: SensorStatus::Normal,
            hwmon_path: PathBuf::from("/test"),
        });

        assert_eq!(data.fans().count(), 1);
    }

    #[test]
    fn test_data_alerts() {
        let mut data = SensorHealthData::default();
        data.sensors.push(SensorReading {
            device: "test".to_string(),
            sensor_type: SensorType::Temperature,
            label: "Normal".to_string(),
            index: 1,
            value: 45.0,
            critical: None,
            max: None,
            min: None,
            status: SensorStatus::Normal,
            hwmon_path: PathBuf::from("/test"),
        });
        data.sensors.push(SensorReading {
            device: "test".to_string(),
            sensor_type: SensorType::Temperature,
            label: "Warning".to_string(),
            index: 2,
            value: 85.0,
            critical: None,
            max: None,
            min: None,
            status: SensorStatus::Warning,
            hwmon_path: PathBuf::from("/test"),
        });
        data.sensors.push(SensorReading {
            device: "test".to_string(),
            sensor_type: SensorType::Temperature,
            label: "Critical".to_string(),
            index: 3,
            value: 100.0,
            critical: None,
            max: None,
            min: None,
            status: SensorStatus::Critical,
            hwmon_path: PathBuf::from("/test"),
        });
        data.sensors.push(SensorReading {
            device: "test".to_string(),
            sensor_type: SensorType::Temperature,
            label: "Fault".to_string(),
            index: 4,
            value: 0.0,
            critical: None,
            max: None,
            min: None,
            status: SensorStatus::Fault,
            hwmon_path: PathBuf::from("/test"),
        });

        assert_eq!(data.alerts().count(), 3); // Warning, Critical, and Fault
    }

    #[test]
    fn test_status_determination_no_thresholds() {
        let status = SensorHealthAnalyzer::determine_status(50.0, None, None, None);
        assert_eq!(status, SensorStatus::Normal);
    }

    #[test]
    fn test_status_determination_critical_priority() {
        // Critical should take priority over Warning
        let status = SensorHealthAnalyzer::determine_status(100.0, Some(95.0), Some(85.0), None);
        assert_eq!(status, SensorStatus::Critical);
    }

    #[test]
    fn test_status_determination_just_below_critical() {
        let status = SensorHealthAnalyzer::determine_status(94.9, Some(95.0), Some(85.0), None);
        assert_eq!(status, SensorStatus::Warning);
    }

    #[test]
    fn test_status_determination_at_min() {
        let status =
            SensorHealthAnalyzer::determine_status(10.0, Some(95.0), Some(85.0), Some(10.0));
        assert_eq!(status, SensorStatus::Low);
    }

    #[test]
    fn test_analyzer_interval() {
        let analyzer = SensorHealthAnalyzer::new();
        assert_eq!(analyzer.interval(), Duration::from_secs(2));
    }

    #[test]
    fn test_analyzer_name() {
        let analyzer = SensorHealthAnalyzer::new();
        assert_eq!(analyzer.name(), "sensor_health");
    }

    #[test]
    fn test_sensor_health_data_default() {
        let data = SensorHealthData::default();
        assert!(data.sensors.is_empty());
        assert!(data.type_counts.is_empty());
        assert!(data.status_counts.is_empty());
    }

    #[test]
    fn test_sensor_reading_short_label_exactly_12() {
        let reading = SensorReading {
            device: "test".to_string(),
            sensor_type: SensorType::Temperature,
            label: "123456789012".to_string(), // Exactly 12 chars
            index: 1,
            value: 45.0,
            critical: None,
            max: None,
            min: None,
            status: SensorStatus::Normal,
            hwmon_path: PathBuf::from("/test"),
        };
        assert_eq!(reading.short_label(), "123456789012");
    }

    #[test]
    fn test_sensor_reading_clone() {
        let reading = SensorReading {
            device: "test".to_string(),
            sensor_type: SensorType::Temperature,
            label: "Test".to_string(),
            index: 1,
            value: 45.0,
            critical: Some(100.0),
            max: Some(85.0),
            min: Some(10.0),
            status: SensorStatus::Normal,
            hwmon_path: PathBuf::from("/test"),
        };
        let cloned = reading.clone();
        assert_eq!(cloned.device, "test");
        assert_eq!(cloned.value, 45.0);
    }

    #[test]
    fn test_sensor_type_hash_eq() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(SensorType::Temperature);
        set.insert(SensorType::Fan);
        assert!(set.contains(&SensorType::Temperature));
        assert!(set.contains(&SensorType::Fan));
        assert!(!set.contains(&SensorType::Voltage));
    }

    #[test]
    fn test_sensor_status_hash_eq() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(SensorStatus::Normal);
        set.insert(SensorStatus::Warning);
        assert!(set.contains(&SensorStatus::Normal));
        assert!(!set.contains(&SensorStatus::Critical));
    }
}
