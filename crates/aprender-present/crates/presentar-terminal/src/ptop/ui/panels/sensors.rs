//! Sensors panel rendering and utilities.
//!
//! Provides sensor panel title building, temperature formatting,
//! and helper functions for rendering thermal and power metrics.

use presentar_core::Color;

// =============================================================================
// SENSORS TITLE BUILDING
// =============================================================================

/// Build sensors panel title string.
///
/// Format: "Sensors │ CPU: 45°C │ GPU: 52°C │ 65W"
#[must_use]
pub fn build_sensors_title(
    cpu_temp: Option<f64>,
    gpu_temp: Option<f64>,
    power_watts: Option<f64>,
) -> String {
    let mut parts = vec!["Sensors".to_string()];

    if let Some(t) = cpu_temp {
        parts.push(format!("CPU: {}°C", t.round() as i32));
    }

    if let Some(t) = gpu_temp {
        parts.push(format!("GPU: {}°C", t.round() as i32));
    }

    if let Some(w) = power_watts {
        parts.push(format!("{}W", w.round() as i32));
    }

    parts.join(" │ ")
}

/// Build compact sensors title for narrow panels.
///
/// Format: "Sensors │ 45°C"
#[must_use]
pub fn build_sensors_title_compact(max_temp: Option<f64>) -> String {
    if let Some(t) = max_temp {
        format!("Sensors │ {}°C", t.round() as i32)
    } else {
        "Sensors │ --".to_string()
    }
}

// =============================================================================
// TEMPERATURE FORMATTING
// =============================================================================

/// Format temperature in Celsius.
#[must_use]
pub fn format_temp_celsius(temp: f64) -> String {
    format!("{}°C", temp.round() as i32)
}

/// Format temperature in Fahrenheit.
#[must_use]
pub fn format_temp_fahrenheit(celsius: f64) -> String {
    let fahrenheit = celsius * 9.0 / 5.0 + 32.0;
    format!("{}°F", fahrenheit.round() as i32)
}

/// Format temperature with unit preference.
#[must_use]
pub fn format_temp(celsius: f64, use_fahrenheit: bool) -> String {
    if use_fahrenheit {
        format_temp_fahrenheit(celsius)
    } else {
        format_temp_celsius(celsius)
    }
}

// =============================================================================
// TEMPERATURE COLORS
// =============================================================================

/// Temperature thresholds for color coding.
#[derive(Debug, Clone, PartialEq)]
pub struct TempThresholds {
    /// Temperature at which warning color starts
    pub warning: f64,
    /// Temperature at which critical color starts
    pub critical: f64,
    /// Maximum safe temperature
    pub max: f64,
}

impl TempThresholds {
    /// Standard CPU temperature thresholds.
    #[must_use]
    pub fn cpu() -> Self {
        Self {
            warning: 70.0,
            critical: 85.0,
            max: 100.0,
        }
    }

    /// Standard GPU temperature thresholds.
    #[must_use]
    pub fn gpu() -> Self {
        Self {
            warning: 75.0,
            critical: 90.0,
            max: 110.0,
        }
    }

    /// Standard NVMe/SSD temperature thresholds.
    #[must_use]
    pub fn nvme() -> Self {
        Self {
            warning: 60.0,
            critical: 70.0,
            max: 85.0,
        }
    }

    /// Get color for temperature using these thresholds.
    #[must_use]
    pub fn color(&self, temp: f64) -> Color {
        if temp >= self.critical {
            Color::new(1.0, 0.3, 0.3, 1.0) // Critical red
        } else if temp >= self.warning {
            Color::new(1.0, 0.7, 0.2, 1.0) // Warning orange
        } else {
            Color::new(0.3, 0.9, 0.5, 1.0) // Normal green
        }
    }
}

impl Default for TempThresholds {
    fn default() -> Self {
        Self::cpu()
    }
}

/// Get color for CPU temperature.
#[must_use]
pub fn cpu_temp_color(celsius: f64) -> Color {
    TempThresholds::cpu().color(celsius)
}

/// Get color for GPU temperature.
#[must_use]
pub fn gpu_temp_color(celsius: f64) -> Color {
    TempThresholds::gpu().color(celsius)
}

/// Get color for generic temperature (uses CPU thresholds).
#[must_use]
pub fn temp_color(celsius: f64) -> Color {
    TempThresholds::cpu().color(celsius)
}

// =============================================================================
// FAN SPEED
// =============================================================================

/// Format fan speed in RPM.
#[must_use]
pub fn format_fan_speed(rpm: u32) -> String {
    if rpm == 0 {
        "Off".to_string()
    } else if rpm >= 1000 {
        format!("{:.1}K RPM", rpm as f64 / 1000.0)
    } else {
        format!("{} RPM", rpm)
    }
}

/// Get color for fan speed.
#[must_use]
pub fn fan_speed_color(rpm: u32, max_rpm: u32) -> Color {
    if max_rpm == 0 {
        return Color::new(0.5, 0.5, 0.5, 1.0);
    }

    let percent = (rpm as f64 / max_rpm as f64) * 100.0;

    if percent > 90.0 {
        Color::new(1.0, 0.4, 0.4, 1.0) // Very high - red
    } else if percent > 70.0 {
        Color::new(1.0, 0.7, 0.3, 1.0) // High - orange
    } else if percent > 40.0 {
        Color::new(0.5, 0.8, 0.5, 1.0) // Normal - green
    } else if rpm > 0 {
        Color::new(0.4, 0.6, 0.8, 1.0) // Low - blue
    } else {
        Color::new(0.4, 0.4, 0.4, 1.0) // Off - gray
    }
}

// =============================================================================
// POWER
// =============================================================================

/// Format power consumption in watts.
#[must_use]
pub fn format_power(watts: f64) -> String {
    if watts >= 1000.0 {
        format!("{:.2}kW", watts / 1000.0)
    } else if watts >= 100.0 {
        format!("{:.0}W", watts)
    } else if watts >= 10.0 {
        format!("{:.1}W", watts)
    } else {
        format!("{:.2}W", watts)
    }
}

/// Get color for power consumption.
#[must_use]
pub fn power_color(watts: f64, tdp: f64) -> Color {
    if tdp <= 0.0 {
        return Color::new(0.5, 0.5, 0.5, 1.0);
    }

    let percent = (watts / tdp) * 100.0;

    if percent > 100.0 {
        Color::new(1.0, 0.3, 0.3, 1.0) // Over TDP - red
    } else if percent > 90.0 {
        Color::new(1.0, 0.6, 0.2, 1.0) // Near TDP - orange
    } else if percent > 70.0 {
        Color::new(1.0, 0.8, 0.2, 1.0) // High - yellow
    } else {
        Color::new(0.3, 0.8, 0.5, 1.0) // Normal - green
    }
}

// =============================================================================
// SENSOR TYPE
// =============================================================================

/// Sensor type for categorization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SensorType {
    /// CPU core temperature
    CpuCore,
    /// CPU package temperature
    CpuPackage,
    /// GPU temperature
    Gpu,
    /// NVMe/SSD temperature
    Storage,
    /// Motherboard/ambient temperature
    Ambient,
    /// Fan speed sensor
    Fan,
    /// Power consumption sensor
    Power,
    /// Voltage sensor
    Voltage,
}

impl SensorType {
    /// Get display icon for sensor type.
    #[must_use]
    pub fn icon(&self) -> &'static str {
        match self {
            Self::CpuCore | Self::CpuPackage => "󰻠",
            Self::Gpu => "󰢮",
            Self::Storage => "󰋊",
            Self::Ambient => "󱃂",
            Self::Fan => "󰈐",
            Self::Power => "󰚥",
            Self::Voltage => "󰚦",
        }
    }

    /// Get unit for sensor type.
    #[must_use]
    pub fn unit(&self) -> &'static str {
        match self {
            Self::CpuCore | Self::CpuPackage | Self::Gpu | Self::Storage | Self::Ambient => "°C",
            Self::Fan => "RPM",
            Self::Power => "W",
            Self::Voltage => "V",
        }
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // build_sensors_title tests
    // =========================================================================

    #[test]
    fn test_build_sensors_title_full() {
        let title = build_sensors_title(Some(45.0), Some(52.0), Some(65.0));
        assert!(title.contains("Sensors"));
        assert!(title.contains("CPU: 45°C"));
        assert!(title.contains("GPU: 52°C"));
        assert!(title.contains("65W"));
    }

    #[test]
    fn test_build_sensors_title_cpu_only() {
        let title = build_sensors_title(Some(45.0), None, None);
        assert!(title.contains("CPU: 45°C"));
        assert!(!title.contains("GPU"));
    }

    #[test]
    fn test_build_sensors_title_empty() {
        let title = build_sensors_title(None, None, None);
        assert_eq!(title, "Sensors");
    }

    #[test]
    fn test_build_sensors_title_rounds() {
        let title = build_sensors_title(Some(45.6), None, Some(65.4));
        assert!(title.contains("46°C")); // Rounded up
        assert!(title.contains("65W")); // Rounded down
    }

    // =========================================================================
    // build_sensors_title_compact tests
    // =========================================================================

    #[test]
    fn test_build_sensors_title_compact_with_temp() {
        let title = build_sensors_title_compact(Some(45.0));
        assert!(title.contains("Sensors"));
        assert!(title.contains("45°C"));
    }

    #[test]
    fn test_build_sensors_title_compact_no_temp() {
        let title = build_sensors_title_compact(None);
        assert!(title.contains("--"));
    }

    // =========================================================================
    // format_temp tests
    // =========================================================================

    #[test]
    fn test_format_temp_celsius() {
        assert_eq!(format_temp_celsius(45.0), "45°C");
        assert_eq!(format_temp_celsius(45.6), "46°C"); // Rounded
    }

    #[test]
    fn test_format_temp_fahrenheit() {
        assert_eq!(format_temp_fahrenheit(0.0), "32°F");
        assert_eq!(format_temp_fahrenheit(100.0), "212°F");
    }

    #[test]
    fn test_format_temp_preference() {
        assert_eq!(format_temp(45.0, false), "45°C");
        assert!(format_temp(45.0, true).contains("°F"));
    }

    // =========================================================================
    // TempThresholds tests
    // =========================================================================

    #[test]
    fn test_temp_thresholds_cpu() {
        let thresh = TempThresholds::cpu();
        assert_eq!(thresh.warning, 70.0);
        assert_eq!(thresh.critical, 85.0);
    }

    #[test]
    fn test_temp_thresholds_gpu() {
        let thresh = TempThresholds::gpu();
        assert!(thresh.warning > TempThresholds::cpu().warning);
    }

    #[test]
    fn test_temp_thresholds_nvme() {
        let thresh = TempThresholds::nvme();
        assert!(thresh.warning < TempThresholds::cpu().warning);
    }

    #[test]
    fn test_temp_thresholds_color_normal() {
        let thresh = TempThresholds::cpu();
        let color = thresh.color(50.0);
        assert!(color.g > 0.8, "Normal temp should be green");
    }

    #[test]
    fn test_temp_thresholds_color_warning() {
        let thresh = TempThresholds::cpu();
        let color = thresh.color(75.0);
        assert!(color.r > 0.9 && color.g > 0.6, "Warning should be orange");
    }

    #[test]
    fn test_temp_thresholds_color_critical() {
        let thresh = TempThresholds::cpu();
        let color = thresh.color(90.0);
        assert!(color.r > 0.9 && color.g < 0.5, "Critical should be red");
    }

    #[test]
    fn test_temp_thresholds_default() {
        let default = TempThresholds::default();
        let cpu = TempThresholds::cpu();
        assert_eq!(default.warning, cpu.warning);
    }

    #[test]
    fn test_temp_thresholds_derive_debug() {
        let thresh = TempThresholds::cpu();
        let debug = format!("{:?}", thresh);
        assert!(debug.contains("TempThresholds"));
    }

    #[test]
    fn test_temp_thresholds_derive_clone() {
        let thresh = TempThresholds::cpu();
        let cloned = thresh.clone();
        assert_eq!(thresh, cloned);
    }

    // =========================================================================
    // temp color function tests
    // =========================================================================

    #[test]
    fn test_cpu_temp_color() {
        let color = cpu_temp_color(50.0);
        assert!(color.g > 0.8);
    }

    #[test]
    fn test_gpu_temp_color() {
        let color = gpu_temp_color(60.0);
        assert!(color.g > 0.8); // 60 is below GPU warning (75)
    }

    #[test]
    fn test_temp_color() {
        let color = temp_color(90.0);
        assert!(color.r > 0.9);
    }

    // =========================================================================
    // format_fan_speed tests
    // =========================================================================

    #[test]
    fn test_format_fan_speed_off() {
        assert_eq!(format_fan_speed(0), "Off");
    }

    #[test]
    fn test_format_fan_speed_low() {
        assert_eq!(format_fan_speed(500), "500 RPM");
    }

    #[test]
    fn test_format_fan_speed_high() {
        let result = format_fan_speed(2500);
        assert!(result.contains("K RPM"));
    }

    // =========================================================================
    // fan_speed_color tests
    // =========================================================================

    #[test]
    fn test_fan_speed_color_off() {
        let color = fan_speed_color(0, 3000);
        assert!(color.r < 0.5, "Off should be gray");
    }

    #[test]
    fn test_fan_speed_color_low() {
        let color = fan_speed_color(500, 3000);
        assert!(color.b > 0.7, "Low speed should be blue");
    }

    #[test]
    fn test_fan_speed_color_normal() {
        let color = fan_speed_color(1500, 3000);
        assert!(color.g > 0.7, "Normal speed should be green");
    }

    #[test]
    fn test_fan_speed_color_high() {
        let color = fan_speed_color(2800, 3000);
        assert!(color.r > 0.9, "High speed should be red-ish");
    }

    #[test]
    fn test_fan_speed_color_zero_max() {
        let color = fan_speed_color(1000, 0);
        assert!((color.r - color.g).abs() < 0.1, "Zero max should be gray");
    }

    // =========================================================================
    // format_power tests
    // =========================================================================

    #[test]
    fn test_format_power_low() {
        assert_eq!(format_power(5.5), "5.50W");
    }

    #[test]
    fn test_format_power_medium() {
        assert_eq!(format_power(65.5), "65.5W");
    }

    #[test]
    fn test_format_power_high() {
        assert_eq!(format_power(250.0), "250W");
    }

    #[test]
    fn test_format_power_kw() {
        let result = format_power(1500.0);
        assert!(result.contains("kW"));
    }

    // =========================================================================
    // power_color tests
    // =========================================================================

    #[test]
    fn test_power_color_normal() {
        let color = power_color(50.0, 150.0);
        assert!(color.g > 0.7, "Normal power should be green");
    }

    #[test]
    fn test_power_color_high() {
        let color = power_color(120.0, 150.0);
        assert!(color.r > 0.9);
    }

    #[test]
    fn test_power_color_over_tdp() {
        let color = power_color(160.0, 150.0);
        assert!(color.r > 0.9 && color.g < 0.5, "Over TDP should be red");
    }

    #[test]
    fn test_power_color_zero_tdp() {
        let color = power_color(100.0, 0.0);
        assert!((color.r - color.g).abs() < 0.1, "Zero TDP should be gray");
    }

    // =========================================================================
    // SensorType tests
    // =========================================================================

    #[test]
    fn test_sensor_type_icon() {
        assert!(!SensorType::CpuCore.icon().is_empty());
        assert!(!SensorType::Gpu.icon().is_empty());
        assert!(!SensorType::Fan.icon().is_empty());
    }

    #[test]
    fn test_sensor_type_unit() {
        assert_eq!(SensorType::CpuCore.unit(), "°C");
        assert_eq!(SensorType::Fan.unit(), "RPM");
        assert_eq!(SensorType::Power.unit(), "W");
        assert_eq!(SensorType::Voltage.unit(), "V");
    }

    #[test]
    fn test_sensor_type_derive_debug() {
        let st = SensorType::CpuCore;
        let debug = format!("{:?}", st);
        assert!(debug.contains("CpuCore"));
    }

    #[test]
    fn test_sensor_type_derive_copy() {
        let st = SensorType::Gpu;
        let copied = st;
        assert_eq!(st, copied);
    }
}
