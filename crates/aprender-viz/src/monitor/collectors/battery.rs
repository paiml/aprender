//! Battery status collector.
//!
//! Reads from `/sys/class/power_supply/` on Linux to collect battery metrics.
//!
//! ## Falsification Criteria
//!
//! - #50: Battery percentage matches `upower` within ±1%

use crate::monitor::error::Result;
use crate::monitor::types::{Collector, MetricValue, Metrics};
use std::path::PathBuf;
use std::time::Duration;

/// Battery charging state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BatteryState {
    /// Battery is charging.
    Charging,
    /// Battery is discharging.
    Discharging,
    /// Battery is full.
    Full,
    /// Battery is not charging (plugged in but not charging).
    NotCharging,
    /// Unknown state.
    #[default]
    Unknown,
}

impl BatteryState {
    /// Parses battery state from sysfs string.
    fn from_str(s: &str) -> Self {
        match s.trim().to_lowercase().as_str() {
            "charging" => Self::Charging,
            "discharging" => Self::Discharging,
            "full" => Self::Full,
            "not charging" => Self::NotCharging,
            _ => Self::Unknown,
        }
    }

    /// Returns a display character for the state.
    #[must_use]
    pub fn symbol(&self) -> char {
        match self {
            Self::Charging => '⚡',
            Self::Discharging => '🔋',
            Self::Full => '✓',
            Self::NotCharging => '⏸',
            Self::Unknown => '?',
        }
    }

    /// Returns true if the battery is charging.
    #[must_use]
    pub fn is_charging(&self) -> bool {
        matches!(self, Self::Charging)
    }
}

/// Information about a battery.
#[derive(Debug, Clone)]
pub struct BatteryInfo {
    /// Battery name (e.g., "BAT0").
    pub name: String,
    /// Current capacity percentage (0-100).
    pub capacity: u8,
    /// Charging state.
    pub state: BatteryState,
    /// Energy now in µWh (if available).
    pub energy_now: Option<u64>,
    /// Energy full in µWh (if available).
    pub energy_full: Option<u64>,
    /// Energy full design in µWh (if available).
    pub energy_full_design: Option<u64>,
    /// Current power draw in µW (if available).
    pub power_now: Option<u64>,
    /// Voltage now in µV (if available).
    pub voltage_now: Option<u64>,
    /// Time to empty in seconds (calculated).
    pub time_to_empty: Option<u64>,
    /// Time to full in seconds (calculated).
    pub time_to_full: Option<u64>,
}

impl BatteryInfo {
    /// Returns the battery health as a percentage.
    #[must_use]
    pub fn health_percent(&self) -> Option<f64> {
        match (self.energy_full, self.energy_full_design) {
            (Some(full), Some(design)) if design > 0 => Some((full as f64 / design as f64) * 100.0),
            _ => None,
        }
    }

    /// Returns the current power draw in Watts.
    #[must_use]
    pub fn power_watts(&self) -> Option<f64> {
        self.power_now.map(|p| p as f64 / 1_000_000.0)
    }

    /// Formats the time remaining as a human-readable string.
    #[must_use]
    pub fn time_remaining_formatted(&self) -> Option<String> {
        let seconds = match self.state {
            BatteryState::Discharging => self.time_to_empty,
            BatteryState::Charging => self.time_to_full,
            _ => None,
        }?;

        let hours = seconds / 3600;
        let minutes = (seconds % 3600) / 60;

        if hours > 0 {
            Some(format!("{hours}h {minutes}m"))
        } else {
            Some(format!("{minutes}m"))
        }
    }
}

/// Collector for battery metrics.
#[derive(Debug)]
pub struct BatteryCollector {
    /// Detected batteries.
    batteries: Vec<BatteryInfo>,
    /// Power supply base path.
    power_supply_path: PathBuf,
}

impl BatteryCollector {
    /// Creates a new battery collector.
    #[must_use]
    pub fn new() -> Self {
        Self { batteries: Vec::new(), power_supply_path: PathBuf::from("/sys/class/power_supply") }
    }

    /// Returns all detected batteries.
    #[must_use]
    pub fn batteries(&self) -> &[BatteryInfo] {
        &self.batteries
    }

    /// Returns the primary battery (first one).
    #[must_use]
    pub fn primary(&self) -> Option<&BatteryInfo> {
        self.batteries.first()
    }

    /// Returns true if any battery is charging.
    #[must_use]
    pub fn is_charging(&self) -> bool {
        self.batteries.iter().any(|b| b.state.is_charging())
    }

    /// Returns the combined capacity percentage.
    #[must_use]
    pub fn total_capacity(&self) -> Option<u8> {
        if self.batteries.is_empty() {
            return None;
        }

        // Weighted average based on energy_full
        let (total_energy, total_full): (u64, u64) = self
            .batteries
            .iter()
            .filter_map(|b| match (b.energy_now, b.energy_full) {
                (Some(now), Some(full)) => Some((now, full)),
                _ => None,
            })
            .fold((0, 0), |(acc_e, acc_f), (e, f)| (acc_e + e, acc_f + f));

        if total_full > 0 {
            Some(((total_energy as f64 / total_full as f64) * 100.0) as u8)
        } else {
            // Fall back to simple average of capacity
            let sum: u32 = self.batteries.iter().map(|b| u32::from(b.capacity)).sum();
            Some((sum / self.batteries.len() as u32) as u8)
        }
    }

    /// Discovers battery devices.
    #[cfg(target_os = "linux")]
    fn discover_batteries(&self) -> Vec<String> {
        if !self.power_supply_path.exists() {
            return Vec::new();
        }

        std::fs::read_dir(&self.power_supply_path)
            .ok()
            .map(|entries| {
                entries
                    .filter_map(std::result::Result::ok)
                    .filter_map(|e| {
                        let name = e.file_name().to_string_lossy().to_string();
                        // Check if it's a battery
                        let type_path = e.path().join("type");
                        let psu_type = std::fs::read_to_string(&type_path)
                            .ok()
                            .map(|s| s.trim().to_lowercase());

                        if psu_type.as_deref() == Some("battery") {
                            Some(name)
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    #[cfg(not(target_os = "linux"))]
    fn discover_batteries(&self) -> Vec<String> {
        Vec::new()
    }

    /// Reads a sysfs file as u64.
    fn read_u64(&self, path: &PathBuf) -> Option<u64> {
        std::fs::read_to_string(path).ok().and_then(|s| s.trim().parse().ok())
    }

    /// Reads a sysfs file as string.
    fn read_string(&self, path: &PathBuf) -> Option<String> {
        std::fs::read_to_string(path).ok().map(|s| s.trim().to_string())
    }

    /// Reads battery information from sysfs.
    #[cfg(target_os = "linux")]
    fn read_battery(&self, name: &str) -> Option<BatteryInfo> {
        let base = self.power_supply_path.join(name);

        // Read capacity (required)
        let capacity = self.read_u64(&base.join("capacity"))? as u8;

        // Read state
        let state = self
            .read_string(&base.join("status"))
            .map(|s| BatteryState::from_str(&s))
            .unwrap_or_default();

        // Read energy values (may not exist on all systems)
        let energy_now = self.read_u64(&base.join("energy_now"));
        let energy_full = self.read_u64(&base.join("energy_full"));
        let energy_full_design = self.read_u64(&base.join("energy_full_design"));
        let power_now = self.read_u64(&base.join("power_now"));
        let voltage_now = self.read_u64(&base.join("voltage_now"));

        // Calculate time remaining
        let (time_to_empty, time_to_full) = match (energy_now, energy_full, power_now) {
            (Some(now), Some(full), Some(power)) if power > 0 => {
                let to_empty = (now as f64 / power as f64 * 3600.0) as u64;
                let to_full = ((full.saturating_sub(now)) as f64 / power as f64 * 3600.0) as u64;
                (Some(to_empty), Some(to_full))
            }
            _ => (None, None),
        };

        Some(BatteryInfo {
            name: name.to_string(),
            capacity,
            state,
            energy_now,
            energy_full,
            energy_full_design,
            power_now,
            voltage_now,
            time_to_empty,
            time_to_full,
        })
    }

    #[cfg(not(target_os = "linux"))]
    fn read_battery(&self, _name: &str) -> Option<BatteryInfo> {
        None
    }
}

impl Default for BatteryCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl Collector for BatteryCollector {
    fn id(&self) -> &'static str {
        "battery"
    }

    fn collect(&mut self) -> Result<Metrics> {
        self.batteries.clear();

        // Discover and read all batteries
        for name in self.discover_batteries() {
            if let Some(info) = self.read_battery(&name) {
                self.batteries.push(info);
            }
        }

        // Build metrics
        let mut metrics = Metrics::new();

        // Battery count
        metrics.insert("battery.count", MetricValue::Counter(self.batteries.len() as u64));

        // Primary battery info
        if let Some(primary) = self.primary() {
            metrics.insert("battery.capacity", MetricValue::Gauge(f64::from(primary.capacity)));
            metrics.insert(
                "battery.charging",
                MetricValue::Gauge(if primary.state.is_charging() { 1.0 } else { 0.0 }),
            );

            if let Some(power) = primary.power_watts() {
                metrics.insert("battery.power_watts", MetricValue::Gauge(power));
            }

            if let Some(health) = primary.health_percent() {
                metrics.insert("battery.health", MetricValue::Gauge(health));
            }
        }

        // Total capacity
        if let Some(total) = self.total_capacity() {
            metrics.insert("battery.total_capacity", MetricValue::Gauge(f64::from(total)));
        }

        Ok(metrics)
    }

    fn is_available(&self) -> bool {
        !self.discover_batteries().is_empty()
    }

    fn interval_hint(&self) -> Duration {
        Duration::from_millis(5000) // Battery state changes slowly
    }

    fn display_name(&self) -> &'static str {
        "Battery"
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
    fn test_battery_state_from_str() {
        assert_eq!(BatteryState::from_str("Charging"), BatteryState::Charging);
        assert_eq!(BatteryState::from_str("Discharging"), BatteryState::Discharging);
        assert_eq!(BatteryState::from_str("Full"), BatteryState::Full);
        assert_eq!(BatteryState::from_str("Not charging"), BatteryState::NotCharging);
        assert_eq!(BatteryState::from_str("random"), BatteryState::Unknown);
    }

    #[test]
    fn test_battery_state_symbol() {
        assert_eq!(BatteryState::Charging.symbol(), '⚡');
        assert_eq!(BatteryState::Discharging.symbol(), '🔋');
        assert_eq!(BatteryState::Full.symbol(), '✓');
    }

    #[test]
    fn test_battery_state_is_charging() {
        assert!(BatteryState::Charging.is_charging());
        assert!(!BatteryState::Discharging.is_charging());
        assert!(!BatteryState::Full.is_charging());
    }

    #[test]
    fn test_battery_state_default() {
        assert_eq!(BatteryState::default(), BatteryState::Unknown);
    }

    #[test]
    fn test_battery_info_health_percent() {
        let info = BatteryInfo {
            name: "BAT0".to_string(),
            capacity: 50,
            state: BatteryState::Discharging,
            energy_now: Some(25_000_000),
            energy_full: Some(50_000_000),
            energy_full_design: Some(60_000_000),
            power_now: Some(10_000_000),
            voltage_now: Some(12_000_000),
            time_to_empty: Some(9000),
            time_to_full: None,
        };

        // health = 50_000_000 / 60_000_000 * 100 = 83.33%
        let health = info.health_percent().expect("Should have health");
        assert!((health - 83.33).abs() < 0.1);
    }

    #[test]
    fn test_battery_info_power_watts() {
        let info = BatteryInfo {
            name: "BAT0".to_string(),
            capacity: 50,
            state: BatteryState::Discharging,
            energy_now: None,
            energy_full: None,
            energy_full_design: None,
            power_now: Some(15_000_000), // 15W
            voltage_now: None,
            time_to_empty: None,
            time_to_full: None,
        };

        let power = info.power_watts().expect("Should have power");
        assert!((power - 15.0).abs() < 0.01);
    }

    #[test]
    fn test_battery_info_time_remaining_formatted() {
        let info = BatteryInfo {
            name: "BAT0".to_string(),
            capacity: 50,
            state: BatteryState::Discharging,
            energy_now: None,
            energy_full: None,
            energy_full_design: None,
            power_now: None,
            voltage_now: None,
            time_to_empty: Some(5400), // 1h 30m
            time_to_full: None,
        };

        let formatted = info.time_remaining_formatted().expect("Should have formatted time");
        assert_eq!(formatted, "1h 30m");
    }

    #[test]
    fn test_battery_info_time_remaining_minutes_only() {
        let info = BatteryInfo {
            name: "BAT0".to_string(),
            capacity: 50,
            state: BatteryState::Discharging,
            energy_now: None,
            energy_full: None,
            energy_full_design: None,
            power_now: None,
            voltage_now: None,
            time_to_empty: Some(1800), // 30m
            time_to_full: None,
        };

        let formatted = info.time_remaining_formatted().expect("Should have formatted time");
        assert_eq!(formatted, "30m");
    }

    #[test]
    fn test_battery_collector_new() {
        let collector = BatteryCollector::new();
        assert!(collector.batteries.is_empty());
    }

    #[test]
    fn test_battery_collector_default() {
        let collector = BatteryCollector::default();
        assert!(collector.batteries.is_empty());
    }

    #[test]
    fn test_battery_collector_interval() {
        let collector = BatteryCollector::new();
        assert_eq!(collector.interval_hint(), Duration::from_millis(5000));
    }

    #[test]
    fn test_battery_collector_display_name() {
        let collector = BatteryCollector::new();
        assert_eq!(collector.display_name(), "Battery");
    }

    #[test]
    fn test_battery_collector_id() {
        let collector = BatteryCollector::new();
        assert_eq!(collector.id(), "battery");
    }

    #[test]
    fn test_battery_collector_total_capacity() {
        let mut collector = BatteryCollector::new();

        // Add mock batteries
        collector.batteries.push(BatteryInfo {
            name: "BAT0".to_string(),
            capacity: 80,
            state: BatteryState::Discharging,
            energy_now: Some(40_000_000),
            energy_full: Some(50_000_000),
            energy_full_design: Some(60_000_000),
            power_now: None,
            voltage_now: None,
            time_to_empty: None,
            time_to_full: None,
        });

        collector.batteries.push(BatteryInfo {
            name: "BAT1".to_string(),
            capacity: 60,
            state: BatteryState::Discharging,
            energy_now: Some(30_000_000),
            energy_full: Some(50_000_000),
            energy_full_design: Some(60_000_000),
            power_now: None,
            voltage_now: None,
            time_to_empty: None,
            time_to_full: None,
        });

        // Total: 70_000_000 / 100_000_000 = 70%
        let total = collector.total_capacity().expect("Should have total");
        assert_eq!(total, 70);
    }

    #[test]
    fn test_battery_collector_is_charging() {
        let mut collector = BatteryCollector::new();

        collector.batteries.push(BatteryInfo {
            name: "BAT0".to_string(),
            capacity: 50,
            state: BatteryState::Discharging,
            energy_now: None,
            energy_full: None,
            energy_full_design: None,
            power_now: None,
            voltage_now: None,
            time_to_empty: None,
            time_to_full: None,
        });

        assert!(!collector.is_charging());

        collector.batteries[0].state = BatteryState::Charging;
        assert!(collector.is_charging());
    }

    // ========================================================================
    // Linux-specific Tests
    // ========================================================================

    #[cfg(target_os = "linux")]
    #[test]
    fn test_battery_collector_collect() {
        let mut collector = BatteryCollector::new();
        let result = collector.collect();

        assert!(result.is_ok());
        let metrics = result.expect("collect should succeed");

        // Should have battery count metric
        assert!(metrics.get_counter("battery.count").is_some());
    }
}
