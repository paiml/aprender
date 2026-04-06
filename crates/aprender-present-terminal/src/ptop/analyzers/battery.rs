//! Battery analyzer for ptop (PMAT-GAP-036 - ttop parity)
//!
//! Reads battery status from /sys/class/power_supply/ on Linux.
//! Updates battery state asynchronously without blocking UI.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// Battery charging state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BatteryState {
    /// Currently charging
    Charging,
    /// Currently discharging (on battery power)
    Discharging,
    /// Battery full
    Full,
    /// Not charging (plugged in but not charging)
    NotCharging,
    /// State unknown
    #[default]
    Unknown,
}

impl BatteryState {
    /// Parse battery state from sysfs string
    pub fn from_str(s: &str) -> Self {
        match s.trim().to_lowercase().as_str() {
            "charging" => Self::Charging,
            "discharging" => Self::Discharging,
            "full" => Self::Full,
            "not charging" => Self::NotCharging,
            _ => Self::Unknown,
        }
    }

    /// Get display name for UI
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::Charging => "Charging",
            Self::Discharging => "Discharging",
            Self::Full => "Full",
            Self::NotCharging => "Plugged",
            Self::Unknown => "Unknown",
        }
    }

    /// Get icon/emoji for state
    #[must_use]
    pub fn icon(&self) -> &'static str {
        match self {
            Self::Charging => "🔌",
            Self::Discharging => "🔋",
            Self::Full => "⚡",
            Self::NotCharging => "🔌",
            Self::Unknown => "❓",
        }
    }
}

/// Information about a single battery
#[derive(Debug, Clone, Default)]
pub struct BatteryInfo {
    /// Battery name (e.g., "BAT0", "BAT1")
    pub name: String,
    /// Current charge percentage (0-100)
    pub percentage: f32,
    /// Current state (Charging, Discharging, etc.)
    pub state: BatteryState,
    /// Current energy in Wh (optional)
    pub energy_now: Option<f64>,
    /// Full charge energy in Wh (optional)
    pub energy_full: Option<f64>,
    /// Design capacity in Wh (optional)
    pub energy_design: Option<f64>,
    /// Power draw in W (optional, positive = charging, negative = discharging)
    pub power_now: Option<f64>,
    /// Estimated time to empty in seconds (if discharging)
    pub time_to_empty: Option<u64>,
    /// Estimated time to full in seconds (if charging)
    pub time_to_full: Option<u64>,
    /// Battery health percentage (energy_full / energy_design * 100)
    pub health: Option<f32>,
    /// Cycle count (optional)
    pub cycle_count: Option<u32>,
    /// Battery temperature in °C (optional)
    pub temperature: Option<f32>,
    /// Voltage in V (optional)
    pub voltage: Option<f64>,
}

impl BatteryInfo {
    /// Format time to empty/full as human-readable string
    #[must_use]
    pub fn format_time_remaining(&self) -> Option<String> {
        let seconds = match self.state {
            BatteryState::Discharging => self.time_to_empty?,
            BatteryState::Charging => self.time_to_full?,
            _ => return None,
        };

        let hours = seconds / 3600;
        let minutes = (seconds % 3600) / 60;

        if hours > 0 {
            Some(format!("{hours}h{minutes:02}m"))
        } else {
            Some(format!("{minutes}m"))
        }
    }
}

/// Battery analyzer data
#[derive(Debug, Clone, Default)]
pub struct BatteryData {
    /// List of batteries detected
    pub batteries: Vec<BatteryInfo>,
    /// Combined percentage across all batteries
    pub combined_percentage: f32,
    /// Combined state (Charging takes priority)
    pub combined_state: BatteryState,
    /// AC adapter connected
    pub ac_connected: bool,
    /// Timestamp of last update
    pub last_update: Option<Instant>,
}

impl BatteryData {
    /// Check if any battery is available
    #[must_use]
    pub fn has_battery(&self) -> bool {
        !self.batteries.is_empty()
    }

    /// Get primary battery (first one)
    #[must_use]
    pub fn primary(&self) -> Option<&BatteryInfo> {
        self.batteries.first()
    }
}

/// Battery analyzer
#[derive(Debug)]
pub struct BatteryAnalyzer {
    /// Collected battery data
    data: BatteryData,
    /// Path to power_supply sysfs directory
    sysfs_path: PathBuf,
    /// Minimum interval between updates
    update_interval: Duration,
    /// Last update time
    last_update: Instant,
}

impl BatteryAnalyzer {
    /// Create new battery analyzer
    #[must_use]
    pub fn new() -> Option<Self> {
        let sysfs_path = PathBuf::from("/sys/class/power_supply");
        if !sysfs_path.exists() {
            return None;
        }

        // Check if there are any batteries
        let has_battery = std::fs::read_dir(&sysfs_path)
            .ok()?
            .filter_map(Result::ok)
            .any(|entry| {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                name.starts_with("BAT") || name.starts_with("battery")
            });

        if !has_battery {
            return None;
        }

        Some(Self {
            data: BatteryData::default(),
            sysfs_path,
            update_interval: Duration::from_secs(5), // Update every 5s
            last_update: Instant::now()
                .checked_sub(Duration::from_secs(10))
                .unwrap_or_else(Instant::now), // Force immediate update
        })
    }

    /// Get current battery data
    #[must_use]
    pub fn data(&self) -> &BatteryData {
        &self.data
    }

    /// Collect battery data (call periodically)
    pub fn collect(&mut self) {
        // Rate limit updates
        if self.last_update.elapsed() < self.update_interval {
            return;
        }
        self.last_update = Instant::now();

        let mut batteries = Vec::new();
        let mut ac_connected = false;

        if let Ok(entries) = std::fs::read_dir(&self.sysfs_path) {
            for entry in entries.filter_map(Result::ok) {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                let path = entry.path();

                if name_str.starts_with("BAT") || name_str.starts_with("battery") {
                    if let Some(info) = Self::read_battery(&path, &name_str) {
                        batteries.push(info);
                    }
                } else if name_str.starts_with("AC") || name_str.starts_with("ACAD") {
                    ac_connected = Self::read_ac_status(&path);
                }
            }
        }

        // Calculate combined values
        let combined_percentage = if batteries.is_empty() {
            0.0
        } else {
            batteries.iter().map(|b| b.percentage).sum::<f32>() / batteries.len() as f32
        };

        let combined_state = if batteries.iter().any(|b| b.state == BatteryState::Charging) {
            BatteryState::Charging
        } else if batteries.iter().all(|b| b.state == BatteryState::Full) {
            BatteryState::Full
        } else if batteries
            .iter()
            .any(|b| b.state == BatteryState::Discharging)
        {
            BatteryState::Discharging
        } else {
            BatteryState::Unknown
        };

        self.data = BatteryData {
            batteries,
            combined_percentage,
            combined_state,
            ac_connected,
            last_update: Some(Instant::now()),
        };
    }

    /// Read battery info from sysfs path
    fn read_battery(path: &Path, name: &str) -> Option<BatteryInfo> {
        let read_file =
            |file: &str| -> Option<String> { std::fs::read_to_string(path.join(file)).ok() };

        let read_int = |file: &str| -> Option<i64> { read_file(file)?.trim().parse().ok() };

        // Read status
        let state = read_file("status")
            .map(|s| BatteryState::from_str(&s))
            .unwrap_or_default();

        // Read capacity (percentage)
        let percentage = read_int("capacity").unwrap_or(0) as f32;

        // Energy values are in µWh, convert to Wh
        let energy_now = read_int("energy_now").map(|e| e as f64 / 1_000_000.0);
        let energy_full = read_int("energy_full").map(|e| e as f64 / 1_000_000.0);
        let energy_design = read_int("energy_full_design").map(|e| e as f64 / 1_000_000.0);

        // Power is in µW, convert to W
        let power_now = read_int("power_now").map(|p| p as f64 / 1_000_000.0);

        // Voltage in µV, convert to V
        let voltage = read_int("voltage_now").map(|v| v as f64 / 1_000_000.0);

        // Calculate health
        let health = match (energy_full, energy_design) {
            (Some(full), Some(design)) if design > 0.0 => Some((full / design * 100.0) as f32),
            _ => None,
        };

        // Calculate time to empty/full
        let (time_to_empty, time_to_full) = match (power_now, energy_now, energy_full) {
            (Some(power), Some(now), Some(full)) if power > 0.0 => match state {
                BatteryState::Discharging => {
                    let hours = now / power;
                    (Some((hours * 3600.0) as u64), None)
                }
                BatteryState::Charging => {
                    let hours = (full - now) / power;
                    (None, Some((hours * 3600.0) as u64))
                }
                _ => (None, None),
            },
            _ => (None, None),
        };

        // Cycle count
        let cycle_count = read_int("cycle_count").map(|c| c as u32);

        // Temperature (not always available)
        let temperature = read_int("temp").map(|t| t as f32 / 10.0); // Usually in 0.1°C

        Some(BatteryInfo {
            name: name.to_string(),
            percentage,
            state,
            energy_now,
            energy_full,
            energy_design,
            power_now,
            time_to_empty,
            time_to_full,
            health,
            cycle_count,
            temperature,
            voltage,
        })
    }

    /// Read AC adapter status
    fn read_ac_status(path: &Path) -> bool {
        std::fs::read_to_string(path.join("online"))
            .ok()
            .and_then(|s| s.trim().parse::<i32>().ok())
            .is_some_and(|v| v == 1)
    }
}

impl Default for BatteryAnalyzer {
    fn default() -> Self {
        Self {
            data: BatteryData::default(),
            sysfs_path: PathBuf::from("/sys/class/power_supply"),
            update_interval: Duration::from_secs(5),
            last_update: Instant::now()
                .checked_sub(Duration::from_secs(10))
                .unwrap_or_else(Instant::now),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_battery_state_from_str() {
        assert_eq!(BatteryState::from_str("Charging"), BatteryState::Charging);
        assert_eq!(
            BatteryState::from_str("Discharging"),
            BatteryState::Discharging
        );
        assert_eq!(BatteryState::from_str("Full"), BatteryState::Full);
        assert_eq!(
            BatteryState::from_str("Not charging"),
            BatteryState::NotCharging
        );
        assert_eq!(BatteryState::from_str("unknown"), BatteryState::Unknown);
        assert_eq!(BatteryState::from_str("random"), BatteryState::Unknown);
    }

    #[test]
    fn test_battery_state_name() {
        assert_eq!(BatteryState::Charging.name(), "Charging");
        assert_eq!(BatteryState::Discharging.name(), "Discharging");
        assert_eq!(BatteryState::Full.name(), "Full");
        assert_eq!(BatteryState::NotCharging.name(), "Plugged");
        assert_eq!(BatteryState::Unknown.name(), "Unknown");
    }

    #[test]
    fn test_battery_state_icon() {
        assert_eq!(BatteryState::Charging.icon(), "🔌");
        assert_eq!(BatteryState::Discharging.icon(), "🔋");
        assert_eq!(BatteryState::Full.icon(), "⚡");
    }

    #[test]
    fn test_battery_info_default() {
        let info = BatteryInfo::default();
        assert_eq!(info.percentage, 0.0);
        assert_eq!(info.state, BatteryState::Unknown);
        assert!(info.name.is_empty());
    }

    #[test]
    fn test_battery_data_has_battery() {
        let mut data = BatteryData::default();
        assert!(!data.has_battery());

        data.batteries.push(BatteryInfo::default());
        assert!(data.has_battery());
    }

    #[test]
    fn test_battery_data_primary() {
        let mut data = BatteryData::default();
        assert!(data.primary().is_none());

        data.batteries.push(BatteryInfo {
            name: "BAT0".to_string(),
            percentage: 75.0,
            ..Default::default()
        });
        assert!(data.primary().is_some());
        assert_eq!(data.primary().unwrap().name, "BAT0");
    }

    #[test]
    fn test_format_time_remaining() {
        let mut info = BatteryInfo::default();
        assert!(info.format_time_remaining().is_none());

        info.state = BatteryState::Discharging;
        info.time_to_empty = Some(3661); // 1h1m1s
        assert_eq!(info.format_time_remaining(), Some("1h01m".to_string()));

        info.time_to_empty = Some(1800); // 30m
        assert_eq!(info.format_time_remaining(), Some("30m".to_string()));

        info.state = BatteryState::Charging;
        info.time_to_full = Some(7200); // 2h
        assert_eq!(info.format_time_remaining(), Some("2h00m".to_string()));
    }

    #[test]
    fn test_battery_analyzer_default() {
        let analyzer = BatteryAnalyzer::default();
        assert!(!analyzer.data().has_battery());
    }
}
