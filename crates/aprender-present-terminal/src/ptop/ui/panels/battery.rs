//! Battery panel rendering and utilities.
//!
//! Provides battery panel title building, charge formatting,
//! and helper functions for rendering battery metrics.

use presentar_core::Color;

// =============================================================================
// BATTERY TITLE BUILDING
// =============================================================================

/// Build battery panel title string.
///
/// Format: "Battery │ 85% │ Charging │ 2h 15m"
#[must_use]
pub fn build_battery_title(
    percent: f64,
    state: BatteryState,
    time_remaining: Option<u64>,
) -> String {
    let state_str = state.display_name();

    if let Some(secs) = time_remaining {
        let time_str = format_time_remaining(secs);
        format!("Battery │ {:.0}% │ {} │ {}", percent, state_str, time_str)
    } else {
        format!("Battery │ {:.0}% │ {}", percent, state_str)
    }
}

/// Build compact battery title for narrow panels.
///
/// Format: "Bat │ 85% ⚡"
#[must_use]
pub fn build_battery_title_compact(percent: f64, is_charging: bool) -> String {
    let icon = if is_charging { " ⚡" } else { "" };
    format!("Bat │ {:.0}%{}", percent, icon)
}

// =============================================================================
// BATTERY STATE
// =============================================================================

/// Battery charging state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BatteryState {
    /// Battery is charging
    Charging,
    /// Battery is discharging (on battery power)
    #[default]
    Discharging,
    /// Battery is full and connected to power
    Full,
    /// Battery status is unknown
    Unknown,
    /// Battery is not present
    NotPresent,
}

impl BatteryState {
    /// Get display name for battery state.
    #[must_use]
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Charging => "Charging",
            Self::Discharging => "Discharging",
            Self::Full => "Full",
            Self::Unknown => "Unknown",
            Self::NotPresent => "N/A",
        }
    }

    /// Get short display name.
    #[must_use]
    pub fn short_name(&self) -> &'static str {
        match self {
            Self::Charging => "CHG",
            Self::Discharging => "DIS",
            Self::Full => "FULL",
            Self::Unknown => "UNK",
            Self::NotPresent => "N/A",
        }
    }

    /// Get icon for battery state.
    #[must_use]
    pub fn icon(&self) -> &'static str {
        match self {
            Self::Charging => "⚡",
            Self::Discharging => "🔋",
            Self::Full => "🔌",
            Self::Unknown => "❓",
            Self::NotPresent => "∅",
        }
    }

    /// Check if battery is charging.
    #[must_use]
    pub fn is_charging(&self) -> bool {
        matches!(self, Self::Charging)
    }

    /// Check if on battery power.
    #[must_use]
    pub fn is_discharging(&self) -> bool {
        matches!(self, Self::Discharging)
    }
}

// =============================================================================
// TIME FORMATTING
// =============================================================================

/// Format time remaining in human-readable form.
///
/// # Examples
/// - 3600 -> "1h 0m"
/// - 5400 -> "1h 30m"
/// - 300 -> "5m"
#[must_use]
pub fn format_time_remaining(seconds: u64) -> String {
    let hours = seconds / 3600;
    let minutes = (seconds % 3600) / 60;

    if hours > 0 {
        format!("{}h {}m", hours, minutes)
    } else {
        format!("{}m", minutes)
    }
}

/// Format time remaining in compact form.
///
/// # Examples
/// - 3600 -> "1:00"
/// - 5400 -> "1:30"
#[must_use]
pub fn format_time_compact(seconds: u64) -> String {
    let hours = seconds / 3600;
    let minutes = (seconds % 3600) / 60;
    format!("{}:{:02}", hours, minutes)
}

// =============================================================================
// BATTERY COLORS
// =============================================================================

/// Get color for battery percentage.
#[must_use]
pub fn battery_percent_color(percent: f64, is_charging: bool) -> Color {
    if is_charging {
        // When charging, always show positive color
        Color::new(0.3, 0.9, 0.5, 1.0) // Green
    } else if percent <= 10.0 {
        Color::new(1.0, 0.3, 0.3, 1.0) // Critical red
    } else if percent <= 20.0 {
        Color::new(1.0, 0.5, 0.2, 1.0) // Warning orange
    } else if percent <= 40.0 {
        Color::new(1.0, 0.8, 0.2, 1.0) // Yellow
    } else {
        Color::new(0.3, 0.9, 0.5, 1.0) // Green
    }
}

/// Get color for battery state.
#[must_use]
pub fn battery_state_color(state: BatteryState) -> Color {
    match state {
        BatteryState::Charging => Color::new(0.3, 0.9, 0.5, 1.0), // Green
        BatteryState::Full => Color::new(0.4, 0.8, 1.0, 1.0),     // Blue
        BatteryState::Discharging => Color::new(1.0, 0.8, 0.3, 1.0), // Yellow
        BatteryState::Unknown => Color::new(0.5, 0.5, 0.5, 1.0),  // Gray
        BatteryState::NotPresent => Color::new(0.3, 0.3, 0.3, 1.0), // Dark gray
    }
}

// =============================================================================
// BATTERY BAR SEGMENTS
// =============================================================================

/// Battery icon segment for visual display.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatteryIcon {
    /// Number of filled segments (0-4)
    pub filled: u8,
    /// Total segments
    pub total: u8,
}

impl BatteryIcon {
    /// Create battery icon from percentage.
    #[must_use]
    pub fn from_percent(percent: f64) -> Self {
        let filled = match percent {
            p if p >= 87.5 => 4,
            p if p >= 62.5 => 3,
            p if p >= 37.5 => 2,
            p if p >= 12.5 => 1,
            _ => 0,
        };

        Self { filled, total: 4 }
    }

    /// Get visual representation.
    #[must_use]
    pub fn display(&self) -> String {
        let filled_char = '█';
        let empty_char = '░';

        let filled: String = std::iter::repeat(filled_char)
            .take(self.filled as usize)
            .collect();
        let empty: String = std::iter::repeat(empty_char)
            .take((self.total - self.filled) as usize)
            .collect();

        format!("[{}{}]", filled, empty)
    }
}

impl Default for BatteryIcon {
    fn default() -> Self {
        Self::from_percent(100.0)
    }
}

// =============================================================================
// HEALTH METRICS
// =============================================================================

/// Battery health status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatteryHealth {
    /// Battery is healthy
    Good,
    /// Battery is degraded but functional
    Fair,
    /// Battery needs replacement
    Poor,
    /// Battery health unknown
    Unknown,
}

impl BatteryHealth {
    /// Determine health from capacity percentage.
    #[must_use]
    pub fn from_capacity(design_capacity: u64, full_capacity: u64) -> Self {
        if design_capacity == 0 {
            return Self::Unknown;
        }

        let percent = (full_capacity as f64 / design_capacity as f64) * 100.0;

        if percent >= 80.0 {
            Self::Good
        } else if percent >= 50.0 {
            Self::Fair
        } else {
            Self::Poor
        }
    }

    /// Get color for health status.
    #[must_use]
    pub fn color(&self) -> Color {
        match self {
            Self::Good => Color::new(0.3, 0.9, 0.5, 1.0), // Green
            Self::Fair => Color::new(1.0, 0.8, 0.3, 1.0), // Yellow
            Self::Poor => Color::new(1.0, 0.4, 0.3, 1.0), // Red
            Self::Unknown => Color::new(0.5, 0.5, 0.5, 1.0), // Gray
        }
    }

    /// Get display name.
    #[must_use]
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Good => "Good",
            Self::Fair => "Fair",
            Self::Poor => "Poor",
            Self::Unknown => "Unknown",
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
    // build_battery_title tests
    // =========================================================================

    #[test]
    fn test_build_battery_title_with_time() {
        let title = build_battery_title(85.0, BatteryState::Charging, Some(8100));
        assert!(title.contains("Battery"));
        assert!(title.contains("85%"));
        assert!(title.contains("Charging"));
        assert!(title.contains("2h 15m"));
    }

    #[test]
    fn test_build_battery_title_no_time() {
        let title = build_battery_title(100.0, BatteryState::Full, None);
        assert!(title.contains("Full"));
        assert!(!title.contains("m")); // No time
    }

    #[test]
    fn test_build_battery_title_discharging() {
        let title = build_battery_title(50.0, BatteryState::Discharging, Some(3600));
        assert!(title.contains("Discharging"));
        assert!(title.contains("1h 0m"));
    }

    // =========================================================================
    // build_battery_title_compact tests
    // =========================================================================

    #[test]
    fn test_build_battery_title_compact_charging() {
        let title = build_battery_title_compact(85.0, true);
        assert!(title.contains("Bat"));
        assert!(title.contains("85%"));
        assert!(title.contains("⚡"));
    }

    #[test]
    fn test_build_battery_title_compact_not_charging() {
        let title = build_battery_title_compact(50.0, false);
        assert!(!title.contains("⚡"));
    }

    // =========================================================================
    // BatteryState tests
    // =========================================================================

    #[test]
    fn test_battery_state_display_name() {
        assert_eq!(BatteryState::Charging.display_name(), "Charging");
        assert_eq!(BatteryState::Discharging.display_name(), "Discharging");
        assert_eq!(BatteryState::Full.display_name(), "Full");
    }

    #[test]
    fn test_battery_state_short_name() {
        assert_eq!(BatteryState::Charging.short_name(), "CHG");
        assert_eq!(BatteryState::Discharging.short_name(), "DIS");
    }

    #[test]
    fn test_battery_state_icon() {
        assert!(!BatteryState::Charging.icon().is_empty());
        assert!(!BatteryState::Discharging.icon().is_empty());
    }

    #[test]
    fn test_battery_state_is_charging() {
        assert!(BatteryState::Charging.is_charging());
        assert!(!BatteryState::Discharging.is_charging());
    }

    #[test]
    fn test_battery_state_is_discharging() {
        assert!(BatteryState::Discharging.is_discharging());
        assert!(!BatteryState::Charging.is_discharging());
    }

    #[test]
    fn test_battery_state_default() {
        assert_eq!(BatteryState::default(), BatteryState::Discharging);
    }

    #[test]
    fn test_battery_state_derive_debug() {
        let state = BatteryState::Charging;
        let debug = format!("{:?}", state);
        assert!(debug.contains("Charging"));
    }

    // =========================================================================
    // format_time tests
    // =========================================================================

    #[test]
    fn test_format_time_remaining_hours_minutes() {
        assert_eq!(format_time_remaining(5400), "1h 30m");
        assert_eq!(format_time_remaining(7200), "2h 0m");
    }

    #[test]
    fn test_format_time_remaining_minutes_only() {
        assert_eq!(format_time_remaining(300), "5m");
        assert_eq!(format_time_remaining(0), "0m");
    }

    #[test]
    fn test_format_time_compact() {
        assert_eq!(format_time_compact(3600), "1:00");
        assert_eq!(format_time_compact(5400), "1:30");
        assert_eq!(format_time_compact(90), "0:01");
    }

    // =========================================================================
    // battery color tests
    // =========================================================================

    #[test]
    fn test_battery_percent_color_charging() {
        let color = battery_percent_color(5.0, true);
        assert!(
            color.g > 0.8,
            "Charging should be green even at low percent"
        );
    }

    #[test]
    fn test_battery_percent_color_critical() {
        let color = battery_percent_color(5.0, false);
        assert!(color.r > 0.9 && color.g < 0.5, "Critical should be red");
    }

    #[test]
    fn test_battery_percent_color_warning() {
        let color = battery_percent_color(15.0, false);
        assert!(color.r > 0.9, "Warning should be orange");
    }

    #[test]
    fn test_battery_percent_color_low() {
        let color = battery_percent_color(30.0, false);
        assert!(color.r > 0.9 && color.g > 0.7, "Low should be yellow");
    }

    #[test]
    fn test_battery_percent_color_normal() {
        let color = battery_percent_color(80.0, false);
        assert!(color.g > 0.8, "Normal should be green");
    }

    #[test]
    fn test_battery_state_color() {
        let color = battery_state_color(BatteryState::Charging);
        assert!(color.g > 0.8);

        let color = battery_state_color(BatteryState::Full);
        assert!(color.b > 0.9);
    }

    // =========================================================================
    // BatteryIcon tests
    // =========================================================================

    #[test]
    fn test_battery_icon_from_percent_full() {
        let icon = BatteryIcon::from_percent(100.0);
        assert_eq!(icon.filled, 4);
    }

    #[test]
    fn test_battery_icon_from_percent_empty() {
        let icon = BatteryIcon::from_percent(5.0);
        assert_eq!(icon.filled, 0);
    }

    #[test]
    fn test_battery_icon_from_percent_half() {
        let icon = BatteryIcon::from_percent(50.0);
        assert_eq!(icon.filled, 2);
    }

    #[test]
    fn test_battery_icon_display() {
        let icon = BatteryIcon::from_percent(75.0);
        let display = icon.display();
        assert!(display.starts_with('['));
        assert!(display.ends_with(']'));
        assert!(display.contains('█'));
    }

    #[test]
    fn test_battery_icon_default() {
        let icon = BatteryIcon::default();
        assert_eq!(icon.filled, 4);
    }

    #[test]
    fn test_battery_icon_derive_debug() {
        let icon = BatteryIcon::from_percent(50.0);
        let debug = format!("{:?}", icon);
        assert!(debug.contains("BatteryIcon"));
    }

    // =========================================================================
    // BatteryHealth tests
    // =========================================================================

    #[test]
    fn test_battery_health_from_capacity_good() {
        let health = BatteryHealth::from_capacity(5000, 4500);
        assert_eq!(health, BatteryHealth::Good);
    }

    #[test]
    fn test_battery_health_from_capacity_fair() {
        let health = BatteryHealth::from_capacity(5000, 3000);
        assert_eq!(health, BatteryHealth::Fair);
    }

    #[test]
    fn test_battery_health_from_capacity_poor() {
        let health = BatteryHealth::from_capacity(5000, 2000);
        assert_eq!(health, BatteryHealth::Poor);
    }

    #[test]
    fn test_battery_health_from_capacity_zero() {
        let health = BatteryHealth::from_capacity(0, 0);
        assert_eq!(health, BatteryHealth::Unknown);
    }

    #[test]
    fn test_battery_health_color() {
        let color = BatteryHealth::Good.color();
        assert!(color.g > 0.8);

        let color = BatteryHealth::Poor.color();
        assert!(color.r > 0.9);
    }

    #[test]
    fn test_battery_health_display_name() {
        assert_eq!(BatteryHealth::Good.display_name(), "Good");
        assert_eq!(BatteryHealth::Fair.display_name(), "Fair");
        assert_eq!(BatteryHealth::Poor.display_name(), "Poor");
    }

    #[test]
    fn test_battery_health_derive_debug() {
        let health = BatteryHealth::Good;
        let debug = format!("{:?}", health);
        assert!(debug.contains("Good"));
    }
}
