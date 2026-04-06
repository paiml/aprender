//! PSI (Pressure Stall Information) panel rendering and utilities.
//!
//! Provides PSI panel title building, pressure formatting,
//! and helper functions for rendering system pressure metrics.

use presentar_core::Color;

// =============================================================================
// PSI TITLE BUILDING
// =============================================================================

/// Build PSI panel title string.
///
/// Format: "PSI │ CPU: ▂ │ IO: ▅ │ Mem: ▁"
#[must_use]
pub fn build_psi_title(cpu_pct: f64, io_pct: f64, mem_pct: f64) -> String {
    format!(
        "PSI │ CPU: {} │ IO: {} │ Mem: {}",
        pressure_symbol(cpu_pct),
        pressure_symbol(io_pct),
        pressure_symbol(mem_pct)
    )
}

/// Build compact PSI title for narrow panels.
///
/// Format: "PSI │ ▂▅▁"
#[must_use]
pub fn build_psi_title_compact(cpu_pct: f64, io_pct: f64, mem_pct: f64) -> String {
    format!(
        "PSI │ {}{}{}",
        pressure_symbol(cpu_pct),
        pressure_symbol(io_pct),
        pressure_symbol(mem_pct)
    )
}

// =============================================================================
// PRESSURE SYMBOLS
// =============================================================================

/// Get Unicode block symbol for pressure percentage (8 levels).
#[must_use]
pub fn pressure_symbol(percent: f64) -> &'static str {
    match percent {
        p if p >= 87.5 => "█",
        p if p >= 75.0 => "▇",
        p if p >= 62.5 => "▆",
        p if p >= 50.0 => "▅",
        p if p >= 37.5 => "▄",
        p if p >= 25.0 => "▃",
        p if p >= 12.5 => "▂",
        p if p > 0.0 => "▁",
        _ => " ",
    }
}

/// Get extended pressure symbol with numeric hint.
#[must_use]
pub fn pressure_symbol_labeled(percent: f64) -> String {
    format!("{}{:.0}%", pressure_symbol(percent), percent)
}

// =============================================================================
// PRESSURE COLORS
// =============================================================================

/// Get color for pressure percentage.
#[must_use]
pub fn pressure_color(percent: f64) -> Color {
    if percent >= 75.0 {
        Color::new(1.0, 0.3, 0.3, 1.0) // Critical red
    } else if percent >= 50.0 {
        Color::new(1.0, 0.6, 0.2, 1.0) // Warning orange
    } else if percent >= 25.0 {
        Color::new(1.0, 0.8, 0.2, 1.0) // Yellow
    } else if percent > 0.0 {
        Color::new(0.3, 0.9, 0.5, 1.0) // Green
    } else {
        Color::new(0.4, 0.4, 0.4, 1.0) // Gray (idle)
    }
}

/// Get background color for pressure bar.
#[must_use]
pub fn pressure_bar_bg_color() -> Color {
    Color::new(0.2, 0.2, 0.2, 1.0)
}

// =============================================================================
// PRESSURE DATA TYPES
// =============================================================================

/// PSI metric type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PsiMetricType {
    /// CPU pressure
    Cpu,
    /// Memory pressure
    Memory,
    /// I/O pressure
    Io,
}

impl PsiMetricType {
    /// Get display name.
    #[must_use]
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Cpu => "CPU",
            Self::Memory => "Memory",
            Self::Io => "I/O",
        }
    }

    /// Get short name.
    #[must_use]
    pub fn short_name(&self) -> &'static str {
        match self {
            Self::Cpu => "CPU",
            Self::Memory => "Mem",
            Self::Io => "IO",
        }
    }

    /// Get icon.
    #[must_use]
    pub fn icon(&self) -> &'static str {
        match self {
            Self::Cpu => "󰻠",
            Self::Memory => "󰍛",
            Self::Io => "󰋊",
        }
    }
}

/// PSI time window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PsiWindow {
    /// 10 second average
    #[default]
    Avg10,
    /// 60 second average
    Avg60,
    /// 300 second (5 minute) average
    Avg300,
}

impl PsiWindow {
    /// Get display label.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::Avg10 => "10s",
            Self::Avg60 => "60s",
            Self::Avg300 => "5m",
        }
    }

    /// Get numeric seconds.
    #[must_use]
    pub fn seconds(&self) -> u32 {
        match self {
            Self::Avg10 => 10,
            Self::Avg60 => 60,
            Self::Avg300 => 300,
        }
    }
}

// =============================================================================
// PSI STATS
// =============================================================================

/// PSI statistics for a single metric.
#[derive(Debug, Clone, Default)]
pub struct PsiStats {
    /// 10-second average percentage
    pub avg10: f64,
    /// 60-second average percentage
    pub avg60: f64,
    /// 300-second average percentage
    pub avg300: f64,
    /// Total stall time in microseconds
    pub total_usec: u64,
}

impl PsiStats {
    /// Create new PSI stats.
    #[must_use]
    pub fn new(avg10: f64, avg60: f64, avg300: f64, total_usec: u64) -> Self {
        Self {
            avg10,
            avg60,
            avg300,
            total_usec,
        }
    }

    /// Get the average for a given window.
    #[must_use]
    pub fn get_avg(&self, window: PsiWindow) -> f64 {
        match window {
            PsiWindow::Avg10 => self.avg10,
            PsiWindow::Avg60 => self.avg60,
            PsiWindow::Avg300 => self.avg300,
        }
    }

    /// Get the highest pressure across all windows.
    #[must_use]
    pub fn max_pressure(&self) -> f64 {
        self.avg10.max(self.avg60).max(self.avg300)
    }

    /// Check if any pressure is significant.
    #[must_use]
    pub fn has_pressure(&self) -> bool {
        self.avg10 > 0.1 || self.avg60 > 0.1 || self.avg300 > 0.1
    }
}

// =============================================================================
// SEVERITY ASSESSMENT
// =============================================================================

/// Assess PSI severity level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PsiSeverity {
    /// No significant pressure
    Normal,
    /// Minor pressure, worth monitoring
    Low,
    /// Noticeable pressure affecting performance
    Medium,
    /// High pressure, system struggling
    High,
    /// Critical pressure, system severely impacted
    Critical,
}

impl PsiSeverity {
    /// Determine severity from percentage.
    #[must_use]
    pub fn from_percent(percent: f64) -> Self {
        if percent >= 75.0 {
            Self::Critical
        } else if percent >= 50.0 {
            Self::High
        } else if percent >= 25.0 {
            Self::Medium
        } else if percent >= 5.0 {
            Self::Low
        } else {
            Self::Normal
        }
    }

    /// Get color for severity.
    #[must_use]
    pub fn color(&self) -> Color {
        match self {
            Self::Critical => Color::new(1.0, 0.2, 0.2, 1.0),
            Self::High => Color::new(1.0, 0.5, 0.2, 1.0),
            Self::Medium => Color::new(1.0, 0.8, 0.2, 1.0),
            Self::Low => Color::new(0.5, 0.9, 0.4, 1.0),
            Self::Normal => Color::new(0.4, 0.7, 0.4, 1.0),
        }
    }

    /// Get display label.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::Critical => "CRITICAL",
            Self::High => "HIGH",
            Self::Medium => "MEDIUM",
            Self::Low => "LOW",
            Self::Normal => "OK",
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
    // build_psi_title tests
    // =========================================================================

    #[test]
    fn test_build_psi_title_basic() {
        let title = build_psi_title(10.0, 50.0, 5.0);
        assert!(title.contains("PSI"));
        assert!(title.contains("CPU"));
        assert!(title.contains("IO"));
        assert!(title.contains("Mem"));
    }

    #[test]
    fn test_build_psi_title_zero() {
        let title = build_psi_title(0.0, 0.0, 0.0);
        assert!(
            title.contains(" "),
            "Zero pressure should have space symbol"
        );
    }

    #[test]
    fn test_build_psi_title_high() {
        let title = build_psi_title(90.0, 90.0, 90.0);
        assert!(title.contains("█"));
    }

    // =========================================================================
    // build_psi_title_compact tests
    // =========================================================================

    #[test]
    fn test_build_psi_title_compact() {
        let title = build_psi_title_compact(25.0, 50.0, 75.0);
        assert!(title.contains("PSI"));
        assert!(title.chars().count() < 20);
    }

    // =========================================================================
    // pressure_symbol tests
    // =========================================================================

    #[test]
    fn test_pressure_symbol_zero() {
        assert_eq!(pressure_symbol(0.0), " ");
    }

    #[test]
    fn test_pressure_symbol_low() {
        assert_eq!(pressure_symbol(5.0), "▁");
    }

    #[test]
    fn test_pressure_symbol_medium() {
        assert_eq!(pressure_symbol(50.0), "▅");
    }

    #[test]
    fn test_pressure_symbol_high() {
        assert_eq!(pressure_symbol(90.0), "█");
    }

    #[test]
    fn test_pressure_symbol_boundaries() {
        assert_eq!(pressure_symbol(12.5), "▂");
        assert_eq!(pressure_symbol(25.0), "▃");
        assert_eq!(pressure_symbol(37.5), "▄");
        assert_eq!(pressure_symbol(62.5), "▆");
        assert_eq!(pressure_symbol(75.0), "▇");
        assert_eq!(pressure_symbol(87.5), "█");
    }

    #[test]
    fn test_pressure_symbol_labeled() {
        let result = pressure_symbol_labeled(50.0);
        assert!(result.contains("▅"));
        assert!(result.contains("50%"));
    }

    // =========================================================================
    // pressure_color tests
    // =========================================================================

    #[test]
    fn test_pressure_color_idle() {
        let color = pressure_color(0.0);
        assert!((color.r - color.g).abs() < 0.1, "Idle should be gray");
    }

    #[test]
    fn test_pressure_color_low() {
        let color = pressure_color(10.0);
        assert!(color.g > 0.8, "Low should be green");
    }

    #[test]
    fn test_pressure_color_medium() {
        let color = pressure_color(30.0);
        assert!(color.r > 0.9 && color.g > 0.7, "Medium should be yellow");
    }

    #[test]
    fn test_pressure_color_high() {
        let color = pressure_color(60.0);
        assert!(color.r > 0.9, "High should be orange");
    }

    #[test]
    fn test_pressure_color_critical() {
        let color = pressure_color(80.0);
        assert!(color.r > 0.9 && color.g < 0.5, "Critical should be red");
    }

    #[test]
    fn test_pressure_bar_bg_color() {
        let color = pressure_bar_bg_color();
        assert!(color.r < 0.3 && color.g < 0.3 && color.b < 0.3);
    }

    // =========================================================================
    // PsiMetricType tests
    // =========================================================================

    #[test]
    fn test_psi_metric_type_display_name() {
        assert_eq!(PsiMetricType::Cpu.display_name(), "CPU");
        assert_eq!(PsiMetricType::Memory.display_name(), "Memory");
        assert_eq!(PsiMetricType::Io.display_name(), "I/O");
    }

    #[test]
    fn test_psi_metric_type_short_name() {
        assert_eq!(PsiMetricType::Cpu.short_name(), "CPU");
        assert_eq!(PsiMetricType::Memory.short_name(), "Mem");
        assert_eq!(PsiMetricType::Io.short_name(), "IO");
    }

    #[test]
    fn test_psi_metric_type_icon() {
        assert!(!PsiMetricType::Cpu.icon().is_empty());
        assert!(!PsiMetricType::Memory.icon().is_empty());
    }

    #[test]
    fn test_psi_metric_type_derive_debug() {
        let metric = PsiMetricType::Cpu;
        let debug = format!("{:?}", metric);
        assert!(debug.contains("Cpu"));
    }

    // =========================================================================
    // PsiWindow tests
    // =========================================================================

    #[test]
    fn test_psi_window_label() {
        assert_eq!(PsiWindow::Avg10.label(), "10s");
        assert_eq!(PsiWindow::Avg60.label(), "60s");
        assert_eq!(PsiWindow::Avg300.label(), "5m");
    }

    #[test]
    fn test_psi_window_seconds() {
        assert_eq!(PsiWindow::Avg10.seconds(), 10);
        assert_eq!(PsiWindow::Avg60.seconds(), 60);
        assert_eq!(PsiWindow::Avg300.seconds(), 300);
    }

    #[test]
    fn test_psi_window_default() {
        assert_eq!(PsiWindow::default(), PsiWindow::Avg10);
    }

    // =========================================================================
    // PsiStats tests
    // =========================================================================

    #[test]
    fn test_psi_stats_new() {
        let stats = PsiStats::new(10.0, 20.0, 30.0, 1000);
        assert_eq!(stats.avg10, 10.0);
        assert_eq!(stats.avg60, 20.0);
        assert_eq!(stats.avg300, 30.0);
        assert_eq!(stats.total_usec, 1000);
    }

    #[test]
    fn test_psi_stats_get_avg() {
        let stats = PsiStats::new(10.0, 20.0, 30.0, 0);
        assert_eq!(stats.get_avg(PsiWindow::Avg10), 10.0);
        assert_eq!(stats.get_avg(PsiWindow::Avg60), 20.0);
        assert_eq!(stats.get_avg(PsiWindow::Avg300), 30.0);
    }

    #[test]
    fn test_psi_stats_max_pressure() {
        let stats = PsiStats::new(10.0, 50.0, 30.0, 0);
        assert_eq!(stats.max_pressure(), 50.0);
    }

    #[test]
    fn test_psi_stats_has_pressure() {
        let stats = PsiStats::new(10.0, 0.0, 0.0, 0);
        assert!(stats.has_pressure());

        let no_pressure = PsiStats::default();
        assert!(!no_pressure.has_pressure());
    }

    #[test]
    fn test_psi_stats_default() {
        let stats = PsiStats::default();
        assert_eq!(stats.avg10, 0.0);
        assert_eq!(stats.total_usec, 0);
    }

    #[test]
    fn test_psi_stats_derive_debug() {
        let stats = PsiStats::new(10.0, 20.0, 30.0, 1000);
        let debug = format!("{:?}", stats);
        assert!(debug.contains("PsiStats"));
    }

    #[test]
    fn test_psi_stats_derive_clone() {
        let stats = PsiStats::new(10.0, 20.0, 30.0, 1000);
        let cloned = stats.clone();
        assert_eq!(cloned.avg10, stats.avg10);
    }

    // =========================================================================
    // PsiSeverity tests
    // =========================================================================

    #[test]
    fn test_psi_severity_from_percent_normal() {
        assert_eq!(PsiSeverity::from_percent(0.0), PsiSeverity::Normal);
        assert_eq!(PsiSeverity::from_percent(4.0), PsiSeverity::Normal);
    }

    #[test]
    fn test_psi_severity_from_percent_low() {
        assert_eq!(PsiSeverity::from_percent(10.0), PsiSeverity::Low);
    }

    #[test]
    fn test_psi_severity_from_percent_medium() {
        assert_eq!(PsiSeverity::from_percent(30.0), PsiSeverity::Medium);
    }

    #[test]
    fn test_psi_severity_from_percent_high() {
        assert_eq!(PsiSeverity::from_percent(60.0), PsiSeverity::High);
    }

    #[test]
    fn test_psi_severity_from_percent_critical() {
        assert_eq!(PsiSeverity::from_percent(80.0), PsiSeverity::Critical);
    }

    #[test]
    fn test_psi_severity_color() {
        let color = PsiSeverity::Critical.color();
        assert!(color.r > 0.9 && color.g < 0.3);

        let color = PsiSeverity::Normal.color();
        assert!(color.g > 0.6);
    }

    #[test]
    fn test_psi_severity_label() {
        assert_eq!(PsiSeverity::Critical.label(), "CRITICAL");
        assert_eq!(PsiSeverity::Normal.label(), "OK");
    }

    #[test]
    fn test_psi_severity_derive_debug() {
        let severity = PsiSeverity::High;
        let debug = format!("{:?}", severity);
        assert!(debug.contains("High"));
    }
}
