//! Colorblind-safe color scheme based on Viridis.

use crate::monitor::memory::PressureLevel;

// ============================================================================
// Color Scheme (Colorblind-Safe Viridis-Based)
// ============================================================================

/// Colorblind-safe color scheme based on Viridis
#[derive(Debug, Clone)]
pub struct ColorScheme {
    /// OK/Success color (teal)
    pub ok: RgbColor,
    /// Warning color (yellow)
    pub warning: RgbColor,
    /// Critical color (red-orange)
    pub critical: RgbColor,
    /// Neutral color (blue)
    pub neutral: RgbColor,
    /// Background color (dark purple)
    pub background: RgbColor,
}

impl Default for ColorScheme {
    fn default() -> Self {
        Self {
            ok: RgbColor::new(0x21, 0x91, 0x8c),         // Teal
            warning: RgbColor::new(0xfd, 0xe7, 0x25),    // Yellow
            critical: RgbColor::new(0xf0, 0x3b, 0x20),   // Red-orange
            neutral: RgbColor::new(0x3b, 0x52, 0x8b),    // Blue
            background: RgbColor::new(0x44, 0x01, 0x54), // Dark purple
        }
    }
}

/// RGB color
#[derive(Debug, Clone, Copy)]
pub struct RgbColor {
    /// Red component (0-255)
    pub r: u8,
    /// Green component (0-255)
    pub g: u8,
    /// Blue component (0-255)
    pub b: u8,
}

impl RgbColor {
    /// Create a new RGB color
    #[must_use]
    pub const fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }

    /// Convert to ANSI true-color escape sequence (foreground)
    #[must_use]
    pub fn to_ansi_fg(&self) -> String {
        format!("\x1b[38;2;{};{};{}m", self.r, self.g, self.b)
    }

    /// Convert to ANSI true-color escape sequence (background)
    #[must_use]
    pub fn to_ansi_bg(&self) -> String {
        format!("\x1b[48;2;{};{};{}m", self.r, self.g, self.b)
    }

    /// Get color for pressure level
    #[must_use]
    pub fn for_pressure_level(level: PressureLevel) -> Self {
        match level {
            PressureLevel::Ok => Self::new(0x21, 0x91, 0x8c), // Teal
            PressureLevel::Elevated => Self::new(0xfd, 0xe7, 0x25), // Yellow
            PressureLevel::Warning => Self::new(0xfd, 0xa6, 0x00), // Orange
            PressureLevel::Critical => Self::new(0xf0, 0x3b, 0x20), // Red
        }
    }
}
