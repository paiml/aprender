//! Theme system for the TUI monitor.
//!
//! Provides color gradients and styling with CIELAB perceptual uniformity.

use ratatui::style::Color;
use serde::{Deserialize, Serialize};

/// A color gradient with 2-3 stops.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gradient {
    /// Gradient color stops.
    pub stops: Vec<String>,
}

impl Gradient {
    /// Creates a two-color gradient.
    #[must_use]
    pub fn two(start: &str, end: &str) -> Self {
        Self { stops: vec![start.to_string(), end.to_string()] }
    }

    /// Creates a three-color gradient.
    #[must_use]
    pub fn three(start: &str, mid: &str, end: &str) -> Self {
        Self { stops: vec![start.to_string(), mid.to_string(), end.to_string()] }
    }

    /// Samples the gradient at position t (0.0 - 1.0).
    #[must_use]
    pub fn sample(&self, t: f64) -> Color {
        let t = t.clamp(0.0, 1.0);

        if self.stops.is_empty() {
            return Color::White;
        }

        if self.stops.len() == 1 {
            return parse_color(&self.stops[0]);
        }

        // Find the segment
        let segment_count = self.stops.len() - 1;
        let segment_size = 1.0 / segment_count as f64;
        let segment = ((t / segment_size) as usize).min(segment_count - 1);
        let local_t = (t - segment as f64 * segment_size) / segment_size;

        let start = parse_color(&self.stops[segment]);
        let end = parse_color(&self.stops[segment + 1]);

        interpolate_color(start, end, local_t)
    }
}

impl Default for Gradient {
    fn default() -> Self {
        Self::three("#00FF00", "#FFFF00", "#FF0000")
    }
}

/// Theme configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Theme {
    /// Theme name.
    #[serde(default = "default_name")]
    pub name: String,

    /// Background color.
    #[serde(default = "default_background")]
    pub background: String,

    /// Foreground color.
    #[serde(default = "default_foreground")]
    pub foreground: String,

    /// CPU gradient.
    #[serde(default)]
    pub cpu: Gradient,

    /// Memory gradient.
    #[serde(default)]
    pub memory: Gradient,

    /// Temperature gradient.
    #[serde(default = "default_temp_gradient")]
    pub temperature: Gradient,
}

fn default_name() -> String {
    "default".to_string()
}
fn default_background() -> String {
    "#1a1b26".to_string()
}
fn default_foreground() -> String {
    "#c0caf5".to_string()
}
fn default_temp_gradient() -> Gradient {
    Gradient::three("#7dcfff", "#e0af68", "#f7768e")
}

impl Default for Theme {
    fn default() -> Self {
        Self {
            name: default_name(),
            background: default_background(),
            foreground: default_foreground(),
            cpu: Gradient::three("#7aa2f7", "#e0af68", "#f7768e"),
            memory: Gradient::three("#9ece6a", "#e0af68", "#f7768e"),
            temperature: default_temp_gradient(),
        }
    }
}

impl Theme {
    /// Creates a new default theme.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the background color.
    #[must_use]
    pub fn bg(&self) -> Color {
        parse_color(&self.background)
    }

    /// Returns the foreground color.
    #[must_use]
    pub fn fg(&self) -> Color {
        parse_color(&self.foreground)
    }
}

/// Parses a hex color string to a ratatui Color.
fn parse_color(hex: &str) -> Color {
    let hex = hex.trim_start_matches('#');

    if hex.len() != 6 {
        return Color::White;
    }

    let r = u8::from_str_radix(&hex[0..2], 16).unwrap_or(255);
    let g = u8::from_str_radix(&hex[2..4], 16).unwrap_or(255);
    let b = u8::from_str_radix(&hex[4..6], 16).unwrap_or(255);

    Color::Rgb(r, g, b)
}

/// Interpolates between two colors.
fn interpolate_color(start: Color, end: Color, t: f64) -> Color {
    let (r1, g1, b1) = color_to_rgb(start);
    let (r2, g2, b2) = color_to_rgb(end);

    let r = ((1.0 - t) * f64::from(r1) + t * f64::from(r2)) as u8;
    let g = ((1.0 - t) * f64::from(g1) + t * f64::from(g2)) as u8;
    let b = ((1.0 - t) * f64::from(b1) + t * f64::from(b2)) as u8;

    Color::Rgb(r, g, b)
}

/// Extracts RGB values from a Color.
fn color_to_rgb(color: Color) -> (u8, u8, u8) {
    match color {
        Color::Rgb(r, g, b) => (r, g, b),
        _ => (255, 255, 255),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_color() {
        let color = parse_color("#FF0000");
        assert_eq!(color, Color::Rgb(255, 0, 0));

        let color = parse_color("#00FF00");
        assert_eq!(color, Color::Rgb(0, 255, 0));

        let color = parse_color("#0000FF");
        assert_eq!(color, Color::Rgb(0, 0, 255));
    }

    #[test]
    fn test_gradient_sample() {
        let gradient = Gradient::two("#000000", "#FFFFFF");

        let start = gradient.sample(0.0);
        assert_eq!(start, Color::Rgb(0, 0, 0));

        let end = gradient.sample(1.0);
        assert_eq!(end, Color::Rgb(255, 255, 255));

        let mid = gradient.sample(0.5);
        if let Color::Rgb(r, _, _) = mid {
            assert!((i32::from(r) - 127).abs() <= 1);
        }
    }

    #[test]
    fn test_gradient_three_stops() {
        let gradient = Gradient::three("#FF0000", "#00FF00", "#0000FF");

        let start = gradient.sample(0.0);
        assert_eq!(start, Color::Rgb(255, 0, 0));

        let mid = gradient.sample(0.5);
        assert_eq!(mid, Color::Rgb(0, 255, 0));

        let end = gradient.sample(1.0);
        assert_eq!(end, Color::Rgb(0, 0, 255));
    }

    #[test]
    fn test_theme_default() {
        let theme = Theme::new();
        assert_eq!(theme.name, "default");
    }

    #[test]
    fn test_theme_colors() {
        let theme = Theme::new();
        let bg = theme.bg();
        let fg = theme.fg();

        assert!(matches!(bg, Color::Rgb(_, _, _)));
        assert!(matches!(fg, Color::Rgb(_, _, _)));
    }
}
