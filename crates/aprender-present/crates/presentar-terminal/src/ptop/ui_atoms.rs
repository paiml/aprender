//! Atomic widget helpers for ptop UI.
//!
//! Provides semantic color functions and text rendering helpers.

use crate::direct::DirectTerminalCanvas;
use crate::widgets::SemanticStatus;
use presentar_core::{Canvas, Color, Point, TextStyle};

// =============================================================================
// SEMANTIC COLORS (derived from SemanticStatus)
// =============================================================================

/// Get color for a usage percentage (high = bad, like CPU/Memory usage).
/// Uses SemanticStatus::from_usage() internally.
#[inline]
pub fn usage_color(pct: f64) -> Color {
    SemanticStatus::from_usage(pct).color()
}

// =============================================================================
// QUICK TEXT RENDERING
// =============================================================================

/// Draw text with a specific color (convenience wrapper).
#[inline]
pub fn draw_colored_text(
    canvas: &mut DirectTerminalCanvas<'_>,
    text: &str,
    x: f32,
    y: f32,
    color: Color,
) {
    canvas.draw_text(
        text,
        Point::new(x, y),
        &TextStyle {
            color,
            ..Default::default()
        },
    );
}

// =============================================================================
// SEVERITY TO COLOR
// =============================================================================

/// Convert a 0.0-1.0 severity to SemanticStatus.
/// 0.0-0.2 = Normal, 0.2-0.4 = Good, 0.4-0.6 = Warning, 0.6-0.8 = High, 0.8-1.0 = Critical
fn severity_to_status(severity: f64) -> SemanticStatus {
    if severity >= 0.8 {
        SemanticStatus::Critical
    } else if severity >= 0.6 {
        SemanticStatus::High
    } else if severity >= 0.4 {
        SemanticStatus::Warning
    } else if severity >= 0.2 {
        SemanticStatus::Good
    } else {
        SemanticStatus::Normal
    }
}

/// Get color from severity (convenience function).
#[inline]
pub fn severity_color(severity: f64) -> Color {
    severity_to_status(severity).color()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::direct::CellBuffer;

    #[test]
    fn test_usage_color_gradient() {
        // Low usage = green
        let low = usage_color(10.0);
        assert!(low.g > low.r, "Low usage should be greenish");

        // High usage = red
        let high = usage_color(95.0);
        assert!(high.r > high.g, "High usage should be reddish");

        // Mid usage checks
        let mid = usage_color(50.0);
        assert!(mid.r > 0.5 && mid.g > 0.5, "Mid usage should be yellowish");
    }

    #[test]
    fn test_severity_color() {
        let low = severity_color(0.1);
        let high = severity_color(0.9);
        assert!(low.g > low.r, "Low severity should be greenish");
        assert!(high.r > high.g, "High severity should be reddish");
    }

    #[test]
    fn test_draw_colored_text() {
        let mut buffer = CellBuffer::new(20, 1);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        draw_colored_text(&mut canvas, "Hello", 0.0, 0.0, Color::RED);
        let cell = buffer.get(0, 0).unwrap();
        assert_eq!(cell.symbol, "H");
    }
}
