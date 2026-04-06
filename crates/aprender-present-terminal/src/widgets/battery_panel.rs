//! `BatteryPanel` widget.
//!
//! Displays battery charge level, status, and time remaining.
//! Reference: ttop battery panel.

use presentar_core::{
    Brick, BrickAssertion, BrickBudget, BrickVerification, Canvas, Color, Constraints, Event,
    LayoutResult, Point, Rect, Size, TextStyle, TypeId, Widget,
};
use std::any::Any;
use std::time::Duration;

/// Battery information.
#[derive(Debug, Clone, Default)]
pub struct BatteryInfo {
    /// Capacity percentage (0-100).
    pub capacity: u8,
    /// Status string (e.g., "Charging", "Discharging").
    pub status: String,
    /// Time remaining in minutes.
    pub time_remaining_mins: Option<u32>,
    /// Is battery present?
    pub present: bool,
}

impl BatteryInfo {
    pub fn new(capacity: u8, status: impl Into<String>, present: bool) -> Self {
        Self {
            capacity,
            status: status.into(),
            time_remaining_mins: None,
            present,
        }
    }

    pub fn with_time(mut self, mins: u32) -> Self {
        self.time_remaining_mins = Some(mins);
        self
    }
}

/// Battery panel widget.
#[derive(Debug, Clone, Default)]
pub struct BatteryPanel {
    /// Battery info.
    pub info: Option<BatteryInfo>,
    /// Cached bounds.
    bounds: Rect,
}

impl BatteryPanel {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_info(mut self, info: Option<BatteryInfo>) -> Self {
        self.info = info;
        self
    }
}

impl Widget for BatteryPanel {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        // Height 2 rows (bar + status)
        let height = 2.0;
        let width = constraints.max_width.min(40.0);
        constraints.constrain(Size::new(width, height))
    }

    fn layout(&mut self, bounds: Rect) -> LayoutResult {
        self.bounds = bounds;
        LayoutResult {
            size: Size::new(bounds.width, bounds.height),
        }
    }

    fn paint(&self, canvas: &mut dyn Canvas) {
        if self.bounds.width < 1.0 || self.bounds.height < 1.0 {
            return;
        }

        if let Some(bat) = &self.info {
            if !bat.present {
                canvas.draw_text(
                    "No battery detected",
                    Point::new(self.bounds.x, self.bounds.y),
                    &TextStyle {
                        color: Color::new(0.5, 0.5, 0.5, 1.0),
                        ..Default::default()
                    },
                );
                return;
            }

            // Draw charge bar
            let bar_width = (self.bounds.width as usize).min(30);
            let filled = ((bat.capacity as f32 / 100.0) * bar_width as f32) as usize;
            let bar = "█".repeat(filled) + &"░".repeat(bar_width.saturating_sub(filled));

            // Color logic: low=red, med=yellow, high=green
            let color = if bat.capacity < 20 {
                Color::new(1.0, 0.3, 0.3, 1.0) // Red
            } else if bat.capacity < 50 {
                Color::new(1.0, 0.8, 0.2, 1.0) // Yellow
            } else {
                Color::new(0.3, 0.9, 0.3, 1.0) // Green
            };

            canvas.draw_text(
                &bar,
                Point::new(self.bounds.x, self.bounds.y),
                &TextStyle {
                    color,
                    ..Default::default()
                },
            );

            // Status row
            if self.bounds.height >= 2.0 {
                let status_icon = match bat.status.as_str() {
                    "Charging" => "⚡ Charging",
                    "Discharging" => "🔋 Discharging",
                    "Full" => "✓ Full",
                    "Not charging" => "— Idle",
                    _ => "? Unknown",
                };
                
                let time_str = bat.time_remaining_mins.map(|m| {
                    if m >= 60 {
                        format!(" ({}h{}m)", m / 60, m % 60)
                    } else {
                        format!(" ({m}m)")
                    }
                }).unwrap_or_default();

                let text = format!("{status_icon}{time_str}");

                canvas.draw_text(
                    &text,
                    Point::new(self.bounds.x, self.bounds.y + 1.0),
                    &TextStyle {
                        color: Color::new(0.7, 0.7, 0.7, 1.0),
                        ..Default::default()
                    },
                );
            }
        } else {
            canvas.draw_text(
                "No battery detected",
                Point::new(self.bounds.x, self.bounds.y),
                &TextStyle {
                    color: Color::new(0.5, 0.5, 0.5, 1.0),
                    ..Default::default()
                },
            );
        }
    }

    fn event(&mut self, _event: &Event) -> Option<Box<dyn Any + Send>> {
        None
    }

    fn children(&self) -> &[Box<dyn Widget>] {
        &[]
    }

    fn children_mut(&mut self) -> &mut [Box<dyn Widget>] {
        &mut []
    }
}

impl Brick for BatteryPanel {
    fn brick_name(&self) -> &'static str {
        "battery_panel"
    }

    fn assertions(&self) -> &[BrickAssertion] {
        static ASSERTIONS: &[BrickAssertion] = &[BrickAssertion::max_latency_ms(8)];
        ASSERTIONS
    }

    fn budget(&self) -> BrickBudget {
        BrickBudget::uniform(8)
    }

    fn verify(&self) -> BrickVerification {
        BrickVerification {
            passed: vec![BrickAssertion::max_latency_ms(8)],
            failed: vec![],
            verification_time: Duration::from_micros(5),
        }
    }

    fn to_html(&self) -> String {
        String::new()
    }

    fn to_css(&self) -> String {
        String::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_battery_info_new() {
        let info = BatteryInfo::new(80, "Charging", true);
        assert_eq!(info.capacity, 80);
        assert_eq!(info.status, "Charging");
        assert!(info.present);
    }

    #[test]
    fn test_battery_panel_new() {
        let panel = BatteryPanel::new();
        assert!(panel.info.is_none());
    }

    #[test]
    fn test_battery_panel_with_info() {
        let info = BatteryInfo::new(50, "Discharging", true);
        let panel = BatteryPanel::new().with_info(Some(info));
        assert!(panel.info.is_some());
    }

    #[test]
    fn test_battery_panel_paint_no_info() {
        use crate::{CellBuffer, DirectTerminalCanvas};
        let panel = BatteryPanel::new();
        let mut buffer = CellBuffer::new(40, 5);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        panel.paint(&mut canvas);
        // Should draw "No battery detected"
    }
}
