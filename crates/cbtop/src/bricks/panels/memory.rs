//! Memory panel brick (Layer 3)
//!
//! Displays RAM, swap, and per-process memory usage.

use crate::brick::{Brick, BrickAssertion, BrickBudget, BrickVerification};
use presentar_core::{Canvas, Point, Rect, TextStyle, Widget};
use presentar_terminal::{Meter, Theme};
use std::any::Any;

/// Memory metrics for display
#[derive(Debug, Clone, Default)]
pub struct MemoryMetrics {
    /// Total RAM in bytes
    pub total_ram: u64,
    /// Used RAM in bytes
    pub used_ram: u64,
    /// Total swap in bytes
    pub total_swap: u64,
    /// Used swap in bytes
    pub used_swap: u64,
    /// Cached memory in bytes
    pub cached: u64,
    /// Buffer memory in bytes
    pub buffers: u64,
}

impl MemoryMetrics {
    /// RAM usage as percentage
    pub fn ram_percent(&self) -> f64 {
        if self.total_ram > 0 {
            (self.used_ram as f64 / self.total_ram as f64) * 100.0
        } else {
            0.0
        }
    }

    /// Swap usage as percentage
    pub fn swap_percent(&self) -> f64 {
        if self.total_swap > 0 {
            (self.used_swap as f64 / self.total_swap as f64) * 100.0
        } else {
            0.0
        }
    }

    /// Format bytes as human-readable
    pub fn format_bytes(bytes: u64) -> String {
        batuta_common::fmt::format_bytes(bytes)
    }
}

/// Memory panel for system memory monitoring
pub struct MemoryPanelBrick {
    /// Current memory metrics
    pub metrics: MemoryMetrics,
    /// Theme for rendering
    pub theme: Theme,
}

impl MemoryPanelBrick {
    /// Create a new memory panel
    pub fn new() -> Self {
        Self { metrics: MemoryMetrics::default(), theme: Theme::tokyo_night() }
    }

    /// Paint the memory panel
    pub fn paint(&self, canvas: &mut dyn Canvas, width: f32, _height: f32) {
        let label_style = TextStyle { color: self.theme.foreground, ..Default::default() };
        let dim_style = TextStyle { color: self.theme.dim, ..Default::default() };

        canvas.draw_text("Memory Monitor", Point::new(2.0, 2.0), &label_style);

        // RAM usage
        let ram_pct = self.metrics.ram_percent();
        canvas.draw_text("RAM:", Point::new(2.0, 4.0), &dim_style);

        let mut ram_meter = Meter::new(ram_pct, 100.0).with_color(self.theme.memory_color(ram_pct));
        ram_meter.layout(Rect::new(10.0, 4.0, width - 30.0, 1.0));
        ram_meter.paint(canvas);

        let ram_info = format!(
            "{} / {} ({:.1}%)",
            MemoryMetrics::format_bytes(self.metrics.used_ram),
            MemoryMetrics::format_bytes(self.metrics.total_ram),
            ram_pct
        );
        let ram_style = TextStyle { color: self.theme.memory_color(ram_pct), ..Default::default() };
        canvas.draw_text(&ram_info, Point::new(2.0, 5.0), &ram_style);

        // Swap usage
        let swap_pct = self.metrics.swap_percent();
        canvas.draw_text("Swap:", Point::new(2.0, 7.0), &dim_style);

        let mut swap_meter =
            Meter::new(swap_pct, 100.0).with_color(self.theme.memory_color(swap_pct));
        swap_meter.layout(Rect::new(10.0, 7.0, width - 30.0, 1.0));
        swap_meter.paint(canvas);

        let swap_info = format!(
            "{} / {} ({:.1}%)",
            MemoryMetrics::format_bytes(self.metrics.used_swap),
            MemoryMetrics::format_bytes(self.metrics.total_swap),
            swap_pct
        );
        let swap_style =
            TextStyle { color: self.theme.memory_color(swap_pct), ..Default::default() };
        canvas.draw_text(&swap_info, Point::new(2.0, 8.0), &swap_style);

        // Cache/Buffer info
        canvas.draw_text("Cached:", Point::new(2.0, 10.0), &dim_style);
        canvas.draw_text(
            &MemoryMetrics::format_bytes(self.metrics.cached),
            Point::new(12.0, 10.0),
            &label_style,
        );

        canvas.draw_text("Buffers:", Point::new(2.0, 11.0), &dim_style);
        canvas.draw_text(
            &MemoryMetrics::format_bytes(self.metrics.buffers),
            Point::new(12.0, 11.0),
            &label_style,
        );
    }
}

impl Default for MemoryPanelBrick {
    fn default() -> Self {
        Self::new()
    }
}

impl Brick for MemoryPanelBrick {
    fn brick_name(&self) -> &'static str {
        "memory_panel"
    }

    fn assertions(&self) -> Vec<BrickAssertion> {
        vec![
            BrickAssertion::MinWidth(40),
            BrickAssertion::MinHeight(12),
            BrickAssertion::max_latency_ms(8),
        ]
    }

    fn budget(&self) -> BrickBudget {
        BrickBudget::FRAME_60FPS
    }

    fn verify(&self) -> BrickVerification {
        let mut v = BrickVerification::new();
        for assertion in self.assertions() {
            v.check(&assertion);
        }
        v
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_panel_brick_name() {
        let panel = MemoryPanelBrick::new();
        assert_eq!(panel.brick_name(), "memory_panel");
    }

    #[test]
    fn test_memory_panel_has_assertions() {
        let panel = MemoryPanelBrick::new();
        assert!(!panel.assertions().is_empty());
    }

    #[test]
    fn test_memory_metrics_percent() {
        let metrics = MemoryMetrics {
            total_ram: 16_000_000_000,
            used_ram: 8_000_000_000,
            total_swap: 4_000_000_000,
            used_swap: 1_000_000_000,
            cached: 2_000_000_000,
            buffers: 500_000_000,
        };
        assert!((metrics.ram_percent() - 50.0).abs() < 0.01);
        assert!((metrics.swap_percent() - 25.0).abs() < 0.01);
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(MemoryMetrics::format_bytes(500), "500 B");
        assert_eq!(MemoryMetrics::format_bytes(1024), "1.0 KB");
        assert_eq!(MemoryMetrics::format_bytes(1_048_576), "1.0 MB");
        assert_eq!(MemoryMetrics::format_bytes(1_073_741_824), "1.0 GB");
    }
}
