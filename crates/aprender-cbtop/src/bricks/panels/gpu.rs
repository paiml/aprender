//! GPU panel brick (Layer 3)
//!
//! Displays GPU metrics from GpuCollectorBrick (Genchi Genbutsu: real data).
//!
//! Integrates with trueno-gpu/CUPTI for:
//! - GPU utilization (SM activity)
//! - VRAM usage (used/total)
//! - Temperature monitoring
//! - Power consumption

use crate::brick::{Brick, BrickAssertion, BrickBudget, BrickVerification};
use crate::bricks::collectors::gpu::GpuMetrics;
use presentar_core::{Canvas, Point, Rect, TextStyle, Widget};
use presentar_terminal::{BrailleGraph, GraphMode, Theme};
use std::any::Any;

/// GPU panel displaying real-time GPU metrics
pub struct GpuPanelBrick {
    /// GPU utilization history for graph
    pub gpu_data: Vec<f64>,
    /// Theme for styling
    pub theme: Theme,
    /// Current GPU metrics from collector
    pub current_metrics: Option<GpuMetrics>,
    /// Temperature (from nvidia-smi or CUPTI when available)
    pub temperature_c: Option<u32>,
    /// Power usage in watts (from nvidia-smi or CUPTI when available)
    pub power_watts: Option<u32>,
    /// Power limit in watts
    pub power_limit_watts: Option<u32>,
}

impl GpuPanelBrick {
    pub fn new() -> Self {
        Self {
            gpu_data: Vec::new(),
            theme: Theme::tokyo_night(),
            current_metrics: None,
            temperature_c: None,
            power_watts: None,
            power_limit_watts: None,
        }
    }

    /// Update panel with metrics from GpuCollectorBrick
    pub fn update_from_metrics(&mut self, metrics: &GpuMetrics) {
        // Add utilization to history (keep last 120 samples)
        self.gpu_data.push(metrics.utilization_gpu as f64);
        if self.gpu_data.len() > 120 {
            self.gpu_data.remove(0);
        }
        self.current_metrics = Some(metrics.clone());
    }

    /// Update temperature (from nvidia-smi or CUPTI)
    pub fn update_temperature(&mut self, temp_c: u32) {
        self.temperature_c = Some(temp_c);
    }

    /// Update power metrics (from nvidia-smi or CUPTI)
    pub fn update_power(&mut self, watts: u32, limit_watts: u32) {
        self.power_watts = Some(watts);
        self.power_limit_watts = Some(limit_watts);
    }

    pub fn paint(&self, canvas: &mut dyn Canvas, width: f32, _height: f32) {
        let label_style = TextStyle {
            color: self.theme.foreground,
            ..Default::default()
        };
        let dim_style = TextStyle {
            color: self.theme.dim,
            ..Default::default()
        };

        canvas.draw_text("GPU Monitor", Point::new(2.0, 2.0), &label_style);

        // Main graph
        if !self.gpu_data.is_empty() {
            let gpu_usage = self.gpu_data.last().copied().unwrap_or(0.0);
            let mut graph = BrailleGraph::new(self.gpu_data.clone())
                .with_color(self.theme.gpu_color(gpu_usage))
                .with_range(0.0, 100.0)
                .with_mode(GraphMode::Braille);
            graph.layout(Rect::new(2.0, 3.0, width - 4.0, 8.0));
            graph.paint(canvas);
        }

        // GPU info - real data from collector when available
        let (device_name, vram_used_gb, vram_total_gb, util_pct) =
            if let Some(ref m) = self.current_metrics {
                (
                    m.device_name.as_str(),
                    m.memory_used_mb as f64 / 1024.0,
                    m.memory_total_mb as f64 / 1024.0,
                    m.utilization_gpu as f64,
                )
            } else {
                ("No GPU detected", 0.0, 0.0, 0.0)
            };

        let data_source = if self.current_metrics.is_some() {
            "CUPTI"
        } else {
            "none"
        };
        canvas.draw_text(
            &format!("GPU Info ({})", data_source),
            Point::new(2.0, 12.0),
            &label_style,
        );

        canvas.draw_text("Device: ", Point::new(2.0, 13.0), &dim_style);
        canvas.draw_text(device_name, Point::new(10.0, 13.0), &label_style);

        // VRAM with memory gradient
        canvas.draw_text("VRAM: ", Point::new(2.0, 14.0), &dim_style);
        let vram_pct = if vram_total_gb > 0.0 {
            (vram_used_gb / vram_total_gb) * 100.0
        } else {
            0.0
        };
        let vram_style = TextStyle {
            color: self.theme.memory_color(vram_pct),
            ..Default::default()
        };
        canvas.draw_text(
            &format!(
                "{:.1} / {:.1} GB ({:.0}%)",
                vram_used_gb, vram_total_gb, vram_pct
            ),
            Point::new(8.0, 14.0),
            &vram_style,
        );

        // Utilization
        canvas.draw_text("Util: ", Point::new(2.0, 15.0), &dim_style);
        let util_style = TextStyle {
            color: self.theme.gpu_color(util_pct),
            ..Default::default()
        };
        canvas.draw_text(
            &format!("{:.0}%", util_pct),
            Point::new(8.0, 15.0),
            &util_style,
        );

        // Temperature display with color gradient
        canvas.draw_text("Temp: ", Point::new(16.0, 15.0), &dim_style);
        let temp = self.temperature_c.unwrap_or(0);
        let temp_style = TextStyle {
            color: self.theme.temp_color(temp as f64, 100.0),
            ..Default::default()
        };
        canvas.draw_text(&format!("{} C", temp), Point::new(22.0, 15.0), &temp_style);

        // Power with cpu gradient (reuse for power)
        canvas.draw_text("Power: ", Point::new(2.0, 16.0), &dim_style);
        let power = self.power_watts.unwrap_or(0);
        let power_limit = self.power_limit_watts.unwrap_or(1);
        let power_pct = (power as f64 / power_limit as f64) * 100.0;
        let power_style = TextStyle {
            color: self.theme.cpu_color(power_pct),
            ..Default::default()
        };
        canvas.draw_text(
            &format!("{}W / {}W ({:.0}%)", power, power_limit, power_pct),
            Point::new(9.0, 16.0),
            &power_style,
        );
    }
}

impl Brick for GpuPanelBrick {
    fn brick_name(&self) -> &'static str {
        "gpu_panel"
    }

    fn assertions(&self) -> Vec<BrickAssertion> {
        vec![
            BrickAssertion::MinWidth(40),
            BrickAssertion::MinHeight(15),
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

impl Default for GpuPanelBrick {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use presentar_core::RecordingCanvas;
    use std::time::Instant;

    #[test]
    fn test_gpu_panel_brick_name() {
        let panel = GpuPanelBrick::new();
        assert_eq!(panel.brick_name(), "gpu_panel");
    }

    #[test]
    fn test_gpu_panel_paint_empty() {
        let panel = GpuPanelBrick::new();
        let mut canvas = RecordingCanvas::new();

        panel.paint(&mut canvas, 80.0, 24.0);

        // Should draw header text even with no data
        assert!(!canvas.is_empty());
        assert!(canvas.command_count() >= 1);
    }

    #[test]
    fn test_gpu_panel_paint_with_data() {
        let mut panel = GpuPanelBrick::new();

        // Add some GPU data
        let metrics = GpuMetrics {
            timestamp: Instant::now(),
            device_index: 0,
            device_name: "RTX 4090".to_string(),
            utilization_gpu: 75,
            memory_used_mb: 8192,
            memory_total_mb: 24576,
        };
        panel.update_from_metrics(&metrics);
        panel.update_temperature(72);
        panel.update_power(250, 350);

        let mut canvas = RecordingCanvas::new();
        panel.paint(&mut canvas, 80.0, 24.0);

        // Should have more commands with data
        assert!(canvas.command_count() >= 5);
    }

    #[test]
    fn test_gpu_panel_paint_with_graph() {
        let mut panel = GpuPanelBrick::new();

        // Add multiple data points for graph
        for i in 0..10 {
            let metrics = GpuMetrics {
                timestamp: Instant::now(),
                device_index: 0,
                device_name: "Test GPU".to_string(),
                utilization_gpu: (i * 10) as u32,
                memory_used_mb: 1000,
                memory_total_mb: 2000,
            };
            panel.update_from_metrics(&metrics);
        }

        let mut canvas = RecordingCanvas::new();
        panel.paint(&mut canvas, 80.0, 24.0);

        // Graph rendering adds more commands
        assert!(canvas.command_count() >= 5);
    }

    #[test]
    fn test_gpu_panel_has_assertions() {
        let panel = GpuPanelBrick::new();
        assert!(!panel.assertions().is_empty());
    }

    #[test]
    fn test_gpu_panel_update_from_metrics() {
        let mut panel = GpuPanelBrick::new();

        let metrics = GpuMetrics {
            timestamp: Instant::now(),
            device_index: 0,
            device_name: "RTX 4090".to_string(),
            utilization_gpu: 75,
            memory_used_mb: 8192,
            memory_total_mb: 24576,
        };

        panel.update_from_metrics(&metrics);

        assert_eq!(panel.gpu_data.len(), 1);
        assert_eq!(panel.gpu_data[0], 75.0);
        assert!(panel.current_metrics.is_some());
        let m = panel.current_metrics.as_ref().unwrap();
        assert_eq!(m.device_name, "RTX 4090");
    }

    #[test]
    fn test_gpu_panel_update_temperature() {
        let mut panel = GpuPanelBrick::new();
        panel.update_temperature(72);
        assert_eq!(panel.temperature_c, Some(72));
    }

    #[test]
    fn test_gpu_panel_update_power() {
        let mut panel = GpuPanelBrick::new();
        panel.update_power(250, 350);
        assert_eq!(panel.power_watts, Some(250));
        assert_eq!(panel.power_limit_watts, Some(350));
    }

    #[test]
    fn test_gpu_panel_history_limit() {
        let mut panel = GpuPanelBrick::new();

        // Add 130 samples (should cap at 120)
        for i in 0..130 {
            let metrics = GpuMetrics {
                timestamp: Instant::now(),
                device_index: 0,
                device_name: "Test GPU".to_string(),
                utilization_gpu: (i % 100) as u32,
                memory_used_mb: 1000,
                memory_total_mb: 2000,
            };
            panel.update_from_metrics(&metrics);
        }

        assert_eq!(panel.gpu_data.len(), 120);
        // First entry should be from i=10 (value 10.0), last from i=129 (value 29.0)
        assert_eq!(panel.gpu_data[0], 10.0);
        assert_eq!(panel.gpu_data[119], 29.0);
    }

    #[test]
    fn test_gpu_panel_default() {
        let panel = GpuPanelBrick::default();
        assert!(panel.gpu_data.is_empty());
        assert!(panel.current_metrics.is_none());
        assert!(panel.temperature_c.is_none());
        assert!(panel.power_watts.is_none());
    }
}
