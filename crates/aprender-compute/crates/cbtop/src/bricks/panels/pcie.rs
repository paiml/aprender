//! PCIe panel brick (Layer 3)
//!
//! Displays PCIe bandwidth utilization and transfer statistics.

use crate::brick::{Brick, BrickAssertion, BrickBudget, BrickVerification};
use presentar_core::{Canvas, Point, Rect, TextStyle, Widget};
use presentar_terminal::{BrailleGraph, GraphMode, Meter, Sparkline, Theme};
use std::any::Any;

/// PCIe generation capabilities
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcieGen {
    Gen3,
    Gen4,
    Gen5,
}

impl PcieGen {
    /// Theoretical max bandwidth per lane in GB/s
    pub fn bandwidth_per_lane(&self) -> f64 {
        match self {
            Self::Gen3 => 0.985, // ~1 GB/s
            Self::Gen4 => 1.969, // ~2 GB/s
            Self::Gen5 => 3.938, // ~4 GB/s
        }
    }

    /// Display name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Gen3 => "Gen3",
            Self::Gen4 => "Gen4",
            Self::Gen5 => "Gen5",
        }
    }
}

/// PCIe device information
#[derive(Debug, Clone)]
pub struct PcieDevice {
    /// Device name
    pub name: String,
    /// PCIe generation
    pub gen: PcieGen,
    /// Number of lanes
    pub lanes: u8,
    /// Current TX bandwidth in GB/s
    pub tx_bandwidth: f64,
    /// Current RX bandwidth in GB/s
    pub rx_bandwidth: f64,
}

impl PcieDevice {
    /// Maximum theoretical bandwidth in GB/s
    pub fn max_bandwidth(&self) -> f64 {
        self.gen.bandwidth_per_lane() * self.lanes as f64
    }

    /// TX utilization percentage
    pub fn tx_utilization(&self) -> f64 {
        (self.tx_bandwidth / self.max_bandwidth() * 100.0).min(100.0)
    }

    /// RX utilization percentage
    pub fn rx_utilization(&self) -> f64 {
        (self.rx_bandwidth / self.max_bandwidth() * 100.0).min(100.0)
    }
}

/// PCIe panel for bandwidth monitoring
pub struct PciePanelBrick {
    /// GPU PCIe device
    pub gpu_device: Option<PcieDevice>,
    /// TX bandwidth history (GB/s)
    pub tx_history: Vec<f64>,
    /// RX bandwidth history (GB/s)
    pub rx_history: Vec<f64>,
    /// Total bytes transferred
    pub total_tx_bytes: u64,
    /// Total bytes received
    pub total_rx_bytes: u64,
    /// Theme for rendering
    pub theme: Theme,
}

impl PciePanelBrick {
    /// Create a new PCIe panel
    pub fn new() -> Self {
        Self {
            gpu_device: None,
            tx_history: Vec::new(),
            rx_history: Vec::new(),
            total_tx_bytes: 0,
            total_rx_bytes: 0,
            theme: Theme::tokyo_night(),
        }
    }

    /// Format bandwidth as human-readable
    fn format_bandwidth(gb_per_sec: f64) -> String {
        if gb_per_sec >= 1.0 {
            format!("{:.2} GB/s", gb_per_sec)
        } else {
            format!("{:.0} MB/s", gb_per_sec * 1024.0)
        }
    }

    /// Format bytes as human-readable
    fn format_bytes(bytes: u64) -> String {
        batuta_common::fmt::format_bytes_full(bytes)
    }

    /// Paint the PCIe panel
    pub fn paint(&self, canvas: &mut dyn Canvas, width: f32, _height: f32) {
        let label_style = TextStyle { color: self.theme.foreground, ..Default::default() };
        let dim_style = TextStyle { color: self.theme.dim, ..Default::default() };

        canvas.draw_text("PCIe Monitor", Point::new(2.0, 2.0), &label_style);

        if let Some(ref device) = self.gpu_device {
            // Device info
            canvas.draw_text(
                &format!(
                    "{} - PCIe {} x{} (Max: {})",
                    device.name,
                    device.gen.name(),
                    device.lanes,
                    Self::format_bandwidth(device.max_bandwidth())
                ),
                Point::new(2.0, 4.0),
                &label_style,
            );

            // TX (Host → GPU)
            canvas.draw_text("TX (H→G):", Point::new(2.0, 6.0), &dim_style);
            let tx_util = device.tx_utilization();
            let tx_color = self.theme.cpu_color(tx_util);
            let mut tx_meter = Meter::new(tx_util, 100.0).with_color(tx_color);
            tx_meter.layout(Rect::new(14.0, 6.0, width - 40.0, 1.0));
            tx_meter.paint(canvas);
            canvas.draw_text(
                &format!("{} ({:.1}%)", Self::format_bandwidth(device.tx_bandwidth), tx_util),
                Point::new(width - 24.0, 6.0),
                &TextStyle { color: tx_color, ..Default::default() },
            );

            // RX (GPU → Host)
            canvas.draw_text("RX (G→H):", Point::new(2.0, 7.0), &dim_style);
            let rx_util = device.rx_utilization();
            let rx_color = self.theme.cpu_color(rx_util);
            let mut rx_meter = Meter::new(rx_util, 100.0).with_color(rx_color);
            rx_meter.layout(Rect::new(14.0, 7.0, width - 40.0, 1.0));
            rx_meter.paint(canvas);
            canvas.draw_text(
                &format!("{} ({:.1}%)", Self::format_bandwidth(device.rx_bandwidth), rx_util),
                Point::new(width - 24.0, 7.0),
                &TextStyle { color: rx_color, ..Default::default() },
            );
        } else {
            canvas.draw_text("No PCIe GPU detected", Point::new(2.0, 4.0), &dim_style);
        }

        // Bandwidth history graphs
        if !self.tx_history.is_empty() {
            canvas.draw_text("TX History:", Point::new(2.0, 9.0), &dim_style);
            let max_bw = self.gpu_device.as_ref().map(|d| d.max_bandwidth()).unwrap_or(16.0);
            let mut tx_graph = BrailleGraph::new(self.tx_history.clone())
                .with_color(self.theme.cpu_color(50.0))
                .with_range(0.0, max_bw)
                .with_mode(GraphMode::Braille);
            tx_graph.layout(Rect::new(2.0, 10.0, width - 4.0, 3.0));
            tx_graph.paint(canvas);
        }

        if !self.rx_history.is_empty() {
            canvas.draw_text("RX History:", Point::new(2.0, 14.0), &dim_style);
            let max_bw = self.gpu_device.as_ref().map(|d| d.max_bandwidth()).unwrap_or(16.0);
            let mut rx_sparkline = Sparkline::new(self.rx_history.clone())
                .with_color(self.theme.cpu_color(50.0))
                .with_range(0.0, max_bw);
            rx_sparkline.layout(Rect::new(2.0, 15.0, width - 4.0, 1.0));
            rx_sparkline.paint(canvas);
        }

        // Transfer totals
        canvas.draw_text("Total TX:", Point::new(2.0, 17.0), &dim_style);
        canvas.draw_text(
            &Self::format_bytes(self.total_tx_bytes),
            Point::new(14.0, 17.0),
            &label_style,
        );
        canvas.draw_text("Total RX:", Point::new(2.0, 18.0), &dim_style);
        canvas.draw_text(
            &Self::format_bytes(self.total_rx_bytes),
            Point::new(14.0, 18.0),
            &label_style,
        );
    }
}

impl Default for PciePanelBrick {
    fn default() -> Self {
        Self::new()
    }
}

impl Brick for PciePanelBrick {
    fn brick_name(&self) -> &'static str {
        "pcie_panel"
    }

    fn assertions(&self) -> Vec<BrickAssertion> {
        vec![
            BrickAssertion::MinWidth(50),
            BrickAssertion::MinHeight(18),
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
    fn test_pcie_panel_brick_name() {
        let panel = PciePanelBrick::new();
        assert_eq!(panel.brick_name(), "pcie_panel");
    }

    #[test]
    fn test_pcie_panel_has_assertions() {
        let panel = PciePanelBrick::new();
        assert!(!panel.assertions().is_empty());
    }

    #[test]
    fn test_pcie_gen_bandwidth() {
        assert!((PcieGen::Gen3.bandwidth_per_lane() - 0.985).abs() < 0.01);
        assert!((PcieGen::Gen4.bandwidth_per_lane() - 1.969).abs() < 0.01);
        assert!((PcieGen::Gen5.bandwidth_per_lane() - 3.938).abs() < 0.01);
    }

    #[test]
    fn test_pcie_device_utilization() {
        let device = PcieDevice {
            name: "GPU".to_string(),
            gen: PcieGen::Gen4,
            lanes: 16,
            tx_bandwidth: 15.75, // Half of max
            rx_bandwidth: 7.875, // Quarter of max
        };
        assert!((device.max_bandwidth() - 31.5).abs() < 0.1);
        assert!((device.tx_utilization() - 50.0).abs() < 1.0);
        assert!((device.rx_utilization() - 25.0).abs() < 1.0);
    }

    #[test]
    fn test_format_bandwidth() {
        assert_eq!(PciePanelBrick::format_bandwidth(1.5), "1.50 GB/s");
        assert_eq!(PciePanelBrick::format_bandwidth(0.5), "512 MB/s");
    }
}
