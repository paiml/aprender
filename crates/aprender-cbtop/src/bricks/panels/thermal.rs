//! Thermal panel brick (Layer 3)
//!
//! Displays CPU and GPU temperature monitoring with warning thresholds.

use crate::brick::{Brick, BrickAssertion, BrickBudget, BrickVerification};
use presentar_core::{Canvas, Color, Point, Rect, TextStyle, Widget};
use presentar_terminal::{BrailleGraph, GraphMode, Meter, Theme};
use std::any::Any;

/// Temperature thresholds for thermal warnings
#[derive(Debug, Clone, Copy)]
pub struct ThermalThresholds {
    /// Warning temperature (yellow)
    pub warning: f64,
    /// Critical temperature (red)
    pub critical: f64,
    /// Maximum safe temperature
    pub max_safe: f64,
}

impl Default for ThermalThresholds {
    fn default() -> Self {
        Self {
            warning: 70.0,
            critical: 85.0,
            max_safe: 100.0,
        }
    }
}

/// Thermal metrics for a single sensor
#[derive(Debug, Clone, Default)]
pub struct SensorReading {
    /// Sensor name
    pub name: String,
    /// Current temperature in Celsius
    pub temp_c: f64,
}

/// Thermal panel for temperature monitoring
pub struct ThermalPanelBrick {
    /// CPU temperature history
    pub cpu_temp_history: Vec<f64>,
    /// GPU temperature history
    pub gpu_temp_history: Vec<f64>,
    /// Current CPU temperature
    pub cpu_temp: f64,
    /// Current GPU temperature
    pub gpu_temp: f64,
    /// Additional sensor readings
    pub sensors: Vec<SensorReading>,
    /// Temperature thresholds
    pub thresholds: ThermalThresholds,
    /// Theme for rendering
    pub theme: Theme,
}

impl ThermalPanelBrick {
    /// Create a new thermal panel
    pub fn new() -> Self {
        Self {
            cpu_temp_history: Vec::new(),
            gpu_temp_history: Vec::new(),
            cpu_temp: 0.0,
            gpu_temp: 0.0,
            sensors: Vec::new(),
            thresholds: ThermalThresholds::default(),
            theme: Theme::tokyo_night(),
        }
    }

    /// Get color for temperature value based on thresholds
    fn temp_color(&self, temp: f64) -> Color {
        if temp >= self.thresholds.critical {
            Color::new(1.0, 0.2, 0.2, 1.0) // Red
        } else if temp >= self.thresholds.warning {
            Color::new(1.0, 0.8, 0.2, 1.0) // Yellow/Orange
        } else {
            Color::new(0.3, 1.0, 0.5, 1.0) // Green
        }
    }

    /// Get status string for temperature
    fn temp_status(&self, temp: f64) -> &'static str {
        if temp >= self.thresholds.critical {
            "CRITICAL"
        } else if temp >= self.thresholds.warning {
            "WARNING"
        } else {
            "OK"
        }
    }

    /// Paint the thermal panel
    pub fn paint(&self, canvas: &mut dyn Canvas, width: f32, _height: f32) {
        let label_style = TextStyle {
            color: self.theme.foreground,
            ..Default::default()
        };
        let dim_style = TextStyle {
            color: self.theme.dim,
            ..Default::default()
        };

        canvas.draw_text("Thermal Monitor", Point::new(2.0, 2.0), &label_style);

        // CPU Temperature
        canvas.draw_text("CPU:", Point::new(2.0, 4.0), &dim_style);
        let cpu_color = self.temp_color(self.cpu_temp);
        let cpu_status = self.temp_status(self.cpu_temp);
        canvas.draw_text(
            &format!("{:.1}°C [{}]", self.cpu_temp, cpu_status),
            Point::new(8.0, 4.0),
            &TextStyle {
                color: cpu_color,
                ..Default::default()
            },
        );

        // CPU temperature gauge
        let cpu_pct = (self.cpu_temp / self.thresholds.max_safe * 100.0).min(100.0);
        let mut cpu_meter = Meter::new(cpu_pct, 100.0).with_color(cpu_color);
        cpu_meter.layout(Rect::new(2.0, 5.0, width - 20.0, 1.0));
        cpu_meter.paint(canvas);

        // CPU temperature history graph
        if !self.cpu_temp_history.is_empty() {
            let mut graph = BrailleGraph::new(self.cpu_temp_history.clone())
                .with_color(cpu_color)
                .with_range(0.0, self.thresholds.max_safe)
                .with_mode(GraphMode::Braille);
            graph.layout(Rect::new(2.0, 6.0, width - 4.0, 4.0));
            graph.paint(canvas);
        }

        // GPU Temperature
        canvas.draw_text("GPU:", Point::new(2.0, 11.0), &dim_style);
        let gpu_color = self.temp_color(self.gpu_temp);
        let gpu_status = self.temp_status(self.gpu_temp);
        canvas.draw_text(
            &format!("{:.1}°C [{}]", self.gpu_temp, gpu_status),
            Point::new(8.0, 11.0),
            &TextStyle {
                color: gpu_color,
                ..Default::default()
            },
        );

        // GPU temperature gauge
        let gpu_pct = (self.gpu_temp / self.thresholds.max_safe * 100.0).min(100.0);
        let mut gpu_meter = Meter::new(gpu_pct, 100.0).with_color(gpu_color);
        gpu_meter.layout(Rect::new(2.0, 12.0, width - 20.0, 1.0));
        gpu_meter.paint(canvas);

        // GPU temperature history graph
        if !self.gpu_temp_history.is_empty() {
            let mut graph = BrailleGraph::new(self.gpu_temp_history.clone())
                .with_color(gpu_color)
                .with_range(0.0, self.thresholds.max_safe)
                .with_mode(GraphMode::Braille);
            graph.layout(Rect::new(2.0, 13.0, width - 4.0, 4.0));
            graph.paint(canvas);
        }

        // Additional sensors
        if !self.sensors.is_empty() {
            canvas.draw_text("Other Sensors:", Point::new(2.0, 18.0), &dim_style);
            for (i, sensor) in self.sensors.iter().enumerate().take(4) {
                let y = 19.0 + i as f32;
                let color = self.temp_color(sensor.temp_c);
                canvas.draw_text(
                    &format!("{}: {:.1}°C", sensor.name, sensor.temp_c),
                    Point::new(2.0, y),
                    &TextStyle {
                        color,
                        ..Default::default()
                    },
                );
            }
        }
    }
}

impl Default for ThermalPanelBrick {
    fn default() -> Self {
        Self::new()
    }
}

impl Brick for ThermalPanelBrick {
    fn brick_name(&self) -> &'static str {
        "thermal_panel"
    }

    fn assertions(&self) -> Vec<BrickAssertion> {
        vec![
            BrickAssertion::MinWidth(40),
            BrickAssertion::MinHeight(20),
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
    fn test_thermal_panel_brick_name() {
        let panel = ThermalPanelBrick::new();
        assert_eq!(panel.brick_name(), "thermal_panel");
    }

    #[test]
    fn test_thermal_panel_has_assertions() {
        let panel = ThermalPanelBrick::new();
        assert!(!panel.assertions().is_empty());
    }

    #[test]
    fn test_temp_color_thresholds() {
        let panel = ThermalPanelBrick::new();

        // Below warning should be green-ish
        let ok_color = panel.temp_color(50.0);
        assert!(ok_color.g > ok_color.r);

        // Warning should be yellow-ish
        let warn_color = panel.temp_color(75.0);
        assert!(warn_color.r > 0.5 && warn_color.g > 0.5);

        // Critical should be red-ish
        let crit_color = panel.temp_color(90.0);
        assert!(crit_color.r > crit_color.g);
    }

    #[test]
    fn test_temp_status() {
        let panel = ThermalPanelBrick::new();
        assert_eq!(panel.temp_status(50.0), "OK");
        assert_eq!(panel.temp_status(75.0), "WARNING");
        assert_eq!(panel.temp_status(90.0), "CRITICAL");
    }
}
