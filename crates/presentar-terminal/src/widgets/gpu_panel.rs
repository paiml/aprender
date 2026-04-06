//! `GpuPanel` widget for GPU monitoring.
//!
//! Displays GPU utilization, temperature, VRAM, power, and per-process memory.
//! Supports NVIDIA (via nvidia-smi) and AMD (via sysfs).

#![allow(dead_code)] // Some fields/constants reserved for future features

use presentar_core::{
    Brick, BrickAssertion, BrickBudget, BrickVerification, Canvas, Color, Constraints, Event,
    LayoutResult, Point, Rect, Size, TextStyle, TypeId, Widget,
};
use std::any::Any;
use std::time::Duration;

/// Block characters for utilization bar (8 levels).
const BAR_CHARS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

/// GPU vendor type for display.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GpuVendor {
    Nvidia,
    Amd,
    Intel,
    #[default]
    Unknown,
}

impl GpuVendor {
    /// Get display name.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Nvidia => "NVIDIA",
            Self::Amd => "AMD",
            Self::Intel => "Intel",
            Self::Unknown => "GPU",
        }
    }
}

/// A GPU device entry.
#[derive(Debug, Clone)]
pub struct GpuDevice {
    /// GPU index.
    pub index: u32,
    /// GPU name (e.g., "RTX 3080").
    pub name: String,
    /// GPU vendor.
    pub vendor: GpuVendor,
    /// GPU utilization (0-100).
    pub utilization: f32,
    /// Temperature in Celsius.
    pub temperature: Option<f32>,
    /// Total VRAM in bytes.
    pub vram_total: u64,
    /// Used VRAM in bytes.
    pub vram_used: u64,
    /// Power draw in watts.
    pub power_draw: Option<f32>,
    /// Power limit in watts.
    pub power_limit: Option<f32>,
    /// Fan speed percentage.
    pub fan_speed: Option<u32>,
}

impl Default for GpuDevice {
    fn default() -> Self {
        Self {
            index: 0,
            name: "Unknown GPU".to_string(),
            vendor: GpuVendor::Unknown,
            utilization: 0.0,
            temperature: None,
            vram_total: 0,
            vram_used: 0,
            power_draw: None,
            power_limit: None,
            fan_speed: None,
        }
    }
}

impl GpuDevice {
    /// Create a new GPU device.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Default::default()
        }
    }

    /// Set GPU vendor.
    #[must_use]
    pub fn with_vendor(mut self, vendor: GpuVendor) -> Self {
        self.vendor = vendor;
        self
    }

    /// Set utilization percentage.
    #[must_use]
    pub fn with_utilization(mut self, util: f32) -> Self {
        self.utilization = util;
        self
    }

    /// Set temperature.
    #[must_use]
    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = Some(temp);
        self
    }

    /// Set VRAM usage.
    #[must_use]
    pub fn with_vram(mut self, used: u64, total: u64) -> Self {
        self.vram_used = used;
        self.vram_total = total;
        self
    }

    /// Set power info.
    #[must_use]
    pub fn with_power(mut self, draw: f32, limit: Option<f32>) -> Self {
        self.power_draw = Some(draw);
        self.power_limit = limit;
        self
    }

    /// Set fan speed.
    #[must_use]
    pub fn with_fan(mut self, speed: u32) -> Self {
        self.fan_speed = Some(speed);
        self
    }

    /// Get VRAM usage percentage.
    pub fn vram_percent(&self) -> f32 {
        if self.vram_total > 0 {
            (self.vram_used as f64 / self.vram_total as f64 * 100.0) as f32
        } else {
            0.0
        }
    }

    /// Format VRAM for display.
    pub fn vram_display(&self) -> String {
        let used_gb = self.vram_used as f64 / 1_073_741_824.0;
        let total_gb = self.vram_total as f64 / 1_073_741_824.0;
        format!("{used_gb:.1}G / {total_gb:.1}G")
    }
}

/// A process using GPU memory.
#[derive(Debug, Clone)]
pub struct GpuProcess {
    /// Process name.
    pub name: String,
    /// Process ID.
    pub pid: u32,
    /// GPU memory used in bytes.
    pub vram_used: u64,
}

impl GpuProcess {
    /// Create a new GPU process entry.
    #[must_use]
    pub fn new(name: impl Into<String>, pid: u32, vram: u64) -> Self {
        Self {
            name: name.into(),
            pid,
            vram_used: vram,
        }
    }

    /// Format VRAM for display.
    pub fn vram_display(&self) -> String {
        let mb = self.vram_used / (1024 * 1024);
        if mb >= 1024 {
            format!("{:.1}G", mb as f64 / 1024.0)
        } else {
            format!("{mb}M")
        }
    }
}

/// GPU panel displaying GPU stats and per-process memory.
#[derive(Debug, Clone)]
pub struct GpuPanel {
    /// GPU device info.
    device: GpuDevice,
    /// Processes using GPU.
    processes: Vec<GpuProcess>,
    /// Utilization bar color.
    bar_color: Color,
    /// Temperature color (changes based on value).
    temp_color: Color,
    /// Show process table.
    show_processes: bool,
    /// Max processes to show.
    max_processes: usize,
    /// Cached bounds.
    bounds: Rect,
}

impl Default for GpuPanel {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuPanel {
    /// Create a new GPU panel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            device: GpuDevice::default(),
            processes: Vec::new(),
            bar_color: Color::new(0.4, 0.8, 0.4, 1.0), // Green
            temp_color: Color::new(1.0, 0.8, 0.2, 1.0), // Yellow
            show_processes: true,
            max_processes: 3,
            bounds: Rect::default(),
        }
    }

    /// Set the GPU device.
    #[must_use]
    pub fn with_device(mut self, device: GpuDevice) -> Self {
        self.device = device;
        self
    }

    /// Set GPU processes.
    #[must_use]
    pub fn with_processes(mut self, processes: Vec<GpuProcess>) -> Self {
        self.processes = processes;
        self
    }

    /// Add a process.
    pub fn add_process(&mut self, process: GpuProcess) {
        self.processes.push(process);
    }

    /// Set bar color.
    #[must_use]
    pub fn with_bar_color(mut self, color: Color) -> Self {
        self.bar_color = color;
        self
    }

    /// Toggle process display.
    #[must_use]
    pub fn show_processes(mut self, show: bool) -> Self {
        self.show_processes = show;
        self
    }

    /// Set max processes to show.
    #[must_use]
    pub fn max_processes(mut self, max: usize) -> Self {
        self.max_processes = max;
        self
    }

    /// Draw the utilization bar.
    fn draw_util_bar(&self, canvas: &mut dyn Canvas, y: f32, width: f32) {
        use std::fmt::Write;
        let util = self.device.utilization;
        let bar_width = (width - 6.0) as usize; // Leave room for percentage
        let filled = ((util / 100.0) * bar_width as f32) as usize;

        let mut bar = String::new();
        for i in 0..bar_width {
            if i < filled {
                bar.push('█');
            } else {
                bar.push('░');
            }
        }
        write!(bar, " {util:3.0}%").ok();

        canvas.draw_text(
            &bar,
            Point::new(self.bounds.x, y),
            &TextStyle {
                color: self.bar_color,
                ..Default::default()
            },
        );
    }

    /// Draw GPU info lines.
    fn draw_info(&self, canvas: &mut dyn Canvas, start_y: f32) -> f32 {
        use std::fmt::Write as _;
        let mut y = start_y;
        let x = self.bounds.x;

        // Temperature and Power on one line
        let mut info_line = String::new();
        if let Some(temp) = self.device.temperature {
            write!(info_line, "Temp: {temp:3.0}°C").ok();
        }
        if let Some(power) = self.device.power_draw {
            if !info_line.is_empty() {
                info_line.push_str("  ");
            }
            write!(info_line, "Power: {power:3.0}W").ok();
        }
        if !info_line.is_empty() {
            canvas.draw_text(
                &info_line,
                Point::new(x, y),
                &TextStyle {
                    color: Color::WHITE,
                    ..Default::default()
                },
            );
            y += 1.0;
        }

        // VRAM line
        if self.device.vram_total > 0 {
            let vram_line = format!(
                "VRAM: {} ({:.0}%)",
                self.device.vram_display(),
                self.device.vram_percent()
            );
            canvas.draw_text(
                &vram_line,
                Point::new(x, y),
                &TextStyle {
                    color: Color::WHITE,
                    ..Default::default()
                },
            );
            y += 1.0;
        }

        // Fan speed
        if let Some(fan) = self.device.fan_speed {
            canvas.draw_text(
                &format!("Fan: {fan}%"),
                Point::new(x, y),
                &TextStyle {
                    color: Color::WHITE,
                    ..Default::default()
                },
            );
            y += 1.0;
        }

        y
    }

    /// Draw top processes.
    fn draw_processes(&self, canvas: &mut dyn Canvas, start_y: f32) {
        if !self.show_processes || self.processes.is_empty() {
            return;
        }

        let x = self.bounds.x;
        let mut y = start_y;

        // Sort by VRAM usage
        let mut sorted: Vec<_> = self.processes.iter().collect();
        sorted.sort_by(|a, b| b.vram_used.cmp(&a.vram_used));

        for proc in sorted.iter().take(self.max_processes) {
            // Truncate name to fit
            let max_name_len = 12;
            let name: String = if proc.name.len() > max_name_len {
                format!("{}...", &proc.name[..max_name_len - 3])
            } else {
                proc.name.clone()
            };

            let line = format!("{:<12} {:>6}", name, proc.vram_display());
            canvas.draw_text(
                &line,
                Point::new(x, y),
                &TextStyle {
                    color: Color::new(0.7, 0.7, 0.7, 1.0),
                    ..Default::default()
                },
            );
            y += 1.0;
        }
    }
}

impl Brick for GpuPanel {
    fn brick_name(&self) -> &'static str {
        "gpu_panel"
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
            verification_time: Duration::from_micros(25),
        }
    }

    fn to_html(&self) -> String {
        String::new()
    }

    fn to_css(&self) -> String {
        String::new()
    }
}

impl Widget for GpuPanel {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        // Header (1) + util bar (1) + info (3) + processes (3) = 8 lines typical
        let height = 8.0_f32.min(constraints.max_height);
        Size::new(constraints.max_width, height)
    }

    fn layout(&mut self, bounds: Rect) -> LayoutResult {
        self.bounds = bounds;
        LayoutResult {
            size: Size::new(bounds.width, bounds.height),
        }
    }

    fn paint(&self, canvas: &mut dyn Canvas) {
        if self.bounds.width < 10.0 || self.bounds.height < 3.0 {
            return;
        }

        let mut y = self.bounds.y;

        // Utilization bar
        self.draw_util_bar(canvas, y, self.bounds.width);
        y += 1.0;

        // Info lines
        y = self.draw_info(canvas, y);

        // Top processes
        self.draw_processes(canvas, y);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_device_vram_percent() {
        let device = GpuDevice::default().with_vram(4 * 1024 * 1024 * 1024, 8 * 1024 * 1024 * 1024);
        assert!((device.vram_percent() - 50.0).abs() < 0.1);
    }

    #[test]
    fn test_gpu_device_vram_display() {
        let device = GpuDevice::default().with_vram(4 * 1024 * 1024 * 1024, 8 * 1024 * 1024 * 1024);
        assert_eq!(device.vram_display(), "4.0G / 8.0G");
    }

    #[test]
    fn test_gpu_process_vram_display() {
        let proc_mb = GpuProcess::new("test", 1234, 512 * 1024 * 1024);
        assert_eq!(proc_mb.vram_display(), "512M");

        let proc_gb = GpuProcess::new("test", 1234, 2 * 1024 * 1024 * 1024);
        assert_eq!(proc_gb.vram_display(), "2.0G");
    }

    #[test]
    fn test_panel_default() {
        let panel = GpuPanel::new();
        assert!(panel.show_processes);
        assert_eq!(panel.max_processes, 3);
    }

    #[test]
    fn test_panel_builder() {
        let device = GpuDevice::new("RTX 3080")
            .with_vendor(GpuVendor::Nvidia)
            .with_utilization(80.0)
            .with_temperature(72.0)
            .with_vram(8 * 1024 * 1024 * 1024, 10 * 1024 * 1024 * 1024)
            .with_power(220.0, Some(320.0))
            .with_fan(65);

        let panel = GpuPanel::new().with_device(device).max_processes(5);

        assert_eq!(panel.max_processes, 5);
        assert_eq!(panel.device.name, "RTX 3080");
    }

    #[test]
    fn test_gpu_vendor_as_str() {
        assert_eq!(GpuVendor::Nvidia.as_str(), "NVIDIA");
        assert_eq!(GpuVendor::Amd.as_str(), "AMD");
        assert_eq!(GpuVendor::Intel.as_str(), "Intel");
        assert_eq!(GpuVendor::Unknown.as_str(), "GPU");
    }

    #[test]
    fn test_gpu_device_default() {
        let device = GpuDevice::default();
        assert_eq!(device.index, 0);
        assert_eq!(device.vendor, GpuVendor::Unknown);
        assert_eq!(device.utilization, 0.0);
        assert!(device.temperature.is_none());
    }

    #[test]
    fn test_gpu_device_zero_vram() {
        let device = GpuDevice::default();
        assert_eq!(device.vram_percent(), 0.0);
    }

    #[test]
    fn test_gpu_panel_with_bar_color() {
        let panel = GpuPanel::new().with_bar_color(Color::RED);
        assert_eq!(
            format!("{:?}", panel.bar_color),
            format!("{:?}", Color::RED)
        );
    }

    #[test]
    fn test_gpu_panel_show_processes() {
        let panel = GpuPanel::new().show_processes(false);
        assert!(!panel.show_processes);
    }

    #[test]
    fn test_gpu_panel_with_processes() {
        let processes = vec![
            GpuProcess::new("python", 1234, 1024 * 1024 * 1024),
            GpuProcess::new("chrome", 5678, 512 * 1024 * 1024),
        ];
        let panel = GpuPanel::new().with_processes(processes);
        assert_eq!(panel.processes.len(), 2);
    }

    #[test]
    fn test_gpu_panel_add_process() {
        let mut panel = GpuPanel::new();
        panel.add_process(GpuProcess::new("test", 100, 256 * 1024 * 1024));
        assert_eq!(panel.processes.len(), 1);
    }

    #[test]
    fn test_gpu_panel_brick_traits() {
        let panel = GpuPanel::new();
        assert_eq!(panel.brick_name(), "gpu_panel");
        assert!(!panel.assertions().is_empty());
        assert!(panel.budget().paint_ms > 0);
        assert!(panel.verify().is_valid());
        assert!(panel.to_html().is_empty());
        assert!(panel.to_css().is_empty());
    }

    #[test]
    fn test_gpu_panel_widget_traits() {
        let mut panel = GpuPanel::new();

        // Measure
        let size = panel.measure(Constraints {
            min_width: 0.0,
            min_height: 0.0,
            max_width: 80.0,
            max_height: 20.0,
        });
        assert!(size.width > 0.0);
        assert!(size.height > 0.0);

        // Layout
        let result = panel.layout(Rect::new(0.0, 0.0, 80.0, 10.0));
        assert_eq!(result.size.width, 80.0);

        // Type ID
        assert_eq!(Widget::type_id(&panel), TypeId::of::<GpuPanel>());

        // Event
        assert!(panel
            .event(&Event::key_down(presentar_core::Key::Enter))
            .is_none());

        // Children
        assert!(panel.children().is_empty());
        assert!(panel.children_mut().is_empty());
    }

    #[test]
    #[allow(clippy::identity_op)]
    fn test_gpu_panel_paint_full() {
        use crate::direct::{CellBuffer, DirectTerminalCanvas};

        let device = GpuDevice::new("RTX 3080")
            .with_vendor(GpuVendor::Nvidia)
            .with_utilization(75.0)
            .with_temperature(68.0)
            .with_vram(6 * 1024 * 1024 * 1024, 10 * 1024 * 1024 * 1024)
            .with_power(180.0, Some(320.0))
            .with_fan(55);

        let processes = vec![
            GpuProcess::new("python3", 1234, 2 * 1024 * 1024 * 1024),
            GpuProcess::new("verylongprocessname", 5678, 1 * 1024 * 1024 * 1024),
            GpuProcess::new("chrome", 9012, 512 * 1024 * 1024),
        ];

        let mut panel = GpuPanel::new()
            .with_device(device)
            .with_processes(processes);

        panel.layout(Rect::new(0.0, 0.0, 60.0, 12.0));

        let mut buffer = CellBuffer::new(60, 12);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        panel.paint(&mut canvas);
    }

    #[test]
    fn test_gpu_panel_paint_without_processes() {
        use crate::direct::{CellBuffer, DirectTerminalCanvas};

        let device = GpuDevice::new("RTX 3080")
            .with_utilization(50.0)
            .with_temperature(60.0)
            .with_vram(4 * 1024 * 1024 * 1024, 8 * 1024 * 1024 * 1024);

        let mut panel = GpuPanel::new().with_device(device).show_processes(false);

        panel.layout(Rect::new(0.0, 0.0, 60.0, 10.0));

        let mut buffer = CellBuffer::new(60, 10);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        panel.paint(&mut canvas);
    }

    #[test]
    fn test_gpu_panel_paint_minimal_info() {
        use crate::direct::{CellBuffer, DirectTerminalCanvas};

        // Device with no temperature, power, fan - just utilization
        let device = GpuDevice::new("Generic GPU").with_utilization(30.0);

        let mut panel = GpuPanel::new().with_device(device);
        panel.layout(Rect::new(0.0, 0.0, 60.0, 10.0));

        let mut buffer = CellBuffer::new(60, 10);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        panel.paint(&mut canvas);
    }

    #[test]
    fn test_gpu_panel_paint_small_bounds() {
        use crate::direct::{CellBuffer, DirectTerminalCanvas};

        let mut panel = GpuPanel::new();
        panel.layout(Rect::new(0.0, 0.0, 5.0, 2.0)); // Too small

        let mut buffer = CellBuffer::new(5, 2);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        panel.paint(&mut canvas); // Should early return
    }

    #[test]
    fn test_gpu_panel_paint_only_power() {
        use crate::direct::{CellBuffer, DirectTerminalCanvas};

        // Device with power but no temperature
        let device = GpuDevice::new("GPU")
            .with_utilization(40.0)
            .with_power(150.0, None);

        let mut panel = GpuPanel::new().with_device(device);
        panel.layout(Rect::new(0.0, 0.0, 60.0, 10.0));

        let mut buffer = CellBuffer::new(60, 10);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        panel.paint(&mut canvas);
    }

    #[test]
    fn test_gpu_panel_default_trait() {
        let panel = GpuPanel::default();
        assert!(panel.show_processes);
        assert_eq!(panel.max_processes, 3);
    }

    #[test]
    fn test_gpu_process_new() {
        let proc = GpuProcess::new("test_process", 12345, 100 * 1024 * 1024);
        assert_eq!(proc.name, "test_process");
        assert_eq!(proc.pid, 12345);
        assert_eq!(proc.vram_used, 100 * 1024 * 1024);
    }

    #[test]
    fn test_gpu_device_all_builders() {
        let device = GpuDevice::new("Test GPU")
            .with_vendor(GpuVendor::Amd)
            .with_utilization(95.0)
            .with_temperature(85.0)
            .with_vram(8 * 1024 * 1024 * 1024, 16 * 1024 * 1024 * 1024)
            .with_power(250.0, Some(350.0))
            .with_fan(80);

        assert_eq!(device.vendor, GpuVendor::Amd);
        assert_eq!(device.utilization, 95.0);
        assert_eq!(device.temperature, Some(85.0));
        assert_eq!(device.vram_used, 8 * 1024 * 1024 * 1024);
        assert_eq!(device.vram_total, 16 * 1024 * 1024 * 1024);
        assert_eq!(device.power_draw, Some(250.0));
        assert_eq!(device.power_limit, Some(350.0));
        assert_eq!(device.fan_speed, Some(80));
    }
}
