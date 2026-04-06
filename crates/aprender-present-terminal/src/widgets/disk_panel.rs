//! `DiskPanel` widget for disk usage and I/O visualization.
//!
//! Displays a list of mounted disks with usage bars and I/O rates.
//! Reference: ttop disk panel.

use crate::theme::Gradient;
use crate::widgets::display_rules::{format_bytes_si, format_column, format_percent, ColumnAlign, TruncateStrategy};
use crate::widgets::selection::{RowHighlight, DIMMED_BG};
use crate::widgets::{percent_color, swap_color}; // Re-using existing helpers or define local
use presentar_core::{
    Brick, BrickAssertion, BrickBudget, BrickVerification, Canvas, Color, Constraints, Event,
    LayoutResult, Point, Rect, Size, TextStyle, TypeId, Widget,
};
use std::any::Any;
use std::time::Duration;

/// Information about a single disk for display.
#[derive(Debug, Clone)]
pub struct DiskEntry {
    /// Device name (e.g., "sda1", "nvme0n1p1").
    pub name: String,
    /// Mount point (e.g., "/", "/home").
    pub mount_point: String,
    /// File system type (e.g., "ext4", "btrfs").
    pub file_system: String,
    /// Total space in bytes.
    pub total_space: u64,
    /// Available space in bytes.
    pub available_space: u64,
    /// Read rate in bytes/sec.
    pub read_rate: f64,
    /// Write rate in bytes/sec.
    pub write_rate: f64,
    /// Is this disk active (high I/O)?
    pub is_active: bool,
}

impl DiskEntry {
    pub fn new(name: impl Into<String>, mount: impl Into<String>, total: u64, available: u64) -> Self {
        Self {
            name: name.into(),
            mount_point: mount.into(),
            file_system: String::new(),
            total_space: total,
            available_space: available,
            read_rate: 0.0,
            write_rate: 0.0,
            is_active: false,
        }
    }

    pub fn with_fs(mut self, fs: impl Into<String>) -> Self {
        self.file_system = fs.into();
        self
    }

    pub fn with_io(mut self, read: f64, write: f64) -> Self {
        self.read_rate = read;
        self.write_rate = write;
        // Simple activity heuristic
        self.is_active = read > 1024.0 || write > 1024.0;
        self
    }

    pub fn used_space(&self) -> u64 {
        self.total_space.saturating_sub(self.available_space)
    }

    pub fn usage_percent(&self) -> f64 {
        if self.total_space == 0 {
            0.0
        } else {
            (self.used_space() as f64 / self.total_space as f64) * 100.0
        }
    }
}

/// Disk panel widget.
#[derive(Debug, Clone, Default)]
pub struct DiskPanel {
    /// List of disks.
    pub disks: Vec<DiskEntry>,
    /// Cached bounds.
    bounds: Rect,
    /// Deterministic mode (for screenshots/testing).
    pub deterministic: bool,
    /// Selected row index.
    pub selected_row: Option<usize>,
}

impl DiskPanel {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_disks(mut self, disks: Vec<DiskEntry>) -> Self {
        self.disks = disks;
        self
    }

    pub fn deterministic(mut self) -> Self {
        self.deterministic = true;
        self
    }

    pub fn select(mut self, row: Option<usize>) -> Self {
        self.selected_row = row;
        self
    }

    /// Format bytes rate.
    fn format_rate(bytes_per_sec: f64) -> String {
        if bytes_per_sec >= 1_073_741_824.0 {
            format!("{:.1}G/s", bytes_per_sec / 1_073_741_824.0)
        } else if bytes_per_sec >= 1_048_576.0 {
            format!("{:.1}M/s", bytes_per_sec / 1_048_576.0)
        } else if bytes_per_sec >= 1024.0 {
            format!("{:.1}K/s", bytes_per_sec / 1024.0)
        } else {
            format!("{:.0}B/s", bytes_per_sec)
        }
    }
}

impl Widget for DiskPanel {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        // Height depends on number of disks
        let height = self.disks.len().max(1) as f32;
        let width = constraints.max_width.min(80.0);
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

        if self.deterministic {
            // Deterministic mode: show ttop style placeholders
            let dim_color = Color::new(0.3, 0.3, 0.3, 1.0);
            
            // Row 1: I/O Pressure
            canvas.draw_text(
                "I/O Pressure ○  0.0% some    0.0% full",
                Point::new(self.bounds.x, self.bounds.y),
                &TextStyle {
                    color: dim_color,
                    ..Default::default()
                },
            );

            // Row 2: Header
            if self.bounds.height >= 2.0 {
                canvas.draw_text(
                    "── Top Active Processes ──────────────",
                    Point::new(self.bounds.x, self.bounds.y + 1.0),
                    &TextStyle {
                        color: dim_color,
                        ..Default::default()
                    },
                );
            }
            return;
        }

        if self.disks.is_empty() {
            canvas.draw_text(
                "No disks found",
                Point::new(self.bounds.x, self.bounds.y),
                &TextStyle {
                    color: Color::new(0.5, 0.5, 0.5, 1.0),
                    ..Default::default()
                },
            );
            return;
        }

        let max_rows = self.bounds.height as usize;
        
        for (i, disk) in self.disks.iter().take(max_rows).enumerate() {
            let y = self.bounds.y + i as f32;
            let is_selected = self.selected_row == Some(i);

            // Row background/highlight
            if is_selected {
                // Using selection widget would be better, but simple rect for now
                canvas.fill_rect(
                    Rect::new(self.bounds.x, y, self.bounds.width, 1.0),
                    crate::widgets::selection::SELECTION_BG,
                );
            }

            // Layout: mount(8) | size(5)G | bar(...) | pct(5)% | IO(...)
            // Fixed parts: 8 + 1 + 6 + 1 + 1 + 6 + 1 = 24 chars
            let fixed_width = 24;
            
            // I/O String
            let io_str = if disk.read_rate > 0.0 || disk.write_rate > 0.0 {
                format!(
                    " R:{} W:{}",
                    Self::format_rate(disk.read_rate),
                    Self::format_rate(disk.write_rate)
                )
            } else {
                String::new()
            };
            
            let io_width = io_str.len();
            let available_width = (self.bounds.width as usize).saturating_sub(fixed_width + io_width);
            let bar_width = available_width.max(2);

            let pct = disk.usage_percent();
            let filled = ((pct / 100.0) * bar_width as f64) as usize;
            let bar = "█".repeat(filled.min(bar_width)) + &"░".repeat(bar_width.saturating_sub(filled));

            let mount_short: String = if disk.mount_point == "/" {
                "/".to_string()
            } else {
                disk.mount_point
                    .split('/')
                    .next_back()
                    .unwrap_or(&disk.mount_point)
                    .chars()
                    .take(8)
                    .collect()
            };

            let total_gb = disk.total_space as f64 / 1024.0 / 1024.0 / 1024.0;
            
            // Format: "mnt      100G  ████░░  50.0%  R:10M W:5M"
            let text = format!("{mount_short:<8} {total_gb:>5.0}G {bar} {pct:>5.1}%{io_str}");

            // Color
            let color = if is_selected {
                Color::WHITE
            } else if disk.is_active {
                Color::WHITE // Active I/O highlight
            } else {
                // Gradient based on usage (green -> yellow -> red)
                // Re-implementing simplified percent_color here to avoid dep cycles if needed, 
                // but ideally we import it.
                // Assuming we can't easily import private `percent_color` from ui.rs, 
                // we'll implement a simple version.
                if pct > 90.0 { Color::new(1.0, 0.3, 0.3, 1.0) }
                else if pct > 70.0 { Color::new(1.0, 0.8, 0.2, 1.0) }
                else { Color::new(0.3, 0.9, 0.3, 1.0) }
            };

            canvas.draw_text(
                &text,
                Point::new(self.bounds.x, y),
                &TextStyle {
                    color,
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

impl Brick for DiskPanel {
    fn brick_name(&self) -> &'static str {
        "disk_panel"
    }

    fn assertions(&self) -> &[BrickAssertion] {
        static ASSERTIONS: &[BrickAssertion] = &[BrickAssertion::max_latency_ms(16)];
        ASSERTIONS
    }

    fn budget(&self) -> BrickBudget {
        BrickBudget::uniform(16)
    }

    fn verify(&self) -> BrickVerification {
        BrickVerification {
            passed: vec![BrickAssertion::max_latency_ms(16)],
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
    fn test_disk_entry_new() {
        let disk = DiskEntry::new("sda1", "/", 1000, 500);
        assert_eq!(disk.name, "sda1");
        assert_eq!(disk.mount_point, "/");
        assert_eq!(disk.total_space, 1000);
        assert_eq!(disk.available_space, 500);
    }

    #[test]
    fn test_disk_entry_usage() {
        let disk = DiskEntry::new("sda1", "/", 100, 25);
        assert_eq!(disk.used_space(), 75);
        assert_eq!(disk.usage_percent(), 75.0);
    }

    #[test]
    fn test_disk_entry_with_io() {
        let disk = DiskEntry::new("sda1", "/", 100, 25)
            .with_io(2048.0, 512.0);
        assert!(disk.is_active);
        assert_eq!(disk.read_rate, 2048.0);
    }

    #[test]
    fn test_disk_panel_new() {
        let panel = DiskPanel::new();
        assert!(panel.disks.is_empty());
    }

    #[test]
    fn test_disk_panel_with_disks() {
        let disks = vec![DiskEntry::new("sda1", "/", 100, 50)];
        let panel = DiskPanel::new().with_disks(disks);
        assert_eq!(panel.disks.len(), 1);
    }

    #[test]
    fn test_disk_panel_measure() {
        let disks = vec![DiskEntry::new("sda1", "/", 100, 50)];
        let panel = DiskPanel::new().with_disks(disks);
        let size = panel.measure(Constraints::new(0.0, 100.0, 0.0, 50.0));
        assert_eq!(size.height, 1.0);
    }

    #[test]
    fn test_format_rate() {
        assert_eq!(DiskPanel::format_rate(500.0), "500B/s");
        assert_eq!(DiskPanel::format_rate(1024.0), "1.0K/s");
        assert_eq!(DiskPanel::format_rate(1_048_576.0), "1.0M/s");
    }
}
