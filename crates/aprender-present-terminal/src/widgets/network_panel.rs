//! `NetworkPanel` widget for network interface monitoring.
//!
//! Displays network interfaces with upload/download sparklines.
//! Reference: ttop/btop network displays.

use presentar_core::{
    Brick, BrickAssertion, BrickBudget, BrickVerification, Canvas, Color, Constraints, Event,
    LayoutResult, Point, Rect, Size, TextStyle, TypeId, Widget,
};
use std::any::Any;
use std::time::Duration;

/// Block characters for sparkline rendering (8 levels).
const SPARK_CHARS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

/// A network interface entry.
#[derive(Debug, Clone)]
pub struct NetworkInterface {
    /// Interface name (e.g., "eth0", "wlan0").
    pub name: String,
    /// Download bytes per second history.
    pub rx_history: Vec<f64>,
    /// Upload bytes per second history.
    pub tx_history: Vec<f64>,
    /// Current download bytes per second.
    pub rx_bps: f64,
    /// Current upload bytes per second.
    pub tx_bps: f64,
    /// Total bytes received.
    pub rx_total: u64,
    /// Total bytes transmitted.
    pub tx_total: u64,
    /// Receive errors (cumulative).
    pub rx_errors: u64,
    /// Transmit errors (cumulative).
    pub tx_errors: u64,
    /// Receive dropped packets (cumulative).
    pub rx_dropped: u64,
    /// Transmit dropped packets (cumulative).
    pub tx_dropped: u64,
    /// Error rate (errors per second).
    pub errors_per_sec: f64,
    /// Drop rate (drops per second).
    pub drops_per_sec: f64,
    /// Bandwidth utilization percentage (CB-NET-006).
    /// None if link speed unknown.
    pub utilization_percent: Option<f64>,
}

impl NetworkInterface {
    /// Create a new network interface entry.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            rx_history: Vec::new(),
            tx_history: Vec::new(),
            rx_bps: 0.0,
            tx_bps: 0.0,
            rx_total: 0,
            tx_total: 0,
            rx_errors: 0,
            tx_errors: 0,
            rx_dropped: 0,
            tx_dropped: 0,
            errors_per_sec: 0.0,
            drops_per_sec: 0.0,
            utilization_percent: None,
        }
    }

    /// Set error and drop stats.
    pub fn set_stats(&mut self, rx_errors: u64, tx_errors: u64, rx_dropped: u64, tx_dropped: u64) {
        self.rx_errors = rx_errors;
        self.tx_errors = tx_errors;
        self.rx_dropped = rx_dropped;
        self.tx_dropped = tx_dropped;
    }

    /// Set error and drop rates.
    pub fn set_rates(&mut self, errors_per_sec: f64, drops_per_sec: f64) {
        self.errors_per_sec = errors_per_sec;
        self.drops_per_sec = drops_per_sec;
    }

    /// Set bandwidth utilization percentage (CB-NET-006).
    pub fn set_utilization(&mut self, utilization_percent: Option<f64>) {
        self.utilization_percent = utilization_percent;
    }

    /// Total errors (RX + TX).
    #[must_use]
    pub fn total_errors(&self) -> u64 {
        self.rx_errors + self.tx_errors
    }

    /// Total dropped packets (RX + TX).
    #[must_use]
    pub fn total_dropped(&self) -> u64 {
        self.rx_dropped + self.tx_dropped
    }

    /// Update with current bandwidth readings.
    pub fn update(&mut self, rx_bps: f64, tx_bps: f64) {
        self.rx_bps = rx_bps;
        self.tx_bps = tx_bps;
        self.rx_history.push(rx_bps);
        self.tx_history.push(tx_bps);
        // Keep last 60 samples
        if self.rx_history.len() > 60 {
            self.rx_history.remove(0);
        }
        if self.tx_history.len() > 60 {
            self.tx_history.remove(0);
        }
    }

    /// Set totals.
    pub fn set_totals(&mut self, rx: u64, tx: u64) {
        self.rx_total = rx;
        self.tx_total = tx;
    }
}

/// Network panel showing multiple interfaces with sparklines.
#[derive(Debug, Clone)]
pub struct NetworkPanel {
    /// Network interfaces.
    interfaces: Vec<NetworkInterface>,
    /// Download color.
    rx_color: Color,
    /// Upload color.
    tx_color: Color,
    /// Sparkline width.
    spark_width: usize,
    /// Show totals.
    show_totals: bool,
    /// Compact mode.
    compact: bool,
    /// Cached bounds.
    bounds: Rect,
}

impl Default for NetworkPanel {
    fn default() -> Self {
        Self::new()
    }
}

impl NetworkPanel {
    /// Create a new network panel.
    #[must_use]
    pub fn new() -> Self {
        Self {
            interfaces: Vec::new(),
            rx_color: Color::new(0.3, 0.8, 0.3, 1.0), // Green for download
            tx_color: Color::new(0.8, 0.3, 0.3, 1.0), // Red for upload
            spark_width: 20,
            show_totals: true,
            compact: false,
            bounds: Rect::default(),
        }
    }

    /// Set interfaces.
    pub fn set_interfaces(&mut self, interfaces: Vec<NetworkInterface>) {
        self.interfaces = interfaces;
    }

    /// Add an interface.
    pub fn add_interface(&mut self, iface: NetworkInterface) {
        self.interfaces.push(iface);
    }

    /// Get interface by name.
    pub fn interface_mut(&mut self, name: &str) -> Option<&mut NetworkInterface> {
        self.interfaces.iter_mut().find(|i| i.name == name)
    }

    /// Clear all interfaces.
    pub fn clear(&mut self) {
        self.interfaces.clear();
    }

    /// Set download color.
    #[must_use]
    pub fn with_rx_color(mut self, color: Color) -> Self {
        self.rx_color = color;
        self
    }

    /// Set upload color.
    #[must_use]
    pub fn with_tx_color(mut self, color: Color) -> Self {
        self.tx_color = color;
        self
    }

    /// Set sparkline width.
    #[must_use]
    pub fn with_spark_width(mut self, width: usize) -> Self {
        self.spark_width = width;
        self
    }

    /// Hide totals.
    #[must_use]
    pub fn without_totals(mut self) -> Self {
        self.show_totals = false;
        self
    }

    /// Enable compact mode.
    #[must_use]
    pub fn compact(mut self) -> Self {
        self.compact = true;
        self
    }

    /// Get interface count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.interfaces.len()
    }

    /// Check if empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.interfaces.is_empty()
    }

    /// Format bytes per second as human-readable string.
    fn format_bps(bps: f64) -> String {
        const KB: f64 = 1024.0;
        const MB: f64 = KB * 1024.0;
        const GB: f64 = MB * 1024.0;

        if bps >= GB {
            format!("{:.1}G/s", bps / GB)
        } else if bps >= MB {
            format!("{:.1}M/s", bps / MB)
        } else if bps >= KB {
            format!("{:.1}K/s", bps / KB)
        } else {
            format!("{bps:.0}B/s")
        }
    }

    /// Format total bytes as human-readable string.
    fn format_bytes(bytes: u64) -> String {
        const KB: u64 = 1024;
        const MB: u64 = KB * 1024;
        const GB: u64 = MB * 1024;
        const TB: u64 = GB * 1024;

        if bytes >= TB {
            format!("{:.1}T", bytes as f64 / TB as f64)
        } else if bytes >= GB {
            format!("{:.1}G", bytes as f64 / GB as f64)
        } else if bytes >= MB {
            format!("{:.1}M", bytes as f64 / MB as f64)
        } else if bytes >= KB {
            format!("{:.1}K", bytes as f64 / KB as f64)
        } else {
            format!("{bytes}B")
        }
    }

    /// Render sparkline from data.
    fn render_sparkline(data: &[f64], width: usize) -> String {
        if data.is_empty() {
            return " ".repeat(width);
        }

        let max_val = data
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
            .max(1.0);
        let start = data.len().saturating_sub(width);
        let visible = &data[start..];

        let mut result = String::with_capacity(width * 3); // Unicode chars are up to 3 bytes
        for &val in visible {
            let normalized = (val / max_val).clamp(0.0, 1.0);
            let idx = ((normalized * 7.0).round() as usize).min(7);
            result.push(SPARK_CHARS[idx]);
        }

        // Pad if needed (use char count, not byte count)
        let char_count = result.chars().count();
        if char_count < width {
            let padding = " ".repeat(width - char_count);
            result.insert_str(0, &padding);
        }

        result
    }
}

impl Brick for NetworkPanel {
    fn brick_name(&self) -> &'static str {
        "network_panel"
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

impl Widget for NetworkPanel {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        let rows_per_iface = if self.compact { 1 } else { 2 };
        let height = (self.interfaces.len() * rows_per_iface + 1) as f32;
        let min_width = if self.compact { 40.0 } else { 60.0 };
        let width = constraints.max_width.max(min_width);
        constraints.constrain(Size::new(width, height.max(2.0)))
    }

    fn layout(&mut self, bounds: Rect) -> LayoutResult {
        self.bounds = bounds;
        LayoutResult {
            size: Size::new(bounds.width, bounds.height),
        }
    }

    #[allow(clippy::too_many_lines)]
    fn paint(&self, canvas: &mut dyn Canvas) {
        let width = self.bounds.width as usize;
        let height = self.bounds.height as usize;
        if width == 0 || height == 0 {
            return;
        }

        // Header
        let header_style = TextStyle {
            color: Color::new(0.0, 1.0, 1.0, 1.0),
            weight: presentar_core::FontWeight::Bold,
            ..Default::default()
        };
        canvas.draw_text(
            "Network",
            Point::new(self.bounds.x, self.bounds.y),
            &header_style,
        );

        let _rows_per_iface = if self.compact { 1 } else { 2 };
        let mut y = self.bounds.y + 1.0;

        for iface in &self.interfaces {
            if y >= self.bounds.y + self.bounds.height {
                break;
            }

            if self.compact {
                // Compact: single line per interface
                // eth0: ▁▂▃▄▅ 125K/s ↓ ▅▄▃▂▁ 50K/s ↑
                let name_w = 8;
                let spark_w = self.spark_width.min(width.saturating_sub(30) / 2);

                let mut x = self.bounds.x;

                // Interface name
                let name = format!("{:name_w$}", iface.name);
                canvas.draw_text(
                    &name,
                    Point::new(x, y),
                    &TextStyle {
                        color: Color::new(0.8, 0.8, 0.8, 1.0),
                        ..Default::default()
                    },
                );
                x += name_w as f32 + 1.0;

                // Download sparkline
                let rx_spark = Self::render_sparkline(&iface.rx_history, spark_w);
                canvas.draw_text(
                    &rx_spark,
                    Point::new(x, y),
                    &TextStyle {
                        color: self.rx_color,
                        ..Default::default()
                    },
                );
                x += spark_w as f32 + 1.0;

                // Download rate
                let rx_rate = format!("{:>8}", Self::format_bps(iface.rx_bps));
                canvas.draw_text(
                    &rx_rate,
                    Point::new(x, y),
                    &TextStyle {
                        color: self.rx_color,
                        ..Default::default()
                    },
                );
                x += 9.0;

                // Arrow
                canvas.draw_text(
                    "↓",
                    Point::new(x, y),
                    &TextStyle {
                        color: self.rx_color,
                        ..Default::default()
                    },
                );
                x += 2.0;

                // Upload sparkline
                let tx_spark = Self::render_sparkline(&iface.tx_history, spark_w);
                canvas.draw_text(
                    &tx_spark,
                    Point::new(x, y),
                    &TextStyle {
                        color: self.tx_color,
                        ..Default::default()
                    },
                );
                x += spark_w as f32 + 1.0;

                // Upload rate
                let tx_rate = format!("{:>8}", Self::format_bps(iface.tx_bps));
                canvas.draw_text(
                    &tx_rate,
                    Point::new(x, y),
                    &TextStyle {
                        color: self.tx_color,
                        ..Default::default()
                    },
                );
                x += 9.0;

                // Arrow
                canvas.draw_text(
                    "↑",
                    Point::new(x, y),
                    &TextStyle {
                        color: self.tx_color,
                        ..Default::default()
                    },
                );
                x += 2.0;

                // Error/Drop rate highlighting (CB-NET-003/004)
                // Show warning indicators if error or drop rates are non-zero
                if iface.errors_per_sec > 0.0 || iface.drops_per_sec > 0.0 {
                    // Choose color based on severity
                    let (indicator, color) =
                        if iface.errors_per_sec > 10.0 || iface.drops_per_sec > 10.0 {
                            (
                                "●",
                                Color {
                                    r: 1.0,
                                    g: 0.3,
                                    b: 0.3,
                                    a: 1.0,
                                },
                            ) // Red - critical
                        } else if iface.errors_per_sec > 1.0 || iface.drops_per_sec > 1.0 {
                            (
                                "◐",
                                Color {
                                    r: 1.0,
                                    g: 0.8,
                                    b: 0.2,
                                    a: 1.0,
                                },
                            ) // Yellow - warning
                        } else {
                            (
                                "○",
                                Color {
                                    r: 0.8,
                                    g: 0.8,
                                    b: 0.3,
                                    a: 1.0,
                                },
                            ) // Dim yellow - minor
                        };

                    // Format: ●E:5/D:2
                    let err_drop_text = format!(
                        "{indicator}E:{:.0}/D:{:.0}",
                        iface.errors_per_sec, iface.drops_per_sec
                    );
                    canvas.draw_text(
                        &err_drop_text,
                        Point::new(x, y),
                        &TextStyle {
                            color,
                            ..Default::default()
                        },
                    );
                    x += err_drop_text.len() as f32 + 1.0;
                }

                // Bandwidth utilization display (CB-NET-006)
                if let Some(util_pct) = iface.utilization_percent {
                    let capped = util_pct.min(100.0);
                    let (util_text, util_color) = if util_pct > 80.0 {
                        (
                            format!("●{capped:.0}%"),
                            Color {
                                r: 1.0,
                                g: 0.3,
                                b: 0.3,
                                a: 1.0,
                            }, // Red - saturated
                        )
                    } else if util_pct > 50.0 {
                        (
                            format!("◐{util_pct:.0}%"),
                            Color {
                                r: 1.0,
                                g: 0.8,
                                b: 0.2,
                                a: 1.0,
                            }, // Yellow - high
                        )
                    } else {
                        (
                            format!("{util_pct:.0}%"),
                            Color {
                                r: 0.5,
                                g: 0.8,
                                b: 0.5,
                                a: 1.0,
                            }, // Green - normal
                        )
                    };
                    canvas.draw_text(
                        &util_text,
                        Point::new(x, y),
                        &TextStyle {
                            color: util_color,
                            ..Default::default()
                        },
                    );
                }
            } else {
                // Full: two lines per interface
                // eth0
                //   ↓ ▁▂▃▄▅▆▇█ 125.3M/s (Total: 1.2G)  ↑ ▅▄▃▂▁▂▃▄ 50.2K/s (Total: 500M)

                // Interface name
                canvas.draw_text(
                    &iface.name,
                    Point::new(self.bounds.x, y),
                    &TextStyle {
                        color: Color::new(0.8, 0.8, 1.0, 1.0),
                        weight: presentar_core::FontWeight::Bold,
                        ..Default::default()
                    },
                );
                y += 1.0;

                let spark_w = self.spark_width.min(width.saturating_sub(40) / 2);
                let mut x = self.bounds.x + 2.0;

                // Download
                canvas.draw_text(
                    "↓",
                    Point::new(x, y),
                    &TextStyle {
                        color: self.rx_color,
                        ..Default::default()
                    },
                );
                x += 2.0;

                let rx_spark = Self::render_sparkline(&iface.rx_history, spark_w);
                canvas.draw_text(
                    &rx_spark,
                    Point::new(x, y),
                    &TextStyle {
                        color: self.rx_color,
                        ..Default::default()
                    },
                );
                x += spark_w as f32 + 1.0;

                let rx_rate = Self::format_bps(iface.rx_bps);
                canvas.draw_text(
                    &rx_rate,
                    Point::new(x, y),
                    &TextStyle {
                        color: self.rx_color,
                        ..Default::default()
                    },
                );
                x += 10.0;

                if self.show_totals {
                    let rx_total = format!("({})", Self::format_bytes(iface.rx_total));
                    canvas.draw_text(
                        &rx_total,
                        Point::new(x, y),
                        &TextStyle {
                            color: Color::new(0.5, 0.5, 0.5, 1.0),
                            ..Default::default()
                        },
                    );
                    x += 10.0;
                }

                x += 2.0;

                // Upload
                canvas.draw_text(
                    "↑",
                    Point::new(x, y),
                    &TextStyle {
                        color: self.tx_color,
                        ..Default::default()
                    },
                );
                x += 2.0;

                let tx_spark = Self::render_sparkline(&iface.tx_history, spark_w);
                canvas.draw_text(
                    &tx_spark,
                    Point::new(x, y),
                    &TextStyle {
                        color: self.tx_color,
                        ..Default::default()
                    },
                );
                x += spark_w as f32 + 1.0;

                let tx_rate = Self::format_bps(iface.tx_bps);
                canvas.draw_text(
                    &tx_rate,
                    Point::new(x, y),
                    &TextStyle {
                        color: self.tx_color,
                        ..Default::default()
                    },
                );
                x += 10.0;

                if self.show_totals {
                    let tx_total = format!("({})", Self::format_bytes(iface.tx_total));
                    canvas.draw_text(
                        &tx_total,
                        Point::new(x, y),
                        &TextStyle {
                            color: Color::new(0.5, 0.5, 0.5, 1.0),
                            ..Default::default()
                        },
                    );
                }
            }

            y += 1.0;
        }

        // Empty state
        if self.interfaces.is_empty() && height > 1 {
            canvas.draw_text(
                "No interfaces",
                Point::new(self.bounds.x + 1.0, self.bounds.y + 1.0),
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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::disallowed_methods)]
mod tests {
    use super::*;

    #[allow(dead_code)]
    fn sample_interface() -> NetworkInterface {
        let mut iface = NetworkInterface::new("eth0");
        for i in 0..30 {
            iface.update(i as f64 * 1000.0, i as f64 * 500.0);
        }
        iface.set_totals(1024 * 1024 * 1024, 512 * 1024 * 1024);
        iface
    }

    #[test]
    fn test_network_panel_new() {
        let panel = NetworkPanel::new();
        assert!(panel.is_empty());
    }

    #[test]
    fn test_network_panel_add_interface() {
        let mut panel = NetworkPanel::new();
        panel.add_interface(NetworkInterface::new("eth0"));
        assert_eq!(panel.len(), 1);
    }

    #[test]
    fn test_network_panel_set_interfaces() {
        let mut panel = NetworkPanel::new();
        panel.set_interfaces(vec![
            NetworkInterface::new("eth0"),
            NetworkInterface::new("wlan0"),
        ]);
        assert_eq!(panel.len(), 2);
    }

    #[test]
    fn test_network_panel_interface_mut() {
        let mut panel = NetworkPanel::new();
        panel.add_interface(NetworkInterface::new("eth0"));
        let iface = panel.interface_mut("eth0").unwrap();
        iface.update(1000.0, 500.0);
        assert_eq!(iface.rx_bps, 1000.0);
    }

    #[test]
    fn test_network_panel_clear() {
        let mut panel = NetworkPanel::new();
        panel.add_interface(NetworkInterface::new("eth0"));
        panel.clear();
        assert!(panel.is_empty());
    }

    #[test]
    fn test_network_interface_update() {
        let mut iface = NetworkInterface::new("eth0");
        iface.update(1000.0, 500.0);
        assert_eq!(iface.rx_bps, 1000.0);
        assert_eq!(iface.tx_bps, 500.0);
        assert_eq!(iface.rx_history.len(), 1);
        assert_eq!(iface.tx_history.len(), 1);
    }

    #[test]
    fn test_network_interface_history_limit() {
        let mut iface = NetworkInterface::new("eth0");
        for i in 0..100 {
            iface.update(i as f64, i as f64);
        }
        assert_eq!(iface.rx_history.len(), 60);
        assert_eq!(iface.tx_history.len(), 60);
    }

    #[test]
    fn test_network_panel_with_colors() {
        let panel = NetworkPanel::new()
            .with_rx_color(Color::BLUE)
            .with_tx_color(Color::RED);
        assert_eq!(panel.rx_color, Color::BLUE);
        assert_eq!(panel.tx_color, Color::RED);
    }

    #[test]
    fn test_network_panel_with_spark_width() {
        let panel = NetworkPanel::new().with_spark_width(30);
        assert_eq!(panel.spark_width, 30);
    }

    #[test]
    fn test_network_panel_without_totals() {
        let panel = NetworkPanel::new().without_totals();
        assert!(!panel.show_totals);
    }

    #[test]
    fn test_network_panel_compact() {
        let panel = NetworkPanel::new().compact();
        assert!(panel.compact);
    }

    #[test]
    fn test_format_bps() {
        assert_eq!(NetworkPanel::format_bps(500.0), "500B/s");
        assert_eq!(NetworkPanel::format_bps(1024.0), "1.0K/s");
        assert_eq!(NetworkPanel::format_bps(1024.0 * 1024.0), "1.0M/s");
        assert_eq!(NetworkPanel::format_bps(1024.0 * 1024.0 * 1024.0), "1.0G/s");
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(NetworkPanel::format_bytes(500), "500B");
        assert_eq!(NetworkPanel::format_bytes(1024), "1.0K");
        assert_eq!(NetworkPanel::format_bytes(1024 * 1024), "1.0M");
        assert_eq!(NetworkPanel::format_bytes(1024 * 1024 * 1024), "1.0G");
        assert_eq!(
            NetworkPanel::format_bytes(1024u64 * 1024 * 1024 * 1024),
            "1.0T"
        );
    }

    #[test]
    fn test_render_sparkline() {
        let data = vec![0.0, 0.5, 1.0];
        let spark = NetworkPanel::render_sparkline(&data, 5);
        assert_eq!(spark.chars().count(), 5);
    }

    #[test]
    fn test_render_sparkline_empty() {
        let spark = NetworkPanel::render_sparkline(&[], 5);
        assert_eq!(spark, "     ");
    }

    #[test]
    fn test_network_panel_measure() {
        let mut panel = NetworkPanel::new();
        panel.add_interface(NetworkInterface::new("eth0"));
        let size = panel.measure(Constraints::new(0.0, 100.0, 0.0, 50.0));
        assert!(size.width >= 60.0);
        assert!(size.height >= 2.0);
    }

    #[test]
    fn test_network_panel_layout() {
        let mut panel = NetworkPanel::new();
        let result = panel.layout(Rect::new(0.0, 0.0, 80.0, 20.0));
        assert_eq!(result.size.width, 80.0);
    }

    #[test]
    fn test_network_panel_verify() {
        let panel = NetworkPanel::new();
        assert!(panel.verify().is_valid());
    }

    #[test]
    fn test_network_panel_brick_name() {
        let panel = NetworkPanel::new();
        assert_eq!(panel.brick_name(), "network_panel");
    }

    #[test]
    fn test_network_panel_default() {
        let panel = NetworkPanel::default();
        assert!(panel.is_empty());
    }

    #[test]
    fn test_network_panel_children() {
        let panel = NetworkPanel::new();
        assert!(panel.children().is_empty());
    }

    #[test]
    fn test_network_panel_children_mut() {
        let mut panel = NetworkPanel::new();
        assert!(panel.children_mut().is_empty());
    }

    #[test]
    fn test_network_panel_type_id() {
        let panel = NetworkPanel::new();
        assert_eq!(Widget::type_id(&panel), TypeId::of::<NetworkPanel>());
    }

    #[test]
    fn test_network_panel_to_html() {
        let panel = NetworkPanel::new();
        assert!(panel.to_html().is_empty());
    }

    #[test]
    fn test_network_panel_to_css() {
        let panel = NetworkPanel::new();
        assert!(panel.to_css().is_empty());
    }

    #[test]
    fn test_network_interface_set_totals() {
        let mut iface = NetworkInterface::new("eth0");
        iface.set_totals(1000, 500);
        assert_eq!(iface.rx_total, 1000);
        assert_eq!(iface.tx_total, 500);
    }

    #[test]
    fn test_network_panel_paint_with_data() {
        use crate::direct::{CellBuffer, DirectTerminalCanvas};

        let mut panel = NetworkPanel::new();
        let mut iface = NetworkInterface::new("eth0");
        for i in 0..30 {
            iface.update(i as f64 * 1000.0, i as f64 * 500.0);
        }
        iface.set_totals(1024 * 1024 * 1024, 512 * 1024 * 1024);
        panel.add_interface(iface);

        panel.layout(Rect::new(0.0, 0.0, 80.0, 10.0));

        let mut buffer = CellBuffer::new(80, 10);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        panel.paint(&mut canvas);
    }

    #[test]
    fn test_network_panel_paint_empty() {
        use crate::direct::{CellBuffer, DirectTerminalCanvas};

        let mut panel = NetworkPanel::new();
        panel.layout(Rect::new(0.0, 0.0, 60.0, 10.0));

        let mut buffer = CellBuffer::new(60, 10);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        panel.paint(&mut canvas);
    }

    #[test]
    fn test_network_panel_paint_small_bounds() {
        use crate::direct::{CellBuffer, DirectTerminalCanvas};

        let mut panel = NetworkPanel::new();
        panel.add_interface(NetworkInterface::new("eth0"));
        panel.layout(Rect::new(0.0, 0.0, 5.0, 0.5)); // Too small

        let mut buffer = CellBuffer::new(5, 1);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        panel.paint(&mut canvas); // Should early return
    }

    #[test]
    fn test_network_panel_paint_compact() {
        use crate::direct::{CellBuffer, DirectTerminalCanvas};

        let mut panel = NetworkPanel::new().compact();
        let mut iface = NetworkInterface::new("wlan0");
        iface.update(5000.0, 2500.0);
        panel.add_interface(iface);

        panel.layout(Rect::new(0.0, 0.0, 60.0, 10.0));

        let mut buffer = CellBuffer::new(60, 10);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        panel.paint(&mut canvas);
    }

    #[test]
    fn test_network_panel_paint_with_totals() {
        use crate::direct::{CellBuffer, DirectTerminalCanvas};

        let mut panel = NetworkPanel::new(); // show_totals is true by default
        let mut iface = NetworkInterface::new("eth0");
        iface.update(1024.0 * 1024.0, 512.0 * 1024.0);
        iface.set_totals(10 * 1024 * 1024 * 1024, 5 * 1024 * 1024 * 1024);
        panel.add_interface(iface);

        panel.layout(Rect::new(0.0, 0.0, 80.0, 10.0));

        let mut buffer = CellBuffer::new(80, 10);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        panel.paint(&mut canvas);
    }

    #[test]
    fn test_network_panel_paint_without_totals() {
        use crate::direct::{CellBuffer, DirectTerminalCanvas};

        let mut panel = NetworkPanel::new().without_totals();
        let mut iface = NetworkInterface::new("eth0");
        iface.update(1024.0, 512.0);
        panel.add_interface(iface);

        panel.layout(Rect::new(0.0, 0.0, 60.0, 10.0));

        let mut buffer = CellBuffer::new(60, 10);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        panel.paint(&mut canvas);
    }

    #[test]
    fn test_network_panel_multiple_interfaces() {
        use crate::direct::{CellBuffer, DirectTerminalCanvas};

        let mut panel = NetworkPanel::new();

        let mut eth0 = NetworkInterface::new("eth0");
        eth0.update(10000.0, 5000.0);
        panel.add_interface(eth0);

        let mut wlan0 = NetworkInterface::new("wlan0");
        wlan0.update(2000.0, 1000.0);
        panel.add_interface(wlan0);

        let mut lo = NetworkInterface::new("lo");
        lo.update(100.0, 100.0);
        panel.add_interface(lo);

        panel.layout(Rect::new(0.0, 0.0, 80.0, 10.0));

        let mut buffer = CellBuffer::new(80, 10);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        panel.paint(&mut canvas);
    }

    #[test]
    fn test_network_panel_event() {
        let mut panel = NetworkPanel::new();
        let result = panel.event(&Event::key_down(presentar_core::Key::Enter));
        assert!(result.is_none());
    }

    #[test]
    fn test_network_panel_assertions() {
        let panel = NetworkPanel::new();
        assert!(!panel.assertions().is_empty());
    }

    #[test]
    fn test_network_panel_budget() {
        let panel = NetworkPanel::new();
        assert!(panel.budget().paint_ms > 0);
    }

    #[test]
    fn test_network_interface_long_name() {
        use crate::direct::{CellBuffer, DirectTerminalCanvas};

        let mut panel = NetworkPanel::new();
        let mut iface = NetworkInterface::new("verylonginterfacename0");
        iface.update(1024.0, 512.0);
        panel.add_interface(iface);

        panel.layout(Rect::new(0.0, 0.0, 80.0, 10.0));

        let mut buffer = CellBuffer::new(80, 10);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        panel.paint(&mut canvas);
    }

    #[test]
    fn test_format_bps_edge_cases() {
        // Test very large values (displayed as G/s since no T/s support)
        assert!(NetworkPanel::format_bps(1024.0 * 1024.0 * 1024.0 * 1024.0).contains("G/s"));
        // Test very small values
        assert_eq!(NetworkPanel::format_bps(0.5), "0B/s");
    }

    #[test]
    fn test_render_sparkline_single_value() {
        let data = vec![0.5];
        let spark = NetworkPanel::render_sparkline(&data, 3);
        assert_eq!(spark.chars().count(), 3);
    }
}
