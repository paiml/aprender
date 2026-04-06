//! Network panel rendering and utilities.
//!
//! Provides network panel title building, traffic formatting,
//! and helper functions for rendering network metrics.

use presentar_core::Color;

// =============================================================================
// NETWORK TITLE BUILDING
// =============================================================================

/// Build network panel title string.
///
/// Format: "Network │ eth0 │ ↓1.5MB/s ↑500KB/s"
#[must_use]
pub fn build_network_title(interface: &str, rx_rate: u64, tx_rate: u64) -> String {
    let rx_str = format_traffic_rate(rx_rate);
    let tx_str = format_traffic_rate(tx_rate);
    format!("Network │ {} │ ↓{} ↑{}", interface, rx_str, tx_str)
}

/// Build compact network title for narrow panels.
///
/// Format: "Net │ ↓1.5M ↑500K"
#[must_use]
pub fn build_network_title_compact(rx_rate: u64, tx_rate: u64) -> String {
    let rx_str = format_traffic_rate_short(rx_rate);
    let tx_str = format_traffic_rate_short(tx_rate);
    format!("Net │ ↓{} ↑{}", rx_str, tx_str)
}

/// Build network panel title with connection count (GAP-NET-002).
///
/// Format: "Network │ eth0 │ ↓1.5MB/s ↑500KB/s │ 42 conn"
#[must_use]
pub fn build_network_title_with_conns(
    interface: &str,
    rx_rate: u64,
    tx_rate: u64,
    connection_count: usize,
) -> String {
    let rx_str = format_traffic_rate(rx_rate);
    let tx_str = format_traffic_rate(tx_rate);
    format!(
        "Network │ {} │ ↓{} ↑{} │ {} conn",
        interface, rx_str, tx_str, connection_count
    )
}

/// Build compact network title with connection count.
///
/// Format: "Net │ ↓1.5M ↑500K │ 42"
#[must_use]
pub fn build_network_title_compact_with_conns(
    rx_rate: u64,
    tx_rate: u64,
    connection_count: usize,
) -> String {
    let rx_str = format_traffic_rate_short(rx_rate);
    let tx_str = format_traffic_rate_short(tx_rate);
    format!("Net │ ↓{} ↑{} │ {}", rx_str, tx_str, connection_count)
}

// =============================================================================
// TRAFFIC FORMATTING
// =============================================================================

/// Format traffic rate as human-readable string.
///
/// # Examples
/// - 0 -> "0B/s"
/// - 1024 -> "1.0KB/s"
/// - 1048576 -> "1.0MB/s"
#[must_use]
pub fn format_traffic_rate(bytes_per_sec: u64) -> String {
    if bytes_per_sec == 0 {
        return "0B/s".to_string();
    }

    const KB: f64 = 1024.0;
    const MB: f64 = 1024.0 * 1024.0;
    const GB: f64 = 1024.0 * 1024.0 * 1024.0;

    let bytes = bytes_per_sec as f64;

    if bytes >= GB {
        format!("{:.1}GB/s", bytes / GB)
    } else if bytes >= MB {
        format!("{:.1}MB/s", bytes / MB)
    } else if bytes >= KB {
        format!("{:.1}KB/s", bytes / KB)
    } else {
        format!("{}B/s", bytes_per_sec)
    }
}

/// Format traffic rate in short form (no /s suffix).
///
/// # Examples
/// - 1024 -> "1.0K"
/// - 1048576 -> "1.0M"
#[must_use]
pub fn format_traffic_rate_short(bytes_per_sec: u64) -> String {
    if bytes_per_sec == 0 {
        return "0".to_string();
    }

    const KB: f64 = 1024.0;
    const MB: f64 = 1024.0 * 1024.0;
    const GB: f64 = 1024.0 * 1024.0 * 1024.0;

    let bytes = bytes_per_sec as f64;

    if bytes >= GB {
        format!("{:.1}G", bytes / GB)
    } else if bytes >= MB {
        format!("{:.1}M", bytes / MB)
    } else if bytes >= KB {
        format!("{:.0}K", bytes / KB)
    } else {
        format!("{}B", bytes_per_sec)
    }
}

/// Format total bytes transferred.
#[must_use]
pub fn format_total_bytes(bytes: u64) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = 1024.0 * 1024.0;
    const GB: f64 = 1024.0 * 1024.0 * 1024.0;
    const TB: f64 = 1024.0 * 1024.0 * 1024.0 * 1024.0;

    let b = bytes as f64;

    if b >= TB {
        format!("{:.2}TB", b / TB)
    } else if b >= GB {
        format!("{:.2}GB", b / GB)
    } else if b >= MB {
        format!("{:.1}MB", b / MB)
    } else if b >= KB {
        format!("{:.0}KB", b / KB)
    } else {
        format!("{}B", bytes)
    }
}

// =============================================================================
// NETWORK COLORS
// =============================================================================

/// Get color for download (RX) traffic.
#[must_use]
pub fn rx_color() -> Color {
    Color::new(0.3, 0.9, 0.5, 1.0) // Green
}

/// Get color for upload (TX) traffic.
#[must_use]
pub fn tx_color() -> Color {
    Color::new(0.9, 0.5, 0.3, 1.0) // Orange
}

/// Get color for traffic rate based on intensity.
#[must_use]
pub fn traffic_intensity_color(bytes_per_sec: u64) -> Color {
    const MB: u64 = 1024 * 1024;

    if bytes_per_sec > 100 * MB {
        Color::new(1.0, 0.4, 0.4, 1.0) // Very high - red
    } else if bytes_per_sec > 10 * MB {
        Color::new(1.0, 0.8, 0.3, 1.0) // High - yellow
    } else if bytes_per_sec > MB {
        Color::new(0.4, 0.9, 0.5, 1.0) // Moderate - green
    } else if bytes_per_sec > 0 {
        Color::new(0.5, 0.7, 0.5, 1.0) // Low - dim green
    } else {
        Color::new(0.4, 0.4, 0.4, 1.0) // Idle - gray
    }
}

// =============================================================================
// INTERFACE UTILITIES
// =============================================================================

/// Network interface type detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterfaceType {
    /// Physical Ethernet
    Ethernet,
    /// Wireless
    Wifi,
    /// Loopback
    Loopback,
    /// Virtual/Bridge
    Virtual,
    /// VPN tunnel
    Vpn,
    /// Docker/Container
    Docker,
    /// Unknown type
    Unknown,
}

impl InterfaceType {
    /// Detect interface type from name.
    #[must_use]
    pub fn from_name(name: &str) -> Self {
        let lower = name.to_lowercase();

        if lower == "lo" {
            Self::Loopback
        } else if lower.starts_with("eth") || lower.starts_with("en") {
            Self::Ethernet
        } else if lower.starts_with("wl") || lower.starts_with("wifi") {
            Self::Wifi
        } else if lower.starts_with("docker") || lower.starts_with("br-") {
            Self::Docker
        } else if lower.starts_with("veth") || lower.starts_with("virbr") {
            Self::Virtual
        } else if lower.starts_with("tun") || lower.starts_with("tap") || lower.starts_with("wg") {
            Self::Vpn
        } else {
            Self::Unknown
        }
    }

    /// Get icon for interface type.
    #[must_use]
    pub fn icon(&self) -> &'static str {
        match self {
            Self::Ethernet => "󰈀",
            Self::Wifi => "󰖩",
            Self::Loopback => "󰑐",
            Self::Virtual => "󰒍",
            Self::Vpn => "󰒃",
            Self::Docker => "󰡨",
            Self::Unknown => "󰈁",
        }
    }

    /// Get display name for interface type.
    #[must_use]
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Ethernet => "Ethernet",
            Self::Wifi => "Wi-Fi",
            Self::Loopback => "Loopback",
            Self::Virtual => "Virtual",
            Self::Vpn => "VPN",
            Self::Docker => "Docker",
            Self::Unknown => "Network",
        }
    }
}

// =============================================================================
// CONNECTION STATE
// =============================================================================

/// TCP connection state display.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionState {
    Established,
    Listen,
    TimeWait,
    CloseWait,
    SynSent,
    SynRecv,
    FinWait1,
    FinWait2,
    Closing,
    LastAck,
    Closed,
}

impl ConnectionState {
    /// Get short display name.
    #[must_use]
    pub fn short_name(&self) -> &'static str {
        match self {
            Self::Established => "ESTAB",
            Self::Listen => "LISTEN",
            Self::TimeWait => "TIME_W",
            Self::CloseWait => "CLOSE_W",
            Self::SynSent => "SYN_S",
            Self::SynRecv => "SYN_R",
            Self::FinWait1 => "FIN_W1",
            Self::FinWait2 => "FIN_W2",
            Self::Closing => "CLOSING",
            Self::LastAck => "LAST_A",
            Self::Closed => "CLOSED",
        }
    }

    /// Get color for state.
    #[must_use]
    pub fn color(&self) -> Color {
        match self {
            Self::Established => Color::new(0.3, 0.9, 0.3, 1.0), // Green
            Self::Listen => Color::new(0.5, 0.7, 1.0, 1.0),      // Blue
            Self::TimeWait | Self::CloseWait => Color::new(0.8, 0.8, 0.3, 1.0), // Yellow
            Self::Closed => Color::new(0.5, 0.5, 0.5, 1.0),      // Gray
            _ => Color::new(0.7, 0.5, 0.3, 1.0),                 // Orange for others
        }
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // build_network_title tests
    // =========================================================================

    #[test]
    fn test_build_network_title_basic() {
        let title = build_network_title("eth0", 1024 * 1024, 512 * 1024);
        assert!(title.contains("Network"));
        assert!(title.contains("eth0"));
        assert!(title.contains("↓"));
        assert!(title.contains("↑"));
    }

    #[test]
    fn test_build_network_title_zero() {
        let title = build_network_title("lo", 0, 0);
        assert!(title.contains("lo"));
        assert!(title.contains("0B/s"));
    }

    #[test]
    fn test_build_network_title_high_traffic() {
        let title = build_network_title("eth0", 100 * 1024 * 1024, 50 * 1024 * 1024);
        assert!(title.contains("MB/s"));
    }

    // =========================================================================
    // build_network_title_compact tests
    // =========================================================================

    #[test]
    fn test_build_network_title_compact_basic() {
        let title = build_network_title_compact(1024 * 1024, 512 * 1024);
        assert!(title.contains("Net"));
        assert!(title.contains("↓"));
        assert!(title.contains("↑"));
        assert!(!title.contains("/s")); // Short format
    }

    #[test]
    fn test_build_network_title_compact_short() {
        let title = build_network_title_compact(0, 0);
        assert!(title.chars().count() < 20);
    }

    // =========================================================================
    // build_network_title_with_conns tests (GAP-NET-002)
    // =========================================================================

    #[test]
    fn test_build_network_title_with_conns_basic() {
        let title = build_network_title_with_conns("eth0", 1024 * 1024, 512 * 1024, 42);
        assert!(title.contains("Network"));
        assert!(title.contains("eth0"));
        assert!(title.contains("42 conn"));
    }

    #[test]
    fn test_build_network_title_with_conns_zero() {
        let title = build_network_title_with_conns("lo", 0, 0, 0);
        assert!(title.contains("0 conn"));
    }

    #[test]
    fn test_build_network_title_with_conns_high_count() {
        let title = build_network_title_with_conns("eth0", 1024, 1024, 1000);
        assert!(title.contains("1000 conn"));
    }

    #[test]
    fn test_build_network_title_compact_with_conns_basic() {
        let title = build_network_title_compact_with_conns(1024 * 1024, 512 * 1024, 42);
        assert!(title.contains("Net"));
        assert!(title.contains("42"));
        assert!(!title.contains("conn")); // Compact doesn't say "conn"
    }

    #[test]
    fn test_build_network_title_compact_with_conns_zero() {
        let title = build_network_title_compact_with_conns(0, 0, 0);
        assert!(title.contains("│ 0"));
    }

    // =========================================================================
    // format_traffic_rate tests
    // =========================================================================

    #[test]
    fn test_format_traffic_rate_zero() {
        assert_eq!(format_traffic_rate(0), "0B/s");
    }

    #[test]
    fn test_format_traffic_rate_bytes() {
        assert_eq!(format_traffic_rate(500), "500B/s");
    }

    #[test]
    fn test_format_traffic_rate_kb() {
        assert_eq!(format_traffic_rate(1024), "1.0KB/s");
    }

    #[test]
    fn test_format_traffic_rate_mb() {
        assert_eq!(format_traffic_rate(1024 * 1024), "1.0MB/s");
    }

    #[test]
    fn test_format_traffic_rate_gb() {
        assert_eq!(format_traffic_rate(1024 * 1024 * 1024), "1.0GB/s");
    }

    // =========================================================================
    // format_traffic_rate_short tests
    // =========================================================================

    #[test]
    fn test_format_traffic_rate_short_zero() {
        assert_eq!(format_traffic_rate_short(0), "0");
    }

    #[test]
    fn test_format_traffic_rate_short_kb() {
        assert_eq!(format_traffic_rate_short(1024), "1K");
    }

    #[test]
    fn test_format_traffic_rate_short_mb() {
        assert_eq!(format_traffic_rate_short(1024 * 1024), "1.0M");
    }

    #[test]
    fn test_format_traffic_rate_short_no_suffix() {
        let result = format_traffic_rate_short(5 * 1024 * 1024);
        assert!(!result.contains("/s"));
    }

    // =========================================================================
    // format_total_bytes tests
    // =========================================================================

    #[test]
    fn test_format_total_bytes_bytes() {
        assert_eq!(format_total_bytes(500), "500B");
    }

    #[test]
    fn test_format_total_bytes_kb() {
        let result = format_total_bytes(2048);
        assert!(result.contains("KB"));
    }

    #[test]
    fn test_format_total_bytes_mb() {
        let result = format_total_bytes(5 * 1024 * 1024);
        assert!(result.contains("MB"));
    }

    #[test]
    fn test_format_total_bytes_gb() {
        let result = format_total_bytes(10 * 1024 * 1024 * 1024);
        assert!(result.contains("GB"));
    }

    #[test]
    fn test_format_total_bytes_tb() {
        let result = format_total_bytes(2 * 1024 * 1024 * 1024 * 1024);
        assert!(result.contains("TB"));
    }

    // =========================================================================
    // color tests
    // =========================================================================

    #[test]
    fn test_rx_color_is_green() {
        let color = rx_color();
        assert!(color.g > 0.8);
    }

    #[test]
    fn test_tx_color_is_orange() {
        let color = tx_color();
        assert!(color.r > 0.8);
    }

    #[test]
    fn test_traffic_intensity_color_idle() {
        let color = traffic_intensity_color(0);
        assert!((color.r - color.g).abs() < 0.1, "Idle should be gray");
    }

    #[test]
    fn test_traffic_intensity_color_low() {
        let color = traffic_intensity_color(512 * 1024);
        assert!(color.g > 0.6);
    }

    #[test]
    fn test_traffic_intensity_color_high() {
        let color = traffic_intensity_color(50 * 1024 * 1024);
        assert!(color.r > 0.9);
    }

    #[test]
    fn test_traffic_intensity_color_very_high() {
        let color = traffic_intensity_color(200 * 1024 * 1024);
        assert!(color.r > 0.9 && color.g < 0.5);
    }

    // =========================================================================
    // InterfaceType tests
    // =========================================================================

    #[test]
    fn test_interface_type_from_name_ethernet() {
        assert_eq!(InterfaceType::from_name("eth0"), InterfaceType::Ethernet);
        assert_eq!(InterfaceType::from_name("enp0s3"), InterfaceType::Ethernet);
    }

    #[test]
    fn test_interface_type_from_name_wifi() {
        assert_eq!(InterfaceType::from_name("wlan0"), InterfaceType::Wifi);
        assert_eq!(InterfaceType::from_name("wlp2s0"), InterfaceType::Wifi);
    }

    #[test]
    fn test_interface_type_from_name_loopback() {
        assert_eq!(InterfaceType::from_name("lo"), InterfaceType::Loopback);
    }

    #[test]
    fn test_interface_type_from_name_docker() {
        assert_eq!(InterfaceType::from_name("docker0"), InterfaceType::Docker);
        assert_eq!(InterfaceType::from_name("br-abc123"), InterfaceType::Docker);
    }

    #[test]
    fn test_interface_type_from_name_vpn() {
        assert_eq!(InterfaceType::from_name("tun0"), InterfaceType::Vpn);
        assert_eq!(InterfaceType::from_name("wg0"), InterfaceType::Vpn);
    }

    #[test]
    fn test_interface_type_from_name_virtual() {
        assert_eq!(InterfaceType::from_name("veth123"), InterfaceType::Virtual);
        assert_eq!(InterfaceType::from_name("virbr0"), InterfaceType::Virtual);
    }

    #[test]
    fn test_interface_type_from_name_unknown() {
        assert_eq!(InterfaceType::from_name("custom0"), InterfaceType::Unknown);
    }

    #[test]
    fn test_interface_type_icon() {
        assert!(!InterfaceType::Ethernet.icon().is_empty());
        assert!(!InterfaceType::Wifi.icon().is_empty());
    }

    #[test]
    fn test_interface_type_display_name() {
        assert_eq!(InterfaceType::Ethernet.display_name(), "Ethernet");
        assert_eq!(InterfaceType::Wifi.display_name(), "Wi-Fi");
        assert_eq!(InterfaceType::Loopback.display_name(), "Loopback");
    }

    #[test]
    fn test_interface_type_derive_debug() {
        let itype = InterfaceType::Ethernet;
        let debug = format!("{:?}", itype);
        assert!(debug.contains("Ethernet"));
    }

    #[test]
    fn test_interface_type_derive_clone() {
        let itype = InterfaceType::Wifi;
        let cloned = itype;
        assert_eq!(itype, cloned);
    }

    // =========================================================================
    // ConnectionState tests
    // =========================================================================

    #[test]
    fn test_connection_state_short_name() {
        assert_eq!(ConnectionState::Established.short_name(), "ESTAB");
        assert_eq!(ConnectionState::Listen.short_name(), "LISTEN");
        assert_eq!(ConnectionState::TimeWait.short_name(), "TIME_W");
    }

    #[test]
    fn test_connection_state_color_established() {
        let color = ConnectionState::Established.color();
        assert!(color.g > 0.8, "Established should be green");
    }

    #[test]
    fn test_connection_state_color_listen() {
        let color = ConnectionState::Listen.color();
        assert!(color.b > 0.9, "Listen should be blue");
    }

    #[test]
    fn test_connection_state_color_closed() {
        let color = ConnectionState::Closed.color();
        assert!((color.r - color.g).abs() < 0.1, "Closed should be gray");
    }

    #[test]
    fn test_connection_state_derive_debug() {
        let state = ConnectionState::Established;
        let debug = format!("{:?}", state);
        assert!(debug.contains("Established"));
    }
}
