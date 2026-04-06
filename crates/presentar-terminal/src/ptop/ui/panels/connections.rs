//! Connections panel rendering and utilities.
//!
//! Provides connections panel title building, connection formatting,
//! and helper functions for rendering network connection metrics.

use presentar_core::Color;

// =============================================================================
// CONNECTIONS TITLE BUILDING
// =============================================================================

/// Build connections panel title string.
///
/// Format: "Connections │ 45 │ ESTAB: 30 │ LISTEN: 5"
#[must_use]
pub fn build_connections_title(total: usize, established: usize, listening: usize) -> String {
    format!(
        "Connections │ {} │ ESTAB: {} │ LISTEN: {}",
        total, established, listening
    )
}

/// Build compact connections title for narrow panels.
///
/// Format: "Conn │ 45"
#[must_use]
pub fn build_connections_title_compact(total: usize) -> String {
    format!("Conn │ {}", total)
}

// =============================================================================
// CONNECTION STATE
// =============================================================================

/// TCP connection state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TcpConnectionState {
    /// Connection established
    #[default]
    Established,
    /// Listening for connections
    Listen,
    /// Connection is in TIME_WAIT state
    TimeWait,
    /// Connection is in CLOSE_WAIT state
    CloseWait,
    /// SYN sent, waiting for response
    SynSent,
    /// SYN received, responding
    SynRecv,
    /// FIN_WAIT_1 state
    FinWait1,
    /// FIN_WAIT_2 state
    FinWait2,
    /// CLOSING state
    Closing,
    /// LAST_ACK state
    LastAck,
    /// Connection closed
    Closed,
}

impl TcpConnectionState {
    /// Get display name for connection state.
    #[must_use]
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Established => "ESTABLISHED",
            Self::Listen => "LISTEN",
            Self::TimeWait => "TIME_WAIT",
            Self::CloseWait => "CLOSE_WAIT",
            Self::SynSent => "SYN_SENT",
            Self::SynRecv => "SYN_RECV",
            Self::FinWait1 => "FIN_WAIT_1",
            Self::FinWait2 => "FIN_WAIT_2",
            Self::Closing => "CLOSING",
            Self::LastAck => "LAST_ACK",
            Self::Closed => "CLOSED",
        }
    }

    /// Get short display name.
    #[must_use]
    pub fn short_name(&self) -> &'static str {
        match self {
            Self::Established => "ESTAB",
            Self::Listen => "LSTN",
            Self::TimeWait => "TIME_W",
            Self::CloseWait => "CLOSE_W",
            Self::SynSent => "SYN_S",
            Self::SynRecv => "SYN_R",
            Self::FinWait1 => "FIN_W1",
            Self::FinWait2 => "FIN_W2",
            Self::Closing => "CLOSE",
            Self::LastAck => "LAST_A",
            Self::Closed => "CLOSED",
        }
    }

    /// Get color for connection state.
    #[must_use]
    pub fn color(&self) -> Color {
        match self {
            Self::Established => Color::new(0.3, 0.9, 0.4, 1.0), // Green
            Self::Listen => Color::new(0.4, 0.7, 1.0, 1.0),      // Blue
            Self::TimeWait | Self::CloseWait => Color::new(1.0, 0.8, 0.3, 1.0), // Yellow
            Self::SynSent | Self::SynRecv => Color::new(0.9, 0.6, 0.3, 1.0), // Orange
            Self::FinWait1 | Self::FinWait2 | Self::Closing => Color::new(0.7, 0.5, 0.3, 1.0), // Brown
            Self::LastAck => Color::new(0.6, 0.4, 0.4, 1.0), // Dark red
            Self::Closed => Color::new(0.5, 0.5, 0.5, 1.0),  // Gray
        }
    }

    /// Check if connection is active.
    #[must_use]
    pub fn is_active(&self) -> bool {
        matches!(self, Self::Established | Self::Listen)
    }

    /// Check if connection is closing.
    #[must_use]
    pub fn is_closing(&self) -> bool {
        matches!(
            self,
            Self::TimeWait
                | Self::CloseWait
                | Self::FinWait1
                | Self::FinWait2
                | Self::Closing
                | Self::LastAck
        )
    }
}

// =============================================================================
// PORT/SERVICE MAPPING
// =============================================================================

/// Map well-known port to service name.
#[must_use]
pub fn port_to_service_name(port: u16) -> &'static str {
    match port {
        20 => "FTP-D",
        21 => "FTP",
        22 => "SSH",
        23 => "Telnet",
        25 => "SMTP",
        53 => "DNS",
        80 => "HTTP",
        110 => "POP3",
        119 => "NNTP",
        123 => "NTP",
        143 => "IMAP",
        161 => "SNMP",
        443 => "HTTPS",
        445 => "SMB",
        465 => "SMTPS",
        587 => "Subm",
        993 => "IMAPS",
        995 => "POP3S",
        1433 => "MSSQL",
        3306 => "MySQL",
        3389 => "RDP",
        5432 => "PgSQL",
        5900..=5999 => "VNC",
        6379 => "Redis",
        8080 => "Alt-HTTP",
        8443 => "Alt-HTTPS",
        9000..=9099 => "App",
        27017 => "MongoDB",
        _ => "",
    }
}

/// Check if port is a well-known service port.
#[must_use]
pub fn is_well_known_port(port: u16) -> bool {
    port <= 1023
}

/// Check if port is ephemeral (dynamic).
#[must_use]
pub fn is_ephemeral_port(port: u16) -> bool {
    port >= 49152
}

// =============================================================================
// ADDRESS FORMATTING
// =============================================================================

/// Format IP address for display (truncate if needed).
#[must_use]
pub fn format_address(addr: &str, max_width: usize) -> String {
    if addr.len() <= max_width {
        return addr.to_string();
    }

    if max_width < 4 {
        return addr.chars().take(max_width).collect();
    }

    // Truncate with ellipsis
    let truncated: String = addr.chars().take(max_width - 1).collect();
    format!("{}~", truncated)
}

/// Format socket address (IP:port) for display.
#[must_use]
pub fn format_socket_address(ip: &str, port: u16, max_width: usize) -> String {
    let full = format!("{}:{}", ip, port);
    format_address(&full, max_width)
}

/// Abbreviate localhost addresses.
#[must_use]
pub fn abbreviate_localhost(addr: &str) -> &str {
    match addr {
        "127.0.0.1" => "localhost",
        "::1" => "localhost",
        "0.0.0.0" => "*",
        "::" => "*",
        _ => addr,
    }
}

// =============================================================================
// CONNECTION PROTOCOL
// =============================================================================

/// Network protocol type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Protocol {
    #[default]
    Tcp,
    Udp,
    Tcp6,
    Udp6,
}

impl Protocol {
    /// Get display name.
    #[must_use]
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Tcp => "TCP",
            Self::Udp => "UDP",
            Self::Tcp6 => "TCP6",
            Self::Udp6 => "UDP6",
        }
    }

    /// Check if IPv6.
    #[must_use]
    pub fn is_ipv6(&self) -> bool {
        matches!(self, Self::Tcp6 | Self::Udp6)
    }

    /// Check if TCP.
    #[must_use]
    pub fn is_tcp(&self) -> bool {
        matches!(self, Self::Tcp | Self::Tcp6)
    }
}

// =============================================================================
// CONNECTION COLUMN WIDTHS
// =============================================================================

/// Column widths for connection display.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConnectionColumnWidths {
    /// Protocol column width
    pub proto: usize,
    /// Local address column width
    pub local: usize,
    /// Remote address column width
    pub remote: usize,
    /// State column width
    pub state: usize,
    /// Process column width
    pub process: usize,
}

impl ConnectionColumnWidths {
    /// Calculate column widths for available width.
    #[must_use]
    pub fn calculate(available_width: usize) -> Self {
        const PROTO_WIDTH: usize = 5;
        const STATE_WIDTH: usize = 8;
        const FIXED: usize = PROTO_WIDTH + STATE_WIDTH;

        let remaining = available_width.saturating_sub(FIXED);
        let addr_width = remaining / 3;
        let process_width = remaining - (addr_width * 2);

        Self {
            proto: PROTO_WIDTH,
            local: addr_width.max(10),
            remote: addr_width.max(10),
            state: STATE_WIDTH,
            process: process_width.max(8),
        }
    }

    /// Get total width.
    #[must_use]
    pub fn total(&self) -> usize {
        self.proto + self.local + self.remote + self.state + self.process
    }
}

impl Default for ConnectionColumnWidths {
    fn default() -> Self {
        Self::calculate(80)
    }
}

// =============================================================================
// CONNECTION DISPLAY COLORS
// =============================================================================

/// Dim color for muted labels.
pub const DIM_COLOR: Color = Color {
    r: 0.5,
    g: 0.5,
    b: 0.5,
    a: 1.0,
};

/// Active connection color (green).
pub const ACTIVE_COLOR: Color = Color {
    r: 0.3,
    g: 0.9,
    b: 0.3,
    a: 1.0,
};

/// Listen state color (blue).
pub const LISTEN_COLOR: Color = Color {
    r: 0.3,
    g: 0.7,
    b: 1.0,
    a: 1.0,
};

// =============================================================================
// SPARKLINE HELPERS
// =============================================================================

/// Build sparkline string from normalized values (0.0-1.0).
///
/// Uses braille-style characters: ▁▂▃▄▅▆▇█
#[must_use]
pub fn build_sparkline(values: &[f64], max_chars: usize) -> String {
    const CHARS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

    if values.len() < 3 {
        return String::new();
    }

    let chars: String = values
        .iter()
        .rev()
        .take(max_chars)
        .rev()
        .map(|&v| {
            let idx = ((v * 7.0).round() as usize).min(7);
            CHARS[idx]
        })
        .collect();

    if chars.is_empty() {
        String::new()
    } else {
        format!(" {chars}")
    }
}

/// Check if connection data has enough history for sparkline.
#[must_use]
pub fn has_sparkline_data(history_len: usize) -> bool {
    history_len >= 3
}

// =============================================================================
// STATE ABBREVIATIONS (extracted to reduce draw_connections_panel complexity)
// =============================================================================

/// Single-character state abbreviation for compact display.
///
/// # Arguments
/// * `state` - TCP connection state
///
/// # Returns
/// Single character: E=Established, L=Listen, T=TimeWait, C=CloseWait, S=SynSent, ?=Other
#[must_use]
pub fn state_abbreviation(state: TcpConnectionState) -> &'static str {
    match state {
        TcpConnectionState::Established => "E",
        TcpConnectionState::Listen => "L",
        TcpConnectionState::TimeWait => "T",
        TcpConnectionState::CloseWait => "C",
        TcpConnectionState::SynSent => "S",
        TcpConnectionState::SynRecv => "R",
        TcpConnectionState::FinWait1 => "F",
        TcpConnectionState::FinWait2 => "f",
        TcpConnectionState::Closing => "X",
        TcpConnectionState::LastAck => "A",
        TcpConnectionState::Closed => "-",
    }
}

/// Geo indicator for connection locality.
///
/// # Arguments
/// * `is_listening` - Whether connection is in LISTEN state
/// * `is_private` - Whether remote IP is private/local (127.x, 192.168.x, 10.x, etc.)
///
/// # Returns
/// "-" for listen, "L" for local/private, "R" for remote/public
#[must_use]
pub fn geo_indicator(is_listening: bool, is_private: bool) -> &'static str {
    if is_listening {
        "-"
    } else if is_private {
        "L"
    } else {
        "R"
    }
}

/// Determine if an IP address is private/local.
///
/// Private ranges: 127.x.x.x, 10.x.x.x, 172.16-31.x.x, 192.168.x.x, link-local, IPv6 loopback
#[must_use]
pub fn is_private_ip(ip: &std::net::IpAddr) -> bool {
    match ip {
        std::net::IpAddr::V4(ipv4) => {
            ipv4.is_loopback() || ipv4.is_private() || ipv4.is_link_local()
        }
        std::net::IpAddr::V6(ipv6) => ipv6.is_loopback(),
    }
}

// =============================================================================
// HOT INDICATOR COLORS (extracted to reduce draw_connections_panel complexity)
// =============================================================================

/// Hot indicator color based on activity level.
///
/// # Arguments
/// * `indicator` - Hot indicator string ("●" for hot, "◐" for warm, etc.)
///
/// # Returns
/// Orange for hot (●), yellow for warm, dim for inactive
#[must_use]
pub fn hot_indicator_color(indicator: &str) -> Color {
    if indicator == "●" {
        // Hot - orange
        Color {
            r: 1.0,
            g: 0.4,
            b: 0.2,
            a: 1.0,
        }
    } else if indicator == "◐" || indicator == "○" {
        // Warm - yellow
        Color {
            r: 1.0,
            g: 0.7,
            b: 0.3,
            a: 1.0,
        }
    } else {
        // Inactive - dim
        DIM_COLOR
    }
}

/// Truncate process name for display in connection table.
///
/// # Arguments
/// * `name` - Process name
/// * `max_len` - Maximum length (includes ellipsis if truncated)
#[must_use]
pub fn truncate_process_name(name: &str, max_len: usize) -> String {
    if name.len() <= max_len {
        name.to_string()
    } else if max_len > 1 {
        format!("{}…", &name[..max_len - 1])
    } else {
        name.chars().take(max_len).collect()
    }
}

// =============================================================================
// SERVICE AUTO-DETECTION (GAP-CONN-003)
// =============================================================================

/// Sorted port-to-service lookup table (GAP-CONN-003).
///
/// Maps well-known ports to service names for quick identification.
/// Based on IANA port assignments and common conventions.
/// Sorted by port number for binary search.
static PORT_SERVICES: &[(u16, &str)] = &[
    // System services
    (20, "ftp-data"),
    (21, "ftp"),
    (22, "ssh"),
    (23, "telnet"),
    (25, "smtp"),
    (53, "dns"),
    (67, "dhcp"),
    (68, "dhcp"),
    (69, "tftp"),
    (80, "http"),
    (110, "pop3"),
    (119, "nntp"),
    (123, "ntp"),
    (143, "imap"),
    (161, "snmp"),
    (162, "snmptrap"),
    (179, "bgp"),
    (194, "irc"),
    (443, "https"),
    (445, "smb"),
    (465, "smtps"),
    (514, "syslog"),
    (587, "submission"),
    (636, "ldaps"),
    (873, "rsync"),
    (993, "imaps"),
    (995, "pop3s"),
    // Other common
    (1080, "socks"),
    // Database services
    (1433, "mssql"),
    (1521, "oracle"),
    // Message queues
    (1883, "mqtt"),
    // Container/orchestration
    (2375, "docker"),
    (2376, "docker-tls"),
    // Development
    (3000, "dev-http"),
    // Monitoring
    (3100, "loki"),
    (3306, "mysql"),
    (3389, "rdp"),
    (4000, "dev-http"),
    (5000, "dev-http"),
    (5432, "postgres"),
    (5601, "kibana"),
    (5672, "amqp"),
    (5900, "vnc"),
    (6379, "redis"),
    (6443, "k8s-api"),
    (8000, "dev-http"),
    (8080, "http-alt"),
    (8443, "https-alt"),
    (9000, "dev-http"),
    (9042, "cassandra"),
    (9090, "prometheus"),
    (9092, "kafka"),
    (9100, "node-exp"),
    (9200, "elastic"),
    (10250, "kubelet"),
    (27017, "mongodb"),
];

/// Port-to-service name lookup via binary search (GAP-CONN-003).
#[must_use]
pub fn port_to_service(port: u16) -> Option<&'static str> {
    // Handle X11 range specially (6000..=6063)
    if (6000..=6063).contains(&port) {
        return Some("x11");
    }
    PORT_SERVICES
        .binary_search_by_key(&port, |&(p, _)| p)
        .map(|i| PORT_SERVICES[i].1)
        .ok()
}

/// Get service name with fallback to port number.
#[must_use]
pub fn service_name_or_port(port: u16) -> String {
    port_to_service(port)
        .map(String::from)
        .unwrap_or_else(|| port.to_string())
}

/// Service category for color coding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServiceCategory {
    /// Web services (HTTP, HTTPS)
    Web,
    /// Database services (MySQL, PostgreSQL, etc.)
    Database,
    /// Security-sensitive (SSH, RDP, VNC)
    Secure,
    /// System services (DNS, NTP, etc.)
    System,
    /// Development/local services
    Dev,
    /// Unknown/other
    Other,
}

impl ServiceCategory {
    /// Categorize a port.
    #[must_use]
    pub fn from_port(port: u16) -> Self {
        match port {
            80 | 443 | 8080 | 8443 => Self::Web,
            1433 | 1521 | 3306 | 5432 | 6379 | 9042 | 27017 => Self::Database,
            22 | 23 | 3389 | 5900..=5999 => Self::Secure,
            20..=25 | 53 | 67 | 68 | 123 | 161 | 162 | 514 => Self::System,
            3000 | 4000 | 5000 | 8000 | 9000 => Self::Dev,
            _ => Self::Other,
        }
    }

    /// Get display color for category.
    #[must_use]
    pub fn color(&self) -> Color {
        match self {
            Self::Web => Color::new(0.4, 0.8, 1.0, 1.0),      // Cyan
            Self::Database => Color::new(1.0, 0.6, 0.2, 1.0), // Orange
            Self::Secure => Color::new(1.0, 0.3, 0.3, 1.0),   // Red
            Self::System => Color::new(0.6, 0.6, 0.6, 1.0),   // Gray
            Self::Dev => Color::new(0.5, 1.0, 0.5, 1.0),      // Green
            Self::Other => Color::new(0.8, 0.8, 0.8, 1.0),    // Light gray
        }
    }

    /// Get short label for category.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::Web => "WEB",
            Self::Database => "DB",
            Self::Secure => "SEC",
            Self::System => "SYS",
            Self::Dev => "DEV",
            Self::Other => "-",
        }
    }
}

/// Format connection for display in table row.
///
/// # Returns
/// Formatted line for connection display
#[must_use]
pub fn format_connection_row(
    service: &str,
    local_port: u16,
    remote: &str,
    geo: &str,
    state_short: &str,
    age: &str,
    proc_name: &str,
) -> String {
    let local = format!(":{}", local_port);
    format!(
        "{:<5} {:<12} {:<17} {:<2} {:<3} {:<5} {}",
        service, local, remote, geo, state_short, age, proc_name
    )
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // build_connections_title tests
    // =========================================================================

    #[test]
    fn test_build_connections_title_basic() {
        let title = build_connections_title(45, 30, 5);
        assert!(title.contains("Connections"));
        assert!(title.contains("45"));
        assert!(title.contains("ESTAB: 30"));
        assert!(title.contains("LISTEN: 5"));
    }

    #[test]
    fn test_build_connections_title_zero() {
        let title = build_connections_title(0, 0, 0);
        assert!(title.contains("0"));
    }

    // =========================================================================
    // build_connections_title_compact tests
    // =========================================================================

    #[test]
    fn test_build_connections_title_compact_basic() {
        let title = build_connections_title_compact(45);
        assert!(title.contains("Conn"));
        assert!(title.contains("45"));
        assert!(!title.contains("ESTAB"));
    }

    // =========================================================================
    // TcpConnectionState tests
    // =========================================================================

    #[test]
    fn test_tcp_connection_state_display_name() {
        assert_eq!(
            TcpConnectionState::Established.display_name(),
            "ESTABLISHED"
        );
        assert_eq!(TcpConnectionState::Listen.display_name(), "LISTEN");
        assert_eq!(TcpConnectionState::TimeWait.display_name(), "TIME_WAIT");
    }

    #[test]
    fn test_tcp_connection_state_short_name() {
        assert_eq!(TcpConnectionState::Established.short_name(), "ESTAB");
        assert_eq!(TcpConnectionState::Listen.short_name(), "LSTN");
    }

    #[test]
    fn test_tcp_connection_state_color() {
        let color = TcpConnectionState::Established.color();
        assert!(color.g > 0.8, "Established should be green");

        let color = TcpConnectionState::Listen.color();
        assert!(color.b > 0.9, "Listen should be blue");

        let color = TcpConnectionState::Closed.color();
        assert!((color.r - color.g).abs() < 0.1, "Closed should be gray");
    }

    #[test]
    fn test_tcp_connection_state_is_active() {
        assert!(TcpConnectionState::Established.is_active());
        assert!(TcpConnectionState::Listen.is_active());
        assert!(!TcpConnectionState::TimeWait.is_active());
        assert!(!TcpConnectionState::Closed.is_active());
    }

    #[test]
    fn test_tcp_connection_state_is_closing() {
        assert!(TcpConnectionState::TimeWait.is_closing());
        assert!(TcpConnectionState::CloseWait.is_closing());
        assert!(TcpConnectionState::FinWait1.is_closing());
        assert!(!TcpConnectionState::Established.is_closing());
    }

    #[test]
    fn test_tcp_connection_state_default() {
        assert_eq!(
            TcpConnectionState::default(),
            TcpConnectionState::Established
        );
    }

    #[test]
    fn test_tcp_connection_state_derive_debug() {
        let state = TcpConnectionState::Listen;
        let debug = format!("{:?}", state);
        assert!(debug.contains("Listen"));
    }

    // =========================================================================
    // port_to_service_name tests
    // =========================================================================

    #[test]
    fn test_port_to_service_name_common() {
        assert_eq!(port_to_service_name(22), "SSH");
        assert_eq!(port_to_service_name(80), "HTTP");
        assert_eq!(port_to_service_name(443), "HTTPS");
    }

    #[test]
    fn test_port_to_service_name_databases() {
        assert_eq!(port_to_service_name(3306), "MySQL");
        assert_eq!(port_to_service_name(5432), "PgSQL");
        assert_eq!(port_to_service_name(6379), "Redis");
        assert_eq!(port_to_service_name(27017), "MongoDB");
    }

    #[test]
    fn test_port_to_service_name_vnc_range() {
        assert_eq!(port_to_service_name(5900), "VNC");
        assert_eq!(port_to_service_name(5901), "VNC");
        assert_eq!(port_to_service_name(5999), "VNC");
    }

    #[test]
    fn test_port_to_service_name_unknown() {
        assert_eq!(port_to_service_name(12345), "");
        assert_eq!(port_to_service_name(0), "");
    }

    // =========================================================================
    // port classification tests
    // =========================================================================

    #[test]
    fn test_is_well_known_port() {
        assert!(is_well_known_port(80));
        assert!(is_well_known_port(443));
        assert!(is_well_known_port(1023));
        assert!(!is_well_known_port(1024));
        assert!(!is_well_known_port(8080));
    }

    #[test]
    fn test_is_ephemeral_port() {
        assert!(!is_ephemeral_port(80));
        assert!(!is_ephemeral_port(49151));
        assert!(is_ephemeral_port(49152));
        assert!(is_ephemeral_port(65535));
    }

    // =========================================================================
    // format_address tests
    // =========================================================================

    #[test]
    fn test_format_address_fits() {
        let result = format_address("192.168.1.1", 20);
        assert_eq!(result, "192.168.1.1");
    }

    #[test]
    fn test_format_address_truncates() {
        let result = format_address("192.168.1.100", 10);
        assert_eq!(result.len(), 10);
        assert!(result.ends_with('~'));
    }

    #[test]
    fn test_format_address_very_short() {
        let result = format_address("192.168.1.1", 3);
        assert_eq!(result, "192");
    }

    #[test]
    fn test_format_socket_address() {
        let result = format_socket_address("192.168.1.1", 8080, 20);
        assert!(result.contains("192.168.1.1:8080"));
    }

    // =========================================================================
    // abbreviate_localhost tests
    // =========================================================================

    #[test]
    fn test_abbreviate_localhost_ipv4() {
        assert_eq!(abbreviate_localhost("127.0.0.1"), "localhost");
    }

    #[test]
    fn test_abbreviate_localhost_ipv6() {
        assert_eq!(abbreviate_localhost("::1"), "localhost");
    }

    #[test]
    fn test_abbreviate_localhost_any() {
        assert_eq!(abbreviate_localhost("0.0.0.0"), "*");
        assert_eq!(abbreviate_localhost("::"), "*");
    }

    #[test]
    fn test_abbreviate_localhost_other() {
        assert_eq!(abbreviate_localhost("192.168.1.1"), "192.168.1.1");
    }

    // =========================================================================
    // Protocol tests
    // =========================================================================

    #[test]
    fn test_protocol_display_name() {
        assert_eq!(Protocol::Tcp.display_name(), "TCP");
        assert_eq!(Protocol::Udp.display_name(), "UDP");
        assert_eq!(Protocol::Tcp6.display_name(), "TCP6");
    }

    #[test]
    fn test_protocol_is_ipv6() {
        assert!(!Protocol::Tcp.is_ipv6());
        assert!(!Protocol::Udp.is_ipv6());
        assert!(Protocol::Tcp6.is_ipv6());
        assert!(Protocol::Udp6.is_ipv6());
    }

    #[test]
    fn test_protocol_is_tcp() {
        assert!(Protocol::Tcp.is_tcp());
        assert!(Protocol::Tcp6.is_tcp());
        assert!(!Protocol::Udp.is_tcp());
        assert!(!Protocol::Udp6.is_tcp());
    }

    #[test]
    fn test_protocol_default() {
        assert_eq!(Protocol::default(), Protocol::Tcp);
    }

    // =========================================================================
    // ConnectionColumnWidths tests
    // =========================================================================

    #[test]
    fn test_connection_column_widths_default() {
        let widths = ConnectionColumnWidths::default();
        assert_eq!(widths.proto, 5);
        assert_eq!(widths.state, 8);
    }

    #[test]
    fn test_connection_column_widths_calculate() {
        let widths = ConnectionColumnWidths::calculate(100);
        assert!(widths.total() <= 100);
    }

    #[test]
    fn test_connection_column_widths_narrow() {
        let widths = ConnectionColumnWidths::calculate(40);
        // Should have minimum widths
        assert!(widths.local >= 10);
        assert!(widths.remote >= 10);
    }

    #[test]
    fn test_connection_column_widths_derive_debug() {
        let widths = ConnectionColumnWidths::default();
        let debug = format!("{:?}", widths);
        assert!(debug.contains("ConnectionColumnWidths"));
    }

    #[test]
    fn test_connection_column_widths_derive_clone() {
        let widths = ConnectionColumnWidths::calculate(100);
        let cloned = widths.clone();
        assert_eq!(widths, cloned);
    }

    // =========================================================================
    // Color constant tests
    // =========================================================================

    #[test]
    fn test_dim_color() {
        assert!((DIM_COLOR.r - 0.5).abs() < 0.01);
        assert!((DIM_COLOR.g - 0.5).abs() < 0.01);
        assert!((DIM_COLOR.b - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_active_color_is_green() {
        assert!(ACTIVE_COLOR.g > 0.8);
        assert!(ACTIVE_COLOR.r < 0.5);
    }

    #[test]
    fn test_listen_color_is_blue() {
        assert!(LISTEN_COLOR.b > 0.9);
    }

    // =========================================================================
    // Sparkline tests
    // =========================================================================

    #[test]
    fn test_sparkline_basic() {
        let values = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let sparkline = build_sparkline(&values, 12);
        assert!(!sparkline.is_empty());
        assert!(sparkline.starts_with(' '));
    }

    #[test]
    fn test_sparkline_empty_short_input() {
        let values = vec![0.5, 0.5]; // Less than 3 values
        let sparkline = build_sparkline(&values, 12);
        assert!(sparkline.is_empty());
    }

    #[test]
    fn test_sparkline_contains_chars() {
        let values = vec![0.0, 0.0, 0.0];
        let sparkline = build_sparkline(&values, 12);
        assert!(sparkline.contains('▁')); // Low value char
    }

    #[test]
    fn test_sparkline_max_chars() {
        let values: Vec<f64> = (0..20).map(|i| i as f64 / 20.0).collect();
        let sparkline = build_sparkline(&values, 5);
        // Should be limited to max 5 chars plus space prefix
        assert!(sparkline.chars().count() <= 6);
    }

    #[test]
    fn test_has_sparkline_data_true() {
        assert!(has_sparkline_data(3));
        assert!(has_sparkline_data(10));
    }

    #[test]
    fn test_has_sparkline_data_false() {
        assert!(!has_sparkline_data(0));
        assert!(!has_sparkline_data(2));
    }

    #[test]
    fn test_sparkline_high_values() {
        let values = vec![1.0, 1.0, 1.0, 1.0];
        let sparkline = build_sparkline(&values, 12);
        assert!(sparkline.contains('█')); // High value char
    }

    // =========================================================================
    // state_abbreviation tests
    // =========================================================================

    #[test]
    fn f_conn_state_001_established() {
        assert_eq!(state_abbreviation(TcpConnectionState::Established), "E");
    }

    #[test]
    fn f_conn_state_002_listen() {
        assert_eq!(state_abbreviation(TcpConnectionState::Listen), "L");
    }

    #[test]
    fn f_conn_state_003_time_wait() {
        assert_eq!(state_abbreviation(TcpConnectionState::TimeWait), "T");
    }

    #[test]
    fn f_conn_state_004_close_wait() {
        assert_eq!(state_abbreviation(TcpConnectionState::CloseWait), "C");
    }

    #[test]
    fn f_conn_state_005_syn_sent() {
        assert_eq!(state_abbreviation(TcpConnectionState::SynSent), "S");
    }

    #[test]
    fn f_conn_state_006_all_states_single_char() {
        // All state abbreviations should be single character
        for state in [
            TcpConnectionState::Established,
            TcpConnectionState::Listen,
            TcpConnectionState::TimeWait,
            TcpConnectionState::CloseWait,
            TcpConnectionState::SynSent,
            TcpConnectionState::SynRecv,
            TcpConnectionState::FinWait1,
            TcpConnectionState::FinWait2,
            TcpConnectionState::Closing,
            TcpConnectionState::LastAck,
            TcpConnectionState::Closed,
        ] {
            assert_eq!(
                state_abbreviation(state).chars().count(),
                1,
                "State {:?} should have single-char abbreviation",
                state
            );
        }
    }

    // =========================================================================
    // geo_indicator tests
    // =========================================================================

    #[test]
    fn f_conn_geo_001_listen() {
        assert_eq!(geo_indicator(true, false), "-");
        assert_eq!(geo_indicator(true, true), "-");
    }

    #[test]
    fn f_conn_geo_002_local() {
        assert_eq!(geo_indicator(false, true), "L");
    }

    #[test]
    fn f_conn_geo_003_remote() {
        assert_eq!(geo_indicator(false, false), "R");
    }

    // =========================================================================
    // port_to_service tests (GAP-CONN-003)
    // =========================================================================

    #[test]
    fn f_conn_svc_001_ssh() {
        assert_eq!(port_to_service(22), Some("ssh"));
    }

    #[test]
    fn f_conn_svc_002_http() {
        assert_eq!(port_to_service(80), Some("http"));
    }

    #[test]
    fn f_conn_svc_003_https() {
        assert_eq!(port_to_service(443), Some("https"));
    }

    #[test]
    fn f_conn_svc_004_mysql() {
        assert_eq!(port_to_service(3306), Some("mysql"));
    }

    #[test]
    fn f_conn_svc_005_postgres() {
        assert_eq!(port_to_service(5432), Some("postgres"));
    }

    #[test]
    fn f_conn_svc_006_redis() {
        assert_eq!(port_to_service(6379), Some("redis"));
    }

    #[test]
    fn f_conn_svc_007_unknown() {
        assert_eq!(port_to_service(12345), None);
    }

    #[test]
    fn f_conn_svc_008_x11_range() {
        assert_eq!(port_to_service(6000), Some("x11"));
        assert_eq!(port_to_service(6010), Some("x11"));
        assert_eq!(port_to_service(6063), Some("x11"));
    }

    #[test]
    fn f_conn_svc_009_service_name_or_port_known() {
        assert_eq!(service_name_or_port(22), "ssh");
        assert_eq!(service_name_or_port(80), "http");
    }

    #[test]
    fn f_conn_svc_010_service_name_or_port_unknown() {
        assert_eq!(service_name_or_port(12345), "12345");
    }

    // =========================================================================
    // ServiceCategory tests (GAP-CONN-003)
    // =========================================================================

    #[test]
    fn f_conn_cat_001_web() {
        assert_eq!(ServiceCategory::from_port(80), ServiceCategory::Web);
        assert_eq!(ServiceCategory::from_port(443), ServiceCategory::Web);
        assert_eq!(ServiceCategory::from_port(8080), ServiceCategory::Web);
    }

    #[test]
    fn f_conn_cat_002_database() {
        assert_eq!(ServiceCategory::from_port(3306), ServiceCategory::Database);
        assert_eq!(ServiceCategory::from_port(5432), ServiceCategory::Database);
        assert_eq!(ServiceCategory::from_port(6379), ServiceCategory::Database);
    }

    #[test]
    fn f_conn_cat_003_secure() {
        assert_eq!(ServiceCategory::from_port(22), ServiceCategory::Secure);
        assert_eq!(ServiceCategory::from_port(3389), ServiceCategory::Secure);
        assert_eq!(ServiceCategory::from_port(5900), ServiceCategory::Secure);
    }

    #[test]
    fn f_conn_cat_004_system() {
        assert_eq!(ServiceCategory::from_port(53), ServiceCategory::System);
        assert_eq!(ServiceCategory::from_port(123), ServiceCategory::System);
    }

    #[test]
    fn f_conn_cat_005_dev() {
        assert_eq!(ServiceCategory::from_port(3000), ServiceCategory::Dev);
        assert_eq!(ServiceCategory::from_port(8000), ServiceCategory::Dev);
    }

    #[test]
    fn f_conn_cat_006_other() {
        assert_eq!(ServiceCategory::from_port(12345), ServiceCategory::Other);
    }

    #[test]
    fn f_conn_cat_007_label() {
        assert_eq!(ServiceCategory::Web.label(), "WEB");
        assert_eq!(ServiceCategory::Database.label(), "DB");
        assert_eq!(ServiceCategory::Secure.label(), "SEC");
    }

    #[test]
    fn f_conn_cat_008_color_web_is_cyan() {
        let color = ServiceCategory::Web.color();
        assert!(color.b > 0.9, "Web category should be cyan");
    }

    #[test]
    fn f_conn_cat_009_color_secure_is_red() {
        let color = ServiceCategory::Secure.color();
        assert!(
            color.r > 0.9 && color.g < 0.5,
            "Secure category should be red"
        );
    }

    #[test]
    fn f_conn_cat_010_derive_debug() {
        let cat = ServiceCategory::Web;
        let debug = format!("{:?}", cat);
        assert!(debug.contains("Web"));
    }

    // =========================================================================
    // is_private_ip tests
    // =========================================================================

    #[test]
    fn f_conn_ip_001_loopback_v4() {
        use std::net::{IpAddr, Ipv4Addr};
        let ip = IpAddr::V4(Ipv4Addr::LOCALHOST);
        assert!(is_private_ip(&ip));
    }

    #[test]
    fn f_conn_ip_002_loopback_v6() {
        use std::net::{IpAddr, Ipv6Addr};
        let ip = IpAddr::V6(Ipv6Addr::LOCALHOST);
        assert!(is_private_ip(&ip));
    }

    #[test]
    fn f_conn_ip_003_private_192() {
        use std::net::{IpAddr, Ipv4Addr};
        let ip = IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1));
        assert!(is_private_ip(&ip));
    }

    #[test]
    fn f_conn_ip_004_private_10() {
        use std::net::{IpAddr, Ipv4Addr};
        let ip = IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1));
        assert!(is_private_ip(&ip));
    }

    #[test]
    fn f_conn_ip_005_public() {
        use std::net::{IpAddr, Ipv4Addr};
        let ip = IpAddr::V4(Ipv4Addr::new(8, 8, 8, 8));
        assert!(!is_private_ip(&ip));
    }

    // =========================================================================
    // hot_indicator_color tests
    // =========================================================================

    #[test]
    fn f_conn_hot_001_hot_is_orange() {
        let color = hot_indicator_color("●");
        assert!(color.r > 0.9, "Hot should have high red");
        assert!(color.g < 0.5, "Hot should have low green");
    }

    #[test]
    fn f_conn_hot_002_warm_is_yellow() {
        let color = hot_indicator_color("◐");
        assert!(color.r > 0.9, "Warm should have high red");
        assert!(color.g > 0.6, "Warm should have medium green");
    }

    #[test]
    fn f_conn_hot_003_inactive_is_dim() {
        let color = hot_indicator_color("");
        assert!(color.r < 0.6, "Inactive should be dim");
        assert!(color.g < 0.6, "Inactive should be dim");
    }

    // =========================================================================
    // truncate_process_name tests
    // =========================================================================

    #[test]
    fn f_conn_proc_001_short_name() {
        assert_eq!(truncate_process_name("chrome", 10), "chrome");
    }

    #[test]
    fn f_conn_proc_002_exact_length() {
        assert_eq!(truncate_process_name("0123456789", 10), "0123456789");
    }

    #[test]
    fn f_conn_proc_003_truncate_long() {
        let result = truncate_process_name("very_long_process", 10);
        assert_eq!(result.chars().count(), 10);
        assert!(result.ends_with('…'));
    }

    #[test]
    fn f_conn_proc_004_truncate_to_one() {
        let result = truncate_process_name("test", 1);
        assert_eq!(result, "t");
    }

    // =========================================================================
    // format_connection_row tests
    // =========================================================================

    #[test]
    fn f_conn_row_001_basic_format() {
        let row = format_connection_row("SSH", 22, "*", "-", "L", "1h", "sshd");
        assert!(row.contains("SSH"));
        assert!(row.contains(":22"));
        assert!(row.contains("sshd"));
    }

    #[test]
    fn f_conn_row_002_established() {
        let row = format_connection_row("HTTPS", 443, "1.2.3.4:8080", "R", "E", "5m", "chrome");
        assert!(row.contains("HTTPS"));
        assert!(row.contains("1.2.3.4:8080"));
        assert!(row.contains("chrome"));
    }

    #[test]
    fn f_conn_row_003_alignment() {
        let row = format_connection_row("SSH", 22, "*", "-", "L", "1h", "sshd");
        // Check that fixed-width columns create consistent spacing
        assert!(row.len() > 30, "Row should be reasonable width");
    }
}
