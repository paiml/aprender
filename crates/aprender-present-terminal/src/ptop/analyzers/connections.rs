//! Network Connections Analyzer
//!
//! Parses `/proc/net/tcp` and `/proc/net/tcp6` to show active network connections.
//! Maps sockets to processes via `/proc/[pid]/fd/` for PID/process name resolution.

#![allow(clippy::uninlined_format_args)]
#![allow(clippy::map_unwrap_or)]
#![allow(clippy::redundant_closure_for_method_calls)]

use std::collections::{HashMap, VecDeque};
use std::fs;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
use std::path::Path;
use std::time::{Duration, Instant};

use super::{Analyzer, AnalyzerError};

/// TCP connection state (from Linux kernel)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TcpState {
    Established = 1,
    SynSent = 2,
    SynRecv = 3,
    FinWait1 = 4,
    FinWait2 = 5,
    TimeWait = 6,
    Close = 7,
    CloseWait = 8,
    LastAck = 9,
    Listen = 10,
    Closing = 11,
    Unknown = 0,
}

impl TcpState {
    /// Parse state from hex string
    pub fn from_hex(hex: &str) -> Self {
        match u8::from_str_radix(hex, 16) {
            Ok(1) => Self::Established,
            Ok(2) => Self::SynSent,
            Ok(3) => Self::SynRecv,
            Ok(4) => Self::FinWait1,
            Ok(5) => Self::FinWait2,
            Ok(6) => Self::TimeWait,
            Ok(7) => Self::Close,
            Ok(8) => Self::CloseWait,
            Ok(9) => Self::LastAck,
            Ok(10) => Self::Listen,
            Ok(11) => Self::Closing,
            _ => Self::Unknown,
        }
    }

    /// Get display name
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Established => "ESTABLISHED",
            Self::SynSent => "SYN_SENT",
            Self::SynRecv => "SYN_RECV",
            Self::FinWait1 => "FIN_WAIT1",
            Self::FinWait2 => "FIN_WAIT2",
            Self::TimeWait => "TIME_WAIT",
            Self::Close => "CLOSE",
            Self::CloseWait => "CLOSE_WAIT",
            Self::LastAck => "LAST_ACK",
            Self::Listen => "LISTEN",
            Self::Closing => "CLOSING",
            Self::Unknown => "UNKNOWN",
        }
    }

    /// Short form for display
    pub fn short(&self) -> &'static str {
        match self {
            Self::Established => "ESTAB",
            Self::SynSent => "SYN_S",
            Self::SynRecv => "SYN_R",
            Self::FinWait1 => "FIN1",
            Self::FinWait2 => "FIN2",
            Self::TimeWait => "TIME_W",
            Self::Close => "CLOSE",
            Self::CloseWait => "CLOSEW",
            Self::LastAck => "LACK",
            Self::Listen => "LISTEN",
            Self::Closing => "CLOSNG",
            Self::Unknown => "UNK",
        }
    }
}

/// A single TCP connection
#[derive(Debug, Clone)]
pub struct TcpConnection {
    /// Local IP address
    pub local_addr: IpAddr,
    /// Local port
    pub local_port: u16,
    /// Remote IP address
    pub remote_addr: IpAddr,
    /// Remote port
    pub remote_port: u16,
    /// Connection state
    pub state: TcpState,
    /// Socket inode (for process mapping)
    pub inode: u64,
    /// User ID
    pub uid: u32,
    /// Process ID (if resolved)
    pub pid: Option<u32>,
    /// Process name (if resolved)
    pub process_name: Option<String>,
    /// IPv6 connection
    pub is_ipv6: bool,
    /// Time when connection was first seen (CB-CONN-001)
    pub first_seen: Option<Instant>,
}

/// Unique key for tracking connection age
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ConnectionKey {
    local_addr: IpAddr,
    local_port: u16,
    remote_addr: IpAddr,
    remote_port: u16,
}

impl TcpConnection {
    /// Get unique key for this connection (for tracking age)
    fn key(&self) -> ConnectionKey {
        ConnectionKey {
            local_addr: self.local_addr,
            local_port: self.local_port,
            remote_addr: self.remote_addr,
            remote_port: self.remote_port,
        }
    }

    /// Get connection age as Duration (CB-CONN-001)
    pub fn age(&self) -> Option<Duration> {
        self.first_seen.map(|t| t.elapsed())
    }

    /// Format age for display (e.g., "5s", "2m", "1h")
    pub fn age_display(&self) -> String {
        match self.age() {
            Some(age) => {
                let secs = age.as_secs();
                if secs < 60 {
                    format!("{}s", secs)
                } else if secs < 3600 {
                    format!("{}m", secs / 60)
                } else if secs < 86400 {
                    format!("{}h", secs / 3600)
                } else {
                    format!("{}d", secs / 86400)
                }
            }
            None => "-".to_string(),
        }
    }

    /// Format local address for display
    pub fn local_display(&self) -> String {
        format!("{}:{}", self.local_addr, self.local_port)
    }

    /// Format remote address for display
    pub fn remote_display(&self) -> String {
        if self.state == TcpState::Listen {
            "*:*".to_string()
        } else {
            format!("{}:{}", self.remote_addr, self.remote_port)
        }
    }

    /// Format process info for display
    pub fn process_display(&self) -> String {
        match (&self.pid, &self.process_name) {
            (Some(pid), Some(name)) => format!("{}/{}", pid, name),
            (Some(pid), None) => format!("{}", pid),
            _ => "-".to_string(),
        }
    }

    /// Check if connection is "hot" (recently established, likely active) - CB-CONN-006
    /// Returns true if connection is ESTABLISHED and was first seen within the last 5 seconds
    pub fn is_hot(&self) -> bool {
        if self.state != TcpState::Established {
            return false;
        }
        match self.age() {
            Some(age) => age.as_secs() < 5,
            None => false,
        }
    }

    /// Get hot indicator for display
    /// Returns symbol and color hint: ("🔥", "hot"), ("●", "warm"), ("", "normal")
    pub fn hot_indicator(&self) -> (&'static str, &'static str) {
        if !matches!(self.state, TcpState::Established) {
            return ("", "normal");
        }
        match self.age() {
            Some(age) => {
                let secs = age.as_secs();
                if secs < 5 {
                    ("●", "hot") // Just established, likely active
                } else if secs < 30 {
                    ("◐", "warm") // Recently active
                } else {
                    ("", "normal")
                }
            }
            None => ("", "normal"),
        }
    }

    /// Check if remote address is local (loopback or private) (CB-CONN-003)
    /// Returns true for 127.x.x.x, ::1, 10.x.x.x, 172.16-31.x.x, 192.168.x.x, fd00::/8
    pub fn is_remote_local(&self) -> bool {
        match self.remote_addr {
            IpAddr::V4(ip) => {
                ip.is_loopback() || ip.is_private() || ip.is_link_local() || ip.is_unspecified()
            }
            IpAddr::V6(ip) => {
                ip.is_loopback()
                    || ip.is_unspecified()
                    // Check for unique local address (fc00::/7, covers fd00::/8)
                    || {
                        let segs = ip.segments();
                        (segs[0] & 0xfe00) == 0xfc00
                    }
                    // Check for link-local (fe80::/10)
                    || {
                        let segs = ip.segments();
                        (segs[0] & 0xffc0) == 0xfe80
                    }
            }
        }
    }

    /// Get locality indicator for display (CB-CONN-003)
    /// Returns "L" for local/private, "R" for remote/internet
    pub fn locality_indicator(&self) -> &'static str {
        if self.state == TcpState::Listen {
            "" // Listen sockets don't have a meaningful remote
        } else if self.is_remote_local() {
            "L"
        } else {
            "R"
        }
    }

    /// Get locality with color hint (CB-CONN-003)
    /// Returns (indicator, color_hint) where color_hint is "local" or "remote"
    pub fn locality_display(&self) -> (&'static str, &'static str) {
        if self.state == TcpState::Listen {
            ("", "none")
        } else if self.is_remote_local() {
            ("L", "local")
        } else {
            ("R", "remote")
        }
    }
}

/// Connection count sample for sparkline (CB-CONN-007)
#[derive(Debug, Clone, Copy, Default)]
pub struct ConnectionCountSample {
    /// Total established connections
    pub established: usize,
    /// Total listening sockets
    pub listening: usize,
    /// Total connections (all states)
    pub total: usize,
}

/// Connections data
#[derive(Debug, Clone, Default)]
pub struct ConnectionsData {
    /// All TCP connections
    pub connections: Vec<TcpConnection>,
    /// Count by state
    pub state_counts: HashMap<TcpState, usize>,
    /// Connection count history for sparkline (CB-CONN-007)
    /// 60 samples = 60 seconds at 1 sample/sec, or 2 minutes at 2 sec intervals
    pub count_history: Vec<ConnectionCountSample>,
}

impl ConnectionsData {
    /// Get connections filtered by state
    pub fn by_state(&self, state: TcpState) -> impl Iterator<Item = &TcpConnection> {
        self.connections.iter().filter(move |c| c.state == state)
    }

    /// Get established connections only
    pub fn established(&self) -> impl Iterator<Item = &TcpConnection> {
        self.by_state(TcpState::Established)
    }

    /// Get listening sockets only
    pub fn listening(&self) -> impl Iterator<Item = &TcpConnection> {
        self.by_state(TcpState::Listen)
    }

    /// Total connection count
    pub fn total(&self) -> usize {
        self.connections.len()
    }

    /// Get sparkline data for established connections (CB-CONN-007)
    /// Returns values normalized to 0.0-1.0 range based on max in history
    pub fn established_sparkline(&self) -> Vec<f64> {
        if self.count_history.is_empty() {
            return vec![];
        }
        let max = self
            .count_history
            .iter()
            .map(|s| s.established)
            .max()
            .unwrap_or(1)
            .max(1);
        self.count_history
            .iter()
            .map(|s| s.established as f64 / max as f64)
            .collect()
    }

    /// Get sparkline data for total connections (CB-CONN-007)
    pub fn total_sparkline(&self) -> Vec<f64> {
        if self.count_history.is_empty() {
            return vec![];
        }
        let max = self
            .count_history
            .iter()
            .map(|s| s.total)
            .max()
            .unwrap_or(1)
            .max(1);
        self.count_history
            .iter()
            .map(|s| s.total as f64 / max as f64)
            .collect()
    }
}

/// Maximum history samples for sparkline (CB-CONN-007)
const MAX_HISTORY_SAMPLES: usize = 60;

/// Analyzer for network connections
pub struct ConnectionsAnalyzer {
    data: ConnectionsData,
    interval: Duration,
    /// Cache of inode -> (pid, name) mappings
    inode_cache: HashMap<u64, (u32, String)>,
    /// Track when connections were first seen (CB-CONN-001)
    connection_ages: HashMap<ConnectionKey, Instant>,
    /// Connection count history for sparkline (CB-CONN-007)
    count_history: VecDeque<ConnectionCountSample>,
    /// Last collection time — skip until interval elapses
    last_collect: Option<Instant>,
}

impl Default for ConnectionsAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl ConnectionsAnalyzer {
    /// Create a new connections analyzer
    pub fn new() -> Self {
        Self {
            data: ConnectionsData::default(),
            interval: Duration::from_secs(2),
            inode_cache: HashMap::new(),
            connection_ages: HashMap::new(),
            last_collect: None,
            count_history: VecDeque::with_capacity(MAX_HISTORY_SAMPLES),
        }
    }

    /// Get the current connections data
    pub fn data(&self) -> &ConnectionsData {
        &self.data
    }

    /// Parse /proc/net/tcp or /proc/net/tcp6
    fn parse_tcp_file(
        &mut self,
        path: &str,
        is_ipv6: bool,
    ) -> Result<Vec<TcpConnection>, AnalyzerError> {
        let contents = fs::read_to_string(path)
            .map_err(|e| AnalyzerError::IoError(format!("Failed to read {}: {}", path, e)))?;

        let mut connections = Vec::new();

        for line in contents.lines().skip(1) {
            // Skip header
            if let Some(conn) = self.parse_tcp_line(line, is_ipv6) {
                connections.push(conn);
            }
        }

        Ok(connections)
    }

    /// Parse a single line from /proc/net/tcp
    fn parse_tcp_line(&self, line: &str, is_ipv6: bool) -> Option<TcpConnection> {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 10 {
            return None;
        }

        // local_address: ip:port in hex
        let (local_addr, local_port) = Self::parse_address(parts[1], is_ipv6)?;
        let (remote_addr, remote_port) = Self::parse_address(parts[2], is_ipv6)?;

        let state = TcpState::from_hex(parts[3]);
        let uid: u32 = parts[7].parse().ok()?;
        let inode: u64 = parts[9].parse().ok()?;

        // Try to resolve process from cache
        let (pid, process_name) = self.inode_cache.get(&inode).cloned().unzip();

        Some(TcpConnection {
            local_addr,
            local_port,
            remote_addr,
            remote_port,
            state,
            inode,
            uid,
            pid,
            process_name,
            is_ipv6,
            first_seen: None, // Set by collect() based on connection_ages
        })
    }

    /// Parse hex address:port
    fn parse_address(s: &str, is_ipv6: bool) -> Option<(IpAddr, u16)> {
        let mut parts = s.split(':');
        let addr_hex = parts.next()?;
        let port_hex = parts.next()?;

        let port = u16::from_str_radix(port_hex, 16).ok()?;

        let addr = if is_ipv6 {
            Self::parse_ipv6(addr_hex)?
        } else {
            Self::parse_ipv4(addr_hex)?
        };

        Some((addr, port))
    }

    /// Parse IPv4 from hex (little-endian)
    fn parse_ipv4(hex: &str) -> Option<IpAddr> {
        if hex.len() != 8 {
            return None;
        }
        let bytes = u32::from_str_radix(hex, 16).ok()?;
        // Network byte order is big-endian, but /proc shows little-endian
        Some(IpAddr::V4(Ipv4Addr::from(bytes.to_be())))
    }

    /// Parse IPv6 from hex
    fn parse_ipv6(hex: &str) -> Option<IpAddr> {
        if hex.len() != 32 {
            return None;
        }

        // IPv6 in /proc is stored as 4 32-bit words in little-endian
        let mut segments = [0u16; 8];
        for i in 0..4 {
            let word_hex = &hex[i * 8..(i + 1) * 8];
            let word = u32::from_str_radix(word_hex, 16).ok()?.to_be();
            segments[i * 2] = (word >> 16) as u16;
            segments[i * 2 + 1] = word as u16;
        }

        Some(IpAddr::V6(Ipv6Addr::from(segments)))
    }

    /// Build inode to process mapping from /proc/[pid]/fd/
    fn build_inode_cache(&mut self) {
        self.inode_cache.clear();

        let proc_path = Path::new("/proc");
        let Ok(entries) = fs::read_dir(proc_path) else {
            return;
        };

        for entry in entries.flatten() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();

            // Only process numeric directories (PIDs)
            let Ok(pid) = name_str.parse::<u32>() else {
                continue;
            };

            // Read process name
            let pid_path = proc_path.join(name_str.as_ref());
            let comm_path = pid_path.join("comm");
            let process_name = fs::read_to_string(&comm_path)
                .map(|s| s.trim().to_string())
                .unwrap_or_default();

            // Scan fd directory for socket inodes
            let fd_path = pid_path.join("fd");
            let Ok(fd_entries) = fs::read_dir(&fd_path) else {
                continue;
            };

            for fd_entry in fd_entries.flatten() {
                let link = fd_entry.path();
                let Ok(target) = fs::read_link(&link) else {
                    continue;
                };

                let target_str = target.to_string_lossy();
                // Socket links look like: socket:[12345]
                if target_str.starts_with("socket:[") {
                    if let Some(inode_str) = target_str
                        .strip_prefix("socket:[")
                        .and_then(|s| s.strip_suffix(']'))
                    {
                        if let Ok(inode) = inode_str.parse::<u64>() {
                            self.inode_cache.insert(inode, (pid, process_name.clone()));
                        }
                    }
                }
            }
        }
    }

    /// Resolve process info for a connection
    fn resolve_process(&self, conn: &mut TcpConnection) {
        if let Some((pid, name)) = self.inode_cache.get(&conn.inode) {
            conn.pid = Some(*pid);
            conn.process_name = Some(name.clone());
        }
    }
}

impl Analyzer for ConnectionsAnalyzer {
    fn name(&self) -> &'static str {
        "connections"
    }

    fn collect(&mut self) -> Result<(), AnalyzerError> {
        // Interval guard: /proc/net/tcp + inode cache rebuild is expensive
        if let Some(last) = self.last_collect {
            if last.elapsed() < self.interval {
                return Ok(());
            }
        }
        self.last_collect = Some(Instant::now());

        // First, build the inode cache
        self.build_inode_cache();

        let mut all_connections = Vec::new();
        let now = Instant::now();

        // Parse IPv4 connections
        if Path::new("/proc/net/tcp").exists() {
            if let Ok(conns) = self.parse_tcp_file("/proc/net/tcp", false) {
                all_connections.extend(conns);
            }
        }

        // Parse IPv6 connections
        if Path::new("/proc/net/tcp6").exists() {
            if let Ok(conns) = self.parse_tcp_file("/proc/net/tcp6", true) {
                all_connections.extend(conns);
            }
        }

        // Resolve process info and set first_seen for each connection (CB-CONN-001)
        let mut current_keys = std::collections::HashSet::new();
        for conn in &mut all_connections {
            self.resolve_process(conn);

            // Track connection age
            let key = conn.key();
            current_keys.insert(key.clone());

            // Get or insert first_seen time
            let first_seen = *self.connection_ages.entry(key).or_insert(now);
            conn.first_seen = Some(first_seen);
        }

        // Clean up stale connection ages (connections that no longer exist)
        self.connection_ages.retain(|k, _| current_keys.contains(k));

        // Count by state
        let mut state_counts: HashMap<TcpState, usize> = HashMap::new();
        for conn in &all_connections {
            *state_counts.entry(conn.state).or_insert(0) += 1;
        }

        // Record connection count sample for sparkline (CB-CONN-007)
        let established = *state_counts.get(&TcpState::Established).unwrap_or(&0);
        let listening = *state_counts.get(&TcpState::Listen).unwrap_or(&0);
        let sample = ConnectionCountSample {
            established,
            listening,
            total: all_connections.len(),
        };

        // Maintain fixed-size history buffer
        if self.count_history.len() >= MAX_HISTORY_SAMPLES {
            self.count_history.pop_front();
        }
        self.count_history.push_back(sample);

        // Copy history to data for UI access
        let count_history: Vec<ConnectionCountSample> =
            self.count_history.iter().copied().collect();

        self.data = ConnectionsData {
            connections: all_connections,
            state_counts,
            count_history,
        };

        Ok(())
    }

    fn interval(&self) -> Duration {
        self.interval
    }

    fn available(&self) -> bool {
        Path::new("/proc/net/tcp").exists()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tcp_state_parsing() {
        assert_eq!(TcpState::from_hex("01"), TcpState::Established);
        assert_eq!(TcpState::from_hex("0A"), TcpState::Listen);
        assert_eq!(TcpState::from_hex("06"), TcpState::TimeWait);
        assert_eq!(TcpState::from_hex("FF"), TcpState::Unknown);
    }

    #[test]
    fn test_ipv4_parsing() {
        // 127.0.0.1 in little-endian hex is 0100007F
        let addr = ConnectionsAnalyzer::parse_ipv4("0100007F");
        assert!(addr.is_some());
        if let Some(IpAddr::V4(v4)) = addr {
            assert_eq!(v4, Ipv4Addr::new(127, 0, 0, 1));
        }

        // 0.0.0.0
        let addr = ConnectionsAnalyzer::parse_ipv4("00000000");
        assert!(addr.is_some());
        if let Some(IpAddr::V4(v4)) = addr {
            assert_eq!(v4, Ipv4Addr::new(0, 0, 0, 0));
        }
    }

    #[test]
    fn test_address_parsing() {
        // 127.0.0.1:53 (port 0x0035 = 53)
        let result = ConnectionsAnalyzer::parse_address("0100007F:0035", false);
        assert!(result.is_some());
        let (addr, port) = result.unwrap();
        assert_eq!(port, 53);
        if let IpAddr::V4(v4) = addr {
            assert_eq!(v4, Ipv4Addr::new(127, 0, 0, 1));
        }
    }

    #[test]
    fn test_analyzer_available() {
        let analyzer = ConnectionsAnalyzer::new();
        // Should be available on Linux
        #[cfg(target_os = "linux")]
        assert!(analyzer.available());
    }

    #[test]
    fn test_analyzer_collect() {
        let mut analyzer = ConnectionsAnalyzer::new();
        let result = analyzer.collect();

        // Should not fail even if we don't have access
        assert!(result.is_ok());

        // On Linux, we should have some connections
        #[cfg(target_os = "linux")]
        {
            let data = analyzer.data();
            // At least the test process should have sockets
            // (but we might not have permission to see them)
            let _ = data.total();
        }
    }

    #[test]
    fn test_tcp_state_display() {
        assert_eq!(TcpState::Established.as_str(), "ESTABLISHED");
        assert_eq!(TcpState::Listen.short(), "LISTEN");
        assert_eq!(TcpState::TimeWait.short(), "TIME_W");
    }

    #[test]
    fn test_connection_display() {
        let conn = TcpConnection {
            local_addr: IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)),
            local_port: 8080,
            remote_addr: IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1)),
            remote_port: 443,
            state: TcpState::Established,
            inode: 12345,
            uid: 1000,
            pid: Some(1234),
            process_name: Some("firefox".to_string()),
            is_ipv6: false,
            first_seen: None,
        };

        assert_eq!(conn.local_display(), "127.0.0.1:8080");
        assert_eq!(conn.remote_display(), "192.168.1.1:443");
        assert_eq!(conn.process_display(), "1234/firefox");
    }

    #[test]
    fn test_connection_age_display() {
        // F-CONN-001: Connection age must be non-negative
        let conn = TcpConnection {
            local_addr: IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)),
            local_port: 8080,
            remote_addr: IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1)),
            remote_port: 443,
            state: TcpState::Established,
            inode: 12345,
            uid: 1000,
            pid: None,
            process_name: None,
            is_ipv6: false,
            first_seen: Some(Instant::now()),
        };

        // Age should be very small (< 1 second)
        assert_eq!(conn.age_display(), "0s");

        // Connection without first_seen should show "-"
        let conn_no_age = TcpConnection {
            local_addr: IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)),
            local_port: 8080,
            remote_addr: IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1)),
            remote_port: 443,
            state: TcpState::Listen,
            inode: 0,
            uid: 0,
            pid: None,
            process_name: None,
            is_ipv6: false,
            first_seen: None,
        };
        assert_eq!(conn_no_age.age_display(), "-");
    }

    #[test]
    fn test_connection_hot_indicator() {
        // CB-CONN-006: Hot connection indicator
        // Recently established connection should be hot
        let hot_conn = TcpConnection {
            local_addr: IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)),
            local_port: 8080,
            remote_addr: IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1)),
            remote_port: 443,
            state: TcpState::Established,
            inode: 12345,
            uid: 1000,
            pid: None,
            process_name: None,
            is_ipv6: false,
            first_seen: Some(Instant::now()),
        };

        // Should be hot (just established)
        assert!(hot_conn.is_hot());
        let (indicator, level) = hot_conn.hot_indicator();
        assert_eq!(indicator, "●");
        assert_eq!(level, "hot");

        // Listen connections should never be hot
        let listen_conn = TcpConnection {
            local_addr: IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)),
            local_port: 80,
            remote_addr: IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)),
            remote_port: 0,
            state: TcpState::Listen,
            inode: 0,
            uid: 0,
            pid: None,
            process_name: None,
            is_ipv6: false,
            first_seen: Some(Instant::now()),
        };

        assert!(!listen_conn.is_hot());
        let (indicator, level) = listen_conn.hot_indicator();
        assert_eq!(indicator, "");
        assert_eq!(level, "normal");

        // Connection without first_seen should not be hot
        let no_age_conn = TcpConnection {
            local_addr: IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)),
            local_port: 8080,
            remote_addr: IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1)),
            remote_port: 443,
            state: TcpState::Established,
            inode: 0,
            uid: 0,
            pid: None,
            process_name: None,
            is_ipv6: false,
            first_seen: None,
        };

        assert!(!no_age_conn.is_hot());
        let (indicator, level) = no_age_conn.hot_indicator();
        assert_eq!(indicator, "");
        assert_eq!(level, "normal");
    }

    #[test]
    fn test_connection_count_sparkline() {
        // CB-CONN-007: Connection count sparkline
        let mut data = ConnectionsData::default();

        // Empty history should return empty sparkline
        assert!(data.established_sparkline().is_empty());
        assert!(data.total_sparkline().is_empty());

        // Add some samples
        data.count_history = vec![
            ConnectionCountSample {
                established: 5,
                listening: 2,
                total: 10,
            },
            ConnectionCountSample {
                established: 10,
                listening: 2,
                total: 15,
            },
            ConnectionCountSample {
                established: 8,
                listening: 3,
                total: 12,
            },
            ConnectionCountSample {
                established: 10,
                listening: 2,
                total: 14,
            },
        ];

        let sparkline = data.established_sparkline();
        assert_eq!(sparkline.len(), 4);

        // Max is 10, so values should be: 0.5, 1.0, 0.8, 1.0
        assert!((sparkline[0] - 0.5).abs() < 0.01);
        assert!((sparkline[1] - 1.0).abs() < 0.01);
        assert!((sparkline[2] - 0.8).abs() < 0.01);
        assert!((sparkline[3] - 1.0).abs() < 0.01);

        let total_sparkline = data.total_sparkline();
        assert_eq!(total_sparkline.len(), 4);
        // Max total is 15
        assert!((total_sparkline[1] - 1.0).abs() < 0.01); // 15/15 = 1.0
    }

    // ========================================================================
    // Locality indicator tests (CB-CONN-003)
    // ========================================================================

    #[test]
    fn test_connection_is_remote_local_loopback_v4() {
        let conn = TcpConnection {
            local_addr: IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)),
            local_port: 8080,
            remote_addr: IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)),
            remote_port: 9000,
            state: TcpState::Established,
            inode: 0,
            uid: 0,
            pid: None,
            process_name: None,
            is_ipv6: false,
            first_seen: None,
        };
        assert!(conn.is_remote_local());
        assert_eq!(conn.locality_indicator(), "L");
    }

    #[test]
    fn test_connection_is_remote_local_private_10() {
        let conn = TcpConnection {
            local_addr: IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1)),
            local_port: 8080,
            remote_addr: IpAddr::V4(Ipv4Addr::new(10, 1, 2, 3)),
            remote_port: 443,
            state: TcpState::Established,
            inode: 0,
            uid: 0,
            pid: None,
            process_name: None,
            is_ipv6: false,
            first_seen: None,
        };
        assert!(conn.is_remote_local());
        assert_eq!(conn.locality_indicator(), "L");
    }

    #[test]
    fn test_connection_is_remote_local_private_192_168() {
        let conn = TcpConnection {
            local_addr: IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1)),
            local_port: 8080,
            remote_addr: IpAddr::V4(Ipv4Addr::new(192, 168, 1, 100)),
            remote_port: 22,
            state: TcpState::Established,
            inode: 0,
            uid: 0,
            pid: None,
            process_name: None,
            is_ipv6: false,
            first_seen: None,
        };
        assert!(conn.is_remote_local());
    }

    #[test]
    fn test_connection_is_remote_internet() {
        let conn = TcpConnection {
            local_addr: IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1)),
            local_port: 54321,
            remote_addr: IpAddr::V4(Ipv4Addr::new(8, 8, 8, 8)), // Google DNS
            remote_port: 443,
            state: TcpState::Established,
            inode: 0,
            uid: 0,
            pid: None,
            process_name: None,
            is_ipv6: false,
            first_seen: None,
        };
        assert!(!conn.is_remote_local());
        assert_eq!(conn.locality_indicator(), "R");
        let (indicator, color) = conn.locality_display();
        assert_eq!(indicator, "R");
        assert_eq!(color, "remote");
    }

    #[test]
    fn test_connection_locality_listen() {
        let conn = TcpConnection {
            local_addr: IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)),
            local_port: 80,
            remote_addr: IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)),
            remote_port: 0,
            state: TcpState::Listen,
            inode: 0,
            uid: 0,
            pid: None,
            process_name: None,
            is_ipv6: false,
            first_seen: None,
        };
        assert_eq!(conn.locality_indicator(), "");
        let (indicator, color) = conn.locality_display();
        assert_eq!(indicator, "");
        assert_eq!(color, "none");
    }

    #[test]
    fn test_connection_is_remote_local_ipv6_loopback() {
        let conn = TcpConnection {
            local_addr: IpAddr::V6(Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0, 1)),
            local_port: 8080,
            remote_addr: IpAddr::V6(Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0, 1)),
            remote_port: 9000,
            state: TcpState::Established,
            inode: 0,
            uid: 0,
            pid: None,
            process_name: None,
            is_ipv6: true,
            first_seen: None,
        };
        assert!(conn.is_remote_local());
        assert_eq!(conn.locality_indicator(), "L");
    }

    #[test]
    fn test_connection_is_remote_local_ipv6_unique_local() {
        // fd00::/8 is a unique local address
        let conn = TcpConnection {
            local_addr: IpAddr::V6(Ipv6Addr::new(0xfd00, 0, 0, 0, 0, 0, 0, 1)),
            local_port: 8080,
            remote_addr: IpAddr::V6(Ipv6Addr::new(0xfd00, 0, 0, 0, 0, 0, 0, 2)),
            remote_port: 9000,
            state: TcpState::Established,
            inode: 0,
            uid: 0,
            pid: None,
            process_name: None,
            is_ipv6: true,
            first_seen: None,
        };
        assert!(conn.is_remote_local());
    }

    #[test]
    fn test_connection_is_remote_internet_ipv6() {
        // 2607:f8b0::1 is a Google IPv6 address
        let conn = TcpConnection {
            local_addr: IpAddr::V6(Ipv6Addr::new(0xfd00, 0, 0, 0, 0, 0, 0, 1)),
            local_port: 54321,
            remote_addr: IpAddr::V6(Ipv6Addr::new(0x2607, 0xf8b0, 0, 0, 0, 0, 0, 1)),
            remote_port: 443,
            state: TcpState::Established,
            inode: 0,
            uid: 0,
            pid: None,
            process_name: None,
            is_ipv6: true,
            first_seen: None,
        };
        assert!(!conn.is_remote_local());
        assert_eq!(conn.locality_indicator(), "R");
    }

    // ========================================================================
    // Falsification Tests for CB-CONN-003 (Phase 7 QA Gate)
    // ========================================================================

    /// F-LOC-001: RFC 1918 Private Network Audit
    /// All private addresses MUST be classified as Local
    #[test]
    fn test_f_loc_001_rfc1918_compliance() {
        // Test all RFC 1918 ranges:
        // 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16

        let test_cases = vec![
            // (remote_addr, expected_local, description)
            (
                Ipv4Addr::new(10, 0, 0, 1),
                true,
                "10.0.0.1 (Class A private)",
            ),
            (
                Ipv4Addr::new(10, 255, 255, 255),
                true,
                "10.255.255.255 (Class A edge)",
            ),
            (
                Ipv4Addr::new(172, 16, 0, 1),
                true,
                "172.16.0.1 (Class B private start)",
            ),
            (
                Ipv4Addr::new(172, 31, 255, 255),
                true,
                "172.31.255.255 (Class B private end)",
            ),
            (
                Ipv4Addr::new(172, 15, 0, 1),
                false,
                "172.15.0.1 (NOT private)",
            ),
            (
                Ipv4Addr::new(172, 32, 0, 1),
                false,
                "172.32.0.1 (NOT private)",
            ),
            (
                Ipv4Addr::new(192, 168, 0, 1),
                true,
                "192.168.0.1 (Class C private)",
            ),
            (
                Ipv4Addr::new(192, 168, 255, 255),
                true,
                "192.168.255.255 (Class C edge)",
            ),
            (Ipv4Addr::new(127, 0, 0, 1), true, "127.0.0.1 (Loopback)"),
            (
                Ipv4Addr::new(8, 8, 8, 8),
                false,
                "8.8.8.8 (Google DNS - public)",
            ),
            (
                Ipv4Addr::new(1, 1, 1, 1),
                false,
                "1.1.1.1 (Cloudflare - public)",
            ),
        ];

        for (remote, expected_local, desc) in test_cases {
            let conn = TcpConnection {
                local_addr: IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1)),
                local_port: 54321,
                remote_addr: IpAddr::V4(remote),
                remote_port: 443,
                state: TcpState::Established,
                inode: 0,
                uid: 0,
                pid: None,
                process_name: None,
                is_ipv6: false,
                first_seen: None,
            };

            assert_eq!(
                conn.is_remote_local(),
                expected_local,
                "RFC 1918 FAIL: {} should be local={}, got={}",
                desc,
                expected_local,
                conn.is_remote_local()
            );

            let expected_indicator = if expected_local { "L" } else { "R" };
            assert_eq!(
                conn.locality_indicator(),
                expected_indicator,
                "Indicator FAIL: {} should be '{}', got '{}'",
                desc,
                expected_indicator,
                conn.locality_indicator()
            );
        }
    }

    /// F-LOC-002: IPv6 Link-Local Edge Case (fe80::/10)
    /// Link-local addresses MUST be classified as Local
    #[test]
    fn test_f_loc_002_ipv6_link_local() {
        // fe80::/10 is link-local
        let conn = TcpConnection {
            local_addr: IpAddr::V6(Ipv6Addr::new(
                0xfe80, 0, 0, 0, 0x1234, 0x5678, 0x9abc, 0xdef0,
            )),
            local_port: 8080,
            remote_addr: IpAddr::V6(Ipv6Addr::new(
                0xfe80, 0, 0, 0, 0xabcd, 0xef01, 0x2345, 0x6789,
            )),
            remote_port: 9000,
            state: TcpState::Established,
            inode: 0,
            uid: 0,
            pid: None,
            process_name: None,
            is_ipv6: true,
            first_seen: None,
        };

        assert!(
            conn.is_remote_local(),
            "fe80::/10 link-local MUST be classified as Local"
        );
        assert_eq!(
            conn.locality_indicator(),
            "L",
            "fe80::/10 indicator MUST be 'L'"
        );
    }

    /// F-LOC-003: IPv6 ULA (fd00::/8) and fc00::/7 coverage
    #[test]
    fn test_f_loc_003_ipv6_ula_coverage() {
        let ula_cases = vec![
            (0xfd00, "fd00:: (common ULA)"),
            (0xfd12, "fd12:: (ULA variant)"),
            (0xfc00, "fc00:: (fc00::/7 start)"),
        ];

        for (prefix, desc) in ula_cases {
            let conn = TcpConnection {
                local_addr: IpAddr::V6(Ipv6Addr::new(prefix, 0, 0, 0, 0, 0, 0, 1)),
                local_port: 8080,
                remote_addr: IpAddr::V6(Ipv6Addr::new(prefix, 0x1234, 0, 0, 0, 0, 0, 2)),
                remote_port: 9000,
                state: TcpState::Established,
                inode: 0,
                uid: 0,
                pid: None,
                process_name: None,
                is_ipv6: true,
                first_seen: None,
            };

            assert!(
                conn.is_remote_local(),
                "{} MUST be classified as Local",
                desc
            );
        }
    }

    /// F-LOC-004: Global unicast IPv6 is Remote
    #[test]
    fn test_f_loc_004_ipv6_global_unicast_is_remote() {
        // 2000::/3 is global unicast
        let global_cases = vec![
            (0x2001, 0x0db8, "2001:db8:: (documentation)"),
            (0x2607, 0xf8b0, "2607:f8b0:: (Google)"),
            (0x2606, 0x4700, "2606:4700:: (Cloudflare)"),
        ];

        for (prefix1, prefix2, desc) in global_cases {
            let conn = TcpConnection {
                local_addr: IpAddr::V6(Ipv6Addr::new(0xfd00, 0, 0, 0, 0, 0, 0, 1)),
                local_port: 54321,
                remote_addr: IpAddr::V6(Ipv6Addr::new(prefix1, prefix2, 0, 0, 0, 0, 0, 1)),
                remote_port: 443,
                state: TcpState::Established,
                inode: 0,
                uid: 0,
                pid: None,
                process_name: None,
                is_ipv6: true,
                first_seen: None,
            };

            assert!(
                !conn.is_remote_local(),
                "{} MUST be classified as Remote",
                desc
            );
            assert_eq!(conn.locality_indicator(), "R");
        }
    }
}
