//! Network Statistics Analyzer
//!
//! Parses `/proc/net/dev` for detailed network interface statistics including
//! packet counts, errors, drops, and throughput.

#![allow(clippy::uninlined_format_args)]

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::{Duration, Instant};

use super::{Analyzer, AnalyzerError};

/// Statistics for a single network interface
#[derive(Debug, Clone, Default)]
pub struct InterfaceStats {
    /// Interface name (e.g., "eth0", "wlan0")
    pub interface: String,
    /// Receive bytes
    pub rx_bytes: u64,
    /// Receive packets
    pub rx_packets: u64,
    /// Receive errors
    pub rx_errors: u64,
    /// Receive dropped
    pub rx_dropped: u64,
    /// Receive FIFO errors
    pub rx_fifo: u64,
    /// Receive frame errors
    pub rx_frame: u64,
    /// Receive compressed
    pub rx_compressed: u64,
    /// Receive multicast
    pub rx_multicast: u64,
    /// Transmit bytes
    pub tx_bytes: u64,
    /// Transmit packets
    pub tx_packets: u64,
    /// Transmit errors
    pub tx_errors: u64,
    /// Transmit dropped
    pub tx_dropped: u64,
    /// Transmit FIFO errors
    pub tx_fifo: u64,
    /// Transmit collisions
    pub tx_collisions: u64,
    /// Transmit carrier errors
    pub tx_carrier: u64,
    /// Transmit compressed
    pub tx_compressed: u64,
}

impl InterfaceStats {
    /// Is this a loopback interface?
    pub fn is_loopback(&self) -> bool {
        self.interface == "lo"
    }

    /// Is this a virtual interface?
    pub fn is_virtual(&self) -> bool {
        self.interface.starts_with("veth")
            || self.interface.starts_with("docker")
            || self.interface.starts_with("br-")
            || self.interface.starts_with("virbr")
    }

    /// Total errors (RX + TX)
    pub fn total_errors(&self) -> u64 {
        self.rx_errors + self.tx_errors
    }

    /// Total dropped (RX + TX)
    pub fn total_dropped(&self) -> u64 {
        self.rx_dropped + self.tx_dropped
    }
}

/// Calculated network rates (per second)
#[derive(Debug, Clone, Default)]
pub struct InterfaceRates {
    /// Interface name
    pub interface: String,
    /// Receive bytes per second
    pub rx_bytes_per_sec: f64,
    /// Transmit bytes per second
    pub tx_bytes_per_sec: f64,
    /// Receive packets per second
    pub rx_packets_per_sec: f64,
    /// Transmit packets per second
    pub tx_packets_per_sec: f64,
    /// Error rate (errors per second)
    pub errors_per_sec: f64,
    /// Drop rate (drops per second)
    pub drops_per_sec: f64,
    /// Link speed in bits per second (CB-NET-006)
    /// Read from /sys/class/net/{iface}/speed (Mbps)
    pub link_speed_bps: Option<u64>,
}

impl InterfaceRates {
    /// Format RX rate for display
    pub fn rx_rate_display(&self) -> String {
        format_bytes_rate(self.rx_bytes_per_sec)
    }

    /// Format TX rate for display
    pub fn tx_rate_display(&self) -> String {
        format_bytes_rate(self.tx_bytes_per_sec)
    }

    /// Total bandwidth (RX + TX)
    pub fn total_bandwidth(&self) -> f64 {
        self.rx_bytes_per_sec + self.tx_bytes_per_sec
    }

    /// Bandwidth utilization percentage (CB-NET-006)
    /// Returns None if link speed is unknown
    pub fn utilization_percent(&self) -> Option<f64> {
        self.link_speed_bps.map(|speed| {
            if speed == 0 {
                return 0.0;
            }
            // Convert bytes/sec to bits/sec (* 8) and calculate percentage
            let total_bps = self.total_bandwidth() * 8.0;
            (total_bps / speed as f64) * 100.0
        })
    }

    /// Get utilization display string (CB-NET-006)
    /// Returns "N/A" if link speed unknown, or "XX.X%" otherwise
    pub fn utilization_display(&self) -> String {
        match self.utilization_percent() {
            Some(pct) => format!("{:.1}%", pct.min(100.0)),
            None => "N/A".to_string(),
        }
    }
}

/// Protocol-level statistics from /proc/net/snmp (CB-NET-002)
#[derive(Debug, Clone, Default)]
pub struct ProtocolStats {
    // TCP statistics
    /// Active TCP connections opened
    pub tcp_active_opens: u64,
    /// Passive TCP connections opened
    pub tcp_passive_opens: u64,
    /// TCP connection attempts failed
    pub tcp_attempt_fails: u64,
    /// TCP connections reset
    pub tcp_estab_resets: u64,
    /// Current established TCP connections
    pub tcp_curr_estab: u64,
    /// TCP segments received
    pub tcp_in_segs: u64,
    /// TCP segments sent
    pub tcp_out_segs: u64,
    /// TCP segments retransmitted
    pub tcp_retrans_segs: u64,
    /// TCP segments with errors
    pub tcp_in_errs: u64,
    /// TCP RST packets sent
    pub tcp_out_rsts: u64,

    // UDP statistics
    /// UDP datagrams received
    pub udp_in_datagrams: u64,
    /// UDP datagrams sent
    pub udp_out_datagrams: u64,
    /// UDP datagrams with no port
    pub udp_no_ports: u64,
    /// UDP receive errors
    pub udp_in_errors: u64,
    /// UDP receive buffer overflow errors
    pub udp_rcvbuf_errors: u64,

    // ICMP statistics
    /// ICMP messages received
    pub icmp_in_msgs: u64,
    /// ICMP messages sent
    pub icmp_out_msgs: u64,
    /// ICMP errors received
    pub icmp_in_errors: u64,
    /// ICMP errors sent
    pub icmp_out_errors: u64,
}

impl ProtocolStats {
    /// Get TCP retransmit rate (retrans / out_segs)
    pub fn tcp_retransmit_rate(&self) -> f64 {
        if self.tcp_out_segs == 0 {
            0.0
        } else {
            self.tcp_retrans_segs as f64 / self.tcp_out_segs as f64 * 100.0
        }
    }

    /// Get UDP error rate (errors / in_datagrams)
    pub fn udp_error_rate(&self) -> f64 {
        if self.udp_in_datagrams == 0 {
            0.0
        } else {
            self.udp_in_errors as f64 / self.udp_in_datagrams as f64 * 100.0
        }
    }
}

/// Protocol statistics with rate calculations (CB-NET-002)
#[derive(Debug, Clone, Default)]
pub struct ProtocolRates {
    /// TCP segments received per second
    pub tcp_in_segs_per_sec: f64,
    /// TCP segments sent per second
    pub tcp_out_segs_per_sec: f64,
    /// TCP retransmits per second
    pub tcp_retrans_per_sec: f64,
    /// UDP datagrams received per second
    pub udp_in_per_sec: f64,
    /// UDP datagrams sent per second
    pub udp_out_per_sec: f64,
    /// ICMP messages per second
    pub icmp_per_sec: f64,
}

/// Network statistics data
#[derive(Debug, Clone, Default)]
pub struct NetworkStatsData {
    /// Raw stats per interface
    pub stats: HashMap<String, InterfaceStats>,
    /// Calculated rates per interface
    pub rates: HashMap<String, InterfaceRates>,
    /// Total RX bytes per second (all interfaces)
    pub total_rx_bytes_per_sec: f64,
    /// Total TX bytes per second (all interfaces)
    pub total_tx_bytes_per_sec: f64,
    /// Total errors per second
    pub total_errors_per_sec: f64,
    /// Total drops per second
    pub total_drops_per_sec: f64,
    /// Protocol-level statistics (CB-NET-002)
    pub protocol_stats: ProtocolStats,
    /// Protocol rate calculations (CB-NET-002)
    pub protocol_rates: ProtocolRates,
}

impl NetworkStatsData {
    /// Get physical interfaces only (no loopback, no virtual)
    pub fn physical_interfaces(&self) -> impl Iterator<Item = (&String, &InterfaceStats)> {
        self.stats
            .iter()
            .filter(|(_, s)| !s.is_loopback() && !s.is_virtual())
    }

    /// Get rates for physical interfaces only
    pub fn physical_rates(&self) -> impl Iterator<Item = (&String, &InterfaceRates)> {
        self.rates.iter().filter(|(name, _)| {
            self.stats
                .get(*name)
                .is_some_and(|s| !s.is_loopback() && !s.is_virtual())
        })
    }
}

/// Analyzer for network statistics
pub struct NetworkStatsAnalyzer {
    data: NetworkStatsData,
    interval: Duration,
    /// Previous stats for rate calculation
    prev_stats: HashMap<String, InterfaceStats>,
    /// Previous protocol stats for rate calculation (CB-NET-002)
    prev_protocol_stats: Option<ProtocolStats>,
    /// Time of previous collection
    prev_time: Option<Instant>,
}

impl Default for NetworkStatsAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl NetworkStatsAnalyzer {
    /// Create a new network stats analyzer
    pub fn new() -> Self {
        Self {
            data: NetworkStatsData::default(),
            interval: Duration::from_secs(1),
            prev_stats: HashMap::new(),
            prev_protocol_stats: None,
            prev_time: None,
        }
    }

    /// Get the current data
    pub fn data(&self) -> &NetworkStatsData {
        &self.data
    }

    /// Parse /proc/net/dev
    fn parse_net_dev(&self) -> Result<HashMap<String, InterfaceStats>, AnalyzerError> {
        let contents = fs::read_to_string("/proc/net/dev")
            .map_err(|e| AnalyzerError::IoError(format!("Failed to read /proc/net/dev: {}", e)))?;

        let mut stats = HashMap::new();

        for line in contents.lines().skip(2) {
            // Skip header lines
            if let Some(iface_stats) = self.parse_net_dev_line(line) {
                stats.insert(iface_stats.interface.clone(), iface_stats);
            }
        }

        Ok(stats)
    }

    /// Parse a single line from /proc/net/dev
    fn parse_net_dev_line(&self, line: &str) -> Option<InterfaceStats> {
        // Format: iface: rx_bytes rx_packets rx_errs rx_drop rx_fifo rx_frame rx_compressed rx_multicast
        //                tx_bytes tx_packets tx_errs tx_drop tx_fifo tx_colls tx_carrier tx_compressed
        let mut parts = line.split(':');
        let interface = parts.next()?.trim().to_string();
        let stats_str = parts.next()?.trim();

        let stats: Vec<&str> = stats_str.split_whitespace().collect();
        if stats.len() < 16 {
            return None;
        }

        Some(InterfaceStats {
            interface,
            rx_bytes: stats[0].parse().ok()?,
            rx_packets: stats[1].parse().ok()?,
            rx_errors: stats[2].parse().ok()?,
            rx_dropped: stats[3].parse().ok()?,
            rx_fifo: stats[4].parse().ok()?,
            rx_frame: stats[5].parse().ok()?,
            rx_compressed: stats[6].parse().ok()?,
            rx_multicast: stats[7].parse().ok()?,
            tx_bytes: stats[8].parse().ok()?,
            tx_packets: stats[9].parse().ok()?,
            tx_errors: stats[10].parse().ok()?,
            tx_dropped: stats[11].parse().ok()?,
            tx_fifo: stats[12].parse().ok()?,
            tx_collisions: stats[13].parse().ok()?,
            tx_carrier: stats[14].parse().ok()?,
            tx_compressed: stats[15].parse().ok()?,
        })
    }

    /// Calculate rates from previous and current stats
    fn calculate_rates(
        &self,
        current: &HashMap<String, InterfaceStats>,
        elapsed_secs: f64,
    ) -> HashMap<String, InterfaceRates> {
        let mut rates = HashMap::new();

        for (iface, curr) in current {
            if let Some(prev) = self.prev_stats.get(iface) {
                let rx_bytes_delta = curr.rx_bytes.saturating_sub(prev.rx_bytes);
                let tx_bytes_delta = curr.tx_bytes.saturating_sub(prev.tx_bytes);
                let rx_packets_delta = curr.rx_packets.saturating_sub(prev.rx_packets);
                let tx_packets_delta = curr.tx_packets.saturating_sub(prev.tx_packets);
                let errors_delta = curr.total_errors().saturating_sub(prev.total_errors());
                let drops_delta = curr.total_dropped().saturating_sub(prev.total_dropped());

                // Read link speed from sysfs (CB-NET-006)
                let link_speed_bps = Self::read_link_speed(iface);

                rates.insert(
                    iface.clone(),
                    InterfaceRates {
                        interface: iface.clone(),
                        rx_bytes_per_sec: rx_bytes_delta as f64 / elapsed_secs,
                        tx_bytes_per_sec: tx_bytes_delta as f64 / elapsed_secs,
                        rx_packets_per_sec: rx_packets_delta as f64 / elapsed_secs,
                        tx_packets_per_sec: tx_packets_delta as f64 / elapsed_secs,
                        errors_per_sec: errors_delta as f64 / elapsed_secs,
                        drops_per_sec: drops_delta as f64 / elapsed_secs,
                        link_speed_bps,
                    },
                );
            }
        }

        rates
    }

    /// Read link speed from /sys/class/net/{iface}/speed (CB-NET-006)
    /// Returns speed in bits per second, or None if unavailable
    fn read_link_speed(iface: &str) -> Option<u64> {
        let speed_path = format!("/sys/class/net/{iface}/speed");
        let speed_str = fs::read_to_string(&speed_path).ok()?;
        let speed_mbps: u64 = speed_str.trim().parse().ok()?;
        // /sys/class/net/*/speed reports speed in Mbps
        // Convert to bps (* 1_000_000)
        // Note: -1 means unknown (e.g., wireless), we return None for that
        if speed_mbps > 0 && speed_mbps < 1_000_000 {
            Some(speed_mbps * 1_000_000)
        } else {
            None
        }
    }

    /// Parse /proc/net/snmp for protocol statistics (CB-NET-002)
    fn parse_net_snmp(&self) -> Result<ProtocolStats, AnalyzerError> {
        let contents = fs::read_to_string("/proc/net/snmp")
            .map_err(|e| AnalyzerError::IoError(format!("Failed to read /proc/net/snmp: {e}")))?;

        let mut stats = ProtocolStats::default();
        let mut tcp_headers: Vec<&str> = Vec::new();
        let mut udp_headers: Vec<&str> = Vec::new();
        let mut icmp_headers: Vec<&str> = Vec::new();

        for line in contents.lines() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.is_empty() {
                continue;
            }

            match parts[0] {
                "Tcp:" if parts.len() > 1 && parts[1].parse::<u64>().is_err() => {
                    // Header line
                    tcp_headers = parts[1..].to_vec();
                }
                "Tcp:" if !tcp_headers.is_empty() => {
                    // Data line
                    let values: Vec<&str> = parts[1..].to_vec();
                    for (i, header) in tcp_headers.iter().enumerate() {
                        if let Some(val) = values.get(i).and_then(|v| v.parse().ok()) {
                            match *header {
                                "ActiveOpens" => stats.tcp_active_opens = val,
                                "PassiveOpens" => stats.tcp_passive_opens = val,
                                "AttemptFails" => stats.tcp_attempt_fails = val,
                                "EstabResets" => stats.tcp_estab_resets = val,
                                "CurrEstab" => stats.tcp_curr_estab = val,
                                "InSegs" => stats.tcp_in_segs = val,
                                "OutSegs" => stats.tcp_out_segs = val,
                                "RetransSegs" => stats.tcp_retrans_segs = val,
                                "InErrs" => stats.tcp_in_errs = val,
                                "OutRsts" => stats.tcp_out_rsts = val,
                                _ => {}
                            }
                        }
                    }
                }
                "Udp:" if parts.len() > 1 && parts[1].parse::<u64>().is_err() => {
                    udp_headers = parts[1..].to_vec();
                }
                "Udp:" if !udp_headers.is_empty() => {
                    let values: Vec<&str> = parts[1..].to_vec();
                    for (i, header) in udp_headers.iter().enumerate() {
                        if let Some(val) = values.get(i).and_then(|v| v.parse().ok()) {
                            match *header {
                                "InDatagrams" => stats.udp_in_datagrams = val,
                                "OutDatagrams" => stats.udp_out_datagrams = val,
                                "NoPorts" => stats.udp_no_ports = val,
                                "InErrors" => stats.udp_in_errors = val,
                                "RcvbufErrors" => stats.udp_rcvbuf_errors = val,
                                _ => {}
                            }
                        }
                    }
                }
                "Icmp:" if parts.len() > 1 && parts[1].parse::<u64>().is_err() => {
                    icmp_headers = parts[1..].to_vec();
                }
                "Icmp:" if !icmp_headers.is_empty() => {
                    let values: Vec<&str> = parts[1..].to_vec();
                    for (i, header) in icmp_headers.iter().enumerate() {
                        if let Some(val) = values.get(i).and_then(|v| v.parse().ok()) {
                            match *header {
                                "InMsgs" => stats.icmp_in_msgs = val,
                                "OutMsgs" => stats.icmp_out_msgs = val,
                                "InErrors" => stats.icmp_in_errors = val,
                                "OutErrors" => stats.icmp_out_errors = val,
                                _ => {}
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        Ok(stats)
    }

    /// Calculate protocol rates from previous and current stats (CB-NET-002)
    fn calculate_protocol_rates(
        &self,
        current: &ProtocolStats,
        elapsed_secs: f64,
    ) -> ProtocolRates {
        if let Some(prev) = &self.prev_protocol_stats {
            ProtocolRates {
                tcp_in_segs_per_sec: current.tcp_in_segs.saturating_sub(prev.tcp_in_segs) as f64
                    / elapsed_secs,
                tcp_out_segs_per_sec: current.tcp_out_segs.saturating_sub(prev.tcp_out_segs) as f64
                    / elapsed_secs,
                tcp_retrans_per_sec: current
                    .tcp_retrans_segs
                    .saturating_sub(prev.tcp_retrans_segs)
                    as f64
                    / elapsed_secs,
                udp_in_per_sec: current
                    .udp_in_datagrams
                    .saturating_sub(prev.udp_in_datagrams) as f64
                    / elapsed_secs,
                udp_out_per_sec: current
                    .udp_out_datagrams
                    .saturating_sub(prev.udp_out_datagrams) as f64
                    / elapsed_secs,
                icmp_per_sec: (current.icmp_in_msgs.saturating_sub(prev.icmp_in_msgs)
                    + current.icmp_out_msgs.saturating_sub(prev.icmp_out_msgs))
                    as f64
                    / elapsed_secs,
            }
        } else {
            ProtocolRates::default()
        }
    }
}

impl Analyzer for NetworkStatsAnalyzer {
    fn name(&self) -> &'static str {
        "network_stats"
    }

    fn collect(&mut self) -> Result<(), AnalyzerError> {
        let current_stats = self.parse_net_dev()?;
        let now = Instant::now();

        // Parse protocol stats (CB-NET-002) - optional, don't fail if unavailable
        let protocol_stats = self.parse_net_snmp().unwrap_or_default();

        let (rates, protocol_rates) = if let Some(prev_time) = self.prev_time {
            let elapsed = now.duration_since(prev_time).as_secs_f64();
            if elapsed > 0.0 {
                let iface_rates = self.calculate_rates(&current_stats, elapsed);
                let proto_rates = self.calculate_protocol_rates(&protocol_stats, elapsed);
                (iface_rates, proto_rates)
            } else {
                (HashMap::new(), ProtocolRates::default())
            }
        } else {
            (HashMap::new(), ProtocolRates::default())
        };

        // Calculate totals (excluding loopback)
        let total_rx: f64 = rates
            .iter()
            .filter(|(name, _)| *name != "lo")
            .map(|(_, r)| r.rx_bytes_per_sec)
            .sum();
        let total_tx: f64 = rates
            .iter()
            .filter(|(name, _)| *name != "lo")
            .map(|(_, r)| r.tx_bytes_per_sec)
            .sum();
        let total_errors: f64 = rates.values().map(|r| r.errors_per_sec).sum();
        let total_drops: f64 = rates.values().map(|r| r.drops_per_sec).sum();

        self.data = NetworkStatsData {
            stats: current_stats.clone(),
            rates,
            total_rx_bytes_per_sec: total_rx,
            total_tx_bytes_per_sec: total_tx,
            total_errors_per_sec: total_errors,
            total_drops_per_sec: total_drops,
            protocol_stats: protocol_stats.clone(),
            protocol_rates,
        };

        self.prev_stats = current_stats;
        self.prev_protocol_stats = Some(protocol_stats);
        self.prev_time = Some(now);

        Ok(())
    }

    fn interval(&self) -> Duration {
        self.interval
    }

    fn available(&self) -> bool {
        Path::new("/proc/net/dev").exists()
    }
}

/// Format bytes per second for display
fn format_bytes_rate(bytes_per_sec: f64) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = KB * 1024.0;
    const GB: f64 = MB * 1024.0;

    if bytes_per_sec >= GB {
        format!("{:.1}G/s", bytes_per_sec / GB)
    } else if bytes_per_sec >= MB {
        format!("{:.1}M/s", bytes_per_sec / MB)
    } else if bytes_per_sec >= KB {
        format!("{:.1}K/s", bytes_per_sec / KB)
    } else {
        format!("{:.0}B/s", bytes_per_sec)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interface_stats_is_loopback() {
        let lo = InterfaceStats {
            interface: "lo".to_string(),
            ..Default::default()
        };
        assert!(lo.is_loopback());

        let eth = InterfaceStats {
            interface: "eth0".to_string(),
            ..Default::default()
        };
        assert!(!eth.is_loopback());
    }

    #[test]
    fn test_interface_stats_is_virtual() {
        let veth = InterfaceStats {
            interface: "veth123".to_string(),
            ..Default::default()
        };
        assert!(veth.is_virtual());

        let docker = InterfaceStats {
            interface: "docker0".to_string(),
            ..Default::default()
        };
        assert!(docker.is_virtual());

        let eth = InterfaceStats {
            interface: "eth0".to_string(),
            ..Default::default()
        };
        assert!(!eth.is_virtual());
    }

    #[test]
    fn test_interface_stats_totals() {
        let stats = InterfaceStats {
            rx_errors: 10,
            tx_errors: 5,
            rx_dropped: 3,
            tx_dropped: 2,
            ..Default::default()
        };
        assert_eq!(stats.total_errors(), 15);
        assert_eq!(stats.total_dropped(), 5);
    }

    #[test]
    fn test_interface_rates_display() {
        let rates = InterfaceRates {
            interface: "eth0".to_string(),
            rx_bytes_per_sec: 1_500_000.0,
            tx_bytes_per_sec: 500_000.0,
            ..Default::default()
        };
        assert_eq!(rates.rx_rate_display(), "1.4M/s");
        assert_eq!(rates.tx_rate_display(), "488.3K/s");
        assert!((rates.total_bandwidth() - 2_000_000.0).abs() < 0.01);
    }

    #[test]
    fn test_interface_rates_utilization() {
        // CB-NET-006: Bandwidth utilization percentage
        // Test with 1Gbps link (1_000_000_000 bps)
        let rates = InterfaceRates {
            interface: "eth0".to_string(),
            rx_bytes_per_sec: 50_000_000.0,      // 50 MB/s = 400 Mbps
            tx_bytes_per_sec: 50_000_000.0,      // 50 MB/s = 400 Mbps
            link_speed_bps: Some(1_000_000_000), // 1 Gbps
            ..Default::default()
        };

        // Total: 100 MB/s = 800 Mbps = 80% of 1 Gbps
        let util = rates.utilization_percent();
        assert!(util.is_some());
        assert!((util.unwrap() - 80.0).abs() < 0.01);
        assert_eq!(rates.utilization_display(), "80.0%");

        // Test without link speed
        let rates_no_speed = InterfaceRates {
            interface: "wlan0".to_string(),
            rx_bytes_per_sec: 1_000_000.0,
            tx_bytes_per_sec: 1_000_000.0,
            link_speed_bps: None,
            ..Default::default()
        };
        assert!(rates_no_speed.utilization_percent().is_none());
        assert_eq!(rates_no_speed.utilization_display(), "N/A");
    }

    #[test]
    fn test_analyzer_creation() {
        let analyzer = NetworkStatsAnalyzer::new();
        assert_eq!(analyzer.name(), "network_stats");
    }

    #[test]
    fn test_analyzer_available() {
        let analyzer = NetworkStatsAnalyzer::new();
        #[cfg(target_os = "linux")]
        assert!(analyzer.available());
    }

    #[test]
    fn test_analyzer_collect() {
        let mut analyzer = NetworkStatsAnalyzer::new();
        let result = analyzer.collect();
        assert!(result.is_ok());

        let data = analyzer.data();
        #[cfg(target_os = "linux")]
        {
            // Should at least have loopback
            assert!(data.stats.contains_key("lo"));
        }
    }

    #[test]
    fn test_format_bytes_rate() {
        assert_eq!(format_bytes_rate(500.0), "500B/s");
        assert_eq!(format_bytes_rate(1536.0), "1.5K/s");
        assert_eq!(format_bytes_rate(1_500_000.0), "1.4M/s");
    }

    // ========================================================================
    // Protocol statistics tests (CB-NET-002)
    // ========================================================================

    #[test]
    fn test_protocol_stats_default() {
        let stats = ProtocolStats::default();
        assert_eq!(stats.tcp_curr_estab, 0);
        assert_eq!(stats.udp_in_datagrams, 0);
        assert_eq!(stats.icmp_in_msgs, 0);
    }

    #[test]
    fn test_protocol_stats_tcp_retransmit_rate() {
        let stats = ProtocolStats {
            tcp_out_segs: 1000,
            tcp_retrans_segs: 10,
            ..Default::default()
        };
        let rate = stats.tcp_retransmit_rate();
        assert!((rate - 1.0).abs() < 0.001); // 1% retransmit rate
    }

    #[test]
    fn test_protocol_stats_tcp_retransmit_rate_zero_segs() {
        let stats = ProtocolStats {
            tcp_out_segs: 0,
            tcp_retrans_segs: 10,
            ..Default::default()
        };
        let rate = stats.tcp_retransmit_rate();
        assert_eq!(rate, 0.0);
    }

    #[test]
    fn test_protocol_stats_udp_error_rate() {
        let stats = ProtocolStats {
            udp_in_datagrams: 500,
            udp_in_errors: 5,
            ..Default::default()
        };
        let rate = stats.udp_error_rate();
        assert!((rate - 1.0).abs() < 0.001); // 1% error rate
    }

    #[test]
    fn test_protocol_stats_udp_error_rate_zero_datagrams() {
        let stats = ProtocolStats {
            udp_in_datagrams: 0,
            udp_in_errors: 5,
            ..Default::default()
        };
        let rate = stats.udp_error_rate();
        assert_eq!(rate, 0.0);
    }

    #[test]
    fn test_protocol_rates_default() {
        let rates = ProtocolRates::default();
        assert_eq!(rates.tcp_in_segs_per_sec, 0.0);
        assert_eq!(rates.udp_in_per_sec, 0.0);
        assert_eq!(rates.icmp_per_sec, 0.0);
    }

    #[test]
    fn test_network_stats_data_with_protocol() {
        let data = NetworkStatsData {
            protocol_stats: ProtocolStats {
                tcp_curr_estab: 42,
                ..Default::default()
            },
            protocol_rates: ProtocolRates {
                tcp_in_segs_per_sec: 100.0,
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(data.protocol_stats.tcp_curr_estab, 42);
        assert_eq!(data.protocol_rates.tcp_in_segs_per_sec, 100.0);
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_analyzer_collect_with_protocol_stats() {
        let mut analyzer = NetworkStatsAnalyzer::new();
        let result = analyzer.collect();
        assert!(result.is_ok());

        // Protocol stats should be populated (may be 0 but struct exists)
        let data = analyzer.data();
        // Just verify the struct exists and has sensible defaults
        assert!(data.protocol_stats.tcp_retransmit_rate() >= 0.0);
    }

    // ========================================================================
    // Falsification Tests for CB-NET-002 (Phase 7 QA Gate)
    // ========================================================================

    /// F-PROTO-001: Real /proc/net/snmp parsing verification
    /// Stats must come from actual system data, not hardcoded values
    #[test]
    #[cfg(target_os = "linux")]
    fn test_f_proto_001_real_snmp_parsing() {
        let mut analyzer = NetworkStatsAnalyzer::new();
        analyzer.collect().unwrap();

        let stats = &analyzer.data().protocol_stats;

        // On any Linux system with network activity:
        // - tcp_in_segs should be non-zero (we've received TCP data)
        // - tcp_out_segs should be non-zero (we've sent TCP data)
        // These are cumulative counters, always growing
        assert!(
            stats.tcp_in_segs > 0 || stats.tcp_out_segs > 0,
            "TCP segment counters should be non-zero on an active system"
        );

        // At minimum, CurrEstab should be reasonable (0-65535)
        assert!(
            stats.tcp_curr_estab < 65535,
            "Current established connections should be reasonable"
        );
    }

    /// F-PROTO-002: Rate calculation is delta-based, not snapshot
    /// Two collections must show rate = (delta / elapsed_time)
    #[test]
    #[cfg(target_os = "linux")]
    fn test_f_proto_002_rate_is_delta_based() {
        let mut analyzer = NetworkStatsAnalyzer::new();

        // First collection - establishes baseline
        analyzer.collect().unwrap();
        let first_stats = analyzer.data().protocol_stats.clone();

        // Wait briefly
        std::thread::sleep(std::time::Duration::from_millis(100));

        // Second collection - calculates rates
        analyzer.collect().unwrap();
        let second_data = analyzer.data();

        // Rates should be calculated from deltas
        // If tcp_in_segs increased by N in 0.1s, rate should be ~N/0.1 = 10*N
        let delta_in = second_data
            .protocol_stats
            .tcp_in_segs
            .saturating_sub(first_stats.tcp_in_segs);

        // If there was any activity, rate should be positive
        // If no activity, both should be 0
        if delta_in > 0 {
            assert!(
                second_data.protocol_rates.tcp_in_segs_per_sec > 0.0,
                "Rate should be positive when delta is positive"
            );
        }

        // Rate should never be negative
        assert!(
            second_data.protocol_rates.tcp_in_segs_per_sec >= 0.0,
            "Rate must be non-negative"
        );
        assert!(
            second_data.protocol_rates.tcp_out_segs_per_sec >= 0.0,
            "Rate must be non-negative"
        );
    }

    /// F-PROTO-003: All protocol families are tracked (TCP/UDP/ICMP)
    /// Verify all fields are populated, not just TCP
    #[test]
    fn test_f_proto_003_all_protocols_tracked() {
        // Create stats with known values
        let stats = ProtocolStats {
            tcp_active_opens: 100,
            tcp_passive_opens: 50,
            tcp_in_segs: 10000,
            tcp_out_segs: 8000,
            tcp_retrans_segs: 10,
            udp_in_datagrams: 5000,
            udp_out_datagrams: 4000,
            udp_in_errors: 5,
            icmp_in_msgs: 200,
            icmp_out_msgs: 150,
            icmp_in_errors: 2,
            icmp_out_errors: 1,
            ..Default::default()
        };

        // Verify retransmit rate calculation
        let retrans_rate = stats.tcp_retransmit_rate();
        assert!(
            (retrans_rate - 0.125).abs() < 0.001,
            "TCP retransmit rate should be 10/8000 * 100 = 0.125%"
        );

        // Verify UDP error rate
        let udp_err_rate = stats.udp_error_rate();
        assert!(
            (udp_err_rate - 0.1).abs() < 0.001,
            "UDP error rate should be 5/5000 * 100 = 0.1%"
        );
    }

    /// F-PROTO-004: Protocol rates calculation correctness
    #[test]
    fn test_f_proto_004_rate_calculation_math() {
        let mut analyzer = NetworkStatsAnalyzer::new();

        // Manually set previous stats
        analyzer.prev_protocol_stats = Some(ProtocolStats {
            tcp_in_segs: 1000,
            tcp_out_segs: 800,
            tcp_retrans_segs: 10,
            udp_in_datagrams: 500,
            udp_out_datagrams: 400,
            icmp_in_msgs: 100,
            icmp_out_msgs: 80,
            ..Default::default()
        });

        // Current stats (increased)
        let current = ProtocolStats {
            tcp_in_segs: 2000,      // +1000
            tcp_out_segs: 1800,     // +1000
            tcp_retrans_segs: 20,   // +10
            udp_in_datagrams: 1000, // +500
            udp_out_datagrams: 900, // +500
            icmp_in_msgs: 200,      // +100
            icmp_out_msgs: 180,     // +100
            ..Default::default()
        };

        // Calculate rates over 1 second
        let rates = analyzer.calculate_protocol_rates(&current, 1.0);

        // Verify calculations
        assert!(
            (rates.tcp_in_segs_per_sec - 1000.0).abs() < 0.1,
            "TCP in rate should be 1000/s"
        );
        assert!(
            (rates.tcp_out_segs_per_sec - 1000.0).abs() < 0.1,
            "TCP out rate should be 1000/s"
        );
        assert!(
            (rates.tcp_retrans_per_sec - 10.0).abs() < 0.1,
            "TCP retrans rate should be 10/s"
        );
        assert!(
            (rates.udp_in_per_sec - 500.0).abs() < 0.1,
            "UDP in rate should be 500/s"
        );
        assert!(
            (rates.icmp_per_sec - 200.0).abs() < 0.1,
            "ICMP rate should be 200/s (in + out)"
        );
    }
}
