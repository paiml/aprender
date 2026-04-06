//! Network metrics collector.
//!
//! Parses `/proc/net/dev` on Linux to collect network interface metrics.
//!
//! ## Falsification Criteria
//!
//! - #38: Network throughput matches `iftop` within ±5%
//! - #44: Network packet counts are monotonically increasing

use crate::monitor::error::{MonitorError, Result};
use crate::monitor::ring_buffer::RingBuffer;
#[cfg(target_os = "macos")]
use crate::monitor::subprocess::run_with_timeout;
use crate::monitor::types::{Collector, MetricValue, Metrics};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Raw statistics for a network interface.
#[derive(Debug, Clone, Default)]
pub struct NetStats {
    /// Interface name.
    pub name: String,
    /// Bytes received.
    pub rx_bytes: u64,
    /// Packets received.
    pub rx_packets: u64,
    /// Receive errors.
    pub rx_errors: u64,
    /// Receive drops.
    pub rx_drops: u64,
    /// Bytes transmitted.
    pub tx_bytes: u64,
    /// Packets transmitted.
    pub tx_packets: u64,
    /// Transmit errors.
    pub tx_errors: u64,
    /// Transmit drops.
    pub tx_drops: u64,
}

/// Calculated network rates.
#[derive(Debug, Clone, Default)]
pub struct NetRates {
    /// Interface name.
    pub name: String,
    /// Download rate in bytes/sec.
    pub rx_bytes_per_sec: f64,
    /// Upload rate in bytes/sec.
    pub tx_bytes_per_sec: f64,
    /// Receive packets per second.
    pub rx_packets_per_sec: f64,
    /// Transmit packets per second.
    pub tx_packets_per_sec: f64,
}

impl NetRates {
    /// Returns download rate in bits per second.
    #[must_use]
    pub fn rx_bits_per_sec(&self) -> f64 {
        self.rx_bytes_per_sec * 8.0
    }

    /// Returns upload rate in bits per second.
    #[must_use]
    pub fn tx_bits_per_sec(&self) -> f64 {
        self.tx_bytes_per_sec * 8.0
    }

    /// Formats the download rate as a human-readable string.
    #[must_use]
    pub fn rx_formatted(&self) -> String {
        format_bytes_rate(self.rx_bytes_per_sec)
    }

    /// Formats the upload rate as a human-readable string.
    #[must_use]
    pub fn tx_formatted(&self) -> String {
        format_bytes_rate(self.tx_bytes_per_sec)
    }
}

/// Formats bytes per second as a human-readable string.
fn format_bytes_rate(bytes_per_sec: f64) -> String {
    batuta_common::fmt::format_bytes_rate(bytes_per_sec)
}

/// Collector for network metrics.
#[derive(Debug)]
pub struct NetworkCollector {
    /// Previous stats for delta calculation.
    prev_stats: HashMap<String, NetStats>,
    /// Previous collection time.
    prev_time: Option<Instant>,
    /// Calculated rates per interface.
    rates: HashMap<String, NetRates>,
    /// Download history per interface (normalized 0-1).
    rx_history: HashMap<String, RingBuffer<f64>>,
    /// Upload history per interface (normalized 0-1).
    tx_history: HashMap<String, RingBuffer<f64>>,
    /// Current interface for primary display.
    current_interface: Option<String>,
    /// Auto-select interface with most traffic.
    auto_select: bool,
    /// Maximum observed throughput (for normalization).
    max_throughput: f64,
}

impl NetworkCollector {
    /// Creates a new network collector.
    #[must_use]
    pub fn new() -> Self {
        Self {
            prev_stats: HashMap::new(),
            prev_time: None,
            rates: HashMap::new(),
            rx_history: HashMap::new(),
            tx_history: HashMap::new(),
            current_interface: None,
            auto_select: true,
            max_throughput: 125_000_000.0, // 1 Gbps default max
        }
    }

    /// Sets the current interface for primary display.
    pub fn set_interface(&mut self, name: impl Into<String>) {
        self.current_interface = Some(name.into());
        self.auto_select = false;
    }

    /// Enables auto-selection of interface with most traffic.
    pub fn enable_auto_select(&mut self) {
        self.auto_select = true;
        self.current_interface = None;
    }

    /// Returns the list of available interface names.
    #[must_use]
    pub fn interfaces(&self) -> Vec<String> {
        self.rates.keys().cloned().collect()
    }

    /// Returns rates for all interfaces.
    #[must_use]
    pub fn all_rates(&self) -> &HashMap<String, NetRates> {
        &self.rates
    }

    /// Returns rates for the current interface.
    #[must_use]
    pub fn current_rates(&self) -> Option<&NetRates> {
        self.current_interface.as_ref().and_then(|name| self.rates.get(name))
    }

    /// Returns the current interface name.
    #[must_use]
    pub fn current_interface(&self) -> Option<&str> {
        self.current_interface.as_deref()
    }

    /// Returns download history for the current interface.
    #[must_use]
    pub fn rx_history(&self) -> Option<&RingBuffer<f64>> {
        self.current_interface.as_ref().and_then(|name| self.rx_history.get(name))
    }

    /// Returns upload history for the current interface.
    #[must_use]
    pub fn tx_history(&self) -> Option<&RingBuffer<f64>> {
        self.current_interface.as_ref().and_then(|name| self.tx_history.get(name))
    }

    /// Reads network statistics from /proc/net/dev.
    #[cfg(target_os = "linux")]
    fn read_net_dev(&self) -> Result<HashMap<String, NetStats>> {
        let content = std::fs::read_to_string("/proc/net/dev").map_err(|e| {
            MonitorError::CollectionFailed {
                collector: "network",
                message: format!("Failed to read /proc/net/dev: {e}"),
            }
        })?;

        let mut stats = HashMap::new();

        for line in content.lines().skip(2) {
            // Skip header lines
            let line = line.trim();
            let parts: Vec<&str> = line.split(':').collect();
            if parts.len() != 2 {
                continue;
            }

            let name = parts[0].trim().to_string();
            let values: Vec<u64> =
                parts[1].split_whitespace().filter_map(|s| s.parse().ok()).collect();

            if values.len() < 16 {
                continue;
            }

            // Skip loopback
            if name == "lo" {
                continue;
            }

            stats.insert(
                name.clone(),
                NetStats {
                    name,
                    rx_bytes: values[0],
                    rx_packets: values[1],
                    rx_errors: values[2],
                    rx_drops: values[3],
                    tx_bytes: values[8],
                    tx_packets: values[9],
                    tx_errors: values[10],
                    tx_drops: values[11],
                },
            );
        }

        Ok(stats)
    }

    #[cfg(target_os = "macos")]
    fn read_net_dev(&self) -> Result<HashMap<String, NetStats>> {
        // Use netstat -ib to get interface statistics on macOS
        // Wrap in timeout to prevent hangs on slow/unresponsive systems
        let result = run_with_timeout("netstat", &["-ib"], Duration::from_secs(5));

        let content = match result.stdout_string() {
            Some(s) => s,
            None => {
                // Timeout or error - return empty stats rather than hanging
                return Ok(HashMap::new());
            }
        };
        let mut stats = HashMap::new();

        // Parse netstat -ib output:
        // Name  Mtu   Network       Address            Ipkts Ierrs     Ibytes    Opkts Oerrs     Obytes  Coll
        // en0   1500  <Link#6>      xx:xx:xx:xx:xx:xx 123456     0  123456789   654321     0  987654321     0

        for line in content.lines().skip(1) {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 11 {
                continue;
            }

            let name = parts[0].to_string();

            // Skip loopback and non-physical interfaces
            if name.starts_with("lo") || name.starts_with("gif") || name.starts_with("stf") {
                continue;
            }

            // Only consider interfaces with link-level addresses (contain <Link#)
            let has_link = parts.iter().any(|p| p.starts_with("<Link#"));
            if !has_link {
                continue;
            }

            // Find the byte columns (they're the large numbers after error counts)
            // Format: Ipkts Ierrs Ibytes Opkts Oerrs Obytes Coll
            // Position varies based on whether Address field is present
            let (rx_bytes, rx_packets, rx_errors, tx_bytes, tx_packets, tx_errors) =
                if parts.len() >= 11 {
                    // Standard format with MAC address
                    let ipkts: u64 = parts[4].parse().unwrap_or(0);
                    let ierrs: u64 = parts[5].parse().unwrap_or(0);
                    let ibytes: u64 = parts[6].parse().unwrap_or(0);
                    let opkts: u64 = parts[7].parse().unwrap_or(0);
                    let oerrs: u64 = parts[8].parse().unwrap_or(0);
                    let obytes: u64 = parts[9].parse().unwrap_or(0);
                    (ibytes, ipkts, ierrs, obytes, opkts, oerrs)
                } else {
                    continue;
                };

            // Skip interfaces with no traffic
            if rx_bytes == 0 && tx_bytes == 0 {
                continue;
            }

            stats.insert(
                name.clone(),
                NetStats {
                    name,
                    rx_bytes,
                    rx_packets,
                    rx_errors,
                    rx_drops: 0, // Not available in netstat -ib
                    tx_bytes,
                    tx_packets,
                    tx_errors,
                    tx_drops: 0, // Not available in netstat -ib
                },
            );
        }

        Ok(stats)
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    fn read_net_dev(&self) -> Result<HashMap<String, NetStats>> {
        Ok(HashMap::new())
    }

    /// Calculates rates from delta between samples.
    fn calculate_rates(
        &self,
        current: &HashMap<String, NetStats>,
        elapsed_secs: f64,
    ) -> HashMap<String, NetRates> {
        let mut rates = HashMap::new();

        for (name, curr) in current {
            if let Some(prev) = self.prev_stats.get(name) {
                let rx_delta = curr.rx_bytes.saturating_sub(prev.rx_bytes);
                let tx_delta = curr.tx_bytes.saturating_sub(prev.tx_bytes);
                let rx_pkt_delta = curr.rx_packets.saturating_sub(prev.rx_packets);
                let tx_pkt_delta = curr.tx_packets.saturating_sub(prev.tx_packets);

                rates.insert(
                    name.clone(),
                    NetRates {
                        name: name.clone(),
                        rx_bytes_per_sec: rx_delta as f64 / elapsed_secs,
                        tx_bytes_per_sec: tx_delta as f64 / elapsed_secs,
                        rx_packets_per_sec: rx_pkt_delta as f64 / elapsed_secs,
                        tx_packets_per_sec: tx_pkt_delta as f64 / elapsed_secs,
                    },
                );
            }
        }

        rates
    }

    /// Auto-selects the interface with most traffic.
    fn auto_select_interface(&mut self) {
        if !self.auto_select {
            return;
        }

        let best = self
            .rates
            .iter()
            .max_by(|a, b| {
                let a_total = a.1.rx_bytes_per_sec + a.1.tx_bytes_per_sec;
                let b_total = b.1.rx_bytes_per_sec + b.1.tx_bytes_per_sec;
                a_total.partial_cmp(&b_total).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(name, _)| name.clone());

        if best.is_some() {
            self.current_interface = best;
        }
    }

    /// Updates history buffers.
    fn update_history(&mut self) {
        for (name, rate) in &self.rates {
            // Normalize to 0-1 range
            let rx_norm = (rate.rx_bytes_per_sec / self.max_throughput).min(1.0);
            let tx_norm = (rate.tx_bytes_per_sec / self.max_throughput).min(1.0);

            self.rx_history
                .entry(name.clone())
                .or_insert_with(|| RingBuffer::new(300))
                .push(rx_norm);

            self.tx_history
                .entry(name.clone())
                .or_insert_with(|| RingBuffer::new(300))
                .push(tx_norm);
        }
    }
}

impl Default for NetworkCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl Collector for NetworkCollector {
    fn id(&self) -> &'static str {
        "network"
    }

    fn collect(&mut self) -> Result<Metrics> {
        let now = Instant::now();
        let current_stats = self.read_net_dev()?;

        // Calculate rates if we have previous data
        if let Some(prev_time) = self.prev_time {
            let elapsed = now.duration_since(prev_time);
            let elapsed_secs = elapsed.as_secs_f64();

            if elapsed_secs > 0.0 {
                self.rates = self.calculate_rates(&current_stats, elapsed_secs);
                self.auto_select_interface();
                self.update_history();
            }
        }

        // Update previous state
        self.prev_stats = current_stats;
        self.prev_time = Some(now);

        // Build metrics
        let mut metrics = Metrics::new();

        // Total rates across all interfaces
        let total_rx: f64 = self.rates.values().map(|r| r.rx_bytes_per_sec).sum();
        let total_tx: f64 = self.rates.values().map(|r| r.tx_bytes_per_sec).sum();

        metrics.insert("network.rx_bytes_per_sec", MetricValue::Gauge(total_rx));
        metrics.insert("network.tx_bytes_per_sec", MetricValue::Gauge(total_tx));

        // Interface count
        metrics.insert("network.interface_count", MetricValue::Counter(self.rates.len() as u64));

        // Current interface rates
        if let Some(rates) = self.current_rates() {
            metrics.insert(
                "network.current.rx_bytes_per_sec",
                MetricValue::Gauge(rates.rx_bytes_per_sec),
            );
            metrics.insert(
                "network.current.tx_bytes_per_sec",
                MetricValue::Gauge(rates.tx_bytes_per_sec),
            );
        }

        Ok(metrics)
    }

    fn is_available(&self) -> bool {
        #[cfg(target_os = "linux")]
        {
            std::path::Path::new("/proc/net/dev").exists()
        }
        #[cfg(target_os = "macos")]
        {
            true // macOS uses netstat which is always available
        }
        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        {
            false
        }
    }

    fn interval_hint(&self) -> Duration {
        Duration::from_millis(1000)
    }

    fn display_name(&self) -> &'static str {
        "Network"
    }
}

// ============================================================================
// Tests (TDD - Written First)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Unit Tests
    // ========================================================================

    #[test]
    fn test_network_collector_new() {
        let collector = NetworkCollector::new();
        assert!(collector.prev_stats.is_empty());
        assert!(collector.prev_time.is_none());
        assert!(collector.auto_select);
    }

    #[test]
    fn test_network_collector_default() {
        let collector = NetworkCollector::default();
        assert!(collector.auto_select);
    }

    #[test]
    fn test_net_stats_default() {
        let stats = NetStats::default();
        assert!(stats.name.is_empty());
        assert_eq!(stats.rx_bytes, 0);
        assert_eq!(stats.tx_bytes, 0);
    }

    #[test]
    fn test_net_rates_default() {
        let rates = NetRates::default();
        assert!(rates.name.is_empty());
        assert_eq!(rates.rx_bytes_per_sec, 0.0);
        assert_eq!(rates.tx_bytes_per_sec, 0.0);
    }

    #[test]
    fn test_net_rates_bits_conversion() {
        let rates = NetRates {
            name: "eth0".to_string(),
            rx_bytes_per_sec: 1000.0,
            tx_bytes_per_sec: 500.0,
            rx_packets_per_sec: 10.0,
            tx_packets_per_sec: 5.0,
        };

        assert_eq!(rates.rx_bits_per_sec(), 8000.0);
        assert_eq!(rates.tx_bits_per_sec(), 4000.0);
    }

    #[test]
    fn test_format_bytes_rate() {
        assert_eq!(format_bytes_rate(500.0), "500B/s");
        assert_eq!(format_bytes_rate(1500.0), "1.50K/s");
        assert_eq!(format_bytes_rate(1_500_000.0), "1.50M/s");
        assert_eq!(format_bytes_rate(1_500_000_000.0), "1.50G/s");
    }

    #[test]
    fn test_net_rates_formatted() {
        let rates = NetRates {
            name: "eth0".to_string(),
            rx_bytes_per_sec: 1_500_000.0,
            tx_bytes_per_sec: 500_000.0,
            ..Default::default()
        };

        assert_eq!(rates.rx_formatted(), "1.50M/s");
        assert_eq!(rates.tx_formatted(), "500K/s");
    }

    #[test]
    fn test_set_interface() {
        let mut collector = NetworkCollector::new();
        assert!(collector.auto_select);

        collector.set_interface("eth0");
        assert!(!collector.auto_select);
        assert_eq!(collector.current_interface(), Some("eth0"));
    }

    #[test]
    fn test_enable_auto_select() {
        let mut collector = NetworkCollector::new();
        collector.set_interface("eth0");
        assert!(!collector.auto_select);

        collector.enable_auto_select();
        assert!(collector.auto_select);
        assert!(collector.current_interface.is_none());
    }

    #[test]
    fn test_network_collector_interval() {
        let collector = NetworkCollector::new();
        assert_eq!(collector.interval_hint(), Duration::from_millis(1000));
    }

    #[test]
    fn test_network_collector_display_name() {
        let collector = NetworkCollector::new();
        assert_eq!(collector.display_name(), "Network");
    }

    #[test]
    fn test_network_collector_id() {
        let collector = NetworkCollector::new();
        assert_eq!(collector.id(), "network");
    }

    #[test]
    fn test_calculate_rates() {
        let mut collector = NetworkCollector::new();

        // Set up previous stats
        collector.prev_stats.insert(
            "eth0".to_string(),
            NetStats {
                name: "eth0".to_string(),
                rx_bytes: 1000,
                tx_bytes: 500,
                rx_packets: 100,
                tx_packets: 50,
                ..Default::default()
            },
        );

        // Current stats (after 1 second)
        let mut current = HashMap::new();
        current.insert(
            "eth0".to_string(),
            NetStats {
                name: "eth0".to_string(),
                rx_bytes: 2000,  // +1000 bytes
                tx_bytes: 1000,  // +500 bytes
                rx_packets: 200, // +100 packets
                tx_packets: 100, // +50 packets
                ..Default::default()
            },
        );

        let rates = collector.calculate_rates(&current, 1.0);

        let eth0_rates = rates.get("eth0").expect("Should have eth0 rates");
        assert!((eth0_rates.rx_bytes_per_sec - 1000.0).abs() < 0.01);
        assert!((eth0_rates.tx_bytes_per_sec - 500.0).abs() < 0.01);
        assert!((eth0_rates.rx_packets_per_sec - 100.0).abs() < 0.01);
        assert!((eth0_rates.tx_packets_per_sec - 50.0).abs() < 0.01);
    }

    // ========================================================================
    // Linux-specific Tests
    // ========================================================================

    #[cfg(target_os = "linux")]
    #[test]
    fn test_network_collector_is_available() {
        let collector = NetworkCollector::new();
        assert!(collector.is_available());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_network_collector_collect() {
        let mut collector = NetworkCollector::new();

        // First collection (no rates yet)
        let result = collector.collect();
        assert!(result.is_ok());

        // Wait a bit and collect again
        std::thread::sleep(Duration::from_millis(100));

        let result = collector.collect();
        assert!(result.is_ok());

        let metrics = result.expect("collect should succeed");
        assert!(metrics.get_gauge("network.rx_bytes_per_sec").is_some());
        assert!(metrics.get_gauge("network.tx_bytes_per_sec").is_some());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_network_collector_interfaces() {
        let mut collector = NetworkCollector::new();
        let _ = collector.collect();
        std::thread::sleep(Duration::from_millis(50));
        let _ = collector.collect();

        // Should find at least one interface (excluding lo)
        let interfaces = collector.interfaces();
        // May be empty in containerized environments
        // Just verify it doesn't panic
        let _ = interfaces;
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_network_packet_monotonic() {
        // Falsification criterion #44: packet counts are monotonically increasing
        let mut collector = NetworkCollector::new();

        let _ = collector.collect();
        let stats1 = collector.prev_stats.clone();

        std::thread::sleep(Duration::from_millis(50));

        let _ = collector.collect();
        let stats2 = collector.prev_stats.clone();

        // Verify packet counts never decreased
        for (name, s2) in &stats2 {
            if let Some(s1) = stats1.get(name) {
                assert!(
                    s2.rx_packets >= s1.rx_packets,
                    "RX packets should not decrease for {name}"
                );
                assert!(
                    s2.tx_packets >= s1.tx_packets,
                    "TX packets should not decrease for {name}"
                );
            }
        }
    }
}
