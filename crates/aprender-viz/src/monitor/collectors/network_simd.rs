//! SIMD-accelerated Network metrics collector.
//!
//! This module provides a high-performance network collector using SIMD operations
//! for parsing `/proc/net/dev` and computing throughput metrics.
//!
//! ## Performance Targets (Falsifiable)
//!
//! - Standard interfaces (≤8): < 40μs
//! - Many interfaces (≤32): < 80μs
//!
//! ## Design
//!
//! Uses Structure-of-Arrays (SoA) layout for SIMD-friendly access to
//! per-interface statistics. Delta calculations and rate computations
//! use vectorized operations.

use crate::monitor::error::{MonitorError, Result};
use crate::monitor::simd::ring_buffer::SimdRingBuffer;
use crate::monitor::simd::soa::NetworkMetricsSoA;
use crate::monitor::simd::{kernels, SimdStats};
use crate::monitor::types::{Collector, MetricValue, Metrics};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// SIMD-accelerated network collector.
///
/// Uses SoA layout and SIMD operations for high-performance metric collection.
#[derive(Debug)]
pub struct SimdNetworkCollector {
    /// Current metrics in SoA layout.
    current: NetworkMetricsSoA,
    /// Previous metrics for delta calculation.
    previous: NetworkMetricsSoA,
    /// Previous collection time.
    prev_time: Option<Instant>,
    /// Calculated rates per interface (bytes per second).
    rx_rates: Vec<f64>,
    /// Calculated TX rates per interface.
    tx_rates: Vec<f64>,
    /// RX history per interface (normalized 0-1).
    rx_history: HashMap<String, SimdRingBuffer>,
    /// TX history per interface (normalized 0-1).
    tx_history: HashMap<String, SimdRingBuffer>,
    /// Current interface for primary display.
    current_interface: Option<String>,
    /// Auto-select interface with most traffic.
    auto_select: bool,
    /// Maximum observed throughput (for normalization).
    max_throughput: f64,
    /// Pre-allocated read buffer (8KB).
    #[cfg(target_os = "linux")]
    read_buffer: Vec<u8>,
    /// Whether we have previous data for delta calculation.
    has_previous: bool,
}

impl SimdNetworkCollector {
    /// Creates a new SIMD network collector.
    #[must_use]
    pub fn new() -> Self {
        Self {
            current: NetworkMetricsSoA::new(32),
            previous: NetworkMetricsSoA::new(32),
            prev_time: None,
            rx_rates: Vec::with_capacity(32),
            tx_rates: Vec::with_capacity(32),
            rx_history: HashMap::new(),
            tx_history: HashMap::new(),
            current_interface: None,
            auto_select: true,
            max_throughput: 125_000_000.0, // 1 Gbps default max
            #[cfg(target_os = "linux")]
            read_buffer: vec![0u8; 8192],
            has_previous: false,
        }
    }

    /// Parses /proc/net/dev using SIMD-optimized reading.
    #[cfg(target_os = "linux")]
    fn parse_net_dev(&mut self) -> Result<()> {
        use std::fs::File;
        use std::io::Read;

        let mut file = File::open("/proc/net/dev").map_err(|e| MonitorError::CollectionFailed {
            collector: "network_simd",
            message: format!("Failed to open /proc/net/dev: {e}"),
        })?;

        let bytes_read =
            file.read(&mut self.read_buffer).map_err(|e| MonitorError::CollectionFailed {
                collector: "network_simd",
                message: format!("Failed to read /proc/net/dev: {e}"),
            })?;

        // Reset current metrics for fresh parse
        self.current = NetworkMetricsSoA::new(32);

        // Copy to local buffer to avoid borrow conflict (self.read_buffer vs &mut self)
        #[allow(clippy::unnecessary_to_owned)]
        let buffer = self.read_buffer[..bytes_read].to_vec();
        self.parse_net_dev_buffer(&buffer)
    }

    /// Parses net/dev buffer using SIMD-assisted line finding.
    #[cfg(target_os = "linux")]
    fn parse_net_dev_buffer(&mut self, buffer: &[u8]) -> Result<()> {
        // Find line boundaries using SIMD
        let newlines = kernels::simd_find_newlines(buffer);

        let mut line_start = 0;
        let mut line_num = 0;

        for &newline_pos in &newlines {
            line_num += 1;

            // Skip header lines (first 2 lines)
            if line_num <= 2 {
                line_start = newline_pos + 1;
                continue;
            }

            let line = &buffer[line_start..newline_pos];

            // Find the colon separator
            if let Some(colon_pos) = line.iter().position(|&b| b == b':') {
                let name_bytes = &line[..colon_pos];
                let name = std::str::from_utf8(name_bytes).unwrap_or("").trim().to_string();

                // Skip loopback
                if name == "lo" {
                    line_start = newline_pos + 1;
                    continue;
                }

                // Parse the values after the colon
                let values = kernels::simd_parse_integers(&line[colon_pos + 1..]);

                if values.len() >= 16 {
                    self.current.set_interface(
                        &name, values[0],  // rx_bytes
                        values[1],  // rx_packets
                        values[2],  // rx_errors
                        values[3],  // rx_drops
                        values[8],  // tx_bytes
                        values[9],  // tx_packets
                        values[10], // tx_errors
                        values[11], // tx_drops
                    );
                }
            }

            line_start = newline_pos + 1;
        }

        Ok(())
    }

    #[cfg(not(target_os = "linux"))]
    fn parse_net_dev(&mut self) -> Result<()> {
        // Non-Linux platforms: generate dummy data
        self.current = NetworkMetricsSoA::new(32);
        self.current.set_interface("eth0", 1000, 100, 0, 0, 500, 50, 0, 0);
        Ok(())
    }

    /// Computes rates from current and previous metrics.
    fn compute_rates(&mut self, elapsed_secs: f64) {
        if !self.has_previous || elapsed_secs <= 0.0 {
            return;
        }

        let count = self.current.interface_count;
        let prev_count = self.previous.interface_count;

        // Only calculate for interfaces that exist in both samples
        let min_count = count.min(prev_count);

        // Calculate RX deltas using SIMD
        let rx_delta = kernels::simd_delta(
            &self.current.rx_bytes[..min_count],
            &self.previous.rx_bytes[..min_count],
        );

        // Calculate TX deltas using SIMD
        let tx_delta = kernels::simd_delta(
            &self.current.tx_bytes[..min_count],
            &self.previous.tx_bytes[..min_count],
        );

        // Compute rates
        self.rx_rates.clear();
        self.tx_rates.clear();

        for (&rx, &tx) in rx_delta.iter().zip(tx_delta.iter()) {
            self.rx_rates.push(rx as f64 / elapsed_secs);
            self.tx_rates.push(tx as f64 / elapsed_secs);
        }
    }

    /// Auto-selects the interface with most traffic.
    fn auto_select_interface(&mut self) {
        if !self.auto_select || self.current.interface_count == 0 {
            return;
        }

        let mut best_idx = 0;
        let mut best_traffic = 0.0;

        for i in 0..self.rx_rates.len() {
            let total = self.rx_rates.get(i).unwrap_or(&0.0) + self.tx_rates.get(i).unwrap_or(&0.0);
            if total > best_traffic {
                best_traffic = total;
                best_idx = i;
            }
        }

        if best_idx < self.current.names.len() {
            self.current_interface = Some(self.current.names[best_idx].clone());
        }
    }

    /// Updates history buffers.
    fn update_history(&mut self) {
        for (i, name) in self.current.names.iter().enumerate() {
            let rx_rate = self.rx_rates.get(i).copied().unwrap_or(0.0);
            let tx_rate = self.tx_rates.get(i).copied().unwrap_or(0.0);

            // Normalize to 0-1 range
            let rx_norm = (rx_rate / self.max_throughput).min(1.0);
            let tx_norm = (tx_rate / self.max_throughput).min(1.0);

            self.rx_history
                .entry(name.clone())
                .or_insert_with(|| SimdRingBuffer::new(300))
                .push(rx_norm);

            self.tx_history
                .entry(name.clone())
                .or_insert_with(|| SimdRingBuffer::new(300))
                .push(tx_norm);
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
    pub fn interfaces(&self) -> &[String] {
        &self.current.names
    }

    /// Returns the current interface name.
    #[must_use]
    pub fn current_interface(&self) -> Option<&str> {
        self.current_interface.as_deref()
    }

    /// Returns RX history for the current interface.
    #[must_use]
    pub fn rx_history(&self) -> Option<&SimdRingBuffer> {
        self.current_interface.as_ref().and_then(|name| self.rx_history.get(name))
    }

    /// Returns TX history for the current interface.
    #[must_use]
    pub fn tx_history(&self) -> Option<&SimdRingBuffer> {
        self.current_interface.as_ref().and_then(|name| self.tx_history.get(name))
    }

    /// Returns RX rate statistics for the current interface.
    #[must_use]
    pub fn rx_stats(&self) -> Option<&SimdStats> {
        self.rx_history().map(super::super::simd::ring_buffer::SimdRingBuffer::statistics)
    }

    /// Returns TX rate statistics for the current interface.
    #[must_use]
    pub fn tx_stats(&self) -> Option<&SimdStats> {
        self.tx_history().map(super::super::simd::ring_buffer::SimdRingBuffer::statistics)
    }

    /// Returns total RX bytes using SIMD sum.
    #[must_use]
    pub fn total_rx_bytes(&self) -> u64 {
        self.current.total_rx_bytes()
    }

    /// Returns total TX bytes using SIMD sum.
    #[must_use]
    pub fn total_tx_bytes(&self) -> u64 {
        self.current.total_tx_bytes()
    }

    /// Returns RX rate for a specific interface.
    #[must_use]
    pub fn rx_rate(&self, idx: usize) -> f64 {
        self.rx_rates.get(idx).copied().unwrap_or(0.0)
    }

    /// Returns TX rate for a specific interface.
    #[must_use]
    pub fn tx_rate(&self, idx: usize) -> f64 {
        self.tx_rates.get(idx).copied().unwrap_or(0.0)
    }
}

impl Default for SimdNetworkCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl Collector for SimdNetworkCollector {
    fn id(&self) -> &'static str {
        "network_simd"
    }

    fn collect(&mut self) -> Result<Metrics> {
        let now = Instant::now();

        // Parse /proc/net/dev
        self.parse_net_dev()?;

        // Calculate rates if we have previous data
        if let Some(prev_time) = self.prev_time {
            let elapsed = now.duration_since(prev_time);
            let elapsed_secs = elapsed.as_secs_f64();

            if elapsed_secs > 0.0 {
                self.compute_rates(elapsed_secs);
                self.auto_select_interface();
                self.update_history();
            }
        }

        // Swap buffers for next delta calculation
        std::mem::swap(&mut self.current, &mut self.previous);
        self.current = NetworkMetricsSoA::new(32);
        self.prev_time = Some(now);
        self.has_previous = true;

        // Build metrics
        let mut metrics = Metrics::new();

        // Total rates across all interfaces
        let total_rx: f64 = self.rx_rates.iter().sum();
        let total_tx: f64 = self.tx_rates.iter().sum();

        metrics.insert("network.rx_bytes_per_sec", MetricValue::Gauge(total_rx));
        metrics.insert("network.tx_bytes_per_sec", MetricValue::Gauge(total_tx));

        // Interface count
        metrics.insert(
            "network.interface_count",
            MetricValue::Counter(self.previous.interface_count as u64),
        );

        // Current interface rates
        if let Some(ref iface) = self.current_interface {
            if let Some(idx) = self.previous.names.iter().position(|n| n == iface) {
                metrics.insert(
                    "network.current.rx_bytes_per_sec",
                    MetricValue::Gauge(self.rx_rate(idx)),
                );
                metrics.insert(
                    "network.current.tx_bytes_per_sec",
                    MetricValue::Gauge(self.tx_rate(idx)),
                );
            }
        }

        Ok(metrics)
    }

    fn is_available(&self) -> bool {
        #[cfg(target_os = "linux")]
        {
            std::path::Path::new("/proc/net/dev").exists()
        }
        #[cfg(not(target_os = "linux"))]
        {
            false
        }
    }

    fn interval_hint(&self) -> Duration {
        Duration::from_millis(1000)
    }

    fn display_name(&self) -> &'static str {
        "Network (SIMD)"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_network_collector_new() {
        let collector = SimdNetworkCollector::new();
        assert!(collector.current_interface.is_none());
        assert!(collector.auto_select);
    }

    #[test]
    fn test_simd_network_collector_id() {
        let collector = SimdNetworkCollector::new();
        assert_eq!(collector.id(), "network_simd");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_simd_network_collector_available() {
        let collector = SimdNetworkCollector::new();
        assert!(collector.is_available());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_simd_network_collector_collect() {
        let mut collector = SimdNetworkCollector::new();

        // First collection establishes baseline
        let result1 = collector.collect();
        assert!(result1.is_ok());

        // Wait and collect again for rate calculation
        std::thread::sleep(Duration::from_millis(100));
        let result2 = collector.collect();
        assert!(result2.is_ok());

        let metrics = result2.expect("collect should succeed");
        assert!(metrics.get_gauge("network.rx_bytes_per_sec").is_some());
        assert!(metrics.get_gauge("network.tx_bytes_per_sec").is_some());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_simd_network_parse_buffer() {
        let mut collector = SimdNetworkCollector::new();
        let test_data = b"Inter-|   Receive                                                |  Transmit\n face |bytes    packets errs drop fifo frame compressed multicast|bytes    packets errs drop fifo colls carrier compressed\n  eth0: 1000      100    0    0    0     0          0         0     500       50    0    0    0     0       0          0\n    lo:    0        0    0    0    0     0          0         0       0        0    0    0    0     0       0          0\n";

        let result = collector.parse_net_dev_buffer(test_data);
        assert!(result.is_ok());

        assert_eq!(collector.current.interface_count, 1);
        assert_eq!(collector.current.names[0], "eth0");
        assert_eq!(collector.current.rx_bytes[0], 1000);
        assert_eq!(collector.current.tx_bytes[0], 500);
    }

    #[test]
    fn test_simd_network_set_interface() {
        let mut collector = SimdNetworkCollector::new();
        assert!(collector.auto_select);

        collector.set_interface("eth0");
        assert!(!collector.auto_select);
        assert_eq!(collector.current_interface(), Some("eth0"));
    }

    #[test]
    fn test_simd_network_enable_auto_select() {
        let mut collector = SimdNetworkCollector::new();
        collector.set_interface("eth0");
        assert!(!collector.auto_select);

        collector.enable_auto_select();
        assert!(collector.auto_select);
        assert!(collector.current_interface.is_none());
    }

    #[test]
    fn test_simd_network_display_name() {
        let collector = SimdNetworkCollector::new();
        assert_eq!(collector.display_name(), "Network (SIMD)");
    }

    #[test]
    fn test_simd_network_interval() {
        let collector = SimdNetworkCollector::new();
        assert_eq!(collector.interval_hint(), Duration::from_millis(1000));
    }

    #[test]
    fn test_simd_network_rates() {
        let mut collector = SimdNetworkCollector::new();

        // Set up test data
        collector.rx_rates = vec![1000.0, 2000.0];
        collector.tx_rates = vec![500.0, 1000.0];

        assert!((collector.rx_rate(0) - 1000.0).abs() < 0.01);
        assert!((collector.tx_rate(0) - 500.0).abs() < 0.01);
        assert!((collector.rx_rate(1) - 2000.0).abs() < 0.01);
        assert!((collector.tx_rate(2) - 0.0).abs() < 0.01); // Out of bounds
    }

    #[test]
    fn test_network_metrics_soa() {
        let mut net = NetworkMetricsSoA::new(4);
        net.set_interface("eth0", 1000, 100, 0, 0, 500, 50, 0, 0);
        net.set_interface("eth1", 2000, 200, 0, 0, 1000, 100, 0, 0);

        assert_eq!(net.interface_count, 2);
        assert_eq!(net.total_rx_bytes(), 3000);
        assert_eq!(net.total_tx_bytes(), 1500);
    }
}
