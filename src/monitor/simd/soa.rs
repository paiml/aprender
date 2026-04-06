//! Structure-of-Arrays (SoA) metric layouts for SIMD processing.
//!
//! ## Design Philosophy
//!
//! Traditional Array-of-Structures (AoS) layout:
//! ```text
//! [Core0{user,nice,sys,idle}, Core1{user,nice,sys,idle}, ...]
//! ```
//!
//! SIMD-friendly Structure-of-Arrays (SoA) layout:
//! ```text
//! user:   [Core0.user, Core1.user, Core2.user, ...]
//! nice:   [Core0.nice, Core1.nice, Core2.nice, ...]
//! system: [Core0.sys,  Core1.sys,  Core2.sys,  ...]
//! idle:   [Core0.idle, Core1.idle, Core2.idle, ...]
//! ```
//!
//! ## Falsifiable Hypothesis (H₃)
//!
//! SoA layout reduces L2 cache misses by ≥50% for multi-core CPU metric iteration.
//!
//! ## References
//!
//! - Drepper (2007): "What Every Programmer Should Know About Memory"
//! - Fog (2024): "Optimizing Software in C++"

use super::{MAX_CORES, MAX_DISKS, MAX_INTERFACES};

/// CPU metrics in SoA layout for SIMD processing.
#[repr(C, align(64))]
#[derive(Debug, Clone)]
pub struct CpuMetricsSoA {
    /// User time per core.
    pub user: Vec<u64>,
    /// Nice time per core.
    pub nice: Vec<u64>,
    /// System time per core.
    pub system: Vec<u64>,
    /// Idle time per core.
    pub idle: Vec<u64>,
    /// I/O wait time per core.
    pub iowait: Vec<u64>,
    /// IRQ time per core.
    pub irq: Vec<u64>,
    /// Soft IRQ time per core.
    pub softirq: Vec<u64>,
    /// Steal time per core.
    pub steal: Vec<u64>,
    /// Total time per core (computed).
    pub total: Vec<u64>,
    /// Usage percentage per core (computed).
    pub usage_pct: Vec<f64>,
    /// Number of cores.
    pub core_count: usize,
}

impl CpuMetricsSoA {
    /// Creates a new SoA structure for the given core count.
    #[must_use]
    pub fn new(core_count: usize) -> Self {
        let count = core_count.min(MAX_CORES);
        // Round up to multiple of 8 for SIMD alignment
        let aligned_count = count.div_ceil(8) * 8;

        Self {
            user: vec![0; aligned_count],
            nice: vec![0; aligned_count],
            system: vec![0; aligned_count],
            idle: vec![0; aligned_count],
            iowait: vec![0; aligned_count],
            irq: vec![0; aligned_count],
            softirq: vec![0; aligned_count],
            steal: vec![0; aligned_count],
            total: vec![0; aligned_count],
            usage_pct: vec![0.0; aligned_count],
            core_count: count,
        }
    }

    /// Sets metrics for a specific core.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn set_core(
        &mut self,
        core: usize,
        user: u64,
        nice: u64,
        system: u64,
        idle: u64,
        iowait: u64,
        irq: u64,
        softirq: u64,
        steal: u64,
    ) {
        if core >= self.user.len() {
            return;
        }

        self.user[core] = user;
        self.nice[core] = nice;
        self.system[core] = system;
        self.idle[core] = idle;
        self.iowait[core] = iowait;
        self.irq[core] = irq;
        self.softirq[core] = softirq;
        self.steal[core] = steal;

        // Compute total
        self.total[core] = user + nice + system + idle + iowait + irq + softirq + steal;
    }

    /// Computes usage percentages from deltas (prev -> current).
    pub fn compute_usage_from_delta(&mut self, prev: &Self) {
        use super::kernels::{simd_delta, simd_percentage};

        // Calculate deltas for total and idle
        let total_delta =
            simd_delta(&self.total[..self.core_count], &prev.total[..prev.core_count]);
        let idle_delta = simd_delta(&self.idle[..self.core_count], &prev.idle[..prev.core_count]);

        // Calculate used = total - idle
        let used_delta: Vec<u64> =
            total_delta.iter().zip(idle_delta.iter()).map(|(&t, &i)| t.saturating_sub(i)).collect();

        // Calculate percentage
        let percentages = simd_percentage(&used_delta, &total_delta);

        for (i, pct) in percentages.into_iter().enumerate() {
            if i < self.usage_pct.len() {
                self.usage_pct[i] = pct;
            }
        }
    }

    /// Returns usage percentage for a specific core.
    #[must_use]
    pub fn usage(&self, core: usize) -> f64 {
        self.usage_pct.get(core).copied().unwrap_or(0.0)
    }

    /// Returns average usage across all cores.
    #[must_use]
    pub fn avg_usage(&self) -> f64 {
        if self.core_count == 0 {
            return 0.0;
        }
        let sum: f64 = self.usage_pct[..self.core_count].iter().sum();
        sum / self.core_count as f64
    }

    /// Clears all metrics.
    pub fn clear(&mut self) {
        for v in &mut self.user {
            *v = 0;
        }
        for v in &mut self.nice {
            *v = 0;
        }
        for v in &mut self.system {
            *v = 0;
        }
        for v in &mut self.idle {
            *v = 0;
        }
        for v in &mut self.iowait {
            *v = 0;
        }
        for v in &mut self.irq {
            *v = 0;
        }
        for v in &mut self.softirq {
            *v = 0;
        }
        for v in &mut self.steal {
            *v = 0;
        }
        for v in &mut self.total {
            *v = 0;
        }
        for v in &mut self.usage_pct {
            *v = 0.0;
        }
    }
}

impl Default for CpuMetricsSoA {
    fn default() -> Self {
        Self::new(8) // Default to 8 cores
    }
}

/// Memory metrics in SoA layout.
#[repr(C, align(64))]
#[derive(Debug, Clone, Default)]
pub struct MemoryMetricsSoA {
    /// Total memory in bytes.
    pub total: u64,
    /// Free memory in bytes.
    pub free: u64,
    /// Available memory in bytes.
    pub available: u64,
    /// Buffers in bytes.
    pub buffers: u64,
    /// Cached memory in bytes.
    pub cached: u64,
    /// Swap total in bytes.
    pub swap_total: u64,
    /// Swap free in bytes.
    pub swap_free: u64,
    /// Dirty pages in bytes.
    pub dirty: u64,
    /// Slab memory in bytes.
    pub slab: u64,
    /// Padding to 64 bytes.
    _padding: [u8; 8],
}

impl MemoryMetricsSoA {
    /// Creates new memory metrics.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            total: 0,
            free: 0,
            available: 0,
            buffers: 0,
            cached: 0,
            swap_total: 0,
            swap_free: 0,
            dirty: 0,
            slab: 0,
            _padding: [0; 8],
        }
    }

    /// Returns used memory in bytes.
    #[must_use]
    pub fn used(&self) -> u64 {
        self.total.saturating_sub(self.available)
    }

    /// Returns memory usage percentage.
    #[must_use]
    pub fn usage_pct(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        (self.used() as f64 / self.total as f64) * 100.0
    }

    /// Returns swap usage percentage.
    #[must_use]
    pub fn swap_usage_pct(&self) -> f64 {
        if self.swap_total == 0 {
            return 0.0;
        }
        let swap_used = self.swap_total.saturating_sub(self.swap_free);
        (swap_used as f64 / self.swap_total as f64) * 100.0
    }
}

/// Network interface metrics in SoA layout.
#[repr(C, align(64))]
#[derive(Debug, Clone)]
pub struct NetworkMetricsSoA {
    /// Interface names (not SIMD, but needed for identification).
    pub names: Vec<String>,
    /// Received bytes per interface.
    pub rx_bytes: Vec<u64>,
    /// Received packets per interface.
    pub rx_packets: Vec<u64>,
    /// Receive errors per interface.
    pub rx_errors: Vec<u64>,
    /// Receive drops per interface.
    pub rx_drops: Vec<u64>,
    /// Transmitted bytes per interface.
    pub tx_bytes: Vec<u64>,
    /// Transmitted packets per interface.
    pub tx_packets: Vec<u64>,
    /// Transmit errors per interface.
    pub tx_errors: Vec<u64>,
    /// Transmit drops per interface.
    pub tx_drops: Vec<u64>,
    /// Number of interfaces.
    pub interface_count: usize,
}

impl NetworkMetricsSoA {
    /// Creates a new SoA structure for the given interface count.
    #[must_use]
    pub fn new(interface_count: usize) -> Self {
        let count = interface_count.min(MAX_INTERFACES);
        let aligned_count = count.div_ceil(8) * 8;

        Self {
            names: Vec::with_capacity(count),
            rx_bytes: vec![0; aligned_count],
            rx_packets: vec![0; aligned_count],
            rx_errors: vec![0; aligned_count],
            rx_drops: vec![0; aligned_count],
            tx_bytes: vec![0; aligned_count],
            tx_packets: vec![0; aligned_count],
            tx_errors: vec![0; aligned_count],
            tx_drops: vec![0; aligned_count],
            interface_count: 0,
        }
    }

    /// Adds or updates an interface.
    #[allow(clippy::too_many_arguments)]
    pub fn set_interface(
        &mut self,
        name: &str,
        rx_bytes: u64,
        rx_packets: u64,
        rx_errors: u64,
        rx_drops: u64,
        tx_bytes: u64,
        tx_packets: u64,
        tx_errors: u64,
        tx_drops: u64,
    ) {
        // Find or add interface
        let idx = if let Some(pos) = self.names.iter().position(|n| n == name) {
            pos
        } else {
            if self.interface_count >= self.rx_bytes.len() {
                return; // At capacity
            }
            self.names.push(name.to_string());
            let idx = self.interface_count;
            self.interface_count += 1;
            idx
        };

        self.rx_bytes[idx] = rx_bytes;
        self.rx_packets[idx] = rx_packets;
        self.rx_errors[idx] = rx_errors;
        self.rx_drops[idx] = rx_drops;
        self.tx_bytes[idx] = tx_bytes;
        self.tx_packets[idx] = tx_packets;
        self.tx_errors[idx] = tx_errors;
        self.tx_drops[idx] = tx_drops;
    }

    /// Returns total received bytes across all interfaces.
    #[must_use]
    pub fn total_rx_bytes(&self) -> u64 {
        use super::kernels::simd_sum;
        let values: Vec<f64> =
            self.rx_bytes[..self.interface_count].iter().map(|&v| v as f64).collect();
        simd_sum(&values) as u64
    }

    /// Returns total transmitted bytes across all interfaces.
    #[must_use]
    pub fn total_tx_bytes(&self) -> u64 {
        use super::kernels::simd_sum;
        let values: Vec<f64> =
            self.tx_bytes[..self.interface_count].iter().map(|&v| v as f64).collect();
        simd_sum(&values) as u64
    }
}

impl Default for NetworkMetricsSoA {
    fn default() -> Self {
        Self::new(16)
    }
}

/// Disk metrics in SoA layout.
#[repr(C, align(64))]
#[derive(Debug, Clone)]
pub struct DiskMetricsSoA {
    /// Disk names.
    pub names: Vec<String>,
    /// Read operations completed.
    pub reads_completed: Vec<u64>,
    /// Read sectors (512 bytes each).
    pub sectors_read: Vec<u64>,
    /// Read time in milliseconds.
    pub read_time_ms: Vec<u64>,
    /// Write operations completed.
    pub writes_completed: Vec<u64>,
    /// Written sectors.
    pub sectors_written: Vec<u64>,
    /// Write time in milliseconds.
    pub write_time_ms: Vec<u64>,
    /// I/O operations in progress.
    pub io_in_progress: Vec<u64>,
    /// I/O time in milliseconds.
    pub io_time_ms: Vec<u64>,
    /// Number of disks.
    pub disk_count: usize,
}

impl DiskMetricsSoA {
    /// Creates a new SoA structure for the given disk count.
    #[must_use]
    pub fn new(disk_count: usize) -> Self {
        let count = disk_count.min(MAX_DISKS);
        let aligned_count = count.div_ceil(8) * 8;

        Self {
            names: Vec::with_capacity(count),
            reads_completed: vec![0; aligned_count],
            sectors_read: vec![0; aligned_count],
            read_time_ms: vec![0; aligned_count],
            writes_completed: vec![0; aligned_count],
            sectors_written: vec![0; aligned_count],
            write_time_ms: vec![0; aligned_count],
            io_in_progress: vec![0; aligned_count],
            io_time_ms: vec![0; aligned_count],
            disk_count: 0,
        }
    }

    /// Adds or updates a disk.
    #[allow(clippy::too_many_arguments)]
    pub fn set_disk(
        &mut self,
        name: &str,
        reads: u64,
        sectors_read: u64,
        read_time: u64,
        writes: u64,
        sectors_written: u64,
        write_time: u64,
        io_in_progress: u64,
        io_time: u64,
    ) {
        let idx = if let Some(pos) = self.names.iter().position(|n| n == name) {
            pos
        } else {
            if self.disk_count >= self.reads_completed.len() {
                return;
            }
            self.names.push(name.to_string());
            let idx = self.disk_count;
            self.disk_count += 1;
            idx
        };

        self.reads_completed[idx] = reads;
        self.sectors_read[idx] = sectors_read;
        self.read_time_ms[idx] = read_time;
        self.writes_completed[idx] = writes;
        self.sectors_written[idx] = sectors_written;
        self.write_time_ms[idx] = write_time;
        self.io_in_progress[idx] = io_in_progress;
        self.io_time_ms[idx] = io_time;
    }

    /// Returns total bytes read across all disks.
    #[must_use]
    pub fn total_bytes_read(&self) -> u64 {
        use super::kernels::simd_sum;
        let values: Vec<f64> = self.sectors_read[..self.disk_count]
            .iter()
            .map(|&v| (v * 512) as f64) // 512 bytes per sector
            .collect();
        simd_sum(&values) as u64
    }

    /// Returns total bytes written across all disks.
    #[must_use]
    pub fn total_bytes_written(&self) -> u64 {
        use super::kernels::simd_sum;
        let values: Vec<f64> =
            self.sectors_written[..self.disk_count].iter().map(|&v| (v * 512) as f64).collect();
        simd_sum(&values) as u64
    }
}

impl Default for DiskMetricsSoA {
    fn default() -> Self {
        Self::new(16)
    }
}

/// Battery metrics (not SoA - single battery typically).
#[repr(C, align(64))]
#[derive(Debug, Clone, Default)]
pub struct BatteryMetrics {
    /// Charge percentage (0-100).
    pub capacity: u8,
    /// Charging status.
    pub status: BatteryStatus,
    /// Current energy in microWh.
    pub energy_now: u64,
    /// Full energy in microWh.
    pub energy_full: u64,
    /// Design capacity in microWh.
    pub energy_full_design: u64,
    /// Current power draw in microW.
    pub power_now: u64,
    /// Voltage in microV.
    pub voltage_now: u64,
    /// Time to empty in seconds (if discharging).
    pub time_to_empty: Option<u64>,
    /// Time to full in seconds (if charging).
    pub time_to_full: Option<u64>,
    /// Battery health percentage.
    pub health_pct: f64,
}

/// Battery charging status.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum BatteryStatus {
    /// Status unknown or not available.
    #[default]
    Unknown,
    /// Battery is currently charging.
    Charging,
    /// Battery is discharging (on battery power).
    Discharging,
    /// Battery is not charging (plugged in but full or paused).
    NotCharging,
    /// Battery is fully charged.
    Full,
}

impl BatteryMetrics {
    /// Creates new battery metrics.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            capacity: 0,
            status: BatteryStatus::Unknown,
            energy_now: 0,
            energy_full: 0,
            energy_full_design: 0,
            power_now: 0,
            voltage_now: 0,
            time_to_empty: None,
            time_to_full: None,
            health_pct: 100.0,
        }
    }

    /// Calculates battery health.
    pub fn calculate_health(&mut self) {
        if self.energy_full_design > 0 {
            self.health_pct = (self.energy_full as f64 / self.energy_full_design as f64) * 100.0;
        }
    }

    /// Calculates time remaining based on current power draw.
    pub fn calculate_time_remaining(&mut self) {
        if self.power_now == 0 {
            self.time_to_empty = None;
            self.time_to_full = None;
            return;
        }

        match self.status {
            BatteryStatus::Discharging => {
                // Time to empty = energy_now / power_now (in hours -> seconds)
                let hours = self.energy_now as f64 / self.power_now as f64;
                self.time_to_empty = Some((hours * 3600.0) as u64);
                self.time_to_full = None;
            }
            BatteryStatus::Charging => {
                // Time to full = (energy_full - energy_now) / power_now
                let remaining = self.energy_full.saturating_sub(self.energy_now);
                let hours = remaining as f64 / self.power_now as f64;
                self.time_to_full = Some((hours * 3600.0) as u64);
                self.time_to_empty = None;
            }
            _ => {
                self.time_to_empty = None;
                self.time_to_full = None;
            }
        }
    }
}

/// Temperature sensor reading.
#[repr(C)]
#[derive(Debug, Clone, Default)]
pub struct TempReading {
    /// Sensor label/name.
    pub label: String,
    /// Current temperature in Celsius.
    pub current: f64,
    /// High threshold in Celsius.
    pub high: Option<f64>,
    /// Critical threshold in Celsius.
    pub critical: Option<f64>,
}

/// Sensor metrics in SoA layout.
#[repr(C, align(64))]
#[derive(Debug, Clone, Default)]
pub struct SensorMetricsSoA {
    /// Temperature readings.
    pub temps: Vec<TempReading>,
    /// Fan speeds (RPM).
    pub fan_speeds: Vec<(String, u64)>,
}

impl SensorMetricsSoA {
    /// Creates new sensor metrics.
    #[must_use]
    pub fn new() -> Self {
        Self { temps: Vec::with_capacity(32), fan_speeds: Vec::with_capacity(8) }
    }

    /// Adds a temperature reading.
    pub fn add_temp(
        &mut self,
        label: &str,
        current: f64,
        high: Option<f64>,
        critical: Option<f64>,
    ) {
        self.temps.push(TempReading { label: label.to_string(), current, high, critical });
    }

    /// Adds a fan speed reading.
    pub fn add_fan(&mut self, label: &str, rpm: u64) {
        self.fan_speeds.push((label.to_string(), rpm));
    }

    /// Returns average temperature across all sensors.
    #[must_use]
    pub fn avg_temp(&self) -> f64 {
        if self.temps.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.temps.iter().map(|t| t.current).sum();
        sum / self.temps.len() as f64
    }

    /// Returns maximum temperature.
    #[must_use]
    pub fn max_temp(&self) -> f64 {
        self.temps.iter().map(|t| t.current).fold(f64::MIN, f64::max)
    }

    /// Clears all readings.
    pub fn clear(&mut self) {
        self.temps.clear();
        self.fan_speeds.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_metrics_soa() {
        let mut metrics = CpuMetricsSoA::new(4);
        metrics.set_core(0, 100, 10, 50, 800, 20, 5, 5, 10);
        metrics.set_core(1, 200, 20, 100, 600, 40, 10, 10, 20);

        assert_eq!(metrics.user[0], 100);
        assert_eq!(metrics.user[1], 200);
        assert_eq!(metrics.total[0], 1000);
        assert_eq!(metrics.total[1], 1000);
    }

    #[test]
    fn test_memory_metrics_soa() {
        let mut mem = MemoryMetricsSoA::new();
        mem.total = 16_000_000_000;
        mem.available = 8_000_000_000;

        assert_eq!(mem.used(), 8_000_000_000);
        assert!((mem.usage_pct() - 50.0).abs() < 0.1);
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

    #[test]
    fn test_disk_metrics_soa() {
        let mut disk = DiskMetricsSoA::new(4);
        disk.set_disk("sda", 100, 1000, 50, 200, 2000, 100, 0, 150);

        assert_eq!(disk.disk_count, 1);
        assert_eq!(disk.total_bytes_read(), 512_000);
        assert_eq!(disk.total_bytes_written(), 1_024_000);
    }

    #[test]
    fn test_battery_metrics() {
        let mut battery = BatteryMetrics::new();
        battery.capacity = 80;
        battery.status = BatteryStatus::Discharging;
        battery.energy_now = 40_000_000; // 40 Wh
        battery.energy_full = 50_000_000; // 50 Wh
        battery.energy_full_design = 60_000_000; // 60 Wh design
        battery.power_now = 10_000_000; // 10 W

        battery.calculate_health();
        battery.calculate_time_remaining();

        assert!((battery.health_pct - 83.33).abs() < 0.1);
        assert!(battery.time_to_empty.is_some());
        // 40 Wh / 10 W = 4 hours = 14400 seconds
        assert_eq!(battery.time_to_empty, Some(14400));
    }

    #[test]
    fn test_sensor_metrics() {
        let mut sensors = SensorMetricsSoA::new();
        sensors.add_temp("CPU", 45.0, Some(80.0), Some(100.0));
        sensors.add_temp("GPU", 55.0, Some(85.0), Some(105.0));
        sensors.add_fan("CPU Fan", 1500);

        assert_eq!(sensors.temps.len(), 2);
        assert!((sensors.avg_temp() - 50.0).abs() < 0.1);
        assert!((sensors.max_temp() - 55.0).abs() < 0.1);
    }

    #[test]
    fn test_alignment() {
        assert_eq!(std::mem::align_of::<CpuMetricsSoA>(), 64);
        assert_eq!(std::mem::align_of::<MemoryMetricsSoA>(), 64);
        assert_eq!(std::mem::align_of::<NetworkMetricsSoA>(), 64);
        assert_eq!(std::mem::align_of::<DiskMetricsSoA>(), 64);
        assert_eq!(std::mem::align_of::<BatteryMetrics>(), 64);
        assert_eq!(std::mem::align_of::<SensorMetricsSoA>(), 64);
    }
}
