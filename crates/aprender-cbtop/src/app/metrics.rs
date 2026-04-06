//! Real system metrics collection from /proc filesystem.

use std::time::Instant;

use super::hardware::{DiskMetrics, MemoryBreakdown, NetworkMetrics};
use super::CbtopApp;

impl CbtopApp {
    /// Read memory breakdown from /proc/meminfo (PMAT-012 UI-04)
    pub(super) fn read_memory() -> MemoryBreakdown {
        #[cfg(target_os = "linux")]
        {
            if let Ok(contents) = std::fs::read_to_string("/proc/meminfo") {
                let mut mem = MemoryBreakdown::default();
                for line in contents.lines() {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        let value: u64 = parts[1].parse().unwrap_or(0);
                        match parts[0] {
                            "MemTotal:" => mem.total_kb = value,
                            "MemAvailable:" => mem.available_kb = value,
                            "Buffers:" => mem.buffers_kb = value,
                            "Cached:" => mem.cached_kb = value,
                            _ => {}
                        }
                    }
                }
                mem.used_kb = mem.total_kb.saturating_sub(mem.available_kb);
                return mem;
            }
        }
        MemoryBreakdown::default()
    }

    /// Read network metrics from /proc/net/dev (PMAT-012 UI-07 P2)
    pub(super) fn read_network(&mut self) -> NetworkMetrics {
        #[cfg(target_os = "linux")]
        {
            if let Ok(contents) = std::fs::read_to_string("/proc/net/dev") {
                let mut total_rx: u64 = 0;
                let mut total_tx: u64 = 0;

                for line in contents.lines().skip(2) {
                    // Skip header lines
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 10 {
                        let iface = parts[0].trim_end_matches(':');
                        // Skip loopback
                        if iface == "lo" {
                            continue;
                        }
                        let rx: u64 = parts[1].parse().unwrap_or(0);
                        let tx: u64 = parts[9].parse().unwrap_or(0);
                        total_rx += rx;
                        total_tx += tx;
                    }
                }

                let now = Instant::now();
                let (rx_rate, tx_rate) =
                    if let Some((prev_rx, prev_tx, prev_time)) = self.last_network_stat {
                        let elapsed = now.duration_since(prev_time).as_secs_f64();
                        if elapsed > 0.0 {
                            let rx_delta = total_rx.saturating_sub(prev_rx) as f64;
                            let tx_delta = total_tx.saturating_sub(prev_tx) as f64;
                            (rx_delta / elapsed, tx_delta / elapsed)
                        } else {
                            (0.0, 0.0)
                        }
                    } else {
                        (0.0, 0.0)
                    };

                self.last_network_stat = Some((total_rx, total_tx, now));

                return NetworkMetrics {
                    rx_bytes: total_rx,
                    tx_bytes: total_tx,
                    rx_rate,
                    tx_rate,
                };
            }
        }
        NetworkMetrics::default()
    }

    /// Read disk metrics using statvfs (PMAT-012 UI-08 P2)
    pub(super) fn read_disks() -> Vec<DiskMetrics> {
        let mut disks = Vec::new();

        #[cfg(target_os = "linux")]
        {
            // Read mounts from /proc/mounts and get stats for common ones
            if let Ok(contents) = std::fs::read_to_string("/proc/mounts") {
                for line in contents.lines() {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        let mount = parts[1];
                        let fstype = parts.get(2).unwrap_or(&"");

                        // Only include real filesystems on common mounts
                        if !matches!(*fstype, "ext4" | "xfs" | "btrfs" | "zfs" | "ntfs" | "vfat") {
                            continue;
                        }
                        // Skip if not a standard mount
                        if !mount.starts_with("/home") && mount != "/" && !mount.starts_with("/mnt")
                        {
                            continue;
                        }

                        // Use nix or libc statvfs
                        #[cfg(unix)]
                        {
                            use std::ffi::CString;
                            use std::mem::MaybeUninit;

                            if let Ok(c_path) = CString::new(mount) {
                                let mut stat = MaybeUninit::<libc::statvfs>::uninit();
                                let result =
                                    unsafe { libc::statvfs(c_path.as_ptr(), stat.as_mut_ptr()) };
                                if result == 0 {
                                    let stat = unsafe { stat.assume_init() };
                                    let block_size = stat.f_frsize;
                                    let total = stat.f_blocks * block_size;
                                    let available = stat.f_bavail * block_size;
                                    let used = total.saturating_sub(available);
                                    let usage_pct = if total > 0 {
                                        (used as f64 / total as f64) * 100.0
                                    } else {
                                        0.0
                                    };

                                    disks.push(DiskMetrics {
                                        mount: mount.to_string(),
                                        total_bytes: total,
                                        used_bytes: used,
                                        usage_percent: usage_pct,
                                    });

                                    // Limit to 3 disks for display
                                    if disks.len() >= 3 {
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        disks
    }

    /// Read real CPU usage from /proc/stat (aggregate and per-core)
    pub(super) fn read_cpu_usage(&mut self) -> f64 {
        #[cfg(target_os = "linux")]
        {
            if let Ok(contents) = std::fs::read_to_string("/proc/stat") {
                let mut aggregate_usage = 0.0;
                let mut per_core: Vec<f64> = Vec::new();

                for line in contents.lines() {
                    if line.starts_with("cpu") {
                        let parts: Vec<&str> = line.split_whitespace().collect();
                        if parts.len() >= 5 {
                            let user: u64 = parts[1].parse().unwrap_or(0);
                            let nice: u64 = parts[2].parse().unwrap_or(0);
                            let system: u64 = parts[3].parse().unwrap_or(0);
                            let idle: u64 = parts[4].parse().unwrap_or(0);

                            let total = user + nice + system + idle;
                            let active = user + nice + system;

                            if parts[0] == "cpu" {
                                // Aggregate CPU line
                                if let Some((prev_active, prev_total)) = self.last_cpu_stat {
                                    let delta_active = active.saturating_sub(prev_active);
                                    let delta_total = total.saturating_sub(prev_total);
                                    if delta_total > 0 {
                                        aggregate_usage =
                                            (delta_active as f64 / delta_total as f64) * 100.0;
                                    }
                                }
                                self.last_cpu_stat = Some((active, total));
                            } else if parts[0].starts_with("cpu") {
                                // Per-core CPU line (cpu0, cpu1, etc.)
                                // Calculate instantaneous usage (simplified - no delta tracking per core)
                                if total > 0 {
                                    per_core.push((active as f64 / total as f64) * 100.0);
                                }
                            }
                        }
                    }
                }

                self.load_metrics.per_core_usage = per_core;
                return aggregate_usage;
            }
        }
        // Fallback for non-Linux or on error
        0.0
    }
}
