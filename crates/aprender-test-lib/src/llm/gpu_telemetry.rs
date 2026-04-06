//! GPU telemetry collection via nvidia-smi during benchmarks.
//!
//! Spawns `nvidia-smi` as a background process, parses CSV output,
//! and aggregates into summary statistics for the benchmark result.

use super::loadtest::{GpuTelemetry, TelemetryStat};
use std::process::Stdio;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;

/// A single GPU telemetry sample from nvidia-smi.
#[derive(Debug, Clone)]
pub struct GpuSample {
    /// GPU compute utilization (%).
    pub gpu_utilization_pct: f64,
    /// GPU memory utilization (%).
    pub memory_utilization_pct: f64,
    /// GPU memory used (MB).
    pub memory_used_mb: f64,
    /// Total GPU memory (MB).
    pub memory_total_mb: f64,
    /// Power draw (watts).
    pub power_draw_w: f64,
    /// GPU temperature (Celsius).
    pub temperature_c: f64,
    /// GPU clock speed (MHz).
    pub clock_gpu_mhz: f64,
    /// Memory clock speed (MHz).
    pub clock_mem_mhz: f64,
}

/// Collects GPU telemetry by polling nvidia-smi.
///
/// When `gpu_host` is set and is not localhost, telemetry is collected
/// from the remote machine via SSH (GH-34).
#[allow(missing_debug_implementations)]
pub struct GpuTelemetryCollector {
    samples: Vec<GpuSample>,
    child: Option<tokio::process::Child>,
    rx: Option<tokio::sync::mpsc::Receiver<GpuSample>>,
    expected_clock_mhz: Option<u32>,
    poll_interval_s: u64,
    gpu_host: Option<String>,
}

impl GpuTelemetryCollector {
    /// Create a new collector.
    pub fn new(poll_interval_s: u64, expected_clock_mhz: Option<u32>) -> Self {
        Self {
            samples: Vec::new(),
            child: None,
            rx: None,
            expected_clock_mhz,
            poll_interval_s,
            gpu_host: None,
        }
    }

    /// Set the remote host for GPU telemetry collection (GH-34).
    ///
    /// When the target URL points to a remote machine, extract the hostname
    /// and pass it here. Telemetry will use `ssh <host> nvidia-smi ...`.
    pub fn with_host(mut self, host: Option<String>) -> Self {
        self.gpu_host = host;
        self
    }

    /// Start collecting GPU telemetry in the background.
    pub async fn start(&mut self) -> Result<(), String> {
        let interval = self.poll_interval_s;
        let nvsmi_args = format!(
            "nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,clocks.gr,clocks.mem --format=csv,noheader,nounits -l {interval}"
        );

        // GH-34: if gpu_host is set and not localhost, use SSH
        let mut child = if let Some(ref host) = self.gpu_host {
            if is_remote_host(host) {
                Command::new("ssh")
                    .args([
                        "-o",
                        "StrictHostKeyChecking=no",
                        "-o",
                        "ConnectTimeout=5",
                        host,
                        &nvsmi_args,
                    ])
                    .stdout(Stdio::piped())
                    .stderr(Stdio::null())
                    .spawn()
                    .map_err(|e| format!("Failed to start ssh {host} nvidia-smi: {e}"))?
            } else {
                spawn_local_nvidia_smi(interval)?
            }
        } else {
            spawn_local_nvidia_smi(interval)?
        };

        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| "Failed to capture nvidia-smi stdout".to_string())?;

        let (tx, rx) = tokio::sync::mpsc::channel(1024);
        self.rx = Some(rx);
        self.child = Some(child);

        tokio::spawn(async move {
            let reader = BufReader::new(stdout);
            let mut lines = reader.lines();
            while let Ok(Some(line)) = lines.next_line().await {
                if let Some(sample) = parse_nvidia_smi_line(&line) {
                    if tx.send(sample).await.is_err() {
                        break;
                    }
                }
            }
        });

        Ok(())
    }

    /// Stop collecting and aggregate results.
    pub async fn stop(
        &mut self,
        completion_tokens: u64,
        total_requests: u64,
    ) -> Option<GpuTelemetry> {
        // Kill nvidia-smi process
        if let Some(ref mut child) = self.child {
            let _ = child.kill().await;
        }
        self.child = None;

        // Drain remaining samples
        if let Some(ref mut rx) = self.rx {
            while let Ok(sample) = rx.try_recv() {
                self.samples.push(sample);
            }
        }
        self.rx = None;

        if self.samples.is_empty() {
            return None;
        }

        Some(self.aggregate(completion_tokens, total_requests))
    }

    fn aggregate(&self, completion_tokens: u64, total_requests: u64) -> GpuTelemetry {
        let n = self.samples.len();
        let interval_s = self.poll_interval_s as f64;

        let gpu_util = stat(&self.samples, |s| s.gpu_utilization_pct);
        let mem_used = stat(&self.samples, |s| s.memory_used_mb);
        let power = stat(&self.samples, |s| s.power_draw_w);
        let temp = stat(&self.samples, |s| s.temperature_c);
        let clock = stat(&self.samples, |s| s.clock_gpu_mhz);

        let memory_total_mb = self.samples.first().map_or(0.0, |s| s.memory_total_mb);

        // Energy: sum(power_w * interval_s) / 3600 = Wh
        let energy_total_wh: f64 = self
            .samples
            .iter()
            .map(|s| s.power_draw_w * interval_s)
            .sum::<f64>()
            / 3600.0;

        let energy_j = energy_total_wh * 3600.0;
        let energy_per_token_mj = if completion_tokens > 0 {
            energy_j * 1000.0 / completion_tokens as f64
        } else {
            0.0
        };
        let energy_per_request_mj = if total_requests > 0 {
            energy_j * 1000.0 / total_requests as f64
        } else {
            0.0
        };

        // Throttle detection: clock drop >10% from expected or max observed
        let expected_clock = self.expected_clock_mhz.map(f64::from).unwrap_or(clock.max);
        let throttle_threshold = expected_clock * 0.9;
        let throttle_events = self
            .samples
            .iter()
            .filter(|s| s.clock_gpu_mhz < throttle_threshold && expected_clock > 0.0)
            .count();

        GpuTelemetry {
            samples: n,
            gpu_utilization_pct: gpu_util,
            memory_used_mb: mem_used,
            memory_total_mb,
            power_draw_w: power,
            temperature_c: temp,
            clock_gpu_mhz: clock,
            throttle_events,
            energy_total_wh,
            energy_per_token_mj,
            energy_per_request_mj,
        }
    }
}

fn stat<F: Fn(&GpuSample) -> f64>(samples: &[GpuSample], f: F) -> TelemetryStat {
    if samples.is_empty() {
        return TelemetryStat {
            mean: 0.0,
            max: 0.0,
            min: 0.0,
        };
    }
    let values: Vec<f64> = samples.iter().map(&f).collect();
    let sum: f64 = values.iter().sum();
    let mean = sum / values.len() as f64;
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    TelemetryStat { mean, max, min }
}

fn spawn_local_nvidia_smi(interval: u64) -> Result<tokio::process::Child, String> {
    Command::new("nvidia-smi")
        .args([
            "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,clocks.gr,clocks.mem",
            "--format=csv,noheader,nounits",
            "-l",
            &interval.to_string(),
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| format!("Failed to start nvidia-smi: {e}"))
}

fn is_remote_host(host: &str) -> bool {
    !matches!(host, "localhost" | "127.0.0.1" | "::1" | "0.0.0.0")
}

/// Extract hostname from a URL for GPU telemetry collection.
pub fn extract_host_from_url(url: &str) -> Option<String> {
    url.split("//")
        .nth(1)
        .and_then(|rest| rest.split(':').next())
        .map(String::from)
}

fn parse_nvidia_smi_line(line: &str) -> Option<GpuSample> {
    let parts: Vec<&str> = line.split(',').map(str::trim).collect();
    if parts.len() < 8 {
        return None;
    }
    Some(GpuSample {
        gpu_utilization_pct: parts[0].parse().ok()?,
        memory_utilization_pct: parts[1].parse().ok()?,
        memory_used_mb: parts[2].parse().ok()?,
        memory_total_mb: parts[3].parse().ok()?,
        power_draw_w: parts[4].parse().ok()?,
        temperature_c: parts[5].parse().ok()?,
        clock_gpu_mhz: parts[6].parse().ok()?,
        clock_mem_mhz: parts[7].parse().ok()?,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_nvidia_smi_line() {
        let line = "82, 45, 1842, 8192, 45.2, 72, 1485, 6000";
        let sample = parse_nvidia_smi_line(line).unwrap();
        assert!((sample.gpu_utilization_pct - 82.0).abs() < 0.1);
        assert!((sample.memory_used_mb - 1842.0).abs() < 0.1);
        assert!((sample.power_draw_w - 45.2).abs() < 0.1);
        assert!((sample.temperature_c - 72.0).abs() < 0.1);
        assert!((sample.clock_gpu_mhz - 1485.0).abs() < 0.1);
    }

    #[test]
    fn test_parse_nvidia_smi_line_bad() {
        assert!(parse_nvidia_smi_line("bad data").is_none());
        assert!(parse_nvidia_smi_line("1, 2, 3").is_none());
    }

    #[test]
    fn test_stat_empty() {
        let samples: Vec<GpuSample> = Vec::new();
        let s = stat(&samples, |s| s.gpu_utilization_pct);
        assert_eq!(s.mean, 0.0);
    }

    #[test]
    fn test_is_remote_host() {
        assert!(!is_remote_host("localhost"));
        assert!(!is_remote_host("127.0.0.1"));
        assert!(!is_remote_host("::1"));
        assert!(is_remote_host("192.168.50.38"));
        assert!(is_remote_host("yoga"));
        assert!(is_remote_host("jetson"));
    }

    #[test]
    fn test_extract_host_from_url() {
        assert_eq!(
            extract_host_from_url("http://192.168.50.38:8081"),
            Some("192.168.50.38".to_string())
        );
        assert_eq!(
            extract_host_from_url("http://localhost:8081"),
            Some("localhost".to_string())
        );
        assert_eq!(extract_host_from_url("bad"), None);
    }

    #[test]
    fn test_aggregate() {
        let collector = GpuTelemetryCollector {
            samples: vec![
                GpuSample {
                    gpu_utilization_pct: 80.0,
                    memory_utilization_pct: 40.0,
                    memory_used_mb: 1800.0,
                    memory_total_mb: 8192.0,
                    power_draw_w: 45.0,
                    temperature_c: 70.0,
                    clock_gpu_mhz: 1500.0,
                    clock_mem_mhz: 6000.0,
                },
                GpuSample {
                    gpu_utilization_pct: 90.0,
                    memory_utilization_pct: 50.0,
                    memory_used_mb: 1900.0,
                    memory_total_mb: 8192.0,
                    power_draw_w: 50.0,
                    temperature_c: 75.0,
                    clock_gpu_mhz: 1200.0, // throttled
                    clock_mem_mhz: 6000.0,
                },
            ],
            child: None,
            rx: None,
            expected_clock_mhz: Some(1500),
            poll_interval_s: 1,
            gpu_host: None,
        };

        let telemetry = collector.aggregate(100, 10);
        assert_eq!(telemetry.samples, 2);
        assert!((telemetry.gpu_utilization_pct.mean - 85.0).abs() < 0.1);
        assert_eq!(telemetry.throttle_events, 1); // 1200 < 1500*0.9=1350
        assert!(telemetry.energy_total_wh > 0.0);
        assert!(telemetry.energy_per_token_mj > 0.0);
    }
}
