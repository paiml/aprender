//! System health and VRAM collection via nvidia-smi and /proc.
//! Spec sections 9.8 (VRAM), 9.10 (System Health), 9.11 (Energy).

use crate::metrics::catalog::{EnergyMetrics, SystemHealthMetrics, VramMetrics};
use std::process::Command;

/// Collect system health metrics from nvidia-smi (NVML) and /proc.
pub fn collect_system_health() -> Option<SystemHealthMetrics> {
    let gpu = query_nvidia_smi(&[
        "temperature.gpu",
        "power.draw",
        "clocks.current.sm",
        "clocks.current.memory",
        "memory.used",
        "memory.total",
    ])?;

    let fields: Vec<&str> = gpu.split(", ").collect();
    if fields.len() < 6 {
        return None;
    }

    let cpu_freq = read_cpu_frequency().unwrap_or(0.0);
    let cpu_temp = read_cpu_temperature().unwrap_or(0.0);

    // Unified-memory platforms report memory.total as N/A (parses to 0); fall back to system RAM.
    let mut gpu_mem_total = parse_nvidia_val(fields[5]);
    if gpu_mem_total <= 0.0 {
        gpu_mem_total = read_system_memory_total_mb().unwrap_or(0.0);
    }

    Some(SystemHealthMetrics {
        gpu_temperature_celsius: parse_nvidia_val(fields[0]),
        gpu_power_watts: parse_nvidia_val(fields[1]),
        gpu_clock_mhz: parse_nvidia_val(fields[2]),
        gpu_memory_clock_mhz: parse_nvidia_val(fields[3]),
        cpu_frequency_mhz: cpu_freq,
        cpu_temperature_celsius: cpu_temp,
        gpu_memory_used_mb: parse_nvidia_val(fields[4]),
        gpu_memory_total_mb: gpu_mem_total,
    })
}

/// Collect VRAM metrics from nvidia-smi.
pub fn collect_vram() -> Option<VramMetrics> {
    let gpu = query_nvidia_smi(&["memory.used", "memory.total", "memory.free"])?;

    let fields: Vec<&str> = gpu.split(", ").collect();
    if fields.len() < 3 {
        return None;
    }

    let used = parse_nvidia_val(fields[0]);
    let mut total = parse_nvidia_val(fields[1]);
    let free = parse_nvidia_val(fields[2]);
    // Unified-memory NVIDIA platforms (GB10/GH200/Jetson) report VRAM total as [N/A] via nvidia-smi
    // because the GPU shares system RAM. Fall back to total system memory so the profiler reports the
    // (unified) memory budget instead of 0.
    if total <= 0.0 {
        total = read_system_memory_total_mb().unwrap_or(0.0);
    }
    let utilization = if total > 0.0 {
        used / total * 100.0
    } else {
        0.0
    };

    Some(VramMetrics {
        vram_used_mb: used,
        vram_total_mb: total,
        vram_free_mb: free,
        vram_utilization_pct: utilization,
        vram_peak_mb: used, // snapshot — no tracking history
        vram_allocation_count: 0,
        vram_fragmentation_pct: 0.0,
    })
}

/// Compute energy efficiency from power and throughput.
pub fn compute_energy(power_watts: f64, tflops: f64, duration_us: f64) -> Option<EnergyMetrics> {
    if power_watts <= 0.0 {
        return None;
    }
    let tflops_per_watt = if power_watts > 0.0 {
        tflops / power_watts
    } else {
        0.0
    };
    let joules = power_watts * duration_us * 1e-6;
    Some(EnergyMetrics {
        tflops_per_watt,
        joules_per_inference: joules,
    })
}

/// Run nvidia-smi --query-gpu and return the CSV row.
fn query_nvidia_smi(fields: &[&str]) -> Option<String> {
    let query = fields.join(",");
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu", &query, "--format=csv,noheader,nounits"])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let line = stdout.trim();
    if line.is_empty() || line.contains("[N/A]") && line.chars().all(|c| c == ',' || c == ' ') {
        return None;
    }
    Some(line.to_string())
}

/// Parse a numeric value from nvidia-smi output (handles "123 W", "45 MiB", etc.)
fn parse_nvidia_val(s: &str) -> f64 {
    let s = s.trim();
    if s == "[N/A]" || s == "N/A" {
        return 0.0;
    }
    // Take the first token that looks numeric
    s.split_whitespace()
        .next()
        .and_then(|token| token.parse::<f64>().ok())
        .unwrap_or(0.0)
}

/// Total system RAM in MB, read from `/proc/meminfo` (`MemTotal`).
///
/// Used as a fallback for GPU memory total on unified-memory NVIDIA platforms (GB10 / Grace-Blackwell,
/// GH200, Jetson), where `nvidia-smi --query-gpu=memory.total` reports `[N/A]` because the GPU shares
/// system RAM rather than exposing dedicated VRAM. On dedicated GPUs nvidia-smi returns a real value,
/// so this fallback is never reached and behavior is unchanged.
fn read_system_memory_total_mb() -> Option<f64> {
    let content = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in content.lines() {
        // Format: "MemTotal:       65780480 kB"
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            let kb = rest.split_whitespace().next()?.parse::<f64>().ok()?;
            return Some(kb / 1024.0);
        }
    }
    None
}

/// Read current CPU frequency from /proc/cpuinfo (MHz).
fn read_cpu_frequency() -> Option<f64> {
    let content = std::fs::read_to_string("/proc/cpuinfo").ok()?;
    // Take average across all cores
    let mut total = 0.0;
    let mut count = 0;
    for line in content.lines() {
        if line.starts_with("cpu MHz") {
            if let Some(val) = line.split(':').nth(1) {
                if let Ok(mhz) = val.trim().parse::<f64>() {
                    total += mhz;
                    count += 1;
                }
            }
        }
    }
    if count > 0 {
        Some(total / count as f64)
    } else {
        None
    }
}

/// Read CPU temperature from /sys thermal zones.
fn read_cpu_temperature() -> Option<f64> {
    // Try thermal_zone0 first (usually CPU package)
    for i in 0..10 {
        let path = format!("/sys/class/thermal/thermal_zone{i}/temp");
        if let Ok(content) = std::fs::read_to_string(&path) {
            if let Ok(millidegrees) = content.trim().parse::<f64>() {
                return Some(millidegrees / 1000.0);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_nvidia_val() {
        assert!((parse_nvidia_val("285.32 W") - 285.32).abs() < 0.01);
        assert!((parse_nvidia_val("24564 MiB") - 24564.0).abs() < 1.0);
        assert!((parse_nvidia_val("62") - 62.0).abs() < 0.01);
        assert!((parse_nvidia_val("[N/A]")).abs() < 0.01);
        assert!((parse_nvidia_val("N/A")).abs() < 0.01);
    }

    #[test]
    fn test_compute_energy() {
        let e = compute_energy(300.0, 11.6, 23.2).unwrap();
        assert!((e.tflops_per_watt - 11.6 / 300.0).abs() < 0.001);
        assert!((e.joules_per_inference - 300.0 * 23.2e-6).abs() < 0.001);
    }

    #[test]
    fn test_compute_energy_zero_power() {
        assert!(compute_energy(0.0, 11.6, 23.2).is_none());
    }

    /// System health collection should not panic even without nvidia-smi.
    #[test]
    fn test_collect_system_health_no_panic() {
        let _ = collect_system_health();
    }

    /// VRAM collection should not panic even without nvidia-smi.
    #[test]
    fn test_collect_vram_no_panic() {
        let _ = collect_vram();
    }

    #[test]
    fn test_read_cpu_frequency_no_panic() {
        let _ = read_cpu_frequency();
    }

    /// On Linux `/proc/meminfo` always reports a positive `MemTotal`. This backs the
    /// unified-memory VRAM fallback (GB10/GH200/Jetson report memory.total = N/A).
    #[test]
    fn test_read_system_memory_total_mb() {
        let total = read_system_memory_total_mb();
        assert!(total.is_some(), "/proc/meminfo MemTotal should be readable");
        assert!(total.unwrap() > 0.0, "system memory total should be > 0 MB");
    }

    #[test]
    fn test_read_cpu_temperature_no_panic() {
        let _ = read_cpu_temperature();
    }

    /// How many GPUs `nvidia-smi` actually REPORTS -- not whether the binary exists.
    ///
    /// `which::which("nvidia-smi").is_ok()` was the precondition for both GPU tests below,
    /// and it is the wrong question. The intel clean-room runner has the NVIDIA userland
    /// installed and no visible device, so `nvidia-smi` resolves, the tests decided a GPU
    /// was present, the collectors correctly returned `None`, and Coverage Nightly failed
    /// with "nvidia-smi exists but no health data" -- a red build reporting an ENVIRONMENT
    /// fact as a code defect. A binary on `PATH` is not a device on the bus.
    fn gpus_reported() -> usize {
        let Ok(out) = std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=name", "--format=csv,noheader"])
            .output()
        else {
            return 0; // not installed, or not executable: no device either way
        };
        if !out.status.success() {
            return 0; // installed, but the driver has nothing to report
        }
        String::from_utf8_lossy(&out.stdout)
            .lines()
            .filter(|l| !l.trim().is_empty())
            .count()
    }

    /// When a GPU IS reported, the collectors must return sane data.
    ///
    /// Paired with `test_gpu_collectors_are_none_without_a_reporting_gpu` so that exactly
    /// one of the two is meaningful on any given box and NEITHER is vacuous: a machine
    /// with a GPU proves the parse works, a machine without one proves the absence path
    /// returns `None` instead of panicking or fabricating a reading.
    #[test]
    fn test_gpu_collectors_have_valid_data_when_a_gpu_is_reported() {
        if gpus_reported() == 0 {
            eprintln!(
                "skip: nvidia-smi reports no GPU on this host; the paired negative test covers it"
            );
            return;
        }
        let health = collect_system_health().expect("a reporting GPU must yield health data");
        assert!(
            health.gpu_temperature_celsius > 0.0,
            "GPU temp should be > 0"
        );
        assert!(
            health.gpu_memory_total_mb > 0.0,
            "GPU memory total should be > 0"
        );

        let vram = collect_vram().expect("a reporting GPU must yield VRAM data");
        assert!(vram.vram_total_mb > 0.0, "VRAM total should be > 0");
        assert!(vram.vram_utilization_pct >= 0.0 && vram.vram_utilization_pct <= 100.0);
    }

    /// When no GPU is reported, the collectors must return `None` -- never a fabricated
    /// zero, and never a panic. This is the half that runs in the clean room.
    #[test]
    fn test_gpu_collectors_are_none_without_a_reporting_gpu() {
        if gpus_reported() > 0 {
            eprintln!("skip: this host reports a GPU; the paired positive test covers it");
            return;
        }
        assert!(
            collect_system_health().is_none(),
            "with no GPU reported, health must be None rather than a fabricated reading"
        );
        assert!(
            collect_vram().is_none(),
            "with no GPU reported, VRAM must be None rather than a fabricated reading"
        );
    }
}
