//! CPU SIMD profiling via perf stat + renacer + trueno-explain.
//! Spec section 4.2.

use anyhow::Result;
use std::collections::HashMap;
use std::process::Command;

/// perf stat hardware counters for SIMD analysis.
pub const SIMD_PERF_EVENTS: &[&str] = &[
    "cycles",
    "instructions",
    "cache-references",
    "cache-misses",
    "L1-dcache-load-misses",
    "LLC-loads",
    "branches",
    "branch-misses",
];

/// Architecture-specific perf events for SIMD utilization.
pub const AVX2_EVENTS: &[&str] = &[
    "fp_arith_inst_retired.scalar_single",
    "fp_arith_inst_retired.128b_packed_single",
    "fp_arith_inst_retired.256b_packed_single",
];

pub const AVX512_EVENTS: &[&str] = &[
    "fp_arith_inst_retired.scalar_single",
    "fp_arith_inst_retired.256b_packed_single",
    "fp_arith_inst_retired.512b_packed_single",
];

/// Parsed perf stat output.
#[derive(Debug, Clone, Default)]
pub struct PerfStatResult {
    pub counters: HashMap<String, u64>,
    pub wall_time_secs: f64,
}

impl PerfStatResult {
    /// Compute IPC (instructions per cycle).
    pub fn ipc(&self) -> f64 {
        let cycles = *self.counters.get("cycles").unwrap_or(&0) as f64;
        let instructions = *self.counters.get("instructions").unwrap_or(&0) as f64;
        if cycles > 0.0 {
            instructions / cycles
        } else {
            0.0
        }
    }

    /// Compute cache miss rate.
    pub fn cache_miss_rate(&self) -> f64 {
        let refs = *self.counters.get("cache-references").unwrap_or(&0) as f64;
        let misses = *self.counters.get("cache-misses").unwrap_or(&0) as f64;
        if refs > 0.0 {
            misses / refs * 100.0
        } else {
            0.0
        }
    }

    /// Compute branch misprediction rate.
    pub fn branch_miss_rate(&self) -> f64 {
        let branches = *self.counters.get("branches").unwrap_or(&0) as f64;
        let misses = *self.counters.get("branch-misses").unwrap_or(&0) as f64;
        if branches > 0.0 {
            misses / branches * 100.0
        } else {
            0.0
        }
    }

    /// Compute SIMD utilization: vector_ops / (vector_ops + scalar_ops) * 100.
    pub fn simd_utilization(&self) -> Option<f64> {
        let scalar = *self.counters.get("fp_arith_inst_retired.scalar_single").unwrap_or(&0) as f64;
        let vec128 =
            *self.counters.get("fp_arith_inst_retired.128b_packed_single").unwrap_or(&0) as f64;
        let vec256 =
            *self.counters.get("fp_arith_inst_retired.256b_packed_single").unwrap_or(&0) as f64;
        let vec512 =
            *self.counters.get("fp_arith_inst_retired.512b_packed_single").unwrap_or(&0) as f64;

        let vector = vec128 + vec256 + vec512;
        let total = scalar + vector;
        if total > 0.0 {
            Some(vector / total * 100.0)
        } else {
            None
        }
    }
}

/// Run perf stat and parse the output.
pub fn run_perf_stat(binary: &str, args: &[&str], events: &[&str]) -> Result<PerfStatResult> {
    let event_str = events.join(",");
    let mut cmd = Command::new("perf");
    cmd.arg("stat")
        .arg("-e")
        .arg(&event_str)
        .arg("-x")
        .arg(",") // CSV separator
        .arg(binary)
        .args(args);

    let output = cmd.output()?;
    let stderr = String::from_utf8_lossy(&output.stderr);

    parse_perf_stat_csv(&stderr)
}

/// Parse perf stat CSV output (perf writes stats to stderr).
/// Format: value,unit,event_name,... (with -x ,)
pub fn parse_perf_stat_csv(output: &str) -> Result<PerfStatResult> {
    let mut result = PerfStatResult::default();

    for line in output.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("Performance") {
            continue;
        }

        // Extract wall time from "X.YZ seconds time elapsed" lines
        if line.contains("seconds time elapsed") {
            if let Some(time_str) = line.split_whitespace().next() {
                if let Ok(t) = time_str.parse::<f64>() {
                    result.wall_time_secs = t;
                }
            }
            continue;
        }

        // CSV format: value,unit,event-name,...
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() >= 3 {
            let value_str = fields[0].trim().replace(' ', "");
            let event_name = fields[2].trim();

            if let Ok(value) = value_str.parse::<u64>() {
                result.counters.insert(event_name.to_string(), value);
            }
        }
    }

    Ok(result)
}

/// Profile a SIMD function.
pub fn profile_simd(function: &str, size: u32, arch: &str) -> Result<()> {
    println!("\n=== CGP SIMD Profile: {function} (size={size}, arch={arch}) ===\n");

    // Validate architecture
    let simd_events: &[&str] = match arch {
        "avx2" => {
            #[cfg(target_arch = "x86_64")]
            {
                if !std::arch::is_x86_feature_detected!("avx2") {
                    println!("  Warning: AVX2 not available on this CPU.");
                }
            }
            AVX2_EVENTS
        }
        "avx512" => {
            #[cfg(target_arch = "x86_64")]
            {
                if !std::arch::is_x86_feature_detected!("avx512f") {
                    println!("  Warning: AVX-512 not available on this CPU.");
                }
            }
            AVX512_EVENTS
        }
        "neon" => {
            #[cfg(not(target_arch = "aarch64"))]
            {
                println!("  NEON not available -- use --cross-profile for QEMU-based analysis");
                return Ok(());
            }
            #[cfg(target_arch = "aarch64")]
            {
                &["INST_RETIRED", "CPU_CYCLES", "ASE_SPEC"][..]
            }
        }
        "sse2" => {
            &["fp_arith_inst_retired.scalar_single", "fp_arith_inst_retired.128b_packed_single"][..]
        }
        _ => {
            anyhow::bail!("Unknown SIMD architecture: {arch}. Supported: avx2, avx512, neon, sse2")
        }
    };

    let has_perf = which::which("perf").is_ok();
    if !has_perf {
        println!("  perf not found. Install linux-tools-common for hardware counter profiling.");
        println!("  Showing static analysis only.");
        println!("\n  Function: {function}");
        println!("  Architecture: {arch}");
        return Ok(());
    }

    // Try to find a trueno benchmark binary
    let bin_path = find_bench_binary();
    match bin_path {
        Some(binary) => {
            println!("  Backend: perf stat");
            println!("  Binary: {binary}");

            // Collect base + SIMD counters
            let mut all_events: Vec<&str> = SIMD_PERF_EVENTS.to_vec();
            all_events.extend_from_slice(simd_events);

            match run_perf_stat(&binary, &[], &all_events) {
                Ok(result) => {
                    // Check if counters are all zero (paranoid mode)
                    let cycles = *result.counters.get("cycles").unwrap_or(&0);
                    if cycles == 0 && !result.counters.is_empty() {
                        let paranoid =
                            std::fs::read_to_string("/proc/sys/kernel/perf_event_paranoid")
                                .ok()
                                .and_then(|s| s.trim().parse::<i32>().ok())
                                .unwrap_or(-1);
                        if paranoid > 2 {
                            println!("  \x1b[33m[WARN]\x1b[0m perf_event_paranoid={paranoid} — hardware counters blocked.");
                            println!("  Fix: sudo sysctl kernel.perf_event_paranoid=2");
                            println!("  Or run: sudo cgp profile simd ...\n");
                        }
                    }

                    println!("\n  Hardware Counters:");
                    println!("    Cycles:       {:>14}", format_count(cycles));
                    println!(
                        "    Instructions: {:>14}",
                        format_count(*result.counters.get("instructions").unwrap_or(&0))
                    );
                    println!("    IPC:          {:>14.2}", result.ipc());
                    println!("    Cache miss:   {:>13.1}%", result.cache_miss_rate());
                    println!("    Branch miss:  {:>13.1}%", result.branch_miss_rate());

                    if let Some(simd_pct) = result.simd_utilization() {
                        println!("\n  SIMD Utilization:");
                        println!("    Vector ops:    {simd_pct:.1}%");
                        println!("    Scalar ops:    {:.1}%", 100.0 - simd_pct);
                        if simd_pct < 50.0 {
                            println!(
                                "    [WARN] Low SIMD utilization — check for scalar fallbacks"
                            );
                        } else {
                            println!("    [OK] Good SIMD utilization");
                        }
                    }

                    if result.wall_time_secs > 0.0 {
                        println!("\n  Wall time: {:.3}s", result.wall_time_secs);
                    }
                }
                Err(e) => {
                    println!("  perf stat failed: {e}");
                    println!("  Try: sudo sysctl kernel.perf_event_paranoid=2");
                }
            }
        }
        None => {
            println!("  No benchmark binary found.");
            println!("  Build with: cargo build --release --bench vector_ops");
            println!("  Then re-run cgp profile simd.");
        }
    }

    println!();
    Ok(())
}

/// Format a large number with comma separators.
fn format_count(n: u64) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}

/// Find a trueno benchmark binary.
/// Checks CARGO_TARGET_DIR, standard locations, and glob for bench deps.
fn find_bench_binary() -> Option<String> {
    // Check CARGO_TARGET_DIR first (user's zsh function sets this)
    let target_dir = std::env::var("CARGO_TARGET_DIR").unwrap_or_default();

    let mut candidates: Vec<String> = Vec::new();
    if !target_dir.is_empty() {
        candidates.push(format!("{target_dir}/release/examples/benchmark_matrix_suite"));
    }
    candidates.extend_from_slice(&[
        "/mnt/nvme-raid0/targets/trueno/release/examples/benchmark_matrix_suite".to_string(),
        "./target/release/examples/benchmark_matrix_suite".to_string(),
    ]);

    for path in &candidates {
        if std::path::Path::new(path).exists() {
            return Some(path.clone());
        }
    }

    // Try glob for bench binaries
    let glob_dirs = if !target_dir.is_empty() {
        vec![format!("{target_dir}/release/deps")]
    } else {
        vec![
            "/mnt/nvme-raid0/targets/trueno/release/deps".to_string(),
            "./target/release/deps".to_string(),
        ]
    };
    for dir in &glob_dirs {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                if name_str.starts_with("vector_ops-") && !name_str.contains('.') {
                    return Some(entry.path().display().to_string());
                }
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_events_defined() {
        assert!(!SIMD_PERF_EVENTS.is_empty());
        assert!(!AVX2_EVENTS.is_empty());
        assert!(!AVX512_EVENTS.is_empty());
    }

    #[test]
    fn test_invalid_arch_rejected() {
        let result = profile_simd("test_fn", 1024, "invalid_arch");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_perf_stat_csv() {
        let output = "1234567,,cycles,,,\n456789,,instructions,,,\n100,,cache-references,,,\n5,,cache-misses,,,\n";
        let result = parse_perf_stat_csv(output).unwrap();
        assert_eq!(*result.counters.get("cycles").unwrap(), 1234567);
        assert_eq!(*result.counters.get("instructions").unwrap(), 456789);
    }

    #[test]
    fn test_perf_stat_ipc() {
        let mut result = PerfStatResult::default();
        result.counters.insert("cycles".to_string(), 1000);
        result.counters.insert("instructions".to_string(), 2000);
        assert!((result.ipc() - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_perf_stat_cache_miss_rate() {
        let mut result = PerfStatResult::default();
        result.counters.insert("cache-references".to_string(), 1000);
        result.counters.insert("cache-misses".to_string(), 50);
        assert!((result.cache_miss_rate() - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_simd_utilization() {
        let mut result = PerfStatResult::default();
        result.counters.insert("fp_arith_inst_retired.scalar_single".to_string(), 100);
        result.counters.insert("fp_arith_inst_retired.256b_packed_single".to_string(), 900);
        let util = result.simd_utilization().unwrap();
        assert!((util - 90.0).abs() < 0.01);
    }

    #[test]
    fn test_format_count() {
        assert_eq!(format_count(0), "0");
        assert_eq!(format_count(999), "999");
        assert_eq!(format_count(1000), "1,000");
        assert_eq!(format_count(1234567), "1,234,567");
    }
}
