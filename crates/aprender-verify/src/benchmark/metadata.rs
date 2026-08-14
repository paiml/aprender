/// System metadata collection for benchmark reproducibility
///
/// This module provides utilities to collect complete environmental metadata
/// for reproducible benchmarking.

use crate::benchmark::*;
use std::process::Command;
use sysinfo::System;

/// Collect complete benchmark metadata
pub fn collect_metadata(
    benchmark_suite: String,
    git_commit: Option<String>,
    git_branch: Option<String>,
) -> std::io::Result<BenchmarkMetadata> {
    Ok(BenchmarkMetadata {
        benchmark_suite,
        timestamp: chrono::Utc::now().to_rfc3339(),
        git_commit: git_commit.unwrap_or_else(get_git_commit),
        git_branch: git_branch.unwrap_or_else(get_git_branch),
        operator: "automated-ci".to_string(),
        hardware: collect_hardware_info(),
        software: collect_software_info()?,
        environment: collect_environment_config(),
    })
}

/// Collect hardware information using sysinfo
fn collect_hardware_info() -> HardwareInfo {
    let mut sys = System::new_all();
    sys.refresh_all();

    let cpu = if let Some(cpu) = sys.cpus().first() {
        CpuInfo {
            model: cpu.brand().to_string(),
            cores: sys.physical_core_count().unwrap_or(1),
            threads: sys.cpus().len(),
            frequency_mhz: cpu.frequency(),
            cache_l1: "Unknown".to_string(), // sysinfo doesn't provide cache info
            cache_l2: "Unknown".to_string(),
            cache_l3: "Unknown".to_string(),
        }
    } else {
        CpuInfo {
            model: "Unknown".to_string(),
            cores: 1,
            threads: 1,
            frequency_mhz: 0,
            cache_l1: "Unknown".to_string(),
            cache_l2: "Unknown".to_string(),
            cache_l3: "Unknown".to_string(),
        }
    };

    let memory = MemoryInfo {
        total_gb: (sys.total_memory() as f64) / 1_073_741_824.0, // Convert KB to GB
        memory_type: "Unknown".to_string(), // sysinfo doesn't provide memory type
        frequency_mhz: None,
    };

    HardwareInfo {
        cpu,
        memory,
        storage: None, // Optional, would require additional system introspection
    }
}

/// Collect software environment information
fn collect_software_info() -> std::io::Result<SoftwareInfo> {
    Ok(SoftwareInfo {
        os: get_os_info(),
        kernel: get_kernel_version(),
        rustc: get_rustc_version(),
        cargo: get_cargo_version(),
        llvm: get_llvm_version(),
    })
}

/// Collect environment configuration
fn collect_environment_config() -> EnvironmentConfig {
    EnvironmentConfig {
        cpu_governor: get_cpu_governor(),
        turbo_boost: "unknown".to_string(),
        swap: get_swap_status(),
        isolation: None,
    }
}

/// Get git commit hash
fn get_git_commit() -> String {
    Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|output| {
            if output.status.success() {
                String::from_utf8(output.stdout).ok().map(|s| s.trim().to_string())
            } else {
                None
            }
        })
        .unwrap_or_else(|| "unknown".to_string())
}

/// Get git branch name
fn get_git_branch() -> String {
    Command::new("git")
        .args(["rev-parse", "--abbrev-ref", "HEAD"])
        .output()
        .ok()
        .and_then(|output| {
            if output.status.success() {
                String::from_utf8(output.stdout).ok().map(|s| s.trim().to_string())
            } else {
                None
            }
        })
        .unwrap_or_else(|| "unknown".to_string())
}

/// Get OS information
fn get_os_info() -> String {
    let sys = System::new_all();
    format!(
        "{} {}",
        sys.name().unwrap_or_else(|| "Unknown".to_string()),
        sys.os_version().unwrap_or_else(|| "Unknown".to_string())
    )
}

/// Get kernel version
fn get_kernel_version() -> String {
    let sys = System::new_all();
    sys.kernel_version().unwrap_or_else(|| "Unknown".to_string())
}

/// Get rustc version
fn get_rustc_version() -> String {
    Command::new("rustc")
        .arg("--version")
        .output()
        .ok()
        .and_then(|output| {
            if output.status.success() {
                String::from_utf8(output.stdout).ok().map(|s| s.trim().to_string())
            } else {
                None
            }
        })
        .unwrap_or_else(|| "unknown".to_string())
}

/// Get cargo version
fn get_cargo_version() -> String {
    Command::new("cargo")
        .arg("--version")
        .output()
        .ok()
        .and_then(|output| {
            if output.status.success() {
                String::from_utf8(output.stdout).ok().map(|s| s.trim().to_string())
            } else {
                None
            }
        })
        .unwrap_or_else(|| "unknown".to_string())
}

/// Get LLVM version (extracted from rustc --version --verbose)
fn get_llvm_version() -> String {
    Command::new("rustc")
        .args(["--version", "--verbose"])
        .output()
        .ok()
        .and_then(|output| {
            if output.status.success() {
                String::from_utf8(output.stdout)
                    .ok()
                    .and_then(|s| {
                        s.lines()
                            .find(|line| line.starts_with("LLVM version:"))
                            .map(|line| line.replace("LLVM version:", "").trim().to_string())
                    })
            } else {
                None
            }
        })
        .unwrap_or_else(|| "unknown".to_string())
}

/// Get CPU governor setting (Linux-specific)
fn get_cpu_governor() -> String {
    #[cfg(target_os = "linux")]
    {
        std::fs::read_to_string("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
            .ok()
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| "unknown".to_string())
    }

    #[cfg(not(target_os = "linux"))]
    {
        "not-applicable".to_string()
    }
}

/// Get swap status
fn get_swap_status() -> String {
    let sys = System::new_all();
    if sys.total_swap() > 0 {
        "enabled".to_string()
    } else {
        "disabled".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collect_metadata() {
        let metadata = collect_metadata(
            "test-suite".to_string(),
            Some("abc123".to_string()),
            Some("main".to_string()),
        );

        assert!(metadata.is_ok());
        let metadata = metadata.unwrap();

        assert_eq!(metadata.benchmark_suite, "test-suite");
        assert_eq!(metadata.git_commit, "abc123");
        assert_eq!(metadata.git_branch, "main");
        assert!(!metadata.timestamp.is_empty());

        // Verify hardware info is collected
        assert!(!metadata.hardware.cpu.model.is_empty());
        assert!(metadata.hardware.cpu.cores > 0);
        assert!(metadata.hardware.memory.total_gb > 0.0);

        // Verify software info is collected
        assert!(!metadata.software.os.is_empty());
        assert!(!metadata.software.kernel.is_empty());
        assert!(!metadata.software.rustc.is_empty());
        assert!(!metadata.software.cargo.is_empty());
    }

    #[test]
    fn test_get_git_commit() {
        let commit = get_git_commit();
        // Should return either a valid commit hash or "unknown"
        assert!(!commit.is_empty());
    }

    #[test]
    fn test_get_git_branch() {
        let branch = get_git_branch();
        // Should return either a valid branch name or "unknown"
        assert!(!branch.is_empty());
    }
}
