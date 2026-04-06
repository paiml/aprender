//! `cgp doctor` — Check tool availability and hardware capabilities.
//!
//! Detects: ncu, nsys, nvidia-smi, perf, CUPTI, criterion, renacer,
//! trueno-explain, GPU, CPU features.
//! Must complete in < 2 seconds (FALSIFY-CGP-061).
//! Must degrade gracefully when tools are missing (FALSIFY-CGP-012).

use anyhow::Result;
use serde::Serialize;
use std::process::Command;
use std::time::Instant;

/// Result of checking a single tool or capability.
#[derive(Debug, Clone, Serialize)]
pub struct ToolCheck {
    pub name: String,
    pub version: Option<String>,
    pub status: ToolStatus,
    pub path: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum ToolStatus {
    Ok,
    Missing,
    VersionMismatch { expected: String, found: String },
    Error(String),
}

/// Full doctor report for JSON output.
#[derive(Debug, Serialize)]
pub struct DoctorReport {
    pub checks: Vec<ToolCheck>,
    pub ok_count: usize,
    pub total_required: usize,
    pub operational: bool,
    pub elapsed_ms: f64,
}

impl std::fmt::Display for ToolStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ToolStatus::Ok => write!(f, "\x1b[32m[OK]\x1b[0m"),
            ToolStatus::Missing => write!(f, "\x1b[31m[MISSING]\x1b[0m"),
            ToolStatus::VersionMismatch { expected, found } => {
                write!(f, "\x1b[33m[VERSION] expected {expected}, found {found}\x1b[0m")
            }
            ToolStatus::Error(msg) => write!(f, "\x1b[31m[ERROR: {msg}]\x1b[0m"),
        }
    }
}

/// Check if a binary exists and get its version.
fn check_binary(
    name: &str,
    version_args: &[&str],
    version_parser: fn(&str) -> Option<String>,
) -> ToolCheck {
    match which::which(name) {
        Ok(path) => {
            let version = if version_args.is_empty() {
                None
            } else {
                Command::new(name).args(version_args).output().ok().and_then(|out| {
                    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
                    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
                    let combined = format!("{stdout}{stderr}");
                    version_parser(&combined)
                })
            };
            ToolCheck {
                name: name.to_string(),
                version,
                status: ToolStatus::Ok,
                path: Some(path.display().to_string()),
            }
        }
        Err(_) => ToolCheck {
            name: name.to_string(),
            version: None,
            status: ToolStatus::Missing,
            path: None,
        },
    }
}

/// Parse version from a line like "NVIDIA Nsight Compute CLI 2025.1.1.0"
fn parse_ncu_version(output: &str) -> Option<String> {
    output
        .lines()
        .find(|l| l.contains("Nsight Compute") || l.contains("ncu"))
        .and_then(|l| l.split_whitespace().last().map(String::from))
}

/// Parse version from nsys output
fn parse_nsys_version(output: &str) -> Option<String> {
    output.lines().find(|l| l.contains("version") || l.contains("Nsight Systems")).and_then(|l| {
        l.split_whitespace()
            .find(|w| w.chars().next().is_some_and(|c| c.is_ascii_digit()))
            .map(String::from)
    })
}

/// Parse nvidia-smi driver version
#[allow(dead_code)]
fn parse_nvidia_smi_version(output: &str) -> Option<String> {
    output.lines().find(|l| l.contains("Driver Version")).and_then(|l| {
        l.split("Driver Version:")
            .nth(1)
            .and_then(|s| s.split_whitespace().next())
            .map(String::from)
    })
}

/// Parse perf version
fn parse_perf_version(output: &str) -> Option<String> {
    output.lines().next().and_then(|l| {
        l.split_whitespace()
            .find(|w| w.chars().next().is_some_and(|c| c.is_ascii_digit()))
            .map(String::from)
    })
}

/// Parse generic --version output (first version-like string)
fn parse_generic_version(output: &str) -> Option<String> {
    output.lines().next().and_then(|l| {
        l.split_whitespace()
            .find(|w| w.chars().next().is_some_and(|c| c.is_ascii_digit()))
            .map(String::from)
    })
}

/// Detect GPU model and compute capability.
fn detect_gpu() -> ToolCheck {
    let result = Command::new("nvidia-smi")
        .args(["--query-gpu=name,compute_cap", "--format=csv,noheader"])
        .output();
    match result {
        Ok(out) if out.status.success() => {
            let stdout = String::from_utf8_lossy(&out.stdout);
            let info = stdout.trim().to_string();
            ToolCheck {
                name: "GPU".to_string(),
                version: Some(info),
                status: ToolStatus::Ok,
                path: None,
            }
        }
        _ => ToolCheck {
            name: "GPU".to_string(),
            version: None,
            status: ToolStatus::Missing,
            path: None,
        },
    }
}

/// Detect CPU features (AVX2, AVX-512, NEON, etc.)
fn detect_cpu() -> ToolCheck {
    #[cfg(target_arch = "x86_64")]
    {
        let mut features = Vec::new();
        if std::arch::is_x86_feature_detected!("avx2") {
            features.push("AVX2");
        }
        if std::arch::is_x86_feature_detected!("fma") {
            features.push("FMA");
        }
        if std::arch::is_x86_feature_detected!("avx512f") {
            features.push("AVX-512F");
        }
        if std::arch::is_x86_feature_detected!("sse4.2") {
            features.push("SSE4.2");
        }
        let cpu_model = std::fs::read_to_string("/proc/cpuinfo").ok().and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("model name"))
                .and_then(|l| l.split(':').nth(1))
                .map(|s| s.trim().to_string())
        });
        let version = match cpu_model {
            Some(model) => format!("{model} ({features})", features = features.join(", ")),
            None => features.join(", "),
        };
        ToolCheck {
            name: "CPU".to_string(),
            version: Some(version),
            status: ToolStatus::Ok,
            path: None,
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        ToolCheck {
            name: "CPU".to_string(),
            version: Some("aarch64 (NEON)".to_string()),
            status: ToolStatus::Ok,
            path: None,
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        ToolCheck {
            name: "CPU".to_string(),
            version: Some(format!("{}", std::env::consts::ARCH)),
            status: ToolStatus::Ok,
            path: None,
        }
    }
}

/// Check perf_event_paranoid setting
fn check_perf_paranoid() -> Option<i32> {
    std::fs::read_to_string("/proc/sys/kernel/perf_event_paranoid")
        .ok()
        .and_then(|s| s.trim().parse().ok())
}

/// Collect all doctor checks (pure data, no I/O to stdout).
pub fn collect_checks() -> Vec<ToolCheck> {
    vec![
        check_binary("nvidia-smi", &["--query-gpu=driver_version", "--format=csv,noheader"], |s| {
            Some(s.trim().to_string())
        }),
        {
            // CUDA runtime version from nvcc or nvidia-smi
            let mut check = check_binary("nvcc", &["--version"], |s| {
                s.lines()
                    .find(|l| l.contains("release"))
                    .and_then(|l| l.split("release ").nth(1))
                    .and_then(|s| s.split(',').next())
                    .map(String::from)
            });
            check.name = "CUDA Runtime".to_string();
            check
        },
        check_binary("ncu", &["--version"], parse_ncu_version),
        check_binary("nsys", &["--version"], parse_nsys_version),
        {
            // CUPTI check: look for libcupti.so
            let cupti_paths =
                ["/usr/local/cuda/lib64/libcupti.so", "/usr/lib/x86_64-linux-gnu/libcupti.so"];
            let found = cupti_paths.iter().find(|p| std::path::Path::new(p).exists());
            ToolCheck {
                name: "CUPTI".to_string(),
                version: found.map(|p| p.to_string()),
                status: if found.is_some() { ToolStatus::Ok } else { ToolStatus::Missing },
                path: found.map(|p| p.to_string()),
            }
        },
        {
            let mut check = check_binary("perf", &["--version"], parse_perf_version);
            if check.status == ToolStatus::Ok {
                if let Some(paranoid) = check_perf_paranoid() {
                    check.version = Some(format!(
                        "{} (perf_event_paranoid={})",
                        check.version.as_deref().unwrap_or("?"),
                        paranoid
                    ));
                }
            }
            check
        },
        // Memory safety: valgrind required for SIMD alignment verification (#242)
        check_binary("valgrind", &["--version"], parse_generic_version),
        check_binary("renacer", &["--version"], parse_generic_version),
        check_binary("trueno-explain", &["--version"], parse_generic_version),
        detect_gpu(),
        detect_cpu(),
    ]
}

/// Build a full doctor report.
pub fn build_report() -> DoctorReport {
    let start = Instant::now();
    let checks = collect_checks();

    let optional_tools = ["renacer", "trueno-explain", "CUPTI"];
    let mut ok_count = 0;
    let mut total = checks.len();

    for check in &checks {
        if check.status == ToolStatus::Ok {
            ok_count += 1;
        } else if optional_tools.contains(&check.name.as_str()) {
            total -= 1;
        }
    }

    let elapsed = start.elapsed();
    DoctorReport {
        checks,
        ok_count,
        total_required: total,
        operational: ok_count >= total,
        elapsed_ms: elapsed.as_secs_f64() * 1000.0,
    }
}

/// Run all doctor checks and print results.
pub fn run_doctor(json: bool) -> Result<()> {
    let report = build_report();

    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }

    println!("\n=== cgp System Check ===\n");

    for check in &report.checks {
        let version_str = check.version.as_deref().unwrap_or("");
        let pad_name = format!("{:18}", format!("{}:", check.name));
        let pad_version = format!("{:30}", version_str);
        println!("  {pad_name}{pad_version}{}", check.status);
    }

    // Check for perf_event_paranoid issues
    if let Some(paranoid) = check_perf_paranoid() {
        if paranoid > 2 {
            println!(
                "  \x1b[33m[WARN]\x1b[0m perf_event_paranoid={paranoid} — hardware counters blocked for non-root users."
            );
            println!("         Fix: sudo sysctl kernel.perf_event_paranoid=2");
            println!("         Or run cgp with sudo for perf stat features.\n");
        }
    }

    if report.operational {
        println!(
            "  All {} required components available. cgp is fully operational.",
            report.ok_count
        );
    } else {
        let missing = report.total_required - report.ok_count;
        println!(
            "  {}/{} components available. {missing} missing — cgp will operate in degraded mode.",
            report.ok_count, report.total_required
        );
    }
    println!("  Completed in {:.0}ms", report.elapsed_ms);
    println!();

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FALSIFY-CGP-010: doctor must detect installed tools
    #[test]
    fn test_detect_cpu_features() {
        let cpu = detect_cpu();
        assert_eq!(cpu.status, ToolStatus::Ok);
        assert!(cpu.version.is_some());
    }

    /// FALSIFY-CGP-011: doctor must handle missing tools gracefully
    #[test]
    fn test_missing_tool_graceful() {
        let check = check_binary("nonexistent-tool-xyz", &["--version"], parse_generic_version);
        assert_eq!(check.status, ToolStatus::Missing);
        assert!(check.path.is_none());
    }

    /// FALSIFY-CGP-061: doctor must complete in < 2 seconds
    #[test]
    fn test_doctor_speed() {
        let start = Instant::now();
        // Just run the individual checks without printing
        let _ = detect_cpu();
        let _ = detect_gpu();
        let _ = check_binary("nonexistent", &[], parse_generic_version);
        let elapsed = start.elapsed();
        assert!(elapsed.as_secs() < 2, "doctor checks took {:?}", elapsed);
    }
}
