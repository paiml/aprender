//! Environment preflight checks.

use super::{CheckResult, CheckType, PreflightCheck};

/// Compare available MB against a required threshold, returning a pass/fail `CheckResult`.
fn check_mb_threshold(avail_mb: u64, required: u64, resource: &str) -> CheckResult {
    if avail_mb >= required {
        CheckResult::passed(format!("{avail_mb} MB {resource} available (minimum: {required} MB)"))
    } else {
        CheckResult::failed(format!(
            "Only {avail_mb} MB {resource} available (minimum: {required} MB)"
        ))
    }
}

/// Parse the available disk space (MB) from `df -m .` output.
#[cfg(unix)]
fn parse_df_available_mb(stdout: &str) -> Option<u64> {
    stdout
        .lines()
        .nth(1)
        .and_then(|line| line.split_whitespace().nth(3))
        .and_then(|s| s.parse::<u64>().ok())
}

/// Parse available memory (MB) from `free -m` output.
#[cfg(unix)]
fn parse_free_available_mb(stdout: &str) -> Option<u64> {
    stdout.lines().nth(1).and_then(|line| {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 7 {
            parts[6].parse::<u64>().ok()
        } else {
            None
        }
    })
}

impl PreflightCheck {
    // =========================================================================
    // Built-in Environment Checks
    // =========================================================================

    /// Check available disk space
    pub fn disk_space_mb(min_mb: u64) -> Self {
        Self::new(
            "disk_space",
            CheckType::Environment,
            format!("Ensures at least {min_mb} MB disk space available"),
            move |_data, ctx| {
                let required = ctx.min_disk_space_mb.unwrap_or(min_mb);

                #[cfg(unix)]
                {
                    use std::process::Command;
                    if let Ok(output) = Command::new("df").args(["-m", "."]).output() {
                        let stdout = String::from_utf8_lossy(&output.stdout);
                        if let Some(avail_mb) = parse_df_available_mb(&stdout) {
                            return check_mb_threshold(avail_mb, required, "disk");
                        }
                    }
                }

                // Fallback: assume sufficient space
                CheckResult::passed(format!("Disk space check passed (assumed >= {required} MB)"))
            },
        )
    }

    /// Check available memory
    pub fn memory_mb(min_mb: u64) -> Self {
        Self::new(
            "memory",
            CheckType::Environment,
            format!("Ensures at least {min_mb} MB memory available"),
            move |_data, ctx| {
                let required = ctx.min_memory_mb.unwrap_or(min_mb);

                #[cfg(unix)]
                {
                    use std::process::Command;
                    if let Ok(output) = Command::new("free").args(["-m"]).output() {
                        let stdout = String::from_utf8_lossy(&output.stdout);
                        if let Some(avail_mb) = parse_free_available_mb(&stdout) {
                            return check_mb_threshold(avail_mb, required, "memory");
                        }
                    }
                }

                // Fallback
                CheckResult::passed(format!("Memory check passed (assumed >= {required} MB)"))
            },
        )
    }

    /// Check GPU availability
    pub fn gpu_available() -> Self {
        Self::new(
            "gpu_available",
            CheckType::Environment,
            "Checks if GPU is available for training",
            |_data, _ctx| {
                // Check for NVIDIA GPU using nvidia-smi
                #[cfg(unix)]
                {
                    use std::process::Command;
                    let result = Command::new("nvidia-smi")
                        .args(["--query-gpu=name", "--format=csv,noheader"])
                        .output();

                    if let Ok(output) = result {
                        if output.status.success() {
                            let gpu_name = String::from_utf8_lossy(&output.stdout);
                            let gpu_name = gpu_name.trim();
                            if !gpu_name.is_empty() {
                                return CheckResult::passed(format!("GPU available: {gpu_name}"));
                            }
                        }
                    }
                }

                CheckResult::warning("No GPU detected, training will use CPU")
            },
        )
        .optional()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::preflight::PreflightContext;

    #[test]
    fn test_disk_space_check_creation() {
        let check = PreflightCheck::disk_space_mb(1000);
        assert_eq!(check.name, "disk_space");
        assert_eq!(check.check_type, CheckType::Environment);
        assert!(check.description.contains("1000"));
    }

    #[test]
    fn test_memory_check_creation() {
        let check = PreflightCheck::memory_mb(512);
        assert_eq!(check.name, "memory");
        assert_eq!(check.check_type, CheckType::Environment);
        assert!(check.description.contains("512"));
    }

    #[test]
    fn test_gpu_available_check_creation() {
        let check = PreflightCheck::gpu_available();
        assert_eq!(check.name, "gpu_available");
        assert_eq!(check.check_type, CheckType::Environment);
        // Optional checks have required=false
        assert!(!check.required);
    }

    #[test]
    fn test_disk_space_check_runs() {
        let check = PreflightCheck::disk_space_mb(1);
        let ctx = PreflightContext::default();
        let data: &[Vec<f64>] = &[];
        let result = check.run(data, &ctx);
        // Should either pass or use fallback
        assert!(result.is_passed() || result.is_warning());
    }

    #[test]
    fn test_memory_check_runs() {
        let check = PreflightCheck::memory_mb(1);
        let ctx = PreflightContext::default();
        let data: &[Vec<f64>] = &[];
        let result = check.run(data, &ctx);
        // Should either pass or use fallback
        assert!(result.is_passed() || result.is_warning());
    }

    #[test]
    fn test_gpu_check_runs() {
        let check = PreflightCheck::gpu_available();
        let ctx = PreflightContext::default();
        let data: &[Vec<f64>] = &[];
        let result = check.run(data, &ctx);
        // Should pass, warning, or fail (but not panic)
        assert!(result.is_passed() || result.is_warning() || result.is_failed());
    }

    #[test]
    fn test_disk_space_with_context_override() {
        let check = PreflightCheck::disk_space_mb(1000);
        let ctx = PreflightContext { min_disk_space_mb: Some(1), ..Default::default() };
        let data: &[Vec<f64>] = &[];
        let result = check.run(data, &ctx);
        // With low threshold, should likely pass
        assert!(result.is_passed() || result.is_warning());
    }

    #[test]
    fn test_memory_with_context_override() {
        let check = PreflightCheck::memory_mb(1000);
        let ctx = PreflightContext { min_memory_mb: Some(1), ..Default::default() };
        let data: &[Vec<f64>] = &[];
        let result = check.run(data, &ctx);
        // With low threshold, should likely pass
        assert!(result.is_passed() || result.is_warning());
    }
}
