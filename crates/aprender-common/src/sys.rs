//! System information utilities.
//!
//! Provides cross-platform detection for cgroups, CPU info, and other
//! system-level queries used across the Batuta stack.

/// Check if cgroup v1 or v2 CPU controllers are available.
///
/// Returns `true` on Linux when either `/sys/fs/cgroup/cpu` (v1) or
/// `/sys/fs/cgroup/unified` (v2) exists. Always returns `false` on non-Linux.
///
/// # Examples
/// ```
/// use batuta_common::sys::is_cgroup_available;
/// // Returns true on Linux with cgroup support
/// let _available = is_cgroup_available();
/// ```
#[must_use]
pub fn is_cgroup_available() -> bool {
    #[cfg(target_os = "linux")]
    {
        std::path::Path::new("/sys/fs/cgroup/cpu").exists()
            || std::path::Path::new("/sys/fs/cgroup/unified").exists()
    }

    #[cfg(not(target_os = "linux"))]
    {
        false
    }
}

/// Get CPU model name (best effort).
///
/// On Linux, reads `/proc/cpuinfo` for the "model name" field.
/// Returns `"Unknown CPU"` on other platforms or if parsing fails.
///
/// # Examples
/// ```
/// use batuta_common::sys::get_cpu_info;
/// let info = get_cpu_info();
/// assert!(!info.is_empty());
/// ```
#[must_use]
pub fn get_cpu_info() -> String {
    #[cfg(target_os = "linux")]
    {
        if let Ok(content) = std::fs::read_to_string("/proc/cpuinfo") {
            for line in content.lines() {
                if line.starts_with("model name")
                    && let Some(name) = line.split(':').nth(1)
                {
                    return name.trim().to_string();
                }
            }
        }
    }
    "Unknown CPU".to_string()
}

/// Get the number of available CPU cores.
///
/// Uses `std::thread::available_parallelism()`, falling back to 1.
#[must_use]
pub fn cpu_count() -> usize {
    std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_cgroup_available_returns_bool() {
        // Just verify it doesn't panic
        let _ = is_cgroup_available();
    }

    #[test]
    fn test_get_cpu_info_non_empty() {
        let info = get_cpu_info();
        assert!(!info.is_empty());
    }

    #[test]
    fn test_cpu_count_positive() {
        assert!(cpu_count() >= 1);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_cgroup_on_linux() {
        // On Linux, cgroup should typically be available
        // but we don't assert true since containers may not have it
        let _ = is_cgroup_available();
    }
}
