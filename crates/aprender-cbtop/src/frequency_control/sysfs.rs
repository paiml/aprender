use std::path::PathBuf;

/// Read and trim a sysfs file. Returns None on I/O error.
fn read_sysfs_trimmed(path: &std::path::Path) -> Option<String> {
    std::fs::read_to_string(path).ok().map(|s| s.trim().to_string())
}

/// Read u64 value from sysfs
pub(crate) fn read_sysfs_value(path: &std::path::Path) -> Option<u64> {
    read_sysfs_trimmed(path).and_then(|s| s.parse().ok())
}

/// Read string from sysfs
pub(crate) fn read_sysfs_string(path: &std::path::Path) -> Option<String> {
    read_sysfs_trimmed(path)
}

/// Build sysfs governor path for a given CPU.
pub(crate) fn governor_path(sysfs_path: &std::path::Path, cpu_id: usize) -> PathBuf {
    sysfs_path.join(format!("cpu{cpu_id}/cpufreq/scaling_governor"))
}
