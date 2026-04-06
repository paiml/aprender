use serde::{Deserialize, Serialize};

/// Severity level for cluster anomalies
#[derive(Debug, Deserialize, Serialize, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    Low,
    Medium,
    High,
    Critical,
}

/// Filter for context-aware syscall classification
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ArgsFilter {
    /// Match file descriptor path pattern (e.g., "/dev/nvidia.*" for GPU calls)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fd_path_pattern: Option<String>,

    /// Match if any argument contains substring (e.g., "libtensorflow")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arg_contains: Option<String>,
}

/// User-defined syscall cluster loaded from TOML configuration
///
/// # Example TOML
/// ```toml
/// [[cluster]]
/// name = "GPU"
/// description = "CUDA/ROCm kernel launches"
/// syscalls = ["ioctl"]
/// args_filter = { fd_path_pattern = "/dev/nvidia.*" }
/// expected_for_transpiler = false
/// anomaly_threshold = 0.0
/// severity = "medium"
/// ```
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ClusterDefinition {
    /// Unique cluster name (e.g., "`MemoryAllocation`", "GPU")
    pub name: String,

    /// Human-readable description
    pub description: String,

    /// List of syscalls in this cluster (e.g., ["mmap", "munmap", "brk"])
    pub syscalls: Vec<String>,

    /// Whether this cluster is expected in single-shot compile workflows
    pub expected_for_transpiler: bool,

    /// Percentage change threshold before flagging anomaly (0.0-1.0)
    ///
    /// Examples:
    /// - 0.50 = 50% increase acceptable
    /// - 0.0 = ANY occurrence is RED FLAG (e.g., Networking)
    pub anomaly_threshold: f64,

    /// Severity level for anomalies in this cluster
    pub severity: Severity,

    /// Optional filter for context-aware classification
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub args_filter: Option<ArgsFilter>,
}

impl ClusterDefinition {
    /// Check if a syscall count/time change exceeds this cluster's threshold
    pub fn is_anomalous(&self, baseline_count: usize, current_count: usize) -> bool {
        if baseline_count == 0 {
            // New cluster appeared
            return current_count > 0;
        }

        let delta = current_count as f64 - baseline_count as f64;
        let pct_change = delta / baseline_count as f64;

        pct_change.abs() > self.anomaly_threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_anomalous_new_cluster() {
        let cluster = ClusterDefinition {
            name: "Networking".to_string(),
            description: "HTTP/network calls".to_string(),
            syscalls: vec!["socket".to_string()],
            expected_for_transpiler: false,
            anomaly_threshold: 0.0,
            severity: Severity::Critical,
            args_filter: None,
        };

        assert!(cluster.is_anomalous(0, 1)); // New cluster = anomaly
        assert!(!cluster.is_anomalous(0, 0)); // No change = not anomaly
    }

    #[test]
    fn test_is_anomalous_threshold() {
        let cluster = ClusterDefinition {
            name: "MemoryAllocation".to_string(),
            description: "Heap management".to_string(),
            syscalls: vec!["mmap".to_string()],
            expected_for_transpiler: true,
            anomaly_threshold: 0.50, // 50% increase acceptable
            severity: Severity::Medium,
            args_filter: None,
        };

        assert!(!cluster.is_anomalous(100, 140)); // +40% = under threshold
        assert!(cluster.is_anomalous(100, 160)); // +60% = exceeds threshold
        assert!(cluster.is_anomalous(100, 40)); // -60% = exceeds threshold (reduction)
    }

    #[test]
    fn test_severity_ordering() {
        assert!(Severity::Critical > Severity::High);
        assert!(Severity::High > Severity::Medium);
        assert!(Severity::Medium > Severity::Low);
    }
}
