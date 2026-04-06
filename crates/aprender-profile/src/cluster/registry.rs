use crate::cluster::{ArgsFilter, ClusterDefinition};
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Registry of user-defined syscall clusters loaded from TOML configuration
///
/// Implements Open-Closed Principle: extensible via configuration without recompilation.
///
/// # Example Usage
/// ```ignore
/// use renacer::cluster::{ClusterRegistry, FdTable};
///
/// let registry = ClusterRegistry::from_toml("clusters.toml")?;
/// if let Some(cluster) = registry.classify("mmap", &[], &FdTable::new()) {
///     println!("Cluster: {} (severity: {:?})", cluster.name, cluster.severity);
/// }
/// # Ok::<(), anyhow::Error>(())
/// ```
#[derive(Debug)]
pub struct ClusterRegistry {
    /// All defined clusters
    clusters: Vec<ClusterDefinition>,

    /// Fast lookup: syscall name → cluster name
    syscall_to_cluster: HashMap<String, String>,
}

impl ClusterRegistry {
    /// Load cluster definitions from TOML configuration file
    ///
    /// # Errors
    /// Returns error if file doesn't exist, has invalid TOML syntax, or contains
    /// duplicate syscall mappings.
    ///
    /// # Example TOML
    /// ```toml
    /// [[cluster]]
    /// name = "MemoryAllocation"
    /// description = "Heap management"
    /// syscalls = ["mmap", "munmap", "brk"]
    /// expected_for_transpiler = true
    /// anomaly_threshold = 0.50
    /// severity = "medium"
    /// ```
    pub fn from_toml<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = fs::read_to_string(path.as_ref()).with_context(|| {
            format!("Failed to read clusters file: {}", path.as_ref().display())
        })?;

        #[derive(serde::Deserialize)]
        struct ClusterFile {
            cluster: Vec<ClusterDefinition>,
        }

        let file: ClusterFile =
            toml::from_str(&content).with_context(|| "Failed to parse TOML cluster definitions")?;
        let definitions = file.cluster;

        // Build reverse index: syscall name → cluster name
        let mut syscall_to_cluster = HashMap::new();
        for cluster in &definitions {
            for syscall in &cluster.syscalls {
                if let Some(existing) =
                    syscall_to_cluster.insert(syscall.clone(), cluster.name.clone())
                {
                    anyhow::bail!(
                        "Duplicate syscall '{}' in clusters '{}' and '{}'",
                        syscall,
                        existing,
                        cluster.name
                    );
                }
            }
        }

        Ok(Self { clusters: definitions, syscall_to_cluster })
    }

    /// Load default cluster pack for single-shot compile workflows
    ///
    /// Uses embedded clusters-default.toml compiled into binary for zero-config operation.
    pub fn default_transpiler_clusters() -> Result<Self> {
        const DEFAULT_TOML: &str = include_str!("../../clusters-default.toml");

        #[derive(serde::Deserialize)]
        struct ClusterFile {
            cluster: Vec<ClusterDefinition>,
        }

        let file: ClusterFile = toml::from_str(DEFAULT_TOML)
            .context("Failed to parse embedded clusters-default.toml")?;
        let definitions = file.cluster;

        let mut syscall_to_cluster = HashMap::new();
        for cluster in &definitions {
            for syscall in &cluster.syscalls {
                syscall_to_cluster.insert(syscall.clone(), cluster.name.clone());
            }
        }

        Ok(Self { clusters: definitions, syscall_to_cluster })
    }

    /// Classify a syscall into its semantic cluster
    ///
    /// # Arguments
    /// * `syscall` - Syscall name (e.g., "mmap", "socket")
    /// * `args` - Syscall arguments as strings
    /// * `fds` - File descriptor table for path resolution (e.g., fd 3 → "/dev/nvidia0")
    ///
    /// # Returns
    /// * `Some(&ClusterDefinition)` if syscall matches a cluster (potentially after filtering)
    /// * `None` if no cluster matches
    ///
    /// # Poka-Yoke (Error Proofing)
    /// Callers should log warnings for unmatched syscalls and suggest TOML additions.
    pub fn classify<'a>(
        &'a self,
        syscall: &str,
        args: &[String],
        fds: &FdTable,
    ) -> Option<&'a ClusterDefinition> {
        // Fast path: lookup by syscall name
        let cluster_name = self.syscall_to_cluster.get(syscall)?;
        let cluster = self.clusters.iter().find(|c| &c.name == cluster_name)?;

        // Apply args filter if specified
        if let Some(filter) = &cluster.args_filter {
            if !Self::matches_filter(syscall, args, fds, filter) {
                return None;
            }
        }

        Some(cluster)
    }

    /// Apply argument-based filter for context-aware classification
    fn matches_filter(syscall: &str, args: &[String], fds: &FdTable, filter: &ArgsFilter) -> bool {
        // Filter by file descriptor path (e.g., /dev/nvidia*)
        if let Some(pattern) = &filter.fd_path_pattern {
            if syscall == "ioctl" {
                if let Some(fd_str) = args.first() {
                    if let Ok(fd) = fd_str.parse::<i32>() {
                        if let Some(path) = fds.get_path(fd) {
                            // Simplified pattern matching (use regex in production)
                            return path.contains(pattern.trim_end_matches('*'));
                        }
                    }
                }
            }
            return false;
        }

        // Filter by argument substring (e.g., "libtensorflow")
        if let Some(substring) = &filter.arg_contains {
            return args.iter().any(|arg| arg.contains(substring));
        }

        true
    }

    /// Get cluster definition by name
    pub fn get_cluster(&self, name: &str) -> Option<&ClusterDefinition> {
        self.clusters.iter().find(|c| c.name == name)
    }

    /// Get all defined clusters
    pub fn clusters(&self) -> &[ClusterDefinition] {
        &self.clusters
    }

    /// Simple classification without file descriptor tracking
    ///
    /// For use cases like time attribution where FD tracking isn't available.
    /// Returns cluster name or None if unclassified.
    pub fn classify_simple(&self, syscall: &str, _args: &[String]) -> Option<String> {
        // Simple lookup by syscall name without args filtering
        self.syscall_to_cluster.get(syscall).cloned()
    }
}

/// File descriptor table for mapping fd numbers to paths
///
/// In production, this would be populated from /proc/self/fd/* or ptrace data.
#[derive(Debug, Clone, Default)]
pub struct FdTable {
    table: HashMap<i32, String>,
}

impl FdTable {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, fd: i32, path: String) {
        self.table.insert(fd, path);
    }

    pub fn get_path(&self, fd: i32) -> Option<&str> {
        self.table.get(&fd).map(std::string::String::as_str)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn create_test_toml() -> Result<NamedTempFile> {
        let mut file = NamedTempFile::new()?;
        writeln!(
            file,
            r#"
[[cluster]]
name = "MemoryAllocation"
description = "Heap management"
syscalls = ["mmap", "munmap", "brk"]
expected_for_transpiler = true
anomaly_threshold = 0.50
severity = "medium"

[[cluster]]
name = "GPU"
description = "CUDA kernel launches"
syscalls = ["ioctl"]
expected_for_transpiler = false
anomaly_threshold = 0.0
severity = "medium"

[cluster.args_filter]
fd_path_pattern = "/dev/nvidia*"
"#
        )?;
        file.flush()?;
        Ok(file)
    }

    #[test]
    fn test_from_toml() -> Result<()> {
        let file = create_test_toml()?;
        let registry = ClusterRegistry::from_toml(file.path())?;

        assert_eq!(registry.clusters.len(), 2);
        assert!(registry.get_cluster("MemoryAllocation").is_some());
        assert!(registry.get_cluster("GPU").is_some());

        Ok(())
    }

    #[test]
    fn test_classify_simple() -> Result<()> {
        let file = create_test_toml()?;
        let registry = ClusterRegistry::from_toml(file.path())?;
        let fds = FdTable::new();

        // mmap → MemoryAllocation cluster
        let cluster = registry.classify("mmap", &[], &fds);
        assert!(cluster.is_some());
        assert_eq!(cluster.expect("test").name, "MemoryAllocation");

        // Unmatched syscall
        let cluster = registry.classify("socket", &[], &fds);
        assert!(cluster.is_none());

        Ok(())
    }

    #[test]
    fn test_classify_with_filter() -> Result<()> {
        let file = create_test_toml()?;
        let registry = ClusterRegistry::from_toml(file.path())?;

        // ioctl without GPU fd → no match
        let fds = FdTable::new();
        let cluster = registry.classify("ioctl", &["3".to_string()], &fds);
        assert!(cluster.is_none());

        // ioctl with GPU fd → GPU cluster
        let mut fds = FdTable::new();
        fds.insert(3, "/dev/nvidia0".to_string());
        let cluster = registry.classify("ioctl", &["3".to_string()], &fds);
        assert!(cluster.is_some());
        assert_eq!(cluster.expect("test").name, "GPU");

        Ok(())
    }

    #[test]
    fn test_duplicate_syscall_error() {
        let mut file = NamedTempFile::new().expect("test");
        writeln!(
            file,
            r#"
[[cluster]]
name = "ClusterA"
description = "Test cluster A"
syscalls = ["mmap"]
expected_for_transpiler = true
anomaly_threshold = 0.5
severity = "medium"

[[cluster]]
name = "ClusterB"
description = "Test cluster B"
syscalls = ["mmap"]
expected_for_transpiler = true
anomaly_threshold = 0.5
severity = "medium"
"#
        )
        .expect("test");
        file.flush().expect("test");

        let result = ClusterRegistry::from_toml(file.path());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Duplicate syscall"));
    }
}
