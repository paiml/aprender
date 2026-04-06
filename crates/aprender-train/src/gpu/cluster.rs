//! Cluster configuration for multi-node GPU training (GPU-SHARE Phase 3, §3.2).
//!
//! Parses `cluster.yaml` files describing heterogeneous training clusters
//! with mixed GPU types (RTX 4090, Jetson, CPU-only nodes).
//!
//! # Example
//!
//! ```yaml
//! nodes:
//!   - name: desktop
//!     host: localhost
//!     gpus:
//!       - uuid: GPU-abcd-1234
//!         type: rtx-4090
//!         vram_mb: 24564
//!         memory_type: discrete
//!     max_adapters: 3
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::Path;

/// Top-level cluster configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterConfig {
    /// Cluster nodes (at least one required).
    pub nodes: Vec<NodeConfig>,
}

/// Configuration for a single training node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeConfig {
    /// Human-readable node name (must be unique within cluster).
    pub name: String,
    /// Hostname or IP address.
    pub host: String,
    /// Transport method for remote nodes.
    #[serde(default)]
    pub transport: Transport,
    /// SSH user for remote nodes (defaults to current user).
    #[serde(default)]
    pub user: Option<String>,
    /// GPUs available on this node (empty = CPU-only).
    #[serde(default)]
    pub gpus: Vec<GpuConfig>,
    /// Maximum number of concurrent adapters on this node.
    #[serde(default = "default_max_adapters")]
    pub max_adapters: usize,
    /// CPU cores available (for CPU-only nodes).
    #[serde(default)]
    pub cpu_cores: Option<u32>,
    /// RAM in MB (for CPU-only nodes).
    #[serde(default)]
    pub ram_mb: Option<u64>,
}

/// Transport method for connecting to a node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[derive(Default)]
pub enum Transport {
    /// Local node (no transport needed).
    #[default]
    Local,
    /// SSH transport via forjar.
    Ssh,
}

/// Configuration for a single GPU on a node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuConfig {
    /// GPU UUID from nvidia-smi (e.g., GPU-abcd-1234).
    pub uuid: String,
    /// GPU type identifier (e.g., rtx-4090, jetson-orin).
    #[serde(rename = "type")]
    pub gpu_type: String,
    /// Total VRAM in MB.
    pub vram_mb: u64,
    /// Memory architecture (affects reserve factor).
    #[serde(default)]
    pub memory_type: MemoryType,
}

/// GPU memory architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[derive(Default)]
pub enum MemoryType {
    /// Discrete GPU memory (85% reserve factor).
    #[default]
    Discrete,
    /// Unified memory shared with CPU (60% reserve factor).
    Unified,
}

impl MemoryType {
    /// Reserve factor: fraction of VRAM usable for training.
    #[must_use]
    pub fn reserve_factor(self) -> f32 {
        match self {
            Self::Discrete => 0.85,
            Self::Unified => 0.60,
        }
    }
}

fn default_max_adapters() -> usize {
    1
}

/// Cluster configuration validation errors.
#[derive(Debug, thiserror::Error)]
pub enum ClusterValidationError {
    #[error("cluster must have at least one node")]
    NoNodes,
    #[error("duplicate node name: {0}")]
    DuplicateNodeName(String),
    #[error("node '{name}': max_adapters must be >= 1")]
    ZeroMaxAdapters { name: String },
    #[error("node '{node}': GPU '{uuid}' has zero VRAM")]
    ZeroVram { node: String, uuid: String },
    #[error("node '{node}': duplicate GPU UUID '{uuid}'")]
    DuplicateGpuUuid { node: String, uuid: String },
    #[error("node '{node}': SSH transport requires a host other than localhost")]
    SshLocalhost { node: String },
}

impl ClusterConfig {
    /// Load cluster config from a YAML file.
    ///
    /// # Errors
    /// Returns error if file cannot be read or parsed, or if validation fails.
    pub fn from_file(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let contents = std::fs::read_to_string(path)?;
        let config: Self = serde_yaml::from_str(&contents)?;
        config.validate()?;
        Ok(config)
    }

    /// Parse cluster config from a YAML string.
    ///
    /// # Errors
    /// Returns error if parsing or validation fails.
    pub fn from_yaml(yaml: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let config: Self = serde_yaml::from_str(yaml)?;
        config.validate()?;
        Ok(config)
    }

    /// Validate cluster configuration.
    ///
    /// # Errors
    /// Returns the first validation error found.
    pub fn validate(&self) -> Result<(), ClusterValidationError> {
        if self.nodes.is_empty() {
            return Err(ClusterValidationError::NoNodes);
        }

        let mut names = HashSet::new();
        for node in &self.nodes {
            if !names.insert(&node.name) {
                return Err(ClusterValidationError::DuplicateNodeName(node.name.clone()));
            }
            if node.max_adapters == 0 {
                return Err(ClusterValidationError::ZeroMaxAdapters { name: node.name.clone() });
            }
            if node.transport == Transport::Ssh
                && (node.host == "localhost" || node.host == "127.0.0.1")
            {
                return Err(ClusterValidationError::SshLocalhost { node: node.name.clone() });
            }
            validate_node_gpus(node)?;
        }
        Ok(())
    }

    /// Total number of adapters the cluster can train concurrently.
    #[must_use]
    pub fn total_adapter_capacity(&self) -> usize {
        self.nodes.iter().map(|n| n.max_adapters).sum()
    }

    /// Find a node by name.
    #[must_use]
    pub fn find_node(&self, name: &str) -> Option<&NodeConfig> {
        self.nodes.iter().find(|n| n.name == name)
    }
}

fn validate_node_gpus(node: &NodeConfig) -> Result<(), ClusterValidationError> {
    let mut gpu_uuids = HashSet::new();
    for gpu in &node.gpus {
        if gpu.vram_mb == 0 {
            return Err(ClusterValidationError::ZeroVram {
                node: node.name.clone(),
                uuid: gpu.uuid.clone(),
            });
        }
        if !gpu_uuids.insert(&gpu.uuid) {
            return Err(ClusterValidationError::DuplicateGpuUuid {
                node: node.name.clone(),
                uuid: gpu.uuid.clone(),
            });
        }
    }
    Ok(())
}

impl NodeConfig {
    /// Total VRAM across all GPUs on this node (in MB).
    #[must_use]
    pub fn total_vram_mb(&self) -> u64 {
        self.gpus.iter().map(|g| g.vram_mb).sum()
    }

    /// Usable VRAM (total × reserve_factor) across all GPUs.
    #[must_use]
    pub fn usable_vram_mb(&self) -> u64 {
        self.gpus
            .iter()
            .map(|g| (g.vram_mb as f64 * f64::from(g.memory_type.reserve_factor())) as u64)
            .sum()
    }

    /// Whether this node is local (no transport needed).
    #[must_use]
    pub fn is_local(&self) -> bool {
        self.transport == Transport::Local
    }

    /// Whether this is a CPU-only node (no GPUs).
    #[must_use]
    pub fn is_cpu_only(&self) -> bool {
        self.gpus.is_empty()
    }
}

impl GpuConfig {
    /// Usable VRAM after applying reserve factor.
    #[must_use]
    pub fn usable_vram_mb(&self) -> u64 {
        (self.vram_mb as f64 * f64::from(self.memory_type.reserve_factor())) as u64
    }
}

impl std::fmt::Display for ClusterConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "Cluster: {} node(s), {} adapter slots",
            self.nodes.len(),
            self.total_adapter_capacity()
        )?;
        for node in &self.nodes {
            write!(f, "  {}: {} ({})", node.name, node.host, node.transport)?;
            if node.gpus.is_empty() {
                write!(f, " [CPU-only]")?;
            } else {
                for gpu in &node.gpus {
                    write!(f, " [{} {} MB {:?}]", gpu.gpu_type, gpu.vram_mb, gpu.memory_type)?;
                }
            }
            writeln!(f, " max_adapters={}", node.max_adapters)?;
        }
        Ok(())
    }
}

impl std::fmt::Display for Transport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Local => write!(f, "local"),
            Self::Ssh => write!(f, "ssh"),
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    fn sample_yaml() -> &'static str {
        r"
nodes:
  - name: desktop
    host: localhost
    gpus:
      - uuid: GPU-abcd-1234
        type: rtx-4090
        vram_mb: 24564
        memory_type: discrete
    max_adapters: 3

  - name: jetson
    host: jetson.local
    transport: ssh
    gpus:
      - uuid: GPU-efgh-5678
        type: jetson-orin
        vram_mb: 8192
        memory_type: unified
    max_adapters: 1

  - name: intel-box
    host: 10.0.0.5
    transport: ssh
    user: noah
    gpus: []
    cpu_cores: 16
    ram_mb: 65536
    max_adapters: 1
"
    }

    #[test]
    fn test_parse_cluster_yaml() {
        let config = ClusterConfig::from_yaml(sample_yaml()).unwrap();
        assert_eq!(config.nodes.len(), 3);

        let desktop = &config.nodes[0];
        assert_eq!(desktop.name, "desktop");
        assert_eq!(desktop.host, "localhost");
        assert_eq!(desktop.transport, Transport::Local);
        assert_eq!(desktop.gpus.len(), 1);
        assert_eq!(desktop.gpus[0].uuid, "GPU-abcd-1234");
        assert_eq!(desktop.gpus[0].gpu_type, "rtx-4090");
        assert_eq!(desktop.gpus[0].vram_mb, 24564);
        assert_eq!(desktop.gpus[0].memory_type, MemoryType::Discrete);
        assert_eq!(desktop.max_adapters, 3);

        let jetson = &config.nodes[1];
        assert_eq!(jetson.transport, Transport::Ssh);
        assert_eq!(jetson.gpus[0].memory_type, MemoryType::Unified);

        let intel = &config.nodes[2];
        assert!(intel.is_cpu_only());
        assert_eq!(intel.user, Some("noah".to_string()));
        assert_eq!(intel.cpu_cores, Some(16));
        assert_eq!(intel.ram_mb, Some(65536));
    }

    #[test]
    fn test_total_adapter_capacity() {
        let config = ClusterConfig::from_yaml(sample_yaml()).unwrap();
        assert_eq!(config.total_adapter_capacity(), 5); // 3 + 1 + 1
    }

    #[test]
    fn test_node_vram_calculations() {
        let config = ClusterConfig::from_yaml(sample_yaml()).unwrap();
        let desktop = &config.nodes[0];
        assert_eq!(desktop.total_vram_mb(), 24564);
        // 24564 * 0.85 = 20879.4 → 20879
        assert_eq!(desktop.usable_vram_mb(), 20879);

        let jetson = &config.nodes[1];
        assert_eq!(jetson.total_vram_mb(), 8192);
        // 8192 * 0.60 = 4915.2 → 4915
        assert_eq!(jetson.usable_vram_mb(), 4915);
    }

    #[test]
    fn test_gpu_usable_vram() {
        let gpu = GpuConfig {
            uuid: "GPU-test".to_string(),
            gpu_type: "rtx-4090".to_string(),
            vram_mb: 24000,
            memory_type: MemoryType::Discrete,
        };
        assert_eq!(gpu.usable_vram_mb(), 20400); // 24000 * 0.85
    }

    #[test]
    fn test_find_node() {
        let config = ClusterConfig::from_yaml(sample_yaml()).unwrap();
        assert!(config.find_node("desktop").is_some());
        assert!(config.find_node("jetson").is_some());
        assert!(config.find_node("nonexistent").is_none());
    }

    #[test]
    fn test_validation_no_nodes() {
        let yaml = "nodes: []";
        let result = ClusterConfig::from_yaml(yaml);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("at least one node"));
    }

    #[test]
    fn test_validation_duplicate_names() {
        let yaml = r"
nodes:
  - name: box1
    host: localhost
    max_adapters: 1
  - name: box1
    host: 10.0.0.2
    transport: ssh
    max_adapters: 1
";
        let result = ClusterConfig::from_yaml(yaml);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("duplicate node name"));
    }

    #[test]
    fn test_validation_zero_max_adapters() {
        let yaml = r"
nodes:
  - name: bad
    host: localhost
    max_adapters: 0
";
        let result = ClusterConfig::from_yaml(yaml);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("max_adapters"));
    }

    #[test]
    fn test_validation_zero_vram() {
        let yaml = r"
nodes:
  - name: bad
    host: localhost
    gpus:
      - uuid: GPU-bad
        type: unknown
        vram_mb: 0
    max_adapters: 1
";
        let result = ClusterConfig::from_yaml(yaml);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("zero VRAM"));
    }

    #[test]
    fn test_validation_duplicate_gpu_uuid() {
        let yaml = r"
nodes:
  - name: dupes
    host: localhost
    gpus:
      - uuid: GPU-same
        type: rtx-4090
        vram_mb: 24000
      - uuid: GPU-same
        type: rtx-4090
        vram_mb: 24000
    max_adapters: 2
";
        let result = ClusterConfig::from_yaml(yaml);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("duplicate GPU UUID"));
    }

    #[test]
    fn test_validation_ssh_localhost() {
        let yaml = r"
nodes:
  - name: bad-ssh
    host: localhost
    transport: ssh
    max_adapters: 1
";
        let result = ClusterConfig::from_yaml(yaml);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("SSH transport"));
    }

    #[test]
    fn test_display() {
        let config = ClusterConfig::from_yaml(sample_yaml()).unwrap();
        let display = format!("{config}");
        assert!(display.contains("3 node(s)"));
        assert!(display.contains("5 adapter slots"));
        assert!(display.contains("desktop"));
        assert!(display.contains("rtx-4090"));
        assert!(display.contains("CPU-only"));
    }

    #[test]
    fn test_reserve_factor() {
        assert!((MemoryType::Discrete.reserve_factor() - 0.85).abs() < f32::EPSILON);
        assert!((MemoryType::Unified.reserve_factor() - 0.60).abs() < f32::EPSILON);
    }

    #[test]
    fn test_minimal_config() {
        let yaml = r"
nodes:
  - name: single
    host: localhost
";
        let config = ClusterConfig::from_yaml(yaml).unwrap();
        assert_eq!(config.nodes.len(), 1);
        assert_eq!(config.nodes[0].max_adapters, 1); // default
        assert!(config.nodes[0].gpus.is_empty()); // default
    }

    #[test]
    fn test_serialization_roundtrip() {
        let config = ClusterConfig::from_yaml(sample_yaml()).unwrap();
        let yaml = serde_yaml::to_string(&config).unwrap();
        let reparsed = ClusterConfig::from_yaml(&yaml).unwrap();
        assert_eq!(reparsed.nodes.len(), config.nodes.len());
        assert_eq!(reparsed.total_adapter_capacity(), config.total_adapter_capacity());
    }
}

/// GPU dispatch cost_model (PW-01: 5× PCIe Rule)
///
/// Determines when GPU dispatch is beneficial based on compute-to-transfer
/// ratio. The crossover point (dispatch_threshold) is 5× the PCIe transfer cost.
pub struct GpuCostModel {
    /// PCIe transfer cost per MB (microseconds)
    pub pcie_cost_per_mb: f64,
    /// GPU compute cost per MFLOP (microseconds)
    pub gpu_compute_per_mflop: f64,
    /// Dispatch threshold multiplier (default: 5×)
    pub dispatch_threshold: f64,
}

impl Default for GpuCostModel {
    fn default() -> Self {
        Self {
            pcie_cost_per_mb: 40.0,      // PCIe 4.0 ~25 GB/s → ~40 µs/MB
            gpu_compute_per_mflop: 0.01, // RTX 4090 ~80 TFLOPS → ~0.01 µs/MFLOP
            dispatch_threshold: 5.0,     // 5× PCIe rule
        }
    }
}

impl GpuCostModel {
    /// Check if GPU dispatch is beneficial for the given workload.
    ///
    /// Returns true when compute time > dispatch_threshold × transfer time (crossover).
    pub fn should_dispatch_gpu(&self, data_mb: f64, compute_mflops: f64) -> bool {
        let transfer_cost = data_mb * self.pcie_cost_per_mb;
        let compute_cost = compute_mflops * self.gpu_compute_per_mflop;
        compute_cost > self.dispatch_threshold * transfer_cost
    }
}

#[cfg(test)]
mod cost_model_tests {
    use super::*;

    /// cost_test: small workloads stay on CPU (PW-13 prediction_accuracy)
    #[test]
    fn cost_test_small_workload_stays_cpu() {
        let model = GpuCostModel::default();
        // 1 MB data, 100 MFLOPS → transfer dominates, prediction_accuracy: CPU
        assert!(!model.should_dispatch_gpu(1.0, 100.0));
    }

    /// cost_test: large workloads go to GPU (PW-13 prediction_accuracy)
    #[test]
    fn cost_test_large_workload_goes_gpu() {
        let model = GpuCostModel::default();
        // 1 MB data, 1_000_000 MFLOPS → compute dominates, prediction_accuracy: GPU
        assert!(model.should_dispatch_gpu(1.0, 1_000_000.0));
    }
}
