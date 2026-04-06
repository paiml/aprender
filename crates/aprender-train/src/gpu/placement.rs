//! Job placement algorithm for multi-node adapter training (GPU-SHARE Phase 3, §3.3).
//!
//! Scores each node for each adapter job and assigns greedily:
//!
//! ```text
//! score = (free_vram / adapter_budget) × gpu_flops_factor × (1 / current_load)
//! ```
//!
//! Where `gpu_flops_factor` normalizes different GPU types:
//! - RTX 4090: 1.0 (reference)
//! - Jetson Orin: 0.06 (8 SMs vs 128)
//! - CPU (Intel): 0.01

use super::cluster::{ClusterConfig, NodeConfig};

/// FLOPS factor for known GPU types, normalized to RTX 4090.
fn gpu_flops_factor(gpu_type: &str) -> f64 {
    match gpu_type.to_lowercase().as_str() {
        "rtx-4090" | "rtx4090" | "geforce-rtx-4090" => 1.0,
        "rtx-4080" | "rtx4080" => 0.72,
        "rtx-3090" | "rtx3090" => 0.55,
        "rtx-3080" | "rtx3080" => 0.45,
        "a100" | "a100-80gb" | "a100-40gb" => 1.2,
        "h100" | "h100-80gb" => 2.0,
        "jetson-orin" | "orin" => 0.06,
        "jetson-nano" | "nano" => 0.02,
        _ => 0.5, // unknown GPU, conservative estimate
    }
}

/// A pending adapter job to place on a cluster node.
#[derive(Debug, Clone)]
pub struct AdapterJob {
    /// Index of this adapter in the adapters config.
    pub adapter_idx: usize,
    /// Estimated VRAM budget in MB.
    pub budget_mb: u64,
    /// Human-readable label.
    pub label: String,
}

/// Result of placing an adapter on a node.
#[derive(Debug, Clone)]
pub struct PlacementDecision {
    /// Adapter index.
    pub adapter_idx: usize,
    /// Node name assigned.
    pub node_name: String,
    /// Placement score (higher = better fit).
    pub score: f64,
}

/// Current load state of a node for placement scoring.
#[derive(Debug, Clone, Default)]
pub struct NodeLoad {
    /// Number of adapters currently assigned.
    pub active_adapters: usize,
    /// VRAM already reserved (MB).
    pub reserved_vram_mb: u64,
}

/// Place adapter jobs across cluster nodes greedily.
///
/// Each adapter is assigned to the highest-scoring eligible node.
/// Nodes are updated with load after each assignment.
///
/// # Returns
/// A vector of placement decisions. Unplaceable adapters are omitted.
pub fn place_adapters(
    cluster: &ClusterConfig,
    jobs: &[AdapterJob],
    initial_load: &[NodeLoad],
) -> Vec<PlacementDecision> {
    let mut loads: Vec<NodeLoad> = cluster
        .nodes
        .iter()
        .enumerate()
        .map(|(i, _)| initial_load.get(i).cloned().unwrap_or_default())
        .collect();

    let mut placements = Vec::new();

    for job in jobs {
        let best = find_best_node(cluster, job, &loads);
        if let Some((node_idx, score)) = best {
            let node = &cluster.nodes[node_idx];
            placements.push(PlacementDecision {
                adapter_idx: job.adapter_idx,
                node_name: node.name.clone(),
                score,
            });
            loads[node_idx].active_adapters += 1;
            loads[node_idx].reserved_vram_mb += job.budget_mb;
        }
    }

    placements
}

fn find_best_node(
    cluster: &ClusterConfig,
    job: &AdapterJob,
    loads: &[NodeLoad],
) -> Option<(usize, f64)> {
    let mut best: Option<(usize, f64)> = None;

    for (i, node) in cluster.nodes.iter().enumerate() {
        let load = &loads[i];

        // Check adapter capacity
        if load.active_adapters >= node.max_adapters {
            continue;
        }

        let score = score_node(node, job.budget_mb, load);
        if score <= 0.0 {
            continue;
        }

        match best {
            None => best = Some((i, score)),
            Some((_, best_score)) if score > best_score => best = Some((i, score)),
            _ => {}
        }
    }

    best
}

/// Score a node for a given adapter budget.
///
/// `score = (free_vram / adapter_budget) × gpu_flops_factor × (1 / current_load)`
///
/// Returns 0.0 if the node cannot fit the adapter.
pub fn score_node(node: &NodeConfig, budget_mb: u64, load: &NodeLoad) -> f64 {
    if budget_mb == 0 {
        return 0.0;
    }

    let free_vram = free_vram_mb(node, load);
    if free_vram < budget_mb {
        return 0.0;
    }

    let vram_ratio = free_vram as f64 / budget_mb as f64;
    let flops = node_flops_factor(node);
    let load_factor = 1.0 / (1.0 + load.active_adapters as f64);

    vram_ratio * flops * load_factor
}

/// Available VRAM on a node after accounting for reserves and current load.
fn free_vram_mb(node: &NodeConfig, load: &NodeLoad) -> u64 {
    let usable = node.usable_vram_mb();
    usable.saturating_sub(load.reserved_vram_mb)
}

/// Aggregate FLOPS factor for a node (max GPU if multi-GPU).
fn node_flops_factor(node: &NodeConfig) -> f64 {
    if node.gpus.is_empty() {
        return 0.01; // CPU-only
    }
    node.gpus.iter().map(|g| gpu_flops_factor(&g.gpu_type)).fold(0.0_f64, f64::max)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use crate::gpu::cluster::ClusterConfig;

    fn test_cluster() -> ClusterConfig {
        ClusterConfig::from_yaml(
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
",
        )
        .unwrap()
    }

    #[test]
    fn test_gpu_flops_known_types() {
        assert!((gpu_flops_factor("rtx-4090") - 1.0).abs() < f64::EPSILON);
        assert!((gpu_flops_factor("jetson-orin") - 0.06).abs() < f64::EPSILON);
        assert!((gpu_flops_factor("h100") - 2.0).abs() < f64::EPSILON);
        assert!((gpu_flops_factor("unknown-gpu") - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_score_node_desktop() {
        let cluster = test_cluster();
        let desktop = &cluster.nodes[0];
        let load = NodeLoad::default();
        // usable = 24564 * 0.85 = 20879
        // score = (20879 / 8000) * 1.0 * (1 / 1) = 2.609
        let score = score_node(desktop, 8000, &load);
        assert!(score > 2.5);
        assert!(score < 2.7);
    }

    #[test]
    fn test_score_node_insufficient_vram() {
        let cluster = test_cluster();
        let desktop = &cluster.nodes[0];
        let load = NodeLoad::default();
        // Request more VRAM than usable
        let score = score_node(desktop, 25000, &load);
        assert!((score - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_score_node_with_load() {
        let cluster = test_cluster();
        let desktop = &cluster.nodes[0];
        let load = NodeLoad { active_adapters: 1, reserved_vram_mb: 8000 };
        // free = 20879 - 8000 = 12879
        // score = (12879 / 8000) * 1.0 * (1 / 2) = 0.804
        let score = score_node(desktop, 8000, &load);
        assert!(score > 0.7);
        assert!(score < 0.9);
    }

    #[test]
    fn test_score_cpu_only_node() {
        let cluster = test_cluster();
        let intel = &cluster.nodes[2];
        let load = NodeLoad::default();
        // CPU-only: 0 VRAM, budget > 0 → score = 0
        let score = score_node(intel, 8000, &load);
        assert!((score - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_place_single_adapter() {
        let cluster = test_cluster();
        let jobs =
            vec![AdapterJob { adapter_idx: 0, budget_mb: 8000, label: "adapter-0".to_string() }];
        let placements = place_adapters(&cluster, &jobs, &[]);
        assert_eq!(placements.len(), 1);
        assert_eq!(placements[0].node_name, "desktop"); // highest score
        assert_eq!(placements[0].adapter_idx, 0);
    }

    #[test]
    fn test_place_multiple_adapters_greedy() {
        let cluster = test_cluster();
        let jobs: Vec<AdapterJob> = (0..4)
            .map(|i| AdapterJob { adapter_idx: i, budget_mb: 6000, label: format!("adapter-{i}") })
            .collect();
        let placements = place_adapters(&cluster, &jobs, &[]);

        // Desktop has 3 slots and enough VRAM for 3 × 6 GB
        // Jetson has 1 slot with ~4915 MB usable — too small for 6 GB budget
        // So only 3 should be placed on desktop
        assert_eq!(placements.len(), 3);
        for p in &placements {
            assert_eq!(p.node_name, "desktop");
        }
    }

    #[test]
    fn test_place_small_adapters_across_nodes() {
        let cluster = test_cluster();
        let jobs: Vec<AdapterJob> = (0..4)
            .map(|i| AdapterJob {
                adapter_idx: i,
                budget_mb: 2000, // small enough for Jetson
                label: format!("adapter-{i}"),
            })
            .collect();
        let placements = place_adapters(&cluster, &jobs, &[]);

        // Desktop: 3 slots, Jetson: 1 slot → all 4 placed
        assert_eq!(placements.len(), 4);
        let desktop_count = placements.iter().filter(|p| p.node_name == "desktop").count();
        let jetson_count = placements.iter().filter(|p| p.node_name == "jetson").count();
        assert_eq!(desktop_count, 3);
        assert_eq!(jetson_count, 1);
    }

    #[test]
    fn test_place_no_capacity() {
        let cluster = test_cluster();
        let jobs = vec![AdapterJob {
            adapter_idx: 0,
            budget_mb: 30000, // too large for any node
            label: "too-big".to_string(),
        }];
        let placements = place_adapters(&cluster, &jobs, &[]);
        assert!(placements.is_empty());
    }

    #[test]
    fn test_place_with_initial_load() {
        let cluster = test_cluster();
        let jobs =
            vec![AdapterJob { adapter_idx: 0, budget_mb: 6000, label: "adapter-0".to_string() }];
        // Desktop already has 3 adapters (full)
        let load = vec![
            NodeLoad { active_adapters: 3, reserved_vram_mb: 18000 },
            NodeLoad::default(),
            NodeLoad::default(),
        ];
        let placements = place_adapters(&cluster, &jobs, &load);
        // Desktop full, Jetson too small for 6 GB → nothing placed
        assert!(placements.is_empty());
    }

    #[test]
    fn test_node_flops_factor_multi_gpu() {
        // A hypothetical node with two different GPUs should use the max
        let node = NodeConfig {
            name: "multi".to_string(),
            host: "localhost".to_string(),
            transport: super::super::cluster::Transport::Local,
            user: None,
            gpus: vec![
                super::super::cluster::GpuConfig {
                    uuid: "GPU-1".to_string(),
                    gpu_type: "rtx-3080".to_string(),
                    vram_mb: 10240,
                    memory_type: super::super::cluster::MemoryType::Discrete,
                },
                super::super::cluster::GpuConfig {
                    uuid: "GPU-2".to_string(),
                    gpu_type: "rtx-4090".to_string(),
                    vram_mb: 24564,
                    memory_type: super::super::cluster::MemoryType::Discrete,
                },
            ],
            max_adapters: 4,
            cpu_cores: None,
            ram_mb: None,
        };
        let flops = node_flops_factor(&node);
        assert!((flops - 1.0).abs() < f64::EPSILON); // max of 0.45 and 1.0
    }
}
