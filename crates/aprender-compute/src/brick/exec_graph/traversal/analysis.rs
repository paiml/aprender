//! ExecutionGraph analysis: critical path, slack, roofline, ping-pong detection.

use std::collections::HashMap;

use super::core::ExecutionGraph;
use crate::brick::exec_graph::node::{EdgeType, ExecutionNode, ExecutionNodeId, TransferDirection};

impl ExecutionGraph {
    // ========================
    // Phase 9: Critical Path Analysis (CPA)
    // ========================

    /// Get timing for a node (ns). Returns 0 for non-timed nodes.
    fn node_timing_ns(&self, id: ExecutionNodeId) -> u64 {
        debug_assert!(
            (id.0 as usize) < self.nodes.len(),
            "CB-BUDGET: node id {} out of bounds (graph has {} nodes)",
            id.0,
            self.nodes.len()
        );
        match &self.nodes[id.0 as usize] {
            ExecutionNode::Brick { timing_ns, .. } => *timing_ns,
            ExecutionNode::Kernel { timing_ns, .. } => timing_ns.unwrap_or(0),
            ExecutionNode::Transfer { timing_ns, .. } => timing_ns.unwrap_or(0),
            ExecutionNode::Function { .. }
            | ExecutionNode::Layer { .. }
            | ExecutionNode::AsyncTask { .. } => 0,
        }
    }

    /// Compute critical path through execution graph using longest-path algorithm.
    ///
    /// Returns (critical_path_nodes, total_time_ns). The critical path represents
    /// the longest chain of dependencies that determines total execution time.
    ///
    /// Reference: Graham et al. (1979) "Scheduling Algorithms for Multi-Processor Systems"
    pub fn critical_path(&self) -> (Vec<ExecutionNodeId>, u64) {
        if self.nodes.is_empty() {
            return (vec![], 0);
        }

        // Build adjacency list for DependsOn and Sequence edges
        let mut adj: Vec<Vec<(u32, u64)>> = vec![vec![]; self.nodes.len()];
        for edge in &self.edges {
            match &edge.edge_type {
                EdgeType::DependsOn | EdgeType::Sequence => {
                    let weight = self.node_timing_ns(edge.dst);
                    adj[edge.src.0 as usize].push((edge.dst.0, weight));
                }
                EdgeType::Contains | EdgeType::Calls | EdgeType::Launches => {
                    // Hierarchical edges: children contribute to parent time
                    let weight = self.node_timing_ns(edge.dst);
                    adj[edge.src.0 as usize].push((edge.dst.0, weight));
                }
                EdgeType::Transfer { .. } => {
                    // Transfer edges carry their own timing
                    let weight = self.node_timing_ns(edge.dst);
                    adj[edge.src.0 as usize].push((edge.dst.0, weight));
                }
            }
        }

        // Topological sort using Kahn's algorithm
        let mut in_degree = vec![0u32; self.nodes.len()];
        for edges in &adj {
            for (dst, _) in edges {
                in_degree[*dst as usize] += 1;
            }
        }

        let mut queue: Vec<u32> =
            (0..self.nodes.len() as u32).filter(|&i| in_degree[i as usize] == 0).collect();
        let mut topo_order = Vec::with_capacity(self.nodes.len());

        while let Some(u) = queue.pop() {
            topo_order.push(u);
            for (v, _) in &adj[u as usize] {
                in_degree[*v as usize] -= 1;
                if in_degree[*v as usize] == 0 {
                    queue.push(*v);
                }
            }
        }

        // Longest path DP
        let mut dist = vec![0u64; self.nodes.len()];
        let mut pred = vec![None::<u32>; self.nodes.len()];

        // Initialize with node's own timing for roots
        for &node in &topo_order {
            if self.edges.iter().all(|e| e.dst.0 != node) {
                dist[node as usize] = self.node_timing_ns(ExecutionNodeId(node));
            }
        }

        for &u in &topo_order {
            for (v, weight) in &adj[u as usize] {
                let new_dist = dist[u as usize] + weight;
                if new_dist > dist[*v as usize] {
                    dist[*v as usize] = new_dist;
                    pred[*v as usize] = Some(u);
                }
            }
        }

        // Find endpoint with maximum distance
        let (end_node, &total_time) =
            dist.iter().enumerate().max_by_key(|(_, &d)| d).unwrap_or((0, &0));

        // Reconstruct path
        let mut path = vec![];
        let mut current = Some(end_node as u32);
        while let Some(node) = current {
            path.push(ExecutionNodeId(node));
            current = pred[node as usize];
        }
        path.reverse();

        (path, total_time)
    }

    /// Compute slack for each node (how much it can be delayed without affecting total time).
    ///
    /// Returns map from node ID to slack in nanoseconds. Nodes on critical path have slack = 0.
    pub fn compute_slack(&self) -> HashMap<ExecutionNodeId, u64> {
        let (critical_path, total_time) = self.critical_path();
        let critical_set: std::collections::HashSet<_> = critical_path.iter().copied().collect();

        let mut slack = HashMap::new();

        // Build reverse adjacency
        let mut reverse_adj: Vec<Vec<u32>> = vec![vec![]; self.nodes.len()];
        for edge in &self.edges {
            reverse_adj[edge.dst.0 as usize].push(edge.src.0);
        }

        // Forward pass: earliest start time
        let mut earliest = vec![0u64; self.nodes.len()];
        for i in 0..self.nodes.len() {
            let mut max_pred = 0u64;
            for &pred in &reverse_adj[i] {
                max_pred = max_pred
                    .max(earliest[pred as usize] + self.node_timing_ns(ExecutionNodeId(pred)));
            }
            earliest[i] = max_pred;
        }

        // Backward pass: latest start time
        let mut latest = vec![total_time; self.nodes.len()];
        for i in (0..self.nodes.len()).rev() {
            let timing = self.node_timing_ns(ExecutionNodeId(i as u32));
            let mut min_succ = total_time;
            for edge in &self.edges {
                if edge.src.0 == i as u32 {
                    min_succ = min_succ.min(latest[edge.dst.0 as usize]);
                }
            }
            latest[i] = min_succ.saturating_sub(timing);
        }

        // Slack = latest - earliest
        for i in 0..self.nodes.len() {
            let node_id = ExecutionNodeId(i as u32);
            let node_slack = if critical_set.contains(&node_id) {
                0
            } else {
                latest[i].saturating_sub(earliest[i])
            };
            slack.insert(node_id, node_slack);
        }

        slack
    }

    /// Compute roofline distance for kernel nodes.
    ///
    /// Returns map from kernel node ID to distance from roofline (0.0 = optimal).
    /// Distance = 1.0 - min(achieved/peak_compute, achieved/peak_bandwidth).
    ///
    /// Reference: Williams et al. (2009) "Roofline: An Insightful Visual Performance Model"
    pub fn roofline_distance(
        &self,
        peak_tflops: f32,
        peak_bandwidth_gb_s: f32,
    ) -> HashMap<ExecutionNodeId, f32> {
        let mut distances = HashMap::new();

        for (i, node) in self.nodes.iter().enumerate() {
            if let ExecutionNode::Kernel { arithmetic_intensity, achieved_tflops, .. } = node {
                if let (Some(ai), Some(achieved)) = (arithmetic_intensity, achieved_tflops) {
                    // Roofline model: achievable = min(peak_compute, ai * bandwidth)
                    let bandwidth_bound = *ai * peak_bandwidth_gb_s / 1000.0; // Convert GB/s to TFLOP/s
                    let roofline_bound = peak_tflops.min(bandwidth_bound);
                    let efficiency = achieved / roofline_bound;
                    let distance = 1.0 - efficiency.min(1.0);
                    distances.insert(ExecutionNodeId(i as u32), distance);
                }
            }
        }

        distances
    }

    /// Detect ping-pong memory transfer patterns (wasteful H2D followed by D2H).
    ///
    /// Returns pairs of transfer node IDs that exhibit ping-pong behavior.
    pub fn detect_ping_pong(&self) -> Vec<(ExecutionNodeId, ExecutionNodeId)> {
        let mut patterns = Vec::new();

        // Find transfer nodes
        let transfers: Vec<(usize, &ExecutionNode)> = self
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| matches!(n, ExecutionNode::Transfer { .. }))
            .collect();

        // Check for H2D followed by D2H on same data
        for i in 0..transfers.len() {
            for j in (i + 1)..transfers.len() {
                if let (
                    ExecutionNode::Transfer {
                        src: src1,
                        dst: dst1,
                        direction: dir1,
                        bytes: bytes1,
                        ..
                    },
                    ExecutionNode::Transfer {
                        src: src2,
                        dst: dst2,
                        direction: dir2,
                        bytes: bytes2,
                        ..
                    },
                ) = (&transfers[i].1, &transfers[j].1)
                {
                    // Ping-pong: H2D then D2H with matching src/dst and same size
                    let is_ping_pong = (*dir1 == TransferDirection::H2D
                        && *dir2 == TransferDirection::D2H
                        && dst1 == src2
                        && bytes1 == bytes2)
                        || (*dir1 == TransferDirection::D2H
                            && *dir2 == TransferDirection::H2D
                            && src1 == dst2
                            && bytes1 == bytes2);

                    if is_ping_pong {
                        patterns.push((
                            ExecutionNodeId(transfers[i].0 as u32),
                            ExecutionNodeId(transfers[j].0 as u32),
                        ));
                    }
                }
            }
        }

        patterns
    }

    /// Get critical path analysis summary as formatted string.
    pub fn critical_path_summary(&self) -> String {
        let (path, total_ns) = self.critical_path();
        let slack = self.compute_slack();

        let mut output = String::new();
        output.push_str(&format!(
            "Critical Path: {:.2}ms ({} nodes)\n",
            total_ns as f64 / 1_000_000.0,
            path.len()
        ));
        output.push_str("─".repeat(50).as_str());
        output.push('\n');

        for (i, node_id) in path.iter().enumerate() {
            let node = &self.nodes[node_id.0 as usize];
            let timing = self.node_timing_ns(*node_id);
            let node_name = Self::format_node_name(node);

            let prefix = if i == 0 {
                "┌"
            } else if i == path.len() - 1 {
                "└"
            } else {
                "│"
            };
            output.push_str(&format!(
                "{} {} ({:.1}µs)\n",
                prefix,
                node_name,
                timing as f64 / 1000.0
            ));
        }

        // Show nodes with most slack (parallelization opportunities)
        let mut slack_vec: Vec<_> = slack.iter().collect();
        slack_vec.sort_by(|a, b| b.1.cmp(a.1));

        if slack_vec.iter().any(|(_, &s)| s > 0) {
            output.push_str("\nParallelization Opportunities (high slack):\n");
            for (node_id, &node_slack) in slack_vec.iter().take(5) {
                if node_slack > 0 {
                    let node = &self.nodes[node_id.0 as usize];
                    let node_name = Self::format_node_name(node);
                    output.push_str(&format!(
                        "  {} slack={:.1}µs\n",
                        node_name,
                        node_slack as f64 / 1000.0
                    ));
                }
            }
        }

        output
    }

    /// Format a node name for display in critical path summaries.
    fn format_node_name(node: &ExecutionNode) -> String {
        match node {
            ExecutionNode::Layer { index } => format!("Layer {}", index),
            ExecutionNode::Brick { id, .. } => id.name().to_string(),
            ExecutionNode::Kernel { name, .. } => name.clone(),
            ExecutionNode::Function { name, .. } => name.clone(),
            ExecutionNode::Transfer { direction, src, dst, .. } => {
                format!("{:?} {} → {}", direction, src, dst)
            }
            ExecutionNode::AsyncTask { name, poll_count, .. } => {
                format!("{} ({}polls)", name, poll_count)
            }
        }
    }
}
