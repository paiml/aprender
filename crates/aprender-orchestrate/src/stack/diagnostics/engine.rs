//! Stack diagnostics engine
//!
//! This module contains the main `StackDiagnostics` struct which serves as the
//! primary analysis engine for stack health monitoring. It includes graph
//! algorithms like PageRank and Betweenness Centrality.

use super::types::{
    AndonStatus, Anomaly, ComponentNode, GraphMetrics, HealthStatus, HealthSummary,
};
use crate::stack::DependencyGraph;
use anyhow::Result;
use std::collections::HashMap;

// ============================================================================
// Stack Diagnostics Engine
// ============================================================================

/// Main diagnostics engine for stack analysis
#[derive(Debug)]
pub struct StackDiagnostics {
    /// Component nodes
    components: HashMap<String, ComponentNode>,
    /// Dependency graph
    graph: Option<DependencyGraph>,
    /// Computed graph metrics
    metrics: GraphMetrics,
    /// Detected anomalies
    anomalies: Vec<Anomaly>,
}

impl StackDiagnostics {
    /// Create a new diagnostics engine
    pub fn new() -> Self {
        Self {
            components: HashMap::new(),
            graph: None,
            metrics: GraphMetrics::default(),
            anomalies: Vec::new(),
        }
    }

    /// Add a component to the knowledge graph
    pub fn add_component(&mut self, node: ComponentNode) {
        self.components.insert(node.name.clone(), node);
    }

    /// Get a component by name
    pub fn get_component(&self, name: &str) -> Option<&ComponentNode> {
        self.components.get(name)
    }

    /// Get all components
    pub fn components(&self) -> impl Iterator<Item = &ComponentNode> {
        self.components.values()
    }

    /// Get component count
    pub fn component_count(&self) -> usize {
        self.components.len()
    }

    /// Set the dependency graph
    pub fn set_graph(&mut self, graph: DependencyGraph) {
        self.graph = Some(graph);
    }

    /// Get the dependency graph
    pub fn graph(&self) -> Option<&DependencyGraph> {
        self.graph.as_ref()
    }

    /// Compute graph metrics (PageRank, Betweenness, etc.)
    pub fn compute_metrics(&mut self) -> Result<&GraphMetrics> {
        let n = self.components.len();
        if n == 0 {
            return Ok(&self.metrics);
        }

        self.metrics.total_nodes = n;

        // Build adjacency from dependency graph if available
        let adjacency = self.build_adjacency();

        // Compute PageRank
        self.compute_pagerank(&adjacency, 0.85, 100);

        // Compute Betweenness Centrality
        self.compute_betweenness(&adjacency);

        // Compute depth from roots
        self.compute_depth(&adjacency);

        // Compute graph-level metrics
        self.metrics.total_edges = adjacency.values().map(|v| v.len()).sum();
        let max_edges = n * (n.saturating_sub(1));
        self.metrics.density =
            if max_edges > 0 { self.metrics.total_edges as f64 / max_edges as f64 } else { 0.0 };
        self.metrics.avg_degree =
            if n > 0 { self.metrics.total_edges as f64 / n as f64 } else { 0.0 };
        self.metrics.max_depth = self.metrics.depth_map.values().copied().max().unwrap_or(0);

        Ok(&self.metrics)
    }

    /// Build adjacency list from dependency graph
    fn build_adjacency(&self) -> HashMap<String, Vec<String>> {
        let mut adjacency: HashMap<String, Vec<String>> = HashMap::new();

        // Initialize all nodes
        for name in self.components.keys() {
            adjacency.insert(name.clone(), Vec::new());
        }

        // Add edges from dependency graph
        if let Some(graph) = &self.graph {
            for crate_info in graph.all_crates() {
                let from = &crate_info.name;
                for dep in &crate_info.paiml_dependencies {
                    if self.components.contains_key(&dep.name) {
                        adjacency.entry(from.clone()).or_default().push(dep.name.clone());
                    }
                }
            }
        }

        adjacency
    }

    /// Compute PageRank using power iteration
    fn compute_pagerank(
        &mut self,
        adjacency: &HashMap<String, Vec<String>>,
        damping: f64,
        max_iter: usize,
    ) {
        let n = self.components.len();
        if n == 0 {
            return;
        }

        let initial = 1.0 / n as f64;
        let mut scores: HashMap<String, f64> =
            self.components.keys().map(|k| (k.clone(), initial)).collect();

        // Find dangling nodes (nodes with no outgoing edges)
        let dangling_nodes: Vec<_> = adjacency
            .iter()
            .filter(|(_, targets)| targets.is_empty())
            .map(|(node, _)| node.clone())
            .collect();

        // Power iteration
        for _ in 0..max_iter {
            let mut new_scores: HashMap<String, f64> = HashMap::new();
            let teleport = (1.0 - damping) / n as f64;

            // Dangling nodes contribute their rank equally to all nodes
            let dangling_sum: f64 =
                dangling_nodes.iter().map(|node| scores.get(node).unwrap_or(&0.0)).sum();
            let dangling_contrib = damping * dangling_sum / n as f64;

            for node in self.components.keys() {
                let mut incoming_score = 0.0;

                // Find nodes that link to this node
                for (source, targets) in adjacency {
                    if targets.contains(node) {
                        let out_degree = targets.len();
                        if out_degree > 0 {
                            incoming_score +=
                                scores.get(source).unwrap_or(&0.0) / out_degree as f64;
                        }
                    }
                }

                new_scores
                    .insert(node.clone(), teleport + damping * incoming_score + dangling_contrib);
            }

            // Check convergence
            let diff: f64 =
                new_scores.iter().map(|(k, v)| (v - scores.get(k).unwrap_or(&0.0)).abs()).sum();

            scores = new_scores;

            if diff < 1e-6 {
                break;
            }
        }

        self.metrics.pagerank = scores;
    }

    /// Compute Betweenness Centrality using Brandes algorithm (simplified)
    fn compute_betweenness(&mut self, adjacency: &HashMap<String, Vec<String>>) {
        let nodes: Vec<_> = self.components.keys().cloned().collect();
        let n = nodes.len();

        // Initialize betweenness
        let mut betweenness: HashMap<String, f64> =
            nodes.iter().map(|n| (n.clone(), 0.0)).collect();

        // For each source, compute shortest paths and accumulate
        for source in &nodes {
            // BFS from source
            let mut dist: HashMap<String, i32> = HashMap::new();
            let mut sigma: HashMap<String, f64> = HashMap::new();
            let mut predecessors: HashMap<String, Vec<String>> = HashMap::new();

            for n in &nodes {
                dist.insert(n.clone(), -1);
                sigma.insert(n.clone(), 0.0);
                predecessors.insert(n.clone(), Vec::new());
            }

            dist.insert(source.clone(), 0);
            sigma.insert(source.clone(), 1.0);

            let mut queue = vec![source.clone()];
            let mut order = Vec::new();

            while !queue.is_empty() {
                let v = queue.remove(0);
                order.push(v.clone());

                if let Some(neighbors) = adjacency.get(&v) {
                    for w in neighbors {
                        let d_v = dist[&v];
                        let d_w = dist.get(w).copied().unwrap_or(-1);

                        if d_w < 0 {
                            dist.insert(w.clone(), d_v + 1);
                            queue.push(w.clone());
                        }

                        if dist.get(w).copied().unwrap_or(-1) == d_v + 1 {
                            let sigma_v = sigma.get(&v).copied().unwrap_or(0.0);
                            if let Some(s) = sigma.get_mut(w) {
                                *s += sigma_v;
                            }
                            if let Some(p) = predecessors.get_mut(w) {
                                p.push(v.clone());
                            }
                        }
                    }
                }
            }

            // Back-propagation
            let mut delta: HashMap<String, f64> = nodes.iter().map(|n| (n.clone(), 0.0)).collect();

            for w in order.iter().rev() {
                for v in predecessors.get(w).cloned().unwrap_or_default() {
                    let sigma_v = sigma.get(&v).copied().unwrap_or(1.0);
                    let sigma_w = sigma.get(w).copied().unwrap_or(1.0);
                    let delta_w = delta.get(w).copied().unwrap_or(0.0);

                    if sigma_w > 0.0 {
                        if let Some(d) = delta.get_mut(&v) {
                            *d += (sigma_v / sigma_w) * (1.0 + delta_w);
                        }
                    }
                }

                if w != source {
                    if let Some(b) = betweenness.get_mut(w) {
                        *b += delta.get(w).copied().unwrap_or(0.0);
                    }
                }
            }
        }

        // Normalize
        let norm = if n > 2 { (n - 1) * (n - 2) } else { 1 };
        for v in betweenness.values_mut() {
            *v /= norm as f64;
        }

        self.metrics.betweenness = betweenness;
    }

    /// Compute depth from root nodes (nodes with no incoming edges)
    fn compute_depth(&mut self, adjacency: &HashMap<String, Vec<String>>) {
        let mut depth: HashMap<String, u32> = HashMap::new();
        let nodes: Vec<_> = self.components.keys().cloned().collect();

        // Find incoming edges for each node
        let mut has_incoming: HashMap<String, bool> =
            nodes.iter().map(|n| (n.clone(), false)).collect();
        for targets in adjacency.values() {
            for t in targets {
                has_incoming.insert(t.clone(), true);
            }
        }

        // Roots are nodes with no incoming edges
        let roots: Vec<_> =
            nodes.iter().filter(|n| !has_incoming.get(*n).unwrap_or(&false)).cloned().collect();

        // BFS from roots
        let mut queue: Vec<(String, u32)> = roots.into_iter().map(|r| (r, 0)).collect();

        while let Some((node, d)) = queue.pop() {
            if let std::collections::hash_map::Entry::Vacant(e) = depth.entry(node.clone()) {
                e.insert(d);
                if let Some(neighbors) = adjacency.get(&node) {
                    for neighbor in neighbors {
                        if !depth.contains_key(neighbor) {
                            queue.push((neighbor.clone(), d + 1));
                        }
                    }
                }
            }
        }

        // Assign depth 0 to any unreachable nodes
        for node in &nodes {
            depth.entry(node.clone()).or_insert(0);
        }

        self.metrics.depth_map = depth;
    }

    /// Get computed metrics
    pub fn metrics(&self) -> &GraphMetrics {
        &self.metrics
    }

    /// Get detected anomalies
    pub fn anomalies(&self) -> &[Anomaly] {
        &self.anomalies
    }

    /// Add an anomaly
    pub fn add_anomaly(&mut self, anomaly: Anomaly) {
        self.anomalies.push(anomaly);
    }

    /// Compute stack health summary
    pub fn health_summary(&self) -> HealthSummary {
        let total = self.components.len();
        let green = self.components.values().filter(|c| c.health == HealthStatus::Green).count();
        let yellow = self.components.values().filter(|c| c.health == HealthStatus::Yellow).count();
        let red = self.components.values().filter(|c| c.health == HealthStatus::Red).count();

        let avg_score = if total > 0 {
            self.components.values().map(|c| c.metrics.demo_score).sum::<f64>() / total as f64
        } else {
            0.0
        };

        HealthSummary {
            total_components: total,
            green_count: green,
            yellow_count: yellow,
            red_count: red,
            unknown_count: total.saturating_sub(green + yellow + red),
            avg_demo_score: avg_score,
            avg_coverage: self.avg_metric(|c| c.metrics.coverage),
            andon_status: self.compute_andon_status(green, yellow, red, total),
        }
    }

    fn avg_metric<F>(&self, f: F) -> f64
    where
        F: Fn(&ComponentNode) -> f64,
    {
        let total = self.components.len();
        if total == 0 {
            return 0.0;
        }
        self.components.values().map(f).sum::<f64>() / total as f64
    }

    fn compute_andon_status(
        &self,
        green: usize,
        yellow: usize,
        red: usize,
        total: usize,
    ) -> AndonStatus {
        if red > 0 {
            AndonStatus::Red
        } else if yellow > 0 {
            AndonStatus::Yellow
        } else if green == total && total > 0 {
            AndonStatus::Green
        } else {
            AndonStatus::Unknown
        }
    }
}

impl Default for StackDiagnostics {
    fn default() -> Self {
        Self::new()
    }
}
