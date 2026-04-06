//! Dependency Graph Analysis
//!
//! Uses trueno-graph for cycle detection and topological sorting.
//! Edge metadata stored separately since trueno-graph only stores f32 weights.
//!
//! When the `trueno-graph` feature is disabled, a lightweight fallback
//! implementation is provided using only the standard library.

use crate::stack::is_paiml_crate;
use crate::stack::types::*;
use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::path::Path;
#[cfg(feature = "trueno-graph")]
use trueno_graph::{is_cyclic, toposort, CsrGraph, NodeId};

// ============================================================================
// Fallback graph primitives when trueno-graph is not available
// ============================================================================

#[cfg(not(feature = "trueno-graph"))]
mod fallback {
    use std::collections::{HashMap, HashSet, VecDeque};

    /// Lightweight node identifier (mirrors trueno_graph::NodeId)
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct NodeId(pub u32);

    /// Minimal adjacency-list graph (replaces CsrGraph)
    #[derive(Debug, Clone)]
    pub struct CsrGraph {
        outgoing: HashMap<u32, Vec<u32>>,
        incoming: HashMap<u32, Vec<u32>>,
        names: HashMap<u32, String>,
    }

    impl CsrGraph {
        pub fn new() -> Self {
            Self { outgoing: HashMap::new(), incoming: HashMap::new(), names: HashMap::new() }
        }

        pub fn from_edge_list(edges: &[(NodeId, NodeId, f32)]) -> Result<Self, &'static str> {
            let mut g = Self::new();
            for &(from, to, _) in edges {
                let _ = g.add_edge(from, to, 1.0);
            }
            Ok(g)
        }

        pub fn set_node_name(&mut self, id: NodeId, name: String) {
            self.names.insert(id.0, name);
        }

        pub fn add_edge(
            &mut self,
            from: NodeId,
            to: NodeId,
            _weight: f32,
        ) -> Result<(), &'static str> {
            self.outgoing.entry(from.0).or_default().push(to.0);
            self.incoming.entry(to.0).or_default().push(from.0);
            Ok(())
        }

        pub fn outgoing_neighbors(&self, id: NodeId) -> Result<&[u32], &'static str> {
            Ok(self.outgoing.get(&id.0).map(|v| v.as_slice()).unwrap_or(&[]))
        }

        pub fn incoming_neighbors(&self, id: NodeId) -> Result<&[u32], &'static str> {
            Ok(self.incoming.get(&id.0).map(|v| v.as_slice()).unwrap_or(&[]))
        }

        fn all_nodes(&self) -> HashSet<u32> {
            let mut nodes = HashSet::new();
            for (&k, vs) in &self.outgoing {
                nodes.insert(k);
                for &v in vs {
                    nodes.insert(v);
                }
            }
            for (&k, vs) in &self.incoming {
                nodes.insert(k);
                for &v in vs {
                    nodes.insert(v);
                }
            }
            nodes
        }
    }

    /// Detect cycles via DFS
    pub fn is_cyclic(graph: &CsrGraph) -> bool {
        let nodes = graph.all_nodes();
        let mut visited = HashSet::new();
        let mut on_stack = HashSet::new();

        for &node in &nodes {
            if !visited.contains(&node) && dfs_cycle(graph, node, &mut visited, &mut on_stack) {
                return true;
            }
        }
        false
    }

    fn dfs_cycle(
        graph: &CsrGraph,
        node: u32,
        visited: &mut HashSet<u32>,
        on_stack: &mut HashSet<u32>,
    ) -> bool {
        visited.insert(node);
        on_stack.insert(node);
        if let Ok(neighbors) = graph.outgoing_neighbors(NodeId(node)) {
            for &neighbor in neighbors {
                if !visited.contains(&neighbor) {
                    if dfs_cycle(graph, neighbor, visited, on_stack) {
                        return true;
                    }
                } else if on_stack.contains(&neighbor) {
                    return true;
                }
            }
        }
        on_stack.remove(&node);
        false
    }

    /// Topological sort via Kahn's algorithm
    pub fn toposort(graph: &CsrGraph) -> Result<Vec<NodeId>, &'static str> {
        let nodes = graph.all_nodes();
        let mut in_degree: HashMap<u32, usize> = HashMap::new();

        for &node in &nodes {
            in_degree.entry(node).or_insert(0);
            if let Ok(neighbors) = graph.outgoing_neighbors(NodeId(node)) {
                for &neighbor in neighbors {
                    *in_degree.entry(neighbor).or_insert(0) += 1;
                }
            }
        }

        let mut queue: VecDeque<u32> =
            in_degree.iter().filter(|(_, &deg)| deg == 0).map(|(&node, _)| node).collect();

        let mut result = Vec::new();
        while let Some(node) = queue.pop_front() {
            result.push(NodeId(node));
            if let Ok(neighbors) = graph.outgoing_neighbors(NodeId(node)) {
                for &neighbor in neighbors {
                    if let Some(deg) = in_degree.get_mut(&neighbor) {
                        *deg -= 1;
                        if *deg == 0 {
                            queue.push_back(neighbor);
                        }
                    }
                }
            }
        }

        if result.len() != nodes.len() {
            return Err("cycle detected");
        }
        Ok(result)
    }
}

#[cfg(not(feature = "trueno-graph"))]
use fallback::{is_cyclic, toposort, CsrGraph, NodeId};

/// Dependency graph for the PAIML stack
#[derive(Debug, Clone)]
pub struct DependencyGraph {
    /// The underlying trueno-graph CSR graph
    graph: CsrGraph,

    /// Map from crate name to node ID
    name_to_id: HashMap<String, NodeId>,

    /// Map from node ID to crate name
    id_to_name: HashMap<NodeId, String>,

    /// Edge metadata (not stored in trueno-graph which only has f32 weights)
    edge_data: HashMap<(NodeId, NodeId), DependencyEdge>,

    /// Crate information for each node
    crate_info: HashMap<String, CrateInfo>,

    /// Next node ID to assign
    next_id: u32,
}

/// Edge data in the dependency graph
#[derive(Debug, Clone)]
pub struct DependencyEdge {
    /// Version requirement
    pub version_req: String,

    /// Whether this is a path dependency
    pub is_path: bool,

    /// Dependency kind
    pub kind: DependencyKind,
}

impl DependencyGraph {
    /// Create a new empty dependency graph
    pub fn new() -> Self {
        Self {
            graph: CsrGraph::new(),
            name_to_id: HashMap::new(),
            id_to_name: HashMap::new(),
            edge_data: HashMap::new(),
            crate_info: HashMap::new(),
            next_id: 0,
        }
    }

    /// Build graph from cargo metadata, falling back to binary detection
    /// for non-Rust projects.
    #[cfg(feature = "native")]
    pub fn from_workspace(workspace_path: &Path) -> Result<Self> {
        use cargo_metadata::MetadataCommand;
        use std::collections::HashSet;

        let cargo_toml = workspace_path.join("Cargo.toml");
        if !cargo_toml.exists() {
            return Self::from_installed_binaries();
        }

        let metadata = MetadataCommand::new()
            .manifest_path(cargo_toml)
            .exec()
            .map_err(|e| anyhow!("Failed to read cargo metadata: {}", e))?;

        let mut graph = Self::new();

        // Collect workspace member package IDs so we only add edges from
        // crates WE control.  Resolved transitive deps (e.g. trueno pulled
        // in by aprender) may have optional reverse deps that create false
        // cycles in the graph (GH-25).
        let workspace_member_ids: HashSet<&cargo_metadata::PackageId> =
            metadata.workspace_members.iter().collect();

        // First pass: add all PAIML crates as nodes
        for package in &metadata.packages {
            if is_paiml_crate(&package.name) {
                let version = package.version.clone();
                let semver_version =
                    semver::Version::new(version.major, version.minor, version.patch);

                let crate_info = CrateInfo::new(
                    &package.name,
                    semver_version,
                    package.manifest_path.clone().into_std_path_buf(),
                );

                graph.add_crate(crate_info);
            }
        }

        // Second pass: add edges ONLY from workspace members.
        // Non-workspace PAIML packages are external — their own deps
        // (which may include optional reverse deps like trueno → aprender)
        // must NOT be added, as they create false cycles (GH-25).
        for package in &metadata.packages {
            if !is_paiml_crate(&package.name) {
                continue;
            }
            if !workspace_member_ids.contains(&package.id) {
                continue;
            }

            for dep in &package.dependencies {
                if is_paiml_crate(&dep.name) {
                    let is_path = dep.path.is_some();
                    let version_req = dep.req.to_string();

                    let kind = match dep.kind {
                        cargo_metadata::DependencyKind::Normal => DependencyKind::Normal,
                        cargo_metadata::DependencyKind::Development => DependencyKind::Dev,
                        cargo_metadata::DependencyKind::Build => DependencyKind::Build,
                        _ => DependencyKind::Normal,
                    };

                    let edge = DependencyEdge { version_req, is_path, kind };

                    // Add edge: package depends on dep
                    graph.add_dependency(&package.name, &dep.name, edge);

                    // Update crate info with dependency
                    if let Some(info) = graph.crate_info.get_mut(&package.name) {
                        let dep_info = if is_path {
                            let path = dep
                                .path
                                .as_ref()
                                .map(|p| p.clone().into_std_path_buf())
                                .unwrap_or_default();
                            DependencyInfo::path(&dep.name, path)
                        } else {
                            DependencyInfo::new(&dep.name, dep.req.to_string())
                        };
                        info.paiml_dependencies.push(dep_info);
                    }
                }
            }
        }

        Ok(graph)
    }

    /// Build a graph from installed PAIML binaries when no Cargo.toml is present.
    /// This allows `batuta stack status` to work in non-Rust projects that use
    /// sovereign stack tools via CLI.
    #[cfg(feature = "native")]
    fn from_installed_binaries() -> Result<Self> {
        use std::process::Command;

        let mut graph = Self::new();

        // Map binary names to crate names for common PAIML tools
        let binary_crates: &[(&str, &str)] = &[
            ("apr", "aprender"),
            ("pv", "provable-contracts"),
            ("forjar", "forjar"),
            ("alimentar", "alimentar"),
            ("batuta", "batuta"),
            ("pmat", "pmat"),
            ("bashrs", "bashrs"),
            ("realizar", "realizar"),
            ("entrenar", "entrenar"),
            ("certeza", "certeza"),
            ("ttop", "ttop"),
            ("depyler", "depyler"),
        ];

        for (binary, crate_name) in binary_crates {
            if let Ok(output) = Command::new("which").arg(binary).output() {
                if output.status.success() {
                    let bin_path = String::from_utf8_lossy(&output.stdout).trim().to_string();
                    let version = get_binary_version(binary);
                    let crate_info =
                        CrateInfo::new(*crate_name, version, std::path::PathBuf::from(&bin_path));
                    graph.add_crate(crate_info);
                }
            }
        }

        Ok(graph)
    }

    /// Add a crate to the graph
    pub fn add_crate(&mut self, info: CrateInfo) {
        let name = info.name.clone();
        if !self.name_to_id.contains_key(&name) {
            let id = NodeId(self.next_id);
            self.next_id += 1;
            self.name_to_id.insert(name.clone(), id);
            self.id_to_name.insert(id, name.clone());
            self.graph.set_node_name(id, name.clone());
        }
        self.crate_info.insert(name, info);
    }

    /// Ensure a node exists for a crate name
    fn ensure_node(&mut self, name: &str) -> NodeId {
        if let Some(&id) = self.name_to_id.get(name) {
            id
        } else {
            let id = NodeId(self.next_id);
            self.next_id += 1;
            self.name_to_id.insert(name.to_string(), id);
            self.id_to_name.insert(id, name.to_string());
            self.graph.set_node_name(id, name.to_string());
            id
        }
    }

    /// Add a dependency edge between crates
    pub fn add_dependency(&mut self, from: &str, to: &str, edge: DependencyEdge) {
        let from_id = self.ensure_node(from);
        let to_id = self.ensure_node(to);

        // Store edge metadata
        self.edge_data.insert((from_id, to_id), edge);

        // Add edge to trueno-graph (weight = 1.0)
        let _ = self.graph.add_edge(from_id, to_id, 1.0);
    }

    /// Build a release graph (excludes dev-dependencies)
    fn build_release_graph(&self) -> CsrGraph {
        let mut edges: Vec<(NodeId, NodeId, f32)> = Vec::new();

        // Collect non-dev edges
        for ((from_id, to_id), edge_data) in &self.edge_data {
            if edge_data.kind != DependencyKind::Dev {
                edges.push((*from_id, *to_id, 1.0));
            }
        }

        CsrGraph::from_edge_list(&edges).unwrap_or_else(|_| CsrGraph::new())
    }

    /// Check if the graph has cycles (excluding dev-dependencies)
    ///
    /// Dev-dependencies are excluded because they don't create real dependency
    /// cycles for release ordering. A crate can have a dev-dependency on another
    /// crate that depends on it (common for integration testing).
    pub fn has_cycles(&self) -> bool {
        let release_graph = self.build_release_graph();
        is_cyclic(&release_graph)
    }

    /// Get topological order for releases (dependencies first)
    ///
    /// Uses the release graph which excludes dev-dependencies to avoid
    /// false cycle detection (fixes issue #13).
    pub fn topological_order(&self) -> Result<Vec<String>> {
        let release_graph = self.build_release_graph();

        if is_cyclic(&release_graph) {
            return Err(anyhow!("Circular dependency detected in the graph"));
        }

        let sorted =
            toposort(&release_graph).map_err(|_| anyhow!("Failed to compute topological order"))?;

        // Map NodeIds back to names and reverse
        // Reverse because toposort gives dependents first, we want dependencies first
        let names: Vec<String> = sorted
            .iter()
            .rev() // Dependencies should come before dependents
            .filter_map(|id| self.id_to_name.get(id).cloned())
            .collect();

        Ok(names)
    }

    /// Get release order for a specific crate and its dependencies
    pub fn release_order_for(&self, crate_name: &str) -> Result<Vec<String>> {
        let full_order = self.topological_order()?;
        let deps = self.all_dependencies(crate_name);

        // Filter to only include the target crate and its dependencies
        let mut order: Vec<String> = full_order
            .into_iter()
            .filter(|name| name == crate_name || deps.contains(name))
            .collect();

        // Ensure the target crate is at the end
        if let Some(pos) = order.iter().position(|n| n == crate_name) {
            order.remove(pos);
            order.push(crate_name.to_string());
        }

        Ok(order)
    }

    /// Get all dependencies (transitive) for a crate
    pub fn all_dependencies(&self, crate_name: &str) -> Vec<String> {
        let mut deps = Vec::new();
        let mut visited = std::collections::HashSet::new();

        if let Some(&id) = self.name_to_id.get(crate_name) {
            self.collect_dependencies(id, &mut deps, &mut visited);
        }

        deps
    }

    fn collect_dependencies(
        &self,
        id: NodeId,
        deps: &mut Vec<String>,
        visited: &mut std::collections::HashSet<NodeId>,
    ) {
        if visited.contains(&id) {
            return;
        }
        visited.insert(id);

        // Get outgoing neighbors from trueno-graph
        if let Ok(neighbors) = self.graph.outgoing_neighbors(id) {
            for &neighbor_id in neighbors {
                let neighbor = NodeId(neighbor_id);
                if let Some(name) = self.id_to_name.get(&neighbor) {
                    if !deps.contains(name) {
                        deps.push(name.clone());
                    }
                    self.collect_dependencies(neighbor, deps, visited);
                }
            }
        }
    }

    /// Get crates that depend on this crate (reverse dependencies)
    pub fn dependents(&self, crate_name: &str) -> Vec<String> {
        let mut dependents = Vec::new();

        if let Some(&id) = self.name_to_id.get(crate_name) {
            if let Ok(neighbors) = self.graph.incoming_neighbors(id) {
                for &neighbor_id in neighbors {
                    if let Some(name) = self.id_to_name.get(&NodeId(neighbor_id)) {
                        dependents.push(name.clone());
                    }
                }
            }
        }

        dependents
    }

    /// Get all path dependencies in the graph
    pub fn find_path_dependencies(&self) -> Vec<PathDependencyIssue> {
        let mut issues = Vec::new();

        for ((from_id, to_id), edge_data) in &self.edge_data {
            if edge_data.is_path {
                if let (Some(from), Some(to)) =
                    (self.id_to_name.get(from_id), self.id_to_name.get(to_id))
                {
                    issues.push(PathDependencyIssue {
                        crate_name: from.clone(),
                        dependency: to.clone(),
                        current: format!("path = \"../{}\"", to),
                        recommended: None, // Will be filled in by checker
                    });
                }
            }
        }

        issues
    }

    /// Detect version conflicts for external dependencies
    pub fn detect_conflicts(&self) -> Vec<VersionConflict> {
        let mut dep_versions: HashMap<String, Vec<ConflictUsage>> = HashMap::new();

        // Collect all external dependency versions
        for (crate_name, info) in &self.crate_info {
            for dep in &info.external_dependencies {
                dep_versions.entry(dep.name.clone()).or_default().push(ConflictUsage {
                    crate_name: crate_name.clone(),
                    version_req: dep.version_req.clone(),
                });
            }
        }

        // Find conflicts (different versions of same dependency)
        let mut conflicts = Vec::new();
        for (dep_name, usages) in dep_versions {
            if usages.len() > 1 {
                let versions: std::collections::HashSet<_> =
                    usages.iter().map(|u| &u.version_req).collect();

                if versions.len() > 1 {
                    conflicts.push(VersionConflict {
                        dependency: dep_name,
                        usages,
                        recommendation: None,
                    });
                }
            }
        }

        conflicts
    }

    /// Get crate info by name
    pub fn get_crate(&self, name: &str) -> Option<&CrateInfo> {
        self.crate_info.get(name)
    }

    /// Get mutable crate info by name
    pub fn get_crate_mut(&mut self, name: &str) -> Option<&mut CrateInfo> {
        self.crate_info.get_mut(name)
    }

    /// Get all crates in the graph
    pub fn all_crates(&self) -> impl Iterator<Item = &CrateInfo> {
        self.crate_info.values()
    }

    /// Get number of crates in the graph
    pub fn crate_count(&self) -> usize {
        self.crate_info.len()
    }

    /// Check if a node exists (for compatibility with tests)
    pub(crate) fn node_indices_contains(&self, name: &str) -> bool {
        self.name_to_id.contains_key(name)
    }
}

impl Default for DependencyGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Path dependency issue for reporting
#[derive(Debug, Clone)]
pub struct PathDependencyIssue {
    /// Crate that has the path dependency
    pub crate_name: String,

    /// Dependency that is path-based
    pub dependency: String,

    /// Current specification
    pub current: String,

    /// Recommended crates.io version
    pub recommended: Option<String>,
}

/// Extract a semver version from a binary's --version output.
#[cfg(feature = "native")]
fn get_binary_version(binary: &str) -> semver::Version {
    let output = std::process::Command::new(binary).arg("--version").output().ok();

    let version_str =
        output.as_ref().map(|o| String::from_utf8_lossy(&o.stdout).to_string()).unwrap_or_default();

    // Parse version from output like "batuta 0.7.2" or "bashrs v6.65.0"
    let version = version_str
        .split_whitespace()
        .find_map(|word| {
            let w = word.trim_start_matches('v');
            semver::Version::parse(w).ok()
        })
        .unwrap_or_else(|| semver::Version::new(0, 0, 0));

    version
}

#[cfg(test)]
#[path = "graph_tests.rs"]
mod tests;
