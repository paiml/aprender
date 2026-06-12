//! ExecutionGraph export and visualization: DOT, CSR, TUI tree, ASCII tree.

use std::collections::HashMap;

use super::core::ExecutionGraph;
use crate::brick::exec_graph::node::{EdgeType, ExecutionNode, TransferDirection};

/// Build parent-child relationship map and identify root nodes.
fn build_children_map(
    edges: &[crate::brick::exec_graph::node::ExecutionEdge],
    node_count: usize,
) -> (HashMap<u32, Vec<u32>>, Vec<u32>) {
    let mut children_map: HashMap<u32, Vec<u32>> = HashMap::new();
    let mut has_parent: std::collections::HashSet<u32> = std::collections::HashSet::new();

    for edge in edges {
        if edge.edge_type == EdgeType::Contains || edge.edge_type == EdgeType::Launches {
            children_map.entry(edge.src.0).or_default().push(edge.dst.0);
            has_parent.insert(edge.dst.0);
        }
    }

    let root_ids: Vec<u32> = (0..node_count as u32).filter(|id| !has_parent.contains(id)).collect();

    (children_map, root_ids)
}

/// Format an execution node as (label, info) for ASCII tree rendering.
fn format_ascii_node(node: &ExecutionNode) -> (String, String) {
    match node {
        ExecutionNode::Layer { index } => (format!("Layer {}", index), String::new()),
        ExecutionNode::Brick { id: brick_id, timing_ns, elements } => (
            brick_id.name().to_string(),
            format!("  {:.1}µs ({} elem)", *timing_ns as f64 / 1000.0, elements),
        ),
        ExecutionNode::Kernel { name, grid, block, shared_mem, .. } => (
            name.clone(),
            format!("  <<<{},{},{}>>> smem={}B", grid.0, block.0, block.1, shared_mem),
        ),
        ExecutionNode::Function { name, file, line } => {
            let loc = match (file, line) {
                (Some(f), Some(l)) => format!(" ({}:{})", f, l),
                (None, _) | (_, None) => String::new(),
            };
            (format!("{}{}", name, loc), String::new())
        }
        ExecutionNode::Transfer { src, dst, bytes, direction, timing_ns } => {
            let timing_str =
                timing_ns.map(|ns| format!(" {:.1}µs", ns as f64 / 1000.0)).unwrap_or_default();
            (format!("{:?}: {} → {}", direction, src, dst), format!("  {}B{}", bytes, timing_str))
        }
        ExecutionNode::AsyncTask { name, poll_count, yield_count, total_poll_ns } => {
            let efficiency = if *poll_count > 0 { 100.0 / *poll_count as f64 } else { 0.0 };
            (
                name.clone(),
                format!(
                    "  polls:{} yields:{} {:.1}µs ({:.0}% eff)",
                    poll_count,
                    yield_count,
                    *total_poll_ns as f64 / 1000.0,
                    efficiency
                ),
            )
        }
    }
}

/// Recursively build ASCII tree string for a node and its children.
fn build_ascii_tree(
    graph: &ExecutionGraph,
    id: u32,
    children_map: &HashMap<u32, Vec<u32>>,
    prefix: &str,
    connector: &str,
    output: &mut String,
) {
    let (label, info) = format_ascii_node(&graph.nodes[id as usize]);
    output.push_str(&format!("{}{}{}{}\n", prefix, connector, label, info));

    if let Some(child_ids) = children_map.get(&id) {
        let child_count = child_ids.len();
        for (i, &child_id) in child_ids.iter().enumerate() {
            let is_last = i == child_count - 1;
            let new_connector = if is_last { "└── " } else { "├── " };
            let new_prefix = if connector.is_empty() {
                prefix.to_string()
            } else if connector == "└── " {
                format!("{}    ", prefix)
            } else {
                format!("{}│   ", prefix)
            };
            build_ascii_tree(graph, child_id, children_map, &new_prefix, new_connector, output);
        }
    }
}

/// Map an `ExecutionNode` to its DOT label text and style attribute string.
fn node_to_dot_label(node: &ExecutionNode) -> (String, &'static str) {
    match node {
        ExecutionNode::Layer { index } => {
            (format!("Layer {}", index), "style=filled,fillcolor=lightblue")
        }
        ExecutionNode::Brick { id, timing_ns, .. } => (
            format!("{}\\n{:.1}µs", id.name(), *timing_ns as f64 / 1000.0),
            "style=filled,fillcolor=lightgreen",
        ),
        ExecutionNode::Kernel { name, grid, block, .. } => (
            format!("{}\\n<<<{},{},{}>>>", name, grid.0, block.0, block.1),
            "style=filled,fillcolor=lightyellow",
        ),
        ExecutionNode::Function { name, file, line } => {
            let loc = match (file, line) {
                (Some(f), Some(l)) => format!("\\n{}:{}", f, l),
                (None, _) | (_, None) => String::new(),
            };
            (format!("{}{}", name, loc), "style=filled,fillcolor=lightgray")
        }
        ExecutionNode::Transfer { src, dst, bytes, direction, .. } => {
            let dir = match direction {
                TransferDirection::H2D => "H2D",
                TransferDirection::D2H => "D2H",
                TransferDirection::D2D => "D2D",
            };
            (
                format!("{}\\n{}->{}\\n{:.1}MB", dir, src, dst, *bytes as f64 / 1e6),
                "style=filled,fillcolor=lightsalmon",
            )
        }
        ExecutionNode::AsyncTask { name, poll_count, yield_count, total_poll_ns } => {
            let efficiency = if *poll_count > 0 { 100.0 / *poll_count as f64 } else { 0.0 };
            (
                format!(
                    "{}\\npolls:{} yields:{}\\n{:.1}µs ({:.0}%)",
                    name,
                    poll_count,
                    yield_count,
                    *total_poll_ns as f64 / 1000.0,
                    efficiency
                ),
                "style=filled,fillcolor=lightcyan",
            )
        }
    }
}

/// Map an `EdgeType` to its DOT style attribute string.
fn edge_to_dot_style(edge_type: &EdgeType) -> &'static str {
    match edge_type {
        EdgeType::Calls => "style=solid",
        EdgeType::Contains => "style=dashed",
        EdgeType::Launches => "style=bold,color=red",
        EdgeType::Sequence => "style=dotted",
        EdgeType::DependsOn => "style=solid,color=blue",
        EdgeType::Transfer { .. } => "style=bold,color=orange",
    }
}

impl ExecutionGraph {
    /// Export to DOT format for Graphviz visualization.
    pub fn to_dot(&self) -> String {
        debug_assert!(
            self.nodes.len() <= u32::MAX as usize,
            "CB-BUDGET: graph node count {} exceeds u32 range",
            self.nodes.len()
        );
        let mut dot = String::from("digraph ExecutionGraph {\n");
        dot.push_str("  rankdir=TB;\n");
        dot.push_str("  node [shape=box];\n\n");

        // Add nodes with styling based on type
        for (i, node) in self.nodes.iter().enumerate() {
            let (label, style) = node_to_dot_label(node);
            dot.push_str(&format!("  n{} [label=\"{}\",{}];\n", i, label, style));
        }

        dot.push('\n');

        // Add edges with styling based on type
        for edge in &self.edges {
            let style = edge_to_dot_style(&edge.edge_type);
            dot.push_str(&format!("  n{} -> n{} [{}];\n", edge.src.0, edge.dst.0, style));
        }

        dot.push_str("}\n");
        dot
    }

    // NOTE: `to_csr()` (export to trueno-graph CsrGraph) removed (APR-MONO
    // self-containment): aprender-graph depends on trueno, so compute→graph closed a
    // compute→graph→compute cycle. CSR export belongs in aprender-graph (which already
    // depends on compute), not in the compute foundation. `to_dot`/`to_ascii_tree` remain.

    /// Convert to presentar-terminal TreeNode for TUI visualization.
    ///
    /// PAR-201: Renders the execution graph as a collapsible tree in the terminal.
    #[cfg(feature = "presentar-tui")]
    pub fn to_tree_node(&self) -> presentar_terminal::TreeNode {
        use presentar_terminal::{Color, TreeNode};

        let layer_color = Color::new(0.4, 0.6, 1.0, 1.0);
        let brick_color = Color::new(0.4, 0.8, 0.4, 1.0);
        let kernel_color = Color::new(1.0, 0.8, 0.3, 1.0);
        let func_color = Color::new(0.7, 0.7, 0.7, 1.0);

        let (children_map, root_ids) = build_children_map(&self.edges, self.nodes.len());

        // Recursive function to build TreeNode
        fn build_node(
            graph: &ExecutionGraph,
            id: u32,
            children_map: &HashMap<u32, Vec<u32>>,
            layer_color: Color,
            brick_color: Color,
            kernel_color: Color,
            func_color: Color,
        ) -> TreeNode {
            let node = &graph.nodes[id as usize];
            let (label, info, color) = match node {
                ExecutionNode::Layer { index } => (format!("Layer {}", index), None, layer_color),
                ExecutionNode::Brick { id: brick_id, timing_ns, elements } => (
                    brick_id.name().to_string(),
                    Some(format!("{:.1}µs ({} elem)", *timing_ns as f64 / 1000.0, elements)),
                    brick_color,
                ),
                ExecutionNode::Kernel { name, grid, block, shared_mem, .. } => (
                    name.clone(),
                    Some(format!("<<<{},{},{}>>> smem={}B", grid.0, block.0, block.1, shared_mem)),
                    kernel_color,
                ),
                ExecutionNode::Function { name, file, line } => {
                    let loc = match (file, line) {
                        (Some(f), Some(l)) => format!(" ({}:{})", f, l),
                        (None, _) | (_, None) => String::new(),
                    };
                    (format!("{}{}", name, loc), None, func_color)
                }
                ExecutionNode::Transfer { src, dst, bytes, direction, timing_ns } => {
                    let timing_str = timing_ns
                        .map(|ns| format!(" {:.1}µs", ns as f64 / 1000.0))
                        .unwrap_or_default();
                    (
                        format!("{:?}: {} → {}", direction, src, dst),
                        Some(format!("{}B{}", bytes, timing_str)),
                        Color::new(0.8, 0.4, 0.8, 1.0), // Transfer color (magenta)
                    )
                }
                ExecutionNode::AsyncTask { name, poll_count, yield_count, total_poll_ns } => {
                    let efficiency = if *poll_count > 0 { 100.0 / *poll_count as f64 } else { 0.0 };
                    (
                        name.clone(),
                        Some(format!(
                            "polls:{} yields:{} {:.1}µs ({:.0}% eff)",
                            poll_count,
                            yield_count,
                            *total_poll_ns as f64 / 1000.0,
                            efficiency
                        )),
                        Color::new(0.4, 0.8, 0.8, 1.0), // Async task color (cyan)
                    )
                }
            };

            let mut tree_node = TreeNode::new(id as u64, label).with_color(color);
            if let Some(info_str) = info {
                tree_node = tree_node.with_info(info_str);
            }

            // Add children
            if let Some(child_ids) = children_map.get(&id) {
                for &child_id in child_ids {
                    let child = build_node(
                        graph,
                        child_id,
                        children_map,
                        layer_color,
                        brick_color,
                        kernel_color,
                        func_color,
                    );
                    tree_node = tree_node.with_child(child);
                }
            }

            tree_node
        }

        // Build root node
        if root_ids.is_empty() {
            TreeNode::new(0, "Empty Graph")
        } else if root_ids.len() == 1 {
            build_node(
                self,
                root_ids[0],
                &children_map,
                layer_color,
                brick_color,
                kernel_color,
                func_color,
            )
        } else {
            // Multiple roots: wrap in a synthetic root
            let mut root = TreeNode::new(u64::MAX, "Execution Graph")
                .with_color(Color::new(0.9, 0.9, 0.9, 1.0));
            for &root_id in &root_ids {
                let child = build_node(
                    self,
                    root_id,
                    &children_map,
                    layer_color,
                    brick_color,
                    kernel_color,
                    func_color,
                );
                root = root.with_child(child);
            }
            root
        }
    }

    /// Render graph to ASCII tree string (headless mode for testing/automation).
    ///
    /// PAR-201: Zero-dependency tree visualization for CI/CD, logging, and snapshot tests.
    #[must_use]
    pub fn to_ascii_tree(&self) -> String {
        let (children_map, root_ids) = build_children_map(&self.edges, self.nodes.len());
        let mut output = String::new();

        if root_ids.is_empty() {
            output.push_str("(empty graph)\n");
        } else if root_ids.len() == 1 {
            build_ascii_tree(self, root_ids[0], &children_map, "", "", &mut output);
        } else {
            output.push_str("Execution Graph\n");
            let root_count = root_ids.len();
            for (i, &root_id) in root_ids.iter().enumerate() {
                let is_last = i == root_count - 1;
                let connector = if is_last { "└── " } else { "├── " };
                build_ascii_tree(self, root_id, &children_map, "", connector, &mut output);
            }
        }

        if output.ends_with('\n') {
            output.pop();
        }
        output
    }
}
