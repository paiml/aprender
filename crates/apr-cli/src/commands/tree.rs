//! Model architecture tree visualization (GH-122)
//!
//! Toyota Way: Visualization - Make the structure visible.
//! Display model architecture as a hierarchical tree with sizes.
//!
//! Usage:
//!   apr tree model.apr
//!   apr tree model.apr --filter "decoder"
//!   apr tree model.apr --format dot > model.dot
//!   apr tree model.apr --format mermaid

use crate::error::CliError;
use aprender::format::rosetta::RosettaStone;
use colored::Colorize;
use std::collections::BTreeMap;
use std::path::Path;

/// Output format for tree
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TreeFormat {
    /// ASCII tree (default)
    Ascii,
    /// Graphviz DOT format
    Dot,
    /// Mermaid diagram format
    Mermaid,
    /// JSON structure
    Json,
}

impl std::str::FromStr for TreeFormat {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "ascii" | "text" => Ok(Self::Ascii),
            "dot" | "graphviz" => Ok(Self::Dot),
            "mermaid" | "md" => Ok(Self::Mermaid),
            "json" => Ok(Self::Json),
            _ => Err(format!("Unknown format: {s}")),
        }
    }
}

/// Tree node representing a tensor or group
#[derive(Debug, Clone)]
struct TreeNode {
    name: String,
    full_path: String,
    shape: Option<Vec<usize>>,
    size_bytes: usize,
    children: BTreeMap<String, TreeNode>,
    is_leaf: bool,
}

impl TreeNode {
    fn new(name: &str, full_path: &str) -> Self {
        Self {
            name: name.to_string(),
            full_path: full_path.to_string(),
            shape: None,
            size_bytes: 0,
            children: BTreeMap::new(),
            is_leaf: false,
        }
    }

    fn total_size(&self) -> usize {
        if self.is_leaf {
            self.size_bytes
        } else {
            self.children.values().map(TreeNode::total_size).sum()
        }
    }

    fn tensor_count(&self) -> usize {
        if self.is_leaf {
            1
        } else {
            self.children.values().map(TreeNode::tensor_count).sum()
        }
    }
}

/// Collect tensors from Rosetta Stone inspection report into tree
fn build_tree_from_rosetta(
    report: &aprender::format::rosetta::InspectionReport,
    filter: Option<&str>,
) -> TreeNode {
    let mut root = TreeNode::new("model", "");
    for tensor in &report.tensors {
        if let Some(f) = filter {
            if !tensor.name.contains(f) {
                continue;
            }
        }
        insert_tensor(&mut root, &tensor.name, &tensor.shape, tensor.size_bytes);
    }
    root
}

/// Insert a single tensor into the tree
fn insert_tensor(root: &mut TreeNode, name: &str, shape: &[usize], size: usize) {
    let parts: Vec<&str> = name.split('.').collect();
    let mut current = root;

    for (i, part) in parts.iter().enumerate() {
        let path = parts[..=i].join(".");

        if !current.children.contains_key(*part) {
            current
                .children
                .insert((*part).to_string(), TreeNode::new(part, &path));
        }

        current = current.children.get_mut(*part).expect("just inserted");

        if i == parts.len() - 1 {
            current.is_leaf = true;
            current.shape = Some(shape.to_vec());
            current.size_bytes = size;
        }
    }
}

/// Run the tree command
pub(crate) fn run(
    apr_path: &Path,
    filter: Option<&str>,
    format: TreeFormat,
    show_sizes: bool,
    max_depth: Option<usize>,
) -> Result<(), CliError> {
    if !apr_path.exists() {
        return Err(CliError::FileNotFound(apr_path.to_path_buf()));
    }

    // All formats go through Rosetta Stone (handles GGUF, SafeTensors, APR v2)
    let rosetta = RosettaStone::new();
    let report = rosetta
        .inspect(apr_path)
        .map_err(|e| CliError::InvalidFormat(format!("Failed to inspect: {e}")))?;
    let root = build_tree_from_rosetta(&report, filter);

    // Output based on format
    match format {
        TreeFormat::Ascii => print_ascii_tree(&root, apr_path, show_sizes, max_depth),
        TreeFormat::Dot => print_dot_graph(&root),
        TreeFormat::Mermaid => print_mermaid_graph(&root),
        TreeFormat::Json => print_json_tree(&root),
    }

    Ok(())
}

/// Print ASCII tree
fn print_ascii_tree(root: &TreeNode, path: &Path, show_sizes: bool, max_depth: Option<usize>) {
    let total_size = root.total_size();
    let tensor_count = root.tensor_count();

    // Header
    println!(
        "{} ({} tensors, {})",
        path.file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("model")
            .cyan()
            .bold(),
        tensor_count.to_string().green(),
        format_size(total_size).yellow()
    );

    // Print children
    let children: Vec<_> = root.children.values().collect();
    for (i, child) in children.iter().enumerate() {
        let is_last = i == children.len() - 1;
        print_tree_node(child, "", is_last, show_sizes, 0, max_depth);
    }
}

#[allow(clippy::disallowed_methods)] // unwrap_or_default is safe for formatting
fn print_tree_node(
    node: &TreeNode,
    prefix: &str,
    is_last: bool,
    show_sizes: bool,
    depth: usize,
    max_depth: Option<usize>,
) {
    // Check depth limit
    if let Some(max) = max_depth {
        if depth >= max {
            return;
        }
    }

    let connector = if is_last { "└── " } else { "├── " };
    let child_prefix = if is_last { "    " } else { "│   " };

    // Node name
    let name_str = if node.is_leaf {
        format!("{}", node.name.green())
    } else {
        format!("{}", node.name.blue().bold())
    };

    // Shape and size info
    let info_str = if node.is_leaf {
        let shape_str = node
            .shape
            .as_ref()
            .map(|s| format!("{s:?}"))
            .unwrap_or_default();
        let size_str = if show_sizes {
            format!(" ━━━ {}", format_size(node.size_bytes).dimmed())
        } else {
            String::new()
        };
        format!(" {}{}", shape_str.dimmed(), size_str)
    } else if show_sizes {
        let total = node.total_size();
        let count = node.tensor_count();
        format!(
            " [{} tensors, {}]",
            count.to_string().dimmed(),
            format_size(total).dimmed()
        )
    } else {
        String::new()
    };

    println!("{prefix}{connector}{name_str}{info_str}");

    // Print children
    let children: Vec<_> = node.children.values().collect();
    for (i, child) in children.iter().enumerate() {
        let child_is_last = i == children.len() - 1;
        let new_prefix = format!("{prefix}{child_prefix}");
        print_tree_node(
            child,
            &new_prefix,
            child_is_last,
            show_sizes,
            depth + 1,
            max_depth,
        );
    }
}

/// Print Graphviz DOT format
fn print_dot_graph(root: &TreeNode) {
    println!("digraph model {{");
    println!("  rankdir=TB;");
    println!("  node [shape=box, fontname=\"Helvetica\"];");
    println!("  edge [fontname=\"Helvetica\", fontsize=10];");
    println!();

    // Generate nodes and edges
    print_dot_nodes(root, "root");

    println!("}}");
}

#[allow(clippy::disallowed_methods)] // unwrap_or_default is safe for formatting
fn print_dot_nodes(node: &TreeNode, parent_id: &str) {
    for (name, child) in &node.children {
        let node_id = format!("{}_{}", parent_id, name.replace('.', "_"));

        // Node style based on type
        let (shape, color) = if child.is_leaf {
            ("box", "lightblue")
        } else if child.children.len() > 3 {
            ("folder", "lightyellow")
        } else {
            ("box", "white")
        };

        let label = if child.is_leaf {
            let shape_str = child
                .shape
                .as_ref()
                .map(|s| format!("{s:?}"))
                .unwrap_or_default();
            format!("{name}\\n{shape_str}")
        } else {
            format!("{}\\n({} tensors)", name, child.tensor_count())
        };

        println!(
            "  {node_id} [label=\"{label}\", shape={shape}, fillcolor=\"{color}\", style=filled];"
        );
        println!("  {parent_id} -> {node_id};");

        print_dot_nodes(child, &node_id);
    }
}

/// Print Mermaid diagram format
fn print_mermaid_graph(root: &TreeNode) {
    println!("```mermaid");
    println!("graph TD");

    print_mermaid_nodes(root, "root", "Model");

    println!("```");
}

#[allow(clippy::disallowed_methods)] // unwrap_or_default is safe for formatting
fn print_mermaid_nodes(node: &TreeNode, parent_id: &str, parent_label: &str) {
    if parent_id == "root" {
        println!("  {parent_id}[{parent_label}]");
    }

    for (i, (name, child)) in node.children.iter().enumerate() {
        let node_id = format!("{parent_id}_{i}");

        let label = if child.is_leaf {
            let shape_str = child
                .shape
                .as_ref()
                .map(|s| format!("{s:?}"))
                .unwrap_or_default();
            format!("{name}<br/>{shape_str}")
        } else {
            name.clone()
        };

        // Use different shapes for leaves vs containers
        let node_def = if child.is_leaf {
            format!("{node_id}[{label}]")
        } else {
            format!("{node_id}{{{{ {label} }}}}")
        };

        println!("  {parent_id} --> {node_def}");

        print_mermaid_nodes(child, &node_id, name);
    }
}

/// Print JSON tree
#[allow(clippy::disallowed_methods)] // json! macro uses infallible unwrap internally
fn print_json_tree(root: &TreeNode) {
    fn to_json(node: &TreeNode) -> serde_json::Value {
        if node.is_leaf {
            serde_json::json!({
                "name": node.name,
                "path": node.full_path,
                "shape": node.shape,
                "size_bytes": node.size_bytes,
                "type": "tensor"
            })
        } else {
            let children: Vec<_> = node.children.values().map(to_json).collect();
            serde_json::json!({
                "name": node.name,
                "path": node.full_path,
                "total_size": node.total_size(),
                "tensor_count": node.tensor_count(),
                "type": "group",
                "children": children
            })
        }
    }

    if let Ok(json) = serde_json::to_string_pretty(&to_json(root)) {
        println!("{json}");
    }
}

/// Format bytes as human-readable size
fn format_size(bytes: usize) -> String {
    batuta_common::fmt::format_bytes(bytes as u64)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[path = "tree_tests.rs"]
mod tests;
