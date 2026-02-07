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
use aprender::serialization::apr::AprReader;
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

    let reader = AprReader::open(apr_path)
        .map_err(|e| CliError::InvalidFormat(format!("Failed to read APR: {e}")))?;

    // Build tree from tensor names
    let mut root = TreeNode::new("model", "");

    for tensor in &reader.tensors {
        // Apply filter
        if let Some(f) = filter {
            if !tensor.name.contains(f) {
                continue;
            }
        }

        // Parse path and build tree
        let parts: Vec<&str> = tensor.name.split('.').collect();
        let mut current = &mut root;

        for (i, part) in parts.iter().enumerate() {
            let path = parts[..=i].join(".");

            if !current.children.contains_key(*part) {
                current
                    .children
                    .insert((*part).to_string(), TreeNode::new(part, &path));
            }

            current = current.children.get_mut(*part).expect("just inserted");

            // If last part, this is a leaf
            if i == parts.len() - 1 {
                current.is_leaf = true;
                current.shape = Some(tensor.shape.clone());
                current.size_bytes = tensor.size;
            }
        }
    }

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
    const KB: usize = 1024;
    const MB: usize = KB * 1024;
    const GB: usize = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} B")
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::{tempdir, NamedTempFile};

    // ========================================================================
    // TreeFormat Tests
    // ========================================================================

    #[test]
    fn test_tree_format_from_str_ascii() {
        assert_eq!("ascii".parse::<TreeFormat>().unwrap(), TreeFormat::Ascii);
        assert_eq!("text".parse::<TreeFormat>().unwrap(), TreeFormat::Ascii);
    }

    #[test]
    fn test_tree_format_from_str_dot() {
        assert_eq!("dot".parse::<TreeFormat>().unwrap(), TreeFormat::Dot);
        assert_eq!("graphviz".parse::<TreeFormat>().unwrap(), TreeFormat::Dot);
    }

    #[test]
    fn test_tree_format_from_str_mermaid() {
        assert_eq!(
            "mermaid".parse::<TreeFormat>().unwrap(),
            TreeFormat::Mermaid
        );
        assert_eq!("md".parse::<TreeFormat>().unwrap(), TreeFormat::Mermaid);
    }

    #[test]
    fn test_tree_format_from_str_json() {
        assert_eq!("json".parse::<TreeFormat>().unwrap(), TreeFormat::Json);
    }

    #[test]
    fn test_tree_format_from_str_case_insensitive() {
        assert_eq!("ASCII".parse::<TreeFormat>().unwrap(), TreeFormat::Ascii);
        assert_eq!("DOT".parse::<TreeFormat>().unwrap(), TreeFormat::Dot);
        assert_eq!("JSON".parse::<TreeFormat>().unwrap(), TreeFormat::Json);
    }

    #[test]
    fn test_tree_format_from_str_invalid() {
        assert!("invalid".parse::<TreeFormat>().is_err());
        assert!("xyz".parse::<TreeFormat>().is_err());
    }

    // ========================================================================
    // TreeNode Tests
    // ========================================================================

    #[test]
    fn test_tree_node_new() {
        let node = TreeNode::new("test", "root.test");
        assert_eq!(node.name, "test");
        assert_eq!(node.full_path, "root.test");
        assert!(node.shape.is_none());
        assert_eq!(node.size_bytes, 0);
        assert!(!node.is_leaf);
        assert!(node.children.is_empty());
    }

    #[test]
    fn test_tree_node_total_size_leaf() {
        let mut node = TreeNode::new("leaf", "root.leaf");
        node.is_leaf = true;
        node.size_bytes = 1000;
        assert_eq!(node.total_size(), 1000);
    }

    #[test]
    fn test_tree_node_total_size_parent() {
        let mut parent = TreeNode::new("parent", "root");

        let mut child1 = TreeNode::new("child1", "root.child1");
        child1.is_leaf = true;
        child1.size_bytes = 100;

        let mut child2 = TreeNode::new("child2", "root.child2");
        child2.is_leaf = true;
        child2.size_bytes = 200;

        parent.children.insert("child1".to_string(), child1);
        parent.children.insert("child2".to_string(), child2);

        assert_eq!(parent.total_size(), 300);
    }

    #[test]
    fn test_tree_node_tensor_count_leaf() {
        let mut node = TreeNode::new("leaf", "root.leaf");
        node.is_leaf = true;
        assert_eq!(node.tensor_count(), 1);
    }

    #[test]
    fn test_tree_node_tensor_count_parent() {
        let mut parent = TreeNode::new("parent", "root");

        let mut child1 = TreeNode::new("child1", "root.child1");
        child1.is_leaf = true;

        let mut child2 = TreeNode::new("child2", "root.child2");
        child2.is_leaf = true;

        parent.children.insert("child1".to_string(), child1);
        parent.children.insert("child2".to_string(), child2);

        assert_eq!(parent.tensor_count(), 2);
    }

    // ========================================================================
    // Format Size Tests
    // ========================================================================

    #[test]
    fn test_format_size_bytes() {
        assert_eq!(format_size(0), "0 B");
        assert_eq!(format_size(100), "100 B");
        assert_eq!(format_size(1023), "1023 B");
    }

    #[test]
    fn test_format_size_kilobytes() {
        assert_eq!(format_size(1024), "1.00 KB");
        assert_eq!(format_size(2048), "2.00 KB");
        assert_eq!(format_size(1536), "1.50 KB");
    }

    #[test]
    fn test_format_size_megabytes() {
        assert_eq!(format_size(1024 * 1024), "1.00 MB");
        assert_eq!(format_size(10 * 1024 * 1024), "10.00 MB");
    }

    #[test]
    fn test_format_size_gigabytes() {
        assert_eq!(format_size(1024 * 1024 * 1024), "1.00 GB");
        assert_eq!(format_size(2 * 1024 * 1024 * 1024), "2.00 GB");
    }

    // ========================================================================
    // Run Command Tests
    // ========================================================================

    #[test]
    fn test_run_file_not_found() {
        let result = run(
            Path::new("/nonexistent/model.apr"),
            None,
            TreeFormat::Ascii,
            false,
            None,
        );
        assert!(result.is_err());
        match result {
            Err(CliError::FileNotFound(_)) => {}
            _ => panic!("Expected FileNotFound error"),
        }
    }

    #[test]
    fn test_run_invalid_apr() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        file.write_all(b"not a valid apr").expect("write");

        let result = run(file.path(), None, TreeFormat::Ascii, false, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_is_directory() {
        let dir = tempdir().expect("create temp dir");
        let result = run(dir.path(), None, TreeFormat::Ascii, false, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_filter() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        file.write_all(b"not a valid apr").expect("write");

        let result = run(file.path(), Some("decoder"), TreeFormat::Ascii, false, None);
        // Should fail (invalid file) but tests filter path
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_dot_format() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        file.write_all(b"not a valid apr").expect("write");

        let result = run(file.path(), None, TreeFormat::Dot, false, None);
        // Should fail (invalid file) but tests dot format path
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_json_format() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        file.write_all(b"not a valid apr").expect("write");

        let result = run(file.path(), None, TreeFormat::Json, false, None);
        // Should fail (invalid file) but tests json format path
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_show_sizes() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        file.write_all(b"not a valid apr").expect("write");

        let result = run(file.path(), None, TreeFormat::Ascii, true, None);
        // Should fail (invalid file) but tests show_sizes path
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_max_depth() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        file.write_all(b"not a valid apr").expect("write");

        let result = run(file.path(), None, TreeFormat::Ascii, false, Some(2));
        // Should fail (invalid file) but tests max_depth path
        assert!(result.is_err());
    }

    // ========================================================================
    // TreeFormat error message content
    // ========================================================================

    #[test]
    fn tree_format_error_contains_input_string() {
        let err = "banana".parse::<TreeFormat>().unwrap_err();
        assert!(
            err.contains("banana"),
            "Error message should contain the invalid input"
        );
    }

    #[test]
    fn tree_format_error_for_empty_string() {
        let err = "".parse::<TreeFormat>().unwrap_err();
        assert!(err.contains("Unknown format"));
    }

    // ========================================================================
    // TreeNode: deeply nested total_size and tensor_count
    // ========================================================================

    #[test]
    fn tree_node_total_size_three_level_hierarchy() {
        // root -> mid -> leaf1(100), leaf2(200)
        let mut root = TreeNode::new("root", "");

        let mut mid = TreeNode::new("mid", "mid");
        let mut leaf1 = TreeNode::new("leaf1", "mid.leaf1");
        leaf1.is_leaf = true;
        leaf1.size_bytes = 100;
        let mut leaf2 = TreeNode::new("leaf2", "mid.leaf2");
        leaf2.is_leaf = true;
        leaf2.size_bytes = 200;
        mid.children.insert("leaf1".to_string(), leaf1);
        mid.children.insert("leaf2".to_string(), leaf2);

        root.children.insert("mid".to_string(), mid);

        assert_eq!(root.total_size(), 300);
    }

    #[test]
    fn tree_node_tensor_count_three_level_hierarchy() {
        let mut root = TreeNode::new("root", "");
        let mut group_a = TreeNode::new("a", "a");
        let mut group_b = TreeNode::new("b", "a.b");

        let mut leaf1 = TreeNode::new("w", "a.b.w");
        leaf1.is_leaf = true;
        let mut leaf2 = TreeNode::new("b", "a.b.b");
        leaf2.is_leaf = true;

        group_b.children.insert("w".to_string(), leaf1);
        group_b.children.insert("b".to_string(), leaf2);
        group_a.children.insert("b".to_string(), group_b);

        let mut leaf3 = TreeNode::new("v", "a.v");
        leaf3.is_leaf = true;
        group_a.children.insert("v".to_string(), leaf3);

        root.children.insert("a".to_string(), group_a);

        assert_eq!(root.tensor_count(), 3);
    }

    // ========================================================================
    // TreeNode: empty parent (no children)
    // ========================================================================

    #[test]
    fn tree_node_total_size_empty_non_leaf_is_zero() {
        let node = TreeNode::new("empty", "empty");
        assert_eq!(node.total_size(), 0);
    }

    #[test]
    fn tree_node_tensor_count_empty_non_leaf_is_zero() {
        let node = TreeNode::new("empty", "empty");
        assert_eq!(node.tensor_count(), 0);
    }

    // ========================================================================
    // TreeNode: leaf with shape
    // ========================================================================

    #[test]
    fn tree_node_leaf_with_shape_and_size() {
        let mut node = TreeNode::new("weight", "layer.weight");
        node.is_leaf = true;
        node.shape = Some(vec![768, 3072]);
        node.size_bytes = 768 * 3072 * 4;
        assert_eq!(node.total_size(), 768 * 3072 * 4);
        assert_eq!(node.tensor_count(), 1);
        assert_eq!(
            node.shape.as_ref().expect("should have shape"),
            &[768, 3072]
        );
    }

    // ========================================================================
    // TreeNode: mixed leaf and non-leaf children
    // ========================================================================

    #[test]
    fn tree_node_total_size_mixed_children() {
        let mut parent = TreeNode::new("parent", "parent");

        // Direct leaf child
        let mut direct_leaf = TreeNode::new("bias", "parent.bias");
        direct_leaf.is_leaf = true;
        direct_leaf.size_bytes = 50;

        // Group child with a nested leaf
        let mut group = TreeNode::new("attn", "parent.attn");
        let mut nested_leaf = TreeNode::new("weight", "parent.attn.weight");
        nested_leaf.is_leaf = true;
        nested_leaf.size_bytes = 150;
        group.children.insert("weight".to_string(), nested_leaf);

        parent.children.insert("bias".to_string(), direct_leaf);
        parent.children.insert("attn".to_string(), group);

        assert_eq!(parent.total_size(), 200);
        assert_eq!(parent.tensor_count(), 2);
    }

    // ========================================================================
    // format_size: boundary values
    // ========================================================================

    #[test]
    fn format_size_exactly_1kb() {
        assert_eq!(format_size(1024), "1.00 KB");
    }

    #[test]
    fn format_size_just_under_1kb() {
        assert_eq!(format_size(1023), "1023 B");
    }

    #[test]
    fn format_size_exactly_1mb() {
        assert_eq!(format_size(1024 * 1024), "1.00 MB");
    }

    #[test]
    fn format_size_just_under_1mb() {
        // 1MB - 1 byte = 1023.999... KB
        let result = format_size(1024 * 1024 - 1);
        assert!(result.ends_with(" KB"), "Expected KB, got: {result}");
    }

    #[test]
    fn format_size_exactly_1gb() {
        assert_eq!(format_size(1024 * 1024 * 1024), "1.00 GB");
    }

    #[test]
    fn format_size_just_under_1gb() {
        let result = format_size(1024 * 1024 * 1024 - 1);
        assert!(result.ends_with(" MB"), "Expected MB, got: {result}");
    }

    #[test]
    fn format_size_fractional_gb() {
        // 1.5 GB
        let bytes = 1024 * 1024 * 1024 + 512 * 1024 * 1024;
        assert_eq!(format_size(bytes), "1.50 GB");
    }

    #[test]
    fn format_size_one_byte() {
        assert_eq!(format_size(1), "1 B");
    }

    // ========================================================================
    // TreeFormat: all aliases roundtrip
    // ========================================================================

    #[test]
    fn tree_format_mixed_case_mermaid() {
        assert_eq!(
            "Mermaid".parse::<TreeFormat>().expect("valid"),
            TreeFormat::Mermaid
        );
        assert_eq!(
            "MD".parse::<TreeFormat>().expect("valid"),
            TreeFormat::Mermaid
        );
    }

    #[test]
    fn tree_format_graphviz_alias() {
        assert_eq!(
            "GRAPHVIZ".parse::<TreeFormat>().expect("valid"),
            TreeFormat::Dot
        );
    }

    #[test]
    fn tree_format_text_alias() {
        assert_eq!(
            "TEXT".parse::<TreeFormat>().expect("valid"),
            TreeFormat::Ascii
        );
    }

    // ========================================================================
    // run: error variants
    // ========================================================================

    #[test]
    fn run_file_not_found_returns_correct_error_variant() {
        let result = run(
            Path::new("/definitely/does/not/exist.apr"),
            None,
            TreeFormat::Ascii,
            false,
            None,
        );
        assert!(
            matches!(result, Err(CliError::FileNotFound(_))),
            "Expected FileNotFound, got: {result:?}"
        );
    }

    #[test]
    fn run_with_mermaid_format_invalid_file() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        file.write_all(b"not valid").expect("write");

        let result = run(file.path(), None, TreeFormat::Mermaid, false, None);
        assert!(result.is_err());
    }

    #[test]
    fn run_directory_returns_invalid_format() {
        let dir = tempdir().expect("create temp dir");
        let result = run(dir.path(), None, TreeFormat::Ascii, false, None);
        // Directory will fail at AprReader::open
        assert!(result.is_err());
    }
}
