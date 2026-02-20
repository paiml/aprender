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
    assert_eq!(format_size(1024), "1.0 KB");
    assert_eq!(format_size(2048), "2.0 KB");
    assert_eq!(format_size(1536), "1.5 KB");
}

#[test]
fn test_format_size_megabytes() {
    assert_eq!(format_size(1024 * 1024), "1.0 MB");
    assert_eq!(format_size(10 * 1024 * 1024), "10.0 MB");
}

#[test]
fn test_format_size_gigabytes() {
    assert_eq!(format_size(1024 * 1024 * 1024), "1.0 GB");
    assert_eq!(format_size(2 * 1024 * 1024 * 1024), "2.0 GB");
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
    assert_eq!(format_size(1024), "1.0 KB");
}

#[test]
fn format_size_just_under_1kb() {
    assert_eq!(format_size(1023), "1023 B");
}

#[test]
fn format_size_exactly_1mb() {
    assert_eq!(format_size(1024 * 1024), "1.0 MB");
}

#[test]
fn format_size_just_under_1mb() {
    // 1MB - 1 byte = 1023.999... KB
    let result = format_size(1024 * 1024 - 1);
    assert!(result.ends_with(" KB"), "Expected KB, got: {result}");
}

#[test]
fn format_size_exactly_1gb() {
    assert_eq!(format_size(1024 * 1024 * 1024), "1.0 GB");
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
    assert_eq!(format_size(bytes), "1.5 GB");
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
