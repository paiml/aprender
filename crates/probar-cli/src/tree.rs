//! File Tree Visualization
//!
//! Displays the directory structure of served files with MIME types and sizes.
//!
//! ## Example Output
//!
//! ```text
//! demos/realtime-transcription/
//! ├── index.html (2.3 KB) [text/html]
//! ├── styles.css (1.1 KB) [text/css]
//! ├── pkg/
//! │   ├── realtime_wasm.js (45 KB) [text/javascript]
//! │   └── realtime_wasm_bg.wasm (1.2 MB) [application/wasm]
//! └── worker.js (5.6 KB) [text/javascript]
//!
//! Total: 5 files, 1.3 MB
//! ```

#![allow(clippy::must_use_candidate)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::use_self)]
#![allow(clippy::format_push_string)]
#![allow(clippy::missing_const_for_fn)]
#![allow(clippy::needless_continue)]
#![allow(clippy::map_unwrap_or)]

use crate::dev_server::get_mime_type;
use glob::Pattern;
use std::path::{Path, PathBuf};

/// File node in the tree
#[derive(Debug, Clone)]
pub struct FileNode {
    /// File or directory name
    pub name: String,
    /// Full path
    pub path: PathBuf,
    /// File size in bytes (0 for directories)
    pub size: u64,
    /// MIME type (empty for directories)
    pub mime_type: String,
    /// Whether this is a directory
    pub is_dir: bool,
    /// Child nodes
    pub children: Vec<FileNode>,
}

impl FileNode {
    /// Create a new file node
    #[must_use]
    pub fn new_file(path: PathBuf, size: u64) -> Self {
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        let mime_type = get_mime_type(&path);

        Self {
            name,
            path,
            size,
            mime_type,
            is_dir: false,
            children: Vec::new(),
        }
    }

    /// Create a new directory node
    #[must_use]
    pub fn new_dir(path: PathBuf) -> Self {
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| path.to_string_lossy().to_string());

        Self {
            name,
            path,
            size: 0,
            mime_type: String::new(),
            is_dir: true,
            children: Vec::new(),
        }
    }

    /// Get total size including children
    #[must_use]
    pub fn total_size(&self) -> u64 {
        if self.is_dir {
            self.children.iter().map(FileNode::total_size).sum()
        } else {
            self.size
        }
    }

    /// Count total files (excluding directories)
    #[must_use]
    pub fn file_count(&self) -> usize {
        if self.is_dir {
            self.children.iter().map(FileNode::file_count).sum()
        } else {
            1
        }
    }
}

/// Configuration for tree display
#[derive(Debug, Clone)]
pub struct TreeConfig {
    /// Maximum depth to display (None = unlimited)
    pub max_depth: Option<usize>,
    /// Filter pattern (glob)
    pub filter: Option<Pattern>,
    /// Show file sizes
    pub show_sizes: bool,
    /// Show MIME types
    pub show_mime_types: bool,
    /// Use colors
    pub use_colors: bool,
}

impl Default for TreeConfig {
    fn default() -> Self {
        Self {
            max_depth: None,
            filter: None,
            show_sizes: true,
            show_mime_types: true,
            use_colors: atty::is(atty::Stream::Stdout),
        }
    }
}

impl TreeConfig {
    /// Set maximum depth
    #[must_use]
    pub fn with_depth(mut self, depth: Option<usize>) -> Self {
        self.max_depth = depth;
        self
    }

    /// Set filter pattern
    #[must_use]
    pub fn with_filter(mut self, pattern: Option<&str>) -> Self {
        self.filter = pattern.and_then(|p| Pattern::new(p).ok());
        self
    }

    /// Set whether to show sizes
    #[must_use]
    pub const fn with_sizes(mut self, show: bool) -> Self {
        self.show_sizes = show;
        self
    }

    /// Set whether to show MIME types
    #[must_use]
    pub const fn with_mime_types(mut self, show: bool) -> Self {
        self.show_mime_types = show;
        self
    }
}

/// Build a file tree from a directory
///
/// # Errors
///
/// Returns an error if the path cannot be read or doesn't exist.
pub fn build_tree(root: &Path, config: &TreeConfig) -> Result<FileNode, std::io::Error> {
    build_tree_recursive(root, config, 0)
}

fn build_tree_recursive(
    path: &Path,
    config: &TreeConfig,
    current_depth: usize,
) -> Result<FileNode, std::io::Error> {
    let metadata = std::fs::metadata(path)?;

    if metadata.is_file() {
        // Check filter
        if let Some(ref pattern) = config.filter {
            let name = path
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default();
            if !pattern.matches(&name) {
                // Return empty node that will be filtered
                return Ok(FileNode {
                    name: String::new(),
                    path: path.to_path_buf(),
                    size: 0,
                    mime_type: String::new(),
                    is_dir: false,
                    children: Vec::new(),
                });
            }
        }

        return Ok(FileNode::new_file(path.to_path_buf(), metadata.len()));
    }

    // Directory
    let mut node = FileNode::new_dir(path.to_path_buf());

    // Check depth limit
    if let Some(max_depth) = config.max_depth {
        if current_depth >= max_depth {
            return Ok(node);
        }
    }

    // Read directory contents
    let mut entries: Vec<_> = std::fs::read_dir(path)?.filter_map(Result::ok).collect();

    // Sort: directories first, then alphabetically
    entries.sort_by(|a, b| {
        let a_is_dir = a.path().is_dir();
        let b_is_dir = b.path().is_dir();
        match (a_is_dir, b_is_dir) {
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            _ => a.file_name().cmp(&b.file_name()),
        }
    });

    for entry in entries {
        let child_path = entry.path();

        // Skip hidden files and common ignore patterns
        let name = child_path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();

        if name.starts_with('.') || name == "node_modules" || name == "target" {
            continue;
        }

        match build_tree_recursive(&child_path, config, current_depth + 1) {
            Ok(child) => {
                // Filter out empty nodes (filtered files)
                if !child.name.is_empty() {
                    node.children.push(child);
                }
            }
            Err(_) => continue, // Skip unreadable entries
        }
    }

    Ok(node)
}

/// Format a file size for display
#[must_use]
pub fn format_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} B")
    }
}

/// Render the tree to a string
#[must_use]
pub fn render_tree(root: &FileNode, config: &TreeConfig) -> String {
    let mut output = String::new();

    // Root directory name
    output.push_str(&root.name);
    output.push_str("/\n");

    // Render children
    render_node_children(&root.children, config, "", &mut output);

    // Summary line
    let total_files = root.file_count();
    let total_size = root.total_size();
    output.push('\n');
    output.push_str(&format!(
        "Total: {} files, {}\n",
        total_files,
        format_size(total_size)
    ));

    output
}

fn render_node_children(
    children: &[FileNode],
    config: &TreeConfig,
    prefix: &str,
    output: &mut String,
) {
    let len = children.len();

    for (i, child) in children.iter().enumerate() {
        let is_last = i == len - 1;
        let connector = if is_last { "└── " } else { "├── " };
        let child_prefix = if is_last { "    " } else { "│   " };

        // Build line
        output.push_str(prefix);
        output.push_str(connector);
        output.push_str(&child.name);

        if child.is_dir {
            output.push('/');
        } else {
            // File info
            if config.show_sizes {
                output.push_str(&format!(" ({})", format_size(child.size)));
            }
            if config.show_mime_types && !child.mime_type.is_empty() {
                output.push_str(&format!(" [{}]", child.mime_type));
            }
        }

        output.push('\n');

        // Recurse for directories
        if child.is_dir && !child.children.is_empty() {
            let new_prefix = format!("{prefix}{child_prefix}");
            render_node_children(&child.children, config, &new_prefix, output);
        }
    }
}

/// Display the file tree to stdout
///
/// # Errors
///
/// Returns an error if the path cannot be read.
pub fn display_tree(root: &Path, config: &TreeConfig) -> Result<(), std::io::Error> {
    let tree = build_tree(root, config)?;
    let output = render_tree(&tree, config);
    print!("{output}");
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_format_size() {
        assert_eq!(format_size(0), "0 B");
        assert_eq!(format_size(512), "512 B");
        assert_eq!(format_size(1024), "1.0 KB");
        assert_eq!(format_size(1536), "1.5 KB");
        assert_eq!(format_size(1_048_576), "1.0 MB");
        assert_eq!(format_size(1_073_741_824), "1.0 GB");
    }

    #[test]
    fn test_file_node_new_file() {
        let node = FileNode::new_file(PathBuf::from("test.html"), 1024);
        assert_eq!(node.name, "test.html");
        assert_eq!(node.size, 1024);
        assert_eq!(node.mime_type, "text/html");
        assert!(!node.is_dir);
    }

    #[test]
    fn test_file_node_new_dir() {
        let node = FileNode::new_dir(PathBuf::from("src"));
        assert_eq!(node.name, "src");
        assert!(node.is_dir);
        assert!(node.children.is_empty());
    }

    #[test]
    fn test_tree_config_default() {
        let config = TreeConfig::default();
        assert!(config.max_depth.is_none());
        assert!(config.filter.is_none());
        assert!(config.show_sizes);
        assert!(config.show_mime_types);
    }

    #[test]
    fn test_tree_config_builder() {
        let config = TreeConfig::default()
            .with_depth(Some(2))
            .with_filter(Some("*.rs"))
            .with_sizes(false)
            .with_mime_types(false);

        assert_eq!(config.max_depth, Some(2));
        assert!(config.filter.is_some());
        assert!(!config.show_sizes);
        assert!(!config.show_mime_types);
    }

    #[test]
    fn test_build_tree_simple() {
        let temp = TempDir::new().unwrap();
        std::fs::write(temp.path().join("index.html"), "<html></html>").unwrap();
        std::fs::write(temp.path().join("style.css"), "body {}").unwrap();

        let config = TreeConfig::default();
        let tree = build_tree(temp.path(), &config).unwrap();

        assert!(tree.is_dir);
        assert_eq!(tree.children.len(), 2);
    }

    #[test]
    fn test_build_tree_nested() {
        let temp = TempDir::new().unwrap();
        let subdir = temp.path().join("pkg");
        std::fs::create_dir(&subdir).unwrap();
        std::fs::write(subdir.join("app.js"), "console.log('hi')").unwrap();

        let config = TreeConfig::default();
        let tree = build_tree(temp.path(), &config).unwrap();

        assert_eq!(tree.children.len(), 1);
        assert!(tree.children[0].is_dir);
        assert_eq!(tree.children[0].children.len(), 1);
    }

    #[test]
    fn test_build_tree_with_depth_limit() {
        let temp = TempDir::new().unwrap();
        let subdir = temp.path().join("deep");
        std::fs::create_dir(&subdir).unwrap();
        std::fs::write(subdir.join("file.txt"), "content").unwrap();

        let config = TreeConfig::default().with_depth(Some(0));
        let tree = build_tree(temp.path(), &config).unwrap();

        // Should not recurse into directories
        assert!(tree.children.is_empty() || tree.children.iter().all(|c| c.children.is_empty()));
    }

    #[test]
    fn test_build_tree_with_filter() {
        let temp = TempDir::new().unwrap();
        std::fs::write(temp.path().join("app.js"), "js").unwrap();
        std::fs::write(temp.path().join("style.css"), "css").unwrap();
        std::fs::write(temp.path().join("index.html"), "html").unwrap();

        let config = TreeConfig::default().with_filter(Some("*.js"));
        let tree = build_tree(temp.path(), &config).unwrap();

        // Should only include .js files
        assert_eq!(tree.file_count(), 1);
    }

    #[test]
    fn test_render_tree() {
        let mut root = FileNode::new_dir(PathBuf::from("project"));
        root.children.push(FileNode::new_file(
            PathBuf::from("project/index.html"),
            1024,
        ));
        root.children
            .push(FileNode::new_file(PathBuf::from("project/app.js"), 2048));

        let config = TreeConfig::default();
        let output = render_tree(&root, &config);

        assert!(output.contains("project/"));
        assert!(output.contains("index.html"));
        assert!(output.contains("app.js"));
        assert!(output.contains("Total:"));
    }

    #[test]
    fn test_file_node_total_size() {
        let mut root = FileNode::new_dir(PathBuf::from("root"));
        root.children
            .push(FileNode::new_file(PathBuf::from("a.txt"), 100));
        root.children
            .push(FileNode::new_file(PathBuf::from("b.txt"), 200));

        assert_eq!(root.total_size(), 300);
    }

    #[test]
    fn test_file_node_file_count() {
        let mut root = FileNode::new_dir(PathBuf::from("root"));
        let mut subdir = FileNode::new_dir(PathBuf::from("sub"));
        subdir
            .children
            .push(FileNode::new_file(PathBuf::from("a.txt"), 100));
        root.children.push(subdir);
        root.children
            .push(FileNode::new_file(PathBuf::from("b.txt"), 100));

        assert_eq!(root.file_count(), 2);
    }

    // Additional coverage tests

    #[test]
    fn test_file_node_new_file_empty_name() {
        let node = FileNode::new_file(PathBuf::from("/"), 0);
        assert_eq!(node.name, "");
    }

    #[test]
    fn test_file_node_new_dir_no_filename() {
        // Path like "/" has no file_name component
        let node = FileNode::new_dir(PathBuf::from("/"));
        assert_eq!(node.name, "/");
    }

    #[test]
    fn test_file_node_total_size_single_file() {
        let node = FileNode::new_file(PathBuf::from("test.txt"), 500);
        assert_eq!(node.total_size(), 500);
    }

    #[test]
    fn test_file_node_file_count_single_file() {
        let node = FileNode::new_file(PathBuf::from("test.txt"), 100);
        assert_eq!(node.file_count(), 1);
    }

    #[test]
    fn test_file_node_file_count_empty_dir() {
        let node = FileNode::new_dir(PathBuf::from("empty"));
        assert_eq!(node.file_count(), 0);
    }

    #[test]
    fn test_tree_config_invalid_filter() {
        let config = TreeConfig::default().with_filter(Some("[invalid"));
        assert!(config.filter.is_none());
    }

    #[test]
    fn test_tree_config_none_filter() {
        let config = TreeConfig::default().with_filter(None);
        assert!(config.filter.is_none());
    }

    #[test]
    fn test_render_tree_nested_directories() {
        let mut root = FileNode::new_dir(PathBuf::from("project"));
        let mut subdir = FileNode::new_dir(PathBuf::from("project/src"));
        subdir.children.push(FileNode::new_file(
            PathBuf::from("project/src/main.rs"),
            512,
        ));
        root.children.push(subdir);
        root.children
            .push(FileNode::new_file(PathBuf::from("project/README.md"), 256));

        let config = TreeConfig::default();
        let output = render_tree(&root, &config);

        assert!(output.contains("src/"));
        assert!(output.contains("main.rs"));
        assert!(output.contains("README.md"));
        assert!(output.contains("│"));
    }

    #[test]
    fn test_render_tree_no_sizes() {
        let mut root = FileNode::new_dir(PathBuf::from("project"));
        root.children
            .push(FileNode::new_file(PathBuf::from("project/test.txt"), 1024));

        let config = TreeConfig::default().with_sizes(false);
        let output = render_tree(&root, &config);

        assert!(output.contains("test.txt"));
        // Size should not be in parentheses next to filename
        // (summary line still includes total size)
        assert!(!output.contains("(1.0 KB)"));
    }

    #[test]
    fn test_render_tree_no_mime_types() {
        let mut root = FileNode::new_dir(PathBuf::from("project"));
        root.children
            .push(FileNode::new_file(PathBuf::from("project/test.html"), 1024));

        let config = TreeConfig::default().with_mime_types(false);
        let output = render_tree(&root, &config);

        assert!(output.contains("test.html"));
        assert!(!output.contains("[text/html]"));
    }

    #[test]
    fn test_build_tree_hidden_files() {
        let temp = TempDir::new().unwrap();
        std::fs::write(temp.path().join(".hidden"), "secret").unwrap();
        std::fs::write(temp.path().join("visible.txt"), "public").unwrap();

        let config = TreeConfig::default();
        let tree = build_tree(temp.path(), &config).unwrap();

        // Hidden files should be excluded
        assert_eq!(tree.file_count(), 1);
        assert_eq!(tree.children[0].name, "visible.txt");
    }

    #[test]
    fn test_build_tree_ignores_node_modules() {
        let temp = TempDir::new().unwrap();
        let nm = temp.path().join("node_modules");
        std::fs::create_dir(&nm).unwrap();
        std::fs::write(nm.join("package.json"), "{}").unwrap();
        std::fs::write(temp.path().join("index.js"), "code").unwrap();

        let config = TreeConfig::default();
        let tree = build_tree(temp.path(), &config).unwrap();

        // node_modules should be excluded
        assert_eq!(tree.children.len(), 1);
        assert_eq!(tree.children[0].name, "index.js");
    }

    #[test]
    fn test_build_tree_ignores_target() {
        let temp = TempDir::new().unwrap();
        let target = temp.path().join("target");
        std::fs::create_dir(&target).unwrap();
        std::fs::write(target.join("debug"), "binary").unwrap();
        std::fs::write(temp.path().join("Cargo.toml"), "[package]").unwrap();

        let config = TreeConfig::default();
        let tree = build_tree(temp.path(), &config).unwrap();

        // target should be excluded
        assert_eq!(tree.children.len(), 1);
        assert_eq!(tree.children[0].name, "Cargo.toml");
    }

    #[test]
    fn test_build_tree_nonexistent_path() {
        let config = TreeConfig::default();
        let result = build_tree(Path::new("/nonexistent/path"), &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_build_tree_file_instead_of_directory() {
        let temp = TempDir::new().unwrap();
        let file_path = temp.path().join("file.txt");
        std::fs::write(&file_path, "content").unwrap();

        let config = TreeConfig::default();
        let tree = build_tree(&file_path, &config).unwrap();

        assert!(!tree.is_dir);
        assert_eq!(tree.name, "file.txt");
    }

    #[test]
    fn test_render_tree_empty_mime_type() {
        let mut root = FileNode::new_dir(PathBuf::from("project"));
        let mut file = FileNode::new_file(PathBuf::from("project/unknown"), 100);
        file.mime_type = String::new(); // Empty mime type
        root.children.push(file);

        let config = TreeConfig::default().with_mime_types(true);
        let output = render_tree(&root, &config);

        // Should not have brackets for empty mime type
        assert!(output.contains("unknown"));
        assert!(!output.contains("[]"));
    }

    #[test]
    fn test_display_tree() {
        let temp = TempDir::new().unwrap();
        std::fs::write(temp.path().join("test.txt"), "content").unwrap();

        let config = TreeConfig::default();
        let result = display_tree(temp.path(), &config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_display_tree_error() {
        let config = TreeConfig::default();
        let result = display_tree(Path::new("/nonexistent/path"), &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_format_size_large_gigabytes() {
        assert_eq!(format_size(10_737_418_240), "10.0 GB");
    }

    #[test]
    fn test_format_size_precise() {
        assert_eq!(format_size(1_572_864), "1.5 MB");
    }

    #[test]
    fn test_tree_directories_sorted_first() {
        let temp = TempDir::new().unwrap();
        std::fs::write(temp.path().join("aaa.txt"), "content").unwrap();
        let dir = temp.path().join("bbb");
        std::fs::create_dir(&dir).unwrap();

        let config = TreeConfig::default();
        let tree = build_tree(temp.path(), &config).unwrap();

        // Directory should come before file despite alphabetical order
        assert_eq!(tree.children.len(), 2);
        assert!(tree.children[0].is_dir);
        assert!(!tree.children[1].is_dir);
    }

    #[test]
    fn test_render_tree_multiple_files_last_item() {
        let mut root = FileNode::new_dir(PathBuf::from("project"));
        root.children
            .push(FileNode::new_file(PathBuf::from("a.txt"), 100));
        root.children
            .push(FileNode::new_file(PathBuf::from("b.txt"), 100));
        root.children
            .push(FileNode::new_file(PathBuf::from("c.txt"), 100));

        let config = TreeConfig::default();
        let output = render_tree(&root, &config);

        // Last item should use └── connector
        assert!(output.contains("└── c.txt"));
        // Middle items should use ├── connector
        assert!(output.contains("├── a.txt"));
        assert!(output.contains("├── b.txt"));
    }
}
