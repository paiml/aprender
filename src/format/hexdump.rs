//! Hex dump and data flow visualization for tensor inspection (GH-122).
//!
// Allow format_push_string: cleaner code for string building without I/O concerns
#![allow(clippy::format_push_string)]
//!
//! Implements Toyota Way Principle 12 (Genchi Genbutsu): Go and see the actual
//! tensor values, not abstractions.
//!
//! # Features
//!
//! - Hex dump with ASCII sidebar
//! - Data flow visualization
//! - Model hierarchy tree view
//! - Tensor statistics summary
//!
//! # Example
//!
//! ```rust
//! use aprender::format::hexdump::{hex_dump, HexDumpConfig};
//!
//! let data = [0x41, 0x50, 0x52, 0x31]; // "APR1"
//! let dump = hex_dump(&data, &HexDumpConfig::default());
//! assert!(dump.contains("41 50 52 31"));
//! assert!(dump.contains("APR1"));
//! ```
//!
//! # PMAT Compliance
//!
//! - Zero `unwrap()` calls
//! - All string formatting is safe

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for hex dump display
#[derive(Debug, Clone)]
pub struct HexDumpConfig {
    /// Bytes per line (default: 16)
    pub bytes_per_line: usize,
    /// Show ASCII sidebar (default: true)
    pub show_ascii: bool,
    /// Show offset column (default: true)
    pub show_offset: bool,
    /// Group bytes (e.g., 2 = pairs, 4 = quads)
    pub group_size: usize,
    /// Maximum bytes to dump (0 = unlimited)
    pub max_bytes: usize,
}

impl Default for HexDumpConfig {
    fn default() -> Self {
        Self {
            bytes_per_line: 16,
            show_ascii: true,
            show_offset: true,
            group_size: 1,
            max_bytes: 0,
        }
    }
}

impl HexDumpConfig {
    /// Compact hex dump (8 bytes per line, no offset)
    #[must_use]
    pub fn compact() -> Self {
        Self {
            bytes_per_line: 8,
            show_ascii: true,
            show_offset: false,
            group_size: 1,
            max_bytes: 256,
        }
    }

    /// Wide hex dump (32 bytes per line)
    #[must_use]
    pub fn wide() -> Self {
        Self {
            bytes_per_line: 32,
            show_ascii: true,
            show_offset: true,
            group_size: 4,
            max_bytes: 0,
        }
    }
}

// ============================================================================
// Hex Dump
// ============================================================================

/// Generate hex dump of byte slice.
///
/// Returns multi-line string with hex values and optional ASCII sidebar.
#[must_use]
pub fn hex_dump(data: &[u8], config: &HexDumpConfig) -> String {
    let mut result = String::new();
    let bytes_per_line = config.bytes_per_line.max(1);
    let max_bytes = if config.max_bytes > 0 {
        config.max_bytes.min(data.len())
    } else {
        data.len()
    };

    for (line_idx, chunk) in data[..max_bytes].chunks(bytes_per_line).enumerate() {
        if config.show_offset {
            result.push_str(&format!("{:08x}  ", line_idx * bytes_per_line));
        }
        format_hex_bytes(&mut result, chunk, bytes_per_line, config.group_size);
        format_ascii_sidebar(&mut result, chunk, bytes_per_line, config.show_ascii);
        result.push('\n');
    }

    if max_bytes < data.len() {
        result.push_str(&format!(
            "... ({} more bytes truncated)\n",
            data.len() - max_bytes
        ));
    }

    result
}

/// Format hex bytes with optional grouping and padding for incomplete lines.
fn format_hex_bytes(result: &mut String, chunk: &[u8], bytes_per_line: usize, group_size: usize) {
    for (i, byte) in chunk.iter().enumerate() {
        if group_size > 1 && i > 0 && i % group_size == 0 {
            result.push(' ');
        }
        result.push_str(&format!("{byte:02x} "));
    }
    for i in 0..(bytes_per_line - chunk.len()) {
        if group_size > 1 && (chunk.len() + i) % group_size == 0 {
            result.push(' ');
        }
        result.push_str("   ");
    }
}

/// Format ASCII sidebar with padding for incomplete lines.
fn format_ascii_sidebar(result: &mut String, chunk: &[u8], bytes_per_line: usize, show: bool) {
    if !show {
        return;
    }
    result.push_str(" |");
    for byte in chunk {
        if byte.is_ascii_graphic() || *byte == b' ' {
            result.push(*byte as char);
        } else {
            result.push('.');
        }
    }
    for _ in chunk.len()..bytes_per_line {
        result.push(' ');
    }
    result.push('|');
}

/// Generate hex dump of f32 tensor values.
///
/// Shows both raw bytes and float interpretation.
#[must_use]
pub fn tensor_hex_dump(tensor: &[f32], config: &HexDumpConfig) -> String {
    let mut result = String::new();
    let values_per_line = (config.bytes_per_line / 4).max(1);
    let max_values = if config.max_bytes > 0 {
        (config.max_bytes / 4).min(tensor.len())
    } else {
        tensor.len()
    };

    let truncated = max_values < tensor.len();

    for (line_idx, chunk) in tensor[..max_values].chunks(values_per_line).enumerate() {
        // Offset column (in float indices)
        if config.show_offset {
            result.push_str(&format!("[{:06}]  ", line_idx * values_per_line));
        }

        // Float values
        for val in chunk {
            result.push_str(&format!("{val:>12.6e} "));
        }

        // Hex representation
        result.push_str("  |");
        for val in chunk {
            let bytes = val.to_le_bytes();
            result.push_str(&format!(
                " {:02x}{:02x}{:02x}{:02x}",
                bytes[3], bytes[2], bytes[1], bytes[0]
            ));
        }
        result.push_str(" |");

        result.push('\n');
    }

    if truncated {
        result.push_str(&format!(
            "... ({} more values truncated)\n",
            tensor.len() - max_values
        ));
    }

    result
}

// ============================================================================
// Data Flow Visualization
// ============================================================================

/// Layer information for data flow visualization
#[derive(Debug, Clone)]
pub struct LayerInfo {
    /// Layer name
    pub name: String,
    /// Layer type (e.g., "Conv2d", "Linear", "`LayerNorm`")
    pub layer_type: String,
    /// Input shape
    pub input_shape: Vec<usize>,
    /// Output shape
    pub output_shape: Vec<usize>,
    /// Parameter count
    pub params: usize,
}

impl LayerInfo {
    /// Create new layer info
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        layer_type: impl Into<String>,
        input_shape: Vec<usize>,
        output_shape: Vec<usize>,
        params: usize,
    ) -> Self {
        Self {
            name: name.into(),
            layer_type: layer_type.into(),
            input_shape,
            output_shape,
            params,
        }
    }

    /// Format shape as string
    fn format_shape(shape: &[usize]) -> String {
        if shape.is_empty() {
            "()".to_string()
        } else {
            format!(
                "({})",
                shape
                    .iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        }
    }
}

/// Generate data flow visualization for model layers.
///
/// Returns ASCII art showing layer sequence with shapes.
#[must_use]
pub fn data_flow_diagram(layers: &[LayerInfo]) -> String {
    let mut result = String::new();

    result.push_str("Data Flow Diagram\n");
    result.push_str("=================\n\n");

    if layers.is_empty() {
        result.push_str("(no layers)\n");
        return result;
    }

    // Find max widths for alignment
    let max_name_len = layers.iter().map(|l| l.name.len()).max().unwrap_or(10);
    let max_type_len = layers
        .iter()
        .map(|l| l.layer_type.len())
        .max()
        .unwrap_or(10);

    for (i, layer) in layers.iter().enumerate() {
        let input_str = LayerInfo::format_shape(&layer.input_shape);
        let output_str = LayerInfo::format_shape(&layer.output_shape);

        // Input arrow (except first layer)
        if i == 0 {
            result.push_str(&format!("    Input: {input_str}\n"));
            result.push_str("       |\n");
            result.push_str("       v\n");
        }

        // Layer box
        result.push_str(&format!(
            "  +{:-<width$}+\n",
            "",
            width = max_name_len + max_type_len + 10
        ));
        result.push_str(&format!(
            "  | {:name_width$} [{:type_width$}] {:>8} |\n",
            layer.name,
            layer.layer_type,
            format_params(layer.params),
            name_width = max_name_len,
            type_width = max_type_len
        ));
        result.push_str(&format!(
            "  +{:-<width$}+\n",
            "",
            width = max_name_len + max_type_len + 10
        ));

        // Output arrow
        result.push_str("       |\n");
        if i == layers.len() - 1 {
            result.push_str("       v\n");
            result.push_str(&format!("    Output: {output_str}\n"));
        } else {
            result.push_str(&format!("       | {output_str}\n"));
            result.push_str("       v\n");
        }
    }

    // Summary
    let total_params: usize = layers.iter().map(|l| l.params).sum();
    result.push_str(&format!(
        "\nTotal parameters: {}\n",
        format_params(total_params)
    ));

    result
}

/// Format parameter count with K/M/B suffixes
fn format_params(count: usize) -> String {
    if count >= 1_000_000_000 {
        format!("{:.1}B", count as f64 / 1_000_000_000.0)
    } else if count >= 1_000_000 {
        format!("{:.1}M", count as f64 / 1_000_000.0)
    } else if count >= 1_000 {
        format!("{:.1}K", count as f64 / 1_000.0)
    } else {
        count.to_string()
    }
}

// ============================================================================
// Model Tree View
// ============================================================================

/// Node in model hierarchy tree
#[derive(Debug, Clone)]
pub struct TreeNode {
    /// Node name
    pub name: String,
    /// Node type
    pub node_type: String,
    /// Child nodes
    pub children: Vec<TreeNode>,
    /// Tensor shape (if leaf node)
    pub shape: Option<Vec<usize>>,
    /// Data type (if leaf node)
    pub dtype: Option<String>,
}

impl TreeNode {
    /// Create new tree node
    #[must_use]
    pub fn new(name: impl Into<String>, node_type: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            node_type: node_type.into(),
            children: Vec::new(),
            shape: None,
            dtype: None,
        }
    }

    /// Create leaf node (tensor)
    #[must_use]
    pub fn tensor(name: impl Into<String>, shape: Vec<usize>, dtype: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            node_type: "Tensor".to_string(),
            children: Vec::new(),
            shape: Some(shape),
            dtype: Some(dtype.into()),
        }
    }

    /// Add child node
    pub fn add_child(&mut self, child: TreeNode) {
        self.children.push(child);
    }

    /// Count total nodes
    #[must_use]
    pub fn count_nodes(&self) -> usize {
        1 + self
            .children
            .iter()
            .map(TreeNode::count_nodes)
            .sum::<usize>()
    }
}

/// Generate tree view of model hierarchy.
///
/// Returns ASCII tree representation.
#[must_use]
pub fn tree_view(root: &TreeNode) -> String {
    let mut result = String::new();
    tree_view_recursive(root, "", true, &mut result);
    result
}

include!("hexdump_part_02.rs");
