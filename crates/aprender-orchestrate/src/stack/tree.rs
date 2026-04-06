//! Stack Tree View - Visual hierarchical representation of PAIML stack
//!
//! Implements spec: docs/specifications/stack-tree-view.md

use serde::{Deserialize, Serialize};
use std::fmt;

// ============================================================================
// TREE-001: Core Types
// ============================================================================

/// Health status of a component
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HealthStatus {
    /// Local and remote versions match
    Synced,
    /// Local version is behind remote
    Behind,
    /// Local version is ahead of remote
    Ahead,
    /// Crate not found on crates.io
    NotFound,
    /// Error checking status
    Error(String),
}

impl fmt::Display for HealthStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Synced => write!(f, "✓"),
            Self::Behind => write!(f, "⚠"),
            Self::Ahead => write!(f, "↑"),
            Self::NotFound => write!(f, "?"),
            Self::Error(_) => write!(f, "✗"),
        }
    }
}

/// A component in the PAIML stack
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Component {
    /// Crate name
    pub name: String,
    /// Short description
    pub description: String,
    /// Local version if found
    pub version_local: Option<semver::Version>,
    /// Remote version from crates.io
    pub version_remote: Option<semver::Version>,
    /// Health status
    pub health: HealthStatus,
}

impl Component {
    /// Create a new component
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            version_local: None,
            version_remote: None,
            health: HealthStatus::NotFound,
        }
    }

    /// Set local version
    pub fn with_local_version(mut self, version: semver::Version) -> Self {
        self.version_local = Some(version);
        self.update_health();
        self
    }

    /// Set remote version
    pub fn with_remote_version(mut self, version: semver::Version) -> Self {
        self.version_remote = Some(version);
        self.update_health();
        self
    }

    /// Update health based on versions
    fn update_health(&mut self) {
        self.health = match (&self.version_local, &self.version_remote) {
            (Some(local), Some(remote)) => {
                if local == remote {
                    HealthStatus::Synced
                } else if local < remote {
                    HealthStatus::Behind
                } else {
                    HealthStatus::Ahead
                }
            }
            (Some(_), None) => HealthStatus::NotFound,
            (None, Some(_)) => HealthStatus::NotFound,
            (None, None) => HealthStatus::NotFound,
        };
    }
}

/// A layer in the PAIML stack
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StackLayer {
    /// Layer name (e.g., "core", "ml")
    pub name: String,
    /// Components in this layer
    pub components: Vec<Component>,
}

impl StackLayer {
    /// Create a new layer
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into(), components: Vec::new() }
    }

    /// Add a component to this layer
    pub fn add_component(mut self, component: Component) -> Self {
        self.components.push(component);
        self
    }
}

/// The complete PAIML stack tree
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StackTree {
    /// Stack name
    pub name: String,
    /// Total crate count
    pub total_crates: usize,
    /// Layers in the stack
    pub layers: Vec<StackLayer>,
}

impl StackTree {
    /// Create a new stack tree
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into(), total_crates: 0, layers: Vec::new() }
    }

    /// Add a layer to the tree
    pub fn add_layer(mut self, layer: StackLayer) -> Self {
        self.total_crates += layer.components.len();
        self.layers.push(layer);
        self
    }

    /// Get total synced count
    pub fn synced_count(&self) -> usize {
        self.layers
            .iter()
            .flat_map(|l| &l.components)
            .filter(|c| c.health == HealthStatus::Synced)
            .count()
    }

    /// Get total behind count
    pub fn behind_count(&self) -> usize {
        self.layers
            .iter()
            .flat_map(|l| &l.components)
            .filter(|c| c.health == HealthStatus::Behind)
            .count()
    }
}

// ============================================================================
// TREE-002: Output Formats
// ============================================================================

/// Output format for the tree
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OutputFormat {
    /// ASCII tree (default)
    #[default]
    Ascii,
    /// JSON output
    Json,
    /// Graphviz DOT format
    Dot,
}

impl std::str::FromStr for OutputFormat {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "ascii" => Ok(Self::Ascii),
            "json" => Ok(Self::Json),
            "dot" => Ok(Self::Dot),
            _ => Err(format!("Unknown format: {s}")),
        }
    }
}

// ============================================================================
// TREE-003: Formatters
// ============================================================================

/// Format a single component line for ASCII tree output.
fn format_component_line(
    comp: &Component,
    show_health: bool,
    comp_prefix: &str,
    comp_branch: &str,
) -> String {
    if !show_health {
        return format!("{}{}{}\n", comp_prefix, comp_branch, comp.name);
    }
    let version_str = match (&comp.version_local, &comp.version_remote) {
        (Some(local), Some(remote)) if local != remote => {
            format!("v{} → {}", local, remote)
        }
        (Some(local), _) => format!("v{}", local),
        _ => String::new(),
    };
    format!("{}{}{} {} {}\n", comp_prefix, comp_branch, comp.name, comp.health, version_str)
}

/// Format tree as ASCII
pub fn format_ascii(tree: &StackTree, show_health: bool) -> String {
    let mut output = format!("{} ({} crates)\n", tree.name, tree.total_crates);

    for (layer_idx, layer) in tree.layers.iter().enumerate() {
        let is_last_layer = layer_idx == tree.layers.len() - 1;
        let layer_prefix = if is_last_layer { "└── " } else { "├── " };
        output.push_str(&format!("{}{}\n", layer_prefix, layer.name));

        let comp_prefix = if is_last_layer { "    " } else { "│   " };
        for (comp_idx, comp) in layer.components.iter().enumerate() {
            let comp_branch =
                if comp_idx == layer.components.len() - 1 { "└── " } else { "├── " };
            output.push_str(&format_component_line(comp, show_health, comp_prefix, comp_branch));
        }
    }

    output
}

/// Format tree as JSON
pub fn format_json(tree: &StackTree) -> Result<String, serde_json::Error> {
    serde_json::to_string_pretty(tree)
}

/// Format tree as Graphviz DOT
pub fn format_dot(tree: &StackTree) -> String {
    let mut output = String::from("digraph paiml_stack {\n");
    output.push_str("  rankdir=TB;\n");
    output.push_str("  node [shape=box];\n\n");

    for layer in &tree.layers {
        output.push_str(&format!("  subgraph cluster_{} {{\n", layer.name.replace(' ', "_")));
        output.push_str(&format!("    label=\"{}\";\n", layer.name));

        for comp in &layer.components {
            let color = match comp.health {
                HealthStatus::Synced => "green",
                HealthStatus::Behind => "orange",
                HealthStatus::Ahead => "blue",
                HealthStatus::NotFound => "gray",
                HealthStatus::Error(_) => "red",
            };
            output.push_str(&format!("    {} [color={}];\n", comp.name.replace('-', "_"), color));
        }

        output.push_str("  }\n\n");
    }

    output.push_str("}\n");
    output
}

// ============================================================================
// TREE-004: Builder
// ============================================================================

/// Layer definitions for PAIML stack
pub const LAYER_DEFINITIONS: &[(&str, &[&str])] = &[
    ("core", &["trueno", "trueno-viz", "trueno-db", "trueno-graph", "trueno-rag", "trueno-zram"]),
    ("ml", &["aprender", "aprender-shell", "aprender-tsp"]),
    ("inference", &["realizar", "renacer", "alimentar", "entrenar"]),
    ("orchestration", &["batuta", "certeza", "presentar", "pacha"]),
    ("distributed", &["repartir", "pepita"]),
    ("inference", &["whisper-apr"]),
    ("transpilation", &["ruchy", "decy", "depyler", "bashrs"]),
    ("docs", &["sovereign-ai-stack-book"]),
];

/// Component descriptions
pub fn get_component_description(name: &str) -> &'static str {
    match name {
        "trueno" => "SIMD tensor operations",
        "trueno-viz" => "Visualization",
        "trueno-db" => "Vector database",
        "trueno-graph" => "Graph algorithms",
        "trueno-rag" => "RAG framework",
        "trueno-zram" => "SIMD memory compression",
        "aprender" => "ML algorithms",
        "aprender-shell" => "REPL",
        "aprender-tsp" => "TSP solver",
        "realizar" => "Inference engine",
        "renacer" => "Model lifecycle",
        "alimentar" => "Data pipelines",
        "entrenar" => "Experiment tracking",
        "batuta" => "Orchestrator",
        "certeza" => "Quality gates",
        "presentar" => "Presentation",
        "pacha" => "Knowledge base",
        "repartir" => "Distributed computing",
        "pepita" => "io_uring kernel interfaces",
        "whisper-apr" => "Speech-to-text inference",
        "ruchy" => "Rust-Python bridge",
        "decy" => "Decision engine",
        "depyler" => "Python→Rust transpiler",
        "bashrs" => "Bash→Rust transpiler",
        "sovereign-ai-stack-book" => "Documentation",
        _ => "Unknown component",
    }
}

/// Build the default PAIML stack tree (without version info)
pub fn build_tree() -> StackTree {
    let mut tree = StackTree::new("PAIML Stack");

    for (layer_name, components) in LAYER_DEFINITIONS {
        let mut layer = StackLayer::new(*layer_name);
        for comp_name in *components {
            let component = Component::new(*comp_name, get_component_description(comp_name));
            layer = layer.add_component(component);
        }
        tree = tree.add_layer(layer);
    }

    tree
}

// ============================================================================
// Tests - Extreme TDD
// ============================================================================

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    // ========================================================================
    // TREE-001: HealthStatus Tests
    // ========================================================================

    #[test]
    fn test_TREE_001_health_status_display_synced() {
        assert_eq!(format!("{}", HealthStatus::Synced), "✓");
    }

    #[test]
    fn test_TREE_001_health_status_display_behind() {
        assert_eq!(format!("{}", HealthStatus::Behind), "⚠");
    }

    #[test]
    fn test_TREE_001_health_status_display_ahead() {
        assert_eq!(format!("{}", HealthStatus::Ahead), "↑");
    }

    #[test]
    fn test_TREE_001_health_status_display_not_found() {
        assert_eq!(format!("{}", HealthStatus::NotFound), "?");
    }

    #[test]
    fn test_TREE_001_health_status_display_error() {
        assert_eq!(format!("{}", HealthStatus::Error("test".into())), "✗");
    }

    #[test]
    fn test_TREE_001_health_status_serialize() {
        let json = serde_json::to_string(&HealthStatus::Synced).expect("json serialize failed");
        assert_eq!(json, "\"synced\"");
    }

    #[test]
    fn test_TREE_001_health_status_deserialize() {
        let status: HealthStatus =
            serde_json::from_str("\"behind\"").expect("json deserialize failed");
        assert_eq!(status, HealthStatus::Behind);
    }

    // ========================================================================
    // TREE-002: Component Tests
    // ========================================================================

    #[test]
    fn test_TREE_002_component_new() {
        let comp = Component::new("trueno", "SIMD ops");
        assert_eq!(comp.name, "trueno");
        assert_eq!(comp.description, "SIMD ops");
        assert_eq!(comp.health, HealthStatus::NotFound);
    }

    #[test]
    fn test_TREE_002_component_with_local_version() {
        let comp =
            Component::new("trueno", "SIMD").with_local_version(semver::Version::new(1, 0, 0));
        assert_eq!(comp.version_local, Some(semver::Version::new(1, 0, 0)));
    }

    #[test]
    fn test_TREE_002_component_health_synced() {
        let comp = Component::new("trueno", "SIMD")
            .with_local_version(semver::Version::new(1, 0, 0))
            .with_remote_version(semver::Version::new(1, 0, 0));
        assert_eq!(comp.health, HealthStatus::Synced);
    }

    #[test]
    fn test_TREE_002_component_health_behind() {
        let comp = Component::new("trueno", "SIMD")
            .with_local_version(semver::Version::new(1, 0, 0))
            .with_remote_version(semver::Version::new(1, 1, 0));
        assert_eq!(comp.health, HealthStatus::Behind);
    }

    #[test]
    fn test_TREE_002_component_health_ahead() {
        let comp = Component::new("trueno", "SIMD")
            .with_local_version(semver::Version::new(2, 0, 0))
            .with_remote_version(semver::Version::new(1, 0, 0));
        assert_eq!(comp.health, HealthStatus::Ahead);
    }

    // ========================================================================
    // TREE-003: StackLayer Tests
    // ========================================================================

    #[test]
    fn test_TREE_003_stack_layer_new() {
        let layer = StackLayer::new("core");
        assert_eq!(layer.name, "core");
        assert!(layer.components.is_empty());
    }

    #[test]
    fn test_TREE_003_stack_layer_add_component() {
        let layer =
            StackLayer::new("core").add_component(Component::new("trueno", "SIMD tensor ops"));
        assert_eq!(layer.components.len(), 1);
        assert_eq!(layer.components[0].name, "trueno");
    }

    // ========================================================================
    // TREE-004: StackTree Tests
    // ========================================================================

    #[test]
    fn test_TREE_004_stack_tree_new() {
        let tree = StackTree::new("PAIML Stack");
        assert_eq!(tree.name, "PAIML Stack");
        assert_eq!(tree.total_crates, 0);
        assert!(tree.layers.is_empty());
    }

    #[test]
    fn test_TREE_004_stack_tree_add_layer() {
        let layer =
            StackLayer::new("core").add_component(Component::new("trueno", "SIMD tensor ops"));
        let tree = StackTree::new("PAIML Stack").add_layer(layer);
        assert_eq!(tree.total_crates, 1);
        assert_eq!(tree.layers.len(), 1);
    }

    #[test]
    fn test_TREE_004_stack_tree_synced_count() {
        let layer = StackLayer::new("core").add_component(
            Component::new("trueno", "SIMD")
                .with_local_version(semver::Version::new(1, 0, 0))
                .with_remote_version(semver::Version::new(1, 0, 0)),
        );
        let tree = StackTree::new("Test").add_layer(layer);
        assert_eq!(tree.synced_count(), 1);
    }

    #[test]
    fn test_TREE_004_stack_tree_behind_count() {
        let layer = StackLayer::new("core").add_component(
            Component::new("trueno", "SIMD")
                .with_local_version(semver::Version::new(1, 0, 0))
                .with_remote_version(semver::Version::new(2, 0, 0)),
        );
        let tree = StackTree::new("Test").add_layer(layer);
        assert_eq!(tree.behind_count(), 1);
    }

    // ========================================================================
    // TREE-005: OutputFormat Tests
    // ========================================================================

    #[test]
    fn test_TREE_005_output_format_from_str_ascii() {
        assert_eq!("ascii".parse::<OutputFormat>().expect("parse failed"), OutputFormat::Ascii);
    }

    #[test]
    fn test_TREE_005_output_format_from_str_json() {
        assert_eq!("json".parse::<OutputFormat>().expect("parse failed"), OutputFormat::Json);
    }

    #[test]
    fn test_TREE_005_output_format_from_str_dot() {
        assert_eq!("dot".parse::<OutputFormat>().expect("parse failed"), OutputFormat::Dot);
    }

    #[test]
    fn test_TREE_005_output_format_from_str_case_insensitive() {
        assert_eq!("JSON".parse::<OutputFormat>().expect("parse failed"), OutputFormat::Json);
    }

    #[test]
    fn test_TREE_005_output_format_from_str_invalid() {
        assert!("xml".parse::<OutputFormat>().is_err());
    }

    #[test]
    fn test_TREE_005_output_format_default() {
        assert_eq!(OutputFormat::default(), OutputFormat::Ascii);
    }

    // ========================================================================
    // TREE-006: ASCII Formatter Tests
    // ========================================================================

    #[test]
    fn test_TREE_006_format_ascii_header() {
        let tree = StackTree::new("Test Stack");
        let output = format_ascii(&tree, false);
        assert!(output.starts_with("Test Stack (0 crates)"));
    }

    #[test]
    fn test_TREE_006_format_ascii_with_layer() {
        let layer = StackLayer::new("core").add_component(Component::new("trueno", "SIMD"));
        let tree = StackTree::new("Test").add_layer(layer);
        let output = format_ascii(&tree, false);
        assert!(output.contains("└── core"));
        assert!(output.contains("trueno"));
    }

    #[test]
    fn test_TREE_006_format_ascii_with_health() {
        let layer = StackLayer::new("core").add_component(
            Component::new("trueno", "SIMD")
                .with_local_version(semver::Version::new(1, 0, 0))
                .with_remote_version(semver::Version::new(1, 0, 0)),
        );
        let tree = StackTree::new("Test").add_layer(layer);
        let output = format_ascii(&tree, true);
        assert!(output.contains("✓"));
        assert!(output.contains("v1.0.0"));
    }

    #[test]
    fn test_TREE_006_format_ascii_version_diff() {
        let layer = StackLayer::new("core").add_component(
            Component::new("trueno", "SIMD")
                .with_local_version(semver::Version::new(1, 0, 0))
                .with_remote_version(semver::Version::new(2, 0, 0)),
        );
        let tree = StackTree::new("Test").add_layer(layer);
        let output = format_ascii(&tree, true);
        assert!(output.contains("v1.0.0 → 2.0.0"));
    }

    // ========================================================================
    // TREE-007: JSON Formatter Tests
    // ========================================================================

    #[test]
    fn test_TREE_007_format_json_valid() {
        let tree = StackTree::new("Test");
        let json = format_json(&tree).expect("unexpected failure");
        assert!(json.contains("\"name\": \"Test\""));
    }

    #[test]
    fn test_TREE_007_format_json_roundtrip() {
        let layer = StackLayer::new("core").add_component(Component::new("trueno", "SIMD"));
        let tree = StackTree::new("Test").add_layer(layer);
        let json = format_json(&tree).expect("unexpected failure");
        let parsed: StackTree = serde_json::from_str(&json).expect("json deserialize failed");
        assert_eq!(parsed.name, "Test");
        assert_eq!(parsed.layers[0].components[0].name, "trueno");
    }

    // ========================================================================
    // TREE-008: DOT Formatter Tests
    // ========================================================================

    #[test]
    fn test_TREE_008_format_dot_header() {
        let tree = StackTree::new("Test");
        let dot = format_dot(&tree);
        assert!(dot.starts_with("digraph paiml_stack {"));
        assert!(dot.contains("rankdir=TB"));
    }

    #[test]
    fn test_TREE_008_format_dot_cluster() {
        let layer = StackLayer::new("core").add_component(Component::new("trueno", "SIMD"));
        let tree = StackTree::new("Test").add_layer(layer);
        let dot = format_dot(&tree);
        assert!(dot.contains("subgraph cluster_core"));
        assert!(dot.contains("label=\"core\""));
    }

    #[test]
    fn test_TREE_008_format_dot_health_colors() {
        let layer = StackLayer::new("core").add_component(
            Component::new("trueno", "SIMD")
                .with_local_version(semver::Version::new(1, 0, 0))
                .with_remote_version(semver::Version::new(1, 0, 0)),
        );
        let tree = StackTree::new("Test").add_layer(layer);
        let dot = format_dot(&tree);
        assert!(dot.contains("color=green"));
    }

    // ========================================================================
    // TREE-009: Builder Tests
    // ========================================================================

    #[test]
    fn test_TREE_009_build_tree_creates_all_layers() {
        let tree = build_tree();
        assert_eq!(tree.layers.len(), 8);
    }

    #[test]
    fn test_TREE_009_build_tree_total_crates() {
        let tree = build_tree();
        assert_eq!(tree.total_crates, 25);
    }

    #[test]
    fn test_TREE_009_build_tree_core_layer() {
        let tree = build_tree();
        let core = &tree.layers[0];
        assert_eq!(core.name, "core");
        assert_eq!(core.components.len(), 6);
        assert_eq!(core.components[0].name, "trueno");
    }

    #[test]
    fn test_TREE_009_get_component_description() {
        assert_eq!(get_component_description("trueno"), "SIMD tensor operations");
        assert_eq!(get_component_description("batuta"), "Orchestrator");
        assert_eq!(get_component_description("unknown"), "Unknown component");
    }

    // ========================================================================
    // TREE-010: Integration Tests
    // ========================================================================

    #[test]
    fn test_TREE_010_full_tree_ascii_output() {
        let tree = build_tree();
        let output = format_ascii(&tree, false);
        assert!(output.contains("PAIML Stack (25 crates)"));
        assert!(output.contains("core"));
        assert!(output.contains("ml"));
        assert!(output.contains("orchestration"));
        assert!(output.contains("trueno"));
        assert!(output.contains("batuta"));
    }

    #[test]
    fn test_TREE_010_full_tree_json_output() {
        let tree = build_tree();
        let json = format_json(&tree).expect("unexpected failure");
        let parsed: serde_json::Value =
            serde_json::from_str(&json).expect("json deserialize failed");
        assert_eq!(parsed["total_crates"], 25);
    }

    #[test]
    fn test_TREE_010_full_tree_dot_output() {
        let tree = build_tree();
        let dot = format_dot(&tree);
        assert!(dot.contains("digraph"));
        assert!(dot.contains("cluster_core"));
        assert!(dot.contains("cluster_ml"));
    }
}
