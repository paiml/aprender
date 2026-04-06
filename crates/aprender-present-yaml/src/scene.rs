//! Presentar Scene Format (.prs) parser.
//!
//! This module implements the `.prs` format specification for shareable
//! visualization manifests. The format is YAML-based and declarative,
//! enabling WASM-native parsing without runtime interpreters.
//!
//! # Design Philosophy
//!
//! A `.prs` file is a *bill of materials* for a visualization—it declares
//! **what** to display and **where** data lives, not **how** to fetch or render it.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

/// Presentar Scene - top-level structure for `.prs` files.
///
/// A Scene is a declarative manifest that references external resources
/// (models, datasets) and defines widget layout and interactions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scene {
    /// Format version (semver, e.g., "1.0")
    pub prs_version: String,

    /// Scene metadata
    pub metadata: SceneMetadata,

    /// External resources (models, datasets)
    #[serde(default)]
    pub resources: Resources,

    /// Widget layout configuration
    pub layout: SceneLayout,

    /// Widget definitions
    pub widgets: Vec<SceneWidget>,

    /// Event → action bindings
    #[serde(default)]
    pub bindings: Vec<Binding>,

    /// Theme configuration
    #[serde(default)]
    pub theme: Option<SceneTheme>,

    /// Security permissions
    #[serde(default)]
    pub permissions: Permissions,

    /// Header bar configuration (for tmux layout)
    #[serde(default)]
    pub header: Option<HeaderFooter>,

    /// Footer bar configuration (for tmux layout)
    #[serde(default)]
    pub footer: Option<HeaderFooter>,

    /// Keyboard sequence bindings (for tmux layout)
    #[serde(default)]
    pub key_bindings: Option<KeyBindings>,
}

/// Scene metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneMetadata {
    /// Unique scene identifier (kebab-case, e.g., "sentiment-analysis-demo")
    pub name: String,

    /// Human-readable title
    #[serde(default)]
    pub title: Option<String>,

    /// Description
    #[serde(default)]
    pub description: Option<String>,

    /// Author email or identifier
    #[serde(default)]
    pub author: Option<String>,

    /// Creation timestamp (ISO 8601)
    #[serde(default)]
    pub created: Option<String>,

    /// License identifier (e.g., "MIT", "Apache-2.0")
    #[serde(default)]
    pub license: Option<String>,

    /// Tags for categorization
    #[serde(default)]
    pub tags: Vec<String>,
}

/// External resources container.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Resources {
    /// Model resources
    #[serde(default)]
    pub models: HashMap<String, ModelResource>,

    /// Dataset resources
    #[serde(default)]
    pub datasets: HashMap<String, DatasetResource>,
}

/// Model resource reference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelResource {
    /// Model format (apr, gguf, safetensors)
    #[serde(rename = "type")]
    pub resource_type: ModelType,

    /// Source URL or path (can be array for fallback)
    pub source: ResourceSource,

    /// Content hash for verification (blake3:<hex>)
    #[serde(default)]
    pub hash: Option<String>,

    /// File size in bytes (for progress indication)
    #[serde(default)]
    pub size_bytes: Option<u64>,
}

/// Dataset resource reference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetResource {
    /// Dataset format (ald, parquet, csv)
    #[serde(rename = "type")]
    pub resource_type: DatasetType,

    /// Source URL or path (can be array for fallback)
    pub source: ResourceSource,

    /// Content hash for verification (blake3:<hex>)
    #[serde(default)]
    pub hash: Option<String>,

    /// File size in bytes (for progress indication)
    #[serde(default)]
    pub size_bytes: Option<u64>,
}

/// Model format types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelType {
    /// Aprender model format
    Apr,
    /// GGUF quantized format
    Gguf,
    /// SafeTensors format
    Safetensors,
}

/// Dataset format types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DatasetType {
    /// Alimentar dataset format
    Ald,
    /// Apache Parquet
    Parquet,
    /// Comma-separated values
    Csv,
}

/// Resource source - single URL/path or array of fallbacks.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResourceSource {
    /// Single source
    Single(String),
    /// Multiple sources (tried in order)
    Multiple(Vec<String>),
}

impl ResourceSource {
    /// Get all sources as a slice.
    #[must_use]
    pub fn sources(&self) -> Vec<&str> {
        match self {
            Self::Single(s) => vec![s.as_str()],
            Self::Multiple(v) => v.iter().map(String::as_str).collect(),
        }
    }

    /// Get primary source.
    #[must_use]
    pub fn primary(&self) -> &str {
        match self {
            Self::Single(s) => s.as_str(),
            Self::Multiple(v) => v.first().map_or("", String::as_str),
        }
    }
}

/// Scene layout configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneLayout {
    /// Layout type
    #[serde(rename = "type")]
    pub layout_type: LayoutType,

    /// Number of columns (for grid layout)
    #[serde(default)]
    pub columns: Option<u32>,

    /// Number of rows (for grid layout)
    #[serde(default)]
    pub rows: Option<u32>,

    /// Gap between widgets in pixels
    #[serde(default = "default_gap")]
    pub gap: u32,

    /// Flex direction (for flex layout)
    #[serde(default)]
    pub direction: Option<FlexDirection>,

    /// Flex wrap (for flex layout)
    #[serde(default)]
    pub wrap: Option<bool>,

    /// Canvas width (for absolute layout)
    #[serde(default)]
    pub width: Option<u32>,

    /// Canvas height (for absolute layout)
    #[serde(default)]
    pub height: Option<u32>,
}

const fn default_gap() -> u32 {
    16
}

/// Layout type enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LayoutType {
    /// CSS Grid layout
    Grid,
    /// Flexbox layout
    Flex,
    /// Absolute positioning
    Absolute,
    /// TMUX-style multi-pane terminal layout
    Tmux,
}

/// Flex direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FlexDirection {
    /// Horizontal (left to right)
    Row,
    /// Vertical (top to bottom)
    Column,
}

/// Scene widget definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneWidget {
    /// Unique widget identifier
    pub id: String,

    /// Widget type
    #[serde(rename = "type")]
    pub widget_type: WidgetType,

    /// Grid position (for grid layout)
    #[serde(default)]
    pub position: Option<GridPosition>,

    /// Widget-specific configuration
    #[serde(default)]
    pub config: WidgetConfig,
}

/// Widget types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WidgetType {
    /// Text input field
    Textbox,
    /// Numeric slider
    Slider,
    /// Selection dropdown
    Dropdown,
    /// Clickable button
    Button,
    /// Image display
    Image,
    /// Bar chart visualization
    BarChart,
    /// Line chart visualization
    LineChart,
    /// Single-value gauge
    Gauge,
    /// Data table
    Table,
    /// Markdown content
    Markdown,
    /// Model inference runner
    Inference,
    /// Interactive terminal pane (APR shell, WOS, or static)
    Terminal,
}

/// Grid position for widgets.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridPosition {
    /// Row index (0-based)
    pub row: u32,
    /// Column index (0-based)
    pub col: u32,
    /// Column span (defaults to 1)
    #[serde(default = "default_span")]
    pub colspan: u32,
    /// Row span (defaults to 1)
    #[serde(default = "default_span")]
    pub rowspan: u32,
}

const fn default_span() -> u32 {
    1
}

/// Widget configuration - varies by widget type.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WidgetConfig {
    // Common fields
    /// Label text
    #[serde(default)]
    pub label: Option<String>,
    /// Title text
    #[serde(default)]
    pub title: Option<String>,

    // Textbox fields
    /// Placeholder text
    #[serde(default)]
    pub placeholder: Option<String>,
    /// Maximum input length
    #[serde(default)]
    pub max_length: Option<u32>,

    // Slider fields
    /// Minimum value
    #[serde(default)]
    pub min: Option<f64>,
    /// Maximum value
    #[serde(default)]
    pub max: Option<f64>,
    /// Step increment
    #[serde(default)]
    pub step: Option<f64>,
    /// Default value
    #[serde(default)]
    pub default: Option<f64>,

    // Dropdown fields
    /// Selection options
    #[serde(default)]
    pub options: Option<String>,
    /// Allow multiple selection
    #[serde(default)]
    pub multi_select: Option<bool>,

    // Button fields
    /// Button action
    #[serde(default)]
    pub action: Option<String>,

    // Image fields
    /// Image source URL/path
    #[serde(default)]
    pub source: Option<String>,
    /// Alt text
    #[serde(default)]
    pub alt: Option<String>,
    /// Upload mode
    #[serde(default)]
    pub mode: Option<String>,
    /// Accepted MIME types
    #[serde(default)]
    pub accept: Option<Vec<String>>,

    // Chart fields
    /// Data source expression
    #[serde(default)]
    pub data: Option<String>,
    /// X-axis field/expression
    #[serde(default)]
    pub x_axis: Option<String>,
    /// Y-axis field/expression
    #[serde(default)]
    pub y_axis: Option<String>,

    // Gauge fields
    /// Gauge value expression
    #[serde(default)]
    pub value: Option<String>,
    /// Gauge thresholds
    #[serde(default)]
    pub thresholds: Option<Vec<Threshold>>,

    // Table fields
    /// Column definitions
    #[serde(default)]
    pub columns: Option<Vec<String>>,
    /// Sortable flag
    #[serde(default)]
    pub sortable: Option<bool>,

    // Markdown fields
    /// Markdown content
    #[serde(default)]
    pub content: Option<String>,

    // Inference fields
    /// Model reference
    #[serde(default)]
    pub model: Option<String>,
    /// Input expression
    #[serde(default)]
    pub input: Option<String>,
    /// Output field
    #[serde(default)]
    pub output: Option<String>,

    // Terminal fields (mode field shared with Image)
    /// URL to APR model file
    #[serde(default)]
    pub model_url: Option<String>,
    /// Terminal prompt string
    #[serde(default)]
    pub prompt: Option<String>,
    /// Enable search bar at bottom of pane
    #[serde(default)]
    pub search_bar: Option<bool>,
    /// Scrollback history size (lines)
    #[serde(default)]
    pub history_size: Option<u32>,
    /// Script path for WOS mode
    #[serde(default)]
    pub script: Option<String>,
    /// Auto-run script on load
    #[serde(default)]
    pub auto_run: Option<bool>,
    /// Hint text shown when auto-run is enabled
    #[serde(default)]
    pub auto_run_hint: Option<String>,
}

/// Gauge threshold.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Threshold {
    /// Threshold value
    pub value: f64,
    /// Color at/below threshold
    pub color: String,
}

/// Event binding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Binding {
    /// Event trigger (e.g., "text_input.change")
    pub trigger: String,

    /// Debounce delay in milliseconds
    #[serde(default)]
    pub debounce_ms: Option<u32>,

    /// Actions to execute
    pub actions: Vec<BindingAction>,
}

/// Binding action.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BindingAction {
    /// Target (widget ID or inference.model)
    pub target: String,

    /// Action type (refresh, set, etc.)
    #[serde(default)]
    pub action: Option<String>,

    /// Input expression
    #[serde(default)]
    pub input: Option<String>,
}

/// Theme configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneTheme {
    /// Theme preset (light, dark)
    #[serde(default)]
    pub preset: Option<String>,

    /// Custom theme values
    #[serde(default)]
    pub custom: HashMap<String, String>,
}

/// Security permissions.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Permissions {
    /// Allowed network URLs (glob patterns)
    #[serde(default)]
    pub network: Vec<String>,

    /// Allowed filesystem paths (glob patterns)
    #[serde(default)]
    pub filesystem: Vec<String>,

    /// Clipboard access
    #[serde(default)]
    pub clipboard: bool,

    /// Camera access
    #[serde(default)]
    pub camera: bool,
}

/// Header or footer bar configuration for tmux layout.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeaderFooter {
    /// Height in pixels
    #[serde(default = "default_header_height")]
    pub height: u32,

    /// Background color
    #[serde(default)]
    pub background: Option<String>,

    /// Content sections (left, center, right)
    #[serde(default)]
    pub content: HeaderContent,
}

fn default_header_height() -> u32 {
    48
}

/// Header/footer content layout.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HeaderContent {
    /// Left-aligned items
    #[serde(default)]
    pub left: Vec<ContentItem>,
    /// Center-aligned items
    #[serde(default)]
    pub center: Vec<ContentItem>,
    /// Right-aligned items
    #[serde(default)]
    pub right: Vec<ContentItem>,
}

/// A content item within header/footer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentItem {
    /// Item type (text, nav, pane_tabs)
    #[serde(rename = "type")]
    pub item_type: String,
    /// Text content
    #[serde(default)]
    pub content: Option<String>,
    /// Style name
    #[serde(default)]
    pub style: Option<String>,
    /// Navigation items (for nav type)
    #[serde(default)]
    pub items: Vec<NavItem>,
}

/// Navigation link item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NavItem {
    /// Display label
    pub label: String,
    /// Link target
    pub href: String,
    /// Open in new tab
    #[serde(default)]
    pub external: bool,
}

/// Keyboard binding configuration for tmux-style prefix keys.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyBindings {
    /// Prefix key (e.g., "ctrl+b")
    #[serde(default = "default_prefix_key")]
    pub prefix_key: String,

    /// Timeout in ms after prefix key before returning to normal mode
    #[serde(default = "default_prefix_timeout")]
    pub prefix_timeout_ms: u32,

    /// Two-key sequences (prefix + follow-up)
    #[serde(default)]
    pub sequences: Vec<KeySequence>,

    /// Single-key global bindings (no prefix needed)
    #[serde(default)]
    pub global: Vec<GlobalKeyBinding>,
}

fn default_prefix_key() -> String {
    "ctrl+b".into()
}

fn default_prefix_timeout() -> u32 {
    500
}

/// A two-key sequence binding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeySequence {
    /// Key sequence (e.g., `["ctrl+b", "0"]`)
    pub keys: Vec<String>,
    /// Action to perform
    pub action: serde_yaml_ng::Value,
}

/// A single global key binding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalKeyBinding {
    /// Key name
    pub key: String,
    /// Action to perform
    pub action: serde_yaml_ng::Value,
}

/// Error type for scene parsing and validation.
#[derive(Debug)]
pub enum SceneError {
    /// YAML parsing error
    Yaml(serde_yaml_ng::Error),

    /// Invalid prs_version format
    InvalidVersion(String),

    /// Duplicate widget ID
    DuplicateWidgetId(String),

    /// Invalid binding target (references non-existent widget)
    InvalidBindingTarget {
        /// The binding trigger
        trigger: String,
        /// The invalid target
        target: String,
    },

    /// Invalid hash format
    InvalidHashFormat {
        /// Resource name
        resource: String,
        /// The invalid hash
        hash: String,
    },

    /// Missing required hash for remote resource
    MissingRemoteHash {
        /// Resource name
        resource: String,
    },

    /// Invalid expression syntax
    InvalidExpression {
        /// Widget ID or context
        context: String,
        /// The invalid expression
        expression: String,
        /// Error message
        message: String,
    },

    /// Invalid metadata name (must be kebab-case)
    InvalidMetadataName(String),

    /// Layout validation error
    LayoutError(String),
}

impl fmt::Display for SceneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Yaml(e) => write!(f, "YAML error: {e}"),
            Self::InvalidVersion(v) => write!(f, "Invalid prs_version: {v}"),
            Self::DuplicateWidgetId(id) => write!(f, "Duplicate widget id: {id}"),
            Self::InvalidBindingTarget { trigger, target } => {
                write!(
                    f,
                    "Invalid binding target '{target}' in trigger '{trigger}'"
                )
            }
            Self::InvalidHashFormat { resource, hash } => {
                write!(f, "Invalid hash format for '{resource}': {hash}")
            }
            Self::MissingRemoteHash { resource } => {
                write!(f, "Missing hash for remote resource: {resource}")
            }
            Self::InvalidExpression {
                context,
                expression,
                message,
            } => {
                write!(
                    f,
                    "Invalid expression in {context}: '{expression}' - {message}"
                )
            }
            Self::InvalidMetadataName(name) => {
                write!(f, "Invalid metadata name '{name}': must be kebab-case")
            }
            Self::LayoutError(msg) => write!(f, "Layout error: {msg}"),
        }
    }
}

impl std::error::Error for SceneError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Yaml(e) => Some(e),
            _ => None,
        }
    }
}

impl From<serde_yaml_ng::Error> for SceneError {
    fn from(e: serde_yaml_ng::Error) -> Self {
        Self::Yaml(e)
    }
}

impl Scene {
    /// Parse a scene from YAML string.
    ///
    /// # Errors
    ///
    /// Returns an error if the YAML is invalid or fails validation.
    pub fn from_yaml(yaml: &str) -> Result<Self, SceneError> {
        let scene: Self = serde_yaml_ng::from_str(yaml)?;
        scene.validate()?;
        Ok(scene)
    }

    /// Serialize scene to YAML string.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails.
    pub fn to_yaml(&self) -> Result<String, serde_yaml_ng::Error> {
        serde_yaml_ng::to_string(self)
    }

    /// Validate the scene structure.
    ///
    /// Checks:
    /// 1. prs_version format (semver)
    /// 2. metadata.name is kebab-case
    /// 3. Widget IDs are unique
    /// 4. Binding targets reference valid widgets/resources
    /// 5. Remote resources have hashes
    /// 6. Hash formats are valid (blake3:<hex>)
    ///
    /// # Errors
    ///
    /// Returns the first validation error found.
    pub fn validate(&self) -> Result<(), SceneError> {
        self.validate_version()?;
        self.validate_metadata_name()?;
        self.validate_widget_ids()?;
        self.validate_bindings()?;
        self.validate_resource_hashes()?;
        self.validate_layout()?;
        Ok(())
    }

    fn validate_version(&self) -> Result<(), SceneError> {
        // Version should be "X.Y" format
        let parts: Vec<&str> = self.prs_version.split('.').collect();
        if parts.len() != 2 {
            return Err(SceneError::InvalidVersion(self.prs_version.clone()));
        }
        for part in parts {
            if part.parse::<u32>().is_err() {
                return Err(SceneError::InvalidVersion(self.prs_version.clone()));
            }
        }
        Ok(())
    }

    fn validate_metadata_name(&self) -> Result<(), SceneError> {
        let name = &self.metadata.name;
        // Must be kebab-case: lowercase letters, numbers, hyphens
        if !name
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
        {
            return Err(SceneError::InvalidMetadataName(name.clone()));
        }
        // Cannot start or end with hyphen
        if name.starts_with('-') || name.ends_with('-') {
            return Err(SceneError::InvalidMetadataName(name.clone()));
        }
        // Cannot have consecutive hyphens
        if name.contains("--") {
            return Err(SceneError::InvalidMetadataName(name.clone()));
        }
        Ok(())
    }

    fn validate_widget_ids(&self) -> Result<(), SceneError> {
        let mut seen = std::collections::HashSet::new();
        for widget in &self.widgets {
            if !seen.insert(&widget.id) {
                return Err(SceneError::DuplicateWidgetId(widget.id.clone()));
            }
        }
        Ok(())
    }

    fn validate_bindings(&self) -> Result<(), SceneError> {
        let widget_ids: std::collections::HashSet<&str> =
            self.widgets.iter().map(|w| w.id.as_str()).collect();
        let model_ids: std::collections::HashSet<&str> =
            self.resources.models.keys().map(String::as_str).collect();

        for binding in &self.bindings {
            for action in &binding.actions {
                let target = &action.target;

                // Check if target is a widget ID
                if widget_ids.contains(target.as_str()) {
                    continue;
                }

                // Check if target is inference.<model_name>
                if let Some(model_name) = target.strip_prefix("inference.") {
                    if model_ids.contains(model_name) {
                        continue;
                    }
                }

                return Err(SceneError::InvalidBindingTarget {
                    trigger: binding.trigger.clone(),
                    target: target.clone(),
                });
            }
        }
        Ok(())
    }

    fn validate_resource_hashes(&self) -> Result<(), SceneError> {
        // Validate model hashes
        for (name, resource) in &self.resources.models {
            if is_remote_source(&resource.source) && resource.hash.is_none() {
                return Err(SceneError::MissingRemoteHash {
                    resource: name.clone(),
                });
            }
            if let Some(hash) = &resource.hash {
                validate_hash_format(name, hash)?;
            }
        }

        // Validate dataset hashes
        for (name, resource) in &self.resources.datasets {
            if is_remote_source(&resource.source) && resource.hash.is_none() {
                return Err(SceneError::MissingRemoteHash {
                    resource: name.clone(),
                });
            }
            if let Some(hash) = &resource.hash {
                validate_hash_format(name, hash)?;
            }
        }

        Ok(())
    }

    fn validate_layout(&self) -> Result<(), SceneError> {
        match self.layout.layout_type {
            LayoutType::Grid => {
                if self.layout.columns.is_none() {
                    return Err(SceneError::LayoutError(
                        "Grid layout requires 'columns' field".to_string(),
                    ));
                }
            }
            LayoutType::Absolute => {
                if self.layout.width.is_none() || self.layout.height.is_none() {
                    return Err(SceneError::LayoutError(
                        "Absolute layout requires 'width' and 'height' fields".to_string(),
                    ));
                }
            }
            LayoutType::Flex => {
                // Flex layout has optional fields
            }
            LayoutType::Tmux => {
                // Tmux layout uses rows/cols from SceneLayout (optional)
            }
        }
        Ok(())
    }

    /// Get all widget IDs.
    #[must_use]
    pub fn widget_ids(&self) -> Vec<&str> {
        self.widgets.iter().map(|w| w.id.as_str()).collect()
    }

    /// Get a widget by ID.
    #[must_use]
    pub fn get_widget(&self, id: &str) -> Option<&SceneWidget> {
        self.widgets.iter().find(|w| w.id == id)
    }

    /// Get a model resource by name.
    #[must_use]
    pub fn get_model(&self, name: &str) -> Option<&ModelResource> {
        self.resources.models.get(name)
    }

    /// Get a dataset resource by name.
    #[must_use]
    pub fn get_dataset(&self, name: &str) -> Option<&DatasetResource> {
        self.resources.datasets.get(name)
    }
}

/// Check if a resource source is remote (https://).
fn is_remote_source(source: &ResourceSource) -> bool {
    source.sources().iter().any(|s| s.starts_with("https://"))
}

/// Validate hash format (blake3:<64-hex-chars>).
fn validate_hash_format(resource: &str, hash: &str) -> Result<(), SceneError> {
    if let Some(hex) = hash.strip_prefix("blake3:") {
        // BLAKE3 produces 256-bit (32-byte) hashes = 64 hex characters
        if hex.len() >= 12 && hex.chars().all(|c| c.is_ascii_hexdigit()) {
            return Ok(());
        }
    }
    Err(SceneError::InvalidHashFormat {
        resource: resource.to_string(),
        hash: hash.to_string(),
    })
}

#[cfg(test)]
#[path = "scene_tests.rs"]
mod tests;
