//! Plugin System - Extensible content type handling
//!
//! Provides a plugin architecture for extending alimentar with new content
//! types. Built-in plugins include Dataset and Course (assetgen integration).

use std::{collections::HashMap, sync::Arc};

use arrow::{
    array::{ArrayRef, RecordBatch, StringArray},
    datatypes::{DataType, Field, Schema},
};
use serde::{Deserialize, Serialize};

use crate::{
    dataset::{ArrowDataset, Dataset},
    error::{Error, Result},
    serve::{
        content::{
            BoxedContent, ContentMetadata, ContentTypeId, ServeableContent, ValidationReport,
        },
        schema::{Constraint, ContentSchema, FieldDefinition, FieldType},
    },
};

/// Rendering hints for trueno-viz integration
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RenderHints {
    /// Preferred chart type (e.g., "scatter", "line", "table")
    pub chart_type: Option<String>,
    /// X-axis column name
    pub x_column: Option<String>,
    /// Y-axis column name
    pub y_column: Option<String>,
    /// Color column name
    pub color_column: Option<String>,
    /// Additional rendering options
    #[serde(default)]
    pub options: HashMap<String, serde_json::Value>,
}

impl RenderHints {
    /// Create new render hints
    pub fn new() -> Self {
        Self::default()
    }

    /// Set chart type
    pub fn with_chart_type(mut self, chart_type: impl Into<String>) -> Self {
        self.chart_type = Some(chart_type.into());
        self
    }

    /// Set X column
    pub fn with_x_column(mut self, column: impl Into<String>) -> Self {
        self.x_column = Some(column.into());
        self
    }

    /// Set Y column
    pub fn with_y_column(mut self, column: impl Into<String>) -> Self {
        self.y_column = Some(column.into());
        self
    }

    /// Set color column
    pub fn with_color_column(mut self, column: impl Into<String>) -> Self {
        self.color_column = Some(column.into());
        self
    }

    /// Add a custom option
    pub fn with_option(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.options.insert(key.into(), value);
        self
    }
}

/// Plugin interface for extending alimentar with new content types
///
/// Implement this trait to add support for custom content types
/// that can be served, validated, and visualized.
pub trait ContentPlugin: Send + Sync {
    /// Returns the content type ID this plugin handles
    fn content_type(&self) -> ContentTypeId;

    /// Returns the schema for this content type
    fn schema(&self) -> ContentSchema;

    /// Parses raw content into ServeableContent
    ///
    /// # Errors
    ///
    /// Returns an error if the data cannot be parsed into the expected content
    /// type.
    fn parse(&self, data: &[u8]) -> Result<BoxedContent>;

    /// Serializes ServeableContent back to bytes
    ///
    /// # Errors
    ///
    /// Returns an error if the content cannot be serialized.
    fn serialize(&self, content: &dyn ServeableContent) -> Result<Vec<u8>>;

    /// Returns UI rendering hints for trueno-viz integration
    fn render_hints(&self) -> RenderHints;

    /// Plugin version for compatibility
    fn version(&self) -> &str;

    /// Plugin name for display
    fn name(&self) -> &str;

    /// Plugin description
    fn description(&self) -> &str;
}

/// Registry for content plugins
pub struct PluginRegistry {
    plugins: HashMap<ContentTypeId, Box<dyn ContentPlugin>>,
}

impl PluginRegistry {
    /// Create a new plugin registry with built-in plugins
    pub fn new() -> Self {
        let mut registry = Self {
            plugins: HashMap::new(),
        };

        // Register built-in plugins
        registry.register(Box::new(DatasetPlugin::new()));
        registry.register(Box::new(RawPlugin::new()));

        registry
    }

    /// Register a plugin
    pub fn register(&mut self, plugin: Box<dyn ContentPlugin>) {
        self.plugins.insert(plugin.content_type(), plugin);
    }

    /// Get a plugin by content type
    pub fn get(&self, content_type: &ContentTypeId) -> Option<&dyn ContentPlugin> {
        self.plugins.get(content_type).map(|p| p.as_ref())
    }

    /// List all registered content types
    pub fn content_types(&self) -> Vec<ContentTypeId> {
        self.plugins.keys().cloned().collect()
    }

    /// Check if a content type is registered
    pub fn has(&self, content_type: &ContentTypeId) -> bool {
        self.plugins.contains_key(content_type)
    }

    /// Get plugin count
    pub fn len(&self) -> usize {
        self.plugins.len()
    }

    /// Check if registry is empty
    pub fn is_empty(&self) -> bool {
        self.plugins.is_empty()
    }
}

impl Default for PluginRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Built-in Plugins
// ============================================================================

/// Dataset plugin for Arrow/Parquet datasets
pub struct DatasetPlugin;

impl DatasetPlugin {
    /// Create a new dataset plugin
    pub fn new() -> Self {
        Self
    }
}

impl Default for DatasetPlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl ContentPlugin for DatasetPlugin {
    fn content_type(&self) -> ContentTypeId {
        ContentTypeId::dataset()
    }

    fn schema(&self) -> ContentSchema {
        ContentSchema::new(ContentTypeId::dataset(), "1.0")
            .with_field(
                FieldDefinition::new("name", FieldType::String)
                    .with_description("Dataset name")
                    .with_constraint(Constraint::min_length(1)),
            )
            .with_field(
                FieldDefinition::new("format", FieldType::String)
                    .with_description("Data format (parquet, csv, json)")
                    .with_constraint(Constraint::enum_values(vec![
                        serde_json::json!("parquet"),
                        serde_json::json!("csv"),
                        serde_json::json!("json"),
                        serde_json::json!("arrow"),
                    ])),
            )
            .with_field(
                FieldDefinition::new("rows", FieldType::Integer).with_description("Number of rows"),
            )
            .with_field(
                FieldDefinition::new("columns", FieldType::Integer)
                    .with_description("Number of columns"),
            )
            .with_required("name")
    }

    fn parse(&self, data: &[u8]) -> Result<BoxedContent> {
        // Contract: configuration-v1.yaml precondition (pv codegen)
        contract_pre_configuration!(data);
        // Try to parse as parquet first
        if data.len() >= 4 && &data[0..4] == b"PAR1" {
            let dataset = ArrowDataset::from_parquet_bytes(data)?;
            return Ok(Box::new(DatasetContent::new(dataset)));
        }

        // Try to parse as JSON
        if let Ok(text) = std::str::from_utf8(data) {
            let trimmed = text.trim();
            if trimmed.starts_with('{') || trimmed.starts_with('[') {
                let dataset = ArrowDataset::from_json_str(text)?;
                return Ok(Box::new(DatasetContent::new(dataset)));
            }

            // Try CSV
            let dataset = ArrowDataset::from_csv_str(text)?;
            return Ok(Box::new(DatasetContent::new(dataset)));
        }

        Err(Error::parse("Unable to detect dataset format"))
    }

    fn serialize(&self, content: &dyn ServeableContent) -> Result<Vec<u8>> {
        content.to_bytes()
    }

    fn render_hints(&self) -> RenderHints {
        RenderHints::new().with_chart_type("table")
    }

    fn version(&self) -> &'static str {
        "1.0.0"
    }

    fn name(&self) -> &'static str {
        "Dataset Plugin"
    }

    fn description(&self) -> &'static str {
        "Handles Arrow/Parquet/CSV/JSON datasets"
    }
}

/// Wrapper for ArrowDataset as ServeableContent
struct DatasetContent {
    dataset: ArrowDataset,
    name: String,
}

impl DatasetContent {
    fn new(dataset: ArrowDataset) -> Self {
        Self {
            dataset,
            name: "dataset".to_string(),
        }
    }

    #[allow(dead_code)]
    fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }
}

impl ServeableContent for DatasetContent {
    fn schema(&self) -> ContentSchema {
        ContentSchema::new(ContentTypeId::dataset(), "1.0")
    }

    fn validate(&self) -> Result<ValidationReport> {
        Ok(ValidationReport::success())
    }

    fn to_arrow(&self) -> Result<RecordBatch> {
        self.dataset
            .get(0)
            .ok_or_else(|| Error::data("Empty dataset"))
    }

    fn metadata(&self) -> ContentMetadata {
        ContentMetadata::new(ContentTypeId::dataset(), &self.name, 0)
            .with_row_count(self.dataset.len())
    }

    fn content_type(&self) -> ContentTypeId {
        ContentTypeId::dataset()
    }

    fn chunks(&self, _chunk_size: usize) -> Box<dyn Iterator<Item = Result<RecordBatch>> + Send> {
        let batches: Vec<_> = self.dataset.iter().collect();
        Box::new(batches.into_iter().map(Ok))
    }

    fn to_bytes(&self) -> Result<Vec<u8>> {
        // Return parquet bytes
        self.dataset.to_parquet_bytes()
    }
}

/// Raw data plugin for pasted/clipboard content
pub struct RawPlugin;

impl RawPlugin {
    /// Create a new raw plugin
    pub fn new() -> Self {
        Self
    }
}

impl Default for RawPlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl ContentPlugin for RawPlugin {
    fn content_type(&self) -> ContentTypeId {
        ContentTypeId::raw()
    }

    fn schema(&self) -> ContentSchema {
        ContentSchema::new(ContentTypeId::raw(), "1.0")
            .with_field(
                FieldDefinition::new("data", FieldType::String)
                    .with_description("Raw data content"),
            )
            .with_field(
                FieldDefinition::new("source", FieldType::String)
                    .with_description("Data source (clipboard, stdin, etc.)"),
            )
            .with_field(
                FieldDefinition::new("format", FieldType::String)
                    .with_description("Detected format"),
            )
    }

    fn parse(&self, data: &[u8]) -> Result<BoxedContent> {
        use crate::serve::raw_source::{RawSource, SourceType};

        let text =
            std::str::from_utf8(data).map_err(|e| Error::parse(format!("Invalid UTF-8: {e}")))?;

        let source = RawSource::from_string(text, SourceType::Direct);
        Ok(Box::new(source))
    }

    fn serialize(&self, content: &dyn ServeableContent) -> Result<Vec<u8>> {
        content.to_bytes()
    }

    fn render_hints(&self) -> RenderHints {
        RenderHints::new().with_chart_type("table")
    }

    fn version(&self) -> &'static str {
        "1.0.0"
    }

    fn name(&self) -> &'static str {
        "Raw Data Plugin"
    }

    fn description(&self) -> &'static str {
        "Handles raw/pasted data with automatic format detection"
    }
}

// ============================================================================
// Course Plugin (assetgen integration)
// ============================================================================

/// Course plugin for assetgen course content
#[allow(dead_code)]
pub struct CoursePlugin;

impl CoursePlugin {
    /// Create a new course plugin
    pub fn new() -> Self {
        Self
    }
}

impl Default for CoursePlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl ContentPlugin for CoursePlugin {
    fn content_type(&self) -> ContentTypeId {
        ContentTypeId::course()
    }

    fn schema(&self) -> ContentSchema {
        ContentSchema::new(ContentTypeId::course(), "1.0")
            .with_field(
                FieldDefinition::new("id", FieldType::String)
                    .with_description("Unique course identifier")
                    .with_constraint(Constraint::pattern(r"^[a-z0-9-]+$"))
                    .with_constraint(Constraint::max_length(64)),
            )
            .with_field(
                FieldDefinition::new("title", FieldType::String)
                    .with_description("Course title")
                    .with_constraint(Constraint::min_length(1))
                    .with_constraint(Constraint::max_length(256)),
            )
            .with_field(
                FieldDefinition::new("description", FieldType::String)
                    .with_description("Full course description"),
            )
            .with_field(
                FieldDefinition::new("short_description", FieldType::String)
                    .with_description("Brief course summary")
                    .with_constraint(Constraint::max_length(500)),
            )
            .with_field(
                FieldDefinition::new("categories", FieldType::array(FieldType::String))
                    .with_description("Course categories"),
            )
            .with_field(
                FieldDefinition::new("weeks", FieldType::Integer)
                    .with_description("Number of weeks")
                    .with_constraint(Constraint::min(1.0))
                    .with_constraint(Constraint::max(52.0)),
            )
            .with_field(
                FieldDefinition::new("featured", FieldType::Boolean)
                    .with_description("Whether course is featured")
                    .with_default(serde_json::json!(false)),
            )
            .with_required("id")
            .with_required("title")
            .with_required("description")
            .with_required("weeks")
    }

    fn parse(&self, data: &[u8]) -> Result<BoxedContent> {
        let text =
            std::str::from_utf8(data).map_err(|e| Error::parse(format!("Invalid UTF-8: {e}")))?;

        // Parse as JSON (course outline format)
        let course: CourseContent = serde_json::from_str(text)
            .map_err(|e| Error::parse(format!("Invalid course JSON: {e}")))?;

        Ok(Box::new(course))
    }

    fn serialize(&self, content: &dyn ServeableContent) -> Result<Vec<u8>> {
        content.to_bytes()
    }

    fn render_hints(&self) -> RenderHints {
        RenderHints::new()
            .with_chart_type("course")
            .with_option("show_progress", serde_json::json!(true))
    }

    fn version(&self) -> &'static str {
        "1.0.0"
    }

    fn name(&self) -> &'static str {
        "Course Plugin"
    }

    fn description(&self) -> &'static str {
        "Handles assetgen course content"
    }
}

/// Course content structure (aligned with assetgen)
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CourseContent {
    /// Course ID
    pub id: String,
    /// Course title
    pub title: String,
    /// Full description
    pub description: String,
    /// Short description
    #[serde(default)]
    pub short_description: String,
    /// Categories
    #[serde(default)]
    pub categories: Vec<String>,
    /// Number of weeks
    pub weeks: u32,
    /// Featured flag
    #[serde(default)]
    pub featured: bool,
    /// Difficulty level
    #[serde(default)]
    pub difficulty: Option<String>,
    /// Prerequisites
    #[serde(default)]
    pub prerequisites: Vec<String>,
    /// Course outline
    #[serde(default)]
    pub outline: Option<CourseOutline>,
}

/// Course outline structure
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CourseOutline {
    /// Outline title
    pub title: String,
    /// Weeks in the course
    #[serde(default)]
    pub weeks: Vec<Week>,
}

/// Week structure
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Week {
    /// Week number
    pub number: u32,
    /// Week title
    pub title: String,
    /// Lessons in this week
    #[serde(default)]
    pub lessons: Vec<Lesson>,
}

/// Lesson structure
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lesson {
    /// Lesson number (e.g., "1.1", "1.2")
    pub number: String,
    /// Lesson title
    pub title: String,
    /// Lesson assets
    #[serde(default)]
    pub assets: Vec<Asset>,
}

/// Asset structure
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Asset {
    /// Asset filename
    pub filename: String,
    /// Asset type
    #[serde(rename = "type")]
    pub kind: AssetType,
    /// Asset description
    #[serde(default)]
    pub description: Option<String>,
}

/// Asset type enumeration
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum AssetType {
    /// Video asset
    Video,
    /// Key terms asset
    KeyTerms,
    /// Quiz asset
    Quiz,
    /// Lab asset
    Lab,
    /// Reflection asset
    Reflection,
}

impl ServeableContent for CourseContent {
    fn schema(&self) -> ContentSchema {
        CoursePlugin::new().schema()
    }

    fn validate(&self) -> Result<ValidationReport> {
        let mut report = ValidationReport::success();

        if self.id.is_empty() {
            report = report.with_error(crate::serve::content::ValidationError::new(
                "id",
                "Course ID is required",
            ));
        }

        if self.title.is_empty() {
            report = report.with_error(crate::serve::content::ValidationError::new(
                "title",
                "Course title is required",
            ));
        }

        if self.weeks == 0 {
            report = report.with_error(crate::serve::content::ValidationError::new(
                "weeks",
                "Course must have at least 1 week",
            ));
        }

        Ok(report)
    }

    fn to_arrow(&self) -> Result<RecordBatch> {
        // Convert course to a simple table representation
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("title", DataType::Utf8, false),
            Field::new("description", DataType::Utf8, false),
            Field::new("weeks", DataType::Utf8, false),
            Field::new("categories", DataType::Utf8, false),
        ]));

        let id_array: ArrayRef = Arc::new(StringArray::from(vec![self.id.as_str()]));
        let title_array: ArrayRef = Arc::new(StringArray::from(vec![self.title.as_str()]));
        let desc_array: ArrayRef = Arc::new(StringArray::from(vec![self.description.as_str()]));
        let weeks_array: ArrayRef = Arc::new(StringArray::from(vec![self.weeks.to_string()]));
        let cats_array: ArrayRef = Arc::new(StringArray::from(vec![self.categories.join(", ")]));

        RecordBatch::try_new(
            schema,
            vec![id_array, title_array, desc_array, weeks_array, cats_array],
        )
        .map_err(|e| Error::data(format!("Failed to create course batch: {e}")))
    }

    fn metadata(&self) -> ContentMetadata {
        ContentMetadata::new(ContentTypeId::course(), &self.title, 0)
            .with_description(&self.description)
            .with_custom("id", serde_json::json!(&self.id))
            .with_custom("weeks", serde_json::json!(self.weeks))
            .with_custom("categories", serde_json::json!(&self.categories))
    }

    fn content_type(&self) -> ContentTypeId {
        ContentTypeId::course()
    }

    fn chunks(&self, _chunk_size: usize) -> Box<dyn Iterator<Item = Result<RecordBatch>> + Send> {
        let batch_result = self.to_arrow();
        Box::new(std::iter::once(batch_result))
    }

    fn to_bytes(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(self)
            .map_err(|e| Error::data(format!("Failed to serialize course: {e}")))
    }
}

// ============================================================================
// Book Plugin (assetgen book/chapter integration)
// ============================================================================

/// Book plugin for assetgen book content with chapters
#[allow(dead_code)]
pub struct BookPlugin;

impl BookPlugin {
    /// Create a new book plugin
    pub fn new() -> Self {
        Self
    }
}

impl Default for BookPlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl ContentPlugin for BookPlugin {
    fn content_type(&self) -> ContentTypeId {
        ContentTypeId::new("assetgen.book")
    }

    fn schema(&self) -> ContentSchema {
        ContentSchema::new(ContentTypeId::new("assetgen.book"), "1.0")
            .with_field(
                FieldDefinition::new("id", FieldType::String).with_description("Book identifier"),
            )
            .with_field(
                FieldDefinition::new("title", FieldType::String).with_description("Book title"),
            )
            .with_field(
                FieldDefinition::new("author", FieldType::String).with_description("Book author"),
            )
            .with_field(
                FieldDefinition::new("description", FieldType::String)
                    .with_description("Book description"),
            )
            .with_field(
                FieldDefinition::new("version", FieldType::String).with_description("Book version"),
            )
            .with_field(
                FieldDefinition::new(
                    "chapters",
                    FieldType::array(FieldType::Object {
                        schema: Box::new(ContentSchema::new(
                            ContentTypeId::new("assetgen.chapter"),
                            "1.0",
                        )),
                    }),
                )
                .with_description("Book chapters"),
            )
            .with_required("id")
            .with_required("title")
    }

    fn parse(&self, data: &[u8]) -> Result<BoxedContent> {
        let text =
            std::str::from_utf8(data).map_err(|e| Error::parse(format!("Invalid UTF-8: {e}")))?;

        let book: BookContent = serde_json::from_str(text)
            .map_err(|e| Error::parse(format!("Invalid book JSON: {e}")))?;

        Ok(Box::new(book))
    }

    fn serialize(&self, content: &dyn ServeableContent) -> Result<Vec<u8>> {
        content.to_bytes()
    }

    fn render_hints(&self) -> RenderHints {
        RenderHints::new()
            .with_chart_type("book")
            .with_option("show_progress", serde_json::json!(true))
            .with_option("enable_bookmarks", serde_json::json!(true))
    }

    fn version(&self) -> &'static str {
        "1.0.0"
    }

    fn name(&self) -> &'static str {
        "Book Plugin"
    }

    fn description(&self) -> &'static str {
        "Handles assetgen book content with chapters"
    }
}

/// Book content structure (aligned with assetgen books)
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BookContent {
    /// Book ID
    pub id: String,
    /// Book title
    pub title: String,
    /// Book author
    pub author: String,
    /// Book description
    #[serde(default)]
    pub description: String,
    /// Book version
    #[serde(default)]
    pub version: String,
    /// Source URL
    #[serde(default)]
    pub source_url: Option<String>,
    /// Book settings
    #[serde(default)]
    pub settings: Option<BookSettings>,
    /// Book chapters
    #[serde(default)]
    pub chapters: Vec<Chapter>,
}

/// Book feature enumeration
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum BookFeature {
    /// Enable terminal component
    Terminal,
    /// Enable quizzes
    Quizzes,
    /// Enable labs
    Labs,
    /// Show progress indicator
    Progress,
    /// Enable bookmarks
    Bookmarks,
}

/// Book settings
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BookSettings {
    /// Enabled features (set of feature flags)
    #[serde(default)]
    pub features: std::collections::HashSet<BookFeature>,
    /// Default Python version
    #[serde(default)]
    pub default_python_version: Option<String>,
    /// Required packages
    #[serde(default)]
    pub required_packages: Vec<String>,
    /// Navigation type (sequential, free)
    #[serde(default)]
    pub navigation_type: Option<String>,
}

/// Chapter structure
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chapter {
    /// Chapter ID
    pub id: String,
    /// Chapter title
    pub title: String,
    /// Chapter order
    #[serde(default)]
    pub order: u32,
    /// Source file path
    #[serde(default)]
    pub source_file: Option<String>,
    /// Interactive components
    #[serde(default)]
    pub components: Vec<ChapterComponent>,
    /// Chapter settings
    #[serde(default)]
    pub settings: Option<ChapterSettings>,
}

/// Chapter settings
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ChapterSettings {
    /// Estimated reading time
    #[serde(default)]
    pub estimated_time: Option<String>,
    /// Difficulty level
    #[serde(default)]
    pub difficulty: Option<String>,
    /// Prerequisites (other chapter IDs)
    #[serde(default)]
    pub prerequisites: Vec<String>,
}

/// Interactive chapter component
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChapterComponent {
    /// Component type (terminal, quiz, lab)
    #[serde(rename = "type")]
    pub kind: ComponentType,
    /// Component ID
    pub id: String,
    /// Position in chapter
    #[serde(default)]
    pub position: Option<String>,
    /// Component configuration
    #[serde(default)]
    pub config: Option<serde_json::Value>,
}

/// Component type enumeration
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ComponentType {
    /// Interactive terminal
    Terminal,
    /// Quiz component
    Quiz,
    /// Lab exercise
    Lab,
    /// Code editor
    Editor,
    /// Visualization
    Visualization,
}

impl ServeableContent for BookContent {
    fn schema(&self) -> ContentSchema {
        BookPlugin::new().schema()
    }

    fn validate(&self) -> Result<ValidationReport> {
        let mut report = ValidationReport::success();

        if self.id.is_empty() {
            report = report.with_error(crate::serve::content::ValidationError::new(
                "id",
                "Book ID is required",
            ));
        }

        if self.title.is_empty() {
            report = report.with_error(crate::serve::content::ValidationError::new(
                "title",
                "Book title is required",
            ));
        }

        if self.author.is_empty() {
            report = report.with_error(crate::serve::content::ValidationError::new(
                "author",
                "Book author is required",
            ));
        }

        // Validate chapters
        for (i, chapter) in self.chapters.iter().enumerate() {
            if chapter.id.is_empty() {
                report = report.with_error(crate::serve::content::ValidationError::new(
                    format!("chapters[{}].id", i),
                    "Chapter ID is required",
                ));
            }
            if chapter.title.is_empty() {
                report = report.with_error(crate::serve::content::ValidationError::new(
                    format!("chapters[{}].title", i),
                    "Chapter title is required",
                ));
            }
        }

        Ok(report)
    }

    fn to_arrow(&self) -> Result<RecordBatch> {
        // Convert book chapters to a table representation
        let schema = Arc::new(Schema::new(vec![
            Field::new("chapter_id", DataType::Utf8, false),
            Field::new("chapter_title", DataType::Utf8, false),
            Field::new("order", DataType::Utf8, false),
            Field::new("difficulty", DataType::Utf8, true),
            Field::new("estimated_time", DataType::Utf8, true),
        ]));

        if self.chapters.is_empty() {
            // Return empty batch with schema
            return Ok(RecordBatch::new_empty(schema));
        }

        let chapter_ids: Vec<&str> = self.chapters.iter().map(|c| c.id.as_str()).collect();
        let chapter_titles: Vec<&str> = self.chapters.iter().map(|c| c.title.as_str()).collect();
        let orders: Vec<String> = self.chapters.iter().map(|c| c.order.to_string()).collect();
        let difficulties: Vec<Option<&str>> = self
            .chapters
            .iter()
            .map(|c| c.settings.as_ref().and_then(|s| s.difficulty.as_deref()))
            .collect();
        let times: Vec<Option<&str>> = self
            .chapters
            .iter()
            .map(|c| {
                c.settings
                    .as_ref()
                    .and_then(|s| s.estimated_time.as_deref())
            })
            .collect();

        let id_array: ArrayRef = Arc::new(StringArray::from(chapter_ids));
        let title_array: ArrayRef = Arc::new(StringArray::from(chapter_titles));
        let order_array: ArrayRef = Arc::new(StringArray::from(orders));
        let diff_array: ArrayRef = Arc::new(StringArray::from(difficulties));
        let time_array: ArrayRef = Arc::new(StringArray::from(times));

        RecordBatch::try_new(
            schema,
            vec![id_array, title_array, order_array, diff_array, time_array],
        )
        .map_err(|e| Error::data(format!("Failed to create book batch: {e}")))
    }

    fn metadata(&self) -> ContentMetadata {
        ContentMetadata::new(ContentTypeId::new("assetgen.book"), &self.title, 0)
            .with_description(&self.description)
            .with_custom("id", serde_json::json!(&self.id))
            .with_custom("author", serde_json::json!(&self.author))
            .with_custom("version", serde_json::json!(&self.version))
            .with_custom("chapter_count", serde_json::json!(self.chapters.len()))
    }

    fn content_type(&self) -> ContentTypeId {
        ContentTypeId::new("assetgen.book")
    }

    fn chunks(&self, _chunk_size: usize) -> Box<dyn Iterator<Item = Result<RecordBatch>> + Send> {
        let batch_result = self.to_arrow();
        Box::new(std::iter::once(batch_result))
    }

    fn to_bytes(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(self).map_err(|e| Error::data(format!("Failed to serialize book: {e}")))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::manual_string_new)]
mod tests {
    use super::*;

    #[test]
    fn test_plugin_registry() {
        let registry = PluginRegistry::new();

        assert!(registry.has(&ContentTypeId::dataset()));
        assert!(registry.has(&ContentTypeId::raw()));
        assert!(!registry.has(&ContentTypeId::course()));
    }

    #[test]
    fn test_register_course_plugin() {
        let mut registry = PluginRegistry::new();
        registry.register(Box::new(CoursePlugin::new()));

        assert!(registry.has(&ContentTypeId::course()));
        assert_eq!(registry.len(), 3);
    }

    #[test]
    fn test_dataset_plugin_schema() {
        let plugin = DatasetPlugin::new();
        let schema = plugin.schema();

        assert_eq!(schema.content_type, ContentTypeId::dataset());
        assert!(schema.is_required("name"));
    }

    #[test]
    fn test_course_plugin_schema() {
        let plugin = CoursePlugin::new();
        let schema = plugin.schema();

        assert_eq!(schema.content_type, ContentTypeId::course());
        assert!(schema.is_required("id"));
        assert!(schema.is_required("title"));
    }

    #[test]
    fn test_render_hints() {
        let hints = RenderHints::new()
            .with_chart_type("scatter")
            .with_x_column("x")
            .with_y_column("y")
            .with_option("point_size", serde_json::json!(5));

        assert_eq!(hints.chart_type, Some("scatter".to_string()));
        assert_eq!(hints.x_column, Some("x".to_string()));
        assert!(hints.options.contains_key("point_size"));
    }

    #[test]
    fn test_course_content_validation() {
        let valid_course = CourseContent {
            id: "rust-fundamentals".to_string(),
            title: "Rust Fundamentals".to_string(),
            description: "Learn Rust programming".to_string(),
            short_description: "Learn Rust".to_string(),
            categories: vec!["programming".to_string()],
            weeks: 4,
            featured: false,
            difficulty: Some("beginner".to_string()),
            prerequisites: vec![],
            outline: None,
        };

        let report = valid_course.validate().unwrap();
        assert!(report.valid);

        let invalid_course = CourseContent {
            id: "".to_string(),
            title: "".to_string(),
            description: "".to_string(),
            short_description: "".to_string(),
            categories: vec![],
            weeks: 0,
            featured: false,
            difficulty: None,
            prerequisites: vec![],
            outline: None,
        };

        let report = invalid_course.validate().unwrap();
        assert!(!report.valid);
        assert!(!report.errors.is_empty());
    }

    #[test]
    fn test_course_to_arrow() {
        let course = CourseContent {
            id: "test-course".to_string(),
            title: "Test Course".to_string(),
            description: "A test course".to_string(),
            short_description: "Test".to_string(),
            categories: vec!["test".to_string()],
            weeks: 2,
            featured: false,
            difficulty: None,
            prerequisites: vec![],
            outline: None,
        };

        let batch = course.to_arrow().unwrap();
        assert_eq!(batch.num_rows(), 1);
        assert_eq!(batch.num_columns(), 5);
    }

    #[test]
    fn test_dataset_plugin_parse_csv() {
        let plugin = DatasetPlugin::new();
        let csv_data = b"name,age\nAlice,30\nBob,25";
        let content = plugin.parse(csv_data).unwrap();
        let batch = content.to_arrow().unwrap();
        // CSV with header has 2 data rows
        assert!(batch.num_rows() >= 1);
        assert!(batch.num_columns() >= 1);
    }

    #[test]
    fn test_dataset_plugin_parse_json() {
        let plugin = DatasetPlugin::new();
        // Arrow JSON reader expects newline-delimited JSON objects
        let json_data = b"{\"name\":\"Alice\",\"age\":30}\n{\"name\":\"Bob\",\"age\":25}";
        let content = plugin.parse(json_data).unwrap();
        let batch = content.to_arrow().unwrap();
        assert!(batch.num_rows() >= 1);
    }

    #[test]
    fn test_dataset_plugin_serialize() {
        use crate::ArrowDataset;

        let csv_data = "name,value\na,1\nb,2";
        let dataset = ArrowDataset::from_csv_str(csv_data).unwrap();
        let content = DatasetContent::new(dataset);

        let plugin = DatasetPlugin::new();
        let bytes = plugin.serialize(&content).unwrap();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_dataset_plugin_version_and_name() {
        let plugin = DatasetPlugin::new();
        assert_eq!(plugin.version(), "1.0.0");
        assert_eq!(plugin.name(), "Dataset Plugin");
        assert!(!plugin.description().is_empty());
    }

    #[test]
    fn test_dataset_content_metadata() {
        use crate::ArrowDataset;

        let csv_data = "name,value\na,1\nb,2";
        let dataset = ArrowDataset::from_csv_str(csv_data).unwrap();
        let content = DatasetContent::new(dataset);

        let meta = content.metadata();
        assert_eq!(meta.content_type, ContentTypeId::dataset());
        assert_eq!(meta.row_count, Some(2));
    }

    #[test]
    fn test_dataset_content_chunks() {
        use crate::ArrowDataset;

        let csv_data = "name,value\na,1\nb,2";
        let dataset = ArrowDataset::from_csv_str(csv_data).unwrap();
        let content = DatasetContent::new(dataset);

        let chunks: Vec<_> = content.chunks(100).collect();
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].is_ok());
    }

    #[test]
    fn test_raw_plugin_parse() {
        let plugin = RawPlugin::new();
        let data = b"line1\nline2\nline3";
        let content = plugin.parse(data).unwrap();
        let batch = content.to_arrow().unwrap();
        assert!(batch.num_rows() > 0);
    }

    #[test]
    fn test_raw_plugin_version_and_name() {
        let plugin = RawPlugin::new();
        assert_eq!(plugin.version(), "1.0.0");
        assert_eq!(plugin.name(), "Raw Data Plugin");
        assert!(!plugin.description().is_empty());
    }

    #[test]
    fn test_raw_plugin_render_hints() {
        let plugin = RawPlugin::new();
        let hints = plugin.render_hints();
        assert_eq!(hints.chart_type, Some("table".to_string()));
    }

    #[test]
    fn test_course_plugin_parse() {
        let plugin = CoursePlugin::new();
        let json = r#"{"id":"test","title":"Test","description":"Desc","weeks":4}"#;
        let content = plugin.parse(json.as_bytes()).unwrap();
        let batch = content.to_arrow().unwrap();
        assert_eq!(batch.num_rows(), 1);
    }

    #[test]
    fn test_course_plugin_version_and_name() {
        let plugin = CoursePlugin::new();
        assert_eq!(plugin.version(), "1.0.0");
        assert_eq!(plugin.name(), "Course Plugin");
        assert!(!plugin.description().is_empty());
    }

    #[test]
    fn test_course_plugin_render_hints() {
        let plugin = CoursePlugin::new();
        let hints = plugin.render_hints();
        assert_eq!(hints.chart_type, Some("course".to_string()));
        assert!(hints.options.contains_key("show_progress"));
    }

    #[test]
    fn test_course_content_metadata() {
        let course = CourseContent {
            id: "test".to_string(),
            title: "Test".to_string(),
            description: "Description".to_string(),
            short_description: "Short".to_string(),
            categories: vec!["cat1".to_string()],
            weeks: 3,
            featured: true,
            difficulty: Some("intermediate".to_string()),
            prerequisites: vec!["prereq1".to_string()],
            outline: None,
        };

        let meta = course.metadata();
        assert_eq!(meta.content_type, ContentTypeId::course());
        assert!(meta.custom.contains_key("weeks"));
    }

    #[test]
    fn test_course_content_chunks() {
        let course = CourseContent {
            id: "test".to_string(),
            title: "Test".to_string(),
            description: "Desc".to_string(),
            short_description: "Short".to_string(),
            categories: vec![],
            weeks: 1,
            featured: false,
            difficulty: None,
            prerequisites: vec![],
            outline: None,
        };

        let chunks: Vec<_> = course.chunks(100).collect();
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn test_course_content_to_bytes() {
        let course = CourseContent {
            id: "test".to_string(),
            title: "Test".to_string(),
            description: "Desc".to_string(),
            short_description: "".to_string(),
            categories: vec![],
            weeks: 1,
            featured: false,
            difficulty: None,
            prerequisites: vec![],
            outline: None,
        };

        let bytes = course.to_bytes().unwrap();
        assert!(!bytes.is_empty());
        let parsed: CourseContent = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(parsed.id, "test");
    }

    #[test]
    fn test_course_content_schema() {
        let course = CourseContent {
            id: "test".to_string(),
            title: "Test".to_string(),
            description: "Desc".to_string(),
            short_description: "".to_string(),
            categories: vec![],
            weeks: 1,
            featured: false,
            difficulty: None,
            prerequisites: vec![],
            outline: None,
        };

        let schema = course.schema();
        assert_eq!(schema.content_type, ContentTypeId::course());
    }

    #[test]
    fn test_plugin_registry_get() {
        let registry = PluginRegistry::new();
        let plugin = registry.get(&ContentTypeId::dataset()).unwrap();
        assert_eq!(plugin.name(), "Dataset Plugin");
    }

    #[test]
    fn test_plugin_registry_content_types() {
        let registry = PluginRegistry::new();
        let types = registry.content_types();
        assert!(types.contains(&ContentTypeId::dataset()));
        assert!(types.contains(&ContentTypeId::raw()));
    }

    #[test]
    fn test_plugin_registry_is_empty() {
        let registry = PluginRegistry::new();
        assert!(!registry.is_empty());
    }

    // Book Plugin Tests
    #[test]
    fn test_book_plugin_schema() {
        let plugin = BookPlugin::new();
        let schema = plugin.schema();
        assert_eq!(schema.content_type, ContentTypeId::new("assetgen.book"));
        assert!(schema.is_required("id"));
        assert!(schema.is_required("title"));
    }

    #[test]
    fn test_book_plugin_version_and_name() {
        let plugin = BookPlugin::new();
        assert_eq!(plugin.version(), "1.0.0");
        assert_eq!(plugin.name(), "Book Plugin");
        assert!(!plugin.description().is_empty());
    }

    #[test]
    fn test_book_plugin_render_hints() {
        let plugin = BookPlugin::new();
        let hints = plugin.render_hints();
        assert_eq!(hints.chart_type, Some("book".to_string()));
        assert!(hints.options.contains_key("show_progress"));
        assert!(hints.options.contains_key("enable_bookmarks"));
    }

    #[test]
    fn test_book_plugin_parse() {
        let plugin = BookPlugin::new();
        let json = r#"{"id":"test-book","title":"Test Book","author":"Author"}"#;
        let content = plugin.parse(json.as_bytes()).unwrap();
        let batch = content.to_arrow().unwrap();
        // Empty book has no chapters, returns empty batch
        assert_eq!(batch.num_rows(), 0);
    }

    #[test]
    fn test_book_content_validation_valid() {
        let book = BookContent {
            id: "test-book".to_string(),
            title: "Test Book".to_string(),
            author: "Test Author".to_string(),
            description: "A test book".to_string(),
            version: "1.0".to_string(),
            source_url: None,
            settings: None,
            chapters: vec![Chapter {
                id: "ch1".to_string(),
                title: "Chapter 1".to_string(),
                order: 1,
                source_file: None,
                components: vec![],
                settings: None,
            }],
        };

        let report = book.validate().unwrap();
        assert!(report.valid);
    }

    #[test]
    fn test_book_content_validation_invalid() {
        let book = BookContent {
            id: "".to_string(),
            title: "".to_string(),
            author: "".to_string(),
            description: "".to_string(),
            version: "".to_string(),
            source_url: None,
            settings: None,
            chapters: vec![],
        };

        let report = book.validate().unwrap();
        assert!(!report.valid);
        assert!(!report.errors.is_empty());
    }

    #[test]
    fn test_book_content_to_arrow() {
        let book = BookContent {
            id: "test-book".to_string(),
            title: "Test Book".to_string(),
            author: "Author".to_string(),
            description: "Desc".to_string(),
            version: "1.0".to_string(),
            source_url: None,
            settings: None,
            chapters: vec![
                Chapter {
                    id: "ch1".to_string(),
                    title: "Introduction".to_string(),
                    order: 1,
                    source_file: Some("01-intro.md".to_string()),
                    components: vec![],
                    settings: Some(ChapterSettings {
                        estimated_time: Some("15 minutes".to_string()),
                        difficulty: Some("beginner".to_string()),
                        prerequisites: vec![],
                    }),
                },
                Chapter {
                    id: "ch2".to_string(),
                    title: "Getting Started".to_string(),
                    order: 2,
                    source_file: None,
                    components: vec![],
                    settings: None,
                },
            ],
        };

        let batch = book.to_arrow().unwrap();
        assert_eq!(batch.num_rows(), 2);
        assert_eq!(batch.num_columns(), 5);
    }

    #[test]
    fn test_book_content_metadata() {
        let book = BookContent {
            id: "minimal-python".to_string(),
            title: "Minimal Python".to_string(),
            author: "Noah Gift".to_string(),
            description: "A minimal Python book".to_string(),
            version: "1.0.0".to_string(),
            source_url: Some("https://example.com".to_string()),
            settings: None,
            chapters: vec![Chapter {
                id: "ch1".to_string(),
                title: "Ch1".to_string(),
                order: 1,
                source_file: None,
                components: vec![],
                settings: None,
            }],
        };

        let meta = book.metadata();
        assert_eq!(meta.content_type, ContentTypeId::new("assetgen.book"));
        assert!(meta.custom.contains_key("author"));
        assert!(meta.custom.contains_key("chapter_count"));
    }

    #[test]
    fn test_book_content_to_bytes() {
        let book = BookContent {
            id: "test".to_string(),
            title: "Test".to_string(),
            author: "Author".to_string(),
            description: "".to_string(),
            version: "".to_string(),
            source_url: None,
            settings: None,
            chapters: vec![],
        };

        let bytes = book.to_bytes().unwrap();
        assert!(!bytes.is_empty());
        let parsed: BookContent = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(parsed.id, "test");
    }

    #[test]
    fn test_chapter_with_components() {
        let chapter = Chapter {
            id: "intro".to_string(),
            title: "Introduction".to_string(),
            order: 1,
            source_file: Some("content/01-intro.md".to_string()),
            components: vec![
                ChapterComponent {
                    kind: ComponentType::Terminal,
                    id: "intro-terminal".to_string(),
                    position: Some("after-paragraph-2".to_string()),
                    config: Some(serde_json::json!({
                        "initial_code": "print('Hello!')",
                        "height": "300px"
                    })),
                },
                ChapterComponent {
                    kind: ComponentType::Quiz,
                    id: "intro-quiz".to_string(),
                    position: Some("end-of-chapter".to_string()),
                    config: None,
                },
            ],
            settings: Some(ChapterSettings {
                estimated_time: Some("15 minutes".to_string()),
                difficulty: Some("beginner".to_string()),
                prerequisites: vec![],
            }),
        };

        assert_eq!(chapter.components.len(), 2);
        assert_eq!(chapter.components[0].kind, ComponentType::Terminal);
        assert_eq!(chapter.components[1].kind, ComponentType::Quiz);
    }

    #[test]
    fn test_register_book_plugin() {
        let mut registry = PluginRegistry::new();
        registry.register(Box::new(BookPlugin::new()));

        assert!(registry.has(&ContentTypeId::new("assetgen.book")));
        assert_eq!(registry.len(), 3); // dataset, raw, book
    }
}
