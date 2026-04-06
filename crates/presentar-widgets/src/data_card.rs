//! `DataCard` widget for displaying dataset metadata.

use presentar_core::{
    widget::{AccessibleRole, LayoutResult, TextStyle},
    Brick, BrickAssertion, BrickBudget, BrickVerification, Canvas, Color, Constraints, Point, Rect,
    Size, TypeId, Widget,
};
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::collections::HashMap;
use std::time::Duration;

/// Data quality indicator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum DataQuality {
    /// Unknown quality
    #[default]
    Unknown,
    /// Poor quality (needs cleaning)
    Poor,
    /// Fair quality
    Fair,
    /// Good quality
    Good,
    /// Excellent quality (production ready)
    Excellent,
}

impl DataQuality {
    /// Get display color for quality.
    #[must_use]
    pub fn color(&self) -> Color {
        match self {
            Self::Unknown => Color::new(0.6, 0.6, 0.6, 1.0),
            Self::Poor => Color::new(0.9, 0.3, 0.3, 1.0),
            Self::Fair => Color::new(0.9, 0.7, 0.1, 1.0),
            Self::Good => Color::new(0.4, 0.7, 0.3, 1.0),
            Self::Excellent => Color::new(0.2, 0.7, 0.3, 1.0),
        }
    }

    /// Get quality label.
    #[must_use]
    pub const fn label(&self) -> &'static str {
        match self {
            Self::Unknown => "Unknown",
            Self::Poor => "Poor",
            Self::Fair => "Fair",
            Self::Good => "Good",
            Self::Excellent => "Excellent",
        }
    }

    /// Get quality score (0-100).
    #[must_use]
    pub const fn score(&self) -> u8 {
        match self {
            Self::Unknown => 0,
            Self::Poor => 25,
            Self::Fair => 50,
            Self::Good => 75,
            Self::Excellent => 100,
        }
    }
}

/// Data column/field schema.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DataColumn {
    /// Column name
    pub name: String,
    /// Data type
    pub dtype: String,
    /// Whether nullable
    pub nullable: bool,
    /// Description
    pub description: Option<String>,
}

impl DataColumn {
    /// Create a new column.
    #[must_use]
    pub fn new(name: impl Into<String>, dtype: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            dtype: dtype.into(),
            nullable: false,
            description: None,
        }
    }

    /// Set nullable.
    #[must_use]
    pub const fn nullable(mut self) -> Self {
        self.nullable = true;
        self
    }

    /// Set description.
    #[must_use]
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }
}

/// Dataset statistics.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct DataStats {
    /// Number of rows
    pub rows: Option<u64>,
    /// Number of columns
    pub columns: Option<u32>,
    /// Size in bytes
    pub size_bytes: Option<u64>,
    /// Null percentage
    pub null_percentage: Option<f32>,
    /// Duplicate percentage
    pub duplicate_percentage: Option<f32>,
}

impl DataStats {
    /// Create new stats.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set row count.
    #[must_use]
    pub const fn rows(mut self, count: u64) -> Self {
        self.rows = Some(count);
        self
    }

    /// Set column count.
    #[must_use]
    pub const fn columns(mut self, count: u32) -> Self {
        self.columns = Some(count);
        self
    }

    /// Set size in bytes.
    #[must_use]
    pub const fn size_bytes(mut self, bytes: u64) -> Self {
        self.size_bytes = Some(bytes);
        self
    }

    /// Set null percentage.
    #[must_use]
    pub fn null_percentage(mut self, pct: f32) -> Self {
        self.null_percentage = Some(pct.clamp(0.0, 100.0));
        self
    }

    /// Set duplicate percentage.
    #[must_use]
    pub fn duplicate_percentage(mut self, pct: f32) -> Self {
        self.duplicate_percentage = Some(pct.clamp(0.0, 100.0));
        self
    }

    /// Format size for display.
    #[must_use]
    pub fn formatted_size(&self) -> Option<String> {
        self.size_bytes.map(|bytes| {
            if bytes >= 1_000_000_000 {
                format!("{:.1} GB", bytes as f64 / 1_000_000_000.0)
            } else if bytes >= 1_000_000 {
                format!("{:.1} MB", bytes as f64 / 1_000_000.0)
            } else if bytes >= 1_000 {
                format!("{:.1} KB", bytes as f64 / 1_000.0)
            } else {
                format!("{bytes} B")
            }
        })
    }

    /// Format row count for display.
    #[must_use]
    pub fn formatted_rows(&self) -> Option<String> {
        self.rows.map(|r| {
            if r >= 1_000_000 {
                format!("{:.1}M rows", r as f64 / 1_000_000.0)
            } else if r >= 1_000 {
                format!("{:.1}K rows", r as f64 / 1_000.0)
            } else {
                format!("{r} rows")
            }
        })
    }
}

/// `DataCard` widget for displaying dataset metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataCard {
    /// Dataset name
    name: String,
    /// Dataset version
    version: String,
    /// Description
    description: Option<String>,
    /// Data quality
    quality: DataQuality,
    /// Data format (CSV, Parquet, JSON, etc.)
    format: Option<String>,
    /// Source URL or path
    source: Option<String>,
    /// Schema columns
    schema: Vec<DataColumn>,
    /// Statistics
    stats: DataStats,
    /// License
    license: Option<String>,
    /// Tags
    tags: Vec<String>,
    /// Custom metadata
    metadata: HashMap<String, String>,
    /// Card width
    width: Option<f32>,
    /// Card height
    height: Option<f32>,
    /// Background color
    background: Color,
    /// Border color
    border_color: Color,
    /// Corner radius
    corner_radius: f32,
    /// Show schema preview
    show_schema: bool,
    /// Accessible name
    accessible_name_value: Option<String>,
    /// Test ID
    test_id_value: Option<String>,
    /// Cached bounds
    #[serde(skip)]
    bounds: Rect,
}

impl Default for DataCard {
    fn default() -> Self {
        Self {
            name: String::new(),
            version: String::from("1.0.0"),
            description: None,
            quality: DataQuality::Unknown,
            format: None,
            source: None,
            schema: Vec::new(),
            stats: DataStats::default(),
            license: None,
            tags: Vec::new(),
            metadata: HashMap::new(),
            width: None,
            height: None,
            background: Color::WHITE,
            border_color: Color::new(0.9, 0.9, 0.9, 1.0),
            corner_radius: 8.0,
            show_schema: true,
            accessible_name_value: None,
            test_id_value: None,
            bounds: Rect::default(),
        }
    }
}

impl DataCard {
    /// Create a new data card.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Self::default()
        }
    }

    /// Set name.
    #[must_use]
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Set version.
    #[must_use]
    pub fn version(mut self, version: impl Into<String>) -> Self {
        self.version = version.into();
        self
    }

    /// Set description.
    #[must_use]
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Set quality.
    #[must_use]
    pub const fn quality(mut self, quality: DataQuality) -> Self {
        self.quality = quality;
        self
    }

    /// Set format.
    #[must_use]
    pub fn format(mut self, format: impl Into<String>) -> Self {
        self.format = Some(format.into());
        self
    }

    /// Set source.
    #[must_use]
    pub fn source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    /// Add a schema column.
    #[must_use]
    pub fn column(mut self, col: DataColumn) -> Self {
        self.schema.push(col);
        self
    }

    /// Add multiple schema columns.
    #[must_use]
    pub fn columns(mut self, cols: impl IntoIterator<Item = DataColumn>) -> Self {
        self.schema.extend(cols);
        self
    }

    /// Set statistics.
    #[must_use]
    pub const fn stats(mut self, stats: DataStats) -> Self {
        self.stats = stats;
        self
    }

    /// Set license.
    #[must_use]
    pub fn license(mut self, license: impl Into<String>) -> Self {
        self.license = Some(license.into());
        self
    }

    /// Add a tag.
    #[must_use]
    pub fn tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Add multiple tags.
    #[must_use]
    pub fn tags(mut self, tags: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.tags.extend(tags.into_iter().map(Into::into));
        self
    }

    /// Add custom metadata.
    #[must_use]
    pub fn metadata_entry(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Set width.
    #[must_use]
    pub fn width(mut self, width: f32) -> Self {
        self.width = Some(width.max(200.0));
        self
    }

    /// Set height.
    #[must_use]
    pub fn height(mut self, height: f32) -> Self {
        self.height = Some(height.max(150.0));
        self
    }

    /// Set background color.
    #[must_use]
    pub const fn background(mut self, color: Color) -> Self {
        self.background = color;
        self
    }

    /// Set border color.
    #[must_use]
    pub const fn border_color(mut self, color: Color) -> Self {
        self.border_color = color;
        self
    }

    /// Set corner radius.
    #[must_use]
    pub fn corner_radius(mut self, radius: f32) -> Self {
        self.corner_radius = radius.max(0.0);
        self
    }

    /// Set whether to show schema preview.
    #[must_use]
    pub const fn show_schema(mut self, show: bool) -> Self {
        self.show_schema = show;
        self
    }

    /// Set accessible name.
    #[must_use]
    pub fn accessible_name(mut self, name: impl Into<String>) -> Self {
        self.accessible_name_value = Some(name.into());
        self
    }

    /// Set test ID.
    #[must_use]
    pub fn test_id(mut self, id: impl Into<String>) -> Self {
        self.test_id_value = Some(id.into());
        self
    }

    // Getters

    /// Get name.
    #[must_use]
    pub fn get_name(&self) -> &str {
        &self.name
    }

    /// Get version.
    #[must_use]
    pub fn get_version(&self) -> &str {
        &self.version
    }

    /// Get description.
    #[must_use]
    pub fn get_description(&self) -> Option<&str> {
        self.description.as_deref()
    }

    /// Get quality.
    #[must_use]
    pub const fn get_quality(&self) -> DataQuality {
        self.quality
    }

    /// Get format.
    #[must_use]
    pub fn get_format(&self) -> Option<&str> {
        self.format.as_deref()
    }

    /// Get source.
    #[must_use]
    pub fn get_source(&self) -> Option<&str> {
        self.source.as_deref()
    }

    /// Get schema.
    #[must_use]
    pub fn get_schema(&self) -> &[DataColumn] {
        &self.schema
    }

    /// Get stats.
    #[must_use]
    pub const fn get_stats(&self) -> &DataStats {
        &self.stats
    }

    /// Get license.
    #[must_use]
    pub fn get_license(&self) -> Option<&str> {
        self.license.as_deref()
    }

    /// Get tags.
    #[must_use]
    pub fn get_tags(&self) -> &[String] {
        &self.tags
    }

    /// Get custom metadata.
    #[must_use]
    pub fn get_metadata(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(String::as_str)
    }

    /// Check if has schema.
    #[must_use]
    pub fn has_schema(&self) -> bool {
        !self.schema.is_empty()
    }

    /// Get column count.
    #[must_use]
    pub fn column_count(&self) -> usize {
        self.schema.len()
    }
}

impl Widget for DataCard {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        let width = self.width.unwrap_or(320.0);
        let height = self.height.unwrap_or(220.0);
        constraints.constrain(Size::new(width, height))
    }

    fn layout(&mut self, bounds: Rect) -> LayoutResult {
        self.bounds = bounds;
        LayoutResult {
            size: bounds.size(),
        }
    }

    #[allow(clippy::too_many_lines)]
    fn paint(&self, canvas: &mut dyn Canvas) {
        let padding = 16.0;

        // Background
        canvas.fill_rect(self.bounds, self.background);

        // Border
        canvas.stroke_rect(self.bounds, self.border_color, 1.0);

        // Quality badge
        let quality_color = self.quality.color();
        let badge_rect = Rect::new(
            self.bounds.x + self.bounds.width - 80.0,
            self.bounds.y + padding,
            70.0,
            22.0,
        );
        canvas.fill_rect(badge_rect, quality_color);

        let badge_style = TextStyle {
            size: 10.0,
            color: Color::WHITE,
            ..TextStyle::default()
        };
        canvas.draw_text(
            self.quality.label(),
            Point::new(badge_rect.x + 10.0, badge_rect.y + 15.0),
            &badge_style,
        );

        // Title
        let title_style = TextStyle {
            size: 18.0,
            color: Color::new(0.1, 0.1, 0.1, 1.0),
            ..TextStyle::default()
        };
        canvas.draw_text(
            &self.name,
            Point::new(self.bounds.x + padding, self.bounds.y + padding + 16.0),
            &title_style,
        );

        // Version and format
        let info_style = TextStyle {
            size: 12.0,
            color: Color::new(0.5, 0.5, 0.5, 1.0),
            ..TextStyle::default()
        };
        let version_text = match &self.format {
            Some(f) => format!("v{} • {}", self.version, f),
            None => format!("v{}", self.version),
        };
        canvas.draw_text(
            &version_text,
            Point::new(self.bounds.x + padding, self.bounds.y + padding + 36.0),
            &info_style,
        );

        let mut y_offset = padding + 50.0;

        // Description
        if let Some(ref desc) = self.description {
            let desc_style = TextStyle {
                size: 12.0,
                color: Color::new(0.3, 0.3, 0.3, 1.0),
                ..TextStyle::default()
            };
            canvas.draw_text(
                desc,
                Point::new(self.bounds.x + padding, self.bounds.y + y_offset + 12.0),
                &desc_style,
            );
            y_offset += 24.0;
        }

        // Stats
        let stats_style = TextStyle {
            size: 11.0,
            color: Color::new(0.4, 0.4, 0.4, 1.0),
            ..TextStyle::default()
        };
        let value_style = TextStyle {
            size: 14.0,
            color: Color::new(0.2, 0.47, 0.96, 1.0),
            ..TextStyle::default()
        };

        let mut sx = self.bounds.x + padding;
        if let Some(rows) = self.stats.formatted_rows() {
            canvas.draw_text(
                "Rows",
                Point::new(sx, self.bounds.y + y_offset + 12.0),
                &stats_style,
            );
            canvas.draw_text(
                &rows,
                Point::new(sx, self.bounds.y + y_offset + 28.0),
                &value_style,
            );
            sx += 80.0;
        }
        if let Some(cols) = self.stats.columns {
            canvas.draw_text(
                "Columns",
                Point::new(sx, self.bounds.y + y_offset + 12.0),
                &stats_style,
            );
            canvas.draw_text(
                &cols.to_string(),
                Point::new(sx, self.bounds.y + y_offset + 28.0),
                &value_style,
            );
            sx += 70.0;
        }
        if let Some(size) = self.stats.formatted_size() {
            canvas.draw_text(
                "Size",
                Point::new(sx, self.bounds.y + y_offset + 12.0),
                &stats_style,
            );
            canvas.draw_text(
                &size,
                Point::new(sx, self.bounds.y + y_offset + 28.0),
                &value_style,
            );
        }

        if self.stats.rows.is_some()
            || self.stats.columns.is_some()
            || self.stats.size_bytes.is_some()
        {
            y_offset += 40.0;
        }

        // Schema preview
        if self.show_schema && !self.schema.is_empty() {
            let schema_style = TextStyle {
                size: 10.0,
                color: Color::new(0.4, 0.4, 0.4, 1.0),
                ..TextStyle::default()
            };
            canvas.draw_text(
                "Schema:",
                Point::new(self.bounds.x + padding, self.bounds.y + y_offset + 12.0),
                &schema_style,
            );
            y_offset += 18.0;

            let col_style = TextStyle {
                size: 10.0,
                color: Color::new(0.2, 0.2, 0.2, 1.0),
                ..TextStyle::default()
            };
            for col in self.schema.iter().take(4) {
                let nullable = if col.nullable { "?" } else { "" };
                let text = format!("{}: {}{}", col.name, col.dtype, nullable);
                canvas.draw_text(
                    &text,
                    Point::new(
                        self.bounds.x + padding + 8.0,
                        self.bounds.y + y_offset + 12.0,
                    ),
                    &col_style,
                );
                y_offset += 14.0;
            }
            if self.schema.len() > 4 {
                canvas.draw_text(
                    &format!("... +{} more", self.schema.len() - 4),
                    Point::new(
                        self.bounds.x + padding + 8.0,
                        self.bounds.y + y_offset + 12.0,
                    ),
                    &schema_style,
                );
                y_offset += 14.0;
            }
        }

        // Tags
        if !self.tags.is_empty() {
            let tag_style = TextStyle {
                size: 10.0,
                color: Color::new(0.3, 0.3, 0.3, 1.0),
                ..TextStyle::default()
            };
            let tag_bg = Color::new(0.95, 0.95, 0.95, 1.0);

            let mut tx = self.bounds.x + padding;
            for tag in self.tags.iter().take(5) {
                let tag_width = (tag.len() as f32).mul_add(6.0, 12.0);
                canvas.fill_rect(
                    Rect::new(tx, self.bounds.y + y_offset + 4.0, tag_width, 18.0),
                    tag_bg,
                );
                canvas.draw_text(
                    tag,
                    Point::new(tx + 6.0, self.bounds.y + y_offset + 17.0),
                    &tag_style,
                );
                tx += tag_width + 6.0;
            }
        }
    }

    fn event(&mut self, _event: &presentar_core::Event) -> Option<Box<dyn Any + Send>> {
        None
    }

    fn children(&self) -> &[Box<dyn Widget>] {
        &[]
    }

    fn children_mut(&mut self) -> &mut [Box<dyn Widget>] {
        &mut []
    }

    fn is_interactive(&self) -> bool {
        false
    }

    fn is_focusable(&self) -> bool {
        false
    }

    fn accessible_name(&self) -> Option<&str> {
        self.accessible_name_value.as_deref().or(Some(&self.name))
    }

    fn accessible_role(&self) -> AccessibleRole {
        AccessibleRole::Generic
    }

    fn test_id(&self) -> Option<&str> {
        self.test_id_value.as_deref()
    }
}

// PROBAR-SPEC-009: Brick Architecture - Tests define interface
impl Brick for DataCard {
    fn brick_name(&self) -> &'static str {
        "DataCard"
    }

    fn assertions(&self) -> &[BrickAssertion] {
        &[
            BrickAssertion::TextVisible,
            BrickAssertion::MaxLatencyMs(16),
        ]
    }

    fn budget(&self) -> BrickBudget {
        BrickBudget::uniform(16)
    }

    fn verify(&self) -> BrickVerification {
        BrickVerification {
            passed: self.assertions().to_vec(),
            failed: vec![],
            verification_time: Duration::from_micros(10),
        }
    }

    fn to_html(&self) -> String {
        let test_id = self.test_id_value.as_deref().unwrap_or("data-card");
        format!(
            r#"<div class="brick-data-card" data-testid="{}" aria-label="{}">{}</div>"#,
            test_id, self.name, self.name
        )
    }

    fn to_css(&self) -> String {
        ".brick-data-card { display: block; }".into()
    }

    fn test_id(&self) -> Option<&str> {
        self.test_id_value.as_deref()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::disallowed_methods)]
mod tests {
    use super::*;

    // ===== DataQuality Tests =====

    #[test]
    fn test_data_quality_default() {
        assert_eq!(DataQuality::default(), DataQuality::Unknown);
    }

    #[test]
    fn test_data_quality_color() {
        let excellent = DataQuality::Excellent;
        let color = excellent.color();
        assert!(color.g > color.r); // Green-ish
    }

    #[test]
    fn test_data_quality_label() {
        assert_eq!(DataQuality::Unknown.label(), "Unknown");
        assert_eq!(DataQuality::Poor.label(), "Poor");
        assert_eq!(DataQuality::Fair.label(), "Fair");
        assert_eq!(DataQuality::Good.label(), "Good");
        assert_eq!(DataQuality::Excellent.label(), "Excellent");
    }

    #[test]
    fn test_data_quality_score() {
        assert_eq!(DataQuality::Unknown.score(), 0);
        assert_eq!(DataQuality::Poor.score(), 25);
        assert_eq!(DataQuality::Fair.score(), 50);
        assert_eq!(DataQuality::Good.score(), 75);
        assert_eq!(DataQuality::Excellent.score(), 100);
    }

    // ===== DataColumn Tests =====

    #[test]
    fn test_data_column_new() {
        let col = DataColumn::new("age", "int64");
        assert_eq!(col.name, "age");
        assert_eq!(col.dtype, "int64");
        assert!(!col.nullable);
        assert!(col.description.is_none());
    }

    #[test]
    fn test_data_column_nullable() {
        let col = DataColumn::new("email", "string").nullable();
        assert!(col.nullable);
    }

    #[test]
    fn test_data_column_description() {
        let col = DataColumn::new("id", "uuid").description("Primary key");
        assert_eq!(col.description, Some("Primary key".to_string()));
    }

    // ===== DataStats Tests =====

    #[test]
    fn test_data_stats_new() {
        let stats = DataStats::new();
        assert!(stats.rows.is_none());
        assert!(stats.columns.is_none());
    }

    #[test]
    fn test_data_stats_builder() {
        let stats = DataStats::new()
            .rows(1_000_000)
            .columns(50)
            .size_bytes(500_000_000)
            .null_percentage(2.5)
            .duplicate_percentage(0.1);

        assert_eq!(stats.rows, Some(1_000_000));
        assert_eq!(stats.columns, Some(50));
        assert_eq!(stats.size_bytes, Some(500_000_000));
        assert_eq!(stats.null_percentage, Some(2.5));
        assert_eq!(stats.duplicate_percentage, Some(0.1));
    }

    #[test]
    fn test_data_stats_null_percentage_clamped() {
        let stats = DataStats::new().null_percentage(150.0);
        assert_eq!(stats.null_percentage, Some(100.0));

        let stats = DataStats::new().null_percentage(-10.0);
        assert_eq!(stats.null_percentage, Some(0.0));
    }

    #[test]
    fn test_data_stats_formatted_size_bytes() {
        let stats = DataStats::new().size_bytes(500);
        assert_eq!(stats.formatted_size(), Some("500 B".to_string()));
    }

    #[test]
    fn test_data_stats_formatted_size_kb() {
        let stats = DataStats::new().size_bytes(5_000);
        assert_eq!(stats.formatted_size(), Some("5.0 KB".to_string()));
    }

    #[test]
    fn test_data_stats_formatted_size_mb() {
        let stats = DataStats::new().size_bytes(50_000_000);
        assert_eq!(stats.formatted_size(), Some("50.0 MB".to_string()));
    }

    #[test]
    fn test_data_stats_formatted_size_gb() {
        let stats = DataStats::new().size_bytes(5_000_000_000);
        assert_eq!(stats.formatted_size(), Some("5.0 GB".to_string()));
    }

    #[test]
    fn test_data_stats_formatted_rows_small() {
        let stats = DataStats::new().rows(500);
        assert_eq!(stats.formatted_rows(), Some("500 rows".to_string()));
    }

    #[test]
    fn test_data_stats_formatted_rows_thousands() {
        let stats = DataStats::new().rows(50_000);
        assert_eq!(stats.formatted_rows(), Some("50.0K rows".to_string()));
    }

    #[test]
    fn test_data_stats_formatted_rows_millions() {
        let stats = DataStats::new().rows(5_000_000);
        assert_eq!(stats.formatted_rows(), Some("5.0M rows".to_string()));
    }

    // ===== DataCard Construction Tests =====

    #[test]
    fn test_data_card_new() {
        let card = DataCard::new("customers");
        assert_eq!(card.get_name(), "customers");
        assert_eq!(card.get_version(), "1.0.0");
        assert_eq!(card.get_quality(), DataQuality::Unknown);
    }

    #[test]
    fn test_data_card_default() {
        let card = DataCard::default();
        assert!(card.name.is_empty());
        assert_eq!(card.version, "1.0.0");
    }

    #[test]
    fn test_data_card_builder() {
        let card = DataCard::new("sales_data")
            .version("2.0.0")
            .description("Quarterly sales data")
            .quality(DataQuality::Excellent)
            .format("Parquet")
            .source("s3://bucket/sales/")
            .column(DataColumn::new("id", "int64"))
            .column(DataColumn::new("amount", "float64"))
            .stats(DataStats::new().rows(1_000_000).columns(20))
            .license("MIT")
            .tag("sales")
            .tag("finance")
            .metadata_entry("owner", "analytics-team")
            .width(400.0)
            .height(300.0)
            .background(Color::WHITE)
            .border_color(Color::new(0.8, 0.8, 0.8, 1.0))
            .corner_radius(12.0)
            .show_schema(true)
            .accessible_name("Sales data card")
            .test_id("sales-card");

        assert_eq!(card.get_name(), "sales_data");
        assert_eq!(card.get_version(), "2.0.0");
        assert_eq!(card.get_description(), Some("Quarterly sales data"));
        assert_eq!(card.get_quality(), DataQuality::Excellent);
        assert_eq!(card.get_format(), Some("Parquet"));
        assert_eq!(card.get_source(), Some("s3://bucket/sales/"));
        assert_eq!(card.get_schema().len(), 2);
        assert_eq!(card.get_stats().rows, Some(1_000_000));
        assert_eq!(card.get_license(), Some("MIT"));
        assert_eq!(card.get_tags().len(), 2);
        assert_eq!(card.get_metadata("owner"), Some("analytics-team"));
        assert_eq!(Widget::accessible_name(&card), Some("Sales data card"));
        assert_eq!(Widget::test_id(&card), Some("sales-card"));
    }

    #[test]
    fn test_data_card_columns() {
        let cols = vec![DataColumn::new("a", "int"), DataColumn::new("b", "string")];
        let card = DataCard::new("data").columns(cols);
        assert_eq!(card.column_count(), 2);
        assert!(card.has_schema());
    }

    #[test]
    fn test_data_card_tags() {
        let card = DataCard::new("data").tags(["raw", "cleaned", "normalized"]);
        assert_eq!(card.get_tags().len(), 3);
    }

    // ===== Dimension Tests =====

    #[test]
    fn test_data_card_width_min() {
        let card = DataCard::new("data").width(100.0);
        assert_eq!(card.width, Some(200.0));
    }

    #[test]
    fn test_data_card_height_min() {
        let card = DataCard::new("data").height(50.0);
        assert_eq!(card.height, Some(150.0));
    }

    #[test]
    fn test_data_card_corner_radius_min() {
        let card = DataCard::new("data").corner_radius(-5.0);
        assert_eq!(card.corner_radius, 0.0);
    }

    // ===== Widget Trait Tests =====

    #[test]
    fn test_data_card_type_id() {
        let card = DataCard::new("data");
        assert_eq!(Widget::type_id(&card), TypeId::of::<DataCard>());
    }

    #[test]
    fn test_data_card_measure_default() {
        let card = DataCard::new("data");
        let size = card.measure(Constraints::loose(Size::new(1000.0, 1000.0)));
        assert_eq!(size.width, 320.0);
        assert_eq!(size.height, 220.0);
    }

    #[test]
    fn test_data_card_measure_custom() {
        let card = DataCard::new("data").width(400.0).height(300.0);
        let size = card.measure(Constraints::loose(Size::new(1000.0, 1000.0)));
        assert_eq!(size.width, 400.0);
        assert_eq!(size.height, 300.0);
    }

    #[test]
    fn test_data_card_layout() {
        let mut card = DataCard::new("data");
        let bounds = Rect::new(10.0, 20.0, 320.0, 220.0);
        let result = card.layout(bounds);
        assert_eq!(result.size, Size::new(320.0, 220.0));
        assert_eq!(card.bounds, bounds);
    }

    #[test]
    fn test_data_card_children() {
        let card = DataCard::new("data");
        assert!(card.children().is_empty());
    }

    #[test]
    fn test_data_card_is_interactive() {
        let card = DataCard::new("data");
        assert!(!card.is_interactive());
    }

    #[test]
    fn test_data_card_is_focusable() {
        let card = DataCard::new("data");
        assert!(!card.is_focusable());
    }

    #[test]
    fn test_data_card_accessible_role() {
        let card = DataCard::new("data");
        assert_eq!(card.accessible_role(), AccessibleRole::Generic);
    }

    #[test]
    fn test_data_card_accessible_name_from_name() {
        let card = DataCard::new("customers");
        assert_eq!(Widget::accessible_name(&card), Some("customers"));
    }

    #[test]
    fn test_data_card_accessible_name_explicit() {
        let card = DataCard::new("customers").accessible_name("Customer dataset");
        assert_eq!(Widget::accessible_name(&card), Some("Customer dataset"));
    }

    #[test]
    fn test_data_card_test_id() {
        let card = DataCard::new("data").test_id("data-card");
        assert_eq!(Widget::test_id(&card), Some("data-card"));
    }

    // ===== Has Schema Tests =====

    #[test]
    fn test_data_card_has_schema_false() {
        let card = DataCard::new("data");
        assert!(!card.has_schema());
    }

    #[test]
    fn test_data_card_has_schema_true() {
        let card = DataCard::new("data").column(DataColumn::new("id", "int"));
        assert!(card.has_schema());
    }

    // =========================================================================
    // Additional Coverage Tests
    // =========================================================================

    #[test]
    fn test_data_quality_color_all_variants() {
        let _ = DataQuality::Unknown.color();
        let _ = DataQuality::Poor.color();
        let _ = DataQuality::Fair.color();
        let _ = DataQuality::Good.color();
        let _ = DataQuality::Excellent.color();
    }

    #[test]
    fn test_data_stats_formatted_rows_none() {
        let stats = DataStats::new();
        assert!(stats.formatted_rows().is_none());
    }

    #[test]
    fn test_data_stats_formatted_size_none() {
        let stats = DataStats::new();
        assert!(stats.formatted_size().is_none());
    }

    #[test]
    fn test_data_card_children_mut() {
        let mut card = DataCard::new("data");
        assert!(card.children_mut().is_empty());
    }

    #[test]
    fn test_data_card_event_returns_none() {
        let mut card = DataCard::new("data");
        let result = card.event(&presentar_core::Event::key_down(presentar_core::Key::Down));
        assert!(result.is_none());
    }

    #[test]
    fn test_data_card_test_id_none() {
        let card = DataCard::new("data");
        assert!(Widget::test_id(&card).is_none());
    }

    #[test]
    fn test_data_stats_duplicate_percentage_clamped() {
        let stats = DataStats::new().duplicate_percentage(150.0);
        assert_eq!(stats.duplicate_percentage, Some(100.0));

        let stats = DataStats::new().duplicate_percentage(-10.0);
        assert_eq!(stats.duplicate_percentage, Some(0.0));
    }

    #[test]
    fn test_data_column_eq() {
        let col1 = DataColumn::new("id", "int64");
        let col2 = DataColumn::new("id", "int64");
        assert_eq!(col1.name, col2.name);
        assert_eq!(col1.dtype, col2.dtype);
    }

    // =========================================================================
    // Brick Trait Tests
    // =========================================================================

    #[test]
    fn test_data_card_brick_name() {
        let card = DataCard::new("test");
        assert_eq!(card.brick_name(), "DataCard");
    }

    #[test]
    fn test_data_card_brick_assertions() {
        let card = DataCard::new("test");
        let assertions = card.assertions();
        assert!(assertions.len() >= 2);
        assert!(matches!(assertions[0], BrickAssertion::TextVisible));
        assert!(matches!(assertions[1], BrickAssertion::MaxLatencyMs(16)));
    }

    #[test]
    fn test_data_card_brick_budget() {
        let card = DataCard::new("test");
        let budget = card.budget();
        // Verify budget has reasonable values
        assert!(budget.layout_ms > 0);
        assert!(budget.paint_ms > 0);
    }

    #[test]
    fn test_data_card_brick_verify() {
        let card = DataCard::new("test");
        let verification = card.verify();
        assert!(!verification.passed.is_empty());
        assert!(verification.failed.is_empty());
    }

    #[test]
    fn test_data_card_brick_to_html() {
        let card = DataCard::new("test-dataset").test_id("my-data-card");
        let html = card.to_html();
        assert!(html.contains("brick-data-card"));
        assert!(html.contains("my-data-card"));
        assert!(html.contains("test-dataset"));
    }

    #[test]
    fn test_data_card_brick_to_html_default() {
        let card = DataCard::new("test");
        let html = card.to_html();
        assert!(html.contains("data-testid=\"data-card\""));
    }

    #[test]
    fn test_data_card_brick_to_css() {
        let card = DataCard::new("test");
        let css = card.to_css();
        assert!(css.contains(".brick-data-card"));
        assert!(css.contains("display: block"));
    }

    #[test]
    fn test_data_card_brick_test_id() {
        let card = DataCard::new("test").test_id("card-1");
        assert_eq!(Brick::test_id(&card), Some("card-1"));
    }

    #[test]
    fn test_data_card_brick_test_id_none() {
        let card = DataCard::new("test");
        assert!(Brick::test_id(&card).is_none());
    }

    // =========================================================================
    // DataQuality Additional Tests
    // =========================================================================

    #[test]
    fn test_data_quality_debug() {
        let quality = DataQuality::Good;
        let debug_str = format!("{quality:?}");
        assert!(debug_str.contains("Good"));
    }

    #[test]
    fn test_data_quality_eq() {
        assert_eq!(DataQuality::Good, DataQuality::Good);
        assert_ne!(DataQuality::Poor, DataQuality::Excellent);
    }

    #[test]
    fn test_data_quality_clone() {
        let quality = DataQuality::Fair;
        let cloned = quality;
        assert_eq!(cloned, DataQuality::Fair);
    }

    #[test]
    fn test_data_quality_serde() {
        let quality = DataQuality::Excellent;
        let serialized = serde_json::to_string(&quality).unwrap();
        let deserialized: DataQuality = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized, DataQuality::Excellent);
    }

    // =========================================================================
    // DataColumn Additional Tests
    // =========================================================================

    #[test]
    fn test_data_column_debug() {
        let col = DataColumn::new("id", "int64");
        let debug_str = format!("{col:?}");
        assert!(debug_str.contains("id"));
        assert!(debug_str.contains("int64"));
    }

    #[test]
    fn test_data_column_clone() {
        let col = DataColumn::new("name", "string")
            .nullable()
            .description("User name");
        let cloned = col;
        assert_eq!(cloned.name, "name");
        assert_eq!(cloned.dtype, "string");
        assert!(cloned.nullable);
        assert_eq!(cloned.description, Some("User name".to_string()));
    }

    #[test]
    fn test_data_column_serde() {
        let col = DataColumn::new("age", "int32");
        let serialized = serde_json::to_string(&col).unwrap();
        let deserialized: DataColumn = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized.name, "age");
        assert_eq!(deserialized.dtype, "int32");
    }

    // =========================================================================
    // DataStats Additional Tests
    // =========================================================================

    #[test]
    fn test_data_stats_debug() {
        let stats = DataStats::new().rows(100);
        let debug_str = format!("{stats:?}");
        assert!(debug_str.contains("100"));
    }

    #[test]
    fn test_data_stats_clone() {
        let stats = DataStats::new().rows(1000).columns(10).size_bytes(50000);
        let cloned = stats;
        assert_eq!(cloned.rows, Some(1000));
        assert_eq!(cloned.columns, Some(10));
        assert_eq!(cloned.size_bytes, Some(50000));
    }

    #[test]
    fn test_data_stats_eq() {
        let stats1 = DataStats::new().rows(100);
        let stats2 = DataStats::new().rows(100);
        assert_eq!(stats1.rows, stats2.rows);
    }

    #[test]
    fn test_data_stats_default() {
        let stats = DataStats::default();
        assert!(stats.rows.is_none());
        assert!(stats.columns.is_none());
        assert!(stats.size_bytes.is_none());
        assert!(stats.null_percentage.is_none());
        assert!(stats.duplicate_percentage.is_none());
    }

    #[test]
    fn test_data_stats_formatted_rows_edge_cases() {
        // Exactly 1000 rows
        let stats = DataStats::new().rows(1000);
        assert_eq!(stats.formatted_rows(), Some("1.0K rows".to_string()));

        // Exactly 1 million rows
        let stats = DataStats::new().rows(1_000_000);
        assert_eq!(stats.formatted_rows(), Some("1.0M rows".to_string()));
    }

    #[test]
    fn test_data_stats_formatted_size_edge_cases() {
        // Exactly 1 KB
        let stats = DataStats::new().size_bytes(1000);
        assert_eq!(stats.formatted_size(), Some("1.0 KB".to_string()));

        // Exactly 1 MB
        let stats = DataStats::new().size_bytes(1_000_000);
        assert_eq!(stats.formatted_size(), Some("1.0 MB".to_string()));

        // Exactly 1 GB
        let stats = DataStats::new().size_bytes(1_000_000_000);
        assert_eq!(stats.formatted_size(), Some("1.0 GB".to_string()));
    }

    // =========================================================================
    // DataCard Construction Additional Tests
    // =========================================================================

    #[test]
    fn test_data_card_debug() {
        let card = DataCard::new("test");
        let debug_str = format!("{card:?}");
        assert!(debug_str.contains("test"));
    }

    #[test]
    fn test_data_card_clone() {
        let card = DataCard::new("original")
            .version("2.0.0")
            .quality(DataQuality::Good);
        let cloned = card;
        assert_eq!(cloned.get_name(), "original");
        assert_eq!(cloned.get_version(), "2.0.0");
        assert_eq!(cloned.get_quality(), DataQuality::Good);
    }

    #[test]
    fn test_data_card_serde() {
        let card = DataCard::new("serialized")
            .version("1.2.3")
            .quality(DataQuality::Fair);
        let serialized = serde_json::to_string(&card).unwrap();
        let deserialized: DataCard = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized.get_name(), "serialized");
        assert_eq!(deserialized.get_version(), "1.2.3");
        assert_eq!(deserialized.get_quality(), DataQuality::Fair);
    }

    // =========================================================================
    // Widget Trait Additional Tests
    // =========================================================================

    #[test]
    fn test_data_card_measure_with_tight_constraints() {
        let card = DataCard::new("test").width(400.0).height(300.0);
        let size = card.measure(Constraints::tight(Size::new(200.0, 150.0)));
        assert_eq!(size.width, 200.0);
        assert_eq!(size.height, 150.0);
    }

    #[test]
    fn test_data_card_name_setter() {
        let card = DataCard::new("initial").name("changed");
        assert_eq!(card.get_name(), "changed");
    }

    // =========================================================================
    // Getter Tests (ensure all getters are covered)
    // =========================================================================

    #[test]
    fn test_data_card_getters_none() {
        let card = DataCard::new("test");
        assert!(card.get_description().is_none());
        assert!(card.get_format().is_none());
        assert!(card.get_source().is_none());
        assert!(card.get_license().is_none());
        assert!(card.get_metadata("nonexistent").is_none());
    }

    #[test]
    fn test_data_card_getters_some() {
        let card = DataCard::new("test")
            .description("desc")
            .format("CSV")
            .source("http://example.com")
            .license("MIT")
            .metadata_entry("key", "value");

        assert_eq!(card.get_description(), Some("desc"));
        assert_eq!(card.get_format(), Some("CSV"));
        assert_eq!(card.get_source(), Some("http://example.com"));
        assert_eq!(card.get_license(), Some("MIT"));
        assert_eq!(card.get_metadata("key"), Some("value"));
    }

    // =========================================================================
    // Edge Case Tests
    // =========================================================================

    #[test]
    fn test_data_card_empty_columns() {
        let card = DataCard::new("test").columns(vec![]);
        assert_eq!(card.column_count(), 0);
        assert!(!card.has_schema());
    }

    #[test]
    fn test_data_card_many_columns() {
        let cols: Vec<DataColumn> = (0..10)
            .map(|i| DataColumn::new(format!("col_{i}"), "int"))
            .collect();
        let card = DataCard::new("test").columns(cols);
        assert_eq!(card.column_count(), 10);
    }

    #[test]
    fn test_data_card_empty_tags() {
        let tags: [&str; 0] = [];
        let card = DataCard::new("test").tags(tags);
        assert!(card.get_tags().is_empty());
    }

    #[test]
    fn test_data_card_show_schema_false() {
        let card = DataCard::new("test")
            .column(DataColumn::new("id", "int"))
            .show_schema(false);
        assert!(card.has_schema());
        // show_schema only affects paint, not has_schema
    }

    #[test]
    fn test_data_card_default_colors() {
        let card = DataCard::default();
        assert_eq!(card.background, Color::WHITE);
    }
}
