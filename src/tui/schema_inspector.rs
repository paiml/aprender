//! Schema inspector widget for displaying Arrow schema information
//!
//! Displays field names, types, and nullable status in a table format.

use super::{adapter::DatasetAdapter, format::truncate_string};

/// Schema inspector widget for displaying dataset schema
///
/// Shows field names, Arrow types, and nullable status.
///
/// # Example
///
/// ```ignore
/// let adapter = DatasetAdapter::from_dataset(&dataset)?;
/// let inspector = SchemaInspector::new(&adapter);
///
/// for line in inspector.render_lines() {
///     println!("{}", line);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct SchemaInspector {
    /// Field information: (name, type, nullable)
    fields: Vec<FieldInfo>,
    /// Display width
    display_width: u16,
}

/// Information about a single field
#[derive(Debug, Clone)]
struct FieldInfo {
    name: String,
    type_name: String,
    nullable: bool,
}

impl SchemaInspector {
    /// Create a new schema inspector from a dataset adapter
    pub fn new(adapter: &DatasetAdapter) -> Self {
        Self::with_width(adapter, 80)
    }

    /// Create a new schema inspector with specific width
    pub fn with_width(adapter: &DatasetAdapter, width: u16) -> Self {
        let schema = adapter.schema();
        let fields: Vec<FieldInfo> = schema
            .fields()
            .iter()
            .map(|f| FieldInfo {
                name: f.name().clone(),
                type_name: format_type_name(f.data_type()),
                nullable: f.is_nullable(),
            })
            .collect();

        Self {
            fields,
            display_width: width,
        }
    }

    /// Get the number of fields
    pub fn field_count(&self) -> usize {
        self.fields.len()
    }

    /// Check if schema is empty
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    /// Get the display width
    pub fn display_width(&self) -> u16 {
        self.display_width
    }

    /// Render the schema as lines
    pub fn render_lines(&self) -> Vec<String> {
        let mut lines = Vec::with_capacity(self.fields.len() + 3);

        // Calculate column widths
        let name_width = self
            .fields
            .iter()
            .map(|f| f.name.len())
            .max()
            .unwrap_or(5)
            .max(5);
        let type_width = self
            .fields
            .iter()
            .map(|f| f.type_name.len())
            .max()
            .unwrap_or(4)
            .max(4);

        // Header
        let header = format!(
            "{:<name_width$}  {:<type_width$}  Nullable",
            "Field",
            "Type",
            name_width = name_width,
            type_width = type_width
        );
        lines.push(header);

        // Separator
        let sep = format!(
            "{:-<name_width$}  {:-<type_width$}  --------",
            "",
            "",
            name_width = name_width,
            type_width = type_width
        );
        lines.push(sep);

        // Fields
        for field in &self.fields {
            let nullable_str = if field.nullable { "Yes" } else { "No" };
            let line = format!(
                "{:<name_width$}  {:<type_width$}  {}",
                truncate_string(&field.name, name_width),
                truncate_string(&field.type_name, type_width),
                nullable_str,
                name_width = name_width,
                type_width = type_width
            );
            lines.push(line);
        }

        lines
    }

    /// Get field info by index
    pub fn field(&self, index: usize) -> Option<(&str, &str, bool)> {
        self.fields
            .get(index)
            .map(|f| (f.name.as_str(), f.type_name.as_str(), f.nullable))
    }

    /// Get all field names
    pub fn field_names(&self) -> Vec<&str> {
        self.fields.iter().map(|f| f.name.as_str()).collect()
    }

    /// Get all type names
    pub fn type_names(&self) -> Vec<&str> {
        self.fields.iter().map(|f| f.type_name.as_str()).collect()
    }
}

/// Format Arrow data type as human-readable string
fn format_type_name(dt: &arrow::datatypes::DataType) -> String {
    use arrow::datatypes::DataType;

    match dt {
        DataType::Null => "Null".to_string(),
        DataType::Boolean => "Boolean".to_string(),
        DataType::Int8 => "Int8".to_string(),
        DataType::Int16 => "Int16".to_string(),
        DataType::Int32 => "Int32".to_string(),
        DataType::Int64 => "Int64".to_string(),
        DataType::UInt8 => "UInt8".to_string(),
        DataType::UInt16 => "UInt16".to_string(),
        DataType::UInt32 => "UInt32".to_string(),
        DataType::UInt64 => "UInt64".to_string(),
        DataType::Float16 => "Float16".to_string(),
        DataType::Float32 => "Float32".to_string(),
        DataType::Float64 => "Float64".to_string(),
        DataType::Utf8 => "Utf8".to_string(),
        DataType::LargeUtf8 => "LargeUtf8".to_string(),
        DataType::Binary => "Binary".to_string(),
        DataType::LargeBinary => "LargeBinary".to_string(),
        DataType::Date32 => "Date32".to_string(),
        DataType::Date64 => "Date64".to_string(),
        DataType::Timestamp(unit, tz) => {
            let unit_str = match unit {
                arrow::datatypes::TimeUnit::Second => "s",
                arrow::datatypes::TimeUnit::Millisecond => "ms",
                arrow::datatypes::TimeUnit::Microsecond => "us",
                arrow::datatypes::TimeUnit::Nanosecond => "ns",
            };
            match tz {
                Some(tz) => format!("Timestamp[{unit_str}, {tz}]"),
                None => format!("Timestamp[{unit_str}]"),
            }
        }
        DataType::List(inner) => format!("List<{}>", format_type_name(inner.data_type())),
        DataType::LargeList(inner) => {
            format!("LargeList<{}>", format_type_name(inner.data_type()))
        }
        DataType::Struct(fields) => {
            format!("Struct({})", fields.len())
        }
        DataType::Dictionary(key, value) => {
            format!(
                "Dict<{}, {}>",
                format_type_name(key),
                format_type_name(value)
            )
        }
        DataType::Map(field, _) => {
            format!("Map<{}>", format_type_name(field.data_type()))
        }
        _ => format!("{dt:?}"),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::datatypes::{DataType, Field, Schema};

    use super::*;

    fn create_test_adapter() -> DatasetAdapter {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("value", DataType::Int32, false),
            Field::new("score", DataType::Float32, true),
            Field::new("timestamp", DataType::Int64, false),
        ]));

        DatasetAdapter::from_batches(vec![], schema).unwrap()
    }

    #[test]
    fn f_inspector_new() {
        let adapter = create_test_adapter();
        let inspector = SchemaInspector::new(&adapter);
        assert_eq!(inspector.field_count(), 4);
    }

    #[test]
    fn f_inspector_field_count() {
        let adapter = create_test_adapter();
        let inspector = SchemaInspector::new(&adapter);
        assert_eq!(inspector.field_count(), 4);
    }

    #[test]
    fn f_inspector_is_empty() {
        let adapter = DatasetAdapter::empty();
        let inspector = SchemaInspector::new(&adapter);
        assert!(inspector.is_empty());
    }

    #[test]
    fn f_inspector_field_names() {
        let adapter = create_test_adapter();
        let inspector = SchemaInspector::new(&adapter);
        let names = inspector.field_names();
        assert_eq!(names, vec!["id", "value", "score", "timestamp"]);
    }

    #[test]
    fn f_inspector_type_names() {
        let adapter = create_test_adapter();
        let inspector = SchemaInspector::new(&adapter);
        let types = inspector.type_names();
        assert_eq!(types[0], "Utf8");
        assert_eq!(types[1], "Int32");
        assert_eq!(types[2], "Float32");
    }

    #[test]
    fn f_inspector_field_info() {
        let adapter = create_test_adapter();
        let inspector = SchemaInspector::new(&adapter);

        let (name, type_name, nullable) = inspector.field(0).unwrap();
        assert_eq!(name, "id");
        assert_eq!(type_name, "Utf8");
        assert!(!nullable);

        let (name, _, nullable) = inspector.field(2).unwrap();
        assert_eq!(name, "score");
        assert!(nullable);
    }

    #[test]
    fn f_inspector_field_out_of_bounds() {
        let adapter = create_test_adapter();
        let inspector = SchemaInspector::new(&adapter);
        assert!(inspector.field(100).is_none());
    }

    #[test]
    fn f_inspector_render_lines() {
        let adapter = create_test_adapter();
        let inspector = SchemaInspector::new(&adapter);
        let lines = inspector.render_lines();

        // Header + separator + 4 fields = 6 lines
        assert_eq!(lines.len(), 6);

        // Header contains "Field" and "Type"
        assert!(lines[0].contains("Field"));
        assert!(lines[0].contains("Type"));

        // Separator line
        assert!(lines[1].contains("---"));

        // Field lines
        assert!(lines[2].contains("id"));
        assert!(lines[2].contains("Utf8"));
    }

    #[test]
    fn f_inspector_render_nullable() {
        let adapter = create_test_adapter();
        let inspector = SchemaInspector::new(&adapter);
        let lines = inspector.render_lines();

        // score is nullable
        assert!(lines[4].contains("Yes"));
        // id is not nullable
        assert!(lines[2].contains("No"));
    }

    #[test]
    fn f_inspector_clone() {
        let adapter = create_test_adapter();
        let inspector = SchemaInspector::new(&adapter);
        let cloned = inspector.clone();
        assert_eq!(inspector.field_count(), cloned.field_count());
    }

    #[test]
    fn f_format_type_utf8() {
        assert_eq!(format_type_name(&DataType::Utf8), "Utf8");
    }

    #[test]
    fn f_format_type_int32() {
        assert_eq!(format_type_name(&DataType::Int32), "Int32");
    }

    #[test]
    fn f_format_type_float32() {
        assert_eq!(format_type_name(&DataType::Float32), "Float32");
    }

    #[test]
    fn f_format_type_list() {
        let list_type = DataType::List(Arc::new(Field::new("item", DataType::Int32, true)));
        let formatted = format_type_name(&list_type);
        assert!(formatted.contains("List"));
        assert!(formatted.contains("Int32"));
    }

    #[test]
    fn f_format_type_timestamp() {
        let ts_type = DataType::Timestamp(arrow::datatypes::TimeUnit::Millisecond, None);
        let formatted = format_type_name(&ts_type);
        assert!(formatted.contains("Timestamp"));
        assert!(formatted.contains("ms"));
    }

    #[test]
    fn f_inspector_display_width() {
        let adapter = create_test_adapter();
        let inspector = SchemaInspector::new(&adapter);
        // display_width should be reasonable (non-zero)
        assert!(inspector.display_width() > 0);
    }

    #[test]
    fn f_format_type_null() {
        assert_eq!(format_type_name(&DataType::Null), "Null");
    }

    #[test]
    fn f_format_type_boolean() {
        assert_eq!(format_type_name(&DataType::Boolean), "Boolean");
    }

    #[test]
    fn f_format_type_int8() {
        assert_eq!(format_type_name(&DataType::Int8), "Int8");
    }

    #[test]
    fn f_format_type_int16() {
        assert_eq!(format_type_name(&DataType::Int16), "Int16");
    }

    #[test]
    fn f_format_type_int64() {
        assert_eq!(format_type_name(&DataType::Int64), "Int64");
    }

    #[test]
    fn f_format_type_uint8() {
        assert_eq!(format_type_name(&DataType::UInt8), "UInt8");
    }

    #[test]
    fn f_format_type_uint16() {
        assert_eq!(format_type_name(&DataType::UInt16), "UInt16");
    }

    #[test]
    fn f_format_type_uint32() {
        assert_eq!(format_type_name(&DataType::UInt32), "UInt32");
    }

    #[test]
    fn f_format_type_uint64() {
        assert_eq!(format_type_name(&DataType::UInt64), "UInt64");
    }

    #[test]
    fn f_format_type_float16() {
        assert_eq!(format_type_name(&DataType::Float16), "Float16");
    }

    #[test]
    fn f_format_type_float64() {
        assert_eq!(format_type_name(&DataType::Float64), "Float64");
    }

    #[test]
    fn f_format_type_large_utf8() {
        assert_eq!(format_type_name(&DataType::LargeUtf8), "LargeUtf8");
    }

    #[test]
    fn f_format_type_binary() {
        assert_eq!(format_type_name(&DataType::Binary), "Binary");
    }

    #[test]
    fn f_format_type_large_binary() {
        assert_eq!(format_type_name(&DataType::LargeBinary), "LargeBinary");
    }

    #[test]
    fn f_format_type_date32() {
        assert_eq!(format_type_name(&DataType::Date32), "Date32");
    }

    #[test]
    fn f_format_type_date64() {
        assert_eq!(format_type_name(&DataType::Date64), "Date64");
    }

    #[test]
    fn f_format_type_timestamp_second() {
        let ts = DataType::Timestamp(arrow::datatypes::TimeUnit::Second, None);
        let formatted = format_type_name(&ts);
        assert!(formatted.contains("[s]"));
    }

    #[test]
    fn f_format_type_timestamp_microsecond() {
        let ts = DataType::Timestamp(arrow::datatypes::TimeUnit::Microsecond, None);
        let formatted = format_type_name(&ts);
        assert!(formatted.contains("[us]"));
    }

    #[test]
    fn f_format_type_timestamp_nanosecond() {
        let ts = DataType::Timestamp(arrow::datatypes::TimeUnit::Nanosecond, None);
        let formatted = format_type_name(&ts);
        assert!(formatted.contains("[ns]"));
    }

    #[test]
    fn f_format_type_timestamp_with_tz() {
        let ts = DataType::Timestamp(
            arrow::datatypes::TimeUnit::Millisecond,
            Some(Arc::from("UTC")),
        );
        let formatted = format_type_name(&ts);
        assert!(formatted.contains("UTC"));
    }

    #[test]
    fn f_format_type_large_list() {
        let list_type = DataType::LargeList(Arc::new(Field::new("item", DataType::Utf8, true)));
        let formatted = format_type_name(&list_type);
        assert!(formatted.contains("LargeList"));
        assert!(formatted.contains("Utf8"));
    }

    #[test]
    fn f_format_type_struct() {
        let fields = vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Utf8, true),
        ];
        let struct_type = DataType::Struct(fields.into());
        let formatted = format_type_name(&struct_type);
        assert!(formatted.contains("Struct"));
        assert!(formatted.contains('2'));
    }

    #[test]
    fn f_format_type_dictionary() {
        let dict_type = DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8));
        let formatted = format_type_name(&dict_type);
        assert!(formatted.contains("Dict"));
        assert!(formatted.contains("Int32"));
        assert!(formatted.contains("Utf8"));
    }

    #[test]
    fn f_format_type_map() {
        let map_type = DataType::Map(
            Arc::new(Field::new(
                "entries",
                DataType::Struct(
                    vec![
                        Field::new("key", DataType::Utf8, false),
                        Field::new("value", DataType::Int32, true),
                    ]
                    .into(),
                ),
                false,
            )),
            false,
        );
        let formatted = format_type_name(&map_type);
        assert!(formatted.contains("Map"));
    }

    #[test]
    fn f_format_type_fallback() {
        // Use a type that falls through to the debug fallback
        let duration_type = DataType::Duration(arrow::datatypes::TimeUnit::Second);
        let formatted = format_type_name(&duration_type);
        // Should contain "Duration" from debug output
        assert!(formatted.contains("Duration"));
    }
}
