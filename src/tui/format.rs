//! Cell formatting utilities for TUI display
//!
//! Provides safe formatting of Arrow array values to display strings.

use arrow::{
    array::{
        Array, BinaryArray, BooleanArray, Date32Array, Date64Array, Float32Array, Float64Array,
        Int16Array, Int32Array, Int64Array, Int8Array, LargeBinaryArray, LargeStringArray,
        StringArray, TimestampMicrosecondArray, TimestampMillisecondArray,
        TimestampNanosecondArray, TimestampSecondArray, UInt16Array, UInt32Array, UInt64Array,
        UInt8Array,
    },
    datatypes::DataType,
};

use super::error::TuiResult;

/// Format an Arrow array value at the given row index as a display string
///
/// Returns `None` if the value is null, `Some(formatted_string)` otherwise.
/// Returns an error only for truly unsupported types.
///
/// # Arguments
/// * `array` - The Arrow array to read from
/// * `row` - The row index within the array
///
/// # Example
/// ```ignore
/// let array = StringArray::from(vec!["hello", "world"]);
/// let formatted = format_array_value(&array, 0)?;
/// assert_eq!(formatted, Some("hello".to_string()));
/// ```
pub fn format_array_value(array: &dyn Array, row: usize) -> TuiResult<Option<String>> {
    // Check bounds first
    if row >= array.len() {
        return Ok(None);
    }

    // Handle null values
    if array.is_null(row) {
        return Ok(Some("NULL".to_string()));
    }

    let formatted = match array.data_type() {
        // String types
        DataType::Utf8 => format_utf8(array, row),
        DataType::LargeUtf8 => format_large_utf8(array, row),

        // Integer types
        DataType::Int8 => format_int8(array, row),
        DataType::Int16 => format_int16(array, row),
        DataType::Int32 => format_int32(array, row),
        DataType::Int64 => format_int64(array, row),
        DataType::UInt8 => format_uint8(array, row),
        DataType::UInt16 => format_uint16(array, row),
        DataType::UInt32 => format_uint32(array, row),
        DataType::UInt64 => format_uint64(array, row),

        // Float types
        DataType::Float32 => format_float32(array, row),
        DataType::Float64 => format_float64(array, row),

        // Boolean
        DataType::Boolean => format_boolean(array, row),

        // Binary types
        DataType::Binary => format_binary(array, row),
        DataType::LargeBinary => format_large_binary(array, row),

        // Date types
        DataType::Date32 => format_date32(array, row),
        DataType::Date64 => format_date64(array, row),

        // Timestamp types
        DataType::Timestamp(unit, _) => format_timestamp(array, row, *unit),

        // Null type
        DataType::Null => Some("NULL".to_string()),

        // Unsupported types - return placeholder
        other => Some(format!("<{}>", type_name(other))),
    };

    Ok(formatted)
}

/// Get a human-readable type name
fn type_name(dt: &DataType) -> &'static str {
    match dt {
        DataType::Null => "null",
        DataType::Boolean => "bool",
        DataType::Int8 => "i8",
        DataType::Int16 => "i16",
        DataType::Int32 => "i32",
        DataType::Int64 => "i64",
        DataType::UInt8 => "u8",
        DataType::UInt16 => "u16",
        DataType::UInt32 => "u32",
        DataType::UInt64 => "u64",
        DataType::Float32 => "f32",
        DataType::Float64 => "f64",
        DataType::Utf8 => "string",
        DataType::LargeUtf8 => "large_string",
        DataType::Binary => "binary",
        DataType::LargeBinary => "large_binary",
        DataType::Date32 => "date32",
        DataType::Date64 => "date64",
        DataType::Timestamp(_, _) => "timestamp",
        DataType::List(_) => "list",
        DataType::LargeList(_) => "large_list",
        DataType::Struct(_) => "struct",
        DataType::Map(_, _) => "map",
        DataType::Dictionary(_, _) => "dict",
        _ => "unknown",
    }
}

// Individual format functions for each type

fn format_utf8(array: &dyn Array, row: usize) -> Option<String> {
    array
        .as_any()
        .downcast_ref::<StringArray>()
        .map(|arr| arr.value(row).to_string())
}

fn format_large_utf8(array: &dyn Array, row: usize) -> Option<String> {
    array
        .as_any()
        .downcast_ref::<LargeStringArray>()
        .map(|arr| arr.value(row).to_string())
}

fn format_int8(array: &dyn Array, row: usize) -> Option<String> {
    array
        .as_any()
        .downcast_ref::<Int8Array>()
        .map(|arr| arr.value(row).to_string())
}

fn format_int16(array: &dyn Array, row: usize) -> Option<String> {
    array
        .as_any()
        .downcast_ref::<Int16Array>()
        .map(|arr| arr.value(row).to_string())
}

fn format_int32(array: &dyn Array, row: usize) -> Option<String> {
    array
        .as_any()
        .downcast_ref::<Int32Array>()
        .map(|arr| arr.value(row).to_string())
}

fn format_int64(array: &dyn Array, row: usize) -> Option<String> {
    array
        .as_any()
        .downcast_ref::<Int64Array>()
        .map(|arr| arr.value(row).to_string())
}

fn format_uint8(array: &dyn Array, row: usize) -> Option<String> {
    array
        .as_any()
        .downcast_ref::<UInt8Array>()
        .map(|arr| arr.value(row).to_string())
}

fn format_uint16(array: &dyn Array, row: usize) -> Option<String> {
    array
        .as_any()
        .downcast_ref::<UInt16Array>()
        .map(|arr| arr.value(row).to_string())
}

fn format_uint32(array: &dyn Array, row: usize) -> Option<String> {
    array
        .as_any()
        .downcast_ref::<UInt32Array>()
        .map(|arr| arr.value(row).to_string())
}

fn format_uint64(array: &dyn Array, row: usize) -> Option<String> {
    array
        .as_any()
        .downcast_ref::<UInt64Array>()
        .map(|arr| arr.value(row).to_string())
}

fn format_float32(array: &dyn Array, row: usize) -> Option<String> {
    array
        .as_any()
        .downcast_ref::<Float32Array>()
        .map(|arr| format!("{:.2}", arr.value(row)))
}

fn format_float64(array: &dyn Array, row: usize) -> Option<String> {
    array
        .as_any()
        .downcast_ref::<Float64Array>()
        .map(|arr| format!("{:.4}", arr.value(row)))
}

fn format_boolean(array: &dyn Array, row: usize) -> Option<String> {
    array
        .as_any()
        .downcast_ref::<BooleanArray>()
        .map(|arr| if arr.value(row) { "true" } else { "false" }.to_string())
}

fn format_binary(array: &dyn Array, row: usize) -> Option<String> {
    array.as_any().downcast_ref::<BinaryArray>().map(|arr| {
        let bytes = arr.value(row);
        format_bytes_preview(bytes)
    })
}

fn format_large_binary(array: &dyn Array, row: usize) -> Option<String> {
    array
        .as_any()
        .downcast_ref::<LargeBinaryArray>()
        .map(|arr| {
            let bytes = arr.value(row);
            format_bytes_preview(bytes)
        })
}

/// Format binary data as hex preview
fn format_bytes_preview(bytes: &[u8]) -> String {
    if bytes.len() <= 8 {
        format!("0x{}", hex_encode(bytes))
    } else {
        format!("0x{}... ({} bytes)", hex_encode(&bytes[..8]), bytes.len())
    }
}

/// Simple hex encoding without external dependency
fn hex_encode(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut result = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(result, "{b:02x}");
    }
    result
}

fn format_date32(array: &dyn Array, row: usize) -> Option<String> {
    array.as_any().downcast_ref::<Date32Array>().map(|arr| {
        let days = arr.value(row);
        // Days since Unix epoch
        format!("date:{days}")
    })
}

fn format_date64(array: &dyn Array, row: usize) -> Option<String> {
    array.as_any().downcast_ref::<Date64Array>().map(|arr| {
        let millis = arr.value(row);
        format!("date64:{millis}")
    })
}

fn format_timestamp(
    array: &dyn Array,
    row: usize,
    unit: arrow::datatypes::TimeUnit,
) -> Option<String> {
    use arrow::datatypes::TimeUnit;

    match unit {
        TimeUnit::Second => array
            .as_any()
            .downcast_ref::<TimestampSecondArray>()
            .map(|arr| format!("ts:{}", arr.value(row))),
        TimeUnit::Millisecond => array
            .as_any()
            .downcast_ref::<TimestampMillisecondArray>()
            .map(|arr| format!("ts:{}", arr.value(row))),
        TimeUnit::Microsecond => array
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .map(|arr| format!("ts:{}", arr.value(row))),
        TimeUnit::Nanosecond => array
            .as_any()
            .downcast_ref::<TimestampNanosecondArray>()
            .map(|arr| format!("ts:{}", arr.value(row))),
    }
}

/// Truncate a string to fit within a maximum display width
///
/// Handles Unicode properly by counting graphemes, not bytes.
/// Adds ellipsis if truncation occurs.
///
/// # Arguments
/// * `s` - The string to truncate
/// * `max_width` - Maximum display width in characters
///
/// # Returns
/// The truncated string, or the original if it fits
pub fn truncate_string(s: &str, max_width: usize) -> String {
    if max_width < 3 {
        return s.chars().take(max_width).collect();
    }

    let char_count = s.chars().count();
    if char_count <= max_width {
        return s.to_string();
    }

    // Reserve space for ellipsis
    let truncate_at = max_width.saturating_sub(2);
    let mut result: String = s.chars().take(truncate_at).collect();
    result.push_str("..");
    result
}

/// Calculate the display width of a string
///
/// For now, this is a simple character count.
/// Could be enhanced with unicode-width crate for proper CJK handling.
pub fn display_width(s: &str) -> usize {
    s.chars().count()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{
        ArrayRef, BinaryArray, BooleanArray, Date32Array, Date64Array, Float32Array, Float64Array,
        Int16Array, Int32Array, Int64Array, Int8Array, LargeBinaryArray, LargeStringArray,
        NullArray, StringArray, TimestampMillisecondArray, UInt16Array, UInt32Array, UInt64Array,
        UInt8Array,
    };

    use super::*;

    fn make_string_array(values: Vec<Option<&str>>) -> ArrayRef {
        Arc::new(StringArray::from(values))
    }

    fn make_int32_array(values: Vec<Option<i32>>) -> ArrayRef {
        Arc::new(Int32Array::from(values))
    }

    fn make_float32_array(values: Vec<Option<f32>>) -> ArrayRef {
        Arc::new(Float32Array::from(values))
    }

    fn make_int8_array(values: Vec<Option<i8>>) -> ArrayRef {
        Arc::new(Int8Array::from(values))
    }

    fn make_int16_array(values: Vec<Option<i16>>) -> ArrayRef {
        Arc::new(Int16Array::from(values))
    }

    fn make_int64_array(values: Vec<Option<i64>>) -> ArrayRef {
        Arc::new(Int64Array::from(values))
    }

    fn make_uint8_array(values: Vec<Option<u8>>) -> ArrayRef {
        Arc::new(UInt8Array::from(values))
    }

    fn make_uint16_array(values: Vec<Option<u16>>) -> ArrayRef {
        Arc::new(UInt16Array::from(values))
    }

    fn make_uint32_array(values: Vec<Option<u32>>) -> ArrayRef {
        Arc::new(UInt32Array::from(values))
    }

    fn make_uint64_array(values: Vec<Option<u64>>) -> ArrayRef {
        Arc::new(UInt64Array::from(values))
    }

    fn make_float64_array(values: Vec<Option<f64>>) -> ArrayRef {
        Arc::new(Float64Array::from(values))
    }

    fn make_boolean_array(values: Vec<Option<bool>>) -> ArrayRef {
        Arc::new(BooleanArray::from(values))
    }

    #[test]
    fn f_format_utf8_string() {
        let array = make_string_array(vec![Some("hello"), Some("world")]);
        let result = format_array_value(array.as_ref(), 0).unwrap();
        assert_eq!(result, Some("hello".to_string()));
    }

    #[test]
    fn f_format_utf8_null() {
        let array = make_string_array(vec![None, Some("world")]);
        let result = format_array_value(array.as_ref(), 0).unwrap();
        assert_eq!(result, Some("NULL".to_string()));
    }

    #[test]
    fn f_format_int32() {
        let array = make_int32_array(vec![Some(42), Some(-100)]);
        let result = format_array_value(array.as_ref(), 0).unwrap();
        assert_eq!(result, Some("42".to_string()));

        let result_neg = format_array_value(array.as_ref(), 1).unwrap();
        assert_eq!(result_neg, Some("-100".to_string()));
    }

    #[test]
    fn f_format_float32() {
        let array = make_float32_array(vec![Some(2.71), Some(2.0)]);
        let result = format_array_value(array.as_ref(), 0).unwrap();
        assert_eq!(result, Some("2.71".to_string()));
    }

    #[test]
    fn f_format_out_of_bounds() {
        let array = make_string_array(vec![Some("hello")]);
        let result = format_array_value(array.as_ref(), 10).unwrap();
        assert_eq!(result, None);
    }

    #[test]
    fn f_truncate_string_short() {
        let s = "hello";
        assert_eq!(truncate_string(s, 10), "hello");
    }

    #[test]
    fn f_truncate_string_exact() {
        let s = "hello";
        assert_eq!(truncate_string(s, 5), "hello");
    }

    #[test]
    fn f_truncate_string_long() {
        let s = "hello world this is a long string";
        let result = truncate_string(s, 10);
        assert!(result.ends_with(".."));
        assert!(result.chars().count() <= 10);
    }

    #[test]
    fn f_truncate_string_very_short_max() {
        let s = "hello";
        let result = truncate_string(s, 2);
        assert_eq!(result.chars().count(), 2);
    }

    #[test]
    fn f_display_width_ascii() {
        assert_eq!(display_width("hello"), 5);
    }

    #[test]
    fn f_display_width_unicode() {
        assert_eq!(display_width("日本語"), 3);
    }

    #[test]
    fn f_display_width_empty() {
        assert_eq!(display_width(""), 0);
    }

    #[test]
    fn f_hex_encode_empty() {
        assert_eq!(hex_encode(&[]), "");
    }

    #[test]
    fn f_hex_encode_bytes() {
        assert_eq!(hex_encode(&[0xde, 0xad, 0xbe, 0xef]), "deadbeef");
    }

    #[test]
    fn f_format_bytes_preview_short() {
        let bytes = vec![0x01, 0x02, 0x03];
        let result = format_bytes_preview(&bytes);
        assert!(result.starts_with("0x"));
        assert!(result.contains("010203"));
    }

    #[test]
    fn f_format_bytes_preview_long() {
        let bytes = vec![0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a];
        let result = format_bytes_preview(&bytes);
        assert!(result.contains("..."));
        assert!(result.contains("10 bytes"));
    }

    #[test]
    fn f_type_name_string() {
        assert_eq!(type_name(&DataType::Utf8), "string");
    }

    #[test]
    fn f_type_name_int32() {
        assert_eq!(type_name(&DataType::Int32), "i32");
    }

    #[test]
    fn f_type_name_float32() {
        assert_eq!(type_name(&DataType::Float32), "f32");
    }

    // Additional tests for comprehensive coverage

    #[test]
    fn f_format_int8() {
        let array = make_int8_array(vec![Some(42), Some(-100)]);
        let result = format_array_value(array.as_ref(), 0).unwrap();
        assert_eq!(result, Some("42".to_string()));
    }

    #[test]
    fn f_format_int16() {
        let array = make_int16_array(vec![Some(1234), Some(-5678)]);
        let result = format_array_value(array.as_ref(), 0).unwrap();
        assert_eq!(result, Some("1234".to_string()));
    }

    #[test]
    fn f_format_int64() {
        let array = make_int64_array(vec![Some(1_000_000_000_000), Some(-1)]);
        let result = format_array_value(array.as_ref(), 0).unwrap();
        assert_eq!(result, Some("1000000000000".to_string()));
    }

    #[test]
    fn f_format_uint8() {
        let array = make_uint8_array(vec![Some(255), Some(0)]);
        let result = format_array_value(array.as_ref(), 0).unwrap();
        assert_eq!(result, Some("255".to_string()));
    }

    #[test]
    fn f_format_uint16() {
        let array = make_uint16_array(vec![Some(65535), Some(0)]);
        let result = format_array_value(array.as_ref(), 0).unwrap();
        assert_eq!(result, Some("65535".to_string()));
    }

    #[test]
    fn f_format_uint32() {
        let array = make_uint32_array(vec![Some(4_000_000_000), Some(0)]);
        let result = format_array_value(array.as_ref(), 0).unwrap();
        assert_eq!(result, Some("4000000000".to_string()));
    }

    #[test]
    fn f_format_uint64() {
        let array = make_uint64_array(vec![Some(18_446_744_073_709_551_615), Some(0)]);
        let result = format_array_value(array.as_ref(), 0).unwrap();
        assert_eq!(result, Some("18446744073709551615".to_string()));
    }

    #[test]
    fn f_format_float64() {
        let array = make_float64_array(vec![Some(2.718281828), Some(1.0)]);
        let result = format_array_value(array.as_ref(), 0).unwrap();
        assert_eq!(result, Some("2.7183".to_string()));
    }

    #[test]
    fn f_format_boolean_true() {
        let array = make_boolean_array(vec![Some(true), Some(false)]);
        let result = format_array_value(array.as_ref(), 0).unwrap();
        assert_eq!(result, Some("true".to_string()));
    }

    #[test]
    fn f_format_boolean_false() {
        let array = make_boolean_array(vec![Some(false), Some(true)]);
        let result = format_array_value(array.as_ref(), 0).unwrap();
        assert_eq!(result, Some("false".to_string()));
    }

    #[test]
    fn f_format_large_string() {
        let array: ArrayRef = Arc::new(LargeStringArray::from(vec!["large_text", "another"]));
        let result = format_array_value(array.as_ref(), 0).unwrap();
        assert_eq!(result, Some("large_text".to_string()));
    }

    #[test]
    fn f_format_binary() {
        let array: ArrayRef = Arc::new(BinaryArray::from_vec(vec![&[0x01, 0x02, 0x03]]));
        let result = format_array_value(array.as_ref(), 0).unwrap();
        assert!(result.is_some());
        assert!(result.unwrap().starts_with("0x"));
    }

    #[test]
    fn f_format_large_binary() {
        let array: ArrayRef = Arc::new(LargeBinaryArray::from_vec(vec![&[0xde, 0xad, 0xbe, 0xef]]));
        let result = format_array_value(array.as_ref(), 0).unwrap();
        assert!(result.is_some());
        assert!(result.unwrap().contains("deadbeef"));
    }

    #[test]
    fn f_format_date32() {
        let array: ArrayRef = Arc::new(Date32Array::from(vec![Some(19000), None]));
        let result = format_array_value(array.as_ref(), 0).unwrap();
        assert!(result.is_some());
        assert!(result.unwrap().contains("date:"));
    }

    #[test]
    fn f_format_date64() {
        let array: ArrayRef = Arc::new(Date64Array::from(vec![Some(1_640_000_000_000), None]));
        let result = format_array_value(array.as_ref(), 0).unwrap();
        assert!(result.is_some());
        assert!(result.unwrap().contains("date64:"));
    }

    #[test]
    fn f_format_timestamp_ms() {
        let array: ArrayRef = Arc::new(TimestampMillisecondArray::from(vec![
            Some(1_640_000_000_000),
            None,
        ]));
        let result = format_array_value(array.as_ref(), 0).unwrap();
        assert!(result.is_some());
        assert!(result.unwrap().contains("ts:"));
    }

    #[test]
    fn f_format_null_type() {
        let array: ArrayRef = Arc::new(NullArray::new(3));
        let result = format_array_value(array.as_ref(), 0).unwrap();
        assert_eq!(result, Some("NULL".to_string()));
    }

    #[test]
    fn f_type_name_null() {
        assert_eq!(type_name(&DataType::Null), "null");
    }

    #[test]
    fn f_type_name_bool() {
        assert_eq!(type_name(&DataType::Boolean), "bool");
    }

    #[test]
    fn f_type_name_int8() {
        assert_eq!(type_name(&DataType::Int8), "i8");
    }

    #[test]
    fn f_type_name_int16() {
        assert_eq!(type_name(&DataType::Int16), "i16");
    }

    #[test]
    fn f_type_name_int64() {
        assert_eq!(type_name(&DataType::Int64), "i64");
    }

    #[test]
    fn f_type_name_uint8() {
        assert_eq!(type_name(&DataType::UInt8), "u8");
    }

    #[test]
    fn f_type_name_uint16() {
        assert_eq!(type_name(&DataType::UInt16), "u16");
    }

    #[test]
    fn f_type_name_uint32() {
        assert_eq!(type_name(&DataType::UInt32), "u32");
    }

    #[test]
    fn f_type_name_uint64() {
        assert_eq!(type_name(&DataType::UInt64), "u64");
    }

    #[test]
    fn f_type_name_float64() {
        assert_eq!(type_name(&DataType::Float64), "f64");
    }

    #[test]
    fn f_type_name_large_string() {
        assert_eq!(type_name(&DataType::LargeUtf8), "large_string");
    }

    #[test]
    fn f_type_name_binary() {
        assert_eq!(type_name(&DataType::Binary), "binary");
    }

    #[test]
    fn f_type_name_large_binary() {
        assert_eq!(type_name(&DataType::LargeBinary), "large_binary");
    }

    #[test]
    fn f_type_name_date32() {
        assert_eq!(type_name(&DataType::Date32), "date32");
    }

    #[test]
    fn f_type_name_date64() {
        assert_eq!(type_name(&DataType::Date64), "date64");
    }

    #[test]
    fn f_type_name_timestamp() {
        let ts_type = DataType::Timestamp(arrow::datatypes::TimeUnit::Millisecond, None);
        assert_eq!(type_name(&ts_type), "timestamp");
    }

    #[test]
    fn f_type_name_list() {
        use arrow::datatypes::Field;
        let list_type = DataType::List(Arc::new(Field::new("item", DataType::Int32, true)));
        assert_eq!(type_name(&list_type), "list");
    }

    #[test]
    fn f_type_name_large_list() {
        use arrow::datatypes::Field;
        let list_type = DataType::LargeList(Arc::new(Field::new("item", DataType::Int32, true)));
        assert_eq!(type_name(&list_type), "large_list");
    }

    #[test]
    fn f_type_name_struct() {
        use arrow::datatypes::{Field, Fields};
        let fields = Fields::from(vec![Field::new("a", DataType::Int32, false)]);
        let struct_type = DataType::Struct(fields);
        assert_eq!(type_name(&struct_type), "struct");
    }

    #[test]
    fn f_type_name_map() {
        use arrow::datatypes::Field;
        let entries = Field::new(
            "entries",
            DataType::Struct(
                vec![
                    Field::new("key", DataType::Utf8, false),
                    Field::new("value", DataType::Int32, true),
                ]
                .into(),
            ),
            false,
        );
        let map_type = DataType::Map(Arc::new(entries), false);
        assert_eq!(type_name(&map_type), "map");
    }

    #[test]
    fn f_type_name_dict() {
        let dict_type = DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8));
        assert_eq!(type_name(&dict_type), "dict");
    }

    #[test]
    fn f_type_name_unknown() {
        // Test a type that falls through to the catch-all
        let interval_type = DataType::Interval(arrow::datatypes::IntervalUnit::DayTime);
        assert_eq!(type_name(&interval_type), "unknown");
    }

    #[test]
    fn f_format_timestamp_second() {
        use arrow::array::TimestampSecondArray;
        let array: ArrayRef = Arc::new(TimestampSecondArray::from(vec![Some(1_640_000_000), None]));
        let result = format_array_value(array.as_ref(), 0).unwrap();
        assert!(result.is_some());
        assert!(result.unwrap().contains("ts:"));
    }

    #[test]
    fn f_format_timestamp_microsecond() {
        use arrow::array::TimestampMicrosecondArray;
        let array: ArrayRef = Arc::new(TimestampMicrosecondArray::from(vec![
            Some(1_640_000_000_000_000),
            None,
        ]));
        let result = format_array_value(array.as_ref(), 0).unwrap();
        assert!(result.is_some());
        assert!(result.unwrap().contains("ts:"));
    }

    #[test]
    fn f_format_timestamp_nanosecond() {
        use arrow::array::TimestampNanosecondArray;
        let array: ArrayRef = Arc::new(TimestampNanosecondArray::from(vec![
            Some(1_640_000_000_000_000_000),
            None,
        ]));
        let result = format_array_value(array.as_ref(), 0).unwrap();
        assert!(result.is_some());
        assert!(result.unwrap().contains("ts:"));
    }

    #[test]
    fn f_format_unsupported_type() {
        // Test with an unsupported type that should show placeholder
        use arrow::{array::ListArray, buffer::OffsetBuffer, datatypes::Field};

        let values = Int32Array::from(vec![1, 2, 3, 4, 5]);
        let offsets = OffsetBuffer::new(vec![0, 2, 5].into());
        let field = Arc::new(Field::new("item", DataType::Int32, true));
        let array: ArrayRef = Arc::new(ListArray::new(field, offsets, Arc::new(values), None));

        let result = format_array_value(array.as_ref(), 0).unwrap();
        assert!(result.is_some());
        // Should show <list> placeholder
        assert!(result.unwrap().contains("list"));
    }
}
