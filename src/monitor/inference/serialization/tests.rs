//! Tests for trace serialization.

use super::*;
use crate::monitor::inference::path::LinearPath;
use crate::monitor::inference::DecisionTrace;
use std::error::Error;

fn make_test_trace() -> DecisionTrace<LinearPath> {
    let path = LinearPath::new(vec![0.5, -0.3, 0.2], 0.1, 0.5, 0.87).with_probability(0.87);
    DecisionTrace::new(1_000_000, 42, 0xdeadbeef, path, 0.87, 500)
}

#[test]
fn test_path_type_from_u8() {
    assert_eq!(PathType::from(0), PathType::Linear);
    assert_eq!(PathType::from(1), PathType::Tree);
    assert_eq!(PathType::from(4), PathType::Neural);
    assert_eq!(PathType::from(100), PathType::Custom);
}

#[test]
fn test_serialize_binary() {
    let serializer = TraceSerializer::new(TraceFormat::Binary);
    let trace = make_test_trace();

    let bytes = serializer.serialize(&trace, PathType::Linear).expect("Serialization failed");

    // Check header
    assert_eq!(&bytes[0..4], &APRT_MAGIC);
    assert_eq!(bytes[4], APRT_VERSION);
    assert_eq!(bytes[5], PathType::Linear as u8);
}

#[test]
fn test_binary_roundtrip() {
    let serializer = TraceSerializer::new(TraceFormat::Binary);
    let trace = make_test_trace();

    let bytes = serializer.serialize(&trace, PathType::Linear).expect("Serialization failed");

    let restored: DecisionTrace<LinearPath> =
        serializer.deserialize(&bytes).expect("Deserialization failed");

    assert_eq!(trace.timestamp_ns, restored.timestamp_ns);
    assert_eq!(trace.sequence, restored.sequence);
    assert_eq!(trace.input_hash, restored.input_hash);
    assert!((trace.output - restored.output).abs() < 1e-6);
}

#[test]
fn test_serialize_json() {
    let serializer = TraceSerializer::new(TraceFormat::Json);
    let trace = make_test_trace();

    let bytes = serializer.serialize(&trace, PathType::Linear).expect("Serialization failed");

    let json = String::from_utf8(bytes).expect("Invalid UTF-8");
    assert!(json.contains("timestamp_ns"));
    assert!(json.contains("sequence"));
    assert!(json.contains("path"));
}

#[test]
fn test_json_roundtrip() {
    let serializer = TraceSerializer::new(TraceFormat::Json);
    let trace = make_test_trace();

    let bytes = serializer.serialize(&trace, PathType::Linear).expect("Serialization failed");

    let restored: DecisionTrace<LinearPath> =
        serializer.deserialize(&bytes).expect("Deserialization failed");

    assert_eq!(trace.sequence, restored.sequence);
}

#[test]
fn test_serialize_json_lines() {
    let serializer = TraceSerializer::new(TraceFormat::JsonLines);
    let trace = make_test_trace();

    let bytes = serializer.serialize(&trace, PathType::Linear).expect("Serialization failed");

    // Should end with newline
    assert_eq!(bytes.last(), Some(&b'\n'));

    // Should be single line
    let json = String::from_utf8(bytes).expect("Invalid UTF-8");
    assert_eq!(json.lines().count(), 1);
}

#[test]
fn test_deserialize_invalid_magic() {
    let serializer = TraceSerializer::new(TraceFormat::Binary);
    let bytes = vec![0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00];

    let result: Result<DecisionTrace<LinearPath>, _> = serializer.deserialize(&bytes);
    assert!(matches!(result, Err(SerializationError::InvalidFormat(_))));
}

#[test]
fn test_deserialize_version_mismatch() {
    let serializer = TraceSerializer::new(TraceFormat::Binary);
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&APRT_MAGIC);
    bytes.push(99); // Invalid version
    bytes.extend_from_slice(&[0, 0, 0]);

    let result: Result<DecisionTrace<LinearPath>, _> = serializer.deserialize(&bytes);
    assert!(matches!(result, Err(SerializationError::VersionMismatch { .. })));
}

#[test]
fn test_deserialize_insufficient_data() {
    let serializer = TraceSerializer::new(TraceFormat::Binary);
    let bytes = vec![0x41, 0x50]; // Only 2 bytes

    let result: Result<DecisionTrace<LinearPath>, _> = serializer.deserialize(&bytes);
    assert!(matches!(result, Err(SerializationError::InvalidFormat(_))));
}

#[test]
fn test_serializer_default() {
    let serializer = TraceSerializer::default();
    assert_eq!(serializer.format(), TraceFormat::Binary);
}

#[test]
fn test_error_display() {
    let err = SerializationError::InvalidFormat("test".to_string());
    assert!(err.to_string().contains("Invalid format"));

    let err = SerializationError::VersionMismatch { expected: 1, actual: 2 };
    assert!(err.to_string().contains("Version mismatch"));
}

// Additional coverage tests

#[test]
fn test_path_type_all_variants() {
    assert_eq!(PathType::from(2), PathType::Forest);
    assert_eq!(PathType::from(3), PathType::KNN);
    assert_eq!(PathType::from(255), PathType::Custom);
}

#[test]
fn test_trace_format_serde() {
    let format = TraceFormat::Json;
    let json = serde_json::to_string(&format).expect("JSON serialization should succeed");
    let restored: TraceFormat =
        serde_json::from_str(&json).expect("JSON deserialization should succeed");
    assert_eq!(format, restored);

    let format = TraceFormat::JsonLines;
    let json = serde_json::to_string(&format).expect("JSON serialization should succeed");
    let restored: TraceFormat =
        serde_json::from_str(&json).expect("JSON deserialization should succeed");
    assert_eq!(format, restored);

    let format = TraceFormat::Binary;
    let json = serde_json::to_string(&format).expect("JSON serialization should succeed");
    let restored: TraceFormat =
        serde_json::from_str(&json).expect("JSON deserialization should succeed");
    assert_eq!(format, restored);
}

#[test]
fn test_serialization_error_json_display() {
    let invalid_json = "not valid json";
    let err: Result<serde_json::Value, _> = serde_json::from_str(invalid_json);
    if let Err(json_err) = err {
        let serialization_err = SerializationError::Json(json_err);
        let display = serialization_err.to_string();
        assert!(display.contains("JSON error"));
    }
}

#[test]
fn test_serialization_error_io_display() {
    let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
    let serialization_err = SerializationError::Io(io_err);
    let display = serialization_err.to_string();
    assert!(display.contains("IO error"));
}

#[test]
fn test_serialization_error_source_json() {
    let invalid_json = "not valid json";
    let err: Result<serde_json::Value, _> = serde_json::from_str(invalid_json);
    if let Err(json_err) = err {
        let serialization_err = SerializationError::Json(json_err);
        assert!(serialization_err.source().is_some());
    }
}

#[test]
fn test_serialization_error_source_io() {
    let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "test");
    let serialization_err = SerializationError::Io(io_err);
    assert!(serialization_err.source().is_some());
}

#[test]
fn test_serialization_error_source_none() {
    let err = SerializationError::InvalidFormat("test".to_string());
    assert!(err.source().is_none());

    let err = SerializationError::VersionMismatch { expected: 1, actual: 2 };
    assert!(err.source().is_none());
}

#[test]
fn test_serialization_error_from_json() {
    let invalid_json = "not valid json";
    let json_err = serde_json::from_str::<serde_json::Value>(invalid_json).unwrap_err();
    let serialization_err: SerializationError = json_err.into();
    assert!(matches!(serialization_err, SerializationError::Json(_)));
}

#[test]
fn test_serialization_error_from_io() {
    let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "test");
    let serialization_err: SerializationError = io_err.into();
    assert!(matches!(serialization_err, SerializationError::Io(_)));
}

#[test]
fn test_deserialize_binary_invalid_trace_data() {
    let serializer = TraceSerializer::new(TraceFormat::Binary);
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&APRT_MAGIC);
    bytes.push(APRT_VERSION);
    bytes.extend_from_slice(&[0, 0, 0]); // path type + reserved
    bytes.extend_from_slice(&[0xFF, 0xFF, 0xFF]); // Invalid trace data

    let result: Result<DecisionTrace<LinearPath>, _> = serializer.deserialize(&bytes);
    assert!(matches!(result, Err(SerializationError::InvalidFormat(_))));
}

#[test]
fn test_json_lines_deserialize() {
    let serializer = TraceSerializer::new(TraceFormat::JsonLines);
    let trace = make_test_trace();

    let bytes = serializer.serialize(&trace, PathType::Linear).expect("Serialization failed");

    let restored: DecisionTrace<LinearPath> =
        serializer.deserialize(&bytes).expect("Deserialization failed");

    assert_eq!(trace.sequence, restored.sequence);
}

#[test]
fn test_trace_format_debug() {
    let format = TraceFormat::Binary;
    let debug = format!("{format:?}");
    assert!(debug.contains("Binary"));
}

#[test]
fn test_path_type_debug() {
    let pt = PathType::Neural;
    let debug = format!("{pt:?}");
    assert!(debug.contains("Neural"));
}

#[test]
fn test_serialization_error_debug() {
    let err = SerializationError::InvalidFormat("test".to_string());
    let debug = format!("{err:?}");
    assert!(debug.contains("InvalidFormat"));
}
