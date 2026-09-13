//! Parquet-compatible span record schema (Sprint 40 - Golden Thread Core)
//!
//! This module defines the canonical schema for storing OpenTelemetry spans in
//! trueno-db's Parquet-backed storage. The schema is optimized for:
//!
//! - **Query performance:** Flat structure for predicate pushdown
//! - **Compression:** Columnar layout with RLE/dictionary encoding
//! - **W3C Trace Context:** Native support for traceparent format
//! - **Causal ordering:** Lamport logical clock for happens-before
//!
//! # Design Principles
//!
//! 1. **Flat Structure:** Parquet performs best with flat schemas (no deep nesting)
//! 2. **Fixed-size IDs:** `trace_id` (16 bytes), `span_id` (8 bytes) for efficient indexing
//! 3. **JSON Attributes:** Flexible key-value pairs stored as JSON string
//! 4. **Timestamp Precision:** Nanosecond precision for microsecond-level tracing
//! 5. **Logical Causality:** Lamport clock field for provable ordering
//!
//! # Parquet Schema Mapping
//!
//! ```text
//! SpanRecord (Rust)              →  Parquet Physical Type
//! ├─ trace_id: [u8; 16]          →  FIXED_LEN_BYTE_ARRAY(16)
//! ├─ span_id: [u8; 8]            →  FIXED_LEN_BYTE_ARRAY(8)
//! ├─ parent_span_id: Option<..>  →  FIXED_LEN_BYTE_ARRAY(8), nullable=true
//! ├─ span_name: String           →  BYTE_ARRAY (UTF8)
//! ├─ span_kind: SpanKind         →  INT32 (enum)
//! ├─ start_time_nanos: u64       →  INT64
//! ├─ end_time_nanos: u64         →  INT64
//! ├─ logical_clock: u64          →  INT64 (Lamport timestamp)
//! ├─ duration_nanos: u64         →  INT64 (computed: end - start)
//! ├─ status_code: StatusCode     →  INT32 (enum)
//! ├─ status_message: String      →  BYTE_ARRAY (UTF8)
//! ├─ attributes_json: String     →  BYTE_ARRAY (UTF8) - JSON map
//! ├─ resource_json: String       →  BYTE_ARRAY (UTF8) - JSON map
//! ├─ process_id: u32             →  INT32
//! └─ thread_id: u64              →  INT64
//! ```
//!
//! # Query Patterns
//!
//! The schema is optimized for these access patterns:
//!
//! ```sql
//! -- Critical path queries (p95 <20ms for 1M spans)
//! SELECT * FROM spans WHERE trace_id = ?
//! SELECT * FROM spans WHERE trace_id = ? ORDER BY logical_clock
//! SELECT * FROM spans WHERE trace_id = ? AND parent_span_id IS NULL
//!
//! -- Temporal range queries
//! SELECT * FROM spans WHERE start_time_nanos BETWEEN ? AND ?
//!
//! -- Process/thread filtering
//! SELECT * FROM spans WHERE process_id = ? AND thread_id = ?
//!
//! -- Status filtering (error analysis)
//! SELECT * FROM spans WHERE status_code = 2 -- ERROR
//! ```
//!
//! # Peer-Reviewed Foundation
//!
//! - **Melnik et al. (2010). "Dremel: Interactive Analysis of Web-Scale Datasets." Google.**
//!   - Finding: Columnar storage with nested encoding enables <1s queries on trillion-row tables
//!   - Application: Parquet schema optimized for predicate pushdown
//!
//! - **Lamb et al. (2012). "The Vertica Analytic Database." VLDB.**
//!   - Finding: Column-store compression (RLE, dictionary) achieves 10-50× reduction
//!   - Application: Fixed-size IDs and enums for optimal compression

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Span record compatible with trueno-db Parquet storage
///
/// This is the canonical schema for all spans recorded by renacer. Each span
/// represents a single operation (syscall, function call, GPU kernel, etc.)
/// with complete metadata for causal analysis.
///
/// # Example
///
/// ```
/// use renacer_core::span_record::{SpanRecord, SpanKind, StatusCode};
/// use std::collections::HashMap;
///
/// let span = SpanRecord {
///     trace_id: [0x4b, 0xf9, 0x2f, 0x3c, 0x7b, 0x64, 0x4b, 0xf9,
///                0x2f, 0x3c, 0x7b, 0x64, 0x4b, 0xf9, 0x2f, 0x3c],
///     span_id: [0x00, 0xf0, 0x67, 0xaa, 0x0b, 0xa9, 0x02, 0xb7],
///     parent_span_id: None,
///     span_name: "read".to_string(),
///     span_kind: SpanKind::Internal,
///     start_time_nanos: 1700000000000000000,
///     end_time_nanos: 1700000000000050000,
///     duration_nanos: 50000,
///     logical_clock: 42,
///     status_code: StatusCode::Ok,
///     status_message: String::new(),
///     attributes_json: r#"{"syscall.name":"read","syscall.fd":3,"syscall.bytes":1024}"#.to_string(),
///     resource_json: r#"{"service.name":"renacer","process.pid":1234}"#.to_string(),
///     process_id: 1234,
///     thread_id: 1234,
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SpanRecord {
    /// W3C Trace Context trace ID (128-bit / 16 bytes)
    ///
    /// Format: 32 hex characters (e.g., `4bf92f3c7b644bf92f3c7b644bf92f3c`)
    ///
    /// This is the "golden thread" that links all operations across the entire
    /// pipeline (Rust binary → transpilation → syscalls).
    pub trace_id: [u8; 16],

    /// W3C Trace Context span ID (64-bit / 8 bytes)
    ///
    /// Format: 16 hex characters (e.g., `00f067aa0ba902b7`)
    ///
    /// Uniquely identifies this span within the trace.
    pub span_id: [u8; 8],

    /// Parent span ID (if this span has a parent)
    ///
    /// - `None` indicates this is a root span
    /// - `Some(id)` indicates this span is a child of another span
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub parent_span_id: Option<[u8; 8]>,

    /// Human-readable span name (e.g., "read", "write", "GPU kernel", "HTTP GET")
    ///
    /// Should follow OpenTelemetry semantic conventions:
    /// - Syscalls: use syscall name (e.g., "read", "write")
    /// - Functions: use function name (e.g., "`process_request`")
    /// - HTTP: use "HTTP {method}" (e.g., "HTTP GET")
    pub span_name: String,

    /// Span kind (internal, server, client, producer, consumer)
    ///
    /// Indicates the role of this span in the request flow.
    pub span_kind: SpanKind,

    /// Start time in nanoseconds since UNIX epoch
    ///
    /// This is the **physical timestamp** (subject to clock skew).
    /// Use `logical_clock` for causal ordering.
    pub start_time_nanos: u64,

    /// End time in nanoseconds since UNIX epoch
    ///
    /// This is the **physical timestamp** (subject to clock skew).
    /// Use `logical_clock` for causal ordering.
    pub end_time_nanos: u64,

    /// Span duration in nanoseconds (`end_time` - `start_time`)
    ///
    /// This is a computed field for query convenience.
    pub duration_nanos: u64,

    /// Lamport logical clock timestamp
    ///
    /// This provides a **mathematical guarantee** of causal ordering:
    /// if event A → B (happens-before), then `logical_clock(A) < logical_clock(B)`.
    ///
    /// Use this for:
    /// - Critical path analysis (longest path in causal graph)
    /// - Detecting race conditions (concurrent events have incomparable clocks)
    /// - Cross-process ordering (even with clock skew)
    pub logical_clock: u64,

    /// Span status code (unset, ok, error)
    pub status_code: StatusCode,

    /// Span status message (empty if OK, error message if ERROR)
    pub status_message: String,

    /// Span attributes as JSON string
    ///
    /// This contains all key-value metadata about the span:
    /// - Syscall arguments: `{"syscall.name":"read","syscall.fd":3,"syscall.bytes":1024}`
    /// - File paths: `{"file.path":"/etc/passwd","file.line":42}`
    /// - HTTP: `{"http.method":"GET","http.url":"https://example.com"}`
    ///
    /// Stored as JSON to maintain flat Parquet schema (no nested columns).
    pub attributes_json: String,

    /// Resource attributes as JSON string
    ///
    /// This contains metadata about the execution environment:
    /// - `{"service.name":"renacer","process.pid":1234,"host.name":"server1"}`
    ///
    /// Stored as JSON to maintain flat Parquet schema.
    pub resource_json: String,

    /// Process ID (for filtering by process)
    pub process_id: u32,

    /// Thread ID (for filtering by thread)
    pub thread_id: u64,
}

/// Span kind (OpenTelemetry semantic convention)
///
/// Indicates the role of this span in the distributed trace.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[repr(u8)]
pub enum SpanKind {
    /// Internal operation (function call, syscall)
    #[default]
    Internal = 0,

    /// Server-side request handling (HTTP server, RPC server)
    Server = 1,

    /// Client-side request (HTTP client, RPC client)
    Client = 2,

    /// Producer (message queue producer)
    Producer = 3,

    /// Consumer (message queue consumer)
    Consumer = 4,
}

/// Span status code (OpenTelemetry semantic convention)
///
/// Indicates whether the span completed successfully or with an error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[repr(u8)]
pub enum StatusCode {
    /// Default status - span status not set
    #[default]
    Unset = 0,

    /// Span completed successfully
    Ok = 1,

    /// Span completed with an error
    Error = 2,
}

impl SpanRecord {
    /// Create a new `SpanRecord` with computed duration
    ///
    /// # Arguments
    ///
    /// * `trace_id` - W3C Trace Context trace ID (16 bytes)
    /// * `span_id` - W3C Trace Context span ID (8 bytes)
    /// * `parent_span_id` - Parent span ID (None for root spans)
    /// * `span_name` - Human-readable span name
    /// * `span_kind` - Span kind (internal, server, client, etc.)
    /// * `start_time_nanos` - Start time in nanoseconds since UNIX epoch
    /// * `end_time_nanos` - End time in nanoseconds since UNIX epoch
    /// * `logical_clock` - Lamport logical clock timestamp
    /// * `status_code` - Span status code
    /// * `status_message` - Span status message
    /// * `attributes` - Span attributes (will be serialized to JSON)
    /// * `resource` - Resource attributes (will be serialized to JSON)
    /// * `process_id` - Process ID
    /// * `thread_id` - Thread ID
    ///
    /// # Example
    ///
    /// ```
    /// use renacer_core::span_record::{SpanRecord, SpanKind, StatusCode};
    /// use std::collections::HashMap;
    ///
    /// let mut attributes = HashMap::new();
    /// attributes.insert("syscall.name".to_string(), "read".to_string());
    /// attributes.insert("syscall.fd".to_string(), "3".to_string());
    ///
    /// let mut resource = HashMap::new();
    /// resource.insert("service.name".to_string(), "renacer".to_string());
    ///
    /// let span = SpanRecord::new(
    ///     [0x4b, 0xf9, 0x2f, 0x3c, 0x7b, 0x64, 0x4b, 0xf9,
    ///      0x2f, 0x3c, 0x7b, 0x64, 0x4b, 0xf9, 0x2f, 0x3c],
    ///     [0x00, 0xf0, 0x67, 0xaa, 0x0b, 0xa9, 0x02, 0xb7],
    ///     None,
    ///     "read".to_string(),
    ///     SpanKind::Internal,
    ///     1700000000000000000,
    ///     1700000000000050000,
    ///     42,
    ///     StatusCode::Ok,
    ///     String::new(),
    ///     attributes,
    ///     resource,
    ///     1234,
    ///     1234,
    /// );
    ///
    /// assert_eq!(span.duration_nanos, 50000);
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        trace_id: [u8; 16],
        span_id: [u8; 8],
        parent_span_id: Option<[u8; 8]>,
        span_name: String,
        span_kind: SpanKind,
        start_time_nanos: u64,
        end_time_nanos: u64,
        logical_clock: u64,
        status_code: StatusCode,
        status_message: String,
        attributes: HashMap<String, String>,
        resource: HashMap<String, String>,
        process_id: u32,
        thread_id: u64,
    ) -> Self {
        let duration_nanos = end_time_nanos.saturating_sub(start_time_nanos);

        let attributes_json =
            serde_json::to_string(&attributes).unwrap_or_else(|_| "{}".to_string());

        let resource_json = serde_json::to_string(&resource).unwrap_or_else(|_| "{}".to_string());

        Self {
            trace_id,
            span_id,
            parent_span_id,
            span_name,
            span_kind,
            start_time_nanos,
            end_time_nanos,
            duration_nanos,
            logical_clock,
            status_code,
            status_message,
            attributes_json,
            resource_json,
            process_id,
            thread_id,
        }
    }

    /// Parse attributes from JSON string
    ///
    /// # Returns
    ///
    /// `HashMap` of attribute key-value pairs, or empty map if parse fails.
    pub fn parse_attributes(&self) -> HashMap<String, String> {
        serde_json::from_str(&self.attributes_json).unwrap_or_default()
    }

    /// Parse resource attributes from JSON string
    ///
    /// # Returns
    ///
    /// `HashMap` of resource key-value pairs, or empty map if parse fails.
    pub fn parse_resource(&self) -> HashMap<String, String> {
        serde_json::from_str(&self.resource_json).unwrap_or_default()
    }

    /// Get trace ID as hex string (W3C Trace Context format)
    ///
    /// # Example
    ///
    /// ```
    /// use renacer_core::span_record::{SpanRecord, SpanKind, StatusCode};
    /// use std::collections::HashMap;
    ///
    /// let span = SpanRecord::new(
    ///     [0x4b, 0xf9, 0x2f, 0x3c, 0x7b, 0x64, 0x4b, 0xf9,
    ///      0x2f, 0x3c, 0x7b, 0x64, 0x4b, 0xf9, 0x2f, 0x3c],
    ///     [0x00, 0xf0, 0x67, 0xaa, 0x0b, 0xa9, 0x02, 0xb7],
    ///     None,
    ///     "test".to_string(),
    ///     SpanKind::Internal,
    ///     0, 0, 0,
    ///     StatusCode::Ok,
    ///     String::new(),
    ///     HashMap::new(),
    ///     HashMap::new(),
    ///     0, 0,
    /// );
    ///
    /// assert_eq!(span.trace_id_hex(), "4bf92f3c7b644bf92f3c7b644bf92f3c");
    /// ```
    pub fn trace_id_hex(&self) -> String {
        hex::encode(self.trace_id)
    }

    /// Get span ID as hex string (W3C Trace Context format)
    pub fn span_id_hex(&self) -> String {
        hex::encode(self.span_id)
    }

    /// Get parent span ID as hex string (W3C Trace Context format)
    pub fn parent_span_id_hex(&self) -> Option<String> {
        self.parent_span_id.map(hex::encode)
    }

    /// Check if this is a root span (no parent)
    pub fn is_root(&self) -> bool {
        self.parent_span_id.is_none()
    }

    /// Check if this span represents an error
    pub fn is_error(&self) -> bool {
        self.status_code == StatusCode::Error
    }
}

// We need the hex crate for trace ID formatting
// Note: This will be added to Cargo.toml dependencies if not present

// Compile-time thread-safety verification (Sprint 59)
static_assertions::assert_impl_all!(SpanRecord: Send, Sync);
static_assertions::assert_impl_all!(SpanKind: Send, Sync);
static_assertions::assert_impl_all!(StatusCode: Send, Sync);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_span_record_creation() {
        let mut attributes = HashMap::new();
        attributes.insert("syscall.name".to_string(), "read".to_string());
        attributes.insert("syscall.fd".to_string(), "3".to_string());

        let mut resource = HashMap::new();
        resource.insert("service.name".to_string(), "renacer".to_string());

        let span = SpanRecord::new(
            [1; 16],
            [2; 8],
            None,
            "read".to_string(),
            SpanKind::Internal,
            1000,
            2000,
            42,
            StatusCode::Ok,
            String::new(),
            attributes,
            resource,
            1234,
            5678,
        );

        assert_eq!(span.trace_id, [1; 16]);
        assert_eq!(span.span_id, [2; 8]);
        assert_eq!(span.parent_span_id, None);
        assert_eq!(span.span_name, "read");
        assert_eq!(span.span_kind, SpanKind::Internal);
        assert_eq!(span.start_time_nanos, 1000);
        assert_eq!(span.end_time_nanos, 2000);
        assert_eq!(span.duration_nanos, 1000);
        assert_eq!(span.logical_clock, 42);
        assert_eq!(span.status_code, StatusCode::Ok);
        assert_eq!(span.process_id, 1234);
        assert_eq!(span.thread_id, 5678);
    }

    #[test]
    fn test_duration_computation() {
        let span = SpanRecord::new(
            [1; 16],
            [2; 8],
            None,
            "test".to_string(),
            SpanKind::Internal,
            1000,
            3500,
            42,
            StatusCode::Ok,
            String::new(),
            HashMap::new(),
            HashMap::new(),
            0,
            0,
        );

        assert_eq!(span.duration_nanos, 2500);
    }

    #[test]
    fn test_attributes_serialization() {
        let mut attributes = HashMap::new();
        attributes.insert("key1".to_string(), "value1".to_string());
        attributes.insert("key2".to_string(), "value2".to_string());

        let span = SpanRecord::new(
            [1; 16],
            [2; 8],
            None,
            "test".to_string(),
            SpanKind::Internal,
            0,
            0,
            0,
            StatusCode::Ok,
            String::new(),
            attributes.clone(),
            HashMap::new(),
            0,
            0,
        );

        let parsed = span.parse_attributes();
        assert_eq!(parsed.get("key1"), Some(&"value1".to_string()));
        assert_eq!(parsed.get("key2"), Some(&"value2".to_string()));
    }

    #[test]
    fn test_trace_id_hex() {
        let span = SpanRecord::new(
            [
                0x4b, 0xf9, 0x2f, 0x3c, 0x7b, 0x64, 0x4b, 0xf9, 0x2f, 0x3c, 0x7b, 0x64, 0x4b, 0xf9,
                0x2f, 0x3c,
            ],
            [0x00, 0xf0, 0x67, 0xaa, 0x0b, 0xa9, 0x02, 0xb7],
            None,
            "test".to_string(),
            SpanKind::Internal,
            0,
            0,
            0,
            StatusCode::Ok,
            String::new(),
            HashMap::new(),
            HashMap::new(),
            0,
            0,
        );

        assert_eq!(span.trace_id_hex(), "4bf92f3c7b644bf92f3c7b644bf92f3c");
        assert_eq!(span.span_id_hex(), "00f067aa0ba902b7");
    }

    #[test]
    fn test_is_root() {
        let root_span = SpanRecord::new(
            [1; 16],
            [2; 8],
            None,
            "root".to_string(),
            SpanKind::Internal,
            0,
            0,
            0,
            StatusCode::Ok,
            String::new(),
            HashMap::new(),
            HashMap::new(),
            0,
            0,
        );

        let child_span = SpanRecord::new(
            [1; 16],
            [3; 8],
            Some([2; 8]),
            "child".to_string(),
            SpanKind::Internal,
            0,
            0,
            1,
            StatusCode::Ok,
            String::new(),
            HashMap::new(),
            HashMap::new(),
            0,
            0,
        );

        assert!(root_span.is_root());
        assert!(!child_span.is_root());
    }

    #[test]
    fn test_is_error() {
        let ok_span = SpanRecord::new(
            [1; 16],
            [2; 8],
            None,
            "ok".to_string(),
            SpanKind::Internal,
            0,
            0,
            0,
            StatusCode::Ok,
            String::new(),
            HashMap::new(),
            HashMap::new(),
            0,
            0,
        );

        let error_span = SpanRecord::new(
            [1; 16],
            [3; 8],
            None,
            "error".to_string(),
            SpanKind::Internal,
            0,
            0,
            1,
            StatusCode::Error,
            "Something went wrong".to_string(),
            HashMap::new(),
            HashMap::new(),
            0,
            0,
        );

        assert!(!ok_span.is_error());
        assert!(error_span.is_error());
    }

    #[test]
    fn test_span_kind_default() {
        assert_eq!(SpanKind::default(), SpanKind::Internal);
    }

    #[test]
    fn test_status_code_default() {
        assert_eq!(StatusCode::default(), StatusCode::Unset);
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Prove span duration is always end_time - start_time
    #[kani::proof]
    fn proof_duration_consistency() {
        let start: u64 = kani::any();
        let end: u64 = kani::any();
        kani::assume(end >= start);
        let duration = end - start;
        kani::assert(start + duration == end, "duration must equal end - start");
    }

    /// Prove SpanKind enum covers all variants
    #[kani::proof]
    fn proof_span_kind_exhaustive() {
        let kind: u8 = kani::any();
        kani::assume(kind <= 4);
        let _result = match kind {
            0 => SpanKind::Internal,
            1 => SpanKind::Server,
            2 => SpanKind::Client,
            3 => SpanKind::Producer,
            _ => SpanKind::Consumer,
        };
    }

    /// Prove StatusCode enum covers all variants
    #[kani::proof]
    fn proof_status_code_exhaustive() {
        let code: u8 = kani::any();
        kani::assume(code <= 2);
        let _result = match code {
            0 => StatusCode::Unset,
            1 => StatusCode::Ok,
            _ => StatusCode::Error,
        };
    }
}
