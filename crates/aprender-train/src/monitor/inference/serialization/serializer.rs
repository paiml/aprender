//! Trace serializer implementation.

use super::error::SerializationError;
use super::format::{PathType, TraceFormat, APRT_MAGIC, APRT_VERSION};
use crate::monitor::inference::path::DecisionPath;
use crate::monitor::inference::trace::DecisionTrace;
use serde::{Deserialize, Serialize};

/// Trace serializer
pub struct TraceSerializer {
    format: TraceFormat,
}

impl TraceSerializer {
    /// Create a new serializer
    pub fn new(format: TraceFormat) -> Self {
        Self { format }
    }

    /// Get the format
    pub fn format(&self) -> TraceFormat {
        self.format
    }

    /// Serialize a trace to bytes
    pub fn serialize<P: DecisionPath + Serialize>(
        &self,
        trace: &DecisionTrace<P>,
        path_type: PathType,
    ) -> Result<Vec<u8>, SerializationError> {
        match self.format {
            TraceFormat::Binary => self.serialize_binary(trace, path_type),
            TraceFormat::Json => self.serialize_json(trace),
            TraceFormat::JsonLines => self.serialize_json_line(trace),
        }
    }

    /// Serialize to APRT binary format
    fn serialize_binary<P: DecisionPath + Serialize>(
        &self,
        trace: &DecisionTrace<P>,
        path_type: PathType,
    ) -> Result<Vec<u8>, SerializationError> {
        let trace_bytes = trace.to_bytes();

        let mut bytes = Vec::with_capacity(8 + trace_bytes.len());

        // Header
        bytes.extend_from_slice(&APRT_MAGIC);
        bytes.push(APRT_VERSION);
        bytes.push(path_type as u8);
        bytes.extend_from_slice(&[0, 0]); // Reserved (alignment)

        // Trace data
        bytes.extend_from_slice(&trace_bytes);

        Ok(bytes)
    }

    /// Serialize to JSON
    fn serialize_json<P: DecisionPath + Serialize>(
        &self,
        trace: &DecisionTrace<P>,
    ) -> Result<Vec<u8>, SerializationError> {
        serde_json::to_vec_pretty(trace).map_err(SerializationError::Json)
    }

    /// Serialize to JSON line (no trailing newline)
    fn serialize_json_line<P: DecisionPath + Serialize>(
        &self,
        trace: &DecisionTrace<P>,
    ) -> Result<Vec<u8>, SerializationError> {
        let mut bytes = serde_json::to_vec(trace).map_err(SerializationError::Json)?;
        bytes.push(b'\n');
        Ok(bytes)
    }

    /// Deserialize from bytes
    pub fn deserialize<P: DecisionPath + for<'de> Deserialize<'de>>(
        &self,
        bytes: &[u8],
    ) -> Result<DecisionTrace<P>, SerializationError> {
        match self.format {
            TraceFormat::Binary => self.deserialize_binary(bytes),
            TraceFormat::Json | TraceFormat::JsonLines => self.deserialize_json(bytes),
        }
    }

    /// Deserialize from APRT binary format
    fn deserialize_binary<P: DecisionPath>(
        &self,
        bytes: &[u8],
    ) -> Result<DecisionTrace<P>, SerializationError> {
        if bytes.len() < 8 {
            return Err(SerializationError::InvalidFormat("Insufficient header bytes".to_string()));
        }

        // Check magic
        if bytes[0..4] != APRT_MAGIC {
            return Err(SerializationError::InvalidFormat("Invalid APRT magic bytes".to_string()));
        }

        // Check version
        let version = bytes[4];
        if version != APRT_VERSION {
            return Err(SerializationError::VersionMismatch {
                expected: APRT_VERSION,
                actual: version,
            });
        }

        // Skip path type and reserved bytes (bytes[5], bytes[6], bytes[7])

        // Deserialize trace
        DecisionTrace::from_bytes(bytes.get(8..).unwrap_or_default()).map_err(|e| {
            SerializationError::InvalidFormat(format!("Path deserialization failed: {e}"))
        })
    }

    /// Deserialize from JSON
    fn deserialize_json<P: DecisionPath + for<'de> Deserialize<'de>>(
        &self,
        bytes: &[u8],
    ) -> Result<DecisionTrace<P>, SerializationError> {
        serde_json::from_slice(bytes).map_err(SerializationError::Json)
    }
}

impl Default for TraceSerializer {
    fn default() -> Self {
        Self::new(TraceFormat::Binary)
    }
}
