//! Trace Serialization (ENT-108)
//!
//! Binary and JSON serialization for decision traces.

mod error;
mod format;
mod serializer;

pub use error::SerializationError;
pub use format::{PathType, TraceFormat, APRT_MAGIC, APRT_VERSION};
pub use serializer::TraceSerializer;

// Re-export trace types for convenience in tests

#[cfg(test)]
mod tests;
