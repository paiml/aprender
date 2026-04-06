//! ENT-034: Auto-feature type inference from data
//!
//! Automatically infers feature types from training data by analyzing column statistics.
//! Supports: numeric, categorical, text, datetime, embedding types.

mod config;
mod inference;
mod schema;
mod stats;
mod types;

#[cfg(test)]
mod tests;

// Re-export all public types and functions
pub use config::InferenceConfig;
pub use inference::{collect_stats_from_samples, infer_schema, infer_schema_from_path, infer_type};
pub use schema::InferredSchema;
pub use stats::ColumnStats;
pub use types::FeatureType;
