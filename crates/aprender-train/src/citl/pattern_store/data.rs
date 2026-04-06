//! Serializable data structures for pattern store persistence.

use super::{FixPattern, PatternStoreConfig};
use serde::{Deserialize, Serialize};

/// Serializable wrapper for pattern store data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternStoreData {
    /// Format version for future compatibility
    pub version: u32,
    /// Store configuration
    pub config: PatternStoreConfig,
    /// All indexed patterns
    pub patterns: Vec<FixPattern>,
}
