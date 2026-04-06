//! Stage transition records

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::stage::ModelStage;

/// Stage transition record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageTransition {
    /// Model name
    pub model_name: String,
    /// Version
    pub version: u32,
    /// Previous stage
    pub from_stage: ModelStage,
    /// New stage
    pub to_stage: ModelStage,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// User who made the transition
    pub user: Option<String>,
    /// Reason for transition
    pub reason: Option<String>,
}
