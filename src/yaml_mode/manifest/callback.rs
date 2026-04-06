//! Callback Configuration
//!
//! Contains callback configuration types for training manifests.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Callback configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallbackConfig {
    /// Callback type
    #[serde(rename = "type")]
    pub callback_type: CallbackType,

    /// Trigger event
    pub trigger: String,

    /// Interval (for step-based triggers)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interval: Option<usize>,

    /// Callback-specific configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub config: Option<HashMap<String, serde_json::Value>>,

    /// Custom script (for custom callbacks)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub script: Option<String>,
}

/// Callback type
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CallbackType {
    Checkpoint,
    LrMonitor,
    GradientMonitor,
    SamplePredictions,
    Custom,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_callback_type_serde() {
        let json = r#""checkpoint""#;
        let ct: CallbackType =
            serde_json::from_str(json).expect("JSON deserialization should succeed");
        assert_eq!(ct, CallbackType::Checkpoint);
    }
}
