//! Emergency conditions that trigger immediate action

use crate::monitor::andon::AlertLevel;
use serde::{Deserialize, Serialize};

/// Emergency condition that triggers immediate action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EmergencyCondition {
    /// Collision imminent (AV)
    CollisionImminent {
        /// Time to collision in milliseconds
        time_to_collision_ms: f32,
    },

    /// Sensor degraded
    SensorDegraded {
        /// Sensor identifier
        sensor: String,
        /// Quality score (0.0 - 1.0)
        quality: f32,
    },

    /// Audit chain integrity failure
    AuditChainBroken,

    /// Decision system timeout
    DecisionTimeout {
        /// Maximum allowed latency in milliseconds
        max_ms: f32,
    },

    /// Repeated low-confidence decisions
    ConsecutiveLowConfidence {
        /// Number of consecutive low-confidence decisions
        count: usize,
        /// Threshold for low confidence
        threshold: f32,
    },

    /// NaN or Inf detected in output
    InvalidOutput,
}

impl EmergencyCondition {
    /// Get alert level for this condition
    pub fn alert_level(&self) -> AlertLevel {
        match self {
            EmergencyCondition::CollisionImminent { .. } => AlertLevel::Critical,
            EmergencyCondition::SensorDegraded { quality, .. } if *quality < 0.3 => {
                AlertLevel::Critical
            }
            EmergencyCondition::SensorDegraded { .. } => AlertLevel::Error,
            EmergencyCondition::AuditChainBroken => AlertLevel::Critical,
            EmergencyCondition::DecisionTimeout { .. } => AlertLevel::Error,
            EmergencyCondition::ConsecutiveLowConfidence { .. } => AlertLevel::Warning,
            EmergencyCondition::InvalidOutput => AlertLevel::Critical,
        }
    }

    /// Generate alert message
    pub fn message(&self) -> String {
        match self {
            EmergencyCondition::CollisionImminent { time_to_collision_ms } => {
                format!("Collision imminent in {time_to_collision_ms:.1}ms")
            }
            EmergencyCondition::SensorDegraded { sensor, quality } => {
                let quality_pct = quality * 100.0;
                format!("Sensor {sensor} degraded: quality={quality_pct:.1}%")
            }
            EmergencyCondition::AuditChainBroken => "Audit chain integrity failure".to_string(),
            EmergencyCondition::DecisionTimeout { max_ms } => {
                format!("Decision timeout: exceeded {max_ms:.1}ms limit")
            }
            EmergencyCondition::ConsecutiveLowConfidence { count, threshold } => {
                let threshold_pct = threshold * 100.0;
                format!("{count} consecutive decisions below {threshold_pct:.1}% confidence")
            }
            EmergencyCondition::InvalidOutput => "NaN or Inf detected in model output".to_string(),
        }
    }
}
