//! Agent loop phase tracking.
//!
//! Defines the discrete phases of the perceive-reason-act loop.
//! Modeled as a finite state machine — transitions are deterministic
//! and bounded (arXiv:2512.10350 — contractive dynamics).

use serde::{Deserialize, Serialize};

/// Phase of the agent loop FSM.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LoopPhase {
    /// Recalling memories and context for the current query.
    Perceive,
    /// Generating a completion via the LLM driver.
    Reason,
    /// Executing a tool call.
    Act {
        /// Name of the tool being executed.
        tool_name: String,
    },
    /// Agent has produced a final response.
    Done,
    /// Agent encountered an unrecoverable error.
    Error {
        /// Human-readable error description.
        message: String,
    },
}

impl std::fmt::Display for LoopPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Perceive => write!(f, "perceive"),
            Self::Reason => write!(f, "reason"),
            Self::Act { tool_name } => write!(f, "act:{tool_name}"),
            Self::Done => write!(f, "done"),
            Self::Error { message } => write!(f, "error:{message}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phase_display() {
        assert_eq!(LoopPhase::Perceive.to_string(), "perceive");
        assert_eq!(LoopPhase::Reason.to_string(), "reason");
        assert_eq!(LoopPhase::Act { tool_name: "rag".to_string() }.to_string(), "act:rag");
        assert_eq!(LoopPhase::Done.to_string(), "done");
        assert_eq!(LoopPhase::Error { message: "budget".to_string() }.to_string(), "error:budget");
    }

    #[test]
    fn test_phase_equality() {
        assert_eq!(LoopPhase::Perceive, LoopPhase::Perceive);
        assert_ne!(LoopPhase::Perceive, LoopPhase::Reason);
        assert_ne!(
            LoopPhase::Act { tool_name: "a".into() },
            LoopPhase::Act { tool_name: "b".into() }
        );
    }

    #[test]
    fn test_phase_serialization_roundtrip() {
        let phases = vec![
            LoopPhase::Perceive,
            LoopPhase::Reason,
            LoopPhase::Act { tool_name: "memory".into() },
            LoopPhase::Done,
            LoopPhase::Error { message: "out of budget".into() },
        ];
        for phase in &phases {
            let json = serde_json::to_string(phase).expect("serialize failed");
            let back: LoopPhase = serde_json::from_str(&json).expect("deserialize failed");
            assert_eq!(*phase, back);
        }
    }
}
