//! Memory substrate — agent persistent state.
//!
//! Defines the `MemorySubstrate` trait for storing and recalling
//! agent memories. Phase 1 provides `InMemorySubstrate` (ephemeral,
//! substring matching). Phase 2 adds `TruenoMemory` (durable,
//! semantic similarity via trueno-rag vector search).
//!
//! See: arXiv:2512.13564 (memory survey), arXiv:2602.19320 (taxonomy).

pub mod in_memory;
#[cfg(feature = "rag")]
pub mod trueno;

pub use in_memory::InMemorySubstrate;
#[cfg(feature = "rag")]
pub use trueno::TruenoMemory;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::agent::result::AgentError;

/// Unique identifier for a stored memory fragment.
pub type MemoryId = String;

/// Filter for memory recall queries.
#[derive(Debug, Clone, Default)]
pub struct MemoryFilter {
    /// Filter by agent ID.
    pub agent_id: Option<String>,
    /// Filter by memory source type.
    pub source: Option<MemorySource>,
    /// Filter memories created after this time.
    pub since: Option<chrono::DateTime<chrono::Utc>>,
}

/// Source of a memory fragment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemorySource {
    /// From agent conversation.
    Conversation,
    /// From tool execution result.
    ToolResult,
    /// System-injected memory.
    System,
    /// User-provided memory.
    User,
}

/// A recalled memory fragment with relevance score.
#[derive(Debug, Clone)]
pub struct MemoryFragment {
    /// Unique identifier.
    pub id: MemoryId,
    /// Memory content text.
    pub content: String,
    /// How the memory was created.
    pub source: MemorySource,
    /// Relevance score (0.0-1.0, higher = more relevant).
    pub relevance_score: f32,
    /// When the memory was created.
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// Unified structured + semantic memory store.
#[async_trait]
pub trait MemorySubstrate: Send + Sync {
    /// Store a memory fragment.
    async fn remember(
        &self,
        agent_id: &str,
        content: &str,
        source: MemorySource,
        embedding: Option<&[f32]>,
    ) -> Result<MemoryId, AgentError>;

    /// Recall relevant memories.
    async fn recall(
        &self,
        query: &str,
        limit: usize,
        filter: Option<MemoryFilter>,
        query_embedding: Option<&[f32]>,
    ) -> Result<Vec<MemoryFragment>, AgentError>;

    /// Store structured key-value data.
    async fn set(
        &self,
        agent_id: &str,
        key: &str,
        value: serde_json::Value,
    ) -> Result<(), AgentError>;

    /// Retrieve structured key-value data.
    async fn get(&self, agent_id: &str, key: &str)
        -> Result<Option<serde_json::Value>, AgentError>;

    /// Delete a memory fragment.
    async fn forget(&self, id: MemoryId) -> Result<(), AgentError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_source_serialization() {
        let sources = vec![
            MemorySource::Conversation,
            MemorySource::ToolResult,
            MemorySource::System,
            MemorySource::User,
        ];
        for source in &sources {
            let json = serde_json::to_string(source).expect("serialize failed");
            let back: MemorySource = serde_json::from_str(&json).expect("deserialize failed");
            assert_eq!(*source, back);
        }
    }

    #[test]
    fn test_memory_filter_default() {
        let filter = MemoryFilter::default();
        assert!(filter.agent_id.is_none());
        assert!(filter.source.is_none());
        assert!(filter.since.is_none());
    }
}
