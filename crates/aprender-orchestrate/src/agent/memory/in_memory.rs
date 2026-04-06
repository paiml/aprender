//! In-memory substrate — ephemeral, substring-matching memory.
//!
//! Phase 1 implementation. Uses `HashMap` for key-value and `Vec` for
//! fragment storage. Recall uses case-insensitive substring matching
//! (NOT semantic similarity — that requires `TruenoMemory` in Phase 2).

use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Mutex;

use super::{MemoryFilter, MemoryFragment, MemoryId, MemorySource, MemorySubstrate};
use crate::agent::result::AgentError;

/// In-memory substrate (ephemeral, no persistence).
pub struct InMemorySubstrate {
    /// Fragment storage.
    fragments: Mutex<Vec<StoredFragment>>,
    /// Key-value storage.
    kv: Mutex<HashMap<String, serde_json::Value>>,
    /// Counter for generating unique IDs.
    next_id: Mutex<u64>,
}

struct StoredFragment {
    id: MemoryId,
    agent_id: String,
    content: String,
    source: MemorySource,
    created_at: chrono::DateTime<chrono::Utc>,
}

impl InMemorySubstrate {
    /// Create an empty in-memory substrate.
    pub fn new() -> Self {
        Self {
            fragments: Mutex::new(Vec::new()),
            kv: Mutex::new(HashMap::new()),
            next_id: Mutex::new(1),
        }
    }

    fn gen_id(&self) -> String {
        let mut id = self.next_id.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let current = *id;
        *id += 1;
        format!("mem-{current}")
    }

    fn kv_key(agent_id: &str, key: &str) -> String {
        format!("{agent_id}:{key}")
    }
}

impl Default for InMemorySubstrate {
    fn default() -> Self {
        Self::new()
    }
}

/// Acquire a mutex lock, mapping poison errors to `AgentError`.
fn lock<T>(mutex: &Mutex<T>) -> Result<std::sync::MutexGuard<'_, T>, AgentError> {
    mutex.lock().map_err(|e| AgentError::Memory(format!("lock: {e}")))
}

/// Check if a stored fragment passes the optional filter.
fn matches_filter(f: &StoredFragment, filter: Option<&MemoryFilter>) -> bool {
    let Some(filter) = filter else { return true };
    if let Some(ref aid) = filter.agent_id {
        if f.agent_id != *aid {
            return false;
        }
    }
    if let Some(ref src) = filter.source {
        if f.source != *src {
            return false;
        }
    }
    if let Some(since) = filter.since {
        if f.created_at < since {
            return false;
        }
    }
    true
}

/// Score a stored fragment for relevance based on query length ratio.
fn score_fragment(f: &StoredFragment, query: &str) -> MemoryFragment {
    let score = if f.content.is_empty() {
        0.0
    } else {
        // Precision loss acceptable: string lengths fit easily in f32
        #[allow(clippy::cast_precision_loss)]
        let s = (query.len() as f32 / f.content.len() as f32).min(1.0);
        s
    };
    MemoryFragment {
        id: f.id.clone(),
        content: f.content.clone(),
        source: f.source.clone(),
        relevance_score: score,
        created_at: f.created_at,
    }
}

#[async_trait]
impl MemorySubstrate for InMemorySubstrate {
    async fn remember(
        &self,
        agent_id: &str,
        content: &str,
        source: MemorySource,
        _embedding: Option<&[f32]>,
    ) -> Result<MemoryId, AgentError> {
        let id = self.gen_id();
        let fragment = StoredFragment {
            id: id.clone(),
            agent_id: agent_id.to_string(),
            content: content.to_string(),
            source,
            created_at: chrono::Utc::now(),
        };
        lock(&self.fragments)?.push(fragment);
        Ok(id)
    }

    async fn recall(
        &self,
        query: &str,
        limit: usize,
        filter: Option<MemoryFilter>,
        _query_embedding: Option<&[f32]>,
    ) -> Result<Vec<MemoryFragment>, AgentError> {
        let fragments = lock(&self.fragments)?;

        let query_lower = query.to_lowercase();

        let mut results: Vec<MemoryFragment> = fragments
            .iter()
            .filter(|f| {
                matches_filter(f, filter.as_ref())
                    && f.content.to_lowercase().contains(&query_lower)
            })
            .map(|f| score_fragment(f, query))
            .collect();

        results.sort_by(|a, b| {
            b.relevance_score.partial_cmp(&a.relevance_score).unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(limit);

        Ok(results)
    }

    async fn set(
        &self,
        agent_id: &str,
        key: &str,
        value: serde_json::Value,
    ) -> Result<(), AgentError> {
        lock(&self.kv)?.insert(Self::kv_key(agent_id, key), value);
        Ok(())
    }

    async fn get(
        &self,
        agent_id: &str,
        key: &str,
    ) -> Result<Option<serde_json::Value>, AgentError> {
        let kv = lock(&self.kv)?;
        Ok(kv.get(&Self::kv_key(agent_id, key)).cloned())
    }

    async fn forget(&self, id: MemoryId) -> Result<(), AgentError> {
        lock(&self.fragments)?.retain(|f| f.id != id);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_remember_and_recall() {
        let substrate = InMemorySubstrate::new();
        substrate
            .remember("agent1", "Rust is fast", MemorySource::User, None)
            .await
            .expect("remember failed");

        let results = substrate.recall("Rust", 10, None, None).await.expect("recall failed");
        assert_eq!(results.len(), 1);
        assert!(results[0].content.contains("Rust is fast"));
    }

    #[tokio::test]
    async fn test_recall_case_insensitive() {
        let substrate = InMemorySubstrate::new();
        substrate
            .remember("a", "HELLO WORLD", MemorySource::System, None)
            .await
            .expect("remember failed");

        let results = substrate.recall("hello", 10, None, None).await.expect("recall failed");
        assert_eq!(results.len(), 1);
    }

    #[tokio::test]
    async fn test_recall_no_match() {
        let substrate = InMemorySubstrate::new();
        substrate.remember("a", "apples", MemorySource::User, None).await.expect("remember failed");

        let results = substrate.recall("oranges", 10, None, None).await.expect("recall failed");
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_recall_limit() {
        let substrate = InMemorySubstrate::new();
        for i in 0..10 {
            substrate
                .remember("a", &format!("item {i} with keyword"), MemorySource::Conversation, None)
                .await
                .expect("remember failed");
        }

        let results = substrate.recall("keyword", 3, None, None).await.expect("recall failed");
        assert_eq!(results.len(), 3);
    }

    #[tokio::test]
    async fn test_filter_by_agent_id() {
        let substrate = InMemorySubstrate::new();
        substrate
            .remember("agent1", "secret data", MemorySource::User, None)
            .await
            .expect("remember failed");
        substrate
            .remember("agent2", "other data", MemorySource::User, None)
            .await
            .expect("remember failed");

        let filter = MemoryFilter { agent_id: Some("agent1".into()), ..Default::default() };
        let results =
            substrate.recall("data", 10, Some(filter), None).await.expect("recall failed");
        assert_eq!(results.len(), 1);
        assert!(results[0].content.contains("secret"));
    }

    #[tokio::test]
    async fn test_kv_set_get() {
        let substrate = InMemorySubstrate::new();
        substrate.set("a", "key1", serde_json::json!(42)).await.expect("set failed");

        let val = substrate.get("a", "key1").await.expect("get failed");
        assert_eq!(val, Some(serde_json::json!(42)));

        let missing = substrate.get("a", "nonexistent").await.expect("get failed");
        assert!(missing.is_none());
    }

    #[tokio::test]
    async fn test_kv_isolation() {
        let substrate = InMemorySubstrate::new();
        substrate.set("agent1", "key", serde_json::json!("one")).await.expect("set failed");
        substrate.set("agent2", "key", serde_json::json!("two")).await.expect("set failed");

        let v1 = substrate.get("agent1", "key").await.expect("get failed");
        let v2 = substrate.get("agent2", "key").await.expect("get failed");
        assert_eq!(v1, Some(serde_json::json!("one")));
        assert_eq!(v2, Some(serde_json::json!("two")));
    }

    #[tokio::test]
    async fn test_forget() {
        let substrate = InMemorySubstrate::new();
        let id = substrate
            .remember("a", "forget me", MemorySource::User, None)
            .await
            .expect("remember failed");

        substrate.forget(id).await.expect("forget failed");

        let results = substrate.recall("forget", 10, None, None).await.expect("recall failed");
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_unique_ids() {
        let substrate = InMemorySubstrate::new();
        let id1 = substrate
            .remember("a", "one", MemorySource::User, None)
            .await
            .expect("remember failed");
        let id2 = substrate
            .remember("a", "two", MemorySource::User, None)
            .await
            .expect("remember failed");
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_default() {
        let substrate = InMemorySubstrate::default();
        assert_eq!(substrate.gen_id(), "mem-1");
    }

    #[tokio::test]
    async fn test_filter_by_source() {
        let substrate = InMemorySubstrate::new();
        substrate
            .remember("a", "user msg", MemorySource::User, None)
            .await
            .expect("remember failed");
        substrate
            .remember("a", "system msg", MemorySource::System, None)
            .await
            .expect("remember failed");

        let filter = MemoryFilter { source: Some(MemorySource::System), ..Default::default() };
        let results = substrate.recall("msg", 10, Some(filter), None).await.expect("recall failed");
        assert_eq!(results.len(), 1);
        assert!(results[0].content.contains("system"));
    }

    #[tokio::test]
    async fn test_filter_by_since() {
        let substrate = InMemorySubstrate::new();
        substrate
            .remember("a", "old memory", MemorySource::User, None)
            .await
            .expect("remember failed");

        let after_first = chrono::Utc::now();

        substrate
            .remember("a", "new memory", MemorySource::User, None)
            .await
            .expect("remember failed");

        let filter = MemoryFilter { since: Some(after_first), ..Default::default() };
        let results =
            substrate.recall("memory", 10, Some(filter), None).await.expect("recall failed");
        assert_eq!(results.len(), 1);
        assert!(results[0].content.contains("new"));
    }

    #[test]
    fn test_score_empty_content() {
        let f = StoredFragment {
            id: "mem-1".into(),
            agent_id: "a".into(),
            content: String::new(),
            source: MemorySource::User,
            created_at: chrono::Utc::now(),
        };
        let scored = score_fragment(&f, "query");
        assert_eq!(scored.relevance_score, 0.0);
    }

    #[test]
    fn test_score_long_content() {
        let f = StoredFragment {
            id: "mem-1".into(),
            agent_id: "a".into(),
            content: "a very long content string that is much longer than the query".into(),
            source: MemorySource::User,
            created_at: chrono::Utc::now(),
        };
        let scored = score_fragment(&f, "short");
        assert!(scored.relevance_score > 0.0);
        assert!(scored.relevance_score < 1.0);
    }
}
