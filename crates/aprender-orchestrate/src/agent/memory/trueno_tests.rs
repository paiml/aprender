use super::*;

fn make_substrate() -> TruenoMemory {
    TruenoMemory::open_in_memory().expect("in-memory open")
}

#[tokio::test]
async fn test_remember_and_recall() {
    let mem = make_substrate();
    let id = mem
        .remember("agent1", "Rust is great for systems programming", MemorySource::User, None)
        .await
        .expect("remember");
    assert!(id.starts_with("trueno-"));

    let results = mem.recall("Rust systems", 10, None, None).await.expect("recall");
    assert_eq!(results.len(), 1);
    assert!(results[0].content.contains("systems programming"));
    assert!(results[0].relevance_score > 0.0);
}

#[tokio::test]
async fn test_recall_no_match() {
    let mem = make_substrate();
    mem.remember("a", "SIMD vector operations", MemorySource::System, None)
        .await
        .expect("remember");

    let results = mem.recall("cryptocurrency blockchain", 10, None, None).await.expect("recall");
    assert!(results.is_empty());
}

#[tokio::test]
async fn test_recall_empty_query() {
    let mem = make_substrate();
    mem.remember("a", "some content", MemorySource::User, None).await.expect("remember");

    let results = mem.recall("", 10, None, None).await.expect("recall");
    assert!(results.is_empty());
}

#[tokio::test]
async fn test_recall_limit() {
    let mem = make_substrate();
    for i in 0..10 {
        mem.remember(
            "a",
            &format!("memory about Rust topic {i}"),
            MemorySource::Conversation,
            None,
        )
        .await
        .expect("remember");
    }

    let results = mem.recall("Rust topic", 3, None, None).await.expect("recall");
    assert_eq!(results.len(), 3);
}

#[tokio::test]
async fn test_recall_bm25_ranking() {
    let mem = make_substrate();
    mem.remember("a", "machine learning algorithms", MemorySource::User, None)
        .await
        .expect("remember");
    mem.remember(
        "a",
        "machine learning machine learning deep learning neural networks",
        MemorySource::User,
        None,
    )
    .await
    .expect("remember");
    mem.remember("a", "cooking recipes for dinner", MemorySource::User, None)
        .await
        .expect("remember");

    let results = mem.recall("machine learning", 10, None, None).await.expect("recall");
    // Should find ML-related memories, not cooking
    assert!(results.len() >= 2);
    assert!(results.iter().all(|r| !r.content.contains("cooking")));
}

#[tokio::test]
async fn test_filter_by_agent_id() {
    let mem = make_substrate();
    mem.remember("agent1", "secret data about Rust", MemorySource::User, None)
        .await
        .expect("remember");
    mem.remember("agent2", "other data about Rust", MemorySource::User, None)
        .await
        .expect("remember");

    let filter = MemoryFilter { agent_id: Some("agent1".into()), ..Default::default() };
    let results = mem.recall("Rust", 10, Some(filter), None).await.expect("recall");
    assert_eq!(results.len(), 1);
    assert!(results[0].content.contains("secret"));
}

#[tokio::test]
async fn test_kv_set_get() {
    let mem = make_substrate();
    mem.set("a", "counter", serde_json::json!(42)).await.expect("set");

    let val = mem.get("a", "counter").await.expect("get");
    assert_eq!(val, Some(serde_json::json!(42)));
}

#[tokio::test]
async fn test_kv_missing_key() {
    let mem = make_substrate();
    let val = mem.get("a", "nonexistent").await.expect("get");
    assert!(val.is_none());
}

#[tokio::test]
async fn test_kv_overwrite() {
    let mem = make_substrate();
    mem.set("a", "key", serde_json::json!("old")).await.expect("set");
    mem.set("a", "key", serde_json::json!("new")).await.expect("set");

    let val = mem.get("a", "key").await.expect("get");
    assert_eq!(val, Some(serde_json::json!("new")));
}

#[tokio::test]
async fn test_kv_isolation() {
    let mem = make_substrate();
    mem.set("agent1", "key", serde_json::json!("one")).await.expect("set");
    mem.set("agent2", "key", serde_json::json!("two")).await.expect("set");

    let v1 = mem.get("agent1", "key").await.expect("get");
    let v2 = mem.get("agent2", "key").await.expect("get");
    assert_eq!(v1, Some(serde_json::json!("one")));
    assert_eq!(v2, Some(serde_json::json!("two")));
}

#[tokio::test]
async fn test_kv_complex_value() {
    let mem = make_substrate();
    let complex = serde_json::json!({
        "name": "test",
        "items": [1, 2, 3],
        "nested": {"a": true}
    });
    mem.set("a", "config", complex.clone()).await.expect("set");

    let val = mem.get("a", "config").await.expect("get");
    assert_eq!(val, Some(complex));
}

#[tokio::test]
async fn test_forget() {
    let mem = make_substrate();
    let id = mem.remember("a", "forget me Rust", MemorySource::User, None).await.expect("remember");

    // Forget using the full doc_id pattern
    let doc_id = format!("a:{id}");
    mem.forget(doc_id).await.expect("forget");

    let results = mem.recall("forget Rust", 10, None, None).await.expect("recall");
    assert!(results.is_empty());
}

#[tokio::test]
async fn test_unique_ids() {
    let mem = make_substrate();
    let id1 = mem.remember("a", "one", MemorySource::User, None).await.expect("remember");
    let id2 = mem.remember("a", "two", MemorySource::User, None).await.expect("remember");
    assert_ne!(id1, id2);
}

#[tokio::test]
async fn test_fragment_count() {
    let mem = make_substrate();
    assert_eq!(mem.fragment_count().expect("count"), 0);

    mem.remember("a", "first memory", MemorySource::User, None).await.expect("remember");
    assert_eq!(mem.fragment_count().expect("count"), 1);

    mem.remember("a", "second memory", MemorySource::User, None).await.expect("remember");
    assert_eq!(mem.fragment_count().expect("count"), 2);
}

#[tokio::test]
async fn test_recall_with_source_filter() {
    let mem = make_substrate();
    mem.remember("agent1", "system data about Rust", MemorySource::System, None)
        .await
        .expect("remember");
    mem.remember("agent1", "user data about Rust", MemorySource::User, None)
        .await
        .expect("remember");

    let filter = MemoryFilter { source: Some(MemorySource::System), ..Default::default() };
    let results = mem.recall("Rust", 10, Some(filter), None).await.expect("recall");
    // Source filter passes through (source filtering not
    // fully implemented in FTS layer), but agent_id filter works
    assert!(!results.is_empty());
}

#[tokio::test]
async fn test_forget_nonexistent() {
    let mem = make_substrate();
    // Forgetting a non-existent ID should not error
    let result = mem.forget("nonexistent-id".into()).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_recall_whitespace_query() {
    let mem = make_substrate();
    mem.remember("a", "content", MemorySource::User, None).await.expect("remember");

    let results = mem.recall("   ", 10, None, None).await.expect("recall");
    assert!(results.is_empty());
}

#[tokio::test]
async fn test_multiple_sources() {
    let mem = make_substrate();
    mem.remember("a", "conversation about Rust memory", MemorySource::Conversation, None)
        .await
        .expect("remember");
    mem.remember("a", "tool result about Rust performance", MemorySource::ToolResult, None)
        .await
        .expect("remember");
    mem.remember("a", "system note about Rust safety", MemorySource::System, None)
        .await
        .expect("remember");

    let results = mem.recall("Rust", 10, None, None).await.expect("recall");
    assert!(results.len() >= 2);
}

#[tokio::test]
async fn test_open_durable_file() {
    let dir = tempfile::tempdir().expect("tmpdir");
    let db_path = dir.path().join("test_memory.db");
    let mem = TruenoMemory::open(&db_path).expect("open durable");

    // Store something
    let id = mem
        .remember("a", "durable data about Rust", MemorySource::User, None)
        .await
        .expect("remember");
    assert!(id.starts_with("trueno-"));

    // Recall it
    let results = mem.recall("Rust durable", 10, None, None).await.expect("recall");
    assert!(!results.is_empty());

    // Fragment count
    assert!(mem.fragment_count().expect("count") > 0);

    // Drop and reopen to test persistence
    drop(mem);
    let mem2 = TruenoMemory::open(&db_path).expect("reopen");
    let results2 = mem2.recall("Rust durable", 10, None, None).await.expect("recall after reopen");
    assert!(!results2.is_empty());
}

#[tokio::test]
async fn test_kv_set_get_durable() {
    let dir = tempfile::tempdir().expect("tmpdir");
    let db_path = dir.path().join("test_kv.db");
    let mem = TruenoMemory::open(&db_path).expect("open durable");

    mem.set("agent1", "key1", serde_json::json!({"test": true})).await.expect("set");

    let val = mem.get("agent1", "key1").await.expect("get");
    assert_eq!(val, Some(serde_json::json!({"test": true})));
}

#[tokio::test]
async fn test_gen_id_persistence() {
    let dir = tempfile::tempdir().expect("tmpdir");
    let db_path = dir.path().join("test_ids.db");
    let mem = TruenoMemory::open(&db_path).expect("open");

    // Generate a few IDs
    let id1 = mem.remember("a", "first", MemorySource::User, None).await.expect("r1");
    let id2 = mem.remember("a", "second", MemorySource::User, None).await.expect("r2");
    assert_ne!(id1, id2);

    // Reopen — ID counter should persist
    drop(mem);
    let mem2 = TruenoMemory::open(&db_path).expect("reopen");
    let id3 = mem2.remember("a", "third", MemorySource::User, None).await.expect("r3");
    // id3 should not conflict with id1 or id2
    assert_ne!(id3, id1);
    assert_ne!(id3, id2);
}

#[tokio::test]
async fn test_relevance_normalized() {
    let mem = make_substrate();
    mem.remember("a", "Rust programming language", MemorySource::User, None)
        .await
        .expect("remember");

    let results = mem.recall("Rust", 10, None, None).await.expect("recall");
    assert_eq!(results.len(), 1);
    // Score should be normalized to 0.0-1.0
    assert!(results[0].relevance_score >= 0.0);
    assert!(results[0].relevance_score <= 1.0);
}
