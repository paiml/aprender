use std::sync::Arc;

use super::*;
use crate::agent::driver::mock::MockDriver;
use crate::agent::driver::CompletionResponse;
use crate::agent::result::{StopReason, TokenUsage};

fn test_manifest(name: &str) -> AgentManifest {
    let mut m = AgentManifest::default();
    m.name = name.to_string();
    m
}

/// Create a mock driver with N identical responses.
fn mock_driver(text: &str, count: usize) -> Arc<MockDriver> {
    let responses: Vec<_> = (0..count)
        .map(|_| CompletionResponse {
            text: text.to_string(),
            stop_reason: StopReason::EndTurn,
            tool_calls: vec![],
            usage: TokenUsage { input_tokens: 10, output_tokens: 5 },
        })
        .collect();
    Arc::new(MockDriver::new(responses))
}

#[tokio::test]
async fn test_pool_spawn_single() {
    let driver = mock_driver("agent-1 done", 1);
    let mut pool = AgentPool::new(driver, 4);

    let id = pool
        .spawn(SpawnConfig { manifest: test_manifest("agent-1"), query: "hello".into() })
        .expect("spawn failed");

    assert_eq!(id, 1);
    assert_eq!(pool.active_count(), 1);
}

#[tokio::test]
async fn test_pool_join_all() {
    let driver = mock_driver("done", 4);
    let mut pool = AgentPool::new(driver, 4);

    pool.spawn(SpawnConfig { manifest: test_manifest("a1"), query: "q1".into() })
        .expect("spawn a1");

    pool.spawn(SpawnConfig { manifest: test_manifest("a2"), query: "q2".into() })
        .expect("spawn a2");

    let results = pool.join_all().await;
    assert_eq!(results.len(), 2);

    for (_, result) in &results {
        let r = result.as_ref().expect("agent should succeed");
        assert_eq!(r.text, "done");
    }
}

#[tokio::test]
async fn test_pool_capacity_limit() {
    let driver = mock_driver("done", 4);
    let mut pool = AgentPool::new(driver, 1);

    pool.spawn(SpawnConfig { manifest: test_manifest("a1"), query: "q1".into() })
        .expect("spawn a1");

    // Second spawn should fail — pool at capacity
    let err =
        pool.spawn(SpawnConfig { manifest: test_manifest("a2"), query: "q2".into() }).unwrap_err();

    assert!(matches!(err, AgentError::CircuitBreak(_)), "expected CircuitBreak, got: {err}");
}

#[tokio::test]
async fn test_pool_fan_out_fan_in() {
    let driver = mock_driver("result", 3);
    let mut pool = AgentPool::new(driver, 4);

    let configs = vec![
        SpawnConfig { manifest: test_manifest("w1"), query: "task1".into() },
        SpawnConfig { manifest: test_manifest("w2"), query: "task2".into() },
        SpawnConfig { manifest: test_manifest("w3"), query: "task3".into() },
    ];

    let ids = pool.fan_out(configs).expect("fan_out");
    assert_eq!(ids.len(), 3);

    let results = pool.join_all().await;
    assert_eq!(results.len(), 3);
}

#[tokio::test]
async fn test_pool_join_next() {
    let driver = mock_driver("one", 1);
    let mut pool = AgentPool::new(driver, 4);

    pool.spawn(SpawnConfig { manifest: test_manifest("single"), query: "q".into() })
        .expect("spawn");

    let (id, result) = pool.join_next().await.expect("should have result");
    assert_eq!(id, 1);
    assert_eq!(result.expect("agent ok").text, "one");

    // No more agents
    assert!(pool.join_next().await.is_none());
}

#[tokio::test]
async fn test_pool_abort_all() {
    let driver = mock_driver("done", 4);
    let mut pool = AgentPool::new(driver, 4);

    pool.spawn(SpawnConfig { manifest: test_manifest("abort-me"), query: "q".into() })
        .expect("spawn");

    pool.abort_all();
    // After abort, join_all returns whatever completed
    let results = pool.join_all().await;
    // Aborted tasks may or may not have completed
    assert!(results.len() <= 1);
}

#[tokio::test]
async fn test_pool_with_shared_memory() {
    let driver = mock_driver("memorized", 1);
    let memory = Arc::new(InMemorySubstrate::new());
    let mut pool = AgentPool::new(driver, 4).with_memory(memory);

    pool.spawn(SpawnConfig { manifest: test_manifest("mem-agent"), query: "remember this".into() })
        .expect("spawn");

    let results = pool.join_all().await;
    assert_eq!(results.len(), 1);
}

#[test]
fn test_pool_defaults() {
    let driver = mock_driver("x", 2);
    let pool = AgentPool::new(driver, 8);
    assert_eq!(pool.max_concurrent(), 8);
    assert_eq!(pool.active_count(), 0);
}

#[test]
fn test_agent_message_fields() {
    let msg = AgentMessage { from: 0, to: 1, content: "hello sub-agent".into() };
    assert_eq!(msg.from, 0);
    assert_eq!(msg.to, 1);
    assert_eq!(msg.content, "hello sub-agent");
}

#[tokio::test]
async fn test_pool_increments_ids() {
    let driver = mock_driver("x", 2);
    let mut pool = AgentPool::new(driver, 4);

    let id1 =
        pool.spawn(SpawnConfig { manifest: test_manifest("a"), query: "q".into() }).expect("spawn");

    // Drain first to free capacity tracking
    let _ = pool.join_all().await;

    let id2 =
        pool.spawn(SpawnConfig { manifest: test_manifest("b"), query: "q".into() }).expect("spawn");

    assert_eq!(id1, 1);
    assert_eq!(id2, 2);
}

#[test]
fn test_router_register_unregister() {
    let router = MessageRouter::new(8);
    let _rx = router.register(1);
    assert_eq!(router.agent_count(), 1);

    let _rx2 = router.register(2);
    assert_eq!(router.agent_count(), 2);

    router.unregister(1);
    assert_eq!(router.agent_count(), 1);
}

#[tokio::test]
async fn test_router_send_receive() {
    let router = MessageRouter::new(8);
    let mut rx = router.register(42);

    let msg = AgentMessage { from: 0, to: 42, content: "hello agent".into() };
    router.send(msg).await.expect("send ok");

    let received = rx.recv().await.expect("recv ok");
    assert_eq!(received.from, 0);
    assert_eq!(received.to, 42);
    assert_eq!(received.content, "hello agent");
}

#[tokio::test]
async fn test_router_send_to_unknown() {
    let router = MessageRouter::new(8);
    let msg = AgentMessage { from: 0, to: 99, content: "nobody home".into() };
    let result = router.send(msg).await;
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("not registered"));
}

#[tokio::test]
async fn test_router_multiple_messages() {
    let router = MessageRouter::new(8);
    let mut rx = router.register(1);

    for i in 0..5 {
        let msg = AgentMessage { from: 0, to: 1, content: format!("msg-{i}") };
        router.send(msg).await.expect("send");
    }

    for i in 0..5 {
        let received = rx.recv().await.expect("recv");
        assert_eq!(received.content, format!("msg-{i}"));
    }
}

#[tokio::test]
async fn test_router_cross_agent() {
    let router = MessageRouter::new(8);
    let mut rx1 = router.register(1);
    let mut rx2 = router.register(2);

    // Agent 1 → Agent 2
    router.send(AgentMessage { from: 1, to: 2, content: "from-1".into() }).await.expect("send");

    // Agent 2 → Agent 1
    router.send(AgentMessage { from: 2, to: 1, content: "from-2".into() }).await.expect("send");

    let m1 = rx1.recv().await.expect("recv");
    assert_eq!(m1.content, "from-2");
    assert_eq!(m1.from, 2);

    let m2 = rx2.recv().await.expect("recv");
    assert_eq!(m2.content, "from-1");
    assert_eq!(m2.from, 1);
}

#[test]
fn test_pool_has_router() {
    let driver = mock_driver("x", 1);
    let pool = AgentPool::new(driver, 4);
    assert_eq!(pool.router().agent_count(), 0);
}

#[tokio::test]
async fn test_pool_registers_agents_in_router() {
    let driver = mock_driver("done", 2);
    let mut pool = AgentPool::new(driver, 4);

    pool.spawn(SpawnConfig { manifest: test_manifest("r1"), query: "q".into() }).expect("spawn");

    // Agent is registered immediately
    assert_eq!(pool.router().agent_count(), 1);

    pool.spawn(SpawnConfig { manifest: test_manifest("r2"), query: "q".into() }).expect("spawn");
    assert_eq!(pool.router().agent_count(), 2);

    // After join, agents unregister themselves
    let _ = pool.join_all().await;
    // Give a brief moment for cleanup tasks
    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    assert_eq!(pool.router().agent_count(), 0);
}

#[tokio::test]
async fn test_pool_with_tool_builder() {
    use super::ToolBuilder;
    use std::sync::atomic::{AtomicU32, Ordering};

    let call_count = Arc::new(AtomicU32::new(0));
    let cc = Arc::clone(&call_count);

    let builder: ToolBuilder = Arc::new(move |_manifest| {
        cc.fetch_add(1, Ordering::SeqCst);
        ToolRegistry::new()
    });

    let driver = mock_driver("built", 1);
    let mut pool = AgentPool::new(driver, 4).with_tool_builder(builder);

    pool.spawn(SpawnConfig { manifest: test_manifest("tb"), query: "test".into() }).expect("spawn");

    let results = pool.join_all().await;
    assert_eq!(results.len(), 1);
    assert_eq!(call_count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn test_pool_with_stream_events() {
    let driver = mock_driver("streamed", 1);
    let (tx, mut rx) = mpsc::channel(64);
    let mut pool = AgentPool::new(driver, 4).with_stream(tx);

    pool.spawn(SpawnConfig { manifest: test_manifest("stream-agent"), query: "hi".into() })
        .expect("spawn");

    let results = pool.join_all().await;
    assert_eq!(results.len(), 1);

    // Should have received stream events
    let mut event_count = 0;
    while rx.try_recv().is_ok() {
        event_count += 1;
    }
    assert!(event_count > 0, "should emit stream events");
}

/// BH-MUT-0002: Zero-capacity pool blocks all spawns.
#[tokio::test]
async fn test_pool_zero_capacity_blocks_all() {
    let driver = mock_driver("nope", 1);
    let mut pool = AgentPool::new(driver, 0);

    let err = pool
        .spawn(SpawnConfig { manifest: test_manifest("blocked"), query: "q".into() })
        .unwrap_err();

    assert!(matches!(err, AgentError::CircuitBreak(_)), "zero-capacity pool must block: {err}");
}

/// BH-MUT-0002: Pool at exact capacity N blocks spawn N+1.
#[tokio::test]
async fn test_pool_exact_capacity_boundary() {
    let driver = mock_driver("ok", 3);
    let mut pool = AgentPool::new(driver, 2);

    // Fill to capacity
    pool.spawn(SpawnConfig { manifest: test_manifest("a1"), query: "q1".into() })
        .expect("spawn 1/2");

    pool.spawn(SpawnConfig { manifest: test_manifest("a2"), query: "q2".into() })
        .expect("spawn 2/2");

    assert_eq!(pool.active_count(), 2);

    // Spawn 3 must fail at capacity=2
    let err =
        pool.spawn(SpawnConfig { manifest: test_manifest("a3"), query: "q3".into() }).unwrap_err();

    assert!(matches!(err, AgentError::CircuitBreak(_)));
}
