//! Agent Pool Multi-Agent Demo
//!
//! Demonstrates the AgentPool orchestration:
//! - Create pool with bounded concurrency
//! - Spawn individual agents
//! - Fan-out: spawn multiple agents concurrently
//! - Fan-in: collect all results with join_all
//! - Capacity enforcement (Muda: bounded resources)
//! - Message routing between agents
//!
//! Run with: `cargo run --example agent_pool --features agents`

#[cfg(feature = "agents")]
#[tokio::main]
async fn main() {
    use std::sync::Arc;

    use batuta::agent::driver::mock::MockDriver;
    use batuta::agent::driver::{CompletionResponse, LlmDriver};
    use batuta::agent::pool::{AgentMessage, AgentPool, SpawnConfig};
    use batuta::agent::result::StopReason;

    println!("Agent Pool Multi-Agent Demo");
    println!("==========================");
    println!();

    // Create mock driver that returns different responses
    let driver: Arc<dyn LlmDriver> = Arc::new(MockDriver::new(vec![
        CompletionResponse {
            text: "Agent 1: SIMD analysis complete".into(),
            stop_reason: StopReason::EndTurn,
            tool_calls: vec![],
            usage: Default::default(),
        },
        CompletionResponse {
            text: "Agent 2: Memory profiling done".into(),
            stop_reason: StopReason::EndTurn,
            tool_calls: vec![],
            usage: Default::default(),
        },
        CompletionResponse {
            text: "Agent 3: Code review finished".into(),
            stop_reason: StopReason::EndTurn,
            tool_calls: vec![],
            usage: Default::default(),
        },
        CompletionResponse {
            text: "Agent 4: Documentation updated".into(),
            stop_reason: StopReason::EndTurn,
            tool_calls: vec![],
            usage: Default::default(),
        },
    ]));

    // Pool with max 3 concurrent agents
    let mut pool = AgentPool::new(Arc::clone(&driver), 3);

    // --- Spawn individual agents ---
    println!("--- Spawn Individual Agents ---");
    let id1 = pool
        .spawn(SpawnConfig {
            manifest: make_manifest("analyzer"),
            query: "Analyze SIMD performance".into(),
        })
        .expect("spawn failed");
    println!("  Spawned agent {id1} (analyzer)");

    let id2 = pool
        .spawn(SpawnConfig {
            manifest: make_manifest("profiler"),
            query: "Profile memory usage".into(),
        })
        .expect("spawn failed");
    println!("  Spawned agent {id2} (profiler)");
    println!("  Active: {}/{}", pool.active_count(), pool.max_concurrent());
    println!();

    // --- Message routing ---
    println!("--- Message Router ---");
    println!("  Registered agents: {}", pool.router().agent_count());
    let msg = AgentMessage { from: 0, to: id1, content: "Prioritize AVX-512 analysis".into() };
    match pool.router().send(msg).await {
        Ok(()) => println!("  Sent message to agent {id1}"),
        Err(e) => println!("  Send failed: {e}"),
    }
    println!();

    // --- Capacity enforcement (Muda) ---
    println!("--- Capacity Enforcement ---");
    let id3 = pool
        .spawn(SpawnConfig {
            manifest: make_manifest("reviewer"),
            query: "Review code quality".into(),
        })
        .expect("spawn failed");
    println!("  Spawned agent {id3} (reviewer) — at capacity");

    // This should fail: pool is at max_concurrent (3)
    let over_capacity =
        pool.spawn(SpawnConfig { manifest: make_manifest("excess"), query: "Should fail".into() });
    match over_capacity {
        Ok(_) => println!("  ERROR: Should have been rejected!"),
        Err(e) => println!("  Correctly rejected: {e}"),
    }
    println!();

    // --- Fan-in: collect results ---
    println!("--- Join All (Fan-In) ---");
    let results = pool.join_all().await;
    for (id, result) in &results {
        match result {
            Ok(r) => println!("  Agent {id}: {} ({} iterations)", r.text, r.iterations,),
            Err(e) => println!("  Agent {id}: ERROR — {e}"),
        }
    }
    println!("  Collected {} results", results.len());
    println!();

    // --- Fan-out demo ---
    println!("--- Fan-Out Demo ---");

    let driver2: Arc<dyn LlmDriver> = Arc::new(MockDriver::new(vec![
        CompletionResponse {
            text: "Batch agent A done".into(),
            stop_reason: StopReason::EndTurn,
            tool_calls: vec![],
            usage: Default::default(),
        },
        CompletionResponse {
            text: "Batch agent B done".into(),
            stop_reason: StopReason::EndTurn,
            tool_calls: vec![],
            usage: Default::default(),
        },
    ]));

    let mut pool2 = AgentPool::new(driver2, 4);

    let configs = vec![
        SpawnConfig { manifest: make_manifest("batch-a"), query: "Task A".into() },
        SpawnConfig { manifest: make_manifest("batch-b"), query: "Task B".into() },
    ];

    let ids = pool2.fan_out(configs).expect("fan_out failed");
    println!("  Fan-out spawned {} agents: {:?}", ids.len(), ids);

    let results = pool2.join_all().await;
    for (id, result) in &results {
        match result {
            Ok(r) => println!("  Agent {id}: {}", r.text),
            Err(e) => println!("  Agent {id}: ERROR — {e}"),
        }
    }
    println!();

    println!("All pool operations completed.");
}

#[cfg(feature = "agents")]
fn make_manifest(name: &str) -> batuta::agent::AgentManifest {
    let mut m = batuta::agent::AgentManifest::default();
    m.name = name.to_string();
    m
}

#[cfg(not(feature = "agents"))]
fn main() {
    eprintln!(
        "Enable `agents` feature: \
         cargo run --example agent_pool --features agents"
    );
    std::process::exit(1);
}
