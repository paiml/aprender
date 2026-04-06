#![allow(
    clippy::unwrap_used,
    clippy::float_cmp,
    dead_code,
    clippy::needless_borrows_for_generic_args,
    clippy::match_single_binding,
    clippy::single_char_pattern
)]
//! Agent Memory Substrate Demo
//!
//! Demonstrates the MemorySubstrate trait with InMemorySubstrate:
//! - Store memories (remember)
//! - Recall by substring query
//! - Filter by source and agent ID
//! - Key-value structured storage (set/get)
//! - Forget (delete) memories
//!
//! Run with: `cargo run --example agent_memory --features agents`

#[cfg(feature = "agents")]
#[tokio::main]
async fn main() {
    use batuta::agent::memory::{InMemorySubstrate, MemoryFilter, MemorySource, MemorySubstrate};

    println!("Agent Memory Substrate Demo");
    println!("==========================");
    println!();

    let memory = InMemorySubstrate::new();
    let agent = "research-agent";

    // Store conversation memories
    println!("--- Storing memories ---");
    let id1 = memory
        .remember(
            agent,
            "SIMD vectorization improves throughput by 4-8x on AVX2",
            MemorySource::Conversation,
            None,
        )
        .await
        .expect("remember failed");
    println!("  Stored [{id1}]: SIMD vectorization...");

    let id2 = memory
        .remember(
            agent,
            "Rust's ownership model prevents data races at compile time",
            MemorySource::Conversation,
            None,
        )
        .await
        .expect("remember failed");
    println!("  Stored [{id2}]: Rust ownership...");

    let id3 = memory
        .remember(
            agent,
            "Q4K quantization reduces model size by 75% with <1% quality loss",
            MemorySource::ToolResult,
            None,
        )
        .await
        .expect("remember failed");
    println!("  Stored [{id3}]: Q4K quantization...");

    memory
        .remember(
            "other-agent",
            "This memory belongs to a different agent",
            MemorySource::System,
            None,
        )
        .await
        .expect("remember failed");
    println!("  Stored memory for other-agent");
    println!();

    // Recall by substring query
    println!("--- Recall: 'SIMD' ---");
    let results = memory.recall("SIMD", 5, None, None).await.expect("recall failed");
    for frag in &results {
        println!("  [{:.2}] {}: {}", frag.relevance_score, frag.id, truncate(&frag.content, 60),);
    }
    println!("  ({} results)", results.len());
    println!();

    // Recall with filter
    println!("--- Recall with ToolResult filter ---");
    let filter = MemoryFilter { source: Some(MemorySource::ToolResult), ..Default::default() };
    let results =
        memory.recall("quantization", 5, Some(filter), None).await.expect("recall failed");
    for frag in &results {
        println!(
            "  [{:.2}] {:?}: {}",
            frag.relevance_score,
            frag.source,
            truncate(&frag.content, 60),
        );
    }
    println!("  ({} tool results)", results.len());
    println!();

    // Key-value storage
    println!("--- Key-Value Storage ---");
    let config = serde_json::json!({
        "model": "llama-3.2-8b",
        "temperature": 0.7,
        "session_count": 42,
    });
    memory.set(agent, "config", config.clone()).await.expect("set failed");
    println!("  Set 'config': {config}");

    let retrieved = memory.get(agent, "config").await.expect("get failed");
    println!("  Get 'config': {}", retrieved.unwrap());

    let missing = memory.get(agent, "nonexistent").await.expect("get failed");
    println!("  Get 'nonexistent': {:?}", missing);
    println!();

    // Forget a memory
    println!("--- Forget ---");
    memory.forget(id1.clone()).await.expect("forget failed");
    println!("  Forgot [{id1}]");

    let after_forget = memory.recall("SIMD", 5, None, None).await.expect("recall failed");
    println!("  Recall 'SIMD' after forget: {} results", after_forget.len());
    println!();

    println!("All memory operations completed.");
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}...", &s[..max])
    }
}

#[cfg(not(feature = "agents"))]
fn main() {
    eprintln!(
        "Enable `agents` feature: \
         cargo run --example agent_memory --features agents"
    );
    std::process::exit(1);
}
