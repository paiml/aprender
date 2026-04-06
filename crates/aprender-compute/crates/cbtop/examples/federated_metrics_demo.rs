#![allow(clippy::disallowed_methods)]
//! Federated Metrics Aggregation Demo
//!
//! Demonstrates CRDT-based multi-host metrics aggregation.
//!
//! Run with: cargo run --example federated_metrics_demo -p cbtop

use cbtop::{FederationConfig, GCounter, LwwRegister, MetricsFederation, OrSet};
use std::time::Duration;

fn main() {
    println!("=== Federated Metrics Aggregation Demo ===\n");

    // Create federation for local host
    let config = FederationConfig {
        host_timeout: Duration::from_secs(60),
        memory_limit_bytes: 10 * 1024 * 1024, // 10MB
        ..Default::default()
    };
    let mut federation = MetricsFederation::new("host-1", config);

    // Add remote hosts to federation
    println!("Setting up 3-node cluster...");
    federation.add_host("host-2");
    federation.add_host("host-3");
    println!("Active hosts: {}\n", federation.active_host_count());

    // Record metrics from local host
    println!("Recording CPU metrics from local host...");
    for i in 0..10 {
        let cpu_usage = 45.0 + (i as f64 * 2.0);
        federation.record("cpu_usage", cpu_usage).unwrap();
    }
    println!("Total samples: {}\n", federation.total_samples());

    // Get aggregated metrics
    println!("Aggregated metrics...");
    if let Some(agg) = federation.get_aggregated("cpu_usage") {
        println!("  Mean: {:.2}%", agg.mean());
        println!("  Min:  {:.2}%", agg.min);
        println!("  Max:  {:.2}%", agg.max);
        println!("  P50:  {:.2}%", agg.p50());
        println!("  P95:  {:.2}%", agg.p95());
        println!("  P99:  {:.2}%", agg.p99());
    }

    // Demonstrate CRDT types
    println!("\n=== CRDT Type Demonstrations ===");

    // G-Counter (grow-only counter)
    println!("\nG-Counter (grow-only counter):");
    let mut counter = GCounter::new();
    counter.increment("node-a", 5);
    counter.increment("node-b", 3);
    counter.increment("node-a", 2); // node-a now has 7
    println!("  After increments: node-a=7, node-b=3");
    println!("  Total value: {}", counter.value());

    // Merge two counters (simulating partition recovery)
    let mut counter2 = GCounter::new();
    counter2.increment("node-c", 10);
    counter.merge(&counter2);
    println!("  After merge with node-c=10: {}", counter.value());

    // LWW-Register (last-writer-wins)
    println!("\nLWW-Register (last-writer-wins):");
    let mut reg = LwwRegister::new("initial_value", 0, "writer1");
    println!("  Initial: {:?} (t=0, by writer1)", reg.value());

    reg.update("updated_value", 5, "writer2");
    println!("  After update: {:?} (t=5, by writer2)", reg.value());

    // Older update ignored
    reg.update("stale_value", 3, "writer1");
    println!("  After stale update (t=3): {:?} (unchanged)", reg.value());

    // OR-Set (observed-remove set)
    println!("\nOR-Set (observed-remove set):");
    let mut set: OrSet<String> = OrSet::new();
    set.add("alice".to_string(), "tag1".to_string());
    set.add("bob".to_string(), "tag2".to_string());
    set.add("charlie".to_string(), "tag3".to_string());
    println!("  Added: alice, bob, charlie");
    println!("  Elements: {:?}", set.elements());

    set.remove(&"bob".to_string());
    println!("  After removing bob: {:?}", set.elements());

    println!("\n✅ Federated metrics demo complete!");
}
