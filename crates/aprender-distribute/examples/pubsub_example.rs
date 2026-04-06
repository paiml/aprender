#![allow(clippy::all, clippy::pedantic, clippy::nursery)]
//! Publish-Subscribe (PUB/SUB) Messaging Example
//!
//! Demonstrates one-to-many broadcast messaging where multiple subscribers
//! receive copies of published messages.
//!
//! # Use Cases
//!
//! - Event notifications
//! - Logging and monitoring
//! - Real-time updates
//! - Distributed state synchronization
//!
//! # Run
//!
//! ```bash
//! cargo run --example pubsub_example
//! ```

use repartir::error::Result;
use repartir::messaging::{Message, PubSubChannel};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║  Publish-Subscribe (PUB/SUB) Messaging Pattern              ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();

    // Create a PUB/SUB channel
    let channel = PubSubChannel::new();
    println!("✓ Created PUB/SUB channel");
    println!();

    // Scenario 1: Multiple subscribers on same topic
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ Scenario 1: Multiple Subscribers (Broadcast)                ");
    println!("└─────────────────────────────────────────────────────────────┘");
    println!();

    let mut worker1 = channel.subscribe("tasks").await;
    let mut worker2 = channel.subscribe("tasks").await;
    let mut worker3 = channel.subscribe("tasks").await;

    println!("  Subscribers: {}", channel.subscriber_count("tasks").await);
    println!();

    // Publish a task notification
    let msg = Message::text("New task available: compute-pi")
        .with_metadata("priority", "high")
        .with_metadata("timestamp", "2025-11-22T15:30:00Z");

    let count = channel.publish("tasks", msg).await?;
    println!("  ✓ Published to {} subscribers", count);
    println!();

    // All subscribers receive the message
    if let Ok(m) = worker1.try_recv() {
        println!("  Worker 1 received: {}", m.as_text()?);
        println!("    Priority: {}", m.get_metadata("priority").unwrap_or("none"));
    }

    if let Ok(m) = worker2.try_recv() {
        println!("  Worker 2 received: {}", m.as_text()?);
        println!("    Priority: {}", m.get_metadata("priority").unwrap_or("none"));
    }

    if let Ok(m) = worker3.try_recv() {
        println!("  Worker 3 received: {}", m.as_text()?);
        println!("    Priority: {}", m.get_metadata("priority").unwrap_or("none"));
    }

    println!();

    // Scenario 2: Topic isolation
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ Scenario 2: Topic Isolation                                 ");
    println!("└─────────────────────────────────────────────────────────────┘");
    println!();

    let mut events_sub = channel.subscribe("events").await;
    let mut alerts_sub = channel.subscribe("alerts").await;

    println!("  Topics: {}", channel.topic_count().await);
    println!();

    // Publish to different topics
    channel.publish("events", Message::text("Task completed: job-123")).await?;
    channel.publish("alerts", Message::text("CPU usage above 80%")).await?;

    // Each subscriber only receives from their topic
    if let Ok(m) = events_sub.try_recv() {
        println!("  Events subscriber: {}", m.as_text()?);
    }

    if let Ok(m) = alerts_sub.try_recv() {
        println!("  Alerts subscriber: {}", m.as_text()?);
    }

    println!();

    // Scenario 3: Real-time monitoring
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ Scenario 3: Real-Time Monitoring                            ");
    println!("└─────────────────────────────────────────────────────────────┘");
    println!();

    let mut monitor1 = channel.subscribe("metrics").await;
    let mut monitor2 = channel.subscribe("metrics").await;

    // Simulate publishing metrics
    for i in 1..=3 {
        let msg = Message::text(format!("CPU: {}%", 50 + i * 10))
            .with_metadata("source", "worker-1")
            .with_metadata("type", "gauge");

        channel.publish("metrics", msg).await?;
    }

    println!("  ✓ Published 3 metrics");
    println!();

    // Both monitors receive all metrics
    println!("  Monitor 1 received:");
    while let Ok(m) = monitor1.try_recv() {
        println!(
            "    - {} (source: {})",
            m.as_text()?,
            m.get_metadata("source").unwrap_or("unknown")
        );
    }

    println!();
    println!("  Monitor 2 received:");
    while let Ok(m) = monitor2.try_recv() {
        println!(
            "    - {} (source: {})",
            m.as_text()?,
            m.get_metadata("source").unwrap_or("unknown")
        );
    }

    println!();
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║  ✅ PUB/SUB Pattern Demonstrated Successfully                 ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();
    println!("Key Characteristics:");
    println!("  • One publisher → Many subscribers (broadcast)");
    println!("  • Each subscriber receives all messages");
    println!("  • Topic-based isolation");
    println!("  • Useful for events, logging, monitoring");
    println!();

    Ok(())
}
