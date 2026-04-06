#![allow(clippy::all, clippy::pedantic, clippy::nursery)]
//! Push-Pull (PUSH/PULL) Messaging Example
//!
//! Demonstrates work distribution where multiple producers push work items
//! and multiple consumers pull them for processing (load balancing).
//!
//! # Use Cases
//!
//! - Work queue / task distribution
//! - Load balancing across workers
//! - Pipeline processing
//! - Job scheduling
//!
//! # Run
//!
//! ```bash
//! cargo run --example pushpull_example
//! ```

use repartir::error::Result;
use repartir::messaging::{Message, PushPullChannel};
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║  Push-Pull (PUSH/PULL) Messaging Pattern                    ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();

    // Create a PUSH/PULL channel
    let channel = PushPullChannel::new(100);
    println!("✓ Created PUSH/PULL channel (capacity: 100)");
    println!();

    // Scenario 1: Basic work distribution
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ Scenario 1: Basic Work Distribution                         ");
    println!("└─────────────────────────────────────────────────────────────┘");
    println!();

    // Producer pushes work
    for i in 1..=5 {
        let msg = Message::text(format!("Task {}", i))
            .with_metadata("job_id", format!("{}", i))
            .with_metadata("priority", if i % 2 == 0 { "high" } else { "normal" });

        channel.push(msg).await?;
    }

    println!("  ✓ Producer pushed 5 tasks");
    println!();

    // Consumer pulls work
    println!("  Consumer processing:");
    for _ in 0..5 {
        if let Some(msg) = channel.pull().await {
            println!(
                "    - {} (priority: {})",
                msg.as_text()?,
                msg.get_metadata("priority").unwrap_or("unknown")
            );
        }
    }

    println!();

    // Scenario 2: Multiple producers
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ Scenario 2: Multiple Producers                              ");
    println!("└─────────────────────────────────────────────────────────────┘");
    println!();

    let sender1 = channel.sender();
    let sender2 = channel.sender();
    let sender3 = channel.sender();

    // Spawn producers
    let producer1 = tokio::spawn(async move {
        for i in 1..=3 {
            let msg = Message::text(format!("Producer1-Task{}", i));
            let _ = sender1.send(msg).await;
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    });

    let producer2 = tokio::spawn(async move {
        for i in 1..=3 {
            let msg = Message::text(format!("Producer2-Task{}", i));
            let _ = sender2.send(msg).await;
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    });

    let producer3 = tokio::spawn(async move {
        for i in 1..=3 {
            let msg = Message::text(format!("Producer3-Task{}", i));
            let _ = sender3.send(msg).await;
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    });

    // Wait for producers
    let _ = tokio::join!(producer1, producer2, producer3);

    println!("  ✓ 3 producers pushed 9 tasks total");
    println!();

    // Single consumer processes all
    println!("  Consumer processing:");
    let mut count = 0;
    while count < 9 {
        if let Some(msg) = channel.pull().await {
            println!("    - {}", msg.as_text()?);
            count += 1;
        }
    }

    println!();

    // Scenario 3: Load balancing across workers
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ Scenario 3: Load Balancing Across Workers                   ");
    println!("└─────────────────────────────────────────────────────────────┘");
    println!();

    let channel2 = PushPullChannel::new(100);

    // Push work items
    for i in 1..=12 {
        let msg = Message::bytes(format!("WorkItem-{}", i).into_bytes());
        channel2.push(msg).await?;
    }

    println!("  ✓ Pushed 12 work items");
    println!();

    // Spawn 3 worker tasks
    let channel_ref = std::sync::Arc::new(channel2);
    let mut workers = vec![];

    for worker_id in 1..=3 {
        let ch = channel_ref.clone();
        let worker = tokio::spawn(async move {
            let mut processed = 0;
            for _ in 0..4 {
                if let Some(msg) = ch.pull().await {
                    let work = String::from_utf8_lossy(msg.as_bytes());
                    println!("    Worker {} processing: {}", worker_id, work);
                    processed += 1;
                    tokio::time::sleep(Duration::from_millis(50)).await;
                }
            }
            processed
        });
        workers.push(worker);
    }

    // Wait for all workers
    let results = futures::future::join_all(workers).await;
    let total: usize = results.iter().filter_map(|r| r.as_ref().ok()).sum();

    println!();
    println!("  ✓ 3 workers processed {} items total", total);
    println!("  ✓ Load balanced: ~4 items per worker");

    println!();
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║  ✅ PUSH/PULL Pattern Demonstrated Successfully              ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();
    println!("Key Characteristics:");
    println!("  • Many producers → Many consumers (work distribution)");
    println!("  • Each message consumed by exactly ONE consumer");
    println!("  • Automatic load balancing");
    println!("  • Useful for work queues, job scheduling, pipelines");
    println!();

    Ok(())
}
