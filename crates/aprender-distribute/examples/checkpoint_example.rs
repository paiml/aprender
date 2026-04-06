//! Checkpoint example demonstrating task state persistence (v2.0).
//!
//! This example shows how to:
//! 1. Enable checkpointing for a task
//! 2. Set checkpoint intervals
//! 3. Restore from checkpoints on failure
//!
//! Run with:
//! ```bash
//! cargo run --example checkpoint_example --features checkpoint
//! ```

use repartir::checkpoint::{CheckpointManager, TaskState};
use repartir::task::{Backend, Task};
use repartir::Pool;
use std::path::PathBuf;
use std::time::{Duration, SystemTime};

#[tokio::main]
async fn main() -> repartir::error::Result<()> {
    println!("=== Repartir v2.0: Checkpoint Example ===\n");

    // Create checkpoint directory
    let checkpoint_dir = PathBuf::from("./tmp/checkpoints");
    std::fs::create_dir_all(&checkpoint_dir)?;

    // Initialize checkpoint manager
    let manager = CheckpointManager::new(checkpoint_dir)?;

    println!("1. Creating a task with checkpointing enabled...");

    // Create a task
    let task = Task::builder().binary("/bin/sleep").arg("1").backend(Backend::Cpu).build()?;

    println!("   ✓ Task created with ID: {}", task.id());

    // Simulate task execution and checkpointing
    println!("2. Simulating task execution with periodic checkpoints...");

    let task_id = *task.id().as_uuid();

    // Checkpoint at iteration 10
    let state1 = TaskState {
        task_id,
        iteration: 10,
        data: b"State at iteration 10".to_vec(),
        timestamp: SystemTime::now(),
    };
    let checkpoint1_id = manager.checkpoint(task_id, &state1).await?;
    println!("   ✓ Checkpoint 1 created: {checkpoint1_id}");

    tokio::time::sleep(Duration::from_millis(100)).await;

    // Checkpoint at iteration 20
    let state2 = TaskState {
        task_id,
        iteration: 20,
        data: b"State at iteration 20".to_vec(),
        timestamp: SystemTime::now(),
    };
    let checkpoint2_id = manager.checkpoint(task_id, &state2).await?;
    println!("   ✓ Checkpoint 2 created: {checkpoint2_id}\n");

    // List all checkpoints
    println!("3. Listing all checkpoints...");
    let checkpoints = manager.list_checkpoints(task_id).await?;
    println!("   Found {} checkpoints:", checkpoints.len());
    for (i, checkpoint) in checkpoints.iter().enumerate() {
        println!("   [{} Checkpoint {}", i + 1, checkpoint.checkpoint_id);
        println!("      Iteration: {}", checkpoint.iteration);
        println!("      Size: {} bytes", checkpoint.size_bytes);
        println!("      Created: {:?}", checkpoint.created_at);
    }
    println!();

    // Restore from checkpoint
    println!("4. Simulating task failure and restore...");
    let restored = manager.restore(task_id).await?;

    match restored {
        Some(state) => {
            println!("   ✓ Restored from checkpoint:");
            println!("     Task ID: {}", state.task_id);
            println!("     Iteration: {}", state.iteration);
            println!("     Data: {}", String::from_utf8_lossy(&state.data));
            println!("     Timestamp: {:?}\n", state.timestamp);
        }
        None => {
            println!("   ✗ No checkpoint found\n");
        }
    }

    // Demonstrate data dependencies
    println!("5. Creating task with data dependencies (for locality)...");
    let task_with_locality = Task::builder()
        .binary("/bin/echo")
        .arg("Processing data")
        .backend(Backend::Cpu)
        .data_dependency("tensor_batch_42")
        .data_dependency("checkpoint_123")
        .data_dependency("model_weights_v2")
        .build()?;

    println!("   ✓ Task created with data dependencies:");
    for dep in task_with_locality.data_dependencies() {
        println!("     - {dep}");
    }
    println!();

    // Execute a simple task
    println!("6. Executing task...");
    let pool = Pool::builder().cpu_workers(2).build()?;

    let simple_task = Task::builder()
        .binary("/bin/echo")
        .arg("Hello from Repartir v2.0!")
        .backend(Backend::Cpu)
        .build()?;

    let result = pool.submit(simple_task).await?;

    if result.is_success() {
        println!("   ✓ Task completed successfully");
        println!("   ✓ Output: {}\n", result.stdout_str()?.trim());
    }

    pool.shutdown().await;

    // Cleanup demonstration
    println!("7. Checkpoint cleanup (retention policy)...");
    println!("   Keeping checkpoints newer than 30 days...");
    let deleted = manager.cleanup(30).await?;
    println!("   ✓ Deleted {deleted} old checkpoints\n");

    println!("=== Example Complete ===");
    println!("\nKey Takeaways:");
    println!("  • Checkpointing enables fault-tolerant long-running tasks");
    println!("  • Checkpoints are stored persistently and survive restarts");
    println!("  • Data dependencies enable locality-aware scheduling");
    println!("  • Automatic cleanup manages storage with retention policies");

    // Cleanup temp directory
    std::fs::remove_dir_all("./tmp/checkpoints").ok();

    Ok(())
}
