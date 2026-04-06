#![allow(clippy::all, clippy::pedantic, clippy::nursery)]
//! Hello Repartir Example
//!
//! Demonstrates basic usage of the repartir distributed computing library.

use repartir::task::{Backend, Task};
use repartir::Pool;

#[tokio::main]
async fn main() -> repartir::error::Result<()> {
    // Initialize tracing for logs
    tracing_subscriber::fmt::init();

    println!("🚀 Repartir v1.0 - Sovereign AI Distributed Computing");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // Create a pool with 4 CPU workers
    let pool = Pool::builder().cpu_workers(4).max_queue_size(100).build()?;

    println!("✓ Pool initialized with {} workers\n", pool.capacity());

    // Example 1: Simple echo command
    #[cfg(unix)]
    {
        println!("Example 1: Running echo command");
        let task = Task::builder()
            .binary("/bin/echo")
            .arg("Hello from Repartir!")
            .backend(Backend::Cpu)
            .build()?;

        let result = pool.submit(task).await?;

        if result.is_success() {
            println!("✓ Output: {}", result.stdout_str()?.trim());
        } else {
            println!("✗ Task failed with exit code: {}", result.exit_code());
        }
    }

    println!();

    // Example 2: Running a shell command
    #[cfg(unix)]
    {
        println!("Example 2: Running shell command");
        let task = Task::builder()
            .binary("/bin/sh")
            .arg("-c")
            .arg("echo 'Pure Rust' && echo 'Zero C/C++' && echo 'Iron Lotus Quality'")
            .backend(Backend::Cpu)
            .build()?;

        let result = pool.submit(task).await?;

        if result.is_success() {
            println!("✓ Output:");
            for line in result.stdout_str()?.lines() {
                println!("  {}", line);
            }
        }
    }

    println!();

    // Example 3: Environment variables
    #[cfg(unix)]
    {
        println!("Example 3: Using environment variables");
        let task = Task::builder()
            .binary("/usr/bin/env")
            .env_var("REPARTIR_MODE", "sovereign")
            .env_var("QUALITY_LEVEL", "iron-lotus")
            .backend(Backend::Cpu)
            .build()?;

        let result = pool.submit(task).await?;

        if result.is_success() {
            let output = result.stdout_str()?;
            for line in output.lines() {
                if line.contains("REPARTIR") || line.contains("QUALITY") {
                    println!("  ✓ {}", line);
                }
            }
        }
    }

    println!();

    // Example 4: Error handling
    #[cfg(unix)]
    {
        println!("Example 4: Handling command failure");
        let task = Task::builder()
            .binary("/bin/sh")
            .arg("-c")
            .arg("exit 42") // Deliberately fail
            .backend(Backend::Cpu)
            .build()?;

        let result = pool.submit(task).await?;

        if !result.is_success() {
            println!("✓ Caught expected failure:");
            println!("  Exit code: {}", result.exit_code());
        }
    }

    println!();

    // Example 5: Parallel execution
    #[cfg(unix)]
    {
        println!("Example 5: Parallel task execution");
        use std::time::Instant;

        let start = Instant::now();

        // Submit 10 tasks that each sleep for 100ms
        let mut results = Vec::new();
        for i in 0..10 {
            let task = Task::builder()
                .binary("/bin/sh")
                .arg("-c")
                .arg(format!("sleep 0.1 && echo 'Task {}'", i))
                .backend(Backend::Cpu)
                .build()?;

            results.push(pool.submit(task));
        }

        // Wait for all tasks
        let outputs = futures::future::join_all(results).await;

        let elapsed = start.elapsed();

        println!("  ✓ Completed 10 tasks in {:?}", elapsed);
        println!("  ✓ Parallel speedup achieved!");

        // Verify all succeeded
        let successes = outputs
            .iter()
            .filter(|r| r.as_ref().map(|res| res.is_success()).unwrap_or(false))
            .count();
        println!("  ✓ {} / 10 tasks succeeded", successes);
    }

    println!();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("All examples completed! Shutting down pool...");

    // Graceful shutdown
    pool.shutdown().await;

    println!("✓ Pool shutdown complete");

    Ok(())
}
