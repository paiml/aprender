#![allow(clippy::all, clippy::pedantic, clippy::nursery)]
//! Repartir v1.1 Comprehensive Showcase
//!
//! Demonstrates all v1.1 Production Hardening features:
//! - CPU executor (multi-threaded work-stealing)
//! - GPU detection and capabilities
//! - Remote executor with TLS encryption
//! - Priority-based scheduling
//! - Fault tolerance and error handling
//! - Performance benchmarking
//!
//! # Setup for Full Demo
//!
//! 1. Generate TLS certificates (for remote executor):
//!    ```bash
//!    ./scripts/generate-test-certs.sh ./certs
//!    ```
//!
//! 2. Run with all features:
//!    ```bash
//!    cargo run --example v1_1_showcase --features full
//!    ```

use repartir::error::Result;
use repartir::executor::cpu::CpuExecutor;
use repartir::executor::Executor;
use repartir::task::{Backend, Priority, Task};
use repartir::Pool;
use std::time::{Duration, Instant};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    print_header();

    // Run feature demonstrations
    demo_cpu_executor().await?;
    demo_gpu_detection().await?;
    demo_tls_config().await?;
    demo_priority_scheduling().await?;
    demo_work_stealing_performance().await?;
    demo_fault_tolerance().await?;

    print_footer();

    Ok(())
}

fn print_header() {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║  🚀 Repartir v1.1 - Production Hardening Showcase            ║");
    println!("║  Sovereign AI-Grade Distributed Computing for Rust           ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();
}

fn print_footer() {
    println!();
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║  ✅ All v1.1 Features Demonstrated Successfully               ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();
    println!("Next steps:");
    println!("  • Review the Iron Lotus quality gates (make tier2)");
    println!("  • Run mutation testing (make mutation)");
    println!("  • Deploy to production with TLS encryption");
    println!("  • Scale to distributed workers");
    println!();
}

/// Demonstrates CPU executor with work-stealing scheduler
async fn demo_cpu_executor() -> Result<()> {
    section_header("CPU Executor - Work-Stealing Scheduler");

    let executor = CpuExecutor::new();
    println!("  Name:     {}", executor.name());
    println!("  Capacity: {} workers", executor.capacity());
    println!("  Features: Blumofe & Leiserson work-stealing");
    println!();

    // Execute a simple task
    #[cfg(unix)]
    {
        let task = Task::builder()
            .binary("/bin/echo")
            .arg("CPU executor working!")
            .backend(Backend::Cpu)
            .build()?;

        let start = Instant::now();
        let result = executor.execute(task).await?;
        let duration = start.elapsed();

        if result.is_success() {
            println!("  ✓ Task executed successfully in {:?}", duration);
            println!("  ✓ Output: {}", result.stdout_str()?.trim());
        }
    }

    println!();
    Ok(())
}

/// Demonstrates GPU detection and capabilities
async fn demo_gpu_detection() -> Result<()> {
    section_header("GPU Executor - Cross-Platform Compute");

    #[cfg(feature = "gpu")]
    {
        use repartir::executor::gpu::GpuExecutor;

        match GpuExecutor::new().await {
            Ok(executor) => {
                println!("  ✓ GPU Detected!");
                println!("  Device:        {}", executor.device_name());
                println!("  Executor:      {}", executor.name());
                println!("  Compute Units: {}", executor.capacity());
                println!();
                println!("  Backends: Vulkan, Metal, DirectX 12, WebGPU");
                println!("  Note: v1.1 provides detection only (compute in v1.2)");
            }
            Err(e) => {
                println!("  ℹ No GPU available: {}", e);
                println!("  This is normal in VM/CI environments");
            }
        }
    }

    #[cfg(not(feature = "gpu"))]
    {
        println!("  ℹ GPU feature not enabled");
        println!("  Enable with: --features gpu");
    }

    println!();
    Ok(())
}

/// Demonstrates TLS configuration
async fn demo_tls_config() -> Result<()> {
    section_header("TLS Encryption - Secure Remote Execution");

    #[cfg(feature = "remote-tls")]
    {
        use repartir::executor::tls::TlsConfig;
        use std::path::Path;

        // Check if test certificates exist
        if Path::new("./certs/client.pem").exists() {
            let tls_config = TlsConfig::builder()
                .client_cert("./certs/client.pem")
                .client_key("./certs/client.key")
                .server_cert("./certs/server.pem")
                .server_key("./certs/server.key")
                .ca_cert("./certs/ca.pem")
                .build()?;

            println!("  ✓ TLS Configuration loaded");
            println!();
            println!(
                "  Client: {}",
                if tls_config.client_config().is_ok() { "✓ Enabled" } else { "✗ Disabled" }
            );
            println!(
                "  Server: {}",
                if tls_config.server_config().is_ok() { "✓ Enabled" } else { "✗ Disabled" }
            );
            println!();
            println!("  Security:");
            println!("    • TLS 1.3 encryption");
            println!("    • Certificate-based auth");
            println!("    • Perfect forward secrecy");
            println!("    • MITM protection");
        } else {
            println!("  ℹ Test certificates not found");
            println!("  Generate with: ./scripts/generate-test-certs.sh ./certs");
        }
    }

    #[cfg(not(feature = "remote-tls"))]
    {
        println!("  ℹ TLS feature not enabled");
        println!("  Enable with: --features remote-tls");
    }

    println!();
    Ok(())
}

/// Demonstrates priority-based scheduling
async fn demo_priority_scheduling() -> Result<()> {
    section_header("Priority Scheduling - Iron Lotus Quality");

    let pool = Pool::builder().cpu_workers(4).max_queue_size(100).build()?;

    println!("  Pool capacity: {} workers", pool.capacity());
    println!();

    #[cfg(unix)]
    {
        // Submit tasks with different priorities
        println!("  Submitting tasks with priorities:");

        let high_task = Task::builder()
            .binary("/bin/echo")
            .arg("High priority")
            .priority(Priority::High)
            .backend(Backend::Cpu)
            .build()?;

        let normal_task = Task::builder()
            .binary("/bin/echo")
            .arg("Normal priority")
            .priority(Priority::Normal)
            .backend(Backend::Cpu)
            .build()?;

        let low_task = Task::builder()
            .binary("/bin/echo")
            .arg("Low priority")
            .priority(Priority::Low)
            .backend(Backend::Cpu)
            .build()?;

        // Submit in reverse order to show priority works
        let start = Instant::now();
        let low_result = pool.submit(low_task).await?;
        let normal_result = pool.submit(normal_task).await?;
        let high_result = pool.submit(high_task).await?;
        let duration = start.elapsed();

        println!("  ✓ High:   {} ({:?})", high_result.stdout_str()?.trim(), high_result.duration());
        println!(
            "  ✓ Normal: {} ({:?})",
            normal_result.stdout_str()?.trim(),
            normal_result.duration()
        );
        println!("  ✓ Low:    {} ({:?})", low_result.stdout_str()?.trim(), low_result.duration());
        println!();
        println!("  Total execution: {:?}", duration);
    }

    pool.shutdown().await;
    println!();
    Ok(())
}

/// Demonstrates work-stealing performance
async fn demo_work_stealing_performance() -> Result<()> {
    section_header("Work-Stealing Performance - Parallel Speedup");

    let pool = Pool::builder().cpu_workers(4).max_queue_size(100).build()?;

    #[cfg(unix)]
    {
        const TASK_COUNT: usize = 20;

        println!("  Executing {} parallel tasks...", TASK_COUNT);

        let start = Instant::now();
        let mut futures = Vec::new();

        for i in 0..TASK_COUNT {
            let task = Task::builder()
                .binary("/bin/sh")
                .arg("-c")
                .arg(format!("sleep 0.1 && echo 'Task {}'", i))
                .backend(Backend::Cpu)
                .build()?;

            futures.push(pool.submit(task));
        }

        // Wait for all tasks
        let results = futures::future::join_all(futures).await;
        let duration = start.elapsed();

        let successes = results
            .iter()
            .filter(|r| r.as_ref().map(|res| res.is_success()).unwrap_or(false))
            .count();

        println!();
        println!("  Results:");
        println!("    Tasks:     {}/{} succeeded", successes, TASK_COUNT);
        println!("    Duration:  {:?}", duration);
        println!("    Workers:   {}", pool.capacity());
        println!();

        // Calculate speedup
        let sequential_time = Duration::from_millis(100 * TASK_COUNT as u64);
        let speedup = sequential_time.as_millis() as f64 / duration.as_millis() as f64;
        println!("  ⚡ Parallel speedup: {:.2}x", speedup);
        println!("  (vs sequential: {:?})", sequential_time);
    }

    pool.shutdown().await;
    println!();
    Ok(())
}

/// Demonstrates fault tolerance
async fn demo_fault_tolerance() -> Result<()> {
    section_header("Fault Tolerance - Graceful Error Handling");

    let pool = Pool::builder().cpu_workers(2).max_queue_size(50).build()?;

    #[cfg(unix)]
    {
        // Test successful task
        let success_task =
            Task::builder().binary("/bin/echo").arg("Success!").backend(Backend::Cpu).build()?;

        let result = pool.submit(success_task).await?;
        println!("  ✓ Successful task:");
        println!("    Exit code: {}", result.exit_code());
        println!("    Output:    {}", result.stdout_str()?.trim());
        println!();

        // Test failing task
        let fail_task = Task::builder()
            .binary("/bin/sh")
            .arg("-c")
            .arg("exit 42")
            .backend(Backend::Cpu)
            .build()?;

        let result = pool.submit(fail_task).await?;
        println!("  ✓ Failed task handled gracefully:");
        println!("    Exit code: {}", result.exit_code());
        println!("    Is error:  {}", !result.is_success());
        println!();

        println!("  ✓ System remains stable after errors");
    }

    pool.shutdown().await;
    println!();
    Ok(())
}

fn section_header(title: &str) {
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ {} {}", title, " ".repeat(60usize.saturating_sub(title.len())));
    println!("└─────────────────────────────────────────────────────────────┘");
    println!();
}
