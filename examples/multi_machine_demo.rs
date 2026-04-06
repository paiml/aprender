//! Multi-Machine Distributed Computing Demo
//!
//! Demonstrates running distributed tasks across multiple machines
//! (e.g., Linux server + Mac workstation).
//!
//! # Setup
//!
//! ## Step 1: Start worker on Linux machine
//! ```bash
//! # On Linux (this machine)
//! cd ~/src/repartir
//! cargo run --bin repartir-worker --features remote --release -- --bind 0.0.0.0:9000
//! ```
//!
//! ## Step 2: Start worker on Mac
//! ```bash
//! # On Mac
//! cd ~/src/repartir
//! cargo run --bin repartir-worker --features remote --release -- --bind 0.0.0.0:9000
//! ```
//!
//! ## Step 3: Run coordinator
//! ```bash
//! # On either machine (update IPs accordingly)
//! cargo run --example multi_machine_demo --features distributed -- \
//!     --linux 192.168.1.100:9000 \
//!     --mac 192.168.1.101:9000
//! ```
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    Coordinator (this process)               │
//! │                    RemoteExecutor                           │
//! ├─────────────────────────────┬───────────────────────────────┤
//! │      Linux Worker           │         Mac Worker            │
//! │      (192.168.x.x:9000)     │       (192.168.x.x:9000)      │
//! │      ┌─────────────────┐    │       ┌─────────────────┐     │
//! │      │ 48 CPU cores    │    │       │ 10 CPU cores    │     │
//! │      │ AVX-512 SIMD    │    │       │ NEON SIMD       │     │
//! │      │ Optional GPU    │    │       │ Metal GPU       │     │
//! │      └─────────────────┘    │       └─────────────────┘     │
//! └─────────────────────────────┴───────────────────────────────┘
//! ```

#[cfg(feature = "distributed")]
use repartir::{
    error::Result,
    executor::remote::RemoteExecutor,
    executor::Executor,
    task::{Backend, ExecutionResult, Task},
};

#[cfg(feature = "distributed")]
use std::time::Instant;

#[cfg(feature = "distributed")]
#[tokio::main]
async fn main() -> Result<()> {
    // Parse command line args
    let args: Vec<String> = std::env::args().collect();

    let (linux_addr, mac_addr) = if args.len() >= 5 {
        // --linux <addr> --mac <addr>
        let linux = args
            .iter()
            .position(|a| a == "--linux")
            .map(|i| args.get(i + 1).cloned())
            .flatten()
            .unwrap_or_else(|| "localhost:9000".to_string());
        let mac = args
            .iter()
            .position(|a| a == "--mac")
            .map(|i| args.get(i + 1).cloned())
            .flatten()
            .unwrap_or_else(|| "localhost:9001".to_string());
        (linux, mac)
    } else {
        // Demo mode - use localhost with different ports
        println!("╔════════════════════════════════════════════════════════════════╗");
        println!("║          MULTI-MACHINE DISTRIBUTED COMPUTING DEMO              ║");
        println!("╠════════════════════════════════════════════════════════════════╣");
        println!("║                                                                ║");
        println!("║  To run across Linux + Mac:                                    ║");
        println!("║                                                                ║");
        println!("║  1. On Linux machine:                                          ║");
        println!("║     cd ~/src/repartir                                          ║");
        println!("║     cargo run --bin repartir-worker --features remote \\        ║");
        println!("║           --release -- --bind 0.0.0.0:9000                     ║");
        println!("║                                                                ║");
        println!("║  2. On Mac machine:                                            ║");
        println!("║     cd ~/src/repartir                                          ║");
        println!("║     cargo run --bin repartir-worker --features remote \\        ║");
        println!("║           --release -- --bind 0.0.0.0:9000                     ║");
        println!("║                                                                ║");
        println!("║  3. Run coordinator with actual IPs:                           ║");
        println!("║     cargo run --example multi_machine_demo --features \\        ║");
        println!("║           distributed -- --linux 10.0.0.1:9000 \\               ║");
        println!("║           --mac 10.0.0.2:9000                                  ║");
        println!("║                                                                ║");
        println!("╚════════════════════════════════════════════════════════════════╝");
        println!();
        println!("Running in LOCAL DEMO mode (single machine simulation)...\n");
        ("localhost:9000".to_string(), "localhost:9001".to_string())
    };

    println!("═══════════════════════════════════════════════════════════════");
    println!("  Connecting to workers...");
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Linux worker: {}", linux_addr);
    println!("  Mac worker:   {}", mac_addr);
    println!();

    // Try to connect to workers
    let executor = RemoteExecutor::new().await?;

    // Attempt to connect to both workers
    let mut connected = 0;
    for addr in [&linux_addr, &mac_addr] {
        match executor.add_worker(addr).await {
            Ok(()) => {
                println!("  ✓ Connected to {}", addr);
                connected += 1;
            }
            Err(e) => {
                println!("  ✗ Failed to connect to {}: {}", addr, e);
            }
        }
    }
    println!();

    if connected > 0 {
        println!("Connected to {} workers\n", executor.capacity());
        run_distributed_demo(&executor).await?;
    } else {
        println!("Could not connect to any workers.");
        println!();
        println!("Make sure workers are running:");
        println!(
            "  Linux: cargo run --bin repartir-worker --features remote -- --bind 0.0.0.0:9000"
        );
        println!(
            "  Mac:   cargo run --bin repartir-worker --features remote -- --bind 0.0.0.0:9000"
        );
        println!();

        // Fall back to local demo
        println!("Falling back to LOCAL CPU pool demo...\n");
        run_local_demo().await?;
    }

    Ok(())
}

#[cfg(feature = "distributed")]
async fn run_distributed_demo(executor: &RemoteExecutor) -> Result<()> {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Running distributed tasks across machines");
    println!("═══════════════════════════════════════════════════════════════\n");

    // Task 1: Get system info from each node
    println!("1. Querying system info from each node...\n");

    let tasks = vec![
        ("Linux", Task::builder().binary("uname").arg("-a").backend(Backend::Cpu).build()?),
        ("Mac", Task::builder().binary("uname").arg("-a").backend(Backend::Cpu).build()?),
    ];

    for (name, task) in tasks {
        let start = Instant::now();
        let exec_result: Result<ExecutionResult> = executor.execute(task).await;
        match exec_result {
            Ok(result) if result.is_success() => {
                let output = result.stdout_str().unwrap_or_default();
                println!("   {} ({:.1}ms):", name, start.elapsed().as_secs_f64() * 1000.0);
                println!("   └─ {}", output.trim());
            }
            Ok(result) => {
                println!("   {} failed: {:?}", name, result.exit_code());
            }
            Err(e) => {
                println!("   {} error: {}", name, e);
            }
        }
    }
    println!();

    // Task 2: Parallel computation
    println!("2. Running parallel computation on both nodes...\n");

    let computations = vec![
        ("Shard 0 (Linux)", "echo 'Computing shard 0...' && sleep 0.5 && echo 'Shard 0 complete'"),
        ("Shard 1 (Mac)", "echo 'Computing shard 1...' && sleep 0.5 && echo 'Shard 1 complete'"),
    ];

    let start = Instant::now();
    let mut handles = vec![];

    for (name, cmd) in &computations {
        let task =
            Task::builder().binary("sh").arg("-c").arg(*cmd).backend(Backend::Cpu).build()?;

        // Note: In real usage, you'd want async parallel execution
        // For demo, we show sequential but explain parallel pattern
        let result: ExecutionResult = executor.execute(task).await?;
        handles.push((name, result));
    }

    let total_time = start.elapsed();

    for (name, result) in handles {
        if result.is_success() {
            let output = result.stdout_str().unwrap_or_default();
            println!("   {}: {}", name, output.trim().replace('\n', " → "));
        }
    }
    println!();
    println!("   Total time: {:.2}s (parallel would be ~0.5s)", total_time.as_secs_f64());
    println!();

    // Task 3: Architecture comparison
    println!("3. Comparing architectures...\n");

    let arch_task = Task::builder()
        .binary("sh")
        .arg("-c")
        .arg("echo \"Arch: $(uname -m), CPUs: $(nproc 2>/dev/null || sysctl -n hw.ncpu)\"")
        .backend(Backend::Cpu)
        .build()?;

    let result: ExecutionResult = executor.execute(arch_task).await?;
    if result.is_success() {
        println!("   {}", result.stdout_str().unwrap_or_default().trim());
    }

    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Distributed demo complete!");
    println!("═══════════════════════════════════════════════════════════════");

    Ok(())
}

#[cfg(feature = "distributed")]
async fn run_local_demo() -> Result<()> {
    use repartir::Pool;

    println!("═══════════════════════════════════════════════════════════════");
    println!("  Running local CPU pool demo");
    println!("═══════════════════════════════════════════════════════════════\n");

    let cpu_count = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(4);

    let pool = Pool::builder().cpu_workers(cpu_count.min(8)).build()?;

    println!("   Pool: {} workers on {} cores\n", pool.capacity(), cpu_count);

    // Run some local tasks
    let tasks = vec![
        ("System", "uname -a"),
        ("CPU Info", "grep -c processor /proc/cpuinfo 2>/dev/null || sysctl -n hw.ncpu"),
        ("Memory", "free -h 2>/dev/null | head -2 || vm_stat | head -5"),
    ];

    for (name, cmd) in tasks {
        let task = Task::builder().binary("sh").arg("-c").arg(cmd).backend(Backend::Cpu).build()?;

        let start = Instant::now();
        let result = pool.submit(task).await?;
        let elapsed = start.elapsed();

        if result.is_success() {
            let output = result.stdout_str().unwrap_or_default();
            let first_line = output.lines().next().unwrap_or("").trim();
            println!("   {} ({:.1}ms): {}", name, elapsed.as_secs_f64() * 1000.0, first_line);
        }
    }

    pool.shutdown().await;
    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Local demo complete!");
    println!("═══════════════════════════════════════════════════════════════");

    Ok(())
}

#[cfg(not(feature = "distributed"))]
fn main() {
    println!("This example requires the 'distributed' feature.");
    println!("Run with: cargo run --example multi_machine_demo --features distributed");
}
