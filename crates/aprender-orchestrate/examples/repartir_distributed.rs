//! Distributed Computing with repartir
//!
//! This example demonstrates how to use repartir for distributed
//! CPU/GPU/Remote task execution across multiple machines.
//!
//! # Features Demonstrated
//!
//! - **CPU Executor**: Local multi-core execution with work-stealing
//! - **GPU Executor**: wgpu-based GPU compute (Vulkan/Metal/DX12)
//! - **Remote Executor**: TCP-based distributed execution
//! - **Tensor Operations**: SIMD-accelerated via trueno integration
//!
//! # Running
//!
//! ```bash
//! # CPU execution (default)
//! cargo run --example repartir_distributed --features distributed
//!
//! # With GPU support
//! cargo run --example repartir_distributed --features distributed,repartir/gpu
//!
//! # For remote execution, first start workers on each node:
//! # cargo run --bin repartir-worker --features remote -- --bind 0.0.0.0:9000
//! ```
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    repartir Pool                            │
//! ├─────────────────────────────────────────────────────────────┤
//! │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
//! │  │ CpuExecutor │  │ GpuExecutor │  │   RemoteExecutor    │  │
//! │  │ (work-steal)│  │ (wgpu)      │  │   (TCP/TLS)         │  │
//! │  │  AVX2/512   │  │ Vulkan/Metal│  │  Node1 ─► NodeN     │  │
//! │  └─────────────┘  └─────────────┘  └─────────────────────┘  │
//! └─────────────────────────────────────────────────────────────┘
//! ```

#[cfg(feature = "distributed")]
use repartir::{
    error::Result,
    task::{Backend, Task},
    Pool,
};

#[cfg(feature = "distributed")]
#[tokio::main]
async fn main() -> Result<()> {
    println!("=== repartir Distributed Computing Demo ===\n");

    // =========================================================================
    // Section 1: CPU Pool Execution
    // =========================================================================
    println!("1. CPU Pool Execution");
    println!("   ─────────────────────────────────────────");

    // Use available parallelism (no external dep needed)
    let cpu_count = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(4);
    println!("   Detected {} CPU cores", cpu_count);

    let pool = Pool::builder()
        .cpu_workers(cpu_count.min(8)) // Use up to 8 workers
        .max_queue_size(1000)
        .build()?;

    println!("   Pool created with {} workers", pool.capacity());
    println!("   Pending tasks: {}\n", pool.pending_tasks().await);

    // Submit a simple task
    #[cfg(unix)]
    {
        let task = Task::builder()
            .binary("/bin/echo")
            .arg("Hello from repartir!")
            .backend(Backend::Cpu)
            .build()?;

        println!("   Submitting task: echo 'Hello from repartir!'");
        let result = pool.submit(task).await?;

        if result.is_success() {
            println!("   Result: {}", result.stdout_str().unwrap_or("".into()).trim());
        } else {
            println!("   Task failed with exit code: {:?}", result.exit_code());
        }
    }

    #[cfg(not(unix))]
    {
        println!("   (Skipping task execution on non-Unix platform)");
    }

    // Graceful shutdown
    pool.shutdown().await;
    println!("   Pool shut down gracefully\n");

    // =========================================================================
    // Section 2: Backend Selection Strategy
    // =========================================================================
    println!("2. Backend Selection Strategy");
    println!("   ─────────────────────────────────────────");
    println!("   Available backends:");
    println!("   • CPU  - Multi-core with work-stealing (Blumofe & Leiserson)");
    println!("   • GPU  - wgpu compute (Vulkan/Metal/DX12/WebGPU)");
    println!("   • Remote - TCP-based distributed across machines\n");

    println!("   Selection criteria (batuta MoE router):");
    println!("   ┌─────────────┬──────────┬──────────┬──────────┐");
    println!("   │ Complexity  │  Scalar  │   SIMD   │   GPU    │");
    println!("   ├─────────────┼──────────┼──────────┼──────────┤");
    println!("   │ Low (O(n))  │   <1M    │   >1M    │  Never   │");
    println!("   │ Medium      │  <10K    │ 10K-100K │  >100K   │");
    println!("   │ High (O(n³))│   <1K    │  1K-10K  │  >10K    │");
    println!("   └─────────────┴──────────┴──────────┴──────────┘\n");

    // =========================================================================
    // Section 3: Multi-Machine Pattern
    // =========================================================================
    println!("3. Multi-Machine Distributed Pattern");
    println!("   ─────────────────────────────────────────");
    println!("   To run across multiple machines:\n");
    println!("   Step 1: Start workers on each node");
    println!("   $ repartir-worker --bind 0.0.0.0:9000\n");
    println!("   Step 2: Connect from coordinator");
    println!("   ```rust");
    println!("   use repartir::executor::remote::RemoteExecutor;");
    println!();
    println!("   let executor = RemoteExecutor::builder()");
    println!("       .add_worker(\"192.168.1.10:9000\")  // Node 1");
    println!("       .add_worker(\"192.168.1.11:9000\")  // Node 2");
    println!("       .add_worker(\"192.168.1.12:9000\")  // Node 3");
    println!("       .build().await?;");
    println!();
    println!("   let task = Task::builder()");
    println!("       .binary(\"./gpu-workload\")");
    println!("       .backend(Backend::Gpu)");
    println!("       .build()?;");
    println!();
    println!("   let result = executor.execute(task).await?;");
    println!("   ```\n");

    // =========================================================================
    // Section 4: Integration with Sovereign AI Stack
    // =========================================================================
    println!("4. Sovereign AI Stack Integration");
    println!("   ─────────────────────────────────────────");
    println!("   repartir integrates with:");
    println!("   • trueno  - SIMD tensor ops (tensor feature)");
    println!("   • trueno-db - Checkpoint persistence (checkpoint feature)");
    println!("   • batuta  - Orchestration and MoE routing");
    println!("   • aprender - Distributed ML training\n");

    println!("   Feature flags:");
    println!("   ┌──────────────┬──────────────────────────────────────┐");
    println!("   │ Feature      │ Description                          │");
    println!("   ├──────────────┼──────────────────────────────────────┤");
    println!("   │ cpu          │ Multi-core execution (default)       │");
    println!("   │ gpu          │ wgpu GPU compute                     │");
    println!("   │ remote       │ TCP distributed execution            │");
    println!("   │ remote-tls   │ TLS-secured remote                   │");
    println!("   │ tensor       │ trueno SIMD integration              │");
    println!("   │ checkpoint   │ trueno-db + Parquet persistence      │");
    println!("   │ tui          │ Job flow TUI visualization           │");
    println!("   │ full         │ All features enabled                 │");
    println!("   └──────────────┴──────────────────────────────────────┘\n");

    println!("=== Demo Complete ===");
    Ok(())
}

#[cfg(not(feature = "distributed"))]
fn main() {
    println!("This example requires the 'distributed' feature.");
    println!("Run with: cargo run --example repartir_distributed --features distributed");
}
