#![allow(clippy::all, clippy::pedantic, clippy::nursery)]
//! GPU Detection Example
//!
//! Demonstrates GPU adapter detection and metadata inspection.
//! This example shows what GPUs are available on the system.

#[cfg(feature = "gpu")]
use repartir::executor::gpu::GpuExecutor;
#[cfg(feature = "gpu")]
use repartir::executor::Executor;

#[tokio::main]
async fn main() -> repartir::error::Result<()> {
    // Initialize tracing for logs
    tracing_subscriber::fmt::init();

    println!("🎮 Repartir v1.1 - GPU Detection");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    #[cfg(feature = "gpu")]
    {
        println!("Detecting GPU adapters...");
        println!();

        match GpuExecutor::new().await {
            Ok(executor) => {
                println!("✓ GPU Detected:");
                println!("  Device: {}", executor.device_name());
                println!("  Executor: {}", executor.name());
                println!("  Compute Units: {}", executor.capacity());
                println!();
                println!("  Note: v1.1 provides GPU detection only.");
                println!("        Binary task execution on GPU not yet supported.");
                println!("        Full GPU compute coming in v1.2 with rust-gpu.");
                println!();
                println!("✓ GPU executor initialized successfully");
            }
            Err(e) => {
                println!("✗ No GPU detected: {}", e);
                println!();
                println!("  This is normal if:");
                println!("  - Running in a VM without GPU passthrough");
                println!("  - Running in CI/CD environment");
                println!("  - No GPU hardware available");
                println!();
                println!("  For GPU support, ensure:");
                println!("  - GPU drivers are installed (NVIDIA/AMD/Intel)");
                println!("  - Vulkan/Metal/DirectX 12 runtime available");
                println!("  - GPU is not exclusively claimed by another process");
            }
        }
    }

    #[cfg(not(feature = "gpu"))]
    {
        println!("❌ GPU feature not enabled");
        println!();
        println!("To enable GPU support, rebuild with:");
        println!("  cargo run --example gpu_detect --features gpu");
        println!();
        println!("Or enable in your Cargo.toml:");
        println!("  repartir = {{ version = \"0.1\", features = [\"gpu\"] }}");
    }

    println!();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    Ok(())
}
