//! Debug CUDA initialization - tests multiple executor creation/destruction cycles
//!
//! Run with: cargo run --features cuda --example cuda_debug

use realizar::cuda::CudaExecutor;
use std::io::Write;

fn main() {
    println!("=== CUDA Debug Test (Multiple Cycles) ===\n");

    println!("Step 1: is_available() = {}", CudaExecutor::is_available());
    println!("Step 2: num_devices() = {}", CudaExecutor::num_devices());

    // Test multiple create/destroy cycles (like tests do)
    for i in 1..=3 {
        println!("\n=== Cycle {} ===", i);
        std::io::stdout().flush().ok();

        println!("Creating executor...");
        match CudaExecutor::new(0) {
            Ok(mut executor) => {
                println!("  SUCCESS: {}", executor.device_name().unwrap_or_default());

                // Do some work like the tests do
                let a = vec![1.0f32; 16];
                let b = vec![1.0f32; 16];
                let mut c = vec![0.0f32; 16];

                println!("  Running GEMM...");
                match executor.gemm(&a, &b, &mut c, 4, 4, 4) {
                    Ok(()) => println!("  GEMM OK: c[0]={:.1}", c[0]),
                    Err(e) => println!("  GEMM failed: {}", e),
                }

                println!("  Dropping executor...");
            },
            Err(e) => {
                println!("  FAILED: {}", e);
            },
        }
        println!("  Executor dropped");
    }

    println!("\n=== All cycles completed successfully! ===");
}
