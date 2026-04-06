//! Minimal test for CUDA error 716

use trueno_gpu::driver::{CudaContext, CudaModule};
use trueno_gpu::kernels::{Kernel, Q5_0GemvKernel, RmsNormKernel};

fn main() {
    println!("=== Testing CUDA Module Loading ===\n");

    // Initialize context
    let ctx = match CudaContext::new(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Failed to create CUDA context: {}", e);
            return;
        },
    };
    println!("✓ CUDA context created");

    // Test RmsNorm first (works)
    println!("\n--- Testing RmsNorm kernel ---");
    let rms = RmsNormKernel::new(896);
    let ptx = rms.emit_ptx();
    println!("PTX length: {} bytes", ptx.len());
    match CudaModule::from_ptx(&ctx, &ptx) {
        Ok(_) => println!("✓ RmsNorm module loaded"),
        Err(e) => println!("✗ RmsNorm failed: {}", e),
    }

    // Test Q5_0 (fails)
    println!("\n--- Testing Q5_0 GEMV kernel ---");
    let q5 = Q5_0GemvKernel::new(896, 896);
    let ptx = q5.emit_ptx();
    println!("PTX length: {} bytes", ptx.len());
    match CudaModule::from_ptx(&ctx, &ptx) {
        Ok(_) => println!("✓ Q5_0 module loaded"),
        Err(e) => println!("✗ Q5_0 failed: {}", e),
    }

    // Test with larger dimensions
    println!("\n--- Testing Q5_0 GEMV with different sizes ---");
    for (k, n) in [(256, 256), (512, 512), (896, 896), (1024, 1024)] {
        let q5 = Q5_0GemvKernel::new(k, n);
        let ptx = q5.emit_ptx();
        match CudaModule::from_ptx(&ctx, &ptx) {
            Ok(_) => println!("✓ Q5_0 {}x{} loaded", k, n),
            Err(e) => println!("✗ Q5_0 {}x{} failed: {}", k, n, e),
        }
    }

    println!("\n=== Test Complete ===");
}
