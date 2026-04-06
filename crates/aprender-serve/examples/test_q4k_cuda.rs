//! Test Q4K CUDA kernel loading
//!
//! Run: cargo run --release --features cuda --example test_q4k_cuda

#[cfg(feature = "cuda")]
use realizar::cuda::{CudaExecutor, CudaKernels, KernelType};

fn main() {
    #[cfg(feature = "cuda")]
    {
        println!("Testing Q4K CUDA kernel...");

        // First dump the PTX
        let kernels = CudaKernels::new();
        let kernel_type = KernelType::QuantizedGemm { m: 1, n: 1, k: 32 };
        let ptx = kernels.generate_ptx(&kernel_type);

        println!("=== Generated PTX ===");
        println!("{}", ptx);
        println!("=== End PTX ===");

        if !CudaExecutor::is_available() {
            println!("CUDA not available");
            return;
        }

        let mut executor = CudaExecutor::new(0).expect("test");
        println!(
            "CUDA executor created: {}",
            executor.device_name().unwrap_or_default()
        );

        // Create minimal Q4K data (18 bytes per 32-value block)
        let k = 32u32; // Single block
        let m = 1u32; // Single output

        // Simplified Q4K block: 18 bytes
        let weights = vec![0u8; 18];
        let input = vec![1.0f32; k as usize];
        let mut output = vec![0.0f32; m as usize];

        println!("Calling q4k_matvec with m={}, k={}...", m, k);

        match executor.q4k_matvec(&weights, &input, &mut output, m, k) {
            Ok(()) => println!("SUCCESS! Output: {:?}", output),
            Err(e) => println!("FAILED: {:?}", e),
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        println!("CUDA feature not enabled, skipping test");
    }
}
