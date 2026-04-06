//! Test Q4K with phi-2 dimensions
#[cfg(feature = "cuda")]
use realizar::cuda::CudaExecutor;

fn main() {
    #[cfg(feature = "cuda")]
    {
        println!("Testing Q4K with phi-2 dimensions...");

        if !CudaExecutor::is_available() {
            println!("CUDA not available");
            return;
        }

        let mut executor = CudaExecutor::new(0).expect("Failed to create executor");
        println!("GPU: {}", executor.device_name().unwrap_or_default());

        // phi-2: hidden_dim=2560
        let m = 2560u32;
        let k = 2560u32;

        // Q4_K block size is 256 values, so we need k to be a multiple
        // Each Q4_K superblock is 144 bytes for 256 values
        let num_blocks = (k as usize).div_ceil(256);
        let weight_bytes = num_blocks * 144 * m as usize;

        println!("Testing q4k_matvec with m={}, k={}", m, k);
        println!(
            "Weight buffer: {} bytes ({} blocks)",
            weight_bytes,
            num_blocks * m as usize
        );

        let weights = vec![0u8; weight_bytes];
        let input = vec![1.0f32; k as usize];
        let mut output = vec![0.0f32; m as usize];

        match executor.q4k_matvec(&weights, &input, &mut output, m, k) {
            Ok(()) => println!("SUCCESS! output[0] = {}", output[0]),
            Err(e) => println!("FAILED: {}", e),
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        println!("CUDA feature not enabled, skipping test");
    }
}
