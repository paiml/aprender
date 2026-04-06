//! Test multi-head attention with phi-2 dimensions
#[cfg(feature = "cuda")]
use realizar::cuda::CudaExecutor;

fn main() {
    #[cfg(feature = "cuda")]
    {
        println!("Testing multi-head attention with phi-2 dimensions...");

        if !CudaExecutor::is_available() {
            println!("CUDA not available");
            return;
        }

        let mut executor = CudaExecutor::new(0).expect("Failed to create executor");
        println!("GPU: {}", executor.device_name().unwrap_or_default());

        // Try minimal dimensions like test_tc_attention
        let seq_len = 4u32;
        let head_dim = 64u32;
        let n_heads = 1u32; // Single head
        let causal = true;

        let total_size = (seq_len * head_dim * n_heads) as usize;
        println!(
            "Testing flash_attention_multi_head with seq_len={}, head_dim={}, n_heads={}",
            seq_len, head_dim, n_heads
        );
        println!("Total size: {} floats", total_size);

        let q = vec![1.0f32; total_size];
        let k = vec![1.0f32; total_size];
        let v = vec![1.0f32; total_size];
        let mut output = vec![0.0f32; total_size];

        match executor.flash_attention_multi_head(
            &q,
            &k,
            &v,
            &mut output,
            seq_len,
            head_dim,
            n_heads,
            causal,
        ) {
            Ok(()) => println!("SUCCESS! output[0] = {}", output[0]),
            Err(e) => println!("FAILED: {}", e),
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        println!("CUDA feature not enabled, skipping test");
    }
}
