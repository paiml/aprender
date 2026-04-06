//! Debug multi-head attention kernel
#[cfg(feature = "cuda")]
use realizar::cuda::{CudaExecutor, CudaKernels, KernelType};

fn main() {
    #[cfg(feature = "cuda")]
    {
        println!("=== Debug Multi-Head Attention Kernel ===\n");

        let seq_len = 4u32;
        let head_dim = 64u32;
        let n_heads = 1u32;
        let causal = true;

        // Generate PTX for both kernel types
        let kernels = CudaKernels::new();

        // Standard multi-head attention
        let mha_type = KernelType::MultiHeadAttention {
            seq_len,
            head_dim,
            n_heads,
            causal,
        };
        let mha_ptx = kernels.generate_ptx(&mha_type);
        let mha_name = kernels.kernel_name(&mha_type);

        println!("MultiHeadAttention kernel:");
        println!("  Name: {}", mha_name);
        println!("  PTX size: {} bytes", mha_ptx.len());

        // Calculate launch config (same as flash_attention_multi_head)
        let max_tile = (48 * 1024) / (head_dim * 12);
        let tile_q = max_tile.min(64).min(seq_len);
        let num_q_blocks = seq_len.div_ceil(tile_q);
        let threads_per_block = (tile_q * head_dim).min(1024);

        println!("  max_tile: {}", max_tile);
        println!("  tile_q: {}", tile_q);
        println!("  num_q_blocks: {}", num_q_blocks);
        println!("  threads_per_block: {}", threads_per_block);
        println!("  grid: ({}, {}, 1)", num_q_blocks, n_heads);
        println!("\n  PTX:\n{}\n", mha_ptx);

        // Validate with ptxas
        println!("Validating PTX with ptxas...");
        std::fs::write("/tmp/mha_debug.ptx", &mha_ptx).expect("test");
        let output = std::process::Command::new("ptxas")
            .args([
                "--gpu-name",
                "sm_89",
                "/tmp/mha_debug.ptx",
                "-o",
                "/dev/null",
            ])
            .output()
            .expect("ptxas failed");

        if output.status.success() {
            println!("PTX validation: PASS");
        } else {
            println!("PTX validation: FAIL");
            println!("stderr: {}", String::from_utf8_lossy(&output.stderr));
        }

        // Now try to execute
        println!("\n=== Testing execution ===");
        if !CudaExecutor::is_available() {
            println!("CUDA not available");
            return;
        }

        let mut executor = CudaExecutor::new(0).expect("Failed to create executor");
        println!("GPU: {}", executor.device_name().unwrap_or_default());

        let total_size = (seq_len * head_dim * n_heads) as usize;
        let q = vec![0.1f32; total_size];
        let k = vec![0.1f32; total_size];
        let v = vec![0.1f32; total_size];
        let mut attn_output = vec![0.0f32; total_size];

        match executor.flash_attention_multi_head(
            &q,
            &k,
            &v,
            &mut attn_output,
            seq_len,
            head_dim,
            n_heads,
            causal,
        ) {
            Ok(()) => println!("SUCCESS! output[0] = {}", attn_output[0]),
            Err(e) => println!("FAILED: {}", e),
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        println!("CUDA feature not enabled, skipping test");
    }
}
