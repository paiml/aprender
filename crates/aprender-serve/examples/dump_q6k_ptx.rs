//! CORRECTNESS-002: Dump Q6K kernel PTX for manual inspection
//!
//! Run with: cargo run --release --features cuda --example dump_q6k_ptx

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use realizar::cuda::{CudaKernels, KernelType};

    let kernels = CudaKernels::default();

    // Generate Q6K GEMV kernel PTX for typical LM head dimensions
    // hidden_dim = 1536, vocab_size = 151936
    let kernel_type = KernelType::Q6KGemv {
        k: 1536, // hidden_dim
        n: 100,  // small n for readable PTX
    };

    let ptx = kernels.generate_ptx(&kernel_type);

    println!("// Q6K GEMV PTX for k=1536, n=100");
    println!("// Look for issues in:");
    println!("// 1. Scale indexing (scale_idx calculation)");
    println!("// 2. ql/qh byte offset calculation");
    println!("// 3. Nibble extraction (high vs low)");
    println!("// 4. qh bit shifting");
    println!("// 5. Activation index calculation");
    println!("//");
    println!("{}", ptx);

    Ok(())
}
