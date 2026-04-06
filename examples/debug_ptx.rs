//! Debug PTX generation for multi-head attention
use realizar::cuda::{CudaKernels, KernelType};

fn main() {
    let kernels = CudaKernels::new();

    // Use same dimensions as phi-2: seq_len=128, head_dim=80, n_heads=32
    let kernel = KernelType::MultiHeadAttention {
        seq_len: 128,
        head_dim: 80,
        n_heads: 32,
        causal: true,
    };

    let ptx = kernels.generate_ptx(&kernel);

    println!("=== Multi-Head Attention PTX for phi-2 ===");
    println!("seq_len=128, head_dim=80, n_heads=32, causal=true");
    println!("PTX size: {} bytes\n", ptx.len());
    println!("{}", ptx);

    // Save to file
    std::fs::write("/tmp/attention_phi2.ptx", &ptx).expect("test");
    println!("\nPTX saved to /tmp/attention_phi2.ptx");
}
