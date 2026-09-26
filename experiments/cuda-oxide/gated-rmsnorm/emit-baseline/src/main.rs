//! `cargo run --release -- <sm_XX> <head_dim> <num_heads> <eps>` → the hand-PTX `gdn_gated_rmsnorm` on stdout.
use trueno_gpu::kernels::{GatedRmsNormKernel, Kernel};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let arg = |i: usize, d: &str| a.get(i).cloned().unwrap_or_else(|| d.to_string());
    let target = arg(1, "sm_89");
    let head_dim: u32 = arg(2, "128").parse().expect("head_dim");
    let num_heads: u32 = arg(3, "16").parse().expect("num_heads");
    let eps: f32 = arg(4, "1e-6").parse().expect("eps");
    print!(
        "{}",
        GatedRmsNormKernel::new(head_dim, num_heads, eps).emit_ptx_for_target(&target)
    );
}
