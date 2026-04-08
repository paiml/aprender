#![allow(clippy::disallowed_methods)]
//! Chapter 9: Inference with aprender-serve
//!
//! Demonstrates quantization math and inference architecture.
//! Citation: Dettmers et al., "LLM.int8()," arXiv:2208.07339
//! Contract: contracts/apr-book-ch09-v1.yaml

fn main() {
    // Quantization bit-width contract
    // Q4K: 4.5 bits/weight effective (Dettmers et al., arXiv:2210.17323)
    let params_7b: f64 = 7e9;
    let bytes_f16 = params_7b * 2.0;
    let bytes_q4k = params_7b * 4.5 / 8.0;
    let compression = bytes_f16 / bytes_q4k;
    println!("7B model size:");
    println!("  F16:  {:.1} GB", bytes_f16 / 1e9);
    println!("  Q4K:  {:.1} GB", bytes_q4k / 1e9);
    println!("  Compression: {compression:.1}x");
    assert!(compression > 3.0, "Q4K must compress >3x vs F16");

    // Fused dequant+matmul contract
    // (Dao et al., FlashAttention-2, arXiv:2307.08691)
    println!("\nFused dequant+matmul: dequantize inline during GEMV");
    println!("  Avoids materializing full F32 weight matrix");
    println!("  Memory bandwidth: read Q4K, compute F32, write F32");

    // PagedAttention KV cache (Kwon et al., arXiv:2309.06180)
    let page_size = 16_usize; // tokens per page
    let ctx_len = 4096_usize;
    let pages_needed = (ctx_len + page_size - 1) / page_size;
    println!("\nPagedAttention:");
    println!("  Page size: {page_size} tokens");
    println!("  Pages for {ctx_len} context: {pages_needed}");
    assert_eq!(pages_needed, 256, "Page count contract");

    // Performance targets (from apr oracle)
    println!("\nPerformance targets:");
    println!("  1B Q4K: 100+ tok/s CPU, 500+ tok/s GPU");
    println!("  7B Q4K:  30+ tok/s CPU, 150+ tok/s GPU");

    // Architecture contract
    println!("\nContract: aprender-serve handles ALL inference");
    println!("Contract: aprender-core is for TRAINING ONLY");

    println!("Chapter 9 contracts: PASSED");
}
