#![allow(clippy::disallowed_methods)]
//! Chapter 8: Transformer Architecture
//!
//! Demonstrates GQA, RoPE, SwiGLU architecture parameters.
//! Verified by: apr oracle --family qwen2 --explain --stats
//! Citation: Vaswani et al., "Attention Is All You Need," arXiv:1706.03762
//! Contract: contracts/apr-book-ch08-v1.yaml

fn main() {
    // Qwen2-7B architecture (from: apr oracle --family qwen2 --size 7b --stats)
    let hidden_dim: usize = 3584;
    let num_layers: usize = 28;
    let num_heads: usize = 28;
    let num_kv_heads: usize = 4;
    let intermediate_dim: usize = 18944;
    let head_dim: usize = 128;
    let rope_theta: f64 = 1_000_000.0;

    // GQA ratio contract (Ainslie et al., 2023, arXiv:2305.13245)
    let gqa_ratio = num_kv_heads as f64 / num_heads as f64;
    let kv_reduction = 1.0 - gqa_ratio;
    println!("GQA ratio: {gqa_ratio:.2} -> {:.0}% KV cache reduction", kv_reduction * 100.0);
    assert!(
        kv_reduction > 0.5,
        "GQA must reduce KV cache by >50%"
    );

    // SwiGLU expansion ratio (Shazeer, 2020, arXiv:2002.05202)
    let ffn_ratio = intermediate_dim as f64 / hidden_dim as f64;
    println!("SwiGLU expansion: {ffn_ratio:.2}x (compensates for gating)");
    assert!(ffn_ratio > 2.0, "SwiGLU expansion must exceed 2x");

    // Head dimension contract (Vaswani et al., 2017)
    assert_eq!(
        hidden_dim,
        num_heads * head_dim,
        "hidden_dim = num_heads * head_dim"
    );

    // RoPE theta contract (Su et al., 2021, arXiv:2104.09864)
    println!("RoPE theta: {rope_theta:.0} (higher = longer context)");
    assert!(rope_theta > 0.0, "RoPE theta must be positive");

    // KV cache budget (Pope et al., 2022, arXiv:2211.05102)
    let ctx_len: usize = 4096;
    let kv_bytes = 2 * num_layers * num_kv_heads * head_dim * ctx_len * 2;
    let kv_mb = kv_bytes as f64 / (1024.0 * 1024.0);
    println!("KV cache at {ctx_len} context: {kv_mb:.0} MB");

    // RMSNorm vs LayerNorm (Zhang & Sennrich, 2019, arXiv:1910.07467)
    println!("Normalization: RMSNorm (no mean subtraction, faster)");

    println!("Chapter 8 contracts: PASSED");
}
