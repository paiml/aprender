#![allow(clippy::disallowed_methods)]
//! Chapter 3: The APR Model Format
//!
//! Demonstrates TensorStats, layout contracts, and format validation.
//! Citation: Safronov et al., "SafeTensors," HuggingFace 2023
//! Contract: contracts/apr-book-ch03-v1.yaml (v2 — api_calls enforced)

use aprender::format::validated_tensors::TensorStats;

fn main() {
    // --- TensorStats: compute statistics on real tensor data ---
    let weight_data: Vec<f32> = (0..768).map(|i| ((i as f32 / 768.0) - 0.5) * 2.0).collect();

    let stats = TensorStats::compute(&weight_data);
    println!("TensorStats on 768-element weight tensor:");
    println!("  min:      {:.4}", stats.min);
    println!("  max:      {:.4}", stats.max);
    println!("  mean:     {:.4}", stats.mean);
    println!("  l2_norm:  {:.4}", stats.l2_norm);
    println!("  zero_pct: {:.2}%", stats.zero_pct() * 100.0);

    assert!(stats.min < 0.0, "Weight tensor should have negative values");
    assert!(stats.max > 0.0, "Weight tensor should have positive values");
    assert!(
        stats.l2_norm > 0.0,
        "Non-constant tensor must have l2_norm > 0"
    );
    assert!(
        (stats.mean).abs() < 0.1,
        "Symmetric data should have mean near 0"
    );

    // All-zero tensor
    let zero_stats = TensorStats::compute(&vec![0.0_f32; 100]);
    println!(
        "\nAll-zero tensor: zero_pct={:.0}%",
        zero_stats.zero_pct() * 100.0
    );
    assert!(
        (zero_stats.zero_pct() - 100.0).abs() < 1e-4,
        "All-zero tensor must have 100% zeros, got {:.2}%",
        zero_stats.zero_pct()
    );

    // --- Layout contract (LAYOUT-001) ---
    let gguf_shape = [4096_usize, 11008];
    let apr_shape = [gguf_shape[1], gguf_shape[0]];
    println!("\nLayout contract (LAYOUT-001):");
    println!("  GGUF {:?} -> APR {:?}", gguf_shape, apr_shape);
    assert_eq!(apr_shape[0], 11008, "Rows = ne1 after transpose");
    assert_eq!(apr_shape[1], 4096, "Cols = ne0 after transpose");

    // APR magic bytes
    let magic_v2 = b"APR\0";
    assert_eq!(magic_v2[3], 0, "v2 null-terminated");
    println!("  Magic: {:?}", magic_v2);

    println!("\nChapter 3 contracts: PASSED");
}
