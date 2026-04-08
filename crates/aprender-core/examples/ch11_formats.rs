#![allow(clippy::disallowed_methods)]
//! Chapter 11: Model Formats and Conversion
//!
//! Demonstrates format contracts and layout invariants.
//! Citation: Frantar et al., "GPTQ," arXiv:2210.17323
//! Contract: contracts/apr-book-ch11-v1.yaml

fn main() {
    // Format contract: APR is ALWAYS row-major
    println!("Tensor layout contract (LAYOUT-001):");
    println!("  GGUF [ne0, ne1] col-major -> transpose -> APR [rows, cols] row-major");
    println!("  SafeTensors native -> APR [rows, cols] row-major");

    // Shape reversal contract
    let gguf_shape = [4096_usize, 11008]; // [ne0, ne1] in GGUF
    let apr_shape = [gguf_shape[1], gguf_shape[0]]; // [rows, cols] in APR
    println!(
        "\nGGUF shape: {:?} -> APR shape: {:?}",
        gguf_shape, apr_shape
    );
    assert_eq!(apr_shape[0], 11008, "Rows = ne1");
    assert_eq!(apr_shape[1], 4096, "Cols = ne0");

    // 2D tensor transpose contract
    // Only 2D tensors are transposed; 1D tensors (bias, norm) pass through
    let is_2d = |shape: &[usize]| shape.len() == 2;
    assert!(is_2d(&[11008, 4096]), "Weight matrices are 2D");
    assert!(!is_2d(&[4096]), "Bias vectors are 1D");

    // APR magic bytes contract
    let magic_v2 = b"APR\0";
    let magic_v1 = b"APRN";
    println!("\nMagic bytes: v2={:?}, v1={:?}", magic_v2, magic_v1);

    // Quantization format sizes
    let q4k_block_size = 256_usize;
    let q4k_bytes_per_block = 144_usize; // K-quant superblock
    let bits_per_weight = (q4k_bytes_per_block * 8) as f64 / q4k_block_size as f64;
    println!("\nQ4K: {bits_per_weight:.1} bits/weight (block size {q4k_block_size})");
    assert!(bits_per_weight < 5.0, "Q4K must be under 5 bits/weight");
    assert!(bits_per_weight > 4.0, "Q4K must be over 4 bits/weight");

    // Import vs convert architecture
    println!("\nArchitecture contract:");
    println!("  apr import = PASSTHROUGH ONLY");
    println!("  apr convert = TRANSFORMATIONS");

    println!("\nChapter 11 contracts: PASSED");
}
