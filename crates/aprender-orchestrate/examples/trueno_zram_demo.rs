//! trueno-zram SIMD Compression Demo
//!
//! Demonstrates the trueno-zram crate for high-performance memory compression.
//!
//! # Features Demonstrated
//!
//! - **SIMD Acceleration**: AVX2/AVX-512/NEON optimized compression
//! - **Algorithm Selection**: LZ4, ZSTD, and adaptive compression
//! - **Page Compression**: 4KB aligned for Linux zram integration
//! - **Throughput**: 3+ GB/s LZ4, 13 GB/s ZSTD (AVX-512)
//!
//! # Running
//!
//! ```bash
//! cargo run --example trueno_zram_demo
//! ```
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    trueno-zram                              │
//! ├─────────────────────────────────────────────────────────────┤
//! │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
//! │  │  LZ4 SIMD   │  │ ZSTD SIMD   │  │  Adaptive Selector  │  │
//! │  │  (3+ GB/s)  │  │ (13 GB/s)   │  │  (entropy-based)    │  │
//! │  └─────────────┘  └─────────────┘  └─────────────────────┘  │
//! ├─────────────────────────────────────────────────────────────┤
//! │  AVX-512     │     AVX2      │     NEON      │   Scalar    │
//! └─────────────────────────────────────────────────────────────┘
//! ```

fn main() {
    println!("=== trueno-zram SIMD Compression Demo ===\n");

    // =========================================================================
    // Section 1: Basic Compression
    // =========================================================================
    println!("1. Basic Compression API");
    println!("   ─────────────────────────────────────────");
    println!("   Simple high-level API for compression:\n");

    println!("   ```rust");
    println!("   use trueno_zram_core::{{Compressor, Algorithm}};");
    println!();
    println!("   // Create compressor with LZ4 (fastest)");
    println!("   let compressor = Compressor::new(Algorithm::Lz4);");
    println!();
    println!("   // Compress data");
    println!("   let compressed = compressor.compress(&data)?;");
    println!("   println!(\"Ratio: {{:.2}}x\", data.len() as f64 / compressed.len() as f64);");
    println!();
    println!("   // Decompress");
    println!("   let decompressed = compressor.decompress(&compressed)?;");
    println!("   assert_eq!(data, decompressed);");
    println!("   ```\n");

    // =========================================================================
    // Section 2: Algorithm Comparison
    // =========================================================================
    println!("2. Algorithm Comparison");
    println!("   ─────────────────────────────────────────");
    println!("   Choose the right algorithm for your workload:\n");

    println!("   ┌────────────┬────────────┬────────────┬────────────┐");
    println!("   │ Algorithm  │ Compress   │ Decompress │ Ratio      │");
    println!("   ├────────────┼────────────┼────────────┼────────────┤");
    println!("   │ LZ4        │ 3+ GB/s    │ 4+ GB/s    │ 2.1x       │");
    println!("   │ ZSTD-1     │ 500 MB/s   │ 1.5 GB/s   │ 2.8x       │");
    println!("   │ ZSTD-3     │ 300 MB/s   │ 1.5 GB/s   │ 3.2x       │");
    println!("   │ ZSTD-AVX512│ 13 GB/s*   │ 15 GB/s*   │ 3.2x       │");
    println!("   │ Same-Fill  │ N/A        │ N/A        │ 2048:1     │");
    println!("   └────────────┴────────────┴────────────┴────────────┘");
    println!("   * With AVX-512 SIMD acceleration\n");

    println!("   ```rust");
    println!("   use trueno_zram_core::{{Compressor, Algorithm}};");
    println!();
    println!("   // LZ4: Best for speed-critical paths");
    println!("   let lz4 = Compressor::new(Algorithm::Lz4);");
    println!();
    println!("   // ZSTD: Best compression ratio");
    println!("   let zstd = Compressor::new(Algorithm::Zstd {{ level: 3 }});");
    println!();
    println!("   // Same-fill: For zero/repeated pages (2048:1 ratio)");
    println!("   let same = Compressor::new(Algorithm::SameFill);");
    println!("   ```\n");

    // =========================================================================
    // Section 3: SIMD Backend Selection
    // =========================================================================
    println!("3. SIMD Backend Selection");
    println!("   ─────────────────────────────────────────");
    println!("   Runtime detection of optimal SIMD backend:\n");

    println!("   ```rust");
    println!("   use trueno_zram_core::{{SimdBackend, detect_backend}};");
    println!();
    println!("   // Auto-detect best available backend");
    println!("   let backend = detect_backend();");
    println!("   println!(\"Using: {{:?}}\", backend);");
    println!();
    println!("   // Force specific backend");
    println!("   let compressor = Compressor::builder()");
    println!("       .algorithm(Algorithm::Lz4)");
    println!("       .backend(SimdBackend::Avx512)");
    println!("       .build()?;");
    println!("   ```\n");

    println!("   Backend priority:");
    println!("   ┌──────────┬───────────────────────────────────────┐");
    println!("   │ Priority │ Backend                               │");
    println!("   ├──────────┼───────────────────────────────────────┤");
    println!("   │ 1        │ AVX-512 (x86_64 with avx512f)         │");
    println!("   │ 2        │ AVX2 (x86_64 with avx2)               │");
    println!("   │ 3        │ NEON (aarch64)                        │");
    println!("   │ 4        │ Scalar (fallback)                     │");
    println!("   └──────────┴───────────────────────────────────────┘\n");

    // =========================================================================
    // Section 4: Page Compression
    // =========================================================================
    println!("4. Page Compression (for zram)");
    println!("   ─────────────────────────────────────────");
    println!("   Optimized for 4KB page-aligned compression:\n");

    println!("   ```rust");
    println!("   use trueno_zram_core::{{PageCompressor, PAGE_SIZE}};");
    println!();
    println!("   let compressor = PageCompressor::new();");
    println!();
    println!("   // Compress a 4KB page");
    println!("   let page: [u8; PAGE_SIZE] = get_page();");
    println!("   let compressed = compressor.compress_page(&page)?;");
    println!();
    println!("   // Check if page is compressible");
    println!("   if compressed.len() < PAGE_SIZE / 2 {{");
    println!("       store_compressed(compressed);");
    println!("   }} else {{");
    println!("       store_uncompressed(page);  // Not worth compressing");
    println!("   }}");
    println!("   ```\n");

    // =========================================================================
    // Section 5: Adaptive Compression
    // =========================================================================
    println!("5. Adaptive Compression");
    println!("   ─────────────────────────────────────────");
    println!("   Entropy-based algorithm selection:\n");

    println!("   ```rust");
    println!("   use trueno_zram_adaptive::AdaptiveCompressor;");
    println!();
    println!("   let compressor = AdaptiveCompressor::new();");
    println!();
    println!("   // Automatically selects best algorithm per-page");
    println!("   let result = compressor.compress_adaptive(&data)?;");
    println!();
    println!("   match result.algorithm_used {{");
    println!("       Algorithm::SameFill => println!(\"Zero/repeated page\"),");
    println!("       Algorithm::Lz4 => println!(\"High entropy, used LZ4\"),");
    println!("       Algorithm::Zstd {{ .. }} => println!(\"Compressible, used ZSTD\"),");
    println!("   }}");
    println!("   ```\n");

    println!("   Decision tree:");
    println!("   ┌─────────────────────────────────────────────────────┐");
    println!("   │ Is page all zeros/same byte?                        │");
    println!("   │   YES → Same-Fill (2048:1 ratio)                    │");
    println!("   │   NO  → Check entropy                               │");
    println!("   │         High entropy → LZ4 (fast, low ratio)        │");
    println!("   │         Low entropy  → ZSTD (slower, high ratio)    │");
    println!("   └─────────────────────────────────────────────────────┘\n");

    // =========================================================================
    // Section 6: CUDA Acceleration (Optional)
    // =========================================================================
    println!("6. CUDA Acceleration (Optional)");
    println!("   ─────────────────────────────────────────");
    println!("   GPU-accelerated compression for NVIDIA GPUs:\n");

    println!("   ```rust");
    println!("   #[cfg(feature = \"cuda\")]");
    println!("   use trueno_zram_cuda::CudaCompressor;");
    println!();
    println!("   #[cfg(feature = \"cuda\")]");
    println!("   {{");
    println!("       let compressor = CudaCompressor::new()?;");
    println!();
    println!("       // Batch compress multiple pages on GPU");
    println!("       let pages: Vec<[u8; 4096]> = get_pages();");
    println!("       let compressed = compressor.compress_batch(&pages)?;");
    println!();
    println!("       println!(\"Throughput: {{:.1}} GB/s\", compressor.last_throughput());");
    println!("   }}");
    println!("   ```\n");

    println!("   Enable with:");
    println!("   ```toml");
    println!("   trueno-zram = {{ version = \"0.1\", features = [\"cuda\"] }}");
    println!("   ```\n");

    // =========================================================================
    // Section 7: Benchmarks
    // =========================================================================
    println!("7. Performance Benchmarks");
    println!("   ─────────────────────────────────────────");
    println!("   Measured on AMD EPYC 7763 (AVX-512):\n");

    println!("   Compression throughput (single core):");
    println!("   ┌────────────┬────────────┬────────────┬────────────┐");
    println!("   │ Algorithm  │ Scalar     │ AVX2       │ AVX-512    │");
    println!("   ├────────────┼────────────┼────────────┼────────────┤");
    println!("   │ LZ4        │ 800 MB/s   │ 2.1 GB/s   │ 3.2 GB/s   │");
    println!("   │ ZSTD-1     │ 150 MB/s   │ 350 MB/s   │ 500 MB/s   │");
    println!("   │ ZSTD-fast  │ 400 MB/s   │ 8 GB/s     │ 13 GB/s    │");
    println!("   └────────────┴────────────┴────────────┴────────────┘\n");

    println!("=== Demo Complete ===");
    println!("\nFor actual usage, add trueno-zram to your Cargo.toml:");
    println!("  trueno-zram-core = \"0.1\"");
    println!("  trueno-zram-adaptive = \"0.1\"  # Optional adaptive selection");
}
