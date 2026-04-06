//! Minimal test using `GpuBatchCompressor` directly

#[cfg(feature = "cuda")]
fn main() {
    use trueno_zram_core::gpu::{GpuBatchCompressor, GpuBatchConfig};
    use trueno_zram_core::Algorithm;

    println!("=== DEBUG: GpuBatchCompressor Decompression ===\n");

    // Create compressor
    let config = GpuBatchConfig {
        device_index: 0,
        algorithm: Algorithm::Lz4,
        batch_size: 1000,
        async_dma: true,
        ring_buffer_slots: 4,
    };

    println!("Step 1: Creating GpuBatchCompressor...");
    let mut compressor = GpuBatchCompressor::new(config).expect("Should create compressor");
    println!("  ✓ Created\n");

    // Create test data - use same pattern as benchmark
    const NUM_PAGES: usize = 10_000; // Match benchmark scale
    println!("Step 2: Creating {} test pages...", NUM_PAGES);

    let pages: Vec<[u8; 4096]> = (0..NUM_PAGES)
        .map(|i| {
            let mut page = [0u8; 4096];
            match i % 4 {
                0 => {} // Zero page
                1 => {
                    for (j, byte) in page.iter_mut().enumerate() {
                        *byte = (j % 256) as u8;
                    }
                }
                _ => {
                    let mut rng = (i as u64).wrapping_mul(0x0005_DEEC_E66D);
                    for byte in &mut page {
                        rng = rng.wrapping_mul(0x0005_DEEC_E66D).wrapping_add(0xB);
                        *byte = (rng >> 33) as u8;
                    }
                }
            }
            page
        })
        .collect();
    println!("  ✓ Created\n");

    // Compress using CPU (like benchmark does)
    println!("Step 3: Compressing with CPU...");
    let compress_result = compressor.compress_batch(&pages).expect("Compression should work");

    let compressed: Vec<Vec<u8>> = compress_result.pages.iter().map(|p| p.data.clone()).collect();
    let sizes: Vec<u32> = compressed.iter().map(|c| c.len() as u32).collect();

    let total_compressed: usize = compressed.iter().map(|c| c.len()).sum();
    println!(
        "  Compressed {} pages: {} bytes ({:.2}x ratio)",
        NUM_PAGES,
        total_compressed,
        (NUM_PAGES * 4096) as f64 / total_compressed as f64
    );
    println!(
        "  Size range: {} - {} bytes",
        sizes.iter().min().unwrap(),
        sizes.iter().max().unwrap()
    );

    // DEBUG: Print first 16 bytes of first few compressed pages
    println!("  First 3 compressed pages:");
    for i in 0..3.min(NUM_PAGES) {
        let bytes: Vec<String> =
            compressed[i].iter().take(16).map(|b| format!("{:02x}", b)).collect();
        println!("    Page {}: [{}...] ({} bytes)", i, bytes.join(" "), compressed[i].len());
    }
    println!();

    // Try to decompress using GPU
    println!("Step 4: Decompressing with GPU...");
    match compressor.decompress_batch_gpu(&compressed, &sizes) {
        Ok(result) => {
            println!("  ✓ Decompressed {} pages\n", result.pages.len());

            // Verify
            let mut mismatches = 0;
            for (i, page) in result.pages.iter().enumerate() {
                if page != &pages[i] {
                    mismatches += 1;
                    if mismatches <= 3 {
                        println!("  Page {} mismatch", i);
                    }
                }
            }

            if mismatches == 0 {
                println!("✓ SUCCESS: All {} pages verified!", NUM_PAGES);
            } else {
                println!("✗ FAILED: {} pages with mismatches", mismatches);
            }
        }
        Err(e) => {
            println!("  ✗ FAILED: {}\n", e);
        }
    }
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("Requires --features cuda");
}
