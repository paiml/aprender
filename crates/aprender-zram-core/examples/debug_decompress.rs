//! Minimal debug test for GPU decompression kernel

#[cfg(feature = "cuda")]
fn main() {
    use std::ffi::c_void;
    use trueno_gpu::driver::{CudaContext, CudaModule, CudaStream, GpuBuffer, LaunchConfig};
    use trueno_gpu::kernels::lz4::Lz4DecompressKernel;
    use trueno_gpu::kernels::Kernel;
    use trueno_zram_core::lz4::compress as lz4_compress_crate; // Use crate's LZ4

    println!("=== DEBUG: GPU Decompression Kernel ===\n");

    // Step 1: Create CUDA context
    println!("Step 1: Creating CUDA context...");
    let ctx = CudaContext::new(0).expect("CUDA context");
    let stream = CudaStream::new(&ctx).expect("CUDA stream");
    println!("  ✓ Context created\n");

    // Step 2: Load BOTH kernels (like GpuBatchCompressor does)
    println!("Step 2: Loading kernels (mimicking GpuBatchCompressor)...");

    // Load compression kernel first (like init_cuda)
    use trueno_gpu::kernels::lz4::Lz4WarpShuffleKernel;
    let compress_kernel = Lz4WarpShuffleKernel::new(65536);
    let compress_ptx = compress_kernel.emit_ptx();
    println!("  Compression PTX: {} bytes", compress_ptx.len());
    let _compress_module =
        CudaModule::from_ptx(&ctx, &compress_ptx).expect("Compression PTX should load");
    println!("  ✓ Compression kernel loaded");

    // Then load decompression kernel (use 65536 like GpuBatchCompressor)
    let kernel = Lz4DecompressKernel::new(65536);
    let ptx = kernel.emit_ptx();
    println!("  Decompression PTX: {} bytes", ptx.len());
    let mut module = CudaModule::from_ptx(&ctx, &ptx).expect("Decompression PTX should load");
    println!("  ✓ Decompression kernel loaded\n");

    // Step 3: Create test data - MULTIPLE pages (like the benchmark)
    const NUM_PAGES: usize = 10_000; // Test large batch like benchmark
    println!("Step 3: Creating test data ({} pages)...", NUM_PAGES);

    let mut originals: Vec<[u8; 4096]> = Vec::with_capacity(NUM_PAGES);
    let mut compressed_pages: Vec<Vec<u8>> = Vec::with_capacity(NUM_PAGES);

    // Use same pattern mix as gpu_decompress_benchmark:
    // 25% zero, 25% sequential, 50% semi-random
    for page_idx in 0..NUM_PAGES {
        let mut original = [0u8; 4096];
        match page_idx % 4 {
            0 => {} // Zero page
            1 => {
                // Sequential pattern
                for (j, byte) in original.iter_mut().enumerate() {
                    *byte = (j % 256) as u8;
                }
            }
            _ => {
                // Semi-random (deterministic PRNG)
                let mut rng = (page_idx as u64).wrapping_mul(0x0005_DEEC_E66D);
                for byte in &mut original {
                    rng = rng.wrapping_mul(0x0005_DEEC_E66D).wrapping_add(0xB);
                    *byte = (rng >> 33) as u8;
                }
            }
        }
        originals.push(original);

        // Use crate's LZ4 compression (same as benchmark)
        let compressed = lz4_compress_crate(&original).expect("Compression should succeed");
        compressed_pages.push(compressed);
    }

    let total_compressed: usize = compressed_pages.iter().map(|p| p.len()).sum();
    println!("  Original: {} bytes total", NUM_PAGES * 4096);
    println!(
        "  Compressed: {} bytes total ({:.2}x ratio)",
        total_compressed,
        (NUM_PAGES * 4096) as f64 / total_compressed as f64
    );

    // DEBUG: Print first 3 compressed pages for comparison
    println!("  First 3 compressed pages:");
    for i in 0..3.min(NUM_PAGES) {
        let bytes: Vec<String> =
            compressed_pages[i].iter().take(16).map(|b| format!("{:02x}", b)).collect();
        println!("    Page {}: [{}...] ({} bytes)", i, bytes.join(" "), compressed_pages[i].len());
    }

    // Step 4: Allocate GPU buffers
    println!("\nStep 4: Allocating GPU buffers...");

    // Input buffer at stride 4352 (max compressed size per page)
    let input_stride = 4352usize;
    let batch_size = NUM_PAGES;

    // Pack all compressed pages into flat buffer at stride positions
    let mut input_flat = vec![0u8; batch_size * input_stride];
    for (i, compressed) in compressed_pages.iter().enumerate() {
        let offset = i * input_stride;
        input_flat[offset..offset + compressed.len()].copy_from_slice(compressed);
    }

    let input_dev: GpuBuffer<u8> = GpuBuffer::from_host(&ctx, &input_flat).expect("Input buffer");
    println!(
        "  Input buffer: {} bytes ({} pages at stride {})",
        input_flat.len(),
        batch_size,
        input_stride
    );

    // Sizes buffer - one u32 per page
    let sizes: Vec<u32> = compressed_pages.iter().map(|p| p.len() as u32).collect();
    let sizes_dev: GpuBuffer<u32> = GpuBuffer::from_host(&ctx, &sizes).expect("Sizes buffer");
    println!("  Sizes buffer: {} entries", sizes.len());
    println!(
        "  Size range: {} - {} bytes",
        sizes.iter().min().unwrap(),
        sizes.iter().max().unwrap()
    );

    // Output buffer - 4096 bytes per page
    let output_dev: GpuBuffer<u8> = GpuBuffer::new(&ctx, batch_size * 4096).expect("Output buffer");
    println!("  Output buffer: {} bytes ({} pages)", batch_size * 4096, batch_size);

    // Step 5: Launch kernel
    println!("\nStep 5: Launching kernel...");
    let batch_size_u32 = batch_size as u32;

    // 1 thread per page: page_id = blockIdx.x * 256 + threadIdx.x
    // Grid: ceil(batch_size / 256) blocks
    let num_blocks = (batch_size_u32 + 255) / 256;
    let config = LaunchConfig {
        grid: (num_blocks, 1, 1), // ceil(batch_size / 256) blocks
        block: (256, 1, 1),       // 256 threads per block
        shared_mem: 0,
    };

    println!("  Grid: ({}, {}, {})", config.grid.0, config.grid.1, config.grid.2);
    println!("  Block: ({}, {}, {})", config.block.0, config.block.1, config.block.2);
    println!("  Batch size: {}", batch_size_u32);

    // Debug: print pointer values
    println!("\n  Kernel arguments:");
    println!("    input_dev ptr: {:p}", input_dev.as_kernel_arg());
    println!("    sizes_dev ptr: {:p}", sizes_dev.as_kernel_arg());
    println!("    output_dev ptr: {:p}", output_dev.as_kernel_arg());

    let mut args: [*mut c_void; 4] = [
        input_dev.as_kernel_arg(),
        sizes_dev.as_kernel_arg(),
        output_dev.as_kernel_arg(),
        &batch_size_u32 as *const u32 as *mut c_void,
    ];

    let kernel_start = std::time::Instant::now();
    unsafe {
        stream
            .launch_kernel(&mut module, "lz4_decompress", &config, &mut args)
            .expect("Kernel launch");
    }

    stream.synchronize().expect("Sync");
    let kernel_time = kernel_start.elapsed();
    let output_bytes = batch_size * 4096;
    let kernel_gbps = (output_bytes as f64) / kernel_time.as_secs_f64() / 1e9;
    println!(
        "  ✓ Kernel completed in {:.2} ms ({:.2} GB/s)\n",
        kernel_time.as_secs_f64() * 1000.0,
        kernel_gbps
    );

    // Step 6: Read output and verify all pages
    println!("Step 6: Reading output and verifying...");
    let mut output = vec![0u8; batch_size * 4096];
    output_dev.copy_to_host(&mut output).expect("D2H");

    let mut total_matches = 0;
    let mut total_mismatches = 0;
    let mut failed_pages = Vec::new();

    for page_idx in 0..NUM_PAGES {
        let out_offset = page_idx * 4096;
        let mut page_matches = 0;
        let mut page_mismatches = 0;

        for i in 0..4096 {
            if output[out_offset + i] == originals[page_idx][i] {
                page_matches += 1;
            } else {
                page_mismatches += 1;
            }
        }

        total_matches += page_matches;
        total_mismatches += page_mismatches;

        if page_mismatches > 0 {
            failed_pages.push((page_idx, page_mismatches));
        }
    }

    println!("\n  Total matches: {} / {}", total_matches, NUM_PAGES * 4096);
    println!("  Total mismatches: {}", total_mismatches);
    println!("  Failed pages: {} / {}", failed_pages.len(), NUM_PAGES);

    if !failed_pages.is_empty() && failed_pages.len() <= 10 {
        println!("\n  Failed page details:");
        for (page_idx, mismatch_count) in &failed_pages {
            println!(
                "    Page {}: {} mismatches (compressed size: {} bytes)",
                page_idx,
                mismatch_count,
                compressed_pages[*page_idx].len()
            );
        }
    }

    if total_mismatches == 0 {
        println!("\n✓ SUCCESS: All {} pages decompressed correctly!", NUM_PAGES);
    } else {
        println!("\n✗ FAILED: {} pages with mismatches", failed_pages.len());
    }
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("Requires --features cuda");
}
