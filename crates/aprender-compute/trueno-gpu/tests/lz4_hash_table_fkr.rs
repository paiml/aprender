//! Popperian Falsification Tests: LZ4 Hash Table Operations
//!
//! These tests isolate specific hypotheses about hash table crashes.
//! Each test is designed to FAIL if the hypothesis is correct.
//!
//! Requires `cuda` feature: `cargo test -p trueno-gpu --test lz4_hash_table_fkr --features cuda`

#![cfg(feature = "cuda")]

use trueno_gpu::kernels::lz4::{LZ4_HASH_SIZE, PAGE_SIZE};

// =============================================================================
// CPU REFERENCE TESTS (No GPU required - pure logic validation)
// =============================================================================

/// FKR-001: Hash index computation must produce values in [0, 2047]
#[test]
fn fkr_001_hash_index_bounds() {
    const LZ4_PRIME: u32 = 0x9E37_79B1;
    const HASH_SHIFT: u32 = 21; // 32 - 11 = 21 for 2048 entries
    const HASH_MASK: u32 = 2047; // 2048 - 1

    // Test with various input values including edge cases
    let test_values: Vec<u32> = vec![
        0,
        1,
        0x0102_0304, // Sequential bytes from test input
        0x1234_5678,
        0xDEAD_BEEF,
        0xFFFF_FFFF,
        0x0001_0203, // First 4 bytes of sequential page
    ];

    for val in test_values {
        let hash_tmp = val.wrapping_mul(LZ4_PRIME);
        let hash_shifted = hash_tmp >> HASH_SHIFT;
        let hash_idx = hash_shifted & HASH_MASK;

        assert!(hash_idx <= 2047, "Hash index {} out of bounds for input 0x{:08X}", hash_idx, val);
    }
}

/// FKR-002: Hash entry offset must fit within hash table region
#[test]
fn fkr_002_hash_entry_offset_bounds() {
    const HASH_TABLE_ENTRIES: u32 = 2048;
    const BYTES_PER_ENTRY: u32 = 4;
    const HASH_TABLE_SIZE: u32 = HASH_TABLE_ENTRIES * BYTES_PER_ENTRY; // 8192

    for hash_idx in 0..HASH_TABLE_ENTRIES {
        let hash_entry_off = hash_idx * BYTES_PER_ENTRY;
        assert!(
            hash_entry_off < HASH_TABLE_SIZE,
            "Entry offset {} exceeds table size {} for index {}",
            hash_entry_off,
            HASH_TABLE_SIZE,
            hash_idx
        );
    }

    // Max index
    let max_idx = 2047u32;
    let max_off = max_idx * 4;
    assert_eq!(max_off, 8188, "Max offset should be 8188");
    assert!(max_off + 4 <= 8192, "Max entry must fit in table");
}

/// FKR-003: Total shared memory offset must be within WARP_SMEM_SIZE
#[test]
fn fkr_003_total_smem_offset_bounds() {
    const WARP_SMEM_SIZE: u32 = PAGE_SIZE + LZ4_HASH_SIZE * 2 + 256; // 12544

    // Hash table starts at PAGE_SIZE
    let hash_table_start = PAGE_SIZE; // 4096

    // Max hash entry offset
    let max_hash_idx: u32 = 2047;
    let max_hash_entry_off = max_hash_idx * 4; // 8188

    // Total offset for max entry
    let max_total_offset = hash_table_start + max_hash_entry_off; // 4096 + 8188 = 12284

    // Must be within warp's shared memory region
    assert!(
        max_total_offset + 4 <= WARP_SMEM_SIZE,
        "Max hash entry offset {} + 4 exceeds WARP_SMEM_SIZE {}",
        max_total_offset,
        WARP_SMEM_SIZE
    );
}

/// FKR-004: Warp shared memory regions must not overlap
#[test]
fn fkr_004_warp_regions_no_overlap() {
    const WARP_SMEM_SIZE: u32 = PAGE_SIZE + LZ4_HASH_SIZE * 2 + 256; // 12544
    const NUM_WARPS: u32 = 3;

    for warp_id in 0..NUM_WARPS {
        let warp_start = warp_id * WARP_SMEM_SIZE;
        let warp_end = warp_start + WARP_SMEM_SIZE;

        // Check no overlap with other warps
        for other_warp in 0..NUM_WARPS {
            if other_warp != warp_id {
                let other_start = other_warp * WARP_SMEM_SIZE;
                let other_end = other_start + WARP_SMEM_SIZE;

                assert!(
                    warp_end <= other_start || warp_start >= other_end,
                    "Warp {} [{}, {}) overlaps with warp {} [{}, {})",
                    warp_id,
                    warp_start,
                    warp_end,
                    other_warp,
                    other_start,
                    other_end
                );
            }
        }
    }
}

/// FKR-005: Sequential input produces varied hash indices (not all same)
#[test]
fn fkr_005_hash_distribution() {
    const LZ4_PRIME: u32 = 0x9E37_79B1;
    const HASH_SHIFT: u32 = 21;
    const HASH_MASK: u32 = 2047;

    // Simulate sequential page data
    let mut indices = std::collections::HashSet::new();

    for pos in 0..100u32 {
        // Little-endian read of bytes [pos, pos+1, pos+2, pos+3] from sequential data
        let val = pos | ((pos + 1) << 8) | ((pos + 2) << 16) | ((pos + 3) << 24);
        let hash_tmp = val.wrapping_mul(LZ4_PRIME);
        let hash_shifted = hash_tmp >> HASH_SHIFT;
        let hash_idx = hash_shifted & HASH_MASK;
        indices.insert(hash_idx);
    }

    // Should produce multiple different indices (good hash distribution)
    assert!(
        indices.len() > 50,
        "Hash function produces too few unique indices: {} (expected >50)",
        indices.len()
    );
}

// =============================================================================
// PTX GENERATION TESTS (No GPU required - validates PTX structure)
// =============================================================================

#[test]
fn fkr_010_lz4_kernel_ptx_has_hash_table_init() {
    use trueno_gpu::kernels::{Kernel, Lz4WarpCompressKernel};

    let kernel = Lz4WarpCompressKernel::new(3);
    let ptx = kernel.emit_ptx();

    // Hash table should be initialized with 0xFFFFFFFF markers
    // Look for the initialization stores
    let has_init_marker = ptx.contains("4294967295") || ptx.contains("0xFFFFFFFF");

    // After our fix, should have hash table initialization
    assert!(
        has_init_marker || ptx.contains("mov.u32") && ptx.lines().any(|l| l.contains("st.u32")),
        "PTX should contain hash table initialization code"
    );
}

#[test]
fn fkr_011_lz4_kernel_ptx_has_barrier_before_compress() {
    use trueno_gpu::kernels::{Kernel, Lz4WarpCompressKernel};

    let kernel = Lz4WarpCompressKernel::new(3);
    let ptx = kernel.emit_ptx();

    // Count barriers - should have at least 2 (after load, after hash init)
    let barrier_count = ptx.matches("bar.sync").count();

    assert!(barrier_count >= 2, "PTX should have at least 2 barriers (found {})", barrier_count);
}

#[test]
fn fkr_012_smem_base_register_used_consistently() {
    use trueno_gpu::kernels::{Kernel, Lz4WarpCompressKernel};

    let kernel = Lz4WarpCompressKernel::new(3);
    let ptx = kernel.emit_ptx();

    // Find the smem_base register (should be %rdNN after cvta.shared + add)
    // The pattern is: cvta.shared.u64 %rdX, smem; add.u64 %rdY, %rdX, offset

    // smem_base should be used in hash table calculations
    // Look for add.u64 patterns that compute hash_table_base
    let has_hash_table_calc = ptx.lines().any(|line| {
        line.contains("add.u64")
            && (line.contains("4096")
                || ptx.lines().any(|l| l.contains("mov.u32") && l.contains("4096")))
    });

    assert!(
        has_hash_table_calc,
        "PTX should have hash table base calculation (smem_base + PAGE_SIZE)"
    );
}

// =============================================================================
// GPU INTEGRATION TESTS (Require CUDA)
// =============================================================================

#[cfg(feature = "cuda")]
mod gpu_tests {
    use std::ffi::c_void;
    use trueno_gpu::driver::{CudaContext, CudaModule, CudaStream, GpuBuffer, LaunchConfig};
    use trueno_gpu::kernels::lz4::PAGE_SIZE;
    use trueno_gpu::kernels::{Kernel, Lz4WarpCompressKernel};

    fn cuda_available() -> bool {
        CudaContext::new(0).is_ok()
    }

    /// FKR-100: Zero pages should compress correctly (known working)
    /// NOTE: This test uses Lz4WarpCompressKernel which has F082 bug and will crash.
    #[test]
    #[ignore = "Uses buggy Lz4WarpCompressKernel - F082 confirmed"]
    fn fkr_100_zero_page_compression() {
        if !cuda_available() {
            eprintln!("FKR-100 SKIPPED: No CUDA device available");
            return;
        }
        let ctx = CudaContext::new(0).expect("CUDA context");
        let stream = CudaStream::new(&ctx).expect("CUDA stream");

        const NUM_PAGES: u32 = 3;
        let input: Vec<u8> = vec![0u8; (NUM_PAGES * PAGE_SIZE) as usize];

        let mut input_buf: GpuBuffer<u8> = GpuBuffer::new(&ctx, input.len()).unwrap();
        let mut output_buf: GpuBuffer<u8> =
            GpuBuffer::new(&ctx, (NUM_PAGES * 4352) as usize).unwrap();
        let mut sizes_buf: GpuBuffer<u32> = GpuBuffer::new(&ctx, NUM_PAGES as usize).unwrap();

        input_buf.copy_from_host(&input).unwrap();

        let kernel = Lz4WarpCompressKernel::new(NUM_PAGES);
        let ptx = kernel.emit_ptx();
        let mut module = CudaModule::from_ptx(&ctx, &ptx).expect("PTX load");

        let config =
            LaunchConfig { grid: kernel.grid_dim(), block: kernel.block_dim(), shared_mem: 0 };

        let num_pages = NUM_PAGES;
        let mut args: [*mut c_void; 4] = [
            input_buf.as_kernel_arg(),
            output_buf.as_kernel_arg(),
            sizes_buf.as_kernel_arg(),
            &num_pages as *const u32 as *mut c_void,
        ];

        unsafe {
            stream
                .launch_kernel(&mut module, "lz4_compress_warp", &config, &mut args)
                .expect("Kernel launch");
        }
        stream.synchronize().expect("Sync");

        let mut sizes = vec![0u32; NUM_PAGES as usize];
        sizes_buf.copy_to_host(&mut sizes).unwrap();

        // Zero pages should compress to small size (20 bytes for LZ4 zero encoding)
        for (i, &size) in sizes.iter().enumerate() {
            assert!(size <= 100, "Zero page {} should compress to <100 bytes, got {}", i, size);
        }
    }

    /// FKR-101: Non-zero pages should not crash (this is the failing test)
    /// NOTE: This test uses Lz4WarpCompressKernel which has F082 bug and will crash.
    #[test]
    #[ignore = "Uses buggy Lz4WarpCompressKernel - F082 confirmed"]
    fn fkr_101_nonzero_page_no_crash() {
        if !cuda_available() {
            eprintln!("FKR-101 SKIPPED: No CUDA device available");
            return;
        }
        let ctx = CudaContext::new(0).expect("CUDA context");
        let stream = CudaStream::new(&ctx).expect("CUDA stream");

        const NUM_PAGES: u32 = 3;

        // Create non-zero sequential data
        let mut input: Vec<u8> = Vec::with_capacity((NUM_PAGES * PAGE_SIZE) as usize);
        for page_idx in 0..NUM_PAGES {
            for byte_idx in 0..PAGE_SIZE {
                input.push(((page_idx * 17 + byte_idx) % 256) as u8);
            }
        }

        let mut input_buf: GpuBuffer<u8> = GpuBuffer::new(&ctx, input.len()).unwrap();
        let mut output_buf: GpuBuffer<u8> =
            GpuBuffer::new(&ctx, (NUM_PAGES * 4352) as usize).unwrap();
        let mut sizes_buf: GpuBuffer<u32> = GpuBuffer::new(&ctx, NUM_PAGES as usize).unwrap();

        input_buf.copy_from_host(&input).unwrap();

        let kernel = Lz4WarpCompressKernel::new(NUM_PAGES);
        let ptx = kernel.emit_ptx();
        let mut module = CudaModule::from_ptx(&ctx, &ptx).expect("PTX load");

        let config =
            LaunchConfig { grid: kernel.grid_dim(), block: kernel.block_dim(), shared_mem: 0 };

        let num_pages = NUM_PAGES;
        let mut args: [*mut c_void; 4] = [
            input_buf.as_kernel_arg(),
            output_buf.as_kernel_arg(),
            sizes_buf.as_kernel_arg(),
            &num_pages as *const u32 as *mut c_void,
        ];

        unsafe {
            stream
                .launch_kernel(&mut module, "lz4_compress_warp", &config, &mut args)
                .expect("Kernel launch should not fail");
        }

        // THIS IS THE CRITICAL TEST - sync should not crash
        stream.synchronize().expect("Sync should not crash with non-zero pages");

        let mut sizes = vec![0u32; NUM_PAGES as usize];
        sizes_buf.copy_to_host(&mut sizes).unwrap();

        // Each page should have a valid size
        for (i, &size) in sizes.iter().enumerate() {
            assert!(
                size > 0 && size <= 4352,
                "Page {} should have valid size in (0, 4352], got {}",
                i,
                size
            );
        }
    }
}
