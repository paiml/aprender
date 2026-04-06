//! LZ4 Compression Kernel PTX FKR (Falsification Kernel Regression) Tests
//!
//! Tests generated PTX for LZ4 compression - catches CUDA_ERROR_INVALID_PTX bugs.
//! Follows pattern from pixel_fkr.rs (Issue #67 prevention).
//!
//! # Running
//! ```bash
//! cargo test -p trueno-gpu --test lz4_fkr --features "cuda"
//! ```
//!
//! # Phases
//! - Phase 0: Static PTX analysis (no GPU required)
//! - Phase 1: Scalar baseline validation
//! - Phase 2: PTX vs Scalar comparison (requires CUDA)

#![cfg(feature = "cuda")]

use trueno_gpu::kernels::{Kernel, Lz4WarpCompressKernel};

// Import LZ4 internals directly from the module
mod lz4_internal {
    pub use trueno_gpu::kernels::lz4::{
        lz4_compress_block, lz4_decompress_block, lz4_hash, LZ4_HASH_MULT, LZ4_HASH_SIZE,
        LZ4_MIN_MATCH, PAGE_SIZE,
    };
}
use lz4_internal::*;

#[cfg(feature = "gpu-pixels")]
use jugar_probar::gpu_pixels::{validate_ptx, PtxBugClass};

// ============================================================================
// PHASE 0: PTX STATIC ANALYSIS (no GPU required)
// ============================================================================

/// LZ4-FKR-001: PTX has valid entry point
#[test]
fn lz4_fkr_ptx_has_entry_point() {
    let kernel = Lz4WarpCompressKernel::new(100);
    let ptx = kernel.emit_ptx();

    assert!(
        ptx.contains(".entry") || ptx.contains(".visible"),
        "LZ4 kernel missing PTX entry point"
    );
    assert!(
        ptx.contains("lz4_compress_warp"),
        "LZ4 kernel entry point should be named lz4_compress_warp"
    );
}

/// LZ4-FKR-002: PTX has required parameters
#[test]
fn lz4_fkr_ptx_has_parameters() {
    let kernel = Lz4WarpCompressKernel::new(100);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains("input_batch"), "Missing input_batch param");
    assert!(ptx.contains("output_batch"), "Missing output_batch param");
    assert!(ptx.contains("output_sizes"), "Missing output_sizes param");
    assert!(ptx.contains("batch_size"), "Missing batch_size param");
}

/// LZ4-FKR-003: PTX has shared memory declaration
#[test]
fn lz4_fkr_ptx_has_shared_memory() {
    let kernel = Lz4WarpCompressKernel::new(100);
    let ptx = kernel.emit_ptx();

    assert!(
        ptx.contains(".shared"),
        "LZ4 kernel must use shared memory for page data and hash table"
    );
}

/// LZ4-FKR-004: PTX has barrier synchronization
#[test]
fn lz4_fkr_ptx_has_barriers() {
    let kernel = Lz4WarpCompressKernel::new(100);
    let ptx = kernel.emit_ptx();

    let bar_count = ptx.matches("bar.sync").count();
    assert!(
        bar_count >= 3,
        "LZ4 kernel needs at least 3 barrier syncs (load, reduction, store), found {}",
        bar_count
    );
}

/// LZ4-FKR-005: PTX barrier safety analysis
#[test]
fn lz4_fkr_ptx_barrier_safety() {
    let kernel = Lz4WarpCompressKernel::new(100);
    let result = kernel.analyze_barrier_safety();

    assert!(result.is_safe, "LZ4 kernel barrier safety failed: {:?}", result.violations);
}

/// LZ4-FKR-006: PTX has hash multiply constant (0x9E3779B1 = 2654435761)
/// EXPECTED TO FAIL until LZ4 compression is implemented
#[test]
fn lz4_fkr_ptx_has_hash_multiply() {
    let kernel = Lz4WarpCompressKernel::new(100);
    let ptx = kernel.emit_ptx();

    // LZ4 hash uses Knuth multiplicative hash: 0x9E3779B1
    assert!(
        ptx.contains("2654435761") || ptx.contains("0x9e3779b1") || ptx.contains("0x9E3779B1"),
        "LZ4 kernel missing hash multiplier constant (0x9E3779B1)"
    );
}

/// LZ4-FKR-007: PTX has compression loop
/// EXPECTED TO FAIL until LZ4 compression is implemented
#[test]
fn lz4_fkr_ptx_has_compression_loop() {
    let kernel = Lz4WarpCompressKernel::new(100);
    let ptx = kernel.emit_ptx();

    assert!(
        ptx.contains("L_compress_loop")
            || ptx.contains("L_main_loop")
            || ptx.contains("L_compress"),
        "LZ4 kernel missing main compression loop label"
    );
}

/// LZ4-FKR-008: PTX has match finding logic
/// EXPECTED TO FAIL until LZ4 compression is implemented
#[test]
fn lz4_fkr_ptx_has_match_finding() {
    let kernel = Lz4WarpCompressKernel::new(100);
    let ptx = kernel.emit_ptx();

    assert!(
        ptx.contains("L_check_match") || ptx.contains("L_found_match") || ptx.contains("match"),
        "LZ4 kernel missing match finding logic"
    );
}

/// LZ4-FKR-009: PTX validates with ptxas (if available)
#[test]
fn lz4_fkr_ptx_validates_with_ptxas() {
    use std::io::Write;
    use std::process::Command;

    // Check if ptxas is available
    let ptxas_check = Command::new("which").arg("ptxas").output();
    if ptxas_check.is_err() || !ptxas_check.expect("test").status.success() {
        eprintln!("ptxas not available, skipping validation");
        return;
    }

    let kernel = Lz4WarpCompressKernel::new(100);
    let ptx = kernel.emit_ptx();

    // Write PTX to temp file
    let mut tmpfile = std::env::temp_dir();
    tmpfile.push("lz4_fkr_test.ptx");
    let mut f = std::fs::File::create(&tmpfile).expect("Failed to create temp file");
    f.write_all(ptx.as_bytes()).expect("Failed to write PTX");

    // Validate with ptxas
    let output = Command::new("ptxas")
        .args(["-arch=sm_89", tmpfile.to_str().expect("test"), "-o", "/dev/null"])
        .output()
        .expect("Failed to run ptxas");

    // Clean up
    let _ = std::fs::remove_file(&tmpfile);

    assert!(
        output.status.success(),
        "ptxas validation failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[cfg(feature = "gpu-pixels")]
mod ptx_analysis {
    use super::*;

    /// LZ4-FKR-010: No shared memory u64 addressing bug
    #[test]
    fn lz4_fkr_no_shared_mem_u64() {
        let kernel = Lz4WarpCompressKernel::new(100);
        let ptx = kernel.emit_ptx();
        let result = validate_ptx(&ptx);

        assert!(
            !result.has_bug(&PtxBugClass::SharedMemU64Addressing),
            "LZ4 kernel uses u64 for shared memory (should use u32 offset + cvta)"
        );
    }

    /// LZ4-FKR-011: No missing barrier sync bug
    #[test]
    fn lz4_fkr_no_missing_barrier() {
        let kernel = Lz4WarpCompressKernel::new(100);
        let ptx = kernel.emit_ptx();
        let result = validate_ptx(&ptx);

        assert!(
            !result.has_bug(&PtxBugClass::MissingBarrierSync),
            "LZ4 kernel missing barrier synchronization"
        );
    }
}

// ============================================================================
// PHASE 1: SCALAR BASELINE VALIDATION
// ============================================================================

/// LZ4-FKR-020: Hash function produces 12-bit output
#[test]
fn lz4_fkr_scalar_hash_12bit() {
    for val in [0u32, 1, 0x12345678, 0xFFFFFFFF, 0xDEADBEEF] {
        let h = lz4_hash(val);
        assert!(h < LZ4_HASH_SIZE, "Hash {} >= 4096 for input {}", h, val);
    }
}

/// LZ4-FKR-021: Hash function is deterministic
#[test]
fn lz4_fkr_scalar_hash_deterministic() {
    let val = 0x12345678u32;
    assert_eq!(lz4_hash(val), lz4_hash(val));
}

/// LZ4-FKR-022: Compression/decompression roundtrip - small data
#[test]
fn lz4_fkr_scalar_roundtrip_small() {
    let input = b"HELLO WORLD";
    let mut compressed = [0u8; 64];
    let mut decompressed = [0u8; 64];

    let comp_size = lz4_compress_block(input, &mut compressed).expect("test");
    let decomp_size =
        lz4_decompress_block(&compressed[..comp_size], &mut decompressed).expect("test");

    assert_eq!(decomp_size, input.len());
    assert_eq!(&decompressed[..decomp_size], input.as_slice());
}

/// LZ4-FKR-023: Compression/decompression roundtrip - repeated pattern
#[test]
fn lz4_fkr_scalar_roundtrip_repeated() {
    let input = [b'A'; 512];
    let mut compressed = [0u8; 1024];
    let mut decompressed = [0u8; 512];

    let comp_size = lz4_compress_block(&input, &mut compressed).expect("test");
    let decomp_size =
        lz4_decompress_block(&compressed[..comp_size], &mut decompressed).expect("test");

    assert_eq!(decomp_size, input.len());
    assert_eq!(&decompressed[..], &input[..]);

    // Repeated pattern should compress well
    assert!(
        comp_size < 52,
        "Repeated 512 bytes should achieve >10:1 ratio, got {} bytes",
        comp_size
    );
}

/// LZ4-FKR-024: Zero page compresses to minimal size
#[test]
fn lz4_fkr_scalar_zero_page() {
    let input = [0u8; PAGE_SIZE as usize];
    let mut compressed = [0u8; PAGE_SIZE as usize];

    let comp_size = lz4_compress_block(&input, &mut compressed).expect("test");

    assert!(comp_size < 100, "Zero page should compress to <100 bytes, got {}", comp_size);
}

/// LZ4-FKR-025: Full page roundtrip
#[test]
fn lz4_fkr_scalar_roundtrip_page() {
    let mut input = [0u8; PAGE_SIZE as usize];
    for (i, byte) in input.iter_mut().enumerate() {
        *byte = ((i * 7) % 256) as u8;
    }
    let mut compressed = [0u8; PAGE_SIZE as usize + 1024];
    let mut decompressed = [0u8; PAGE_SIZE as usize];

    let comp_size = lz4_compress_block(&input, &mut compressed).expect("test");
    let decomp_size =
        lz4_decompress_block(&compressed[..comp_size], &mut decompressed).expect("test");

    assert_eq!(decomp_size, PAGE_SIZE as usize);
    assert_eq!(&decompressed[..], &input[..]);
}

/// LZ4-FKR-026: Compression is deterministic
#[test]
fn lz4_fkr_scalar_deterministic() {
    let input = b"Deterministic compression test data pattern";
    let mut compressed1 = [0u8; 128];
    let mut compressed2 = [0u8; 128];

    let size1 = lz4_compress_block(input, &mut compressed1).expect("test");
    let size2 = lz4_compress_block(input, &mut compressed2).expect("test");

    assert_eq!(size1, size2);
    assert_eq!(&compressed1[..size1], &compressed2[..size2]);
}

/// LZ4-FKR-027: Constants are correct per LZ4 spec
#[test]
fn lz4_fkr_constants() {
    assert_eq!(LZ4_MIN_MATCH, 4, "LZ4 minimum match is 4 bytes");
    assert_eq!(LZ4_HASH_SIZE, 4096, "LZ4 hash table is 4096 entries");
    assert_eq!(LZ4_HASH_MULT, 2654435761, "LZ4 hash multiplier is 0x9E3779B1");
    assert_eq!(PAGE_SIZE, 4096, "Page size is 4KB");
}

// ============================================================================
// PHASE 2: PTX vs SCALAR COMPARISON (requires CUDA)
// ============================================================================

#[cfg(feature = "cuda")]
mod ptx_runtime {
    #[allow(unused_imports)]
    use super::*;
    use trueno_gpu::driver::CudaContext;

    fn cuda_available() -> bool {
        CudaContext::new(0).is_ok()
    }

    /// LZ4-FKR-030: GPU compressed data decompresses correctly
    #[test]
    #[ignore] // Enable after full LZ4 PTX implementation
    fn lz4_fkr_gpu_decompresses() {
        if !cuda_available() {
            eprintln!("Skipping: no CUDA device");
            return;
        }

        // TODO: Execute GPU kernel and verify output decompresses
        // This test will be enabled after PTX implementation
    }

    /// LZ4-FKR-031: GPU matches scalar compression ratio
    #[test]
    #[ignore] // Enable after full LZ4 PTX implementation
    fn lz4_fkr_gpu_matches_scalar_ratio() {
        if !cuda_available() {
            eprintln!("Skipping: no CUDA device");
            return;
        }

        // TODO: Compare GPU vs scalar compression ratios
        // Should be within 5% of each other
    }
}

// ============================================================================
// SUMMARY
// ============================================================================

#[test]
fn lz4_fkr_summary() {
    println!();
    println!("========================================");
    println!("  LZ4 Compression Kernel FKR Suite");
    println!("========================================");
    println!();
    println!("  Phase 0 - PTX Static Analysis:");
    println!("    - entry_point, parameters, shared_memory");
    println!("    - barriers, barrier_safety");
    println!("    - hash_multiply, compression_loop, match_finding");
    println!("    - ptxas_validation");
    println!();
    println!("  Phase 1 - Scalar Baseline:");
    println!("    - hash_12bit, hash_deterministic");
    println!("    - roundtrip_small, roundtrip_repeated");
    println!("    - zero_page, roundtrip_page");
    println!("    - deterministic, constants");
    println!();
    println!("  Phase 2 - PTX Runtime (CUDA):");
    println!("    - gpu_decompresses [PENDING]");
    println!("    - gpu_matches_scalar_ratio [PENDING]");
    println!();
    println!("========================================");
}
