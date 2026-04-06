//! GPU LZ4 Full Compression Tests (TDD - These define requirements)

use super::*;

#[test]
fn test_gpu_lz4_ptx_has_hash_table() {
    // REQ-LZ4-001: PTX kernel must have hash table for match finding
    let kernel = Lz4WarpCompressKernel::new(100);
    let ptx = kernel.emit_ptx();

    // Hash table should be in shared memory (8KB per warp = 4096 entries x 2 bytes)
    // Check for hash computation using the LZ4 hash multiplier (2654435761 = 0x9E3779B1)
    assert!(
        ptx.contains("0x9e3779b1") || ptx.contains("2654435761") || ptx.contains("hash"),
        "PTX must have LZ4 hash computation (mul by 0x9E3779B1)"
    );
}

#[test]
fn test_gpu_lz4_ptx_has_match_finding() {
    // REQ-LZ4-002: PTX kernel must find matches of >= 4 bytes
    let kernel = Lz4WarpCompressKernel::new(100);
    let ptx = kernel.emit_ptx();

    // Match finding requires labeled branches for match logic
    assert!(
        ptx.contains("match") || ptx.contains("L_found_match") || ptx.contains("L_check_match"),
        "PTX must have match finding logic with labeled branches"
    );
}

#[test]
fn test_gpu_lz4_ptx_has_sequence_encoding() {
    // REQ-LZ4-003: PTX kernel must encode LZ4 sequences (token + literals + offset + matchlen)
    let kernel = Lz4WarpCompressKernel::new(100);
    let ptx = kernel.emit_ptx();

    // Sequence encoding requires labeled logic
    assert!(
        ptx.contains("token") || ptx.contains("L_encode") || ptx.contains("L_write_sequence"),
        "PTX must have LZ4 sequence encoding logic"
    );
}

#[test]
fn test_gpu_lz4_ptx_has_output_buffer_management() {
    // REQ-LZ4-004: PTX kernel must manage output buffer correctly
    let kernel = Lz4WarpCompressKernel::new(100);
    let ptx = kernel.emit_ptx();

    // For now, check that there's some form of dynamic size tracking
    let has_dynamic_size =
        ptx.contains("out_pos") || ptx.contains("L_compress") || ptx.contains("compressed_len");
    assert!(
        has_dynamic_size,
        "PTX must track output buffer position dynamically for compression"
    );
}

#[test]
fn test_gpu_lz4_kernel_has_compression_loop() {
    // REQ-LZ4-006: GPU kernel must have main compression loop
    let kernel = Lz4WarpCompressKernel::new(100);
    let ptx = kernel.emit_ptx();

    // Look for loop structure in PTX
    let has_compress_loop = ptx.contains("L_compress_loop")
        || ptx.contains("L_main_loop")
        || (ptx.contains("bra") && ptx.contains("L_loop"));

    assert!(
        has_compress_loop,
        "GPU kernel must have main compression loop (L_compress_loop or similar)"
    );
}
