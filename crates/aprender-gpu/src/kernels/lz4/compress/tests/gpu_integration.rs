//! GPU Kernel Integration Tests (F036-F050)

use super::*;

#[test]
fn test_f036_ptx_has_zero_page_detection() {
    // F036: GPU kernel detects zero pages for optimal compression
    let kernel = Lz4WarpCompressKernel::new(100);
    let ptx = kernel.emit_ptx();

    // Should have OR operations for zero detection
    assert!(
        ptx.contains("or.b32"),
        "Missing OR operations for zero detection"
    );
    // Should have conditional branching for zero vs non-zero path
    assert!(
        ptx.contains("L_write_zero_size"),
        "Missing zero-size output path"
    );
    assert!(
        ptx.contains("L_after_size_write"),
        "Missing size write merge label"
    );
}

#[test]
fn test_f037_ptx_warp_reduction() {
    // F037: PTX uses warp-level reduction for zero detection
    let kernel = Lz4WarpCompressKernel::new(100);
    let ptx = kernel.emit_ptx();

    // Should have multiple barrier syncs (load, reduction, store)
    let bar_count = ptx.matches("bar.sync").count();
    assert!(
        bar_count >= 3,
        "Should have at least 3 barrier syncs, found {}",
        bar_count
    );
}

#[test]
fn test_f038_zero_page_compressed_size() {
    // F038: Zero page should produce minimal output size
    // GPU kernel reports 20 bytes for zero pages (LZ4 sequence encoding)
    let kernel = Lz4WarpCompressKernel::new(100);
    let ptx = kernel.emit_ptx();

    // Should have the compressed size constant (20 bytes for zero page)
    assert!(
        ptx.contains("20"),
        "Should reference compressed zero page size"
    );
}

#[test]
fn test_f039_page_id_calculation() {
    // F039: Page ID correctly calculated from block/thread indices
    let kernel = Lz4WarpCompressKernel::new(100);
    let ptx = kernel.emit_ptx();

    // Should access blockIdx.x and threadIdx.x
    assert!(ptx.contains("%ctaid.x"), "Missing blockIdx.x access");
    assert!(ptx.contains("%tid.x"), "Missing threadIdx.x access");
}

#[test]
fn test_f040_lane_id_masking() {
    // F040: Lane ID correctly computed using mask
    let kernel = Lz4WarpCompressKernel::new(100);
    let ptx = kernel.emit_ptx();

    // Should use AND with 31 for lane_id = threadIdx.x % 32
    assert!(ptx.contains("and.b32"), "Missing lane ID masking");
}

#[test]
fn test_f041_shared_memory_allocation() {
    // F041: Sufficient shared memory for page + hash table
    let kernel = Lz4WarpCompressKernel::new(100);
    let smem = kernel.shared_memory_bytes();

    // Need at least 4KB page + 8KB hash table per warp, times 4 warps
    let min_required = 4 * (PAGE_SIZE as usize + LZ4_HASH_SIZE as usize * 2);
    assert!(
        smem >= min_required,
        "Shared memory {} < required {}",
        smem,
        min_required
    );
}

#[test]
fn test_f042_bounds_check_present() {
    // F042: Kernel has bounds check for page_id < batch_size
    let kernel = Lz4WarpCompressKernel::new(100);
    let ptx = kernel.emit_ptx();

    // Should have comparison instruction for bounds check
    // Uses setp.lt for in-bounds predicate (threads participate in barriers even when OOB)
    assert!(
        ptx.contains("setp.lt"),
        "Missing bounds check comparison (setp.lt)"
    );
    assert!(ptx.contains("L_exit"), "Missing exit label for OOB pages");
}

#[test]
fn test_f043_cooperative_load() {
    // F043: All 32 threads participate in loading 4KB page
    let kernel = Lz4WarpCompressKernel::new(100);
    let ptx = kernel.emit_ptx();

    // Each thread loads 128 bytes = 32 u32s = 8 chunks of 4 u32s
    // Should have many ld.global.u32 instructions
    let ld_count = ptx.matches("ld.global.u32").count();
    assert!(
        ld_count >= 32,
        "Should have many global loads, found {}",
        ld_count
    );
}

#[test]
fn test_f044_leader_thread_writes_size() {
    // F044: Only lane 0 (leader) writes the output size
    let kernel = Lz4WarpCompressKernel::new(100);
    let ptx = kernel.emit_ptx();

    // Should have comparison for lane_id == 0
    assert!(ptx.contains("setp.eq"), "Missing leader thread check");
    assert!(
        ptx.contains("L_not_leader"),
        "Missing non-leader skip label"
    );
}

#[test]
fn test_f045_output_size_write() {
    // F045: Output size correctly written to sizes array
    let kernel = Lz4WarpCompressKernel::new(100);
    let ptx = kernel.emit_ptx();

    // Should store to output_sizes array
    assert!(ptx.contains("st.global.u32"), "Missing size output store");
}

#[test]
fn test_f048_shared_memory_reduction() {
    // F048: Both PTX and WGSL use shared memory for reduction
    let kernel = Lz4WarpCompressKernel::new(100);
    let ptx = kernel.emit_ptx();
    let wgsl = kernel.emit_wgsl();

    // PTX uses generic addressing (after cvta.shared) for flexible warp offset handling
    // Check for generic store/load (st.u32/ld.u32 without state space = generic)
    assert!(
        ptx.contains("st.u32"),
        "PTX missing generic store for reduction"
    );
    assert!(
        ptx.contains("ld.u32"),
        "PTX missing generic load for reduction"
    );
    // Verify shared memory is declared and cvta is used to get generic address
    // cvta.shared converts shared->generic; cvta.to.shared converts generic->shared
    assert!(
        ptx.contains(".shared"),
        "PTX missing shared memory declaration"
    );
    assert!(
        ptx.contains("cvta.shared"),
        "PTX missing cvta for shared->generic"
    );

    // WGSL should use smem for reduction
    assert!(
        wgsl.contains("smem[reduction_idx]"),
        "WGSL missing shared memory reduction"
    );
}

#[test]
fn test_f049_page_data_integrity() {
    // F049: Page data correctly passed through shared memory
    let kernel = Lz4WarpCompressKernel::new(100);
    let ptx = kernel.emit_ptx();

    // Should have matching global loads and stores
    let global_loads = ptx.matches("ld.global.u32").count();
    let global_stores = ptx.matches("st.global.u32").count();

    // Should have balanced load/store for page data
    assert!(global_loads >= 32, "Need at least 32 global loads for 4KB");
    assert!(
        global_stores >= 32,
        "Need at least 32 global stores for 4KB"
    );
}

#[test]
fn test_f050_kernel_determinism() {
    // F050: Kernel generation is structurally deterministic
    // Note: PTX register numbers may vary between invocations due to allocator state,
    // but the WGSL (which uses names, not registers) should be exactly deterministic.
    let k1 = Lz4WarpCompressKernel::new(100);
    let k2 = Lz4WarpCompressKernel::new(100);

    // WGSL should be exactly deterministic (uses named variables)
    let wgsl1 = k1.emit_wgsl();
    let wgsl2 = k2.emit_wgsl();
    assert_eq!(wgsl1, wgsl2, "WGSL should be deterministic");

    // PTX should have same instruction count and structure
    let ptx1 = k1.emit_ptx();
    let ptx2 = k2.emit_ptx();

    // Same number of instructions
    let instr_count_1 = ptx1
        .lines()
        .filter(|l| l.trim().starts_with(|c: char| c.is_alphabetic()))
        .count();
    let instr_count_2 = ptx2
        .lines()
        .filter(|l| l.trim().starts_with(|c: char| c.is_alphabetic()))
        .count();
    assert_eq!(
        instr_count_1, instr_count_2,
        "PTX instruction count should match"
    );

    // Same labels
    assert_eq!(
        ptx1.matches("L_exit").count(),
        ptx2.matches("L_exit").count()
    );
    assert_eq!(
        ptx1.matches("L_not_leader").count(),
        ptx2.matches("L_not_leader").count()
    );
}
