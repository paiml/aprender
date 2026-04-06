use super::*;

#[test]
fn test_softmax_kernel_name() {
    let kernel = SoftmaxKernel::new(4096);
    assert_eq!(kernel.name(), "softmax_warp_shuffle");

    let kernel_shared = SoftmaxKernel::new(4096).without_warp_shuffle();
    assert_eq!(kernel_shared.name(), "softmax_shared");
}

#[test]
fn test_long_row_softmax_ptx_generation() {
    let kernel = LongRowSoftmaxKernel::new(1500);
    let ptx = kernel.emit_ptx();

    // Verify kernel name
    assert!(ptx.contains("softmax_long_row"), "Missing kernel name");

    // Verify parameters
    assert!(ptx.contains(".param .u64 input_ptr"), "Missing input_ptr param");
    assert!(ptx.contains(".param .u64 output_ptr"), "Missing output_ptr param");
    assert!(ptx.contains(".param .u32 row_size"), "Missing row_size param");

    // Verify has grid-stride loops (multiple branch labels)
    assert!(ptx.contains("max_loop:"), "Missing max_loop label");
    assert!(ptx.contains("max_loop_done:"), "Missing max_loop_done label");
    assert!(ptx.contains("sum_loop:"), "Missing sum_loop label");
    assert!(ptx.contains("write_loop:"), "Missing write_loop label");

    // Verify has barrier syncs for inter-warp reduction
    assert!(ptx.contains("bar.sync"), "Missing barrier sync");

    // Verify has warp shuffles for intra-warp reduction
    assert!(
        ptx.contains("shfl") || ptx.contains("shfl.down") || ptx.contains("shfl.sync.down"),
        "Missing warp shuffle"
    );

    // Print first 300 lines for debugging
    for (i, line) in ptx.lines().enumerate().take(300) {
        println!("{:4}: {}", i + 1, line);
    }
}

#[test]
fn test_softmax_ptx_generation() {
    let kernel = SoftmaxKernel::new(4096);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains(".param .u64 input_ptr"));
    assert!(ptx.contains(".param .u64 output_ptr"));
    assert!(ptx.contains(".param .u32 length"));
}

#[test]
fn test_softmax_shared_memory() {
    let kernel = SoftmaxKernel::new(4096).without_warp_shuffle();
    let ptx_kernel = kernel.build_ptx();
    assert!(ptx_kernel.shared_memory_bytes() > 0);
}

#[test]
fn test_softmax_warp_shuffle_ptx() {
    let kernel = SoftmaxKernel::new(32);
    let ptx = kernel.emit_ptx();

    // Verify warp shuffle operations are present
    assert!(ptx.contains("shfl") || ptx.contains("shfl.down"));

    // Verify max operation
    assert!(ptx.contains("max.f32"));

    // Verify exp operation (ex2)
    assert!(ptx.contains("ex2.f32") || ptx.contains("ex2"));

    // Verify division
    assert!(ptx.contains("div.rn.f32")); // div requires rounding mode for floats

    // Verify memory operations
    assert!(ptx.contains("ld.global.f32"));
    assert!(ptx.contains("st.global.f32"));
}

#[test]
fn test_softmax_shared_memory_ptx() {
    let kernel = SoftmaxKernel::new(256).without_warp_shuffle();
    let ptx = kernel.emit_ptx();

    // Verify shared memory usage
    assert!(ptx.contains("ld.shared.f32") || ptx.contains("ld.f32"));
    assert!(ptx.contains("st.shared.f32") || ptx.contains("st.f32"));

    // Verify barrier synchronization
    assert!(ptx.contains("bar"));

    // Verify exp and divide
    assert!(ptx.contains("ex2.f32") || ptx.contains("ex2"));
    assert!(ptx.contains("div.rn.f32")); // div requires rounding mode for floats
}

#[test]
fn test_softmax_kernel_variants() {
    let warp_kernel = SoftmaxKernel::new(32);
    let shared_kernel = SoftmaxKernel::new(256).without_warp_shuffle();

    // Both should produce valid PTX
    let warp_ptx = warp_kernel.emit_ptx();
    let shared_ptx = shared_kernel.emit_ptx();

    assert!(!warp_ptx.is_empty());
    assert!(!shared_ptx.is_empty());

    // Verify different kernel names in output
    assert!(warp_ptx.contains("softmax_warp_shuffle"));
    assert!(shared_ptx.contains("softmax_shared"));
}

#[test]
fn test_softmax_numerical_stability() {
    // Verify the implementation uses numerically stable softmax
    // (subtracts max before exp)
    let kernel = SoftmaxKernel::new(32);
    let ptx = kernel.emit_ptx();

    // Should have sub operation (for val - max)
    assert!(ptx.contains("sub.f32"));

    // Should have mul for log2(e) scaling
    assert!(ptx.contains("mul.f32"));
}

// =========================================================================
// SATD REMEDIATION TESTS (EXTREME TDD)
// These tests verify the max-reduce loop bug is fixed.
// Falsifiable claims per Popperian methodology.
// =========================================================================

#[test]
fn test_shared_max_reduce_loop_iterates() {
    // FALSIFIABLE CLAIM: Max-reduce loop iterates multiple times for full reduction
    // The SATD bug causes it to exit after one iteration.
    let kernel = SoftmaxKernel::new(256).without_warp_shuffle();
    let ptx = kernel.emit_ptx();

    // The PTX should contain a branch back to max_reduce_loop
    // If it only branches to max_reduce_done, the reduction is incomplete
    let has_loop_back = ptx.contains("bra max_reduce_loop") || ptx.contains("bra\tmax_reduce_loop");

    assert!(
        has_loop_back,
        "FALSIFIED: Max-reduce loop does not branch back to loop start. \
         Found 'bra max_reduce_done' instead of 'bra max_reduce_loop'. \
         This means max reduction only runs once, producing wrong max."
    );
}

#[test]
fn test_shared_max_reduce_stride_halves() {
    // FALSIFIABLE CLAIM: Max-reduce stride is halved each iteration (128->64->32->...)
    // If stride is not updated, loop will be infinite or wrong.
    let kernel = SoftmaxKernel::new(256).without_warp_shuffle();
    let ptx = kernel.emit_ptx();

    // Look for stride manipulation - should see shr.b32 (PTX requires .b32 for shifts) or div
    let has_stride_update =
        ptx.contains("shr.b32") || ptx.contains("shr.u32") || ptx.contains("div.u32");

    assert!(
        has_stride_update,
        "FALSIFIED: Max-reduce stride is not halved. \
         Expected shr.b32, shr.u32 or div.u32 for stride = stride / 2. \
         Without this, tree reduction cannot work correctly."
    );
}

#[test]
fn test_shared_sum_reduce_implemented() {
    // FALSIFIABLE CLAIM: Sum reduction is fully implemented, not a placeholder
    // The SATD bug has: `let block_sum = sum_val; // Placeholder`
    let kernel = SoftmaxKernel::new(256).without_warp_shuffle();
    let ptx = kernel.emit_ptx();

    // Verify sum reduction loop structure exists
    // Should have: sum_reduce_loop label, branch back, and sum_reduce_done label
    let has_sum_loop = ptx.contains("sum_reduce_loop");
    let has_sum_done = ptx.contains("sum_reduce_done");
    let has_loop_back = ptx.contains("bra sum_reduce_loop") || ptx.contains("bra\tsum_reduce_loop");

    assert!(
        has_sum_loop && has_sum_done && has_loop_back,
        "FALSIFIED: Sum reduction loop structure is incomplete. \
         has_sum_loop={}, has_sum_done={}, has_loop_back={}. \
         A proper tree reduction needs a complete loop structure.",
        has_sum_loop,
        has_sum_done,
        has_loop_back
    );
}
