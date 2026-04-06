use super::*;

#[test]
fn test_count_params_basic() {
    let ptx = r#"
.visible .entry test(
    .param .u64 a_ptr,
    .param .u64 b_ptr,
    .param .u64 c_ptr
) {
    ret;
}
"#;
    assert_eq!(count_params(ptx), 3);
}

#[test]
fn test_extract_shared_memory_bytes() {
    let ptx = "    .shared .align 16 .b8 smem[32];";
    assert_eq!(extract_shared_memory_bytes(ptx), Some(32));

    let ptx_none = "    .reg .f32 %f<10>;";
    assert_eq!(extract_shared_memory_bytes(ptx_none), None);
}

#[test]
fn test_extract_loop_labels() {
    let ptx = r#"
sum_loop:
    add.u32 %r6, %r6, 256;
    bra sum_loop;
sum_loop_end:
norm_loop:
    bra norm_loop;
exit:
    ret;
"#;
    let labels = extract_loop_labels(ptx);
    assert_eq!(labels, vec!["sum_loop", "sum_loop_end", "norm_loop"]);
}

#[test]
fn test_has_batch_dispatch() {
    // Grid.y dispatch
    assert!(has_batch_dispatch("    mov.u32 %r1, %ctaid.y;"));
    // Register unroll dispatch (m_dim parameter)
    assert!(has_batch_dispatch("    .param .u32 m_dim"));
    // Neither
    assert!(!has_batch_dispatch("    mov.u32 %r1, %ctaid.x;"));
}

#[test]
fn test_batch_dispatch_strategies() {
    assert!(has_grid_y_dispatch("    mov.u32 %r1, %ctaid.y;"));
    assert!(!has_grid_y_dispatch("    .param .u32 m_dim"));
    assert!(has_register_unroll_dispatch("    .param .u32 m_dim"));
    assert!(!has_register_unroll_dispatch("    mov.u32 %r1, %ctaid.y;"));
}

#[test]
fn test_has_u64_shared_memory_addressing() {
    // Bad: u64 register for shared memory
    assert!(has_u64_shared_memory_addressing("    st.shared.f32 [%rd3], %f0;"));
    // Good: u32 register for shared memory
    assert!(!has_u64_shared_memory_addressing("    st.shared.f32 [%r3], %f0;"));
}

#[test]
fn test_validate_parity_matching_kernels() {
    let single = r#"
.version 8.0
.target sm_89
.address_size 64
.visible .entry rmsnorm(
    .param .u64 input_ptr,
    .param .u64 output_ptr,
    .param .u64 gamma_ptr
) {
    .shared .align 16 .b8 smem[32];
    mov.u32 %r0, %tid.x;
sum_loop:
    bra sum_loop;
sum_loop_end:
norm_loop:
    bra norm_loop;
exit:
    ret;
}
"#;
    let batched = r#"
.version 8.0
.target sm_89
.address_size 64
.visible .entry batched_rmsnorm(
    .param .u64 input_ptr,
    .param .u64 output_ptr,
    .param .u64 gamma_ptr
) {
    .shared .align 16 .b8 smem[32];
    mov.u32 %r0, %tid.x;
    mov.u32 %r1, %ctaid.y;
sum_loop:
    bra sum_loop;
sum_loop_end:
norm_loop:
    bra norm_loop;
exit:
    ret;
}
"#;
    let result = validate_parity(single, batched, "rmsnorm", "batched_rmsnorm");
    assert!(result.is_compatible, "Should be compatible: {:?}", result.violations);
}

#[test]
fn test_validate_parity_param_mismatch() {
    let single = r#"
.visible .entry test(
    .param .u64 a,
    .param .u64 b
) { ret; }
"#;
    let batched = r#"
.visible .entry test_batched(
    .param .u64 a,
    .param .u64 b,
    .param .u32 batch_size
) {
    mov.u32 %r1, %ctaid.y;
    ret;
}
"#;
    let result = validate_parity(single, batched, "test", "test_batched");
    assert!(!result.is_compatible);
    assert!(result
        .violations
        .iter()
        .any(|v| v.kind == ParityViolationKind::ParameterCountMismatch));
}

#[test]
fn test_validate_parity_missing_ctaid_y() {
    let single = r#"
.visible .entry test(
    .param .u64 a
) { ret; }
"#;
    let batched = r#"
.visible .entry test_batched(
    .param .u64 a
) { ret; }
"#;
    let result = validate_parity(single, batched, "test", "test_batched");
    assert!(!result.is_compatible);
    assert!(result.violations.iter().any(|v| v.kind == ParityViolationKind::MissingBatchDispatch));
}

#[test]
fn test_validate_parity_u64_shared_memory() {
    let single = r#"
.visible .entry test(
    .param .u64 a
) {
    .shared .align 16 .b8 smem[32];
    st.shared.f32 [%r3], %f0;
    ret;
}
"#;
    let batched = r#"
.visible .entry test_batched(
    .param .u64 a
) {
    .shared .align 16 .b8 smem[32];
    mov.u32 %r1, %ctaid.y;
    st.shared.f32 [%rd3], %f0;
    ret;
}
"#;
    let result = validate_parity(single, batched, "test", "test_batched");
    assert!(!result.is_compatible);
    assert!(result
        .violations
        .iter()
        .any(|v| v.kind == ParityViolationKind::SharedMemoryAddressingU64));
}

#[test]
fn test_validate_batched_kernel_standalone() {
    // Grid.y dispatch
    let good_grid = r#"
.visible .entry good_batched(
    .param .u64 a
) {
    mov.u32 %r1, %ctaid.y;
    st.shared.f32 [%r3], %f0;
    ret;
}
"#;
    let result = validate_batched_kernel(good_grid, "good_batched");
    assert!(result.is_compatible);

    // Register-unrolled dispatch
    let good_reg = r#"
.visible .entry good_reg_batched(
    .param .u64 a,
    .param .u32 m_dim
) {
    ret;
}
"#;
    let result = validate_batched_kernel(good_reg, "good_reg_batched");
    assert!(result.is_compatible);

    // Neither dispatch mechanism + u64 shared mem
    let bad = r#"
.visible .entry bad_batched(
    .param .u64 a
) {
    st.shared.f32 [%rd3], %f0;
    ret;
}
"#;
    let result = validate_batched_kernel(bad, "bad_batched");
    assert!(!result.is_compatible);
    assert_eq!(result.violations.len(), 2); // missing dispatch AND u64 shared mem
}

#[test]
fn test_parity_violation_display() {
    assert_eq!(ParityViolationKind::ParameterCountMismatch.to_string(), "PARAM_COUNT");
    assert_eq!(ParityViolationKind::SharedMemoryAddressingU64.to_string(), "SHARED_MEM_U64");
    assert_eq!(ParityViolationKind::MissingBatchDispatch.to_string(), "MISSING_CTAID_Y");
}

#[test]
fn test_parity_violation_display_all_variants() {
    // Cover all Display variants including those not covered above
    assert_eq!(ParityViolationKind::SharedMemoryMismatch.to_string(), "SHARED_MEM_SIZE");
    assert_eq!(ParityViolationKind::LoopStructureMismatch.to_string(), "LOOP_STRUCTURE");
    assert_eq!(ParityViolationKind::RegisterTypeMismatch.to_string(), "REG_TYPE");
}

#[test]
fn test_validate_parity_shared_memory_mismatch() {
    // Single kernel has smem[32], batched has smem[64] -> SharedMemoryMismatch
    let single = r#"
.visible .entry test(
    .param .u64 a
) {
    .shared .align 16 .b8 smem[32];
    mov.u32 %r1, %ctaid.y;
    ret;
}
"#;
    let batched = r#"
.visible .entry test_batched(
    .param .u64 a
) {
    .shared .align 16 .b8 smem[64];
    mov.u32 %r1, %ctaid.y;
    ret;
}
"#;
    let result = validate_parity(single, batched, "test", "test_batched");
    assert!(!result.is_compatible);
    assert!(
        result.violations.iter().any(|v| v.kind == ParityViolationKind::SharedMemoryMismatch),
        "Expected SharedMemoryMismatch, got: {:?}",
        result.violations
    );
    // Verify message content
    let violation = result
        .violations
        .iter()
        .find(|v| v.kind == ParityViolationKind::SharedMemoryMismatch)
        .expect("test");
    assert!(violation.message.contains("32"));
    assert!(violation.message.contains("64"));
}

#[test]
fn test_validate_parity_shared_memory_one_has_none() {
    // Single kernel has shared memory, batched does not -> SharedMemoryMismatch
    let single = r#"
.visible .entry test(
    .param .u64 a
) {
    .shared .align 16 .b8 smem[128];
    mov.u32 %r1, %ctaid.y;
    ret;
}
"#;
    let batched = r#"
.visible .entry test_batched(
    .param .u64 a
) {
    mov.u32 %r1, %ctaid.y;
    ret;
}
"#;
    let result = validate_parity(single, batched, "test", "test_batched");
    assert!(!result.is_compatible);
    assert!(result.violations.iter().any(|v| v.kind == ParityViolationKind::SharedMemoryMismatch));
}

#[test]
fn test_validate_parity_loop_structure_mismatch() {
    // Single kernel has sum_loop, batched has sum_loop AND norm_loop -> LoopStructureMismatch
    let single = r#"
.visible .entry test(
    .param .u64 a
) {
    mov.u32 %r1, %ctaid.y;
sum_loop:
    bra sum_loop;
    ret;
}
"#;
    let batched = r#"
.visible .entry test_batched(
    .param .u64 a
) {
    mov.u32 %r1, %ctaid.y;
sum_loop:
    bra sum_loop;
norm_loop:
    bra norm_loop;
    ret;
}
"#;
    let result = validate_parity(single, batched, "test", "test_batched");
    assert!(!result.is_compatible);
    assert!(
        result.violations.iter().any(|v| v.kind == ParityViolationKind::LoopStructureMismatch),
        "Expected LoopStructureMismatch, got: {:?}",
        result.violations
    );
    // Verify message content includes loop labels
    let violation = result
        .violations
        .iter()
        .find(|v| v.kind == ParityViolationKind::LoopStructureMismatch)
        .expect("test");
    assert!(violation.message.contains("sum_loop"));
    assert!(violation.message.contains("norm_loop"));
}

#[test]
fn test_validate_parity_u64_shared_memory_on_single_kernel() {
    // Both kernels use u64 for shared memory -> both flagged
    let single = r#"
.visible .entry test(
    .param .u64 a
) {
    .shared .align 16 .b8 smem[32];
    ld.shared.f32 %f0, [%rd5];
    ret;
}
"#;
    let batched = r#"
.visible .entry test_batched(
    .param .u64 a
) {
    .shared .align 16 .b8 smem[32];
    mov.u32 %r1, %ctaid.y;
    ld.shared.f32 %f0, [%rd5];
    ret;
}
"#;
    let result = validate_parity(single, batched, "test", "test_batched");
    assert!(!result.is_compatible);
    // Should have TWO u64 shared memory violations (one for each kernel)
    let u64_violations: Vec<_> = result
        .violations
        .iter()
        .filter(|v| v.kind == ParityViolationKind::SharedMemoryAddressingU64)
        .collect();
    assert_eq!(
        u64_violations.len(),
        2,
        "Expected 2 SharedMemoryAddressingU64 violations (one per kernel), got {}: {:?}",
        u64_violations.len(),
        u64_violations
    );
    // One message should mention single kernel, the other batched
    assert!(u64_violations.iter().any(|v| v.message.contains("test_batched")));
    assert!(u64_violations.iter().any(|v| v.message.contains("'test'")));
}

#[test]
fn test_validate_parity_multiple_violations() {
    // Exercise all violation paths simultaneously:
    // Different param count, different smem, no batch dispatch, u64 smem on both, different loops
    let single = r#"
.visible .entry test(
    .param .u64 a,
    .param .u64 b
) {
    .shared .align 16 .b8 smem[32];
    st.shared.f32 [%rd1], %f0;
sum_loop:
    bra sum_loop;
    ret;
}
"#;
    let batched = r#"
.visible .entry test_batched(
    .param .u64 a
) {
    .shared .align 16 .b8 smem[64];
    st.shared.f32 [%rd2], %f0;
norm_loop:
    bra norm_loop;
    ret;
}
"#;
    let result = validate_parity(single, batched, "test", "test_batched");
    assert!(!result.is_compatible);
    assert_eq!(result.single_name, "test");
    assert_eq!(result.batched_name, "test_batched");

    // Should have at least 5 violations:
    // 1. ParameterCountMismatch (2 vs 1)
    // 2. SharedMemoryMismatch (32 vs 64)
    // 3. MissingBatchDispatch (no ctaid.y or m_dim)
    // 4. SharedMemoryAddressingU64 (batched)
    // 5. SharedMemoryAddressingU64 (single)
    // 6. LoopStructureMismatch (sum_loop vs norm_loop)
    assert!(
        result.violations.len() >= 5,
        "Expected at least 5 violations, got {}: {:?}",
        result.violations.len(),
        result.violations
    );

    let kinds: Vec<_> = result.violations.iter().map(|v| &v.kind).collect();
    assert!(kinds.contains(&&ParityViolationKind::ParameterCountMismatch));
    assert!(kinds.contains(&&ParityViolationKind::SharedMemoryMismatch));
    assert!(kinds.contains(&&ParityViolationKind::MissingBatchDispatch));
    assert!(kinds.contains(&&ParityViolationKind::SharedMemoryAddressingU64));
    assert!(kinds.contains(&&ParityViolationKind::LoopStructureMismatch));
}

#[test]
fn test_validate_parity_result_fields() {
    // Verify ParityResult struct fields are populated correctly
    let single = r#"
.visible .entry my_kernel(
    .param .u64 a
) {
    mov.u32 %r1, %ctaid.y;
    ret;
}
"#;
    let batched = r#"
.visible .entry my_batched_kernel(
    .param .u64 a
) {
    mov.u32 %r1, %ctaid.y;
    ret;
}
"#;
    let result = validate_parity(single, batched, "my_kernel", "my_batched_kernel");
    assert!(result.is_compatible);
    assert_eq!(result.single_name, "my_kernel");
    assert_eq!(result.batched_name, "my_batched_kernel");
    assert!(result.violations.is_empty());
}

#[test]
fn test_count_params_empty() {
    assert_eq!(count_params(""), 0);
    assert_eq!(count_params("no params here\njust code"), 0);
}

#[test]
fn test_count_params_with_indentation() {
    // Params with leading whitespace should still be counted
    let ptx = r#"
.visible .entry test(
    .param .u64 a,
    .param .f32 b,
    .param .u32 c,
    .param .u64 d
) { ret; }
"#;
    assert_eq!(count_params(ptx), 4);
}

#[test]
fn test_extract_shared_memory_bytes_various_sizes() {
    assert_eq!(extract_shared_memory_bytes("    .shared .align 4 .b8 smem[16384];"), Some(16384));
    assert_eq!(extract_shared_memory_bytes("    .shared .align 16 .b8 smem[0];"), Some(0));
    // No smem keyword
    assert_eq!(extract_shared_memory_bytes("    .shared .align 16 .b8 buf[32];"), None);
    // Malformed
    assert_eq!(extract_shared_memory_bytes("    .shared smem[abc];"), None);
}

#[test]
fn test_extract_loop_labels_no_loops() {
    let ptx = r#"
entry:
    mov.u32 %r0, %tid.x;
exit:
    ret;
"#;
    let labels = extract_loop_labels(ptx);
    assert!(labels.is_empty(), "Expected no loop labels, got: {:?}", labels);
}

#[test]
fn test_extract_loop_labels_comments_ignored() {
    let ptx = r#"
// sum_loop:
sum_loop:
    bra sum_loop;
"#;
    let labels = extract_loop_labels(ptx);
    assert_eq!(labels, vec!["sum_loop"]);
}

#[test]
fn test_has_u64_shared_memory_ld_pattern() {
    // ld.shared with u64 register
    assert!(has_u64_shared_memory_addressing("    ld.shared.f32 %f0, [%rd5];"));
    // ld.shared with u32 register
    assert!(!has_u64_shared_memory_addressing("    ld.shared.f32 %f0, [%r5];"));
    // Global memory (not shared) with u64 register is fine
    assert!(!has_u64_shared_memory_addressing("    ld.global.f32 %f0, [%rd5];"));
}

#[test]
fn test_validate_batched_kernel_standalone_names() {
    let ptx = r#"
.visible .entry my_batch(
    .param .u64 a
) {
    mov.u32 %r1, %ctaid.y;
    ret;
}
"#;
    let result = validate_batched_kernel(ptx, "my_batch");
    assert!(result.is_compatible);
    assert_eq!(result.single_name, "");
    assert_eq!(result.batched_name, "my_batch");
}

#[test]
fn test_parity_result_clone_and_debug() {
    let result = ParityResult {
        is_compatible: false,
        violations: vec![ParityViolation {
            kind: ParityViolationKind::RegisterTypeMismatch,
            message: "test violation".to_string(),
        }],
        single_name: "single".to_string(),
        batched_name: "batched".to_string(),
    };
    let cloned = result.clone();
    assert_eq!(cloned.is_compatible, result.is_compatible);
    assert_eq!(cloned.violations.len(), result.violations.len());
    assert_eq!(cloned.single_name, result.single_name);
    assert_eq!(cloned.batched_name, result.batched_name);

    // Debug formatting
    let debug = format!("{:?}", result);
    assert!(debug.contains("ParityResult"));
    assert!(debug.contains("RegisterTypeMismatch"));
}

#[test]
fn test_parity_violation_clone_and_debug() {
    let violation = ParityViolation {
        kind: ParityViolationKind::LoopStructureMismatch,
        message: "loops differ".to_string(),
    };
    let cloned = violation.clone();
    assert_eq!(cloned.kind, violation.kind);
    assert_eq!(cloned.message, violation.message);

    let debug = format!("{:?}", violation);
    assert!(debug.contains("LoopStructureMismatch"));
}

#[test]
fn test_batch_dispatch_strategy_clone_debug_eq() {
    let strategy = BatchDispatchStrategy::GridY;
    let cloned = strategy.clone();
    assert_eq!(cloned, BatchDispatchStrategy::GridY);
    assert_ne!(cloned, BatchDispatchStrategy::RegisterUnroll);

    let debug = format!("{:?}", strategy);
    assert!(debug.contains("GridY"));

    let reg = BatchDispatchStrategy::RegisterUnroll;
    let debug_reg = format!("{:?}", reg);
    assert!(debug_reg.contains("RegisterUnroll"));
}
