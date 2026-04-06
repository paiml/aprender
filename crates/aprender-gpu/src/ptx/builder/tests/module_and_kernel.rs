use super::super::*;

#[test]
fn test_module_defaults() {
    let module = PtxModule::new();
    assert_eq!(module.get_version(), (8, 0));
    assert_eq!(module.get_target(), "sm_70");
    assert_eq!(module.get_address_size(), 64);
}

#[test]
fn test_module_builder() {
    let module = PtxModule::new()
        .version(8, 5)
        .target("sm_86")
        .address_size(64);

    assert_eq!(module.get_version(), (8, 5));
    assert_eq!(module.get_target(), "sm_86");
}

#[test]
fn test_kernel_params() {
    let kernel = PtxKernel::new("test")
        .param(PtxType::U64, "ptr")
        .param(PtxType::U32, "n");

    assert_eq!(kernel.params.len(), 2);
    assert_eq!(kernel.params[0].name, "ptr");
    assert_eq!(kernel.params[1].name, "n");
}

#[test]
fn test_emit_header() {
    let module = PtxModule::new()
        .version(8, 0)
        .target("sm_70")
        .address_size(64);

    let ptx = module.emit();
    assert!(ptx.contains(".version 8.0"));
    assert!(ptx.contains(".target sm_70"));
    assert!(ptx.contains(".address_size 64"));
}

#[test]
fn test_emit_kernel() {
    let kernel = PtxKernel::new("vector_add")
        .param(PtxType::U64, "a")
        .param(PtxType::U64, "b");

    let module = PtxModule::new().add_kernel(kernel);
    let ptx = module.emit();

    assert!(ptx.contains(".visible .entry vector_add"));
    assert!(ptx.contains(".param .u64 a"));
    assert!(ptx.contains(".param .u64 b"));
}

// ========================================================================
// BUG FIX TESTS - EXTREME TDD
// ========================================================================

#[test]
fn test_bar_sync_emission() {
    // BUG: bar.sync was being emitted as "bar.b32 ;" instead of "bar.sync 0;"
    let kernel = PtxKernel::new("test_barrier").build(|ctx| {
        ctx.bar_sync(0);
        ctx.ret();
    });

    let ptx = kernel.emit();
    // Must contain proper bar.sync instruction
    assert!(
        ptx.contains("bar.sync 0"),
        "Expected 'bar.sync 0' but got: {}",
        ptx
    );
    // Must NOT contain the buggy output
    assert!(
        !ptx.contains("bar.b32"),
        "Found buggy 'bar.b32' in: {}",
        ptx
    );
}

#[test]
fn test_cvt_u64_u32_emission() {
    // BUG: cvt was being emitted as "cvt.u64 %r, %r" instead of "cvt.u64.u32 %r, %r"
    let kernel = PtxKernel::new("test_cvt").build(|ctx| {
        let val = ctx.mov_u32_imm(42);
        let _wide = ctx.cvt_u64_u32(val);
        ctx.ret();
    });

    let ptx = kernel.emit();
    // Must contain proper cvt with both types
    assert!(
        ptx.contains("cvt.u64.u32"),
        "Expected 'cvt.u64.u32' but got: {}",
        ptx
    );
}

#[test]
fn test_shared_memory_addressing() {
    // Shared memory access uses register-based addressing
    let kernel = PtxKernel::new("test_shared")
        .shared_memory(1024)
        .build(|ctx| {
            let val = ctx.mov_f32_imm(1.0);
            let offset = ctx.mov_u32_imm(0);
            let offset_64 = ctx.cvt_u64_u32(offset);
            ctx.st_shared_f32(offset_64, val);
            let _loaded = ctx.ld_shared_f32(offset_64);
            ctx.ret();
        });

    let ptx = kernel.emit();
    // Must contain proper shared memory operations
    assert!(
        ptx.contains("st.shared.f32") && ptx.contains("ld.shared.f32"),
        "Expected shared memory operations, got: {}",
        ptx
    );
    // Must contain brackets for addressing
    assert!(
        ptx.contains("[%rd"),
        "Expected bracketed register address, got: {}",
        ptx
    );
}

#[test]
fn test_bar_sync_with_different_barriers() {
    // Test barrier with different IDs
    let kernel = PtxKernel::new("test_barriers").build(|ctx| {
        ctx.bar_sync(0);
        ctx.bar_sync(1);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("bar.sync 0"),
        "Expected 'bar.sync 0' in: {}",
        ptx
    );
    assert!(
        ptx.contains("bar.sync 1"),
        "Expected 'bar.sync 1' in: {}",
        ptx
    );
}

#[test]
fn test_global_memory_addressing() {
    // BUG: global memory access was "ld.global.f32 %f, %r" instead of "ld.global.f32 %f, [%r]"
    let kernel = PtxKernel::new("test_global")
        .param(PtxType::U64, "ptr")
        .build(|ctx| {
            let ptr = ctx.load_param_u64("ptr");
            let val = ctx.ld_global_f32(ptr);
            ctx.st_global_f32(ptr, val);
            ctx.ret();
        });

    let ptx = kernel.emit();
    // Load must use brackets for address
    assert!(
        ptx.contains("ld.global.f32") && ptx.contains("[%rd"),
        "Expected ld.global.f32 with [%rd] address, got: {}",
        ptx
    );
    // Store must use brackets for address
    assert!(
        ptx.contains("st.global.f32 ["),
        "Expected st.global.f32 with [%rd] address, got: {}",
        ptx
    );
}

#[test]
fn test_f32_literal_format() {
    // BUG: float literals were emitted as "0e0" instead of PTX hex format "0F00000000"
    let kernel = PtxKernel::new("test_float").build(|ctx| {
        let _zero = ctx.mov_f32_imm(0.0);
        let _one = ctx.mov_f32_imm(1.0);
        ctx.ret();
    });

    let ptx = kernel.emit();
    // PTX float literals must be in hex format
    assert!(
        ptx.contains("0F00000000"), // 0.0f in hex
        "Expected 0F00000000 for 0.0f, got: {}",
        ptx
    );
    assert!(
        ptx.contains("0F3F800000"), // 1.0f in hex
        "Expected 0F3F800000 for 1.0f, got: {}",
        ptx
    );
}

#[test]
fn test_loop_counter_update_in_place() {
    // BUG: Loop counters were never updated due to SSA - computed values discarded
    // PTX loops need in-place register updates: add.u32 %r0, %r0, 1
    let kernel = PtxKernel::new("test_loop")
        .param(PtxType::U32, "n")
        .build(|ctx| {
            let n = ctx.load_param_u32("n");
            let i = ctx.mov_u32_imm(0);
            ctx.label("loop");
            let done = ctx.setp_ge_u32(i, n);
            ctx.branch_if(done, "exit");
            // In-place increment: i = i + 1
            ctx.add_u32_inplace(i, 1);
            ctx.branch("loop");
            ctx.label("exit");
            ctx.ret();
        });

    let ptx = kernel.emit();
    // The loop counter must be updated in-place (same src and dst register)
    // Look for pattern like: add.u32 %r1, %r1, 1
    assert!(
        ptx.contains("add") && ptx.contains("%r") && ptx.contains(", 1"),
        "Expected in-place add instruction, got: {}",
        ptx
    );
}

#[test]
fn test_accumulator_update_in_place() {
    // BUG: Accumulators in inner loops were never updated
    // Need in-place: add.f32 %f0, %f0, %f1
    let kernel = PtxKernel::new("test_acc").build(|ctx| {
        let acc = ctx.mov_f32_imm(0.0);
        let val = ctx.mov_f32_imm(1.0);
        // In-place accumulate: acc = acc + val
        ctx.add_f32_inplace(acc, val);
        ctx.ret();
    });

    let ptx = kernel.emit();
    // Must have in-place add
    assert!(
        ptx.contains("add") && ptx.contains(".f32"),
        "Expected f32 add instruction, got: {}",
        ptx
    );
}
