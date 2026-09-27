use super::super::*;

use crate::ptx::instructions::WmmaLayout;

// ========================================================================
// TENSOR CORE (WMMA) TESTS - IMP-1000a
// ========================================================================

#[test]
fn test_wmma_load_a_f16() {
    let kernel = PtxKernel::new("test_wmma_load_a")
        .param(PtxType::U64, "a_ptr")
        .build(|ctx| {
            let ptr = ctx.load_param_u64("a_ptr");
            let _frag_a = ctx.wmma_load_a_f16(ptr, 16, WmmaLayout::RowMajor);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(
        ptx.contains(".param .u64 a_ptr"),
        "Expected a_ptr param, got: {}",
        ptx
    );
}

#[test]
fn test_wmma_load_b_f16() {
    let kernel = PtxKernel::new("test_wmma_load_b")
        .param(PtxType::U64, "b_ptr")
        .build(|ctx| {
            let ptr = ctx.load_param_u64("b_ptr");
            let _frag_b = ctx.wmma_load_b_f16(ptr, 16, WmmaLayout::ColMajor);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(
        ptx.contains(".param .u64 b_ptr"),
        "Expected b_ptr param, got: {}",
        ptx
    );
}

#[test]
fn test_wmma_mma_f16_f32() {
    let kernel = PtxKernel::new("test_wmma_mma")
        .param(PtxType::U64, "a_ptr")
        .param(PtxType::U64, "b_ptr")
        .param(PtxType::U64, "c_ptr")
        .build(|ctx| {
            let a = ctx.load_param_u64("a_ptr");
            let b = ctx.load_param_u64("b_ptr");
            let c = ctx.load_param_u64("c_ptr");

            let frag_a = ctx.wmma_load_a_f16(a, 16, WmmaLayout::RowMajor);
            let frag_b = ctx.wmma_load_b_f16(b, 16, WmmaLayout::ColMajor);
            let frag_c = ctx.wmma_load_c_f32(c, 16, WmmaLayout::RowMajor);

            let _frag_d = ctx.wmma_mma_f16_f32(&frag_a, &frag_b, &frag_c);
            ctx.ret();
        });

    let ptx = kernel.emit();
    // Verify kernel structure
    assert!(
        ptx.contains(".visible .entry test_wmma_mma"),
        "Expected kernel entry, got: {}",
        ptx
    );
}

#[test]
fn test_wmma_store_d_f32() {
    let kernel = PtxKernel::new("test_wmma_store")
        .param(PtxType::U64, "d_ptr")
        .build(|ctx| {
            let d = ctx.load_param_u64("d_ptr");
            // Create empty fragment for test
            let frag_d = vec![ctx.mov_f32_imm(0.0)];
            ctx.wmma_store_d_f32(d, &frag_d, 16, WmmaLayout::RowMajor);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(
        ptx.contains(".param .u64 d_ptr"),
        "Expected d_ptr param, got: {}",
        ptx
    );
}

#[test]
fn test_cvt_f16_f32() {
    let kernel = PtxKernel::new("test_cvt_f16").build(|ctx| {
        let f32_val = ctx.mov_f32_imm(1.5);
        let _f16_val = ctx.cvt_f16_f32(f32_val);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("cvt"),
        "Expected cvt instruction, got: {}",
        ptx
    );
}

#[test]
fn test_cvt_f32_f16() {
    let kernel = PtxKernel::new("test_cvt_f32")
        .param(PtxType::U64, "ptr")
        .build(|ctx| {
            let ptr = ctx.load_param_u64("ptr");
            let f16_val = ctx.ld_global_f16(ptr);
            let _f32_val = ctx.cvt_f32_f16(f16_val);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(
        ptx.contains(".param .u64 ptr"),
        "Expected ptr param, got: {}",
        ptx
    );
}

#[test]
fn test_ld_st_global_f16() {
    let kernel = PtxKernel::new("test_f16_mem")
        .param(PtxType::U64, "in_ptr")
        .param(PtxType::U64, "out_ptr")
        .build(|ctx| {
            let in_ptr = ctx.load_param_u64("in_ptr");
            let out_ptr = ctx.load_param_u64("out_ptr");
            let val = ctx.ld_global_f16(in_ptr);
            ctx.st_global_f16(out_ptr, val);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(
        ptx.contains(".param .u64 in_ptr") && ptx.contains(".param .u64 out_ptr"),
        "Expected both params, got: {}",
        ptx
    );
}

#[test]
fn test_wmma_load_c_f32_fragment() {
    // Test WMMA load C (accumulator) fragment
    let kernel = PtxKernel::new("test_wmma_load_c")
        .shared_memory(1024)
        .build(|ctx| {
            let addr = ctx.shared_base_addr();
            let _frag_c = ctx.wmma_load_c_f32(addr, 16, WmmaLayout::RowMajor);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("wmma.load.c.sync.aligned"),
        "Expected wmma.load.c in: {}",
        ptx
    );
}

#[test]
fn test_wmma_store_d_empty_fragment() {
    // Test WMMA store with empty fragment (should be no-op)
    let kernel = PtxKernel::new("test_wmma_store_empty").build(|ctx| {
        let addr = ctx.shared_base_addr();
        let empty_frag: Vec<VirtualReg> = Vec::new();
        ctx.wmma_store_d_f32(addr, &empty_frag, 16, WmmaLayout::RowMajor);
        ctx.ret();
    });

    let ptx = kernel.emit();
    // With empty fragment, wmma_store should return early
    assert!(
        !ptx.contains("wmma.store"),
        "Expected no wmma.store with empty fragment"
    );
}

#[test]
fn test_wmma_layout_col_major() {
    // Test column-major WMMA layout
    let kernel = PtxKernel::new("test_wmma_col")
        .shared_memory(1024)
        .build(|ctx| {
            let addr = ctx.shared_base_addr();
            let _frag_a = ctx.wmma_load_a_f16(addr, 16, WmmaLayout::ColMajor);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(ptx.contains(".col."), "Expected .col. layout in: {}", ptx);
}

/// #4376: the int8 mma form is emitted for S32 and assembles on sm_89;
/// the f16 form is unchanged for F32.
#[test]
fn test_mma_sync_m16n8k32_s8_emits_and_assembles() {
    let kernel = PtxKernel::new("test_mma_s8")
        .param(PtxType::U64, "out_ptr")
        .build(|ctx| {
            let out = ctx.load_param_u64("out_ptr");
            let a =
                [0x0101_0101, 0x0202_0202, 0x0303_0303, 0x0404_0404].map(|v| ctx.mov_b32_imm(v));
            let b = [0x0101_0101, 0xFFFF_FFFF].map(|v| ctx.mov_b32_imm(v));
            let c = [0, 0, 0, 0].map(|v| ctx.mov_s32_imm(v));
            ctx.mma_sync_m16n8k32_s8_inplace(&a, &b, &c);
            let fa = [0x3c00_3c00; 4].map(|v| ctx.mov_b32_imm(v));
            let fb = [0x3c00_3c00; 2].map(|v| ctx.mov_b32_imm(v));
            let fc = ctx.wmma_init_c_zero();
            ctx.mma_sync_m16n8k16_inplace(&fa, &fb, &[fc[0], fc[1], fc[2], fc[3]]);
            ctx.st_global_u32(out, c[0]);
            ctx.ret();
        });
    let ptx = PtxModule::new()
        .version(8, 0)
        .target("sm_89")
        .add_kernel(kernel)
        .emit();
    assert!(
        ptx.contains("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {"),
        "int8 mma missing:\n{ptx}"
    );
    assert!(
        ptx.contains("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {"),
        "f16 mma changed:\n{ptx}"
    );

    let Ok(out) = std::process::Command::new("ptxas")
        .arg("--version")
        .output()
    else {
        eprintln!("ptxas not found; emission checked only");
        return;
    };
    assert!(out.status.success());
    let path = std::env::temp_dir().join(format!("mma_s8_{}.ptx", std::process::id()));
    std::fs::write(&path, &ptx).expect("write ptx");
    let res = std::process::Command::new("ptxas")
        .args(["--gpu-name", "sm_89", "-o", "/dev/null"])
        .arg(&path)
        .output()
        .expect("run ptxas");
    let _ = std::fs::remove_file(&path);
    assert!(
        res.status.success(),
        "ptxas rejected:\n{}\n{ptx}",
        String::from_utf8_lossy(&res.stderr)
    );
}
