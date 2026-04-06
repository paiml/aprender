//! DP4A dot product, membar, volatile loads, ballot, popc/bfind/clz,
//! warp shuffle, atomic operations, trig/neg, and sign conversion tests.

use super::*;

#[test]
fn test_dp4a_u32_instruction() {
    let kernel = PtxKernel::new("test_dp4a").build(|ctx| {
        let a = ctx.mov_u32_imm(0x01020304);
        let b = ctx.mov_u32_imm(0x05060708);
        let c = ctx.mov_u32_imm(0);
        let _result = ctx.dp4a_u32(a, b, c);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(
        ptx.contains("dp4a.u32.u32"),
        "Expected dp4a.u32.u32 in: {}",
        ptx
    );
}

#[test]
fn test_dp4a_u32_inplace_instruction() {
    let kernel = PtxKernel::new("test_dp4a_inplace").build(|ctx| {
        let acc = ctx.mov_u32_imm(0);
        let a = ctx.mov_u32_imm(0x01020304);
        let b = ctx.mov_u32_imm(0x05060708);
        ctx.dp4a_u32_inplace(acc, a, b);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("dp4a"), "Expected dp4a in: {}", ptx);
}

#[test]
fn test_dp4a_u32_s32_inplace_instruction() {
    let kernel = PtxKernel::new("test_dp4a_us").build(|ctx| {
        let acc = ctx.mov_u32_imm(0);
        let a = ctx.mov_u32_imm(0x01020304);
        let b = ctx.mov_u32_imm(0x05060708);
        ctx.dp4a_u32_s32_inplace(acc, a, b);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("dp4a"), "Expected dp4a in: {}", ptx);
}

#[test]
fn test_dp4a_s32_inplace_instruction() {
    let kernel = PtxKernel::new("test_dp4a_s32").build(|ctx| {
        let acc = ctx.mov_u32_imm(0);
        let a = ctx.mov_u32_imm(0x01020304);
        let b = ctx.mov_u32_imm(0x05060708);
        ctx.dp4a_s32_inplace(acc, a, b);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("dp4a"), "Expected dp4a in: {}", ptx);
}

#[test]
fn test_membar_cta_instruction() {
    let kernel = PtxKernel::new("test_membar_cta").build(|ctx| {
        ctx.membar_cta();
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(
        ptx.contains("membar.cta"),
        "Expected membar.cta in: {}",
        ptx
    );
}

#[test]
fn test_membar_gl_instruction() {
    let kernel = PtxKernel::new("test_membar_gl").build(|ctx| {
        ctx.membar_gl();
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("membar.gl"), "Expected membar.gl in: {}", ptx);
}

#[test]
fn test_ld_shared_u32_volatile_instruction() {
    let kernel = PtxKernel::new("test_ld_volatile")
        .shared_memory(256)
        .build(|ctx| {
            let addr = ctx.mov_u64_imm(0);
            let _val = ctx.ld_shared_u32_volatile(addr);
            ctx.ret();
        });
    let ptx = kernel.emit();
    assert!(
        ptx.contains("ld.volatile.shared.u32"),
        "Expected ld.volatile.shared.u32 in: {}",
        ptx
    );
}

#[test]
fn test_ballot_sync_instruction() {
    let kernel = PtxKernel::new("test_ballot").build(|ctx| {
        let a = ctx.mov_u32_imm(1);
        let b = ctx.mov_u32_imm(0);
        let pred = ctx.setp_ge_u32(a, b);
        let _ballot = ctx.ballot_sync(pred, 0xFFFFFFFF);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(
        ptx.contains("vote") || ptx.contains("ballot"),
        "Expected ballot in: {}",
        ptx
    );
}

#[test]
fn test_popc_u32_instruction() {
    let kernel = PtxKernel::new("test_popc").build(|ctx| {
        let val = ctx.mov_u32_imm(0xFF);
        let _count = ctx.popc_u32(val);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("popc"), "Expected popc in: {}", ptx);
}

#[test]
fn test_bfind_u32_instruction() {
    let kernel = PtxKernel::new("test_bfind").build(|ctx| {
        let val = ctx.mov_u32_imm(0x80);
        let _pos = ctx.bfind_u32(val);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("bfind"), "Expected bfind in: {}", ptx);
}

#[test]
fn test_clz_u32_instruction() {
    let kernel = PtxKernel::new("test_clz").build(|ctx| {
        let val = ctx.mov_u32_imm(0x80);
        let _lz = ctx.clz_u32(val);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("clz"), "Expected clz in: {}", ptx);
}

#[test]
fn test_shfl_idx_u32_reg_instruction() {
    let kernel = PtxKernel::new("test_shfl_reg").build(|ctx| {
        let val = ctx.mov_u32_imm(42);
        let lane = ctx.mov_u32_imm(0);
        let _shuffled = ctx.shfl_idx_u32_reg(val, lane, 0xFFFFFFFF);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(
        ptx.contains("shfl.sync.idx"),
        "Expected shfl.sync.idx in: {}",
        ptx
    );
}

#[test]
fn test_atom_add_global_u32_instruction() {
    let kernel = PtxKernel::new("test_atom_add")
        .param(PtxType::U64, "ptr")
        .build(|ctx| {
            let ptr = ctx.load_param_u64("ptr");
            let val = ctx.mov_u32_imm(1);
            let _old = ctx.atom_add_global_u32(ptr, val);
            ctx.ret();
        });
    let ptx = kernel.emit();
    assert!(
        ptx.contains("atom.global.add.u32"),
        "Expected atom.global.add.u32 in: {}",
        ptx
    );
}

#[test]
fn test_atom_exch_global_u32_instruction() {
    let kernel = PtxKernel::new("test_atom_exch")
        .param(PtxType::U64, "ptr")
        .build(|ctx| {
            let ptr = ctx.load_param_u64("ptr");
            let val = ctx.mov_u32_imm(42);
            let _old = ctx.atom_exch_global_u32(ptr, val);
            ctx.ret();
        });
    let ptx = kernel.emit();
    assert!(
        ptx.contains("atom.global.exch.u32"),
        "Expected atom.global.exch.u32 in: {}",
        ptx
    );
}

#[test]
fn test_atom_min_global_u32_instruction() {
    let kernel = PtxKernel::new("test_atom_min")
        .param(PtxType::U64, "ptr")
        .build(|ctx| {
            let ptr = ctx.load_param_u64("ptr");
            let val = ctx.mov_u32_imm(10);
            let _old = ctx.atom_min_global_u32(ptr, val);
            ctx.ret();
        });
    let ptx = kernel.emit();
    assert!(
        ptx.contains("atom.global.min.u32"),
        "Expected atom.global.min.u32 in: {}",
        ptx
    );
}

#[test]
fn test_atom_max_global_u32_instruction() {
    let kernel = PtxKernel::new("test_atom_max")
        .param(PtxType::U64, "ptr")
        .build(|ctx| {
            let ptr = ctx.load_param_u64("ptr");
            let val = ctx.mov_u32_imm(100);
            let _old = ctx.atom_max_global_u32(ptr, val);
            ctx.ret();
        });
    let ptx = kernel.emit();
    assert!(
        ptx.contains("atom.global.max.u32"),
        "Expected atom.global.max.u32 in: {}",
        ptx
    );
}

#[test]
fn test_atom_exch_shared_u32_instruction() {
    let kernel = PtxKernel::new("test_atom_exch_shared")
        .shared_memory(256)
        .build(|ctx| {
            let addr = ctx.mov_u64_imm(0);
            let val = ctx.mov_u32_imm(42);
            let _old = ctx.atom_exch_shared_u32(addr, val);
            ctx.ret();
        });
    let ptx = kernel.emit();
    assert!(
        ptx.contains("atom.shared.exch.u32"),
        "Expected atom.shared.exch.u32 in: {}",
        ptx
    );
}

#[test]
fn test_sin_f32_instruction() {
    let kernel = PtxKernel::new("test_sin").build(|ctx| {
        let val = ctx.mov_f32_imm(1.57);
        let _result = ctx.sin_f32(val);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(
        ptx.contains("sin.approx.f32"),
        "Expected sin.approx.f32 in: {}",
        ptx
    );
}

#[test]
fn test_cos_f32_instruction() {
    let kernel = PtxKernel::new("test_cos").build(|ctx| {
        let val = ctx.mov_f32_imm(0.0);
        let _result = ctx.cos_f32(val);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(
        ptx.contains("cos.approx.f32"),
        "Expected cos.approx.f32 in: {}",
        ptx
    );
}

#[test]
fn test_neg_f32_instruction() {
    let kernel = PtxKernel::new("test_neg").build(|ctx| {
        let val = ctx.mov_f32_imm(1.0);
        let _result = ctx.neg_f32(val);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("neg.f32"), "Expected neg.f32 in: {}", ptx);
}

#[test]
fn test_cvt_s32_s8_instruction() {
    let kernel = PtxKernel::new("test_cvt_s8")
        .param(PtxType::U64, "ptr")
        .build(|ctx| {
            let ptr = ctx.load_param_u64("ptr");
            let val = ctx.ld_global_u8(ptr);
            let _signed = ctx.cvt_s32_s8(val);
            ctx.ret();
        });
    let ptx = kernel.emit();
    assert!(
        ptx.contains("setp"),
        "Expected setp for sign extension in: {}",
        ptx
    );
}

#[test]
fn test_cvt_f32_s32_instruction() {
    let kernel = PtxKernel::new("test_cvt_f32_s32").build(|ctx| {
        let val = ctx.mov_u32_imm(42);
        let _float = ctx.cvt_f32_s32(val);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(
        ptx.contains("cvt.rn.f32.s32"),
        "Expected cvt.rn.f32.s32 in: {}",
        ptx
    );
}
