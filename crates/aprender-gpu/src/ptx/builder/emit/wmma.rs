//! WMMA (Tensor Core) operation emission
//!
//! Handles: WmmaLoadA, WmmaLoadB, WmmaLoadC, WmmaMma, WmmaStoreD

use super::operand::emit_operand;
use crate::ptx::instructions::{PtxInstruction, PtxOp};

/// Emit WMMA load instruction with proper register list format
/// Format: wmma.load.{a|b|c}.sync.aligned.m16n16k16.{layout}.{type} {regs}, [ptr], stride
pub(crate) fn emit_wmma_load(prefix: String, instr: &PtxInstruction, matrix: &str) -> String {
    let mut s = prefix;

    // Parse label to get layout, type, stride
    // Label format: "m16n16k16.{layout}.{type}.stride.{stride}"
    let label = instr.label.as_deref().unwrap_or("m16n16k16.row.f16.stride.16");
    let parts: Vec<&str> = label.split('.').collect();

    // Build instruction opcode
    s.push_str(&format!("wmma.load.{}.sync.aligned", matrix));

    // Add shape, layout, type from label (e.g., "m16n16k16.row.f16")
    if parts.len() >= 3 {
        s.push('.');
        s.push_str(parts[0]); // m16n16k16
        s.push('.');
        s.push_str(parts[1]); // row/col
        s.push('.');
        s.push_str(parts[2]); // f16/f32
    } else {
        s.push_str(".m16n16k16.row.f16");
    }

    s.push(' ');

    // Destination registers: {%r0, %r1, ..., %r7}
    s.push('{');
    for (i, dst) in instr.dsts.iter().enumerate() {
        s.push_str(&emit_operand(dst));
        if i < instr.dsts.len() - 1 {
            s.push_str(", ");
        }
    }
    s.push_str("}, ");

    // Source: [ptr]
    if let Some(src) = instr.srcs.first() {
        s.push('[');
        s.push_str(&emit_operand(src));
        s.push_str("], ");
    }

    // Stride
    if let Some(stride) = instr.srcs.get(1) {
        s.push_str(&emit_operand(stride));
    } else {
        // Extract stride from label (last part after "stride.")
        if let Some(stride_pos) = label.find("stride.") {
            s.push_str(&label[stride_pos + 7..]);
        } else {
            s.push_str("16");
        }
    }

    s.push_str(";\n");
    s
}

/// Emit WMMA MMA instruction with proper register list format
/// Format: wmma.mma.sync.aligned.m16n16k16.row.col.f32.f32 {d}, {a}, {b}, {c}
pub(crate) fn emit_wmma_mma(prefix: String, instr: &PtxInstruction) -> String {
    let mut s = prefix;

    // Label format: "m16n16k16.row.col.f32.f32"
    let label = instr.label.as_deref().unwrap_or("m16n16k16.row.col.f32.f32");

    s.push_str("wmma.mma.sync.aligned.");
    s.push_str(label);
    s.push(' ');

    // D registers (first 8 of dsts)
    s.push('{');
    for (i, dst) in instr.dsts.iter().enumerate() {
        s.push_str(&emit_operand(dst));
        if i < instr.dsts.len() - 1 {
            s.push_str(", ");
        }
    }
    s.push_str("}, ");

    // A, B, C registers (each 8 registers from srcs)
    // Total srcs = 24 (8 A + 8 B + 8 C)
    let groups = [
        (0, 8),   // A
        (8, 16),  // B
        (16, 24), // C
    ];

    for (start, end) in groups {
        s.push('{');
        for i in start..end.min(instr.srcs.len()) {
            s.push_str(&emit_operand(&instr.srcs[i]));
            if i < end.min(instr.srcs.len()) - 1 {
                s.push_str(", ");
            }
        }
        s.push('}');
        if end < 24 && end <= instr.srcs.len() {
            s.push_str(", ");
        }
    }

    s.push_str(";\n");
    s
}

/// Emit WMMA store instruction with proper format
/// Format: wmma.store.d.sync.aligned.m16n16k16.{layout}.{type} [ptr], {regs}, stride
pub(crate) fn emit_wmma_store(prefix: String, instr: &PtxInstruction) -> String {
    let mut s = prefix;

    // Label format: "m16n16k16.{layout}.{type}.stride.{stride}"
    let label = instr.label.as_deref().unwrap_or("m16n16k16.row.f32.stride.16");
    let parts: Vec<&str> = label.split('.').collect();

    s.push_str("wmma.store.d.sync.aligned");

    // Add shape, layout, type from label
    if parts.len() >= 3 {
        s.push('.');
        s.push_str(parts[0]); // m16n16k16
        s.push('.');
        s.push_str(parts[1]); // row
        s.push('.');
        s.push_str(parts[2]); // f32
    } else {
        s.push_str(".m16n16k16.row.f32");
    }

    s.push(' ');

    // [ptr]
    if let Some(src) = instr.srcs.first() {
        s.push('[');
        s.push_str(&emit_operand(src));
        s.push_str("], ");
    }

    // {regs} - D fragment (srcs 1-8)
    s.push('{');
    let frag_end = instr.srcs.len().saturating_sub(1).min(9);
    for i in 1..frag_end {
        s.push_str(&emit_operand(&instr.srcs[i]));
        if i < frag_end - 1 {
            s.push_str(", ");
        }
    }
    s.push_str("}, ");

    // Stride (last src)
    if let Some(stride) = instr.srcs.last() {
        s.push_str(&emit_operand(stride));
    } else if let Some(stride_pos) = label.find("stride.") {
        s.push_str(&label[stride_pos + 7..]);
    } else {
        s.push_str("16");
    }

    s.push_str(";\n");
    s
}

/// Check if this is a WMMA operation
pub(crate) fn is_wmma_op(op: &PtxOp) -> bool {
    matches!(
        op,
        PtxOp::WmmaLoadA
            | PtxOp::WmmaLoadB
            | PtxOp::WmmaLoadC
            | PtxOp::WmmaMma
            | PtxOp::WmmaStoreD
            | PtxOp::MmaSync
            | PtxOp::LdMatrix
            | PtxOp::LdMatrixTrans
    )
}

/// Emit mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
///
/// Per-thread register layout:
///   A: d0..d3 (4 regs, 8 FP16 values)
///   B: d4..d5 (2 regs, 4 FP16 values)
///   C: d6..d9 (4 regs, 4 FP32 accumulators)
///   D: dst0..dst3 (4 regs, 4 FP32 results)
///
/// PTX: mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
///        {d0,d1,d2,d3}, {a0,a1,a2,a3}, {b0,b1}, {c0,c1,c2,c3};
pub(crate) fn emit_mma_sync(prefix: String, instr: &PtxInstruction) -> String {
    let mut s = prefix;

    // Destinations (D: 4 FP32 regs — dst + dsts)
    s.push_str("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {");
    if let Some(ref d) = instr.dst {
        s.push_str(&emit_operand(d));
    }
    for d in &instr.dsts {
        s.push_str(", ");
        s.push_str(&emit_operand(d));
    }
    s.push_str("}, {");

    // Sources: first 4 = A regs, next 2 = B regs, last 4 = C regs
    let sources = &instr.srcs;
    for i in 0..4 {
        if i > 0 {
            s.push_str(", ");
        }
        s.push_str(&emit_operand(&sources[i]));
    }
    s.push_str("}, {");
    for i in 4..6 {
        if i > 4 {
            s.push_str(", ");
        }
        s.push_str(&emit_operand(&sources[i]));
    }
    s.push_str("}, {");
    for i in 6..10 {
        if i > 6 {
            s.push_str(", ");
        }
        s.push_str(&emit_operand(&sources[i]));
    }
    s.push_str("};\n");
    s
}

/// Emit ldmatrix.sync.aligned.m8n8.x4.shared.b16
///
/// Loads 4 8×8 FP16 matrices from shared memory. Each thread provides
/// one source address (its row within the 8×8 tile).
///
/// PTX: ldmatrix.sync.aligned.m8n8.x4.shared.b16 {d0,d1,d2,d3}, [addr];
pub(crate) fn emit_ldmatrix(prefix: String, instr: &PtxInstruction) -> String {
    let mut s = prefix;
    s.push_str("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {");
    if let Some(ref d) = instr.dst {
        s.push_str(&emit_operand(d));
    }
    for d in &instr.dsts {
        s.push_str(", ");
        s.push_str(&emit_operand(d));
    }
    s.push_str("}, [");
    s.push_str(&emit_operand(&instr.srcs[0]));
    s.push_str("];\n");
    s
}

/// Emit ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16
/// Transposed variant for B fragment of mma.sync.row.col.
pub(crate) fn emit_ldmatrix_trans(prefix: String, instr: &PtxInstruction) -> String {
    let mut s = prefix;
    s.push_str("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {");
    if let Some(ref d) = instr.dst {
        s.push_str(&emit_operand(d));
    }
    for d in &instr.dsts {
        s.push_str(", ");
        s.push_str(&emit_operand(d));
    }
    s.push_str("}, [");
    s.push_str(&emit_operand(&instr.srcs[0]));
    s.push_str("];\n");
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ptx::instructions::Operand;
    use crate::ptx::registers::VirtualReg;
    use crate::ptx::types::PtxType;

    const INDENT: &str = "    ";

    fn regs(start: u32, count: u32, ty: PtxType) -> Vec<Operand> {
        (start..start + count).map(|i| Operand::Reg(VirtualReg::new(i, ty))).collect()
    }

    fn ptr_reg(id: u32) -> Operand {
        Operand::Reg(VirtualReg::new(id, PtxType::U64))
    }

    fn wmma_instr(
        op: PtxOp,
        ty: PtxType,
        dsts: Vec<Operand>,
        srcs: Vec<Operand>,
        label: Option<&str>,
    ) -> PtxInstruction {
        PtxInstruction {
            op,
            ty,
            src_type: None,
            dst: None,
            dsts,
            srcs,
            label: label.map(String::from),
            predicate: None,
            state_space: None,
            rounding: None,
        }
    }

    fn assert_contains_all(result: &str, patterns: &[&str]) {
        for pat in patterns {
            assert!(result.contains(pat), "missing '{pat}' in: {result}");
        }
    }

    // === is_wmma_op ===

    #[test]
    fn test_is_wmma_op_all_variants() {
        for op in [
            PtxOp::WmmaLoadA,
            PtxOp::WmmaLoadB,
            PtxOp::WmmaLoadC,
            PtxOp::WmmaMma,
            PtxOp::WmmaStoreD,
        ] {
            assert!(is_wmma_op(&op));
        }
        for op in [PtxOp::Add, PtxOp::Ld, PtxOp::Mul, PtxOp::Bra] {
            assert!(!is_wmma_op(&op));
        }
    }

    // === emit_wmma_load ===

    #[test]
    fn test_emit_wmma_load_a_default() {
        let instr = wmma_instr(
            PtxOp::WmmaLoadA,
            PtxType::F16,
            regs(0, 8, PtxType::F16),
            vec![ptr_reg(100), Operand::ImmU64(16)],
            None,
        );
        assert_contains_all(
            &emit_wmma_load(INDENT.into(), &instr, "a"),
            &["wmma.load.a.sync.aligned", ".m16n16k16.row.f16"],
        );
    }

    #[test]
    fn test_emit_wmma_load_b_with_label() {
        let instr = wmma_instr(
            PtxOp::WmmaLoadB,
            PtxType::F16,
            regs(0, 8, PtxType::F16),
            vec![ptr_reg(100)],
            Some("m16n16k16.col.f16.stride.32"),
        );
        assert_contains_all(
            &emit_wmma_load(INDENT.into(), &instr, "b"),
            &["wmma.load.b.sync.aligned", ".m16n16k16.col.f16", "32"],
        );
    }

    #[test]
    fn test_emit_wmma_load_c() {
        let instr = wmma_instr(
            PtxOp::WmmaLoadC,
            PtxType::F32,
            regs(0, 8, PtxType::F32),
            vec![ptr_reg(100), Operand::ImmU64(16)],
            Some("m16n16k16.row.f32"),
        );
        assert_contains_all(
            &emit_wmma_load(INDENT.into(), &instr, "c"),
            &["wmma.load.c.sync.aligned", ".m16n16k16.row.f32"],
        );
    }

    #[test]
    fn test_emit_wmma_load_partial_label() {
        let instr = wmma_instr(
            PtxOp::WmmaLoadA,
            PtxType::F16,
            regs(0, 8, PtxType::F16),
            vec![ptr_reg(100)],
            Some("m16n16k16"),
        );
        assert_contains_all(&emit_wmma_load(INDENT.into(), &instr, "a"), &[".m16n16k16.row.f16"]);
    }

    #[test]
    fn test_emit_wmma_load_no_srcs() {
        let instr =
            wmma_instr(PtxOp::WmmaLoadA, PtxType::F16, regs(0, 8, PtxType::F16), vec![], None);
        assert_contains_all(&emit_wmma_load(INDENT.into(), &instr, "a"), &["wmma.load.a", "16"]);
    }

    // === emit_wmma_mma ===

    #[test]
    fn test_emit_wmma_mma_default() {
        let instr = wmma_instr(
            PtxOp::WmmaMma,
            PtxType::F32,
            regs(0, 8, PtxType::F32),
            regs(100, 24, PtxType::F16),
            None,
        );
        assert_contains_all(
            &emit_wmma_mma(INDENT.into(), &instr),
            &["wmma.mma.sync.aligned.m16n16k16.row.col.f32.f32"],
        );
    }

    #[test]
    fn test_emit_wmma_mma_with_label() {
        let instr = wmma_instr(
            PtxOp::WmmaMma,
            PtxType::F16,
            regs(0, 4, PtxType::F16),
            regs(100, 12, PtxType::F16),
            Some("m8n8k4.row.row.f16.f16"),
        );
        assert_contains_all(
            &emit_wmma_mma(INDENT.into(), &instr),
            &["wmma.mma.sync.aligned.m8n8k4.row.row.f16.f16"],
        );
    }

    #[test]
    fn test_emit_wmma_mma_partial_srcs() {
        let instr = wmma_instr(
            PtxOp::WmmaMma,
            PtxType::F32,
            regs(0, 8, PtxType::F32),
            regs(100, 16, PtxType::F16),
            None,
        );
        assert_contains_all(&emit_wmma_mma(INDENT.into(), &instr), &["wmma.mma.sync.aligned"]);
    }

    // === emit_wmma_store ===

    fn store_srcs(ptr: u32, frag_count: u32, frag_ty: PtxType, stride: u64) -> Vec<Operand> {
        std::iter::once(ptr_reg(ptr))
            .chain(regs(1, frag_count, frag_ty))
            .chain(std::iter::once(Operand::ImmU64(stride)))
            .collect()
    }

    #[test]
    fn test_emit_wmma_store_default() {
        let instr = wmma_instr(
            PtxOp::WmmaStoreD,
            PtxType::F32,
            vec![],
            store_srcs(0, 8, PtxType::F32, 16),
            None,
        );
        assert_contains_all(
            &emit_wmma_store(INDENT.into(), &instr),
            &["wmma.store.d.sync.aligned", ".m16n16k16.row.f32"],
        );
    }

    #[test]
    fn test_emit_wmma_store_with_label() {
        let instr = wmma_instr(
            PtxOp::WmmaStoreD,
            PtxType::F32,
            vec![],
            store_srcs(0, 8, PtxType::F32, 32),
            Some("m16n16k16.col.f32.stride.32"),
        );
        assert_contains_all(&emit_wmma_store(INDENT.into(), &instr), &[".m16n16k16.col.f32"]);
    }

    #[test]
    fn test_emit_wmma_store_partial_label() {
        let instr = wmma_instr(
            PtxOp::WmmaStoreD,
            PtxType::F32,
            vec![],
            store_srcs(0, 4, PtxType::F32, 16),
            Some("m8n8k4"),
        );
        assert_contains_all(&emit_wmma_store(INDENT.into(), &instr), &[".m16n16k16.row.f32"]);
    }

    #[test]
    fn test_emit_wmma_store_stride_from_label() {
        let instr = wmma_instr(
            PtxOp::WmmaStoreD,
            PtxType::F32,
            vec![],
            vec![],
            Some("m16n16k16.row.f32.stride.64"),
        );
        assert_contains_all(&emit_wmma_store(INDENT.into(), &instr), &["64"]);
    }

    #[test]
    fn test_emit_wmma_store_empty_srcs() {
        let instr = wmma_instr(PtxOp::WmmaStoreD, PtxType::F32, vec![], vec![], None);
        assert_contains_all(&emit_wmma_store(INDENT.into(), &instr), &["wmma.store.d"]);
    }
}
