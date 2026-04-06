//! Control flow operation emission
//!
//! Handles: Bra, Ret, Bar, MemBar, Setp, Exit

use crate::ptx::instructions::{PtxInstruction, PtxOp};
use std::fmt::Write;

/// Emit control flow opcode - returns Some(full_instruction) for early-return ops
pub(crate) fn emit_control_opcode(instr: &PtxInstruction, prefix: &str) -> Option<String> {
    match instr.op {
        PtxOp::Bra => {
            if let Some(label) = &instr.label {
                return Some(format!("{}bra {};\n", prefix, label));
            }
            None
        }
        PtxOp::Ret => Some(format!("{}ret;\n", prefix)),
        PtxOp::Bar => {
            let barrier_id = instr.label.as_deref().unwrap_or("sync 0");
            Some(format!("{}bar.{};\n", prefix, barrier_id))
        }
        PtxOp::MemBar => {
            let scope = instr.label.as_deref().unwrap_or("cta");
            Some(format!("{}membar.{};\n", prefix, scope))
        }
        _ => None,
    }
}

/// Emit setp opcode with comparison operator
pub(crate) fn emit_setp_opcode(instr: &PtxInstruction, s: &mut String) {
    let cmp = instr.label.as_deref().unwrap_or("eq");
    let _ = write!(s, "setp.{}", cmp);
}

/// Check if this is a control flow operation
pub(crate) fn is_control_op(op: &PtxOp) -> bool {
    matches!(op, PtxOp::Bra | PtxOp::Ret | PtxOp::Bar | PtxOp::MemBar | PtxOp::Setp)
}

/// Check if this is an early-return control op
pub(crate) fn is_early_return_op(op: &PtxOp) -> bool {
    matches!(op, PtxOp::Bra | PtxOp::Ret | PtxOp::Bar | PtxOp::MemBar)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ptx::types::PtxType;

    fn make_instr(op: PtxOp, label: Option<&str>) -> PtxInstruction {
        PtxInstruction {
            op,
            ty: PtxType::U32,
            src_type: None,
            dst: None,
            dsts: vec![],
            srcs: vec![],
            label: label.map(String::from),
            predicate: None,
            state_space: None,
            rounding: None,
        }
    }

    #[test]
    fn test_emit_bra_with_label() {
        let instr = make_instr(PtxOp::Bra, Some("loop_start"));
        let result = emit_control_opcode(&instr, "    ");
        assert_eq!(result, Some("    bra loop_start;\n".to_string()));
    }

    #[test]
    fn test_emit_bra_without_label() {
        let instr = make_instr(PtxOp::Bra, None);
        let result = emit_control_opcode(&instr, "    ");
        assert_eq!(result, None);
    }

    #[test]
    fn test_emit_ret() {
        let instr = make_instr(PtxOp::Ret, None);
        let result = emit_control_opcode(&instr, "    ");
        assert_eq!(result, Some("    ret;\n".to_string()));
    }

    #[test]
    fn test_emit_bar_with_label() {
        let instr = make_instr(PtxOp::Bar, Some("sync 1"));
        let result = emit_control_opcode(&instr, "    ");
        assert_eq!(result, Some("    bar.sync 1;\n".to_string()));
    }

    #[test]
    fn test_emit_bar_default() {
        let instr = make_instr(PtxOp::Bar, None);
        let result = emit_control_opcode(&instr, "    ");
        assert_eq!(result, Some("    bar.sync 0;\n".to_string()));
    }

    #[test]
    fn test_emit_membar_with_scope() {
        let instr = make_instr(PtxOp::MemBar, Some("gl"));
        let result = emit_control_opcode(&instr, "    ");
        assert_eq!(result, Some("    membar.gl;\n".to_string()));
    }

    #[test]
    fn test_emit_membar_default() {
        let instr = make_instr(PtxOp::MemBar, None);
        let result = emit_control_opcode(&instr, "    ");
        assert_eq!(result, Some("    membar.cta;\n".to_string()));
    }

    #[test]
    fn test_emit_control_opcode_non_control() {
        let instr = make_instr(PtxOp::Add, None);
        let result = emit_control_opcode(&instr, "    ");
        assert_eq!(result, None);
    }

    #[test]
    fn test_emit_setp_with_cmp() {
        let instr = make_instr(PtxOp::Setp, Some("lt"));
        let mut s = String::new();
        emit_setp_opcode(&instr, &mut s);
        assert_eq!(s, "setp.lt");
    }

    #[test]
    fn test_emit_setp_default() {
        let instr = make_instr(PtxOp::Setp, None);
        let mut s = String::new();
        emit_setp_opcode(&instr, &mut s);
        assert_eq!(s, "setp.eq");
    }

    #[test]
    fn test_is_control_op() {
        assert!(is_control_op(&PtxOp::Bra));
        assert!(is_control_op(&PtxOp::Ret));
        assert!(is_control_op(&PtxOp::Bar));
        assert!(is_control_op(&PtxOp::MemBar));
        assert!(is_control_op(&PtxOp::Setp));
        assert!(!is_control_op(&PtxOp::Add));
    }

    #[test]
    fn test_is_early_return_op() {
        assert!(is_early_return_op(&PtxOp::Bra));
        assert!(is_early_return_op(&PtxOp::Ret));
        assert!(is_early_return_op(&PtxOp::Bar));
        assert!(is_early_return_op(&PtxOp::MemBar));
        assert!(!is_early_return_op(&PtxOp::Setp));
        assert!(!is_early_return_op(&PtxOp::Add));
    }
}
