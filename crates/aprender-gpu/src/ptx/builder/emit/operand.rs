//! Operand emission utilities
//!
//! Shared operand formatting for all emission modules

use crate::ptx::instructions::Operand;
use std::fmt::Write;

/// Emit an operand as PTX string (allocating version)
pub(crate) fn emit_operand(op: &Operand) -> String {
    match op {
        Operand::Reg(vreg) => vreg.to_ptx_string(),
        Operand::SpecialReg(sreg) => sreg.to_ptx_string().to_string(),
        Operand::ImmI64(v) => v.to_string(),
        Operand::ImmU64(v) => v.to_string(),
        Operand::ImmF32(v) => format!("0F{:08X}", v.to_bits()),
        Operand::ImmF64(v) => format!("0D{:016X}", v.to_bits()),
        Operand::Param(name) => format!("[{}]", name),
        Operand::Addr { base, offset } => {
            if *offset == 0 {
                format!("[{}]", base.to_ptx_string())
            } else {
                format!("[{}+{}]", base.to_ptx_string(), offset)
            }
        }
        Operand::Label(name) => name.clone(),
    }
}

/// Emit shared memory operand with proper addressing syntax
pub(crate) fn emit_shared_mem_operand(op: &Operand) -> String {
    match op {
        Operand::Reg(vreg) => format!("[{}]", vreg.to_ptx_string()),
        Operand::Addr { base, offset } => {
            if *offset == 0 {
                format!("[{}]", base.to_ptx_string())
            } else {
                format!("[{}+{}]", base.to_ptx_string(), offset)
            }
        }
        _ => emit_operand(op),
    }
}

/// Emit global memory operand with proper [addr] syntax
pub(crate) fn emit_global_mem_operand(op: &Operand) -> String {
    match op {
        Operand::Reg(vreg) => format!("[{}]", vreg.to_ptx_string()),
        Operand::Addr { base, offset } => {
            if *offset == 0 {
                format!("[{}]", base.to_ptx_string())
            } else {
                format!("[{}+{}]", base.to_ptx_string(), offset)
            }
        }
        _ => emit_operand(op),
    }
}

/// Write operand directly to buffer (zero allocation)
#[inline]
pub(crate) fn write_operand(op: &Operand, out: &mut String) {
    match op {
        Operand::Reg(vreg) => {
            let _ = write!(out, "{}", vreg);
        }
        Operand::SpecialReg(sreg) => out.push_str(sreg.to_ptx_string()),
        Operand::ImmI64(v) => {
            let _ = write!(out, "{}", v);
        }
        Operand::ImmU64(v) => {
            let _ = write!(out, "{}", v);
        }
        Operand::ImmF32(v) => {
            let _ = write!(out, "0F{:08X}", v.to_bits());
        }
        Operand::ImmF64(v) => {
            let _ = write!(out, "0D{:016X}", v.to_bits());
        }
        Operand::Param(name) => {
            let _ = write!(out, "[{}]", name);
        }
        Operand::Addr { base, offset } => {
            if *offset == 0 {
                let _ = write!(out, "[{}]", base);
            } else {
                let _ = write!(out, "[{}+{}]", base, offset);
            }
        }
        Operand::Label(name) => out.push_str(name),
    }
}

/// Write memory operand with bracket syntax directly to buffer
#[inline]
pub(crate) fn write_mem_operand(op: &Operand, out: &mut String) {
    match op {
        Operand::Reg(vreg) => {
            let _ = write!(out, "[{}]", vreg);
        }
        Operand::Addr { base, offset } => {
            if *offset == 0 {
                let _ = write!(out, "[{}]", base);
            } else {
                let _ = write!(out, "[{}+{}]", base, offset);
            }
        }
        _ => write_operand(op, out),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ptx::registers::{PtxReg, VirtualReg};
    use crate::ptx::types::PtxType;

    #[test]
    fn test_emit_operand_reg() {
        let vreg = VirtualReg::new(0, PtxType::F32);
        let result = emit_operand(&Operand::Reg(vreg));
        assert!(result.contains('r') || result.contains('f'));
    }

    #[test]
    fn test_emit_operand_special_reg() {
        let result = emit_operand(&Operand::SpecialReg(PtxReg::TidX));
        assert!(result.contains("tid"));
    }

    #[test]
    fn test_emit_operand_imm_i64() {
        assert_eq!(emit_operand(&Operand::ImmI64(-42)), "-42");
    }

    #[test]
    fn test_emit_operand_imm_u64() {
        assert_eq!(emit_operand(&Operand::ImmU64(42)), "42");
    }

    #[test]
    fn test_emit_operand_imm_f32() {
        let result = emit_operand(&Operand::ImmF32(1.0));
        assert!(result.starts_with("0F"));
    }

    #[test]
    fn test_emit_operand_imm_f64() {
        let result = emit_operand(&Operand::ImmF64(1.0));
        assert!(result.starts_with("0D"));
    }

    #[test]
    fn test_emit_operand_param() {
        let result = emit_operand(&Operand::Param("input".to_string()));
        assert_eq!(result, "[input]");
    }

    #[test]
    fn test_emit_operand_addr_zero_offset() {
        let vreg = VirtualReg::new(0, PtxType::U64);
        let result = emit_operand(&Operand::Addr { base: vreg, offset: 0 });
        assert!(result.starts_with('[') && result.ends_with(']'));
        assert!(!result.contains('+'));
    }

    #[test]
    fn test_emit_operand_label() {
        let result = emit_operand(&Operand::Label("loop_start".to_string()));
        assert_eq!(result, "loop_start");
    }

    #[test]
    fn test_emit_shared_mem_operand_reg() {
        let vreg = VirtualReg::new(0, PtxType::U64);
        let result = emit_shared_mem_operand(&Operand::Reg(vreg));
        assert!(result.starts_with('[') && result.ends_with(']'));
    }

    #[test]
    fn test_emit_shared_mem_operand_addr_zero() {
        let vreg = VirtualReg::new(0, PtxType::U64);
        let result = emit_shared_mem_operand(&Operand::Addr { base: vreg, offset: 0 });
        assert!(result.starts_with('[') && result.ends_with(']'));
        assert!(!result.contains('+'));
    }

    #[test]
    fn test_emit_shared_mem_operand_fallback() {
        let result = emit_shared_mem_operand(&Operand::ImmU64(42));
        assert_eq!(result, "42");
    }

    #[test]
    fn test_emit_global_mem_operand_reg() {
        let vreg = VirtualReg::new(0, PtxType::U64);
        let result = emit_global_mem_operand(&Operand::Reg(vreg));
        assert!(result.starts_with('[') && result.ends_with(']'));
    }

    #[test]
    fn test_emit_global_mem_operand_addr_zero() {
        let vreg = VirtualReg::new(0, PtxType::U64);
        let result = emit_global_mem_operand(&Operand::Addr { base: vreg, offset: 0 });
        assert!(!result.contains('+'));
    }

    #[test]
    fn test_emit_global_mem_operand_fallback() {
        let result = emit_global_mem_operand(&Operand::Label("addr".to_string()));
        assert_eq!(result, "addr");
    }

    #[test]
    fn test_write_operand_reg() {
        let vreg = VirtualReg::new(0, PtxType::F32);
        let mut out = String::new();
        write_operand(&Operand::Reg(vreg), &mut out);
        assert!(!out.is_empty());
    }

    #[test]
    fn test_write_operand_special_reg() {
        let mut out = String::new();
        write_operand(&Operand::SpecialReg(PtxReg::TidX), &mut out);
        assert!(out.contains("tid"));
    }

    #[test]
    fn test_write_operand_imm_i64() {
        let mut out = String::new();
        write_operand(&Operand::ImmI64(-99), &mut out);
        assert_eq!(out, "-99");
    }

    #[test]
    fn test_write_operand_imm_u64() {
        let mut out = String::new();
        write_operand(&Operand::ImmU64(99), &mut out);
        assert_eq!(out, "99");
    }

    #[test]
    fn test_write_operand_imm_f32() {
        let mut out = String::new();
        write_operand(&Operand::ImmF32(2.5), &mut out);
        assert!(out.starts_with("0F"));
    }

    #[test]
    fn test_write_operand_imm_f64() {
        let mut out = String::new();
        write_operand(&Operand::ImmF64(2.5), &mut out);
        assert!(out.starts_with("0D"));
    }

    #[test]
    fn test_write_operand_param() {
        let mut out = String::new();
        write_operand(&Operand::Param("ptr".to_string()), &mut out);
        assert_eq!(out, "[ptr]");
    }

    #[test]
    fn test_write_operand_addr_zero() {
        let vreg = VirtualReg::new(0, PtxType::U64);
        let mut out = String::new();
        write_operand(&Operand::Addr { base: vreg, offset: 0 }, &mut out);
        assert!(!out.contains('+'));
    }

    #[test]
    fn test_write_operand_addr_nonzero() {
        let vreg = VirtualReg::new(0, PtxType::U64);
        let mut out = String::new();
        write_operand(&Operand::Addr { base: vreg, offset: 64 }, &mut out);
        assert!(out.contains("+64"));
    }

    #[test]
    fn test_write_operand_label() {
        let mut out = String::new();
        write_operand(&Operand::Label("done".to_string()), &mut out);
        assert_eq!(out, "done");
    }

    #[test]
    fn test_write_mem_operand_reg() {
        let vreg = VirtualReg::new(0, PtxType::U64);
        let mut out = String::new();
        write_mem_operand(&Operand::Reg(vreg), &mut out);
        assert!(out.starts_with('[') && out.ends_with(']'));
    }

    #[test]
    fn test_write_mem_operand_addr_zero() {
        let vreg = VirtualReg::new(0, PtxType::U64);
        let mut out = String::new();
        write_mem_operand(&Operand::Addr { base: vreg, offset: 0 }, &mut out);
        assert!(!out.contains('+'));
    }

    #[test]
    fn test_write_mem_operand_addr_nonzero() {
        let vreg = VirtualReg::new(0, PtxType::U64);
        let mut out = String::new();
        write_mem_operand(&Operand::Addr { base: vreg, offset: 32 }, &mut out);
        assert!(out.contains("+32"));
    }

    #[test]
    fn test_write_mem_operand_fallback() {
        let mut out = String::new();
        write_mem_operand(&Operand::ImmU64(100), &mut out);
        assert_eq!(out, "100");
    }
}
