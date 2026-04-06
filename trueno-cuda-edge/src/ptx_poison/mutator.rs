//! PTX mutation operators for poison testing.
//!
//! [`PtxMutator`] defines 8 mutation operators that corrupt PTX assembly
//! in specific ways. If the test suite still passes after a mutation, the
//! tests have insufficient coverage for that code path.

use serde::{Deserialize, Serialize};

/// A mutation operator that transforms PTX source code.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PtxMutator {
    /// Replace `add` with `sub` (arithmetic inversion).
    FlipAddSub,
    /// Replace `mul` with `div` (arithmetic inversion).
    FlipMulDiv,
    /// Replace `.lo` with `.hi` in mul instructions.
    FlipMulLoHi,
    /// Replace `setp.lt` with `setp.ge` (predicate inversion).
    InvertPredicate,
    /// Remove a `bar.sync` instruction (synchronization removal).
    RemoveBarrier,
    /// Replace a register operand with `%r0` (zero register).
    ZeroRegister,
    /// Change `.f32` to `.f64` (precision widening).
    WidenPrecision,
    /// Replace a shared memory address with a global address.
    SwapMemorySpace,
}

impl PtxMutator {
    /// Apply this mutation to PTX source code, returning the mutated source.
    ///
    /// Returns `None` if the mutation target was not found in the source.
    #[must_use]
    pub fn apply(&self, source: &str) -> Option<String> {
        match self {
            Self::FlipAddSub => {
                if source.contains("add.") {
                    Some(source.replacen("add.", "sub.", 1))
                } else {
                    None
                }
            }
            Self::FlipMulDiv => {
                if source.contains("mul.") {
                    Some(source.replacen("mul.", "div.", 1))
                } else {
                    None
                }
            }
            Self::FlipMulLoHi => {
                if source.contains(".lo.") {
                    Some(source.replacen(".lo.", ".hi.", 1))
                } else {
                    None
                }
            }
            Self::InvertPredicate => {
                if source.contains("setp.lt") {
                    Some(source.replacen("setp.lt", "setp.ge", 1))
                } else if source.contains("setp.gt") {
                    Some(source.replacen("setp.gt", "setp.le", 1))
                } else if source.contains("setp.eq") {
                    Some(source.replacen("setp.eq", "setp.ne", 1))
                } else {
                    None
                }
            }
            Self::RemoveBarrier => {
                if source.contains("bar.sync") {
                    // Remove the first bar.sync line
                    let mut lines: Vec<&str> = source.lines().collect();
                    if let Some(pos) = lines.iter().position(|l| l.contains("bar.sync")) {
                        lines.remove(pos);
                        Some(lines.join("\n"))
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            Self::ZeroRegister => {
                // Replace first non-zero register with %r0
                if source.contains("%r1") {
                    Some(source.replacen("%r1", "%r0", 1))
                } else if source.contains("%r2") {
                    Some(source.replacen("%r2", "%r0", 1))
                } else {
                    None
                }
            }
            Self::WidenPrecision => {
                if source.contains(".f32") {
                    Some(source.replacen(".f32", ".f64", 1))
                } else {
                    None
                }
            }
            Self::SwapMemorySpace => {
                if source.contains(".shared") {
                    Some(source.replacen(".shared", ".global", 1))
                } else {
                    None
                }
            }
        }
    }

    /// Human-readable description of this mutation.
    #[must_use]
    pub fn description(&self) -> &'static str {
        match self {
            Self::FlipAddSub => "flip add to sub",
            Self::FlipMulDiv => "flip mul to div",
            Self::FlipMulLoHi => "flip mul.lo to mul.hi",
            Self::InvertPredicate => "invert predicate comparison",
            Self::RemoveBarrier => "remove barrier synchronization",
            Self::ZeroRegister => "replace register with zero",
            Self::WidenPrecision => "widen f32 to f64 precision",
            Self::SwapMemorySpace => "swap shared to global memory",
        }
    }
}

/// Returns the default set of all 8 mutation operators.
#[must_use]
pub fn default_mutators() -> Vec<PtxMutator> {
    vec![
        PtxMutator::FlipAddSub,
        PtxMutator::FlipMulDiv,
        PtxMutator::FlipMulLoHi,
        PtxMutator::InvertPredicate,
        PtxMutator::RemoveBarrier,
        PtxMutator::ZeroRegister,
        PtxMutator::WidenPrecision,
        PtxMutator::SwapMemorySpace,
    ]
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::disallowed_methods)]
mod tests {
    use super::*;

    #[test]
    fn flip_add_sub() {
        let src = "add.f32 %r1, %r2, %r3;";
        let mutated = PtxMutator::FlipAddSub.apply(src).unwrap();
        assert_eq!(mutated, "sub.f32 %r1, %r2, %r3;");
    }

    #[test]
    fn flip_add_sub_not_found() {
        let src = "mul.f32 %r1, %r2, %r3;";
        assert!(PtxMutator::FlipAddSub.apply(src).is_none());
    }

    #[test]
    fn flip_mul_div() {
        let src = "mul.f32 %r1, %r2, %r3;";
        let mutated = PtxMutator::FlipMulDiv.apply(src).unwrap();
        assert_eq!(mutated, "div.f32 %r1, %r2, %r3;");
    }

    #[test]
    fn flip_mul_lo_hi() {
        let src = "mul.lo.s32 %r1, %r2, %r3;";
        let mutated = PtxMutator::FlipMulLoHi.apply(src).unwrap();
        assert_eq!(mutated, "mul.hi.s32 %r1, %r2, %r3;");
    }

    #[test]
    fn invert_predicate_lt_to_ge() {
        let src = "setp.lt.f32 %p1, %r1, %r2;";
        let mutated = PtxMutator::InvertPredicate.apply(src).unwrap();
        assert!(mutated.contains("setp.ge"));
    }

    #[test]
    fn invert_predicate_eq_to_ne() {
        let src = "setp.eq.s32 %p1, %r1, %r2;";
        let mutated = PtxMutator::InvertPredicate.apply(src).unwrap();
        assert!(mutated.contains("setp.ne"));
    }

    #[test]
    fn remove_barrier() {
        let src = "add.f32 %r1, %r2, %r3;\nbar.sync 0;\nmul.f32 %r4, %r5, %r6;";
        let mutated = PtxMutator::RemoveBarrier.apply(src).unwrap();
        assert!(!mutated.contains("bar.sync"));
        assert!(mutated.contains("add.f32"));
        assert!(mutated.contains("mul.f32"));
    }

    #[test]
    fn zero_register() {
        let src = "add.f32 %r1, %r2, %r3;";
        let mutated = PtxMutator::ZeroRegister.apply(src).unwrap();
        assert!(mutated.contains("%r0"));
    }

    #[test]
    fn widen_precision() {
        let src = "add.f32 %r1, %r2, %r3;";
        let mutated = PtxMutator::WidenPrecision.apply(src).unwrap();
        assert!(mutated.contains(".f64"));
    }

    #[test]
    fn swap_memory_space() {
        let src = "ld.shared.f32 %r1, [%r2];";
        let mutated = PtxMutator::SwapMemorySpace.apply(src).unwrap();
        assert!(mutated.contains(".global"));
        assert!(!mutated.contains(".shared"));
    }

    #[test]
    fn default_mutators_has_8_entries() {
        assert_eq!(default_mutators().len(), 8);
    }
}
