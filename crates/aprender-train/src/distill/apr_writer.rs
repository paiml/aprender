//! APR-writer head-count fidelity guard for distilled students.
//!
//! Contract: `contracts/entrenar/qlora-distillation-v1.yaml`
//! Binding: INV-DISTILL-005.
//!
//! MODEL-1 v2 `.apr` metadata reported `56` attention heads and `8` KV
//! heads for the Qwen2.5-7B student. The teacher checkpoint was
//! `28/4`. The runtime still loaded correctly (shapes are
//! authoritative), but `apr inspect` output was misleading and blocked
//! downstream QA. This module ensures the writer propagates
//! teacher-observed head counts rather than a hardcoded default.

/// Head-count block copied from the teacher manifest and written to
/// the student's APR metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeadCount {
    pub attention_heads: u32,
    pub kv_heads: u32,
}

impl HeadCount {
    pub const fn new(attention_heads: u32, kv_heads: u32) -> Self {
        Self { attention_heads, kv_heads }
    }
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum AprWriterError {
    #[error(
        "head count drift: student apr={student_heads}/{student_kv}, teacher={teacher_heads}/{teacher_kv}"
    )]
    HeadCountDrift { student_heads: u32, student_kv: u32, teacher_heads: u32, teacher_kv: u32 },
}

/// INV-DISTILL-005: a distilled APR metadata block MUST match the
/// teacher's head counts. The writer is expected to CALL this before
/// emitting the `.apr` payload, and the `apr inspect` round-trip is
/// expected to verify post-hoc.
pub fn check_head_count_fidelity(
    student: HeadCount,
    teacher: HeadCount,
) -> Result<(), AprWriterError> {
    if student != teacher {
        return Err(AprWriterError::HeadCountDrift {
            student_heads: student.attention_heads,
            student_kv: student.kv_heads,
            teacher_heads: teacher.attention_heads,
            teacher_kv: teacher.kv_heads,
        });
    }
    Ok(())
}

/// Wrap the writer entry point: given the teacher's head count, produce
/// the block the student APR should write. Explicit "copy from teacher"
/// rather than any config-default, which is what let v2 drift to 56/8.
pub fn inherit_teacher_head_count(teacher: HeadCount) -> HeadCount {
    teacher
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Qwen2.5-Coder-7B-Instruct authoritative head counts.
    const QWEN_7B_TEACHER: HeadCount = HeadCount::new(28, 4);

    /// The v2 regression: metadata said 56/8 despite tensors encoding
    /// the correct 28/4.
    const V2_STUDENT_DRIFT: HeadCount = HeadCount::new(56, 8);

    /// FALSIFY-DISTILL-005: the writer must emit teacher-matching head
    /// counts, not a hardcoded default.
    #[test]
    fn inherits_teacher_head_count() {
        let student = inherit_teacher_head_count(QWEN_7B_TEACHER);
        assert_eq!(student.attention_heads, 28);
        assert_eq!(student.kv_heads, 4);
    }

    #[test]
    fn fidelity_check_rejects_v2_drift() {
        let err = check_head_count_fidelity(V2_STUDENT_DRIFT, QWEN_7B_TEACHER).unwrap_err();
        match err {
            AprWriterError::HeadCountDrift {
                student_heads,
                student_kv,
                teacher_heads,
                teacher_kv,
            } => {
                assert_eq!(student_heads, 56);
                assert_eq!(student_kv, 8);
                assert_eq!(teacher_heads, 28);
                assert_eq!(teacher_kv, 4);
            }
        }
    }

    #[test]
    fn fidelity_check_accepts_matching_heads() {
        let student = inherit_teacher_head_count(QWEN_7B_TEACHER);
        assert!(check_head_count_fidelity(student, QWEN_7B_TEACHER).is_ok());
    }
}
