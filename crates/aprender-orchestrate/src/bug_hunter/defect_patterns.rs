//! Defect pattern tables for bug hunting (QA-002 split).

use super::types::{DefectCategory, FindingSeverity};

/// Base defect patterns. When PMAT SATD is active, TODO/FIXME/HACK/XXX
/// are handled by pmat and excluded here to avoid duplicates.
pub(super) fn base_defect_patterns(
    pmat_satd_active: bool,
) -> Vec<(&'static str, DefectCategory, FindingSeverity, f64)> {
    let mut patterns = Vec::new();
    if !pmat_satd_active {
        patterns.extend([
            ("TODO", DefectCategory::LogicErrors, FindingSeverity::Low, 0.3),
            ("FIXME", DefectCategory::LogicErrors, FindingSeverity::Medium, 0.5),
            ("HACK", DefectCategory::LogicErrors, FindingSeverity::Medium, 0.5),
            ("XXX", DefectCategory::LogicErrors, FindingSeverity::Medium, 0.5),
        ]);
    }
    // Always include runtime fault patterns
    patterns.extend([
        ("unwrap()", DefectCategory::LogicErrors, FindingSeverity::Medium, 0.4),
        ("expect(", DefectCategory::LogicErrors, FindingSeverity::Low, 0.3),
        // SAFETY: no actual unsafe code — string literals for pattern matching
        ("unsafe {", DefectCategory::MemorySafety, FindingSeverity::High, 0.7),
        ("transmute", DefectCategory::MemorySafety, FindingSeverity::High, 0.8),
        ("panic!", DefectCategory::LogicErrors, FindingSeverity::Medium, 0.5),
        ("unreachable!", DefectCategory::LogicErrors, FindingSeverity::Low, 0.3),
    ]);
    patterns
}

/// GPU/CUDA and cross-cutting defect patterns (always active).
pub(super) fn gpu_and_crosscutting_patterns(
) -> Vec<(&'static str, DefectCategory, FindingSeverity, f64)> {
    vec![
        // Detects GPU kernel defects: CUDA errors and invalid PTX code
        ("CUDA_ERROR", DefectCategory::GpuKernelBugs, FindingSeverity::Critical, 0.9),
        ("INVALID_PTX", DefectCategory::GpuKernelBugs, FindingSeverity::Critical, 0.95),
        ("PTX error", DefectCategory::GpuKernelBugs, FindingSeverity::Critical, 0.9),
        ("kernel fail", DefectCategory::GpuKernelBugs, FindingSeverity::High, 0.8),
        ("cuBLAS fallback", DefectCategory::GpuKernelBugs, FindingSeverity::High, 0.7),
        ("cuDNN fallback", DefectCategory::GpuKernelBugs, FindingSeverity::High, 0.7),
        // Silent degradation - error swallowing without alerting
        (".unwrap_or_else(|_|", DefectCategory::SilentDegradation, FindingSeverity::High, 0.7),
        ("if let Err(_) =", DefectCategory::SilentDegradation, FindingSeverity::Medium, 0.5),
        ("Err(_) => {}", DefectCategory::SilentDegradation, FindingSeverity::High, 0.75),
        ("Ok(_) => {}", DefectCategory::SilentDegradation, FindingSeverity::Medium, 0.4),
        ("// fallback", DefectCategory::SilentDegradation, FindingSeverity::Medium, 0.5),
        ("// degraded", DefectCategory::SilentDegradation, FindingSeverity::High, 0.7),
        // Identifies skipped test markers (#[ignore], // skip)
        ("#[ignore]", DefectCategory::TestDebt, FindingSeverity::High, 0.7),
        ("// skip", DefectCategory::TestDebt, FindingSeverity::Medium, 0.5),
        ("// skipped", DefectCategory::TestDebt, FindingSeverity::Medium, 0.5),
        ("// broken", DefectCategory::TestDebt, FindingSeverity::High, 0.8),
        ("// fails", DefectCategory::TestDebt, FindingSeverity::High, 0.75),
        ("// disabled", DefectCategory::TestDebt, FindingSeverity::Medium, 0.6),
        ("test removed", DefectCategory::TestDebt, FindingSeverity::Critical, 0.9),
        ("were removed", DefectCategory::TestDebt, FindingSeverity::Critical, 0.9),
        ("tests hang", DefectCategory::TestDebt, FindingSeverity::Critical, 0.9),
        ("hang during", DefectCategory::TestDebt, FindingSeverity::High, 0.8),
        ("compilation hang", DefectCategory::TestDebt, FindingSeverity::High, 0.8),
        // Dimension-related GPU bugs (hidden_dim limits)
        ("hidden_dim >=", DefectCategory::GpuKernelBugs, FindingSeverity::High, 0.7),
        ("hidden_dim >", DefectCategory::GpuKernelBugs, FindingSeverity::High, 0.7),
        ("// 1536", DefectCategory::GpuKernelBugs, FindingSeverity::Medium, 0.5),
        ("// 2048", DefectCategory::GpuKernelBugs, FindingSeverity::Medium, 0.5),
        ("model dimensions", DefectCategory::GpuKernelBugs, FindingSeverity::Medium, 0.5),
        // Hidden debt - euphemisms that hide technical debt (PMAT issue #149)
        // These appear in doc comments and regular comments
        ("placeholder", DefectCategory::HiddenDebt, FindingSeverity::High, 0.75),
        ("stub", DefectCategory::HiddenDebt, FindingSeverity::High, 0.7),
        ("dummy", DefectCategory::HiddenDebt, FindingSeverity::High, 0.7),
        ("fake", DefectCategory::HiddenDebt, FindingSeverity::Medium, 0.6),
        ("mock", DefectCategory::HiddenDebt, FindingSeverity::Medium, 0.5),
        ("simplified", DefectCategory::HiddenDebt, FindingSeverity::Medium, 0.6),
        ("for demonstration", DefectCategory::HiddenDebt, FindingSeverity::High, 0.75),
        ("demo only", DefectCategory::HiddenDebt, FindingSeverity::High, 0.8),
        ("not implemented", DefectCategory::HiddenDebt, FindingSeverity::Critical, 0.9),
        ("unimplemented", DefectCategory::HiddenDebt, FindingSeverity::Critical, 0.9),
        ("temporary", DefectCategory::HiddenDebt, FindingSeverity::Medium, 0.6),
        ("hardcoded", DefectCategory::HiddenDebt, FindingSeverity::Medium, 0.5),
        ("hard-coded", DefectCategory::HiddenDebt, FindingSeverity::Medium, 0.5),
        ("magic number", DefectCategory::HiddenDebt, FindingSeverity::Medium, 0.5),
        ("workaround", DefectCategory::HiddenDebt, FindingSeverity::Medium, 0.6),
        ("quick fix", DefectCategory::HiddenDebt, FindingSeverity::High, 0.7),
        ("quick-fix", DefectCategory::HiddenDebt, FindingSeverity::High, 0.7),
        ("bandaid", DefectCategory::HiddenDebt, FindingSeverity::High, 0.7),
        ("band-aid", DefectCategory::HiddenDebt, FindingSeverity::High, 0.7),
        ("kludge", DefectCategory::HiddenDebt, FindingSeverity::High, 0.75),
        ("tech debt", DefectCategory::HiddenDebt, FindingSeverity::High, 0.8),
        ("technical debt", DefectCategory::HiddenDebt, FindingSeverity::High, 0.8),
    ]
}
