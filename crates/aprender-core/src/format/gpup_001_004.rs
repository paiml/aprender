// SHIP-TWO-001 — `apr-gpu-presence-v1` algorithm-level PARTIAL
// discharge for FALSIFY-GPU-PRESENCE-001..004.
//
// Contract: `contracts/apr-gpu-presence-v1.yaml`.
// Spec: `docs/specifications/aprender-train/ship-two-models-spec.md`.
//
// ## What this file proves NOW (PARTIAL_ALGORITHM_LEVEL)
//
// Four GPU-presence disambiguation gates from paiml/aprender#624:
//
// - GPU-PRESENCE-001 (JSON has gpu_present field): apr gpu --json
//   output contains a top-level "gpu_present" boolean.
// - GPU-PRESENCE-002 (no-GPU ⇒ false): uuid="GPU-unknown" OR total_mb=0
//   ⇒ gpu_present = false.
// - GPU-PRESENCE-003 (text says "No discrete GPU"): when gpu_present=false,
//   text output contains "No discrete GPU" (case-insensitive).
// - GPU-PRESENCE-004 (real GPU ⇒ true): uuid≠"GPU-unknown" AND total_mb>0
//   ⇒ gpu_present = true.

/// Sentinel value emitted by entrenar's detect_gpu_uuid() fallback.
pub const AC_GPU_UNKNOWN_SENTINEL: &str = "GPU-unknown";

/// Required JSON field name (boolean).
pub const AC_GPU_PRESENT_FIELD: &str = "gpu_present";

/// Text-output substring that must appear when gpu_present=false.
pub const AC_GPU_NO_GPU_TEXT_PHRASES: [&str; 2] = ["no discrete gpu", "no gpu detected"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpupVerdict {
    Pass,
    Fail,
}

// -----------------------------------------------------------------------------
// Verdict 1: GPU-PRESENCE-001 — JSON has gpu_present field.
// -----------------------------------------------------------------------------

/// Pass iff `present_fields` contains `gpu_present`.
#[must_use]
pub fn verdict_from_json_has_gpu_present_field(present_fields: &[&str]) -> GpupVerdict {
    if present_fields.iter().any(|&f| f == AC_GPU_PRESENT_FIELD) {
        GpupVerdict::Pass
    } else {
        GpupVerdict::Fail
    }
}

// -----------------------------------------------------------------------------
// Verdict 2: GPU-PRESENCE-002 — no-GPU host has gpu_present=false.
// -----------------------------------------------------------------------------

/// Implication: uuid="GPU-unknown" OR total_mb=0 ⇒ gpu_present = false.
#[must_use]
pub fn verdict_from_no_gpu_implies_false(
    uuid: &str,
    total_mb: u64,
    gpu_present: bool,
) -> GpupVerdict {
    let is_no_gpu_signal = uuid == AC_GPU_UNKNOWN_SENTINEL || total_mb == 0;
    if is_no_gpu_signal && gpu_present {
        // Sentinel detected but gpu_present still true ⇒ Fail.
        GpupVerdict::Fail
    } else {
        // Either it's not a no-GPU signal, or it is and gpu_present=false.
        GpupVerdict::Pass
    }
}

// -----------------------------------------------------------------------------
// Verdict 3: GPU-PRESENCE-003 — text says "No discrete GPU".
// -----------------------------------------------------------------------------

/// When `gpu_present=false`, text output must contain one of the
/// no-GPU phrases (case-insensitive). When gpu_present=true, no
/// requirement on text content.
#[must_use]
pub fn verdict_from_text_says_no_gpu(text_output: &str, gpu_present: bool) -> GpupVerdict {
    if gpu_present {
        return GpupVerdict::Pass;
    }
    let lower = text_output.to_ascii_lowercase();
    if AC_GPU_NO_GPU_TEXT_PHRASES
        .iter()
        .any(|p| lower.contains(p))
    {
        GpupVerdict::Pass
    } else {
        GpupVerdict::Fail
    }
}

// -----------------------------------------------------------------------------
// Verdict 4: GPU-PRESENCE-004 — real GPU host has gpu_present=true.
// -----------------------------------------------------------------------------

/// Implication: uuid≠"GPU-unknown" AND total_mb>0 ⇒ gpu_present = true.
#[must_use]
pub fn verdict_from_gpu_present_implies_true(
    uuid: &str,
    total_mb: u64,
    gpu_present: bool,
) -> GpupVerdict {
    let is_real_gpu = uuid != AC_GPU_UNKNOWN_SENTINEL && total_mb > 0;
    if is_real_gpu && !gpu_present {
        // Real GPU present but gpu_present=false ⇒ predicate inverted.
        GpupVerdict::Fail
    } else {
        GpupVerdict::Pass
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Provenance pins.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_sentinel_is_gpu_unknown() {
        assert_eq!(AC_GPU_UNKNOWN_SENTINEL, "GPU-unknown");
    }

    #[test]
    fn provenance_field_name_gpu_present() {
        assert_eq!(AC_GPU_PRESENT_FIELD, "gpu_present");
    }

    #[test]
    fn provenance_text_phrases() {
        assert!(AC_GPU_NO_GPU_TEXT_PHRASES.contains(&"no discrete gpu"));
        assert!(AC_GPU_NO_GPU_TEXT_PHRASES.contains(&"no gpu detected"));
    }

    // -------------------------------------------------------------------------
    // Section 2: GPU-PRESENCE-001 — JSON has gpu_present field.
    // -------------------------------------------------------------------------
    #[test]
    fn gp001_pass_field_present() {
        let fields = vec!["gpu_uuid", "total_mb", "gpu_present"];
        assert_eq!(
            verdict_from_json_has_gpu_present_field(&fields),
            GpupVerdict::Pass
        );
    }

    #[test]
    fn gp001_pass_field_present_alone() {
        let fields = vec!["gpu_present"];
        assert_eq!(
            verdict_from_json_has_gpu_present_field(&fields),
            GpupVerdict::Pass
        );
    }

    #[test]
    fn gp001_fail_field_missing() {
        // Pre-fix bug: only old fields.
        let fields = vec!["gpu_uuid", "total_mb", "free_mb"];
        assert_eq!(
            verdict_from_json_has_gpu_present_field(&fields),
            GpupVerdict::Fail
        );
    }

    #[test]
    fn gp001_fail_empty() {
        let fields: Vec<&str> = vec![];
        assert_eq!(
            verdict_from_json_has_gpu_present_field(&fields),
            GpupVerdict::Fail
        );
    }

    #[test]
    fn gp001_fail_typo_field_name() {
        // Bug: misspelled field.
        let fields = vec!["gpu_uuid", "gpus_present"];
        assert_eq!(
            verdict_from_json_has_gpu_present_field(&fields),
            GpupVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 3: GPU-PRESENCE-002 — no-GPU ⇒ false.
    // -------------------------------------------------------------------------
    #[test]
    fn gp002_pass_no_gpu_correctly_false() {
        // CPU-only intel host: sentinel uuid + 0 MB + gpu_present=false.
        assert_eq!(
            verdict_from_no_gpu_implies_false("GPU-unknown", 0, false),
            GpupVerdict::Pass
        );
    }

    #[test]
    fn gp002_pass_zero_mb_correctly_false() {
        // Real uuid but 0 MB ⇒ still no usable GPU.
        assert_eq!(
            verdict_from_no_gpu_implies_false("GPU-abcd1234", 0, false),
            GpupVerdict::Pass
        );
    }

    #[test]
    fn gp002_pass_real_gpu_true() {
        // Real GPU, real memory, gpu_present=true: not in no-GPU branch.
        assert_eq!(
            verdict_from_no_gpu_implies_false("GPU-abcd1234", 24576, true),
            GpupVerdict::Pass
        );
    }

    #[test]
    fn gp002_fail_unknown_uuid_with_true() {
        // The exact regression: sentinel uuid but gpu_present=true.
        assert_eq!(
            verdict_from_no_gpu_implies_false("GPU-unknown", 0, true),
            GpupVerdict::Fail
        );
    }

    #[test]
    fn gp002_fail_zero_mb_with_true() {
        assert_eq!(
            verdict_from_no_gpu_implies_false("GPU-abcd", 0, true),
            GpupVerdict::Fail
        );
    }

    #[test]
    fn gp002_fail_unknown_uuid_with_real_mb_true() {
        // Even with non-zero mb, sentinel uuid invalidates presence.
        assert_eq!(
            verdict_from_no_gpu_implies_false("GPU-unknown", 24576, true),
            GpupVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 4: GPU-PRESENCE-003 — text says "No discrete GPU".
    // -------------------------------------------------------------------------
    #[test]
    fn gp003_pass_text_contains_no_discrete_gpu() {
        let text = "No discrete GPU detected on this host.";
        assert_eq!(
            verdict_from_text_says_no_gpu(text, false),
            GpupVerdict::Pass
        );
    }

    #[test]
    fn gp003_pass_text_contains_no_gpu_detected() {
        let text = "Status: No GPU detected.";
        assert_eq!(
            verdict_from_text_says_no_gpu(text, false),
            GpupVerdict::Pass
        );
    }

    #[test]
    fn gp003_pass_case_insensitive() {
        let text = "NO DISCRETE GPU.";
        assert_eq!(
            verdict_from_text_says_no_gpu(text, false),
            GpupVerdict::Pass
        );
    }

    #[test]
    fn gp003_pass_when_gpu_present_no_text_check() {
        // gpu_present=true ⇒ no requirement on text.
        let text = "GPU: NVIDIA RTX 4090\nTotal: 24576 MB";
        assert_eq!(
            verdict_from_text_says_no_gpu(text, true),
            GpupVerdict::Pass
        );
    }

    #[test]
    fn gp003_fail_phantom_gpu_record() {
        // The exact regression: gpu_present=false but text shows
        // sentinel record.
        let text = "GPU: GPU-unknown\nTotal: 0 MB";
        assert_eq!(
            verdict_from_text_says_no_gpu(text, false),
            GpupVerdict::Fail
        );
    }

    #[test]
    fn gp003_fail_empty_text_when_no_gpu() {
        let text = "";
        assert_eq!(
            verdict_from_text_says_no_gpu(text, false),
            GpupVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 5: GPU-PRESENCE-004 — real GPU ⇒ true.
    // -------------------------------------------------------------------------
    #[test]
    fn gp004_pass_real_gpu_correctly_true() {
        assert_eq!(
            verdict_from_gpu_present_implies_true("GPU-abcd1234", 24576, true),
            GpupVerdict::Pass
        );
    }

    #[test]
    fn gp004_pass_no_gpu_irrelevant() {
        // Sentinel ⇒ not in real-GPU branch; Pass regardless of bool.
        assert_eq!(
            verdict_from_gpu_present_implies_true("GPU-unknown", 0, false),
            GpupVerdict::Pass
        );
    }

    #[test]
    fn gp004_pass_zero_mb_irrelevant() {
        assert_eq!(
            verdict_from_gpu_present_implies_true("GPU-abcd", 0, false),
            GpupVerdict::Pass
        );
    }

    #[test]
    fn gp004_fail_real_gpu_predicate_inverted() {
        // Real GPU but gpu_present=false: predicate inverted.
        assert_eq!(
            verdict_from_gpu_present_implies_true("GPU-abcd1234", 24576, false),
            GpupVerdict::Fail
        );
    }

    #[test]
    fn gp004_fail_h100_predicate_inverted() {
        assert_eq!(
            verdict_from_gpu_present_implies_true("GPU-h100-uuid", 81920, false),
            GpupVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 6: Realistic — full bug regression scenarios.
    // -------------------------------------------------------------------------
    #[test]
    fn realistic_pre_fix_intel_host_caught() {
        // Pre-fix on intel CPU-only host: apr gpu --json emits
        //   {"gpu_uuid": "GPU-unknown", "total_mb": 0}
        // with NO gpu_present field.
        let fields = vec!["gpu_uuid", "total_mb", "free_mb", "used_mb"];
        assert_eq!(
            verdict_from_json_has_gpu_present_field(&fields),
            GpupVerdict::Fail
        );
    }

    #[test]
    fn realistic_post_fix_intel_host() {
        // Post-fix: gpu_present field present and false on intel.
        let fields = vec!["gpu_uuid", "total_mb", "gpu_present"];
        assert_eq!(
            verdict_from_json_has_gpu_present_field(&fields),
            GpupVerdict::Pass
        );
        assert_eq!(
            verdict_from_no_gpu_implies_false("GPU-unknown", 0, false),
            GpupVerdict::Pass
        );
        let text = "No discrete GPU detected. Falling back to CPU.";
        assert_eq!(
            verdict_from_text_says_no_gpu(text, false),
            GpupVerdict::Pass
        );
    }

    #[test]
    fn realistic_post_fix_lambda_vector() {
        // Post-fix on lambda-vector RTX 4090: real GPU correctly
        // reports gpu_present=true.
        let fields = vec!["gpu_uuid", "total_mb", "gpu_present"];
        assert_eq!(
            verdict_from_json_has_gpu_present_field(&fields),
            GpupVerdict::Pass
        );
        assert_eq!(
            verdict_from_no_gpu_implies_false("GPU-rtx-4090-uuid", 24576, true),
            GpupVerdict::Pass
        );
        assert_eq!(
            verdict_from_gpu_present_implies_true("GPU-rtx-4090-uuid", 24576, true),
            GpupVerdict::Pass
        );
    }

    #[test]
    fn realistic_phantom_gpu_unknown_record_caught() {
        // GPU-PRESENCE-003 if_fails: "Text output still shows phantom
        // GPU-unknown/0 MB record".
        let phantom_text = "GPU: GPU-unknown\nTotal: 0 MB\nFree: 0 MB";
        assert_eq!(
            verdict_from_text_says_no_gpu(phantom_text, false),
            GpupVerdict::Fail
        );
    }

    #[test]
    fn realistic_predicate_inversion_caught() {
        // GPU-PRESENCE-004 if_fails: "Real GPU not recognized —
        // predicate inverted".
        assert_eq!(
            verdict_from_gpu_present_implies_true("GPU-real", 16384, false),
            GpupVerdict::Fail
        );
    }

    #[test]
    fn realistic_full_disambiguation_pipeline() {
        // Run all 4 gates simultaneously on a synthetic post-fix
        // intel CPU-only invocation:
        let fields = vec!["gpu_uuid", "total_mb", "gpu_present"];
        let uuid = "GPU-unknown";
        let total_mb: u64 = 0;
        let gpu_present = false;
        let text = "Status: No discrete GPU detected.";

        assert_eq!(
            verdict_from_json_has_gpu_present_field(&fields),
            GpupVerdict::Pass
        );
        assert_eq!(
            verdict_from_no_gpu_implies_false(uuid, total_mb, gpu_present),
            GpupVerdict::Pass
        );
        assert_eq!(
            verdict_from_text_says_no_gpu(text, gpu_present),
            GpupVerdict::Pass
        );
        assert_eq!(
            verdict_from_gpu_present_implies_true(uuid, total_mb, gpu_present),
            GpupVerdict::Pass
        );
    }
}
