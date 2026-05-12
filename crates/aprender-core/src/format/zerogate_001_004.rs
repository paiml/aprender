// `apr-zero-feature-gate-v1` algorithm-level PARTIAL discharge for the 4
// zero-feature-gate falsifiers (all commands respond to --help, no
// feature-gate errors, GPU fallback graceful, default features complete).
//
// Contract: `contracts/apr-zero-feature-gate-v1.yaml`.
// Refs: APR-MONO Rule 5 (Zero Feature-Gating for Users), Rule 6 (GPU
// Auto-Detection at Runtime).

/// Phrases that indicate a feature-gate error in user-facing output.
pub const AC_ZEROGATE_FORBIDDEN_PHRASES: [&str; 4] = [
    "feature not enabled",
    "requires --features",
    "enable the",
    "compile with --features",
];

/// Required features in apr-cli's default = [...] list per the contract.
pub const AC_ZEROGATE_REQUIRED_DEFAULT_FEATURES: [&str; 2] = ["inference", "training"];

/// Features that MUST NOT be in default (compile-time only).
pub const AC_ZEROGATE_FORBIDDEN_DEFAULT_FEATURES: [&str; 2] = ["cuda", "code"];

/// GPU error keywords that MUST NOT appear after --no-gpu fallback.
pub const AC_ZEROGATE_GPU_ERROR_KEYWORDS: [&str; 4] =
    ["cuda", "gpu not found", "gpu missing", "no gpu"];

// =============================================================================
// FALSIFY-ZEROGATEV-001 — every subcommand responds to --help with exit 0
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HelpResponseVerdict {
    /// Every subcommand exits 0 on `--help`.
    Pass,
    /// At least one subcommand failed `--help`.
    Fail,
}

/// `(subcommand_name, exit_code_on_help)` per subcommand.
#[must_use]
pub fn verdict_from_help_response(help_results: &[(&str, i32)]) -> HelpResponseVerdict {
    if help_results.is_empty() {
        return HelpResponseVerdict::Fail;
    }
    for (_cmd, exit) in help_results {
        if *exit != 0 {
            return HelpResponseVerdict::Fail;
        }
    }
    HelpResponseVerdict::Pass
}

// =============================================================================
// FALSIFY-ZEROGATEV-002 — no feature-gate error phrases in output
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoFeatureGateErrorVerdict {
    /// No `--help` output contains any forbidden feature-gate phrase.
    Pass,
    /// At least one output contained a feature-gate phrase.
    Fail,
}

#[must_use]
pub fn verdict_from_no_feature_gate_error(combined_output: &str) -> NoFeatureGateErrorVerdict {
    let lower = combined_output.to_lowercase();
    for phrase in AC_ZEROGATE_FORBIDDEN_PHRASES {
        if lower.contains(phrase) {
            return NoFeatureGateErrorVerdict::Fail;
        }
    }
    NoFeatureGateErrorVerdict::Pass
}

// =============================================================================
// FALSIFY-ZEROGATEV-003 — GPU fallback graceful (no GPU error after --no-gpu)
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuFallbackVerdict {
    /// `apr run --no-gpu` output contains no GPU/CUDA error keywords.
    /// (May still fail on the input model — that's a separate gate.)
    Pass,
    /// Output mentioned GPU/CUDA errors despite --no-gpu.
    Fail,
}

#[must_use]
pub fn verdict_from_gpu_fallback(combined_output: &str) -> GpuFallbackVerdict {
    let lower = combined_output.to_lowercase();
    for kw in AC_ZEROGATE_GPU_ERROR_KEYWORDS {
        if lower.contains(kw) {
            return GpuFallbackVerdict::Fail;
        }
    }
    GpuFallbackVerdict::Pass
}

// =============================================================================
// FALSIFY-ZEROGATEV-004 — default features complete + cuda/code excluded
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DefaultFeaturesVerdict {
    /// `inference` AND `training` present in default; `cuda` AND `code` absent.
    Pass,
    /// At least one required feature missing OR forbidden feature present.
    Fail,
}

#[must_use]
pub fn verdict_from_default_features(default_features: &[&str]) -> DefaultFeaturesVerdict {
    use std::collections::HashSet;
    let set: HashSet<&&str> = default_features.iter().collect();
    for required in AC_ZEROGATE_REQUIRED_DEFAULT_FEATURES {
        if !set.contains(&required) {
            return DefaultFeaturesVerdict::Fail;
        }
    }
    for forbidden in AC_ZEROGATE_FORBIDDEN_DEFAULT_FEATURES {
        if set.contains(&forbidden) {
            return DefaultFeaturesVerdict::Fail;
        }
    }
    DefaultFeaturesVerdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Provenance pins.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_forbidden_phrases_count_4() {
        assert_eq!(AC_ZEROGATE_FORBIDDEN_PHRASES.len(), 4);
    }

    #[test]
    fn provenance_required_features_count_2() {
        assert_eq!(AC_ZEROGATE_REQUIRED_DEFAULT_FEATURES.len(), 2);
    }

    #[test]
    fn provenance_forbidden_features_count_2() {
        assert_eq!(AC_ZEROGATE_FORBIDDEN_DEFAULT_FEATURES.len(), 2);
    }

    #[test]
    fn provenance_required_forbidden_disjoint() {
        for r in AC_ZEROGATE_REQUIRED_DEFAULT_FEATURES {
            assert!(!AC_ZEROGATE_FORBIDDEN_DEFAULT_FEATURES.contains(&r));
        }
    }

    // -------------------------------------------------------------------------
    // Section 2: ZEROGATEV-001 help response.
    // -------------------------------------------------------------------------
    #[test]
    fn fz001_pass_all_commands_zero() {
        let r = [("run", 0), ("serve", 0), ("chat", 0), ("finetune", 0)];
        assert_eq!(verdict_from_help_response(&r), HelpResponseVerdict::Pass);
    }

    #[test]
    fn fz001_fail_one_command_nonzero() {
        let r = [("run", 0), ("finetune", 1)];
        assert_eq!(verdict_from_help_response(&r), HelpResponseVerdict::Fail);
    }

    #[test]
    fn fz001_fail_panic_exit() {
        let r = [("run", 0), ("profile", 101)];
        assert_eq!(verdict_from_help_response(&r), HelpResponseVerdict::Fail);
    }

    #[test]
    fn fz001_fail_empty() {
        let r: [(&str, i32); 0] = [];
        assert_eq!(verdict_from_help_response(&r), HelpResponseVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 3: ZEROGATEV-002 no feature-gate phrases.
    // -------------------------------------------------------------------------
    #[test]
    fn fz002_pass_clean_output() {
        let out = "Usage: apr run [OPTIONS] <MODEL>\n\nOptions:\n  --json   Emit JSON";
        assert_eq!(
            verdict_from_no_feature_gate_error(out),
            NoFeatureGateErrorVerdict::Pass
        );
    }

    #[test]
    fn fz002_fail_feature_not_enabled() {
        let out = "Error: feature not enabled — recompile with --features inference";
        assert_eq!(
            verdict_from_no_feature_gate_error(out),
            NoFeatureGateErrorVerdict::Fail
        );
    }

    #[test]
    fn fz002_fail_requires_features() {
        let out = "This command requires --features cuda";
        assert_eq!(
            verdict_from_no_feature_gate_error(out),
            NoFeatureGateErrorVerdict::Fail
        );
    }

    #[test]
    fn fz002_fail_compile_with_features() {
        let out = "compile with --features training";
        assert_eq!(
            verdict_from_no_feature_gate_error(out),
            NoFeatureGateErrorVerdict::Fail
        );
    }

    #[test]
    fn fz002_fail_enable_the_feature() {
        let out = "enable the inference feature";
        assert_eq!(
            verdict_from_no_feature_gate_error(out),
            NoFeatureGateErrorVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 4: ZEROGATEV-003 GPU fallback.
    // -------------------------------------------------------------------------
    #[test]
    fn fz003_pass_clean_cpu_fallback() {
        let out = "Running on CPU SIMD backend";
        assert_eq!(verdict_from_gpu_fallback(out), GpuFallbackVerdict::Pass);
    }

    #[test]
    fn fz003_pass_invalid_model_error() {
        // Failure on the model path is fine; only GPU-related errors fail.
        let out = "Error: invalid model file at /dev/null";
        assert_eq!(verdict_from_gpu_fallback(out), GpuFallbackVerdict::Pass);
    }

    #[test]
    fn fz003_fail_cuda_error() {
        let out = "Error: CUDA driver not available";
        assert_eq!(verdict_from_gpu_fallback(out), GpuFallbackVerdict::Fail);
    }

    #[test]
    fn fz003_fail_gpu_not_found() {
        let out = "Error: GPU not found";
        assert_eq!(verdict_from_gpu_fallback(out), GpuFallbackVerdict::Fail);
    }

    #[test]
    fn fz003_fail_no_gpu_error() {
        let out = "Error: No GPU detected, refusing to run";
        assert_eq!(verdict_from_gpu_fallback(out), GpuFallbackVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 5: ZEROGATEV-004 default features.
    // -------------------------------------------------------------------------
    #[test]
    fn fz004_pass_canonical_defaults() {
        let f = [
            "hf-hub",
            "safetensors-compare",
            "inference",
            "training",
            "visualization",
            "zram",
        ];
        assert_eq!(verdict_from_default_features(&f), DefaultFeaturesVerdict::Pass);
    }

    #[test]
    fn fz004_pass_minimal_required() {
        let f = ["inference", "training"];
        assert_eq!(verdict_from_default_features(&f), DefaultFeaturesVerdict::Pass);
    }

    #[test]
    fn fz004_fail_missing_inference() {
        let f = ["training", "visualization"];
        assert_eq!(verdict_from_default_features(&f), DefaultFeaturesVerdict::Fail);
    }

    #[test]
    fn fz004_fail_missing_training() {
        let f = ["inference", "visualization"];
        assert_eq!(verdict_from_default_features(&f), DefaultFeaturesVerdict::Fail);
    }

    #[test]
    fn fz004_fail_cuda_in_default() {
        // Compile-time-only feature leaked into default.
        let f = ["inference", "training", "cuda"];
        assert_eq!(verdict_from_default_features(&f), DefaultFeaturesVerdict::Fail);
    }

    #[test]
    fn fz004_fail_code_in_default() {
        // External-dep feature leaked into default.
        let f = ["inference", "training", "code"];
        assert_eq!(verdict_from_default_features(&f), DefaultFeaturesVerdict::Fail);
    }

    #[test]
    fn fz004_fail_empty_default() {
        let f: [&str; 0] = [];
        assert_eq!(verdict_from_default_features(&f), DefaultFeaturesVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 6: Realistic — full healthy install passes all 4.
    // -------------------------------------------------------------------------
    #[test]
    fn realistic_healthy_install_passes_all_4() {
        let cmds = [
            ("run", 0), ("serve", 0), ("chat", 0), ("finetune", 0),
            ("train", 0), ("distill", 0), ("profile", 0), ("trace", 0),
        ];
        assert_eq!(verdict_from_help_response(&cmds), HelpResponseVerdict::Pass);

        let clean_help = "Usage: apr run [OPTIONS] <MODEL>\n\nOptions:\n  --no-gpu";
        assert_eq!(
            verdict_from_no_feature_gate_error(clean_help),
            NoFeatureGateErrorVerdict::Pass
        );

        let cpu_run = "Backend: CPU SIMD (AVX2)\nGenerating...";
        assert_eq!(verdict_from_gpu_fallback(cpu_run), GpuFallbackVerdict::Pass);

        let canonical = [
            "hf-hub", "safetensors-compare", "inference", "training",
            "visualization", "zram",
        ];
        assert_eq!(verdict_from_default_features(&canonical), DefaultFeaturesVerdict::Pass);
    }

    #[test]
    fn realistic_pre_fix_all_4_failures() {
        // 001: finetune missing without --features training.
        let bad = [("finetune", 1)];
        assert_eq!(verdict_from_help_response(&bad), HelpResponseVerdict::Fail);

        // 002: feature-gate error reached user.
        let bad_help = "Error: feature not enabled, requires --features inference";
        assert_eq!(
            verdict_from_no_feature_gate_error(bad_help),
            NoFeatureGateErrorVerdict::Fail
        );

        // 003: GPU error despite --no-gpu.
        let bad_run = "Error: CUDA driver not loaded";
        assert_eq!(verdict_from_gpu_fallback(bad_run), GpuFallbackVerdict::Fail);

        // 004: cuda leaked into default.
        let bad_features = ["inference", "training", "cuda"];
        assert_eq!(
            verdict_from_default_features(&bad_features),
            DefaultFeaturesVerdict::Fail
        );
    }
}
