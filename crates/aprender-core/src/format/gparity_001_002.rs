// `apr-gpu-parity-consistency-v1` algorithm-level PARTIAL discharge for the
// 2 FALSIFY-GPARITY gates (parity scope header, ptx-map scope header).
//
// Contract: `contracts/apr-gpu-parity-consistency-v1.yaml`.
//
// Both gates assert that `apr parity --help` and `apr ptx-map --help` outputs
// describe their *scope* clearly (output/logit comparison vs kernel dispatch)
// so users do not confuse seemingly contradictory results from the two
// commands. Each gate's mechanical test is a `grep -qi` for one of a small
// set of scope keywords. Pinning these as pure verdicts prevents drift if the
// help text is ever rewritten.

/// Keywords that satisfy FALSIFY-GPARITY-001 (parity scope description).
/// Per the contract test: `grep -qi "output\|logit\|inference"`.
pub const AC_GPARITY_001_KEYWORDS: [&str; 3] = ["output", "logit", "inference"];

/// Keywords that satisfy FALSIFY-GPARITY-002 (ptx-map scope description).
/// Per the contract test: `grep -qi "kernel\|dispatch\|launch"`.
pub const AC_GPARITY_002_KEYWORDS: [&str; 3] = ["kernel", "dispatch", "launch"];

// =============================================================================
// FALSIFY-GPARITY-001 — parity scope header present
// =============================================================================

/// Verdict for FALSIFY-GPARITY-001.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParityScopeVerdict {
    /// `apr parity --help` text contains at least one of {output, logit,
    /// inference} (case-insensitive).
    Pass,
    /// None of the scope keywords found.
    Fail,
}

/// Pure verdict for FALSIFY-GPARITY-001.
#[must_use]
pub fn verdict_from_parity_help(help_text: &str) -> ParityScopeVerdict {
    let lower = help_text.to_lowercase();
    for kw in AC_GPARITY_001_KEYWORDS {
        if lower.contains(kw) {
            return ParityScopeVerdict::Pass;
        }
    }
    ParityScopeVerdict::Fail
}

// =============================================================================
// FALSIFY-GPARITY-002 — ptx-map scope header present
// =============================================================================

/// Verdict for FALSIFY-GPARITY-002.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PtxMapScopeVerdict {
    /// `apr ptx-map --help` text contains at least one of {kernel, dispatch,
    /// launch} (case-insensitive).
    Pass,
    /// None of the scope keywords found.
    Fail,
}

/// Pure verdict for FALSIFY-GPARITY-002.
#[must_use]
pub fn verdict_from_ptx_map_help(help_text: &str) -> PtxMapScopeVerdict {
    let lower = help_text.to_lowercase();
    for kw in AC_GPARITY_002_KEYWORDS {
        if lower.contains(kw) {
            return PtxMapScopeVerdict::Pass;
        }
    }
    PtxMapScopeVerdict::Fail
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Provenance pins.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_001_keywords_count_3() {
        assert_eq!(AC_GPARITY_001_KEYWORDS.len(), 3);
    }

    #[test]
    fn provenance_002_keywords_count_3() {
        assert_eq!(AC_GPARITY_002_KEYWORDS.len(), 3);
    }

    #[test]
    fn provenance_keyword_sets_disjoint() {
        // Each gate's keywords must NOT overlap, otherwise a single help
        // text could falsely satisfy both.
        for k1 in AC_GPARITY_001_KEYWORDS {
            assert!(
                !AC_GPARITY_002_KEYWORDS.contains(&k1),
                "keyword sets must be disjoint, found {k1:?} in both"
            );
        }
    }

    // -------------------------------------------------------------------------
    // Section 2: GPARITY-001 pass band.
    // -------------------------------------------------------------------------
    #[test]
    fn p001_pass_explicit_contract_header() {
        let help = "GPU/CPU Output Parity: compares inference OUTPUT (logits/tokens)";
        assert_eq!(verdict_from_parity_help(help), ParityScopeVerdict::Pass);
    }

    #[test]
    fn p001_pass_only_output() {
        assert_eq!(verdict_from_parity_help("Compares output."), ParityScopeVerdict::Pass);
    }

    #[test]
    fn p001_pass_only_logit() {
        assert_eq!(verdict_from_parity_help("Logit comparison."), ParityScopeVerdict::Pass);
    }

    #[test]
    fn p001_pass_only_inference() {
        assert_eq!(verdict_from_parity_help("Inference parity check."), ParityScopeVerdict::Pass);
    }

    #[test]
    fn p001_pass_uppercase_keyword() {
        assert_eq!(verdict_from_parity_help("OUTPUT comparison."), ParityScopeVerdict::Pass);
    }

    #[test]
    fn p001_pass_mixed_case() {
        assert_eq!(verdict_from_parity_help("Logit-Level diff."), ParityScopeVerdict::Pass);
    }

    // -------------------------------------------------------------------------
    // Section 3: GPARITY-001 fail band.
    // -------------------------------------------------------------------------
    #[test]
    fn p001_fail_empty_help() {
        assert_eq!(verdict_from_parity_help(""), ParityScopeVerdict::Fail);
    }

    #[test]
    fn p001_fail_no_scope_words() {
        let help = "Usage: apr parity [OPTIONS] <MODEL>\n\nOptions:\n  --json   Emit JSON";
        assert_eq!(verdict_from_parity_help(help), ParityScopeVerdict::Fail);
    }

    #[test]
    fn p001_fail_only_ptx_keywords() {
        // ptx-map keywords must NOT pass the parity gate.
        let help = "Verifies kernel dispatch and launch params.";
        assert_eq!(verdict_from_parity_help(help), ParityScopeVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 4: GPARITY-002 pass band.
    // -------------------------------------------------------------------------
    #[test]
    fn p002_pass_explicit_contract_header() {
        let help = "PTX Kernel Dispatch Map: verifies kernel LAUNCH configuration";
        assert_eq!(verdict_from_ptx_map_help(help), PtxMapScopeVerdict::Pass);
    }

    #[test]
    fn p002_pass_only_kernel() {
        assert_eq!(verdict_from_ptx_map_help("Kernel verification."), PtxMapScopeVerdict::Pass);
    }

    #[test]
    fn p002_pass_only_dispatch() {
        assert_eq!(verdict_from_ptx_map_help("Dispatch summary."), PtxMapScopeVerdict::Pass);
    }

    #[test]
    fn p002_pass_only_launch() {
        assert_eq!(verdict_from_ptx_map_help("Launch params printout."), PtxMapScopeVerdict::Pass);
    }

    #[test]
    fn p002_pass_uppercase() {
        assert_eq!(verdict_from_ptx_map_help("KERNEL map"), PtxMapScopeVerdict::Pass);
    }

    // -------------------------------------------------------------------------
    // Section 5: GPARITY-002 fail band.
    // -------------------------------------------------------------------------
    #[test]
    fn p002_fail_empty() {
        assert_eq!(verdict_from_ptx_map_help(""), PtxMapScopeVerdict::Fail);
    }

    #[test]
    fn p002_fail_no_scope_words() {
        let help = "Usage: apr ptx-map [OPTIONS]";
        assert_eq!(verdict_from_ptx_map_help(help), PtxMapScopeVerdict::Fail);
    }

    #[test]
    fn p002_fail_only_parity_keywords() {
        // parity keywords must NOT pass the ptx-map gate.
        let help = "Compares output logits across inference passes.";
        assert_eq!(verdict_from_ptx_map_help(help), PtxMapScopeVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 6: Cross-gate disjointness sanity.
    // -------------------------------------------------------------------------
    #[test]
    fn cross_gate_disjointness_via_text_pairs() {
        // Confirm the verdicts are independent: the parity contract header
        // passes 001 and FAILS 002, and vice versa for the ptx-map header.
        let parity_header = "GPU/CPU Output Parity: compares inference OUTPUT (logits/tokens)";
        let ptx_header = "PTX Kernel Dispatch Map: verifies kernel LAUNCH configuration";
        assert_eq!(verdict_from_parity_help(parity_header), ParityScopeVerdict::Pass);
        assert_eq!(verdict_from_ptx_map_help(parity_header), PtxMapScopeVerdict::Fail);
        assert_eq!(verdict_from_parity_help(ptx_header), ParityScopeVerdict::Fail);
        assert_eq!(verdict_from_ptx_map_help(ptx_header), PtxMapScopeVerdict::Pass);
    }

    // -------------------------------------------------------------------------
    // Section 7: Realistic — full help text from each subcommand.
    // -------------------------------------------------------------------------
    #[test]
    fn realistic_full_apr_parity_help() {
        let help = "\
Compare GPU and CPU inference output for parity verification.

GPU/CPU Output Parity: compares inference OUTPUT (logits/tokens) between
the two backends to detect numerical divergence.

Usage: apr parity [OPTIONS] <MODEL>

Options:
  --json    Emit JSON output
  -h, --help   Print help
";
        assert_eq!(verdict_from_parity_help(help), ParityScopeVerdict::Pass);
    }

    #[test]
    fn realistic_full_apr_ptx_map_help() {
        let help = "\
Print the PTX kernel dispatch map for a model.

PTX Kernel Dispatch Map: verifies kernel LAUNCH configuration without
running inference.

Usage: apr ptx-map [OPTIONS] <MODEL>

Options:
  --json    Emit JSON output
  -h, --help   Print help
";
        assert_eq!(verdict_from_ptx_map_help(help), PtxMapScopeVerdict::Pass);
    }

    #[test]
    fn realistic_pre_fix_both_fail() {
        // The exact regression class: help texts that omit the scope header.
        let bad_parity = "Usage: apr parity [OPTIONS] <MODEL>\n\nOptions:\n  --json";
        let bad_ptx = "Usage: apr ptx-map [OPTIONS] <MODEL>\n\nOptions:\n  --json";
        assert_eq!(verdict_from_parity_help(bad_parity), ParityScopeVerdict::Fail);
        assert_eq!(verdict_from_ptx_map_help(bad_ptx), PtxMapScopeVerdict::Fail);
    }
}
