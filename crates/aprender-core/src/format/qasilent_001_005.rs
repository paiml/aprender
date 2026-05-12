// `apr-qa-silent-fallback-v1` algorithm-level PARTIAL discharge for the
// 5 silent-fallback-injection falsifiers (truncated detection, zero-tps
// rejection, unknown-arch handling, missing-tokenizer detection, loud
// failure on bad input).
//
// Contract: `contracts/apr-qa-silent-fallback-v1.yaml`.
// Refs: GH-339 (chat template silent raw-prompt fallback), GH-336
// (benchmark silently swallowed errors), GH-337 (chat server byte-decode),
// GH-338 (probar silent metadata corruption), GH-439 (silent _ => default
// match arms at format boundaries).

/// Keywords that constitute a "loud" failure indication when the command
/// chooses to warn instead of error (legitimate fallback path per the
/// `loud_failure_on_bad_input` formula).
pub const AC_QASILENT_LOUD_KEYWORDS: [&str; 3] = ["WARN", "SKIP", "unsupported"];

/// Keywords that satisfy the truncation detection error message.
pub const AC_QASILENT_TRUNCATION_KEYWORDS: [&str; 3] = ["truncat", "corrupt", "incomplete"];

/// Keywords that satisfy the missing-tokenizer detection (either
/// declares failure OR notes legitimate fallback path).
pub const AC_QASILENT_TOKENIZER_KEYWORDS: [&str; 3] = ["tokenizer", "embedded", "GGUF"];

/// Keywords that satisfy the unknown-arch error message.
pub const AC_QASILENT_UNKNOWN_ARCH_KEYWORDS: [&str; 2] = ["unsupported", "unknown"];

// =============================================================================
// F-SILENT-001 — truncated file detection
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TruncationDetectionVerdict {
    /// `apr validate` exits non-zero AND stderr mentions truncation/corruption.
    Pass,
    /// Silent acceptance, OR non-zero exit without explanatory stderr.
    Fail,
}

#[must_use]
pub fn verdict_from_truncation_detection(exit_code: i32, stderr: &str) -> TruncationDetectionVerdict {
    if exit_code == 0 {
        return TruncationDetectionVerdict::Fail;
    }
    let lower = stderr.to_lowercase();
    for kw in AC_QASILENT_TRUNCATION_KEYWORDS {
        if lower.contains(kw) {
            return TruncationDetectionVerdict::Pass;
        }
    }
    TruncationDetectionVerdict::Fail
}

// =============================================================================
// F-SILENT-002 — zero throughput rejection
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZeroThroughputVerdict {
    /// `apr bench` reporting tok/s == 0.0 ⇒ exit_code != 0.
    Pass,
    /// Reported 0.0 tok/s but exited 0 — the regression class.
    Fail,
}

#[must_use]
pub fn verdict_from_zero_throughput(reported_tps: f64, exit_code: i32) -> ZeroThroughputVerdict {
    if reported_tps > 0.0 {
        // Non-zero throughput is fine regardless of exit code; this gate
        // only catches the zero-as-success regression class.
        return ZeroThroughputVerdict::Pass;
    }
    // tps == 0.0 (or negative — also nonsense): require non-zero exit.
    if exit_code == 0 {
        ZeroThroughputVerdict::Fail
    } else {
        ZeroThroughputVerdict::Pass
    }
}

// =============================================================================
// F-SILENT-003 — unknown architecture handling
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnknownArchVerdict {
    /// Unknown architecture identifier ⇒ non-zero exit AND stderr
    /// mentions "unsupported" or "unknown".
    Pass,
    /// Silent fallback to llama default — the regression class.
    Fail,
}

#[must_use]
pub fn verdict_from_unknown_arch(exit_code: i32, stderr: &str) -> UnknownArchVerdict {
    if exit_code == 0 {
        return UnknownArchVerdict::Fail;
    }
    let lower = stderr.to_lowercase();
    for kw in AC_QASILENT_UNKNOWN_ARCH_KEYWORDS {
        if lower.contains(kw) {
            return UnknownArchVerdict::Pass;
        }
    }
    UnknownArchVerdict::Fail
}

// =============================================================================
// F-SILENT-004 — missing tokenizer detection
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MissingTokenizerVerdict {
    /// Either non-zero exit with "tokenizer" in stderr, OR uses embedded
    /// GGUF tokenizer (legitimate fallback). Never silent-garble.
    Pass,
    /// Silent garble — output emitted without explanation.
    Fail,
}

#[must_use]
pub fn verdict_from_missing_tokenizer(stderr_or_log: &str) -> MissingTokenizerVerdict {
    // The contract test is `grep -qE 'tokenizer|embedded|GGUF'`. Any of
    // these keywords appearing means the tool announced its tokenizer
    // path (either error or legitimate fallback). Absence ⇒ silent.
    for kw in AC_QASILENT_TOKENIZER_KEYWORDS {
        if stderr_or_log.contains(kw) {
            return MissingTokenizerVerdict::Pass;
        }
    }
    MissingTokenizerVerdict::Fail
}

// =============================================================================
// F-SILENT-005 — loud failure on bad input
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoudFailureVerdict {
    /// Either: non-zero exit with non-empty stderr, OR stdout contains
    /// WARN / SKIP / unsupported.
    Pass,
    /// Silent degradation — exit 0, no stderr, no warning keywords.
    Fail,
}

#[must_use]
pub fn verdict_from_loud_failure(exit_code: i32, stderr: &str, stdout: &str) -> LoudFailureVerdict {
    // Path 1: explicit error.
    if exit_code != 0 && !stderr.is_empty() {
        return LoudFailureVerdict::Pass;
    }
    // Path 2: warning emitted on stdout.
    for kw in AC_QASILENT_LOUD_KEYWORDS {
        if stdout.contains(kw) {
            return LoudFailureVerdict::Pass;
        }
    }
    LoudFailureVerdict::Fail
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Provenance pins.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_loud_keywords_count_3() {
        assert_eq!(AC_QASILENT_LOUD_KEYWORDS.len(), 3);
    }

    #[test]
    fn provenance_truncation_keywords_count_3() {
        assert_eq!(AC_QASILENT_TRUNCATION_KEYWORDS.len(), 3);
    }

    #[test]
    fn provenance_tokenizer_keywords_count_3() {
        assert_eq!(AC_QASILENT_TOKENIZER_KEYWORDS.len(), 3);
    }

    #[test]
    fn provenance_unknown_arch_keywords_count_2() {
        assert_eq!(AC_QASILENT_UNKNOWN_ARCH_KEYWORDS.len(), 2);
    }

    // -------------------------------------------------------------------------
    // Section 2: F-SILENT-001 truncation detection.
    // -------------------------------------------------------------------------
    #[test]
    fn fs001_pass_truncated_keyword() {
        let v = verdict_from_truncation_detection(1, "Error: file truncated at offset 1024");
        assert_eq!(v, TruncationDetectionVerdict::Pass);
    }

    #[test]
    fn fs001_pass_corrupt_keyword() {
        let v = verdict_from_truncation_detection(2, "Error: corrupt header magic");
        assert_eq!(v, TruncationDetectionVerdict::Pass);
    }

    #[test]
    fn fs001_pass_incomplete_keyword() {
        let v = verdict_from_truncation_detection(1, "Error: model file is incomplete");
        assert_eq!(v, TruncationDetectionVerdict::Pass);
    }

    #[test]
    fn fs001_fail_silent_zero_exit() {
        let v = verdict_from_truncation_detection(0, "model truncated");
        assert_eq!(v, TruncationDetectionVerdict::Fail);
    }

    #[test]
    fn fs001_fail_nonzero_no_keyword() {
        let v = verdict_from_truncation_detection(1, "Generic I/O error");
        assert_eq!(v, TruncationDetectionVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 3: F-SILENT-002 zero throughput.
    // -------------------------------------------------------------------------
    #[test]
    fn fs002_pass_positive_tps() {
        // Non-zero throughput passes regardless of exit code.
        assert_eq!(verdict_from_zero_throughput(127.5, 0), ZeroThroughputVerdict::Pass);
    }

    #[test]
    fn fs002_pass_zero_tps_nonzero_exit() {
        assert_eq!(verdict_from_zero_throughput(0.0, 1), ZeroThroughputVerdict::Pass);
    }

    #[test]
    fn fs002_fail_zero_tps_zero_exit() {
        // The exact regression class: 0.0 tok/s reported as success.
        assert_eq!(verdict_from_zero_throughput(0.0, 0), ZeroThroughputVerdict::Fail);
    }

    #[test]
    fn fs002_fail_negative_tps_zero_exit() {
        // Negative tps is nonsense; treated like zero.
        assert_eq!(verdict_from_zero_throughput(-1.0, 0), ZeroThroughputVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 4: F-SILENT-003 unknown architecture.
    // -------------------------------------------------------------------------
    #[test]
    fn fs003_pass_unsupported_keyword() {
        let v = verdict_from_unknown_arch(1, "Error: unsupported architecture totally_unknown_arch_v99");
        assert_eq!(v, UnknownArchVerdict::Pass);
    }

    #[test]
    fn fs003_pass_unknown_keyword() {
        let v = verdict_from_unknown_arch(1, "Error: unknown model arch");
        assert_eq!(v, UnknownArchVerdict::Pass);
    }

    #[test]
    fn fs003_fail_silent_zero_exit() {
        // The regression: silently fell through to llama.
        let v = verdict_from_unknown_arch(0, "");
        assert_eq!(v, UnknownArchVerdict::Fail);
    }

    #[test]
    fn fs003_fail_nonzero_no_keyword() {
        let v = verdict_from_unknown_arch(1, "Some other error");
        assert_eq!(v, UnknownArchVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 5: F-SILENT-004 missing tokenizer.
    // -------------------------------------------------------------------------
    #[test]
    fn fs004_pass_tokenizer_keyword() {
        let v = verdict_from_missing_tokenizer("Error: tokenizer.json not found");
        assert_eq!(v, MissingTokenizerVerdict::Pass);
    }

    #[test]
    fn fs004_pass_embedded_fallback() {
        let v = verdict_from_missing_tokenizer("Notice: using embedded GGUF tokenizer");
        assert_eq!(v, MissingTokenizerVerdict::Pass);
    }

    #[test]
    fn fs004_pass_gguf_keyword() {
        let v = verdict_from_missing_tokenizer("Falling back to GGUF-internal vocabulary");
        assert_eq!(v, MissingTokenizerVerdict::Pass);
    }

    #[test]
    fn fs004_fail_silent_garble() {
        // Output emitted without any tokenizer/embedded/GGUF announcement.
        let v = verdict_from_missing_tokenizer("Output: ?�?�?");
        assert_eq!(v, MissingTokenizerVerdict::Fail);
    }

    #[test]
    fn fs004_fail_empty_log() {
        let v = verdict_from_missing_tokenizer("");
        assert_eq!(v, MissingTokenizerVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 6: F-SILENT-005 loud failure.
    // -------------------------------------------------------------------------
    #[test]
    fn fs005_pass_explicit_error() {
        let v = verdict_from_loud_failure(1, "Error: bad metadata", "");
        assert_eq!(v, LoudFailureVerdict::Pass);
    }

    #[test]
    fn fs005_pass_warn_on_stdout() {
        let v = verdict_from_loud_failure(0, "", "WARN: degraded mode");
        assert_eq!(v, LoudFailureVerdict::Pass);
    }

    #[test]
    fn fs005_pass_skip_on_stdout() {
        let v = verdict_from_loud_failure(0, "", "SKIP: unsupported feature");
        assert_eq!(v, LoudFailureVerdict::Pass);
    }

    #[test]
    fn fs005_pass_unsupported_on_stdout() {
        let v = verdict_from_loud_failure(0, "", "Notice: unsupported quant — using f16");
        assert_eq!(v, LoudFailureVerdict::Pass);
    }

    #[test]
    fn fs005_fail_silent_success() {
        // The exact regression: bad input but exit 0, empty stderr, no warning.
        let v = verdict_from_loud_failure(0, "", "Output: garbage");
        assert_eq!(v, LoudFailureVerdict::Fail);
    }

    #[test]
    fn fs005_fail_nonzero_exit_no_stderr() {
        // Exit 1 alone isn't enough — need either stderr OR stdout warning.
        let v = verdict_from_loud_failure(1, "", "");
        assert_eq!(v, LoudFailureVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 7: Realistic — full healthy bad-input run + pre-fix.
    // -------------------------------------------------------------------------
    #[test]
    fn realistic_loud_run_passes_all_5() {
        // Truncated file → non-zero exit + corrupt keyword.
        assert_eq!(
            verdict_from_truncation_detection(1, "Error: file is corrupt"),
            TruncationDetectionVerdict::Pass
        );
        // Non-zero throughput.
        assert_eq!(verdict_from_zero_throughput(150.0, 0), ZeroThroughputVerdict::Pass);
        // Unknown arch → non-zero exit + unsupported.
        assert_eq!(
            verdict_from_unknown_arch(1, "Error: unsupported architecture"),
            UnknownArchVerdict::Pass
        );
        // Missing tokenizer → embedded fallback announced.
        assert_eq!(
            verdict_from_missing_tokenizer("Notice: using embedded tokenizer"),
            MissingTokenizerVerdict::Pass
        );
        // Bad input → loud error.
        assert_eq!(
            verdict_from_loud_failure(1, "Error: bad input", ""),
            LoudFailureVerdict::Pass
        );
    }

    #[test]
    fn realistic_pre_fix_all_5_failures() {
        // 001: silent acceptance of truncated file.
        assert_eq!(
            verdict_from_truncation_detection(0, ""),
            TruncationDetectionVerdict::Fail
        );
        // 002: GH-336 — 0 tok/s reported as success.
        assert_eq!(verdict_from_zero_throughput(0.0, 0), ZeroThroughputVerdict::Fail);
        // 003: GH-439 — unknown arch silently maps to llama.
        assert_eq!(
            verdict_from_unknown_arch(0, "tokens generated successfully"),
            UnknownArchVerdict::Fail
        );
        // 004: GH-337 — chat server degrades to byte-level decode silently.
        assert_eq!(
            verdict_from_missing_tokenizer("Output: ?\u{fffd}?\u{fffd}?"),
            MissingTokenizerVerdict::Fail
        );
        // 005: GH-339 / GH-338 — silent fallback to raw prompt / accepted corruption.
        assert_eq!(
            verdict_from_loud_failure(0, "", "Output: degraded silently"),
            LoudFailureVerdict::Fail
        );
    }
}
