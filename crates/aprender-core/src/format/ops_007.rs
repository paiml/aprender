// SHIP-TWO-001 — `apr-cli-operations-v1` algorithm-level PARTIAL
// discharge for FALSIFY-OPS-007.
//
// Contract: `contracts/apr-cli-operations-v1.yaml`.
// Spec: `docs/specifications/aprender-train/ship-two-models-spec.md`.
//
// ## What FALSIFY-OPS-007 says
//
//   rule: Token count bounded by input length
//   prediction: No input produces more than 4x tokens (BPE worst case)
//   test: Encode adversarial strings (single-byte tokens), verify
//         count <= 4 * len
//   if_fails: Token explosion — unbounded memory usage
//
// ## What this file proves NOW (`PARTIAL_ALGORITHM_LEVEL`)
//
// Decision rule: given (`input_byte_len`, `token_count`), Pass iff:
//
//   input_byte_len > 0 AND
//   token_count <= AC_OPS_007_MAX_RATIO (4) * input_byte_len
//
// Linear-bound verdict via `checked_mul` to prevent overflow at
// adversarial input sizes near `u64::MAX`. Inclusive `<=` matches
// the contract's `<= 4 * len` test wording.

/// Maximum tokens-per-input-byte ratio for BPE worst case.
///
/// Per contract: BPE byte-level fallback emits at most ~4 tokens
/// per input byte in the worst case (UTF-8 4-byte sequences split
/// into 4 byte-fallback tokens). 4× cap is the published BPE
/// worst-case bound.
///
/// Drift to 8× would mask token-explosion bugs (unbounded merge
/// expansion); drift to 2× would over-tighten and reject
/// pathological-but-valid CJK / emoji inputs.
pub const AC_OPS_007_MAX_RATIO: u64 = 4;

/// Binary verdict for `FALSIFY-OPS-007`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ops007Verdict {
    /// `input_byte_len > 0` AND `token_count <= 4 * input_byte_len`.
    Pass,
    /// One or more of:
    /// - `input_byte_len == 0` (caller error — empty input).
    /// - `token_count > 4 * input_byte_len` (token explosion).
    /// - Multiplication overflow on `4 * input_byte_len`.
    Fail,
}

/// Pure verdict function for `FALSIFY-OPS-007`.
///
/// Inputs:
/// - `input_byte_len`: length of the input string in bytes (UTF-8).
/// - `token_count`: number of tokens produced by tokenizing the
///   input.
///
/// Pass iff:
/// 1. `input_byte_len > 0`,
/// 2. `token_count <= input_byte_len.checked_mul(4)` (overflow → Fail).
///
/// Otherwise `Fail`.
///
/// # Examples
///
/// 100-byte input → 50 tokens (well within 4× cap) — `Pass`:
/// ```
/// use aprender::format::ops_007::{
///     verdict_from_token_count_bound, Ops007Verdict,
/// };
/// let v = verdict_from_token_count_bound(100, 50);
/// assert_eq!(v, Ops007Verdict::Pass);
/// ```
///
/// 100-byte input → 401 tokens (exceeds 4× cap = 400) — `Fail`:
/// ```
/// use aprender::format::ops_007::{
///     verdict_from_token_count_bound, Ops007Verdict,
/// };
/// let v = verdict_from_token_count_bound(100, 401);
/// assert_eq!(v, Ops007Verdict::Fail);
/// ```
#[must_use]
pub fn verdict_from_token_count_bound(
    input_byte_len: u64,
    token_count: u64,
) -> Ops007Verdict {
    if input_byte_len == 0 {
        return Ops007Verdict::Fail;
    }
    let max_tokens = match input_byte_len.checked_mul(AC_OPS_007_MAX_RATIO) {
        Some(v) => v,
        None => return Ops007Verdict::Fail,
    };
    if token_count <= max_tokens {
        Ops007Verdict::Pass
    } else {
        Ops007Verdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Provenance pin — 4× ratio.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_max_ratio_is_4() {
        assert_eq!(AC_OPS_007_MAX_RATIO, 4);
    }

    // -------------------------------------------------------------------------
    // Section 2: Pass band — typical compression ratios.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_typical_compression_ratio() {
        // BPE typically compresses ~3:1 on English text — 100 bytes → ~30 tokens.
        let v = verdict_from_token_count_bound(100, 30);
        assert_eq!(v, Ops007Verdict::Pass);
    }

    #[test]
    fn pass_at_exact_4x_cap() {
        // Worst case: every byte → 4 tokens.
        let v = verdict_from_token_count_bound(100, 400);
        assert_eq!(v, Ops007Verdict::Pass);
    }

    #[test]
    fn pass_one_byte_one_token() {
        let v = verdict_from_token_count_bound(1, 1);
        assert_eq!(v, Ops007Verdict::Pass);
    }

    #[test]
    fn pass_realistic_apr_run_input() {
        // 50-byte prompt → 12 tokens (typical English compression).
        let v = verdict_from_token_count_bound(50, 12);
        assert_eq!(v, Ops007Verdict::Pass);
    }

    #[test]
    fn pass_huge_clean_input() {
        let v = verdict_from_token_count_bound(1_000_000, 100_000);
        assert_eq!(v, Ops007Verdict::Pass);
    }

    // -------------------------------------------------------------------------
    // Section 3: Fail band — token explosion.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_just_above_4x_cap() {
        let v = verdict_from_token_count_bound(100, 401);
        assert_eq!(
            v,
            Ops007Verdict::Fail,
            "+1 token over cap must Fail"
        );
    }

    #[test]
    fn fail_5x_token_explosion() {
        let v = verdict_from_token_count_bound(100, 500);
        assert_eq!(v, Ops007Verdict::Fail);
    }

    #[test]
    fn fail_10x_token_explosion() {
        let v = verdict_from_token_count_bound(100, 1_000);
        assert_eq!(v, Ops007Verdict::Fail);
    }

    #[test]
    fn fail_unbounded_memory_attack() {
        // Adversarial input: 10-byte string produces 100k tokens.
        let v = verdict_from_token_count_bound(10, 100_000);
        assert_eq!(
            v,
            Ops007Verdict::Fail,
            "DoS-class token explosion must Fail"
        );
    }

    // -------------------------------------------------------------------------
    // Section 4: Fail band — empty input (caller error).
    // -------------------------------------------------------------------------
    #[test]
    fn fail_zero_input_zero_tokens() {
        // Empty input is degenerate — refuse.
        let v = verdict_from_token_count_bound(0, 0);
        assert_eq!(
            v,
            Ops007Verdict::Fail,
            "zero-byte input must Fail (vacuous Pass refused)"
        );
    }

    #[test]
    fn fail_zero_input_with_tokens() {
        // Counter corruption: 0 bytes but 5 tokens.
        let v = verdict_from_token_count_bound(0, 5);
        assert_eq!(v, Ops007Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 5: Overflow protection — checked_mul on input * 4.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_input_times_4_overflow() {
        // input * 4 overflows u64.
        let huge = u64::MAX / 2 + 1;
        let v = verdict_from_token_count_bound(huge, 0);
        // tokens=0 ≤ overflow → Fail (overflow triggers Fail).
        assert_eq!(
            v,
            Ops007Verdict::Fail,
            "input * 4 overflow must Fail (not silently wrap)"
        );
    }

    #[test]
    fn fail_input_max_overflow_with_tokens() {
        let v = verdict_from_token_count_bound(u64::MAX, 100);
        assert_eq!(v, Ops007Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 6: Boundary sweep — token-count sweep around 4× cap.
    // -------------------------------------------------------------------------
    #[test]
    fn token_count_sweep_at_fixed_input_100() {
        let input_len = 100_u64;
        let probes: Vec<(u64, Ops007Verdict)> = vec![
            (0, Ops007Verdict::Pass),
            (1, Ops007Verdict::Pass),
            (50, Ops007Verdict::Pass),
            (100, Ops007Verdict::Pass),
            (300, Ops007Verdict::Pass),
            (399, Ops007Verdict::Pass),
            (400, Ops007Verdict::Pass), // exact cap (inclusive)
            (401, Ops007Verdict::Fail), // just above cap
            (1_000, Ops007Verdict::Fail),
            (u64::MAX, Ops007Verdict::Fail),
        ];
        for (tokens, expected) in probes {
            let v = verdict_from_token_count_bound(input_len, tokens);
            assert_eq!(
                v, expected,
                "input=100 tokens={tokens} expected {expected:?}"
            );
        }
    }

    // -------------------------------------------------------------------------
    // Section 7: Realistic — UTF-8 byte-fallback worst-case scenarios.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_4_byte_emoji_at_4_tokens() {
        // Single emoji is 4 UTF-8 bytes → up to 4 byte-fallback tokens.
        let v = verdict_from_token_count_bound(4, 4);
        assert_eq!(v, Ops007Verdict::Pass);
    }

    #[test]
    fn pass_3_byte_cjk_at_3_tokens() {
        // CJK char is 3 UTF-8 bytes → up to 3 byte-fallback tokens.
        let v = verdict_from_token_count_bound(3, 3);
        assert_eq!(v, Ops007Verdict::Pass);
    }

    #[test]
    fn fail_adversarial_token_explosion() {
        // Realistic regression: tokenizer produces 5+ tokens per
        // input byte due to broken merge table.
        let v = verdict_from_token_count_bound(1_000, 5_001);
        assert_eq!(
            v,
            Ops007Verdict::Fail,
            "5+ tokens per byte must Fail (DoS class)"
        );
    }
}
