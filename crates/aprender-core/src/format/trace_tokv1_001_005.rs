// Bundles two sister contracts in one verdict module:
//
//   `tracing-observability-v1` (FALSIFY-TRACING_OBSERVABILITY_V1_001..002)
//   `tokenizer-v1` (FALSIFY-TOK-001..003)
//
// Module name `trace_tokv1` disambiguates from `tok_001_007`
// (tokenizer-loading-v1, already bound at task #289).
//
// TRACE-001: ∀ span: started → ended (no orphan spans)
// TRACE-002: ∀ child: child.start ≥ parent.start ∧ child.end ≤ parent.end
// TOK-001: BPE tokenizer encode_decode_roundtrip ≈ identity
// TOK-002: BOS/EOS detected from tokenizer.json when defined in config
// TOK-003: GGUF vocab extraction matches tokenizer.json vocab

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraceTokVerdict {
    Pass,
    Fail,
}

#[derive(Debug, Clone, Copy)]
pub struct Span {
    pub started: bool,
    pub ended: bool,
    pub start_ts: i64,
    pub end_ts: i64,
}

// ----------------------------------------------------------------
// TRACE-001
// ----------------------------------------------------------------

/// TRACE-001: every span that started must have ended (no orphans).
///
/// Pass iff for every span, `started → ended` AND `start_ts <= end_ts`.
#[must_use]
pub fn verdict_from_no_orphan_spans(spans: &[Span]) -> TraceTokVerdict {
    if spans.is_empty() {
        return TraceTokVerdict::Fail;
    }
    for s in spans {
        if s.started && !s.ended {
            return TraceTokVerdict::Fail;
        }
        if s.started && s.start_ts > s.end_ts {
            return TraceTokVerdict::Fail;
        }
    }
    TraceTokVerdict::Pass
}

// ----------------------------------------------------------------
// TRACE-002
// ----------------------------------------------------------------

/// TRACE-002: child span fully contained inside parent.
///
/// Pass iff `child.start_ts ≥ parent.start_ts AND child.end_ts ≤ parent.end_ts`.
#[must_use]
pub fn verdict_from_parent_child_ordering(parent: Span, child: Span) -> TraceTokVerdict {
    if !parent.ended || !child.ended {
        return TraceTokVerdict::Fail;
    }
    if child.start_ts >= parent.start_ts && child.end_ts <= parent.end_ts {
        TraceTokVerdict::Pass
    } else {
        TraceTokVerdict::Fail
    }
}

// ----------------------------------------------------------------
// TOK-001 BPE roundtrip
// ----------------------------------------------------------------

/// TOK-001: encode→decode preserves text (whitespace-normalized).
///
/// `original` is the input text.
/// `roundtrip` is `decode(encode(original))`.
/// Pass iff `whitespace_normalize(original) == whitespace_normalize(roundtrip)`.
#[must_use]
pub fn verdict_from_bpe_roundtrip(original: &str, roundtrip: &str) -> TraceTokVerdict {
    let norm = |s: &str| s.split_whitespace().collect::<Vec<_>>().join(" ");
    if norm(original) == norm(roundtrip) {
        TraceTokVerdict::Pass
    } else {
        TraceTokVerdict::Fail
    }
}

// ----------------------------------------------------------------
// TOK-002 special token detection
// ----------------------------------------------------------------

/// TOK-002: BOS/EOS detected when defined in config.
///
/// Pass iff:
///   - bos_in_config → bos_id was found in vocab (i.e., bos_id is Some)
///   - eos_in_config → eos_id was found in vocab
#[must_use]
pub fn verdict_from_special_token_detection(
    bos_in_config: bool,
    bos_id: Option<u32>,
    eos_in_config: bool,
    eos_id: Option<u32>,
) -> TraceTokVerdict {
    if bos_in_config && bos_id.is_none() {
        return TraceTokVerdict::Fail;
    }
    if eos_in_config && eos_id.is_none() {
        return TraceTokVerdict::Fail;
    }
    TraceTokVerdict::Pass
}

// ----------------------------------------------------------------
// TOK-003 GGUF vocab matches tokenizer.json
// ----------------------------------------------------------------

/// TOK-003: GGUF vocab extraction matches tokenizer.json vocab.
///
/// `gguf_vocab_size` and `tokenizer_json_vocab_size` are vocab counts.
/// Pass iff equal AND non-zero.
#[must_use]
pub fn verdict_from_gguf_vocab_matches(
    gguf_vocab_size: u32,
    tokenizer_json_vocab_size: u32,
) -> TraceTokVerdict {
    if gguf_vocab_size == 0 || tokenizer_json_vocab_size == 0 {
        return TraceTokVerdict::Fail;
    }
    if gguf_vocab_size == tokenizer_json_vocab_size {
        TraceTokVerdict::Pass
    } else {
        TraceTokVerdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn span(started: bool, ended: bool, start: i64, end: i64) -> Span {
        Span {
            started,
            ended,
            start_ts: start,
            end_ts: end,
        }
    }

    // -----------------------------------------------------------------
    // Section 1: TRACE-001..002.
    // -----------------------------------------------------------------
    #[test]
    fn ftrace001_pass_all_started_and_ended() {
        let spans = vec![
            span(true, true, 0, 100),
            span(true, true, 100, 200),
        ];
        let v = verdict_from_no_orphan_spans(&spans);
        assert_eq!(v, TraceTokVerdict::Pass);
    }

    #[test]
    fn ftrace001_fail_orphan_span() {
        let spans = vec![span(true, false, 0, 0)];
        let v = verdict_from_no_orphan_spans(&spans);
        assert_eq!(v, TraceTokVerdict::Fail);
    }

    #[test]
    fn ftrace001_fail_inverted_timestamps() {
        let spans = vec![span(true, true, 200, 100)];
        let v = verdict_from_no_orphan_spans(&spans);
        assert_eq!(v, TraceTokVerdict::Fail);
    }

    #[test]
    fn ftrace001_fail_empty() {
        let v = verdict_from_no_orphan_spans(&[]);
        assert_eq!(v, TraceTokVerdict::Fail);
    }

    #[test]
    fn ftrace002_pass_child_inside_parent() {
        let parent = span(true, true, 0, 1000);
        let child = span(true, true, 100, 800);
        let v = verdict_from_parent_child_ordering(parent, child);
        assert_eq!(v, TraceTokVerdict::Pass);
    }

    #[test]
    fn ftrace002_pass_child_at_boundaries() {
        let parent = span(true, true, 0, 1000);
        let child = span(true, true, 0, 1000);
        let v = verdict_from_parent_child_ordering(parent, child);
        assert_eq!(v, TraceTokVerdict::Pass);
    }

    #[test]
    fn ftrace002_fail_child_starts_before_parent() {
        let parent = span(true, true, 100, 1000);
        let child = span(true, true, 50, 800);
        let v = verdict_from_parent_child_ordering(parent, child);
        assert_eq!(v, TraceTokVerdict::Fail);
    }

    #[test]
    fn ftrace002_fail_child_ends_after_parent() {
        let parent = span(true, true, 0, 500);
        let child = span(true, true, 100, 800);
        let v = verdict_from_parent_child_ordering(parent, child);
        assert_eq!(v, TraceTokVerdict::Fail);
    }

    #[test]
    fn ftrace002_fail_unended_parent() {
        let parent = span(true, false, 0, 0);
        let child = span(true, true, 100, 800);
        let v = verdict_from_parent_child_ordering(parent, child);
        assert_eq!(v, TraceTokVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 2: TOK-001 BPE roundtrip.
    // -----------------------------------------------------------------
    #[test]
    fn ftok001_pass_simple_roundtrip() {
        let v = verdict_from_bpe_roundtrip("Hello world", "Hello world");
        assert_eq!(v, TraceTokVerdict::Pass);
    }

    #[test]
    fn ftok001_pass_whitespace_normalized() {
        // Multiple spaces collapse to single space
        let v = verdict_from_bpe_roundtrip("Hello   world", "Hello world");
        assert_eq!(v, TraceTokVerdict::Pass);
    }

    #[test]
    fn ftok001_fail_token_drift() {
        let v = verdict_from_bpe_roundtrip("Hello world", "Goodbye world");
        assert_eq!(v, TraceTokVerdict::Fail);
    }

    #[test]
    fn ftok001_fail_truncation() {
        let v = verdict_from_bpe_roundtrip("Hello world how are you", "Hello world");
        assert_eq!(v, TraceTokVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 3: TOK-002 special token.
    // -----------------------------------------------------------------
    #[test]
    fn ftok002_pass_bos_eos_detected() {
        let v = verdict_from_special_token_detection(true, Some(1), true, Some(2));
        assert_eq!(v, TraceTokVerdict::Pass);
    }

    #[test]
    fn ftok002_pass_no_special_tokens_in_config() {
        let v = verdict_from_special_token_detection(false, None, false, None);
        assert_eq!(v, TraceTokVerdict::Pass);
    }

    #[test]
    fn ftok002_fail_bos_in_config_but_not_found() {
        let v = verdict_from_special_token_detection(true, None, true, Some(2));
        assert_eq!(v, TraceTokVerdict::Fail);
    }

    #[test]
    fn ftok002_fail_eos_in_config_but_not_found() {
        let v = verdict_from_special_token_detection(true, Some(1), true, None);
        assert_eq!(v, TraceTokVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 4: TOK-003 GGUF vocab match.
    // -----------------------------------------------------------------
    #[test]
    fn ftok003_pass_qwen2_vocab() {
        let v = verdict_from_gguf_vocab_matches(151_936, 151_936);
        assert_eq!(v, TraceTokVerdict::Pass);
    }

    #[test]
    fn ftok003_fail_drift() {
        let v = verdict_from_gguf_vocab_matches(151_936, 32_000);
        assert_eq!(v, TraceTokVerdict::Fail);
    }

    #[test]
    fn ftok003_fail_zero_gguf() {
        let v = verdict_from_gguf_vocab_matches(0, 32_000);
        assert_eq!(v, TraceTokVerdict::Fail);
    }

    #[test]
    fn ftok003_fail_zero_json() {
        let v = verdict_from_gguf_vocab_matches(32_000, 0);
        assert_eq!(v, TraceTokVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 5: Mutation surveys.
    // -----------------------------------------------------------------
    #[test]
    fn mutation_survey_002_containment_table() {
        // Sweep 4 (start, end) pairs against parent (0, 1000)
        let parent = span(true, true, 0, 1000);
        let cases = [
            ((0_i64, 500_i64), TraceTokVerdict::Pass),
            ((500, 1000), TraceTokVerdict::Pass),
            ((0, 1000), TraceTokVerdict::Pass),
            ((-1, 999), TraceTokVerdict::Fail),     // starts before
            ((1, 1001), TraceTokVerdict::Fail),     // ends after
            ((0, 1500), TraceTokVerdict::Fail),     // ends after
        ];
        for ((s, e), expected) in cases {
            let child = span(true, true, s, e);
            let v = verdict_from_parent_child_ordering(parent, child);
            assert_eq!(v, expected, "child=({s}, {e})");
        }
    }

    // -----------------------------------------------------------------
    // Section 6: Realistic.
    // -----------------------------------------------------------------
    #[test]
    fn realistic_healthy_passes_all_5() {
        let v1 = verdict_from_no_orphan_spans(&[
            span(true, true, 0, 100),
            span(true, true, 50, 80),
        ]);
        let v2 = verdict_from_parent_child_ordering(
            span(true, true, 0, 100),
            span(true, true, 50, 80),
        );
        let v3 = verdict_from_bpe_roundtrip("What is 2+2?", "What is 2+2?");
        let v4 = verdict_from_special_token_detection(true, Some(151643), true, Some(151645));
        let v5 = verdict_from_gguf_vocab_matches(151_936, 151_936);
        for v in [v1, v2, v3, v4, v5] {
            assert_eq!(v, TraceTokVerdict::Pass);
        }
    }

    // -----------------------------------------------------------------
    // Section 7: Pre-fix regressions.
    // -----------------------------------------------------------------
    #[test]
    fn realistic_pre_fix_all_5_failures() {
        let v1 = verdict_from_no_orphan_spans(&[span(true, false, 0, 0)]);
        let v2 = verdict_from_parent_child_ordering(
            span(true, true, 100, 200),
            span(true, true, 50, 800), // child starts before parent + ends after
        );
        let v3 = verdict_from_bpe_roundtrip("What is 2+2?", "Goodbye world");
        let v4 = verdict_from_special_token_detection(true, None, true, Some(151645));
        let v5 = verdict_from_gguf_vocab_matches(151_936, 32_000);
        for v in [v1, v2, v3, v4, v5] {
            assert_eq!(v, TraceTokVerdict::Fail);
        }
    }
}
