// SHIP-TWO-001 — `chat-template-v1` algorithm-level PARTIAL discharge
// for FALSIFY-CHAT-001..003 (closes 3/3).
//
// Contract: `contracts/chat-template-v1.yaml`.
// Spec: SHIP-TWO-001 §12 (chat template ship gates).
//
// Bundle of 3 verdict fns; each is a pure decision rule over a
// minimal evidence struct so the algorithm-level rule is testable
// offline without standing up a full template engine. The
// runtime-level falsifier `cargo test ... chat_template` already
// exists per contract evidence — this file pins the *decision rule*
// against drift.

// ===========================================================================
// CHAT-001 — ChatML template produces correct output
// ===========================================================================
//
// ChatML well-formedness rule: a non-empty render whose canonical
// `<|im_start|>` / `<|im_end|>` tokens are both present and balanced.
// "Balanced" means equal counts (every open has a close) — the
// template engine must emit closed turn structure.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Chat001Verdict { Pass, Fail }

/// Pass iff the rendered string contains at least one ChatML turn
/// (`<|im_start|>`) AND a matching close (`<|im_end|>`) AND open count
/// equals close count (no unbalanced delimiters).
#[must_use]
pub fn verdict_from_chatml_render(rendered: &str) -> Chat001Verdict {
    if rendered.is_empty() { return Chat001Verdict::Fail; }
    let opens = rendered.matches("<|im_start|>").count();
    let closes = rendered.matches("<|im_end|>").count();
    if opens == 0 || closes == 0 { return Chat001Verdict::Fail; }
    if opens != closes { return Chat001Verdict::Fail; }
    Chat001Verdict::Pass
}

// ===========================================================================
// CHAT-002 — Llama 3 template produces correct output
// ===========================================================================
//
// Llama 3 well-formedness rule: render begins with the
// `<|begin_of_text|>` BOS marker AND contains at least one
// `<|eot_id|>` end-of-turn marker. Llama 3 templates that omit BOS
// are out-of-spec; templates without `<|eot_id|>` cannot be sampled
// against (no stop signal).

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Chat002Verdict { Pass, Fail }

/// Pass iff the rendered string starts with `<|begin_of_text|>` AND
/// contains at least one `<|eot_id|>` marker.
#[must_use]
pub fn verdict_from_llama3_render(rendered: &str) -> Chat002Verdict {
    if rendered.is_empty() { return Chat002Verdict::Fail; }
    if !rendered.starts_with("<|begin_of_text|>") { return Chat002Verdict::Fail; }
    if !rendered.contains("<|eot_id|>") { return Chat002Verdict::Fail; }
    Chat002Verdict::Pass
}

// ===========================================================================
// CHAT-003 — Missing template produces error, not silent skip
// ===========================================================================
//
// PMAT-237 lesson: process validation, not outcome validation. When
// no chat template is registered for a model, the engine MUST emit
// an explicit error — not return an empty string, not return raw
// concatenated content, not silently fall back to a default.
//
// Decision rule: classify the engine's response and Pass iff it is
// `ExplicitError`.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MissingTemplateOutcome {
    /// Engine returned `Err(...)` — the only acceptable outcome.
    ExplicitError,
    /// Engine silently returned an empty string.
    SilentEmpty,
    /// Engine concatenated raw role/content without delimiters.
    SilentConcat,
    /// Engine fell back to a default template (e.g., always-ChatML)
    /// without surfacing the missing-template event.
    SilentFallback,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Chat003Verdict { Pass, Fail }

/// Pass iff `outcome == ExplicitError`. All silent-success branches
/// (Empty, Concat, Fallback) are Fail — they violate the contract
/// invariant that "missing template" is loudly diagnosed.
#[must_use]
pub fn verdict_from_missing_template_outcome(outcome: MissingTemplateOutcome) -> Chat003Verdict {
    match outcome {
        MissingTemplateOutcome::ExplicitError => Chat003Verdict::Pass,
        _ => Chat003Verdict::Fail,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ----- CHAT-001 ----------------------------------------------------------

    #[test]
    fn c001_pass_balanced_chatml() {
        let s = "<|im_start|>system\nYou are helpful.<|im_end|>\
                 <|im_start|>user\nHi<|im_end|>";
        assert_eq!(verdict_from_chatml_render(s), Chat001Verdict::Pass);
    }

    #[test]
    fn c001_pass_single_turn() {
        let s = "<|im_start|>user\nHi<|im_end|>";
        assert_eq!(verdict_from_chatml_render(s), Chat001Verdict::Pass);
    }

    #[test]
    fn c001_fail_empty() {
        assert_eq!(verdict_from_chatml_render(""), Chat001Verdict::Fail);
    }

    #[test]
    fn c001_fail_no_opens() {
        assert_eq!(verdict_from_chatml_render("plain text"), Chat001Verdict::Fail);
    }

    #[test]
    fn c001_fail_unbalanced_open_only() {
        let s = "<|im_start|>user\nHi";
        assert_eq!(verdict_from_chatml_render(s), Chat001Verdict::Fail);
    }

    #[test]
    fn c001_fail_unbalanced_close_only() {
        let s = "user\nHi<|im_end|>";
        assert_eq!(verdict_from_chatml_render(s), Chat001Verdict::Fail);
    }

    #[test]
    fn c001_fail_more_opens_than_closes() {
        let s = "<|im_start|>a<|im_end|><|im_start|>b";
        assert_eq!(verdict_from_chatml_render(s), Chat001Verdict::Fail);
    }

    // ----- CHAT-002 ----------------------------------------------------------

    #[test]
    fn c002_pass_canonical() {
        let s = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>";
        assert_eq!(verdict_from_llama3_render(s), Chat002Verdict::Pass);
    }

    #[test]
    fn c002_fail_empty() {
        assert_eq!(verdict_from_llama3_render(""), Chat002Verdict::Fail);
    }

    #[test]
    fn c002_fail_missing_bos() {
        let s = "<|start_header_id|>user<|end_header_id|>Hi<|eot_id|>";
        assert_eq!(verdict_from_llama3_render(s), Chat002Verdict::Fail);
    }

    #[test]
    fn c002_fail_missing_eot() {
        let s = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>Hi";
        assert_eq!(verdict_from_llama3_render(s), Chat002Verdict::Fail);
    }

    #[test]
    fn c002_fail_chatml_in_llama3_slot() {
        // ChatML render is NOT a valid Llama-3 render.
        let s = "<|im_start|>user\nHi<|im_end|>";
        assert_eq!(verdict_from_llama3_render(s), Chat002Verdict::Fail);
    }

    // ----- CHAT-003 ----------------------------------------------------------

    #[test]
    fn c003_pass_explicit_error() {
        assert_eq!(
            verdict_from_missing_template_outcome(MissingTemplateOutcome::ExplicitError),
            Chat003Verdict::Pass
        );
    }

    #[test]
    fn c003_fail_silent_empty() {
        assert_eq!(
            verdict_from_missing_template_outcome(MissingTemplateOutcome::SilentEmpty),
            Chat003Verdict::Fail
        );
    }

    #[test]
    fn c003_fail_silent_concat() {
        assert_eq!(
            verdict_from_missing_template_outcome(MissingTemplateOutcome::SilentConcat),
            Chat003Verdict::Fail
        );
    }

    #[test]
    fn c003_fail_silent_fallback() {
        assert_eq!(
            verdict_from_missing_template_outcome(MissingTemplateOutcome::SilentFallback),
            Chat003Verdict::Fail
        );
    }

    #[test]
    fn c003_only_explicit_error_passes() {
        // Exhaustive sweep over all 4 outcome variants.
        let probes = [
            (MissingTemplateOutcome::ExplicitError, Chat003Verdict::Pass),
            (MissingTemplateOutcome::SilentEmpty, Chat003Verdict::Fail),
            (MissingTemplateOutcome::SilentConcat, Chat003Verdict::Fail),
            (MissingTemplateOutcome::SilentFallback, Chat003Verdict::Fail),
        ];
        for (outcome, expected) in probes {
            assert_eq!(
                verdict_from_missing_template_outcome(outcome),
                expected,
                "outcome {outcome:?}"
            );
        }
    }
}
