// SHIP-TWO-001 — `conversation-generation-v1` algorithm-level PARTIAL
// discharge for FALSIFY-CONV-001..005.
//
// Contract: `contracts/conversation-generation-v1.yaml`.
// Spec: `docs/specifications/aprender-train/ship-two-models-spec.md`.
//
// ## What this file proves NOW (PARTIAL_ALGORITHM_LEVEL)
//
// Five SSC v11 §6 conversation-generation gates:
//
// - CONV-001 (ChatML 3-turn structure): turns = [system, user, assistant].
// - CONV-002 (Type D ≥ 30%): safe-confirmation share floor.
// - CONV-003 (no empty responses): every turn has non-trivial content.
// - CONV-004 (variant diversity): no single prompt variant > 20%.
// - CONV-005 (system-prompt honesty): contains "not a replacement" AND
//   "pattern matching" disclaimers.
//
// All five are pure properties of the generated conversation batch.

/// Required Type-D fraction (safe-confirmations).
pub const AC_CONV_002_TYPE_D_FLOOR: f32 = 0.30;

/// Maximum variant share — no single template variant > 20%.
pub const AC_CONV_004_MAX_VARIANT_SHARE: f32 = 0.20;

/// Required substrings in the SSC v11 §6.5 system prompt.
pub const AC_CONV_005_HONESTY_PHRASE_1: &str = "not a replacement";
pub const AC_CONV_005_HONESTY_PHRASE_2: &str = "pattern matching";

/// Required ChatML role names in canonical order.
pub const AC_CONV_001_ROLE_SYSTEM: &str = "system";
pub const AC_CONV_001_ROLE_USER: &str = "user";
pub const AC_CONV_001_ROLE_ASSISTANT: &str = "assistant";

/// Required turn count in a single conversation.
pub const AC_CONV_001_TURN_COUNT: usize = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvVerdict {
    Pass,
    Fail,
}

/// Single conversation turn — role + content.
#[derive(Debug, Clone)]
pub struct Turn {
    pub role: String,
    pub content: String,
}

/// Single ChatML conversation.
#[derive(Debug, Clone)]
pub struct Conversation {
    pub turns: Vec<Turn>,
}

// -----------------------------------------------------------------------------
// Verdict 1: CONV-001 — ChatML 3-turn structure.
// -----------------------------------------------------------------------------

/// Pass iff every conversation has 3 turns with roles
/// [system, user, assistant] in order.
#[must_use]
pub fn verdict_from_chatml_structure(conversations: &[Conversation]) -> ConvVerdict {
    if conversations.is_empty() {
        return ConvVerdict::Fail;
    }
    for c in conversations {
        if c.turns.len() != AC_CONV_001_TURN_COUNT {
            return ConvVerdict::Fail;
        }
        if c.turns[0].role != AC_CONV_001_ROLE_SYSTEM {
            return ConvVerdict::Fail;
        }
        if c.turns[1].role != AC_CONV_001_ROLE_USER {
            return ConvVerdict::Fail;
        }
        if c.turns[2].role != AC_CONV_001_ROLE_ASSISTANT {
            return ConvVerdict::Fail;
        }
    }
    ConvVerdict::Pass
}

// -----------------------------------------------------------------------------
// Verdict 2: CONV-002 — Type D ≥ 30%.
// -----------------------------------------------------------------------------

/// `type_labels` is one entry per conversation, e.g. "A", "B", "C", "D".
/// Pass iff Type D fraction ≥ 0.30.
#[must_use]
pub fn verdict_from_type_d_threshold(type_labels: &[&str]) -> ConvVerdict {
    if type_labels.is_empty() {
        return ConvVerdict::Fail;
    }
    let total = type_labels.len() as f32;
    let count_d = type_labels.iter().filter(|&&t| t == "D").count() as f32;
    let frac = count_d / total;
    if frac >= AC_CONV_002_TYPE_D_FLOOR {
        ConvVerdict::Pass
    } else {
        ConvVerdict::Fail
    }
}

// -----------------------------------------------------------------------------
// Verdict 3: CONV-003 — no empty responses.
// -----------------------------------------------------------------------------

/// Pass iff every turn's `content.trim()` is non-empty.
#[must_use]
pub fn verdict_from_no_empty_responses(conversations: &[Conversation]) -> ConvVerdict {
    if conversations.is_empty() {
        return ConvVerdict::Fail;
    }
    for c in conversations {
        for t in &c.turns {
            if t.content.trim().is_empty() {
                return ConvVerdict::Fail;
            }
        }
    }
    ConvVerdict::Pass
}

// -----------------------------------------------------------------------------
// Verdict 4: CONV-004 — variant diversity (no single variant > 20%).
// -----------------------------------------------------------------------------

/// `variant_ids` is one entry per conversation. Pass iff no single
/// variant id appears with share > 20% of total.
#[must_use]
pub fn verdict_from_variant_diversity(variant_ids: &[u32]) -> ConvVerdict {
    if variant_ids.is_empty() {
        return ConvVerdict::Fail;
    }
    let total = variant_ids.len() as f32;
    let mut counts = std::collections::HashMap::<u32, usize>::new();
    for &id in variant_ids {
        *counts.entry(id).or_insert(0) += 1;
    }
    for &c in counts.values() {
        let share = c as f32 / total;
        if share > AC_CONV_004_MAX_VARIANT_SHARE {
            return ConvVerdict::Fail;
        }
    }
    ConvVerdict::Pass
}

// -----------------------------------------------------------------------------
// Verdict 5: CONV-005 — system-prompt honesty.
// -----------------------------------------------------------------------------

/// Pass iff the system prompt contains BOTH honesty phrases.
#[must_use]
pub fn verdict_from_system_prompt_honesty(system_prompt: &str) -> ConvVerdict {
    if !system_prompt.contains(AC_CONV_005_HONESTY_PHRASE_1) {
        return ConvVerdict::Fail;
    }
    if !system_prompt.contains(AC_CONV_005_HONESTY_PHRASE_2) {
        return ConvVerdict::Fail;
    }
    ConvVerdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_conv(roles: &[&str], contents: &[&str]) -> Conversation {
        Conversation {
            turns: roles
                .iter()
                .zip(contents.iter())
                .map(|(r, c)| Turn {
                    role: r.to_string(),
                    content: c.to_string(),
                })
                .collect(),
        }
    }

    fn canonical_conv() -> Conversation {
        make_conv(
            &["system", "user", "assistant"],
            &[
                "You are a code analyst — not a replacement, just pattern matching.",
                "Is `rm -rf /` safe?",
                "No, that command is unsafe.",
            ],
        )
    }

    // -------------------------------------------------------------------------
    // Section 1: Provenance pins.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_type_d_floor_30() {
        assert_eq!(AC_CONV_002_TYPE_D_FLOOR, 0.30);
    }

    #[test]
    fn provenance_max_variant_share_20() {
        assert_eq!(AC_CONV_004_MAX_VARIANT_SHARE, 0.20);
    }

    #[test]
    fn provenance_turn_count_3() {
        assert_eq!(AC_CONV_001_TURN_COUNT, 3);
    }

    #[test]
    fn provenance_role_strings() {
        assert_eq!(AC_CONV_001_ROLE_SYSTEM, "system");
        assert_eq!(AC_CONV_001_ROLE_USER, "user");
        assert_eq!(AC_CONV_001_ROLE_ASSISTANT, "assistant");
    }

    #[test]
    fn provenance_honesty_phrases() {
        assert_eq!(AC_CONV_005_HONESTY_PHRASE_1, "not a replacement");
        assert_eq!(AC_CONV_005_HONESTY_PHRASE_2, "pattern matching");
    }

    // -------------------------------------------------------------------------
    // Section 2: CONV-001 Pass band.
    // -------------------------------------------------------------------------
    #[test]
    fn conv001_pass_canonical_conversation() {
        let convs = vec![canonical_conv()];
        assert_eq!(verdict_from_chatml_structure(&convs), ConvVerdict::Pass);
    }

    #[test]
    fn conv001_pass_multiple_canonical() {
        let convs = vec![canonical_conv(), canonical_conv(), canonical_conv()];
        assert_eq!(verdict_from_chatml_structure(&convs), ConvVerdict::Pass);
    }

    // -------------------------------------------------------------------------
    // Section 3: CONV-001 Fail band.
    // -------------------------------------------------------------------------
    #[test]
    fn conv001_fail_two_turns_only() {
        let convs = vec![make_conv(&["system", "user"], &["sys", "ask"])];
        assert_eq!(verdict_from_chatml_structure(&convs), ConvVerdict::Fail);
    }

    #[test]
    fn conv001_fail_four_turns() {
        let convs = vec![make_conv(
            &["system", "user", "assistant", "user"],
            &["sys", "ask", "ans", "follow"],
        )];
        assert_eq!(verdict_from_chatml_structure(&convs), ConvVerdict::Fail);
    }

    #[test]
    fn conv001_fail_wrong_first_role() {
        let convs = vec![make_conv(
            &["user", "system", "assistant"],
            &["a", "b", "c"],
        )];
        assert_eq!(verdict_from_chatml_structure(&convs), ConvVerdict::Fail);
    }

    #[test]
    fn conv001_fail_swapped_user_assistant() {
        let convs = vec![make_conv(
            &["system", "assistant", "user"],
            &["a", "b", "c"],
        )];
        assert_eq!(verdict_from_chatml_structure(&convs), ConvVerdict::Fail);
    }

    #[test]
    fn conv001_fail_empty_batch() {
        let convs: Vec<Conversation> = vec![];
        assert_eq!(verdict_from_chatml_structure(&convs), ConvVerdict::Fail);
    }

    #[test]
    fn conv001_fail_one_corrupt_in_batch() {
        let mut convs = vec![canonical_conv(), canonical_conv()];
        convs[1].turns.pop(); // 2 turns
        assert_eq!(verdict_from_chatml_structure(&convs), ConvVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 4: CONV-002 — Type D threshold.
    // -------------------------------------------------------------------------
    #[test]
    fn conv002_pass_at_threshold() {
        // 3/10 = 30%.
        let labels = vec!["A", "B", "C", "D", "D", "D", "A", "B", "C", "A"];
        assert_eq!(verdict_from_type_d_threshold(&labels), ConvVerdict::Pass);
    }

    #[test]
    fn conv002_pass_well_above() {
        let labels = vec!["D", "D", "D", "D", "A", "B"];
        assert_eq!(verdict_from_type_d_threshold(&labels), ConvVerdict::Pass);
    }

    #[test]
    fn conv002_fail_below_threshold() {
        // 2/10 = 20% < 30%.
        let labels = vec!["A", "B", "C", "D", "D", "A", "B", "C", "A", "B"];
        assert_eq!(verdict_from_type_d_threshold(&labels), ConvVerdict::Fail);
    }

    #[test]
    fn conv002_fail_zero_d() {
        let labels = vec!["A", "B", "C", "A"];
        assert_eq!(verdict_from_type_d_threshold(&labels), ConvVerdict::Fail);
    }

    #[test]
    fn conv002_fail_empty() {
        let labels: Vec<&str> = vec![];
        assert_eq!(verdict_from_type_d_threshold(&labels), ConvVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 5: CONV-003 — no empty responses.
    // -------------------------------------------------------------------------
    #[test]
    fn conv003_pass_full_responses() {
        let convs = vec![canonical_conv()];
        assert_eq!(verdict_from_no_empty_responses(&convs), ConvVerdict::Pass);
    }

    #[test]
    fn conv003_fail_empty_assistant_response() {
        let convs = vec![make_conv(
            &["system", "user", "assistant"],
            &["sys", "ask", ""],
        )];
        assert_eq!(verdict_from_no_empty_responses(&convs), ConvVerdict::Fail);
    }

    #[test]
    fn conv003_fail_whitespace_only_response() {
        let convs = vec![make_conv(
            &["system", "user", "assistant"],
            &["sys", "ask", "   \n\t  "],
        )];
        assert_eq!(verdict_from_no_empty_responses(&convs), ConvVerdict::Fail);
    }

    #[test]
    fn conv003_fail_empty_in_one_of_many() {
        let mut convs = vec![canonical_conv(), canonical_conv()];
        convs[1].turns[2].content = String::new();
        assert_eq!(verdict_from_no_empty_responses(&convs), ConvVerdict::Fail);
    }

    #[test]
    fn conv003_fail_empty_batch() {
        let convs: Vec<Conversation> = vec![];
        assert_eq!(verdict_from_no_empty_responses(&convs), ConvVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 6: CONV-004 — variant diversity.
    // -------------------------------------------------------------------------
    #[test]
    fn conv004_pass_uniform_distribution() {
        // 12 variants, evenly distributed → max share ~8% < 20%.
        let mut variants = Vec::new();
        for v in 0..12_u32 {
            for _ in 0..5 {
                variants.push(v); // 5 each = 60 total
            }
        }
        assert_eq!(verdict_from_variant_diversity(&variants), ConvVerdict::Pass);
    }

    #[test]
    fn conv004_pass_mild_skew() {
        // 7 variants, max share = 3/15 = 20% (NOT > 20%).
        let variants = vec![0_u32, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6];
        assert_eq!(verdict_from_variant_diversity(&variants), ConvVerdict::Pass);
    }

    #[test]
    fn conv004_fail_one_variant_dominant() {
        // Single variant 60% of total.
        let mut variants = vec![0_u32; 6];
        variants.extend(vec![1_u32, 2, 3, 4]);
        assert_eq!(verdict_from_variant_diversity(&variants), ConvVerdict::Fail);
    }

    #[test]
    fn conv004_fail_just_above_20() {
        // 3/12 = 25% > 20%.
        let variants = vec![0_u32, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        assert_eq!(verdict_from_variant_diversity(&variants), ConvVerdict::Fail);
    }

    #[test]
    fn conv004_fail_empty() {
        let variants: Vec<u32> = vec![];
        assert_eq!(verdict_from_variant_diversity(&variants), ConvVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 7: CONV-005 — system-prompt honesty.
    // -------------------------------------------------------------------------
    #[test]
    fn conv005_pass_contains_both_phrases() {
        let prompt = "I am a code analyst — not a replacement for human review, \
                      and I work via pattern matching against known unsafe constructs.";
        assert_eq!(
            verdict_from_system_prompt_honesty(prompt),
            ConvVerdict::Pass
        );
    }

    #[test]
    fn conv005_fail_missing_phrase_1() {
        let prompt = "I work via pattern matching only.";
        assert_eq!(
            verdict_from_system_prompt_honesty(prompt),
            ConvVerdict::Fail
        );
    }

    #[test]
    fn conv005_fail_missing_phrase_2() {
        let prompt = "I am not a replacement for code review.";
        assert_eq!(
            verdict_from_system_prompt_honesty(prompt),
            ConvVerdict::Fail
        );
    }

    #[test]
    fn conv005_fail_neither_phrase() {
        let prompt = "Hello, I'm a coding assistant.";
        assert_eq!(
            verdict_from_system_prompt_honesty(prompt),
            ConvVerdict::Fail
        );
    }

    #[test]
    fn conv005_fail_empty_prompt() {
        assert_eq!(
            verdict_from_system_prompt_honesty(""),
            ConvVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 8: Sweep — Type D threshold around 30%.
    // -------------------------------------------------------------------------
    #[test]
    fn sweep_type_d_pct_around_threshold() {
        // 100 labels with varying D fractions.
        for d_count in [29_usize, 30, 31, 50, 99] {
            let mut labels = vec!["D"; d_count];
            for _ in 0..(100 - d_count) {
                labels.push("A");
            }
            let v = verdict_from_type_d_threshold(&labels);
            let expected = if d_count >= 30 {
                ConvVerdict::Pass
            } else {
                ConvVerdict::Fail
            };
            assert_eq!(v, expected, "d_count={d_count}");
        }
    }

    // -------------------------------------------------------------------------
    // Section 9: Realistic — full SSC v11 §6 batch.
    // -------------------------------------------------------------------------
    #[test]
    fn realistic_full_ssc_batch_passes_all_gates() {
        // Build a 10-conversation batch with 4 Type-D, 12-variant rotation,
        // canonical ChatML structure, and S6.5 honesty disclaimer.
        let prompt = "Code analyst — not a replacement for review; uses pattern matching.";
        let convs: Vec<Conversation> = (0..10)
            .map(|i| {
                make_conv(
                    &["system", "user", "assistant"],
                    &[
                        prompt,
                        &format!("Is `rm -rf /tmp/{}` safe?", i),
                        if i < 4 {
                            "Type D — confirmed safe."
                        } else {
                            "Type A — security concern."
                        },
                    ],
                )
            })
            .collect();
        let labels: Vec<&str> = (0..10)
            .map(|i| if i < 4 { "D" } else { "A" })
            .collect();
        let variants: Vec<u32> = (0..10).map(|i| (i as u32) % 12).collect();

        assert_eq!(verdict_from_chatml_structure(&convs), ConvVerdict::Pass);
        assert_eq!(verdict_from_type_d_threshold(&labels), ConvVerdict::Pass);
        assert_eq!(verdict_from_no_empty_responses(&convs), ConvVerdict::Pass);
        assert_eq!(verdict_from_variant_diversity(&variants), ConvVerdict::Pass);
        assert_eq!(
            verdict_from_system_prompt_honesty(prompt),
            ConvVerdict::Pass
        );
    }

    #[test]
    fn realistic_classification_bug_caught() {
        // CONV-002 if_fails: "Classification logic incorrectly marks
        // safe entries as unsafe" — only 10% Type D instead of 30+%.
        let labels = vec!["A"; 9].into_iter().chain(vec!["D"; 1]).collect::<Vec<_>>();
        assert_eq!(verdict_from_type_d_threshold(&labels), ConvVerdict::Fail);
    }

    #[test]
    fn realistic_template_render_empty_caught() {
        // CONV-003 if_fails: "Edge case in template rendering produces
        // empty output".
        let convs = vec![make_conv(
            &["system", "user", "assistant"],
            &["sys", "ask", ""], // template emitted empty
        )];
        assert_eq!(verdict_from_no_empty_responses(&convs), ConvVerdict::Fail);
    }

    #[test]
    fn realistic_seed_modular_clustering_caught() {
        // CONV-004 if_fails: "Seed modular arithmetic clusters on few
        // variants" — only variants 0..3 used.
        let mut variants = Vec::new();
        for _ in 0..30 {
            variants.push(0_u32);
        }
        for _ in 0..30 {
            variants.push(1);
        }
        for _ in 0..40 {
            variants.push(2);
        }
        // Variant 2 is 40% — exceeds 20%.
        assert_eq!(verdict_from_variant_diversity(&variants), ConvVerdict::Fail);
    }
}
