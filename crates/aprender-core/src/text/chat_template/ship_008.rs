// SHIP-TWO-001 AC-SHIP1-008 / FALSIFY-SHIP-008 algorithm-level PARTIAL discharge.
//
// Spec: docs/specifications/aprender-train/ship-two-models-spec.md
// Contract: contracts/chat-template-v1.yaml (GATE-CHAT-SHIP-008, bumped 1.0.0 → 1.1.0)
//
// AC-SHIP1-008 states that the MODEL-1 teacher (Qwen2.5-Coder-7B-Instruct,
// ChatML family) renders the canonical system+user messages to a
// byte-equal golden string. The verdict is a pure byte-equality over
// `&str`: `Pass` iff `rendered == golden`, else `Fail`.
//
// This file discharges the *decision rule* at `PARTIAL_ALGORITHM_LEVEL`:
// given the canonical `CANONICAL_MESSAGES`, the `ChatMLTemplate` engine
// produces `AC_SHIP1_008_CANONICAL_GOLDEN` byte-for-byte. The compute-heavy
// portion of the AC (running `apr run teacher.apr --prompt …` against
// the live teacher weights) is intentionally out of scope here; the
// render rule is what changes the tokens-in-stream for downstream
// generation, and changing either side of the bind breaks this test
// before any teacher inference is launched.
//
// Mirrors the MODEL-2 pattern set by SHIP-016/017/018/019/020: pure
// verdict fn + binary enum + exhaustive mutation survey + provenance
// pin. MODEL-1 is now at 2/10 AC-SHIP1 items touched (SHIP-008 +
// SHIP-009).

/// Canonical system message used as the fixed input to the MODEL-1
/// teacher chat-template render check. Kept ASCII-only and free of
/// angle-brackets so user-content sanitization (`sanitize_user_content`)
/// is a no-op — this isolates the *template rendering* rule from
/// the *sanitization* rule, which has its own SHIP-independent test
/// coverage in `tests_sanitize.rs`.
pub const AC_SHIP1_008_CANONICAL_SYSTEM: &str = "You are a helpful coding assistant.";

/// Canonical user message. Same ASCII-only, no-angle-brackets shape
/// so the sanitize step is idempotent; breaks in sanitize would
/// falsify `tests_sanitize.rs`, not SHIP-008.
pub const AC_SHIP1_008_CANONICAL_USER: &str = "Write a Python function to compute the nth Fibonacci number.";

/// Byte-exact expected ChatML rendering for `CANONICAL_MESSAGES` under
/// `ChatMLTemplate::new().format_conversation(...)`. This IS the
/// contract: a single edit to either side of this bind (the engine's
/// `format_message` template, the special-token strings, the trailing
/// generation-prompt, or this constant itself) fails
/// `falsify_ship_008_chat_template_render_bind` before a single byte
/// of teacher output is generated downstream.
///
/// Derivation: `format_conversation` emits
/// `<|im_start|>{role}\n{content}<|im_end|>\n` for each message, then
/// appends `<|im_start|>assistant\n` as the generation prompt.
pub const AC_SHIP1_008_CANONICAL_GOLDEN: &str = "\
<|im_start|>system
You are a helpful coding assistant.<|im_end|>
<|im_start|>user
Write a Python function to compute the nth Fibonacci number.<|im_end|>
<|im_start|>assistant
";

/// Binary verdict for FALSIFY-SHIP-008 / GATE-CHAT-SHIP-008.
/// `Pass` iff the engine-produced render is byte-equal to the golden.
/// `Fail` otherwise (any single byte divergence — length, ordering,
/// role, delimiter, or content).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ship008Verdict {
    Pass,
    Fail,
}

/// Algorithm-level verdict rule for FALSIFY-SHIP-008 / GATE-CHAT-SHIP-008
/// / AC-SHIP1-008: the ChatML render of the canonical (system, user)
/// messages must be byte-equal to the golden string. Verdict is `Pass`
/// iff `rendered == golden` (strict `&str` equality, no normalization,
/// no whitespace tolerance). This proves the decision rule without
/// invoking the teacher model itself; the full discharge (live
/// `apr run paiml/qwen2.5-coder-7b-apache-q4k-v1 --prompt …` + diff
/// against golden completion) remains blocked on evidence collection.
#[must_use]
pub const fn verdict_from_chat_template_render(
    rendered: &str,
    golden: &str,
) -> Ship008Verdict {
    // const fn: byte slices compared by length + content, not
    // UTF-8 aware but ChatML output is ASCII+UTF-8 multibyte-safe
    // when inputs are.
    let a = rendered.as_bytes();
    let b = golden.as_bytes();
    if a.len() != b.len() {
        return Ship008Verdict::Fail;
    }
    let mut i = 0;
    while i < a.len() {
        if a[i] != b[i] {
            return Ship008Verdict::Fail;
        }
        i += 1;
    }
    Ship008Verdict::Pass
}

// ─────────────────────────────────────────────────────────────
// Unit tests — FALSIFY-SHIP-008 algorithm-level proof
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod ship_008_tests {
    use super::*;

    /// FALSIFY-SHIP-008 algorithm-level PARTIAL discharge: prove the
    /// byte-equality decision rule and the bind from
    /// `ChatMLTemplate::format_conversation` to
    /// `AC_SHIP1_008_CANONICAL_GOLDEN`. Any edit to either side
    /// must break this test before teacher inference runs.
    #[test]
    fn falsify_ship_008_chat_template_render_bind() {
        // Section 1: the verdict fn is a pure byte-equality.
        assert_eq!(
            verdict_from_chat_template_render(
                AC_SHIP1_008_CANONICAL_GOLDEN,
                AC_SHIP1_008_CANONICAL_GOLDEN,
            ),
            Ship008Verdict::Pass,
            "byte-identical strings must yield Pass",
        );

        // Section 2: the ChatML engine renders the canonical messages
        // to exactly the golden string. This is the load-bearing bind:
        // edits to `template.rs` or the constants above flip this.
        let engine = super::ChatMLTemplate::new();
        let messages = vec![
            super::ChatMessage::new("system", AC_SHIP1_008_CANONICAL_SYSTEM),
            super::ChatMessage::new("user", AC_SHIP1_008_CANONICAL_USER),
        ];
        let rendered = engine
            .format_conversation(&messages)
            .expect("canonical messages render without error");
        assert_eq!(
            verdict_from_chat_template_render(&rendered, AC_SHIP1_008_CANONICAL_GOLDEN),
            Ship008Verdict::Pass,
            "ChatML engine must produce byte-exact golden for canonical \
             (system, user) input — divergence means template drift",
        );

        // Section 3: counter-examples — corrupting the render in any
        // direction must flip the verdict to Fail. Each of these
        // mirrors a realistic template-drift failure mode.

        // 3a: empty render (engine crashed / returned "") → Fail.
        assert_eq!(
            verdict_from_chat_template_render("", AC_SHIP1_008_CANONICAL_GOLDEN),
            Ship008Verdict::Fail,
            "empty render against non-empty golden must Fail",
        );

        // 3b: missing trailing generation prompt (engine forgot to
        // append `<|im_start|>assistant\n`) → Fail.
        let without_gen_prompt = "\
<|im_start|>system
You are a helpful coding assistant.<|im_end|>
<|im_start|>user
Write a Python function to compute the nth Fibonacci number.<|im_end|>
";
        assert_eq!(
            verdict_from_chat_template_render(
                without_gen_prompt,
                AC_SHIP1_008_CANONICAL_GOLDEN,
            ),
            Ship008Verdict::Fail,
            "render missing the assistant generation prompt must Fail \
             — this is the class of bug that produced blank output \
             in realizar during the Qwen2 migration (GH-PMAT-593)",
        );

        // 3c: wrong special-token (`<|im_start|>` swapped for
        // `<|user|>` — a Llama2-style drift) → Fail.
        let wrong_delim = "\
<|user|>system
You are a helpful coding assistant.<|im_end|>
<|user|>user
Write a Python function to compute the nth Fibonacci number.<|im_end|>
<|user|>assistant
";
        assert_eq!(
            verdict_from_chat_template_render(wrong_delim, AC_SHIP1_008_CANONICAL_GOLDEN),
            Ship008Verdict::Fail,
            "wrong special-token delimiter must Fail — this is the \
             cross-family drift class (ChatML → Llama2 mix)",
        );

        // 3d: swapped role order (user before system — some templates
        // enforce order) → Fail.
        let swapped_roles = "\
<|im_start|>user
Write a Python function to compute the nth Fibonacci number.<|im_end|>
<|im_start|>system
You are a helpful coding assistant.<|im_end|>
<|im_start|>assistant
";
        assert_eq!(
            verdict_from_chat_template_render(swapped_roles, AC_SHIP1_008_CANONICAL_GOLDEN),
            Ship008Verdict::Fail,
            "role-order swap must Fail — engine must preserve message \
             ordering for prompt-format stability",
        );

        // 3e: single-byte flip at the final character (trailing
        // newline → space) — proves sensitivity to whitespace.
        let single_byte_flipped = {
            let mut bytes = AC_SHIP1_008_CANONICAL_GOLDEN.as_bytes().to_vec();
            *bytes.last_mut().expect("golden is non-empty") = b' ';
            String::from_utf8(bytes).expect("ASCII flip stays valid UTF-8")
        };
        assert_eq!(
            verdict_from_chat_template_render(
                &single_byte_flipped,
                AC_SHIP1_008_CANONICAL_GOLDEN,
            ),
            Ship008Verdict::Fail,
            "single trailing-byte flip (\\n → space) must Fail — \
             byte-equality, not approx-equality",
        );

        // Section 4: symmetry of the decision rule — swapping
        // rendered/golden preserves the verdict (equality is
        // commutative).
        assert_eq!(
            verdict_from_chat_template_render(
                AC_SHIP1_008_CANONICAL_GOLDEN,
                "",
            ),
            Ship008Verdict::Fail,
            "non-empty vs empty must Fail regardless of argument order",
        );
        assert_eq!(
            verdict_from_chat_template_render("", ""),
            Ship008Verdict::Pass,
            "empty == empty is vacuously Pass — the rule is byte-equality, \
             not a minimum-length gate",
        );

        // Section 5: provenance pin — the golden string is load-bearing
        // and must contain the three ChatML delimiters plus one
        // generation-prompt trigger. Drift here flags lockstep with
        // `contracts/model-families/qwen2.yaml::chat_template`.
        assert!(
            AC_SHIP1_008_CANONICAL_GOLDEN.contains("<|im_start|>"),
            "golden must contain ChatML im_start delimiter — drift \
             from qwen2.yaml chat_template field",
        );
        assert!(
            AC_SHIP1_008_CANONICAL_GOLDEN.contains("<|im_end|>"),
            "golden must contain ChatML im_end delimiter",
        );
        assert!(
            AC_SHIP1_008_CANONICAL_GOLDEN.ends_with("<|im_start|>assistant\n"),
            "golden must end with the assistant generation prompt — \
             this is what realizar feeds the sampler",
        );
        assert_eq!(
            AC_SHIP1_008_CANONICAL_GOLDEN.matches("<|im_start|>").count(),
            3,
            "golden must have exactly 3 im_start occurrences \
             (system + user + assistant) for the canonical 2-message input",
        );
        assert_eq!(
            AC_SHIP1_008_CANONICAL_GOLDEN.matches("<|im_end|>").count(),
            2,
            "golden must have exactly 2 im_end occurrences (system + user); \
             assistant is unterminated because generation has not yet run",
        );
    }
}
