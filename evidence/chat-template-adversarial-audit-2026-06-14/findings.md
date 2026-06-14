# Chat prompt-construction — adversarial bug-hunt + hand-triage (2026-06-14)

Adversarial bug-hunt over the chat prompt-construction path (templates / roles / special
tokens): 5 finder dimensions → skeptic verification → **28 findings, 14 marked REAL.** Every
REAL verdict was then HAND-RE-CHECKED against the source before any fix (lesson from the
sampling audit, where a skeptic upheld a false positive). The hand-triage materially changed
the picture — several "REAL" verdicts were false positives, and writing a regression test
surfaced a *new* higher-impact bug the hunt missed.

## FIXED this PR (PMAT-762 — Llama2Template::format_conversation)

All three in `crates/aprender-serve/src/chat_template_llama2.rs`, covered by
`falsify_ct_762_*` (chat_template_contract_tests.rs), mutation-verified:

1. **System-only message silently dropped** (finding 1). A `[system]`-only request (or any
   list ending without a following user) stored the system prompt in a local that was only
   consumed by a user turn → output was just `"<s>"`, the system context lost. Fix: flush an
   unconsumed system prompt after the loop as a trailing `[INST] <<SYS>>…<</SYS>> [/INST]`.
2. **Multiple system messages overwrote** (finding 2). `system_prompt = Some(content)` kept
   only the LAST. Fix: concatenate with `\n\n`.
3. **DOUBLE `<s>` (double-BOS) for `[system, user]`** — NOT in the hunt's 14; found by the
   regression test I wrote for #1/#2. The new-round `<s>` (`if i > 0 && !in_user_turn`) fired
   at the first user turn whenever a system message preceded it, emitting `"<s><s>[INST]…"`.
   This hit the COMMON system+user case → higher impact than #1/#2. Fix: a `turn_started`
   guard so the round-opening `<s>` is added only AFTER a completed assistant turn.

Co-evolution (Rule 7): chat-template-v1 → 1.4.0, FALSIFY-CHAT-762.

## RESOLVED after the initial triage

- **[11] RawTemplate concatenates messages with no separators** — **FIXED (PMAT-763)**.
  `RawTemplate::format_conversation` was `.map(sanitize).collect::<String>()` → `[user "Hello",
  assistant "World"]` became `"HelloWorld"`. Reachable via `format_chat_messages(Some(
  request.model))` on the registry_fallback + chat_completions_stream paths when the model name
  matches no template pattern (e.g. "default"). Now emits `content + '\n'` per message,
  matching the realize_handlers raw fallback. Falsifier `falsify_ct_763_*` (mutation-verified).
- **[10] "Missing/divergent BOS in serve chat vs CLI"** — **RESOLVED as a FALSE POSITIVE via
  code analysis (no model needed)**. `BPETokenizer::encode` (tokenizer.rs:259) maps a template's
  literal `<s>`/`<|im_start|>` to its special-token id (via `bpe_encode`'s special-token split
  when merge_rules are present, else the fallback's greedy `token_to_id` match) and does NOT
  auto-prepend BOS. So serve gets BOS exactly once VIA THE TEMPLATE; the CLI (`infer/mod.rs`)
  prepends only because its non-templated raw-prompt path has no embedded BOS. The two are
  different-but-both-valid mechanisms — a blind serve-side prepend would have CAUSED double-BOS.
  **No code change. Do NOT add a serve-side BOS prepend.** (The "needs evidence, don't patch
  blind" deferral was vindicated — the obvious fix was the wrong one.)

## FALSE POSITIVES (skeptic upheld; rejected on hand re-check)

- **[3,5,7] Llama2 + [4,6] Mistral "missing generation prompt"** — ending in `[/INST]` IS the
  correct generation signal for Llama-2/Mistral (the model generates right after `[/INST]`);
  unlike ChatML/Zephyr these formats have no separate `assistant` marker. Finding 4's own
  skeptic even wrote "NO FIX NEEDED — correct per Mistral spec," yet it was listed REAL.
- **[8] RawTemplate doesn't sanitize** — it DOES (chat_template_helpers.rs:6,13). The finding
  conflated it with the realize_handlers `unwrap_or_else` fallback (which is ~unreachable —
  `format_messages` returns Ok for all templates incl. Raw, so the error fallback never fires).
- **[14] "Unicode/emoji loses encoding context" (claimed critical)** — vague mechanism; the
  BPE encoder handles UTF-8 via byte-level fallback. No concrete malformed-prompt repro.

## LOW / spec-compliant (no action)

- **[9] HF undefined template var → empty string** — minijinja default; matches Jinja2.
- **[12] empty content → empty token vec** — spec-compliant for an empty string.
- **[13] special-token literals encoded as text IF special_tokens map empty** — conditional on
  an empty map (a misconfigured tokenizer), not the normal path.

## Method / lesson
Skeptic verification kept signal high, but HAND-RE-CHECKING the REAL verdicts was essential:
of 14 "REAL", ~5 were false positives and the highest-impact bug (double-BOS) was found only
by writing a regression test. Always hand-verify + regression-test before fixing; the BOS
question (finding 10) needs trace evidence, not a blind patch.
