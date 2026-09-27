# #3801 — one detector: `apr chat` stops putting Qwen3 in thinking mode

**Branch** `PMAT-3801-detector-unification` · **author** aprender-c7 · **2026-09-22**
**Divergence from the issue's stated design is recorded ON THE ISSUE before any round:**
https://github.com/paiml/aprender/issues/3801#issuecomment-5773096497

## What was ruled, and why this is not the guard

The issue's "Decided design (aprender-fd)" is a think-budget guard. The cop ruled otherwise after the
measurement below: *"the detector unification IS #3801's remedy … rather than making the
non-terminating mode terminate"*, and on this row, *"Unify the detector; do not bound the loop."*
This receipt implements the ruling and **does not claim the four guard done_when clauses**.

## The defect

`apr chat` imported `detect_format_from_name` from **aprender-core**; every other verb takes
**realizar's**. aprender-core's `TemplateFormat`
(`crates/aprender-core/src/text/chat_template/mod.rs:195-201`) has seven variants — ChatML, Llama2,
Mistral, Alpaca, Phi, Custom, Raw — and **no `Qwen3NoThink`**. On that path no-think was not merely
unselected, it was **unrepresentable**, and `chat_load_tokenizers.rs:120`'s `template_format_name`
match compiled *with no arm for it* because it was matching the other crate's enum.

| path | detector | `qwen3` got |
|---|---|---|
| `apr serve` | realizar | `Qwen3NoThink` |
| `apr run --chat` | realizar, via `prepare_tokens_gguf` → `apr_arch_to_template_hint` | `Qwen3NoThink` |
| `apr qa` golden gate | realizar, since #3724 | `Qwen3NoThink` |
| **`apr chat`** | **aprender-core** (`chat.rs:34`) | **`ChatML` — it thought** |

## Measured, same model, same flags, same command

`apr chat ~/models/Qwen3-1.7B-Q4_K_M.gguf --no-gpu --temperature 0 --max-tokens 64`, prompt
`What is 2+2?`:

| | binary | banner | answer |
|---|---|---|---|
| BEFORE | `apr 0.69.0 (3ebb1c61e)` | `Chat Template: ChatML` | `Assistant: <think>⏎Okay, the user is asking "What is 2+2?" Let me think. This is a basic arithmetic problem…` (still reasoning at the 64-token cut) |
| AFTER | `apr 0.69.0 (1f992a5e3)`, sha256 `d53a6d7038d0d06a7fe82ba5` | `Chat Template: Qwen3NoThink (thinking off)` | `Assistant: </think>⏎⏎2 + 2 = 4.` |

The leading `</think>` in the answer is the no-think scaffold echo — the same artefact the #3571
unit (2) receipt records for the 0.8B rung, pre-existing and not introduced here.

## What changed

- `apr chat` imports realizar's `detect_format_from_name` / `auto_detect_template` / `ChatMessage` /
  `ChatTemplateEngine` / `TemplateFormat`. One detector across all four verbs, by construction.
- The banner's **private copy** of the name match is deleted; it calls the one
  `template_format_name`, which gained the two variants the old enum could not express
  (`Qwen3NoThink`, `Zephyr`). Two copies of a match over an enum are how a new variant gets handled
  in one place and not the other — which is precisely this defect.
- `qwen3_moe` keeps plain ChatML (PMAT-181: trained without `<think>` blocks), pinned by a test so
  the unification does not sweep it up.

## Tests

Five new, in `commands::chat::realizar_chat::one_detector_tests`:
`chat_gives_qwen3_the_no_think_template_like_every_other_verb` (the defect, asserted both ways) ·
`qwen3_moe_keeps_plain_chatml` · `the_formats_chat_already_reported_are_unchanged` (no re-labelling) ·
`the_variants_the_old_enum_could_not_express_are_named` · `the_banner_and_the_session_agree_on_a_qwen3_file`
(the banner derives from the file stem, the session from `general.architecture`; if they disagree the
line a user reads describes a mode they are not in).

## What this does NOT do

It does not implement the think-budget guard and does not make the non-terminating thinking mode
terminate. It makes that mode **unreachable from the four verbs at their defaults**. The guard stays
the remedy for deliberate `--thinking on` under #3723 — a separate row, not in 0.69.1 — where at
greedy decoding the loop this issue measured will still occur, as pinned llama.cpp d1d3c3396 does on
2 of 3 golden prompts.
