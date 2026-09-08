---
name: goal-mode-lanes-do-run-tools
description: --mode goal lanes DO read files and run commands even though num_turns=1, so use goal (not plan) whenever a brief tells lanes to read a file list; and probe --model eligibility before fan-out because agy 503s on some models
metadata:
  type: feedback
---

When a brief's prompt says "read these files / follow this call chain", compose the lane
with `--mode goal --schema <the pinned quorum schema> --sandbox`, not `--mode plan`.

**Why:** on PMAT-1070 (L0-1b, width 3, goal mode) all three lanes reported `num_turns: 1`
yet had `usage.input_tokens` of 690k+ and their `response` narrated real reads
(`types.rs:195`, `parallel_k.rs:288`, `fused_matmul_into.rs:405`) that were correct against
the tree. So `num_turns == 1` does NOT mean "no tool calls" in goal mode — it means one
assistant turn. The `num_turns=1 ⟹ zero tools` rule in
[[plan-mode-lanes-do-not-run-commands]] is a fact about **plan** mode only. Use
`usage.input_tokens` (a few 10k = it read nothing; several 100k = it read the tree) plus
whether the prose cites line numbers that check out, and verify a sample citation yourself.

`agy-lane.sh` cannot express `--model`, and its arg parser rejects unknown flags. For a
"three different model families" brief, take the exact command from
`agy-lane.sh --mode goal --prompt X --schema <s> --dry-run` and re-emit it with `--model`
added; keep `--sandbox` and `--dangerously-skip-permissions`. Say in the receipt that the
calling form was reproduced rather than invoked through the wrapper.

**How to apply:** also probe eligibility first — `agy --model <m> -p="Reply OK."` costs 3s.
`claude-opus-4-6-thinking` returned `status:ERROR, "Eligibility check failed: UNAVAILABLE
(code 503)"` with `duration_seconds: 0` and `num_turns: 0`, i.e. the lane died instantly and
would have silently cost a family. `claude-sonnet-4-6` was fine. `agy models` lists what
exists, not what you are entitled to run. Record the substituted model id in the receipt.

**Addendum (PMAT-1073, width 1, `--mode goal` + the pinned quorum schema).** The
combination works and the mapping survives: `agy-lane.sh`'s goal preamble hard-codes its
own vocabulary ("report outcome (achieved | partial | blocked)"), but a prompt that opens
with an OUTPUT CONTRACT block saying "this overrides any preamble above about
achieved/partial/blocked", gives the enum mapping, and demands a `DESIGN_VERDICT=` prefix
got a clean `verdict: "do-not-implement-as-written"` plus
`summary: "DESIGN_VERDICT=do-not-implement-as-written | …"`. Put the contract block FIRST
in the prompt, before the task, so it lands adjacent to the preamble it is overriding.

Measured on agy 1.1.27: `num_turns: 1`, `duration_seconds: 527`,
`usage.input_tokens: 277200`, `output_tokens: 33475`, `status: SUCCESS`, rc 0, 0-byte
`.err` — a lane that read a 28-file diff and ~10 source files across two crates. Use the
input-token count, not `num_turns`, to decide whether a goal lane actually read the tree.
