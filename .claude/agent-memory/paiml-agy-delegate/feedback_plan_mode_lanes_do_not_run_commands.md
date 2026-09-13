---
name: plan-mode-lanes-do-not-run-commands
description: --mode plan lanes finish in num_turns=1 and never execute the shell commands a brief asks for, yet still label findings grounding=measured — always check num_turns before trusting "measured"
metadata:
  type: feedback
---

A `--mode plan --sandbox` agy lane returns `num_turns: 1`. One turn is the final
structured answer, so the lane ran **no** tool calls. If the brief told each lane to
"run these commands and report the exit codes", the receipt must say the exit codes
were **not** reported, and every `grounding: "measured"` in that lane is really
`asserted`.

**Why:** on PMAT-1059 (width 3) all three lanes were told to run
`check_hardcoded_paths.sh --self-test`, `check_guards_are_wired.sh` and
`. scripts/pmat_bin.sh`. Every lane came back `num_turns=1`, no rc anywhere in
`response`, and two lanes still tagged findings `measured` — including a claim that a
script "runs successfully in the tree". Reporting those as measured would have handed
the orchestrator a fabricated execution receipt. `agy --mode plan` is read-only
*planning*; it is not an execution mode.

**How to apply:** after collecting lanes, `jq -r '.num_turns'` on every file. `1` means
no execution happened — downgrade the lane's `measured` groundings in the receipt and
add an `open_questions` entry naming the commands that were never run. If a brief
genuinely needs commands executed, either run them yourself before launching and paste
the output into the prompt, or ask the orchestrator for `--mode goal` (which does take
turns). See [[lanes-need-a-cd-wrapper-script]] and
[[read-response-when-findings-are-labels]].
