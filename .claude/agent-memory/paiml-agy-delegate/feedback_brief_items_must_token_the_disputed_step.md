---
name: brief-items-must-token-the-disputed-step
description: Build the lane-reduce --brief items from the exact step a prior round disputed; an uncovered token there is the finding, and unanimous PASS does not erase it
metadata:
  type: feedback
---

When a quorum re-runs a question a previous round split on, the `lane-reduce.sh --brief
items.json` tokens must include **the identity evidence for the disputed step**, not just the
conclusion. Then `uncovered[]` tells you whether the lanes actually closed the gap or merely
agreed with the brief's framing.

**Why:** PMAT-1096 round 1 split 2-close / 1-keep-open because the move from same-shape Coder
parity evidence to the reporter's *exact* base-Instruct GGUF was **asserted, not measured**.
Round 2's brief said that gap was now measured and gave the file's sha256. All three lanes
returned PASS/close and `lane-reduce` reported `agreed=true`, `dissent=[]` — but the sha256
token `6a1a2eb6` appeared in **zero** lane outputs, and the apr build id `611989200` in zero.
`uncovered[]` caught it; the unanimous verdict did not. A quorum can agree on a conclusion
while no lane ever verified the one fact the previous round objected to.

**How to apply:** for each ask in the brief, pick a token only a lane that *looked* could echo
— a sha256 prefix, a build id, a byte count, a command's literal output line ("78 positions") —
not a word from the prompt itself. Then report `uncovered[]` verbatim in the receipt and say in
`open_questions` which asks are conclusion-agreed but identity-uncovered, so the orchestrator
re-runs that one command instead of trusting the consensus.

Related: [[capture-per-lane-exit-codes]]
