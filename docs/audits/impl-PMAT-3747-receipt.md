# PMAT-3747 implementation receipt: a follow-up delta, not the feature

**Read this before judging scope.** The ticket's notes describe #3568 PR 1 as a whole: the
`TokenConstraint` hook and the llguidance engine behind it. **That feature is already folded** into
`release/0.69.1-batch-1` at `ded8a932a`, the base of this round. It was judged on its own diff:
`docs/audits/quorum-PMAT-3747.json`, AGREED 3/3 PASS (gemini-3.1-pro-high, gemini-3.6-flash-high,
gemini-3.1-pro-low) on head `c5047bd94`, base `52f43da71`, committed as `a1a1bc8cc`.

This round judges ONLY the delta `ded8a932a...cf8d0f309`: one file, test code, no feature code. A
reviewer should NOT expect the hook or the engine in this diff. They are in the base, and the base is
not what is being judged.

## What the delta is, and all it is

The cop, aprender-04, asked for it on the 0.69.1 critical path. guard_tree's complexity ratchet was RED
on the batch against main `a9502d992`:

`RED NEW crates/aprender-serve/src/constrain/tests.rs::generate_intent cyclomatic 13 cognitive 27`

The cognitive limit is 25. `generate_intent` is a fake model in this ticket's own case table, not
production code. The fix is a **pure refactor of test helpers with no behaviour change**:

- `greedy` is the argmax that both fake models already computed inline.
- `step` is mask → `greedy` → accept, returning `None` at EOS. Both fake models already ran this loop inline.
- `intent_score` is the intent model's scoring, moved out of the loop unchanged.
- `generate` and `generate_intent` now call these three. No test was added, removed or changed in what it asserts.

## Measured at cf8d0f309

| check | result |
|---|---|
| `git diff --stat ded8a932a cf8d0f309` | `crates/aprender-serve/src/constrain/tests.rs`, 72 insertions, 72 deletions, the only file |
| `cargo test -p aprender-serve --lib --features structured-output -- constrain:: constraint_vocab` | 24 passed, 0 failed: the same 24 rows as the feature's receipt |
| `bash scripts/check_complexity_ratchet.sh` | `generate_intent` is no longer listed. rc stays 1 **only** for `crates/aprender-contracts/src/lint/shapes_gate.rs::run_shapes_gate_with` (cyclomatic 13, cognitive 28). That is another constituent's (PMAT-3715), fixed in its own delta, and it is not in this diff |

## What this delta does NOT do

It changes no production code, no gate, no contract, and no test's assertion. It is `Refs #3747`: the
closing of #3747 rides on the feature's receipt, not on this one.
