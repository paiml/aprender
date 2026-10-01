---
spike: 016
idea: llm-decision-classifier
name: upstream-sync-qwen35
type: standard
validates: "Given the fork is 280 commits behind paiml/aprender, when upstream/main is merged in a worktree, then conflicts are enumerated, the workspace builds, and the forecast and SetFit suites stay green"
verdict: VALIDATED
related: [015]
tags: [upstream, merge, qwen3.5, setfit, build]
---

# Spike 016: Upstream sync for Qwen3.5

## What This Validates
Given the fork (`gsd/phase-2-contract-gate` @ `de64a914f`) is 280 commits behind `upstream/main` @ `49fe19c28`,
when upstream is merged in a separate worktree, then the conflicts are enumerable, the workspace builds, and our
forecast and SetFit suites plus upstream's Qwen3.5 tests are green.

## How to Run
```bash
git worktree add ../aprender-016-upstream-sync -b spike/016-upstream-sync <fork-head>
cd ../aprender-016-upstream-sync && git merge upstream/main          # 34 conflicts
python3 <spike>/resolve_ids.py      # contract YAMLs: take upstream when it is ours + `- id:` / `--features setfit`
python3 <spike>/resolve_rest.py     # the 13 remaining files, one rule per hunk (see script)
CARGO_TARGET_DIR=<main>/target cargo check --workspace --exclude aprender-profile
<spike>/test016.sh                  # six suites, rc captured per suite
```
The merge itself is committed LOCALLY as `895c654de` on `spike/016-upstream-sync` — not pushed.

## Investigation Trail
1. **Merge: 34 conflicts** (16 add/add, 15 both-modified, 3 modify/delete of `.pv/` indexes that upstream dropped).
2. **The 16 add/add are our own SetFit, seen twice.** Upstream carries SetFit as the squashed PR #2618
   (2026-08-26, authored by Guy). Our branch has 1,076 lines of later SetFit work (general-BERT loader, 1 GiB
   artifact bound). Upstream touched those files since only via two sweeps: obligation `id:` lines on every
   contract (`1edcb5521`) and `--features setfit` on contract test commands. `resolve_ids.py` takes upstream's hunk
   only when it equals ours plus those insertions; the rest take ours.
3. **Upstream's feature-flag sweep fixed a real vacancy of ours**: our `multinomial-head-v1.yaml` test commands ran
   `cargo test -p aprender-core --lib falsify_…` without `--features setfit`, and `mod setfit` is
   `#[cfg(feature = "setfit")]` on both sides — those commands compiled the module out and ran zero tests.
4. **GEMM: an early "upstream still has our aarch64 silent-zeros bug" claim was wrong.** Upstream fixed it
   independently: `shared_b_path_available` returns false on every non-x86_64 target. Upstream's hunk was taken.
5. **Semantic conflicts git could not see** (surfaced by `cargo check`): three identical duplicate keys in
   `[workspace.dependencies]` (both sides added `trybuild`, `unicode-normalization`, `tokenizers`, `aprender-rand`),
   fork-only crates pinning path deps at `0.63.0` against a `0.69.0` workspace, and upstream's new
   `AppState.effective` field missing from our SetFit-only `Default` constructor. Three one-line fixes.
6. **`aprender-profile` has a `compile_error!` on non-Linux** — it is upstream's own guard, not merge damage;
   macOS checks need `--exclude aprender-profile`.
7. **One red test, and it is pre-existing:** `setfit::artifact::determinism::the_fixture_artifact_hash_matches_the_committed_golden`
   expects `13e5…` and gets `cc17…`. A detached control worktree at the un-merged fork HEAD produces the **same
   `cc17…`**, so the golden went stale on our branch (most likely `4983ffafe`, the general-BERT loader, which rewrote
   46 lines of `artifact.rs`) and no gate caught it. Not caused by the merge.

## Results

| Suite (release, M4 Pro) | rc | passed | failed | ignored |
|---|---|---|---|---|
| `aprender-forecast` (all targets) | 0 | 240 | 0 | 15 |
| `aprender-core --features setfit --lib setfit` | 101 | 250 | **1 (pre-existing stale golden)** | 0 |
| `aprender-train --features setfit --lib setfit` | 0 | 434 | 0 | 3 |
| `aprender-compute --lib` | 0 | 3332 | 0 | 4 |
| `aprender-serve --lib qwen35` | 0 | 41 | 0 | 0 |
| `aprender-mcp-setfit` | 0 | 8 | 0 | 0 |

**Verdict: VALIDATED ✓.** The upstream sync is tractable: 34 conflicts resolve by five rules plus three one-line
semantic fixes, the workspace type-checks, and every suite is green except one golden that is already red on the
fork. Upstream is now at **0.69/0.70** (six minor versions ahead), with Qwen3.5 CPU+CUDA forward in `aprender-serve`.

**Signal for 017**: upstream's `Qwen35Model` (`gguf/inference/forward/forward_qwen35.rs`) is GGUF-only,
token-at-a-time, logits-only; its parity bar is llama.cpp argmax agreement with a 0.25-logit near-tie allowance —
far looser than the ~1e-5 hidden-state parity Kev's pointer head needs. The fused matmul already dispatches
F32/F16/BF16 weights, so an unquantized GGUF of the LoRA-merged base should load on existing kernels.

**Not done (out of spike scope)**: `CLAUDE.md` took ours wholesale, so upstream's CLAUDE.md edits are not carried;
README counts took upstream's and `readme_contract` was not run; `make tier*`, clippy and the CI job set were not run
on the merged tree; the stale golden needs regenerating on the fork before the real sync PR.
