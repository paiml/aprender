# Upstream Sync (fork ⇄ paiml/aprender)

How to merge `upstream/main` into this fork without losing the fork's SetFit/forecast work. It is a
prerequisite for anything that uses upstream's Qwen3.5 path.

## Requirements

From idea `llm-decision-classifier`:

- **Qwen3.5 support comes from upstream, not a fork-local re-port.** Sync `paiml/aprender` first, then
  extend its `Qwen35Model` rather than writing a second implementation (OPS-03).

## How to Build It

The spike merged `upstream/main` @ `49fe19c28` (280 commits ahead) into the fork at `de64a914f`.
The merge is committed locally as `895c654de` on `spike/016-upstream-sync` (worktree
`../aprender-016-upstream-sync`). It is **not pushed**. The spike-017/020 patches sit on top of it
as `12976da0a` and `32103ba83`.

```bash
git worktree add ../aprender-016-upstream-sync -b spike/016-upstream-sync <fork-head>
cd ../aprender-016-upstream-sync && git merge upstream/main          # 34 conflicts
python3 <spike>/resolve_ids.py      # contract YAMLs
python3 <spike>/resolve_rest.py     # the remaining 13 files, one rule per hunk
CARGO_TARGET_DIR=<main>/target cargo check --workspace --exclude aprender-profile
<spike>/test016.sh                  # six suites, rc captured per suite (not through a pipe)
```

### The five resolution rules

34 conflicts: 16 add/add, 15 both-modified, and 3 modify/delete of `.pv/` indexes.

1. **The 16 add/add are our own SetFit, seen twice.** Upstream carries SetFit as squashed PR #2618
   (2026-08-26). The fork has 1,076 lines of later SetFit work: the general-BERT loader and the
   1 GiB artifact bound. Since then, upstream touched those files only through two sweeps:
   obligation `id:` lines on every contract (`1edcb5521`) and `--features setfit` on contract test
   commands. `resolve_ids.py` takes upstream's hunk **only when it equals ours plus those
   insertions**. Everything else takes ours.
2. **Take upstream's `--features setfit` sweep.** It fixed a real vacancy of ours:
   `multinomial-head-v1.yaml` ran `cargo test -p aprender-core --lib falsify_…` without the feature,
   and `mod setfit` is `#[cfg(feature = "setfit")]`, so those commands compiled the module out and
   ran **zero tests**.
3. **GEMM `blis/parallel.rs`: take upstream's.** Upstream fixed the aarch64 silent-zeros bug
   independently: `shared_b_path_available` returns false on every non-x86_64 target.
4. **`.pv/` indexes: accept upstream's deletion.**
5. **`CLAUDE.md`: take ours; README counts: take upstream's.** Re-run `readme_contract` afterwards.

### Three semantic conflicts git cannot see (only `cargo check` surfaces them)

- Duplicate keys in `[workspace.dependencies]`: both sides added `trybuild`,
  `unicode-normalization`, `tokenizers` and `aprender-rand`. Delete one copy.
- Fork-only crates pin path deps at `0.63.0` against a `0.69.0` workspace. Bump the pins.
- Upstream added an `AppState.effective` field that our SetFit-only `Default` constructor lacks.

## What to Avoid

- **Don't attribute a red test to the merge without a control.** One test is red:
  `setfit::artifact::determinism::the_fixture_artifact_hash_matches_the_committed_golden`, expected
  `13e5…`, got `cc17…`. A detached control worktree at the **un-merged** fork HEAD produces the same
  `cc17…`. The golden went stale on the fork, most likely at `4983ffafe` (the general-BERT loader
  rewrote 46 lines of `artifact.rs`), and no gate caught it. Regenerate it on the fork **before**
  the real sync PR, so the sync PR carries no unrelated red.
- **Don't claim "upstream still has our bug" from memory.** The early aarch64 GEMM claim was wrong;
  upstream had already fixed it. Read upstream's hunk first.
- **`aprender-profile` has a `compile_error!` on non-Linux.** That is upstream's own guard, not merge
  damage. Pass `--exclude aprender-profile` on macOS.
- **Don't read suite status through a pipe** (`test016.sh` captures `rc` per suite).

## Constraints

- Upstream is now at **0.69/0.70**, six minor versions ahead of the fork's 0.63, with Qwen3.5 CPU
  and CUDA forward in `aprender-serve`.
- Suites on the merged tree (release, M4 Pro): forecast 240 pass; core setfit 250 pass + 1
  (pre-existing); train setfit 434; compute 3332; serve qwen35 41; mcp-setfit 8.
- **Not done**: upstream's `CLAUDE.md` edits (ours was taken wholesale), `readme_contract`,
  `make tier*`, clippy and the CI job set on the merged tree. All of these are required before the
  sync is a PR.
- Merging to `main` goes through a PR (branch protection). Pushing the sync branch is a checkpoint.

## Origin

Synthesized from spike 016 (VALIDATED).
Source files: `sources/016-upstream-sync-qwen35/` (README, `resolve_ids.py`, `resolve_rest.py`,
`test016.sh`, `conflicts.txt`, RUN-OUTPUT).
