# #7462: nightly builds HEAD, not the last green main commit

Child 2 of #7017. Finding only. The fix changes `.github/workflows/nightly.yml`, which needs
the operator's yes, so no workflow file is modified here. Refs #7462.

## Where the selection happens (code reading, origin/main e994e617e)

| File | Line | What it does |
|------|------|--------------|
| `.github/workflows/nightly.yml` | 81-83 | `actions/checkout@v7` with `ref: main` and **no `fetch-depth`**, so the clone is depth 1 and sees only HEAD. |
| `.github/workflows/nightly.yml` | 102 | `NIGHTLY_SHA=$(git rev-parse HEAD)`. This is the selection point: the sha is HEAD, unconditionally. |
| `.github/workflows/nightly.yml` | 107 | `nightly_manifest.py gate --sha "$NIGHTLY_SHA"` judges only that sha, wrapped in `nightly_gate_retry.sh decide 7 1200`. |
| `scripts/nightly_manifest.py` | 130-155 | `gate()`. Returns `reused` (138-139), `red-ci` (152), `ci-pending`, or `build` (155). It never looks at any other commit. |
| `scripts/nightly_manifest.py` | 669-678 | `_cmd_gate`. One `--sha` in, one decision out. |
| `.github/workflows/nightly.yml` | 394 | `nightly_gate_retry.sh verdict` fails the run on any decision other than `built` or `reused`. |

No `rev-list`, `first-parent`, or walk-back exists in `nightly.yml`, `nightly_manifest.py`, or
`nightly_gate_retry.sh` (grep, 2026-11-28).

## Behaviour today

- HEAD green (`build`): built. Correct.
- HEAD pending (`ci-pending`): retried 7 times at 20 min, then red. Correct per #5053.
- HEAD red (`red-ci`): **no build at all**. The run ends red. The publish job still runs, so
  the manifest is rewritten with `decision: red-ci` at the HEAD sha, and the previous green assets
  stay in the release. The last green main commit is never built.
- Nothing green in the newest 20 commits: the same as today's red-ci. This must stay that way.

The self-test passes on the current tree: `python3 scripts/nightly_manifest.py --self-test`,
`PASS: 0 failing row(s)`, rc 0. Those cases encode the current behaviour; none covers a walk.

## Minimal fix (proposed, not applied)

Changes are in the workflow only, bash only, no new Python. The walk runs only from `red-ci`,
so a green or pending HEAD is unchanged.

```diff
       - name: Checkout
         uses: actions/checkout@v7
         with:
           ref: main
+          fetch-depth: 20   # the walk needs 20 commits; the default depth 1 sees only HEAD
@@ Decide step
-          NIGHTLY_SHA=$(git rev-parse HEAD)
-          echo "sha=$NIGHTLY_SHA" >> "$GITHUB_OUTPUT"
           bash scripts/nightly_fetch_manifest.sh prev-manifest.json
+          NIGHTLY_SHA=$(git rev-parse HEAD)
           d=$(bash scripts/nightly_gate_retry.sh decide 7 1200 -- \
                 python3 scripts/nightly_manifest.py gate --sha "$NIGHTLY_SHA" --prev prev-manifest.json)
+          # Only red-ci walks back. The first build|reused among the newest 20
+          # first-parent commits wins; if none, d stays red-ci at HEAD.
+          if [ "$d" = red-ci ]; then
+            for sha in $(git rev-list --first-parent -n 20 HEAD | tail -n +2); do
+              e=$(python3 scripts/nightly_manifest.py gate --sha "$sha" --prev prev-manifest.json 2>/dev/null) || continue
+              case "$e" in build|reused) NIGHTLY_SHA=$sha; d=$e; break ;; esac
+            done
+          fi
+          echo "sha=$NIGHTLY_SHA" >> "$GITHUB_OUTPUT"
           echo "decision=$d" >> "$GITHUB_OUTPUT"
```

Notes on the diff:
- `|| continue` keeps `set -e` from aborting on a GateError (rc 3) for one older commit.
- A `reused` at an older sha means the previous manifest already holds that commit green, so
  publish sees no new build. That is the intended outcome.
- Build and publish already check out `needs.gate.outputs.sha`, so an older sha needs no further
  change.

## Open questions for the operator

1. **Ruling needed.** `.github/workflows/nightly.yml` is a CI workflow; changing it needs the
   operator's yes.
2. **Criterion mismatch.** The issue says "`ci / gate` is success". `gate()` requires both
   `ci / gate` and `workspace-test` (`scripts/nightly_manifest.py:63`, `REQUIRED`). The walk above
   reuses `gate()`, so it uses both. Confirm which one is meant.
3. **Self-test.** The issue names `scripts/nightly_green_sha.sh --self-test` (5 cases). That file
   does not exist in this tree. The cases could go into `nightly_manifest.py self_test` or a new
   bash script; the latter needs a ruling.
4. **Cost.** The walk makes up to 19 extra check-run API calls, only on red-ci nights. Unauthenticated
   fallback is 60 per hour (`fetch_check_runs`, scripts/nightly_manifest.py:106-127).
5. **Handoff receipt.** The design receipt `handoff/release-cop/quorum-7017-c2-1659/AGY.out` cited in the
   issue is not in this worktree. It was not read.
6. **pmat item.** `pmat work status` does not list PMAT-7462, so this commit carries no `Pmat-Ticket`
   trailer. A pmat item is needed.
