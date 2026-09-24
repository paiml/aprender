# PMAT-4039 — implementation receipt

- **Ticket:** PMAT-4039 (GH #4039, lever f of epic #4033), kind:code. Branch `fix/4039-sweep-cell-order`.
- **Base:** `origin/chore/0.69.1-merge-back` (#4046). Retarget to `main` once #4046 lands.
- **Verdict:** DONE up to the quorum receipt: 3/3 PASS on `266c7a194` (round 7, agy gemini ×3). Not armed.

## What changed
`scripts/model_ladder.sh` used to measure the ladder's rungs in YAML order and then the inventory models. It now
builds **one plan** over both and measures it in the order `scripts/lib/ladder_order.py` prints:
1. CRUX-certified sha first;
2. then bytes, descending;
3. then kind, then id.
The certified set is the keys of the prompt certification's two admission fields. `--certification` picks the
certification file; the default is this version's file, otherwise the newest by version. With neither, the order is
largest first. The `order:` line on each run names the certification file used, or says there was none.

Both sections had to be merged into one plan: one of 0.69.1's three certified models is inventory-only
(`b252c5610a42`). Both loop bodies are kept verbatim inside the one loop.

## Order is a schedule, never a verdict
- **The planner.** `ladder_order.py` exits 2 before any cell runs unless its output is an exact multiset permutation
  of the plan. It also exits 2 on a malformed line. The selftest has 23 cases, including CLI-level must-REDs.
- **The ladder check.** `check_ladder_order_verdicts.sh` (aprender-36's falsifier) runs the **shipped** ladder under
  a fake `apr`, in opposite orders, and requires:
  - identical rows with a clean fake;
  - differing rows with a leaky fake, a detached orphan that outlives its cell (#4055's shape) — the positive control;
  - each file's own sha and inventory record, `red=5`, `executed=3` across N/A, ABSENT and sha-mismatch rungs;
  - a symlinked rung sized through the link;
  - a rung's file sitting in the inventory dir, measured once, with a substring-colliding name;
  - `--only` keeping every inventory record;
  - a failing planner declining both a real run and a dry run;
  - `--certification` with no value declining;
  - two dry runs through the default certification, where right file, wrong file and no file give distinct orders, and
    no measuring verb runs.
  It takes about 25 s. Its `--self-test` makes the comparison blind, and that turns it RED.
- **Mutants.** Seven quorum rounds each turned up surviving mutants. Every one was then planted and shown RED; the
  commit messages list them all.

## Saving: NOT MEASURED
Total sweep time on one sequential card is **unchanged by construction** (0.0 min). The gain is
**time-to-certified-verdict**. It cannot be measured until receipts carry `t_start`/`t_end` (#4051). "CRUX can start
early" holds only **across hosts**: the ladder and CRUX never share a card. Largest-first cuts the tail only with
host sharding, a sweep that is cut short, or #4034's lane concurrency, which is opt-in.

## Verification (orchestrator re-runs)
| Check | Result |
|---|---|
| `bash scripts/check_ladder_order_verdicts.sh` | exit 0, log sha256 `fc31c51e…74eb8a` |
| `… --self-test` | exit 0 (the planted blind comparison turns it RED) |
| `check_ladder_*` (all 6 prior guards, including only_selection's self-test) | rc 0 |
| `check_bashrs_gate.sh`, `check_shell_lint_ratchet.sh`, roadmap guards | rc 0 |
| `check_model_ladder.sh` | the same 8 release-evidence FAILs as the base, none new |

## Quorum (`docs/audits/quorum-PMAT-4039.json`)
- **Rounds 1–5:** Claude Code `claude-sonnet-5` ×3, degraded: same-family. Every round was 2 PASS / 1 FAIL, and
  every FAIL was measured and fixed.
- **Round 6:** the Sonnet lanes hit the usage limit and returned no verdict. The agy round came back 2/1; its dissent
  was a mislabelled case, since fixed.
- **Round 7:** agy gemini ×3, **3/3 PASS**, no isolation violation. Lane 1 wrote a scratch file and deleted it in
  its own sandboxed clone, against the no-write contract. Its clone came back byte-identical; the vote is counted
  and the violation disclosed here.
