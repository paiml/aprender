# impl receipt — PMAT-1092

| | |
|---|---|
| ticket | PMAT-1092 · kind=code · `orch=opus` (model-gate: `decision=admit basis=file`) |
| branch | `PMAT-1092-release-process-spec` · PR #3056 · epic #3058 |
| base | `origin/main` @ `c04eda87d` |
| discover.json sha256 | `repo_root=/mnt/nvme-raid0/agent-wt/rel-process` · `gate_cmd=make gate` · `gate_cmd_fallback=false` · `required_check=ci / gate,workspace-test` |
| verdict | **PARTIAL(escalate)** — the review is complete and landed; the draft's nine RD decisions are Noah's and are not decided here |

## Why a worktree

`/home/noah/src/aprender`'s `main` shares **no common ancestor** with `origin/main`
(`git merge-base main origin/main` is empty; 702 ahead / 7 behind). All work ran in a clean
worktree off `origin/main`.

## Phases and routing

| # | phase | route (`route.sh`, verbatim) | executor |
|---|---|---|---|
| 1 | teamwork grill of the draft | `route=agy-quorum w=1.08 basis=quota.json@20h` | delegate, lane=teamwork, width 1 |
| 2 | ground truth + review | `route=self w=0.00` | self |
| 3–6 | AD-04 quorum ×3 + corrections | `route=agy-quorum w=1.08` | delegate, lane=quorum, width 3 ×3 |

`K̂=5 basis=docs/audits/impl-estimates.jsonl:L20-L29` was for a 3-phase code change and badly
under-estimated a spec grill with four adversarial rounds. Actual ≫ K̂.

## Dispatch ledger

| dispatch | lane | width | conversations | outcome |
|---|---|---|---|---|
| ph1.delegate | teamwork | 1 | `55b4205c` | `do-not-implement-as-written`, 10 findings |
| ph3.delegate | quorum | 3 | `0e863919`, `2ff50417`, `ccd407ea` | **3 × FAIL** on this review; maxTurns, resumed once (§6.2) |
| ph5.delegate | quorum | 3 | `af3ee618`, `595f4a64`, `4d4ac0ef` | **3 × FAIL**, unanimous on RD-9 |
| round 4 | quorum | 3 | — | operator stopped; lanes killed at 06:36 |

Lanes ran in disposable copy-trees (`PMAT-1092-quorum`, `-quorum2`); `repo_root` verified
byte-identical before and after each round (diff md5 `8d091c762e9dda0712614607e2a0f713`).

## Verification — claimed vs re-run

Every lane finding was re-executed. The record that matters is what the quorum overturned
**in this review**:

| round | objection | disposition |
|---|---|---|
| 1 | `cuobjdump` claimed present on all four hosts | **UPHELD** — the row was `command -v cuobjdump` on `noah-Lambda-Vector` labelled `intel`. `ssh intel` → ABSENT. Overturn withdrawn |
| 1 | merge-queue basis mixed `cancelled` runs | **UPHELD** — re-derived over `conclusion=success`; lanes then disagreed on population, so the budget is now `[U]` |
| 1 | GT-1 "20 of 21 non-new" | **UPHELD** — 19 of 20 |
| 1 | RD-2/6/7/8 dispositions | **UPHELD** — rephrased |
| 1 | draft staged into `docs/specifications/` | **UPHELD** — moved to `docs/audits/release-process-review/` |
| 2 | RD-3 ("sound", stale 3–98 citation), RD-4 ("Unchanged") | **UPHELD** — rewritten |
| 3 | RD-9 ("Closed by Appendix A") | **UPHELD** — plus three more of the same shape found by sweeping rather than waiting |
| 1 | 7B may be unprovisioned | **NOT upheld** — present on both dogfood hosts (4 683 073 536 B) |

## Jidoka

`e7244c3ab`'s commit message claimed three edits its script never applied (the `assert` fired,
`git commit` ran anyway in the same shell). Corrected in `f246fdafb` with the mechanism named.
Same shape as reading `$?` through a pipe: a claim written from intent, not from what ran.

## Gaps

- `gate_cmd` (`make gate`) **NotRun** — docs-only change; CI's `ci / gate` is the instrument
  and was pending at hand-off.
- `pr-review-quorum` **FAIL** — the base-owned gate wants a signed receipt at
  `evidence/pr-review/3056/` under `.github/pr-review.pub`. Not produced; separate skill,
  needs the private key.
- AD-04 quorum: **0/3 PASS**. Rounds 1–3 all FAIL; each objection applied; round 4 stopped by
  the operator. Auto-merge **not armed** and must not be until a round returns 3 PASS.
- `pv_lane` NotRun — no contract changed by this PR.

## Filed

- **paiml/paiml-implement#51** — the delegate's documented `out_dir` is keyed by phase number
  alone. This run's `ph4/` already held three `PASS` verdicts from a different session two days
  earlier on ticket PMAT-246. Reducing the documented path would have armed auto-merge on a
  diff no lane had read.
- **paiml/aprender#3058** — the work breakdown this review produces.
