# impl receipt — PMAT-1092

| | |
|---|---|
| ticket | PMAT-1092 · kind=code · `orch=opus` (model-gate: `decision=admit basis=file`) |
| branch | `PMAT-1092-release-process-spec` · PR #3056 · epic #3058 |
| base | `origin/main` @ `c04eda87d` |
| discover.json sha256 | `repo_root=/mnt/nvme-raid0/agent-wt/rel-process` · `gate_cmd=make gate` · `gate_cmd_fallback=false` · `required_check=ci / gate,workspace-test` |
| verdict | **PARTIAL(escalate)** — ten quorum rounds, never 3 PASS — the review is complete and landed; the draft's nine RD decisions are Noah's and are not decided here |

## Why a worktree

`/home/noah/src/aprender`'s `main` shares **no common ancestor** with `origin/main`
(`git merge-base main origin/main` is empty; 702 ahead / 7 behind). All work ran in a clean
worktree off `origin/main`.

## Phases and routing

| # | phase | route (`route.sh`, verbatim) | executor |
|---|---|---|---|
| 1 | teamwork grill of the draft | `route=agy-quorum w=1.08 basis=quota.json@20h` | delegate, lane=teamwork, width 1 |
| 2 | ground truth + review | `route=self w=0.00` | self |
| 3–n | AD-04 quorum + corrections, one phase per round | `route=agy-quorum w=1.08` | delegate, lane=quorum, width 3 per round — **see the dispatch ledger below for the rounds actually run**; this row deliberately names no count, because every count restated in prose here went stale the next round |

`K̂=5 basis=docs/audits/impl-estimates.jsonl:L20-L29` was for a 3-phase code change and badly
under-estimated a spec grill with an adversarial round per phase — see the ledger for how many.
Actual ≫ K̂.

## Dispatch ledger

| dispatch | lane | width | outcome |
|---|---|---|---|
| ph1 | teamwork | 1 | `do-not-implement-as-written`, 10 findings on the draft |
| round 1 | quorum | 3 | **3 × FAIL** on this review — the mislabelled `cuobjdump` probe, a cancelled-run basis, GT-1 arithmetic, four RD dispositions, the draft's placement |
| round 2 | quorum | 3 | **3 × FAIL**, unanimous — RD-3 and RD-4 still disposed |
| round 3 | quorum | 3 | **3 × FAIL**, unanimous — RD-9 "Closed by Appendix A" |
| round 4 | quorum | 3 | **3 × FAIL**, unanimous — F8 decided RD-5; Appendix A's "now close as" |
| round 5 | quorum | 3 | **2 × FAIL / 1 PASS** — the Method table's own arithmetic; §9 in the verdict table; S0-Y4/Y5 |
| round 6 | quorum | 3 | **3 × FAIL** — §9 carve-out missing from the prose, §4 vs RD-6, RD-1's imperative, S0-N1 per target, a stale ordinal |
| round 7 | quorum | 3 | **3 × FAIL** — F11 contradicted S0-N1; F8's `[V]` label list; F11's surviving imperative; §12; S0-M1 |
| round 8 | quorum | 3 | **2 × FAIL / 1 PASS** — first audit of the findings as a set; **three of my measurements did not reproduce** (AD107M/`nvidia-smi`, `mod.rs:268-279`, F10's rolling window) |
| round 9 | quorum | 3 | **3 × FAIL** — **GT-1 held in full**; GT-6 misattributed C13/R-5 and misquoted "build on" |
| round 10 | quorum | 3 | **3 × FAIL** — GT-3's missing `gx10` label, GT-5's truncated titles, **this receipt's stale tally**, and **the epic still carrying corrected text** |

An earlier revision of this receipt recorded "0/3 PASS" over three rounds with round 4
stopped. Ten rounds ran; rounds 5 and 8 each returned one PASS. The quorum has still never
returned three PASS, which is the fact that matters, but the tally was wrong and the ledger
stopped at round 4 — caught by round 10, which was the first to audit this file at all.

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
- AD-04 quorum: **never 3 PASS in ten rounds.** Rounds 1–4, 6, 7, 9, 10 were 3 × FAIL; rounds
  5 and 8 returned 2 × FAIL / 1 PASS. Every objection was applied rather than argued — forty
  rows over eleven passes, tabulated in the review's Method section. Auto-merge **not armed**
  and must not be until a round returns 3 PASS.
- The findings themselves have now been machine-audited as a set: **GT-1 held in full** (all 21
  paths, `find` included). The defects rounds 8–10 found were in the review's *evidence and
  self-account* — five measurements of mine that did not reproduce — not in the findings about
  the draft, every one of which has been re-run by a lane and stood.
- `pv_lane` NotRun — no contract changed by this PR.

## Filed

- **paiml/paiml-implement#51** — the delegate's documented `out_dir` is keyed by phase number
  alone. This run's `ph4/` already held three `PASS` verdicts from a different session two days
  earlier on ticket PMAT-246. Reducing the documented path would have armed auto-merge on a
  diff no lane had read.
- **paiml/aprender#3058** — the work breakdown this review produces.
