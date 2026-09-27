# FLOW-003 QM-01 — queue-model inputs, 7-day window (aprender#4513)

Window `2026-09-20T07:09:01Z` → `2026-09-27T07:09:01Z` (`raw/window.txt`).

```bash
scripts/release/queue_inputs.sh self-test                               # planted fixtures; empty window = RED
scripts/release/queue_inputs.sh compute evidence/flow-003/qm01-2026-09-27/raw   # re-derives queue-inputs.json
```

`raw/` is the receipt: `compute` reads only those files. `fetch` produced them read-only (runs, PR list,
queue timelines, nextest `TRY n` lines of the 156 merge_group + release-PR CI runs). `ident` produced
`raw/ident.tsv` mechanically (rule in the script header), and a person may overwrite rows.

| input | value | n | note |
|---|---|---|---|
| T | 44.1 min | 50 | median of passed queue entries |
| C | 43.6 min | 95 | median release-PR CI cycle |
| q | 0.140 | 57 | backed out of q' = 0.123 with f (Q4); raw q' = 0.153, 2 of 9 failed entries were flakes |
| φ | 0.019 | 156 | 3 runs: `falsify_h3_attention_correctness` ×2, `g9_roofline_efficiency` ×1 |
| λ | 1.34 /h | 225 | PRs created. λ' (queue entries) = 0.375 /h |
| f | 1.0 | **1** | low n |
| F | 4.0 min | **1** | low n |
| ρ_mq | 0.0014 min | **1** | low n |
| ρ_rel | 0.126 min | 83 | |
| R | 34.4 min | **2** | low n |
| ident_rate | **0.287** | 87 | 25 of the 87 red release cycles name a failing test |

`verdict: GREEN`, 0 `[U]`. Four inputs have n < 5 (`low_n`). The window had only 10 ejections, and 5 of
them never re-entered.

## Findings for the model

- **S-4 trips: ident_rate 0.29 < 0.9.** 62 of 87 red release cycles name no failing test. They are
  infra, guard, lint or timeout failures, so bisection by test cannot find the culprit PR. This is an
  upper bound, since a named test in a k-PR batch can still span several PRs.
- **87 of 95 release-PR CI cycles are red (92%).** Of those, 33 are `release/0.69.1-batch-2`.
- **λ ≫ λ':** 225 PRs were created but there were only 63 queue entries. Most work reaches main through
  batches, not the queue.
- `pr-review-quorum` fails on almost every merge_group entry, yet the PRs merge. It is not a required
  queue check, so the queue outcome is `CI` alone.

## QM-08 — q per service class, Prop 12 (aprender#4519)

The classes are FLOW-003 v1.1 §5.1, built from `raw/pr_meta.jsonl` (228 PRs) and `raw/readset.txt`
(the paths a build/test/doctest reads, taken from tree `8b1efb8dec`). **d**: every file is `.md`,
the file list is complete, and no file hits the read set. **a**: a fork PR carrying `maintainer-attested`
(`ATTEST_LABEL`). **x**: everything else. Prop 12 uses q* = 30/(0.5·240) = 0.25 from the [A] inputs.
The verdict is FAIL if q_c ≥ q*, and PASS if the Wilson 95% upper bound, backed out with f, is below q*.
Every other case, including n = 0, is NOT-DECIDED.

| class | π | π_eff | n | q_c | q_c 95% upper | T̄_c | Prop 12 |
|---|---|---|---|---|---|---|---|
| d docs | 0 | 0 | 0 | — | — | — | NOT-DECIDED |
| a attested fork | 0 | 0 | 0 | — | — | — | NOT-DECIDED |
| x full | 1 | 1 | 57 | 0.140 | 0.303 | 44.1 min | NOT-DECIDED |

The self-test plants one docs entry that fails (q_c = 0.5), and it prints **FAIL**. It also plants 21 clean
docs entries, which print **PASS**.

- **Class d is empty over this window.** Four PRs were all-`.md`: #3566 (CLAUDE.md), #3570
  (docs/specifications/…), #3572 (docs/roadmaps/…) and #4119 (README.md). Tests or guards read every one of those
  paths. `toyota_principles_tests.rs:56`, `falsify.rs:394` and `oracle_indexing.rs:277` read
  `docs/specifications` as a whole directory. `check_roadmap_completion_is_cited.sh` reads `docs/roadmaps`, and
  tests read README.md and CLAUDE.md. So §5.1 puts all four in x. A docs lane only has traffic if those readers
  are narrowed, or if §5.1 exempts them.
- **Class a is empty.** The repo has no attestation label yet.
- **x is not decided.** With n = 57, its upper bound of 0.30 is above q* = 0.25. It can stay in the full lane.
- Measure-only: no gate or ruleset changed. The pv binding (`queue-inputs-v1`, §11.3 entity
  `docs/receipts/flow-003/queue-inputs.json`) waits on QM-00.

## ρ_HOL and per-merged-PR rows

`derived.rho_hol` is Prop 11 (B = 1, r = 0): λ′·q̄′·T̄ = 0.00625/min × 0.1228 × 44.1 min = **0.034**. The split and
blind values agree because every entry is class x. This is a lower bound (Theorem 8 tightness), about 18× below
the ρ_HOL > 0.6 stop line. So the head of line is not the bottleneck at today's λ′.

`derived.merged_pr_rows` has one row per PR that left the queue merged in the window: class, mq_wait_min (first add
→ merged), entries, first_try, ejects. There are 39 rows, all class x. On the first try, 29 passed, 3 failed
(ejected), 6 were removed `manual` and 1 left with a `merge_conflict`. There were 3 ejects in total. Median wait
is 54.3 min and the max is 443.7 min.

## Fold size k (Lemma 1, Thm 1, Cor 3) at the measured inputs

`derived.fold` uses q 0.14, f 1.0 (**n = 1**), C 43.6, F 4.0, φ 0.019 and ρ_rel 0.126 min. The self-test checks
the closed forms against the four §6.1 oracle rows. A mutant that drops `r·ρ` fails two of those rows.

| k | EM | E[T_fold], best r | per PR | bisect upper (ident 0.29) | per PR |
|---|---|---|---|---|---|
| 1 | 0.140 | 50.3 min (r=1) | 50.3 | 50.3 | 50.3 |
| 2 | 0.260 | 56.0 | 28.0 | 64.1 | 32.0 |
| 4 | 0.453 | 65.2 | 16.3 | 93.3 | 23.3 |
| 8 | 0.701 | 77.0 | 9.6 | 142.3 | 17.8 |

- Folding still pays per PR, even at the bisection upper bound. That bound charges ⌈log₂k⌉ extra cycles to each
  defect cycle that names no failing test, which is 71% of them.
- r* = 1 at every k, but the gain from r = 0 to r = 1 is only about 0.8 min. φ* ≈ 0.0004–0.002 because ρ_rel is
  about 8 s (nextest retries a single test, not the whole cycle).
- f = 1 rests on one fix, so EM is optimistic. EM equals 1−(1−q)^k exactly when f = 1.
