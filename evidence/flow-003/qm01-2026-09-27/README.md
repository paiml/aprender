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
