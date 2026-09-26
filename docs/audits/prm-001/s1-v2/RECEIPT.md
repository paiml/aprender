# PRM-S1 v2 — set a4910a65, per-tag receipts (#4447, #4354)

Set: `set-a4910a65.jsonl` (review-replay-v2, seed 4354, 50 items per stratum × 2k/8k/16k/32k = 200).
`sha256sum set-a4910a65.jsonl` = `a4910a65ea53e520841b164876d34dead653771226f8a740e6d3408b08b30632`.
Every row below carries that `set_sha`. The diffs themselves are not here: they are the corpus
items, addressed by `diff_sha256`.

Cell `gx10-cuda` (GB10), GGUF sha256 `00fe7986…ef11a4`. Competitor: llama.cpp `d1d3c3396`, run in
the same session on the same GGUF with identical prompt ids (`prompt_ids_sha` `f3c20136…`).
The rows hold timings, token counts, peak RSS and the parsed verdict state only. There are no
prompts or completions; transcripts stay private.

| apr build | rows (apr/llama) | wall p50 s | wall p95 s [95% CI] | ttft ratio | decode tps ratio |
|---|---|---|---|---|---|
| `96eb79f51` (0.69.3) | 200/200 | 26.61 | 69.36 [59.49, 82.56] | n/a: apr untimed before SRV-TIM-001 | n/a |
| `33b5e5b24` (0.70.0 dev, SRV-TIM-001; apr sha256 `38313994…68ce`) | 200/200 | 26.98 | 68.83 [60.91, 85.64] | 4.40× | 0.216 (≈4.6× slower) |

`33b5e5b24` is the build of record. The `speed-block.json` in each directory is the §9 block,
produced by `replay summary --rows rows-both --voter LANE=FILE`.

## Queue budget (voter p95)

`33b5e5b24/voters.txt` is `replay voters` over `evidence/pr-review/` at main
`8cf336c60a2b5071b855af99c99def618fb1f41b`. Of the 48 quorum receipts there, 23 carry a
`duration_seconds`, and all 23 are `antigravity` lanes; the Claude lanes record none.
Budget = the largest lane p95 = **601.6 s (antigravity)**. apr's wall p95 is 68.8 s, so it
is **within budget** at both builds. A Claude-lane latency, once receipts carry one, can only
change which lane sets the budget.
