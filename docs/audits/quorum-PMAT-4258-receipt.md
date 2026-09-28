# Quorum receipt — batch #4322, delta PMAT-4258 (#4258 stale-Q8 carry + DP4A parity guard)

Author: claude-opus-5-5. Base: 86498a9db, the last head of this branch that a quorum or the GPU had judged; the
PMAT-4316 part is covered by `quorum-PMAT-4316-receipt.md`. Shape: sonnet-5 + 1 agy gemini-3.1-pro-high + haiku-4-5
(operator 2026-09-24).

Why a delta round: the full-batch brief was 52 KB. Everything below 86498a9db was already judged 3/3, and the carried
#4302 code was agreed 3/3 on 4482ab98b. This round judges the cherry-picks as landed here, plus the new 555a6774d.

## Round 1 — AGREED 3/3 PASS on head 36f1ea707
Every lane got the same brief: 33300 B, sha256 aa728804…, diff_sha256 7f8fff34f272….

| lane | model | verdict |
|------|-------|---------|
| agy | gemini-3.1-pro-high (measured) | PASS. Its 3 findings cite the acceptance criteria as met. Artifact: `quorum-PMAT-4258.json` |
| claude | claude-sonnet-5 | PASS. It traced every qwen35 GEMV input. The residual that the host upload writes feeds only rmsnorm/residual_add, never a GEMV, so the (ptr,len) key's host-upload limit does not apply. The dense float path never calls `ensure_q8_activation`. 1 finding: stale test name in `evidence/parity/thresholds.yaml:95` |
| claude | claude-haiku-4-5 | PASS, 0 findings. It checked that the receipt matches the GPU logs |

## After the round
The one post-round commit touches only a comment: it annotates the stale test name in `evidence/parity/thresholds.yaml`,
as sonnet's finding asked. It changes no code or thresholds, and the YAML still parses.

## GPU evidence
GREEN on 555a6774d: `qwen35_cuda*` passed 17/17 and the q8 staleness tests passed 2/2. See
`impl-PMAT-4258-receipt.md` §Gates.
