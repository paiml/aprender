# Quorum receipt — PMAT-4316 (#4316, qwen35 CUDA resident residual)

Author: claude-opus-5-5. Base: origin/main @ aa7c6ef03. Shape: sonnet-5 + 1 agy gemini-3.1-pro-high + haiku-4-5 (operator 2026-09-24).

## Round 1 — AGREED 3/3 PASS on head bba6b0924
Every lane got the same brief: 16294 B, sha256 217e98df…76b121de. The agy artifact records prompt_bytes 16294 and diff_sha256 2b6d7a02….

| lane | model | verdict |
|------|-------|---------|
| agy | gemini-3.1-pro-high (measured) | PASS. Its 2 findings confirm claims and carry `fix: none`. Artifact: `quorum-PMAT-4316.json` |
| claude | claude-sonnet-5 | PASS, 0 findings. It grep-verified that every layer launch and `sync_stream` use `self.stream` |
| claude | claude-haiku-4-5 | PASS, 0 findings |

Earlier attempts did not count. The first agy round (head 635bd6761) was aborted when a concurrent `--brief-only` run shared its lanes dir. The Claude lanes were refused by kind-gate until `kind:code` was added, in bba6b0924.

## GPU evidence
Pending. See `impl-PMAT-4316-receipt.md` §Gates. The `qwen35_cuda*` run is queued in gpu-q behind the rc.2 acceptance run.
