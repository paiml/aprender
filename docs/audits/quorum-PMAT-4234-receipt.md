# Quorum receipt — PMAT-4234 (Qwen3.5 continuous batching in apr serve)

Author: claude-opus-5-5. Base: origin/rc/0.69.3-rc.2. Judged head: 513e81bc1 (`git diff base...HEAD`, 27 files, 165 KB).
Shape: sonnet-5 + 1 agy gemini-3.1-pro-high + haiku-4-5 (operator 2026-09-24). No lane is in the author's model.

## Round 1 — AGREED 3/3 PASS
| lane | model | verdict |
|------|-------|---------|
| agy | gemini-3.1-pro-high (measured) | PASS, 0 findings — `quorum-PMAT-4234.json` |
| claude | claude-sonnet-5 | PASS, 1 advisory finding |
| claude | claude-haiku-4-5 | PASS, 0 findings |

The Claude lanes got a read-only command whitelist and read the full judged diff from a file. Brief sha256 prefix: c88a72dc082e.

## Sonnet's advisory finding (does not flip its PASS)
- `qwen35_decode_batcher.rs:92`: `relock` recovers a poisoned `Mutex<GpuModel>` via `PoisonError::into_inner`.
  - If a step panics while holding it, sibling slots keep decoding on that model without a signal.
  - Reached only when `APR_QWEN35_SERVE_SLOTS` > 1, and the default is 1.
  - **Real, and not fixed in this row.** Recorded as an open item: latch poison and refuse further decodes, instead of recovering silently.
