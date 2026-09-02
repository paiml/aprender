# Parity matrix-run ledger

**PP-9: a cell, once run, is spent; it may not be re-run to green.** That rule is only
enforceable against a written record of what has been spent. This file is that record.

Governing spec: [`docs/specifications/performance-parity-llama.cpp.md`](../../docs/specifications/performance-parity-llama.cpp.md).
Append-only. One row per *cell run* — a **(host, workload, model, quantization, commit)** tuple
driven across its bands. **The commit is part of the key**, and PP-9 now says so too: an earlier
draft keyed the cell without it while this file's own closing paragraph said a later commit
"starts a new row", so the two halves of one file disagreed and PP-9 described a state with no
legal move. A row is added when the run is committed, never edited afterwards; a run
found invalid is superseded by a new row whose `status` explains it, and the original stays.

## Spent cells

| # | date | host | class · accelerator | model · quant | workload | bands | N | commit | receipts | status |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2026-09-01 | `lambda` | cuda · RTX 4090 | qwen2.5-coder-7b-instruct · q4_k_m | W1 | c=1,4,8,16 | 3 | `745fa8588` | [`perf-gate-001-w1-lambda/`](../perf-gate-001-w1-lambda/) | **SPENT — subject lane invalid.** Ratios withdrawn by §2.1: the subject binary was built with continuous batching compiled out. Retained as the counter-measurement of record; the comparator lane and the noise floor in §12.1a survive. |
| 2 | 2026-09-01 | `gx10` | cuda · GB10 (aarch64) | qwen2.5-coder-7b-instruct · q4_k_m | W1 | c=1,4,8,16 | 3 | `745fa8588` | [`perf-gate-001-w1-gx10/`](../perf-gate-001-w1-gx10/) | **SPENT — subject lane invalid** (same build defect). Additionally flags `SUSPECT_DISPATCH` under PP-23: 6.203 tok/s decode is 10.6% of this device's roofline (#2846). c=8 carries a 21.17% MDE traced to a device-wide stall (#2833) — the failure PP-19 exists to prevent. |

**Neither row is a parity measurement.** Both are spent, both are withdrawn as ratios, and
under PP-9 neither host may be re-run at `745fa8588` to obtain a different answer. The next
run on either host requires a commit that fixes the §2.1 build defect, and it starts a new row.

## What a row must carry

A run may be appended only with, at minimum: the receipt directory, `receipt.commit`, the
signature verification result (PP-21), the host and its `compute_class` **as reported by the
server, not as declared by the harness** (PP-16), the comparator pin and its expiry state
(PP-20), the per-band `max_in_flight` on both lanes (PP-24), and `roofline_tok_per_sec` with
its ratio against **decode** (PP-23). A row missing any of these is not a spent cell — it is an
unmeasured one, and it belongs in §12, not here.

**None of those five producers exists in the tree today, so this criterion is not yet
satisfiable — which would make PP-9 unenforceable exactly where it binds.** A review found
`roofline_tok_per_sec` present only in prose, no `signature` field on the receipt (the release
gate already prints `FAIL ArmC-sig UNSIGNED`), no `expiry` in `llama_pin.toml`, no perf
workflow and therefore no concurrency group, and no client sha distinct from the server's. All
five are now owed with owners and dates in §12 rather than asserted as if they existed. **Until
they land, a row carries what it has and names what it lacks**; the two rows above are entered
that way and are marked as such. A criterion nothing can meet is not a stricter rule, it is a
rule that is never applied.

**Receipts carry no timestamp.** Both rows above are dated by their evidence commit rather
than by the run, because `provenance` records `binary_path`, `binary_sha256`, `resolution`,
`compute_class`, `host`, `accelerator`, `model`, `quantization` and `feature_set` — and no
clock. Dating a run by when someone committed it is not dating the run. The producer owes a
UTC start timestamp in `provenance`; until it lands, the `date` column above is an upper bound.
