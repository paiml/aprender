# SUPERSEDED: draft apr-perf-ledger-v1 (la-0.71, 2026-10-03)

`contracts/apr-perf-ledger-v1.yaml` already exists on the minted ticket branches:
- content bde14e59 on #4492 (a01/4492-obs05-ledger-checks) and #4572
- variant a304d19f on #4494

The two variants must converge before either one merges. This draft is NOT a competing contract. FLOW-003: the ticket branch owns the id.

| draft | existing FALSIFY-OBS-PERF-* | note |
|---|---|---|
| LEDGER-002 empty | 002 | same |
| LEDGER-003 missing host | 001 | same |
| LEDGER-004 version/sha | 003 | same |
| LEDGER-005 gpu_proof | 004 | same |
| LEDGER-006 identity ratio | 005 | same |

Proposed ADDITIONS for the ticket owner (fixture + mutant-verified checker in
scripts/lookahead/perf_ledger_check.py, fixtures/ledger/):
- LEDGER-007: the ratio must equal the quotient of the arms' medians
- LEDGER-008: the comparator commit must equal scripts/llama_pin.toml. The existing contract states this as a REQUIRE but has no falsifier for it.
- LEDGER-009: prompt_tokens must be 512±8
- LEDGER-010: the §2.4 seed must be 42
- LEDGER-011: each arm must carry the sha of its apr-raw-samples-v1 document (#4551)
- LEDGER-012: stat n must equal prompts × reps
- LEDGER-013: a zero median is RED even when the ratio is consistent
