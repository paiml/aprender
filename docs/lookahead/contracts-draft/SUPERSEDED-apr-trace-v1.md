# SUPERSEDED: draft apr-trace-v1 (la-0.71, 2026-10-03)

`contracts/apr-trace-v1.yaml` already exists on the minted ticket branches.
Content 04cc701e is on #4492 (a01/4492-obs05-ledger-checks), #4494 and #4572.
Variant 1c31e6c2 is on #4564 (76/4564-tr09-one-schema). That branch has an
id collision that must be resolved before either merges. This draft is NOT
a competing contract. FLOW-003: the ticket branch owns the id.

Mapping onto the existing FALSIFY-OBS-TRACE-* ids:

| draft | existing | note |
|---|---|---|
| TRACE-003 measured-not-run | OBS-TRACE-001 | same rule |
| TRACE-005 layer-sum-over-wall | OBS-TRACE-002 | same rule |

The rest are proposed ADDITIONS for the ticket owner. Each one comes with a
fixture and a mutant-verified checker under scripts/lookahead/ and fixtures/trace/:

- TRACE-004: Measured with zero tracer events
- TRACE-002 and TRACE-006: no double count of nested steps (layer_norm, attention, ffn), and the sum of top-level steps stays at or under wall
- TRACE-007: a sparse layer index is RED
- TRACE-008: a step outside the TraceStep enum is RED
- TRACE-009: timings whose provenance is not Measured
- TRACE-010 and TRACE-011: the §2.1 identity block, and gpu_proof on cuda
- TRACE-012: an empty receipt
- TRACE-013..015: the OBS-11 diff pair (identity match, before and after are not the same binary)
