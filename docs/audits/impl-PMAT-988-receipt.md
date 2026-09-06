---
status: partial
partial_reason: "this PR is not yet merged on the required check; it must land after G-4 (its ci.yml step runs scripts/render_dag.py) — flip to complete with the DAG status write-back after merge"
ticket: PMAT-988
row: SPEC-1.6
issue: 2903
epic: 2873
model: "orchestrator claude-fable-5-1 (direct; docs row, no worker dispatched)"
tokens_used: "orchestrator [U] (not instrumented)"
wall_clock_s: "[U] (not instrumented; the turn count before context compaction was not preserved — see Estimates)"
---
# impl-PMAT-988 — SPEC-1.6 · spec v1.6: the 14 S0-ledger defects, one reading of "credited first", the §5.0 tables rendered from the DAG (#2903; driver STEP B row 3)

## Identity
ticket PMAT-988 · kind docs (the driver routes docs rows through the same rail: RED test first, contract discipline where a contract applies, every A_i re-run) · branch `agent/SPEC-1.6` (worktree, claim held) · base `132ccda56` · quorum `--quorum never` (review-only row) · K̂ = 184 (`basis=docs/audits/impl-estimates.jsonl:L1-L7`).

## What lands
- `docs/specifications/PP-066-release-spec.md` v1.6 — 38 anchored edits: the 14 numbered S0-ledger defects (`docs/audits/pp-066-s0-ledger.md` §"Spec defects"), the S0-19 / S0-12 / S0-3 consequences (R-6 and §8 "no **third** signing scheme"; the MIN-05 Metal correction withdrawn; B-A1 anchored on D-3), §4 "credited first" given exactly one reading (C0 is a *precondition of crediting*, not of *working*; the temporal-precedence reading is named as the falsified one — the plan quorum's dissent, A2), §4 andon I-26, G-4's checker home (in-repo; pmat 3.37.0 has no `--rule obligation-dag`), Track P sourced from #2870, Track I carded in the DAG, the four expiry moves recorded under the §12 amendment rule (P-0.6 → 09-26, T-0 → 10-03, P-1.1 → 10-03, I-17 → 10-16), changelog row 1.6, and the §5.0 block between the `dag:table` markers rendered by `scripts/render_dag.py` from `docs/specifications/pp-066-dag.yaml` (90 rows, byte-identical under `--check`).
- `tests/spec/pp066_v16_defects.sh` — the RED-first test of this docs row: 22 rows, one per corrected sentence; `--v15-red` proves the table is RED on the v1.5 text (`git show 42be1560b:…`). Reverting one correction turns exactly its row RED (the registered mutation).
- `ci.yml` `guard-runner-labels`: the defect table and `python3 scripts/render_dag.py --check`, after `spec_conformance.sh --selftest`.
- `docs/roadmaps/roadmap.yaml`: PMAT-988 labels `kind:docs`, `pp-066` (additive; the entry was minted by #2981 and this branch only carries the label edit the rail requires).
- No new contract: the row is prose over a spec; the contract that governs the rendered block is G-4's `contracts/apr-obligation-dag-v1.yaml`, and the DAG carries this row's `contract: null` by design.

## Verification (orchestrator, every command re-run at HEAD 126e25685)
| check | result |
|---|---|
| `bash tests/spec/pp066_v16_defects.sh` | rc 0, `22/22 rows` |
| `bash tests/spec/pp066_v16_defects.sh --v15-red` | rc 0, "the v1.5 text (42be1560b) is RED under this table (rc=1)" |
| `python3 scripts/render_dag.py --check --dag docs/specifications/pp-066-dag.yaml --spec docs/specifications/PP-066-release-spec.md` (G-4's script; the DAG copied in untracked) | rc 0, "byte-identical … (90 rows)" |
| `bash scripts/check_no_claim_literals.sh` | rc 0 (the rendered B-M3 row cites `evidence/reports/0.66-parity-report-provenance.json`; see Jidoka) |
| `bash scripts/check_perf_claims_cite_receipts.sh` | rc 0 |
| `bash scripts/spec_conformance.sh` | rc 0 |
| DAG A for this row (`tests/spec/pp066_v16_defects.sh && check_no_claim_literals.sh && check_perf_claims_cite_receipts.sh && render_dag.py --check`) | all rc 0 |

## Mutation (RED, then restored GREEN)
Reverting the §4 sentence "C0 is a *precondition of crediting*, not of *working*" turns row 15a RED (`FAIL row 15 (defect 15) §4 credited first has exactly one reading`); the v1.5 text as a whole fails 22/22 (`--v15-red`). Restored → 22/22.

## Jidoka
- The claim-literal guard went RED on the rendered §5.0 block: B-M3's title carries the #2826 f32 wgpu decode figure (UNRECEIPTED [C]; provenance only: `evidence/reports/0.66-parity-report-provenance.json`). Root cause, two layers: the DAG title cited no evidence path (fixed on the plan branch, e94b2effb: the title now names its provenance file), and `render_dag.py` truncated titles to 140 characters, cutting the citation off the rendered line (fixed in G-4 at e6687cb88: titles are never truncated). A guard that passes on the source and fails on the render is a renderer defect, not a docs defect.
- The renderer fix's first form put a trailing comment on the tuple line and swallowed three cells (8 columns under an 11-column header); caught by counting columns in the render before re-verifying. Recorded so the next reader checks column counts, not only `--check`.

## Gaps
- Merge order: after G-4 (the ci.yml step invokes `scripts/render_dag.py`, which G-4 adds) and after #2981 (the DAG file, which `--check` reads); until both land the two new steps are RED by design on this branch.
- `tokens_used` / `wall_clock_s` are [U]: not instrumented, and the pre-compaction turn count was not preserved.
- Receipt for this PR: advisory, not produced (driver A1).

## Estimates
K̂ 184 (`basis=docs/audits/impl-estimates.jsonl:L1-L7 median 46 × 4 phases`); actual turns: P_1–P_3 [U] (lost at context compaction), P_4 (renderer fix, re-render, re-verification, this receipt) = 5 orchestrator bash calls. Rows appended to `docs/audits/impl-estimates.jsonl` carry `actual` only where measured.

## Verdict
PENDING-MERGE (`status: partial`).
