---
status: partial
partial_reason: "BLOCKED upstream: pmat 3.38.0 `work cot derive` emits hollow obligations (statement/hypothesis empty, and pv reads `property`); pv refuses them (SCHEMA-005) and loosening pv would be a contract exemption. paiml/paiml-mcp-agent-toolkit#1200 filed 2026-09-06. Nothing shipped."
ticket: PMAT-977
row: C0-3
issue: 2892
epic: 2873
model: "orchestrator claude-fable-5-1, direct (no worker)"
tokens_used: "[U] (not instrumented)"
wall_clock_s: "[U] (not instrumented); ~07:30Z–07:55Z on 2026-09-06"
---
# impl-PMAT-977 — C0-3 · the ten `contracts/work/GH-663..672.cot.yaml` derivations (CB-1658), reconciled with PVI-3.1 (#2892)

## Identity
ticket PMAT-977 · kind code (data) · branch `agent/C0-3` (worktree, claim held; **unpushed**) · base `027ed889d` · quorum review-only (none dispatched: nothing to review) · K̂ = 3 (`basis=first-run[U]`) · owner spec-owner.

## What was measured (every command run by the orchestrator, 2026-09-06)
| step | command | result |
|---|---|---|
| source | `.pmat-work/GH-663..672/{contract.json,cot-digest.json}` copied from the main checkout into the (gitignored) worktree `.pmat-work/` | 10 tickets, each `contract.json` v5.0 with 2 `chain_of_thought` steps carrying `falsifiable_claim`, no `statement`/`hypothesis` |
| derive | `pmat work cot derive GH-<n> --mode cli` ×10 (pmat 3.38.0 = crates.io latest) | 10 × `contracts/work/GH-<n>.cot.yaml`; every re-recorded digest **identical** to the main checkout's (GH-663 `0a020d65…`) |
| A1 | `pmat comply check --mode cli \| grep CB-1658` (worktree with `.pmat-work`) | `✓ CB-1658: 10 derived ticket(s): one obligation + one claim per step, verbatim` |
| A1 mutation | remove `GH-672.cot.yaml` → same command → restore | `✗ CB-1658: 1 derivation completeness violation(s): GH-672: cot-digest.json exists but ./contracts/work/GH-672.cot.yaml is missing` |
| content probe | append `# probe`, then set `statement: "probe"` in GH-663.cot.yaml → same command | ✓ both times — **CB-1658 checks existence only**, never the derived content |
| chain | `pmat work cot check GH-663` | `✓ Chain integrity holds` |
| pv | `pv validate contracts/work/GH-663.cot.yaml` | `[ERROR] SCHEMA-005: proof_obligations[0].property must not be empty` ×2 (every file: 20 errors) |
| pv lint | `pv lint contracts --strict-test-binding` with / without the ten files | **FAIL 0/10 gates** (Gate 1 validate ✗, 1770 contracts, 20 errors) / PASS |
| repo guard | `bash scripts/check_contract_test_binding.sh` | rc 1: `VACUOUS: strict-test-binding gate was SKIPPED (contract validation failed)` |
| README | count 1811 → 1821 on all three lines; `check_readme_claims.sh` rc 0 | (in the local commit; not shipped) |

## Root cause (five whys, stops at the owning tool)
1. pv refuses the files → 2. each `proof_obligations[i]` has no `property` → 3. pmat writes `statement`, and writes it **empty** → 4. `pmat work cot derive` reads `statement`/`hypothesis` from the steps, and the v5.0 `contract.json` carries `falsifiable_claim` instead → 5. owning tool: pmat (`work cot derive`, and CB-1658 which passes a hollow file). Filed: **paiml/paiml-mcp-agent-toolkit#1200**.

## Why nothing shipped
- Hand-filling the generated files is hand-typing a proof summary into a file whose header says "do not edit by hand" (F-26 class); a re-derive would erase it and CB-1658 would not notice either way.
- Teaching pv to accept an empty obligation is a contract exemption (operator ruling 2026-08-21: none).
- Moving the files out of `contracts/work` (PVI-3.1's direction) is not available: pmat hardcodes `./contracts/work/<ID>.cot.yaml` for CB-1658.
So C0-3 is **BLOCKED** on pmat#1200. The branch keeps the ten generated files + the README/roadmap edits locally so the re-run after the fix is `pmat work cot derive` ×10 → `pv lint` → push.

## Jidoka
- pmat#1200 (upstream): derive emits `""`, CB-1658 passes the hollow file.
- The plan's phase (2) "reconcile with PVI-3.1" cannot precede the pmat fix; PVI-3.1 stays a 0.67 row.

## Dispatch ledger
direct only; slots 0/3; denials 0.

## Estimates
K̂ 3 (`basis=first-run[U]`); actual 6 orchestrator bash calls (`basis=this receipt`).

## Verdict
STOPPED(pv validate) — blocked upstream on pmat#1200; `status: partial`.
