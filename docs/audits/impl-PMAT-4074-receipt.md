---
status: partial
ticket: PMAT-4074
row: ONT-5
issue: 4074
model: "rmedia-82 (claude-opus-5-5) P1-P2; claude-opus-5-5 (infra ONT-001 session) contract, wiring, gates; review quorum: Claude lanes sonnet-5 x2 + haiku-4-5, degraded: same-family"
---
# impl-PMAT-4074 — ONT-5 · relation consistency: Horn encoding, witness, pv-sat (#4074; ONT-001 §3.5, §5 ONT-5, R-1..R-3)

## What lands
- `provable_contracts::ontology::witness`: the Horn `ClauseSet` (live ids are units; `depends_on`/`refines` are `¬A∨B`; `contradicts` is `¬A∨¬B`; `supersedes` withdraws a unit), `census_id_set_sha256`, `relations_sha256`, `Witness`, the linear `check`, and `pc_checker` over the shipped `fixtures/unsat-core-corrupt.json`.
- `lint::consistency_gate`: gate `ont-consistency` — PV-ONT-022 (inconsistent, core named), PV-ONT-023 (witness does not check); declines `NoCheckable` / `WitnessStale` / `PositiveControlFailed` exit 2; malformed Σ exit 3.
- `pv-sat` (bin target; the reasoner is private to it — R-1, F-7) with `pc_reasoner` (the plant A⇒B, B⇒C, contradicts(A,C)); it rewrites nothing when the witness on disk is fresh, so `reasoner_git_sha` does not churn.
- `contracts/ont-consistency-v1.yaml` (kind `pattern`, L2 only, 6 obligations, FALSIFY-ONT5-001..007); `contracts/witness/9c24a906….json`; `make contracts` runs pv-sat then re-checks with `pv lint --gate ont-consistency` and fails when `contracts/witness/` is dirty; `scripts/pv_bin.sh` builds and exports `PV_SAT` beside `PV`.
- CI: `ci/explicit-test-commands.d/510-…ont5-consistency-gate.cmd`, `520-…pv-sat-bin.cmd`. Ordinals 460–500 are claimed by open PRs (#4046, #4136, #4137, #4180, #4230 …) and the checker refuses a shared ordinal. `scripts/tree_reader_tests.txt` +3.

## Probe (§5 ONT-5), this tree
`pv lint contracts/ --gate ont-consistency --format json` → `verdict Pass`, `checkable_n 22`, `units 1765`, `unencoded_edges 0`, `pc_checker fired`, `witness.kind model`, `witness.pc_reasoner fired`, `witness.stale false`.

## Named residuals
- **Gate NOT armed.** `ont-consistency` is absent from `armed_gates` (11 armed); arming is a separate PR.
- The pc_checker fixture is crate-local (`crates/aprender-contracts/fixtures/`), because the published crate excludes `tests/`.
- Σ has no `implies` role, so §3.5's `implies` is not encoded; `unencoded_edges` lists any role without a clause, and it is 0 here.
- PVL EV-9 is not on main; nothing here depends on it.
- The ONT-6 `mod_tests` gate count moved to 17 (bookkeeping only).
- The contract is `kind: pattern`, not `kernel`: as `kernel`, PROVABILITY-001 demanded Kani harnesses, and none are claimed.

## Verification (this tree)
| check | result |
|---|---|
| `cargo test -p aprender-contracts --lib ontology::witness` / `lint::consistency_gate` | 9/9, 8/8 |
| `cargo test -p aprender-contracts-cli --bin pv-sat` / `--test ont5_consistency_gate` | 4/4, 5/5 |
| `cargo clippy -p aprender-contracts -p aprender-contracts-cli --all-targets -D warnings` · `cargo fmt --check` · `cargo deny check advisories` | clean · 0 · ok |
| `pv lint contracts/` | PASS, 11/11 armed |
| check_ont_ratchet · complexity_ratchet · tree_reader_tests · explicit_test_commands · readme_claims · `pv extract --check` · roadmap-aggregate-check · readme_sync | all PASS |
