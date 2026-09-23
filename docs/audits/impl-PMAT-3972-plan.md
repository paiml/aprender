# PMAT-3972 plan: ONT-4c5 capability cells (aprender #3972 + #4047)

Spec row: ONT-4c5, paiml/infra `docs/specifications/paiml-ontology.md` v4.12, merged as 948ae923 (infra#950); the row text is identical to the PMAT-921-v412 draft (diffed).
Base: `origin/chore/0.69.1-merge-back` 6db770d2f (#4046). Kind: code. K̂=64 (basis impl-estimates.jsonl:L49-L52).

## Footprint (M)
- `crates/aprender-contracts/src/ontology/receipts.rs`: ONT-4c1's join. The `expected` host set in `resolve()` IS D's host axis.
- `crates/aprender-contracts/src/ontology/verdict.rs`: `from_label` maps DEFER/MANUAL/NO-VERDICT to `Unknown(NotRun)` (read-only use).
- `crates/aprender-contracts/src/lint/shapes_gate.rs`: report, plant, arming, `GateExtra::Shapes` JSON.
- `contracts/`: new `ont-capability-cells-v1.yaml`, `shapes.ttl` (generated?), `lint-baseline.json` `armed_shapes` (+`capability-cells`; the ratchet only grows).
- `tests/fixtures/ont/`: `capability-cells-plant.yaml` + fixture trees: clean, defer, manual, noverdict, missing, fail-admitted, empty-domain.

## Design
1. **One domain definition.** Factor `expected_hosts(rung, all_hosts)` and `row_is_green(rung,row)` out of `resolve()`/`witness_rows()` unchanged. `resolve()` calls them, so ladder-green behaviour is byte-identical. D = { `<rung.id>@<host>` : rung.required, host ∈ expected_hosts }.
2. **Extractor emits only what it finds.** For each witness row (sha matches; a row without sha or with a wrong hex is NOT a cell), emit `model:capabilityCell "<host>=<Pass|Fail|NotRun>"` on the rung node. Pass = the ladder-green predicate holds on that row. Fail = a witness exists and the predicate does not hold. NotRun = the row carries a fleet label (`verdict`/`label` key) that `Verdict::from_label` maps to `Unknown(NotRun)`. An unrecognised label is refused by name (it is never guessed and never counts as Pass). Pass on any witness for (rung,host) wins over Fail across receipt versions: the same "∃ witness row" quantifier ladder-green uses.
3. **Validator computes the difference.** In the shapes gate, `not_run = D \ {cells with Pass|Fail}`. Every id in `not_run` becomes `model:notRunCell "<host>"` on the rung's RequiredModel node before validation. Shape `capability-cells`: targetClass `model:RequiredModel`, `model:notRunCell maxCount 0` (inside §3.6's subset). An absent cell is NEVER folded into Fail (it is NotRun), and a Fail cell is admitted (this row admits Fail; ladder-green is what refuses it).
4. **Arming rule.** `capability-cells` joins `armed_shapes` in `lint-baseline.json`, so a notRunCell is a violation of an armed shape: Fail, exit 1. Exit 2 stays decline-only. **|D| = 0 → a new `ShapesOutcome` decline** (`decline: capability-cells domain is empty`), exit 2, never Pass.
5. **Plant.** `tests/fixtures/ont/capability-cells-plant.yaml`: one required rung, hosts [plant-host], no receipt. It is evaluated in memory every run through the same function. It must yield exactly `not_run == ["<rung>@plant-host"]`, else `pc_shapes["capability-cells"]="not-fired"` and the gate declines (PositiveControlFailed).
6. **JSON.** `GateExtra::Shapes` gains `capability_cells: {domain: [sorted ids], not_run: [sorted ids]}` and `pc_shapes: {"capability-cells": "fired"|"not-fired"}`. The existing `pc_shape` stays.
7. **Contract** `contracts/ont-capability-cells-v1.yaml`: the shape, equations (D, not_run), invariants in Σ glyphs, falsification_tests = the case table below. `pv validate` clean.

## Phases and acceptance commands
- P1 refactor (no behaviour change): `cargo test -p aprender-contracts --lib ontology::receipts` green and ladder fixtures unchanged: `cargo test -p aprender-contracts --lib lint::shapes_gate`.
- P2 cells + domain + not_run (receipts.rs), RED first: `cargo test -p aprender-contracts --lib ontology::receipts::tests::capability_cells`.
- P3 shape + plant + JSON + |D|=0 decline (shapes_gate.rs), fixtures: `cargo test -p aprender-contracts --lib lint::shapes_gate::tests::capability_cells`.
- P4 contract + arming + ttl: `pv validate contracts/ont-capability-cells-v1.yaml && cargo test -p aprender-contracts-cli --test ont4b_shapes_gate`.
- P5 the row's own probe on this tree (first-green) + the 4 row mutations as a case table (each RED): `bash scripts/check_ont_4c5_probe.sh` (runs the probe jq against a pinned pv, then applies each mutant in a temp copy and expects RED).
- P6 gate: `make gate` (discover gate_cmd), fmt, clippy -D warnings on the crate.

## Case table (row's mutations + extras)
| case | expect |
|---|---|
| fold missing into Fail | plant passes → pc_shapes not-fired → RED |
| drop capability-cells from armed_shapes | probe `.armed_shapes` check RED |
| flip one required rung to `required: false` | B ⊄ D → probe RED |
| label a required cell DEFER / MANUAL / NO-VERDICT | not_run names it → Fail exit 1 |
| a required rung@host with no receipt | not_run names it → exit 1 |
| a witness row not green | Fail cell, admitted, not in not_run |
| row without sha256 | not a cell → not_run |
| |D| = 0 | exit 2 + `decline:` line, never Pass |
| unknown label spelling | refused by name, not Pass |

## Residuals (named)
- D takes ONT-4c1's host axis unchanged (the row's own residual).
- "∃ witness row" spans receipt versions: an old green row with a matching sha keeps a cell Pass even if the latest receipt Fails. This is inherited from ladder-green's quantifier on purpose (one definition). Changing it is a separate row.
- infra#950 merged (948ae923); row text diffed identical.
