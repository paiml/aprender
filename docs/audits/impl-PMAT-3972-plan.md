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
2. **Extractor emits only what it finds, measured AT THE CURRENT RELEASE (grill round 1, 2/3 do-not-implement: fixed).** V* = the greatest `version` over all tracked receipts (semver; a version that does not parse is Unsupported, exit 3, never guessed). A cell `<rung>@<host>` is measured ONLY by a witness row (sha matches; a row with no sha or a wrong hex is not a cell) in the (V*, host) receipt. Older versions are never consulted for cells: the cross-version "∃ witness row" that ladder-green uses would let a 0.68.2 green row hide a 0.69.1 DEFER or a missing 0.69.1 row. The grill measured this on this tree (all 8 required rungs share a sha across 0.68.1/0.68.2/0.69.1). ladder-green itself is unchanged. Per cell, precedence is fixed: **NotRun label > Pass/Fail predicate.** A row whose `verdict` or `label` key maps through `Verdict::from_label` to `Unknown(NotRun)` is NotRun whatever its `green` says. A label that `from_label` does not know is refused by name (a finding, exit 1), never Pass. Otherwise Pass = the ladder-green predicate (`green ∧ capability_passed ∧ backends_ok`) and Fail = its negation. The extractor emits `model:capabilityCell "<host>=<Pass|Fail|NotRun>"` for the cells it finds.
3. **Validator computes the difference.** In the shapes gate, `not_run = D \ {cells with Pass|Fail}`. Every id in `not_run` becomes `model:notRunCell "<host>"` on the rung's RequiredModel node before validation. Shape `capability-cells`: targetClass `model:RequiredModel`, `model:notRunCell maxCount 0` (inside §3.6's subset). An absent cell is NEVER folded into Fail (it is NotRun), and a Fail cell is admitted (this row admits Fail; ladder-green is what refuses it).
4. **Arming rule.** `capability-cells` joins `armed_shapes` in `lint-baseline.json`, so a notRunCell is a violation of an armed shape: Fail, exit 1. Exit 2 stays decline-only. **|D| = 0 → a new `ShapesOutcome` decline** (`decline: capability-cells domain is empty`), exit 2, never Pass.
5. **Plant, through the SHAPE and not only the helper (grill: a plant that checks only the Rust set difference would stay "fired" with the shape deleted).** `tests/fixtures/ont/capability-cells-plant.yaml` (one required rung, hosts [plant-host], no receipt) runs in memory every gate run through the SAME cells/domain functions. It must yield exactly `not_run == ["<rung>@plant-host"]`. The resulting `model:notRunCell` node is then inserted into the planted graph `validate_with_plant` builds, and it must draw a violation of the ARMED shape `capability-cells` on `model:notRunCell`. Both must hold for `pc_shapes["capability-cells"]="fired"`. Either failing gives "not-fired", and the gate declines with PositiveControlFailed (exit 2).
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
| DEFER on the V* row + a green row for the same cell in an older version | not_run names it, exit 1 (round-1 hole) |
| V* receipt missing a required rung's row, older version green | not_run names it, exit 1 |
| a host with older receipts but no V* receipt | every required cell on it in not_run |
| row with `green: true` AND `verdict: DEFER` | NotRun (label precedence) |
| row with `verdict: PASS` and `label: MANUAL` | NotRun (any NotRun label wins) |
| optional rung (`required: false`), no receipt | not in D, not in not_run |
| plant shape deleted from contracts / capability-cells unarmed | pc_shapes not-fired → decline exit 2 (never Pass) |
| receipt version that does not parse as semver | Unsupported exit 3 |
| JSON: `capability_cells.domain`/`not_run` sorted, ids `<rung>@<host>`, `pc_shapes` object | asserted by the gate test |

## Residuals (named)
- D takes ONT-4c1's host axis unchanged (the row's own residual).
- RETIRED by round 1: cells no longer span receipt versions (Design 2). ladder-green keeps its own cross-version quantifier; that is ONT-4c1's, and this row does not change it.
- The host axis is derived from the tracked receipts: the spec row's own named residual ("one definition, one blind spot"). V* narrows it. A host with ANY tracked receipt stays in D, and its cells are NotRun unless the host has a V* receipt. A host that has never had a receipt is not in D. B ⊆ D in the probe guards the 14 named cells.
- infra#950 merged (948ae923); row text diffed identical.
