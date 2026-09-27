# PMAT-3792 implementation receipt: a pure refactor of `run_shapes_gate_with` (#3792)

**This ticket IS the refactor.** It is not #3715's feature: that was folded into release/0.69.1-batch-1 at
ded8a932a. The diff against ded8a932a is only this refactor, its fragment and this receipt.

The cop's request, verbatim: "Please land a pure-refactor commit on a branch OFF release/0.69.1-batch-1 (not your old branch) that brings it under threshold with no behaviour change. Proof: `bash scripts/check_complexity_ratchet.sh` rc 0, `cargo test -p aprender-contracts --lib` green, and `pv lint contracts --gate shapes` 8/8 controls fired."

## What changed (no behaviour change)
`crates/aprender-contracts/src/lint/shapes_gate.rs::run_shapes_gate_with` (cyclomatic 13, cognitive 28; the
limit is 25) now delegates to:
- **`prepare`**: collect shapes → none (NoShapes) → the `armed_shapes` baseline error → `--shape` family selection. The order of answers is the original one, and a named family is armed as before.
- **`order_by_family`**: the `--shape` report ordering, moved as is.
- **`needs_receipts`**: the `resolves: receipt` test, moved as is.
- **`verdict_of`**: violations → Fail, warnings alone → Unknown{Warn}, else Pass. This is the original if/else.
- **`by_shape`** and **`by_entity_type`**: the two report maps, moved as is (including the ONT-4c3 comment).

## Measured at 2a40dda85
| check | result |
|---|---|
| `bash scripts/check_complexity_ratchet.sh` | `run_shapes_gate_with` is no longer RED. rc stays 1 only for `crates/aprender-serve/src/constrain/tests.rs::generate_intent` (cyclomatic 13, cognitive 27), which is not in this diff and not this ticket's |
| `cargo test -p aprender-contracts --lib` | 1701 passed, 0 failed |
| `pv lint contracts --gate shapes` | Pass, 8/8 `pc_extract` fired |
| CLI targets driving this function (`ont_release_readiness`, `ont4b_shapes_gate`, `ont4c1_model_receipts`, `ont4c3_parity_receipts`) | 34 / 11 / 12 / 10 passed |
| `cargo clippy -p aprender-contracts --lib --tests -- -D warnings` | clean |
