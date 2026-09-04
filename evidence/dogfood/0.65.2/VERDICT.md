# v0.65.2 dogfood verdict — written, dated, per-gate

Cut: `v0.65.2` at `8e1e9ad40` (main), 2026-09-04. crates.io: 74/74 crates at 0.65.2 (drain complete 23:20Z).

Pre-publish (`scripts/dogfood.sh --phase pre-publish` on 8e1e9ad40, receipt `receipt-20260904T221607Z.json`): **GO**, 41 rows, no FAIL; DEFER rows were the two owed to post-publish by design (`publish-dry-run`, `declared:check_multiplatform_dogfood`).

Post-publish (`scripts/dogfood.sh --phase post-publish`, receipt `receipt-20260904T232050Z.json`): 41 rows, 3 FAIL. Each RED classified with evidence:

## Gate defect, not a release defect (PMAT-960)
- `version-unpublished` — "aprender 0.65.2 is ALREADY on crates.io — bump the version". The row asserts the **pre-publish** invariant and has no phase inversion; in `--phase post-publish` the published version being present is the expected state, which is exactly what it measured. The same shape was recorded for 0.64.0. Filed as PMAT-960 (post-publish polarity plus case rows for both phases).
- `dogfood-gates` — the roll-up of the row below; it is RED because one declared gate is RED, nothing else.

## Owed by this document (the four host receipts)
- `declared:check_multiplatform_dogfood` — RED until `evidence/dogfood/0.65.2/{lambda,intel,gx10,mini}.json` exist for 0.65.2 with `install_rc: 0`. See the host table below (filled in as each host's `cargo install aprender --version 0.65.2 --locked --force` completes; each receipt records the install's exit, wall time, binary sha256 and `--version` output, and the advertised-vs-usable CLI surface).

## Host receipts
| host | arch | install rc | wall s | `apr --version` | surface (advertised / answering --help) |
|---|---|---|---|---|---|
