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
| lambda | x86_64-unknown-linux-gnu | 0 | 160 | `apr 0.65.2 (v0.65.2+no-git)` | 111 advertised / 110 answer `--help` (0 unusable) |
| intel | x86_64-unknown-linux-gnu | 0 | 202 | `apr 0.65.2 (v0.65.2+no-git)` | 111 advertised / 110 answer `--help` (0 unusable) |
| gx10 | aarch64-unknown-linux-gnu | 0 | 115 | `apr 0.65.2 (v0.65.2+no-git)` | 111 advertised / 110 answer `--help` (0 unusable) |
| mini | aarch64-apple-darwin | 0 | 218 | `apr 0.65.2 (v0.65.2+no-git)` | 111 advertised / 110 answer `--help` (0 unusable) |

## Parity lanes (post-publish, published binary, comparator llama.cpp 39173bcac per `scripts/llama_pin.toml`)
Every row is the published `cargo install aprender --version 0.65.2 --locked --force` binary measured on the host by `scripts/parity_host_receipt.sh` (5 interleaved replicates per band, ladder 1/4/8/16). Numbers are aggregate tok/s medians, subject vs comparator, from the receipt's `parity` block or, where the script refused to emit a block, its `parity_attempt` block. **c>1 columns are STRUCK on every host:** a c>1 ratio is a number only when the band carries a PP-26 witness PASS (`min_agree_tokens 64`, `max_constant_run 16`) and the comparator's `slots_admitted` at that c read from `/props`; no band on any host carries a witness (`witness: null` in every band record), so every c>1 band is `INVALID-CORRECTNESS(#2753/#2776)` (P-4). `slots_admitted` did equal c on every band on every host. Completion counts (n/N requests) are kept because they are correctness facts, not ratios.

| host | lane | c=1 | c=4 | c=8 | c=16 | block | verdict |
|---|---|---|---|---|---|---|---|
| lambda | cpu (default install) | 23.2 vs 75.2 (decode 44.1 vs 74.2; TTFT 2.6 s vs 16 ms) | STRUCK | STRUCK | STRUCK | `parity` | c=1 FAIL — decode 0.59×, prefill 0.005× (PMAT-962). c>1: INVALID-CORRECTNESS(#2753/#2776) — no PP-26 witness was run on any band (`bands[].subject.witness = null`), so no c>1 ratio exists; comparator `slots_admitted` did equal c on every band (`comparator_admission`, from `/props`) |
| lambda | cuda (`--features cuda` install of the same crate, `/tmp/apr-0652-cuda`, 167 s per `lambda.json` `cuda_install` and `lambda-cuda-install.txt`) | 326.8 vs 492.5 (decode 345.4 vs 499.3; prefill 5269 vs 29648; TTFT 24 vs 5 ms) | STRUCK | STRUCK | STRUCK | `parity` (second lane, run e52be880) | c=1 FAIL — decode 0.69×, prefill 0.18×. c>1: INVALID-CORRECTNESS(#2753/#2776) — no witness; the earlier "parity at c=16" is withdrawn |
| intel | cpu | 14.3 vs 42.0 (per-replicate 18.1/6.2/12.3/14.6/14.3) | STRUCK (8/8 completed) | STRUCK (8/8 completed) | replicate 2: 0/16 requests succeeded; replicate 1: 9/16 | `parity_attempt` (refused) | NO BLOCK — zero-throughput band (PMAT-963); c>1 INVALID-CORRECTNESS, no witness |
| gx10 | cpu | 3.5 vs 75.7 (TTFT 17.6 s) | STRUCK (4/4 completed, TTFT 36 s) | 0/8 succeeded, all 5 replicates | 0/16 succeeded, all 5 replicates | `parity_attempt` (refused) | NO BLOCK — 0.046× at c=1 (PMAT-964); every request fails at c≥8 (PMAT-963); c>1 INVALID-CORRECTNESS, no witness |
| mini | cpu | 4.5 vs 90.4 (TTFT 13.9 s) | STRUCK (4/4 completed, TTFT 49.5 s) | 0/8 succeeded, all 5 replicates | 0/16 succeeded, all 5 replicates | `parity_attempt` (refused) | NO BLOCK — 0.05× at c=1 (PMAT-964); every request fails at c≥8 (PMAT-963); c>1 INVALID-CORRECTNESS, no witness. Run under Homebrew bash 5.3 + util-linux flock (hand-installed; declared in paiml/infra machines/mini/forjar.yaml by the sibling PR) |

`check_multiplatform_dogfood.sh` on these receipts: install rows ok on all four hosts; bench rows ok on lambda and intel (REPORT on gx10 and mini: `apr bench` measured 7.7 and 8.1 tok/s, below its own H12 floor of 10, so no block); parity rows FAIL on every host — lambda because both its lanes are below the declared floor at c=1 (the cuda lane exists only because the crate was installed a second time with `--features cuda`; the default install resolves 0 GPU layers on a 4090), intel/gx10/mini because the run could not emit a block.

## Determination
- **Publish: complete.** 74/74 crates at 0.65.2 on crates.io; GitHub release `v0.65.2` at `8e1e9ad40`.
- **Pre-publish dogfood: GO** (41 rows, no FAIL).
- **Post-publish dogfood: NO-GO, on measured evidence, not on a gate defect.** The gate defect (PMAT-960, `version-unpublished` lacks post-publish polarity) is real but is not what decides this: the published default-feature binary is below parity on every CPU lane measured (0.59× decode on x86 lambda, ~0.35× on intel under runner load, 0.046× on aarch64 gx10, 0.05× on aarch64-apple mini) and fails every request at c≥8 on gx10 and mini and intermittently at c=16 on intel. Tickets: PMAT-962 (prefill at decode speed, lambda), PMAT-963 (request failures at c≥8), PMAT-964 (aarch64 published binary below its own H12 floor), PMAT-960 (gate polarity), PMAT-961 (resolver refused every non-CUDA comparator; fixed in #2867, applied on intel and mini to take these measurements at all).
- **Decision (decided_by: noah, 2026-09-05):** "0.65.2 stays published. No yank, no 0.65.3. These four host receipts are the 0.66 baseline." The 0.66 discovery baseline is `8e1e9ad40`.
- **PMAT-960 in one line:** `scripts/dogfood.sh` row `version-unpublished` has pre-publish polarity only (PASS iff the version is absent from index.crates.io); in `--phase post-publish` it FAILs the correct state (0.65.2 present, today's RED) and would PASS a failed publish (absent) — fail-open on the one outcome that phase exists to catch. Today it is red on success, not green on failure.
- **Dogfood runs (PP-9, a cell once run is spent):** run 1 (`receipt-20260904T232050Z`, 3 RED) and run 2 (`receipt-20260905T021928Z`, 4 RED incl. the claim-literal guard on the 0.66 report) are the record, kept under `evidence/release/0.65.2/`. Run 3 (`receipt-20260905T024626Z`) was started to re-check after the claim-literal fix, not as a new row whose `why` is "comparator resolvable on intel/mini post-#2867"; it ran to completion before that ruling, it is NOT a row, its receipt is not kept, and the claim-literal fix is verified by the guard's own PASS line, not by a dogfood rerun.
- **Ledger and matrix:** the five runs are rows 7–11 of `evidence/parity/LEDGER.md` (RECORDED, `validity_by_band`, `what_it_lacks[]`). `scripts/perf-matrix.yaml` carries them as `baselines.<host>.D1` (and `lambda.D1-cuda`) cells seeded at achieved c=1 values (P-6); c>1 bands are `INVALID-CORRECTNESS` and unseeded (P-4); nothing is ARMED (`armed_by` untouched). `perf_gate.sh --workload` accepts only W1|W2 today, so D1 is a recorded seed no arm consumes yet. `check_perf_matrix_schema.sh`, `check_perf_claims_cite_receipts.sh`, `check_no_claim_literals.sh` and `spec_conformance.sh` all PASS on this tree.
- **What this verdict does not say:** it does not withdraw 0.65.2 from crates.io, and the only accelerator number it carries is lambda's cuda lane at c=1: 0.69× decode / 0.18× prefill against llama.cpp CUDA. No c>1 number exists (no PP-26 witness) and no GPU claim beyond that lane is made.
