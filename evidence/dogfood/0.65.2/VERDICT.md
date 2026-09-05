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
| lambda | x86_64-unknown-linux-gnu | 0 | 160 | `apr 0.65.2 (v0.65.2+no-git)` | {'cli_subcommands': 111, 'cli_subcommands_answering_help': 1 |
| intel | x86_64-unknown-linux-gnu | 0 | 202 | `apr 0.65.2 (v0.65.2+no-git)` | {'cli_subcommands': 111, 'cli_subcommands_answering_help': 1 |
| gx10 | aarch64-unknown-linux-gnu | 0 | 115 | `apr 0.65.2 (v0.65.2+no-git)` | {'cli_subcommands': 111, 'cli_subcommands_answering_help': 1 |
| mini | aarch64-apple-darwin | 0 | 218 | `apr 0.65.2 (v0.65.2+no-git)` | {'cli_subcommands': 111, 'cli_subcommands_answering_help': 1 |

## Parity lanes (post-publish, published binary, comparator llama.cpp 39173bcac per `scripts/llama_pin.toml`)
Every row is the published `cargo install aprender --version 0.65.2 --locked --force` binary measured on the host by `scripts/parity_host_receipt.sh` (5 interleaved replicates per band, ladder 1/4/8/16). Numbers are aggregate tok/s medians, subject vs comparator, from the receipt's `parity` block or, where the script refused to emit a block, its `parity_attempt` block.

| host | lane | c=1 | c=4 | c=8 | c=16 | block | verdict |
|---|---|---|---|---|---|---|---|
| lambda | cpu | 23.2 vs 75.2 (decode 44.1 vs 74.2; TTFT 2.6 s vs 16 ms) | 44.9 vs 128.3 | 52.5 vs 155.2 | 54.8 vs 172.4 | `parity` | FAIL — decode 0.59×, prefill 0.005× (PMAT-962) |
| lambda | cuda | measured from a second install of the same published crate with `--features cuda` (`/tmp/apr-0652-cuda`); see the `parity` block's second lane | | | | pending at the time of this line | see row below when landed |
| intel | cpu | 14.3 vs 42.0 (per-replicate 18.1/6.2/12.3/14.6/14.3) | 22.7 vs 93.3 | 10.9 vs 65.8 | replicate 2: 0/16 requests succeeded; replicate 1: 9/16 | `parity_attempt` (refused) | NO BLOCK — zero-throughput band (PMAT-963) |
| gx10 | cpu | 3.5 vs 77.8 (TTFT 17.6 s) | 7.1 vs 104.4 (TTFT 36 s) | 0/8 succeeded, all 5 replicates | 0/16 succeeded, all 5 replicates | `parity_attempt` (refused) | NO BLOCK — 0.045× at c=1 (PMAT-964); every request fails at c≥8 (PMAT-963) |
| mini | cpu | pending (run in progress under Homebrew bash 5.3 + util-linux flock; macOS bash 3.2 exits the script silently and macOS has no flock) | | | | pending | pending |

`check_multiplatform_dogfood.sh` on these receipts: install rows ok on all four hosts; bench rows ok on lambda and intel (REPORT on gx10: `apr bench` measured 7.7 tok/s, below its own H12 floor of 10, so no block; mini pending); parity rows FAIL on every host — lambda because the 4090 host needs a cuda lane and the default-feature binary resolves 0 GPU layers, intel/gx10 because the run could not emit a block, mini pending.

## Determination
- **Publish: complete.** 74/74 crates at 0.65.2 on crates.io; GitHub release `v0.65.2` at `8e1e9ad40`.
- **Pre-publish dogfood: GO** (41 rows, no FAIL).
- **Post-publish dogfood: NO-GO, on measured evidence, not on a gate defect.** The gate defect (PMAT-960, `version-unpublished` lacks post-publish polarity) is real but is not what decides this: the published default-feature binary is below parity on every CPU lane measured (0.59× decode on x86 lambda, ~0.35× on intel under runner load, 0.045× on aarch64 gx10) and fails every request at c≥8 on gx10 and intermittently at c=16 on intel. Tickets: PMAT-962 (prefill at decode speed, lambda), PMAT-963 (request failures at c≥8), PMAT-964 (aarch64 published binary below its own H12 floor), PMAT-960 (gate polarity), PMAT-961 (resolver refused every non-CUDA comparator; fixed in #2867, applied on intel and mini to take these measurements at all).
- **What this verdict does not say:** it does not withdraw 0.65.2 from crates.io (the operator's call, [A] in the receipt: leave it published, ship 0.66 against the 0.66 parity report with these receipts as its baseline), and it does not claim any GPU number: the cuda lane on lambda is the only accelerator measurement attempted and is recorded when it lands.
