## Ticket
#2971 (alfredodeza's report; P0; DAG row **L0-1a**, ticket PMAT-1065, epic #2873). #3017 was closed as its duplicate.

## Claim
Claim 2 of 0.66, the bounded half: every model in `evidence/models/supported.yaml` (derived, never typed) is measured over ≥ 64 positions against a threshold that carries its basis, and a model that fails is refused by the GPU with its reason printed — never silently run on the CPU under a forced request. The op that diverges on Qwen2.5-1.5B and its fix are L0-1b.

## RED test
- The RED-first record, before any kernel edit: `c977cad58` — `apr parity` on lambda (RTX 4090, the 0.65.2 cuda install `c642576eecb62daa`), 78 positions: **1.5B min cosine 0.9508 at position 0 (token 785), 7B 0.9986**; n=5 repeats are bit-for-bit identical (`d2a566642`).
- The sentinel gate: `d56f65430` — `sentinel_1p5b_on_lambda_is_red_under_the_horizon_rule` / `sentinel_7b_on_lambda_is_green_under_the_horizon_rule` (lib tests, ride `workspace-test`).
- The must-RED twin: `tests/fixtures/parity/defective/one-position-at-0.5.json` (C14 case-table row 3).

## Acceptance (`.pr/L0-1/accept.sh`, orchestrator's run, `.pr/L0-1/accept.log`)
```
derive_model_manifest.sh --self-test        rc=0   (6/6)
derive_model_manifest.sh --check            rc=0   (18 models)
check_model_parity.sh --self-test           rc=0   (6/6)
--judge 7B record                           rc=0   PASS min 0.9986
--judge 1.5B record                         rc=1   as required (RED)
--judge one-position-at-0.5 twin            rc=1   as required (RED)
cargo test -p apr-cli --test reg15_admission        rc=0 (7 passed)
cargo test -p apr-cli --lib sentinel_tests          rc=0 (3 passed)
cargo test -p aprender-serve --lib parity_report    rc=0
pv validate contracts/apr-gpu-cpu-parity-v1.yaml    rc=0 (valid)
--manifest: the fleet-verify leg on lambda/gx10 (not this host)
```
`cargo check -p aprender-serve -p apr-cli --features cuda` on lambda (CUDA 12.8): clean at 35c330c16.

## Mutation (I3)
| round | commit | expected RED | run |
|---|---|---|---|
| 1 | `a5db15200` — a hand-typed manifest entry + `min_cosine: 0.90` | guard-runner-labels: "The manifest equals its derivation" FAILS; workspace-test: `sentinel_1p5b_on_lambda_is_red` FAILS | _run id after CI_ |
| 2 | (next) `min_positions: 1` | guard-runner-labels: C14 case-table row 4 FAILS | _run id after CI_ |
| GREEN | the reverts | 6/6 · 6/6 · 3 sentinels | _run id after CI_ |

## Contract
`contracts/apr-gpu-cpu-parity-v1.yaml` — kind: pattern; PAR-OB-001..003 ↔ PAR-F-001..003. `pv validate` (via `scripts/pv_bin.sh`): `0 error(s), 0 warning(s) — Contract is valid.`

## Quorum
review-only row: one agy lane on this diff — _verdict recorded in `.pr/L0-1/quorum.md` and the receipt before arming_. The L0-1 root-cause quorum (three lanes, one family — a recorded gap) refuted the fused-FFN hypothesis on default config; L0-1b answers the rest by measurement.

## Receipt
`docs/audits/impl-PMAT-1065-receipt.md` (v6 DONE-IF ledger: (i) admission level ✓, (ii) ✓, (iii) blocked on R-0a, (iv) one pair measured, (v) ✓, (vi) pending G-11b, (vii) ✓).

## Writes
`scripts/derive_model_manifest.sh`, `scripts/check_model_parity.sh`, `scripts/dogfood.sh` (C14 rows), `evidence/models/supported.yaml`, `evidence/parity/{thresholds.yaml,l0-1/**}`, `evidence/dogfood/0.65.2/{lambda,gx10}.json` (validity relabel), `tests/fixtures/parity/defective/**`, `crates/apr-cli/src/commands/{parity_admission.rs,chat_generate_session_02.rs,comparison.rs,mod.rs}`, `crates/apr-cli/src/error.rs`, `crates/apr-cli/tests/reg15_admission.rs`, `crates/aprender-serve/src/gguf/cuda/{mod.rs,mod_parity_gate.rs}`, `crates/aprender-serve/src/api/effective_config.rs`, `contracts/apr-gpu-cpu-parity-v1.yaml`, `.github/workflows/ci.yml` (three guard-runner-labels steps), `.pr/L0-1/accept.sh`, the receipt. No DAG, roadmap, README or spec edit.
