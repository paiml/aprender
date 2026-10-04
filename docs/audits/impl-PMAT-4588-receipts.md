# #4588 receipts: one section per folded ticket

Tree: split/ont10-rest-0.70 at the split revert (3ec5493f72 + this commit). pv 0.70.0, built from 3ec5493f72.
Measured 2026-09-29 on lambda (load about 2). Cargo tests run on intel only, so rows that need them are
marked NOT_MEASURED here and are not counted as passes.

These tickets already have receipts: PMAT-3715 (`impl-PMAT-3715-receipt.md`,
`impl-PMAT-3715-ollama-qwen35-receipt.md`), PMAT-4431 (`impl-PMAT-4431-receipt.md`), and GH-4189
(`impl-GH-4189-receipt.md`).

The split (cop ruling 19:35Z) moved PMAT-4429 (x86-main intel pin) and PMAT-3668 (docs-only gate skip)
to PR-2b, because neither is on the gate graph of rc.1 or FINAL. PMAT-4016 (Planned, 0 code in this diff)
was dropped from the ticket list.

## PMAT-4502: binary surface extraction

| AC | evidence | command | rc |
|---|---|---|---|
| every shipped bin has a contract | `contracts/binary-surface-v1.yaml` + 31 `contracts/bin-*.yaml` | `pv validate <file>`, run once per file (pv takes one file) | ok=32 bad=0 |
| surface audit gate is green | `scripts/surface_audit_bins_gate.sh` | `bash scripts/surface_audit_bins_gate.sh` | 0, `GREEN 29/29` |
| the gate cannot pass on an empty ledger | same script | `--csv` pointed at a header-only CSV | 1, `RED ... has 0 rows ... never` |
| extracted binary nodes | `evidence/binary/snapshot.jsonl` | `wc -l` | 30 lines |
| aprender-contracts lib tests | — | `cargo test -p aprender-contracts --lib` | NOT_MEASURED (intel only) |

## PMAT-4073: refinement gate (model_of)

| AC | evidence | command | rc |
|---|---|---|---|
| refinement gate never gets worse | `crates/aprender-contracts-staging/lean/formalization.yaml`, `discharge/refinement.rs` | `pv lint contracts/ --gate refinement --format json` | 0; verdict Pass, models 3, resolved 3, unrefined 76 = base 76, new_unrefined 0 |

## PMAT-3712: model ladder

| AC | evidence | command | rc |
|---|---|---|---|
| ladder checker has a case table | `scripts/check_model_ladder.sh` | `bash scripts/check_model_ladder.sh --self-test` | 0, `152 case(s), 0 bad` |
| qwen3-8b-q4km rung is required | already `required: true` at base (#3724) | — | unchanged by this PR |
| live ladder verdict | — | `bash scripts/check_model_ladder.sh` | RED by design until `evidence/crux/0.70.0` receipts exist; it is a release-time gate, not counted as a pass |
| cells producer | `scripts/check_ladder_cells_producer.sh` | `bash scripts/check_ladder_cells_producer.sh` (systemd unit, >170 s) | 0, `all cases and mutants as expected` (25 ok rows, 0 bad) |

## PMAT-4445: session forget_prefix

| AC | evidence | command | rc |
|---|---|---|---|
| API exists | `crates/aprender-serve/src/session.rs:234` `pub fn forget_prefix` | — | — |
| extending prompt prefills whole again | `session_tests.rs:298` | `cargo test -p aprender-serve --lib session` | NOT_MEASURED (intel only) |
| same prompt prefills whole again | `session_tests.rs:436` | same | NOT_MEASURED (intel only) |

## PMAT-4083: PVL EV-9 part 1 of 2

The scope is part 1. The provable-ladder section runs on every PR as an ADVISORY check
(`ci/sections.yml`, the `provable-ladder` section; C78(a)), and a forced skip is RED. Making it a
required check in `gate.needs` is part 2 and an operator check-in. It is not in this PR, and the entry
title now says so.

| AC | evidence | command | rc |
|---|---|---|---|
| a forced ladder skip is RED | `scripts/ci/ladder_skip_is_red.sh`, `docs/findings/ev9-fat-driver-skip-is-success.jsonl` | `bash scripts/ci/ladder_skip_is_red.sh` | 0, `PASS mutant (step deleted) -> section success: the check sees the defect` |
| section runs on every PR | `.github/workflows/ci.yml` `--sections ...,provable-ladder` | read | present |
| required in gate.needs | — | — | OUT OF SCOPE (part 2, operator) |

## CB-200 row: reverted (cop ruling 20:31Z)

This PR no longer re-baselines CB-200. `.pmat-gates.toml` `[tdg] baseline` and `scripts/cb200_baseline.txt`
are back to main's 599. `scripts/check_cb200_tdg_grade.sh`, `scripts/cb200_baseline.rebaseline` and the
one-time re-baseline hooks in `scripts/lib_baseline_ratchet.sh` / `scripts/check_baseline_ratchets.sh` are removed.

| check | command | base 00052c0128 | head | verdict |
|---|---|---|---|---|
| CB-200 grade findings, same scanner, same run (pmat 3.42.0, cold cache) | `pmat comply check --path <tree> --checks CB-200 --format json` | 622 | 621 (before revert) | head never-worse (-1); the 599->617 was a stored-limit raise, not new findings |
| complexity set | `bash scripts/check_complexity_ratchet.sh` | 658 fns | 658 fns | equal; rc 1 on both: host pmat 3.42.0 != recorded 3.41.1 (instrument) |
| baseline ratchets | `bash scripts/check_baseline_ratchets.sh` | rc 1 | rc 1 | same 2 tool_version FAIL rows on both (host pmat), no new row |
| ratchet self-test | `bash scripts/check_baseline_ratchets.sh --self-test` | — | rc 0 | ok |
| guards wired | `bash scripts/check_guards_are_wired.sh` | — | rc 0 | PASS |

Pre-existing on main, reported not fixed here: main measures 622 CB-200 findings against its stored 599.
