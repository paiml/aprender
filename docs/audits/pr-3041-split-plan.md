# PR #3041 split plan (PMAT-3041, phase 1)

Operator ruling on #3041: **"#3041 gets split, not resolved."** Order: mechanical / Cargo.lock /
contract moves first, then backend-registry on top. This document is the measurement the split is cut from.
It changes no code.

## Tree line (every figure below was measured on exactly these refs)

| ref | sha |
|-----|-----|
| HEAD (`PMAT-3041-split-plan`, cut from origin/main) | `809326c0f84dfb91bd3fe6e90a2eabd3878cfbe5` |
| origin/main | `809326c0f84dfb91bd3fe6e90a2eabd3878cfbe5` (`contracts(0.68): 14 obligations had no id …` #3316) |
| origin/agent/R-0b (#3041 head) | `245ca32efbdd04a537c59aec32fcafd1528e274b` |
| merge-base | `c04eda87d2f2b74f893dcb92526119aa71ad1f36` |

Re-checked with `git ls-remote origin refs/heads/main refs/heads/agent/R-0b` at the end of measurement:
unchanged. main moved 1898 files / +108,517 −5,146 past the merge-base; #3041 carries 75 commits and
96 files (+11,460 −584) over it. `gh pr view 3041` reports the same 96 files.

## Headline: most of #3041 has already landed on main

#3041 is stacked: it merged `agent/R-0` (R-0a) and `agent/L0-1` (L0-1a) into itself. Both reached main
by **squash merge**, independently of #3041:

| landed | PR | merged |
|--------|----|--------|
| `ad8f98768` feat(registry): R-0a — BackendRegistry probes and enumerates cpu/cuda/wgpu … | #3004 | 2026-09-14T20:12Z |
| `74e3b2540` feat(parity): L0-1a — every supported model computes the same function on GPU as on CPU … | #3026 | 2026-09-08T18:44Z |

The files seen arriving "from main" in another merge today (`crates/aprender-compute/src/registry/{mod,cuda,wgpu_probe}.rs`,
`contracts/apr-backend-registry-v1.yaml`, `crates/apr-cli/tests/registry_failure_catalogue.rs`,
`crates/aprender-compute/tests/registry_case_table.rs`) are R-0a, landed by `ad8f98768`.

Because those landings were squashes, `git cherry -v origin/main origin/agent/R-0b` marks **all 63**
non-merge commits `+` (not on main). That is a patch-id artefact and says nothing about content. The
content measurement is:

| state vs current origin/main | files | meaning |
|------------------------------|------:|---------|
| IDENTICAL (incl. 2 `target-review/*` absent on both sides) | 42 | already landed — drop |
| DIFFERENT, but R-0b's side is only STALE (3-way merge into main leaves 0 residual, or the conflict hunk is main's later work and no R-0b-own commit touches the file) | 20 | landed; main moved past it — drop, take main |
| ABSENT — **zombie** (`contracts/apr-gpu-cpu-parity-v1.yaml`, deleted on main by `9465af2c7`) | 1 | drop (Decision D1) |
| DIFFERENT with a real R-0b-own delta | 23 | residual |
| ABSENT, genuinely new | 10 | residual |
| **total** | **96** | **63 drop / 33 residual** |

The residual is exactly the R-0b row (#3002, PMAT-1073): ten commits whose subject is
`plan|feat|fix|docs(R-0b)` — `dff0225ac 9610dd55c 99589f98f 805342e18 a0269f3c1 5b5116d6c c93f74ae3
a8faded4b f8a94b382 245ca32ef`. No L0-1a or R-0a content remains outstanding.

**Recommendation: close #3041 with a pointer, and deliver the residual as the ordered slices below,
each cut fresh from origin/main.** Do not merge or rebase `agent/R-0b`. A merge brings back the zombie contract, conflicts in 22 files
(14 of them carry no R-0b change at all), and carries 63 commits of history that main already holds
as two squashes. This is the ruling's split, applied to what is actually left.

### Method (reproducible)

- State: `git cat-file -e origin/main:<p>` and `git diff --numstat origin/main origin/agent/R-0b -- <p>`.
- Landed-by: `git log origin/main --diff-filter=A <merge-base>..origin/main -- <p>` (the add), else the
  oldest main commit touching `<p>` since the merge-base.
- Residual: a trial merge in a detached scratch worktree
  (`git worktree add --detach $SCRATCH origin/main && git merge --no-commit --no-ff origin/agent/R-0b`);
  `git diff --cached --numstat origin/main` for clean files, `git diff --name-only --diff-filter=U`
  for conflicts, then `git merge --abort` and `git worktree remove`. (git here is 2.34, so
  `merge-tree --write-tree` is unavailable.)
- R-0b-own delta per conflicted file: numstat of each of the ten R-0b commits restricted to the file.

## Every file #3041 touches

| # | file | state vs origin/main | R-0b vs main (numstat) | landed on main by | residual after 3-way merge into main | split |
|---|------|------|------|------|------|------|
| 1 | `.claude/skills/apr-dogfood/SKILL.md` | DIFFERENT | 1+/88- | `b0406bc8d` | CONFLICT, stale side only (R-0b-own: none) — take main | drop |
| 2 | `.github/workflows/ci.yml` | DIFFERENT | 403+/776- | `74e3b2540` | CONFLICT; R-0b-own: 9610dd55c +4 (self-test step), 99589f98f +2 (--static step), f8a94b382 1-token (backend_refusal_case_table on :570) | S3a+S3b |
| 3 | `.pr/L0-1/accept.sh` | DIFFERENT | 1+/1- | `74e3b2540` | CONFLICT, stale side only (R-0b-own: none) — take main | drop |
| 4 | `.pr/L0-1/diff_benchmark_report.override.patch` | IDENTICAL | 0 | `74e3b2540` | none | drop |
| 5 | `.pr/L0-1/plan.md` | IDENTICAL | 0 | `74e3b2540` | none | drop |
| 6 | `.pr/L0-1/pr-body.md` | IDENTICAL | 0 | `74e3b2540` | none | drop |
| 7 | `.pr/L0-1/quorum.md` | IDENTICAL | 0 | `74e3b2540` | none | drop |
| 8 | `.pr/R-0b/accept.sh` | ABSENT | - | — | new file | S4 |
| 9 | `.pr/R-0b/discover.json` | ABSENT | - | — | new file | S4 |
| 10 | `.pr/R-0b/plan.md` | ABSENT | - | — | new file | S4 |
| 11 | `.pr/R-0b/quorum.md` | ABSENT | - | — | new file | S4 |
| 12 | `Cargo.lock` | DIFFERENT | 297+/211- | `7a33db8e0` | none — stale side, take main | drop |
| 13 | `README.md` | DIFFERENT | 2+/2- | `74e3b2540` | CONFLICT, stale side only (R-0b-own: none) — take main | drop |
| 14 | `book/src/SUMMARY.md` | IDENTICAL | 0 | `ad8f98768` | none | drop |
| 15 | `book/src/cli/devices.md` | IDENTICAL | 0 | `ad8f98768` | none | drop |
| 16 | `contracts/apr-backend-registry-v1.yaml` | DIFFERENT | 36+/1- | `ad8f98768` | CONFLICT; R-0b-own: 9610dd55c 27+/1-, a0269f3c1 6+/3-, 5b5116d6c 7+/1- (REG-OB-004, REG-F-003, binding_registry) | S3a+S3c |
| 17 | `contracts/apr-cli-commands-v1.yaml` | IDENTICAL | 0 | `ad8f98768` | none | drop |
| 18 | `contracts/apr-devices-schema-v1.yaml` | IDENTICAL | 0 | `ad8f98768` | none | drop |
| 19 | `contracts/apr-dogfood-coverage-v1.yaml` | DIFFERENT | 17+/22- | `b0406bc8d` | CONFLICT, stale side only (R-0b-own: none) — take main | drop |
| 20 | `contracts/apr-gpu-cpu-parity-v1.yaml` | ABSENT — **ZOMBIE** | - | added `74e3b2540` (#3026), DELETED `9465af2c7` (#3025) | a merge RESURRECTS it; byte-identical to 74e3b2540; R-0b-own delta none | drop |
| 21 | `contracts/apr-page-cli-devices-v1.yaml` | IDENTICAL | 0 | `ad8f98768` | none | drop |
| 22 | `contracts/schemas/apr-devices-v1.schema.json` | IDENTICAL | 0 | `ad8f98768` | none | drop |
| 23 | `crates/apr-cli/Cargo.toml` | DIFFERENT | 4+/4- | `7a33db8e0` | none — stale side, take main | drop |
| 24 | `crates/apr-cli/src/accel.rs` | DIFFERENT | 58+/2- | — (main has not touched it since the merge-base) | clean, 58+/2- | S3b |
| 25 | `crates/apr-cli/src/commands/bench.rs` | DIFFERENT | 7+/21- | — (main has not touched it since the merge-base) | clean, 7+/21- | S3b |
| 26 | `crates/apr-cli/src/commands/chat_generate_session_02.rs` | DIFFERENT | 29+/9- | `74e3b2540` | CONFLICT; R-0b-own: 805342e18 3+, 5b5116d6c 25+/6- | S3b+S3c |
| 27 | `crates/apr-cli/src/commands/chat_load_tokenizers.rs` | DIFFERENT | 10+/24- | `bad5869cf` | CONFLICT; R-0b-own: 805342e18 3+ | S3b |
| 28 | `crates/apr-cli/src/commands/comparison.rs` | IDENTICAL | 0 | `74e3b2540` | none | drop |
| 29 | `crates/apr-cli/src/commands/devices.rs` | DIFFERENT | 1+/1- | `ad8f98768` | CONFLICT; R-0b-own: 9610dd55c 1+/1- (reserve_override -> pub(crate)) | S3a |
| 30 | `crates/apr-cli/src/commands/diff_benchmark_report.rs` | IDENTICAL | 0 | `74e3b2540` | none | drop |
| 31 | `crates/apr-cli/src/commands/finetune.rs` | DIFFERENT | 2+/2- | — (main has not touched it since the merge-base) | clean, 2+/2- | S3b |
| 32 | `crates/apr-cli/src/commands/mod.rs` | DIFFERENT | 0+/2- | `74e3b2540` | CONFLICT, stale side only (R-0b-own: none) — take main | drop |
| 33 | `crates/apr-cli/src/commands/parity_admission.rs` | DIFFERENT | 1+/1- | `74e3b2540` | CONFLICT, stale side only (R-0b-own: none) — take main | drop |
| 34 | `crates/apr-cli/src/commands/run_entry.rs` | DIFFERENT | 13+/0- | — (main has not touched it since the merge-base) | clean, 13+/0- | S3c |
| 35 | `crates/apr-cli/src/commands/serve/handler_gpu_completion.rs` | DIFFERENT | 1+/7- | — (main has not touched it since the merge-base) | clean, 1+/7- | S3b |
| 36 | `crates/apr-cli/src/commands/serve/handlers_include_01.rs` | DIFFERENT | 6+/0- | — (main has not touched it since the merge-base) | clean, 6+/0- | S3b |
| 37 | `crates/apr-cli/src/commands/serve/mod.rs` | DIFFERENT | 184+/77- | — (main has not touched it since the merge-base) | clean, 184+/77- | S3b+S3c |
| 38 | `crates/apr-cli/src/commands/serve/tests_offload_report_pp14.rs` | DIFFERENT | 2+/2- | — (main has not touched it since the merge-base) | clean, 2+/2- | S3b |
| 39 | `crates/apr-cli/src/commands_enum.rs` | DIFFERENT | 6+/0- | — (main has not touched it since the merge-base) | clean, 6+/0- | S3a |
| 40 | `crates/apr-cli/src/dispatch.rs` | DIFFERENT | 179+/148- | — (main has not touched it since the merge-base) | clean, 179+/148- | S3b |
| 41 | `crates/apr-cli/src/dispatch_analysis.rs` | DIFFERENT | 3+/14- | `b0406bc8d` | clean, 1+/4- | S3b |
| 42 | `crates/apr-cli/src/error.rs` | DIFFERENT | 8+/0- | `74e3b2540` | CONFLICT; R-0b-own: 9610dd55c 8+ (BackendUnavailable, exit 14) | S3a |
| 43 | `crates/apr-cli/src/extended_commands.rs` | DIFFERENT | 0+/10- | `b0406bc8d` | none — stale side, take main | drop |
| 44 | `crates/apr-cli/src/help_producer_truth.rs` | DIFFERENT | 84+/65- | — (main has not touched it since the merge-base) | clean, 84+/65- | S3b |
| 45 | `crates/apr-cli/src/lib.rs` | DIFFERENT | 2+/1- | — (main has not touched it since the merge-base) | clean, 2+/1- | S3b |
| 46 | `crates/apr-cli/src/registry.rs` | ABSENT | - | — | new file | S3a |
| 47 | `crates/apr-cli/tests/backend_refusal_case_table.rs` | ABSENT | - | — | new file | S3a |
| 48 | `crates/apr-cli/tests/cli_commands.rs` | IDENTICAL | 0 | `ad8f98768` | none | drop |
| 49 | `crates/apr-cli/tests/fixtures/registry/cpu-only.json` | IDENTICAL | 0 | `ad8f98768` | none | drop |
| 50 | `crates/apr-cli/tests/fixtures/registry/defective/missing-metal-line.json` | IDENTICAL | 0 | `ad8f98768` | none | drop |
| 51 | `crates/apr-cli/tests/fixtures/registry/no-accelerator-compiled.json` | ABSENT | - | — | new file | S3b |
| 52 | `crates/apr-cli/tests/fixtures/registry/one-cuda.json` | IDENTICAL | 0 | `ad8f98768` | none | drop |
| 53 | `crates/apr-cli/tests/fixtures/registry/two-vendors.json` | IDENTICAL | 0 | `ad8f98768` | none | drop |
| 54 | `crates/apr-cli/tests/reg15_admission.rs` | DIFFERENT | 1+/1- | `74e3b2540` | CONFLICT, stale side only (R-0b-own: none) — take main | drop |
| 55 | `crates/apr-cli/tests/registry_failure_catalogue.rs` | IDENTICAL | 0 | `ad8f98768` | none | drop |
| 56 | `crates/aprender-compute/src/lib.rs` | IDENTICAL | 0 | `ad8f98768` | none | drop |
| 57 | `crates/aprender-compute/src/registry/cuda.rs` | IDENTICAL | 0 | `ad8f98768` | none | drop |
| 58 | `crates/aprender-compute/src/registry/mod.rs` | IDENTICAL | 0 | `ad8f98768` | none | drop |
| 59 | `crates/aprender-compute/src/registry/wgpu_probe.rs` | DIFFERENT | 19+/109- | `ad8f98768` | CONFLICT, stale side only (R-0b-own: none) — take main | drop |
| 60 | `crates/aprender-compute/tests/registry_case_table.rs` | IDENTICAL | 0 | `ad8f98768` | none | drop |
| 61 | `crates/aprender-serve/src/api/effective_config.rs` | DIFFERENT | 54+/2- | `74e3b2540` | clean, 54+/2- | S3c |
| 62 | `crates/aprender-serve/src/api/tests/effective_config_route_pp2.rs` | DIFFERENT | 32+/1- | `74e3b2540` | CONFLICT; R-0b-own: a0269f3c1 32+/1- | S3c |
| 63 | `crates/aprender-serve/src/gguf/cuda/mod.rs` | DIFFERENT | 5+/24- | `74e3b2540` | CONFLICT, stale side only (R-0b-own: none) — take main | drop |
| 64 | `crates/aprender-serve/src/gguf/cuda/mod_parity_gate.rs` | DIFFERENT | 3+/3- | `74e3b2540` | CONFLICT, stale side only (R-0b-own: none) — take main | drop |
| 65 | `docs/audits/impl-PMAT-1065-receipt.md` | IDENTICAL | 0 | `74e3b2540` | none | drop |
| 66 | `docs/audits/impl-PMAT-1073-receipt.md` | ABSENT | - | — | new file | S4 |
| 67 | `docs/audits/impl-PMAT-989-receipt.md` | IDENTICAL | 0 | `ad8f98768` | none | drop |
| 68 | `docs/audits/impl-estimates.jsonl` | DIFFERENT | 0+/16- | `9465af2c7` | none — stale side, take main | drop |
| 69 | `docs/audits/quorum/PMAT-1073-lane-1.json` | ABSENT | - | — | new file | S4 |
| 70 | `docs/audits/surface_audit.csv` | DIFFERENT | 0+/3- | `b0406bc8d` | none — stale side, take main | drop |
| 71 | `evidence/dogfood/0.65.2/gx10.json` | IDENTICAL | 0 | `74e3b2540` | none | drop |
| 72 | `evidence/dogfood/0.65.2/lambda.json` | IDENTICAL | 0 | `74e3b2540` | none | drop |
| 73 | `evidence/models/supported.yaml` | IDENTICAL | 0 | `74e3b2540` | none | drop |
| 74 | `evidence/parity/l0-1/gx10/RECORD.md` | IDENTICAL | 0 | `74e3b2540` | none | drop |
| 75 | `evidence/parity/l0-1/gx10/n5/DETERMINISM.md` | IDENTICAL | 0 | `74e3b2540` | none | drop |
| 76 | `evidence/parity/l0-1/gx10/qwen2.5-coder-1.5b-instruct-q4_k_m.err` | IDENTICAL | 0 | `74e3b2540` | none | drop |
| 77 | `evidence/parity/l0-1/gx10/qwen2.5-coder-1.5b-instruct-q4_k_m.json` | IDENTICAL | 0 | `74e3b2540` | none | drop |
| 78 | `evidence/parity/l0-1/gx10/qwen2.5-coder-7b-instruct-q4_k_m.err` | IDENTICAL | 0 | `74e3b2540` | none | drop |
| 79 | `evidence/parity/l0-1/gx10/qwen2.5-coder-7b-instruct-q4_k_m.json` | IDENTICAL | 0 | `74e3b2540` | none | drop |
| 80 | `evidence/parity/l0-1/lambda/RECORD.md` | DIFFERENT | 0+/19- | `74e3b2540` | CONFLICT, stale side only (R-0b-own: none) — take main | drop |
| 81 | `evidence/parity/l0-1/lambda/n5/DETERMINISM.md` | IDENTICAL | 0 | `74e3b2540` | none | drop |
| 82 | `evidence/parity/l0-1/lambda/qwen2.5-coder-1.5b-instruct-q4_k_m.err` | IDENTICAL | 0 | `74e3b2540` | none | drop |
| 83 | `evidence/parity/l0-1/lambda/qwen2.5-coder-1.5b-instruct-q4_k_m.json` | IDENTICAL | 0 | `74e3b2540` | none | drop |
| 84 | `evidence/parity/l0-1/lambda/qwen2.5-coder-7b-instruct-q4_k_m.err` | IDENTICAL | 0 | `74e3b2540` | none | drop |
| 85 | `evidence/parity/l0-1/lambda/qwen2.5-coder-7b-instruct-q4_k_m.json` | IDENTICAL | 0 | `74e3b2540` | none | drop |
| 86 | `evidence/parity/thresholds.yaml` | DIFFERENT | 1+/48- | `74e3b2540` | CONFLICT, stale side only (R-0b-own: none) — take main | drop |
| 87 | `scripts/check_backend_registry.sh` | ABSENT | - | — | new file | S3b |
| 88 | `scripts/check_model_parity.sh` | DIFFERENT | 1+/41- | `74e3b2540` | CONFLICT, stale side only (R-0b-own: none) — take main | drop |
| 89 | `scripts/claim_literal_baseline.txt` | DIFFERENT | 6+/1- | `7a33db8e0` | clean, 0+/1- | S3b |
| 90 | `scripts/complexity_baseline.txt` | DIFFERENT | 8+/1- | `74e3b2540` | none — stale side, take main | drop |
| 91 | `scripts/derive_model_manifest.sh` | IDENTICAL | 0 | `74e3b2540` | none | drop |
| 92 | `scripts/dogfood.sh` | DIFFERENT | 6+/63- | `74e3b2540` | CONFLICT, stale side only (R-0b-own: none) — take main | drop |
| 93 | `scripts/tree_reader_tests.txt` | DIFFERENT | 19+/101- | `ebc9e9d81` | CONFLICT; R-0b-own: f8a94b382 +1 row (apr-cli --test backend_refusal_case_table); the a8faded4b row is already on main | S3a |
| 94 | `target-review/.rustc_info.json` | IDENTICAL (absent on both) | - | `9465af2c7` (#3025) | none | drop |
| 95 | `target-review/CACHEDIR.TAG` | IDENTICAL (absent on both) | - | `9465af2c7` (#3025) | none | drop |
| 96 | `tests/fixtures/parity/defective/one-position-at-0.5.json` | IDENTICAL | 0 | `74e3b2540` | none | drop |

Note on rows 2 (`ci.yml`) and 13 (`Cargo.lock`). ci.yml shows 403+/776− against main, but only 7 lines of that are
R-0b's own. Cargo.lock shows 297+/211− against main, and a 3-way merge leaves **zero** residual. So S1
has no Cargo.lock work.

## Slices (ordered)

Every slice is cut from current origin/main, not from `agent/R-0b`. **None of the 96 files is
`docs/roadmaps/roadmap.yaml`**, so no slice pays the roadmap merge-contention tax, provided each slice's
`pmat work` bookkeeping stays out of the slice PR. The ci.yml:570 explicit `--test` list is cited
as "the :570 line".

| id | content | files | depends on | conflicts with main | roadmap | proving target | dark? |
|----|---------|------:|-----------|---------------------|---------|----------------|-------|
| **S1** mechanical | **empty.** Cargo.lock, `crates/apr-cli/Cargo.toml`, `complexity_baseline.txt`, `surface_audit.csv`, `impl-estimates.jsonl` all have 0 residual. The two remaining mechanical rows (`claim_literal_baseline.txt` −1 for `serve/mod.rs:209`, and the tree-reader row) are *derived from* S3 code, and landing them first would turn their ratchets red | 0 | — | — | no | — | — |
| **S2** contract moves | **empty as a standalone PR.** The only contract move is dropping the zombie (a no-op on main). The contract *additions* (REG-OB-004, REG-F-003, `binding_registry` rows) name `registry.rs`, `backend_refusal_case_table.rs` and `check_backend_registry.sh`, none of which exist on main. They ride with the code that discharges them (Decision D2) | 0 | — | — | no | — | — |
| **S3a** resolution library + its falsifier (additive; no call site changes behaviour) | `crates/apr-cli/src/registry.rs` (new, 579), `commands_enum.rs` (+6, `#[path] pub mod registry`), `error.rs` (+8 `BackendUnavailable`/14), `commands/devices.rs` (`reserve_override` → `pub(crate)`), `tests/backend_refusal_case_table.rs` (new, 130), `scripts/tree_reader_tests.txt` (+1 row, regenerated), `contracts/apr-backend-registry-v1.yaml` (9610dd55c hunk: REG-OB-004 + REG-F-003 case-table half + `resolution`/`resolution_case_table` binding rows), `.github/workflows/ci.yml` (append `&& cargo test -p apr-cli --test backend_refusal_case_table` to the :570 line) | 8 | nothing (R-0a is on main: `ad8f98768`) | 5: `error.rs` (content), `devices.rs` (add/add), `ci.yml` (content), `tree_reader_tests.txt` (derived — regenerate, never hand-merge), contract (add/add) | no | `cargo test -p apr-cli --test backend_refusal_case_table` (5 rows) + `cargo test -p apr-cli --lib registry` (workspace-test) + `pv validate contracts/apr-backend-registry-v1.yaml` | case table **dark on main until this slice's ci.yml :570 edit**; `--lib` is covered by workspace-test |
| **S3b** call sites read the registry + `selected:` line + static guard (closes #3040) | `accel.rs`, `commands/bench.rs`, `commands/finetune.rs`, `serve/handler_gpu_completion.rs`, `serve/handlers_include_01.rs`, `serve/mod.rs` (9610dd55c/99589f98f/805342e18/245ca32ef hunks), `serve/tests_offload_report_pp14.rs`, `dispatch.rs`, `dispatch_analysis.rs`, `help_producer_truth.rs`, `lib.rs`, `commands/chat_load_tokenizers.rs`, `commands/chat_generate_session_02.rs` (805342e18 +3 only), `tests/fixtures/registry/no-accelerator-compiled.json` (new, used by serve tests), `scripts/check_backend_registry.sh` (new), `scripts/claim_literal_baseline.txt` (−1), `ci.yml` (`--static` step only; see D4) | 17 | S3a | 3: `chat_load_tokenizers.rs` (main moved it: `bad5869cf` F-1, `cb829fcb9`), `chat_generate_session_02.rs` (same), `ci.yml`. `dispatch.rs`/`help_producer_truth.rs`/`lib.rs`/`accel.rs`/`bench.rs`/`serve/mod.rs` are **clean**: main has not touched them since the merge-base | no | workspace-test `--lib` (apr-cli `serve::tests` over fixture registries, dispatch); guard-tree auto-runs `check_backend_registry.sh --self-test`; `check_backend_registry.sh --static` → must print 0 (main has **17** `cfg!(feature="cuda"\|"wgpu")` reads in `crates/apr-cli/src`, R-0b leaves 1, which is a comment in registry.rs) | `--static` is **dark until an ARG-wired ci.yml line exists** (guard_tree runs only `--self-test`); `--lib` not dark |
| **S3c** effective-config `resolved` + runtime-fallback refusal | `crates/aprender-serve/src/api/effective_config.rs` (+54/−2), `api/tests/effective_config_route_pp2.rs` (+32/−1), `serve/mod.rs` (a0269f3c1 hunk 27+/8−), `commands/chat_generate_session_02.rs` (5b5116d6c 25+/6−), `commands/run_entry.rs` (+13), contract (a0269f3c1 + 5b5116d6c hunks: REG-F-003 effective-config/after_generation text, non-goal line) | 6 | S3b (`serve/mod.rs` hunks stack; `after_generation` is called from the chat/run sites S3b routes) | 3: `effective_config_route_pp2.rs` (main moved it in `74e3b2540`), `chat_generate_session_02.rs`, contract | no | `cargo test -p aprender-serve --lib effective_config_route_pp2` + `cargo test -p apr-cli --lib registry::` (`a_forced_accelerator_that_fell_to_cpu_at_runtime_is_refused…`), both in workspace-test | not dark |
| **S4** row docs / receipt | `.pr/R-0b/{accept.sh,discover.json,plan.md,quorum.md}`, `docs/audits/impl-PMAT-1073-receipt.md`, `docs/audits/quorum/PMAT-1073-lane-1.json` | 6 | S3c (the receipt cites all three) | 0 | no | guard-tree (receipt-marker guard); `accept.sh` re-run on the S3c tree | n/a (docs) |

Residual file universe: S3a 8 + S3b 16 new + S3c 3 new + S4 6 = **33** (ci.yml, serve/mod.rs,
chat_generate_session_02.rs and the contract are shared between slices).

### Why the order is S3a → S3b → S3c and not the commit order

- S3a is purely additive. `registry.rs` is `pub mod` and is imported by the integration test
  (`use apr_cli::registry::{resolve_in, Request}`), so it is not dead code, and its only `crate::` needs are
  `error::CliError` and `commands::devices::reserve_override`, both in S3a. `apr_cli::BACKEND_VALUES`
  already exists on main (`commands_enum.rs`, `lib_parse_serve.rs`). So S3a builds and proves the
  never-downgrade rule without changing one runtime path.
- S3b is the behaviour change plus the #3040 decomposition. `99589f98f` decomposed
  `dispatch_runtime_commands` (cog 43), `dispatch_diagnostic_commands` (30) and
  `help_producer_truth::resolve` (73) *because* the pre-commit complexity hook charges the whole include!
  tree of any staged file (see D3).
- S3c stacks on S3b's `serve/mod.rs` and reads `Resolved` from S3a.
- S4 last: its receipt and `accept.sh` describe S3a–S3c. The claims in it were measured on the R-0b tree
  and must be re-measured on the slice trees, not copied.

## Decisions needed (not resolved here)

**D1 — the zombie contract.** `contracts/apr-gpu-cpu-parity-v1.yaml` (172 lines) was added by `74e3b2540`
(#3026) and deleted by `9465af2c7` (#3025). That commit also grew `apr-cpu-vs-gpu-output-parity-v1.yaml` by 80 lines,
so it reads as a fold. #3041's copy is byte-identical to the one #3026 added, and no R-0b commit touches it. A merge of
`agent/R-0b` silently restores it.
*Recommended:* drop it from every slice, because main deleted it on purpose after it landed. *Open check,
separate from this split:* the ids `PAR-OB-001..003` occur **nowhere** in `contracts/` or
`docs/specifications/` on main. Whether the fold kept those obligations under other ids is unverified.

**D2 — "contract moves first" vs a contract that cites code that does not exist yet.** The ruling orders
contract moves before code. The only real contract change in the residual is *new* obligations
(REG-OB-004 `discharged_by: falsification_tests[2]`, REG-F-003, `binding_registry.resolution` /
`static_guard`) whose bindings are S3 files.
*Recommended:* no standalone S2 PR. Each hunk lands in the slice that discharges it (S3a, then S3c).
Why: a contract that claims a discharge by a test not in the tree is a radar, not a ratchet.
`pv validate` would accept it, and nothing would be proved.

**D3 — #3040 is done on the branch but OPEN on GitHub, and #3041's body contradicts its own branch.** The
PR body says `status: partial`, "`dispatch.rs`/`lib.rs` pristine", A1 `--static` RED. The branch tip
says otherwise: `99589f98f` "…closes #3040" converts `dispatch.rs:173` and `lib.rs:230` and decomposes the
three functions, and R-0b leaves 0 code reads. Issue #3040 is OPEN.
*Recommended:* carve the three decompositions out of `99589f98f` into a behaviour-preserving
**S3b-0** that closes #3040, ahead of S3b's conversion. Why: `dispatch.rs` (+183/−148) and
`help_producer_truth.rs` (+84/−65) are the largest hunks in the residual and the most exposed to main
drift, and a refactor-only PR is judged by "same tests, same output", which is cheap to review. The cost is
hand-separating one mixed commit. If that proves unclean, keep it inside S3b and say so in the PR
body. Either way, each slice PR body is written from measurement, never copied from #3041's.

**D4 — how the new guard is wired, and its ripgrep dependency.**
(a) R-0b adds a `--self-test` ci.yml step anchored after `check_receipt_complete.sh --selftest`. That anchor
no longer exists on main. More importantly, `scripts/guard_tree.sh` (ci.yml job `guard-tree`, :780/:906) already runs
`--self-test` for every `git ls-files 'scripts/check_*.sh'`, and `check_guards_are_wired.sh` requires
each such guard to be named somewhere. So the self-test step is redundant, and only the ARG-wired
`--static` call needs a workflow line.
(b) `check_backend_registry.sh` requires `rg` and exits 2 (ENV) without it. **No** `scripts/check_*.sh` on
main invokes `rg`, **no** workflow provisions ripgrep, and `guard-tree` runs on
`[self-hosted, Linux, clean-room]` (intel/yoga/gx10). Its `--self-test` would then be an env-death inside
guard-tree.
*Recommended:* before S3b, port `scan()` to `grep -rnE` / `git grep -nE` (the regex is ERE-compatible) and
drop the self-test step. Why: it removes a new runner dependency, and the self-test step duplicates guard_tree.
Every slice's ci.yml edit (S3a's :570 token, S3b's `--static` step) is a workflow change and is
an operator check-in item under CLAUDE.md.

**D5 — #3041's L0-1a evidence commits after #3026 merged** (`176b986eb` gx10, `f48cd90dd`, `48053dad8`,
`d4e1f416c`): all of their content is already on main. The gx10 records and n5 series are IDENTICAL via
`74e3b2540`, and `reg15_admission` is on the :570 line. Nothing to decide beyond confirming "drop". Listed so no
slice re-adds them.

## Exact commands

S1 is empty. This proves it (the trial merge leaves zero residual on every mechanical file):

```bash
W=$(mktemp -d)/probe
git worktree add --detach "$W" origin/main
git -C "$W" -c user.name=probe -c user.email=probe@invalid merge --no-commit --no-ff origin/agent/R-0b || true
git -C "$W" diff --cached --numstat origin/main -- Cargo.lock crates/apr-cli/Cargo.toml \
    scripts/complexity_baseline.txt docs/audits/surface_audit.csv docs/audits/impl-estimates.jsonl
# expected: no output (0 residual)
git -C "$W" merge --abort; git worktree remove --force "$W"
```

The first non-empty slice (S3a), cut from main:

```bash
git fetch origin main agent/R-0b
git worktree add -b PMAT-1073-s3a-resolution-library <wt> origin/main
cd <wt>
# new files, whole, from the branch tip
git checkout origin/agent/R-0b -- crates/apr-cli/src/registry.rs crates/apr-cli/tests/backend_refusal_case_table.rs
# R-0b-own hunks only, 3-way onto main
git show 9610dd55c -- crates/apr-cli/src/commands_enum.rs crates/apr-cli/src/error.rs \
    crates/apr-cli/src/commands/devices.rs contracts/apr-backend-registry-v1.yaml | git apply -3
# derived registry: regenerate, never hand-merge
bash scripts/check_tree_reader_tests.sh --update
# ci.yml :570 — append `&& cargo test -p apr-cli --test backend_refusal_case_table` (workflow edit: operator check-in)
git diff --stat origin/main   # expect the 8 S3a files
```

The registry.rs at the tip also contains 805342e18's `selected_line` and 5b5116d6c's `after_generation`.
Both are `pub` and self-contained (no new `crate::` needs), so S3a takes the tip file whole. The only
alternative is reconstructing an intermediate file that never existed.

## Close-out for #3041 (for the orchestrator; nothing here was written to GitHub)

Suggested pointer: "Superseded. R-0a landed as `ad8f98768` (#3004) and L0-1a as `74e3b2540` (#3026); 63 of 96
files are identical or stale against main. The R-0b residual (33 files) ships as S3a → S3b(-0) → S3c → S4 per
`docs/audits/pr-3041-split-plan.md`." No-close for #3002 / #3040 until their slices land.
