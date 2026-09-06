---
status: partial
partial_reason: "this PR is not yet merged on the required check; the gx10/intel/mini dogfood legs of A are pending (lambda done); flip to complete with the DAG status write-back after merge"
ticket: PMAT-989
row: R-0
issue: 2904
epic: 2873
model: "orchestrator claude-fable-5-1 (direct — R-0 is a Fable-owned design row per the driver's ROUTING); design quorum: agy delegate (opus) driving 3 agy lanes"
tokens_used: "design-quorum delegate 74787; orchestrator [U] (not instrumented)"
wall_clock_s: "[U] (not instrumented); design quorum dispatched ~03:05Z, R-0a contract commit ~05:15Z on 2026-09-06 (about 7,800 s, R-3's review round interleaved)"
---
# impl-PMAT-989 — R-0 (= R-0a after the split) · BackendRegistry: probe → enumerate → print (#2904; spec §5 R-0, REG-1..14; design quorum 2026-09-06)

## Identity
ticket PMAT-989 · kind code · branch `agent/R-0` (worktree, claim held) · base `65680cdf8` (+ `origin/main` merged at d36c409bf) · `discover.json` at `$XDG_RUNTIME_DIR/paiml-implement/discover-R-0.json` (`gate_cmd_fallback=true`) · quorum: **N-lane design before P2** (done: 3 lanes, 3/3 implement-with-changes, 3/3 split — `docs/audits/pp-066-r0-design-quorum.md`; also PR #3003) then 3-lane review-only at P3 · K̂ = 5 (`basis=first-run[U]`, first design row).

## Plan (P1, from the card and the quorum's decisions)
| phase | scope | A_i | routing |
|---|---|---|---|
| P_1 | `crates/aprender-compute/tests/registry_case_table.rs` (RED first, 1ef2a22b3) | `cargo test -p aprender-compute --test registry_case_table` | direct |
| P_2 | `crates/aprender-compute/src/registry/{mod,cuda,wgpu_probe}.rs` + `lib.rs` export (5795ccc46) | same, under no features / `--features gpu` / `--features cuda` | direct |
| P_3 | `apr devices [--json]` (`crates/apr-cli/src/commands/devices.rs`, `ExtendedCommands::Devices`, dispatch in `dispatch_analysis.rs`), fixtures + `registry_failure_catalogue.rs`, JSON schema, ci.yml targets, CLI contract + `registered_commands` (121fa439b) | `cargo test -p apr-cli --test registry_failure_catalogue` · `--test cli_commands` | direct |
| P_4 | contracts `apr-backend-registry-v1.yaml`, `apr-devices-schema-v1.yaml`; README counts (8df35191c) | `pv validate` · repo guards | direct |
| P3 review | 3-lane review-only on the diff | verdicts in the PR body / this receipt | delegate |

## What lands
- `trueno::registry` (`crates/aprender-compute/src/registry/`): `BackendKind::ALL = {cpu, cuda, wgpu, metal, hip}`; `BackendEntry {kind, api, device_index, device_uid, device_name, vendor, vendor_id: Option, device_type, mem_total, mem_free, mem_kind: Discrete | Unified{working_set_limit}, compute_class, caps, source: CompiledIn | Dlopen(path) | NotCompiled | Fixture(path), status: Ready | Unavailable(NotCompiled | DriverNotFound | NoDevice | NoBackend | ProbeFailed | ReserveExceedsFree), transport}`; the object-safe `BackendFactory` (`kind()`, `discover()`) with `MockBackendFactory`; `BackendRegistry::{discover, discover_with, from_fixture_json, with_reserve, select_default, distinct_devices, render_block, to_json}`; the CUDA factory through `trueno_gpu::driver` (`libcuda.so.1`, never cudart — REG-2), the wgpu factory over adapters with `transport` (a software rasteriser is `NoBackend`, never a GPU); `cpu` always Ready; every kind an explicit line (REG-11); REG-7 reserve (3,584 MiB `[U] default until master row 6 measures vram_peak`) as `ReserveExceedsFree{reserve, free}` **propagated across every entry sharing a `device_uid`** (a card refused for memory through the cuda driver is refused through its wgpu twin too — found by the catalogue, see Jidoka); REG-8 selection printed with its reason; REG-12 nothing persisted, fixtures named in `source`.
- `apr devices [--json]` (an `ExtendedCommands` variant; category `hardware` in `contracts/apr-cli-commands-v1.yaml`, 111 commands): prints the block or the `apr-devices-v1` JSON; overrides `APR_RESERVE_BYTES` (`<n|nK|nM|nG>`) and `APR_REGISTRY_FIXTURE` are printed when active; a malformed override is exit 4 naming the variable; discovery never fails the process (REG-1).
- Case table (9 rows, hermetic) + CLI failure catalogue (9 rows on fixture registries, every hermetic row with a must-RED twin: `tests/fixtures/registry/defective/missing-metal-line.json`), both wired into ci.yml's integration line. `contracts/schemas/apr-devices-v1.schema.json` validated with the `jsonschema` crate on the fixtures and on the running machine.
- Contracts: `apr-backend-registry-v1.yaml` (invariants (i) and (iii) discharged; (ii), (iv), (v) are R-0b's and are NOT claimed) and `apr-devices-schema-v1.yaml`; README 1811 → 1813 contracts, 110 → 111 commands.
- The design-quorum amendment (DAG R-0 = R-0a, new R-0b #3002 / PMAT-1060, expiry moves, spec §12 llamafile citation) rides as #3003 and is also the first commit of this branch.

## Verification (orchestrator, every command re-run at 8df35191c)
| check | result |
|---|---|
| `cargo test -p aprender-compute --test registry_case_table` | rc 0, 9/9 — no features, `--features gpu`, `--features cuda` |
| `cargo test -p apr-cli --test registry_failure_catalogue` | rc 0, 9/9 |
| `cargo test -p apr-cli --test cli_commands` (FALSIFY-CLI-001..005 with `devices`) | rc 0, 15/15 |
| `cargo test -p aprender-compute --lib` (CI's own step for this crate) | rc 0: 3,510 passed, 4 ignored |
| `cargo clippy -p aprender-compute --lib -- -D warnings` · `--features gpu` · `cargo clippy -p apr-cli --lib --bin apr -- -D warnings` · `cargo fmt --all -- --check` | rc 0 each (`--features cuda,gpu` trips a PRE-EXISTING `unused import GemmOp` at `matrix/ops/arithmetic.rs:413` on main too; not this diff) |
| `pv validate` + `pv lint` on both contracts · `check_contract_test_binding.sh` · `check_contract_enforcement.sh` · `check_readme_claims.sh` (FALSIFY-README-002/003) · `check_no_claim_literals.sh` · `check_perf_claims_cite_receipts.sh` · `check_roadmap_diff_additive.sh` · `check_dag_invariants.sh` · `render_dag.py --check` · `check_receipt_complete.sh --dag` · `check_guards_are_wired.sh` · `check_baseline_ratchets.sh` | rc 0 each |
| A (lambda): `apr devices --json \| jq -e '[.entries[]\|select(.status.state=="ready")]\|length>=1'` | true (default build: cpu Ready + wgpu/Vulkan sees the RTX 4090; `cuda` line reads `NotCompiled` because the default build has no cuda feature — an honest line, the D-9 question) |
| A (gx10, intel, mini) | PENDING — see Gaps |
| A: `scripts/pv_bin.sh validate contracts/apr-devices-schema-v1.yaml` | valid |

## Mutations (RED, then restored GREEN)
1. `missing_entry` no longer produces a line for a kind without a factory → case table 6/9 (`cpu_is_always_ready_and_every_kind_prints_a_line`, `json_carries_the_schema_top_level_keys` + one more FAILED) and catalogue 8/9 (`json_output_validates_against_the_schema…` FAILED). Restored → 9/9, 9/9.
2. Pass 2 of `apply_reserve` removed (no propagation to the device's other-API entries) → `a_reserve_refusal_propagates_to_the_devices_other_api_entries` FAILED (8/9). Restored → 9/9.
3. `cpu_entry` Unavailable → `cpu_is_always_ready_and_every_kind_prints_a_line` FAILED (8/9). Restored → 9/9.
4. `apr devices` prints no `override:` lines → `fx7…` and `fx11…` FAILED (7/9). Restored → 9/9.

## Dispatch ledger
| phase | mode | agent | notes |
|---|---|---|---|
| design quorum | delegate (agy quorum, width 3) | abe9d0e106e9e35b0 (agy 47b84e91…, bb7a2fd1…, ec3bc416…) | 3/3 implement-with-changes, 3/3 split; record + amendment (#3003) |
| P_1–P_4 | direct | — | Fable-owned row |
| P3 review | delegate (agy quorum, width 3) | see the PR body | review-only |
Slots ≤ 2 live at any instant; denials 0.

## Jidoka
- **The catalogue's FX-7 row caught a real gap in the library** (the case table had passed): with a fixture where the cuda entry knows its free memory and its wgpu twin does not, the cuda entry was refused for the reserve and the wgpu twin of the SAME card stayed Ready and got selected. Fixed by propagating the refusal across entries sharing a `device_uid` with the sibling's measured figure; the case table gained row 9 (RED before the fix).
- **The pre-commit complexity gate refuses any file carrying pre-existing debt**: `crates/apr-cli/tests/cli_commands.rs` (`get_help_commands` cognitive 47, `help_subcommands` 36 — on main too) blocked the one-line `registered_commands` addition; both `--help` parsers were decomposed into `help_block` / `row_name` / `looks_like_command` in the same commit (15/15 still). `crates/apr-cli/src/dispatch.rs` (`dispatch_runtime_commands` cognitive 41, `dispatch_diagnostic_commands` 30) blocked the dispatch arm; `apr devices` is therefore an `ExtendedCommands` variant dispatched from `dispatch_analysis.rs` (0 violations). Neither debt was discharged with `#[allow]` or `--no-verify`.
- The card's A writes the jq as `[.[]|select(.status=="ready")]` (a bare array); the document is an object (`entries[].status.state`) so the schema can carry `source`, `reserve` and `selected` — the DAG row's A is corrected in the write-back.
- `wgpu` exposes no VRAM figure: `max_buffer_size` is an allocation cap, reported as a capability string, `mem_total` stays unknown for wgpu entries (a first draft printed the cap as memory: "4294967296MiB").
- `git add` with one non-existent pathspec stages nothing (the pathspec error aborts the whole add) — two commits appeared empty until re-added; and `git checkout -- <file>` restores the INDEX version, so a staged edit survives it (use `git restore --staged --worktree`).

## Gaps
- **Dogfood legs**: gx10, intel and mini not yet run (the branch must be built there; S0 left worktrees on intel/mini). Recorded as PENDING in A; the lambda leg is done on the default build. The `--features cuda` leg on lambda: the case table's cuda arm passed (the CUDA factory saw the RTX 4090 via `libcuda.so.1`); the `apr` binary with `--features cuda` is the D-9 question, not this row's.
- The non-hermetic fixtures FX-1/3/12 (Linux-only: `LD_PRELOAD` stub, read-only `$TMPDIR`, `strace`) and FX-2/4/5/6/8/9 (real drivers, real cards, root) are not in CI by the quorum's decision; they become host-dogfood rows recorded as receipts.
- `caps` is a name list; cuBLAS-as-a-capability (REG-5) lands with R-0b's effective-config.
- REG-1's "a driver fault never takes the process down" for the wgpu path (`DEVICE_INIT_LOCK` on a broken EGL driver, lane 1's open question) is unproven here.
- Receipt for this PR: advisory, not produced (driver A1).

## Estimates
K̂ 5 (`basis=first-run[U]`); actual: design quorum 1 delegate dispatch; P_1–P_4 ≈ 22 orchestrator bash calls (`basis=this receipt`). Rows appended to `docs/audits/impl-estimates.jsonl`.

## Verdict
PENDING-MERGE (`status: partial`).
