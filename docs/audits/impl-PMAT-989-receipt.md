---
status: complete
ticket: PMAT-989
row: R-0
issue: 2904
epic: 2873
turns: "P0-P3 [U] (not instrumented); resume 31 (counted from the transcript at write time)"
model: "P0-P3 orchestrator claude-fable-5-1 (direct — R-0 is a Fable-owned design row per the driver's ROUTING); design quorum: agy delegate (opus) driving 3 agy lanes"
tokens_used: "design-quorum delegate 74787; orchestrator [U] (not instrumented); resume (inst:B, claude-opus-5) [U] (not instrumented)"
wall_clock_s: "[U] (not instrumented); design quorum dispatched ~03:05Z, R-0a contract commit ~05:15Z on 2026-09-06 (about 7,800 s, R-3's review round interleaved); resume 17:33Z-19:20Z 2026-09-06 = 6,420 s, of which ~4,800 s is one workspace-test cycle on the mutant"
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
| P_4 | contracts `apr-backend-registry-v1.yaml`, `apr-devices-schema-v1.yaml` (8df35191c) | `pv validate` · repo guards | direct |
| P3 review | 3-lane review-only on the diff | verdicts in the PR body / this receipt | delegate |

## What lands
- `trueno::registry` (`crates/aprender-compute/src/registry/`): `BackendKind::ALL = {cpu, cuda, wgpu, metal, hip}`; `BackendEntry {kind, api, device_index, device_uid, device_name, vendor, vendor_id: Option, device_type, mem_total, mem_free, mem_kind: Discrete | Unified{working_set_limit}, compute_class, caps, source: CompiledIn | Dlopen(path) | NotCompiled | Fixture(path), status: Ready | Unavailable(NotCompiled | DriverNotFound | NoDevice | NoBackend | ProbeFailed | ReserveExceedsFree), transport}`; the object-safe `BackendFactory` (`kind()`, `discover()`) with `MockBackendFactory`; `BackendRegistry::{discover, discover_with, from_fixture_json, with_reserve, select_default, distinct_devices, render_block, to_json}`; the CUDA factory through `trueno_gpu::driver` (`libcuda.so.1`, never cudart — REG-2), the wgpu factory over adapters with `transport` (a software rasteriser is `NoBackend`, never a GPU); `cpu` always Ready; every kind an explicit line (REG-11); REG-7 reserve (3,584 MiB `[U] default until master row 6 measures vram_peak`) as `ReserveExceedsFree{reserve, free}` **propagated across every entry sharing a `device_uid`** (a card refused for memory through the cuda driver is refused through its wgpu twin too — found by the catalogue, see Jidoka); REG-8 selection printed with its reason; REG-12 nothing persisted, fixtures named in `source`.
- `apr devices [--json]` (an `ExtendedCommands` variant; category `hardware` in `contracts/apr-cli-commands-v1.yaml`, 111 commands): prints the block or the `apr-devices-v1` JSON; overrides `APR_RESERVE_BYTES` (`<n|nK|nM|nG>`) and `APR_REGISTRY_FIXTURE` are printed when active; a malformed override is exit 4 naming the variable; discovery never fails the process (REG-1).
- Case table (10 rows, hermetic) + CLI failure catalogue (10 rows on fixture registries, every hermetic row with a must-RED twin: `tests/fixtures/registry/defective/missing-metal-line.json`), both wired into ci.yml's integration line. `contracts/schemas/apr-devices-v1.schema.json` validated with the `jsonschema` crate on the fixtures and on the running machine.
- Contracts: `apr-backend-registry-v1.yaml` (invariants (i) and (iii) discharged; (ii), (iv), (v) are R-0b's and are NOT claimed) and `apr-devices-schema-v1.yaml`. The README counts are NOT written here: G-11 (#3020) made them a
  ratchet the orchestrator regenerates, and `check_row_pr_write_set.sh` refuses a count line on a row
  branch, so this branch's earlier 1811→1814 / 110→111 bump was dropped when origin/main was merged
  in (the README may lag, never overstate — FALSIFY-README-005/007).
- The design-quorum amendment (DAG R-0 = R-0a, new R-0b #3002 / PMAT-1060, expiry moves, spec §12 llamafile citation) rides as #3003 and is also the first commit of this branch.

## Verification (orchestrator, every command re-run at 8df35191c)
| check | result |
|---|---|
| `cargo test -p aprender-compute --test registry_case_table` | rc 0, 9/9 — no features, `--features gpu`, `--features cuda` |
| `cargo test -p apr-cli --test registry_failure_catalogue` | rc 0, 10/10 (after the review round) |
| `cargo test -p apr-cli --test cli_commands` (FALSIFY-CLI-001..005 with `devices`) | rc 0, 15/15 |
| `cargo test -p aprender-compute --lib` (CI's own step for this crate) | rc 0: 3,510 passed, 4 ignored |
| `cargo clippy -p aprender-compute --lib -- -D warnings` · `--features gpu` · `cargo clippy -p apr-cli --lib --bin apr -- -D warnings` · `cargo fmt --all -- --check` | rc 0 each (`--features cuda,gpu` trips a PRE-EXISTING `unused import GemmOp` at `matrix/ops/arithmetic.rs:413` on main too; not this diff) |
| `pv validate` + `pv lint` on both contracts · `check_contract_test_binding.sh` · `check_contract_enforcement.sh` · `check_readme_claims.sh` (FALSIFY-README-002/003) · `check_no_claim_literals.sh` · `check_perf_claims_cite_receipts.sh` · `check_roadmap_diff_additive.sh` · `check_dag_invariants.sh` · `render_dag.py --check` · `check_receipt_complete.sh --dag` · `check_guards_are_wired.sh` · `check_baseline_ratchets.sh` | rc 0 each |
| A (lambda): `apr devices --json \| jq -e '[.entries[]\|select(.status.state=="ready")]\|length>=1'` | true (default build: cpu Ready + wgpu/Vulkan sees the RTX 4090; `cuda` line reads `NotCompiled` because the default build has no cuda feature — an honest line, the D-9 question) |
| A (gx10, intel, mini): the same jq on `apr devices --json` from a build of this branch on each host | **true, true, true** — gx10: `NVIDIA GB10` Ready through wgpu/vulkan, `kind=unified` (the Vulkan loader printed a freedreno `VK_ERROR_INCOMPATIBLE_DRIVER` line on stderr for an unrelated device node; discovery did not abort — REG-1); intel: two `AMD Unknown (RADV NAVI10)` Ready through vulkan, llvmpipe listed as `NoBackend(software rasteriser …)` (S0-3); mini: `Apple M4` Ready through wgpu/metal, `kind=unified`; on all three `cuda` reads `NotCompiled` (default build) and `metal`/`hip` read `NoBackend`. Blocks pasted below. |
| A: `scripts/pv_bin.sh validate contracts/apr-devices-schema-v1.yaml` | valid |

## Four-host dogfood (printed blocks, default build of this branch)
```
gx10  backend: cpu    ready        aarch64 host cpu, 20 threads class=neon mem=122502MiB kind=unified source=compiled-in
gx10  backend: wgpu   ready        device[0]="NVIDIA GB10" kind=unified transport=vulkan caps={max_buffer_size=4503599627370496} source=compiled-in
gx10  backend: wgpu   unavailable  reason=NoBackend(software rasteriser (llvmpipe (LLVM 20.1.2, 128 bits)) is not a GPU) source=compiled-in
gx10  selected: wgpu device[0]  reserve=3584MiB basis=[U] default until master row 6 measures vram_peak  (first Ready non-cpu entry; 1 physical device(s) Ready)
intel backend: cpu    ready        x86_64 host cpu, 32 threads class=avx512 mem=289934MiB kind=unified source=compiled-in
intel backend: wgpu   ready        device[0]="AMD Unknown (RADV NAVI10)" kind=discrete transport=vulkan caps={max_buffer_size=2147483647} source=compiled-in
intel backend: wgpu   ready        device[1]="AMD Unknown (RADV NAVI10)" kind=discrete transport=vulkan caps={max_buffer_size=2147483647} source=compiled-in
intel backend: wgpu   unavailable  reason=NoBackend(software rasteriser (llvmpipe (LLVM 15.0.7, 256 bits)) is not a GPU) source=compiled-in
mini  backend: cpu    ready        aarch64 host cpu, 10 threads class=neon kind=unified source=compiled-in
mini  backend: wgpu   ready        device[0]="Apple M4" kind=unified transport=metal caps={max_buffer_size=9534832640} source=compiled-in
(every host also prints the cuda NotCompiled, metal NoBackend and hip NoBackend lines)
```
**Finding (REG-9, intel)**: the two W5700X cards enumerate with ONE name through wgpu, so they shared a `device_uid` and `distinct_devices()` counted **1** — two different cards collapsed into one. Fixed on this branch: same-named entries within one API are disambiguated by ordinal (`#k`); case-table row 10 (`two_different_cards_with_one_name_stay_two_devices`, RED before the fix). Cross-API twins still match by ordinal.

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
| P3 review | delegate (agy quorum, width 3) | ac34fe7e671ff6cd3 (agy ff457afe…, 904c34f5…, 141e9820…) | 3/3 FAIL = mergeable-with-changes; applied below |
Slots ≤ 2 live at any instant; denials 0.

## Review quorum (3-lane, review-only, 2026-09-06) — claims re-run, then applied
3/3 lanes: mergeable with changes; no lane called the approach wrong. Unanimous positives (re-read, held): invariant (i) cpu always Ready and (iii) non-empty reason across the cfg combinations; the CUDA factory never aborts (REG-1); contexts are dropped. Claims settled by re-running or re-reading:
- **`devices` duplicated in `contracts/apr-cli-commands-v1.yaml`** (lanes 1+2) — TRUE (an aborted edit script inserted it twice; FALSIFY-CLI-005 counts the Rust list, not the YAML, so it did not catch it). Fixed: 111 entries, matching `apr --help`. Lane 3's "111 is accurate" was reading `apr --help`; lane 1's "~120" was wrong.
- **Dead arms in `cuda.rs`** (lane 1) — TRUE: `cuda_available()` is `device_count().map(|c| c > 0)`, folding a load failure, a probe error and zero devices into one bool. Fixed: `CudaDriver::load()` first (`DriverNotFound`), then `device_count()` (`ProbeFailed` / `NoDevice`) — three reachable reasons.
- **Schema too permissive** (lanes 2+3, two distinct defects) — TRUE: nested objects lacked `additionalProperties: false`, and `unavailable` never required the per-kind payload nor did `ready` forbid `kind`. Fixed, with a twin row (`schema_twin_documents_serde_refuses_do_not_validate`: driver-not-found without `path`, a ready entry with `kind`, a stray field — all refused). Lane 1's "robust" was wrong.
- **Override lines printed from the request** (lanes 2+3) — TRUE in spirit: the lines echoed the env value. Now they print what the registry holds (`reserve_bytes`/`basis`, `source`).
- **`#[cfg(feature = "inference")]` on `apr devices` vs an unconditional `registered_commands()`** (lane 1) — the existing precedent: `gpu`, `train`, `code` are cfg-gated and listed unconditionally; the CLI contract describes the default feature set. Not changed; recorded.
- **`apply_reserve` pass 2 keys on `device_uid`, which can differ across APIs** (lane 2 vs lanes 1+3) — TRUE as a limitation: both factories derive the uid from vendor + normalised name; a host whose APIs name one card differently gets two uids and no propagation, and `distinct_devices()` counts two. Recorded as a contract non_goal and a dogfood check, not hidden.
- **Help-parser decomposition not behaviour-preserving** (lane 1 vs lane 3) — FALSE: the original `help_subcommands` had neither the blank-line break nor the lowercase filter (it filtered only `help`); the decomposition keeps each parser's own rules. 15/15 unchanged.
- **A CUDA context per device during a read-only probe** (lane 3) — TRUE: `cuMemGetInfo` needs a current context; recorded as a contract non_goal (created and dropped; a process-level "no side effects", not a driver-level one).

## Jidoka
- **The catalogue's FX-7 row caught a real gap in the library** (the case table had passed): with a fixture where the cuda entry knows its free memory and its wgpu twin does not, the cuda entry was refused for the reserve and the wgpu twin of the SAME card stayed Ready and got selected. Fixed by propagating the refusal across entries sharing a `device_uid` with the sibling's measured figure; the case table gained row 9 (RED before the fix).
- **The pre-commit complexity gate refuses any file carrying pre-existing debt**: `crates/apr-cli/tests/cli_commands.rs` (`get_help_commands` cognitive 47, `help_subcommands` 36 — on main too) blocked the one-line `registered_commands` addition; both `--help` parsers were decomposed into `help_block` / `row_name` / `looks_like_command` in the same commit (15/15 still). `crates/apr-cli/src/dispatch.rs` (`dispatch_runtime_commands` cognitive 41, `dispatch_diagnostic_commands` 30) blocked the dispatch arm; `apr devices` is therefore an `ExtendedCommands` variant dispatched from `dispatch_analysis.rs` (0 violations). Neither debt was discharged with `#[allow]` or `--no-verify`.
- The card's A writes the jq as `[.[]|select(.status=="ready")]` (a bare array); the document is an object (`entries[].status.state`) so the schema can carry `source`, `reserve` and `selected` — the DAG row's A is corrected in the write-back.
- `wgpu` exposes no VRAM figure: `max_buffer_size` is an allocation cap, reported as a capability string, `mem_total` stays unknown for wgpu entries (a first draft printed the cap as memory: "4294967296MiB").
- `git add` with one non-existent pathspec stages nothing (the pathspec error aborts the whole add) — two commits appeared empty until re-added; and `git checkout -- <file>` restores the INDEX version, so a staged edit survives it (use `git restore --staged --worktree`).

## Gaps
- **Dogfood legs**: all four done on the default build (lambda, gx10, intel, mini — blocks above); the intel leg found the REG-9 uid collision, fixed here. The `--features cuda` leg on lambda: the case table's cuda arm passed (the CUDA factory saw the RTX 4090 via `libcuda.so.1`); the `apr` binary with `--features cuda` is the D-9 question, not this row's.
- The non-hermetic fixtures FX-1/3/12 (Linux-only: `LD_PRELOAD` stub, read-only `$TMPDIR`, `strace`) and FX-2/4/5/6/8/9 (real drivers, real cards, root) are not in CI by the quorum's decision; they become host-dogfood rows recorded as receipts.
- `caps` is a name list; cuBLAS-as-a-capability (REG-5) lands with R-0b's effective-config.
- REG-1's "a driver fault never takes the process down" for the wgpu path (`DEVICE_INIT_LOCK` on a broken EGL driver, lane 1's open question) is unproven here.
- Receipt for this PR: advisory, not produced (driver A1).

## Estimates
K̂ 5 (`basis=first-run[U]`); actual: design quorum 1 delegate dispatch; P_1–P_4 ≈ 22 orchestrator bash calls; review round 1 delegate dispatch + 4 bash calls (`basis=this receipt`). Rows appended to `docs/audits/impl-estimates.jsonl`.

## Resume (2026-09-06, driver v6, inst:B — Opus 5)
The row was implemented and reviewed above, then sat with `gate` RED. What the
resume found and did, each item judged by a command, not by intent:

| # | finding at head `caa0e684d` | fix | judged by |
|---|---|---|---|
| 1 | `gate` RED. It is a fan-in job; the cause was `guard-runner-labels` → `check_complexity_ratchet.sh`: **2 STALE rows**, `cli_commands.rs::get_help_commands` and `::help_subcommands`. This PR's own decomposition of the two `--help` parsers (Jidoka above) dropped them under both thresholds, and the ratchet refuses a kept row for a fixed function — "the next regression at that coordinate lands for free" | the two rows deleted | `check_complexity_ratchet.sh` rc=0, 689 recorded offenders, **2 removed**, none new, none grown, none stale |
| 2 | the branch was 2 commits behind a main that had moved under it: G-10 (#3011) and **G-11 (#3020)** | `origin/main` merged in | merge commit `4e90fb4ea` |
| 3 | G-11 landed `scripts/check_row_pr_write_set.sh` AFTER this branch was cut: a row branch `agent/<row>` may not write a README count line. This branch carried `1811→1814` contracts and `110→111` commands | the README conflict resolved to main's side; `git diff origin/main HEAD -- README.md` is empty | `check_row_pr_write_set.sh --branch agent/R-0 --event pull_request` → PASS, "row PR agent/R-0 writes no shared file (27 changed paths)" |
| 4 | the mutation evidence was LOCAL only; driver v6 requires the mutant PUSHED and the PR's own CI RED | mutant `4a66e20a7` → revert | run ids in the table below |

**Consequence of (3), stated rather than hidden:** the README now UNDERSTATES —
it claims 1811 contracts against 1814 present and 110 commands against 111. That
is legal and deliberate: FALSIFY-README-005/007 make the counts a ratchet ("may
lag, never overstate") and the orchestrator's docs commit regenerates them with
`scripts/check_readme_claims.sh --exact` after the merge. The book chapter
`book/src/cli/devices.md` and its page contract stay, because the book parity
falsifier (FALSIFY-BOOK-CLI-PARITY-001) is RED without them.

**Mutation — pushed, PR CI RED, reverted, PR CI GREEN**

| | commit | mutation | CI run | result |
|---|---|---|---|---|
| RED | `4a66e20a7` | `apply_reserve` pass 1's refusals cleared before pass 2, so a reserve refusal does not propagate to the same device's other-API entries | [34049785730](https://github.com/paiml/aprender/actions/runs/34049785730) — job `workspace-test` `101531455145`, step 11 "Integration tests" | **FAILURE**, exit 101. `fx7_reserve_exceeding_free_memory_is_a_named_refusal` FAILED (9 passed; 1 failed). The printed block shows the defect exactly: `cuda unavailable reason=ReserveExceedsFree{reserve=1072668082176, free=21474836480}` and then `selected: wgpu device[0]` — the same RTX 4090 the driver had just refused |
| GREEN | this commit (a `git revert` of `4a66e20a7`) | the mutant reverted, nothing else changed | the run on this commit; id in the PR body (a run id cannot be written into the commit it names) | `gate` + `workspace-test` **SUCCESS** |

The mutant is one line and moves exactly one test
(`a_reserve_refusal_propagates_to_the_devices_other_api_entries`: "the twin
carries the sibling's measured free memory: Ready"), so the RED is attributable
to it and to nothing else. It was aimed at the job that actually runs the test:
the integration line carrying `--test registry_case_table` is in
**`workspace-test`** (ci.yml:408), not in `ci / test` — `ci / test` went GREEN on
the mutant commit and proves nothing here.

**Not fixed, stated:** `present` (workflow `pr-review-quorum`) is RED — it wants
a signed receipt at `evidence/pr-review/3004` and there is none. It is **not** one
of the two required checks (`gate`, `workspace-test`), and #3020 merged with it
red on 2026-09-06. Recorded, not routed around.

## Verdict
COMPLETE (`status: complete`). `.pr/R-0/accept.sh` — every A_i as a command, its
status read directly and never through a pipe — is **15/15 green** on the reverted
tree, including A8, which runs the binary built from HEAD through
`. scripts/apr_bin.sh` (never a bare `apr`: four have coexisted on this box) and
asserts `apr devices --json` carries at least one Ready entry. Auto-merge armed on
the two required checks, `gate` and `workspace-test`.
