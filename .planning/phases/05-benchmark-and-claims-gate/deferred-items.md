# Phase 5 — Deferred Items (out-of-scope discoveries)

Discoveries made while executing Phase 5 plans that are NOT caused by this phase's
changes. Logged rather than fixed, per the executor scope boundary.

> Merge note: plans 05-04 and 05-06 each created this file independently in their own
> worktrees (an add/add conflict at wave-1 merge-back). Both sets of entries are kept and
> renumbered; nothing was dropped in favour of one side.

---

## D-ITEM-05-01: `aprender-serve` test target `driver_cpu` does not compile

**Found during:** plan 05-04, Task 3 (`cargo check --workspace --all-targets` run to prove
the new `AprenderError::ZeroVarianceDifferences` variant broke no downstream match).

**Symptom:** 12 × `E0063` in `crates/aprender-serve/tests/driver_cpu.rs`:

```
error[E0063]: missing field `query_pre_attn_scalar` in initializer of `GGUFConfig`   (×10)
error[E0063]: missing fields `post_attn_norm_weight` and `post_ffw_norm_weight`
              in initializer of `OwnedQuantizedLayer`                                (×2)
error: could not compile `aprender-serve` (test "driver_cpu") due to 12 previous errors
```

**Proven pre-existing, not caused by 05-04:**

- `query_pre_attn_scalar` was added to `GGUFConfig` on 2026-06-19 in `366f3c275`
  (*fix(serve): batched-GPU path crashed on every GQA model*, PMAT-841, #2125), which is
  an ancestor of this plan's base `350b08575`. The test target has been red since then.
- The workspace check output contains **zero** occurrences of `AprenderError`,
  `non-exhaustive` or `ZeroVarianceDifferences`, so the added error variant is not
  implicated. Adding it broke nothing: only `error.rs`'s own `Display` match is
  exhaustive over that enum.

**Why not fixed here:** it is in a different crate, on a surface plan 05-04 does not
touch, and the fix is to update a fixture-construction site for two struct fields added
by a GQA dispatch change — an `aprender-serve` concern with its own review context.

**Suggested owner:** an `aprender-serve` ticket. Note that `make tier2`/`tier3` and the
`workspace-test` required status check may already be masking this by not building
`--all-targets` for that crate; if so, that gap is the more important half of the fix
(CR-02: a gate that runs nothing passes).

---

## D-ITEM-05-02: 24 pre-existing failures in `cargo test -p aprender-train --lib`

**Found during:** plan 05-06, while running the full `aprender-train` lib suite for
regression coverage. The plan's own scoped verify is green — `--lib classify_trainer`
(330 passed) and `apr-cli --lib --features setfit finetune` (81 passed).

Failing modules, none of which plan 05-06 touches:

| Module | Count | Example |
|--------|-------|---------|
| `gpu::ledger::tests` | 12 | `test_reserve_and_release` — `assertion left == right failed: left: 0, right: 8000` |
| `gpu::guard::tests` | 8 | `test_guard_update_actual` |
| `gpu::wait::tests` | 1 | `test_timeout_when_full` |
| `prune::snapshot_tests::tests` | 3 | `snapshot_all_prune_methods` |

**Evidence they are not caused by 05-06:**

- 05-06 modifies `finetune/classify_trainer.rs` (+ its tests) and apr-cli command files.
  `gpu/ledger.rs`, `gpu/guard.rs`, `gpu/wait.rs` and `prune/snapshot_tests.rs` are
  untouched and reference neither `TrainResult` nor `TrainingConfig`.
- `test_reserve_and_release` fails in ISOLATION (`cargo test -p aprender-train --lib
  test_reserve_and_release` → 0 passed; 1 failed), so it is not a parallel-execution
  interaction with the new tests either.
- The assertion is a VRAM reservation figure (0 vs 8000 MB) on a host with no reservable
  GPU ledger state — an environment dependency, not a logic regression.

**Action:** needs its own ticket (GPU-ledger tests assume a reservable GPU / writable
ledger directory; the prune snapshot tests need their snapshots reviewed). Not Phase 5
work.

**Gate relevance:** these 24 failures are the baseline for Phase 5's post-merge test
gate. A wave that reports exactly these 24 has introduced no regression; any *additional*
failure is merge-induced and must be treated as such. Recording the count here so a later
wave cannot quietly absorb a new failure into "the usual 24".

---

## D-ITEM-05-14-A — `apr_reload.rs` is unformatted at HEAD (PRE-EXISTING, out of scope)

Surfaced by plan 05-14's `cargo fmt -p aprender-train -- --check` gate.

`crates/aprender-train/src/train/setfit/apr_reload.rs:331` produces a rustfmt diff (a
`let recorded: ProvenanceRecord = …` binding rustfmt wants on one line). It is
**pre-existing**, not caused by 05-14: the working tree during that plan contained only
`evidence.rs`, and `apr_reload.rs` is byte-identical to `b3386b063`.

Left alone per the scope boundary — only issues DIRECTLY caused by the plan's own changes
are auto-fixed. `rustfmt --check` on `evidence.rs` alone is clean, which is what 05-14's
acceptance criterion asks for ("clean for this file").

**Action:** whoever next runs a crate-wide `cargo fmt` on `aprender-train` should land the
reformat as its own commit rather than folding it into a feature diff. Worth confirming
first whether it is rustfmt-version drift on this host rather than a genuine omission —
if it is drift, reformatting here would make CI red instead.

## D-ITEM-05-14-B — `cargo test` output is rewritten on this host, and the plan's greps assume it is not

Measured during 05-14, recorded because it will bite every later plan in this phase whose
verify block reads a libtest count.

The `rtk` hook rewrites `cargo test` output into a one-line summary
(`cargo test: 313 passed, 3 ignored, 7661 filtered out (1 suite, 37.59s)`). There is **no
`test result: ok. N passed` line anywhere in the captured log** — `grep -c "test result"`
returned **0** on a run that exited **0**. Any verify block asserting that line fails on a
GREEN run, and any count floor fed from it reads 0.

**Workaround in use:** run through `rtk proxy`, which yields the raw libtest stream. 05-14
used it for every `cargo test`, and for `git status` / `git diff` / `git log` in its
checks — the same standing Phase 2 ruling that porcelain-emptiness assertions must go
through `rtk proxy`.

**Action:** plans 05-03, 05-07, 05-09..05-13 should be read as the `rtk proxy` form
wherever they grep a libtest line. Worth deciding once, at phase level, rather than
rediscovering per plan.

## D-ITEM-05-14-C — `make test` cannot compile the workspace on macOS (PRE-EXISTING, blocks the post-merge gate)

Surfaced by the wave-1 post-merge test gate, which is `make test`
(`cargo nextest run --workspace -j 2`). It exited non-zero having run **zero** tests: the
`cargo test --no-run --workspace` compile aborted. Two independent, unrelated breaks:

1. `crates/aprender-profile/examples/{validate_golden_trace,process_tracer_demo}.rs` import
   `renacer::{validate, process_tracer}`, which are `#[cfg(target_os = "linux")]` in
   `crates/aprender-profile/src/lib.rs:86-87`. On macOS the imports do not resolve
   (`E0432`), plus an `E0282` behind them.
2. `crates/aprender-serve/tests/{gguf_config_coverage,gguf_kv_cache_coverage}.rs` build
   `GGUFConfig { .. }` literals that omit the `query_pre_attn_scalar` field (`E0063`, 9
   sites). The field exists in the struct; the test initializers were never updated.

**Proven pre-existing, not caused by 05-14** — three independent controls:
- The identical `aprender-profile` errors reproduce from a detached worktree at the base
  commit `b3386b063` (`cargo check -p aprender-profile --examples` → exit 101).
- All four failing files are byte-identical between `b3386b063` and the merge commit
  (`git rev-parse <rev>:<path>` matches on each).
- The merge changed exactly 4 files: 3 planning docs and
  `crates/aprender-train/src/train/setfit/evidence.rs`. Neither `aprender-profile` nor
  `aprender-serve` declares an `aprender-train` dependency, so no path exists by which the
  change could reach them.

**Why the wave still closed:** the merged tree is byte-identical to the tree the executor
tested in its worktree (`git diff c69a5fcba HEAD` is empty — the merge added no content
beyond the executor's own commits), and wave 1 ran a single plan, so there is no cross-plan
integration surface for the gate to detect. The gate's purpose was satisfied by that
equivalence, not by waiving it.

**Action:** these two breaks make the project's standard test gate unrunnable on a macOS dev
host — every later wave in this phase inherits it, and any plan whose verify block shells
out to `make test` will read a non-zero exit that has nothing to do with its own work. Fix
by feature-gating the two `aprender-profile` examples to Linux (`required-features` or a
`#![cfg]` guard) and adding the missing field to the 9 `GGUFConfig` literals. Until then,
scope post-merge gates to the crates a plan touches.

## D-ITEM-05-14-D — the dev host is at 95% disk with a 117 GB `target/debug`

Surfaced when a full-workspace control compile exhausted the volume mid-run (`ENOSPC`,
which also killed the tool harness's own output writes until space was reclaimed).

`df -h /` reports 926Gi total, ~760Mi free. `target/debug` alone is 117 GB (`target/release`
is 1.6 GB).

**Action:** this phase's remaining waves compile heavily and waves 05-11/05-12 execute 80
benchmark cells that write per-row artifacts under `benchmarks/tweeteval-stance/`. At
~760Mi headroom those will fail on write, and an ENOSPC failure mid-benchmark is
indistinguishable from a measurement failure — exactly the confusion the claims gate exists
to prevent. Reclaim space before dispatching wave 2.

## D-ITEM-05-05-A — 24 `aprender-train` lib tests are red before this plan touched anything

Surfaced by plan 05-05's post-implementation control run of the FULL crate suite
(`CARGO_INCREMENTAL=0 cargo test -p aprender-train --lib --features setfit`, status captured
directly: rc=101, `7969 passed; 24 failed; 17 ignored`). None is reachable from this plan's
change: `bench_row` is a new leaf module under `train::setfit`, and `git diff` against this
plan's base is EMPTY for both `crates/aprender-compute/` and every failing file's crate path.
Two independent pre-existing causes, not one:

**(a) 21 non-hermetic GPU tests** — `gpu::guard::*` (8), `gpu::ledger::*` (12), `gpu::wait::*`
(1). `default_ledger_path()` (`crates/aprender-train/src/gpu/ledger.rs:33-38`) resolves to
`~/.cache/entrenar/gpu-ledger.json` — a MACHINE-GLOBAL path, not a per-test temp dir. Every
process on the box shares one ledger file, so two concurrent `cargo test` runs (this phase
dispatches parallel executors in separate worktrees, on one machine) contend on it. Verified
not to be an ordering interaction with the new tests: running `gpu::ledger` ALONE still fails
(`rc=101, 25 passed; 12 failed`). Failures are assertion mismatches on reserved/available MB
(`left: 17000, right: 10000`), which is the signature of state another process wrote.

**(b) 3 stale insta snapshots** — `prune::snapshot_tests::{snapshot_all_prune_methods,
snapshot_pipeline_stages, snapshot_schedule_validation_errors}`. The strongest evidence that
these predate this plan is that their `.snap.new` rejection artifacts are already TRACKED AND
COMMITTED in git (`git ls-files crates/aprender-train/src/prune/snapshots/ | grep snap.new`
returns three paths) and are UNMODIFIED by this plan's run. A committed `.snap.new` is a
recorded, unresolved snapshot disagreement.

**Action:** neither is this plan's to fix (SCOPE BOUNDARY — auto-fix only what the current
task's changes caused). (a) needs the ledger tests to take a per-test `with_path(tempdir)`,
which the type already supports (`AccessLedger::with_path`, `:139-141`); leaving it means any
plan in this phase that runs the full crate suite reads a red exit that has nothing to do with
its work, and any plan that runs it CONCURRENTLY with a sibling executor makes it worse. (b)
needs someone to review the three diffs and either accept (`cargo insta accept`) or fix the
regression, then delete the committed `.snap.new` files — a committed rejection file makes the
next reader think the disagreement is expected.

**Consequence for gating:** until (a) and (b) are fixed, a Phase 5 plan touching
`aprender-train` must scope its verification to a test FILTER over its own module and quote the
matched count, not to the whole-crate suite. 05-05 did exactly that
(`... --features setfit bench_row` -> rc=0, 24 passed).

## D-ITEM-05-05-B — `cargo clippy -- -D warnings` cannot pass on this workspace

Measured by plan 05-05 (`CARGO_INCREMENTAL=0 cargo clippy -p aprender-train --lib --features
setfit -- -D warnings`, status captured directly: rc=101). Every finding is in
`crates/aprender-compute/` — dead code (`compute_chunk_scalar`, `pack_a_block_generic`,
`pack_b_block_nr16`, `extract_q6k_values`, `NT_STORE_THRESHOLD_BYTES`, `PREFETCH_DISTANCE`,
...), unused imports (`NeonBackend`, `SUPER_BLOCK_BYTES`/`SUPER_BLOCK_SIZE`, `MR_512V2`/
`NR_512V2`) and unused variables (`mr_block`, `nr_block`) in the BLIS/SIMD kernels. `git diff`
against this plan's base for `crates/aprender-compute/` is EMPTY, so none of it is this plan's.
Findings attributable to `crates/aprender-train/src` in that run: **zero** (grep count 0).

**Action:** not this plan's to fix, and it is a genuine hole rather than noise — CLAUDE.md
lists `cargo clippy -- -D warnings` as a standing gate and `ci.yml` runs a lint job, so a gate
this red is either not actually running with `-D warnings` on this path or is scoped narrower
than the docs claim. Worth resolving as its own ticket: either clear the kernel crate or
record explicitly which crates the `-D warnings` gate covers. A plan-level clippy check in this
phase should scope to `crates/aprender-train/src` and read the finding count for its own files.

---

## D-ITEM-05-15: the 9B LoRA comparison arm — Qwen3.5 hybrid forward path is unimplemented

**Found during:** plan 05-11, Task 1 checkpoint (2026-09-07), before any compute was spent.
No LoRA cell was ever run.

**Symptom:** the phase's comparison arm cannot be built. Two independent causes.

**(a) No GPU host.** `lambda-vector`/`gx10`, hardcoded by every `scripts/dispatch-*.sh`
(`GX10_HOST=gx10`, `GX10_USER=noah`, `/home/noah/src/aprender`), is unreachable — 12 enumerated
candidates, 12 failures, rc captured per attempt (8 × `Could not resolve hostname`, 4 × TCP timeout
against the two RFC1918 addresses in `known_hosts`). Human ruling: the host belongs to the
project's upstream maintainer and is not accessible to us. AWS fallback refuted, not assumed — all
five instances in the account are stopped and none is a GPU instance.

**(b) The architecture is unimplemented — the blocking cause.** Weights are NOT the obstacle:
`Qwen/Qwen3.5-9B` is public and ungated, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`,
19.31 GB bf16 across 4 shards. Three structural gaps, each independently fatal:

| Checkpoint declares | `crates/aprender-train/src/transformer/config.rs` has |
|---|---|
| `layer_types`: 24 `linear_attention` + 8 `full_attention` over 32 layers, `full_attention_interval: 4` | no `layer_types` field (14 `pub` fields, uniform layers) |
| `attn_output_gate: true` | no such field |
| `Qwen3_5ForConditionalGeneration`, `image_token_id: 248056`, image+video preprocessor configs, text hyperparameters nested under `text_config` | `qwen3_5_9b()` is flat and text-only |

The only hybrid-forward artifact is
`crates/aprender-contracts-staging/generated/qwen35-hybrid-forward-v1_scaffold.rs`;
`aprender-contracts-staging` has no `Cargo.toml` (CLAUDE.md: one of the two non-crate directories
under `crates/`), so it never compiles.

**Why `qwen35-e2e-verification-v1.yaml` is not counter-evidence:** all seven falsification tests
are analytical — parameter count, FLOPs-per-token, quantized-memory ordering, roofline, obligation
coverage, per-block shape preservation, layer composition. None loads a weight or compares against
the reference implementation. It verifies the architecture's *description*.

**Scale corroboration:** 05-06 Task 3's reload preflight — the precondition that licenses 9B
compute — passed against a 783,236-byte base model. The LoRA path is proven at fixture scale only.

**Confidence:** (b) is an inference from *structural absence*, not an observed loader failure —
Verification Discipline rule 6. The falsifier is cheap and needs no GPU: point aprender's loader at
the real `config.json` and see whether it accepts or rejects the checkpoint. **It has not been
run.** Run it first when picking this up; if it loads, (b) is refuted and only (a) remains.

**Not caused by Phase 5.** `TransformerConfig` has never modelled hybrid layers; `qwen3_5_9b()` and
the staging scaffold both predate this phase. Phase 5 is where the gap became load-bearing.

**To close:** implement the Qwen3.5 hybrid forward path (linear attention + gated output + the
full-attention interval) with a real weight-loading conformance fixture, then restore EVAL-02's
"both SetFit and the 9B LoRA baseline" clause and EVAL-04's paired-delta clause and run the 40 LoRA
cells against the retained selection manifests. The SetFit half does not need re-running — the
pairing key is recorded per cell precisely so the arm can be added later.

**Blocks:** the descoped halves of EVAL-02 and EVAL-04 (`05-CONTEXT.md` D-19; the Phase 5 amendment
table in `.planning/REQUIREMENTS.md`).

---

## D-ITEM-05-16: three PRE-EXISTING red gates found by plan 05-11, out of its scope

**Found during:** plan 05-11 (2026-09-08), while running the wider suites to check for
collateral damage from the 40-cell retarget. **None is caused by this plan** — measured, not
assumed: `git diff --name-only <plan-base>..HEAD` touches no file under `gpu/`, `prune/` or the
`data` command surface. Logged here rather than fixed, per the executor scope boundary: only
issues directly caused by the current task's changes are auto-fixed.

**(a) `gpu::guard` / `gpu::ledger` / `gpu::wait` — 21 failing unit tests.**
`cargo test -p aprender-train --lib --features setfit gpu::ledger -- --test-threads=1` → rc=101,
"25 passed; 12 failed". Serial execution does NOT fix them, so this is not a parallelism
artifact; e.g. `test_capacity_invariant_prevents_overallocation` fails
`assert!(result.is_err())` at `crates/aprender-train/src/gpu/ledger.rs:608`. The capacity
invariant that is supposed to REFUSE an overallocation is accepting one — worth treating as a
real defect rather than a flake.

**(b) `prune::snapshot_tests` — 3 failing tests.** Also fails serially.

**(c) FALSIFY-CLI-006 red: three undeclared depth-2 commands.**
`cargo test -p apr-cli --test cli_commands` → rc=101,
`FALSIFY-CLI-006: the binary offers depth-2 commands the contract does not declare:
["data tweet-eval-stance", "data select", "data pairs"]`. The binary ships three `apr data`
subcommands that `contracts/apr-cli-commands-v1.yaml` does not list. The fix is to add them
under `data`'s `subcommands:` — a contract edit, which is exactly the kind of edit this phase
requires a checkpoint for, so it is not something to slip into an unrelated plan.

Note that 05-11's own `apr setfit bench verify-cell` is NOT among the undeclared: the gate
checks depth-2 paths and `verify-cell` sits at depth 3.

**(d) 15 clippy findings under `-D warnings`.** All in `aprender-compute` (14) and
`aprender-present-terminal` (1) — crates 05-11 never touched. `cargo clippy -p aprender-train
--lib --features setfit -- -D warnings` → rc=101, but zero findings in `aprender-train` or
`aprender-core`.

**Why this matters beyond bookkeeping:** (a) and (c) are gates that are RED in the tree today.
A phase whose thesis is "refuse incomplete or unequal claims" should not leave red gates
unrecorded, and CLAUDE.md's own lesson is that a gate nobody runs is a gate that stops being
run. `make setfit-bench-tests` and `make contract-audit-phase5` — the two gates 05-11 owns —
are both green; these three are not, and they were already not before this plan started.

---

## D-ITEM-05-17-A — the BENCH ROW SEAL is build-graph dependent (`serde_json/preserve_order`)

**Found during:** plan 05-17, Task 2, while writing the 40-committed-row agreement
measurement. It is the third independent measurement of the `bench_row.rs:37-43` comment
05-15 and 05-16 both flagged, and the first one to find a CONSEQUENCE rather than stale prose.

**Symptom, measured on `benchmarks/tweeteval-stance/rows/setfit-s16-seed13.json`:**

| statement | value |
|---|---|
| the envelope's own `semantic_hash` | `1c54f3b4e38540040a2a2224a424969bae473e7b28da6b33ee2fb4be9f1eae69` |
| `sha256` over the payload in **declaration order** | `1c54f3b4…` — reproduces |
| `sha256` over the payload **key-sorted** | `fafda6485f47531aaf277a095ea782cd86ba148e694956bb902e3e0d9c5ad04c` |
| what `BenchRow::from_bytes` computed under `cargo test -p aprender-train --lib --features setfit` | `fafda648…` — the KEY-SORTED one |

So the SAME committed row parses under the shipped `apr` binary and is refused as
`row_digest_mismatch` under the `aprender-train` test binary. Measured over all 40 rows:
declaration order reproduces **40/40**, key-sorted **0/40**.

**Cause, named rather than inferred.** `BenchRowPayload::to_canonical_bytes` serializes through
`serde_json::Value`, whose `Map` is an `IndexMap` when the `serde_json/preserve_order` feature
is enabled anywhere in the binary's dependency graph and a `BTreeMap` when it is not.
`cargo tree -p apr-cli --features setfit -e features -i serde_json` shows
`serde_json feature "preserve_order" <- pmcp v2.19.3`; the same query against
`aprender-train` shows no `preserve_order` at all. `pmcp` is in `apr-cli`'s graph (the MCP
servers) and absent from `aprender-train`'s.

**Why the module comment is worse than wrong.** `bench_row.rs:37-43` says the digest is
key-sorted "no workspace crate enables `preserve_order`" and concludes it is "independent of
Rust field-declaration order, so adding a field in a different position cannot silently change
every historical digest". Both halves are false today, and the second is the load-bearing one:
declaration order IS what the committed digests encode, so reordering a struct field WOULD
silently invalidate all 40 rows and 40 manifests — and so would ADDING OR REMOVING A DEPENDENCY
that touches `serde_json/preserve_order` anywhere in a binary's graph. A cargo feature-unification
change, with no code change at all, can flip the whole committed evidence set between "verifies"
and "refused". An auditor who builds a different binary does not reproduce the guarantee.

**Why 05-17 did not fix it.** `crates/aprender-train/src/train/setfit/bench_row.rs` is outside
this plan's `files_modified`, and a real fix — pinning the canonicalization explicitly rather
than inheriting it from feature unification — re-seals 40 rows, 40 selection manifests and the
run manifest, which is a plan of its own with its own evidence obligations. 05-17's measurement
test routes around it deliberately and says so in its own doc comment, so the next reader cannot
lose the finding: it deserializes the `payload` object directly instead of calling
`BenchRow::from_bytes`.

**Action.** Own ticket. The fix is to make `to_canonical_bytes` state its key order rather than
inherit it (serialize through a type whose ordering is fixed, or sort explicitly), re-seal the
committed evidence in the same commit, and add a test that the digest of a fixed payload equals
a pinned constant — which is the only shape of test that can catch feature-unification drift,
because every test that recomputes the digest the same way the code does will agree with
whatever the code currently produces.

---

## D-ITEM-05-17-B — `uuid_v4()` in `aprender-test-lib` is a timestamp, and its uniqueness test fails on this host

**Found during:** plan 05-17's post-implementation workspace regression. The run reported
**85 failed** against a baseline of 84; the one beyond baseline is
`aprender-test-lib brick::pipeline::tests::test_uuid_v4_generates_unique_ids`.

**Proven not caused by 05-17** — three independent controls:
- `git diff --stat 49832d05c..HEAD -- crates/aprender-test-lib/` is EMPTY: the file is
  byte-identical to the commit the 84-failure baseline was measured at, yet it passed there
  and fails here.
- `cargo tree -p aprender-test-lib` declares no dependency on `aprender-train`, `apr-cli` or
  `aprender-core` — no path exists by which this plan's change could reach it.
- Zero of the 85 failures are in `bench_gate`, `bench_metrics`, `bench_row` or `setfit_bench`.

**The real defect, which is worth a ticket rather than a rerun.**
`crates/aprender-test-lib/src/brick/pipeline.rs`'s `uuid_v4()` is
`format!("{:x}{:x}", SystemTime::now().duration_since(UNIX_EPOCH).as_nanos(), process::id())`
— a TIMESTAMP wearing a UUID's name. `test_uuid_v4_generates_unique_ids` generates 100 in a
tight loop and asserts `ids.len() >= 90`, so its verdict is a function of the host's clock
resolution and loop speed, not of the code. It failed **5/5** on re-run here (deterministic on
this box today, not flaky), and it passed on the same bytes three hours earlier.

**Action.** Either give `uuid_v4()` a real entropy source or a monotonic counter, or delete a
test whose pass depends on the machine being slow enough. A "unique ID" that collides under a
fast loop is a latent defect in anything that uses it as a key, not only in its own test.

---

## D-ITEM-05-17-C — `verify_tests.rs:628` fails `clippy::search_is_some` under `--all-targets`

**Found during:** plan 05-17, Task 1, running
`cargo clippy -p aprender-train --lib --features setfit --no-deps --all-targets -- -D warnings`
(rc=101).

`crates/aprender-train/src/train/setfit/verify_tests.rs:628` writes
`src[door_at + 1..].find("…").is_none()`, which clippy wants as `!…contains(…)`.

**Pre-existing and out of scope:** the file is untouched by 05-17, and the plan's own literal
clippy line (`--lib`, without `--all-targets`) does not compile `#[cfg(test)]` modules and is
rc=0. Recorded because a later plan that widens its clippy scope to `--all-targets` will hit it
and should not mistake it for its own.
