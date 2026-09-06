# PP-066 progress report — aprender 0.66, as of 2026-09-06

Epic #2873. Spec `docs/specifications/PP-066-release-spec.md` (v1.6). DAG `docs/specifications/pp-066-dag.yaml`. Every number below names the artifact it was read from; a number without one is marked `[U]`.

## 0. Summary

- **Where 0.66 stands.** `main` is at `027ed889d`, version `0.65.2`; the last cut is `v0.65.2` (`8e1e9ad40`, 2026-09-04, 74/74 crates on crates.io). The 0.66 DAG on `main` carries 91 rows (61 in the 0.66 lane, 30 carried to 0.67); 5 are complete (G-6, G-4, SPEC-1.6, C0-5, C0-7). The tag row `TAG-0.66.0` lists 52 blockers, 5 of them complete. Eight more rows are implemented and sit in armed PRs (#3001, #3003–#3009); none has merged because `main` is red on one CI step.
- **Why nothing merges today.** The shipped-path ratchet in `guard-runner-labels` compares a count of machine-specific paths against a stored baseline of 277 that named no analyser version. The fleet's pmat moved 3.31.0 → 3.37.0 on 2026-09-06 (paiml/infra forjar pin), the self-arming guard armed, and the unchanged tree counts 317. Every PR went red for a defect none introduced. The root-cause fix is complete on branch `agent/G-10` (pinned analyser, stamped baseline, differential verdict, 281 unpinned references swept to 0) and is the only PR allowed into the queue until it lands.
- **The key issue — GPU discovery and the release binary.** *Solved on a branch, not yet on `main`*: `apr devices` and the backend registry (R-0a, PR #3004) enumerate cpu/cuda/wgpu on every host, print one line per backend kind with a reason when unavailable, and were dogfooded on all four hosts (gx10 GB10, intel 2× W5700X, mini M4, lambda RTX 4090). *On track*: resolution reading the registry instead of compile-time flags (R-0b), cuda in the published crate's default features (R-2, gated on decision D-9), release assets on tagged releases with a promotion gate from four host receipts (R-5), a verifying one-line installer (R-6), README (R-7). *Blocked*: decisions D-3, D-9, D-11 are blank; D-9 gates R-2 and therefore the "user gets the GPU by default" outcome.
- **Nearest cliffs.** Rows expiring 2026-09-12: R-3 (armed in #3001), DEC-D-3, DEC-D-11. Rows expiring 2026-09-19: twenty, including R-0, I-1, C0-1/3/4/6, T-2, T-0h, G-5, G-7, P-0.3, DEC-D-9.

## 1. The release criterion and the DAG

The tag is cut when C0–C13 of spec §4 hold, each as a command that exits 0 on the release commit; C0 (the required check is readable and strict) is credited first. Rows are data in the DAG; `scripts/check_dag_invariants.sh` holds 0 cycles, ≥ 6 days slack on every blocker pair, and one expiry form per row (G-4, landed). Rendered §5.0 tables are byte-identical to the YAML (`scripts/render_dag.py --check`, wired).

| track | tag blockers | complete | notes |
|---|---|---|---|
| I (instrument chain) | 8 | 0 | I-24, I-25, I-26 armed in #3006/#3008/#3009; I-1 (gx10, land #2809 + Blackwell guard) open; I-15/I-18 derive from I-1 |
| C0 (required check) | 7 | 2 | C0-5, C0-7 landed; C0-4 armed (#3007); C0-3 blocked upstream (pmat#1200); C0-1, C0-2, C0-6 open |
| G (guards) | 8 | 2 | G-6, G-4 landed; G-10 (the ratchet) fixed on `agent/G-10`; G-5, G-7, G-1, G-8, G-9, G-3 open |
| R (runtime discovery, refusal, install) | 7 | 0 | R-0a armed (#3004); R-3 armed (#3001); R-2, R-4, R-5, R-6, R-7 open |
| P (provable contracts) | 8 | 0 | every row blocked on DEC-D-11 (blank) |
| S (speed) | 4 | 0 | blocked on I-1 / I-15 / I-17 |
| T (training) | 5 | 0 | T-2 armed (#3005); T-0h, T-1, T-0, T-3 open |
| B (backends) | 2 | 0 | B-A1 blocked on DEC-D-3; B-G1 on R-0, R-2 |
| D (docs) | 3 | 1 | SPEC-1.6 landed; D-1doc, D-2doc open (non-blocking) |

Derived from `pp-066-dag.yaml` on `main` (`TAG-0.66.0.blockers`, 2026-09-06).

## 2. The key issue: GPU discovery and the binary a user gets

### 2.1 The problem, measured

`evidence/dogfood/0.65.2/VERDICT.md` is the post-publish dogfood of the last release, run on the binary a user installs (`cargo install aprender --version 0.65.2 --locked`), on all four hosts:

- The published default-feature binary is CPU-only. The root facade's `[features]` on `main` reads `default = ["cli"]`; `cuda = ["cli", "apr-cli/cuda"]` is opt-in (`Cargo.toml:668-678`).
- Against llama.cpp (comparator pinned in `scripts/llama_pin.toml`) at c=1 on the D1 protocol: decode 0.59× on lambda (x86), ~0.35× on intel under runner load, 0.046× on gx10 and 0.05× on mini (both aarch64). A `--features cuda` install on lambda (167 s) measured 0.69× decode and 0.18× prefill against llama.cpp CUDA at c=1; every c>1 band is INVALID-CORRECTNESS (no PP-26 witness). Source: the parity-lane table and Determination in `evidence/dogfood/0.65.2/VERDICT.md`.
- The verdict is **NO-GO on measured evidence**, and the operator's recorded decision was "0.65.2 stays published; these four host receipts are the 0.66 baseline".
- The nightly cross-compiled assets (5 targets, `nightly.yml`) carry `--no-default-features --features inference` and none carries `cuda` (S0 ledger row S0-16). `binary-release.yml` builds four Linux targets on `release: published`.

### 2.2 Five whys (from the spec's own chain; D-9 is named "the why-1 countermeasure", R-0 "the why-3/why-4 fix")

1. **Why does a user get a CPU-only `apr`?** Because the default feature set excludes `cuda`, and `--gpu` on that binary has nothing to select.
2. **Why is `cuda` not a default feature?** Because a cuda-featured binary on a host without `libcuda.so.1` would crash or silently fall back; without runtime discovery the default cannot be made safe (spec §0 D-9).
3. **Why was there no runtime discovery?** Because backend choice was a compile-time decision — `crates/apr-cli/src/accel.rs:28` reads `cfg!(any(feature = "cuda", feature = "wgpu"))`; the binary asked what it was built with, never what the machine has.
4. **Why compile-time?** Because backends were added one feature flag at a time with no registry and no printed device list; the "Backend: CUDA" line came from the request, not from the device (R-0 card, mutation ii).
5. **Why did this ship?** Because no release criterion required the binary a user gets to be dogfooded on real hosts, and the pre-publish gate graded a different artifact (PMAT-967: a dev checkout with `feature_set [cuda]`). 0.66 adds C4 (install through the installer on four hosts, record `apr devices --json`), C11 (the registry is the only authority; zero `cfg!` reads in backend decisions), and C13 (assets, checksums, signature on the tag).

### 2.3 What is solved (on branch `agent/R-0`, PR #3004, armed)

`trueno::registry` (`crates/aprender-compute/src/registry/{mod.rs,cuda.rs,wgpu_probe.rs}`, 32 functions) and `apr devices [--json]` (`crates/apr-cli/src/commands/devices.rs`), with `contracts/apr-backend-registry-v1.yaml`, `contracts/apr-devices-schema-v1.yaml` and the JSON schema `contracts/schemas/apr-devices-v1.schema.json`. What it does today:

- Enumerates every kind in `{cpu, cuda, wgpu, metal, hip}` on every host and always prints one line per kind: `ready` with device, memory kind and transport, or `unavailable reason=…` (REG-11: the absence of a backend is a line the user reads, never a silence). cpu is always Ready (invariant i).
- Probes CUDA through the driver API only (`libcuda.so.1` via `CudaDriver::load()`, then `device_count()` and a context) — never cudart (REG-2); probes wgpu adapters and refuses software rasterisers (`NoBackend(software rasteriser (llvmpipe …) is not a GPU)`).
- Prints the reserve with its basis (REG-7: `reserve=3584MiB basis=[U] default until master row 6 measures vram_peak`) and propagates a reserve refusal across a device's other-API entries; prints the default selection with its reason (REG-8).
- Is covered without a GPU: `crates/aprender-compute/tests/registry_case_table.rs` (10 tests) and `crates/apr-cli/tests/registry_failure_catalogue.rs` (10 tests) run on fixtures under `crates/apr-cli/tests/fixtures/registry/` including a must-RED twin under `defective/`. Three mutations were shown RED then restored GREEN in the receipt.
- Was dogfooded on all four hosts (receipt `docs/audits/impl-PMAT-989-receipt.md`, "Four-host dogfood"): gx10 `NVIDIA GB10` Ready through wgpu/vulkan (unified, 122502 MiB); intel two `AMD … (RADV NAVI10)` Ready through wgpu/vulkan plus llvmpipe refused; mini `Apple M4` Ready through wgpu/metal; lambda cpu Ready and wgpu/Vulkan sees the RTX 4090, the cuda line printed as unavailable under the default build rather than omitted. The catalogue's FX-7 row found a real library gap (reserve refusal not propagated to the API twin) and the intel run found REG-9's same-name collapse (two W5700X shared one `device_uid`), both fixed on the branch.
- A three-lane design quorum preceded implementation (`docs/audits/pp-066-r0-design-quorum.md`, PR #3003: 3/3 implement-with-changes, 3/3 split R-0 into R-0a/R-0b), and a three-lane review of the diff returned 3/3 mergeable-with-changes; the changes were applied.

### 2.4 What is on track, and what it waits on

| step | row | status | waits on | expiry |
|---|---|---|---|---|
| resolution reads the registry: `--backend/--gpu/--device` refused by enumeration with the registry's reason; `accel.rs`, `dispatch.rs`, `serve/mod.rs` stop consulting `cfg!` | R-0b (issue #3002, PMAT-1060; DAG row lands with #3003) | open; 12 `cfg!(feature = "cuda")` reads across 7 `apr-cli` files remain on `agent/R-0` (C11 requires 0) | R-0a merged | 2026-09-19 (R-0) |
| `cuda` joins the published crate's default features iff S0-14 held on all four hosts, R-0 merged, clean-room p1 green | R-2 (#2905) | open | R-0, **DEC-D-9 (blank)** | 2026-09-26 |
| `apr-*` assets for the 5 targets on every tagged release, built per D-10 (`cli,cuda` where cuda compiles without a toolkit), sha256 + minisign-signed manifest (S0-19: public-key scheme, no third scheme), cut as prerelease, promoted from four host receipts + H12 | R-5 (#2908) | open | R-0, DEC-D-10 (default stands) | 2026-10-09 |
| one-line installer at `/releases/latest/download/install.sh`: detect OS/arch, verify checksum and signature, end by printing `apr devices` | R-6 (#2909) | open | R-5 | 2026-10-16 |
| README leads with the installer; backend line generated from the registry | R-7 (#2910) | open | R-6 | 2026-10-23 |
| the tag | TAG-0.66.0 | open | all 52 | 2026-10-30 |

The honest reading: the *mechanism* that makes a GPU-capable default safe exists and is measured on the real hosts; the *outcome* a user sees (a GPU-capable binary by default, installed by one line, printing what it found) depends on D-9 and D-10 being decided and on R-0b, R-2, R-5, R-6 landing in that order. Nothing in that chain is blocked on unknown engineering; it is blocked on the queue being red (today) and on two operator decisions.

### 2.5 The same lesson, applied twice today

The ratchet incident (§4) is the GPU problem one layer up: a gate whose number depended on whichever binary PATH resolved. The fix has the same shape as the registry — one pinned instrument, printed provenance, no comparison across instruments — which is why `scripts/pmat_bin.sh` follows `scripts/apr_bin.sh` and `scripts/pv_bin.sh` exactly.

## 3. What was accomplished, and why (five-whys per item)

Landed on `main` (all via PRs through `ci / gate` + `workspace-test`):

| item | PR | what | why (the five-whys, compressed) |
|---|---|---|---|
| Spec v1.5 → v1.6 | #2872, #2987 | the release spec: 13 criteria, 91-row DAG, decisions register, refusals, prior-art register | 0.65.2 shipped NO-GO because no written criterion required the shipped artifact to be measured → the criterion must be commands, not prose → the rows must be data with expiries → a spec |
| S0 discovery ledger | #2875 | 23 premises confirmed/falsified before ticket #1 (S0-3 falsified: intel has 2 AMD GPUs; S0-12: no Metal dispatcher; S0-19: two signing schemes already exist; S0-23: strict protection applied) | plans built on unverified premises re-plan mid-flight → verify every premise read-only first → a ledger with evidence paths |
| Plan + DAG | #2981 | `pp-066-dag.yaml`, `pp-066-plan.md`, three quorum lanes on the plan | rows typed into prose drifted (three expiry dates had zero slack) → invariants need a checker → data |
| C0-5 | #2985 | baselines ratchet shrink-only against a ref a PR cannot rewrite | a PR could baseline its own violation and pass → compare against merge-base, never the working tree |
| G-4 | #2987 | `check_dag_invariants.sh` + `render_dag.py` (rendered tables byte-identical) | `pmat comply` has no `--rule obligation-dag` → in-repo checker; hand-typed tables drift → render |
| G-6 | #2987 | `check_roadmap_diff_additive.sh`: a PR's roadmap diff adds entries only; base resolved on three CI checkout shapes | `pmat work add` re-serialised the 2046-line roadmap and minted colliding ids → forbid non-additive diffs mechanically |
| C0-7 | #2987 | receipts carry `status: complete\|partial` in front matter; DAG completion requires a complete receipt; torn `.tmp` receipts refused | a half-written receipt resumed as finished → the marker is set last, the writer writes `.tmp` then `mv` |
| Write-back + G-10 row | #3000 | receipts flipped to complete, PMAT-966 duplicate resolved, the ratchet landmine (#2999) recorded as a DAG row | the landmine was known on 2026-09-05; recording it as a row with an expiry is how it stops being a comment nobody re-reads |

Implemented, verified, armed (auto-merge on), waiting on `main` going green:

| row | PR | what landed on the branch | why |
|---|---|---|---|
| R-3 | #3001 | honest training banner: `--gpu-backend cuda` refuses on a CPU build with the code from `error.rs`; the cuBLAS-backward line prints only when a device-side backward launched (`entrenar::backward_kernel_launches()`); contract `apr-train-banner-truth-v1.yaml` | the banner printed by path, not by event → a user was told "cuBLAS backward" on a CPU run → count launches, print from the count |
| R-0 amendment | #3003 | design-quorum record, DAG R-0a/R-0b split, §12 llamafile correction, §5.0 re-rendered | the design row is the one architecturally significant row of 0.66 → decide the shape before P2, record the dissent |
| R-0a | #3004 | the registry and `apr devices` (§2.3); plus, fixed after CI: the missing book chapter `book/src/cli/devices.md` + page contract | §2.2 |
| T-2 | #3005 | `--max-seq-len` honoured or refused on every finetune path (the wgpu pipeline still clamped 512 at `finetune.rs:717`); `effective max_seq_len=` printed; contract `apr-finetune-config-truth-v1.yaml` | a requested config silently replaced by a clamp is `requested ≠ effective`, the exact class T-0h's `INVALID-CONFIG` refuses |
| I-24 | #3006 | `parity_block.py`: a zero or empty comparator band is a named refusal, never a traceback; `check_parity_block_refusals.sh` wired; contract; plus, fixed after CI: the selftest's `/usr/bin/apr` fixture path (ABS-APR) | a traceback in the gate is an unanswered gate → every refusal is a named row |
| C0-4 | #3007 | `perf_gate.sh` arm A on a c=1-only receipt is `REPORT`, never `VERDICT PASS` (#2830); invariant + FALSIFY-…-012 | an arm that emits nothing must not read as passing (F-28) |
| I-25 | #3008 | `--workload` bound to the prompt corpus sha256 in the receipt; `receipt_accepts_workload`; contract `apr-workload-corpus-binding-v1.yaml` | a workload name without the corpus hash is a label, not a measurement |
| I-26 | #3009 | PP-29 scanner: a §12 expiry is the `Expires **date**` marker, never the first date in the cell; rows past expiry go RED; `derived_expiries.json` regenerated (row 1 → 2026-09-19) | five master rows sat past a mis-parsed expiry with nothing RED (S0-2) → the andon the spec promises did not fire |

Every one of these carries a contract in the same PR, its mutation shown RED then GREEN in the PR body, and every acceptance command re-run by the orchestrator (a worker PASS is a claim).

Blocked, with the block recorded:

- **C0-3** (the ten `contracts/work/GH-663..672.cot.yaml` derivations): `pmat work cot derive` writes them and `CB-1658` turns ✓ (✗ naming the file when one is removed), but every derived obligation is hollow (`statement: ""`), `pv validate` refuses each with `SCHEMA-005`, and CB-1658 checks existence only. Hand-filling a generated file or teaching `pv` to accept an empty obligation are both contract exemptions; pmat hardcodes the path. Filed as paiml/paiml-mcp-agent-toolkit#1200; branch `agent/C0-3` (pushed) holds the receipt (`status: partial`) and the generated files. Ordered today: DAG row U-1 for pmat#1200 with C0-3 blocked by it, and CB-1658 to validate each file with `pv`.

Done today (2026-09-06), not yet on `main`:

- **G-10, the ratchet root cause** (branch `agent/G-10`, five commits, unpushed at the time of writing — §4).
- **Housekeeping ordered by the driver:** the three unpushed fixes pushed (#3004 book chapter, #3006 fixture path, `agent/C0-3`); the throwaway `batch-dry` and `main-probe` worktrees and the private rebuild-helper directory deleted; the seven-PR batch plan abandoned (rows merge serially through the queue).

## 4. Why `main` is red, and the fix

**Mechanism.** `scripts/check_hardcoded_paths.sh --full-if-capable` compared `pmat analyze hardcoded-paths` `shipped_count` against `scripts/hardcoded_path_shipped_baseline.txt`, a single number (277) recorded on 2026-09-05 by #2985 with no instrument named. The mode was "self-arming": it skipped while the fleet's pmat (3.31.0) lacked the subcommand. paiml/infra's `machines/intel/forjar.yaml` moved the fleet to 3.37.0 (`3.31.0 -> 3.37.0 (BSE-10b, PMAT-231)`); the guard armed on `intel-clean-room-9` and counted 317. Re-scanning C0-5's own commit (`cdc0acb99`) under 3.38.0 also gives 317: the tree never changed, the instrument did. First red: #3007 at 07:47Z; then #3003, #3008; #3001's merge-queue run will hit the same step.

**Five whys.** (1) PRs are red because the count exceeds the baseline. (2) The count grew because a different analyser scanned. (3) The baseline had no version because nothing required one. (4) Nothing required one because the guard treated the number as a property of the tree. (5) It could treat it that way because the analyser was whatever PATH resolved — the same defect the repo already fixed for `apr` and `pv`.

**Fix (branch `agent/G-10`, ticket PMAT-1059).**

1. `scripts/pmat_bin.sh` — ONE pin (`PMAT_PIN="3.37.0"`, matching forjar), sourced, option-neutral, exports `$PMAT`/`$PMAT_VERSION`, refuses off-pin or absent binaries naming both versions.
2. `scripts/check_pmat_pinned.sh` — the operator's assertion verbatim, `grep -rEn '(^|[^_/])pmat ' scripts/ .github/workflows/ | grep -v pmat_bin` must be empty, prose included; 15-row case table. RED at HEAD with 281 lines; 0 after the sweep of ~38 scripts and 5 workflows (presence probes became ENV failures; `dogfood.sh`'s pmat-verify WARN became FAIL; `verifier_pin.sh`'s fleet case takes the pin; python callers read `$PMAT`; workflow installs are at the pin).
3. The baseline is stamped (`count:` / `pmat_version:` / `basis:`); missing or unparseable is INVALID and not a number. Today's 277 is stamped INVALID until a re-baseline ticket measures it under the pin — never a raise.
4. The ratchet compares absolutes only under a matching stamp; otherwise it prints `REPORT BASELINE-STALE{old,new}` (or `BASELINE-INVALID`) and decides HEAD vs merge-base under the same binary (`scripts/lib/resolve_base.sh`, extracted from G-6's guard so both judge the same base; base materialised as a detached worktree because the analyser enumerates with `git ls-files`), delta ≤ 0 passes and delta > 0 names the new paths. A stamp that moves while the count stands is refused.
5. Evidence: 13/13 self-test rows (8 new fixture rows, both polarities); live on the branch: `REPORT BASELINE-INVALID{stamp=none,binary=3.37.0}` then `PASS (differential, delta +0)`; live mutation (a `/home/probe/…` string literal appended to `crates/apr-cli/src/main.rs`) → `delta +1` naming the file, rc 1; every step of `guard-runner-labels` replayed locally — the only reds (docker, apr-bin resolution, four self-tests) fail identically on a clean-`main` worktree. Contract `contracts/apr-pinned-analyser-ratchet-v1.yaml`.

**Other reds seen on the armed PRs, classified.** `present` (pr-review-quorum) fails on every PR including merged ones — advisory, pre-existing. `pr-review-receipt` CANCELLED on #3001 — the C0-6 class (mutation step exceeds the job timeout), a DAG row. `Build Book` on #3004 — real: `apr devices` had no chapter; fixed on the branch. `guard-runner-labels` on #3006 before the ratchet — real: an absolute `/usr/bin/apr` fixture path; fixed on the branch.

## 5. What is left for the release

**Decisions (operator; a blank keeps its rows blocked, never defaulted):** D-3 (CI home for non-CUDA lanes; expiry 2026-09-12; blocks B-A1), D-9 (cuda in default features; 2026-09-19; blocks R-2 → I-18, B-G1, C4), D-11 (PV-IMPROVE-001 scope; 2026-09-12; blocks all eight Track P rows). D-10 stands on its default.

**Sequence to green `main` and the first merges:** G-10 PR → queue → `main` green; then #3001, #3003, #3004, #3005, #3006, #3007, #3008, #3009 serially through the queue (one entry at a time; a queue run takes 1–2 h on the shared fleet).

**Process changes ordered 2026-09-06 (R2–R5), all open:**
- G-11: a row PR's write set is code, tests, contracts, its receipt and its book page — never the DAG, the roadmap or README counts; DAG `status` derived from receipts; roadmap/README/estimates written by one orchestrator docs commit after each merge; guard `check_row_pr_write_set.sh` with its mutation. Acceptance: after any of the seven armed PRs merges the rest stay MERGEABLE, rebuilds = 0.
- No batching of row PRs; G-6's labels-only carve-out either lands with a falsifier or does not exist.
- DAG row U-1 (pmat#1200), C0-3 blocked by it; CB-1658 validates every derivation with `pv`; the fix sequence pmat#1200 → pmat release → forjar pin bump on four hosts + runner → `pmat_bin.sh` bump PR → stamped re-baseline. Until then every C-criterion is `[U]`.
- Basis: turns, tokens and wall-clock per row from the eight receipts into `impl-estimates.jsonl`; expiries recomputed under the §12 amendment rule; G-6, R-3 and SPEC-1.6 each stated as landed / amended / expired-RED by 2026-09-12 (G-6 and SPEC-1.6 landed; R-3 is armed and will not land by 2026-09-12 unless the queue clears — an amendment is owed).
- Remote worktrees on gx10/intel/mini removed by a make target (none exists yet).

**Rows by nearest expiry (open):**
- 2026-09-12: R-3 (armed), DEC-D-3, DEC-D-11.
- 2026-09-19: I-1, I-24, I-25, I-26 (armed), C0-1, C0-3 (blocked), C0-4 (armed), C0-6, G-7, G-5, G-2, R-0 (armed), R-4, P-0.3 (blocked on D-11), T-0h, T-2 (armed), DEC-D-9.
- 2026-09-26: R-2, C0-2, G-1, G-8, G-9, P-0.6/0.2/0.4/0.5, S-3, T-1, D-2doc.
- October: R-5 (10-09), R-6 (10-16), R-7 (10-23), T-0, T-3, S-1, S-2, S-3g, I-16, I-17, G-3, B-G1, P-0.1, P-1.1/1.2, D-1doc; TAG-0.66.0 (10-30).

**Not yet started (0.66 lane):** I-1, I-15, I-18, I-16, I-17, C0-1, C0-2, C0-6, G-7, G-5, G-1, G-8, G-9, G-3, R-2, R-4, R-5, R-6, R-7, all of P, S, T-0h, T-1, T-0, T-3, B-A1, B-G1, D-1doc, D-2doc.

## 6. Risks

- **Queue throughput.** One merge-queue entry at a time and 1–2 h per run on a fleet shared across repos: nine PRs are at least a day of serial merges after `main` goes green. If throughput is the constraint, merge-queue batching is a ruleset change and is raised as a STOP, not applied.
- **Expiries.** Twenty rows expire on 2026-09-19; at the measured queue rate the armed rows land but the unstarted ones (I-1, C0-1, C0-6, G-5, G-7, R-4, T-0h) need amendments or go RED under the §4 andon that I-26 now wires.
- **Upstream dependency.** C0 is credited only when the ten derivations validate, which waits on a pmat release (#1200); every C-criterion is `[U]` until then.
- **Decisions.** D-9 and D-10 sit on the critical path of the GPU outcome; D-3 and D-11 expire on 2026-09-12.

## 7. Provenance

`git rev-parse origin/main` = `027ed889d`; DAG and receipts read from `origin/main` with `git show`; branch facts from `origin/agent/{R-0,R-3,T-2,I-24,C0-4,I-25,I-26,C0-3}` and the local `agent/G-10`; 0.65.2 numbers from `evidence/dogfood/0.65.2/VERDICT.md`; the fleet pin from paiml/infra `machines/intel/forjar.yaml` via the GitHub API; the ratchet counts from `pmat analyze hardcoded-paths -p . -f json` under `~/.local/pmat/3.37.0/bin/pmat` (3.37.0) and `~/.cargo/bin/pmat` (3.38.0) on clean checkouts of `027ed889d` and `cdc0acb99`; PR and check states from `gh pr view`/`gh pr checks` and the in-session monitor, 2026-09-06 07:00–09:30Z. Written by the PP-066 orchestrator session; independent readers and adversarial refuters (workflow `pp066-progress-verify`) re-checked the GPU-discovery, release-binary and CI facts — their corrections are folded into the text above where they differed.
