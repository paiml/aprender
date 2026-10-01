# Phase 8 — Deferred Items (out-of-scope discoveries)

Logged by executors under the scope-boundary rule: found while verifying a plan, not caused by it,
not fixed by it.

## From plan 08-01

### D-ITEM-08-01-A: CI's strict-test-binding guard is VACUOUS on this branch, and red underneath

- **Found during:** 08-01 Task 1 verify (`bash scripts/check_contract_test_binding.sh`).
- **Symptom:** rc=1, `VACUOUS: strict-test-binding gate was SKIPPED (contract validation failed); nothing was measured.`
- **Cause 1 (the skip):** `contracts/spectral-indices-v1.yaml` has no `kani_harnesses:` section, so
  `pv lint`'s validate gate reports `PROVABILITY-001: Kernel contract has no kani_harnesses
  (spectral-indices-v1)` and every downstream gate is skipped. Introduced by `fdf6b1802`
  ("feat: add spectral indices and aprender-mcp-chronos-lambda ..."), which is on this branch
  and NOT on `origin/main` (`git merge-base --is-ancestor fdf6b1802 origin/main` rc=1). The file
  has no `contract:` key and is not in the Makefile `CONTRACTS` list, so `make contract-validate`
  never sees it.
- **Cause 2 (underneath):** with the skip lifted in a temporary copy (a declared-not-executed
  `KANI-SPECTRAL-001` naming `test_falsify_spectral_001_bounds`, which exists in
  `crates/aprender-image/src/tests.rs`), the guard runs and reports
  `Resolved 585 test references; 44 dangling across 15 contract(s)` and FAILS on two contracts
  outside the baseline: `contracts/chronos-bolt-parity-v1.yaml` (9 dangling, baseline 0) and
  `contracts/setfit-encoder-conformance-v1.yaml` (8 dangling, baseline 0).
- **Phase 8 status:** in both the as-is and the lifted runs, NO Phase 8 contract is named. The four
  Phase 8 contracts cite no Rust test at all (staged binding); the only test reference is
  `FALSIFY-DECIDE-TOOL-009`, which is `LIVE-PENDING` (unbindable by design).
- **Why not fixed here:** the skip fix alone does not turn the guard green (17 pre-existing dangling
  references in Phase 1 / Phase 6 contracts), so it would be a partial repair to two other phases'
  contracts. It belongs in its own change: add `kani_harnesses:` (and a `contract:` key) to
  spectral-indices-v1, then fix or re-cite the 17 dangling references — never raise the baseline.
- **Impact on 08-01 acceptance:** the "prints its PASS line" clause of the strict-binding criterion
  could not be satisfied by this plan; the "does not name the new contract" clause was verified in
  a run where the gate actually measured.

- **CLOSED 2026-09-29 by plan 08-33 (status: closed).** Cause 1: spectral-indices-v1 now declares
  `kani_harnesses` KANI-SPECTRAL-001..004 in the branch's DECLARED, NOT EXECUTED convention (each names its
  existing `test_falsify_spectral_00n_*` test; cargo-kani is not installed, no proof is claimed), a
  `metadata.kind: kernel` and a `metadata.valid_under` world, at version 1.1.0 (the bump `pv diff` suggested),
  and neon-blis-v1's dead top-level `kind:` is gone, so `pv validate contracts` reports 0 failed of 1850 and
  `pv lint contracts` runs every armed gate. Cause 2: chronos-bolt-parity-v1's nine dangling citations were five
  existing tests whose `#[test]` sat above a four-line `#[cfg_attr(not(chronos_weights), ignore = ...)]`, which
  the resolver's line-based harvest does not see through; `#[test]` now sits directly above `fn` in
  aprender-forecast (bolt.rs x4, chronos.rs x1). The guard measures 27 dangling references, equal to the
  baseline sum of 27 (scripts/contract_test_binding_baseline.txt), with no contract above its own line; the
  baseline file is byte-identical to upstream's 2817c6d97. Evidence: 08-33-HYGIENE-EVIDENCE.json.

### D-ITEM-08-01-B: README CLI command count is stale (FALSIFY-README-003)

- **Found during:** 08-01 Task 1 (`bash scripts/check_readme_claims.sh`).
- **Symptom:** `FAIL FALSIFY-README-003 cli_command_count: README claims 110, contracts/apr-cli-commands-v1.yaml lists 111 commands`.
- **Not caused by 08-01**, unrelated to contracts. FALSIFY-README-002 (contract count) was ALSO
  failing before 08-01 (README said 1778 in two prose lines against 1791 on disk); 08-01 fixed that
  one because it moves the contract count, and it now passes at 1795.

## From plan 08-03

### D-ITEM-08-03-A: D-ITEM-08-01-A re-measured; still blocks the strict-binding PASS line

- **Found during:** 08-03 Task 1 and Task 2 verify (`bash scripts/check_contract_test_binding.sh`).
- **Symptom:** unchanged: rc=1, `VACUOUS: strict-test-binding gate was SKIPPED (contract validation failed)`,
  because `contracts/spectral-indices-v1.yaml` has no `kani_harnesses:`.
- **Measured around it:** `pv lint --strict-test-binding` was run on a temporary copy of `contracts/` with
  the skip lifted. The copy was a sibling dir inside the repo, because the source scan is rooted at the
  contract dir's parent, and it was deleted right after. Result: 587 refs resolved (585 at 08-01, plus
  this plan's two bindings), and `laya-parity-v1` has **0** dangling references. The other 15 contracts
  and 44 dangling references are identical to D-ITEM-08-01-A. A mutated binding (`tiny_parity_mutant_zz`)
  was reported dangling in the same run, so the resolver discriminates.
- **Why not fixed here:** same reason as D-ITEM-08-01-A. The fix belongs to other phases' contracts.
- **Also found:** `.planning/WINDOWS.md` refuses every append (`Ledger entry 24 has invalid status:
  "resolved"`), so this plan's deviation could not be recorded there. That was not caused by 08-03 and
  is not fixed here.

## From plan 08-04

### D-ITEM-08-04-A: D-ITEM-08-01-A re-measured; the strict-binding PASS line is still unreachable

- **Found during:** 08-04 Task 1 and Task 2 verify (`bash scripts/check_contract_test_binding.sh`).
- **Symptom:** unchanged: rc=1, `VACUOUS: strict-test-binding gate was SKIPPED`, because
  `contracts/spectral-indices-v1.yaml` has no `kani_harnesses:`.
- **Measured around it** (temporary lifted copy `contracts-lift-0804/`, deleted after each run): 593 refs
  resolved (589 after Task 1 — 587 at 08-03 plus the two laya-parity-v1 bindings — and 4 more from the
  decide-apr-v1 legs), **0 dangling in laya-parity-v1 and decide-apr-v1**; the other 15 contracts and 44
  dangling refs are identical to D-ITEM-08-01-A. Mutated binding names were flagged in both Phase 8
  contracts, so the resolver discriminates.
- **Why not fixed here:** same reason as D-ITEM-08-01-A.

### D-ITEM-08-04-B: `cargo fmt --all -- --check` fails on files from `fdf6b1802`

- **Found during:** 08-04 final checks.
- **Symptom:** diffs in `crates/aprender-image/src/{lib,spectral,tests}.rs` and
  `crates/aprender-mcp-chronos/{build.rs,src/lib.rs}` — all from `fdf6b1802` (the commit behind
  D-ITEM-08-01-A). `cargo fmt -p aprender-decide -- --check` exits 0.
- **Why not fixed here:** out of scope (other phases' crates); a one-line `cargo fmt` in its own change.

## From plan 08-05

### D-ITEM-08-05-A: D-ITEM-08-01-A re-measured; the strict-binding PASS line is still unreachable

- **Found during:** 08-05 Task 2 verify 2 (`bash scripts/check_contract_test_binding.sh`).
- **Symptom:** unchanged: rc=1, `VACUOUS: strict-test-binding gate was SKIPPED`, because
  `contracts/spectral-indices-v1.yaml` has no `kani_harnesses:`.
- **Measured around it** (temporary lifted copy `contracts-lift-0805/`, deleted after each run): 622 refs
  resolved (593 at 08-04, plus 28 decide-apr-v1 legs for FALSIFY-001/002/003/004/005/008/009/011 in the
  GREEN commit, plus FALSIFY-006 in the trybuild commit), **0 dangling in decide-apr-v1 and
  laya-parity-v1**; the other 44 dangling refs are identical to D-ITEM-08-01-A. Two mutated binding names
  (`nan_weight_mutant_zz`, `index_capacity_mutant_zz`) were both flagged, so the resolver discriminates.
- **Also:** `.planning/WINDOWS.md` still refuses every append (`Ledger entry 24 has invalid status:
  "resolved"`), so this unrun-verify item is recorded here instead of the ledger.
- **Why not fixed here:** same reason as D-ITEM-08-01-A.

## From plan 08-06

### D-ITEM-08-06-A: D-ITEM-08-01-A re-measured; the strict-binding PASS line is still unreachable

- **Found during:** 08-06 Task 2 verify 1 (`bash scripts/check_contract_test_binding.sh`).
- **Symptom:** unchanged: rc=1, `VACUOUS: strict-test-binding gate was SKIPPED (contract validation failed)`,
  because `contracts/spectral-indices-v1.yaml` has no `kani_harnesses:`.
- **Measured around it** (temporary lifted copy `contracts-lift-0806/`, deleted after the runs): 644 refs
  resolved (622 at 08-05 plus exactly this plan's 22: 20 `aprender-mcp-decide` legs for TOOL-001..008 and the
  two `aprender-decide` stance-order legs TOOL-004 binds), **0 dangling in decide-tool-boundary-v1**; the other
  44 dangling refs are identical to D-ITEM-08-01-A. Two mutated binding names
  (`bounds_match_contract_mutant_zz`, `admission_refuses_over_pending_mutant_zz`) were both flagged, so the
  resolver discriminates on this contract.
- **Why not fixed here:** same reason as D-ITEM-08-01-A.

### D-ITEM-08-06-B: README's layout tree still says "(82 crates total)"

- **Found during:** 08-06 Task 1 (README crate count).
- **Symptom:** README.md line 224 (`└── ... (82 crates total)`) disagrees with the gated metrics row, which this
  plan moved 88 -> 89 from `cargo metadata --no-deps`. `readme_contract` only checks the metrics row, so the
  tree line has drifted unnoticed across several phases.
- **Why not fixed here:** pre-existing drift in prose no gate reads; the plan scoped the edit to the gated count.

## From plan 08-07

### D-ITEM-08-07-A: the bootstrap handler has not run under a Lambda runtime

- **Found during:** 08-07 final verification.
- **What is proven locally:** every piece the handler composes. That covers the stateless loopback
  (identity, cold-first and both maximal shapes over real HTTP), the S3 loader through an injected
  fetcher, `LoadOnce` re-arming after a failure, and the probe-id filter and load-evidence strings.
  The `bootstrap` bin compiles for aarch64 (zigbuild, glibc 2.34), and `resolve_model_future_is_send`
  pins the `Send` property lambda_http needs.
- **What is not:** `main.rs::handler` itself. That includes the 503-on-failed-load path, the
  `x-decide-load` header on a proxied response, and the `decide.load` line in CloudWatch. No
  Lambda runtime emulator (`cargo lambda watch`) was run, and a live deploy belongs to 08-10 and 08-11.
- **Also:** `.planning/WINDOWS.md` still refuses every append (`Ledger entry 24 has invalid status:
  "resolved"`), so this unrun-verify item is recorded here instead.

### D-ITEM-08-07-B: a per-crate deploy root builds the wrong `*-lambda` package

- status: resolved
- **Resolved by:** plan 08-10 (user decision shared-crates-root, 2026-09-26). `just laya-deploy`
  deploys from `crates` with server `aprender-mcp-decide`, and `just laya-resolver-proof` executed
  cargo-pmcp 0.24.3's own resolver on this workspace: root `crates` -> `crates/aprender-mcp-decide-lambda`;
  the per-crate root -> `crates/aprender-mcp-chronos-lambda` (the trap, confirmed). The durable
  upstream fix is D-ITEM-08-10-B.
- **Found during:** 08-07 Task 3, while writing `.pmcp/deploy.toml.template`.
- **Symptom:** cargo-pmcp 0.24.2's `find_lambda_package_dir` tries `<deploy-root>/{server}-lambda`
  first, then falls back to the first workspace `*-lambda` package with a `bootstrap` bin. With
  `crates/aprender-mcp-decide-lambda` as the root, the first branch cannot match. In
  workspace-member order, `cargo metadata --no-deps` lists `aprender-mcp-setfit-lambda` first
  (measured), and in alphabetical order `aprender-mcp-chronos-lambda` comes first. Either way,
  `cargo pmcp deploy --manifest-path crates/aprender-mcp-decide-lambda` would ship another server's
  binary.
- **Owner:** plan 08-10 (its resolver-proof task and deploy-root decision already cover it). The
  template now documents the trap instead of prescribing the per-crate command.

## From plan 08-09

### D-ITEM-08-09-A: the stdio server's real-model leg is DEFERRED (user approval, 2026-09-26)

- status: resolved
- **Resolved 2026-09-27 (plan 08-16):** the one declared demo_s64 run passed the unchanged gate
  (gate_pass), and `just laya-verify` printed `deploy_eligible true` on
  `models/decide/laya-stance-64.apr` (sha256 `24a44d7e050166c9b64e2716f2bcb3ce91747f7a3b927d03d6eeae5f89b6275a`).
  `a_real_decide_model_classifies_over_live_stdio` then passed on that exact file over live stdio: one
  tool, served identity `24a44d7e…` equal to the file's sha256 (D-11), K probabilities summing to 1,
  labels from the task. Record: `08-GATE-RUN-EVIDENCE.json` `e2e_stdio`. Invocation note: the env var
  must be an ABSOLUTE path, because `cargo test` runs the test with the crate dir as its cwd.
- **Not run:** `APR_MCP_E2E_DECIDE_MODEL=<real .apr> cargo test -p aprender-mcp-decide --release --test e2e_stdio`,
  test `a_real_decide_model_classifies_over_live_stdio` (crates/aprender-mcp-decide/tests/e2e_stdio.rs).
  It would assert one tool, the served identity equal to the file's sha256 (D-11), K probabilities
  summing to 1, and labels from the task, over live stdio MCP.
- **Why:** no artifact that `pack_laya verify` accepts exists (option 3: both D-19 demo runs are
  fail-closed vectors). decide-apr-v1 knows only `production` and `synthetic-fixture`, so a base
  en-root evidence pack has no honest variant. Serving a low-level pack of a fail-closed vector would
  contradict `demo.fail_closed_rule`. The leg is not run on a stand-in.
- **What covers the gap meanwhile:**
  - the tiny-fixture stdio leg in the same test file;
  - `verify_path` loading real-weights bytes from a FILE through `Decider::load_path` (all eight
    rungs, probe replay) with the 280-row re-scores, in `tests/fail_closed_vectors.rs`;
  - full-model parity on the English root, `tests/laya_parity.rs` (ids 14/14, max |dp| 3.841e-6).
- **What re-arms it:** the first artifact `just laya-verify` accepts. That needs the calibration spike
  (`.planning/todos/pending/spike-laya-calibration-slice-and-temperature-cap.md`) and then a declared
  run that passes the gate unchanged.
- **Alternatives that need a user decision:** (1) amend `demo.fail_closed_rule` to allow a local,
  never-uploaded serve of a vector; or (2) add a non-production evidence variant. Option 2 is a
  decide-apr-v1 / laya-finetune-gate-v1 schema change, and D-17 makes it costly.

### D-ITEM-08-09-B: the 1e-5 re-score bar is fragile on real checkpoints (queued, not acted on)

- status: superseded
- **Superseded by:** laya-parity-v1 A1 (plan 08-13, declared 2026-09-27); implemented by 08-14/08-15.
- **Found during:** the 08-09 tracer halt and debug session `.planning/debug/resolved/laya-rescore-drift.md`.
- **Facts:**
  - `pack_rescore_probs_abs` 1e-5 sits at the fp32 noise floor of real Laya checkpoints. Against a
    float64 reference, torch's own fp32 is up to 3.7e-5 off on fixed_epochs (13 rows).
  - early_stopping and the base re-score at 6.7e-6 / 6.8e-6, about 1.5x headroom. One 1-ULP codegen
    change (`__sincosf_stret` fusion lost in 71e2306e5) once consumed that headroom.
  - x86_64 (Lambda) has never been measured.
- **Related headroom seen in 08-09:** the full-model ladder's `final` block measured 9.155e-5
  against `final_norm_abs` 1e-4 (1.09x). Spike 025 measured 3.05e-5 there before the RoPE fix.
  Every other ladder block and the 14-row probabilities keep more than 2.5x headroom.
- **Owner:** the calibration spike todo (section "Added 2026-09-26: is the 1e-5 re-score bar above fp32
  noise"). User decision option A keeps the bar, and nothing is changed here.

### D-ITEM-08-09-C: `demo.fail_closed_rule` prose still says both vectors fail on the ece_post clause

- status: resolved
- **Resolved by:** plan 08-13 rewrote laya-finetune-gate-v1 `demo.fail_closed_rule` to agree with FALSIFY-LAYA-GATE-010 (early_stopping GateFailed[ece_post] exit 3; fixed_epochs RescoreDrift exit 2), under the user's 2026-09-27 instruction to keep both vectors.
- **What:** laya-finetune-gate-v1 `demo.fail_closed_rule` says pack and verify MUST refuse each vector
  "with the gate recomputed in Rust ... reproducing FAIL on the ece_post clause". Under option A,
  fixed_epochs refuses earlier, with RescoreDrift (still fail-closed, nothing written).
  FALSIFY-LAYA-GATE-010 (1.3.0) states the per-vector refusal and is the bound test.
- **Why not changed here:** the plan forbids moving any `demo` value, and the user's option A kept
  contract text other than GATE-010 unchanged. Whether to align the prose with GATE-010 is a one-line
  user call.

### D-ITEM-08-09-D: D-ITEM-08-01-A re-measured; the strict-binding PASS line is still unreachable

- status: open
- **Symptom:** unchanged: `VACUOUS: strict-test-binding gate was SKIPPED`, because
  `contracts/spectral-indices-v1.yaml` has no `kani_harnesses:`.
- **Measured around it** (temporary lifted copy, deleted after each run):
  - Task 2: 658 references resolved (644 at 08-06, plus this plan's 14 verify bindings).
  - Task 3: 661 (plus FALSIFY-LAYA-GATE-010 and the two FALSIFY-LAYA-PARITY-001/-003 legs). A
    mutated GATE-010 binding (`demo_vectors_are_refused_fail_closed_mutant_zz`) was flagged
    dangling in the same kind of run, so the resolver discriminates.
  - No Phase 8 contract dangles. The only FAIL lines are the two pre-existing contracts
    (chronos-bolt-parity-v1, setfit-encoder-conformance-v1).
- **Why not fixed here:** same reason as D-ITEM-08-01-A.

## From plan 08-10

### D-ITEM-08-10-A: one decide model per workspace under the shared-crates-root deploy

- status: open
- **What:** cargo-pmcp resolves `<root>/<server>-lambda` first, so the deploy from `crates` works only
  with server name `aprender-mcp-decide`, the package stem. `laya-deploy-config` and `laya-deploy`
  refuse any other name. A second decide model, such as a second task, cannot deploy from this
  workspace without revisiting the 08-10 decision.
- **Also shared:** `crates/.pmcp/` and `crates/deploy/` belong to the setfit training server.
  `_laya-crates-root-swap` backs them up and restores them byte-identically. The selftest proves
  this on success, forced failure, SIGTERM and an absent root. A SIGKILL mid-deploy cannot run
  the trap, so it leaves the backup at `models/decide/swap-backup/crates`, and the next swap
  refuses until a human restores from it.
- **Owner:** D-ITEM-08-10-B removes the constraint.

### D-ITEM-08-10-B: fix the resolver upstream in cargo-pmcp (recommended future SDK work, not done)

- status: open
- **What:** in `find_lambda_package_dir`, when `project_root` is itself a package whose name ends in
  `-lambda` and has a `bootstrap` bin, return it before `find_workspace_lambda_package_dir`. This
  makes `--manifest-path crates/<pkg>-lambda` correct for every server. It also protects the
  chronos and setfit deploys from the same class of bug, and it removes the config swap.
- **Why not done here:** the user chose shared-crates-root. The SDK checkout is an external repo on
  an unrelated branch with uncommitted work, so this plan only read it (`git archive` into a
  scratch dir).

### D-ITEM-08-10-C: the live halves of the deploy recipes have never run against AWS

- status: superseded (08-18, 2026-09-27). This is NOT `resolved`: the rule is resolved only if every live
  assumption was confirmed, and one was REFUTED. Nothing is left to measure live, because the refuted
  check was replaced and the replacement passed live. The final record is `08-LIVE-DEPLOY-EVIDENCE.json`
  (`final: true`, `outcome: deployed-passed`).
- **08-18 final tally of the live assumptions (08-17, 2026-09-27):**
  - Function name equals the server id: **confirmed.** `get-function` and `get-function-concurrency`
    on `aprender-mcp-decide` answer.
  - `[deployment] endpoint` lands in `deployment.toml`: **confirmed.**
  - A GET on the endpoint reaches the bootstrap's health branch (auth off): **REFUTED.** The pmcp.run
    edge answers GET `/mcp` with 405 and `/health` with platform JSON (D-ITEM-08-17-A). It was replaced by
    the edge `/health` `serverId` check (`3115c690e`), which passed live on resume attempt 2 and on option 1.
  - The `Compiling` lines land in the redirected deploy log: **confirmed.**
  - `LoggingConfig.LogGroup` names `/aws/lambda/aprender-mcp-decide`: **confirmed.** 08-18 read it there.
  - Previously unrun, now run live:
    - the identity probe (22:56:52Z, identity == H);
    - `laya-deploy-verify` (4 cold samples, DEPLOY VERIFY OK);
    - containment inside `laya-deploy` (reserved concurrency 0, verified), and its resume with
      `delete-function-concurrency`.
- **Partly measured by plan 08-17 (2026-09-27, live):** bucket, upload, deploy-config and the
  `cargo pmcp deploy` half of laya-deploy ran. Confirmed: function name == server id; `[deployment]
  endpoint` lands in deployment.toml; the `Compiling` lines land in the redirected log; `LoggingConfig.LogGroup`
  names `/aws/lambda/aprender-mcp-decide`. REFUTED: a GET on the endpoint does not reach the bootstrap
  (D-ITEM-08-17-A). Still unrun: the identity probe, laya-deploy-verify and a live laya-teardown resume.
- **Kind:** unrun-verify. The live deploy is deferred by option 3.
- **Proven offline:** `bash -n` passes on every recipe body. The refusals are proven on the
  synthetic artifact: placeholder, sha-pin, resolver-proof, deploy-eligibility and
  upload-eligibility. An aws recorder with a positive control counted 0 AWS calls. The
  `laya-deploy-verify` DRY_RUN plan was built from the artifact, also with 0 AWS calls.
  `laya-verify` precedes the first AWS call in `laya-deploy` and `laya-upload`.
- **Assumptions only a live run can check:**
  - The pmcp.run function name equals the server id. `pmcp-train-grant` relies on the same.
  - cargo-pmcp writes `[deployment] endpoint` to `crates/.pmcp/deployment.toml`, which the swap
    copies to `models/decide/deploy-<server>.state/`.
  - A GET on that `/mcp` endpoint reaches the bootstrap's health branch, which may not hold with
    auth on.
  - The `Compiling` lines of `cargo lambda build` land in the redirected deploy log.
  - `LoggingConfig.LogGroup` names the function's log group.
- **Owner:** the first post-spike deploy. Plan 08-11 records HOLD.

### D-ITEM-08-10-D: `auth=on` sets only `[auth] enabled`

- status: resolved (08-18, 2026-09-27). The deploy option that ran was `deploy-auth-off-accept-risk`.
  - Auth posture: **off.** The config has `[auth] enabled = false` and `provider = "none"`, and the
    deploy passed `--no-oauth`. pmcp.run reported `oauthEnabled=false`.
  - Provider: **none.**
  - Risk: the user explicitly accepted the cost-amplification risk, as in the chronos precedent.
  - Re-open condition: a future deploy with `auth=on` must still choose the provider that pairs with it.
- **What:** `laya-deploy-config <apr> on` sets `enabled = true` and leaves `provider = "none"` from
  the template. The provider that pairs with an authenticated pmcp.run function is plan 08-11's
  auth decision. It is not assumed here.


## From plan 08-11

### D-ITEM-08-11-A: the D-18 live deploy and the live `accepted_region_cold` falsification are DEFERRED (option 3; HOLD decided by the user, 2026-09-26)

- status: resolved (08-18, 2026-09-27). The final live outcome is **deployed-passed**
  (`08-LIVE-DEPLOY-EVIDENCE.json`, `final: true`).
  - Server: `aprender-mcp-decide` on pmcp.run, at 3,008 MB arm64, pinned to H `24a44d7e…`.
  - Identity == H through the edge.
  - 4 `laya-deploy-verify` cold samples, 2 per maximal shape: 24168-29350 ms, all < 30000.
  - Plan 08-12 binds `accepted_region_cold` from this record.
  - The thin margin is carried by D-ITEM-08-17-E.
- **Deferred:** two things. (1) The D-18 live deploy of a trained decision model on pmcp.run. (2) The
  live falsification of decide-tool-boundary-v1 `accepted_region_cold`, FALSIFY-DECIDE-TOOL-009, which
  stays `LIVE-PENDING`. The user decided HOLD at the 08-11 go/no-go checkpoint, choosing the
  `hold-no-aws` option. No AWS or pmcp.run call of any kind was made, read-only calls included.
  The record is `08-DEPLOY-EVIDENCE.json` (`outcome: hold`, `decided_by: human`, `readiness: null`).
- **Why:** the D-19 demo failed laya-finetune-gate-v1 under both declared recipes. Both runs are
  fail-closed vectors (`d0f4e40d…` fixed_epochs, `3d4b91da…` early_stopping), and 08-09 refuses both.
  No artifact passes `just laya-verify`, so 08-10's AWS-writing recipes would refuse anything that
  exists in this phase.
- **Next direction (user, 2026-09-26):** pursue a deployable model next to confirm the direction:
  first the calibration spike, then a declared gate run. The procedure below is the path once a
  declared run passes.
- **Comes first:** `.planning/todos/pending/spike-laya-calibration-slice-and-temperature-cap.md`. Whatever
  it recommends is declared in laya-finetune-gate-v1 BEFORE the next gate run is read (D-07).
- **Re-open condition:** a DECLARED run passes laya-finetune-gate-v1 unchanged (gate_max_ece 0.10),
  and `just laya-verify` prints `deploy_eligible true` on its exact `.apr`. The deploy is then a NEW
  plan, not a re-run of 08-11.
- re-open condition MET by plan 08-16 (gate_pass, deploy_eligible true; artifact sha256 `24a44d7e050166c9b64e2716f2bcb3ce91747f7a3b927d03d6eeae5f89b6275a`); owner: plans 08-17/08-18
- **Decisions still open for that run:**
  - **Auth posture.** Auth on was recommended for a 10 GB function (RESEARCH §Security Domain V2:
    an open 10 GB function is a cost-amplification vector). `laya-deploy-config` takes `auth` as a
    required argument, and the provider that pairs with `auth=on` is still undecided (D-ITEM-08-10-D).
  - **Account and memory facts, still unmeasured.** hold-no-aws recorded no readiness facts.
    - RESEARCH A10: the `ze-kasher-dev` profile resolves, and pmcp.run functions are visible in it.
    - RESEARCH A2: pmcp.run accepts 10,240 MB.
    - The account has not been checked for a function whose name contains `decide` or `laya`.
    - Measure all three with read-only calls (`aws sts get-caller-identity`,
      `aws lambda list-functions`, `aws lambda get-account-settings`) before that plan's first write.
  - **The live-side assumptions of D-ITEM-08-10-C** (function name equals server id,
    `[deployment] endpoint` location, GET health with auth on, compile-log capture, log-group
    naming).
- **Procedure it will follow, in order, with 08-10's recipes:**
  1. `just laya-weights-bucket`
  2. `just laya-upload <apr> <run> <data> <base> <server>`. This is gated by `laya-verify` before
     any AWS call.
  3. `just laya-deploy-config <apr> <auth>`, with the auth posture decided above.
  4. `just laya-deploy <apr> <run> <data> <base> <server>`. The recipe runs these steps:
     1. eligibility through `laya-verify`;
     2. the resolver proof, pinned to the installed cargo-pmcp;
     3. the deterministic compile-log identity check;
     4. the scoped grant (`laya-grant`);
     5. the GET health body naming the package;
     6. the live identity probe.
     On an identity failure, it runs containment (`laya-teardown`).
  5. `just laya-deploy-verify <apr> <server>`. The cold-sample rules:
     - bump the config before each cold sample;
     - make the maximal `tools/call` the first POST;
     - take at least 2 CONCENTRATED and 2 DISTRIBUTED samples;
     - prove each one cold by CloudWatch `performed_load=true`;
     - require every sample to be under 30000 ms.
  6. On any breach, record the samples unaltered and stop at a blocking-human choice between
     three responses:
     - lower the contract-owned budget through a gap plan (D-10);
     - defer while serving;
     - defer and contain.
- **Full pre-revision text:** `git show 59d9ed07f:.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-11-PLAN.md`
  (Tasks 2-4: deploy with identity asserted, both cold shapes on every sample, the exceeded-region
  response, the outcome record).

## From plan 08-13

### D-ITEM-08-13-A: x86_64 re-score parity is unmeasured

- status: open
- **What:** x86_64 re-score parity is unmeasured (laya-parity-v1 A1 `qa_gate` risk). Every A1 number is
  aarch64 (Apple M4 Pro, NEON 8x6 BLIS, Apple libm). The deploy direction is aarch64 to aarch64 (Lambda
  arm64), but glibc `sinf`/`cosf` differ from Apple libm, and on Lambda only the two-probe replay at
  `probe_probabilities_abs` runs, so a cross-libm mismatch would refuse the load (fail-closed).
- **Settles it:** spike 028's x86 procedure (`.planning/spikes/028-laya-packability-noise-floor/README.md`,
  "x86_64: unmeasured (risk)") against the committed float64 logits in its `results/triad/*.json`; no
  torch is needed on the host. Record the first x86_64 value in laya-parity-v1.
- **Owner:** a later x86 CI run.

## From plan 08-14

### D-ITEM-08-14-A: the plan's median seeds label disagrees with the contract

- status: resolved
- **Resolved 2026-09-27 (plan 08-15, before 08-16's declared run):** the contract now declares a label
  that names what ships. laya-finetune-gate-v1 `seed_policy.rule` and `gate_report_schema.seeds`: gate-report
  `seeds.label` is "single seed" for one seed, "median-ECE seed of N seeds" under seed selection, and the 1.x
  literal "mean ± sd over N seeds" only for a legacy multi-seed run; variance-report.json keeps "mean ± sd over
  N seeds" (that file does report the mean and sd). `contract.seeds_label(n, policy)`, train.py and
  lifecycle.py `check_median_run` follow it. The Rust verifier does not read `seeds.label` (it re-derives the
  shipped seed from the per-seed files instead), so nothing there had to change. pv diff: identical, no
  version bump; `just laya-train-selftest` green; `just laya-fixtures` byte-identical.
- **What:** 08-14-PLAN Task 1 asked `contract.seeds_label` to return "median-ECE seed of 3 seeds" for the
  median policy. laya-finetune-gate-v1 (as amended by 08-13) fixes the label in two places:
  `seed_policy.rule` ("Under both rules gate-report `seeds.label` is ... else \"mean ± sd over N seeds\"")
  and `gate_report_schema.seeds`. The contract wins, so a 1.4.0 median run reports
  `mean ± sd over 3 seeds`. The shipped seed and policy are carried by `seeds.policy` / `seeds.shipped`.
- **Settles it:** plan 08-15's Rust reader accepts the contract's literal. If a different label is
  wanted, it takes a contract amendment declared before 08-16's run, not a trainer change.
- **Owner:** plan 08-15 (reader), or a human decision before 08-16.

### D-ITEM-08-14-B: FALSIFY-LAYA-GATE-006's "legacy variance seeds" clause has no Python leg left

- status: resolved
- **Resolved 2026-09-27 (plan 08-15, before 08-16's declared run):** the clause is RETIRED in the
  GATE-006 prediction with its reason recorded there. No code path can exercise it (no legacy multi-seed run
  is written since 1.4.0, and a legacy run carries no per-seed record from which a verifier could tell which
  seed shipped), and it decides nothing a verifier accepts (every legacy run is refused SeedPolicyMissing once
  its gate passes, FALSIFY-LAYA-GATE-012). The prediction now claims what is tested: legacy single-seed ships
  seed 13; the median run ships the median. `implemented_by` points at the retirement. pv diff: identical.
- **What:** the GATE-006 prediction still says "with legacy variance seeds 13/17/23 the shipped
  checkpoint is seed 13 even when another seed scores higher". Since 08-14, every three-seed run carries
  `seed_selection` and ships the median, so the trainer no longer writes a legacy MULTI-seed run, and
  the lifecycle cannot exercise that clause. 08-14 corrected GATE-006 `implemented_by` to say so. The
  prediction text is unchanged, because changing it would change what the contract claims.
- **Settles it:** 08-15's Rust legacy-rule tests (a recipe.json without `seed_selection`). The gitignored
  1.x run dir `models/decide/tweet-stance-16-var` is a real legacy multi-seed run the verifier can use.
  Otherwise, a contract edit that retires the clause.
- **Owner:** plan 08-15 / plan 08-12's binding sweep.

## From plan 08-17

### D-ITEM-08-17-A: laya-deploy's health-body check cannot reach the function through pmcp.run

- status: resolved (08-18, 2026-09-27). Its one pending condition was the live run of the replacement
  step, and that is now met:
  - the edge `/health` `serverId` check (`3115c690e`) passed live on resume attempt 2 and on option 1;
  - the identity probe then proved identity == H through the edge at 22:56:52Z.
- **Found during:** 08-17 Task 2, the first live `just laya-deploy` (2026-09-27). It refused and contained a
  correctly built function (reserved concurrency 0, grant removed): `08-LIVE-DEPLOY-EVIDENCE.json`
  `outcome: deploy-refused`.
- **What:** the recipe GETs `[deployment] endpoint` (`.../mcp`) and expects the bootstrap's health body
  naming `aprender-mcp-decide-lambda`. The pmcp.run edge answers that GET itself with 405
  (`"SSE streams are not offered at this endpoint. Use POST /mcp."`), and the separate `/health` URL is
  also answered by the platform (`{"status":"healthy","serverId":...,"hasDeployment":true}`, no package
  field). Measured identically on the live chronos-forecaster endpoint. No GET reaches the bootstrap, so
  the check can never pass on pmcp.run.
- **What still proves identity:** the compile-log check (passed) and the live identity probe, a POST
  `tools/call` returning `model.artifact_sha256 == H` plus the task's labels in order. The probe is
  realizable through the edge; the wrong-binary threat the health body guarded (chronos under this name)
  cannot answer it.
- **Options for the human (the resume checkpoint):** (1) replace the health GET with the edge's `/health`
  `serverId == <server>` check and let the identity probe carry package identity; (2) reach the bootstrap
  directly with `aws lambda invoke` and a synthetic GET event, bypassing the edge; (3) drop the health
  step. Resuming also needs `aws lambda delete-function-concurrency`, which laya-teardown leaves to the
  human.
- **Owner:** the 08-17 continuation after the human's choice.
- **Progress (2026-09-27):** the human chose option 1. The recipe was replaced in `3115c690e` (edge
  `/health` `serverId`, offline case table in `laya-deploy-selftest`, AWS CALLS 0). The live run of the
  new step is still pending: the resume re-deploy stopped earlier, at a pmcp.run login gate.

### D-ITEM-08-17-B: pmcp.run invokes the function before laya-grant can run

- status: CLOSED 2026-09-27 (08-17 option 1, commits 52a12d777 and 743eb02ed). The S3 read is a
  `[[iam.statements]]` entry in the deploy config (Allow s3:GetObject on
  `arn:aws:s3:::<weights-bucket>/decide/aprender-mcp-decide/*`, nothing else). cargo-pmcp 0.24.3 renders it
  into the role's default policy `pmcp-declared`, and the function DependsOn that policy. pmcp.run's own
  post-deploy call then LOADED the model (22:55:16Z, load_ms 24554, status success). `laya-grant` is now a
  read-only check, and the propagation sleep is gone.
- history (status before the close): open (blocking 08-17 since resume attempt 2)
- **What:** CloudWatch shows one invocation at 21:12:10Z, right after the deploy and before the grant.
  pmcp.run made it, not the recipe. The load failed at the S3 length lookup (no policy yet), and
  `LoadOnce` re-armed, so the instance is not poisoned. Whatever the platform learned from that call
  (schema discovery or health) saw a load failure. Every redeploy repeats this, because the role is
  created by the deploy and the grant can only follow it.
- **Also fixed in this plan:** `laya-deploy` now passes `--no-post-deploy-test`, because cargo-pmcp's own
  suite would hit the same pre-grant 403 and exit 3 before the recipe's identity chain. It also sleeps
  `LAYA_GRANT_PROPAGATION_S` (default 20 s) after the grant, so IAM propagation is not misread as an
  identity failure (commit b30f437da).
- **Durable fix:** declare the S3 read in the deploy config's `[iam]` (the pmcp-declared policy the
  platform applies at create time), so the role can read the weights before the first call. This needs a
  check that cargo-pmcp 0.24.3 renders `[iam]` for pmcp-run without a preserved stack.ts.
- **Proven decisive on 2026-09-27 (08-17 resume attempt 2):**
  - The redeploy's own pre-grant platform call came at 22:16:47Z, and the load failed the same way.
  - pmcp.run then marked the server as being in an error state. The edge answered the identity
    probe's first POST with `503 {"code":-32004,"message":"Server is in error state"}` and never
    invoked the function.
  - A second POST a minute later got the same answer, so the state is sticky.
  - The edge `/health` kept reporting `healthy`.
  - So on pmcp.run, a grant that follows the deploy can never reach the identity probe. This is no
    longer a later-plan item. It blocks 08-17.
- **Also measured:** the redeploy kept the execution role name of the first deploy. A grant applied
  before `cargo pmcp deploy` would therefore exist when the platform call runs. That option is
  untested: a stack update might drop an out-of-band policy.
- **Owner:** the 08-17 resume. The human chooses the fix (see the 08-17 SUMMARY checkpoint).

### D-ITEM-08-17-C: the sha256 pin runs sha2's SOFTWARE backend on aarch64

- status: open
- **What:** the workspace depends on `sha2 = "0.10"` without the `asm` feature. On aarch64, sha2 0.10.9
  then compiles only the software compressor (`src/sha256.rs`: the aarch64 intrinsics backend is gated
  on `feature = "asm"`). Measured 618-624 MB/s (1370 ms for the 846 MB artifact) on an Apple M4. The
  3,008 MB tier prices this at 4521 ms of the 17000 ms cold budget (x3.3 for Graviton2). Graviton2 and
  later have the SHA-2 extensions, so the `asm` feature (or another hasher) could return most of that to
  the token budget.
- **Why not done here:** it changes the dependency graph of every crate that uses sha2. A re-derived budget
  must follow a measured cold sample, not a projection.
- **Owner:** a perf plan after the first live cold samples (their `sha_ms` is the measurement).
- **Measured live (08-17/08-18):** `sha_ms` was 2655-3521 over 5 cold loads on Lambda (4 rule samples and
  the external cold call): graviton2 about 3.5 s, graviton3 about 2.7 s. Recommended, not acted on
  (a lever for D-ITEM-08-17-E).

### D-ITEM-08-17-D: pmcp.run keeps refusing MCP POSTs after its post-deploy invocation (platform finding)

- status: open. This is a platform issue for the user's own pmcp.run. It is not aprender work, and nothing was
  done about it here.
- **What:** the pmcp.run edge makes its own post-deploy call to a new function. When that call fails,
  the edge answers every later `POST /mcp` with `503 {"code":-32004,"message":"Server is in error state"}`
  and never invokes the function. `GET /health` keeps reporting `{"status":"healthy", ...}` the whole time.
- **Measured both ways on 2026-09-27:**
  - **Failed load** (08-17 resume attempt 2, grant not yet in place): the 503 was sticky. It was still
    there at 22:18:34Z, about 2 minutes after the platform call. Only containment stopped it.
  - **Successful load** (option 1): the post-deploy call took 24.6 s and returned `success`, and 5
    follow-up calls succeeded in 1.5-3 ms. The edge still answered 503 at about 22:55:25Z. By 22:56:09Z
    it was forwarding again (a throttled 500 while contained). The identity probe passed at 22:56:52Z
    with no redeploy.
- **Why it matters:**
  - `/health` cannot see the MCP route's state, so a health-based readiness check passes while every
    MCP call fails.
  - For a model server whose first call is a 20+ s cold load, one slow or failed first call can take
    the server offline at the edge. On a failed load that lasts until something outside the edge clears it.
- **Recommendation:** open a pmcp.run platform issue:
  - `/health` should report the error state.
  - The state should clear once an invocation succeeds, or have a documented TTL or reset.
  - The post-deploy call's timeout should be documented against the 30 s gateway cap.
  Until then, the decide recipe retries exactly this refusal, bounded (commit 743eb02ed).
- **Owner:** the user (pmcp.run platform). Recommend only; do not act.
- **08-18 (2026-09-27): still open, recommended, no action taken.** One more point belongs in the same
  issue. An external cold call took 31.05 s end to end at the client, and the edge still answered 200
  (its in-function duration was 28008 ms). So the edge's effective cutoff is not a strict 30.000 s from
  the client, and it should be documented. See D-ITEM-08-17-E.

### D-ITEM-08-17-E: the 3 GB cold load runs 21.4-25.8 s, 650 ms under the gateway cap at worst

- status: resolved (2026-09-28, plan 08-30: the 3,008 MB tier is superseded; its lever "the 10 GB
  tier" was taken and measured: at 10,240 MB the download fell to 8927-9139 ms and 8 proven-cold samples
  ran 22,647-27,015 ms, 2,985 ms under the cap at worst. The successor risk is D-ITEM-08-30-B)
- **What:** all four cold samples passed (< 30000 ms), which makes the outcome deployed-passed. But:
  - The worst sample was 29350 ms end-to-end.
  - load_ms was 21411-25813 against the contract's 17000 ms 3 GB extrapolation.
  - download_ms was 13494-17831. Parts timed out at 8000 ms and were retried.
- **Where the budget went:** classify compute came in UNDER its own extrapolation: about 26 ms/token on
  graviton2 and about 16-17 ms/token on graviton3, against 40. Max Memory Used was 2481-2483 MB, a
  measured headroom of about 525 MB against the ~350 MB projected. So the S3 download is what uses up the
  cap, not compute and not memory.
- **Owner:** 08-18 / 08-12. Candidates: the S3 part-size/concurrency and the 8 s attempt timeout,
  D-ITEM-08-17-C (software sha256, 2.7-3.5 s), and the 10 GB tier.
- **External observation (orchestrator, 2026-09-27 about 23:10Z, from the user's laptop outside AWS):**
  - **Cold call.** `POST https://aprender-mcp-decide.us-east.true-mcp.com/mcp`, a `tools/call classify`
    of one stance tweet.
    - HTTP 200 with a client-side `time_total` of **31.05 s**, and identity `24a44d7e…` == H.
    - The client time exceeded 30 s, and the edge still answered 200. So the edge's effective cutoff is
      not a strict 30.000 s measured from the client, and **a real remote client can see a cold call
      land at or over 30 s**.
  - **Warm call.** HTTP 200 in 2.16 s. "Every life deserves protection from conception. #prolife"
    returned `against`, with probabilities none 0.072, against 0.835 and favor 0.094. That is correct
    for the TweetEval abortion target (legalization of abortion).
- **CloudWatch correlation (08-18, read-only, matched by time window; `probe_id=none`):**
  - The cold invocation started at 23:10:11Z.
  - `load_ms=26202`, the slowest measured, above all 4 rule samples. It split as download 15125,
    sha 3521 and build 7468. 8 S3 part attempts timed out at 8 s and were retried.
  - It ran on graviton2 and classified 67 tokens in 1791 ms.
  - REPORT: 28008 ms, Max Memory Used 2482 MB, Init 64 ms, `success`.
  - Client minus REPORT is about **3042 ms**. For the 4 rule samples the same gap was 753-785 ms, so
    client overhead is not a constant ~0.8 s.
  - The 2 warm invocations ran 1.80 s each in-function.
- **Risk arithmetic, not a measurement.** A maximal 120-token cold call on that graviton2 environment
  would run about 26202 + 120 x 26.8 = about 29418 ms in-function. That leaves about 580 ms under the
  function's own 30 s timeout, and it comes to about 32.5 s at the client with this call's overhead.
- **The label stays deployed-passed.** The plan's rule is defined on the `laya-deploy-verify` samples:
  CloudWatch-proven cold, maximal, probe-id matched, and all < 30000 ms. The external call is not one of
  them (it was non-maximal, with no probe id and a different client), and its in-function time was
  itself under the cap. It is recorded as additional risk evidence, not a relabel. Note also that the
  rule's `elapsed_ms` is client-side through the same edge, not in-AWS.
- **Levers. These are recommendations only; none was acted on:**
  1. The S3 download: 13.5-17.8 s at 3 GB (15.1 s on the external call), against about 9 s at 10 GB.
     Tune part size, part concurrency and the 8 s attempt timeout.
  2. sha2 `asm` (D-ITEM-08-17-C): about 4.5 s priced, 2.7-3.5 s measured.
  3. Restore the 10,240 MB tier when AWS approves the limit (budget 1024 built tokens, 8 texts).
  4. A warm floor, or an async MCP Task front door.
- **Status: open.** The input to 08-12 is the label (deployed-passed) plus this margin risk.

## From plan 08-18

### D-ITEM-08-18-A: the decide endpoint is left RUNNING, open, by the user's decision

- status: open (the user's call; nothing for an executor to do)
- **Posture, verified read-only at 23:12:15Z on 2026-09-27:**
  - `aprender-mcp-decide` has no reserved concurrency (`get-function-concurrency` returns none).
  - Configuration: 3008 MB, arm64, Timeout 30, `Active` / `Successful`, pin == H.
  - The edge `/health` returns 200 `serverId aprender-mcp-decide`.
  - Auth is off, and the user accepted that risk. Every cold call buys about 25-28 s of 3 GB compute.
- **Why it is not contained:** the user asked to keep it serving for the pmcp.run admin UI. 08-18
  made no AWS write.
- **What containment WOULD be:** `just laya-teardown aprender-mcp-decide dev ze-kasher-dev`. It sets
  reserved concurrency 0, and `get-function-concurrency` must then read 0. Every invocation is refused,
  warm instances included. The stack-declared weights read stays attached, where it is inert. Resume
  with `aws lambda delete-function-concurrency --profile ze-kasher-dev --function-name aprender-mcp-decide`.
- **Full removal (NOT run):**
  - Destroy the deployment:
    `just _laya-crates-root-swap crates crates/aprender-mcp-decide-lambda/.pmcp/deploy.toml models/decide/destroy-aprender-mcp-decide.state cargo pmcp deploy destroy --manifest-path crates`
  - Remove the weights: `aws s3 rm --profile ze-kasher-dev --recursive s3://<weights-bucket>/decide/aprender-mcp-decide/`
- **Owner:** the user. Re-open trigger: cost, abuse, or a finished admin-UI test.

## From plan 08-12

The CI checkpoint (Task 1) was answered on 2026-09-27: **option A, narrowed**. The user said:

> "Option A. Regarding the make question, I'm not sure as the Aprender is designed for Rust, and our
> addition of converting Python algorithm and verifying that it give the same results is something
> that I've added only recently. Not sure that we need to have it as part of the CI, and we can bypass
> them or protect them with a specific flag."

Applied: ci.yml's integration step gained exactly `cargo test -p aprender-decide --test ui` and
`cargo test -p aprender-mcp-decide --test e2e_stdio`, the two pure-Rust targets. Everything below
stays out of CI.

### D-ITEM-08-12-B: the Python-parity / real-weights suite is LOCAL ONLY, by the user's decision

- status: open by design (the user's call, not a gap an executor should close)
- **Not in CI:**
  - the four env-gated targets `-p aprender-decide --test laya_parity`, `--test fail_closed_vectors`,
    `--test demo_run` and `--test python_records`;
  - the torch-free Python self-tests (`metrics.py`, `data.py`, `gate.py --selftest`);
  - the torch lifecycle.
- **What CI covers instead:** the Rust verifier's lib tests re-derive the gate metrics, split
  disjointness, the synthetic-variant refusal and the A1/A3 refusals.
- **How it runs:** `just laya-verify-suite` in one local command, behind the targets' existing env
  flags (`LAYA_MODEL_DIR`, `LAYA_FAIL_CLOSED_VECTORS=1`, `LAYA_DEMO_RUN=1`, `LAYA_PY_RUN_DIR` +
  `LAYA_PY_DATA_DIR`). The model dir defaults to the pinned snapshot `55cf4c4e…`. An armed leg that
  prints `SKIP:` fails the recipe. The 2026-09-27 run's result is recorded in 08-12-SUMMARY.md.
- **Re-open trigger:** a CI image that carries the base snapshot and the Python trainer, or the user
  asking for the Python surface in CI.
- **Obligation until then:** run `just laya-verify-suite` before changing `aprender_decide::{pack,verify,laya}`,
  `scripts/laya_train` or `contracts/laya-*.yaml`.

### D-ITEM-08-12-C: `make contract-audit-phase8` is still tier3-only, so it is NOT a CI gate

- status: open
- **Why:** CI does not run `make tier3` (ci.yml, #2512). No existing CI step invokes `make`, and
  whether `make` exists in the `sovereign-ci:stable` image is unconfirmed, so the audit was
  deliberately not added to the docker chain. The strict audit (any BIND- line refused, the single
  `PHASE8_LIVE_EXEMPT` now empty, row-to-definition resolution) runs locally only, through
  `make tier3` or directly.
- **Closes when either:**
  - the image is confirmed to have `make`, and `make contract-audit-phase8` is appended to the
    integration chain (a CI-workflow change, so it needs a human check-in); or
  - a make-free invocation of the same audit is wired in, for example a script the Makefile target
    and CI both call.

### D-ITEM-08-12-D: 15 FULL-run failures could not be controlled on the base, because the disk was full

- status: open
- **What failed:** in the no-fail-fast FULL run at HEAD, 15 `aprender-qa-runner` tests failed with
  `Os { code: 28, kind: StorageFull, message: "No space left on device" }`:
  - 14 in `dimensional_check::dimensional_check_tests_tokenizer_dtype`;
  - 1 in `layout_contract::tests`.
  Each writes a TempDir holding a `model.safetensors` of up to about 1 GiB.
- **Why they were not controlled:** the base-commit control could not run them safely with about
  4 GiB free. A first full-workspace control was killed by its disk watchdog at 1 GiB free.
- **Attribution by evidence, not by control:** Phase 8 changed no file under
  `crates/aprender-qa-runner` (`git diff --name-only 59d9ed07f..HEAD` is empty for it), and the
  failure is the host's free space.
- **Host debris, left for the user:** the killed control leaked 22 `.tmp*` TempDirs in `$TMPDIR`,
  6.0 GiB, all born at or after 17:27:31 on 2026-09-27. Removing them was refused by the
  permission classifier, so they are still there.
- **Closes when:** the same `-E` selection passes (or fails identically on 59d9ed07f) on a host
  with at least 15 GiB free.

## Phase 8 close-out — open work

Written by plan 08-12 (D-18) from the two outcome records: `08-GATE-RUN-EVIDENCE.json` (`outcome:
gate_pass`, `deploy_eligible: true`, sha256 `24a44d7e…`) and `08-LIVE-DEPLOY-EVIDENCE.json` (`final:
true`, `outcome: deployed-passed`). One line per item that is still open, with what re-opens or
closes it. Nothing below is implied done.

- **D-ITEM-08-01-A** (and its re-measures 08-03-A, 08-04-A, 08-05-A, 08-06-A, 08-09-D): the
  strict-test-binding guard is still VACUOUS (`spectral-indices-v1.yaml` has no `kani_harnesses:`).
  Every Phase 8 verify used the lifted-copy measurement, in which no Phase 8 contract dangles and only
  chronos-bolt-parity-v1 and setfit-encoder-conformance-v1 do. Closes when a change adds the Kani
  section there and fixes or re-cites the 17 pre-existing dangling references (never by raising the
  baseline).
- **D-ITEM-08-13-A:** x86_64 re-score parity is unmeasured; every parity number is aarch64. Closes on
  the first x86_64 value recorded in laya-parity-v1 (spike 028's procedure). Re-opens as a blocker if
  the decide server is ever deployed on x86_64.
- **D-ITEM-08-17-E:** the 3 GB cold path is 650 ms under the gateway cap at worst (29350 ms), and an
  external cold call took 31.05 s at the client. Levers in the item (S3 part size and timeout, sha2
  `asm`, the 10,240 MB tier, a warm floor). Re-opens `accepted_region_cold` if any cold sample at this
  tier reaches 30000 ms.
- **D-ITEM-08-17-C:** the sha256 pin runs sha2's software backend on aarch64 (2.7-3.5 s per cold load).
  Owner: a perf plan; the measured `sha_ms` is its baseline.
- **D-ITEM-08-17-D:** pmcp.run platform finding (sticky edge error state that `/health` cannot see; the
  edge's effective cutoff is undocumented). The user's to raise upstream; recommend only.
- **D-ITEM-08-18-A:** the decide endpoint is left RUNNING with auth off, by the user's decision. Re-open
  trigger: cost, abuse, or a finished admin-UI test (containment: `just laya-teardown`).
- **D-ITEM-08-10-A / D-ITEM-08-10-B:** one decide model per workspace under the shared-crates-root
  deploy, until cargo-pmcp's resolver is fixed upstream (08-10-B). Open as 08-18 left them.
  D-ITEM-08-10-C is superseded (one live assumption refuted and replaced, the replacement passed live)
  and D-ITEM-08-10-D is resolved (auth off, provider none, risk accepted); a future `auth=on` deploy
  re-opens 08-10-D's provider choice.
- **CI wiring (plan 08-12 Task 1, decided 2026-09-27: option A, narrowed by the user):**
  - Now in CI: `-p aprender-decide --test ui` and `-p aprender-mcp-decide --test e2e_stdio`, on
    ci.yml's integration line. The e2e real-model leg SKIPs there.
  - Still dark, on purpose: the four env-gated parity targets and the Python self-tests. They run
    locally through `just laya-verify-suite` (**D-ITEM-08-12-B**).
  - Still tier3-only: `make contract-audit-phase8`, pending confirmation that the CI image has `make`
    or a make-free invocation (**D-ITEM-08-12-C**).
- **D-ITEM-08-12-D:** 15 `aprender-qa-runner` FULL failures (ENOSPC) were not controlled on the
  base, because the disk was full. There are also 6.0 GiB of leaked TempDirs for the user to remove.
  Closes on a re-run with at least 15 GiB free.
- **Fixed in 08-12, not open:** the contract dependency cycle
  `decide-apr-v1 -> laya-finetune-gate-v1 -> laya-parity-v1 -> decide-apr-v1`, present since 08-01
  (commit 7d5c65f4b). It made three `aprender-contracts-cli` tests fail at every Phase 8 commit, and
  so `workspace-test` would have been red on push.
- **D-ITEM-08-12-A (new, follow-up):** `crates/aprender-decide/src/verify.rs` is 2454 lines, one
  module holding every gate, re-score, seed and shift check. Not split in 08-12 (the plan does not
  require it, and a split would move every binding row this plan just resolved). Owner: a refactor
  plan, which must re-run `make contract-audit-phase8` after moving any bound function.
- **Pre-existing, not Phase 8's code, still open:** D-ITEM-08-01-B (README claims 110 CLI commands, the
  registry lists 111; re-measured 2026-09-27), D-ITEM-08-04-B (`cargo fmt --all -- --check` still fails
  only on the `fdf6b1802` files in aprender-image and aprender-mcp-chronos), D-ITEM-08-06-B (README's
  layout tree still says "(82 crates total)"), and `.planning/WINDOWS.md` refusing appends (08-03-A).
- **D-ITEM-08-07-A:** no longer open in substance. The bootstrap handler ran under the real Lambda
  runtime in 08-17/08-18 (the `decide.load` line in CloudWatch, the `x-decide-load` header on every
  cold sample, the identity probe), and its failed-load path ran live too (22:16:47Z, `decide.load
  failed` before the grant, 08-17 resume attempt 2; the HTTP status of that platform call was not
  observed). Left without a status line by 08-07; recorded closed here.

Resolved by the records and therefore not open: D-ITEM-08-11-A (deployed-passed; plan 08-12 bound
`accepted_region_cold` implemented and FALSIFY-DECIDE-TOOL-009 names its live harness and evidence),
D-ITEM-08-09-A (08-16's gate_pass ran the stdio real-model leg), D-ITEM-08-17-A and -B, and
D-ITEM-08-09-B/-C, D-ITEM-08-14-A/-B as their own status lines say. There is no D-ITEM-08-16-A: the
declared run passed.

The calibration-spike todo (`.planning/todos/pending/spike-laya-calibration-slice-and-temperature-cap.md`)
is ANSWERED by spikes 027/028 and amendments A1-A3 (declared in 08-13, run in 08-16); the workflow
moves the todo file, not this plan.

## Gap round 08-19..08-32 (planned 2026-09-28)

One class-wide round, by the user's decision: each defect CLASS gets an invariant (a `must_haves`
truth), a checked-in enumeration of its whole surface, one sweeping test or gate, and a red-side
(mutation) proof per bound. Owning plans:

- **A, provenance binding:** 08-19 (load: every manifest leaf bound to its blob), 08-21 (verify: every
  run-dir field bound or report-only), 08-23 (served fields), 08-25 (probe evidence), 08-27 (f_avg).
- **B, untrusted-input bounds:** 08-20 (apr-format reader, ModernBERT layer), 08-26 (artifact table and
  sweep), 08-23 (request table and sweep), 08-24 (Lambda-owned request rows).
- **Added at plan revision (2026-09-28), found missing by the plan checkers:** A4-6 (Lambda GET health
  said ok:true without reading the config) -> 08-24 Task 2 (`health()`, 503 ok:false on a config error);
  S4 = R5 + AL2 (tests/fail_closed_vectors.rs duplicated tests/common's helpers and a sha256 helper) ->
  08-21 Task 1 (`mod common;`, library `artifact_sha256_hex`); the V4-a resume fix (ea940faec) had no
  red-side proof -> 08-24 Task 3 mutation row. Real-weights legs in waves 17-18 are serialised host-wide
  through one lock (`heavy`), and an OOM-killed leg is re-run, never read as a refusal.
- **C, gate honesty:** 08-22 (`scripts/laya_gates.tsv`, `just laya-gates-selftest`).
- **D, claim honesty:** 08-28 (classify tool claims), 08-31 (`scripts/laya_claims.tsv`, `just laya-claims-check`).
- **E, Rust-Python agreement:** 08-27 (numeric), 08-29 (Python reader, rows, refusals).
- Composition and CI: 08-32 (`just laya-gap-regression`; first CI run of Phase 8 code).

Owner decisions taken inside the round (plan 08-31 records each outcome here): WR-03 and the V5-a
wire-code side note (08-28), V4-b and any redeploy (08-30), D-14 publication (08-31). A "keep" or
"defer" answer leaves the item below with its decision date.

### Deferred at planning time (reason, re-open trigger)

- **V13-e** (NFC Unicode 17 in Rust vs 15.1 in Python 3.13.7): fails closed (a Unicode 16/17 composition
  pair makes Rust see an overlap Python did not, so verify refuses late; it never admits a leak). The fix
  is pinning one Unicode version for both sides, a workspace dependency decision. Re-open on an honest run
  refused for such a pair, or on any unicode-normalization / Python upgrade.
  status: open
  **Owner:** a workspace-dependency plan (pin one Unicode version for Rust and Python). Final status 08-31: still deferred.
- **V14-b** (PLAUSIBLE) and **CV3** (the contract-audit-phase8 resolver re-implements what `pv
  verify-bindings` should do): extending pv is its own ticket (CLAUDE.md contract rule, option 2). The
  shell resolver gains a must-match / must-not-match table in 08-22. Re-open when pv resolves module paths.
  status: open
  **Owner:** an aprender-contracts (pv) ticket. Final status 08-31: still deferred; the 08-22 ERE case table
  (`make contract-audit-phase8-selftest`, gate row resolver-ere) is the interim guard.
- **R1** (`normalize_text` duplicates aprender-contrastive-data's normalized hash): quality only; reuse
  would add a cross-crate dependency edge.
  status: open
  **Owner:** a /simplify follow-up. Re-open if the two hashes ever disagree (verify refuses late). Final status 08-31: still deferred.
- **R3 / AL6** (contract-audit-phase8 is a copy of the phase-6 target): a Makefile refactor spanning a
  non-Phase-8 target.
  status: open
  **Owner:** a Makefile refactor plan. Re-open when either audit target changes again. Final status 08-31: still deferred.
- **R4 / AL4** (third copy of the loopback proxy; an in-process router instead): architectural and shared
  with the chronos and setfit Lambda crates; its own plan.
  status: open
  **Owner:** a thin-server architecture plan (all three Lambda crates). Re-open on a fourth copy. Final status 08-31: still deferred.
- **R7** (fixtures.py re-implements `data.calibration_split`): regenerating the tiny fixture moves the
  committed goldens (laya_tiny.apr.sha256) for no behaviour change.
  status: open
  **Owner:** a fixtures-owning plan, together with the laya_tiny ece_pre drift found in 08-29 (below). Final status 08-31: still deferred.
- **R8 remainder** (bucket naming x4, the duplicated eligibility parse in laya-upload): refactor only; the
  sha256 helper part is fixed in 08-22.
  status: open
  **Owner:** a /simplify follow-up. Final status 08-31: the sha256 part closed in 08-22; the rest still deferred.
- **S2, S3, S5, S7, S8** (redundant re-hash in check_inputs, the lambda build_server alias, duplicated
  report fields, write-only fields, hand-copied NaN-max): simplification only, no defect; a /simplify
  follow-up.
  status: open
  **Owner:** a /simplify follow-up. Final status 08-31: S1 closed for pack (08-31 Task 1 removed
  `pack::sha256_hex`; `crate::digest::sha256_hex` is the one helper) but the Lambda crate's re-export
  `pub use aprender_decide::artifact::artifact_sha256_hex as sha256_hex` is still a second public name
  (simplification only); S6's stale docs and the lock-named test closed in 08-24, its code simplifications
  still deferred; S2, S3, S5, S7, S8 still deferred.
- **EF2..EF8** (per-element F16 widening, double finiteness scan, per-row forwards, full embedding widen,
  double pack probe, verify materialising the checkpoint, base read twice): owner is the perf plan that
  already owns D-ITEM-08-17-C / -E; re-open if any cold sample comes within 1 s of the 30 s cap.
  status: open
  **Owner:** the perf plan that owns D-ITEM-08-17-C. Final status 08-31: still deferred; the 10,240 MB
  samples (worst 27,015 ms) leave 2,985 ms, so the 1 s trigger has not fired.
- **CV6** (twelve new functions above complexity 10): with D-ITEM-08-12-A (the verify.rs split).
  status: open
  **Owner:** the verify.rs split refactor (D-ITEM-08-12-A). Final status 08-31: still deferred.
- **V2-a residual** (apr-format's MAX_METADATA_SIZE is unenforced for non-decide consumers): shared reader
  policy that could refuse legitimate large-metadata APR files; the decide ladder bounds it at rung 2.
  status: open
  **Owner:** an apr-format reader-policy plan. Re-open when a non-decide consumer reads untrusted APR files.
  Final status 08-31: still deferred.
- **Guard offenders that are not Phase 8's**, so the next CI run's red is attributed: hand-rolled argv in
  aprender-mcp-chronos, aprender-mcp-forecast, aprender-mcp-setfit, aprender-mcp-setfit-train; cascade
  TIERS missing aprender-contrastive-data. D-ITEM-08-01-A (strict-binding vacuity) stays open; the Phase 8
  contracts' test references are checked by `just laya-claims-check` from 08-31.
  status: open
  **Final status 08-31:** Phase 8's own share of all three guards is closed (parser: 08-23; duplicate
  `bootstrap` bin: 08-22; cascade TIERS: 08-31 `publish-false`). The offenders above are not Phase 8's and
  are itemised with owners in the final-status section at the end of this file.

### Not open (refuted by the /code-review max verdicts)

V8-a, V8-b, V12-d, V4-d, V9-c, V9-e, and the behaviour half of V14-d (its comment half is 08-31);
raw candidates AL8, C2-8, D2-5 and A2-7 map to those refuted verdicts.

### Found during plan 08-20 (out of scope, pre-existing)

- **apr-format golden_v2_f32_writer_is_byte_identical fails at HEAD 70125a6b1** (writer produces 516 bytes,
  `crates/apr-format/tests/fixtures/golden_v2.apr` holds 1092). Measured with plan 08-20's reader change
  reverted: same failure, so it is writer/fixture drift, not the strict-index change. The other three golden
  targets (which READ the fixtures) pass. Not in CI's explicit `--test` list, which is why it stayed dark.
  status: open

### Found during plan 08-22 (out of scope, pre-existing)

- **aprender-compute, aprender-core and aprender-decide recompile on EVERY `cargo run --release -p aprender-decide`**
  (about 17 s each, measured twice back to back on `just laya-inspect` with no source change). Some build
  script's rerun condition never settles. It makes every pack_laya recipe pay a rebuild, so
  `just laya-gates-selftest` spends most of its non-weights time compiling. Not a correctness defect.
  status: open

### Found during plan 08-24 (out of scope)

- **`crates/aprender-mcp-decide-lambda/examples/probe.rs` still opens with a crate-wide
  `#![allow(clippy::disallowed_methods)]`.** Plan 08-24 removed the lib's and the bootstrap's crate-wide
  allows (IN-05, Lambda half) but its files list did not include the example, which is its own crate
  target: an unwrap added anywhere in the probe CLI still passes clippy. Same fix shape: item-level allows
  on the functions whose `json!` needs them, then a planted-unwrap check.
  status: open
- **decide-tool-boundary-v1 `frame_http` and `probe_id_header` rows name tests that do not run their hostile
  case.** Their `test:` fields still name `tests::server_config_is_stateless` and
  `tests::probe_id_accepts_only_short_safe_ids`; the hostile cases now run in
  `tests::lambda_request_rows_are_swept` (plan 08-24). The contract's own header says `test` is "the test
  that runs the row's hostile case". Plan 08-24 does not own the contract; repointing both rows (and the
  frame_http `note:`) is a contract edit for whichever plan next versions decide-tool-boundary-v1.
  status: resolved
  **Resolved by:** plan 08-28 (commit 890b769e9, decide-tool-boundary-v1 5.0.0): both rows now name
  `tests::lambda_request_rows_are_swept` first, and the frame_http `note:` says the hostile case runs there.
  Plan 08-31 ledgers both rows (`tool-bound.frame_http`, `tool-bound.probe_id_header` in
  scripts/laya_claims.tsv), so a renamed test fails `just laya-claims-check`.

### Found during plan 08-29 (out of scope)

- **`just laya-fixtures` is no longer byte-identical: `laya_tiny/gate-report.json` `fine_tuned.ece_pre`
  regenerates as 0.014317405789640192 (committed 0.014317405789640136).** Measured on a pristine detached
  worktree at 6ea8d0cc6 (before any 08-29 edit), so it is pre-existing: plan 08-27 moved metrics.py's ECE
  to exactly-summed f64 (221556654) and the fixture, last written by d8030eab6 (08-02), was not
  regenerated. 5.6e-17 apart, far inside gate_metric_recompute_abs; Rust recomputes rather than trusting
  it. Regenerating the committed fixture is a fixtures-owning plan's call (it moves a CI fixture's bytes).
  status: open

### Found during plan 08-30

- **D-ITEM-08-30-A: the owner's /tmp (ephemeral storage up to 10 GB) idea is recorded, not acted on.**
  The owner reported the account can configure /tmp up to 10 GB and suggested local storage may load faster
  than the S3 path (08-LIVE-REDEPLOY-EVIDENCE.json `owner_note_10gb`). Caveat, unchanged: /tmp is per
  execution environment and EMPTY on every cold start, so it helps warm reuse, not the cold load this
  boundary is priced for, unless the model is baked into a container image or mounted from EFS. The memory
  half of that note became the `restore-10240-tier` decision (decide-tool-boundary-v1 7.0.0); the /tmp half
  is a separate hosting change with its own measurement.
  status: open

- **D-ITEM-08-30-B: the 10,240 MB budget is accepted on the whole-request rule, not on its per-term derivation.**
  Measured at 10,240 MB (08-LIVE-REDEPLOY-EVIDENCE.json `live`, 8 proven-cold samples): build including the
  probe replay ran 2808-2928 ms against its 2404 ms price (404-524 ms over on every sample), and gateway
  overhead ran 2751 ms once against its 896 ms price. The per-term-max re-derivation gives 648 tokens, below the
  declared 800. The owner kept 800 / 8 texts (`keep-800-apply-10739`, 08-CONTEXT.md D-18) because every cold
  sample finished under 30 s (worst 27,015 ms, 2,985 ms slack); decide-tool-boundary-v1 8.0.0 states both.
  Re-open trigger: any cold sample at or over 30,000 ms (containment first), a Graviton generation change,
  or a re-measure where a second term exceeds its price. Levers: the sha2 `asm` backend (D-ITEM-08-17-C,
  3.5 s of software sha256) and re-pricing (b) from these samples.
  status: open

### Found during plan 08-32 (out of scope, pre-existing)

- **D-ITEM-08-32-A: `aprender-contracts-cli` `every_contract_generates_book_page` (tests/book_coverage.rs) is red
  at the pre-Phase-8 base.** It calls `parse_contract` on every top-level `contracts/*.yaml`, and
  `contracts/binding.yaml` (a binding registry, not a contract, present since 4dcf9b21f #2277, 2026-07-04) has no
  `metadata:` field: `Failed to parse YAML: missing field 'metadata'`. Control: a detached worktree at d37f2fefc
  (origin/main, before any Phase 8 commit) fails identically (rc 101). Phase 8 changed neither the test,
  binding.yaml nor the schema parser. `just laya-gap-regression` skips it by exact name and prints the skip; the
  contract-cycle guard (`certify_on_real_contracts`, `verify_pipeline_on_real_contracts`,
  `verify_pipeline_json_on_real_contracts`) runs by name and passes. Not in CI's explicit `--test` line.
  status: closed (2026-09-29, plan 08-33: resolved by the upstream merge)
  **Owner:** the aprender-contracts maintainers (skip non-contract YAML in the test, or move binding.yaml).
  **Re-open trigger:** book_coverage added to a required CI line.
  **Refreshed 2026-09-29 (08-32 close-out):** after the upstream merge the failure is unchanged (binding.yaml
  still has no `metadata:`); it is one of the five local failures in 08-CI-RUN-EVIDENCE.json (B5). Upstream's
  contract-hygiene gates that now sit beside it ARE in CI's Integration-tests step, and they are the subject
  of plan 08-33; this one is not (it is not in any explicit-test-commands.d fragment).
  **Correction 2026-09-29 (plan 08-33):** the "unchanged" refresh above was wrong. `cargo test -p
  aprender-contracts-cli --test book_coverage` is GREEN on HEAD (1 passed; 0 failed): the upstream merge brought
  upstream's fix (the test's walker now uses `is_contract_yaml`, which skips `binding.yaml`). The "five local
  failures" of B5 were therefore four contract-hygiene reds plus the stale graph, not book_coverage. The stale
  `book_coverage` exemption in the regression recipe is removed by plan 08-34.
- **The gap regression skips apr-format `golden_v2_f32_writer_is_byte_identical` by exact name** (the 08-20
  entry above, still open); every other apr-format test runs and must pass.

## Gap round 08-19..08-32: final status (plan 08-31, 2026-09-29)

Every finding of `08-REVIEW.md` (1 critical, 9 warnings, 8 info) and `08-CODE-REVIEW-FINDINGS.md` (the
ea940faec list, the V-verdicts and the raw candidates) after plans 08-19..08-31. A finding is **closed**
(the plan that closed it and the test or gate that keeps it closed; where the claim is ledgered, its
`claim_id` in `scripts/laya_claims.tsv`, which `just laya-claims-check` checks), **decided** (an owner
decision and its date), **refuted**, or **still open** (a list entry below with its reason, owner and
re-open trigger). Plan 08-32 (`just laya-gap-regression`, the first CI run of Phase 8 code) is the
round's last plan and is not yet run.

### Closed: 08-REVIEW.md

| Finding | Closed by | Kept closed by (ledger claim_id) |
|---|---|---|
| CR-01 | ea940faec (verify: `ArtifactNotFromRun`), 08-19 (load: `check_manifest_bindings`, rung 4 (e)), 08-21 (verify: base identity, base-dir pins, `run_field_bindings`) | FALSIFY-DECIDE-APR-013, apr.manifest.bindings, apr.verify_checks.2, gate.run_field_bindings |
| WR-01 | 08-20 (apr-format reader: names strictly increasing), 08-26 (`DuplicateTensor`, rung 4) | apr-bound.duplicate_names_reader, apr-bound.duplicate_names_ladder, apr.load_ladder.rung4-unique-and-bound |
| WR-02 | 08-21 (`verify_run` is `cfg(test) pub(crate)`; no production caller, so clippy's dead-code lint refuses a second door) | the compiler |
| WR-03 | 08-28 A-derive (owner, 2026-09-28); live since the 08-30 redeploy | mcp.description.truncation-sentence, FALSIFY-DECIDE-TOOL-005, context.D-09.amendment-08-28 |
| WR-04 | 08-20 | apr-bound.layer_row_length@aprender-core:lib |
| WR-05 | 08-29 (`data.jsonl_lines` == Rust `str::lines`) | py.data.jsonl-lines |
| WR-06 | 08-24 (`watch_loopback` exits 1) | lambda.readme.lazy-load |
| WR-07 | 08-22 (grant check reads attached policies, refuses wildcards / NotAction / any other S3 grant) | coverage.iam-read-only; gate rows grant-check, grant-listing-failure |
| WR-08 | 08-21 (`slice_need`) | decide.verify.slice-fraction |
| WR-09 | 08-29 (`contract.number`, k written as a float) | gate.rescore_noise.k-float |
| IN-01 | 08-31 (argmax folds from the first non-NaN element) | decide.argmax.nan-never-wins |
| IN-02 | 08-26 (inspect runs rungs 1-4 on a bounded read) | apr-bound.inspect_read |
| IN-03 | 08-25 (exact labels-segment parse) | lambda.probe.labels-segment |
| IN-04 | 08-25 (deploy.toml.template), 08-31 (tests/ui.rs header) | lambda.template.tell, decide.ui.runs-in-ci, ci.integration-line.ui |
| IN-05 | 08-23 (aprender-mcp-decide), 08-24 (Lambda lib and bootstrap). **Remainder open:** examples/probe.rs (the 08-24 entry above) | planted-unwrap clippy checks recorded in 08-23 / 08-24 |
| IN-06 | 08-24 | `tests::proxied_response_has_one_cors_origin` |
| IN-07 | 08-20 | `safetensors = { workspace = true }` in crates/aprender-core/Cargo.toml |
| IN-08 | 08-29 (`REFUSED early-stopping`, exit 2) | py.train.early-stopping-refusal |

### Closed: 08-CODE-REVIEW-FINDINGS.md

- **In ea940faec** (the list at the top of that file): V11-d, V4-a (its red-side proof is 08-24 mutation
  M7), V7-c, V1-a, V2-a (decide rung 2; the reader residual is deferred above), V5-a (no echo), V3-a,
  V3-b, V5-d, V10-a, V10-c, V13-b, V6-a, and V6-c in part (finished by 08-21).
- **In the round:** V8-c, V8-d, V6-b, V6-c, V6-d, V12-b (08-21); V11-a, V11-b, V11-c, V14-a (08-22);
  V5-c (08-23); V3-c, V3-d, V4-c (08-24); V12-a, V12-c for the probe CLI (08-25); V1-b, V9-a, V9-b,
  V9-d, V12-c for pack_laya (08-26); V6-e, V7-a, V7-b (08-27); V5-a's wire-code side note (B-iserror)
  and V5-b (the admission claim restated to the measured serial dispatch) (08-28); V13-a, V13-c, V13-d
  (08-29); V4-b and the SKIPPED #14 (08-30, by the owner decisions below); V10-b, V2-b, V2-c (08-20).
- **V14-c** (Phase 8 added one offender to each of three CI-required guards): the parser offender
  aprender-mcp-decide (08-23), the duplicate `bootstrap` bin (08-22, allowlist intent line) and the
  cascade offender aprender-decide (08-31, `publish = false`; ledger decide.cargo.publish-false and the
  gate row cascade-guard, which now runs a case) are all closed. The guards' other offenders are not
  Phase 8's (below).
- **V14-d:** the behaviour half was refuted; the comment half ("servers never call pack") closed in 08-31
  Task 1 (`crate::digest`, `safetensors_is_read_only_by_pack`; ledger pack-safetensors-scope,
  cargo-safetensors-scope, claude-md-safetensors-never-served).
- **Raw candidates that are not a verdict:** R2 (08-24: no second copy of rung 1's cap arithmetic; the
  `load_model_from_path` alias is S-group simplification), R5 / AL2 / S4 (08-21), R6 (08-25), R8's
  sha256 part and D3-4 (08-22), AL3 / AL5 (08-20), AL7 / D3-1 / C2-7 (08-22), A4-6 (08-24), A4-7
  (08-28), A4-8 (08-23; the stdio frame is the accepted row frame_stdio), A2-6 (08-22), D3-5, C2-4,
  C2-6 (08-29), CV1 (08-22), CV4 (08-22), CV5 (08-31), EF1 / AL1 / B1 (ea940faec), B2 (= V3-b). Every
  other raw candidate is marked `=` a verdict in that file and carries the verdict's status.
- **Refuted, not open:** V8-a, V8-b, V12-d, V4-d, V9-c, V9-e and V14-d's behaviour half; the raw
  candidates AL8, C2-8, D2-5 and A2-7 map to them.
- **D-ITEM-08-17-E** is resolved (its own status line, plan 08-30: the 3,008 MB tier is superseded); the
  close-out list line above predates that.

### Decided by the owner inside the round

| Item | Decision id (verbatim) | Date | Record |
|---|---|---|---|
| WR-03 and V5-a's wire-code side note | `A-derive B-iserror` | 2026-09-28 | 08-CONTEXT D-09 amendment; ledger context.D-09.amendment-08-28 |
| V4-b, the redeploy | `redeploy-and-measure` | 2026-09-28 | 08-CONTEXT D-18; 08-LIVE-REDEPLOY-EVIDENCE.json `decision` |
| the tier | `restore-10240-tier` | 2026-09-28 | 08-CONTEXT D-18; evidence `tier_decision` |
| pmcp.run's MemorySize cap | `raise-pmcp-run-cap`, then `retry-after-cap-confirmed` | 2026-09-28 | 08-30 SUMMARY |
| the deadline and the budget | `keep-800-apply-10739`: 800 built tokens / 8 texts kept on the acceptance rule (every cold sample < 30 s) although the per-term-max rule gives 648 tokens; DOWNLOAD_DEADLINE 10,739 ms; the stop-guard's reference set corrected to this tier's own downloads (max 9,065 ms), because the 3,008 MB downloads (13,494-17,831 ms) came from a superseded tier | 2026-09-28 | 08-CONTEXT D-18; evidence `deadline_decision`; ledger tool.648-vs-800, context.D-18.keep-800-apply-10739 |
| D-14, publication of aprender-decide | `publish-false` | 2026-09-29 | 08-CONTEXT D-14 amendment; `publish = false` in crates/aprender-decide/Cargo.toml; ledger decide.cargo.publish-false, context.D-14.amendment-08-31 |

Deployed facts at the end of the round: aprender-mcp-decide is live on pmcp.run at 10,240 MB serving
b615d8244, at most 8 texts and 800 built tokens per call (decide-tool-boundary-v1 9.0.0), bound
refusals are `isError` tool results (08-28, live since the 08-30 redeploy), DOWNLOAD_DEADLINE 10,739 ms,
auth off by the owner's decision (D-ITEM-08-18-A). Nothing was published to crates.io.

### Still open after the round

Tracked above with their own status lines, not repeated here: the planning-time deferrals (V13-e, V14-b /
CV3, R1, R3 / AL6, R4 / AL4, R7, R8 remainder, the S group, EF2..EF8, CV6, V2-a residual),
D-ITEM-08-01-A (strict-binding vacuity; Phase 8's three contracts are now covered by
`just laya-claims-check`, which lists every test their FALSIFY commands and bounds tables name),
IN-05's remainder (examples/probe.rs), the apr-format golden_v2 writer drift (08-20), the rebuild on
every `cargo run -p aprender-decide` (08-22), the laya_tiny `ece_pre` fixture drift (08-29),
D-ITEM-08-30-A (/tmp) and D-ITEM-08-30-B (648 vs 800). New entries:

- **tokenizer_pipeline is accepted, not bounded (08-26).** The ladder bounds the tokenizer blob's bytes
  and pins its digest, but not what a tokenizer.json pipeline does: a normalizer can declare an expanding
  replacement. Only a VERIFIED artifact is bound to Laya-en's pinned tokenizer (verify, the base pin).
  status: open
  **Reason:** bounding it needs an allowlist of pipeline components, which is a design decision.
  **Owner:** the user (a pipeline-allowlist decision), then a decide-apr-v1 plan.
  **Re-open trigger:** any server loading an artifact that did not pass `pack_laya verify`, or a second
  tokenizer family.
- **`cargo clippy -p aprender-decide --test laya_parity -- -D warnings` fails on aprender-compute's
  warnings (08-22).** Without `--no-deps` clippy lints the path dependency; aprender-decide's own
  targets are clean with `--no-deps`.
  status: open
  **Reason:** pre-existing, not Phase 8's code (aprender-compute).
  **Owner:** an aprender-compute lint plan (the toolchain-ceiling gate, CLAUDE.md "Linting", is the place
  it would surface).
  **Re-open trigger:** CI or `make tier2` running a dependency-inclusive clippy over aprender-decide.
- **aprender-core's clippy baseline holds one error: `src/demo/reliable/performance.rs:126:5`
  unreachable expression (08-27).** Measured on the unmodified tree before 08-27 touched aprender-core.
  status: open
  **Reason:** pre-existing, outside every Phase 8 file.
  **Owner:** aprender-core maintainers.
  **Re-open trigger:** `cargo clippy -p aprender-core --lib -- -D warnings` becoming a required gate.
- **D-ITEM-08-33-C: Hand-rolled argv parsers in aprender-mcp-chronos, aprender-mcp-forecast,
  aprender-mcp-setfit and aprender-mcp-setfit-train** (`check_no_hand_rolled_parsers.sh`, re-run 2026-09-29: 36 binary crates
  scanned, 4 hand-rolled, exit 1; aprender-mcp-decide is not among them since 08-23).
  status: open
  **Reason:** not Phase 8's crates (the SetFit and forecasting servers of earlier phases); the next CI
  run's red on this guard is theirs.
  **Owner:** the phases that own those servers; the fix shape is 08-23's (a clap derive `Cli`).
  **Re-open trigger:** the guard is CI-required, so it is red until they are fixed.
- **D-ITEM-08-33-D: aprender-contrastive-data is publishable but absent from the release cascade's TIERS**
  (`check_cascade_covers_all_crates.sh`, re-run 2026-09-29 after `publish-false`: 74 publishable crates,
  R1 lists only aprender-contrastive-data, exit 1).
  status: open
  **Reason:** not Phase 8's crate; whether it publishes (and the same one-way name question D-14 answered
  for aprender-decide) is its owner's call.
  **Owner:** the aprender-contrastive-data owner.
  **Re-open trigger:** the guard is CI-required, so it is red until it is added to TIERS or marked
  `publish = false`.

## Plan 08-32 close-out (2026-09-29): merge, draft PR #4634, CI evidence

Context: 08-32's must-have "workspace-test = success on the pushed head" is UNMET. The branch was scrubbed,
merged with upstream main (301 commits) and pushed as one draft PR (#4634, head 30bc2baaa at the time of the
runs). All CI runs are `action_required` (fork PR awaiting a paiml maintainer) and would stop at upstream's
contract-hygiene gates before the decide integration fragments 510/520. Evidence: `08-CI-RUN-EVIDENCE.json`.

- **D-ITEM-08-32-B: upstream's shrink-only contract-hygiene baselines are exceeded by this branch's contracts.**
  Measured locally after the merge (aprender-contracts-cli, CI Integration-tests fragments 336/338/455):
  389 passed, 5 failed. (i) `formal:` entries outside the declared vocabulary 1464 -> 1569 (+105; Phase 8's
  share is 25: decide-apr 6, decide-tool-boundary 6, laya-finetune-gate 9, laya-parity 4); (ii) kernel-kind
  contracts without `metadata.valid_under` 386 -> 398 (+12); (iii) `pv validate` errors: neon-blis-v1
  (`kind:` at top level instead of under `metadata:`) and spectral-indices-v1 (no `kani_harnesses`);
  (iv) `the_tracked_repo_graph_is_fresh` (regenerate); (v) = D-ITEM-08-32-A.
  status: repaired locally by 08-33 (committed, NOT pushed); closes when 08-34 observes workspace-test green
  **Owner decision (2026-09-29):** "Repair all (new gap plan)". Plan 08-33 did it and read the class, not the
  probe: (i) `formal_prose` 1569 -> 1464 (= baseline, 105 formals in 11 contracts, ledger
  08-33-FORMAL-REWRITES.json); (ii) `contracts_without_valid_under` 398 -> 386 (= baseline; twelve worlds
  declared, kind kernel written out where it was a default); (iii) `pv validate contracts` 2 failed -> 0
  (neon-blis-v1 lost its dead top-level `kind:`; spectral-indices-v1 gained declared-not-executed kani harnesses
  and its world at 1.1.0); (iv) `contracts/contracts.nt` regenerated; the hidden reds behind the five
  (aprender-contracts --lib x6, validate_contracts x2, chronos-bolt-parity-v1's nine dangling citations) are
  closed too. Nothing was loosened: the baselines and every gate source are byte-identical to 2817c6d97.
  The numbers and the planted-violation proofs are in 08-33-HYGIENE-EVIDENCE.json.
  **Owner:** plan 08-33 (the orchestrator plans it next).
  **Re-open trigger:** 08-33 does not bring the five failures to zero, or upstream tightens a baseline again.
- **D-ITEM-08-32-C: 17 of the 19 Phase 3/4 "SetFit tests (feature-gated)" CI legs were not re-homed.**
  Upstream restructured ci.yml (ci/sections.yml, #4433/#4441/#4471); this branch's old step (19 cargo commands,
  03-10/04-11) had no place in the new shape. Upstream fragments 010 and 020 already run the aprender-core
  setfit lib tests and the setfit conformance target; the other 17 legs (aprender-train, apr-cli and
  aprender-serve setfit legs and the cargo-check cells) are not run by CI on this branch.
  status: open
  **Owner decision (2026-09-29):** "Defer to deferred-items".
  **Reason:** upstream fragments 010/020 cover the setfit lib tests + conformance; the other 17 legs need new
  `ci/explicit-test-commands.d` fragments; deferring keeps the PR's CI footprint small, since upstream CI takes
  about 90 minutes.
  **Owner:** the Phase 3/4 SetFit owner.
  **Re-open trigger:** a SetFit regression that fragments 010/020 do not catch. (This also means Phase 4's
  SAFE-02 "in CI" clause stays open.)
- **D-ITEM-08-32-D: `contract-audit-phase6` fails identically before the merge.** 9 Phase 6.1 equations are
  unbound. Not Phase 8's; it is a tier3 target, not a CI gate.
  status: open
  **Owner:** Phase 6.1.
  **Re-open trigger:** contract-audit-phase6 becoming a required gate.
- **D-ITEM-08-32-E: two tests are Linux-only and fail on macOS.** `aprender-mcp`
  `exec_marker_bin_survives_a_transient_etxtbsy` (the file is identical to upstream) and the `aprender-profile`
  crate. Both are expected green on CI's Linux runners.
  **Rows (08-34 audit):** fragments 360 (`cargo check --workspace --benches`, aprender-profile bin test:
  `compile_error!("renacer requires Linux")`; the same command with `--exclude aprender-profile` is rc 0) and 490
  (the aprender-profile integration targets); `cargo build --examples --workspace` (aprender-profile example
  `validate_golden_trace` only; rc 0 with the crate excluded); lib test `aprender-mcp`
  `apr_bin::tests::exec_marker_bin_survives_a_transient_etxtbsy`.
  status: open
  **Reason:** platform-specific, not Phase 8's; local macOS runs of the phase regression skip or ignore them.
  **Owner:** their crates' maintainers.
  **Re-open trigger:** a Linux CI run failing either, or a macOS gate that includes them.
- **D-ITEM-08-32-F: `pr-review-quorum` fails on #4634 with "missing signed review receipt".** A paiml process
  (pull_request_target), not Phase 8 code. It needs a signed `pr-review` receipt for the PR.
  status: open
  **Owner:** whoever opens the reviewable PR(s) (see the PR-shape decision below).
  **Re-open trigger:** the PR leaves draft or is split.
- **D-ITEM-08-32-G: pmcp is pinned to 2.19.3 while upstream is on 2.21.0** (`519dc4125`). Phase 8's
  `request_bounds_table_is_swept` tripwire and the tool-boundary contract are verified on 2.19.3; upstream's
  2.21.0 was not measured against them.
  status: open
  **Reason:** the merge would otherwise have moved the transport under a contract proven on 2.19.3.
  **Owner:** Phase 8 / the pmcp upgrade.
  **Re-open trigger:** re-verify the tool-boundary contract on 2.21.0 (run the sweeps, the e2e_stdio target and
  the Lambda probe), then lift the pin; or upstream requiring 2.21.0 for another crate.
- **D-ITEM-08-32-H: PR shape.** #4634 stays ONE draft PR purely to obtain CI evidence ("Keep one draft for CI",
  2026-09-29); the user arranges maintainer approval upstream. Splitting into reviewable PRs (serving and
  contracts first; training code for upstream's .71-.75 window per the maintainer's note) is planned after
  phase verification.
  status: open
  **Owner:** the user, after `/gsd-verify-work` on Phase 8.
  **Re-open trigger:** phase verification finishing, or the draft becoming unmergeable (it is BEHIND main now).
- **D-ITEM-08-30-A (existing, still open): the 10 GB /tmp note.** Unchanged by 08-32.
- **Scrub note.** SHAs cited in 08-01..08-32 SUMMARY and evidence files written before the scrub are pre-scrub;
  resolve them via `08-SCRUB-COMMIT-MAP.tsv`. Entries above that cite `d37f2fefc` or `4dcf9b21f` are upstream
  commits and did not change.

### Found during plan 08-33

- **D-ITEM-08-33-A: neon-blis-v1 is a registry-flagged kernel contract, not a proven kernel.** SCHEMA-018 was
  repaired by deleting its dead top-level `kind: KernelContract` (upstream migrated 72 files identically;
  `metadata.registry: true` governs). It has no proof obligations, falsification tests or kani harnesses and no
  `metadata.kind` was added, because `kernel` would demand them: new claims.
  status: open (owner decision)
  **Owner:** the aprender-compute maintainers. **Options:** leave as a registry contract, or write the proof
  obligations and falsification tests for the NEON 8x6 kernel and declare it a kernel.
  **Re-open trigger:** any gate that starts to count registry contracts.
- **D-ITEM-08-33-B: kind alternatives for contracts that sit oddly as `kernel`.** To stay non-loosening, all
  twelve kernel-kind contracts without a world keep `kind: kernel` and gained `valid_under: {world: committed}`.
  setfit-benchmark-claims-v1 (a benchmark-claims contract) and tweet-eval-stance-benchmark-v1 (a
  dataset-evaluation contract) are the two that fit least; the tool-boundary contracts (decide, forecast) and
  the artifact-schema contract (decide-apr-v1) are candidates for `pattern` or `schema`. Reclassifying exits
  PROVABILITY-001 and the provability invariant they satisfy today, so it is an owner option, not done here.
  status: open (owner option)
  **Owner:** the owner of each contract's phase. **Re-open trigger:** a contract author wanting to drop the
  kani/falsification obligations, or upstream adding a claims-shaped kind.
- **D-ITEM-08-33-E: the aprender-train GPU-ledger tests fail on macOS (Linux `/proc`).** `cargo test -p
  aprender-train --lib` on this macOS host: 7647 passed, 21 failed, every failure in `gpu::guard`,
  `gpu::ledger` and `gpu::wait` (`VramLedger::is_alive` checks `/proc/<pid>/stat`, ledger.rs:77-79, so a
  reservation reads as dead and `total_reserved` returns 0). `crates/aprender-train` is byte-identical between
  3c0f2cf00 and HEAD; no test there reads a contract's `formal` or `valid_under`. Same class as D-ITEM-08-32-E.
  status: open
  **Owner:** aprender-train maintainers. **Re-open trigger:** a Linux CI run failing them, or a macOS gate that
  includes them.

### Found during plan 08-34

- **D-ITEM-08-34-A: three macOS host limits in the workspace lib universe (none Phase 8's).** Measured with CI's
  own nextest line on this macOS host, each byte-identical crate to upstream 2817c6d97, each expected green on
  Linux: `aprender-cgp` `profilers::system::tests::test_read_system_memory_total_mb` (reads `/proc/meminfo`),
  `aprender-orchestrate` `agent::driver::apr_serve::tests::test_find_apr_binary` (needs an `apr` on PATH; this host
  pins its binary), `aprender-test-lib` `brick::pipeline::tests::test_uuid_v4_generates_unique_ids` (run id is
  `SystemTime` nanos; macOS clock granularity gives fewer than 90 unique of 100, 3 of 3 reruns red here).
  status: open
  **Owner:** their crates' maintainers. **Re-open trigger:** a Linux CI run failing any of them.
- **D-ITEM-08-34-B: `aprender-core` `setfit::artifact::determinism::the_fixture_artifact_hash_matches_the_committed_golden`
  failed in the workspace lib universe and was NOT a host limit.** The fixture artifact hashed to `7169eac8...`
  against the committed golden `831c64c5...` whenever serde_json's `preserve_order` (unified in by pmcp) was in the
  build, and matched the golden without it (fragment 010: `-p aprender-core --features setfit`). At upstream
  2817c6d97 the workspace does not enable `aprender-core/setfit`, so the test was not in upstream's workspace lib
  run (upstream CI green on 2817c6d97 and 00052c012); on this branch `aprender-mcp-setfit` enables it, so it joined
  partition 1/3 of CI's workspace-test and was expected RED on Linux.
  status: **closed by commit 0b3ef0164** (orchestrator-authorised Rule-1 fix in plan 08-34): `write_setfit_apr`
  canonicalises the document (recursively sorted keys) before serialising, so the bytes are the sorted ones under
  either backing and the committed golden is unchanged. Proof: the determinism module (24 tests) green standalone
  (`preserve_order=OFF`) and under `-p aprender-core -p aprender-mcp-setfit` (`=ON`); CI's nextest line partition 1/3
  green for it; mutation-checked (dropping the call turns the unified mode red, removing the sort turns three tests
  red). Not in the audit's `registered_ids`: it is no longer a red, so a recurrence is a NEW finding (BLOCKED-PHASE8),
  never attributed.
  **Sweep residue (not an item):** `aprender-serve` `GgufToAprConverter::to_apr_bytes` builds its header metadata with
  `json!`, which is insertion-ordered under `preserve_order`; no hash or golden bears on it and the reader is
  order-insensitive, so it was not changed. Re-open trigger: a pinned digest of those bytes.
- **D-ITEM-08-34-C: `scripts/check_baseline_ratchets.sh` is red on this host for tool versions alone.** Three
  `tool_version` rows (pmat recorded 3.41.1, runner 3.15.0, twice; bashrs recorded 7.4.1, runner 6.66.3) and the
  vacuity row they cause; every ratchet row read against upstream's tip is `ok`.
  status: open (host limit; expected green where the pinned tools are installed)
  **Owner:** the host. **Re-open trigger:** a CI run failing the guard.
