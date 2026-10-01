---
phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server
plan: 10
subsystem: infra
tags: [laya, aprender-mcp-decide-lambda, cargo-pmcp, pmcp-run, deploy, fail-closed, resolver-proof, s3, iam, cold-start, justfile, offline-selftest]

requires:
  - phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server
    provides: "08-07 aprender-mcp-decide-lambda: bootstrap health body naming the package, probe example (--cold-first, --maximal, --probe-id), decide.load performed_load log line, .pmcp/deploy.toml.template"
  - phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server
    provides: "08-09 just laya-verify (the only deploy-eligibility check), laya-pack-fixture and the synthetic tiny artifact models/decide/selftest/laya_tiny.apr (golden 37d65159…, refused SyntheticNotDeployable)"
provides:
  - "Task 1 decision applied: shared-crates-root (deploy root `crates`, server `aprender-mcp-decide`), proven by EXECUTING cargo-pmcp 0.24.3's find_lambda_package_dir on this workspace"
  - "just laya-resolver-proof (git archive of the SDK at a recorded commit into a scratch dir + injected in-module test) -> models/decide/resolver-proof.txt"
  - "just laya-deploy-config / laya-deploy / laya-build-bootstrap / laya-deploy-selftest (Task 2) and laya-weights-bucket / laya-upload / laya-grant / laya-deploy-verify / laya-teardown (Task 3)"
  - "_laya-crates-root-swap: the shared root's setfit-train state restored byte-identically on every exit path"
  - "probe --plan-only: offline sizing of both accepted_region_cold shapes from a local .apr"
affects: [08-11, 08-12, decide deploy after the calibration spike, cargo-pmcp upstream]

actuals:
  tokens: 16625    # chars/4 over the realized diff: git diff d24ea8086..81137329c (61730 chars: justfile, probe.rs, cargo_pmcp_resolver_proof.rs) + deferred-items.md (4769 chars); this SUMMARY excluded
  tasks: 2         # Tasks 2 and 3 executed here; Task 1 was the blocking-human decision checkpoint, resolved by the user with no commit
  commits: 2       # MEASURED: git rev-list --count d24ea8086..HEAD before this SUMMARY commit
plan_head_before: d24ea8086e80e80e32aa9bc830fe59a24a4b5df2

tech-stack:
  added: []
  patterns:
    - "Resolver evidence = the tool's OWN resolver executed on this workspace (git archive at a recorded commit, test module appended in a scratch copy), never a build of the package by name"
    - "Every AWS-writing recipe calls laya-verify on the exact file before its first AWS call; cheap local refusals come first so each is reachable offline"
    - "An aws recorder first on PATH with a positive control, so AWS CALLS: 0 is a measurement rather than an assertion"
    - "Shared deploy root: back up, install the decide config ALONE (deploy/ cleared so cargo-pmcp cannot preserve another server's stack.ts), restore on EXIT/INT/TERM/HUP, verify by sha256, keep the backup and refuse the next swap if the restore is not byte-identical"
    - "Private just helper with [positional-arguments] so a command and its arguments pass through without re-quoting"

key-files:
  created:
    - scripts/laya_deploy/cargo_pmcp_resolver_proof.rs
  modified:
    - justfile
    - crates/aprender-mcp-decide-lambda/examples/probe.rs
    - .planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/deferred-items.md

key-decisions:
  - "USER DECISION shared-crates-root (2026-09-26): deploy root `crates`, server `aprender-mcp-decide`. The resolver proof executed cargo-pmcp 0.24.3 find_lambda_package_dir from SDK commit e0561f8c9 (builder.rs unchanged since c04fb4ccb): root crates -> crates/aprender-mcp-decide-lambda; control per-crate root -> crates/aprender-mcp-chronos-lambda"
  - "The swap also isolates crates/deploy/ (setfit-train's rendered stack.ts and bootstrap) and passes --regenerate-stack: per the cargo-pmcp source, an existing deploy/lib/stack.ts is PRESERVED, so the decide deploy would otherwise synthesize setfit-train's stack and its [environment] (the S3 URI and sha256 pin) would not be auto-applied"
  - "laya-deploy-config takes `auth` as a REQUIRED second argument (apr auth server env profile), because just cannot put a required parameter after a defaulted one. Deploying an open 10 GB function must be a stated choice (ASVS V2 note in the template)"
  - "laya-deploy refuses a resolver proof made for a different cargo-pmcp version than the installed one. The shared-crates-root analogue of the patch option's pin check"
  - "One decide model per workspace, and the upstream cargo-pmcp fix is recommended future SDK work, not done here (D-ITEM-08-10-A/B)"

patterns-established:
  - "Deploy selftest shape: synthetic artifact, DRY_RUN=1, recorder-shadowed aws with a control call, one CASE line per refusal naming its reason, AWS CALLS / DEPLOY MARKERS counters, positive case armed only by a real artifact that laya-verify must accept"

requirements-completed: [D-07, D-11, D-18]

coverage:
  - id: D1
    description: "The cargo-pmcp wrong-package trap is closed by the human-chosen mechanism (shared-crates-root), and cargo-pmcp's own find_lambda_package_dir, executed on this workspace, returns crates/aprender-mcp-decide-lambda for root crates + server aprender-mcp-decide. The per-crate control returns crates/aprender-mcp-chronos-lambda"
    requirement: "D-18"
    verification:
      - kind: integration
        ref: "CARGO_NET_OFFLINE=true just laya-resolver-proof ~/Development/mcp/sdk/rust-mcp-sdk e0561f8c9086aec56a7f70a3a51537f0671d3ffa <scratchpad>/resolver -> test deployment::builder::aprender_resolver_proof::aprender_decide_resolves_from_shared_crates_root ... ok (1 passed)"
        status: pass
    human_judgment: false
  - id: D2
    description: "Every deploy refusal is proven offline on the synthetic artifact with the aws recorder left empty: placeholder, sha-pin, resolver-proof, deploy-eligibility and upload-eligibility (the last two via laya-verify: SyntheticNotDeployable). AWS CALLS: 0 (recorder control passed) and DEPLOY MARKERS: 0. The positive dry run is an explicit SKIP"
    requirement: "D-07"
    verification:
      - kind: integration
        ref: "just laya-deploy-selftest -> DEPLOY SELFTEST OK (CASE 1-5 REFUSED as expected)"
        status: pass
    human_judgment: false
  - id: D3
    description: "crates/.pmcp/deploy.toml (setfit-train) and the whole shared-root state are restored byte-identically on success, forced failure, SIGTERM and an absent root. deploy.toml sha256 is 6661ab08… before and after, and the state sha256 is 7c6f25eb… before and after"
    requirement: "D-18"
    verification:
      - kind: integration
        ref: "just laya-deploy-selftest -> SWAP 1-4 + CRATES ROOT line; cmp against a pre-plan manifest of crates/.pmcp + crates/deploy"
        status: pass
    human_judgment: false
  - id: D4
    description: "laya-verify precedes the first AWS call (and cargo pmcp deploy) in laya-deploy and laya-upload. laya-deploy-verify's DRY_RUN plan names, per sample, the config bump, `--cold-first --maximal <shape>` and the performed_load=true CloudWatch correlation, for both shapes sized from the artifact, and exits 0 with 0 recorded AWS calls"
    requirement: "D-07"
    verification:
      - kind: other
        ref: "plan 08-10 Task 3 <verify> block (run with scratchpad log paths) -> T3 VERIFY PASS"
        status: pass
    human_judgment: false
  - id: D5
    description: "The arm64 decide bootstrap builds as a fresh aarch64 ELF naming aprender-mcp-decide-lambda (a build check, not resolver evidence)"
    requirement: "D-11"
    verification:
      - kind: other
        ref: "just laya-build-bootstrap -> BOOTSTRAP aarch64 OK target/aarch64-unknown-linux-gnu/release/bootstrap (25958736 bytes)"
        status: pass
    human_judgment: false
  - id: D6
    description: "The live halves (bucket, upload, deploy + identity assertions, grant, cold verify, teardown containment) against AWS/pmcp.run"
    requirement: "D-18"
    verification: []
    human_judgment: true
    rationale: "Not run by design. Option 3 defers the D-18 live deploy until a declared run passes laya-finetune-gate-v1, and this plan is offline only. The assumptions only a live run can check are listed in D-ITEM-08-10-C"

duration: 22min
completed: 2026-09-27
status: complete
---

# Phase 8 Plan 10: Fail-Closed Decide Deploy Recipes Summary

**The decide server now deploys from the shared `crates` root under the name `aprender-mcp-decide`. cargo-pmcp 0.24.3's own `find_lambda_package_dir` was executed on this workspace and proves that root resolves to `crates/aprender-mcp-decide-lambda`. The per-crate root resolves to `aprender-mcp-chronos-lambda`, which confirms the trap. Nine deploy recipes and a resolver-proof recipe refuse every non-eligible artifact before their first AWS call. The selftest proves all five refusals offline with 0 recorded AWS calls, and proves that setfit-train's `crates/.pmcp/deploy.toml` is restored byte-identically on every exit path.**

## Performance

- **Duration:** 22 min, from 2026-09-27T03:58:03Z to about 04:20Z. This continuation ran after the Task 1 decision.
- **Tasks:** 2 executed (Tasks 2 and 3). Task 1 was the `blocking-human` decision, which the user resolved with no commit.
- **Files:** 1 created, 3 modified.

## Task 1 decision (recorded)

- **Choice:** `shared-crates-root`, made by the user on 2026-09-26.
- **Package order, re-measured by the prior executor:** the first `bootstrap` bin in a `*-lambda` package, in alphabetical package-id order, is `aprender-mcp-chronos-lambda`. The resolver control below re-confirms it.
- **Installed tool:** `cargo-pmcp 0.24.3 (path+file:///Users/guy/Development/mcp/sdk/rust-mcp-sdk/cargo-pmcp)`.
  - It is not on crates.io and there is no 0.24.3 tag.
  - The binary does not record its build commit. `~/.cargo/.crates2.json` holds only the path, and the binary's mtime is 2026-09-18 20:59.
  - The version moved to 0.24.3 in `ee68bd265`.

## Resolver proof (EXECUTED, not inferred)

| Item | Value |
|---|---|
| SDK commit (source copied) | `e0561f8c9086aec56a7f70a3a51537f0671d3ffa`, the SDK HEAD when the run started. `cargo-pmcp/Cargo.toml` says 0.24.3, the same as the installed tool |
| builder.rs last changed | `c04fb4ccb1418e7f6283954d2686bdbf3fec7ce8`, unchanged at e0561f8c9 and at the later HEAD 1162141f4 |
| How it was copied | `git -C <sdk> archive e0561f8c9 \| tar -x` into the session scratchpad. The SDK worktree was never modified, checked out or branched |
| Injected test | `scripts/laya_deploy/cargo_pmcp_resolver_proof.rs`, appended to the scratch copy's `builder.rs` as child module `aprender_resolver_proof` |
| Test name | `deployment::builder::aprender_resolver_proof::aprender_decide_resolves_from_shared_crates_root` |
| Command | `CARGO_NET_OFFLINE=true just laya-resolver-proof ~/Development/mcp/sdk/rust-mcp-sdk e0561f8c9086aec56a7f70a3a51537f0671d3ffa <scratchpad>/resolver`. It runs `cargo test -p cargo-pmcp --bin cargo-pmcp aprender_resolver_proof -- --nocapture` in the scratch copy |
| Result | `test result: ok. 1 passed; 0 failed; ... 935 filtered out` |
| Returned path | `RESOLVED root=crates server=aprender-mcp-decide -> crates/aprender-mcp-decide-lambda` |
| Control (the trap) | `CONTROL root=crates/aprender-mcp-decide-lambda server=aprender-mcp-decide -> crates/aprender-mcp-chronos-lambda` |
| Lock | The SDK gitignores `Cargo.lock`. The scratch copy was seeded with the checkout's lock (a read-only copy) and resolved offline. `cargo_metadata` was 0.23.1 |
| Written | `models/decide/resolver-proof.txt` (gitignored). Line 1 is the path, then test, command, version, SDK commit, builder.rs commit, root, server, control and timestamp |

- **Concurrent SDK activity.** The SDK HEAD moved from e0561f8c9 to `1162141f4` during the run. The user's other session committed `tests/v2_schema_tripwires.rs` at 21:04 PDT, and that commit is the one change in `git status`.
- Every command this plan ran against the SDK was read-only: `rev-parse`, `show`, `log`, `archive`, `status`. builder.rs is unchanged by that commit.
- The 3.5 GB scratch target dir was deleted after the proof.

## Selftest (final run, `just laya-deploy-selftest`)

```
AWS RECORDER: control call recorded and cleared
CASE 1 placeholder: REFUSED as expected (REFUSED placeholder: crates/aprender-mcp-decide-lambda/.pmcp/deploy.toml still holds an UNSET placeholder -- regenerate it with just laya-deploy-config)
CASE 2 sha-pin: REFUSED as expected (REFUSED sha-pin: APRENDER_DECIDE_SHA256 in crates/aprender-mcp-decide-lambda/.pmcp/deploy.toml is 0000000000000000000000000000000000000000000000000000000000000000 but models/decide/selftest/laya_tiny.apr hashes to 37d65159b2be0fa091aa840cd56c1a84b73c0bcd9e2df5906d1f8218f5448561)
CASE 3 resolver-proof: REFUSED as expected (REFUSED resolver-proof: models/decide/resolver-proof.txt does not exist -- run: just laya-resolver-proof <sdk> <commit> <scratch>)
CASE 4 deploy-eligibility: REFUSED as expected (REFUSED eligibility: REFUSED SyntheticNotDeployable recipe variant "synthetic-fixture" is not deployable (nothing written))
CASE 5 upload-eligibility: REFUSED as expected (REFUSED eligibility: REFUSED SyntheticNotDeployable recipe variant "synthetic-fixture" is not deployable (nothing written))
SWAP 1 success: exit 0, crates root restored byte-identical (state sha256 7c6f25eb46e83347d66e32a9e0cfae39c5556c58383dab2c109b3c061773aa03)
SWAP 2 forced-failure: exit 1, crates root restored byte-identical (state sha256 7c6f25eb46e83347d66e32a9e0cfae39c5556c58383dab2c109b3c061773aa03)
SWAP 3 sigterm: exit 143, crates root restored byte-identical (state sha256 7c6f25eb46e83347d66e32a9e0cfae39c5556c58383dab2c109b3c061773aa03)
SWAP 4 absent-root: exit 0, the decide config was installed and nothing is left behind
SKIP positive dry run: no deploy-eligible artifact (laya-finetune-gate-v1 demo.outcome gate_fail; set LAYA_ELIGIBLE_APR/_RUN/_DATA/_BASE to arm)
RESOLVER PROOF: crates/aprender-mcp-decide-lambda (cargo-pmcp 0.24.3, sdk e0561f8c9086)
BOOTSTRAP aarch64 OK /Users/guy/Development/machine-learning/aprender/target/aarch64-unknown-linux-gnu/release/bootstrap (25958736 bytes)
CRATES ROOT: crates/.pmcp/deploy.toml sha256 before=6661ab08e88ee5058ee01622e8bae064923f778e1adacfab841e8d369cee7b99 after=6661ab08e88ee5058ee01622e8bae064923f778e1adacfab841e8d369cee7b99; state sha256 before=7c6f25eb46e83347d66e32a9e0cfae39c5556c58383dab2c109b3c061773aa03 after=7c6f25eb46e83347d66e32a9e0cfae39c5556c58383dab2c109b3c061773aa03
AWS CALLS: 0
DEPLOY MARKERS: 0
DEPLOY SELFTEST OK
```

- **What SWAP 1 checks inside the command:** the decide config is byte-equal at `crates/.pmcp/deploy.toml`, and `crates/deploy` and `crates/.pmcp/deployment.toml` are absent, so the decide config is installed alone.
- **What the "state" digest covers:** `.pmcp/deploy.toml`, `.pmcp/deployment.toml`, `.pmcp/active-target` and every file under `deploy/`, including setfit-train's `stack.ts` (3dc211f0…) and its 16 MB `bootstrap` (8a9b8b4b…).
- **Independent check:** a pre-plan `shasum` manifest of `crates/.pmcp` and `crates/deploy`, taken before any recipe ran, is `cmp`-identical after all runs.
- **`git status --porcelain` for both `.pmcp` dirs:** empty. No decide `deploy.toml` is left behind; none existed before this plan either.

## Task 3 verification (the plan's `<verify>` block, with scratchpad log paths)

- `just --summary` lists 19 `laya-` recipes: the 9 existing ones, 9 from this plan, and `laya-resolver-proof`. All of them parse, and `bash -n` is clean on every body.
- `eligibility precedes every AWS call in laya-deploy and laya-upload`.
- `DRY_RUN=1 just laya-deploy-verify models/decide/selftest/laya_tiny.apr` exits 0.
  - It sizes `concentrated: 8 texts, 512 tokens planned, max text 16384 bytes` and `distributed: 8 texts, 512 tokens planned, max text 16380 bytes` from the artifact under `ClassifyLimits::CONTRACTED`.
  - For each of 4 samples, alternating shapes, it prints the config bump, the `--cold-first --maximal <shape> --probe-id <fresh uuid>` call, and the `decide.load performed_load=true probe_id=<uuid>` CloudWatch correlation.
  - It ends with `DRY-RUN: stopped before the network`.
  - With the aws recorder on PATH, it recorded **0 bytes**. The same recorder recorded a control call.
- Acceptance greps:

  | Pattern | Count | Required |
  |---|---|---|
  | `s3:prefix` | 2 | >= 1 |
  | `decide/` | 34 | >= 3 |
  | `reserved-concurrent-executions 0` | 2 | >= 1 |
  | `cold-first` | 4 | >= 1 |
  | `performed_load=true` | 5 | >= 1 |
  | `touch crates/aprender-mcp-decide-lambda/src/main.rs` | 2 | >= 1 |
- `cargo clippy -p aprender-mcp-decide-lambda --no-deps --examples --lib -- -D warnings` is clean, and `rustfmt --check` passes on probe.rs.

## Accomplishments

- **Pitfall 1 is closed for the decide deploy, with evidence from the resolver itself.**
  - cargo-pmcp's resolver was executed on this workspace. A zigbuild of the package by name is cited only as a build check.
  - `laya-deploy` refuses without a proof made for the installed cargo-pmcp version.
  - After a live deploy, it asserts the compile log. Its `touch` of main.rs forces a `Compiling aprender-mcp-decide-lambda` line, and no other `*-lambda` package may appear.
  - It also asserts that the GET health body names package and server, and runs the identity probe. Any failure runs `laya-teardown` for containment.
- **D-07 at deploy time.**
  - `laya-deploy` checks, in order: config, placeholder, sha-pin plus the content-addressed key, and the resolver proof. It then calls `laya-verify` on the exact file, and only after that does anything touch AWS, starting with `head-object`.
  - `laya-upload` calls `laya-verify` before anything else, `sts` included, and keys the object by verify's `artifact_sha256`.
  - The run, data and base dirs are required arguments.
- **Least privilege and containment.**
  - The grant allows `GetObject` on `decide/<server>/*` only, plus `ListBucket` conditioned on that `s3:prefix`.
  - `laya-teardown` sets reserved concurrency to 0 and verifies it before removing the grant. It prints the destroy and `s3 rm` commands rather than running them.
- **Cold verify measures the accepted region as defined.** It bumps the config and waits before every sample, sends the maximal request as the first POST, and covers both shapes. CloudWatch cold evidence is required for each probe id, and every sample must be under 30000 ms.

## Task Commits

1. **Task 2 (tracer): the deploy path end to end in dry run on the synthetic artifact** — `dccc37e76` (feat)
2. **Task 3: live-side recipes: bucket, content-addressed upload, scoped grant, cold verify, teardown** — `81137329c` (feat)

Tracer feedback gate: interactive run, `human_verify_mode` end-of-phase, automated-only `<verify>`. I re-ran the verify and it passed: `DEPLOY SELFTEST OK`, all cases, `AWS CALLS: 0`, `DEPLOY MARKERS: 0`, and empty `.pmcp` status. Expansion then continued with no checkpoint.

## Files Created/Modified

- `justfile`: the Laya deploy section.
  - Task 2: `laya-resolver-proof`, `laya-deploy-config`, the private `_laya-crates-root-swap`, `laya-build-bootstrap`, `laya-deploy` and `laya-deploy-selftest`.
  - Task 3: `laya-weights-bucket`, `laya-upload`, `laya-grant`, `laya-deploy-verify` and `laya-teardown`.
- `scripts/laya_deploy/cargo_pmcp_resolver_proof.rs`: the test module appended to a SCRATCH copy of cargo-pmcp's builder.rs. It is never compiled in this workspace.
- `crates/aprender-mcp-decide-lambda/examples/probe.rs`: adds `--plan-only`, which builds the `--maximal` request from `--apr` and prints its size with no network and no `--url`.
- `.planning/phases/08-.../deferred-items.md`: D-ITEM-08-07-B is now resolved, and D-ITEM-08-10-A through D are added.

## Decisions Made

- **The Task 1 choice:** shared-crates-root, made by the user.
- **Two version and argument choices:**
  - `laya-deploy` pins the resolver proof to the installed cargo-pmcp version, the shared-root counterpart of the patch option's commit pin.
  - `auth` is a required argument of `laya-deploy-config`.
- **Other choices, detailed in the frontmatter:**
  - The swap isolates `crates/deploy/`, and the deploy passes `--regenerate-stack`.
  - The swap backup lives under gitignored `models/decide/swap-backup/<root>` rather than in a temp dir, so a SIGKILL mid-deploy leaves something recoverable, and the next swap refuses until a human restores it.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] The swap also isolates `crates/deploy/` and passes `--regenerate-stack`**
- **Found during:** Task 2, reading cargo-pmcp's deploy path before writing the swap.
- **Issue:** The user decision named only `crates/.pmcp/deploy.toml`. But the shared root also holds `crates/deploy/lib/stack.ts`, which setfit-train rendered (it defaults `serverId` to `aprender-setfit-train` and sets `memorySize: 256`), and a 16 MB bootstrap.
  - Per the cargo-pmcp source at e0561f8c9, `validate_and_regenerate_stack_ts` PRESERVES an existing stack.ts (DSTK-01) unless `--regenerate-stack` is passed.
  - A preserved, non-matching stack.ts routes the pmcp-run synth to `cdk synth` with the "declared [iam]/[environment] are not auto-applied" warning.
  - So a decide deploy would have used setfit-train's stack without the decide S3 URI and pin, and cargo-pmcp would have overwritten setfit-train's `deployment.toml`.
- **Fix:** the swap backs up and clears `.pmcp/{deploy,deployment}.toml`, `.pmcp/active-target` and all of `deploy/`, and restores them. `laya-deploy` passes `--regenerate-stack`. The installed binary's `--help` lists that flag.
- **Verification:** SWAP 1-4 plus the CRATES ROOT digest over all of that state, and a `cmp` against the pre-plan manifest.
- **Caveat:** this is a source reading of cargo-pmcp, not exercised against pmcp.run.
- **Committed in:** `dccc37e76`.

**2. [Rule 3 - Blocking] `--locked` cannot work for the SDK copy**
- **Found during:** the first resolver-proof run.
- **Issue:** the SDK gitignores `Cargo.lock`, so the archive has none, and `cargo test --locked` refused to create one.
- **Fix:** seed the scratch copy with the checkout's `Cargo.lock` (a read-only copy), resolve offline, and record the `cargo_metadata` version in the proof.
- **Committed in:** `dccc37e76`.

**3. [Rule 1 - Bug] Selftest mutations produced invalid TOML**
- **Found during:** the first full selftest. CASE 2 failed, refused as `config` instead of `sha-pin`.
- **Issue:** my Python `re.sub` replacement left a literal backslash-quote. CASE 1 had the same defect but passed, because the placeholder check runs before the parse.
- **Fix:** a `mutate` helper with a lambda replacement that writes valid TOML, so each case fails for the check under test and never for a parse error.
- **Committed in:** `dccc37e76`.

**4. [Rule 2 - Missing Critical] The aws recorder needed a positive control**
- **Found during:** Task 3 measurement.
- **Issue:** a hook-filtered `wc` in my own shell printed 0 for a log the recorder had written. A zero proves nothing unless the recorder demonstrably records.
- **Fix:** the selftest makes one control call and requires exit 97 and exactly one recorded line before it clears the log.
- **Committed in:** `81137329c`.

**5. [Scope] Additions beyond the plan's file list**
- **What was added:**
  - `laya-resolver-proof` (a 10th recipe) and `scripts/laya_deploy/cargo_pmcp_resolver_proof.rs`. The plan asks for the proof to be recorded but names no mechanism.
  - The private `_laya-crates-root-swap`.
  - `probe --plan-only`. The plan's dry-run verify sizes both shapes "from the local .apr with ClassifyLimits::CONTRACTED", and the probe had no offline mode.
- **Signature changes:**
  - `laya-deploy-config` takes `apr auth [server env profile]`, not `apr server auth env profile`.
  - `laya-deploy`, `laya-upload` and `laya-deploy-verify` gained trailing `env` and/or `profile` defaults.

**Total deviations:** 4 auto-fixed (2 missing critical, 1 blocking, 1 bug), plus 1 scope note.
**Impact on plan:** the first fix changes what the shared-root decision actually has to protect. It stays inside the chosen option and does not re-open it. No tolerance, threshold, contract or demo value moved.

## Issues Encountered

- **The broken-windows ledger** (`gsd-tools windows append`) still refuses every append (`Ledger entry 24 has invalid status: "resolved"`). The unrun-verify item is recorded as D-ITEM-08-10-C instead.
- **One stray log path:** the first Task 2 selftest log went to `/tmp/p08-10-t2.log`, the path the plan's verify block names. Later runs used the session scratchpad.

## Known Stubs

None. `laya-deploy-verify`'s DRY_RUN endpoint placeholder (`<endpoint from ...>`) is printed plan text, not data flowing anywhere.

## Threat Flags

None beyond the plan's threat model. Every new AWS-facing surface (bucket, upload, grant, cold-verify config bumps, teardown) is covered by T-08-10-01 through T-08-10-10. The config bump in `laya-deploy-verify` is out-of-band drift that the recipe comment names, and the next `laya-deploy` resets it.

## User Setup Required

None for this plan. Nothing touched AWS.

## Next Phase Readiness

- **Plan 08-11** records the deploy go/no-go as HOLD. These recipes are what the first gate-passing run after the calibration spike will use. The order is `laya-weights-bucket` -> `laya-upload` -> `laya-deploy-config <apr> <on|off>` -> `laya-deploy` -> `laya-deploy-verify`.
- **Before that run:**
  - Re-run `just laya-resolver-proof` if cargo-pmcp is reinstalled. `laya-deploy` refuses a proof made for another version.
  - Settle the auth provider (D-ITEM-08-10-D).
- **Recorded limits:**
  - One decide model per workspace (D-ITEM-08-10-A).
  - The upstream cargo-pmcp fix is recommended future SDK work, not done here (D-ITEM-08-10-B).
  - The live halves are unrun (D-ITEM-08-10-C).

---
*Phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server*
*Completed: 2026-09-27*

## Self-Check: PASSED

- Files exist: scripts/laya_deploy/cargo_pmcp_resolver_proof.rs, justfile, crates/aprender-mcp-decide-lambda/examples/probe.rs, models/decide/resolver-proof.txt (gitignored).
- Commits exist: dccc37e76, 81137329c.
- Re-run: both task verify blocks passed on HEAD 81137329c (DEPLOY SELFTEST OK, T3 VERIFY PASS).
