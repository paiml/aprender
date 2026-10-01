---
phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server
plan: 17
subsystem: infra
tags: [laya, aprender-mcp-decide-lambda, pmcp-run, lambda, deploy, tier-policy, decide-tool-boundary, iam, cold-start, 3008mb]
outcome: deployed-passed

requires:
  - phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server
    provides: "08-16: the one deploy-eligible artifact models/decide/laya-stance-64.apr (gate_pass, sha256 24a44d7e…)"
  - phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server
    provides: "08-10: the fail-closed deploy recipes (bucket, upload, config, laya-deploy, grant, teardown, verify)"
provides:
  - "decide-tool-boundary-v1 2.0.0: the classify budget re-priced for the 3,008 MB Lambda tier (lambda_memory_mb 3008, 120 built tokens, 2 texts), with 10,240 MB recorded as the target tier"
  - "a tier mirror: the deploy template's memory_mb is asserted equal to the contract's lambda_memory_mb"
  - "laya-deploy fixed for two pre-grant defects (--no-post-deploy-test, IAM propagation wait) and passing --no-oauth for auth off"
  - "08-LIVE-DEPLOY-EVIDENCE.json: outcome deployed-passed. Identity == H through the pmcp.run edge. 4 cold samples (2 per shape), all < 30000 ms (worst 29350), with Max Memory Used and Init Duration per sample. Warm p50 1418 ms. The admin-UI request shape. The two refused attempts kept as history"
  - "the decide weights read declared IN THE STACK ([[iam.statements]]: s3:GetObject on decide/aprender-mcp-decide/* only). laya-grant is now a read-only check. Teardown never deletes the stack policy"
  - "laya-deploy retries exactly the pmcp.run edge's transient 503 error-state refusal (bounded); any other failure contains"
affects: [08-17 continuation, 08-18, 08-12, decide deploy recipes, pmcp.run edge behaviour]

actuals:
  tokens: 12185    # chars/4 over the realized diff a90a5cd41..working tree (48739 chars), with this SUMMARY and state.json excluded
  tasks: 3         # Task 1 recorded (the user's answer), Task 2 deployed with identity ok (on the option-1 resume), Task 3 cold samples taken
  commits: 10      # MEASURED: git rev-list --count a90a5cd41..HEAD before this SUMMARY's docs commit
plan_head_before: a90a5cd41e91a3d1b153da91d104e5fba6e3822f

tech-stack:
  added: []
  patterns:
    - "Tier policy in the contract: every envelope constant is priced for ONE lambda_memory_mb, and the deploy template's memory is a tested mirror of it"
    - "Extrapolated budgets show every term and its anchor, so the live cold log (download_ms, sha_ms, build_ms) can falsify each term separately"
    - "A maximal request that the budget would refuse is never built (MaximalError::OverBudget)"

key-files:
  created:
    - .planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-LIVE-DEPLOY-EVIDENCE.json
  modified:
    - contracts/decide-tool-boundary-v1.yaml
    - contracts/aprender/binding.yaml
    - crates/aprender-mcp-decide/src/lib.rs
    - crates/aprender-mcp-decide/src/tests.rs
    - crates/aprender-mcp-decide/tests/e2e_stdio.rs
    - crates/aprender-mcp-decide/README.md
    - crates/aprender-mcp-decide-lambda/src/probe.rs
    - crates/aprender-mcp-decide-lambda/src/tests.rs
    - crates/aprender-mcp-decide-lambda/.pmcp/deploy.toml.template
    - justfile
    - .planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/deferred-items.md
    - .planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-LIVE-DEPLOY-EVIDENCE.json

key-decisions:
  - "USER DECISION 2026-09-27: deploy-auth-off-accept-risk. Deploy open like the chronos precedent, and the user explicitly accepts the cost-amplification risk"
  - "USER DECISION 2026-09-27: memory 3,008 MB, not 10,240 MB. The account is capped at 3 GB until AWS Support approves more; profile ze-kasher-dev"
  - "The 3,008 MB budget is an extrapolation from spike 026's Graviton2 anchors (10,240 and 4,096 MB), scaled by vCPU. Per-token uses the larger anchor (40 ms). The fixed cold cost uses the nearer anchor (4,096 MB), because it refutes the 10 GB anchor for the load term. A software sha256 pass is priced explicitly (4521 ms). The result: 120 tokens and 2 texts"
  - "The GET health-body check in laya-deploy is unrealizable through the pmcp.run edge. A 405 from the edge is a recipe-assumption failure, not a model identity failure. Per the user's failure rule, the contained deploy is recorded as deploy-refused and the plan stops for the human's resume decision"
  - "Resume attempt 2: grant-after-deploy cannot work on pmcp.run. The platform's own post-deploy call fails the model load before the grant exists, and the edge then refuses every MCP POST with 503 -32004 'Server is in error state' (D-ITEM-08-17-B, now proven). The outcome stays deploy-refused (contained), and the fix is a human choice"

  - "USER DECISION 2026-09-27 (option 1): declare the S3 read in the deploy config's [iam] so it is part of the stack. Verified from the cargo-pmcp 0.24.3 source first: the policy is the role's default policy, and the function DependsOn it"
  - "Containment stays reserved concurrency 0. The stack-declared read is stack-managed, so teardown never deletes it: it is inert while contained, and `cargo pmcp deploy destroy` removes it"
  - "Outcome labelled by the plan's rule alone: deployed-passed, because every cold sample was < 30000 ms. The thin margin (650 ms) and the load running over its 17000 ms extrapolation are recorded as D-ITEM-08-17-E, not folded into the label"

patterns-established:
  - "A permission a function needs at its FIRST invocation is declared in the stack it is created with, never granted after the deploy"
  - "Budget tiers: re-deriving a door bound for a new memory size changes the contract first, then the Rust mirror, then the deploy memory, in one commit, before any AWS call"

requirements-completed: []

coverage:
  - id: D1
    description: "decide-tool-boundary-v1 2.0.0 re-priced for the 3,008 MB tier (120 tokens, 2 texts, 17000/3900/40 envelope, margin and cap unchanged), with Rust mirrors, the tier mirror test and the OverBudget builder guard. Committed before any AWS call"
    verification:
      - kind: other
        ref: "pv validate contracts/decide-tool-boundary-v1.yaml -> 0 error(s); pv diff suggested major -> 2.0.0"
        status: pass
      - kind: unit
        ref: "cargo test -p aprender-decide -p aprender-mcp-decide -p aprender-mcp-decide-lambda --lib -> 167 passed"
        status: pass
      - kind: unit
        ref: "crates/aprender-mcp-decide-lambda/src/tests.rs#deploy_memory_is_the_contract_tier (red with the template at 10240, green at 3008)"
        status: pass
      - kind: integration
        ref: "cargo test -p aprender-mcp-decide --test e2e_stdio (tiny leg, and the real laya-stance-64 leg: 71 and 66 built tokens, one text per call)"
        status: pass
      - kind: other
        ref: "make contract-audit-phase8 rc 0; probe --plan-only on laya-stance-64.apr: concentrated 1 text / 120 tokens, distributed 2 texts / 120 tokens"
        status: pass
    human_judgment: false
  - id: D2
    description: "laya-deploy fixed for the pre-grant post-deploy suite and IAM propagation, and passes --no-oauth when auth is off. The offline selftest still passes with zero AWS calls"
    verification:
      - kind: integration
        ref: "just laya-deploy-selftest -> DEPLOY SELFTEST OK, AWS CALLS: 0, DEPLOY MARKERS: 0"
        status: pass
      - kind: integration
        ref: "option 1 (52a12d777, 743eb02ed): selftest IAM config table (accept as-generated + literal; refuse s3-star, list-too, bucket-wide, star-resource, other-server, two-statements, bucket-sugar, no-iam), laya-deploy refuses a broadened config, GRANT table, IAM and EDGE SETTLE wiring (mutation-tested). DEPLOY SELFTEST OK, AWS CALLS: 0"
        status: pass
    human_judgment: false
  - id: D3
    description: "Read-only readiness before any write: A10 confirmed, A2 = 3008 (the largest existing MemorySize; no memory field in the account settings; the created function reads back 3008), and no decide or laya function existed"
    verification:
      - kind: other
        ref: "aws sts get-caller-identity / lambda list-functions / lambda get-account-settings (profile ze-kasher-dev); get-function-configuration MemorySize 3008"
        status: pass
    human_judgment: false
  - id: D4
    description: "Live deploy of the gated model on pmcp.run with identity == H proven by a warm classify (D-18, D-11)"
    requirement: "D-18"
    verification:
      - kind: other
        ref: "just laya-deploy -> rc 1: GET health 405 at the pmcp.run edge -> contained (reserved concurrency 0 verified)"
        status: fail
      - kind: other
        ref: "resume attempt 2, just laya-deploy -> rc 1: edge /health serverId ok, then the identity probe's initialize got 503 -32004 'Server is in error state' from the edge -> contained (reserved concurrency 0 verified 22:18:08Z)"
        status: fail
      - kind: other
        ref: "option 1: just laya-deploy (22:48:35-22:55:28Z). Stack grant check ok (pmcp-declared), edge health ok. pmcp.run's post-deploy call loaded the model (load_ms 24554, status success). The probe hit the edge's transient 503 and contained. After the lift, the identity probe passed at 22:56:52Z: artifact_sha256 == H, one tool `classify`, labels none/against/favor in order, 65 tokens"
        status: pass
    human_judgment: false
    rationale: "Passed on the option-1 resume. The history: refused and contained twice. First by the unrealizable GET health check (D-ITEM-08-17-A, fixed in 3115c690e). Then by pmcp.run's sticky error state after its own pre-grant post-deploy call (D-ITEM-08-17-B). Resuming needs a human choice of how the S3 read exists before that call, plus approval to lift the containment"
  - id: D5
    description: "Cold accepted region of both maximal shapes at 3,008 MB, with CloudWatch Max Memory Used and init duration per sample (D-10)"
    requirement: "D-10"
    verification:
      - kind: other
        ref: "just laya-deploy-verify models/decide/laya-stance-64.apr aprender-mcp-decide ze-kasher-dev 2 -> DEPLOY VERIFY OK: 4 cold samples, each with CloudWatch performed_load=true for its probe id. Elapsed 28527/29350/28501/24168 ms (C/D/C/D), load_ms 24550/25460/25813/21411, Max Memory Used 2481-2483 MB, Init 56-66 ms, graviton2 x2 and graviton3 x2. Warm p50 1418 ms, max 1491"
        status: pass
    human_judgment: false
    rationale: "deployed-passed on the plan's rule. The margin is thin (650 ms at worst), and the load runs 4.4-8.8 s over its 17000 ms extrapolation (D-ITEM-08-17-E)"

duration: 51min + 24min (option-1 resume, 22:43-23:07Z)
completed: 2026-09-27
status: complete
---

# Phase 8 Plan 17: Go/No-Go and Live Decide Deploy Summary

**Deployed and passed on the option-1 resume. The decide weights read is now declared in the stack (`[[iam.statements]]`: s3:GetObject on `decide/aprender-mcp-decide/*` only). So pmcp.run's own post-deploy call loaded the model (24.6 s, success) instead of failing it. Through the edge, identity == `24a44d7e…` with labels none/against/favor. All 4 cold samples of both maximal shapes came in under the 30 s cap: 24.2-29.4 s, with Max Memory Used 2481-2483 of 3008 MB and Init 56-66 ms. The margin is thin, and the S3 download, not compute, is what uses it up. The function is left RUNNING for the admin UI.**

**Before that, the plan halted twice.** First on an unrealizable GET health check (fixed in `3115c690e`). Then on pmcp.run's post-deploy call, which ran before the out-of-band grant and left the edge in a sticky error state. Both are kept below as history.

## Performance

- **Duration:** 51 min, from 2026-09-27T20:25:16Z to 21:16:48Z.
- **Tasks:** Task 1 was recorded, and Task 2 executed up to a contained refusal. Task 3 was skipped by the plan's own rule, because a refused deploy takes no cold samples.
- **Files:** 1 created, 11 modified.

## Task 1: the go/no-go answer (recorded)

| Item | Answer |
|---|---|
| Option | `deploy-auth-off-accept-risk`, the user's decision on 2026-09-27 |
| Auth | Off: `[auth] enabled = false` and `--no-oauth`. pmcp.run reported `oauthEnabled=false`. No OAuth provider |
| Risk | **The user explicitly accepts the cost-amplification risk of an open function** (RESEARCH Security Domain V2). This matches the live chronos precedent |
| Memory | **3,008 MB**, not 10,240 MB. The account is capped at 3 GB until AWS Support approves more |
| Profile | `ze-kasher-dev` (A10). The user did not confirm it explicitly, so it was verified read-only below |
| Start sha | `a90a5cd41` (`/tmp/p08-17-start.sha`). Deploy baseline after the amendment: `968d73e99` (`/tmp/p08-17-deploy-base.sha`) |

## The 3,008 MB tier amendment (commit `968d73e99`, before any AWS call)

The contract's own rule applied: lower `classify_max_total_tokens` after re-deriving it here, and never raise the margin. The arithmetic below is in the contract description. It is an extrapolation, and the live cold samples are what falsify it.

vCPU scales with memory at 1 vCPU per 1,769 MB: 6 vCPU at 10,240 MB, 2.315 at 4,096 MB and 1.700 at 3,008 MB. The anchors are spike 026's Graviton2 raw records.

| Term | Derivation | Value |
|---|---|---|
| Per built token at 512 | 10,240 MB: 5660/512 x 6/1.700 = 39.0. 4,096 MB: 12501/512 x 2.315/1.700 = 33.3. The larger, rounded up | `tier_ms_per_token_at_512` **40** (renamed from `g2_ms_per_token_at_512`) |
| Gateway + client overhead | max(client wall - duration), not memory-scaled | 895 |
| Download | 9035-9222 ms, flat between 10 GB and 4 GB | 9222 |
| sha256 pin | sha2 0.10.9 runs its SOFTWARE backend (no `asm`). 1370 ms on M4, x3.3 (spike's G2/M4 ratio) | 4521 |
| Parse + widen | 4 GB: 1675 x 2.315/1.700. The 10 GB anchor over-predicts the 4 GB measurement 2x, so it is not used here | 2281 |
| Cold fixed | 16919, declared UP | `cold_start_budget_ms` **17000** |
| Probe replay | 2 x 48 x 40 = 3840, declared UP | `probe_budget_ms` **3900** |
| Budget | (30000 - 17000 - 3900 - 4000) / 40 = 127, declared DOWN | `classify_max_total_tokens` **120** |
| Count | floor(120 / 57). 57 is the shortest built row of the stance task, measured with `probe --plan-only` | `classify_max_texts` **2** |

- **Unchanged:** `margin_ms` 4000 and `api_gateway_timeout_ms` 30000.
- **Proof obligation:** 120 x 40 + 17000 + 3900 + 4000 = 29700, which is at most 30000.
- **Target tier, recorded:** 10,240 MB with budget 1024 and 8 texts. Restoring it is a re-derivation plus a memory change.
- **Consequences at 3,008 MB:**
  - A legal request is one text of about 63 state tokens (a tweet), or two very short texts.
  - `truncated: true` is unreachable, because a 512-token row is always over the budget.
  - Two real sentences no longer fit in one call: they build to 71 + 66 = 137 tokens.
- **New constants:** `lambda_memory_mb` 3008 and `served_task_min_row_tokens` 57.
- **Mirrors updated:**
  - `ClassifyLimits::CONTRACTED` is now 2 texts and 120 tokens.
  - The deploy template's `memory_mb` is 3008.
  - New tests: `deploy_memory_is_the_contract_tier` (it went red when the template alone was set to 10240) and `max_texts_fit_the_budget_at_the_shortest_row`.
  - The maximal builder now refuses an over-budget request (`MaximalError::OverBudget`, with a new test).
  - The count tests were renamed so they hold at any tier: `one_over_max_texts_refused_naming_max_texts` and `max_texts_accepted_and_classified`. The contract `test:` lines were updated with them.
  - The description test now reads the bounds from the contract.
  - The e2e test sends contract-sized batches.
  - The binding note and the README were updated.
- **Checks, all passing:**

  | Check | Result |
  |---|---|
  | `pv validate` | 0 errors |
  | `pv diff` | suggested major, so 2.0.0 was applied |
  | `make contract-audit-phase8` | rc 0 |
  | Lib tests (decide crates) | 26 in aprender-mcp-decide and 29 in aprender-mcp-decide-lambda. With aprender-decide, 167 across the three crates |
  | `cargo clippy --no-deps --lib --tests --examples -D warnings` | clean on both crates |
  | rustfmt | clean |
  | e2e | tiny leg and real-model leg pass |
  | `probe --plan-only` on the real artifact | concentrated: 1 text, 120 tokens. Distributed: 2 texts, 120 tokens |
  | `just laya-deploy-selftest` | DEPLOY SELFTEST OK, AWS CALLS: 0 |

## Readiness (read-only, before the first write)

- **A10: confirmed.**
  - `ze-kasher-dev` resolves to an IAM user. The account id was seen in the session only and is not recorded anywhere.
  - 228 functions are visible in it, including `chronos-forecaster` and `pmcp-*`.
  - cargo-pmcp resolves the shared `crates` root to target `dev`, which is `ze-kasher-dev` in us-east-1.
- **A2: 3,008 MB.**
  - The largest existing MemorySize is 3008, held by 2 functions.
  - `get-account-settings` has no memory field. It reports only code sizes and ConcurrentExecutions 1000.
  - The proof is the deploy itself: `aprender-mcp-decide` exists at MemorySize 3008 and reads `Active` / `Successful`.
- **No decide or laya function** existed before the deploy.
- **pmcp.run auth:** the cached token had expired, and cargo-pmcp refreshed it on a read-only `outputs` call.

## Task 2: the live sequence

| Step | rc | Result |
|---|---|---|
| Precondition gates on `968d73e99` | 0 | Audit rc 0. Strict-binding guard: the lifted copy resolved 684 refs, and only the 2 pre-existing contracts dangle. Three-crate lib tests pass. Selftest OK. `laya-verify` reports deploy_eligible true with sha H |
| `just laya-weights-bucket dev ze-kasher-dev` | 0 | Bucket created: private, SSE-S3, tagged |
| `just laya-upload ...` | 0 | `laya-verify` ran first. 846,196,868 bytes went to `decide/aprender-mcp-decide/<H>.apr` |
| `just laya-deploy-config <apr> off ...` | 0 | memory 3008, timeout 30, auth false |
| `just laya-deploy ...` (on `b30f437da`) | **1** | Eligibility and the resolver proof passed. `cargo pmcp deploy --regenerate-stack --no-post-deploy-test --no-oauth` returned 0. The compile log names only `aprender-mcp-decide-lambda`. The grant was applied. **GET health returned 405, and the function was contained** |

- **Refusal:** `IDENTITY FAILURE: GET https://aprender-mcp-decide.us-east.true-mcp.com/mcp failed -- containing (reserved concurrency 0, grant removed)` (`curl: (56) ... 405`).
- **Diagnosis, read-only.** The GET never reached the function, for four reasons:
  - The bootstrap answers every GET with 200 and its package body, so a 405 cannot come from it.
  - The pmcp.run edge answers GET `/mcp` itself: `"SSE streams are not offered at this endpoint. Use POST /mcp."`. The live chronos endpoint gives the same 405.
  - The edge's `/health` returns platform JSON (`status`, `serverId`, `hasDeployment`) with no package field.
  - CloudWatch has no invocation for the GET.
- **Containment is verified:** `get-function-concurrency` reads 0, and `list-role-policies` no longer lists the decide weights policy.
- **Left in place for the human:** the deployment, the function and the S3 object.
- **The shared root was restored byte-identical:** state sha256 `7c6f25eb…`, and the porcelain for `crates/.pmcp` and `crates/deploy` is empty.
- **The only invocation** came at 21:12:10Z. pmcp.run made it itself before the grant. The load failed at the S3 length lookup and re-armed. REPORT: 159 ms, Max Memory Used 37 MB, init 67 ms, memorySize 3008. No model load has happened, so there is **no memory, probe-replay or cold-time evidence yet**.
- **Live assumptions (D-ITEM-08-10-C):**
  - Confirmed: function name == server id; the deployment.toml endpoint; compile lines in the log; the log group is `/aws/lambda/aprender-mcp-decide`.
  - **Refuted:** that a GET reaches the bootstrap's health branch.

## Task 3 (option-1 resume): cold accepted region at 3,008 MB

Command: `just laya-deploy-verify models/decide/laya-stance-64.apr aprender-mcp-decide ze-kasher-dev 2`, 22:57:16-23:01:34Z, rc 0, `DEPLOY VERIFY OK`.
- Each sample got a `DECIDE_COLD_BUMP` config change first, then the maximal `tools/call` as the first POST.
- Each was proven cold by its own `decide.load performed_load=true probe_id=<uuid>` line.
- REPORT metrics came from `cw_report.py`.

| # | Shape | Tokens | Elapsed ms | load_ms (download / sha / build) | Max Mem MB | Init ms | Part | ms/token after load |
|---|---|---|---|---|---|---|---|---|
| 1 | concentrated | 120 | 28527 | 24550 (13519 / 3499 / 7444) | 2481 | 66.2 | graviton2 | 26.8 |
| 2 | distributed | 120 | 29350 | 25460 (14324 / 3511 / 7538) | 2481 | 65.2 | graviton2 | 25.9 |
| 3 | concentrated | 120 | 28501 | 25813 (17831 / 2685 / 5225) | 2483 | 56.3 | graviton3 | 15.9 |
| 4 | distributed | 120 | 24168 | 21411 (13494 / 2655 / 5193) | 2481 | 65.6 | graviton3 | 16.7 |

- **Warm:** after the samples, identity passed, and 5 warm calls ran at p50 1418 ms and max 1491 ms.
- **Outcome: `deployed-passed`.** Every judged sample is < 30000 ms, so there is no `exceeded_resolution`. The plan's Task 3 verify printed `deployed-passed samples 4 max ms 29350`.

**Against the contract's 3 GB extrapolations:**

| Term | Extrapolated | Measured | Verdict |
|---|---|---|---|
| Cold (load) | 17000 ms | 21411-25813 ms | **Over by 4.4-8.8 s.** download_ms of 13.5-17.8 s dominates, and parts time out at 8 s and are retried |
| Per token | 40 ms | about 26 (graviton2), about 16-17 (graviton3) | Under |
| Headroom | ~350 MB | 525-527 MB (3008 - 2481/2483) | Better than projected |

- **The cap still held, but only just:** 650 ms at worst. The download is the term to attack (D-ITEM-08-17-E).
- **Init Duration is 56-66 ms,** because the load runs in the first invocation, not in init.
- **Config bumps are safe now.** Every bump re-initialised the environment, and each new environment loaded from S3 under the stack policy and answered through the edge. The stack permission belongs to the role, so a re-init never runs without it.

## Admin-UI call (for the pmcp.run admin UI WASM client)

One warm call through the edge, at 23:02:25Z: HTTP 200 in 1.74 s, `x-decide-load: warm`.

```
POST https://aprender-mcp-decide.us-east.true-mcp.com/mcp
content-type: application/json
accept: application/json, text/event-stream

{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"classify","arguments":{"texts":["We need to stop pretending climate change is a hoax. Act now."]}}}
```

- **Result text:** `{"model":{"artifact_sha256":"24a44d7e…","recipe_id":"6a5489af…","method":"laya","base":"laya-en-root@55cf4c4e"},"labels":["none","against","favor"],"results":[{"label":"none","probabilities":[0.8217,0.1029,0.0753],"tokens":72,"truncated":false}]}`
- **Why `none` is right here:** the tool asks the abortion-stance question, and this tweet is off-topic. Use an abortion-stance tweet to see against or favor.
- **No `initialize` is required.** A client that does initialize, notifications/initialized, tools/list and then tools/call also works; the identity probe does exactly that.
- **Keep each call to 2 texts and 120 built tokens or fewer.** One short tweet builds about 65-72 tokens.

## Resume option 1 (2026-09-27, 22:43-23:07Z): the S3 read in the stack

**Step 1: verified from source, read-only.** Source: cargo-pmcp 0.24.3 at the resolver-proof sdk commit `e0561f8c9086`, plus the working tree. The SDK repo was not modified.
- `deploy_to_pmcp_run` calls `validate_and_regenerate_stack_ts` before any network call. That function runs a fail-closed IAM validator and rewrites stack.ts from `[iam]` under `--regenerate-stack`.
- The pmcp-cfn-renderer appends `[[iam.statements]]` to the role's default `AWS::IAM::Policy` (`pmcp-declared`).
- The Lambda function `DependsOn` that policy and the role, so the grant exists before the first invocation.
- The installed binary carries the same strings.

**Step 2: recipe changes, in `52a12d777` (committed before any AWS write).**
- The template carries one `[[iam.statements]]`, and `laya-deploy-config` fills in the ARN.
- `_laya-iam-check` accepts exactly one Allow `s3:GetObject` on `arn:aws:s3:::<weights-bucket>/decide/<server>/*` and nothing else. `laya-deploy-config` and `laya-deploy` both run it.
- `laya-grant` is now read-only (`_laya-grant-check`). It fails if the scoped read is missing, if anything broader is present, or if the legacy out-of-band policy is still attached.
- `laya-teardown` deletes only the legacy policy, by its exact name, and never the stack-managed one.
- The propagation sleep is gone.

**Step 4: redeploy.**
- The containment was lifted at 22:48Z, then `just laya-deploy` ran from 22:48:35 to 22:55:28Z. The template was rendered by pmcp-cfn-renderer.
- The grant check passed: `pmcp-declared` carries the scoped read, and no legacy policy is on the role.
- Edge `/health` named serverId `aprender-mcp-decide`.
- CloudWatch shows that pmcp.run's own post-deploy call LOADED the model: `performed_load=true load_ms=24554`, REPORT 24569 ms, Max Memory Used 2482 MB, Init 66.6 ms, `success`. 5 follow-up calls also succeeded.
- **The identity probe then got the edge's 503 `Server is in error state` at about 22:55:25Z, and the recipe contained.**
- A read-only investigation followed, with no redeploy:
  - At 22:56:09Z, one POST `initialize` while contained returned a throttled 500. So the edge was forwarding again.
  - The containment was lifted at 22:56:22Z.
  - The identity probe passed at 22:56:52Z.
- So after a successful load the error state is **transient** (under a minute). After the failed load in attempt 2 it was **sticky**.
- `743eb02ed` makes `laya-deploy` retry exactly that refusal (8 x 15 s at most). Any other failure still contains at once. The selftest guards this, and the guard was mutation-tested.

**Final state:** RUNNING, with no reserved concurrency. The `DECIDE_COLD_BUMP` env drift from the samples is reset by the next `laya-deploy`.

**Teardown (not run; the human's call):**
- Contain: `just laya-teardown aprender-mcp-decide dev ze-kasher-dev`. This sets reserved concurrency 0 and leaves the stack read in place, where it is inert.
- Destroy: `just _laya-crates-root-swap crates crates/aprender-mcp-decide-lambda/.pmcp/deploy.toml models/decide/destroy-aprender-mcp-decide.state cargo pmcp deploy destroy --manifest-path crates`
- Remove the weights: `aws s3 rm --profile ze-kasher-dev --recursive s3://<weights-bucket>/decide/aprender-mcp-decide/`

## Task Commits

1. **The 3,008 MB tier amendment** — `968d73e99` (feat)
2. **laya-deploy pre-grant fixes** — `b30f437da` (fix)
3. **Task 2: the deploy-refused evidence record and deferred items** — `371c6bfe7` (feat)
4. **Resume: laya-deploy checks the edge /health serverId** — `3115c690e` (fix)
5. **Resume attempt 1 record (auth gate)** — `f8073e5f5` (docs)
6. **Resume attempt 2 record (edge error state, re-contained)** — `ad57c6ec3` (docs)
7. **Option 1: the S3 read declared in the stack, laya-grant read-only** — `52a12d777` (fix)
8. **laya-deploy retries only the edge's transient 503 error-state refusal** — `743eb02ed` (fix)
9. **Evidence (deployed-passed), deferred items and this SUMMARY** — the docs commit that carries this section

## Deviations from Plan

**1. [User scope change] Memory 3,008 MB and the contract amendment**
- **Source:** the user's go/no-go answer.
- **What changed:** the plan's 10,240 MB became 3,008 MB. Contracts and crate sources changed inside this plan, in commit `968d73e99`.
- **Verify adaptations:**
  - The plan's Task 2 verify asserts `memory_mb == 10240` and no contract or source diff since the start sha. It was run with `3008` and with the post-amendment baseline `968d73e99`.
  - Both passed: no contract or crate source changed after the amendment.

**2. [Rule 3 - Blocking] laya-deploy ran cargo-pmcp's post-deploy suite before its own grant**
- **Found:** before any write, by reading the recipe against `cargo pmcp deploy --help`.
- **Why it blocks:** every MCP request loads the model from S3, and the grant follows the deploy. So the suite could only see a 403 and exit 3 before the grant and before the identity chain. CloudWatch later confirmed that pmcp.run itself invokes the function before the grant (D-ITEM-08-17-B).
- **Fix:** `--no-post-deploy-test`, plus a `LAYA_GRANT_PROPAGATION_S` (20 s) wait after `put-role-policy`, so a not-yet-visible policy is not contained as an identity failure. `--no-oauth` is passed when auth is off, per the user's instruction.
- **Commit:** `b30f437da`. The selftest was re-run: OK, AWS CALLS: 0.

**3. [Rule 1 - Bug, found in the test] The bound-order proptest lost its token branch at 2 texts**
- **Issue:** with 2 texts, 2 x 20 tokens never reaches the shrunk 64-token budget.
- **Fix:** the shrunk instance now pins `max_texts: 8`, and the KANI harness text names it.
- **Commit:** `968d73e99`.

**4. [Rule 2 - Missing critical] The maximal builder could emit an illegal request**
- **Issue:** DISTRIBUTED with a count above floor(budget / shortest row) produced a request over the budget, which the live probe would have timed as "maximal".
- **Fix:** `MaximalError::OverBudget`, with a test.
- **Commit:** `968d73e99`.

**5. [Rule 1 - Bug] laya-deploy contained a working deployment on the edge's transient 503**
- **Found during:** the option-1 redeploy.
- **Issue:** the platform's post-deploy load succeeded, yet the edge refused POSTs with 503 `Server is in error state` for under a minute. The identity probe hit that window, and the recipe contained.
- **Fix:** a bounded retry (8 x 15 s at most) of exactly that refusal. Any other probe failure still contains. The selftest has a wiring guard, which was mutation-tested.
- **Commit:** `743eb02ed`.

**Total deviations:** 1 user scope change, 4 auto-fixed (1 blocking, 2 bugs, 1 missing critical).
**Impact:** no margin, tolerance or artifact moved. The budget was lowered only after re-deriving it.

## Issues Encountered

- **The deploy was refused and contained** by the recipe's unrealizable health check (D-ITEM-08-17-A). This is the blocker below.
- **The `.planning/WINDOWS.md` ledger still refuses every append** (`Ledger entry 24 has invalid status: "resolved"`). The unrun Task 3 is recorded in the evidence and in D-ITEM-08-17-A instead.
- **The rtk hook condenses `aws` output,** so every JSON read used `rtk proxy aws`.

## Threat Flags

| Flag | File | Description |
|------|------|-------------|
| threat_flag: open-endpoint | 08-LIVE-DEPLOY-EVIDENCE.json | `aprender-mcp-decide` is RUNNING on pmcp.run with auth off (user-accepted), left serving for the admin UI. Each cold call buys about 25 s of 3 GB compute. Contain with `just laya-teardown` |

## First checkpoint (resolved: the user chose option 1)

- **Where things stand.** The function is live but contained: reserved concurrency 0, grant removed. The S3 object is in place.
- **What failed.** The 08-10 health-body check cannot pass on pmcp.run, because the edge answers every GET itself.
- **What still proves identity.** The compile log (passed), and the identity probe, a POST `tools/call` that must return `artifact_sha256 == 24a44d7e…` with labels none, against, favor. The probe is still realizable.

**Options:**
1. **Replace the health GET** with the edge's `/health` `serverId == aprender-mcp-decide` check, and let the identity probe carry package identity. (Recommended: it is realizable, and the probe already defeats the wrong-binary threat.)
2. **Reach the bootstrap directly** with `aws lambda invoke` and a synthetic GET event, so the package-naming body is still checked. This needs one more AWS call type.
3. **Stop at deploy-refused.** Tear down with the commands laya-teardown printed, and let 08-18 record the refusal.

**For 1 or 2, the continuation then runs these steps:**
1. Commits the recipe change.
2. Runs `aws lambda delete-function-concurrency --function-name aprender-mcp-decide`. This is the resume step laya-teardown leaves to the human, so it needs the human's approval.
3. Re-runs `just laya-deploy`, which redeploys, re-grants and runs the identity probe.
4. Takes the warm identity classify.
5. Runs Task 3 (`just laya-deploy-verify ... 2`) at the new 3 GB budget.

## Resume attempt 1 (2026-09-27): blocked on a pmcp.run auth gate

- **User decision:** option 1 above, and the user approved lifting the containment.
- **Recipe change, committed before any AWS write (`3115c690e`):**
  - The health step now derives `https://<host>/health` from the endpoint and cross-checks cargo pmcp's logged `health_endpoint`. It requires `serverId == aprender-mcp-decide` and asserts no package field.
  - The identity probe still carries package identity.
  - The logic lives in two pure helpers, `_laya-edge-health-url` and `_laya-edge-health-check`.
  - `laya-deploy-selftest` drives both through a must-accept/must-refuse table: 6 URL cases, plus 8 body cases that include the measured decide, chronos and 405 bodies and the bootstrap's own body. It also checks the wiring, and re-mutating the wiring (an `$ENDPOINT` GET, or a dropped check) turns that check red.
  - Selftest result: DEPLOY SELFTEST OK, AWS CALLS: 0.
- **Containment was already lifted when this continuation started.** CloudTrail shows `DeleteFunctionConcurrency` at 21:54:40Z by the profile's own IAM user. It was read back, not re-run. Current state:
  - Reserved concurrency is unset.
  - The grant is absent, so any invocation fails the S3 load fast.
  - No invocation was logged between the lift and 22:08Z.
  - The edge `/health` already returns `serverId aprender-mcp-decide`.
- **The re-run of `just laya-deploy` (22:01-22:06Z) exited 1 at an auth gate:**
  - Eligibility and compile passed.
  - It then stopped at pmcp.run's `Failed to get upload URLs`, with `UnauthorizedException: Valid authorization header not provided.`
  - Nothing was uploaded, and no AWS resource changed. The shared root was restored byte-identically.
  - A read-only `cargo pmcp deploy outputs` gets the same error. It refreshed the token on the first run, but it does not now.
- **Blocked on the human:** run `cargo pmcp deploy login --target-type pmcp-run` (browser OAuth). The continuation then re-runs `just laya-deploy`, the warm identity classify and Task 3.
- **Resolved at about 22:10Z:** the user logged in. The token is valid until 23:10:14Z, and the active target is `dev` (ze-kasher-dev, us-east-1).

## Resume attempt 2 (2026-09-27, 22:10:50Z-22:17:41Z): refused at the edge, contained

`just laya-deploy models/decide/laya-stance-64.apr models/decide/laya-stance-64 data/decide/tweet-stance-64 <base> aprender-mcp-decide dev ze-kasher-dev` exited 1.

| Step | Result |
|---|---|
| Eligibility | ok: `laya-verify` accepted the artifact, sha256 H |
| `cargo pmcp deploy --regenerate-stack --no-post-deploy-test --no-oauth` | rc 0. Deployment `dep_1790547363088_a3d2572f`, oauth off. The function was updated at 22:16:34Z: 3008 MB, arm64, Active/Successful |
| Compile log identity | ok: `aprender-mcp-decide-lambda`, and no other `*-lambda` package |
| `laya-grant` + 20 s IAM wait | applied |
| Edge health (`3115c690e`, its first live run) | ok: `/health` names serverId `aprender-mcp-decide` |
| Identity probe | **refused:** `initialize: HTTP 503 {"code":-32004,"message":"Server is in error state"}` (probe_id `probe-18d94d6c898037b8-125d1`) |
| Containment | reserved concurrency 0 and the grant removed, both verified read-only at 22:18:08Z |
| Shared root | restored byte-identical (`7c6f25eb…`); porcelain empty |

**Diagnosis. It was measured read-only, plus one diagnostic POST made while the function was contained.**
- **CloudWatch holds exactly one invocation after the redeploy,** at 22:16:47Z. It came 13 s after the update and before the grant, so it is pmcp.run's own post-deploy call (the recipe passes `--no-post-deploy-test`).
  - The load failed at the S3 object-length lookup after 5 attempts: `decide.load failed probe_id=none`.
  - REPORT: durationMs 170.9, Max Memory Used 37 MB, Init Duration 66.7 ms, memorySize 3008.
- **The probe's initialize has no invocation.** The edge answered 503 -32004 itself.
- **The error state is sticky.** A POST initialize at 22:18:34Z got the same 503 body, again with no invocation.
- **The edge `/health` still says healthy** (200, serverId, hasDeployment true). So `/health` does not show the MCP route's error state.
- **Cause (D-ITEM-08-17-B, now proven decisive).** pmcp.run invokes the function after every deploy, before an out-of-band grant can exist. It keeps the failed load as the server's state and refuses MCP traffic at the edge. The grant-after-deploy order therefore can never reach the identity probe on pmcp.run.
- **Not a model failure.** This is not an identity mismatch, an OOM or a probe-replay refusal. No model load has run on Lambda.
- **The redeploy kept the first deploy's execution role name.** So a grant applied before `cargo pmcp deploy` would already exist when the platform makes its call. This is untested.

**Not run:**
- The warm identity classify.
- Task 3 (`just laya-deploy-verify ... 2`): no cold samples, no Max Memory Used under load, and no comparison with the 17000 ms / 40 ms per token / ~350 MB headroom extrapolations.
- The admin-UI call.

Each of these needs a server the edge will route to. FALSIFY-DECIDE-TOOL-009 stays LIVE-PENDING.

**Teardown commands.** They are recorded in the evidence, and none has been run:
- Destroy the deployment: `just _laya-crates-root-swap crates crates/aprender-mcp-decide-lambda/.pmcp/deploy.toml models/decide/destroy-aprender-mcp-decide.state cargo pmcp deploy destroy --manifest-path crates`
- Remove the weights: `aws s3 rm --profile ze-kasher-dev --recursive s3://<weights-bucket>/decide/aprender-mcp-decide/`
- Or resume serving: `aws lambda delete-function-concurrency --profile ze-kasher-dev --function-name aprender-mcp-decide`

## Second checkpoint (resolved: the user chose option 1)

**Where things stand:**
- The function is deployed and contained: reserved concurrency 0, grant removed.
- The S3 object is in place.
- pmcp.run holds the server in an error state.

**What is needed:** the S3 read must exist before pmcp.run's post-deploy call. The fix changes how the weights permission reaches the role, so it is the human's choice. Every option except 4 then needs:
- a redeploy (a live pmcp.run login), which also clears the error state if the platform's call succeeds;
- lifting the containment again, which is your call.

**Options:**
1. **Declare the S3 read in the deploy config's `[iam]`** (recommended; the durable fix in D-ITEM-08-17-B).
   - cargo-pmcp's pmcp-run path runs a fail-closed IAM validator, and `--regenerate-stack` renders `[iam]` into stack.ts. This was read in the 0.24.2 source. 0.24.3, the installed version, is not in the local registry and still needs checking.
   - The policy then exists at create time.
   - `laya-deploy-config` would emit a `[[iam.statements]]` scoped to `decide/aprender-mcp-decide/*`, and `laya-grant` becomes a check.
2. **Pre-grant in `laya-deploy`:** run `laya-grant` before `cargo pmcp deploy` when the function already exists, and re-grant after it.
   - This is the smallest recipe change, and the role name survived the redeploy.
   - Risks: a stack update might drop an out-of-band policy, and a first deploy still has no role to grant to.
3. **Make the bootstrap answer `initialize`/`tools/list` without loading the model,** and load it lazily on `tools/call`.
   - This changes crate source, and so the plan's no-source-edit check.
   - The platform's call would then succeed with no grant at all.
4. **Stop at deploy-refused.** Tear down with the commands above, and let 08-18 record the refusal.

**After options 1-3, the continuation:**
1. Commits the change and re-runs `just laya-deploy-selftest` offline.
2. Lifts the containment.
3. Re-runs `just laya-deploy`.
4. Takes the warm identity classify, keeping it to 2 texts and 120 tokens or fewer.
5. Runs Task 3 (`just laya-deploy-verify ... 2`), taking Max Memory Used and Init Duration per sample with `cw_report.py`.
6. Makes the admin-UI call.

## Next Phase Readiness

- **ROADMAP and STATE move to 16/18.** 08-18 records `accepted_region_cold` from this outcome, and picks up D-ITEM-08-17-E (cold margin) and D-ITEM-08-17-C (software sha256).
- **Platform finding for the user's pmcp.run:** D-ITEM-08-17-D (the error state `/health` cannot see; sticky after a failed first call). Recommend a platform issue.

---
*Phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server*
*Completed: 2026-09-27 (deployed-passed on the option-1 resume)*

## Self-Check: PASSED

- Files exist: 08-LIVE-DEPLOY-EVIDENCE.json, contracts/decide-tool-boundary-v1.yaml (2.0.0), crates/aprender-mcp-decide-lambda/.pmcp/deploy.toml.template (memory_mb 3008).
- Commits exist: 968d73e99, b30f437da, 371c6bfe7.
- Adapted Task 2 verify: rc 0, with no bucket prefix, no account id, reserved=0, no source diff after the amendment, and an empty shared root. Task 3 verify: "Task 3 skipped on deploy-refused".

### Self-Check (resume attempt 2): PASSED

- The adapted Task 2 verify passed:
  - The evidence parses as `deploy-refused` / `deploy-auth-off-accept-risk` with memory 3008 and sha H.
  - There is no bucket prefix and no 12-digit run in it.
  - The live account id (read with sts, never written) is absent from the evidence, this SUMMARY and deferred-items.md.
  - `reserved=0`.
  - `git diff --quiet 968d73e99` over contracts/ and the three decide crates' src/ is clean.
  - The `crates/.pmcp` and `crates/deploy` porcelain is empty.
- The Task 3 verify rule holds: there are no `cold_samples` on a refused branch.

### Self-Check (option-1 resume): PASSED

- The Task 3 verify printed `deployed-passed samples 4 max ms 29350`: 2 or more samples per shape, CloudWatch cold evidence and a probe id for each, and a warm p50 of 1418.
- The evidence carries no bucket prefix. The live account id (read with sts) is absent from it.
- `git diff --quiet 968d73e99` over contracts/ and the three decide crates' src/ is clean (rc 0).
- `just laya-deploy-selftest`: DEPLOY SELFTEST OK, AWS CALLS: 0.
- Commits exist: 52a12d777, 743eb02ed.
- `get-function-concurrency` shows no reservation, so the function is RUNNING.

