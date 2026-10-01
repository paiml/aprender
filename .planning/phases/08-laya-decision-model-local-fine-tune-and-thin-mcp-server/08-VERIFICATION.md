---
phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server
verified: 2026-09-28T01:12:08Z
status: gaps_found
score: 16/18 must-haves verified
covered_files:
  - ".github/workflows/ci.yml"
  - ".gitignore"
  - ".planning/REQUIREMENTS.md"
  - ".planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-01-PLAN.md"
  - ".planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-01-SUMMARY.md"
  - ".planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-02-PLAN.md"
  - ".planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-02-SUMMARY.md"
  - ".planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-03-PLAN.md"
  - ".planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-03-SUMMARY.md"
  - ".planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-04-PLAN.md"
  - ".planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-04-SUMMARY.md"
  - ".planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-05-PLAN.md"
  - ".planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-05-SUMMARY.md"
  - ".planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-06-PLAN.md"
  - ".planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-06-SUMMARY.md"
  - ".planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-07-PLAN.md"
  - ".planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-07-SUMMARY.md"
  - ".planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-08-PLAN.md"
  - ".planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-08-SUMMARY.md"
  - ".planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-09-PLAN.md"
  - ".planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-09-SUMMARY.md"
  - ".planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-10-PLAN.md"
  - ".planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-10-SUMMARY.md"
  - ".planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-11-PLAN.md"
  - ".planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-11-SUMMARY.md"
  - ".planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-12-PLAN.md"
  - ".planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-12-SUMMARY.md"
  - ".planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-13-PLAN.md"
  - ".planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-13-SUMMARY.md"
  - ".planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-14-PLAN.md"
  - ".planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-14-SUMMARY.md"
  - ".planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-15-PLAN.md"
  - ".planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-15-SUMMARY.md"
  - ".planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-16-PLAN.md"
  - ".planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-16-SUMMARY.md"
  - ".planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-17-PLAN.md"
  - ".planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-17-SUMMARY.md"
  - ".planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-18-PLAN.md"
  - ".planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-18-SUMMARY.md"
  - ".planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-CONTEXT.md"
  - ".planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-DEPLOY-EVIDENCE.json"
  - ".planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-GATE-RUN-EVIDENCE.json"
  - ".planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-LIVE-DEPLOY-EVIDENCE.json"
  - ".planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/deferred-items.md"
  - "CLAUDE.md"
  - "Cargo.lock"
  - "Cargo.toml"
  - "Makefile"
  - "README.md"
  - "contracts/aprender/binding.yaml"
  - "contracts/decide-apr-v1.yaml"
  - "contracts/decide-tool-boundary-v1.yaml"
  - "contracts/laya-finetune-gate-v1.yaml"
  - "contracts/laya-parity-v1.yaml"
  - "crates/apr-format/src/v2/mod.rs"
  - "crates/apr-format/src/v2/reader_impl.rs"
  - "crates/apr-format/src/v2/tests.rs"
  - "crates/aprender-core/Cargo.toml"
  - "crates/aprender-core/src/models/mod.rs"
  - "crates/aprender-core/src/models/modernbert/config.rs"
  - "crates/aprender-core/src/models/modernbert/embeddings.rs"
  - "crates/aprender-core/src/models/modernbert/encoder.rs"
  - "crates/aprender-core/src/models/modernbert/gemm.rs"
  - "crates/aprender-core/src/models/modernbert/layer.rs"
  - "crates/aprender-core/src/models/modernbert/load.rs"
  - "crates/aprender-core/src/models/modernbert/mod.rs"
  - "crates/aprender-core/tests/monorepo_invariants.rs"
  - "crates/aprender-decide/Cargo.toml"
  - "crates/aprender-decide/README.md"
  - "crates/aprender-decide/examples/pack_laya.rs"
  - "crates/aprender-decide/src/artifact.rs"
  - "crates/aprender-decide/src/artifact/determinism.rs"
  - "crates/aprender-decide/src/artifact/ladder.rs"
  - "crates/aprender-decide/src/artifact/tests.rs"
  - "crates/aprender-decide/src/laya/builder.rs"
  - "crates/aprender-decide/src/laya/head.rs"
  - "crates/aprender-decide/src/laya/mod.rs"
  - "crates/aprender-decide/src/laya/scorer.rs"
  - "crates/aprender-decide/src/laya/temperature.rs"
  - "crates/aprender-decide/src/laya/tests.rs"
  - "crates/aprender-decide/src/lib.rs"
  - "crates/aprender-decide/src/pack.rs"
  - "crates/aprender-decide/src/task.rs"
  - "crates/aprender-decide/src/test_support.rs"
  - "crates/aprender-decide/src/verify.rs"
  - "crates/aprender-decide/src/verify/tests.rs"
  - "crates/aprender-decide/tests/common/mod.rs"
  - "crates/aprender-decide/tests/demo_run.rs"
  - "crates/aprender-decide/tests/fail_closed_vectors.rs"
  - "crates/aprender-decide/tests/laya_parity.rs"
  - "crates/aprender-decide/tests/python_records.rs"
  - "crates/aprender-decide/tests/ui.rs"
  - "crates/aprender-decide/tests/ui/decider_no_constructor.rs"
  - "crates/aprender-decide/tests/ui/decider_no_constructor.stderr"
  - "crates/aprender-decide/tests/ui/decider_struct_literal.rs"
  - "crates/aprender-decide/tests/ui/decider_struct_literal.stderr"
  - "crates/aprender-mcp-decide-lambda/.pmcp/.gitignore"
  - "crates/aprender-mcp-decide-lambda/.pmcp/deploy.toml.template"
  - "crates/aprender-mcp-decide-lambda/Cargo.toml"
  - "crates/aprender-mcp-decide-lambda/README.md"
  - "crates/aprender-mcp-decide-lambda/examples/probe.rs"
  - "crates/aprender-mcp-decide-lambda/src/lib.rs"
  - "crates/aprender-mcp-decide-lambda/src/main.rs"
  - "crates/aprender-mcp-decide-lambda/src/probe.rs"
  - "crates/aprender-mcp-decide-lambda/src/s3.rs"
  - "crates/aprender-mcp-decide-lambda/src/tests.rs"
  - "crates/aprender-mcp-decide/Cargo.toml"
  - "crates/aprender-mcp-decide/README.md"
  - "crates/aprender-mcp-decide/src/lib.rs"
  - "crates/aprender-mcp-decide/src/main.rs"
  - "crates/aprender-mcp-decide/src/tests.rs"
  - "crates/aprender-mcp-decide/tests/e2e_stdio.rs"
  - "justfile"
  - "scripts/laya_deploy/cargo_pmcp_resolver_proof.rs"
  - "scripts/laya_train/.gitignore"
  - "scripts/laya_train/.python-version"
  - "scripts/laya_train/README.md"
  - "scripts/laya_train/common.py"
  - "scripts/laya_train/contract.py"
  - "scripts/laya_train/data.py"
  - "scripts/laya_train/fixtures.py"
  - "scripts/laya_train/gate.py"
  - "scripts/laya_train/lifecycle.py"
  - "scripts/laya_train/metrics.py"
  - "scripts/laya_train/prepare_stance.py"
  - "scripts/laya_train/pyproject.toml"
  - "scripts/laya_train/train.py"
  - "scripts/laya_train/uv.lock"
covered_digest: "v1:sha256:95cff3524f937db9d3fefc09fb4ae8e4a3b0ad1a9fa1f9356cea6818d60c38af"
behavior_unverified: 0
overrides_applied: 0
gaps:
  - truth: "Deploy eligibility (`pack_laya verify` / `just laya-verify`) enforces every check that contracts/decide-apr-v1.yaml `deploy_eligibility.verify_checks` lists, and so binds the served identity (the D-11 tuple, including `model.base`) to the artifact it serves"
    status: partial
    reason: "Code review CR-01, confirmed by reading the source. decide-apr-v1 verify_checks[2] says eligibility checks 'base sha256 equal to the manifest's declared base'. `check_base` (verify.rs:924-945) compares the contract's base sha256 and the base dir's file against the RUN DIR's recipe (`inputs.recipe.base`). It never compares them against `manifest.base`. Neither `verify_path` (verify.rs:2395-2424) nor ladder rung 4 compares `manifest.base`, `manifest.variant`, `manifest.calibration.*`, `manifest.gate.report_sha256`, or `manifest.inputs_sha256.task_json/.tokenizer_json` against the embedded, sha-bound blobs. `manifest.base.display()` is copied verbatim into `ModelIdentity.base` (artifact.rs:1470), which every classify response serves as `model.base`. A crafted .apr whose weights re-score correctly can therefore carry a false base identity or a fabricated calibration/gate summary and still print `deploy_eligible: true`. What still holds: `artifact_sha256` (the whole-file hash) and `recipe_id` (rung 4 binds it to the embedded recipe blob, and pack.rs:674 binds that to the run's recipe.json) are verified, so the literal D-11 wording (content hash + recipe id) is met. Also, the served behaviour is bound, because the re-score runs on the loaded bytes."
    artifacts:
      - path: "crates/aprender-decide/src/verify.rs"
        issue: "`check_base` reads `inputs.recipe.base`, not `manifest.base`. `verify_path` binds only recipe_id, gate.report_sha256, inputs_sha256 and labels to the run dir, never the manifest to the embedded blobs"
      - path: "crates/aprender-decide/src/artifact.rs"
        issue: "Rung 4 verifies the blob hashes and recipe_id. It does not cross-check manifest.base/variant against the embedded recipe blob, manifest.gate.report_sha256 against the gate-report blob hash, manifest.inputs_sha256.task_json/tokenizer_json against the task/tokenizer blob hashes, or manifest.calibration against the report and agent config"
      - path: "contracts/decide-apr-v1.yaml"
        issue: "`deploy_eligibility.verify_checks[2]` claims a check the code does not perform as worded"
    missing:
      - "Rung-4 cross-checks: manifest.base == recipe_blob.base; manifest.variant == recipe_blob.variant; manifest.gate.report_sha256 == the gate-report blob sha; manifest.inputs_sha256.task_json and .tokenizer_json == the task and tokenizer blob shas; manifest.calibration == the embedded report's calibration block, with t_applied == clamp(agent_config.temperature_by_options[bucket])"
      - "Negative lib tests, one per field, each proving a crafted manifest refuses at load (and therefore at verify)"
      - "Optionally, have `verify_path` require the report's calibration.t_applied to equal the loaded Decider's applied temperature"
      - "No redeploy is needed. The deployed artifact 24a44d7e... was checked field by field in this verification, and every manifest field agrees with its embedded blobs (see Gaps Summary)"
human_verification:
  - test: "Push the branch (134 commits ahead of origin) and watch the first CI run"
    expected: "`workspace-test` runs the aprender-decide, aprender-mcp-decide, aprender-mcp-decide-lambda and aprender-core modernbert lib tests green. The ci.yml integration step runs `cargo test -p aprender-decide --test ui` and `cargo test -p aprender-mcp-decide --test e2e_stdio` green (its real-model leg SKIPs by design)"
    why_human: "The goal says 'CI-tested'. The ci.yml line exists (commit 57e18b541), but no CI run has ever executed Phase 8 code: the branch has not been pushed since. Only the local equivalents ran in this verification"
  - test: "Decide whether the served `classify` tool description may keep its sentence 'Long texts are truncated by the model itself to its window, and each such result reports `truncated: true`' at the 3,008 MB tier"
    expected: "Either the sentence is made tier-aware (review WR-03), or the owner accepts it"
    why_human: "The live tools/list, fetched in this verification, shows both that sentence and 'at most 120 model tokens over the whole request'. decide-tool-boundary-v1 itself says `truncated: true` is unreachable at this tier. So the MCP surface that LLM clients read states the approved 120-token budget, but also promises a behaviour the budget forbids"
---

# Phase 8: Laya Decision Model — Local Fine-Tune and Thin MCP Server — Verification Report

**Phase Goal:** Productise spikes 024–026. A user can fine-tune Laya (the ModernBERT-large decision model) locally on their own 8–64 labelled shots, calibrate it, convert it to .apr, and serve it through a thin, task-bound `classify` MCP server. That server must be in-tree, linted, contracted and CI-tested, and live on pmcp.run (default Lambda) the way the SetFit and Chronos servers are. ModernBERT lands as a reusable aprender-core model. The decision layer is a method-neutral `aprender-decide` crate with Laya as its first method.
**Verified:** 2026-09-28T01:12:08Z
**Status:** gaps_found (one narrow integrity gap; the rest of the goal is achieved)
**Re-verification:** No — initial verification

ROADMAP gives no `success_criteria` for Phase 8, and REQUIREMENTS.md maps no IDs to it ("Requirements: TBD"). The truths below are therefore derived from the goal and the 18 plans' `must_haves`. Every D-01..D-19 that the PLAN frontmatter declares is accounted for against 08-CONTEXT.md and its 2026-09-27 amendments.

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence (gathered in this verification unless noted) |
|---|-------|--------|---------------------------------|
| 1 | A user fine-tunes Laya locally with `just laya-train`. The uv project is in-tree and pinned, training runs on Laya's own code, and the report records the device actually used (D-01..D-04) | ✓ VERIFIED | `scripts/laya_train/pyproject.toml` pins torch==2.14.0, transformers==5.17.0, and laya@4066d5d5…; `.python-version` 3.13.7; `uv.lock` is present. `train.py:175` reads the device from `next(agent.model.parameters()).device`. The run dir `models/decide/laya-stance-64/` exists, and its recipe.json carries the declared base (sha 891102d3…, rev 55cf4c4e) and the spike recipe values. The torch-free self-tests ran here: METRICS/DATA/GATE SELFTEST OK. `contracts/apr-cli-commands-v1.yaml` has no commit since the phase began |
| 2 | `task.json` + `train.jsonl` + a required `eval.jsonl`. Criteria order is the label index, and bad tasks are refused rather than defaulted (D-05) | ✓ VERIFIED | `task.rs` has an order-preserving visitor plus a backing canary. The data.py selftest includes `eval-missing`. The live tools/list shows labels [none, against, favor] in task.json document order. aprender-decide has 112 lib tests, all green |
| 3 | Temperature is refit on a seeded, held-out slice of the train shots, and eval and calibration never overlap (D-06) | ✓ VERIFIED | gate-report `calibration.slice_size` 48, and `slice_ids` are recorded and sha-bound. Normalised eval/train text overlap measured here = 0 of 459. t_applied 3.4278 equals `rl_agent_config.json` `temperature_by_options["choice:3-5"]`. (WR-08: Rust does not enforce the slice fraction; see Anti-Patterns) |
| 4 | The gate is fail-closed. Thresholds are declared in a contract before any run, the eval set is in-distribution by rule (A2), and deploy refuses without a passing, Rust-recomputed gate (D-07) | ✓ VERIFIED | The declaration commits fc0c1dd30, a3e155170, c99761825 and 3cfa83e1b are all ancestors of run commit 39be66265, and recipe.json (11:08) postdates a3e155170 (10:00). Thresholds are unchanged (0.10 / 0.05). `just laya-verify` was re-run here on the exact file: rc 0, `deploy_eligible:true`, recomputed ece_post 0.04425, margin 0.22166, argmax 459/459, bit-identical to 08-GATE-RUN-EVIDENCE.json. Eval set: 459 rows, [111, 291, 57]; shift probe 280 rows reported with `gate_clause: false` |
| 5 | Seed policy: production runs seeds 13/17/23 and ships the median-ECE seed, never the best (D-08 amended, A3). Variance is reported | ✓ VERIFIED | Rank keys 849/442/371 put seed 17 at index 1; 17 ships (checkpoint sha 10264ea0… = seed 17's). The Rust verify re-derived `shipped_seed:17` here. `variance-report.json` says "information only… mean / sd never select anything" |
| 6 | ModernBERT is a reusable aprender-core model at `models/modernbert/`, not built on `models/bert` (D-13) | ✓ VERIFIED | 7 files, 2222 lines. `cargo test -p aprender-core --lib models::modernbert`: 18 passed (tiny_parity, window_mutation, norm_eps_honoured, config_domain, refuses_missing_tensor_by_name, prefix_reuse…). Clippy: 0 diagnostics under models/modernbert |
| 7 | `aprender-decide` is method-neutral: a `DecisionMethod` seam with Laya as the only implementation, reusing core's encoder and primitives, with no bin target (D-14) | ✓ VERIFIED | `trait DecisionMethod` (lib.rs:158) with task/prepare/classify_prepared. laya/mod.rs:36,319 use `aprender::models::modernbert` and `ModernBertEncoder::from_apr(reader, "…encoder.")`. head.rs and scorer.rs import core's Linear, layer_norm, attention and gelu_exact. Cargo.toml has no `[[bin]]` |
| 8 | The served artifact is .apr, stored F16 and widened at load. Parity is torch → .apr → Rust: the fixture bar is 1e-5, and the pack bar is noise-referenced (D-17, A1) | ✓ VERIFIED | Real-weights parity was re-run here (`laya_parity`, LAYA_MODEL_DIR): ids 14/14, argmax 14/14, max \|dp\| 3.841e-6 (bar 1e-5), load 2.19 s through the F16 .apr path, ARCH aarch64. Pack bars from verify: 3.34e-5 (fine-tuned) and 5.98e-5 (zero-shot), both from the float64 record; the observed maxima were 1.13e-5 and 7.35e-6. The manifest carries base, recipe, gate and calibration, plus sha256 of every input. x86_64 is unmeasured (D-ITEM-08-13-A, stated in CLAUDE.md) |
| 9 | Exactly one task-bound `classify` tool. The question and labels come from the artifact, and the caller sends only `texts` (D-09) | ✓ VERIFIED | LIVE tools/list (this verification) returns one tool, `classify`. Its description quotes the artifact's question and lists [none, against, favor]. `ClassifyArgs` has one field, `texts`. e2e_stdio passes (tiny leg in CI; real leg armed here) |
| 10 | `classify` takes a list whose maximum is owned by a contract and priced against the Lambda envelope (D-10) | ✓ VERIFIED | A LIVE 3-text call was refused with "3 texts exceeds classify_max_texts 2 (contracts/decide-tool-boundary-v1.yaml)". The contract mirror tests are in the 26 green aprender-mcp-decide lib tests. The 3,008 MB re-pricing (2 texts / 120 tokens) is derived term by term in decide-tool-boundary-v1 |
| 11 | Each result returns `label` plus calibrated `probabilities` over every label in order, and the response carries the model identity: artifact content hash + recipe id (D-11) | ✓ VERIFIED | LIVE classify (this verification): `artifact_sha256` 24a44d7e… equals `shasum -a 256 models/decide/laya-stance-64.apr`. `recipe_id` 6a5489af… equals sha256(run recipe.json). Probabilities [0.129, 0.245, 0.626] sum to 1, and label=favor is the argmax. The armed stdio real leg passed here with identity == the file sha. (The extra `base` field is covered by truth 18) |
| 12 | Text past the 512-token window is truncated exactly as Laya's builder does, with `truncated: true` (D-12) | ✓ VERIFIED (library) | Real-weights parity rec 11: 512 tokens, "truncated rows 1", ids exact. The tiny fixture's over-window row is in the lib tests. At the served 3 GB tier the flag is unreachable, because the 120-token budget refuses the row first. The contract says so; the tool description does not (human item 2 / WR-03) |
| 13 | `aprender-mcp-decide` (stdio) and `aprender-mcp-decide-lambda` (bootstrap) are in the FALSIFY-MONO-011 `deployment_unit_bins` register with `publish = false` (D-15) | ✓ VERIFIED | monorepo_invariants.rs:320-321. Both Cargo.toml files have `publish = false`. `cargo test -p aprender-core --test monorepo_invariants --test readme_contract`: 11 + 15 passed |
| 14 | CLAUDE.md gains the third realizar-first exception row, argued like SetFit and Forecast, with its claims licensed by the records (D-16) | ✓ VERIFIED | CLAUDE.md:202 (the row) and :239-274 (the paragraph). It states the in-distribution claim scope, shift ECE 0.1896, the 3 GB tier only, the thin cold margin (29.35 s, and 31.05 s external), and x86_64 unmeasured. readme_contract (FALSIFY-DOCS-CLAUDE-001) passes |
| 15 | One trained model is live on pmcp.run default Lambda, with weights fetched from S3 at cold start, and a real classify call verifies it (D-18; 3,008 MB tier and auth off approved by the user) | ✓ VERIFIED | A LIVE read-only POST at 2026-09-28T00:59Z returned HTTP 200 in 29.35 s with header `x-decide-load: cold;load_ms=24657`: a real S3 cold load, and identity == H. The final record (`final: true`, deployed-passed) has 4 CloudWatch-proven cold samples, 24.2-29.4 s. Auth off, the risk acceptance and the RUNNING posture are recorded (D-ITEM-08-18-A) |
| 16 | The demo is TweetEval stance at 64 shots/class, and exactly one declared run passes the gate (D-19 amended) | ✓ VERIFIED | Train is 192 rows, 64/64/64, measured here. The gate evidence records outcome gate_pass for s64-seed13 data, and `demo_s64.outcome: gate_pass` is in the contract. The s16 vectors are kept as fail-closed history |
| 17 | The work is in-tree, linted, contracted and CI-tested | ? UNCERTAIN (CI part) | In-tree: yes. Linted: `cargo clippy -p aprender-decide -p aprender-mcp-decide -p aprender-mcp-decide-lambda --all-targets --no-deps -- -D warnings` rc 0; `cargo fmt --check` rc 0 on those crates and modernbert. Contracted: `pv validate` rc 0 on all four contracts; `make contract-audit-phase8` rc 0 (44 rows resolved, no BIND- line). CI: the ci.yml line has the two approved targets, and the new crates' lib tests fall under workspace-test. But the branch is 134 commits ahead of origin, so no CI run has ever executed this code (human item 1). The narrowing is user-approved and recorded (D-ITEM-08-12-B/-C) |
| 18 | Deploy eligibility enforces every decide-apr-v1 `verify_check`, binding the served identity fields (incl. `model.base`) to the artifact | ✗ FAILED (partial) | CR-01, confirmed in source (see gaps). The deployed artifact itself is consistent: manifest `base` = recipe base (sha 891102d3…, rev 55cf4c4e); `gate.report_sha256` 2c70a663… = the gate_report blob hash = sha(run gate-report.json); `inputs_sha256.task_json` 041c4e38… = the task blob; `tokenizer_json` 6c8aaa9a… = the tokenizer blob; `calibration.t_applied` 3.4278388… = agent config `choice:3-5`. The deployed model's identity is honest; the verifier's general guarantee is not |

**Score:** 16/18 truths verified (0 present-but-behavior-unverified; 1 uncertain; 1 failed-partial)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `contracts/{decide-tool-boundary-v1,laya-finetune-gate-v1,laya-parity-v1,decide-apr-v1}.yaml` | Four Phase 8 contracts | ✓ VERIFIED | v3.0.0 / 3.0.0 / 2.0.0 / 1.0.0, with 8+13+11+12 = 44 equations; `pv validate` rc 0 on each |
| `crates/aprender-core/src/models/modernbert/` | Reusable encoder | ✓ VERIFIED | Registered in models/mod.rs:37 (gated on `parallel`); 18 tests green |
| `crates/aprender-decide/` | Seam, Laya, artifact, pack, verify | ✓ VERIFIED | 6.5k lines of src; 112 lib tests + ui trybuild green; `pack_laya` example drives pack/verify/inspect |
| `crates/aprender-mcp-decide/` | stdio thin server | ✓ VERIFIED | 26 lib + 2 e2e green; real leg armed and green (model load 65 s in a debug build) |
| `crates/aprender-mcp-decide-lambda/` | bootstrap, S3 loader, probe | ✓ VERIFIED | 29 lib tests green; LIVE cold load observed |
| `scripts/laya_train/` | Pinned uv back office | ✓ VERIFIED | train/gate/data/metrics/prepare_stance/lifecycle; self-tests OK |
| `justfile` laya-* recipes | Train, pack, verify, deploy, teardown | ✓ VERIFIED | laya-train:1327, laya-pack:1402, laya-verify:1412, laya-deploy:1743, laya-teardown:2572, laya-verify-suite:1370 |
| `08-GATE-RUN-EVIDENCE.json`, `08-LIVE-DEPLOY-EVIDENCE.json` | Outcome records | ✓ VERIFIED | gate_pass; deployed-passed, `final: true`; the numbers reproduce (verify re-run, live call) |
| `crates/aprender-mcp-decide-lambda/README.md` Deployed section | Endpoint, identity check, claim scope | ✓ VERIFIED | States the 3 GB limits, the 25-31 s cold time, auth off, and in-distribution scope |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| aprender-decide laya/mod.rs | core ModernBertEncoder | `from_apr(reader, "{prefix}encoder.")` | ✓ WIRED | laya/mod.rs:319 |
| pack_laya verify | laya-finetune-gate-v1 / laya-parity-v1 | serde_yaml constants at run time | ✓ WIRED | verify output bounds = 4 x the recorded noise, floor 1e-5 |
| justfile laya-deploy / laya-upload | `just laya-verify` | eligibility before any AWS write | ✓ WIRED | Live evidence sequence: "laya-verify accepted first" on every step |
| stdio server | `Decider::classify_prepared` | `spawn_blocking` after precheck | ✓ WIRED | lib tests + e2e |
| Lambda bootstrap | `load_model_from_bytes` | sha256(buffer) == APRENDER_DECIDE_SHA256, then the ladder | ✓ WIRED | Live header `cold;load_ms=24657`; identity == pin |
| binding.yaml `accepted_region_cold` | 08-LIVE-DEPLOY-EVIDENCE.json | status from the final record | ✓ WIRED | `implemented`, evidence path cited; FALSIFY-DECIDE-TOOL-009 names the live harness |
| verify_path | manifest identity fields ↔ embedded blobs | cross-check | ✗ PARTIAL | CR-01 (truth 18) |

### Data-Flow Trace (Level 4)

| Artifact | Data | Source | Real data | Status |
|----------|------|--------|-----------|--------|
| Live `classify` response `model.artifact_sha256` | file hash | `artifact_sha256_hex` over the S3-loaded buffer, pinned | yes: equals the local file's shasum | ✓ FLOWING |
| Live `probabilities` | calibrated probs | ModernBERT + head + T=3.4278 on the loaded weights | yes: a cold load really happened (24.7 s) | ✓ FLOWING |
| Live `model.base` | display string | `manifest.base.display()`, never cross-checked | correct for this artifact; unverified in general | ⚠️ see truth 18 |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Deploy eligibility on the exact deployed file | `just laya-verify models/decide/laya-stance-64.apr … <base>` | rc 0, `deploy_eligible:true`, sha 24a44d7e…, shipped_seed 17 | ✓ PASS |
| Full-model parity vs spike 025 | `LAYA_MODEL_DIR=… cargo test -p aprender-decide --release --test laya_parity` | 14/14 ids, \|dp\| 3.841e-6, truncated rows 1 | ✓ PASS |
| Real model over live stdio | `APR_MCP_E2E_DECIDE_MODEL=<abs> cargo test -p aprender-mcp-decide --test e2e_stdio a_real` | identity == file sha, 1 passed (116 s debug) | ✓ PASS |
| Live classify (cold) | `rtk proxy curl -X POST …/mcp tools/call classify` (1 text) | 200, 29.35 s, `cold;load_ms=24657`, identity == H | ✓ PASS |
| Live tools/list and over-count refusal | POST tools/list; classify with 3 texts | one `classify` tool; refusal names `classify_max_texts 2` | ✓ PASS |
| Lib tests, 3 new crates | `cargo test -p aprender-decide -p aprender-mcp-decide -p aprender-mcp-decide-lambda --lib` | 112 + 26 + 29 passed | ✓ PASS |
| trybuild private constructor | `cargo test -p aprender-decide --test ui` | 1 passed | ✓ PASS |
| Contract audit | `make contract-audit-phase8` | rc 0, 44 rows resolved, EXEMPT empty | ✓ PASS |

### Probe Execution

Not applicable. The phase declares no `scripts/*/tests/probe-*.sh`. Its live "probe" is `crates/aprender-mcp-decide-lambda/examples/probe.rs`; its recorded results are in 08-LIVE-DEPLOY-EVIDENCE.json, and the live calls above re-observed its identity claim.

### Requirements Coverage (D-IDs declared in PLAN frontmatter)

| D-ID | Declared by | Status | Evidence |
|------|-------------|--------|----------|
| D-01 | 08-02, 08-08, 08-14 | ✓ SATISFIED | Truth 1 |
| D-02 | 08-02, 08-08 | ✓ SATISFIED | Truth 1; CLI registry untouched |
| D-03 | 08-01, 08-08 | ✓ SATISFIED | Truth 1; `device_used: mps:0` read from params |
| D-04 | 08-01, 08-05, 08-08 | ✓ SATISFIED | Base is a declared manifest field; recipe values in recipe.json |
| D-05 | 08-01, 08-02, 08-04, 08-08 | ✓ SATISFIED | Truth 2 |
| D-06 | 08-01, 08-08, 08-09, 08-13, 08-14 | ✓ SATISFIED | Truth 3 (WR-08 warning) |
| D-07 (+A2 amendment) | 08-01, 08-08, 08-09, 08-10, 08-11, 08-13..16 | ✓ SATISFIED | Truth 4; claim scope stated in contract, CLAUDE.md and README |
| D-08 (+A3 amendment) | 08-01, 08-08, 08-13..16 | ✓ SATISFIED | Truth 5 |
| D-09 | 08-01, 08-06 | ✓ SATISFIED | Truth 9 |
| D-10 | 08-01, 08-06, 08-17, 08-18 | ✓ SATISFIED | Truth 10 |
| D-11 | 08-01, 08-05, 08-06, 08-07, 08-09, 08-10, 08-15, 08-16, 08-17 | ✓ SATISFIED (literal: hash + recipe id); ✗ `base` field unverified | Truths 11 and 18 |
| D-12 | 08-01, 08-04, 08-06, 08-09 | ✓ SATISFIED (library); unreachable at the 3 GB tier | Truth 12 |
| D-13 | 08-02, 08-03, 08-04 | ✓ SATISFIED | Truth 6 |
| D-14 | 08-04, 08-05 | ✓ SATISFIED | Truth 7 (crate not yet published; its one-way name is unconfirmed until publish) |
| D-15 | 08-06, 08-07, 08-12 | ✓ SATISFIED | Truth 13 |
| D-16 | 08-12 | ✓ SATISFIED | Truth 14 |
| D-17 (+A1 amendment) | 08-01, 08-02, 08-03, 08-05, 08-09, 08-12..16 | ✓ SATISFIED; ✗ one decide-apr-v1 eligibility claim unenforced | Truths 8 and 18 |
| D-18 | 08-07, 08-10, 08-11, 08-12, 08-17, 08-18 | ✓ SATISFIED (3,008 MB, approved) | Truth 15 |
| D-19 (+amendment) | 08-01, 08-08, 08-09, 08-13, 08-16 | ✓ SATISFIED | Truth 16 |

No orphaned requirements: REQUIREMENTS.md maps no IDs to Phase 8.

**User-approved scope changes. Each is honoured as scope, and each is stated honestly where it is claimed:**
- **A2 in-distribution eval set (option 3, then option 1).** Stated in the laya-finetune-gate-v1 `eval_set.claim`, CLAUDE.md, the Lambda README and CONTEXT D-07. Shift ECE 0.1896 is disclosed.
- **Median-of-3 seeds.** CONTEXT D-08 amendment and the contract say "median, not best" and "selected WITH eval labels".
- **Noise-referenced pack bar.** CONTEXT D-17 amendment, laya-parity-v1 A1 and CLAUDE.md, which says both bars are aarch64-only.
- **3,008 MB tier / 120-token, 2-text budget.** Stated in decide-tool-boundary-v1 2.0.0 (term-by-term extrapolation, and why it is thinner), CLAUDE.md, the README and live evidence `memory_decision`. The one gap is that 08-CONTEXT.md D-18 still reads "10,240 MB" with no amendment note (info). The served tool description is not tier-aware (human item 2).
- **Auth off, risk accepted.** Live evidence `auth.risk_acceptance`, README and D-ITEM-08-18-A.
- **Narrowed CI.** ci.yml plus D-ITEM-08-12-B/-C. `just laya-verify-suite` is documented as local-only.
- **Function left running.** Live evidence `posture`, D-ITEM-08-18-A, and the containment commands printed but not run.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| crates/aprender-decide/src/verify.rs, artifact.rs | 924-945, 2395-2424; rung 4 | CR-01: manifest identity/calibration fields never bound to the embedded blobs | 🛑 Blocker (contract claim unenforced) | Truth 18 / the gap above |
| crates/apr-format/src/v2/reader_impl.rs; artifact.rs | 131; 764-782 | WR-01: duplicate tensor names pass the "every entry" rungs (strict `<` sort check) | ⚠️ Warning | Contradicts decide-apr-v1's "bijection" wording for crafted files. The duplicate is inert, because every lookup takes the first match |
| crates/aprender-mcp-decide/src/lib.rs | 499-505 | WR-03: tool description promises truncation that is unreachable at 120 tokens | ⚠️ Warning | Human item 2; observed live |
| crates/aprender-decide/src/verify.rs | 1072-1138 | WR-08: calibration slice fraction not re-derived in Rust | ⚠️ Warning | The D-06 split is only partly re-derived. The shipped run's slice (48 of 192, 25 %) is correct |
| crates/aprender-decide/src/verify.rs | 2326-2336 | WR-02: public `verify_run` can report eligible without the manifest binding | ⚠️ Warning | No production caller |
| scripts/laya_train/data.py, prepare_stance.py | 147; 60, 102 | WR-05: `splitlines()` vs Rust `lines()` on U+2028/U+0085 | ⚠️ Warning | Latent divergence; the demo data parsed identically |
| crates/aprender-core/src/models/modernbert/layer.rs | 290-350 | WR-04: `ModernBertLayer::forward` panics on an empty row | ⚠️ Warning | The public API promises typed errors |
| justfile | 2341-2419 | WR-07: `_laya-grant-check` ignores managed policies and wildcard patterns | ⚠️ Warning | "Nothing broader" is overclaimed |
| crates/aprender-mcp-decide-lambda/.pmcp/deploy.toml.template, COVERAGE.md, 08-CONTEXT.md D-18 | — | Stale text: GET health body as the tell; ListBucket / put_role_policy grant; 10,240 MB with no amendment note | ℹ️ Info | Documentation drift after 08-17's option 1 and the tier decision |

The scan found no TBD/FIXME/XXX, TODO/HACK or `todo!`/`unimplemented!` in the Phase 8 files. Production code has no `.unwrap()`; the crate-level `disallowed_methods` allow is IN-05.

### Human Verification Required

#### 1. First CI run of Phase 8 code

**Test:** Push `gsd/phase-2-contract-gate` and watch `workspace-test` and the integration step.
**Expected:** The new crates' lib tests are green; `-p aprender-decide --test ui` and `-p aprender-mcp-decide --test e2e_stdio` are green.
**Why human:** "CI-tested" is part of the goal, and no CI run has ever executed this code. Only local equivalents ran.

#### 2. Tier-aware tool description (WR-03)

**Test:** Read the live `tools/list` description.
**Expected:** Decide whether "Long texts are truncated… `truncated: true`" is acceptable next to "at most 120 model tokens", or make it tier-aware.
**Why human:** This is a wording and product call about the published tool contract (D-09 is marked costly to reverse).

### Gaps Summary

The phase goal is achieved in substance. I confirmed each part myself rather than from the SUMMARYs:
- **Training:** a user-runnable local fine-tune and fail-closed gate exist.
- **Verifier:** re-running the Rust verifier on the exact deployed file reproduces `deploy_eligible: true` bit-for-bit.
- **Parity:** real-weights parity reproduces 3.841e-6 against spike 025.
- **Real-model stdio:** the stdio server serves the real model with its own identity.
- **Live endpoint:** it answered a cold call with identity == H and a genuine 24.7 s S3 load.

One narrow gap remains: **CR-01**. decide-apr-v1's `deploy_eligibility` claims that verify checks the base against the manifest's declared base, and the code does not. More broadly, nothing binds the manifest's `base`, `variant`, `calibration` and `gate` summary, or its task/tokenizer input hashes, to the embedded sha-bound blobs.

The effect is confined to *crafted* artifacts:
- a file whose weights genuinely pass the re-score could still serve a false `model.base`, or display a fabricated calibration/gate summary, and be printed `deploy_eligible: true`;
- served behaviour is still bound, because the re-score runs on the loaded bytes;
- the D-11 core (artifact_sha256 + recipe_id) is still bound.

**The deployed artifact is not affected.** Each unbound manifest field of 24a44d7e… was compared here against its embedded blob or run-dir source, and all agree, so the fix needs no redeploy. The fix is local hardening: rung-4 cross-checks plus one negative test per field.

**This looks like a deliberate trade-off, not an oversight.** The embedded fields are documented as "identity summary, NEVER an eligibility input". If the owner prefers to defer it, add to this file's frontmatter:

```yaml
overrides:
  - must_have: "Deploy eligibility enforces every decide-apr-v1 verify_check, binding the served identity fields (incl. model.base) to the artifact"
    reason: "Deployed artifact 24a44d7e verified field-by-field consistent; served behaviour and artifact_sha256/recipe_id are bound; manifest-to-blob cross-checks deferred as hardening (CR-01) to a follow-up plan"
    accepted_by: "<owner>"
    accepted_at: "<ISO timestamp>"
```

and record the item in deferred-items.md, correcting decide-apr-v1 `verify_checks[2]` so the contract no longer claims the check.

---

_Verified: 2026-09-28T01:12:08Z_
_Verifier: Claude (gsd-verifier)_
