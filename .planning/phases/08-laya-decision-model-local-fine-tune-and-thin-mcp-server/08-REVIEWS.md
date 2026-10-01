---
phase: 8
reviewers: [codex, gemini]
reviewer_instances: [fable]
failed_reviewers:
  fable: "claude-fable-5-1 refused: 'Fable 5.1 requires usage credits' — no review produced (gsd-tools recorded ok:true/stubbed:false; that is a false green)"
reviewed_at: 2026-09-25T21:53:01Z
plans_reviewed: [08-01-PLAN.md, 08-02-PLAN.md, 08-03-PLAN.md, 08-04-PLAN.md, 08-05-PLAN.md, 08-06-PLAN.md, 08-07-PLAN.md, 08-08-PLAN.md, 08-09-PLAN.md, 08-10-PLAN.md, 08-11-PLAN.md, 08-12-PLAN.md]
models:
  codex: "gpt-6-astra (reasoning=high)"
  gemini: "unknown"
  fable: "claude-fable-5-1 (reasoning=high)"
model_sources:
  codex: "banner"
  gemini: "unknown"
  fable: "pinned"
---

# Cross-AI Plan Review — Phase 8

> **Incomplete review.** `--fable` was named explicitly and its lane did not run (account lacks Fable 5.1 usage credits). Per ADR-2782 D4 an explicitly requested lane that cannot run is an error, so treat this as a two-reviewer (codex + gemini) review, not the requested three.

<!-- gsd:plan-revision-conflicts:begin -->
## Plan-Revision Conflicts
<!-- gsd:plan-revision-conflicts:end -->

## Codex Review

Read-only review completed for **all 12 plans**. I inspected the referenced repository code and the pinned Laya checkout (`4066d5d…`). No files were changed, and no builds, training runs, skills, or workflows were invoked. Findings below distinguish demonstrated mismatches from implementation risks.

## 08-01

**Summary:** Declaring the contracts first is sound, but two details need correction before this plan can establish usable acceptance criteria.

**Strengths**

- The pending-tolerant audit follows a real precedent: the existing target captures `pv`’s status directly and rejects an empty contract list. This avoids a vacuous successful audit. [Makefile:2188](/Users/guy/Development/machine-learning/aprender/Makefile:2188)

**Concerns**

- **HIGH — The probe budget is impossible for the demo task.** The plan requires each probe to contain at most 48 **built-row** tokens. The pinned stance fixture already needs **57 tokens with an empty state**: its separators occur at positions 16 and 55, followed by the final separator. Consequently, neither synthetic probe can satisfy the contract. [08-01:401](/Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-01-PLAN.md:401), [fixture:1](/Users/guy/Development/machine-learning/aprender/.planning/spikes/025-laya-rust-forward-parity/fixtures/laya-en_fixture.json:1)
- **MEDIUM — Pending equation bindings do not exempt nonexistent test references from CI.** The strict test-binding guard starts new contracts at zero unresolved references. Naming future Rust tests in Wave 1 can therefore fail CI despite passing `pv validate` and the proposed audit. [check_contract_test_binding.sh:37](/Users/guy/Development/machine-learning/aprender/scripts/check_contract_test_binding.sh:37), [ci.yml:803](/Users/guy/Development/machine-learning/aprender/.github/workflows/ci.yml:803)

**Suggestions**

- Derive probe limits from the complete task-bound row, including instructions and options, and recalculate the cold-start allowance.
- Specify an honest staged representation for future test bindings; run the existing strict-binding guard during this plan.

**Risk assessment: HIGH.** An impossible contract would block packing or encourage weakening the check later.

## 08-02

**Summary:** The synthetic oracle approach is appropriate, but fixture semantics and one verification command need tightening.

**Strengths**

- Choosing hidden size 32 deliberately exercises Laya’s actual `max(1, d // 64)` branch, which the spike’s hardcoded head layout misses. This is a meaningful architectural test. [common.py:165](/Users/guy/Development/machine-learning/aprender/.planning/spikes/024-laya-vs-kev-few-shot/vendor/laya/laya/common.py:165)

**Concerns**

- **MEDIUM — The ignore check rejects successful negations.** The plan adds exact-path `!` rules, then expects `git check-ignore -v` to exit 1. In a read-only probe, verbose mode printed the matching negation and returned 0; quiet mode correctly returned 1. Thus the proposed verification can fail precisely when the exceptions work. [08-02:249](/Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-02-PLAN.md:249), [08-02:273](/Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-02-PLAN.md:273)
- **MEDIUM — Synthetic recipe exceptions are undeclared.** The fixture is called schema-valid while using zero shots, zero epochs, another seed, and `tiny-synthetic` as its base. The contract otherwise fixes the production recipe and base. Readers need an explicit distinction between structurally loadable synthetic artifacts and deployment-eligible runs. [08-02:241](/Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-02-PLAN.md:241), [08-01:246](/Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-01-PLAN.md:246)

**Suggestions**

- Check each path with `git check-ignore --no-index -q`, separating ignored status from command errors.
- Declare a synthetic fixture variant and explicitly prohibit it in production gate policy.

**Risk assessment: MEDIUM.** The approach is good; the current checks can obstruct or ambiguously validate its fixtures.

## 08-03

**Summary:** The encoder port has a credible numerical foundation, but its advertised reusable configuration surface exceeds what the planned validation safely supports.

**Strengths**

- A prefix-aware expected tensor set follows the existing BERT import/load contract, where symbolic tensor names prevent converter/loader drift. [bert/load.rs:36](/Users/guy/Development/machine-learning/aprender/crates/aprender-core/src/models/bert/load.rs:36)
- Window mutation directly tests the spike’s inclusive attention bounds rather than merely checking an implementation constant. [laya.rs:111](/Users/guy/Development/machine-learning/aprender/.claude/skills/spike-findings-aprender/sources/025-laya-rust-forward-parity/src/laya.rs:111)

**Concerns**

- **HIGH — Structural configuration validation is incomplete.** The plan validates biases and thetas but omits zero dimensions, zero attention interval, head divisibility, even rotary dimensions, layer-list length, and checked dimension products. The copied primitives use chunk sizes and indexed slices that assume these invariants. [08-03:131](/Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-03-PLAN.md:131), [laya.rs:67](/Users/guy/Development/machine-learning/aprender/.claude/skills/spike-findings-aprender/sources/025-laya-rust-forward-parity/src/laya.rs:67)
- **MEDIUM — `norm_eps` is parsed but normalization is fixed to `1e-5`.** A supported-looking configuration can silently produce the wrong encoder output. [08-03:122](/Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-03-PLAN.md:122)
- **MEDIUM — The clippy check can falsely pass.** It accepts any “could not compile” line, ignores the captured status, and succeeds when no `models/modernbert` path appears—even if a dependency failed before this module was checked. [08-03:181](/Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-03-PLAN.md:181)

**Suggestions**

- Validate a documented supported configuration domain before deriving shapes or allocating.
- Honor `norm_eps` or reject unsupported values.
- Require successful clippy execution, or compare explicitly identified baseline diagnostics after proving this crate was checked.

**Risk assessment: HIGH.** Malformed configurations threaten the typed-error promise; accepted configurations can otherwise drift numerically.

## 08-04

**Summary:** The crate boundary and ordered task representation are sensible. The option-budget refusal, however, misstates the proven source behavior.

**Strengths**

- Parsing criteria directly into an ordered sequence addresses an established repository problem: the forecast crate documents how feature unification changed JSON map ordering. [aprender-forecast/Cargo.toml:37](/Users/guy/Development/machine-learning/aprender/crates/aprender-forecast/Cargo.toml:37)

**Concerns**

- **HIGH — Refusing raw option length above `head_max_len` breaks Laya parity.** Laya first shrinks option token sequences to fit its budget. Spike 026 refuses only when the resulting marker count differs from the option count. The plan explicitly tests refusal when tokenized options exceed the budget, rejecting inputs the reference can successfully shorten. [08-04:232](/Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-04-PLAN.md:232), [common.py:127](/Users/guy/Development/machine-learning/aprender/.planning/spikes/024-laya-vs-kev-few-shot/vendor/laya/laya/common.py:127), [spike-026/lib.rs:233](/Users/guy/Development/machine-learning/aprender/.claude/skills/spike-findings-aprender/sources/026-laya-mcp-default-lambda/src/lib.rs:233)
- **MEDIUM — The dual-backing proof permits both executions to use the same backing.** The acceptance criteria explicitly allow standalone execution to report `ON`; that demonstrates two package selections, not both map implementations. [08-04:266](/Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-04-PLAN.md:266)

**Suggestions**

- Define refusal through the post-build marker-preservation invariant, with separate tests for successful shrinking and actual marker loss.
- Require observed `OFF` and `ON` configurations, using an isolated test build if necessary.

**Risk assessment: HIGH.** The proposed refusal changes externally observable inference semantics.

## 08-05

**Summary:** The artifact design is comprehensive, but the load ladder does not yet establish its claimed allocation safety or portable byte determinism.

**Strengths**

- `AprV2ReaderRef` genuinely borrows the artifact bytes, whereas the owning reader copies them. The chosen reader avoids an unnecessary full-model allocation. [reader_impl.rs:301](/Users/guy/Development/machine-learning/aprender/crates/apr-format/src/v2/reader_impl.rs:301), [reader_impl.rs:166](/Users/guy/Development/machine-learning/aprender/crates/apr-format/src/v2/reader_impl.rs:166)

**Concerns**

- **HIGH — The file-size cap does not bound parser allocations.** Before the planned structural rung runs, `AprV2ReaderRef::from_bytes` calls an index parser that executes `Vec::with_capacity(tensor_count as usize)`. A small artifact with a valid header CRC and enormous declared count can request an enormous allocation. [08-05:150](/Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-05-PLAN.md:150), [reader_impl.rs:95](/Users/guy/Development/machine-learning/aprender/crates/apr-format/src/v2/reader_impl.rs:95)
- **MEDIUM — The golden hash incorporates fresh floating-point inference.** Packing computes probe probabilities and serializes their exact bits. The compute library dispatches different kernels on x86 and ARM; tolerance-level parity does not establish bit identity. A single golden hash across developer and CI machines is therefore unproven. [08-05:138](/Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-05-PLAN.md:138), [compute.rs:100](/Users/guy/Development/machine-learning/aprender/crates/aprender-compute/src/blis/compute.rs:100)

**Suggestions**

- Bound index count before allocation, using the actual index byte extent and checked arithmetic; test forged headers without allocating their declared capacity.
- Store canonical oracle probe values in the run inputs, validate them numerically during packing, and serialize those unchanged.

**Risk assessment: HIGH.** The hostile-artifact guarantee currently has a concrete pre-validation allocation hole.

## 08-06

**Summary:** The server surface is appropriately narrow and follows an established transport pattern. Its resource handling needs a clearer scope.

**Strengths**

- One typed tool over a shared verified model and blocking-thread inference directly follows the SetFit server’s implemented pattern. [aprender-mcp-setfit/src/lib.rs:198](/Users/guy/Development/machine-learning/aprender/crates/aprender-mcp-setfit/src/lib.rs:198)

**Concerns**

- **MEDIUM — Tokenization remains on the async handler path.** `precheck` calls `model.prepare` before entering `spawn_blocking`; only inference is offloaded. Unlike the SetFit precedent’s lightweight precheck, this now performs tokenizer work on caller-controlled input. [08-06:123](/Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-06-PLAN.md:123)
- **MEDIUM — Per-request bounds do not establish process-wide concurrency bounds.** The planned handler spawns blocking inference for each accepted request without specifying admission control, queue limits, or cancellation behavior. This matters particularly for concurrent stdio clients using a large CPU model. [08-06:134](/Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-06-PLAN.md:134)

**Suggestions**

- Keep count/byte checks immediate, then perform preparation and inference in a bounded blocking execution path.
- Define and test concurrent-request behavior, including what happens when a caller disconnects.

**Risk assessment: MEDIUM.** Functional coverage is strong; throughput and runtime responsiveness remain underspecified.

## 08-07

**Summary:** The S3 loader and shared HTTP configuration are well-directed. The proposed dependency declaration and local-file load path need correction.

**Strengths**

- Sharing `server_config()` between the bootstrap and loopback test is a real Chronos precedent, ensuring the test exercises the deployed stateless configuration. [chronos-lambda/src/lib.rs:19](/Users/guy/Development/machine-learning/aprender/crates/aprender-mcp-chronos-lambda/src/lib.rs:19)

**Concerns**

- **MEDIUM — A runtime dependency is declared only for development.** The plan puts `aprender-decide` in dev-dependencies, then references `aprender_decide::artifact::ArtifactLimits` from production `src/s3.rs`. A normal library/bootstrap build cannot rely on that declaration. [08-07:121](/Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-07-PLAN.md:121), [08-07:204](/Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-07-PLAN.md:204)
- **MEDIUM — Local loading bypasses the promised bounded-read mechanism.** `resolve_model` is instructed to read bytes before invoking the from-bytes loader. The existing SetFit implementation explicitly documents why a cap applied after `fs::read` is too late. [08-07:130](/Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-07-PLAN.md:130), [setfit/artifact.rs:1911](/Users/guy/Development/machine-learning/aprender/crates/aprender-core/src/setfit/artifact.rs:1911)

**Suggestions**

- Promote the dependency or expose the required limits through the transport library’s public API.
- Route local sources through the bounded reader before hashing and parsing.
- Give ranged downloads an overall deadline in addition to attempt counts.

**Risk assessment: MEDIUM.** These are concrete, repairable integration gaps.

## 08-08

**Summary:** The training/calibration design aligns with the locked decisions, but the written execution sequence contains two direct API/file-layout failures.

**Strengths**

- The proposed optimizer groups, loss, scheduler, gradient clipping, and `_encode_state` reuse match the actual full fine-tune recipe. [ft_laya.py:39](/Users/guy/Development/machine-learning/aprender/.claude/skills/spike-findings-aprender/sources/024-laya-vs-kev-few-shot/tools/ft_laya.py:39)

**Concerns**

- **HIGH — `expected_sha256` has the wrong argument shape.** The plan passes the contract hash directly. The pinned implementation requires a mapping such as `{"model.safetensors": hash}` and explicitly rejects non-dictionaries. [08-08:141](/Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-08-PLAN.md:141), [revisions.py:64](/Users/guy/Development/machine-learning/aprender/.planning/spikes/024-laya-vs-kev-few-shot/vendor/laya/laya/revisions.py:64)
- **HIGH — The first checkpoint reload precedes creation of its agent config.** Step 5 saves weights and copies encoder/tokenizer directories; step 6 immediately loads `Agent(checkpoint)`, writing `rl_agent_config.json` only afterward. `Agent` refuses a directory lacking that file. [08-08:149](/Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-08-PLAN.md:149), [agent.py:323](/Users/guy/Development/machine-learning/aprender/.planning/spikes/024-laya-vs-kev-few-shot/vendor/laya/laya/agent.py:323)
- **MEDIUM — Row-disjoint calibration can still leak duplicate text.** Normalized hashes protect eval/train separation, but calibration splitting does not specify grouping or rejecting duplicate training texts. Copies can enter both fitting and calibration. [08-08:123](/Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-08-PLAN.md:123)

**Suggestions**

- Correct the digest mapping and write the complete checkpoint directory before its first reload.
- Split by normalized-text groups and reject conflicting labels.
- Add a synthetic end-to-end train/save/reload/calibrate test before the expensive demo.

**Risk assessment: HIGH.** Following the sequence literally stops before calibration.

## 08-09

**Summary:** Full-eval probability parity is valuable, but it does not establish the stronger claim that edited quality reports cannot pass.

**Strengths**

- Reloading the packed bytes before re-scoring correctly tests the artifact actually served, and writing only after successful verification prevents publishing a failed intermediate result. [08-09:124](/Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-09-PLAN.md:124)

**Concerns**

- **HIGH — Gate metrics remain unverified assertions.** `check_gate_report` recomputes the Boolean using reported macro-F1/ECE values. `rescore` checks probability parity, not those metrics against eval labels. Editing reported fine-tuned F1, baseline F1, and ECE can therefore preserve every input hash and probability comparison while changing a failed report into a passing one. [08-09:105](/Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-09-PLAN.md:105)
- **HIGH — Deployment inspection does not prove verified packing occurred.** `inspect` runs only header/manifest checks plus the same report policy. The public low-level packer intentionally bypasses gate policy. Together, these paths allow a structurally packed artifact with plausible metrics to receive `gate_policy_ok` without full-eval verification. [08-09:127](/Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-09-PLAN.md:127), [08-05:76](/Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-05-PLAN.md:76)

**Suggestions**

- Recompute fine-tuned metrics from Rust probabilities and eval labels; retain baseline probabilities so its metrics can also be checked.
- Validate complete row coverage, unique indices, probability dimensions, normalization, and metric domains.
- Define deployment eligibility separately from structural loading, binding verified evaluation evidence to the exact artifact.

**Risk assessment: HIGH.** The current design proves numerical agreement but overstates quality-gate integrity.

## 08-10

**Summary:** The wrong-package issue is real and correctly prioritized. The proposed proof and teardown mechanisms need strengthening.

**Strengths**

- The resolver genuinely checks a preferred subdirectory and then takes the first eligible workspace bootstrap package. Addressing this before deployment prevents shipping a healthy but incorrect server. [builder.rs:333](/Users/guy/Development/mcp/sdk/rust-mcp-sdk/cargo-pmcp/src/deployment/builder.rs:333)

**Concerns**

- **HIGH — The dry-run fallback does not exercise the faulty resolver.** Explicitly running `cargo zigbuild -p aprender-mcp-decide-lambda` proves Cargo can build that package; it says nothing about which package cargo-pmcp selects. The plan permits this as its resolution proof. [08-10:174](/Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-10-PLAN.md:174)
- **HIGH — Removing the S3 grant does not stop a warm deployment.** The teardown recipe only deletes the role policy. A process with its model already loaded can continue answering through its cached server, including after a wrong-identity incident. [08-10:237](/Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-10-PLAN.md:237), [chronos-lambda/main.rs:71](/Users/guy/Development/machine-learning/aprender/crates/aprender-mcp-chronos-lambda/src/main.rs:71)

**Suggestions**

- Require an actual resolver test using the intended workspace/configuration; inspect the selected package and artifact path before upload.
- Make incident containment disable endpoint invocation or remove the deployment.
- Avoid relying on “Compiling …” log lines: cached builds may produce none.

**Risk assessment: HIGH.** Neither the fallback proof nor grant-only teardown provides the stated deployment guarantee.

## 08-11

**Summary:** Live identity verification is necessary, but the planned experiment cannot substantiate its cold maximal-request claim.

**Strengths**

- Matching the live response hash to the local artifact is substantially stronger than accepting a health response. The existing Lambda health handler returns success without loading a model. [chronos-lambda/main.rs:104](/Users/guy/Development/machine-learning/aprender/crates/aprender-mcp-chronos-lambda/src/main.rs:104)

**Concerns**

- **HIGH — The measured request is already warm.** `laya-deploy` runs an identity probe before `laya-deploy-verify`. Furthermore, the probe sends `initialize` before `classify`; the copied Chronos handler starts and loads the server on the first non-health request. Measuring only the later classify call excludes loading. [08-10:171](/Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-10-PLAN.md:171), [08-07:141](/Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-07-PLAN.md:141), [chronos-lambda/main.rs:130](/Users/guy/Development/machine-learning/aprender/crates/aprender-mcp-chronos-lambda/src/main.rs:130)
- **HIGH — Eight texts totaling 1,024 tokens are not necessarily the slowest legal request.** Full attention performs work across token pairs. Two 512-token rows have four times the summed squared lengths of eight 128-token rows. Testing one maximal count/token combination does not cover the accepted region. [laya.rs:111](/Users/guy/Development/machine-learning/aprender/.claude/skills/spike-findings-aprender/sources/025-laya-rust-forward-parity/src/laya.rs:111), [08-11:38](/Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-11-PLAN.md:38)

**Suggestions**

- Define whether the guarantee covers cold initialization, cold classify, or the whole client sequence, then measure that exact path with independently correlated cold-start evidence.
- Test concentrated and distributed token budgets, plus maximum-byte tokenizer inputs.
- Check all recorded samples against the chosen criterion; the current verifier checks only the first sample’s duration.

**Risk assessment: HIGH.** A passing experiment could certify a materially different workload from the promised one.

## 08-12

**Summary:** This plan correctly addresses Rust integration-test reachability, but the phase’s automated coverage remains incomplete.

**Strengths**

- The explicit integration-target additions are necessary: CI runs workspace libraries separately and enumerates integration tests in its command chain. [ci.yml:289](/Users/guy/Development/machine-learning/aprender/.github/workflows/ci.yml:289), [ci.yml:471](/Users/guy/Development/machine-learning/aprender/.github/workflows/ci.yml:471)
- The architectural exception follows the existing argument that parity evidence belongs with the single proven implementation. [CLAUDE.md:203](/Users/guy/Development/machine-learning/aprender/CLAUDE.md:203)

**Concerns**

- **HIGH — The Python production path remains outside CI.** The fixture plan explicitly says CI never runs the Python project, while this plan adds only Rust targets. Training, calibration, split isolation, and Python gate regressions can pass every proposed CI addition. [08-02:260](/Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-02-PLAN.md:260), [08-12:186](/Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-12-PLAN.md:186)
- **HIGH — Wiring the audit into tier3 does not make it a CI gate.** CI explicitly documents that it does not run `make tier3`; the proposed workflow edit adds no Phase 8 audit invocation. Definition-site checking also proves existence, not that the bound training self-tests execute. [ci.yml:799](/Users/guy/Development/machine-learning/aprender/.github/workflows/ci.yml:799), [Makefile:2443](/Users/guy/Development/machine-learning/aprender/Makefile:2443)
- **MEDIUM — Automated “CI-equivalent” verification is narrower than its prose.** The nextest command filters to Phase 8 tests, and the examples command excludes `aprender-profile`. Those are useful local checks, but cannot alone establish whole-workspace equivalence. [08-12:203](/Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-12-PLAN.md:203)

**Suggestions**

- Include lightweight Python data/metrics/gate tests and a tiny checkpoint lifecycle in the concrete CI approval diff.
- Invoke the strict Phase 8 audit explicitly in CI.
- Distinguish focused checks, full workspace results, and documented platform exclusions in completion criteria.

**Risk assessment: HIGH.** The current completion path permits essential production behavior to remain untested by CI.

## Cross-plan assessment

**Coverage: 12/12 plans reviewed. Overall risk: HIGH; revise before execution.**

The architecture is coherent, and the source supports reuse of the numerical port, APR reader/writer, and thin-server templates. The main problems are acceptance guarantees that the proposed mechanisms do not establish.

Prioritize these revisions:

1. Correct the impossible probe budget and Laya option-refusal semantics.
2. Repair the trainer’s digest argument and checkpoint reload ordering.
3. Derive gate metrics from evaluated probabilities and labels.
4. Bound parser allocations before invoking the APR index parser.
5. Redesign cold-start verification around the actual initialization sequence and worst legal request shapes.
6. Put Python tests and the Phase 8 audit into CI.

Also add **08-12 as an explicit dependency of 08-11**. The live deployment currently depends only on 08-10, so its dependency graph does not require CI/documentation/audit closure first. [08-11:6](/Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-11-PLAN.md:6)


---

## Gemini Review

# Cross-AI Plan Review: Phase 8 — Laya Decision Model

## Executive Summary

The implementation plan for **Phase 8: Laya Decision Model: Local Fine-Tune and Thin MCP Server** spans **12 execution plans across 9 waves**. It establishes a complete, contracted path extending [`aprender`](file:///Users/guy/Development/machine-learning/aprender/CLAUDE.md) with ModernBERT and a generalized decision framework ([`aprender-decide`](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-04-PLAN.md)).

Overall quality is **exceptionally high**. The plans adhere to repository disciplines (provable contracts, FALSIFY suites, verification without shell pipe `$?` leakage, ASVS L1 security posture, and zero unverified assumptions). The research and foresight around numerical traps (F16 reload scoring, `serde_json` map order divergence, 512 MB Lambda `/tmp` limits, API Gateway 30-second envelope) are thorough and sound.

However, the review identified **two high-severity structural defects** and **several operational risks** that require adjustment before execution begins:
1. **Wave Ordering / Dependency Inversion**: [`08-12-PLAN.md`](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-12-PLAN.md) (Wave 8) closes binding audits and declares all equations implemented *before* [`08-11-PLAN.md`](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-11-PLAN.md) (Wave 9) executes the live deployment and falsifies the live equation.
2. **Desynchronized Validation Strategy**: [`08-VALIDATION.md`](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-VALIDATION.md) is outdated relative to [`08-01-PLAN.md`](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-01-PLAN.md) and [`08-02-PLAN.md`](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-02-PLAN.md).

---

## 1. Requirements Traceability Matrix (D-01 – D-19)

All locked user decisions from [`08-CONTEXT.md`](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-CONTEXT.md) are mapped and addressed across the 12 plans:

| Decision ID | Description | Target Plan(s) | Review Status | Notes |
|:---|:---|:---|:---:|:---|
| **D-01 / D-02** | Local Python training on Laya code via `just laya-train` with pinned `uv` | [08-02](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-02-PLAN.md), [08-08](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-08-PLAN.md) | **Satisfied** | Pin audit checkpoint included; `contracts/apr-cli-commands-v1.yaml` left untouched. |
| **D-03** | Auto device selection (MPS→CUDA→CPU) with recorded parameter device | [08-08](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-08-PLAN.md) | **Satisfied** | Parameter inspection avoids misleading env variables. |
| **D-04** | English-root base declared in manifest; recipe written before scoring | [08-01](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-01-PLAN.md), [08-05](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-05-PLAN.md), [08-08](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-08-PLAN.md) | **Satisfied** | `recipe_id` sha256 computed and timestamp-ordered prior to eval. |
| **D-05** | `task.json` + `train.jsonl` with strict criteria ordering | [08-01](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-01-PLAN.md), [08-04](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-04-PLAN.md) | **Satisfied** | Uses [`serde::de::Visitor::visit_map`](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-04-PLAN.md#L150) on raw bytes, avoiding `serde_json` map divergence. |
| **D-06** | Required `eval.jsonl`; disjoint seeded calibration slice on train | [08-01](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-01-PLAN.md), [08-08](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-08-PLAN.md) | **Satisfied** | Normalized hash set disjointness enforced and tested. |
| **D-07** | Fail-closed gate: ft beats zs on eval; post-calibrated ECE <= ceiling | [08-01](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-01-PLAN.md), [08-08](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-08-PLAN.md), [08-09](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-09-PLAN.md) | **Satisfied** | Dual-enforced in Python ([`gate.py`](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-08-PLAN.md#L95)) and Rust ([`check_gate_report`](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-09-PLAN.md#L30)). |
| **D-08** | Single declared seed ships; `--seeds N` provides variance report | [08-01](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-01-PLAN.md), [08-08](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-08-PLAN.md) | **Satisfied** | Temporary artifacts pruned; only declared seed checkpoint is retained. |
| **D-09** | Task-bound `classify` MCP tool only | [08-06](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-06-PLAN.md) | **Satisfied** | Question & labels derived from artifact `task.json`. |
| **D-10** | Contract-owned list bounds and maximum token budget | [08-01](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-01-PLAN.md), [08-06](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-06-PLAN.md) | **Satisfied** | Sized to Gateway 30s timeout (`max_texts: 8`, `max_tokens: 1024`). |
| **D-11** | Response returns labels, ordered calibrated probabilities, model identity | [08-01](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-01-PLAN.md), [08-06](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-06-PLAN.md), [08-07](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-07-PLAN.md) | **Satisfied** | Arrays preserve criteria order; `artifact_sha256` + `recipe_id` in response. |
| **D-12** | Builder sequence truncation past 512 with `truncated: true` flag | [08-04](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-04-PLAN.md), [08-06](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-06-PLAN.md), [08-09](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-09-PLAN.md) | **Satisfied** | Tested on tiny fixture and 14-row spike fixture. |
| **D-13** | Reusable ModernBERT in `aprender-core::models::modernbert` | [08-03](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-03-PLAN.md), [08-04](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-04-PLAN.md) | **Satisfied** | Clean separation of encoder from Laya decision head. |
| **D-14** | Method-neutral crate [`aprender-decide`](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-04-PLAN.md) with Laya implementation | [08-04](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-04-PLAN.md), [08-05](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-05-PLAN.md) | **Satisfied** | [`DecisionMethod`](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-04-PLAN.md#L139) trait; no bins; publishable layout. |
| **D-15** | [`aprender-mcp-decide`](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-06-PLAN.md) & [`aprender-mcp-decide-lambda`](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-07-PLAN.md) registered in monorepo invariants | [08-06](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-06-PLAN.md), [08-07](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-07-PLAN.md), [08-12](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-12-PLAN.md) | **Satisfied** | Both added to [`deployment_unit_bins`](file:///Users/guy/Development/machine-learning/aprender/crates/aprender-core/tests/monorepo_invariants.rs#L312) with `publish = false`. |
| **D-16** | Third Realizar-First exception row in [`CLAUDE.md`](file:///Users/guy/Development/machine-learning/aprender/CLAUDE.md) | [08-12](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-12-PLAN.md) | **Satisfied** | Matches SetFit (D-09) and Forecast (D-07) precedent; only real paths cited. |
| **D-17** | `.apr` F16 widened at load; torch→.apr→Rust parity <= 1e-5; schema contract | [08-01](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-01-PLAN.md), [08-03](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-03-PLAN.md), [08-05](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-05-PLAN.md), [08-09](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-09-PLAN.md) | **Satisfied** | Full ladder with trybuild private constructor check and full re-score. |
| **D-18** | Live pmcp.run deploy on default Lambda (10,240 MB) from S3; human checkpoint | [08-07](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-07-PLAN.md), [08-10](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-10-PLAN.md), [08-11](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-11-PLAN.md) | **Satisfied** | In-memory 16-way ranged download; live identity check; go/no-go gate. |
| **D-19** | TweetEval stance demo (16 or 64 shots) | [08-01](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-01-PLAN.md), [08-08](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-08-PLAN.md), [08-09](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-09-PLAN.md), [08-11](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-11-PLAN.md) | **Satisfied** | Sourced from `data/tweet-eval-stance/`; 16 shots / class cell configured. |

---

## 2. High-Severity Findings & Sequencing Defects

### Finding 1: Wave Inversion between Plan 08-11 and Plan 08-12

* **Location:** [`08-11-PLAN.md`](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-11-PLAN.md) (Wave 9) and [`08-12-PLAN.md`](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-12-PLAN.md) (Wave 8).
* **Defect:** 
  1. [`08-12-PLAN.md`](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-12-PLAN.md) Task 2 flips every Phase 8 equation in [`contracts/aprender/binding.yaml`](file:///Users/guy/Development/machine-learning/aprender/contracts/aprender/binding.yaml) from `status: pending` to `status: implemented`, and tightens `contract-audit-phase8` to fail if any `BIND-` line (or pending status) remains.
  2. One of those equations is `accepted_region_cold` from [`contracts/decide-tool-boundary-v1.yaml`](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-01-PLAN.md), which specifically contract-binds that the maximal legal request succeeds live on default Lambda under 30 seconds.
  3. However, the live deploy and measurement happen in [`08-11-PLAN.md`](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-11-PLAN.md), which is scheduled in **Wave 9** (after Wave 8).
  4. Furthermore, [`08-11-PLAN.md`](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-11-PLAN.md) Task 1 presents a human decision checkpoint with an option `hold` ("Do not deploy now"). If the human chooses `hold`, the live deployment does not take place. If 08-12 has already landed in Wave 8, [`CLAUDE.md`](file:///Users/guy/Development/machine-learning/aprender/CLAUDE.md) and [`binding.yaml`](file:///Users/guy/Development/machine-learning/aprender/contracts/aprender/binding.yaml) will have already asserted that all phase requirements are implemented and live-verified.
* **Remediation:**
  Swap the scheduling order:
  * **Wave 8**: [`08-10-PLAN.md`](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-10-PLAN.md) (Cargo-pmcp decision and deploy selftest).
  * **Wave 9**: [`08-11-PLAN.md`](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-11-PLAN.md) (Human checkpoint, live deploy, cold-start verification).
  * **Wave 10**: [`08-12-PLAN.md`](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-12-PLAN.md) (Closeout: CI edit checkpoint, [`CLAUDE.md`](file:///Users/guy/Development/machine-learning/aprender/CLAUDE.md) exception row, flip bindings to implemented, tightened `contract-audit-phase8`, final verification).
  If `hold` is selected in 08-11, 08-12 can cleanly record the hold in `deferred-items.md` and keep the live-deployment equation status pending or adjusted per contract policy.

---

### Finding 2: Outdated [`08-VALIDATION.md`](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-VALIDATION.md) Contract Discrepancies

* **Location:** [`08-VALIDATION.md`](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-VALIDATION.md) §Wave 0 Requirements (lines 71–79).
* **Defect:**
  1. Line 75 requires: `"Copy of spike-025 fixture into crates/aprender-decide/tests/fixtures/"`. However, [`08-02-PLAN.md`](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-02-PLAN.md) and [`08-09-PLAN.md`](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-09-PLAN.md) explicitly reversed this decision to prevent shipping licensed TweetEval text to crates.io upon publishing `aprender-decide`.
  2. Line 76 lists only 3 contracts (`decide-apr-v1`, `laya-parity-v1`, `decide-tool-boundary-v1`), omitting `contracts/laya-finetune-gate-v1.yaml`. [`08-01-PLAN.md`](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-01-PLAN.md) explicitly splits the training gate into its own 4th contract.
  3. Line 74 specifies committing `crates/aprender-decide/tests/fixtures/laya_tiny.json` as a single file, whereas [`08-02-PLAN.md`](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-02-PLAN.md) produces a complete directory fixture `crates/aprender-decide/tests/fixtures/laya_tiny/` containing `checkpoint/model.safetensors`, `task.json`, `recipe.json`, and `oracle.json`.
  4. The file specifies `Wave 0 Requirements`, but there is no Plan 00; these tasks were divided into Wave 1 (08-01) and Wave 2 (08-02).
* **Remediation:**
  Update [`08-VALIDATION.md`](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-VALIDATION.md) to reflect the exact 4-contract architecture, the directory fixture structure, the external spike-025 path reference, and mark `wave_0_complete: true` (or align with Waves 1 & 2).

---

## 3. Medium-Severity Architectural & Operational Risks

### Finding 3: `cargo-pmcp` Resolution Trap & Upstream Maintenance Scope

* **Context:** Probed in [`08-RESEARCH.md`](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-RESEARCH.md) and addressed in [`08-10-PLAN.md`](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-10-PLAN.md).
* **Risk:** The recommended option (`patch-cargo-pmcp`) requires modifying an external repo ([`rust-mcp-sdk/cargo-pmcp`](file:///Users/guy/Development/mcp/sdk/rust-mcp-sdk/cargo-pmcp/src/deployment/builder.rs#L333-L370)) and building a local unreleased binary. If the environment changes or another developer runs `just laya-deploy`, the standard published `cargo-pmcp` (0.24.3) will revert to shipping the wrong binary (`aprender-mcp-chronos-lambda`).
* **Evaluation:**
  Inspection of [`builder.rs:337-340`](file:///Users/guy/Development/mcp/sdk/rust-mcp-sdk/cargo-pmcp/src/deployment/builder.rs#L337-L340) in `cargo-pmcp` shows:
  ```rust
  let preferred_package = format!("{}-lambda", server_name);
  let preferred_dir = self.project_root.join(&preferred_package);
  if preferred_dir.exists() && preferred_dir.join("Cargo.toml").exists() {
      return Ok(preferred_dir);
  }
  ```
  If `project_root` is set to `crates` (via `cargo pmcp deploy --manifest-path crates`), and `server_name` is `aprender-mcp-decide`, `preferred_dir` resolves directly to `crates/aprender-mcp-decide-lambda` **without falling back to workspace search**.
  This means Option 2 (`shared-crates-root`) can be executed cleanly without patching upstream `cargo-pmcp`, provided the config swap trap in `just laya-deploy` is robust. The plan should note this as a zero-patch, reproducible alternative.

---

### Finding 4: API Gateway 30s Envelope Margin on First Cold Call

* **Context:** [`08-01-PLAN.md`](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-01-PLAN.md) and [`08-07-PLAN.md`](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-07-PLAN.md) price the `classify_max_total_tokens` (1024 tokens) against API Gateway's 30,000 ms integration timeout.
* **Risk:**
  - S3 ranged download + sha256 + F16 widening: ~11,300–12,000 ms on Graviton2.
  - In-process probes: ~1,000 ms.
  - Forward inference for 1024 tokens @ 11 ms/token: ~11,260 ms.
  - Gateway / TLS proxy overhead: ~500–1,000 ms.
  - **Total estimated time:** `24.1s – 25.3s`.
  - While this is under 30.0s, the margin is ~4.7s. Any transient AWS S3 jitter or memory contention on Graviton2 could trigger an API Gateway 504.
* **Evaluation:**
  [`08-11-PLAN.md`](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-11-PLAN.md) Task 2 correctly mandates an **accepted-region verification**: sending the maximal legal request on the very first cold call. The plan properly specifies that if this times out, the token budget constant must be adjusted in [`contracts/decide-tool-boundary-v1.yaml`](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-01-PLAN.md) via human decision rather than masked by timeouts.

---

### Finding 5: `constants:` Structure in `contracts/laya-finetune-gate-v1.yaml`

* **Context:** [`08-01-PLAN.md`](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-01-PLAN.md) Task 2 specifies nested maps/lists inside `constants:`:
  ```yaml
  constants:
    variance_seeds: [13, 17, 23]
    recipe:
      encoder_lr: 2.5e-5
  ```
* **Risk:** In existing contracts (e.g. [`chronos-bolt-parity-v1.yaml`](file:///Users/guy/Development/machine-learning/aprender/contracts/chronos-bolt-parity-v1.yaml#L73) and [`forecast-tool-boundary-v1.yaml`](file:///Users/guy/Development/machine-learning/aprender/contracts/forecast-tool-boundary-v1.yaml#L64)), `constants:` is a flat mapping of scalar integers or floats read by helper functions like `constant_u64`. Nested arrays or maps under `constants:` may either fail schema validation in `pv validate` or break existing scalar reader helpers.
* **Remediation:** Keep `constants:` strictly flat (e.g. `gate_min_macro_f1_margin`, `gate_max_ece`, `declared_seed`). Place structured definitions (`recipe:`, `base:`, `demo:`, `schemas:`) as top-level sections in the YAML outside `constants:`, which `pyyaml` in [`scripts/laya_train/contract.py`](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-08-PLAN.md#L91) can parse cleanly without violating contract conventions.

---

## 4. Plan-by-Plan Quality & Discipline Review

| Plan | Quality Score | Strengths | Specific Watch-Outs / Recommendations |
|:---:|:---:|:---|:---|
| **08-01** | 9.5 / 10 | Establishes all 4 contracts and blocking audit before code is written. Pending-tolerant audit handles gradual implementation. | Ensure nested sections in `laya-finetune-gate-v1` don't collide with flat `constants:` parsing in `pv`. |
| **08-02** | 9.5 / 10 | Includes blocking package legitimacy check for PyPI. Generates tiny fixtures from Laya's own code on F16 reload. Small size budget (<1 MB). | Ensure `.gitignore` negations pass include-guard scripts ([`check_package_includes.sh`](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-RESEARCH.md#L168)). |
| **08-03** | 10 / 10 | Lifts spike-025 ModernBERT verbatim onto [`trueno::blis::gemm_blis`](file:///Users/guy/Development/machine-learning/aprender/crates/aprender-compute/src/blis/compute.rs#L838). Reuses house [`erfc_precise`](file:///Users/guy/Development/machine-learning/aprender/crates/aprender-core/src/autograd/ops/activation.rs#L205). Window mutation tests verify `\|i-j\| <= 64`. | Gated on `feature = "parallel"` so `cargo check --no-default-features` remains green. |
| **08-04** | 10 / 10 | Implements [`DecisionMethod`](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-04-PLAN.md#L139) seam. `OrderedCriteria` visitor on raw bytes fixes `preserve_order` divergence. Tested with canary across both backings. | Verify `aprender-decide` has no bin targets to preserve [`FALSIFY-MONO-011`](file:///Users/guy/Development/machine-learning/aprender/crates/aprender-core/tests/monorepo_invariants.rs#L312). |
| **08-05** | 10 / 10 | [`AprV2Writer`](file:///Users/guy/Development/machine-learning/aprender/crates/apr-format/src/v2/writer.rs#L18) container packaging. 8-rung load ladder. Trybuild compile-fail test guarantees [`Decider`](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-05-PLAN.md#L163) private constructor. Golden SHA test. | Tested in both `preserve_order` backings to guarantee byte determinism. |
| **08-06** | 9.5 / 10 | Stdio MCP server with single task-bound `classify` tool. Request bounds checked: Count -> Bytes -> Token Budget. Response identity included. | Prevents text logging in error responses (ASVS V7 compliance). |
| **08-07** | 9.5 / 10 | Lambda custom runtime (`bootstrap`) with in-memory 16-way ranged S3 loader. Avoids 512 MB `/tmp` limit entirely. Probe tool included. | Confirmed `aws-sdk-s3` remains pinned to v1.137.0 to prevent MSRV breaks with Rust 1.93.0. |
| **08-08** | 9.5 / 10 | Full fine-tuning pipeline on Laya code. Fail-closed gate (margin >= 0.05, ECE <= 0.10). Evaluated on F16 reload. Seed variance report. | Verified that TweetEval stance selection manifest matches in-tree files. |
| **08-09** | 10 / 10 | Rust packer enforces gate policy and re-scores all 280 eval rows within 1e-5. Full-model parity test vs spike 025 on real weights. | Real model fixture read from spike path rather than copied into crate. |
| **08-10** | 9.0 / 10 | Human decision on cargo-pmcp package resolution bug. Offline deploy selftest (`just laya-deploy-selftest`) proves all refusals without AWS. | Evaluate zero-patch Option 2 as a backup if external repo changes are blocked. |
| **08-11** | 9.0 / 10 | Live deployment to pmcp.run. Human checkpoint for auth and costs. Accepted-region cold start test. Evidence scrubbing. | Reschedule after 08-12 or make 08-12 Wave 10. |
| **08-12** | 9.0 / 10 | Human checkpoint for CI workflow edit. Documents D-16 exception in [`CLAUDE.md`](file:///Users/guy/Development/machine-learning/aprender/CLAUDE.md). Tightens audit to reject pending rows. | Reschedule to Wave 10 to execute after live deployment. |

---

## 5. Security & Threat Modeling Review (STRIDE & ASVS L1)

The plans implement robust defensive mechanisms across all trust boundaries:
- **Tampering (T-08-05-02, T-08-07-01)**: Content-addressed S3 keys (`decide/<server>/<sha256>.apr`) combined with `APRENDER_DECIDE_SHA256` environment pin enforced before parsing prevents object swapping.
- **Spoofing (T-08-10-01)**: Mitigated by post-deploy identity checks: verifying the build log compiled `aprender-mcp-decide-lambda` and invoking live `classify` to confirm `model.artifact_sha256 == H`.
- **Denial of Service (T-08-06-01, T-08-07-02)**: Layered bounds (Count <= 8 -> Pre-tokenization bytes <= 16 KB -> Built-row tokens <= 1024) prevent memory exhaustion and execution timeouts on both Lambda and stdio. In-memory streaming prevents `/tmp` disk fill attacks.
- **Information Disclosure (T-08-06-03, T-08-08-04, T-08-11-04)**: Error messages explicitly redact input text; artifacts contain normalized text hashes instead of raw tweet content; and [`08-DEPLOY-EVIDENCE.json`](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-11-PLAN.md#L40) has automated verification ensuring AWS account IDs and bucket names are scrubbed.

---

## 6. Actionable Recommendations

Before starting phase execution:

1. **Re-sequence Waves 8, 9, and 10**:
   - Move [`08-12-PLAN.md`](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-12-PLAN.md) from Wave 8 to **Wave 10** (`depends_on: ["08-11"]`).
   - Keep [`08-10-PLAN.md`](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-10-PLAN.md) in **Wave 8** and [`08-11-PLAN.md`](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-11-PLAN.md) in **Wave 9**.
   - This ensures [`binding.yaml`](file:///Users/guy/Development/machine-learning/aprender/contracts/aprender/binding.yaml) and [`CLAUDE.md`](file:///Users/guy/Development/machine-learning/aprender/CLAUDE.md) are closed only after live deployment evidence exists (or after an explicit hold is recorded).

2. **Synchronize [`08-VALIDATION.md`](file:///Users/guy/Development/machine-learning/aprender/.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-VALIDATION.md)**:
   - Remove the requirement to copy the spike-025 fixture into `crates/aprender-decide/`.
   - Update the contract list to all 4 files (adding `laya-finetune-gate-v1.yaml`).
   - Reflect the full directory layout of `laya_tiny`.

3. **Verify Flat Structure for `constants:` in `contracts/laya-finetune-gate-v1.yaml`**:
   - Keep scalar thresholds under `constants:`; place recipe dictionaries, base configurations, and schemas at top-level keys to ensure compatibility with `pv` CLI and YAML parsers.

---

## Conclusion

The Phase 8 implementation plans are **ready for execution** once the wave ordering between plans 08-11 and 08-12 is resolved and `08-VALIDATION.md` is updated. The architectural decisions, risk mitigations, and contract-first verification strategies provide a reliable blueprint for delivering the native Laya decision model and thin MCP servers.


---

## Claude Review (fable)

**LANE FAILED — no review.** The `claude` adapter was invoked with `--model claude-fable-5-1 --as fable` and returned only:

> Fable 5.1 requires usage credits. Switch to another model, or manage usage credits at claude.ai/admin-settings/usage, to continue.

gsd-tools stamped it `[reviewed-without-source-citations]` and reported `ok: true, stubbed: false`. It is neither a review nor a successful lane.

---

## Consensus Summary

Two reviewers ran, codex and gemini (both OpenAI/Google, so the review is still cross-vendor). Codex gave per-plan, source-grounded findings with `file:line` evidence throughout. Gemini was organised by theme with a per-plan table, and cited plan files much more than code, so its per-plan "10/10" scores carry less weight than codex's. The orchestrator spot-checked the load-bearing claims against the repo; results are marked **[verified]** / **[unverified]** below.

### Agreed Strengths
- Contract-first sequencing: all four contracts plus a pending-tolerant `contract-audit-phase8` land in Wave 1, before any code (both).
- The spike-025 numerical port, the zero-copy `AprV2ReaderRef` and the SetFit/Chronos thin-server template are the right things to reuse (both).
- Identity carried in every response (`artifact_sha256`) plus a live identity match beats a health check that never loads a model (both).
- The narrow, task-bound `classify` surface and the layered request bounds (both).

### Agreed Concerns
1. **HIGH — 08-11 and 08-12 are not ordered against each other.** Both plans flag this. 08-11 `depends_on: ["08-10"]` and 08-12 `depends_on: ["08-07","08-09"]` **[verified]**. 08-12 flips every binding (including the live `accepted_region_cold` equation) to `implemented` and tightens the audit, while the live deploy that would falsify that equation runs later in 08-11, which can also end in `hold`. Codex's fix is to make 08-12 a dependency of 08-11. Gemini's fix is to move 08-12 to Wave 10 after 08-11. Gemini's version is the one consistent with "bindings only flip once evidence exists".
2. **HIGH — The cold-start / accepted-region proof (08-10/08-11) does not measure what it claims.** Codex: `laya-deploy` already runs an identity probe (`initialize` then `classify`) before `laya-deploy-verify`, so the "cold" measurement is warm. Also, 8×128 tokens is not the worst legal request for attention cost: 2×512 has 4× the summed squared lengths. Gemini independently puts the margin at only ~4.7 s under the 30 s Gateway limit. **[unverified]** by the orchestrator, but the attention-cost argument is arithmetic.

### Single-reviewer HIGHs worth acting on (codex)
- **08-08: step 6 reloads `Agent(<out>/checkpoint)` before `rl_agent_config.json` exists** (step 5 copies only `encoder/` and `tokenizer/`; `agent.py:322` raises `FileNotFoundError`) **[verified]**.
- **08-08: `expected_sha256=<contract sha>` is a bare string, but `revisions.py:64` requires a dict** `{"model.safetensors": sha}` **[verified]**.
- **08-05: the size cap does not bound parser allocation.** `parse_tensor_index_section` calls `Vec::with_capacity(tensor_count as usize)` from the header before any structural rung runs (`crates/apr-format/src/v2/reader_impl.rs:95`) **[verified]**.
- **08-09: the gate can be forged by editing the report.** `check_gate_report` recomputes pass/fail from *reported* F1/ECE. Re-score checks probability parity, not the metrics against labels. Also, `inspect` never proves verified packing happened **[unverified]**.
- **08-04: refusing options whose raw length exceeds `head_max_len` breaks Laya parity.** Laya shrinks options, and spike 026 refuses only on marker loss (`common.py:127`) **[unverified]**.
- **08-01: the 48-token probe budget may be impossible once the full task-bound row is counted.** Codex measured the spike-025 customer-service fixture, not the stance task **[unverified, plausible]**.
- **08-03: config validation is incomplete** (zero dims, head divisibility, even rotary dims, layer-list length), and `norm_eps` is parsed but hardwired to 1e-5 **[unverified]**.
- **08-10: the `cargo zigbuild -p` fallback does not exercise cargo-pmcp's resolver**, and teardown that only revokes the S3 grant leaves a warm instance serving **[unverified]**.
- **08-12: CI never runs the Python training path or the Phase 8 audit.** tier3 is not run by CI (`ci.yml:799`) **[unverified]**.

### Divergent Views
- **Overall readiness:** codex says HIGH risk, revise before execution. Gemini says "ready for execution" once waves are re-sequenced and VALIDATION.md is synced. Codex's verdict is better grounded. Three of its HIGHs (two in 08-08, one in 08-05) were confirmed in minutes, and gemini missed all three.
- **cargo-pmcp (08-10):** gemini says the `shared-crates-root` option is zero-patch because `builder.rs:337-340` returns the preferred `<name>-lambda` dir first. Codex says the fallback proof never exercises the resolver. The two are compatible: pick the zero-patch option *and* prove selection through the real resolver.
- **Gemini-only:** `08-VALIDATION.md` is stale (it lists three contracts rather than four, has a single-file `laya_tiny.json` instead of a directory, copies the spike-025 fixture, and still mentions "Wave 0"). It also flags that nesting `recipe:`/`variance_seeds` under `constants:` has no precedent: 0 existing contracts nest `constants` **[verified: no precedent; pv behaviour not tested]**.
