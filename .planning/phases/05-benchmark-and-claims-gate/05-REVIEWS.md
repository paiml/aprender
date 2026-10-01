---
phase: 5
reviewers: [codex, gemini]
reviewed_at: 2026-08-17T03:07:26Z
plans_reviewed: [05-01-PLAN.md, , 17.2K, 05-02-PLAN.md, , 10.9K, 05-03-PLAN.md, , 14.7K, 05-04-PLAN.md, , 15.5K, 05-05-PLAN.md, , 15.7K, 05-06-PLAN.md, , 11.8K, 05-07-PLAN.md, , 10.8K, 05-08-PLAN.md, , 11.4K, 05-09-PLAN.md, , 16.6K, 05-10-PLAN.md, , 14.3K, 05-11-PLAN.md, , 11.4K, 05-12-PLAN.md, , 8.9K, 05-13-PLAN.md, , 11.1K]
---

# Cross-AI Plan Review — Phase 5: Benchmark and Claims Gate

Both reviewers received an identical 335 KB bundle: PROJECT.md (first 80 lines), the
roadmap Phase 5 section, REQUIREMENTS.md, 05-CONTEXT.md, 05-RESEARCH.md, 05-VALIDATION.md,
and all 13 plans. Claude was skipped (self-CLI: this review ran inside Claude Code).

Invocation notes:
- **Codex** — `codex exec --ignore-user-config --ignore-rules --sandbox read-only
  --skip-git-repo-check`; the anti-recursion guard was verified engaged (stderr ends at the
  echoed prompt with no tool-call lines, so Codex reviewed the plans rather than re-running
  this workflow).
- **Gemini** — `gemini` on this host is a local shim forwarding to `agy` (Antigravity CLI);
  Google retired the standalone `gemini` binary.

---

## Codex Review

## Summary

The plan set has a strong fail-closed architecture, careful provenance discipline, and unusually good negative testing. However, it is not yet safe to execute as written. Several gates validate self-consistency rather than the underlying claim, one numerical acceptance criterion is mathematically contradictory, expensive benchmark runs can begin before the final claims gate exists, and the LoRA reload/model-size path is insufficiently specified. Those gaps could produce a complete, hash-valid 80-row report whose selection-safety or resource comparison is still false.

## Strengths

- The wave-1 F-10 work is appropriately isolated. Plans 05-01 through 05-03 separate measurement, structural refactoring, and the human-approved contract edit without temporarily relaxing `UncalibratedRegime`.

- 05-02’s single `table_for` lookup is a sound design: regime membership and threshold selection cannot disagree, and the induced-widening mutation is a useful falsification test.

- 05-05 and 05-10 correctly define completeness from a predeclared 80-cell expectation set rather than directory contents. This is the right foundation for detecting selective omission.

- Method-tagged evidence in 05-05 avoids null-padded or falsely SetFit-shaped LoRA rows. The explicit exclusion of bootstrap fields also supports deterministic recomputation.

- 05-08 keeps evaluation and metric semantics in library code, reuses the official `F_avg` implementation, and structurally separates validation calibration from test quality.

- The plans consistently require typed refusals, bounded reads, non-vacuous test filters, executed backend identity, and two-sided mutation controls.

- Human checkpoints are placed around the two genuinely consequential external decisions: the production threshold edit in 05-03 and remote 9B execution in 05-11.

## Concerns

- **HIGH: 05-04 Task 2 specifies an impossible Brier consistency test.** The stated multiclass formula is  
  `mean_i Σ_k (p_ik - y_ik)²`. For binary probabilities `[1-p, p]`, this equals `2 × mean_i(p_i-y_i)²`, not the existing scalar binary Brier score. Requiring equality on a `K=2` case will either fail or pressure the implementation into silently changing the declared multiclass normalization. Also, 05-04 Task 1’s “constant-difference” paired case has zero difference variance, yielding an infinite or undefined t statistic; that is problematic for standard JSON fixtures and contradicts the later finite-value expectations unless explicitly represented as an error case.

- **HIGH: 05-05 and 05-10 cannot actually prove the claimed post-test-selection discipline.** A SetFit row contains only a lock hash plus `role` and `rule`; 05-10 checks those row fields, not the canonical lock bytes, committed selection state, or an access ledger. The LoRA protection is even weaker: `no_selection_attestation`, `epochs_completed == epochs_requested`, and `val_split == 0` do not prove that only one candidate was trained or that canonical test data was not accessed before choosing the recorded candidate. A producer can emit internally consistent assertions and pass the gate while the underlying selection-safe claim is false. The doctored negatives only demonstrate detection of malformed attestations, not truthful provenance.

- **HIGH: The execution dependency graph permits costly evidence collection before the claims gate is finished.** Plan 05-11 depends on 05-09 but runs in parallel with 05-10; plan 05-12 depends on 05-03, 05-09, and 05-11, but not 05-10. Consequently, all 80 expensive cells may be generated before `verify_run`, attestation enforcement, binding rows, and the five dishonesty negatives are complete. A defect discovered by 05-10 could invalidate or require regeneration of the entire matrix. This conflicts with the roadmap statement that wave 6 is blocked on wave 5.

- **HIGH: 05-09 Task 3 assumes a production LoRA reload/evaluation path that the research did not establish.** The plan names `run_classify_core`, training result fields, and GPU accessors, but does not identify or first probe a concrete API that reloads the written 9B base-plus-adapter artifact and produces ordered full probability vectors. That is not a small adapter task; it is essential to EVAL-05 and may require substantial training/inference integration. The same task leaves `artifact_bytes` ambiguous: counting only the adapter makes the 9B method appear artificially small, while SetFit records its complete standalone APR. A defensible model-size comparison must record base bytes, adapter bytes, and total deployable bytes separately.

- **MEDIUM: 05-01/05-03 overstate what six measured calibration cells establish.** Measuring `{s8,s64} × {13,31,53}` can support an engineering margin, but “the seven unmeasured seeds vary only the Philox stream” is not evidence that their minimum real update remains above the frozen threshold. The 40-cell `table_for` test proves only syntactic regime coverage, not that all 40 runs pass the evidence thresholds. Therefore 05-03’s must-have that no benchmark cell can encounter `UncalibratedRegime` or threshold failure mid-matrix is stronger than its evidence.

- **MEDIUM: 05-09’s “cold latency” and peak-RSS measurements are not cleanly isolated.** The same `bench run` process trains, writes, reloads, evaluates, and measures. Its first post-load inference benefits from process and filesystem state created during training, so it is not operationally cold. `VmHWM` is cumulative for the whole process and will largely measure training peak, not inference peak; on macOS, a 10 Hz sampled maximum may miss short peaks. These can be valid metrics if named precisely, but the current labels invite stronger comparisons than the protocol supports.

- **MEDIUM: 05-05’s contract parity check is too weak for the claim “contract-derived expectation.”** Testing that method, shot, and seed literals “appear verbatim” in an `include_str!` contract can pass if values occur in comments, if the contract contains extra values, or if Rust constructs a different Cartesian product. The actual expectation remains duplicated in Rust and YAML. That creates a route for the completeness gate to stay green while disagreeing with the normative contract.

## Suggestions

- In 05-04, choose and contract one Brier normalization explicitly. If retaining the stated unnormalized multiclass definition, assert `multiclass_binary_case == 2 × binary_brier`; alternatively divide the class sum by two for a binary-compatible normalized variant and name it accordingly. Make zero-variance paired cases typed degenerate cases, or define an explicit finite policy rather than serializing `NaN`/infinity.

- Strengthen the claims provenance model before 05-10. Commit and verify canonical SetFit lock records, including the selection/config hashes they bind. For LoRA, emit an append-only candidate/execution ledger committed before test evaluation, or add a benchmark-specific credential/state machine that makes `train exactly one candidate → seal selection → test` enforceable. A Boolean attestation should be supplementary evidence, not the gate.

- Make 05-11 depend on 05-10, or at minimum run one cheap two-method pilot cell through the completed `verify_run` gate before authorizing the 40 LoRA cells. Add 05-10 explicitly to 05-12’s dependencies.

- Split the LoRA work: add a preflight plan or 05-09 Task 0 that proves one tiny adapter can be saved, reloaded in a fresh process, and evaluated into the required probability-vector shape. Do not schedule remote 9B cells until that exact route passes.

- Replace `artifact_bytes` with explicit `base_model_bytes`, `adapter_bytes`, and `deployable_total_bytes` for LoRA; SetFit should record APR bytes as both artifact and deployable total. Report each component without implying adapter-only size is comparable to a standalone APR.

- Treat the six calibration cells as derivation data and reserve several unmeasured contracted cells as prospective validation. Run those after freezing ε but before approval. If compute prevents this, weaken 05-03’s claim to “contract membership covers all cells; empirical threshold success remains a benchmark-run invariant” and make the driver stop immediately on the first evidence failure.

- Measure cold inference in a fresh child process that only loads the artifact and performs one prediction. Record training peak and inference peak separately. If `sysinfo` sampling remains supported, record the actual sampling interval and clearly label it as a lower-bound sampled peak.

- Parse the claims contract into an exact typed expectation set in tests, or generate the Rust constants from a validated contract artifact. Compare exact sets and cardinalities, including rejection of extras and duplicates; substring presence is not sufficient.

## Risk Assessment

**HIGH.** The design has good defensive structure, but three issues directly threaten the phase’s core claim: selection safety is currently self-attested rather than proven, expensive execution is ordered before the definitive gate, and the LoRA production reload/model-size path is assumed rather than demonstrated. The Brier contradiction is also a deterministic implementation blocker. Correcting these issues would reduce the remaining risk substantially; without those corrections, a green report would demonstrate artifact consistency more strongly than benchmark validity.
---

## Gemini Review

# Staff Engineering Assessment: Phase 5 Delivery Plan Set (Benchmark and Claims Gate)

## Executive Summary

Phase 5 transitions Aprender from in-crate unit verification to public, reproducible benchmark claims comparing pure-Rust SetFit with a 9B LoRA baseline across 80 evaluation cells (40 SetFit CPU + 40 LoRA GPU). The plan set demonstrates impressive rigor regarding the F-10 calibration gate, contract immutability, and rejection of unverifiable statistical claims.

However, an adversarial engineering review reveals **five high-severity architectural, statistical, and operational failure modes** across plans `05-01` through `05-13`. Left unaddressed, these flaws will result in:
1. Computation of an invalid headline metric (`F_avg` calculated over the wrong class pair due to an off-by-one label mapping),
2. Silent validation-split leakage and model selection inside the 9B LoRA comparison path,
3. A two-phase manifest deadlock where the 80-cell run manifest cannot be pre-declared or incrementally populated without failing content-hash checks,
4. Non-portable and inconsistent resource metrics due to cross-platform OS differences (`sysinfo` polling on macOS vs. `/proc/self/status` VmHWM on Linux), and
5. A brittle contract coupling that bakes a misleading `minilm-slice-` prefix into production contract regimes.

---

## Critical Findings

### Finding 1: Off-By-One Class Indexing in TweetEval `F_avg` Calculation
* **Citations:** Plan `05-08` (Task 2), Plan `05-10` (Task 1), Plan `05-04` (Task 1), Research Section "Don't Hand-Roll".
* **Severity:** Critical (Results in invalid headline benchmark numbers).
* **Analysis:**
  TweetEval Abortion Stance contains three ordered classes: `0 = against`, `1 = none/neutral`, and `2 = favor`. The contracted benchmark metric (EVAL-01, Project Context) is the official TweetEval `F_avg = (F1_against + F1_favor) / 2`.
  
  In `05-RESEARCH.md` (Don't Hand-Roll table) and the metric assembly specification for `05-08.T2`, the pipeline is instructed to invoke `f1_avg_for_classes(&[1, 2])`. Calling `f1_avg_for_classes(&[1, 2])` computes `(F1_none + F1_favor) / 2`, completely dropping the `against` class (index 0) and replacing it with the `none` neutral class (index 1).
  
  Because `MultiClassMetrics::f1_average_for_classes` takes raw slice indices without asserting class name bindings, this off-by-one index selection will execute without error, emit plausible-looking F1 values, and silently corrupt the primary claim of the entire milestone.
* **Remediation:**
  In `05-08.T2` and `05-10.T1`, bind `f1_avg_for_classes` strictly via an explicit label lookup against `ordered_labels` (e.g., resolving `"against"` -> `idx_against` and `"favor"` -> `idx_favor`, yielding indices `&[0, 2]`). Add a negative test in `05-08.T2` verifying that an evaluator targeting `&[1, 2]` produces a distinct, non-matching metric score against the official TweetEval benchmark fixture.

---

### Finding 2: Uncontrolled Model Selection and Contamination in LoRA Pipeline
* **Citations:** Plan `05-06` (Tasks 1 & 2), Plan `05-11` (Task 2), Decision D-10.
* **Severity:** High (Breaches EVAL-02 and TRN-07 selection-safety contracts).
* **Analysis:**
  While SetFit's path strictly enforces canonical test split isolation via `SelectionLock -> mint_test_token -> CanonicalTestAccess::grant`, the LoRA baseline executed through `apr finetune --task classify` relies on `entrenar::finetune::classify_pipeline::ClassifyTrainer`.
  
  As identified in Research (F-R12 / Pitfall 3), `ClassifyTrainer` defaults to:
  ```rust
  val_split: 0.2, seed: 42, early_stopping_patience: 10
  ```
  This introduces two critical integrity failures:
  1. **Random Internal Split:** A 20% random validation split taken from the few-shot training pool (e.g., 8 examples per class -> 16 total examples -> ~3 examples withheld) destabilizes few-shot training and causes the model to train on fewer examples than the contracted shot count.
  2. **Unattested Early Stopping:** Saving checkpoints based on the minimum validation loss of an ad-hoc random split constitutes uncontracted model selection.
  
  Plan `05-06.T1` proposes allowing CLI overrides for `val_split` and `seed`, but assumption A6 ("ClassifyTrainer tolerates `val_split: 0.0` without division by zero or panic") remains untested before Wave 2. If `ClassifyTrainer` divides by `val_dataset.len()` during metric calculation, setting `val_split = 0.0` will panic. Furthermore, the plan does not define how test predictions are extracted from LoRA: if `ClassifyTrainer` evaluates test data in-process without an explicit post-training evaluation command, it bypasses the decoupled verification gate required by EVAL-04.
* **Remediation:**
  In `05-06.T1` and `05-06.T2`, explicitly refactor `ClassifyTrainer` to support an explicit `ValidationPolicy::None` (or `val_split: 0.0`), disabling early stopping and running for the exact contracted number of epochs. Decouple LoRA test evaluation so that `apr finetune` produces a trained checkpoint, and evaluation occurs through an isolated evaluation entrypoint (`run_classify_eval`) that consumes the manifest-attested test dataset.

---

### Finding 3: Manifest Generation Deadlock and Content Hash Circularity
* **Citations:** Plan `05-05` (Task 2), Plan `05-09` (Task 2), Plan `05-10` (Task 1), Decision D-14.
* **Severity:** High (Blocks autonomous execution and cell resumption).
* **Analysis:**
  Decision D-14 and Plan `05-05.T2` state that `manifest.json` defines completeness by listing all 80 expected cells along with each row's SHA-256 content hash. `SelectionManifest::from_bytes` pattern is adopted ("verifies the digest BEFORE returning").
  
  This creates an impossible ordering dependency:
  * To run cells with incremental resumption, `scripts/run_bench_cells.sh` (`05-09.T3`) needs to know if a cell is completed by checking the manifest.
  * However, the row content hash cannot exist in `manifest.json` until *after* the cell has finished training, evaluated test predictions, recorded resource metrics, and written its `.json` row file.
  * If `manifest.json` is generated only at the very end of the 80 runs, any crash during cell 35 leaves no verifiable manifest, and the directory cannot be validated against omission.
  * If `manifest.json` is updated incrementally after each cell, its top-level SHA-256 digest changes after every run, making concurrent execution of SetFit (CPU) and LoRA (GPU) write-conflicting.
* **Remediation:**
  Split the manifest structure in `05-05.T2` into two formal artifacts:
  1. `BenchmarkMatrixSpec` (`matrix.json`): A deterministic, pre-declared specification containing the 80 expected `(method, shot, seed, sampled_id_hash)` tuples, generated before any cell runs.
  2. `BenchmarkRunManifest` (`run_manifest.json`): The execution record containing cell row hashes, generated by aggregating the row directory once all 80 cells are complete.
  
  In `05-09.T3`, have the cell runner check for the existence and self-hash validity of individual `rows/{method}-{shot}-{seed}.json` files against `matrix.json`, enabling robust resumption without circular manifest dependencies.

---

### Finding 4: Hardware Divergence and Incomparable Resource Accounting
* **Citations:** Plan `05-09` (Task 1), Plan `05-11` (Task 2), Plan `05-12` (Task 1), Research Pattern 6.
* **Severity:** Medium-High (Threatens credibility of EVAL-05 claims).
* **Analysis:**
  SetFit cells (`05-12`) are slated to run on the local macOS developer machine (`Darwin arm64`), while LoRA cells (`05-11`) run remotely on `lambda-vector` (`Linux x86_64`).
  
  According to Pattern 6 and `05-09.T1`:
  * Linux peak memory will be read via `/proc/self/status` `VmHWM` (an exact OS kernel high-water mark).
  * macOS peak memory cannot use `/proc` and must rely on `sysinfo` process polling at discrete intervals (a sampled peak).
  
  This creates an asymmetric measurement: SetFit's transient allocation spikes on macOS may be missed due to polling interval aliasing, while LoRA's `VmHWM` on Linux will capture exact kernel peaks. Comparing SetFit memory to LoRA memory under these conditions violates EVAL-05 ("consistently bounded resource measurements").
  
  Additionally, remote dispatch in `05-11` assumes `lambda-vector` can pull and build branch `gsd/phase-2-contract-gate`. Because no PR is open and local commits are not pushed upstream (02-01 policy), `lambda-vector` cannot fetch the uncommitted local code unless the script uses direct source syncing (`rsync` of the working tree).
* **Remediation:**
  1. Standardize peak memory collection in `05-09.T1` using `mach_task_basic_info` (via safe wrapper or explicit crate abstraction) on macOS to read `resident_size_max`, matching Linux's true `VmHWM` semantics.
  2. In `05-11.T2`, ensure `dispatch-*.sh` uses workspace file synchronization (`rsync` excluding target/cache) rather than `git clone/pull`, ensuring unpushed local changes on `gsd/phase-2-contract-gate` execute seamlessly on `lambda-vector`.

---

### Finding 5: Permanent Semantic Debt in Architecture Fingerprint Grammar
* **Citations:** Plan `05-01` (Task 1 & 3), Plan `05-02` (Task 1), Plan `05-03` (Task 1), Research F-R2.
* **Severity:** Medium (Architectural integrity risk in contracts).
* **Analysis:**
  As revealed in F-R2, `BertSentenceEncoder::architecture_fingerprint()` hardcodes the string `minilm-slice-` regardless of whether the model is a fixture slice or the full 6-layer 30,522-vocab model:
  ```rust
  // encoder.rs:303
  format!("minilm-slice-h{}-l{}-a{}-i{}-v{}", hidden, layers, heads, intermediate, vocab)
  ```
  The full production encoder renders as:
  `minilm-slice-h384-l6-a12-i1536-v30522@1110a243`
  
  Plans `05-01.T3` and `05-03.T1` correctly notice this trap and instruct the executor to copy the rendered string verbatim to avoid breaking the test. However, writing `minilm-slice-` for a full 384-dim production model into the normative `contracts/setfit-train-lifecycle-v1.yaml` contract codifies an architectural misnomer into the long-term contract specification. If Phase 1/2 encoder crates later fix `architecture_fingerprint()` to return `minilm-full-` when `vocab == 30522`, all Phase 5 contracts and gates will fail closed.
* **Remediation:**
  In `05-02.T1`, enhance `RegimeCoordinates::parse` and `RegimeCoordinates::covers` to normalize architecture prefixes (treating `minilm-slice-` and `minilm-full-` as compatible architecture family identifiers when dimensions match), or explicitly document the prefix compatibility rule in `contracts/setfit-train-lifecycle-v1.yaml` before adding the production regime in `05-03`.

---

## Secondary Observations & Refinements

1. **Multiclass Brier Score Scaling (05-04.T2):**
   The standard multiclass Brier score is $BS = \frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K (p_{ik} - y_{ik})^2$. For $K=3$ classes, maximum misclassification produces $BS = 2.0$. The plan must ensure that `05-04.T2` does not apply binary scaling bounds ($[0, 1]$) or divide by $K$ unless explicitly mandated by the contract equation.
2. **Top-Label ECE Bin Clamping (05-04.T2):**
   When computing top-label ECE ($conf_i = \max_k p_{ik}$), probabilities with $conf = 1.0$ mapped via $\lfloor conf \times n\_bins \rfloor$ evaluate to index 10 for 10 bins. Ensure `05-04.T2` explicitly clamps bin indices to $\min(n\_bins - 1, \dots)$ to prevent out-of-bounds panics on saturated predictions.
3. **Execution Wall-Clock on SetFit 40 Cells (05-12.T1):**
   40 sequential SetFit training passes on CPU (especially at $s=64$ with $R=20$ pair iterations) will require 3 to 4 hours of CPU compute. `05-12.T1` should explicitly specify batching or multi-process queueing across CPU cores (e.g., running 4 independent single-threaded cell processes concurrently) to keep execution time under 45 minutes without violating deterministic intra-process Rayon pool constraints.

---

## Plan-by-Plan Action Matrix

| Plan ID | Primary Area | Required Modification |
|---|---|---|
| **05-01** | Calibration Measurement | Document in `05-01.T1` that production regime prefix `minilm-slice-` is accepted under explicit contract aliasing rules. |
| **05-02** | Thresholds Structure | Update `RegimeCoordinates` to handle architecture prefix normalization cleanly. |
| **05-03** | Synchronized Contract Edit | Ensure `pv diff` documentation records the prefix rationale alongside measured boundary windows. |
| **05-04** | Numerics Substrate | Fix top-label ECE bin-10 clamping; pin 3-class Brier range $[0, 2]$; freeze $t_{0.975, 9} = 2.2621571628$ via scipy fixture. |
| **05-05** | Claims Contract & Row Types | Decouple `matrix.json` (expectation matrix) from `run_manifest.json` (completed hash ledger). |
| **05-06** | LoRA Finetune Manifest Door | Add `ValidationPolicy::None` to `ClassifyTrainer`; probe $val\_split = 0.0$ safety; isolate test prediction path. |
| **05-07** | Production Chain Proof | Ensure production ladder test asserts non-empty, valid `model.apr` generation on real weights. |
| **05-08** | Evaluator Door & Metrics | **Fix class index mapping in `f1_avg_for_classes` to `&[0, 2]`** (against=0, favor=2); bind via label names. |
| **05-09** | Bench Run CLI | Use native OS peak memory APIs (`mach_task_basic_info` on macOS, `VmHWM` on Linux); support cell resume via `matrix.json`. |
| **05-10** | Bench Gate & Report | Verify 5 doctored negative fixtures against the decoupled matrix specification. |
| **05-11** | LoRA GPU Dispatch | Use workspace `rsync` for code transport to `lambda-vector` to bypass unpushed git branch limits. |
| **05-12** | SetFit CPU 40 Cells | Implement concurrent cell process dispatch to keep total execution time under 45 minutes. |
| **05-13** | Report & Phase Close | Ensure report aggregation rigorously tests exact bit-level recomputation across both methods. |

---

## Conclusion

The delivery plan set is exceptionally well-structured and displays high engineering maturity. Implementing the remediations above—specifically correcting the `F_avg` class index binding, decoupling the matrix specification from execution hashes, and neutralizing LoRA's internal validation split—will ensure the Phase 5 benchmark suite delivers unassailable, publication-grade empirical claims.

---

## Consensus Summary

Two independent reviewers (Codex `gpt-5.1-codex-max` default, Gemini via `agy`) read the same
335 KB bundle: PROJECT.md, roadmap §5, REQUIREMENTS.md, 05-CONTEXT.md, 05-RESEARCH.md,
05-VALIDATION.md and all 13 plans. Both rate the plan set structurally strong and both land on
HIGH overall risk, but they are **not equally reliable**: the orchestrator spot-checked the
top-severity findings against the plan text, and several of Gemini's findings assert gaps the
plans already close. Findings below are annotated with that verification. Treat the
`[VERIFIED]` items as replanning input; treat `[REFUTED]` items as noise that should NOT drive
plan edits.

### Agreed Strengths

- **Predeclared 80-cell expectation set** (05-05, 05-10) — completeness defined by contract +
  manifest, never a directory listing. Both reviewers name this the right foundation for
  detecting selective omission.
- **Wave-1 F-10 isolation** (05-01 → 05-03) — measurement, structural refactor, and the
  human-approved contract edit are separated, with no inline `UncalibratedRegime` relaxation.
- **Method-tagged evidence** (05-05 `MethodEvidence` serde-tagged enum) — a missing block is
  unrepresentable rather than null-padded.
- **Human checkpoints placed on the two consequential external decisions** — the production
  threshold edit (05-03, D-04) and remote 9B execution (05-11, A1).
- **Negative/mutation testing discipline** — two-sided ordering tests, induced-widening
  mutations, five in-band doctored negatives, non-vacuous test-filter assertions.

### Agreed Concerns

1. **[VERIFIED — HIGH] The LoRA arm is the weakest link, and its production reload path is
   assumed rather than demonstrated.** (05-06, 05-09 T3, 05-11)
   Codex: no concrete API is identified that reloads the written 9B base+adapter artifact in a
   fresh process and yields ordered full probability vectors — yet EVAL-05 and every LoRA
   `QualityBlock` depend on exactly that. Gemini reaches the same crate from a different angle
   (`ClassifyTrainer`'s internal split / early stopping / in-process test evaluation).
   *Partially pre-addressed:* 05-05 already contracts `val_split: 0.0`,
   `early_stopping_disabled: true`, `epochs_completed == epochs_requested`, and 05-06 already
   plans explicit seed/val_split/early-stop control plus the A6 probe. What neither plan proves
   is the **reload → probability-vector** route. Codex's fix — a cheap preflight (05-09 Task 0
   or a new plan) that saves a tiny adapter, reloads it in a fresh process, and evaluates it
   into the required shape, *before* authorizing 40 remote 9B cells — is the highest-value
   change in either review.

2. **[VERIFIED — HIGH] Resource metrics are labelled more strongly than the protocol supports.**
   (05-09 T1, 05-11, 05-12)
   Both reviewers independently flag that SetFit (macOS, `sysinfo` sampled at 10 Hz) and LoRA
   (Linux, `/proc/self/status` VmHWM) measure peak memory by different mechanisms, so an
   EVAL-05 SetFit-vs-LoRA memory comparison mixes a sampled lower bound with an exact kernel
   high-water mark. Codex adds two sharper points the plan does not currently handle: VmHWM is
   **process-cumulative**, so in a `bench run` process that trains *then* reloads *then*
   infers it largely reports the **training** peak, not the inference peak; and "cold" latency
   measured in that same process is not operationally cold. 05-09 T1 does record a mechanism
   string (`vm_hwm` / `sysinfo_sampled_10hz`), which is honest but does not make the two
   numbers comparable. Open Q6 in 05-11 T1 (run SetFit cells on macOS or on lambda-vector) is
   the natural place to resolve the platform half.

3. **[VERIFIED — HIGH, single reviewer but mathematically checkable] 05-04 T2's Brier
   consistency criterion is impossible as written.** (Codex only)
   The plan pins `BS = (1/N) Σ_i Σ_k (p_ik − y_ik)²` *and* requires
   `brier_score_multiclass` to equal the existing binary `brier_score` on a K=2 one-hot case.
   For binary `[1−p, p]` the class sum is `2(p−y)²`, so the multiclass value is exactly **2×**
   the binary one. The criterion will fail, or will pressure the implementer into silently
   renormalizing the declared formula. Note the Task-1 fixture identity
   (`Σ_k sklearn.brier_score_loss(y==k, p[:,k])`) is consistent with the stated formula — it is
   only the K=2 *equality* acceptance criterion that is wrong. Fix: assert
   `multiclass_K2 == 2 × binary`, or adopt a normalized variant and rename it. Codex also flags
   that a "constant-difference" paired-t case has zero difference variance → undefined/infinite
   t, which needs to be a typed degenerate case rather than a JSON `NaN`.

4. **[VERIFIED — HIGH] Expensive evidence can be collected before the gate that judges it
   exists.** (05-10 vs 05-11 vs 05-12)
   Confirmed against the plan frontmatter: 05-10 and 05-11 are **both wave 5** (05-10
   `depends_on: ["05-09"]`, 05-11 `depends_on: ["05-06","05-09"]`), so the 40 GPU cells can be
   generated in parallel with — or before — `bench_gate`'s fail-closed verifier, attestation
   enforcement and five doctored negatives exist. A defect found by 05-10 would invalidate the
   matrix. Separately, 05-12's `depends_on: ["05-03","05-09","05-11"]` omits 05-10; strict
   wave-order execution masks this, but the declared graph does not. Fix: add 05-10 to 05-11's
   and 05-12's `depends_on`, or gate the 40 cells behind one cheap two-method pilot cell driven
   through the completed `verify_run`.

5. **[VERIFIED — MEDIUM] Selection safety is currently self-attested, not proven.** (05-05,
   05-10)
   A SetFit row carries a lock hash plus `role`/`rule`; 05-10 checks those *row fields*, not
   canonical lock bytes or committed selection state. LoRA's protection is a boolean
   `no_selection_attestation` plus `epochs_completed == epochs_requested` — which cannot show
   that exactly one candidate was trained or that canonical test data was untouched before the
   recorded candidate was chosen. The five doctored negatives prove detection of *malformed*
   attestations, not truthful provenance. This is the sharpest version of "a gate could pass
   while the claim is false," which is the phase's own stated bar.

6. **[VERIFIED — MEDIUM] 05-05's `include_str!` contract-parity test is substring-based.**
   Method/shot/seed literals "appearing verbatim" in the YAML passes if the values sit in a
   comment, if the contract holds extras, or if Rust builds a different Cartesian product. The
   expectation set stays duplicated in Rust and YAML. Fix: parse the contract into a typed set
   in the test and compare sets and cardinality, rejecting extras and duplicates.

7. **[PARTIALLY VERIFIED — MEDIUM] Six measured calibration cells are thin support for a
   40-cell no-`UncalibratedRegime` guarantee.** (05-01, 05-03)
   Codex: `{s8,s64} × {13,31,53}` plus "the other seven seeds only vary the Philox stream" is
   an engineering margin, not evidence that every unmeasured cell clears the frozen threshold;
   the 40-cell `table_for` test proves *syntactic* regime coverage only. Either reserve a few
   contracted-but-unmeasured cells as prospective validation after freezing ε, or weaken
   05-03's must-have to "contract membership covers all 40 cells; empirical threshold success
   remains a benchmark-run invariant" and make the driver halt on the first evidence failure.

8. **[VERIFIED — MEDIUM] `artifact_bytes` is not comparable across methods.** (05-09, 05-11)
   Counting only the LoRA adapter makes the 9B method look artificially tiny against SetFit's
   complete standalone APR. Record `base_model_bytes`, `adapter_bytes` and
   `deployable_total_bytes` separately; SetFit records its APR bytes as both artifact and
   deployable total.

9. **[UNVERIFIED — LOW/MEDIUM] Wall-clock feasibility of the 40 SetFit CPU cells.** (05-12)
   Gemini estimates 3–4 h sequential at s=64 and suggests concurrent single-threaded cell
   processes. Not checked by the orchestrator, and it interacts with 05-12's compute pre-auth
   gate and with deterministic Rayon-pool constraints — worth a timing estimate before
   committing to the host chosen in Open Q6.

### Divergent Views

- **`F_avg` class indices — Gemini HIGH vs Codex strength. [REFUTED: Gemini is wrong.]**
  Gemini's top finding claims `f1_average_for_classes(&[1,2])` is an off-by-one that drops
  `against` and should be `&[0,2]`, on the assumption that the label order is
  `0=against, 1=none, 2=favor`. The plans state the ordering explicitly and consistently —
  05-08:19, 05-08:120-121, 05-05:88 all declare `ordered_labels = ["none","against","favor"]`,
  which matches HuggingFace `tweet_eval`'s stance `ClassLabel(names=['none','against','favor'])`.
  Under that ordering `[1,2]` **is** `(F1_against + F1_favor)/2`. Codex, reading the same text,
  cited 05-08's reuse of the official `F_avg` as a strength. **Do not apply Gemini's
  remediation — it would invert the headline metric.** The residual value in the finding is
  real though: nothing in the plans pins `ordered_labels` to the dataset revision's own
  `ClassLabel` order. A fixture test asserting the order against the pinned dataset revision
  would convert an assumption into evidence, and is worth adding to 05-08.

- **Run-manifest "deadlock" — Gemini HIGH. [REFUTED.]** Gemini argues the manifest cannot be
  pre-declared because row content hashes only exist after cells run, and proposes splitting
  `matrix.json` from `run_manifest.json`. 05-05's design already does this within one type:
  `expectation()` is derived **in code** from the methods/shots/seeds constants, each entry
  carries `status pending|complete` with `row_sha256` only when complete, and `record(cell,
  hash)` is idempotent on identical re-record specifically to support resume. The
  concurrent-write half is handled by the wave structure — the roadmap places 05-12 in wave 6
  behind 05-11 in wave 5 *because* they share the run-manifest file. Codex, by contrast, listed
  this same predeclared expectation set as a strength.

- **Remote code transport — Gemini MEDIUM. [REFUTED.]** Gemini asserts lambda-vector cannot
  fetch unpushed local commits and recommends `rsync` of the working tree. 05-11 T2 already
  step (1) pushes the current branch and records the SHA, then checks out that same SHA
  remotely; `rsync` is already the *preferred* transport for the selection manifests
  specifically, so that both methods consume byte-identical pairing keys.

- **ECE bin clamping and Brier range — Gemini secondary items. [REFUTED.]** 05-04 T2 already
  specifies the binary pair's exact bin indexing,
  `((conf * n_bins as f32) as usize).min(n_bins - 1)`, so `conf == 1.0` cannot index out of
  bounds; and it pins the Brier formula with fixtures rather than asserting a `[0,1]` bound.
  Codex's Brier finding (item 3 above) is a *different*, real defect in the same task — the
  K=2 equality criterion — and should not be confused with Gemini's.

- **Architecture-fingerprint prefix — Gemini MEDIUM, Codex silent. [PLAUSIBLE, unverified.]**
  Writing `minilm-slice-h384-l6-a12-i1536-v30522@…` into the normative
  `setfit-train-lifecycle-v1.yaml` for a full production encoder codifies a misnomer, and a
  future correction to `architecture_fingerprint()` would fail every Phase 5 gate closed. The
  plans already knowingly copy the rendered string verbatim; the open question is whether to
  document the prefix-compatibility rule in the contract at 05-03, which costs little.

### Reviewer Reliability Note

Of Gemini's five "critical findings" and three secondary items, four were refuted against the
plan text and one (the `F_avg` index inversion) would have introduced the exact defect it
claimed to prevent. Its genuine contributions are the cross-platform peak-memory asymmetry
(shared with Codex), the fingerprint-prefix debt, and the wall-clock question. Codex's findings
survived spot-checking, including one — the 05-04 Brier factor-of-2 — that is provable by
inspection, and one — the 05-10/05-11/05-12 dependency ordering — that was confirmed directly
against the plans' `depends_on` frontmatter. Weight the two accordingly when replanning.
