# Requirements: Aprender Native SetFit Classification

**Defined:** 2026-08-07
**Core Value:** A small labeled dataset can produce an accurate, fast, reproducible classifier
that trains and runs entirely through Aprender's native Rust and APR lifecycle.

## v1 Requirements

### Encoder Conformance

- [ ] **ENC-01**: A developer can import the pinned
  `sentence-transformers/all-MiniLM-L6-v2` revision into a typed SetFit encoder contract, while
  unsupported architecture, tokenizer, pooling, and configuration variants fail with typed errors

- [ ] **ENC-02**: A developer can batch-tokenize texts once and obtain ordered token IDs, type IDs,
  attention masks, truncation facts, and stable input provenance for training and inference

- [ ] **ENC-03**: A developer can encode single or mixed-length padded batches through one shared
  `Transformer -> masked mean pooling -> L2 normalization` path with fixture-verified outputs

- [ ] **ENC-04**: A developer can enumerate named encoder parameters, select frozen and trainable
  groups, and observe finite non-zero gradients and parameter changes for every contracted trainable
  component after a controlled optimizer step

- [ ] **ENC-05**: A developer can switch the encoder recursively between deterministic evaluation
  behavior and training behavior with dropout, without changing which parameters are registered

- [ ] **ENC-06**: A developer can compute a finite tensor-valued cosine-similarity MSE pair loss
  that remains connected to the encoder graph and matches frozen forward and gradient fixtures

### Data and Pairing

- [x] **DATA-01**: A user can acquire or mirror the pinned TweetEval abortion-stance dataset and
  produce canonical train, validation, and test JSONL with exact labels, counts, hashes, and source
  provenance without committing tweet text

- [x] **DATA-02**: A user receives a typed failure for malformed rows, duplicate IDs, unknown labels,
  invalid class counts, conflicting source roles, any pair or selection that would span splits, and
  a training pool that can no longer supply `shots_per_class` after cross-split exclusion.
  Prepare-time cross-split duplicate *content* is excluded from the training pool and recorded, not
  fatal — see D-27 (resolves the DATA-02 / D-18 conflict; `pv`-checked via the exclusion record)

- [x] **DATA-03**: A user can select exactly 8, 16, 32, or 64 unique canonical-training examples
  per class using each contracted seed and receive a stable selected-ID manifest

- [x] **DATA-04**: A user can replay deterministic positive and negative pair generation where
  positive labels match, negative labels differ, endpoints differ, unordered identities are
  canonical, and singleton-class behavior is explicit

- [x] **DATA-05**: A user can set a maximum pair budget per epoch and pair generation remains
  `O(examples + pair budget)` in state and storage rather than materializing a Cartesian product

- [x] **DATA-06**: A user cannot use validation/test examples as training pairs or use the merged
  SetFit compatibility test split for model selection without an explicit fail-closed error

### Training and Classifier

- [x] **TRN-01**: A developer can run a typed SetFit lifecycle whose legal stages are
  `Prepared -> EncoderTuned -> HeadFitted -> ArtifactReloadedAndVerified`

- [x] **TRN-02**: A user can configure and validate encoder learning rate, epochs, batch size,
  warmup, gradient clipping, maximum length, pair policy/budget, freeze policy, head regularization,
  root seed, and device before training begins
  — **QUALIFIER (03-10):** eleven of the twelve knobs are genuinely configurable; **`max_length`
  is VALIDATED, not configurable — the only accepted value is the tokenizer's pinned 256.**
  `MiniLmTokenizer` hard-truncates at `MAX_SEQUENCE_LENGTH = 256` (tokenizer.rs:52) and
  `encode_texts` takes no length parameter, so knob 6 accepts 256 and rejects everything else
  with `MaxLengthNotSupported`. ROADMAP criterion 1 ("invalid ... length ... configuration fails
  before training begins") IS satisfied; this line read as "choose a value" is not. Making it
  configurable would re-litigate Phase 1's pinned tokenizer and is out of scope.

- [x] **TRN-03**: A user receives proof that named encoder gradients, parameter deltas, embedding
  deltas, and pair-loss behavior passed before a run may identify itself as SetFit or export a model

- [x] **TRN-04**: A user can fit one deterministic L2-regularized multinomial softmax classifier for
  any ordered label set with `K >= 2`, with finite logits/probabilities and explicit convergence or
  failure

- [x] **TRN-05**: The classifier head is fit exactly once per unique selected training example using
  the tuned encoder in evaluation/no-gradient mode, so pair multiplicity cannot reweight the head
  dataset

- [x] **TRN-06**: Two clean CPU runs with identical inputs reproduce selected IDs, pair ordering,
  batch ordering, training step count, semantic hashes, predictions, and the declared deterministic
  portions of the loss trace
  — measured CROSS-PROCESS at fixed rayon pool sizes 1 and 3 with the observed pool sizes asserted
  to differ (`setfit_repro_cross_process`, `make setfit-repro-crossproc`), over the RECORDED
  execution digests; and separately proven to have consumed the intended order
  (`setfit_repro_recorded_matches_expected_replay`)

- [ ] **TRN-07**: A user can select configurations and checkpoints using canonical validation only,
  and a selection-lock record is created before canonical test access is permitted
  — **LEFT UNCHECKED at 03-10, deliberately.** The NEGATIVE half is proven from outside the crate
  at compile time (`tests/ui/setfit_token_without_lock.rs` E0451,
  `tests/ui/setfit_metric_value_asserted.rs` E0451) and the mechanics are fully tested in-crate
  (73 `lock_` + 45 `evaluate_` tests). What is missing is the POSITIVE "a user can" tier: no
  out-of-crate caller and no `apr` surface exercises
  `create_selection_lock -> mint_test_token -> CanonicalTestAccess::grant`, so nothing yet
  demonstrates a user REACHING the lock. Checking this box would put a claim in the table that
  the shipped surface does not support — the policy Phase 2 applied to DATA-01..06.
  — **PHASE 4 AUDIT (04-11): STILL UNCHECKED. Both halves moved; the gap 03-10 named did not
  close.** Two evidence tests, each carrying only the label it earned:
  - **CROSS-PROCESS —** `setfit_cli_lifecycle_trn_07_the_test_split_gate_holds_across_processes`
    (04-15, `crates/apr-cli/tests/setfit_cli_lifecycle.rs`): seven spawned `apr` invocations, each
    a separate process. It proves the **negative** half at the binary tier. L1 vs L2 differ in
    exactly ONE flag and produce two DIFFERENT refusals, which is what makes "the gate fired at
    step (1), before the corpus is opened" a measurement rather than a label; L3 and L6 assert no
    lock file exists on disk after a refused run.
  - **IN-PROCESS —** `apr_evaluate_the_lock_travels_between_two_invocations_as_a_file`
    (04-07, `crates/aprender-train/src/train/setfit/apr_evaluate_tests.rs`): TWO INVOCATIONS
    inside ONE process, mediated by a lock FILE. The second scope reconstructs through
    `SelectionLock::from_canonical_bytes` and asserts the rebuilt `lock_hash` equals the recorded
    one. Its own action text says "ACROSS TWO INVOCATIONS", not two processes. **This is not
    cross-process evidence and must not be relabelled as such.**
  What is still missing is exactly what 03-10 named: the POSITIVE "a user can" tier. 04-16 landed
  `reload_verified_run_from_apr`, a public fresh-process door that does reach
  `create_selection_lock -> mint_test_token -> CanonicalTestAccess::grant` from `.apr` bytes — but
  the only bytes able to drive it come from an in-crate `#[cfg(all(test, feature = "setfit"))]`
  fixture, because no user-reachable path produces a `setfit-apr-v1` artifact (F-10). So the
  positive half is proven IN-CRATE and the negative half is proven CROSS-PROCESS, and no `apr`
  surface creates a lock. 04-14's frontmatter asserts `requirements-completed: [OPS-02, TRN-07]`;
  **that claim is not honoured here** — 04-14 shipped two enabling library doors
  (`SetFitTrainConfig::to_request`, `SelectionLock::from_canonical_bytes`), not the requirement.

### APR Artifact Lifecycle

- [ ] **APR-01**: A user can save one checksummed F32 `setfit-apr-v1` artifact containing all encoder
  tensors, exact tokenizer bytes/hash, pooling/normalization/truncation policy, classifier tensors,
  ordered labels, resolved configuration, training evidence, and data/model provenance
  — **PHASE 4 AUDIT: WRITER half shipped, PRODUCER half blocked (F-10).** `write_setfit_apr` and
  `SetFitArtifactDoc` ship with a storage map covering every APR-01 item (04-01/04-02), and the
  bundle carries a six-field `ProvenanceRecord` with no `Option` and no float (04-13, field 20).
  What no user can do is PRODUCE one from a trained model: `into_artifact_bytes` on the only
  out-of-crate-trainable run yields **`setfit-serde-json-v1`**, not `setfit-apr-v1` (04-12,
  measured over three independent routes). 04-13's frontmatter asserts
  `requirements-completed: [APR-01, APR-05]`; **not honoured** — it shipped fields, not the door.

- [ ] **APR-02**: A user can load the SetFit APR offline without Python, a model hub, or sidecar
  files, and malformed, incomplete, oversized, non-finite, or semantically inconsistent artifacts
  fail before prediction
  — **PHASE 4 AUDIT: REFUSAL half shipped and pinned; ACCEPT half is in-crate only.** The loader
  now walks the contract's EIGHT rungs (`rung2_raw_length` .. `rung8_replay_probes`), pinned by
  `the_rung_numbering_matches_the_contracts_eight_rung_ladder`, which parses the contract's own
  `rungs:` block via `include_str!` rather than a hand-copied list and was shown RED on a real
  rename (F-14 item 1, closed in `fb7904bad`). The accept path is exercised only over an in-crate
  `#[cfg(all(test, feature = "setfit"))]` fixture, since no user-producible artifact exists (F-10).
  OPEN: F-14 item 5 — `read_setfit_apr_bytes_bounded` still pre-reserves the caller-declared length
  clamped to 256 MiB, so a source that lies upward commits a quarter gigabyte before reading a byte.

- [ ] **APR-03**: Training closes the in-memory model, reloads the written APR through the production
  core loader, and verifies exact tokenizer/configuration/tensor state plus tolerance-bounded
  embeddings, logits, and probabilities and exact labels
  — **PHASE 4 AUDIT: executable and green at the `--lib` tier over a SUBSTITUTED encoder.** 04-05
  Task 2 (`06b48cf32`, 7 tests) drives the real `verify_artifact` and the whole trusted verify
  policy, keeping dataset, selection, config and evidence real and substituting **the encoder and
  head only** — the F-10 workaround. So the clause holds for the artifact it is given; it has never
  run over a model the shipped train path produced, because that path cannot produce one.

- [ ] **APR-04**: Evaluation, registration, benchmarking, prediction, and serving accept only an
  `ArtifactReloadedAndVerified` model, never an unpersisted trainer object or training checkpoint
  — **PHASE 4 AUDIT: the NEGATIVE (no-bypass) claim is proven from OUTSIDE the crate at COMPILE
  time** — 04-03's trybuild case (`d0b51815c`) pins non-constructibility, `VerifiedSetFitModel`'s
  only constructor is `load_setfit_apr` (04-12), and the credential trait is sealed with exactly
  two implementors, counted by `credential_seal_is_a_private_supertrait` (04-16). The POSITIVE
  clause — that all five named surfaces actually accept a verified model — is demonstrated only
  over fixtures, so the requirement is not closed at the "a user can" tier.

- [ ] **APR-05**: A user can inspect an APR and recover encoder/tokenizer revision and hashes,
  pooling/truncation policy, label order, head configuration, data fingerprint, seeds, update
  evidence, artifact hash, and compatibility schema version
  — **PHASE 4 AUDIT: the renderer is complete; the thing to render is not user-producible.**
  `apr inspect`'s SetFit section renders every APR-05 field offline in both `--json` and human
  output from the RAW recovered document through ONE renderer, pinned against core's normative
  `SETFIT_ARTIFACT_DOC_FIELDS` so a renamed field fails here instead of surfacing as a `null`
  (04-07, 10 tests). 04-07 itself declined to flip this box. 04-15 confirmed at the spawned tier
  that `apr tensors` names all three schema-owned entries with correct types (including the U8
  `tokenizer.blob`) and `apr inspect` exits 0 with non-empty output. Blocked at the same place as
  APR-01: inspection runs over containers and fixtures, never over an artifact a user produced.

### Rust, CLI, and Serving Interfaces

- [ ] **OPS-01**: A Rust caller can train, save, load, embed, classify, and inspect a SetFit model
  through stable fallible library APIs without depending on CLI implementation modules
  — **PHASE 4 AUDIT: NOT MET. Blocked by F-10, a Phase 5 item. Do not check this box.**
  Rung by rung, measured out-of-crate by `crates/aprender-train/tests/setfit_apr_lifecycle.rs`
  (04-12, 5/5 passing): **train** reachable; **save (bytes)** reachable and re-hashed —
  `into_artifact_bytes` returns 1,824,298 bytes whose SHA-256 equals the digest the trusted policy
  recorded; **save as `setfit-apr-v1`** NO, typed `ProbeComputation{probe:"probe_unicode"}` on
  three independent routes; **load / embed / classify / inspect** NO, all four are methods on
  `VerifiedSetFitModel` whose only constructor is `load_setfit_apr`, which has no admissible input.
  Root cause: `CALIBRATED_REGIMES` holds exactly ONE entry with the architecture compared for exact
  equality, so the only trainable encoder is the phase-3 MiniLM slice, whose 97-row vocabulary
  closure cannot compute `probe_unicode` (canonical id 5915). The MECHANICAL half of OPS-01 — the
  graph property that no library crate depends on `apr-cli` — IS closed by `make
  setfit-api-boundary` (04-10) with an EXECUTED five-leg case table including a MUST-MATCH control,
  so an absence check cannot pass on a dead pattern. That gate must not be read as the requirement.
  Closing F-10 needs a calibration run plus a deliberate edit to
  `contracts/setfit-train-lifecycle-v1.yaml` (D-10(c)).

- [ ] **OPS-02**: A user can complete a CPU `train -> APR -> inspect -> eval -> predict` lifecycle
  through `apr` with structured errors and machine-readable JSON output
  — **PHASE 4 AUDIT: NOT MET. Same blocker, observed through the BINARY. Do not check this box.**
  04-15's spawned ladder walks six real `apr` processes: rung 0 `--version` exit 0 (the mechanism,
  proven before any later exit code is interpreted), rungs 1-3 a genuine three-process chain each
  consuming the previous process's FILES and ending in a `--dry-run` report carrying two DIFFERENT
  64-hex provenance fingerprints and rung 2's seed, and **rung 4 `setfit train` exits 6** with a
  typed `ModelLoadFailed`. `model.apr` is asserted absent afterwards, twice, and rung 5 `inspect`
  is run anyway so the stop is demonstrated rather than described. 04-14's frontmatter asserts
  `requirements-completed: [OPS-02, TRN-07]`; **not honoured** — see TRN-07 above.

- [ ] **OPS-03**: Generic APR inspection, evaluation, and prediction commands auto-detect the SetFit
  architecture and call the shared core model rather than reconstructing tokenizer or pooling logic
  — **PHASE 4 AUDIT: the structural claim holds; the call-through is proven only to the loader.**
  Auto-detection is proven EXECUTED, not asserted: `apr predict` against a tagged container exits
  **6 (ModelLoadFailed) and not 4**, and the distinction is the whole proof — 4 would mean the tag
  never routed, which would make every downstream assertion vacuous (04-15). Non-reconstruction is
  structural: `SetFitBundle::from_run_parts` is `pub(crate)`, so a CLI adapter cannot assemble a
  bundle and re-serialize (04-06). Not closed because no prediction has ever completed through core
  — the chain stops at `load_setfit_apr`'s refusal (F-10).

- [ ] **OPS-04**: A user can classify one or many texts through the Rust API and CLI and receive
  ordered labels, full probability vectors, optional logits, winning margin, token/truncation facts,
  artifact identity, backend identity, and latency
  — **PHASE 4 AUDIT: envelope compared field-by-field across three readers; BACKEND IDENTITY is the
  missing clause.** 04-09's parity gate (`make setfit-parity`, 20 tests) compares core, CLI and HTTP
  over one artifact and is proven able to fail; `latency_ms` was removed from `PartialEq` by a
  manual impl so `assert_eq!` is usable as intended, with latency asserted separately BY BITS so a
  renormalized `-0.0` cannot pass (F-14 item 3, closed in `fb7904bad`). Two reasons this stays
  open: the parity fixture is SYNTHETIC and says so in its own module header, and **`backend_identity`
  is still `status: pending`** in `contracts/aprender/binding.yaml` because the row names
  `aprender::setfit::classify::backend_identity`, which does not exist — the identity comes from
  `ExecutionBackend::identity` in `encoder.rs`. A clause this requirement names by name is bound to
  a symbol that is not there.

- [ ] **OPS-05**: A user can load the same APR into native HTTP serving and submit ordered mixed-
  length batches while readiness and responses report the loaded classifier artifact hash
  — **PHASE 4 AUDIT: the cross-surface agreement is proven over a fixture.** 04-08 added
  `HealthResponse.classifier_artifact_sha256` and `classifier_verified`, and
  `setfit_classify_and_readiness_agree_on_the_artifact` proves the two WIRE values name the same
  artifact — which is the cross-surface claim this requirement makes. 04-09 tied
  `classifier_artifact_sha256` to the fixture FILE's own hash with `classifier_verified` true, and
  ships a spawned-server smoke leg (`make setfit-serve-smoke`, tier3). 04-08 itself declined to
  flip the box: the requirement's "the same APR" presupposes an APR a user produced (F-10).

- [ ] **OPS-06**: CPU training and inference work without accelerator features, and an unavailable
  explicitly requested device fails rather than silently falling back or misreporting the backend
  — **PHASE 4 AUDIT: the CLI half is delivered and tested; the "misreporting" half is unbound.**
  Device resolution is exercised end-to-end at the spawned tier — rung 3's report carries
  `resolved.resolved_device = cpu` (04-15) — and the whole phase runs CPU-only with no accelerator
  feature enabled. The clause "or misreporting the backend" depends on the same `backend_identity`
  binding that OPS-04 needs and that does not resolve to a shipped symbol, so the requirement is
  not demonstrable in full.

### Evaluation and Benchmark Claims

- [ ] **EVAL-01**: A user can evaluate ordered single-label predictions with TweetEval's official
  `F_avg`, per-class metrics, three-class macro-F1, MCC, confusion matrix, and validation-only
  calibration diagnostics

- [ ] **EVAL-02**: A user can run every combination of 8, 16, 32, and 64 shots per class with the ten
  contracted seeds for SetFit, each cell bound to a recorded selection manifest so that any future
  second method can be paired against it on an identical sampled-ID hash
  — **AMENDED 2026-09-07 (Phase 5 D-19): the 9B LoRA arm is descoped.** The original text required
  "both SetFit and the 9B LoRA baseline ... in each comparison cell". aprender cannot run the
  Qwen3.5-9B architecture (hybrid `layer_types` 24 linear + 8 full, `attn_output_gate`, and a
  multimodal `text_config`-nested checkpoint; `TransformerConfig` models none of the three, and the
  only hybrid-forward artifact lives in the non-compiling `aprender-contracts-staging`), and no GPU
  host is reachable. The pairing *mechanism* — the selection manifest and its hash — is retained
  and delivered, so the arm can be added later without re-running the SetFit half. Deferred as
  `D-ITEM-05-15`, not cancelled.

- [ ] **EVAL-03**: A user receives one machine-readable row per method/shot/seed run containing
  dataset/model revisions, selection lock, artifact hash, encoder-update evidence, backend/hardware,
  quality metrics, and resource metrics

- [ ] **EVAL-04**: A user can recompute headline means, dispersion, and uncertainty exactly from all
  40 stored SetFit cells, and any missing, selectively omitted, unmatched, or post-test-selected
  cell invalidates the report
  — **AMENDED 2026-09-07 (Phase 5 D-19): the paired SetFit-versus-LoRA delta clause is descoped**
  for the reason recorded against EVAL-02. **The fail-closed behaviour is NOT weakened**: 05-10's
  `verify_run` and the invalidate-on-omission rule are unchanged, and the declared matrix shrinks
  from 40 comparison cells (80 rows) to 40 SetFit rows. **CORRECTED 2026-09-08:** an earlier
  revision of this entry claimed the six doctored negatives survive "byte-for-byte". That was
  false and is retracted. Two of the six — `UnpairedSelection` and the forged candidate ledger —
  target a *second method's* row; `BenchGateError::UnpairedSelection` carries a `lora_hash` field
  (`bench_gate.rs:449`), so under a 40-cell single-method expectation neither is constructible
  from a production path. Their shapes and variant tags are preserved under a retained deferred
  scope with no production constructor. The other four are re-mutated at the new scope rather
  than inherited — CLAUDE.md Verification Discipline rule 4: extending a guard's scope requires
  re-proving it there, because the 80-cell proof does not transfer. The paired-t machinery built by 05-04 is retained and remains contract-bound; it is
  unexercised by this phase's report rather than removed. No report produced under this amendment
  may state or imply a SetFit-versus-LoRA result.

- [ ] **EVAL-05**: A user can compare training time, warm/cold prediction latency, throughput with
  batch/warmup boundaries, peak memory, artifact size, calibration, and classification quality from
  the reloaded production artifacts

### Contracts and Safety

- [ ] **SAFE-01**: A developer can run executable contracts and numerical fixtures that detect
  detached gradients, invalid masks/IDs/labels/classes, pair/split leakage, non-finite math, label
  drift, artifact mismatches, and train/CLI/HTTP parity failures
  — **PHASE 4 AUDIT: the train/CLI/HTTP parity clause is delivered; the contract registry is not
  yet honest, so the box stays open.** `make setfit-parity` (04-09, 20 tests) is the parity
  detector and was proven able to fail. Eighteen scoped suites now run under their own floors via
  `make setfit-all-tests`, and `make contract-audit-phase4` exits 0 over 15 bound equations with
  ZERO `BIND-001`. But 14 of 15 are `implemented` and **one (`backend_identity`) is `pending`
  because it names a symbol that does not exist** — an equation the audit tolerates rather than
  resolves. The parity fixture is synthetic. Several clauses (detached gradients, mask/ID/label
  validity, pair/split leakage) are Phase 1-3 fixtures outside this phase's evidence.

- [ ] **SAFE-02**: A developer can verify the supported CPU build/test feature matrix in CI without
  Python or network access, while reference-fixture generation remains a separate pinned developer
  workflow
  — **PHASE 4 AUDIT: the LOCAL matrix is complete and green; the words "in CI" are not yet true.**
  `make setfit-feature-matrix` (04-10) spans FOUR crates x THREE CPU profiles with BUILD **and RUN**
  legs — checking is not testing — and exits 0, with no Python and no network. Its graph negatives
  are two-sided (a `tokenizers` marker asserted absent by default AND present with `setfit`), so an
  absence check cannot pass on a dead marker; four of its guards were mutated and each observed red
  before being trusted. Two cells are deliberately NOT wired and both are pre-existing standing
  reds, not gaps: `cargo check -p apr-cli --no-default-features` (D-04-09-A, rc=101 for BOTH minimal
  cells because `setfit` does not imply `inference`) and any whole-crate `aprender-serve` test leg
  (D-04-08-A). **The CI half is now APPLIED** at commit `57f7823ab`, from the reviewed patch
  `.planning/phases/04-apr-artifact-and-production-parity/04-11-ci-setfit.patch`, after explicit
  user approval at the 04-11 checkpoint (CLAUDE.md places `.github/workflows/*.yml` edits outside
  autonomous scope, so the executor authored the patch as a FILE and never touched `ci.yml`;
  `git apply` ran only after that ruling). 16 scoped legs landed, each grep-verified verbatim
  against the Makefile so CI and local gates cannot drift, and the YAML re-parsed with both
  branch-protection required checks (`gate`, `workspace-test`) intact.

  One target remains excluded: `setfit-api-boundary`, on QUOTING grounds, not value grounds
  (D-04-11-A). It is a `cargo tree` gate written as a Make `for` loop whose `$$`-escaped vars and
  single-quoted patterns cannot enter the CI step's `bash -c '...'` without the very rewrite the
  gate exists to detect. Risk accepted by the user: a Linux-only dependency-closure regression
  introduced via `cfg(target_os)` would not be caught, since the gate now runs only locally.

  SAFE-02 is therefore **substantially met**, and is left UNCHECKED only because
  `setfit-api-boundary` has no CI coverage. It is not blocked by F-10.

- [x] **SAFE-03**: A user cannot label a frozen linear probe, centroid classifier, or other
  non-updating encoder baseline as SetFit in artifacts, reports, or benchmark output
  — proven from OUTSIDE the crate at COMPILE time: `tests/ui/setfit_probe_claims_setfit.rs` pins
  E0277 for `FrozenProbeRun -> SetFitRun<_>`, and `SetFitRun`'s constructors are private so no
  out-of-crate conversion can be written either

## v2 Requirements

### Encoder and Objective Expansion

- **EXT-01**: A developer can add another sentence-transformer family through a named adapter and
  architecture-specific parity corpus

- **EXT-02**: A user can select alternative contracted encoder objectives such as InfoNCE, SupCon,
  CoSENT, or triplet loss

### Optimization and Deployment

- **ACC-01**: A user can train and infer with optional accelerator backends that pass declared
  numerical-parity, feature-combination, determinism, and backend-reporting contracts

- **QUANT-01**: A user can derive a quantized SetFit artifact with independent quality, calibration,
  parity, and performance evidence

- **CACHE-01**: A user can reuse token or embedding caches addressed by the complete dataset,
  tokenizer, encoder, pooling, normalization, and configuration fingerprint

### Task and Explanation Expansion

- **TASK-01**: A user can train multilabel or hierarchical SetFit classifiers with task-specific
  losses, metrics, APIs, and artifact semantics

- **EXPL-01**: A user can request separately validated token attribution, counterfactual, or exemplar
  explanations with explicit fidelity and privacy contracts

## Out of Scope

| Feature | Reason |
|---------|--------|
| Frozen probe or centroid presented as SetFit | It omits the defining contrastive encoder update |
| Python/PyTorch/ONNX production fallback | It violates the pure-Rust, offline, single-runtime lifecycle |
| Exhaustive Cartesian pair materialization | It has quadratic memory behavior and implicit weighting |
| Hyperparameter tuning on canonical test or merged compatibility test | It leaks evaluation data and invalidates claims |
| Generic arbitrary Hugging Face/remote-code compatibility | V1 supports one fail-closed contracted encoder family |
| Multiple production heads | One multinomial head keeps binary/multiclass semantics, persistence, and probabilities uniform |
| Multilabel, hierarchical, token/span, or generative classification in v1 | These require different targets, metrics, APIs, and artifacts |
| Quantization before F32 parity | Reduced precision would obscure algorithm and serialization discrepancies |
| GPU-required v1 | Few-shot CPU use is core; accelerators follow complete CPU lifecycle proof |
| Persistent embedding cache in v1 | Stale embeddings can silently survive encoder or preprocessing changes |
| Causal token-level explanation claims in v1 | Score evidence is useful; causal attribution needs separate validation |
| Automated search, continual/distributed training, or registry automation | These broaden orchestration before the core lifecycle is trustworthy |
| Vendored TweetEval text | On-demand pinned acquisition avoids licensing and provenance risk |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| ENC-01 | Phase 1 | Pending |
| ENC-02 | Phase 1 | Pending |
| ENC-03 | Phase 1 | Pending |
| ENC-04 | Phase 1 | Pending |
| ENC-05 | Phase 1 | Pending |
| ENC-06 | Phase 1 | Pending |
| DATA-01 | Phase 2 | Complete |
| DATA-02 | Phase 2 | Complete |
| DATA-03 | Phase 2 | Complete |
| DATA-04 | Phase 2 | Complete |
| DATA-05 | Phase 2 | Complete |
| DATA-06 | Phase 2 | Complete |
| TRN-01 | Phase 3 | Complete |
| TRN-02 | Phase 3 | Complete (qualified: `max_length` validated, not configurable) |
| TRN-03 | Phase 3 | Complete |
| TRN-04 | Phase 3 | Complete |
| TRN-05 | Phase 3 | Complete |
| TRN-06 | Phase 3 | Complete |
| TRN-07 | Phase 3-4 | Partial — negative half proven CROSS-PROCESS (04-15); positive half IN-CRATE only (F-10) |
| APR-01 | Phase 4 | Partial — writer + full field set ship; nothing user-reachable produces a `setfit-apr-v1` (F-10) |
| APR-02 | Phase 4 | Partial — 8-rung refusal ladder pinned to the contract; accept path in-crate only (F-10) |
| APR-03 | Phase 4 | Partial — green at `--lib` over a SUBSTITUTED encoder/head; never over a trained model |
| APR-04 | Phase 4 | Partial — no-bypass proven out-of-crate at COMPILE time; positive half fixture-only |
| APR-05 | Phase 4 | Partial — renderer complete and offline; no user-produced artifact to inspect (F-10) |
| OPS-01 | Phase 4 | **Pending — BLOCKED by F-10 (Phase 5).** Only the graph-boundary half is closed |
| OPS-02 | Phase 4 | **Pending — BLOCKED by F-10 (Phase 5).** Spawned chain stops at rung 4, exit 6 |
| OPS-03 | Phase 4 | Partial — routing proven executed (exit 6 not 4); call-through unproven past the loader |
| OPS-04 | Phase 4 | Partial — envelope parity across 3 readers; `backend_identity` binds to a nonexistent symbol |
| OPS-05 | Phase 4 | Partial — readiness/response artifact agreement proven over a fixture |
| OPS-06 | Phase 4 | Partial — CPU-only path delivered; the "misreporting the backend" clause is unbound |
| EVAL-01 | Phase 5 | Pending |
| EVAL-02 | Phase 5 | Pending |
| EVAL-03 | Phase 5 | Pending |
| EVAL-04 | Phase 5 | Pending |
| EVAL-05 | Phase 5 | Pending |
| SAFE-01 | Phase 4 | Partial — parity clause delivered and falsifiable; one binding row unresolved |
| SAFE-02 | Phase 4 | Partial — local 4x3 matrix green; the CI half is an UNAPPLIED patch awaiting human approval |
| SAFE-03 | Phase 3 | Complete |

## Phase 5 in-flight amendment — the 9B LoRA arm (05-11 checkpoint, 2026-09-07)

Recorded here rather than absorbed silently, because this file outlives the phase.

| # | Item | Where | State |
|---|------|-------|-------|
| 1 | **D-19 descope** — EVAL-02 loses the "both SetFit and the 9B LoRA baseline" clause; EVAL-04 loses "paired SetFit-versus-LoRA deltas". Declared matrix: 40 comparison cells (80 rows) → 40 SetFit rows | `05-CONTEXT.md` D-19 | amended above, in-requirement |
| 2 | **Cause (a): no GPU host** — `lambda-vector`/`gx10` unreachable (12 candidates, 12 failures, rc per attempt); the box is the upstream maintainer's and is not accessible to us. AWS fallback enumerated and refuted: 5 instances, all stopped, none GPU | `05-CONTEXT.md` D-19(a); re-recorded in `05-11-SUMMARY.md` on execution | measured, closed |
| 3 | **Cause (b): the architecture is unimplemented** — Qwen3.5-9B is hybrid-attention (24 linear + 8 full) with `attn_output_gate` and a multimodal `text_config` checkpoint; `TransformerConfig` has no field for any of the three; the hybrid-forward scaffold sits in `aprender-contracts-staging`, which has no `Cargo.toml` and never compiles | `05-CONTEXT.md` D-19(b) | **inference from structural absence, not an observed loader failure** — falsifier named and NOT run |
| 4 | **`qwen35-e2e-verification-v1.yaml` is not counter-evidence** — all 7 falsification tests are analytical (param count, FLOPs, memory ordering, roofline, obligation coverage, shape preservation, layer composition). None loads a weight. It verifies the architecture's *description* | `contracts/qwen35-e2e-verification-v1.yaml` | recorded |
| 5 | **What survives untouched** — 05-10's fail-closed claims gate, its six doctored negatives, the invalidate-on-omission rule, the selection-manifest pairing key, and 05-04's paired-t machinery (retained, contract-bound, unexercised) | 05-10, 05-04 | no change |
| 6 | **Deferred, not cancelled** — the SetFit-versus-LoRA comparison and the 9B hybrid forward path | `D-ITEM-05-15` | open ticket |

**Traceability note.** EVAL-02 and EVAL-04 remain Phase 5 requirements and remain `Pending`; they
are amended in scope, not satisfied by this amendment. A future milestone that implements the 9B
hybrid path should restore the descoped clauses rather than re-deriving them.

## Phase 4 closing audit — recorded amendments and limitations (04-11)

Recorded here rather than absorbed silently, because this file outlives the phase.

| # | Item | Where | State |
|---|------|-------|-------|
| 1 | **D-02(a) typed-key amendment** — `model_type` is the ONLY typed key; `schema`, `schema_version`, `ordered_labels` and `tokenizer_sha256` are first-level document fields, not typed keys | 04-01 item 14 | amended in-contract |
| 2 | **D-01 name extension** — 6 of the 21 HF name templates have no canonical form in `tensor-names-v1` (on the pinned model, 22 of 101 encoder tensors have no name to write). The contract RESERVES those six, with non-collision enumerated against the complete `_fallback` set | 04-01 item 5 | amended in-contract + deferred upstreaming |
| 3 | **Nullable-path allowlist is FOUR paths over FIVE walked sub-documents**, deliberately; an unwalked subtree cannot be allowlisted. Residual: `evidence.epsilon_used` (`Option<f64>`) is the one allowlisted path whose only assignment in the tree is `None` | 04-01 item 3, 04-02, 04-13 | recorded, gate pinned by 04-13 |
| 4 | **`SetFitBundle` field 20** — a six-field `ProvenanceRecord` read off the `Selection` the run actually consumed; no `Option`, no float | 04-13 | shipped |
| 5 | **Backend-identity v1 limitation** — grammar `<device>:<implementation>:<kernel>`, v1 value `cpu:setfit-core:autograd-trueno-matmul`. The binding row still names `aprender::setfit::classify::backend_identity`, **which does not exist**; the identity comes from `ExecutionBackend::identity` in `encoder.rs`. Left `pending` rather than flipped, because flipping it would record a claim nothing can check | 04-01 item 12, 04-10, F-14 item 2 | OPEN — blocks a clause of OPS-04 and OPS-06 |
| 6 | **apr-cli `--no-default-features` run-leg limitation** — the dev-dependency feature-unification threat (T-04-61) that 04-09's plan pre-recorded **does not arise**: 04-09 measured the `realizar` dev-dep unnecessary and none was added. The real, measured weakness is different — apr-cli's `--lib setfit` filter runs **13** tests with the feature OFF because it carries setfit-NAMED tests that are not behind the feature, so that run leg proves the gated surface APPEARS, not that it is ABSENT. SAFE-02's apr-cli gating evidence is therefore the CHECK leg plus the two-sided `tokenizers` graph negative | 04-09, corrected by 04-10 | recorded, not absorbed |
| 7 | **`apr qa` does not apply to any encoder-only APR** — exit 5, "APR missing embedded tokenizer". **The SetFit entries are NOT the cause:** the CONTROL, a plain Bert `slice_model.apr` with no SetFit tag and no U8 blob, produces the identical exit code and identical message. `apr qa` is a GENERATIVE-model gate (`load_embedded_bpe_tokenizer` + `run_inference`). One failing input would have produced the wrong cause | 04-15 finding 1 | OPEN — Phase 5 decision |

**The phase-level blocker.** OPS-01 and OPS-02 are not met, and neither is any requirement whose
"a user can" tier routes through a produced artifact. `CALIBRATED_REGIMES` has exactly one entry
with the architecture compared for exact equality, so the only trainable encoder is the phase-3
MiniLM slice, whose 97-row vocabulary closure cannot compute `probe_unicode` (canonical id 5915).
Measured on three independent routes by 04-12, corroborated cross-process by 04-15, and respected
by 04-09. Closing it is a **Phase 5** item: a calibration run plus a deliberate edit to
`contracts/setfit-train-lifecycle-v1.yaml` (D-10(c)). Three separate executors each declined to
route around it by synthesising an APR-capable encoder — which would have compiled and satisfied
their acceptance criteria while testing a model training never produced.

**Two frontmatter overclaims, not honoured by this audit.** 04-13 asserts
`requirements-completed: [APR-01, APR-05]` and 04-14 asserts `[OPS-02, TRN-07]`. Both shipped real
work — bundle provenance and the allowlist gate; the config-override and durable-lock doors — but
neither delivers the requirement at the "a user can" tier, and 04-15 (later, and the only plan that
drives the shipped binary) states explicitly that OPS-02 must not be checked off. No box was
flipped on the strength of a frontmatter field.

**One gate that was NOT run:** `bashrs` is not installed on this host and is genuinely absent, not
shadowed. No `bashrs` check anywhere in Phase 4 may be reported as passing.

**Coverage:**

- v1 requirements: 38 total
- Mapped to phases: 38
- Unmapped: 0

---
*Requirements defined: 2026-08-07*
*Last updated: 2026-08-07 after roadmap creation*
