# Phase 4: APR Artifact and Production Parity - Research

**Researched:** 2026-08-14
**Domain:** In-tree Rust: APR v2 container codec, fail-closed model loading, CLI/HTTP parity surfaces
**Confidence:** HIGH (nearly all findings verified by direct codebase reading on this branch)

## Summary

Phase 4 is almost entirely an **in-tree integration phase**: every load-bearing mechanism it needs
already exists on branch `gsd/phase-2-contract-gate`. The sealed `SetFitCodec` seam
(`aprender-train/src/train/setfit/verify.rs`) is explicitly shaped for an `AprCodec` adapter; the
APR v2 container (`apr-format/src/v2/`) provides typed metadata + a `custom` JSON map, sorted
tensor index, CRC32 checksums, and deterministic offsets; `SetFitMiniLm::from_bundle_parts` is the
core rebuild door; `tower` 0.5 `util` (`ServiceExt::oneshot`) is already used in aprender-serve
tests for in-process router testing; and `SetFitTrainConfig` already deserializes through a wire
struct that routes every payload through the single validating constructor (so `--config`
TOML/JSON is nearly free — `toml = "0.8"` is already a workspace dependency). **No new external
packages are required.**

The two hardest constraints the planner must design around are both verified in this session:
(1) the trusted verify policy performs a **byte-canonical round-trip closure check** —
`codec.serialize(codec.deserialize(bytes))` must equal the original bytes exactly, or
`SetFitTrainError::ReloadNotFromBytes` fires — which means the APR writer path used by the codec
must be a *pure deterministic function of the bundle* (no timestamps, no environment values, no
post-hoc provenance stamping); and (2) `AprV2Metadata.custom` is a
`HashMap<String, serde_json::Value>` whose serialization order was **empirically shown to vary
across runs** in this session — putting multiple SetFit keys directly in `custom` makes the
artifact bytes (and therefore the artifact hash) nondeterministic and breaks the closure check.
The recommended shape stores the entire D-02 JSON document under **one** custom key whose value is
a `serde_json::Map` (BTreeMap-backed, sorted), uses the existing typed `model_type` field for
family detection, and stores the exact tokenizer bytes as a `U8` tensor entry (byte-exact,
64-byte-aligned, recoverable via `get_tensor_data`).

**Primary recommendation:** Build the phase as five thin layers over existing seams — (a) a
`setfit-apr-v1` writer/loader module in `aprender-core/src/setfit/` over the re-exported
`aprender::format::v2` container, (b) a ~100-line sealed-trait `AprCodec` adapter in
`aprender-train/src/train/setfit/`, (c) a `Setfit` namespace + generic `Predict` command +
auto-detect routing in `apr-cli` following the `apr data` template, (d) a feature-gated classify
route + `AppState` slot in `aprender-serve`, and (e) one shared `ClassifyResponse` envelope family
defined in `aprender-core` — with determinism, fail-closed validation, and the three-surface
parity harness as the real engineering content.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

#### Artifact Internals (`setfit-apr-v1`)

- **D-01: Tensors are stored under canonical `tensor-names-v1.yaml` names, with the HF↔canonical
  name map carried in metadata.** Generic `apr tensors/qa/diff` tooling works unmodified, satisfying
  Ph1 D-19's "declared and validated at the Phase 4 APR write boundary". The HF map lets the loader
  rebuild the encoder's named-parameter view and lets APR-03 compare against Phase 3's HF-keyed
  evidence without name translation at the verification gate.

- **D-02: Metadata is hybrid.** (a) A small set of well-known **typed metadata keys** for what
  generic tooling needs with zero SetFit knowledge: `model_family=setfit`, `schema=setfit-apr-v1`,
  artifact schema version, ordered labels, tokenizer SHA-256. (b) **One schema-versioned JSON
  document** parsed into a single `deny_unknown_fields` struct: resolved configuration, training
  evidence summary + table hash (Ph3 D-12's record), provenance, preprocessing policy, HF name map.
  (c) The **exact tokenizer bytes as a binary blob entry** — byte-identical to the pinned upstream
  file, hash-matched (APR-01 "exact tokenizer bytes/hash"; APR-02 forbids sidecars). One typed
  fail-closed parse for the loader, generic inspectability for identity fields.

- **D-03: "Oversized" is derived-plus-cap.** Two checks: (1) **structural** — every tensor's byte
  size must equal what its declared shape/dtype and the declared architecture config imply; any
  mismatch is a typed "inconsistent artifact" failure before prediction; (2) a **contracted hard
  cap** on total file size as the outer resource-exhaustion bound against hostile inputs. Constants
  are contract-resident per the Ph1 D-14 discipline (a legitimate `setfit-apr-v1` is ~90 MB:
  ~87 MB F32 tensors + ~700 KB tokenizer + metadata).

- **D-04: SetFit detection is explicit-tag-only.** Generic commands read the typed
  `model_family`/`schema` keys. No tag → not SetFit, period; a SetFit-shaped tensor set without the
  tag is a plain APR. Tensor-name sniffing stays where it belongs — importing foreign formats this
  writer did not produce. Config↔tensor consistency is already proven by D-03's structural check,
  so detection needs no shape verification of its own.

#### CLI Surface

- **D-05: Training is `apr setfit train`** — a dedicated namespace following Phase 2's `apr data`
  precedent. This closes the question Phase 3 explicitly left open ("any CLI surface for
  training"). It keeps the 9B LoRA baseline (`apr finetune --task classify`) visibly distinct for
  Phase 5's comparison and gives Phase 5 an obvious home for future subcommands. Inputs are Phase 2
  artifacts: the prepared dataset directory and selection manifest.

- **D-06: predict/eval/inspect are generic-only.** `apr inspect`, `apr eval`, `apr predict`
  auto-detect via D-04's tag and route to the shared core path — exactly one implementation per
  operation (OPS-03's "call the shared core model rather than reconstructing"). The `setfit`
  namespace stays training-only. No setfit-namespaced predict/eval aliases.

- **D-07: Training configuration is config-file-first.** `--config` (TOML or JSON) deserializes
  into Phase 3's validated 12-knob `SetFitConfig` (deny_unknown_fields, fail-before-training per
  TRN-02); flags exist only for filesystem paths (dataset dir, selection manifest, output APR) and
  a small override set (e.g. `--seed`, `--device`). The **merged resolved config** — file plus
  overrides — is what the artifact records, so APR-01's "resolved configuration" provenance is
  exact and a Phase 5 cell is reproducible from one file.

- **D-08: One shared typed response envelope across CLI and HTTP.** A versioned response type
  family (e.g. `ClassifyResponse`) is defined once at the core/serve boundary and serialized
  identically by `apr --json` output and the HTTP route — criterion 4's field parity (labels, full
  probabilities, optional logits, margins, token/truncation facts, latency, backend identity,
  artifact hash) becomes a type-level fact rather than a test obligation. Human-readable CLI
  rendering is a view over the same struct. Schema bumps are deliberate and `pv diff`-visible.

#### Serving Boundary

- **D-09: HTTP lives in `aprender-serve`; the model stays in `aprender-core`.** The serve crate
  owns route, `AppState`, and readiness (its architectural job), and calls aprender-core's verified
  SetFit model through the production loader behind a feature-gated dependency edge. No
  reimplemented tokenizer/pooling/model in serve. **This is a documented exception to the
  realizar-first table**: that rule exists because core's LLM inference was ~750x slower than
  realizar's kernels, but SetFit is a 22M-param encoder whose only conformance-proven
  implementation IS the core one (Phase 1's fixture-verified graph path). The CLAUDE.md
  realizar-first table gets a SetFit row documenting this: classification inference is core-owned
  because the conformance evidence lives there. Serving a second, unproven port would violate
  OPS-03 and re-open every Phase 1 fixture.

- **D-10: `apr serve` auto-detects SetFit — no new serve command.** At startup the server loads the
  APR, reads D-04's tag, and installs the classify route(s); the response body is D-08's shared
  envelope; readiness reports the loaded artifact hash + verified state (OPS-05). A SetFit APR is
  just another model `apr serve` can hold. Route naming details are planning's.

- **D-11: `ArtifactReloadedAndVerified` for a fresh process = integrity + embedded self-test
  probes.** The artifact embeds a few train-time-computed probe records (input text → expected
  embedding/logits/probabilities within contracted tolerances). Every fresh consumer process —
  eval, predict, serve — verifies checksum, schema, finiteness, D-03 consistency, AND replays the
  probes through its own loaded model before it may predict. This makes "the served model is the
  evaluated model" a property each consumer re-proves offline, catching corrupted-but-checksummed
  states, wrong-loader states, and platform math divergence — not a training-time memory. (The full
  train-time round trip — close, reload, re-encode, re-predict, compare — remains the gate that
  mints the state in the first place, per Ph3 D-07.)

- **D-12: Backend identity is reported from execution, never echoed from config.** The identity
  field in responses is produced by the compute path that actually ran; `resolve_device` remains
  the fail-closed startup gate for explicit device requests (OPS-06). v1 is CPU-only so the honest
  value is a CPU/SIMD identity — but the plumbing reads it from execution so a future GPU path
  cannot silently misreport. This is CLAUDE.md Verification Discipline rule 2 ("never label a run
  by intent") applied structurally.

#### Parity Gate

- **D-13: Parity is live three-surface comparison PLUS frozen goldens.** One harness feeds the same
  input set (including mixed-length batches and edge cases) through the library call, a spawned
  `apr` process, and an HTTP request, comparing pairwise: exact labels, tolerance-bounded
  probabilities/logits. Frozen golden fixtures additionally pin the core surface against drift over
  time — live compare proves the surfaces agree, goldens prove nobody moved. An **in-band negative**
  (a deliberately skewed surface variant) must fail the gate in every `cargo test`, per the
  Ph1 D-24 / Ph2 D-25 / Ph3 D-08 discipline. Tolerances are contract-resident and committed before
  any comparison runs (Ph1 D-14).

- **D-14: The HTTP leg runs in-process, with one spawned smoke test.** The parity gate drives the
  real axum `Router` + `AppState` in-process (tower `oneshot` — no ports, deterministic, runs in
  every `cargo test`); ONE tier3 smoke test spawns a real `apr serve` on a loopback port to prove
  binary wiring end-to-end (startup, socket, readiness ordering). All-spawned was rejected as the
  Ph1 D-26 failure mode (a flaky gate becomes a gate that stops being run); in-process-only was
  rejected because nothing would ever prove the installed binary serves.

#### Scope Rulings

- **D-15: Encoder convergence is narrowed to SetFit routing only — a deliberate amendment of
  Ph1 D-02.** Phase 4 delivers exactly what its criteria state: generic commands auto-detect and
  route SetFit artifacts to the shared core path. `apr embed`, `apr rerank`, and `CrossEncoder`
  stay on the existing inference-only BERT stack, untouched and unregressed. The full convergence +
  inference regression sweep becomes an explicitly deferred item with its own future ticket.
  Rationale: the milestone's core value does not need it, and it would add a large regression
  surface to the phase that gates Phase 5. Any plan that "cleans up" the legacy encoder stack this
  phase is out of scope.

- **D-16: Phase 4 closes TRN-07's positive half via `apr eval`.** Canonical-test evaluation through
  `apr eval` exercises `create_selection_lock -> mint_test_token -> CanonicalTestAccess::grant` as
  its real workflow — the missing out-of-crate "a user can" tier Phase 3 deliberately left
  unchecked. TRN-07 gets checked in this phase's traceability, and Phase 5's post-test-selection
  invalidation gains a production-proven mechanism before benchmark pressure. The negative half
  (compile-time E0451 proofs) already exists and is not re-proven here.

#### Carried Forward (not re-litigated)

- **Ph3 D-07 (amended):** Phase 4's APR codec lands as a **thin adapter module inside
  `aprender-train/src/train/setfit/`** that calls `aprender-core`'s APR format code — the FORMAT
  stays in core, only the adapter moves. The codec trait is sealed; verification policy (hashing,
  close, reload, re-encode, re-predict, compare, minting) stays in trusted crate-internal lifecycle
  code. Phase 4 swaps the format behind the seam; it does not redesign the check.
- **Ph3 D-12:** training evidence = compact summary + hash of the full per-parameter table; the APR
  carries the summary + hash (D-02's JSON doc); the APR checksum is the external anchor that
  upgrades "binding" toward tamper-evidence.
- **Ph3 D-14:** selection lock is hash-committing with a typestate token; token minting takes the
  verified run object; canonical test access requires artifact-hash match. D-16 consumes this
  exactly as shipped.
- **Only a closed, production-reloaded, parity-verified F32 APR may reach evaluation, CLI
  prediction, benchmarking, or serving** (roadmap-level Phase 4 decision, already in STATE.md).
- `resolve_device` fails closed on explicitly requested unavailable devices (Ph3/D-05 reuse).
- Contract-resident tolerances committed before comparison (Ph1 D-14); fixture SHA-256 manifests
  (Ph1 D-13); one new phase contract referencing existing ones rather than editing them (Ph1 D-23).
- In-band negatives in every `cargo test`; `cargo-mutants` scoped to new code; trybuild
  non-constructibility where typestates gate behavior; `pv` only, never bash/yq/python.
- Tier wiring: fast gates in tier2, heavy/contract gates in tier3/tier4 — and Phase 3's CR-01
  lesson is now structural: a feature-gated surface MUST have a tier and CI job that actually
  compiles and runs it. SAFE-02's CPU feature matrix (`--no-default-features`,
  `--features setfit`, all-features) extends the Phase 1 D-06 matrix to the new surfaces.
- `aprender-contrastive-data` is bytes-in/typed-out; `apr-cli` owns filesystem adapters (Ph2 D-04).
- `unwrap()` banned, `unsafe_code = "forbid"`, typed errors on all fallible paths.
- Branch/PR policy: Phases 2 and 3 rode `gsd/phase-2-contract-gate`; no PR opened yet — opening it
  is the human's call (02-01 policy). The planner must state Phase 4's branch base explicitly.

### Claude's Discretion

The user selected the recommended option on all 16 questions and delegated nothing explicitly.
The following were surfaced and consciously left to research/planning as implementation detail:

- The exact well-known metadata key names and the JSON document's field schema (D-02), including
  where the full evidence table (vs its hash) is emitted.
- The concrete hard-cap constant (D-03) and probe-record count/selection policy (D-11) — both are
  contract-resident numbers to be derived and committed before comparison.
- HTTP route naming/paths and whether classify rides an existing APR predict route or a new path
  (D-10 fixed the mechanism, not the URL).
- The shape of the new phase contract (e.g. `setfit-apr-v1.yaml` owning artifact schema, load
  validation, parity tolerances, probe policy) vs extending `setfit-train-lifecycle-v1.yaml` —
  Ph1 D-23's one-contract-per-phase pattern is the default.
- Exit-code taxonomy for the new CLI commands (existing `CliError` categories are the baseline).
- Whether `apr setfit train` gains a `--dry-run`/validate-only mode.
- How the LoRA baseline's Phase 5 needs (identical sampled IDs) surface in this phase's CLI, if at
  all — Phase 5's problem unless planning finds a cheap hook.
- Mutation-testing scope boundaries for the new surfaces, given the Phase 3 lesson that
  aprender-core's test binary needs `--timeout >= 120` and tree-copy mode.

### Deferred Ideas (OUT OF SCOPE)

- **Full encoder convergence** — migrating `apr embed` / `apr rerank` / `CrossEncoder` onto the
  graph-connected encoder with an inference regression sweep (Ph1 D-02's original target, narrowed
  by D-15). Needs its own ticket; the legacy inference-only BERT stack remains untouched this
  phase.
- **GPU/accelerator SetFit inference** — v2 (ACC-01); OPS-06's fail-closed handling is the only
  device work this phase.
- **Quantized SetFit artifacts** — v2 (QUANT-01); F32 parity first.
- **The crates.io publish cascade** (`aprender-contrastive-data` then `apr-cli`) — still pending,
  human-run, gates pre-release Gate 5; Phase 4 adds surfaces to `apr-cli` but does not change the
  cascade's shape.
- **Production-encoder calibration regime for the SetFit-identity gate** — the ROADMAP-documented
  Phase 5 blocker: a benchmark run against production `all-MiniLM-L6-v2` returns
  `UncalibratedRegime` until its fingerprint is calibrated into
  `contracts/setfit-train-lifecycle-v1.yaml` via a deliberate `pv diff`-flagged edit (Ph3 D-10(c)).
  Phase 4 does not silently widen the regime; if planning finds it cheap to calibrate here, that is
  a deliberate contract edit to surface, not an inline change.
- **Auth coverage across all serve paths** — the existing `APR_API_KEY*` gate covers only the CPU
  fallback router; unifying auth across router variants is repo-wide serving work, not SetFit
  scope.
- **Repo-wide deferred defects** (D-ITEM-01..04 from Phase 2) — unchanged; the planner should
  expect them when wiring gates (vacuous macOS guards, arm64 clippy red, tier2 zero-test headline,
  contract-audit exit-0).
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description (abridged) | Research Support |
|----|------------------------|------------------|
| APR-01 | Save one checksummed F32 `setfit-apr-v1` with encoder tensors, exact tokenizer bytes/hash, policies, head, labels, resolved config, evidence, provenance | APR v2 container verified: sorted tensor index, CRC32 header+footer, `U8` dtype for tokenizer blob, typed metadata + `custom` map; `SetFitBundle` already carries every required field; determinism pattern in Pitfalls 1–2 |
| APR-02 | Load offline, no Python/hub/sidecars; malformed/incomplete/oversized/non-finite/inconsistent artifacts fail before prediction | Loader validation ladder (Pattern 3); D-03 structural check maps to `TensorIndexEntry` shape/dtype vs byte size; bundle-limits precedent (`BundleLimits::CONTRACTED`, bounds enforced before allocation) verified in `bundle.rs` |
| APR-03 | Train closes in-memory model, reloads via production core loader, verifies exact tokenizer/config/tensor state + tolerance-bounded outputs | `run_verify_policy` verified: close→hash→decode→**byte round-trip closure**→rebuild→re-probe→compare at `Tolerance::EXACT`; `AprCodec` adapter shape prescribed by `verify.rs` module docs; `SetFitMiniLm::from_bundle_parts` exists |
| APR-04 | Only `ArtifactReloadedAndVerified` models reach eval/registration/benchmark/predict/serve | Sealed typestate lifecycle exists (`LifecycleState` sealed, 4 markers); consumer-side gate = D-11 probe replay in the core loader (Pattern 3); trybuild non-constructibility precedent in `aprender-train/tests/ui/` |
| APR-05 | Inspect recovers revisions/hashes, policies, label order, head config, fingerprints, seeds, evidence, artifact hash, schema version | `apr inspect` already parses `AprV2Metadata` incl. `custom` (inspect.rs:525 reads custom keys); SetFit branch reads the typed tag + one-key JSON doc |
| OPS-01 | Rust caller trains/saves/loads/embeds/classifies/inspects via stable fallible APIs, no CLI modules | Layering verified: core owns loader+classify+envelope; train owns lifecycle; apr-cli is adapter-only (Ph2 D-04 precedent `data_contrastive.rs`) |
| OPS-02 | CPU train→APR→inspect→eval→predict via `apr` with structured errors and JSON | `Commands`/`ExtendedCommands` + dispatch wiring pattern verified; `DataCommands` namespace is the template; `CliError::exit_code()` taxonomy exists; **`apr predict` does not exist today — new generic command** |
| OPS-03 | Generic commands auto-detect SetFit, call shared core model | D-04 tag detection via typed `model_type`/custom keys; `apr eval` exists (`--task classify` path exists for LoRA baseline); routing point identified in `dispatch_analysis.rs` |
| OPS-04 | Classify one/many texts: labels, full probabilities, optional logits, margin, token/truncation facts, artifact identity, backend identity, latency | `SetFitMiniLm::tokenize` returns `SentenceBatch` with truncation facts; `MultinomialLogisticRegression::predict_proba` (core) supplies probabilities; trueno runtime backend detection (`detect_x86_backend`/`detect_arm_backend`) supplies execution-derived identity (D-12) |
| OPS-05 | Same APR into native HTTP serving; readiness + responses report artifact hash | `create_router_with_config` + `AppState` slot pattern verified (`apr_model`, `apr_transformer` precedents); `/health/ready` handler exists; recommend NEW `/v1/classify` route (existing `/v1/predict` takes `features: Vec<f32>`, wrong shape) |
| OPS-06 | CPU works without accelerator features; explicit unavailable device fails, backend never misreported | `resolve_device` verified: explicit `cuda`/`cuda:N` hard-fails via `DeviceError::CudaNotAvailable`; `prepare()` already rejects non-CPU resolved devices; D-12 execution-derived identity |
| SAFE-01 | Executable contracts + fixtures detect detached gradients, invalid data/math, leakage, label drift, artifact mismatches, core/CLI/HTTP parity failures | New `setfit-apr-v1.yaml` contract (Ph1 D-23 pattern); parity harness (D-13/D-14) with in-band negative; existing contracts referenced not edited; `$(CONTRACTS)` Makefile list must be extended (explicit list, not glob — Ph2 lesson) |
| SAFE-02 | CPU build/test feature matrix in CI, no Python/network; fixture generation separate | `setfit-feature-matrix` Make target + CI `setfit` steps verified (ci.yml:274–309 runs core+train setfit tests); matrix must grow apr-cli and aprender-serve legs; CR-01 lesson: every new feature-gated surface needs a tier AND CI job that compiles and RUNS it |
</phase_requirements>

## Project Constraints (from CLAUDE.md)

Root `CLAUDE.md` directives that bind this phase (the crate-level
`crates/aprender-train/CLAUDE.md` is the stale pre-monorepo entrenar file — root file governs;
its "Python is PROHIBITED" and contract-first spirit are consistent with phase discipline anyway):

- **Branch protection:** `main` is protected; work on a feature branch; CI `ci / gate` +
  `workspace-test` must pass. Current branch: `gsd/phase-2-contract-gate` @ d66678e7a (Phases 2+3
  ride it; no PR opened — human's call). Planner must state Phase 4's branch base explicitly.
- **Debugging:** use `apr` diagnostic tools first; **pin the binary** via `. scripts/apr_bin.sh`
  — never bare `apr`, never a hardcoded path.
- **Realizar-first table:** ALL inference/serving MUST use realizar — **D-09 is a documented
  exception** and the plan MUST include the CLAUDE.md table edit adding the SetFit row (the
  exception must live where the rule lives).
- **LAYOUT-001/002:** APR is ROW-MAJOR exclusively; the writer must set
  `AprV2Flags::LAYOUT_ROW_MAJOR`; `contracts/tensor-layout-v1.yaml` applies to every tensor
  written.
- **Publishing safety (CB-510):** root-anchored `.gitignore`/exclude paths; after any
  `.gitignore`/`Cargo.toml` exclude change re-run both check scripts (NOTE: both are vacuous on
  macOS — D-ITEM-01; CI covers them on Linux).
- **bashrs for shell scripts**, sourced libraries option-neutral (fail by return status).
- **Verification Discipline (all 8 rules):** never `$?` through a pipe; never label a run by
  intent (D-12 makes rule 2 structural); pin the binary; re-mutate when extending guard scope; a
  gate must scan the surface where the decision is made; vary failing inputs; guard regexes ship a
  case table; check for shadowed artifacts.
- **`pv` dogfooding:** contract validation/lint/diff/score through `pv`
  (`PV_BIN := cargo run --release -p aprender-contracts-cli --bin pv --`), never bash/yq/python.
  `pv diff` takes two filesystem paths, never a git revision.
- **Code search:** `pmat query`, not grep, for intent-level discovery.
- **`unwrap()` banned** (`.clippy.toml` disallowed-methods), `unsafe_code = "forbid"`, tiered
  gates (`make tier1..tier4`), coverage floor 88%.
- **rtk hook:** `git status --porcelain` emptiness assertions must run through `rtk proxy`
  (Ph2 lesson, recorded in STATE.md).
- **Autonomous mode boundaries:** modifying `.github/workflows/*.yml` requires check-in BEFORE
  acting — SAFE-02's CI wiring touches ci.yml, so the plan must surface that edit as a
  human-visible step (Phase 3's CR-01 CI half was item 4 of 03-HUMAN-UAT for exactly this reason).

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| `setfit-apr-v1` byte container (header, index, checksums, alignment) | `apr-format` (v2) | — | ARCHITECTURE.md: "byte-level APR mechanics in apr-format"; container needs NO changes for this phase [VERIFIED: codebase] |
| `setfit-apr-v1` format semantics: writer, production loader, validation ladder, probe replay | `aprender-core/src/setfit/` (new module, `setfit` feature) | uses `aprender::format::v2` re-export | "Format semantics in core" (ARCHITECTURE.md); OPS-03 requires one shared core path; core already owns `SetFitMiniLm`, tokenizer, `MultinomialLogisticRegression`, `from_bundle_parts` |
| Classify API + `ClassifyResponse` envelope family (D-08) | `aprender-core` | consumed by apr-cli + aprender-serve | Core is the only crate both consumers already depend on; dependency direction verified (serve→core optional dep exists; cli→core non-optional) |
| `AprCodec` adapter (sealed `SetFitCodec` impl) | `aprender-train/src/train/setfit/` | calls core writer/loader | Sealed trait forces in-crate impl (verify.rs module docs prescribe exactly this shape) [VERIFIED: codebase] |
| Train→save→reload→verify policy, lifecycle states, lock/token/eval | `aprender-train` (existing, unchanged) | — | Ph3 D-07: Phase 4 swaps the format behind the seam, not the check |
| `apr setfit train`, generic `apr predict` (new), inspect/eval auto-detect routing | `apr-cli` | calls core + train | Ph2 D-04: CLI owns filesystem adapters only; `apr data` namespace is the wiring template |
| Classify HTTP route, `AppState` slot, readiness reporting | `aprender-serve` (new `setfit` feature) | calls core loader/model | D-09; router/AppState/health handlers already exist; tower `oneshot` in-process test pattern already in-tree |
| Device gate + backend identity | `aprender-train::train::device` (gate) + the `ExecutionBackend` value RETURNED by the encode invocation that ran, defined in `aprender-core/src/setfit/encoder.rs` (identity) | surfaced through core classify path | `resolve_device` fail-closed verified. Identity column **SUPERSEDED** — see the note under this table. |
| Parity harness (3-surface + goldens + in-band negative) | new integration test crate location — planner sites it (apr-cli `tests/` is natural: it can dep on core, train, serve) | — | Needs all three surfaces reachable; apr-cli already deps on core and (optionally) train, and can add serve as dev-dep — verify direction at plan time |

> **SUPERSEDED (amended 2026-08-14) — the "Device gate + backend identity" identity source.**
>
> This row originally named "trueno runtime detection (identity)" and called `detect_*_backend()`
> "the execution-derived identity source". That is wrong, and the Phase 4 plan set now FORBIDS it.
> `detect_*_backend()` is CAPABILITY detection: it reports what the host CAN do, never what ran.
> `trueno::Matrix::matmul` dispatches on SIZE (`matmul_naive` below the 64 threshold,
> `gemm_blis_parallel` above it, a GPU path when compiled in), so an AVX2 detection result is
> consistent with a scalar execution — the defect cross-AI review finding B6 raised.
>
> A capability-detection value in the backend field is now a CONTRACT VIOLATION. See **04-01 item
> 12** (backend-identity grammar: the value MUST be produced by the encode invocation that ran and
> returned to the caller; a config/env/CLI value must not reach the field) and **04-04 Task 2**,
> whose acceptance criterion asserts that
> `grep -rn "select_backend\|Backend::AVX\|detect_x86_backend\|detect_arm_backend" crates/aprender-core/src/setfit/`
> returns ZERO matches. The identity channel is `ExecutionBackend`, produced by
> `encode_with_backend` / `encode_texts_traced`, with v1 value `cpu:setfit-core:autograd-trueno-matmul`.
>
> The recorded v1 limitation stands and is not a licence to fall back on detection: trueno exposes
> no per-dispatch execution report, so the identity names the kernel ENTRY POINT the encoder
> invoked, not the innermost dispatch chosen. Upgrading to per-dispatch reporting needs a trueno API
> and is a recorded deferred item. Do not reintroduce detection here.

## Standard Stack

**No new external dependencies are required for this phase.** Everything below is already in the
workspace dependency tree on this branch.

### Core (in-tree)
| Component | Location | Purpose | Status |
|-----------|----------|---------|--------|
| APR v2 container | `crates/apr-format/src/v2/` (re-exported as `aprender::format::v2`) | Header/index/checksum/alignment, `AprV2Writer`, `AprV2Reader`, `AprV2Metadata` with `custom` map, `TensorDType::U8` | [VERIFIED: codebase] |
| Sealed codec seam | `aprender-train/src/train/setfit/verify.rs` | `SetFitCodec` trait (3 methods), trusted `run_verify_policy`, `Tolerance` type, `artifact_hash` (SHA-256 free function) | [VERIFIED: codebase] |
| Bundle | `aprender-train/src/train/setfit/bundle.rs` | `SetFitBundle` — complete normative field list; `from_run_parts`; `BundleLimits` precedent | [VERIFIED: codebase] |
| Core model | `aprender-core/src/setfit/` (`setfit` feature) | `SetFitMiniLm::from_bundle_parts`, `tokenizer_bytes()`, `tokenizer_sha256()`, `named_parameters()`, `encode_texts()`, `architecture()` | [VERIFIED: codebase] |
| Head | `aprender::classification::MultinomialLogisticRegression` | `predict`, `predict_proba`, `weights()`, `intercepts()`, `n_features()` | [VERIFIED: codebase] |
| Config | `aprender-train/.../config.rs` | `SetFitTrainConfig` Deserialize routed through validating wire struct (`try_from = "SetFitTrainConfigWire"`) — `--config` TOML/JSON validation is free | [VERIFIED: codebase] |
| Device gate | `aprender-train/src/train/device.rs` | `resolve_device` — explicit `cuda`/`cuda:N` hard-fails `CudaNotAvailable`; `Device::tag()` | [VERIFIED: codebase] |
| Lock/eval | `aprender-train/.../lock.rs`, `evaluate.rs` | `create_selection_lock`, `mint_test_token`, `CanonicalTestAccess::grant`, `evaluate_validation` — D-16's substrate | [VERIFIED: codebase] |
| Serve router | `aprender-serve/src/api/router.rs`, `api/mod.rs` | `create_router_with_config`, `AppState` (Clone, many `Option<Arc<_>>` model slots), `/health/ready` | [VERIFIED: codebase] |
| CLI wiring | `apr-cli/src/commands_enum.rs` (+ flattened `extended_commands.rs`), `dispatch_analysis.rs`, `commands/data_contrastive.rs` | Namespace subcommand template (`Data { command: DataCommands }` → dispatch → commands module) | [VERIFIED: codebase] |

### Supporting (already-present external deps)
| Library | Version | Purpose | Where |
|---------|---------|---------|-------|
| serde_json | 1.0, **`float_roundtrip` enabled** in core + train | Metadata JSON, envelope serialization; float_roundtrip is REQUIRED for byte-canonical floats | [VERIFIED: Cargo.toml aprender-core:78, aprender-train:109] |
| toml | 0.8 (workspace dep; aprender-train already uses it) | D-07 `--config` TOML parsing — apr-cli adds `toml = { workspace = true }` (no new external dep) | [VERIFIED: root Cargo.toml:141] |
| sha2 | via `aprender/setfit` feature (`dep:sha2`) and aprender-train | SHA-256 artifact hash + tokenizer hash — never hand-roll | [VERIFIED: Cargo.toml] |
| hex | in aprender-train | bit-pattern float encoding precedent (bundle.rs `f32_to_hex`) | [VERIFIED: codebase] |
| axum + tower 0.5 (`util`) + tower-http | in aprender-serve (behind `server` feature) | Router; `ServiceExt::oneshot` for D-14 in-process leg — already used in `api/tests/app_state_default.rs` | [VERIFIED: codebase] |
| trybuild (workspace), proptest 1.4, insta, tempfile | aprender-train dev-deps | Non-constructibility proofs, property tests, snapshots | [VERIFIED: Cargo.toml:162–177] |
| cargo-mutants 25.3.1, cargo-nextest 0.9.102 | installed on host | Mutation + CI test runner | [VERIFIED: local invocation] |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Storing tokenizer bytes as a `U8` tensor entry | `custom` metadata field (hex/base64 string) | Metadata JSON inflates by ~1.4 MB and rides the nondeterministic-map risk; the U8 tensor is byte-exact, 64-byte aligned, recoverable via `get_tensor_data`, and visible to generic tooling. Recommend U8 tensor |
| One custom key holding the whole SetFit JSON doc | Multiple top-level custom keys | Multiple keys serialize in **random order** (empirically verified) → nondeterministic bytes/hash → breaks round-trip closure. One key + `serde_json::Map` value (BTreeMap-sorted) is deterministic without touching apr-format |
| New `/v1/classify` route | Reuse `/v1/predict` | Existing `apr_predict_handler` takes `features: Vec<f32>` (numeric AprModel) — overloading breaks the typed envelope and confuses two model kinds. New route recommended |
| Changing `AprV2Metadata.custom` to `serde_json::Map` repo-wide | Keep `HashMap`, one-key discipline | The type change fixes determinism for everyone but touches 39 files / 77 use sites across crates — out of proportion for this phase; the one-key discipline needs zero apr-format changes |

**Installation:** none — `cargo build` with existing workspace.

## Package Legitimacy Audit

**This phase installs no new external packages.** All dependencies are already resolved in the
workspace lockfile (`toml 0.8` is workspace-declared and used by `aprender-train`; adding
`toml = { workspace = true }` to `apr-cli` introduces no new registry package). slopcheck was
therefore not run; there is nothing to check.

| Package | Registry | Age | Downloads | Source Repo | slopcheck | Disposition |
|---------|----------|-----|-----------|-------------|-----------|-------------|
| — none — | | | | | | |

**Packages removed due to slopcheck [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

## Architecture Patterns

### System Architecture Diagram

```
                       TRAIN-TIME (aprender-train, `setfit` feature)
  Phase-2 artifacts ──► SetFitRun<Prepared> ─► tune_encoder ─► fit_head ─► verify_artifact(AprCodec)
  (dataset dir +                                                              │
   selection manifest)                                          run_verify_policy (TRUSTED):
                                                                probe live ─► serialize ─► SHA-256
                                                                ─► DROP live model ─► deserialize
                                                                ─► re-serialize == bytes? (closure)
                                                                ─► rebuild ─► re-probe ─► compare
                                                                      │
                                                                      ▼
                                                        setfit-apr-v1 file (~90 MB)
                                                        [APR v2 container: header ─ metadata JSON
                                                         (typed fields + ONE "setfit" custom key)
                                                         ─ sorted tensor index ─ F32 tensors under
                                                         canonical names + U8 tokenizer blob ─ CRC32]
                                                                      │
              ┌───────────────────────────────────────────────────────┼───────────────────────────┐
              ▼                                                       ▼                           ▼
   CONSUMER: Rust API (OPS-01)                          CONSUMER: apr CLI (OPS-02/03)   CONSUMER: apr serve (OPS-05)
   aprender-core production loader                      inspect / eval / predict         startup: load + detect tag
      1 read bytes, cap check (D-03)                    detect tag ─► SAME core loader   ─► AppState.setfit slot
      2 header/CRC/schema parse (fail-closed)           ─► SAME classify path            ─► install /v1/classify route
      3 structural: shape×dtype == byte size            eval additionally drives         readiness reports artifact
      4 non-finite scan, tokenizer hash match           lock ─► token ─► grant (D-16)    hash + verified state
      5 rebuild model (from_bundle_parts)                        │                           │
      6 REPLAY EMBEDDED PROBES (D-11)  ──────────► only then: VerifiedSetFitModel ◄──────────┘
              │                                                       │
              ▼                                                       ▼
   classify(texts) ─► ClassifyResponse envelope (D-08, defined ONCE in core):
   labels, full probabilities, optional logits, margins, token/truncation facts,
   latency, backend identity (READ FROM EXECUTION via trueno detection),
   artifact hash  ──► serialized identically by `apr --json` and the HTTP route

   PARITY GATE (D-13/D-14): one harness ─► [in-proc library call] vs [spawned `apr` process]
   vs [in-proc axum Router via tower::oneshot] ─► pairwise compare (exact labels, contracted
   tolerances) + frozen goldens + in-band skewed-surface negative; ONE tier3 spawned-serve smoke
```

### Recommended Project Structure

```
crates/aprender-core/src/setfit/
├── artifact.rs          # NEW: setfit-apr-v1 writer + production loader + validation ladder
│                        #      + probe replay + VerifiedSetFitModel (typestate: only door to classify)
├── classify.rs          # NEW: classify path + ClassifyResponse envelope family (D-08)
└── (existing: encoder.rs, tokenizer.rs, import.rs, mod.rs — untouched per D-15)

crates/aprender-train/src/train/setfit/
├── apr_codec.rs         # NEW: impl sealed::Sealed + SetFitCodec for AprCodec (thin adapter
│                        #      over core's artifact.rs; format id "setfit-apr-v1")
└── (existing verify.rs/bundle.rs/lock.rs/evaluate.rs — format swapped, checks unchanged)

crates/apr-cli/src/
├── setfit_commands.rs   # NEW: SetfitCommands enum (Train {...}) — apr data template
├── commands/setfit_train.rs  # NEW: filesystem adapter (config file → SetFitTrainConfig,
│                        #      dataset dir + manifest in, APR out, atomic write)
├── commands/predict.rs  # NEW: generic `apr predict` (auto-detect; SetFit → core classify)
└── (inspect.rs, eval/ — gain SetFit auto-detect branches)

crates/aprender-serve/src/
├── api/setfit_handlers.rs  # NEW: /v1/classify handler + AppState slot + readiness fields
└── (router.rs — conditional route install; Cargo.toml gains `setfit` feature)

contracts/setfit-apr-v1.yaml  # NEW phase contract: artifact schema, load-validation ladder,
                              # size cap, probe policy/count, parity + probe tolerances
                              # (referencing, not editing, existing contracts; append to $(CONTRACTS))
```

### Pattern 1: The AprCodec adapter (shape is prescribed, not designed)

**What:** `verify.rs` module docs state it verbatim: "phase 4's APR codec lands as a thin ADAPTER
module inside this crate — an `impl Sealed for AprCodec` plus a `SetFitCodec` impl that calls
`aprender-core`'s APR format code." [VERIFIED: codebase, verify.rs:20–29]

```rust
// aprender-train/src/train/setfit/apr_codec.rs (shape; core fns are Phase 4 work)
pub const APR_FORMAT_ID: &str = "setfit-apr-v1";

pub struct AprCodec;
impl super::verify::sealed_marker::Sealed for AprCodec {}   // via the existing private module
impl SetFitCodec for AprCodec {
    fn format_id(&self) -> &'static str { APR_FORMAT_ID }
    fn serialize(&self, bundle: &SetFitBundle) -> Result<Vec<u8>, CodecError> {
        aprender::setfit::artifact::write_setfit_apr(bundle_view(bundle))   // core owns semantics
            .map_err(|source| CodecError::Bundle { format_id: APR_FORMAT_ID.into(), source: source.into() })
    }
    fn deserialize(&self, bytes: &[u8]) -> Result<SetFitBundle, CodecError> { /* core read + map back */ }
}
```

Constraints the adapter inherits from the trusted policy [VERIFIED: verify.rs]:
- **Byte-canonical:** `serialize(deserialize(bytes)) == bytes` exactly, or
  `ReloadNotFromBytes`. See Pitfalls 1–2.
- The format-id is stamped by trusted code (`SetFitBundle::from_run_parts(codec.format_id(), ..)`)
  and re-checked by the trusted `decode` — the adapter must also self-check on its public surface
  (the `SerdeJsonCodec` comment explains why both checks exist).
- `verify_artifact` currently passes `Tolerance::EXACT`. Since APR stores raw LE f32 bytes
  (bit-exact, same as the hex bundle) and the rebuild runs the same CPU path, EXACT should hold
  for the train-time round trip. If planning decides otherwise, the tolerance must be selected by
  trusted crate code keyed on codec — never a trait parameter (the type's doc says exactly this).

Note the seal: `sealed::Sealed` is a private module in `verify.rs`. The adapter lives in the same
crate, so either the module gains `pub(crate)` visibility for the marker or the codec lives in
`verify.rs`'s module tree — a one-line visibility decision for the planner, not a redesign.

### Pattern 2: Deterministic metadata — one custom key + typed `model_type`

**What:** All D-02 content that is not a tensor goes in exactly two places:
1. Typed field `model_type: "setfit"` (first-class, declaration-order-serialized) — D-04's tag.
2. ONE `custom` entry, e.g. `"setfit" -> serde_json::Value::Object(...)`, where the object is
   built as a `serde_json::Map` (BTreeMap-backed by default → sorted keys → deterministic
   [VERIFIED: serde_json default; `preserve_order` not enabled anywhere in workspace]). The object
   carries: `schema: "setfit-apr-v1"`, artifact schema version, ordered labels, tokenizer SHA-256,
   preprocessing/pooling policy record, resolved + requested config, evidence summary + table
   hash, provenance (dataset fingerprints, selection hash, seeds, pinned revision), HF↔canonical
   name map, and the D-11 probe records.

**Why:** `AprV2Metadata.custom` is `HashMap<String, serde_json::Value>` with `#[serde(flatten)]`
[VERIFIED: header_impl.rs:132–300]. HashMap key order was **empirically verified to differ across
three runs of the same binary** in this session. Two or more custom keys ⇒ metadata JSON byte
order varies ⇒ artifact hash varies run-to-run AND `serialize(deserialize(bytes)) != bytes` ⇒ the
round-trip closure fails nondeterministically. One key sidesteps this with zero apr-format changes.
The loader parses that one value into a single `#[serde(deny_unknown_fields)]` struct — D-02's
"one typed fail-closed parse".

Well-known key names (Claude's discretion, recommendation): typed `model_type = "setfit"`;
custom key `"setfit"`; inside the doc use snake_case: `schema`, `schema_version`,
`ordered_labels`, `tokenizer_sha256`, `hf_name_map`, `preprocessing`, `resolved_config`,
`requested_config`, `evidence`, `provenance`, `probes`.

**Float carriage inside the doc:** any f32/f64 that must round-trip bit-exactly (probe
expectations, evidence stats already handled by Ph3 types) should use the bundle.rs bit-pattern
hex precedent rather than decimal JSON — serde_json renders non-finite floats as `null`
(CR-03 lesson) and decimal round-tripping depends on `float_roundtrip` staying enabled.

### Pattern 3: The production loader is a validation ladder that ends in a typestate

**What:** One core function, every consumer calls it, order is load-bearing (mirror of
`SetFitBundle::from_canonical_bytes`):

1. File-size hard cap check on raw length BEFORE parsing (contract-resident constant, D-03).
2. Header parse: magic, version, header CRC (`AprV2Header::verify_checksum`), row-major flag,
   footer CRC over content.
3. Typed tag check (`model_type == "setfit"`), one-key doc parse (`deny_unknown_fields`),
   schema version check (refuse unknown, like `BUNDLE_SCHEMA_VERSION`).
4. Structural consistency: every tensor's `size` equals shape×dtype-width implied by the declared
   `EncoderArchitecture`; tensor set is exactly the canonical expected set (missing = incomplete,
   extra = inconsistent); tokenizer blob SHA-256 equals the recorded `tokenizer_sha256`.
5. Non-finite scan over all F32 payloads (typed failure, before any model exists).
6. Rebuild: canonical→HF name mapping via the embedded map, then
   `SetFitMiniLm::from_bundle_parts` + head from stored coefficients + labels.
7. **Probe replay (D-11):** encode probe inputs, predict, compare against embedded expectations
   at contract-resident tolerances. Only on success construct `VerifiedSetFitModel` — the ONLY
   type exposing `classify`, with private constructors (SAFE-03 / APR-04 house style;
   trybuild-provable non-constructibility from outside the crate).

**When to use:** every consumer — `apr predict`, `apr eval`, `apr serve` startup, and the Rust
API — calls this one door (OPS-03/APR-04). The train-time `verify_artifact` additionally runs the
full close/reload/compare policy that MINTS the state (Ph3 D-07).

### Pattern 4: CLI namespace + generic auto-detect (the `apr data` template)

`ExtendedCommands::Data { command: DataCommands }` → `dispatch_analysis.rs` match arm →
`commands::data_contrastive::run_*` is the exact wiring template [VERIFIED: codebase]. `apr setfit`
clones it. Generic auto-detect: `inspect`/`eval`/`predict` read the header+metadata cheaply (the
existing inspect.rs already parses `AprV2Metadata` incl. custom keys), branch on the typed tag,
and route to the core path. `apr predict` is a NEW generic command (none exists today —
[VERIFIED: no `Predict` variant in Commands/ExtendedCommands]).

Exit codes: extend `CliError` (existing categories; `exit_code()` maps to process codes with a
contract macro already attached). JSON output: serialize the D-08 envelope directly — the CLI
must NOT define its own response struct.

### Pattern 5: Serve integration (slot + conditional route + readiness)

`AppState` is a Clone struct of `Option<Arc<...>>` model slots (`apr_model`, `apr_transformer`
precedents) [VERIFIED: api/mod.rs]. Add `setfit_model: Option<Arc<VerifiedSetFitModel>>` behind a
new `setfit` feature (`setfit = ["dep:aprender", "aprender/setfit", "server"]` — note the crate's
core dep is currently optional via the `aprender-serve` feature). Install `/v1/classify` (and the
readiness fields) when the slot is populated; `apr serve` startup detection happens where format
dispatch already lives (`apr-cli/src/commands/serve/handlers.rs::start_realizar_server` detects
format from magic bytes today). Readiness: extend the `/health/ready` response with
`artifact_sha256` + `verified: true` when a SetFit model is loaded (OPS-05).

**Auth caveat (verified, deferred):** `APR_API_KEY*` protects only the CPU fallback router in
`apr-cli/src/commands/serve/auth.rs`, not the aprender-serve `api::router` paths. Out of scope
per CONTEXT deferred list — the plan should note it, not fix it.

### Pattern 6: Parity harness (D-13/D-14)

- In-process HTTP leg: `create_router_with_config(state, cfg).oneshot(request)` — pattern already
  used in `aprender-serve/src/api/tests/app_state_default.rs` [VERIFIED: codebase].
- CLI leg: spawn the built `apr` binary (pin via `scripts/apr_bin.sh` discipline in Make targets;
  in tests use `env!("CARGO_BIN_EXE_apr")`-style resolution, never PATH).
- Library leg: direct core call.
- Compare pairwise on the SAME input set incl. mixed-length batches; exact labels, contracted
  tolerances for probabilities/logits; frozen goldens with SHA-256 manifests (Ph1 D-13).
- **In-band negative:** a deliberately skewed surface variant (e.g., an envelope with one
  probability perturbed beyond tolerance) must FAIL the gate in every `cargo test`.
- ONE tier3 spawned `apr serve` smoke test on a loopback port proves binary wiring
  (startup → readiness → one classify round trip).

### Anti-Patterns to Avoid

- **Reimplementing tokenizer/pooling/model in serve or CLI** — violates OPS-03/D-09; the entire
  point is one core path.
- **Multiple top-level custom metadata keys** — nondeterministic bytes (empirically verified);
  breaks hash stability and round-trip closure.
- **`stamp_provenance_bytes` / post-hoc metadata patching on a setfit-apr-v1** — any bytes changed
  after hashing invalidates the artifact hash chain; provenance goes in at write time.
- **`created_at` or any environment-derived value in metadata** — breaks determinism; leave
  `created_at: None` (it is `skip_serializing_if = "Option::is_none"` so absence is clean).
- **Tensor-name sniffing for detection** — D-04 forbids it; explicit tag only.
- **A `--features` gate with no tier/CI job that RUNS it** — CR-01; SAFE-02 makes this a
  requirement.
- **Make/CI test filters without a ran-something guard** — CR-02: libtest exits 0 on a zero-match
  filter; every new filter must fail if it ran nothing.
- **Echoing the configured device as backend identity** — D-12; read from execution
  (trueno's runtime backend detection).
- **Editing existing contracts inline** — one NEW phase contract referencing existing ones
  (Ph1 D-23); a calibration-regime widening would be a deliberate `pv diff`-flagged edit.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Container bytes (alignment, index, CRC) | A bespoke SetFit file layout | `AprV2Writer`/`AprV2Reader` (via `aprender::format::v2`) | Deterministic sorted index, 64-byte alignment, header+footer CRC32 already proven by 25k+ tests; D-02 maps 1:1 onto its capabilities |
| Artifact hashing | Custom digest scheme | `sha2` via the existing `artifact_hash` free function | Trusted-policy design: a codec must not hash its own output (verify.rs:198–205) |
| Config validation | Field-by-field deserialization + ad-hoc checks | `SetFitTrainConfig`'s existing `try_from` wire-struct path (TOML via workspace `toml`) | T-3-36: serde cannot route around the single validating constructor; already falsified by tests |
| Model rebuild | New tensor→model assembly in CLI/serve | `SetFitMiniLm::from_bundle_parts` + `MultinomialLogisticRegression` accessors | The rebuild path is the one `bundle_rebuilds_a_bit_identical_encoder` proves bit-exact |
| HTTP in-process testing | Port-binding test servers everywhere | tower `ServiceExt::oneshot` on the real Router | Already in-tree; D-14 chose it to avoid the flaky-gate failure mode |
| Canonical test access | New eval gating for `apr eval` | `create_selection_lock` → `mint_test_token` → `CanonicalTestAccess::grant` (Ph3 D-14) | D-16 exists precisely to consume this as shipped |
| Contract validation tooling | bash/yq/python YAML scripts | `pv` (`$(PV_BIN)` via cargo run) | CLAUDE.md hard rule; workarounds are rejected |
| Device grammar/probing | New device parsing in CLI | `resolve_device` | The grammar+probe split is contract-relevant and already fail-closed |

**Key insight:** the phase's risk is not missing machinery — it is *breaking determinism or
weakening a check while wiring existing machinery together*. Every hard bug class here (CR-01..04,
the HashMap ordering, non-finite JSON nulls) is a check that looks stronger than it is.

## Common Pitfalls

### Pitfall 1: Nondeterministic metadata order breaks hash stability AND round-trip closure
**What goes wrong:** `AprV2Metadata.custom` is `HashMap<String, serde_json::Value>`; with >1 key
the serialized JSON key order varies per process (empirically verified this session: three runs of
one binary produced three orders). Artifact hash then differs across identical runs, and the
trusted policy's `serialize(deserialize(bytes)) == bytes` check (`ReloadNotFromBytes`) fails
intermittently — the worst kind of red.
**Why it happens:** std `HashMap` uses `RandomState`; serde serializes in iteration order; the
existing converter code already inserts many `tokenizer.*` custom keys this way for LLM imports
(nobody there needs byte-determinism).
**How to avoid:** one custom key whose value is a `serde_json::Map` object (BTreeMap-sorted by
default; `preserve_order` is not enabled anywhere in the workspace — verified). All other
identity via typed declaration-order fields (`model_type`).
**Warning signs:** same training run producing different artifact hashes; flaky
`ReloadNotFromBytes` failures.

### Pitfall 2: The byte-canonical obligation is broader than "no timestamps"
**What goes wrong:** `serialize` must be a pure deterministic function of the bundle, and
`deserialize` must recover EVERYTHING `serialize` needs. Any writer input not derivable from the
bundle (creation time, host name, writer version string not pinned, float formatting, map
ordering) breaks closure. The obligation is stated twice in verify.rs on purpose: "a phase-4
writer with padding, unordered metadata or a checksum placed after the payload would fail
`ReloadNotFromBytes` and be blocked until someone edited the contract."
**Why it happens:** container formats accrete provenance conveniences (`created_at`,
`stamp_provenance_bytes`) that are invisible until the closure check runs.
**How to avoid:** the codec's writer sets: `created_at: None`, fixed `version` string from the
schema constant, tensors from the bundle's `BTreeMap` (already sorted; writer re-sorts by name
anyway), one custom key per Pitfall 1; padding is `Vec::resize(_, 0)` — deterministic
[VERIFIED: writer.rs]. Add a dedicated test: write→parse→write, assert byte equality, run it
twice in separate processes (the cross-process half is what catches HashMap ordering).
**Warning signs:** closure test passes locally but fails in CI (different process ⇒ different
random state).

### Pitfall 3: Tokenizer must be byte-exact — the existing `tokenizer.*` metadata pattern is not
**What goes wrong:** the in-tree GGUF/SafeTensors import path stores tokenizer *vocabulary
arrays* in custom metadata (`insert_f32_tokenizer_metadata`) — a lossy re-encoding. APR-01/APR-02
require the exact `tokenizer.json` bytes with matching SHA-256, no sidecars.
**How to avoid:** store the raw bytes as a `TensorDType::U8` entry (dtype exists —
[VERIFIED: tensor_index_impl.rs, U8 = 7]); recover via `get_tensor_data` (raw byte slice); verify
SHA-256 against the recorded hash at load; compare against `SetFitMiniLm::tokenizer_sha256()`
after rebuild. Choose a name that generic tensor tooling tolerates (it will appear in
`apr tensors` listings — acceptable; D-03's structural check must special-case its dtype width 1).
**Warning signs:** a loader that "reconstructs" tokenizer JSON from parts; tokenizer hash checks
that compare recomputed structures instead of bytes.

### Pitfall 4: Vacuous gates — zero-match filters, unwired tests, uncompiled features
**What goes wrong:** three of Phase 3's four review blockers were this class: tests in no tier/CI
job (CR-01), libtest exiting 0 on a zero-match filter (CR-02), a threshold check that stays green
while widening (CR-04).
**How to avoid:** every new Make target/CI filter asserts it ran a nonzero count (parse the
`N passed` line); every new feature-gated surface gets a named tier target AND a ci.yml step that
compiles and RUNS it (apr-cli `setfit` feature and aprender-serve `setfit` feature are the two new
surfaces); `$(CONTRACTS)` in the Makefile is an **explicit list** — `setfit-apr-v1.yaml` must be
appended or it is validated by nothing (Ph2 lesson, verified at Makefile:1138).
**Warning signs:** "green" runs whose logs show 0 tests; a contract file that no target names.

### Pitfall 5: serde_json renders non-finite floats as `null` (CR-03's shape, again)
**What goes wrong:** probe expectations, logits, margins in the JSON doc or the HTTP envelope:
`f32::NAN`/`INFINITY` silently become `null` on serialize and fail (or worse, default) on parse —
non-injective encodings and undetectable divergence.
**How to avoid:** bit-pattern hex for stored floats (bundle.rs precedent); typed rejection of
non-finite values before serialization in the envelope (`within` in verify.rs shows the
NaN-visible comparison idiom); keep `float_roundtrip` (already enabled in core+train — any new
crate serializing envelopes must inherit it via feature unification, which it does workspace-wide).
**Warning signs:** `null` appearing in numeric JSON fields; comparisons written as `delta <= bound`
(false for NaN ⇒ silently accepts).

### Pitfall 6: Host quirks make honest local verification harder
**What goes wrong / how to avoid (all pre-measured, none are regressions):**
- `cargo check --workspace` is RED on Darwin (aprender-profile `compile_error!` on non-Linux) —
  use `--exclude aprender-profile` [STATE.md, control-verified at 02-08].
- `make tier2` is RED on arm64 (24 pre-existing clippy errors in 5 untouched crates, D-ITEM-02) —
  scope clippy assertions to touched crates.
- CB-510 guard scripts pass vacuously on macOS (BSD grep, D-ITEM-01) — CI covers them.
- repo-wide `make contract-audit` prints 132 BIND-001 errors and exits 0 (D-ITEM-04) — use scoped
  audits like `contract-audit-phase2`'s pattern.
- `target/debug/incremental` regrows to ~25 GB (two ENOSPC stops in Phase 2) — set
  `CARGO_INCREMENTAL=0` for long plans.
- `cargo package -p apr-cli` (with or without `--no-verify`) is KNOWN-RED until the human-run
  publish cascade — verify-work must read it as expected.
- rtk hook rewrites `git status --porcelain` — emptiness assertions must use `rtk proxy`.
- aprender-train full suite has a 24-test known-red baseline (byte-for-byte in
  `known-red-baseline.md`) — scoped `setfit::` runs are the honest gates.

### Pitfall 7: Mutation-testing tooling constraints (measured in Phase 3)
**What goes wrong:** `--in-place` conflicts with `--jobs` in cargo-mutants 25.3.1, and
`--timeout 20` kills the BASELINE (aprender-core's 14,285-test binary had not finished linking in
20 s). 1,181 unscoped mutants ≈ 44.6 h.
**How to avoid:** scope `cargo mutants` to new files only (`-f` globs on the new modules),
`--timeout >= 120`, tree-copy mode (no `--in-place` with `--jobs`); inherit 03-HUMAN-UAT's
constraints verbatim; expect proven-equivalent survivors (see memory: most survivors here are
equivalence, not gaps).

### Pitfall 8: Probe records can smuggle dataset text into the artifact
**What goes wrong:** D-11 probes are "input text → expected outputs". If probe inputs are sampled
from the training selection (as train-time `VerifyProbe` rows are), the shipped artifact embeds
tweet text — colliding with the project's deliberate "no vendored TweetEval text" licensing
posture (DATA-01, Out of Scope table).
**How to avoid:** probe-record selection policy is Claude's discretion — recommend a small FIXED
set of synthetic, committed probe strings (contract-resident, e.g., 3–8 strings covering
short/long/truncation-boundary/unicode cases) computed at train time through the verified model.
This also makes probes dataset-independent, which serves Phase 5's cross-cell comparisons.
**Warning signs:** probe inputs derived from `selection.examples()`.

### Pitfall 9: `verify_artifact` mints the state; consumers must not grow a second minting path
**What goes wrong:** APR-04 says every consumer accepts only `ArtifactReloadedAndVerified`. The
consumer-side equivalent (load + probe replay) could drift into a *second* verification policy
with its own tolerances.
**How to avoid:** the core loader's probe-replay tolerances and the train-time policy tolerances
live in ONE contract (`setfit-apr-v1.yaml`) and one Rust constants module; the loader's success
type (`VerifiedSetFitModel`) has private constructors, trybuild-proven non-constructible from
outside — mirroring `SetFitRun`'s house style. The probe tolerances must not contradict
`setfit-encoder-conformance-v1.yaml`'s frozen table (e.g., forward tolerance 1.52587891e-05 for
per-layer, 7.62939453e-06 family for pooled outputs — read the contract, don't re-derive).

## Code Examples

Verified patterns from the codebase (paths absolute from repo root):

### Deterministic one-key metadata construction
```rust
// Pattern for aprender-core/src/setfit/artifact.rs (serde_json::Map is BTreeMap-backed
// by default — sorted keys, deterministic bytes; verified: preserve_order not enabled)
let mut doc = serde_json::Map::new();
doc.insert("schema".into(), "setfit-apr-v1".into());
doc.insert("schema_version".into(), 1u32.into());
doc.insert("ordered_labels".into(), serde_json::to_value(labels)?);
doc.insert("tokenizer_sha256".into(), tokenizer_sha256.into());
// ... hf_name_map, preprocessing, resolved_config, evidence, provenance, probes ...

let mut meta = AprV2Metadata { model_type: "setfit".to_string(), ..Default::default() };
// EXACTLY ONE custom key — see Pitfall 1. created_at stays None — see Pitfall 2.
meta.custom.insert("setfit".to_string(), serde_json::Value::Object(doc));

let mut writer = AprV2Writer::new(meta);
writer.set_header_flags(AprV2Flags::new().with(AprV2Flags::LAYOUT_ROW_MAJOR));
for (canonical_name, (shape, data)) in tensors { writer.add_f32_tensor(canonical_name, shape, &data); }
writer.add_tensor("tokenizer.blob", vec![tok_bytes.len()], TensorDType::U8, &tok_bytes);
let bytes = writer.write()?;   // sorted index, deterministic offsets, CRC32 footer [writer.rs:317]
```

### In-process HTTP leg (existing pattern)
```rust
// Source: crates/aprender-serve/src/api/tests/app_state_default.rs (tower util already a dep)
use tower::util::ServiceExt;
let app = create_router_with_config(state, RouterConfig::default());
let resp = app.oneshot(
    Request::builder().method("POST").uri("/v1/classify")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&req)?))?,
).await?;
```

### Fail-closed limit enforcement before allocation (the house pattern to copy)
```rust
// Source: crates/aprender-train/src/train/setfit/bundle.rs:456-524 — order is load-bearing:
// (1) raw input length vs cap, (2) parse, (3) schema version, (4) counts/sizes computed
// from declared lengths BEFORE decoding payloads. The core loader mirrors this ladder
// against the APR header/index instead of hex strings (D-03 structural check =
// TensorIndexEntry.size == product(shape) * dtype_width, checked before reading data).
```

### Config-file-first training config (already works)
```rust
// SetFitTrainConfig derives Deserialize with #[serde(try_from = "SetFitTrainConfigWire")]
// — every deserialization routes through SetFitTrainConfig::new (config.rs:16-27).
let cfg: SetFitTrainConfig = toml::from_str(&fs::read_to_string(path)?)   // or serde_json
    .map_err(|e| CliError::ValidationFailed(format!("--config: {e}")))?;   // invalid VALUES rejected too
```

### Execution-derived backend identity (D-12 source)
```rust
// Source: crates/aprender-compute/src/lib.rs:281-312 — trueno performs runtime CPU feature
// detection (AVX2+FMA / AVX / SSE2 / NEON / WASM) and selects a Backend. The classify path
// should surface the ACTIVE backend from this layer into the envelope, not echo config.
// resolve_device (aprender-train/src/train/device.rs:110) stays the fail-closed request gate.
```

## State of the Art

| Old Approach (Phase 3) | Current Approach (Phase 4) | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `SerdeJsonCodec` hex-JSON interim format (~182 MB, "never claims to be the shipped container") | `AprCodec` writing real `setfit-apr-v1` APR v2 (~90 MB) behind the same seam | this phase | Format swap only; verify policy, hashing, minting unchanged (Ph3 D-07) |
| Verification is a train-time memory | Every fresh consumer re-proves via embedded probe replay (D-11) | this phase | "Served model is the evaluated model" becomes per-process re-provable |
| `setfit` feature on core+train only | Feature propagates to apr-cli and aprender-serve | this phase | SAFE-02 matrix grows two crates; CR-01 discipline applies to both |
| No CLI training/predict surface | `apr setfit train` + generic `apr predict` + auto-detect | this phase | Closes TRN-07 positive half via `apr eval` (D-16) |

**Deprecated/outdated (do not imitate):**
- `insert_f32_tokenizer_metadata`'s vocabulary-array metadata — lossy; SetFit needs exact bytes.
- `/v1/predict` (`features: Vec<f32>`) — wrong request shape for text classification.
- The crate-level `crates/aprender-train/CLAUDE.md` — pre-monorepo entrenar doc; root CLAUDE.md
  governs.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | serde_json's default `Map` (BTreeMap-backed, sorted) is used when building `serde_json::Value::Object` via `serde_json::Map::new()`, and no workspace crate enables `preserve_order` (grep found none) — so the one-key doc serializes deterministically | Pattern 2, Pitfall 1 | If some dependency enables `preserve_order` via feature unification, Map becomes insertion-ordered (still deterministic if insertion order is fixed — mitigate by inserting in sorted order anyway). Cheap Wave-0 test: write-parse-write byte equality across two processes |
| A2 | `Tolerance::EXACT` will hold for the APR codec's train-time round trip (raw LE f32 bytes are bit-exact; same CPU path re-encodes) | Pattern 1 | If any rebuild-path divergence appears, the tolerance must be introduced in trusted code keyed on codec — a design the `Tolerance` type explicitly anticipates. Falsified immediately by the first `verify_artifact` integration test |
| A3 | A `U8` tensor entry survives generic `apr tensors`/`qa`/`diff` tooling without crashes (dtype is declared in the enum; tools were not exhaustively audited against a U8 entry) | Pitfall 3 | If some generic tool assumes numeric-only tensors, it surfaces as a Phase 4 test failure; fallback is a dedicated blob naming convention + tool allow-list, still no sidecar |
| A4 | apr-cli can take a dev-dependency on aprender-serve (with `setfit` feature) for the parity harness without creating a dependency cycle | Responsibility Map | If a cycle exists (serve → cli somewhere), site the harness in aprender-serve tests or a dedicated test crate instead; verify with `cargo tree` at plan time |
| A5 | The pinned MiniLM pretrained weights are available locally for `apr setfit train` input (Phase 1 acquisition path; `SetFitMiniLm::from_pretrained_dir(dir, root_seed)` is the door) — the CLI will need a `--model-dir`-style flag or a pinned default location | Pattern 4 | If no offline acquisition path exists for users, the train command's UX needs a documented prerequisite; does not affect architecture |

All other factual claims in this document are [VERIFIED: codebase] (read on branch
`gsd/phase-2-contract-gate` @ d66678e7a during this session) or [VERIFIED: empirical test]
(HashMap ordering).

## Open Questions (RESOLVED)

All five questions were resolved during planning; each recommendation was adopted verbatim by the
plan set. Inline markers cite the adopting plan tasks.

1. **Exact canonical tensor names for the BERT family under `tensor-names-v1.yaml`**
   - What we know: the contract maps semantic keys (e.g. `token_embedding_weight`) to per-arch
     alias lists with `bert:` entries and `_fallback` canonical forms (e.g. `token_embd.weight`);
     `map_tensor_names` machinery exists for the GGUF→APR path (Qwen2-focused today).
   - What's unclear: the full 101-tensor canonical name set for this encoder (per-layer patterns
     with `{n}`), and whether `map_tensor_names` supports `Architecture::Bert` or the SetFit
     writer derives names directly from the contract table.
   - Recommendation: the plan's first artifact task derives the complete canonical↔HF table from
     the contract (via `pv status`/reading, not re-invention), commits it as the embedded
     `hf_name_map`, and validates completeness in the writer (missing mapping = typed error).
   - **RESOLVED:** adopted by **04-01 Task 1** — the canonical↔HF name table is derived from
     `tensor-names-v1.yaml` and committed into `contracts/setfit-apr-v1.yaml`; the writer
     validates completeness with a typed error (04-02 Task 1 consumes the table as `hf_name_map`).
2. **The D-03 hard-cap constant**
   - What we know: legitimate artifact ~90 MB; the bundle precedent chose ~2.9x headroom
     (512 MiB over ~182 MB).
   - Recommendation: 256 MiB (~2.8x over 90 MB), contract-resident in `setfit-apr-v1.yaml` with
     the derivation recorded; plus the existing 16 MB `MAX_METADATA_SIZE` as the metadata bound.
   - **RESOLVED:** adopted — 256 MiB (268435456) is contract-resident in `setfit-apr-v1.yaml`
     (**04-01 Task 1**) and enforced as rung 1 of `load_setfit_apr`/`read_setfit_apr_parts`
     before any parse (**04-03 Task 1**).
3. **HTTP route naming**
   - What we know: `/v1/predict` is taken (features-based); D-10 fixed mechanism not URL.
   - Recommendation: `POST /v1/classify` with the D-08 envelope; readiness additions on
     `/health/ready`.
   - **RESOLVED:** adopted — `POST /v1/classify` serializing the core D-08 envelope, plus
     `/health/ready` classifier additions (**04-08 Task 1**; parity-exercised in 04-09).
4. **Where `apr serve` startup detection meets aprender-serve's AppState**
   - What we know: `start_realizar_server` (apr-cli) detects format by magic; aprender-serve owns
     `AppState`/router; the SetFit slot must be populated before router creation.
   - What's unclear: whether the SetFit serve path threads through the existing realizar-server
     startup or a parallel branch in the same function.
   - Recommendation: branch inside the existing APR-format arm after reading the typed tag —
     one detection point, no new serve command (D-10).
   - **RESOLVED:** adopted — the branch lives inside the existing APR-format arm after the typed
     `model_type == "setfit"` tag read; one detection point, no new serve command
     (**04-08 Task 2**).
5. **Contract shape** — default per Ph1 D-23: one new `contracts/setfit-apr-v1.yaml` owning
   artifact schema, load-validation ladder, cap, probe policy, parity tolerances; referencing
   `setfit-train-lifecycle-v1`, `setfit-encoder-conformance-v1`, `tensor-names-v1`,
   `tensor-layout-v1`, `apr-model-lifecycle-v1`. Must be appended to `$(CONTRACTS)` and validated
   via `pv` (schema kinds: if `pv validate` rejects, restructure to `KernelContract` shape per
   CLAUDE.md's three sanctioned options).
   - **RESOLVED:** adopted — one new `contracts/setfit-apr-v1.yaml` authored and pv-validated in
     **04-01 Tasks 1–2**, appended to `$(CONTRACTS)` with the blocking `contract-audit-phase4`
     gate; the KernelContract-restructure fallback is written into 04-01 Task 1's action.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| rustc / cargo | build | ✓ | 1.93.0 | — |
| cargo-nextest | CI-parity local runs | ✓ | 0.9.102 | `cargo test` |
| cargo-mutants | mutation gates | ✓ | 25.3.1 | — (constraints in Pitfall 7) |
| `pv` (contracts CLI) | contract validation | ✓ via `cargo run -p aprender-contracts-cli --bin pv` (`$(PV_BIN)`, Makefile:1136); NOT on PATH as a binary | in-tree | — |
| `apr` binary | dogfood/parity smoke | build from HEAD; MUST pin via `. scripts/apr_bin.sh` | HEAD | — |
| trybuild/proptest/insta/tempfile | tests | ✓ (aprender-train dev-deps) | workspace | — |
| toml crate | D-07 config parsing | ✓ workspace dep 0.8 (add `workspace = true` to apr-cli) | 0.8 | JSON-only (worse UX) |
| tower `util` (oneshot) | D-14 in-process leg | ✓ (aprender-serve dep, used in tests) | 0.5 | — |
| Network / Python / HF hub | — | NOT required (offline phase by design) | — | — |
| Darwin host caveats | local gates | see Pitfall 6 (workspace-check exclude, arm64 clippy red, vacuous CB-510 guards, incremental-cache ENOSPC) | — | scoped commands |

**Missing dependencies with no fallback:** none.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | cargo test (libtest) + trybuild + proptest; cargo-nextest 0.9.102 in CI; cargo-mutants 25.3.1 scoped |
| Config file | Makefile tiers (tier1–tier4) + `.github/workflows/ci.yml` (setfit steps at ci.yml:274–309) |
| Quick run command | `cargo test -p aprender-train --features setfit --lib setfit::` (and `-p aprender-core --features setfit --lib setfit::`) |
| Full suite command | `make tier3` (includes `setfit-tests`, `setfit-repro-*`, `setfit-feature-matrix`, `contrastive-data-boundary`, contract validation via `$(PV_BIN)`) |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command (scoped, <30s where possible) | File Exists? |
|--------|----------|-----------|-------------------------------------------------|-------------|
| APR-01 | Write complete checksummed artifact | unit | `cargo test -p aprender-core --features setfit --lib setfit::artifact::` | ❌ Wave 0 |
| APR-02 | Fail-closed load (malformed/oversized/incomplete/non-finite/inconsistent) | unit (one test per rung + induced corruption) | same filter | ❌ Wave 0 |
| APR-03 | close→reload→verify round trip via AprCodec | integration (train crate) | `cargo test -p aprender-train --features setfit --lib setfit::apr_codec::` | ❌ Wave 0 |
| APR-04 | Consumers reject unverified models | trybuild (non-constructibility) + unit | `cargo test -p aprender-train --features setfit --test ui` (extend) + core trybuild for `VerifiedSetFitModel` | ❌ Wave 0 |
| APR-05 | Inspect recovers all identity fields | unit + CLI integration | `cargo test -p apr-cli --features setfit --lib inspect` (scoped) | ❌ Wave 0 |
| OPS-01 | Stable fallible Rust API, no CLI deps | unit + `cargo tree` boundary check (Ph2 D-04 pattern) | new Make target with ran-something guard | ❌ Wave 0 |
| OPS-02 | Full CPU CLI lifecycle | integration (spawned `apr`) | tier3 target; ONE end-to-end lifecycle test | ❌ Wave 0 |
| OPS-03 | Auto-detect routes to shared core path | unit (detection) + negative (untagged APR is plain) | scoped CLI tests | ❌ Wave 0 |
| OPS-04 | Envelope field parity by type | unit (serialization goldens) + type-level (one struct) | core envelope tests | ❌ Wave 0 |
| OPS-05 | HTTP classify + readiness reports hash | in-process oneshot tests (every `cargo test`) + ONE tier3 spawned smoke | `cargo test -p aprender-serve --features setfit --lib setfit` | ❌ Wave 0 |
| OPS-06 | Explicit unavailable device fails; backend honest | unit (existing `resolve_device` tests + new identity-from-execution test) | scoped | partially ✅ (device.rs tests exist) |
| SAFE-01 | Executable contracts + parity gate + in-band negative | contract (`pv validate`) + parity harness in every `cargo test` | `$(PV_BIN) validate contracts/setfit-apr-v1.yaml` + parity test filter with guard | ❌ Wave 0 |
| SAFE-02 | CPU feature matrix incl. new crates | build matrix | extend `setfit-feature-matrix` (+ ci.yml step — human-visible edit) with apr-cli + aprender-serve legs | partially ✅ (core+train legs exist) |

### Sampling Rate
- **Per task commit:** scoped `setfit::` filters for the touched crate(s) — each with a
  nonzero-ran guard (CR-02).
- **Per wave merge:** `make tier2` (scoped-clippy caveat on arm64) + all scoped setfit suites.
- **Phase gate:** `make tier3` green (setfit targets + contracts + feature matrix) before
  `/gsd:verify-work`; parity harness + in-band negative running in default `cargo test`.

### Wave 0 Gaps
- [ ] `aprender-core/src/setfit/artifact.rs` + tests — APR-01/02, loader ladder, probe replay
- [ ] `aprender-core/src/setfit/classify.rs` + envelope goldens — OPS-04/D-08
- [ ] `aprender-train/src/train/setfit/apr_codec.rs` + closure/determinism tests (two-process
      byte-equality) — APR-03, Pitfalls 1–2
- [ ] trybuild cases for `VerifiedSetFitModel` non-constructibility — APR-04
- [ ] apr-cli: `setfit` feature, `SetfitCommands`, `predict` command, auto-detect branches + tests
- [ ] aprender-serve: `setfit` feature, route, AppState slot, oneshot tests; tier3 spawned smoke
- [ ] `contracts/setfit-apr-v1.yaml` + `$(CONTRACTS)` append + `pv validate` in tier3
- [ ] Parity harness + frozen goldens (SHA-256 manifests) + in-band skewed negative
- [ ] Make targets with ran-something guards; ci.yml matrix extension (flag as human-visible)

## Security Domain

`security_enforcement: true`, ASVS level 1. This phase's attack surface is (a) hostile artifact
bytes handed to the loader, and (b) the new HTTP classify route.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no (deferred by user decision — `APR_API_KEY*` covers only the CPU fallback router; unifying auth is explicitly out of scope) | documented accepted gap |
| V3 Session Management | no | stateless HTTP |
| V4 Access Control | partial | canonical-test access via typestate token (D-16) — capability-based, compile-time enforced |
| V5 Input Validation | yes | serde `deny_unknown_fields` + validated wire structs (config), bounded fail-closed artifact parsing (D-03 cap + structural checks BEFORE allocation, bundle.rs precedent), typed request deserialization on the HTTP route; add a contracted request-body/batch-size bound on `/v1/classify` mirroring the D-03 discipline |
| V6 Cryptography | yes | `sha2` SHA-256 for artifact + tokenizer hashes — never hand-rolled; hash computed by trusted free function, never by the codec |

### Known Threat Patterns for this stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Decompression/allocation bomb in artifact | DoS | Raw-length cap before parse; element counts derived from index BEFORE `Vec` allocation; per-tensor and total bounds (BundleLimits pattern); 16 MB metadata cap already in container |
| Corrupted-but-checksummed artifact (CRC32 is not cryptographic) | Tampering | SHA-256 artifact hash as identity anchor; D-11 probe replay catches semantic corruption CRC/hash cannot |
| Foreign/legacy APR masquerading as SetFit | Spoofing | Explicit-tag-only detection (D-04) + schema-version refusal + structural consistency |
| Non-finite poisoning (NaN weights → NaN probabilities) | Tampering | Load-time non-finite scan (typed failure before prediction); NaN-visible comparisons (`partial_cmp` idiom) |
| Unbounded HTTP batch | DoS | Contracted max batch size / body size on the classify route (envelope rejects oversized requests with a typed error) |
| Backend misreporting | Repudiation | D-12: identity read from execution layer, never config echo |

## Sources

### Primary (HIGH confidence — read directly this session, branch `gsd/phase-2-contract-gate` @ d66678e7a)
- `crates/aprender-train/src/train/setfit/{bundle.rs, verify.rs, config.rs, mod.rs, evidence.rs, lock.rs, evaluate.rs}` — codec seam, trusted policy, closure obligation, limits precedent, lifecycle API
- `crates/apr-format/src/v2/{mod.rs, writer.rs, header_impl.rs, reader_impl.rs, tensor_index_impl.rs}` — container capabilities, determinism analysis, dtypes, metadata struct
- `crates/aprender-core/src/setfit/mod.rs`, `crates/aprender-core/Cargo.toml`, `crates/aprender-core/src/format/mod.rs` — core model API, `setfit` feature, v2 re-export
- `crates/apr-cli/src/{commands_enum.rs, extended_commands.rs, data_commands.rs, dispatch_analysis.rs, error.rs, commands/inspect.rs, commands/serve/mod.rs, commands/serve/handlers.rs}`, `crates/apr-cli/Cargo.toml` — CLI wiring, features, no-Predict-today, serve startup
- `crates/aprender-serve/src/api/{router.rs, mod.rs, apr_handlers.rs}`, `crates/aprender-serve/Cargo.toml`, `src/api/tests/app_state_default.rs` — routes, AppState, oneshot pattern
- `crates/aprender-train/src/train/device.rs`, `crates/aprender-compute/src/lib.rs` — device gate, backend detection
- `Makefile` (tiers, `$(CONTRACTS)`, `PV_BIN`, setfit targets), `.github/workflows/ci.yml` (setfit steps)
- `contracts/` listing + `setfit-encoder-conformance-v1.yaml` tolerance table + `tensor-names-v1.yaml` bert entries
- `.planning/{REQUIREMENTS.md, STATE.md, codebase/ARCHITECTURE.md}`, `04-CONTEXT.md`, Phase 2/3 lessons in STATE.md
- Empirical: HashMap iteration-order test (3 process runs, 3 orders) — scratchpad, this session

### Secondary (MEDIUM)
- `.claude/skills/{apr-dogfood, pre-release}/SKILL.md` — release-gate constraints (KNOWN-RED Gate 5, binary pinning)

### Tertiary (LOW / training knowledge, flagged)
- serde flatten-serialization ordering details (A1) — mitigated by Wave-0 byte-equality test

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all in-tree, read directly; zero new external packages
- Architecture: HIGH — the codec-adapter shape is literally prescribed by verify.rs docs; container capabilities verified against D-02 needs
- Pitfalls: HIGH — determinism pitfall empirically verified; CR-01..04-class pitfalls are measured history on this branch
- Open questions: MEDIUM — canonical bert name table and serve-startup threading need plan-time reading, not new research

**Research date:** 2026-08-14
**Valid until:** stable while the branch head stays at/near d66678e7a — re-verify file:line references if the branch moves substantially (~30 days for the architectural findings)
