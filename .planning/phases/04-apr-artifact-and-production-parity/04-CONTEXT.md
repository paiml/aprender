# Phase 4: APR Artifact and Production Parity - Context

**Gathered:** 2026-08-14
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase delivers **the production half of the milestone**: the real `setfit-apr-v1` write path
plugged into Phase 3's sealed `SetFitCodec` seam, the production core loader with fail-before-predict
validation, the train→save→reload→verify round trip (APR-03), and every consumer surface — Rust API
(OPS-01), `apr` CLI lifecycle (OPS-02), generic-command auto-detection (OPS-03), classify outputs
(OPS-04), and native HTTP serving (OPS-05) — all CPU-first (OPS-06), offline, gated on
`ArtifactReloadedAndVerified` (APR-04), with parity proven across the three surfaces by executable
contracts (SAFE-01) and a CI-verified CPU feature matrix (SAFE-02).

Requirements: APR-01..APR-05, OPS-01..OPS-06, SAFE-01, SAFE-02 — plus, by user decision here,
the **positive half of TRN-07** (see D-16).

**In scope:** the `setfit-apr-v1` schema and writer/loader; APR inspection recovery (APR-05); the
`apr setfit train` command; SetFit auto-detection in generic `apr inspect/eval/predict`; the classify
HTTP surface inside `apr serve`; the three-surface parity gate; load-time self-verification; the CPU
feature-matrix CI wiring.

**Not in scope (and why):**
- The 40-cell benchmark matrix, F_avg claims, LoRA comparison — Phase 5 (EVAL-01..05). Phase 4 only
  guarantees the artifacts and surfaces Phase 5 will measure.
- Converging `apr embed` / `apr rerank` / `CrossEncoder` onto the graph-connected encoder — narrowed
  OUT of this phase by D-15 (a deliberate, recorded amendment of Ph1 D-02's deferral). Deferred item.
- GPU/accelerator inference paths — v2 (ACC-01); OPS-06 requires only that an explicitly requested
  unavailable device fails closed.
- Quantized artifacts — F32 only (APR-01); quantization is v2 (QUANT-01).
- Re-litigating trainer, pair, or encoder semantics — settled in Phases 1–3.

**Phase 3 handoff state:** Phase 3 UAT is COMPLETE (5 passed, 0 issues, commit 81d163c85). All four
verification-layer review blockers are resolved: the SetFit surface now runs in tier3 AND CI
(CR-01, commits a844f6a98 + 52357404b), the replay gate is wired and refuses a zero-match filter
(CR-02, 1038f6414), non-finite digests are typed failures (CR-03, 0158a758d), and the provenance
gate compares regime ids (CR-04, 51db85ace). Phase 4 builds on that verified base — STATE.md's
`human_needed` narrative predates these commits.

</domain>

<decisions>
## Implementation Decisions

### Artifact Internals (`setfit-apr-v1`)

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

### CLI Surface

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

### Serving Boundary

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

### Parity Gate

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

### Scope Rulings

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

### Carried Forward (not re-litigated)

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

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase requirements and prior decisions
- `.planning/ROADMAP.md` § "Phase 4: APR Artifact and Production Parity" — goal and the five
  success criteria this phase is judged against.
- `.planning/REQUIREMENTS.md` — APR-01..APR-05, OPS-01..OPS-06, SAFE-01, SAFE-02 verbatim; plus
  TRN-07's qualifier text (the deliberately unchecked positive half D-16 closes).
- `.planning/PROJECT.md` — Core Value, Constraints (artifact integrity, performance,
  compatibility), Out of Scope list.
- `.planning/phases/03-faithful-two-stage-trainer-and-head/03-CONTEXT.md` — Phase 3 D-01..D-16.
  D-07 (sealed codec seam — the exact seam this phase implements), D-08 (encode-once head input),
  D-12 (evidence record schema), D-14 (selection lock + token minting) are load-bearing here.
- `.planning/phases/03-faithful-two-stage-trainer-and-head/03-HUMAN-UAT.md` — status: complete,
  5 passed, 0 issues; records the mutation-tooling constraints (timeout >= 120, no --in-place with
  --jobs) any Phase 4 mutation plan inherits.
- `.planning/phases/03-faithful-two-stage-trainer-and-head/03-REVIEW.md` — the four
  verification-layer blockers (CR-01..04) and their standard of proof; all resolved on this branch.
- `.planning/phases/02-deterministic-pair-and-data-protocol/02-CONTEXT.md` — Phase 2 D-01..D-27.
  D-04 (bytes boundary), D-08 (selected-ID manifest as materialized artifact), D-16 (split
  typestate + ledger) are load-bearing here.
- `.planning/phases/01-differentiable-minilm-conformance/01-CONTEXT.md` — Phase 1 D-01..D-27.
  D-02 (convergence deferral — amended by this phase's D-15), D-14 (tolerance discipline),
  D-18 (HF dotted names), D-19 (tensor-names mapping at THIS phase's write boundary),
  D-23/D-24/D-26 (contract/negative/tier discipline).
- `.planning/phases/02-deterministic-pair-and-data-protocol/deferred-items.md` — D-ITEM-01..04
  pre-existing repo defects that affect how gates verify locally (vacuous macOS guards, arm64
  clippy red, tier2 zero-test headline, contract-audit exit-0).

### Contracts (use `pv`, never bash/yq/python)
- `contracts/setfit-train-lifecycle-v1.yaml` — Phase 3's gate: lifecycle stages, evidence
  thresholds, calibration regimes. Phase 4's artifact carries its evidence record; the Phase 5
  blocker note in ROADMAP.md (production-encoder regime calibration) lives here.
- `contracts/setfit-encoder-conformance-v1.yaml` — Phase 1's frozen tolerance table; Phase 4's
  probe/parity tolerances must not contradict it.
- `contracts/tensor-names-v1.yaml` — the canonical naming scheme D-01 writes under (Ph1 D-19).
- `contracts/tensor-layout-v1.yaml` — row-major is mandatory; applies to every tensor written.
- `contracts/apr-model-lifecycle-v1.yaml` — existing APR lifecycle phases the new artifact must
  coexist with.
- `contracts/contrastive-pair-protocol-v1.yaml`, `contracts/tweet-eval-stance-benchmark-v1.yaml` —
  data-side contracts whose manifests/fingerprints the artifact's provenance references.
- `contracts/linear-probe-classifier-v1.yaml` — the labeled-baseline contract; SAFE-03's
  distinctly-named types must stay distinct through the artifact and CLI.

### Code the phase builds on
- `crates/aprender-train/src/train/setfit/bundle.rs` — the sealed `SetFitCodec` seam and complete
  bundle (Ph3 05/08); the APR codec adapter implements against this.
- `crates/aprender-train/src/train/setfit/verify.rs` + `verify_tests.rs` — the round-trip
  verification policy that mints `ArtifactReloadedAndVerified`; Phase 4 swaps the format, not this.
- `crates/aprender-train/src/train/setfit/lock.rs`, `evaluate.rs` — selection lock, trusted
  evaluator, token minting (D-16's substrate).
- `crates/aprender-train/src/train/setfit/config.rs` — the validated 12-knob `SetFitConfig` that
  `--config` deserializes into (D-07).
- `crates/aprender-train/src/train/setfit/evidence.rs` — the evidence summary/table types the
  artifact's JSON doc embeds (Ph3 D-12).
- `crates/aprender-core/src/setfit/` — `SetFitMiniLm`, encoder, tokenizer, import; the shared core
  model path every surface routes to (OPS-03, D-09).
- `crates/apr-format/src/v2/` — the APR v2 container (header, typed metadata, checksums,
  reader/writer family) the artifact is written into.
- `crates/aprender-core/src/format/` + `src/serialization/apr/` — format semantics, converter
  (Ph1 D-19's mapping machinery), atomic writer.
- `crates/apr-cli/src/dispatch_run.rs`, `dispatch.rs`, `dispatch_analysis.rs`,
  `commands_enum.rs` — the command wiring pattern for `apr setfit` and the auto-detect routing.
- `crates/apr-cli/src/commands/data_contrastive.rs` — Phase 2's CLI adapter precedent (attested
  ingest, atomic writes, pure filesystem adapter) that `apr setfit train` follows.
- `crates/aprender-serve/src/api/router.rs` + `api/mod.rs` — the router and `AppState` the classify
  route installs into (D-10); note the auth caveat: `APR_API_KEY*` protects only the CPU fallback
  router in `crates/apr-cli/src/commands/serve/auth.rs`, not every serve path.
- `crates/aprender-train/src/train/device.rs` — `resolve_device`, the fail-closed gate D-12 reuses.

### Project rules
- `CLAUDE.md` — Verification Discipline (all 8 rules; rule 2 is structural in D-12), realizar-first
  table (D-09 adds a documented SetFit exception), Publishing Safety, `pv` dogfooding, tiered
  gates, branch protection.
- `.claude/skills/pre-release/SKILL.md` — publishability/MSRV/feature-matrix constraints; note the
  standing KNOWN-RED: `cargo package -p apr-cli` fails until the `aprender-contrastive-data`
  publish cascade (human-run) lands.
- `.planning/codebase/ARCHITECTURE.md` — crate-boundary rules ("byte-level APR mechanics in
  apr-format, format semantics in core, request/runtime inference in serve") that D-09 and the
  codec adapter respect.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **The sealed codec seam is already shaped for this phase** (`bundle.rs`): bytes ↔ complete bundle
  plus a format id, with verification policy crate-internal. The APR codec is an adapter, not an
  architecture.
- **APR v2 container** (`apr-format/src/v2/`): typed metadata, binary blobs, checksums, atomic
  write via the core serialization adapter — D-02's hybrid encoding maps directly onto existing
  container capabilities; no format-level invention needed.
- **`apr serve` infrastructure**: router assembly, `AppState` variants, readiness/health routes,
  and the serve command's model-type detection at startup already exist — D-10 extends a pattern,
  not builds one.
- **CLI dispatch pattern**: `apr data`'s namespace wiring (commands_enum → dispatch → commands/
  module) is the exact template for `apr setfit`.
- **Phase 3's trusted evaluator + lock** (`evaluate.rs`, `lock.rs`): D-16's `apr eval` path is a
  thin CLI adapter over an already-tested workflow.
- **`resolve_device`**: OPS-06's fail-closed half already exists and is contract-relevant.

### Established Patterns
- **Typestate + trybuild non-constructibility** for anything that must be impossible, not just
  rejected (Ph2/Ph3 house style) — applies to "no consumer accepts an unverified model" (APR-04).
- **In-band negatives in every `cargo test`** — D-13's skewed-surface variant is this phase's
  instance.
- **Contract-resident constants committed before comparison** — D-03's cap, D-11's probe
  tolerances, D-13's parity tolerances all follow Ph1 D-14.
- **Feature-gated surface must be compiled and run by a named tier + CI job** — the CR-01 lesson;
  SAFE-02 makes it a requirement, not a habit.
- **Library crates bytes-in/typed-out; `apr-cli` owns filesystem adapters** — the APR file path
  handling lives in the CLI/adapter layer.

### Integration Points
- `aprender-train/src/train/setfit/` gains the APR codec adapter module (Ph3 D-07's designated
  landing spot) calling `aprender-core` format code.
- `apr-cli` gains the `setfit` command namespace and auto-detect routing in generic
  inspect/eval/predict; `aprender-serve` gains a feature-gated classify route + core model edge.
- The `setfit` feature must now propagate through `aprender-serve` as well as core/train — the CPU
  feature matrix (SAFE-02) grows a crate and must stay green in CI.
- D-08's shared envelope type needs a home visible to both `apr-cli` and `aprender-serve` without
  violating the dependency direction — planning must site it (core is the natural meeting point).
- Phase 5 consumes: the artifact hash in every response (EVAL-03), the evidence summary in the APR
  (EVAL-03), machine-readable eval output (EVAL-01), and D-16's lock workflow (EVAL-04's
  post-test-selection invalidation).

</code_context>

<specifics>
## Specific Ideas

- **"The served model is the evaluated model" must be re-provable by every fresh process** — that
  is why D-11 chose embedded self-test probes over trust-the-checksum. The probe replay is the
  production-side mirror of Phase 3's train-time round trip.
- **The realizar-first exception is deliberate and must be documented where the rule lives**
  (CLAUDE.md's table), not just in this file — otherwise a later reader will "fix" serving back
  toward a reimplementation, exactly the drift Ph2's override annotations exist to prevent.
- **Field parity by construction, not by test** (D-08): one response type serialized by both
  surfaces makes criterion 4 structural. The parity gate then checks *values*, not *shapes*.
- **The training CLI consumes Phase 2's artifacts as-is** — prepared dataset dir + selection
  manifest in, APR out. No new data formats at the CLI boundary.
- **A gate that can go vacuous is not a gate** — Phase 3's CR-02 (zero-match filter exits 0)
  applies to every new Make target and CI filter this phase adds; each must fail if it ran nothing.

</specifics>

<deferred>
## Deferred Ideas

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

</deferred>

---

*Phase: 4-APR Artifact and Production Parity*
*Context gathered: 2026-08-14*
