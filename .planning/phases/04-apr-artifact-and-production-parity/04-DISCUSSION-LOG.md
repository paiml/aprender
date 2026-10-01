# Phase 4: APR Artifact and Production Parity - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-08-14
**Phase:** 4-APR Artifact and Production Parity
**Areas discussed:** Artifact internals, CLI surface, Serving boundary, Parity + scope

---

## Artifact internals

### Tensor naming inside the APR

| Option | Description | Selected |
|--------|-------------|----------|
| Canonical + HF map (Recommended) | Write tensors under canonical tensor-names-v1 names; carry the HF↔canonical map in metadata for the loader's named-parameter view and APR-03's HF-keyed evidence comparison | ✓ |
| HF dotted verbatim | Store checkpoint names Phases 1-3 use; zero translation but generic tooling and the naming contract see foreign names | |
| You decide | Leave stored form to planning | |

**User's choice:** Canonical + HF map

### Non-tensor payload encoding

| Option | Description | Selected |
|--------|-------------|----------|
| Hybrid (Recommended) | Typed identity keys (family/schema tag, labels, schema version, tokenizer SHA-256) + one schema-versioned deny_unknown_fields JSON doc (config, evidence, provenance, policy) + exact tokenizer bytes as blob | ✓ |
| One JSON document | Everything in a single doc; simplest loader, but generic inspect needs SetFit-aware parsing | |
| All typed keys | Every field a typed entry; validation scattered across dozens of key reads | |

**User's choice:** Hybrid

### What "oversized" means (APR-01)

| Option | Description | Selected |
|--------|-------------|----------|
| Derived + hard cap (Recommended) | Structural check (tensor bytes must equal declared shape/dtype/config) + a contracted hard cap on file size | ✓ |
| Fixed cap only | One constant; cannot catch shape/byte disagreement early | |
| You decide | Planning picks mechanism and constant | |

**User's choice:** Derived + hard cap

### SetFit auto-detection (OPS-03)

| Option | Description | Selected |
|--------|-------------|----------|
| Explicit tag only (Recommended) | Read typed model_family/schema keys; no tag → not SetFit; never sniff artifacts our own writer produced | ✓ |
| Tag + shape verify | Tag plus tensor-set verification; overlaps the derived size/shape check | |
| You decide | Planning picks, fail-closed required | |

**User's choice:** Explicit tag only

---

## CLI surface

### Training command shape

| Option | Description | Selected |
|--------|-------------|----------|
| apr setfit train (Recommended) | Dedicated namespace per Phase 2's apr data precedent; keeps LoRA baseline distinct for Phase 5 | ✓ |
| Extend apr train | SetFit mode on generic train; two commands wearing one name | |
| You decide | Only constraint is OPS-02's completable lifecycle | |

**User's choice:** apr setfit train

### predict/eval/inspect surface

| Option | Description | Selected |
|--------|-------------|----------|
| Generic only (Recommended) | apr inspect/eval/predict auto-detect and route to shared core path; one code path per operation; setfit namespace stays training-only | ✓ |
| Both, aliases delegate | setfit-namespaced pure delegating aliases for discoverability | |
| You decide | One implementation per operation either way | |

**User's choice:** Generic only

### Training configuration input

| Option | Description | Selected |
|--------|-------------|----------|
| Config file first (Recommended) | --config TOML/JSON into Phase 3's validated SetFitConfig; flags for paths + small overrides; merged resolved config recorded in artifact | ✓ |
| Flags only | All 12 knobs as flags; no canonical source document for provenance | |
| You decide | TRN-02 + APR-01 constraints apply either way | |

**User's choice:** Config file first

### Machine-readable output policy

| Option | Description | Selected |
|--------|-------------|----------|
| Shared typed envelope (Recommended) | One versioned response type family serialized identically by CLI --json and HTTP; parity is type-level | ✓ |
| Per-surface schemas | Idiomatic per surface; parity becomes an ongoing test obligation | |
| You decide | Criterion 4 field parity required either way | |

**User's choice:** Shared typed envelope

---

## Serving boundary

### Crate ownership of serving inference

| Option | Description | Selected |
|--------|-------------|----------|
| HTTP in serve, model in core (Recommended) | serve owns route/AppState/readiness, calls core's verified model via production loader; documented realizar-first exception | ✓ |
| Reimplement in realizar | Consistent with existing table but creates the second path OPS-03 forbids and un-proves Phase 1 fixtures | |
| You decide | OPS-03 + APR-04 are hard constraints | |

**User's choice:** HTTP in serve, model in core

### Route shape and server startup

| Option | Description | Selected |
|--------|-------------|----------|
| apr serve auto-detects (Recommended) | Existing serve command reads the tag at load and installs classify route(s); readiness reports artifact hash + verified state | ✓ |
| Dedicated setfit route family | Distinct endpoint installed only for SetFit; second surface to keep in parity | |
| You decide | OPS-05 constraints apply either way | |

**User's choice:** apr serve auto-detects

### Load-time verification semantics (APR-04)

| Option | Description | Selected |
|--------|-------------|----------|
| Integrity + self-test probes (Recommended) | Artifact embeds train-time probe records; every fresh consumer replays them through its own loaded model before predicting | ✓ |
| Integrity + evidence presence | Verify checksum/schema/consistency + evidence record, no model re-execution | |
| You decide | APR-02's fail-before-prediction list is the floor | |

**User's choice:** Integrity + self-test probes

### Backend identity semantics (OPS-06)

| Option | Description | Selected |
|--------|-------------|----------|
| Report resolved, from execution (Recommended) | Identity produced by the compute path that actually ran; resolve_device stays the fail-closed gate | ✓ |
| Echo resolved config | Simpler but is the "label by intent" pattern CLAUDE.md rule 2 documents | |
| You decide | No-fallback, no-misreporting constraints apply | |

**User's choice:** Report resolved, from execution

---

## Parity + scope

### How core/CLI/HTTP equivalence is gated

| Option | Description | Selected |
|--------|-------------|----------|
| Live compare + goldens (Recommended) | One harness feeds identical inputs through all three surfaces, compares pairwise; frozen goldens pin the core surface; in-band negative must fail in every cargo test | ✓ |
| Frozen goldens only | Surfaces can each pass within tolerance yet disagree by 2x tolerance with each other | |
| You decide | Offline executable detection of any pairwise failure required | |

**User's choice:** Live compare + goldens

### HTTP parity execution mechanics

| Option | Description | Selected |
|--------|-------------|----------|
| In-process gate + spawned smoke (Recommended) | tower oneshot against the real Router in every cargo test; one tier3 smoke spawns real apr serve on loopback | ✓ |
| Spawned process only | Realistic but flaky in fast tiers — the gate-stops-being-run failure mode | |
| In-process only | Never proves the installed binary serves | |

**User's choice:** In-process gate + spawned smoke

### Ph1 D-02 encoder convergence scope

| Option | Description | Selected |
|--------|-------------|----------|
| Narrow: SetFit routing only (Recommended) | Generic commands route SetFit APRs to shared core path; embed/rerank/CrossEncoder untouched; full convergence explicitly deferred with its own ticket | ✓ |
| Full convergence | Honor D-02 literally; roughly doubles blast radius on the critical path to Phase 5 | |
| Middle: embed only | Partial payment, creates a third converged/not-converged state | |

**User's choice:** Narrow: SetFit routing only

### TRN-07 positive-half closure

| Option | Description | Selected |
|--------|-------------|----------|
| Yes, via apr eval (Recommended) | Canonical-test eval exercises create_selection_lock → mint_test_token → grant as its real workflow; TRN-07 checked this phase | ✓ |
| No, leave for Phase 5 | Phase 5's claims gate would depend on a surface built under benchmark pressure | |
| You decide | Closes as a side effect if eval naturally forces the lock path | |

**User's choice:** Yes, via apr eval

---

## Claude's Discretion

- Exact metadata key names and JSON document field schema; full evidence-table placement
- Concrete hard-cap constant and probe-record count/selection policy (contract-resident)
- HTTP route naming/paths
- New phase contract shape (setfit-apr-v1.yaml default per Ph1 D-23) vs extending lifecycle contract
- Exit-code taxonomy; --dry-run mode for setfit train
- Any Phase 5 identical-sampled-ID hooks
- Mutation-testing scope boundaries (Phase 3 tooling constraints inherited)

## Deferred Ideas

- Full encoder convergence (apr embed/rerank/CrossEncoder) — own future ticket (D-15 amendment of Ph1 D-02)
- GPU/accelerator SetFit inference — v2 (ACC-01)
- Quantized artifacts — v2 (QUANT-01)
- crates.io publish cascade — pending, human-run
- Production-encoder calibration regime edit for the SetFit-identity gate — deliberate pv-flagged contract edit, Phase 5 blocker
- Unifying auth across all serve router variants — repo-wide serving work
