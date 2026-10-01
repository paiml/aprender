---
phase: 04-apr-artifact-and-production-parity
plan: 16
subsystem: training-lifecycle
tags: [setfit, apr, typestate, provenance, selection-lock, trn-07, blocked]
status: BLOCKED

# Dependency graph
requires:
  - phase: 04-05
    provides: "AprCodec + APR_FORMAT_ID — the codec the reload would have re-entered the trusted policy through"
  - phase: 04-13
    provides: "bundle field 20 (ProvenanceRecord) — the three recorded identifiers the reload's identity gate compares"
  - phase: 04-03
    provides: "load_setfit_apr / read_setfit_apr_parts / VerifiedSetFitModel — the production loader ladder incl. probe replay"
provides: []
affects: [04-07, 04-12, phase-05]

# Tech tracking
tech-stack:
  added: []
  patterns: []

key-files:
  created: []
  modified: []

key-decisions:
  - "STOPPED before writing code: SetFitRun<HeadFitted> cannot be assembled from artifact bytes, so verify_artifact — the one trusted minting policy — cannot be re-entered from a fresh process. Surfaced as a finding rather than defaulted, per the plan's own HARD CONSTRAINT."
  - "Refused the three available workarounds (default the missing evidence; construct ArtifactReloadedAndVerified locally; mint from a reduced policy) — each is exactly T-04-46 (a second minting path with weaker evidence) or T-04-60 (a check declined on a false rationale)."

patterns-established: []

requirements-completed: []

# Metrics
duration: 41min
completed: 2026-08-15
---

# Phase 4 Plan 16: Fresh-Process Verified Run — Summary

**BLOCKED, with a mechanically proven finding: the `setfit-apr-v1` artifact records the evidence
SUMMARY and not the evidence TABLE, so no fresh process can assemble a `SetFitRun<HeadFitted>` —
and therefore no fresh process can re-enter `verify_artifact`, the one trusted minting policy. The
plan's HARD CONSTRAINT fired exactly as written; no code was shipped.**

## Performance

- **Duration:** 41 min
- **Started:** 2026-08-15T17:06:00Z
- **Completed:** 2026-08-15T17:46:51Z
- **Tasks:** 0 of 1 (stopped at the plan's own STOP condition, before implementation)
- **Files modified:** 0 (source tree is byte-identical to the plan's base commit `87b780ee`)

## The Finding

### What the plan assumed

Plan 04-16 step 6 instructed:

> Assemble `SetFitRun<HeadFitted>` … carrying the rebuilt encoder, the supplied dataset and
> selection, the resolved config, and the HeadFitted evidence assembled from `doc.ordered_labels`,
> the deserialized `doc.evidence` summary and the rebuilt head.

— three values. It then added the STOP condition:

> Every value must be RECOVERED from the artifact or SUPPLIED by the caller — if the evidence type
> requires a value that is neither, STOP and surface it rather than defaulting it.

### What `HeadFittedEvidence` actually requires

`HeadFittedEvidence` (`crates/aprender-train/src/train/setfit/mod.rs:166-174`) has **seven** fields,
not three. The compiler was asked directly rather than the source read — a throwaway probe module was
added, `cargo check -p aprender-train --features setfit --lib` was run, and the probe was deleted:

```
error[E0063]: missing fields `effective_lambda`, `encode_call_count`, `encode_ledger`
              and 2 other fields in initializer of `HeadFittedEvidence`
```

The "2 other fields" are `passed` and `report`. Field-by-field:

| Field | Type | Recoverable from artifact bytes + supplied dataset/selection? |
|-------|------|--------------------------------------------------------------|
| `head` | `MultinomialLogisticRegression` | **Yes** — `from_stored_coefficients` over `doc.head` + head tensors |
| `ordered_labels` | `Vec<String>` | **Yes** — `doc.ordered_labels` |
| `effective_lambda` | `f64` | Recomputable — a deterministic function of `doc.requested_config` and the supplied selection's row count via `head_input::resolve_lambda`. Not *recovered*, but not invented either |
| `report` | `HeadFitReport` | **No** — `{status, iterations, final_grad_norm, objective}` (`aprender-core/src/classification/multinomial.rs:457-466`), a record of an L-BFGS run that happened in another process. Recomputing it means re-fitting, which is a different run |
| `encode_ledger` | `Vec<String>` | **No** — the ordered ids the encode-once path *wrote as it went*. Re-deriving it from the selection is precisely the false green the in-band ledger exists to remove (`mod.rs:211-215`) |
| `encode_call_count` | `usize` | **No** — same; a count of invocations that already happened |
| `passed` | `tune::PassedEvidence` | **No** — see below. This is the hard one |

### Why `passed` is structurally unrecoverable

`PassedEvidence { table: UpdateEvidence, summary: EvidenceSummary }` has private fields. A second
compile probe confirmed a sibling module cannot construct one:

```
error[E0451]: fields `table` and `summary` of struct `PassedEvidence` are private
```

That is by design (`tune.rs:1074-1078`): the only producer is `tune::validate_evidence`, which takes
`&UpdateEvidence` — the full per-parameter table. And **the artifact does not carry that table.**
Four independent confirmations:

1. `SetFitBundle` field 19 is `evidence: EvidenceSummary` (`bundle.rs:500-501`), not `UpdateEvidence`.
2. `SetFitBundle::from_run_parts` takes `evidence: &EvidenceSummary` (`bundle.rs:546`); `verify_artifact`
   passes `parts.passed.summary()` (`mod.rs:785`) and drops the table at the persistence boundary.
3. `SetFitArtifactDoc::evidence` is documented `to_value(EvidenceSummary)` — "opaque here"
   (`aprender-core/src/setfit/artifact.rs:1670-1671`); `WALKED_SUBDOCUMENTS` is the closed five
   `[architecture, requested_config, resolved_config, evidence, provenance]` (artifact.rs:130-136).
4. `apr_codec::bundle_of` reconstructs all twenty bundle fields from the doc and none of them is the
   table (`apr_codec.rs:247-340`).

`EvidenceSummary` carries `table_hash` — a *binding* to a table it does not contain. So the artifact
can prove a supplied table is the right one; it cannot supply it.

`UpdateEvidence` (`evidence.rs:369-406`) holds the per-parameter rows plus `loss_trace_hash`,
`consumed_pair_digest`, `batch_boundary_digest`, `batch_boundary_list`, `parameter_registry_hash`
and `step_count`. Those are the exact values `SetFitRun<ArtifactReloadedAndVerified>`'s
reproducibility accessors return (`mod.rs:851-902`). Minting the state without them would not merely
be incomplete — every one of those six accessors would be answering about a run this process never
observed.

### Why each escape hatch was refused

- **Default the five missing values.** Forbidden by the plan verbatim, and it would make
  `pair_order_digest()`, `loss_trace_hash()`, `step_count()` and `batch_boundaries()` return
  fabrications with the full authority of a verified typestate. T-04-60.
- **Construct `ArtifactReloadedAndVerified` locally from what *is* recoverable.** That is a second
  minting path with weaker evidence — T-04-46, Pitfall 9, and explicitly the thing the plan's
  acceptance criteria grep for.
- **Widen the artifact here.** Carrying the table needs `bundle.rs`, `apr_codec.rs`,
  `aprender-core/src/setfit/artifact.rs`, `contracts/setfit-apr-v1.yaml` and the 20-field bijection
  gate — outside this plan's two-file constraint and inside 04-05's and 04-13's wave-5 ownership.

### The reframing that makes this tractable

`SetFitRun<ArtifactReloadedAndVerified>` is a **train-time** state. It carries the whole stage-one and
stage-two evidence chain, which by construction only the training process ever held. 04-16 tried to
make a fresh process hold it.

But 04-CONTEXT.md **D-11** already defines what a fresh process's verified state means, and it is
narrower:

> **D-11: `ArtifactReloadedAndVerified` for a fresh process = integrity + embedded self-test probes.**

That value already exists and already shipped in 04-03: `VerifiedSetFitModel`, whose own doc says
"APR-04's obligation in one sentence: evaluation, registration, benchmarking, prediction and serving
accept only `ArtifactReloadedAndVerified`, and out-of-crate code cannot mint the consumer-side witness
type" (artifact.rs:1723-1725). It is non-constructible outside core, minted only by `load_setfit_apr`,
and the six-probe replay is part of its ladder.

And it already carries **every value the lock chain actually reads**:

| Door | Reads | Available on `VerifiedSetFitModel`? |
|------|-------|-------------------------------------|
| `create_selection_lock` (lock.rs:659-677) | `artifact_hash()`, `selection_semantic_hash()`, `selection().ledger_hash()` | `artifact_sha256()`; `doc_view().provenance.selection_semantic_hash`; `.selection_ledger_hash` |
| `mint_test_token` (lock.rs:587-610) | `artifact_hash()` | `artifact_sha256()` |
| `CanonicalTestAccess::grant` (lock.rs:754-774) | `artifact_hash()` + the supplied dataset's fingerprint | `artifact_sha256()` + unchanged |

Not one of the three doors touches the evidence table. **The blocker is not missing data — it is that
lock.rs's three doors are typed against the train-time run when the values they read are exactly the
ones a fresh process can hold.**

### 04-CONTEXT.md flagged this in advance

Under "Claude's Discretion" (04-CONTEXT.md:210):

> The exact well-known metadata key names and the JSON document's field schema (D-02), **including
> where the full evidence table (vs its hash) is emitted.**

That discretionary sub-decision was resolved by 04-01/04-02/04-05 as *hash only*. 04-16's requirement
re-opens it. This is a consequence of a recorded decision, not drift.

## Decision Required

**Option A — carry the evidence table in the artifact (bundle field 21+).**
Persist `UpdateEvidence`, `HeadFitReport`, `encode_ledger`, `encode_call_count` and
`effective_lambda`. `PassedEvidence` is then re-minted by re-entering `tune::validate_evidence` — the
real gate, not a copy of it — and `summary.table_hash` gives a free integrity binding on the recovered
table. Plan 04-16 then works as written.
*Cost:* `bundle.rs`, `apr_codec.rs`, `aprender-core/src/setfit/artifact.rs`,
`contracts/setfit-apr-v1.yaml`, the 20→21+ bijection gate, Ph3's committed closure tests. Artifact
grows by one row per trainable parameter. Touches two other wave-5 plans' files.
*Buys:* the reloaded state's reproducibility accessors become true.

**Option B — type the lock doors against the credential a fresh process can hold. (Recommended.)**
Keep 04-16's provenance identity gate (it is sound and independently valuable: it binds a
`VerifiedSetFitModel` to a supplied dataset+selection by all three recorded identifiers). Then make
`create_selection_lock` / `mint_test_token` / `grant` accept either the train-time run or a
fresh-process credential minted from `load_setfit_apr` + that gate — via a small sealed trait, so
there is still exactly one way to obtain each.
*Cost:* `lock.rs` signatures, one new module. No artifact/contract/schema change.
*Buys:* D-11's stated model, implemented. No fabricated evidence anywhere. Nothing claims to know a
run it did not observe.
*Note:* reading the semantic/ledger hashes off `doc.provenance` is the artifact's claim about itself —
which is exactly why the identity gate against the caller's real `Selection` is load-bearing here, not
optional.

**Option C — persist the token, not the run.**
Training writes lock **and** token; `apr eval --split test` consumes both plus a `VerifiedSetFitModel`.
*Cost:* `mint_test_token` and `grant` still take `&SetFitRun<ArtifactReloadedAndVerified>`, so this
collapses into Option B unless a token wire form is added too. Lowest ceiling.

## Files Created/Modified

None. `git status --short` is clean at `87b780ee`; the two compile probes were added and deleted
within this session and never staged.

## Deviations from Plan

None — the plan's HARD CONSTRAINT was executed as written. It said to STOP and surface a finding if a
required value could be neither recovered nor supplied; five such values exist and the finding is
above.

## Issues Encountered

The plan's step 6 was authored against a three-field model of `HeadFittedEvidence`. The type has seven
fields. This was not caught at plan time because `HeadFittedEvidence`'s declaration and
`verify_artifact`'s use of `parts.passed.summary()` are 600 lines apart in `mod.rs`, and the summary is
the only part of `PassedEvidence` the persistence path touches — so a reader following the write path
sees only the summary and can reasonably conclude the summary is all the state carries.

## Next Phase Readiness

**04-07 is blocked on this decision.** Its `<interfaces>` block (04-07-PLAN.md:96) lists
`reload_verified_run_from_apr(bytes, dataset, selection) -> Result<SetFitRun<ArtifactReloadedAndVerified>, SetFitTrainError>`
as "the fresh-process door" it consumes as shipped. That function does not exist and cannot be written
without one of the three options above.

TRN-07's positive tier (D-16) remains unreached. Wave 5's other plans (04-06, 04-12) are unaffected —
this plan touched no shared file.

## Self-Check: PASSED

- `git status --short` clean; `git rev-parse HEAD` = `87b780ee` — no source file created or modified,
  as claimed.
- Both compile-probe results are quoted verbatim from `cargo check -p aprender-train --features setfit
  --lib`; E0063 and E0451 are rustc's, not this document's reading of the source.
- Every file:line citation in this summary was read in this session.

---
*Phase: 04-apr-artifact-and-production-parity*
*Plan: 16 — BLOCKED, awaiting architectural decision*
*Completed: 2026-08-15*
