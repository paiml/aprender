---
phase: 04-apr-artifact-and-production-parity
plan: 12
status: BLOCKED
subsystem: training
tags: [setfit, ops-01, public-api-boundary, integration-test, finding]

# Dependency graph
requires:
  - phase: 04-04
    provides: "ClassifyRequestDocument + VerifiedSetFitModel::classify + ClassifyResponse accessors — VERIFIED PRESENT on this branch"
  - phase: 04-05
    provides: "AprCodec (setfit-apr-v1) behind the sealed codec seam — VERIFIED PRESENT on this branch"
provides: []
affects: [04-06, 04-16]

# Tech tracking
tech-stack:
  added: []
  patterns: []

key-files:
  created: []
  modified: []

key-decisions:
  - "STOPPED rather than patching around the gap: the plan's HARD CONSTRAINT names a public-API gap as an OPS-01 FINDING, not something an integration-test plan may close by editing src/"
  - "The gap was proven by the COMPILER over three independent candidate doors, not by reading the source — three distinct refusals (E0599, E0308, E0624)"
  - "The cargo-tree half of T-04-36 was measured anyway (it is independent of the blocker) and includes a positive control, so its three zeros are measurements rather than a vacuous count"

requirements-completed: []

# Metrics
duration: 25min
completed: 2026-08-15
---

# Phase 04 Plan 12: OPS-01 Public-API Lifecycle Test — BLOCKED (OPS-01 finding)

**The plan cannot be executed as written: the SAVE step of the OPS-01 chain is not reachable through any public API. `verify_artifact` produces the `setfit-apr-v1` bytes internally, hashes them, retains only the hash and a LENGTH, and drops the bytes — so an out-of-crate caller has nothing to hand `load_setfit_apr`. This is exactly the condition the plan's HARD CONSTRAINT says to surface as a finding rather than patch around.**

No source file was created or modified. The worktree is clean at the plan's base commit plus this SUMMARY.

## The Finding — OPS-01-F1: the artifact bytes have no public door

### What OPS-01 asks for

`train -> save -> load -> embed -> classify -> inspect`, purely through public library APIs. The `load` rung is `aprender::setfit::artifact::load_setfit_apr(bytes)`. The `save` rung must therefore hand an out-of-crate caller those `bytes`.

### What the lifecycle actually retains

`SetFitRun::<HeadFitted>::verify_artifact<C: SetFitCodec>(self, codec)` (`crates/aprender-train/src/train/setfit/mod.rs:770`) serializes the bundle inside `verify::run_verify_policy`, hashes exactly those bytes, drops the live model, reloads, closure-checks and compares. What survives into `ArtifactVerifiedEvidence` (`mod.rs:383-395`) is:

| Retained | Type | Public accessor |
|---|---|---|
| `artifact_hash` | `[u8; 32]` | `SetFitRun::<ArtifactReloadedAndVerified>::artifact_hash() -> String` (`mod.rs:916`) |
| `VerifyReport::artifact_bytes` | `usize` — a **LENGTH** | `VerifyReport::artifact_bytes() -> usize` (`mod.rs:325`) |
| the bytes themselves | — | **none** |

The complete public surface of `impl SetFitRun<ArtifactReloadedAndVerified>` is eleven read-only accessors in `mod.rs:838-931` plus `create_selection_lock` in `lock.rs:638`. None returns `Vec<u8>` or `&[u8]`.

### Proven by the compiler, over three independent doors

A temporary probe (`crates/aprender-train/tests/zz_probe_bytes_door.rs`, created, compiled, and deleted — not committed) asked all three candidate doors at once. `cargo check -p aprender-train --features setfit --test zz_probe_bytes_door` returned three distinct refusals:

```
error[E0599]: no method named `artifact_bytes` found for reference
              `&SetFitRun<ArtifactReloadedAndVerified>` in the current scope
  --> crates/aprender-train/tests/zz_probe_bytes_door.rs:12:29

error[E0308]: mismatched types
  --> crates/aprender-train/tests/zz_probe_bytes_door.rs:17:25
   |
17 |     let _bytes: &[u8] = run.evidence().verify_report().artifact_bytes();
   |                 -----   ^^^ expected `&[u8]`, found `usize`

error[E0624]: associated function `from_run_parts` is private
  --> crates/aprender-train/tests/zz_probe_bytes_door.rs:23:31
   |
   ::: crates/aprender-train/src/train/setfit/bundle.rs:539:5
```

Read as a set:

1. **E0599** — the verified run has no bytes accessor at all.
2. **E0308** — the one member that *looks* like the door is a length, not the payload. This is the trap: `run.evidence().verify_report().artifact_bytes()` compiles, and a test that bound it to `let n = ...` would read as if it had the artifact.
3. **E0624** — the public codec seam is not a way around it either. `SetFitCodec::serialize(&SetFitBundle)` is `pub`, but the only constructor that builds a bundle from a run, `SetFitBundle::from_run_parts`, is `pub(crate)`. An out-of-crate caller can serialize a bundle it already has and cannot obtain one from a run.

The remaining theoretical route — hand-building an `aprender::setfit::SetFitArtifactView` from the run's accessors and calling core's public `write_setfit_apr` — is a re-implementation of `AprCodec::serialize` inside the test. It was rejected on sight: the test would then prove that a test-local writer round-trips, which is not OPS-01, and it would drift from the shipped codec silently.

### Why this is not local to plan 04-12

`apr setfit train` hits the identical wall. 04-06 Task 2 step 4 says "run tune_encoder -> fit_head -> verify_artifact(AprCodec); obtain the artifact bytes + recorded hash" and step 6 says "atomic_write the bytes to `--output`". `apr-cli` is out-of-crate with respect to `aprender-train`, so `apr setfit train` cannot write its output file today. **The phase currently has no way to produce a `.apr` file at all from outside the training crate.**

04-16 does not close this. `reload_verified_run_from_apr(bytes, dataset, selection)` **consumes** bytes; it does not produce them. Its own tests run `--lib` (in-crate), so they reach `pub(crate)` doors that neither 04-12 nor 04-06 can.

## What WAS measured — T-04-36's dependency-direction half

Independent of the blocker, and reported because the plan requires it in the SUMMARY:

| Command | `apr-cli` occurrences | Tree size (non-vacuous) |
|---|---|---|
| `cargo tree -p aprender-core -e normal` | **0** | 121 lines |
| `cargo tree -p aprender-train -e normal` | **0** | 633 lines |
| `cargo tree -p aprender-train -e normal --features setfit` | **0** | 707 lines |
| `cargo tree -p aprender -e normal` (**positive control**) | **1** | — |

The positive control is the point. Three zeros from a method that can only ever return zero would be theater (the Phase 3 CR-02 lesson); the root facade genuinely depends on `apr-cli`, the same count returns 1 there, so the three zeros are measurements. The `--features setfit` row is included because that feature is where the new Phase 4 edges appear — checking only default features would not cover the surface where the decision is made.

Neither library crate resolves a dependency on `apr-cli`. The *dependency-direction* half of OPS-01 holds; the *usability* half is what this finding blocks.

## Resolution options (for the decision — not taken here)

All of them edit `crates/aprender-train/src/train/setfit/`, which **04-16 owns in this same wave**, so none may be taken by this plan.

| # | Change | Cost / consequence |
|---|---|---|
| A | Retain the bytes in `ArtifactVerifiedEvidence` + add `pub fn artifact_apr_bytes(&self) -> &[u8]` | Simplest for callers, but a legitimate artifact is ~90 MB held for the run's whole life. Also grows the guarded block: `verify_reproducibility_accessors_are_read_only_and_complete` (`verify_tests.rs:579`) counts that block's `pub fn`s, so this is a deliberate, asserted surface change. Name must differ from the existing `VerifyReport::artifact_bytes` length. |
| B | `verify_artifact` returns `(SetFitRun<ArtifactReloadedAndVerified>, Vec<u8>)` | No retention cost and the bytes cannot be silently ignored, but it is a breaking change to a public signature with call sites in `setfit_repro.rs`, the lib tests and 04-16 step 7. |
| C | `pub fn into_artifact_bytes(self) -> Vec<u8>` on the verified run | Consumes the run, so a caller cannot both write the file and keep the lock/token chain — which is exactly what `apr setfit train` needs to do. |

**Recommendation: fold option A or B into 04-16** (it already edits `mod.rs`, already reasons about artifact bytes, and is the only wave-5 plan permitted to touch these files), then re-run 04-12 and 04-06 in a later wave. Deciding between A and B is a real tradeoff — retention cost vs a breaking signature — and belongs to planning, not to an executor.

## Deviations from Plan

None. The plan's HARD CONSTRAINT was followed exactly: a public-API gap was found, no `src/` file was edited, and the gap is surfaced as an OPS-01 finding.

## Self-Check

- `crates/aprender-train/tests/setfit_apr_lifecycle.rs` — **NOT created** (blocked; a non-compiling test would have broken the build for the whole wave)
- `crates/aprender-train/tests/zz_probe_bytes_door.rs` — created, compiled for evidence, **deleted**; `git status --short` is clean
- `.planning/phases/04-apr-artifact-and-production-parity/04-12-SUMMARY.md` — created (this file)

## Self-Check: BLOCKED

Plan not complete. Requirement OPS-01 is **not** satisfied and must not be marked complete. Missing: the lifecycle integration test, which cannot be written until the artifact-bytes door exists.
