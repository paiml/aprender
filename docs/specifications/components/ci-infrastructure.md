<!-- PCU: ci-infrastructure | contract: contracts/ci-infra-v1.yaml -->
<!-- Example: none (infrastructure specification) -->
<!-- Status: proposed -->
# CI Infrastructure — Permanent Fix Specification

Version: 1.0
Status: proposed
Date: 2026-04-10

**Contract**: `contracts/ci-infra-v1.yaml`
**Parent**: [aprender-spec.md](../aprender-spec.md) §15 Quality Standards

---

## Problem

The CI system has 6 recurring failure modes that cause "whack-a-mole" — fixing
one reveals another, each requiring a push + 10-minute build cycle. The root
cause is architectural: the monorepo is not self-contained.

## Five-Whys Analysis

### RC1: Phantom Workflow Triggers

**Symptom**: `release.yml` (tag-only but no branch filter), `pr-gate.yml`, `nightly-bench.yml`
produce 0-second failures on every push to any branch (~12 false failures/day).
Note: `release.yml` has `tags: ['v*']` but no `branches:` filter, so it evaluates on every push.

1. **Why do they fail?** GitHub evaluates the workflow YAML on every push event.
2. **Why are they evaluated?** No branch filter — `on: push` without `branches:`.
3. **Why no branch filter?** Original design assumed these only run manually or on tags.
4. **Why didn't this break earlier?** GitHub Actions recently changed evaluation behavior for `pull_request_target` and tag-only workflows.
5. **Why wasn't it caught?** No contract enforces that non-CI workflows must have branch filters.

**Root cause**: Missing `branches:` filter on non-CI workflows.
**Permanent fix**: Every workflow MUST specify explicit trigger conditions. No bare `on: push`.
**Contract gate**: `F-INFRA-001`

### RC2: aprender-profile Windows Failure

**Symptom**: 100% of nightly Windows builds fail on `aprender-profile` — `nix` crate not found.

1. **Why does it fail on Windows?** `nix` crate provides Unix system calls (ptrace, setrlimit).
2. **Why is `nix` compiled on Windows?** It's an unconditional dependency in Cargo.toml.
3. **Why unconditional?** The crate was written for Linux profiling — Windows was never a target.
4. **Why does CI build it on Windows?** `nightly.yml` builds `apr-cli` which pulls all workspace deps.
5. **Why isn't it excluded?** No `#[cfg(unix)]` gate, no `--exclude` in the nightly script.

**Root cause**: Platform-specific crate without platform gate.
**Permanent fix**: `[target.'cfg(unix)'.dependencies] nix = ...` in Cargo.toml.
**Contract gate**: `F-INFRA-002`

### RC3: Sibling Repository Dependency Web

**Symptom**: Any change in trueno/realizar/batuta/alimentar can break aprender CI
with zero code change on aprender's side.

1. **Why does a sibling change break us?** CI clones siblings at HEAD of `main`.
2. **Why HEAD of main?** `sovereign-ci.yml` does `git fetch --depth 1 origin main && git reset --hard FETCH_HEAD`.
3. **Why not pinned versions?** Design predates the monorepo — when repos were separate, HEAD was the only option.
4. **Why wasn't this fixed during monorepo consolidation?** The 70 crates were consolidated but CI workflows still reference external repos for `provable-contracts`, `alimentar`, `batuta`, `realizar`.
5. **Why do these external deps still exist?** `[patch.crates-io]` in root Cargo.toml overrides resolution to local paths that point to `../provable-contracts`, `../alimentar`, etc.

**Root cause**: Monorepo consolidation was incomplete — CI still depends on external repos at floating HEAD.
**Permanent fix**: Pin all external repo references to git tags. Update tags deliberately via monthly "deps update" PR.
**Contract gate**: `F-INFRA-003`

### RC4: Floating `cc` Git Patch

**Symptom**: Random build failures when `cc-rs` main branch has regressions.

1. **Why does `cc` break?** `[patch.crates-io] cc = { git = "...", branch = "main" }`.
2. **Why patch `cc`?** A bug fix was needed that hadn't been released to crates.io yet.
3. **Why `branch = "main"` instead of a commit SHA?** Convenience — wanted latest fixes.
4. **Why wasn't it updated when the fix was released?** No tracking mechanism — nobody checks if the upstream fix shipped.
5. **Why no tracking?** No contract requires `[patch.crates-io]` entries to have expiration dates or tracking issues.

**Root cause**: Floating external dependency with no expiration tracking.
**Permanent fix**: Either pin to SHA or remove when upstream releases. Every patch entry MUST have a tracking issue.
**Contract gate**: `F-INFRA-004`

### RC5: Clean-Room Excludes 8 Crates

**Symptom**: `apr-cli`, `aprender-gpu`, `aprender-serve`, `aprender-explain`, and 4 others excluded from CI testing.
Bugs in these crates are invisible until manual testing.

1. **Why excluded?** "shim dep API mismatch" — local path deps differ from crates.io versions.
2. **Why do APIs differ?** Workspace uses local `../realizar` (dev version) but crates.io has older release.
3. **Why local overrides?** `[patch.crates-io]` forces all resolution to local paths.
4. **Why not publish first?** Circular dependency: can't publish without CI green, can't CI without publish.
5. **Why circular?** The monorepo depends on external repos that depend on the monorepo. True self-containment was never achieved.

**Root cause**: Same as RC3 — incomplete monorepo consolidation.
**Permanent fix**: Eliminate `[patch.crates-io]` entries for own crates. Workspace path deps + version field is the correct pattern.
**Contract gate**: `F-INFRA-005`

### RC6: RUSTFLAGS Inconsistency Between Workflows

**Symptom**: Code passes `ci.yml` lint but fails `book-contracts.yml`. Each fix
requires a new push + 10-minute cycle, creating cascading failures.

1. **Why different results?** `ci.yml` uses `-A unused-variables -A missing_docs` but `book-contracts.yml` uses bare `-D warnings`.
2. **Why different flags?** Each workflow was written independently with different strictness assumptions.
3. **Why no alignment?** No single source of truth for RUSTFLAGS across workflows.
4. **Why no local pre-check?** Developers run `cargo clippy` locally but don't replicate exact CI flags.
5. **Why no contract?** No enforcement that all workflows use identical RUSTFLAGS.

**Root cause**: No single source of truth for lint configuration across CI workflows.
**Permanent fix**: Extract RUSTFLAGS to a shared env var or `.cargo/ci-config.toml`. All workflows source the same value.
**Contract gate**: `F-INFRA-006`

## Contract Equations

### Eq 1: `workflow_trigger_explicit`
```
forall workflow W in .github/workflows/*.yml:
  W has explicit branch filter on push/pull_request triggers OR
  W only uses workflow_dispatch/schedule triggers
```

### Eq 2: `platform_deps_gated`
```
forall crate C with platform-specific dependencies:
  dep is gated behind [target.'cfg(...)'.dependencies] OR
  C is excluded from cross-platform builds with documented reason
```

### Eq 3: `external_repos_pinned`
```
forall external repo R cloned by CI:
  R is pinned to a specific tag or SHA (not branch HEAD) AND
  R has a tracking issue for next update
```

### Eq 4: `patches_tracked`
```
forall patch P in [patch.crates-io]:
  P has a tracking GitHub issue AND
  P either pins to commit SHA or has expiration date
```

### Eq 5: `no_untested_exclusions`
```
forall crate C in workspace:
  C is included in CI test suite OR
  C has documented exclusion reason with tracking issue for re-inclusion
```

### Eq 6: `rustflags_consistent`
```
forall workflow W1, W2 that run clippy or compile with -D warnings:
  RUSTFLAGS(W1) == RUSTFLAGS(W2) OR
  difference is documented with rationale
```

## Falsification Tests

| ID | Equation | Test | Automation |
|----|----------|------|------------|
| F-INFRA-001 | trigger_explicit | Parse all workflow YAML; assert no bare `on: push` without `branches:` | Python YAML parse |
| F-INFRA-002 | platform_gated | Find `nix` dep in Cargo.toml; assert under `[target.'cfg(unix)']` | TOML parse |
| F-INFRA-003 | repos_pinned | Parse nightly.yml checkout steps; assert all use `ref:` with tag/SHA | YAML parse |
| F-INFRA-004 | patches_tracked | Parse [patch.crates-io]; assert each has comment with GH issue | TOML parse + regex |
| F-INFRA-005 | no_exclusions | Parse ci.yml `--exclude` flags; assert count <= 2 (GPU-only acceptable) | YAML parse |
| F-INFRA-006 | rustflags_consistent | Extract RUSTFLAGS from all workflows; assert identical set | YAML parse |

## Implementation Priority

| Priority | RC | Fix | Effort | Impact |
|----------|-----|-----|--------|--------|
| P0 | RC1 | Add `branches: [main]` to 3 workflows | 5 min | -12 false failures/day |
| P0 | RC2 | `cfg(unix)` gate on `nix` dep | 15 min | Fixes 100% nightly failures |
| P0 | RC6 | Align RUSTFLAGS across workflows | 10 min | Stops cascading fix cycles |
| P1 | RC4 | Pin or remove `cc` patch | 5 min | Removes floating external dep |
| P1 | RC3 | Pin sibling repos to tags in nightly.yml | 30 min | Stops cross-repo breakage |
| P2 | RC5 | Remove `--exclude` flags (needs API convergence) | 2-4 hrs | Full workspace coverage |

## Relationship to Existing Contracts

| Contract | Overlap | ci-infra-v1 Adds |
|----------|---------|-----------------|
| `ci-entity-v1` | Scores CI quality (permissions, SHA pins) | **Build reproducibility** (trigger filters, platform gates, pinned deps) |
| `repo-entity-v1` | Scores repo structure | **CI-specific** infra (RUSTFLAGS, exclusions, patches) |
| `spec-schema-v1` | Spec system itself | **CI enforcement** of spec contracts |
