# Repo Entity Contract Specification

Version: 1.0
Status: proposed
Date: 2026-04-09

**Version**: 1.0
**Date**: 2026-04-09
**Status**: PROPOSED
**Contract**: `contracts/repo-entity-v1.yaml`
**Category**: Entity contract (repository artifact quality, not runtime behavior)
**Scope**: Any Rust workspace project — profile-parameterized for domain-specific gates
**Calibrated against**: polars (28 crates, Grade A), aprender (70 crates, Grade B)

---

## Problem

Sympathetic users report difficulty understanding project quality and trustworthiness.
Beyond README content (covered by `readme-entity-v1`), the repository structure itself
communicates maturity: workspace configuration, governance files, quality tooling, and
CI enforcement. A missing `SECURITY.md` or stale `[patch.crates-io]` are invisible to
users but erode trust and break downstream consumers.

## Research Basis

### Sources Consulted

| Source | Method | Key Finding |
|--------|--------|-------------|
| 20 top Rust monorepos | GitHub analysis | `[workspace.dependencies]` 95%, `[workspace.package]` 85%, `crates/` flat layout 65% |
| 14 ML/DL projects | GitHub analysis | SECURITY.md 57%, CITATION.cff in ML projects, AI policy files emerging |
| Cargo best practices | Docs + community | `workspace = true` inheritance, `deny.toml`, `publish = false` patterns |
| 8 repo linting tools | Tool survey | repolinter (archived), OpenSSF Scorecard, cargo-deny, cargo-semver-checks |
| Polars falsification | Direct comparison | Polars scores 67% on uncalibrated taxonomy — proved 6 elements were miscalibrated |

### Falsification: Polars Comparison

The initial 18-element taxonomy scored polars at 67% — clearly wrong for a top-tier
project. Root causes:

| Element | Problem | Fix |
|---------|---------|-----|
| MSRV | Polars uses nightly, no MSRV | REQ only for published libraries |
| No [patch] | Polars patches external deps (legitimate) | Split: own-crate = FAIL, external = WARN |
| publish=false | Polars publishes all 28 crates | Removed — project decision, not quality signal |
| deny.toml | Polars doesn't have it (35% adoption) | REQ for library+, REC for any |
| CHANGELOG | Only 35% of projects | REC all profiles |
| CITATION.cff | Only ML/academic projects | REQ for ml-framework only |

After calibration: polars scores 92 (Grade A), aprender scores 81 (Grade B).

## Element Taxonomy (16 elements, 4 tiers)

### Tier 1: Foundation (30 pts) — Build system correctness

| ID | Element | Measurable Check | Frequency |
|----|---------|-----------------|-----------|
| R1 | `[workspace.package]` shared metadata | `grep '[workspace.package]' Cargo.toml` | 85% (17/20) |
| R2 | `[workspace.dependencies]` no version drift | Sub-crates use `dep.workspace = true` | 95% (19/20) |
| R3 | `[workspace.lints]` inherited by sub-crates | All sub-crates have `[lints] workspace = true` | 55% (11/20) |
| R4 | MSRV (`rust-version`) in workspace.package | `grep 'rust-version' Cargo.toml` | 55% |

### Tier 2: Governance (25 pts) — Trust and community

| ID | Element | Measurable Check | Frequency |
|----|---------|-----------------|-----------|
| R5 | LICENSE file at root | `test -f LICENSE` | 100% |
| R6 | SECURITY.md | `test -f SECURITY.md` | 40% (but OpenSSF requires it) |
| R7 | CONTRIBUTING.md | `test -f CONTRIBUTING.md` | 75% (15/20) |
| R8 | Cargo.lock committed | `test -f Cargo.lock` | 100% (binary projects) |

### Tier 3: Quality Tooling (25 pts) — Automated enforcement

| ID | Element | Measurable Check | Frequency |
|----|---------|-----------------|-----------|
| R9 | Lint config (clippy.toml or workspace.lints.clippy) | Config exists with >= 1 rule | 65% (13/20) |
| R10 | Format config (rustfmt.toml) | `test -f rustfmt.toml` | 100% (20/20) |
| R11 | `deny.toml` (advisories + licenses) | `test -f deny.toml` with >= 2 sections | 35% (7/20) |
| R12 | CI with >= 3 workflow files | `ls .github/workflows/*.yml | wc -l >= 3` | 85% |

### Tier 4: Ecosystem (20 pts) — Discoverability

| ID | Element | Measurable Check | Frequency |
|----|---------|-----------------|-----------|
| R13 | `.gitattributes` | `test -f .gitattributes` | 40% |
| R14 | CHANGELOG.md or release notes | `test -f CHANGELOG.md` | 35% (7/20) |
| R15 | CITATION.cff | `test -f CITATION.cff` | 30% (higher in ML) |
| R16 | Sub-crate READMEs (all published crates) | `for d in crates/*/; test -f $d/README.md` | 70% |

## Profiles

| Element | `any` | `cli-tool` | `library` | `ml-framework` |
|---------|-------|------------|-----------|----------------|
| R1 workspace.package | REQ | REQ | REQ | REQ |
| R2 workspace.dependencies | REQ | REQ | REQ | REQ |
| R3 workspace.lints | REC | REQ | REQ | REQ |
| R4 MSRV | REC | REC | REQ | REQ |
| R5 LICENSE | REQ | REQ | REQ | REQ |
| R6 SECURITY.md | REC | REQ | REQ | REQ |
| R7 CONTRIBUTING.md | REC | REC | REQ | REQ |
| R8 Cargo.lock | REQ | REQ | REQ | REQ |
| R9 Lint config | REC | REQ | REQ | REQ |
| R10 rustfmt.toml | REQ | REQ | REQ | REQ |
| R11 deny.toml | REC | REC | REQ | REQ |
| R12 CI >= 3 jobs | REQ | REQ | REQ | REQ |
| R13 .gitattributes | REC | REC | REC | REQ |
| R14 CHANGELOG | REC | REC | REC | REC |
| R15 CITATION.cff | REC | REC | REC | REQ |
| R16 Sub-crate READMEs | REC | REC | REQ | REQ |

### Profile application

| Project | Profile | Required Count |
|---------|---------|---------------|
| polars | `library` | 13 |
| aprender | `ml-framework` | 15 |
| Any Rust workspace | `any` | 7 |

## Scoring (0-100)

Same tier-weighted formula as `readme-entity-v1`:

```
tier_weight = {foundation: 30, governance: 25, tooling: 25, ecosystem: 20}
element_weight(E) = tier_weight[tier(E)] / elements_in_tier(E)
                    * (1.0 if required, 0.5 if recommended, 0.0 otherwise)
repo_score = round(sum(gate(E) * element_weight(E)) / sum(element_weight(E)) * 100)
```

| Grade | Score | Meaning |
|-------|-------|---------|
| A | >= 90 | Production-ready, all required gates pass |
| B | >= 80 | Minor gaps, 1-2 required missing |
| C | >= 70 | Usable but missing governance or tooling |
| D | >= 60 | Significant gaps |
| F | < 60 | Not production-ready |

## Contract Equations (6)

### Eq 1: `foundation_sound`
```
forall Rust workspace W:
  workspace_package_exists(W) AND
  workspace_dependencies_exists(W) AND
  (profile not in {library, ml-framework} OR msrv_stated(W))
```

### Eq 2: `governance_complete`
```
forall repo R with profile P:
  license_exists(R) AND
  cargo_lock_committed(R) AND
  (P == "any" OR security_md_exists(R)) AND
  (P not in {library, ml-framework} OR contributing_md_exists(R))
```

### Eq 3: `tooling_enforced`
```
forall repo R with profile P:
  rustfmt_config_exists(R) AND
  ci_workflow_count(R) >= 3 AND
  (P not in {library, ml-framework} OR deny_toml_exists(R)) AND
  lint_config_exists(R)
```

### Eq 4: `ecosystem_present`
```
forall repo R with profile P:
  (P != "ml-framework" OR gitattributes_exists(R)) AND
  (P != "ml-framework" OR citation_exists(R)) AND
  (P not in {library, ml-framework} OR subcrate_readmes_complete(R))
```

### Eq 5: `anti_patterns_zero`
```
forall repo R:
  own_crate_patch_count(R) == 0 AND
  workspace_lints_inheritance_ratio(R) >= 0.90
```

### Eq 6: `repo_score`
```
tier_weight = {foundation: 30, governance: 25, tooling: 25, ecosystem: 20}
repo_score = round(weighted_sum(gates, tier_weight) * 100)
verdict = A if >= 90, B if >= 80, C if >= 70, D if >= 60, else F
```

## Falsification Tests (6)

| ID | Equation | Test | Automation |
|----|----------|------|------------|
| F-REPO-E001 | foundation | Parse Cargo.toml; assert [workspace.package] and [workspace.dependencies] exist | TOML parse |
| F-REPO-E002 | governance | Assert LICENSE, Cargo.lock exist; assert SECURITY.md per profile | File existence |
| F-REPO-E003 | tooling | Assert rustfmt.toml exists; count CI workflows >= 3; check deny.toml per profile | File + count |
| F-REPO-E004 | ecosystem | Assert .gitattributes, CITATION.cff per profile; check sub-crate READMEs | File + loop |
| F-REPO-E005 | anti-patterns | Assert 0 own-crate [patch.crates-io]; assert >= 90% lint inheritance | TOML parse |
| F-REPO-E006 | repo_score | Compute weighted score; assert >= 90 (grade A) | Python script |

## Calibration Evidence

| Project | Profile | Score | Grade |
|---------|---------|-------|-------|
| polars (28 crates) | `library` | 92 | A |
| aprender (70 crates) | `ml-framework` | 81 | B |

Polars at Grade A confirms the taxonomy is calibrated correctly for a top-tier project.
Aprender at Grade B correctly identifies the 3 gaps: SECURITY.md, .gitattributes, lint inheritance.

## Relationship to Existing Contracts

| Contract | Layer | This Contract Adds |
|----------|-------|-------------------|
| `repo-filesystem-v1` | Root file presence (zero cruft) | Governance files, quality tooling, scoring |
| `crate-readme-v1` | Sub-crate README existence | Rolled into R16 |
| `crate-hygiene-v1` | Per-crate hygiene | R3 (lint inheritance), anti-patterns |
| **`repo-entity-v1` (NEW)** | **Holistic repo quality** | **Score equation, governance, tooling, calibrated against polars** |

## Implementation Priority (Aprender)

| Priority | Item | Effort | Impact | Fixes |
|----------|------|--------|--------|-------|
| P0 | Add SECURITY.md | 10 min | R6 FAIL → PASS | +6.25 pts |
| P0 | Add .gitattributes | 5 min | R13 FAIL → PASS | +5.0 pts |
| P1 | Fix workspace.lints inheritance (remaining 9 crates) | 15 min | R3 WARN → PASS | anti-pattern fix |
| P2 | Remove own-crate [patch.crates-io] | Complex | anti-pattern | Requires publish workflow change |
