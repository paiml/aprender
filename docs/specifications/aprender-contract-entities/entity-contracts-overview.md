# Entity Contracts Overview

Version: 1.0
Status: proposed
Date: 2026-04-09

**Version**: 1.0
**Date**: 2026-04-09
**Status**: PROPOSED
**Category**: Entity contracts — repository artifact quality scoring (not runtime behavior)

---

## What Are Entity Contracts?

Entity contracts validate **repository artifacts** — the files, configurations, and
documentation that make up a project. Unlike runtime contracts (which verify behavior
during execution), entity contracts are static: they can be checked by parsing files,
counting elements, and making HTTP requests. No compilation or execution required.

Every entity contract:
- Scores 0-100 with letter grades (A >= 90, B >= 80, C >= 70, D >= 60, F < 60)
- Uses tier-weighted scoring (important elements contribute more)
- Is profile-parameterized (`any`, `cli-tool`, `library`, `ml-framework`)
- Was calibrated against polars (28-crate Rust monorepo) to prevent false failures
- Has falsification tests that can run in CI

## Contract Registry (14 contracts, 155 elements)

### Repo-Level Entities (9 contracts, 105 elements)

| # | Contract | Elements | Tiers | Scope | Calibration |
|---|----------|----------|-------|-------|-------------|
| 1 | [readme-entity-v1](../../../contracts/readme-entity-v1.yaml) | 15 | identity, proof, orientation, ecosystem | README.md content quality | polars A, aprender A |
| 2 | [repo-entity-v1](../../../contracts/repo-entity-v1.yaml) | 16 | foundation, governance, tooling, ecosystem | Repository structure | polars A, aprender A |
| 3 | [crate-entity-v1](../../../contracts/crate-entity-v1.yaml) | 12 | foundation, discoverability, governance, quality | Per-crate Cargo.toml | polars B, aprender A |
| 4 | [ci-entity-v1](../../../contracts/ci-entity-v1.yaml) | 12 | security, reliability, coverage, operations | GitHub Actions CI | bevy A, aprender A |
| 5 | [changelog-entity-v1](../../../contracts/changelog-entity-v1.yaml) | 10 | structure, content, lifecycle, completeness | CHANGELOG.md quality | ruff A, aprender A |
| 6 | [dep-entity-v1](../../../contracts/dep-entity-v1.yaml) | 10 | security, hygiene, freshness, quality | Dependency health | bevy A, aprender A |
| 7 | [spec-entity-v1](../../../contracts/spec-entity-v1.yaml) | 10 | lifecycle, rigor, hygiene, traceability | Specification documents | k8s KEPs A, aprender A |
| 8 | [api-entity-v1](../../../contracts/api-entity-v1.yaml) | 10 | coverage, quality, safety, discoverability | Public API docs | tokio A, aprender A |
| 9 | [error-entity-v1](../../../contracts/error-entity-v1.yaml) | 10 | correctness, usability, consistency, testing | CLI error messages | rustc A, aprender A |

### Domain-Specific Entities (5 contracts, 50 elements)

| # | Contract | Elements | Tiers | Scope | Calibration |
|---|----------|----------|-------|-------|-------------|
| 10 | [subcommand-entity-v1](../../../contracts/subcommand-entity-v1.yaml) | 10 | completeness, correctness, safety, governance | Per-CLI-subcommand quality | cargo A, ollama B |
| 11 | [model-entity-v1](../../../contracts/model-entity-v1.yaml) | 10 | integrity, metadata, tensors, vocab | Per-model-file readiness | llama.cpp A, HF Hub A |
| 12 | [template-entity-v1](../../../contracts/template-entity-v1.yaml) | 10 | syntax, coverage, tokens, security | Per-chat-template completeness | HF transformers A, ollama C |
| 13 | [binary-entity-v1](../../../contracts/binary-entity-v1.yaml) | 10 | identity, usability, safety, interop | Per-binary CLI quality | ripgrep A, cargo A |
| 14 | [server-entity-v1](../../../contracts/server-entity-v1.yaml) | 10 | reliability, compatibility, correctness | Per-server endpoint quality | vLLM A, TGI A |
| | **Total** | **155** | | | |

## Profile System

Profiles determine which elements are required vs recommended. They inherit:
`ml-framework` includes `library` includes `cli-tool` includes `any`.

| Profile | Use For | Required Elements (typical) |
|---------|---------|----------------------------|
| `any` | Any Rust workspace | ~40-50% of elements |
| `cli-tool` | CLI applications | ~60-70% |
| `library` | Published crates | ~70-80% |
| `ml-framework` | ML/AI projects | ~80-95% |

## Scoring Formula (Shared Across All Contracts)

```
tier_weight = {tier1: W1, tier2: W2, tier3: W3, tier4: W4}  # sum = 100
element_weight(E) = tier_weight[tier(E)] / elements_in_tier(E)
                    * (1.0 if required, 0.5 if recommended, 0.0 otherwise)
score = round(sum(gate(E) * element_weight(E)) / sum(element_weight(E)) * 100)
```

## Relationship to Existing Contracts

Entity contracts operate at the **artifact quality** layer. They complement:
- `document-integrity-v1` — structural markdown/YAML/SVG validation
- `crate-readme-v1` — sub-crate README existence
- `crate-hygiene-v1` — per-crate hygiene rules
- `repo-filesystem-v1` — root file presence / zero cruft
- `apr-docs-v1` — install command correctness

No overlap — entity contracts score **content quality and completeness**,
existing contracts verify **structural correctness and existence**.

## Aprender Current Scores (ml-framework profile)

| Contract | Score | Grade | Top Gap |
|----------|-------|-------|---------|
| readme-entity-v1 | 100 | A | — |
| repo-entity-v1 | 100 | A | — |
| crate-entity-v1 | TBD | C | keywords/categories on ~20 crates |
| ci-entity-v1 | TBD | B | SHA pinning, permissions |
| changelog-entity-v1 | TBD | A | — |
| dep-entity-v1 | TBD | A | — |
| spec-entity-v1 | TBD | C | lifecycle status fields |
| api-entity-v1 | TBD | F | no missing_docs lint |
| error-entity-v1 | TBD | B | JSON error schema |
