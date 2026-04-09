# README Entity Contract Specification

Version: 1.0
Status: proposed
Date: 2026-04-09

**Version**: 1.0
**Date**: 2026-04-09
**Status**: PROPOSED
**Contract**: `contracts/readme-entity-v1.yaml`
**Category**: Entity contract (repository artifact quality, not runtime behavior)
**Scope**: Any software project — profile-parameterized for domain-specific gates

---

## Problem

User feedback on aprender and sibling projects consistently reports:

1. **"Hard to understand what the project does"** — README leads with implementation details, not value
2. **"Unclear what value it adds"** — No comparison or positioning against alternatives
3. **"Can't tell if claims are true or false"** — Benchmarks without reproduction commands, no expected output in examples
4. **"Don't know if it's maintained"** — No freshness signal (last verified date, version recency)

Analysis of 21 top Rust projects (ripgrep, tokio, serde, polars, etc.) and 25 ML/DL
projects (PyTorch, llama.cpp, ollama, candle, burn) reveals 15 discrete, measurable
README elements across 4 tiers. The current aprender README scores 6/15 (40%).

## Goal

A contract that enforces README quality through falsifiable equations. The contract:

- Works for **any project** via profile parameterization
- Is **machine-checkable** (regex, line counting, HTTP HEAD — no ML, no heuristics)
- **Composes** with existing structural contracts (`document-integrity-v1`, `crate-readme-v1`)
- Operates at the **semantic content** layer (existing contracts cover existence and structure)

## Research Basis

### Sources Consulted

| Source | Method | Key Finding |
|--------|--------|-------------|
| Web (README scoring, standard-readme) | Web search | 15-element taxonomy; install + runnable example = highest adoption correlation |
| Rust top projects (21 analyzed) | GitHub README analysis | 100% have badges; 86% have copy-paste examples; 38% have reproducible benchmarks |
| ML/DL projects (25 analyzed) | GitHub README analysis | Supported models table, hardware requirements, "is this real?" trust signals |
| Aprender oracle (stack READMEs) | `batuta oracle --rag` | Stack siblings have ToC, MSRV, quality section — aprender is the outlier |
| README linters (6 tools) | Tool survey | readme-score-api, standard-readme, Vale, remark-lint — none check freshness |

### Existing Tools and Their Gaps

| Tool | Checks | Does NOT Check |
|------|--------|----------------|
| standard-readme | Section presence (Description, Install, Usage, License) | Content quality, freshness, claim verifiability |
| readme-score-api | Code blocks, images, links, length | Semantic clarity, expected output, staleness |
| Vale | Prose quality (passive voice, jargon, readability) | Section completeness, badges, trust signals |
| remark-lint | Markdown structure (heading levels, list style) | Content relevance, freshness, dead links |
| document-integrity-v1 (ours) | Heading hierarchy, link safety, code fences, tables | Semantic content, freshness, claim evidence |

**Gap**: No tool checks whether README **content communicates effectively and truthfully**.
This contract fills that gap.

## Element Taxonomy (15 Elements, 4 Tiers)

### Tier 1: Identity — "What does this project do?"

| ID | Element | Measurable Check | Top-Project Frequency |
|----|---------|------------------|-----------------------|
| E1 | One-line description | First non-HTML paragraph ≤ 150 chars, no implementation jargon | 95% |
| E2 | Value proposition | Section containing comparison table OR "Why" heading | 60% |
| E3 | Status badges | ≥ 4 badge images in first 20 lines (CI, version, license, +1) | 100% |

### Tier 2: Proof — "Is this true or vaporware?"

| ID | Element | Measurable Check | Top-Project Frequency |
|----|---------|------------------|-----------------------|
| E4 | Working install | Code block with install command in first 30 lines, ≤ 3 steps | 95% |
| E5 | Runnable example with output | Code block containing both command AND expected output | 70% |
| E6 | Freshness signal | Date/version within 90 days of HEAD OR CI badge resolves green | 40% explicit |

### Tier 3: Orientation — "Can I use this for my case?"

| ID | Element | Measurable Check | Top-Project Frequency |
|----|---------|------------------|-----------------------|
| E7 | Requirements (MSRV, OS, deps) | Dedicated section or badge stating minimum version | 65% |
| E8 | Supported models/architectures | Table or checklist with ≥ 3 entries (ML projects only) | 80% (ML) |
| E9 | Table of contents | Linked anchor list if README > 100 lines | 57% |

### Tier 4: Ecosystem — Adoption multipliers

| ID | Element | Measurable Check | Top-Project Frequency |
|----|---------|------------------|-----------------------|
| E10 | License (explicit) | Heading or badge with license name | 95% |
| E11 | CI/test evidence | Badge + test count or coverage stated in README body | 75% |
| E12 | Documentation links | Section with ≥ 2 links to docs.rs, book, API ref, specs | 81% |
| E13 | Contributing guide | Section heading or link to CONTRIBUTING.md | 86% |
| E14 | Feature list | Markdown list with ≥ 3 items under Features/capabilities heading | 81% |
| E15 | No anti-patterns | Zero dead links, zero aspirational claims without "planned" qualifier | N/A |

## Profiles

Profiles determine which elements are REQUIRED vs RECOMMENDED for a project type.
Profiles inherit: `ml-framework` ⊃ `library` ⊃ `cli-tool` ⊃ `any`.

| Element | `any` | `cli-tool` | `library` | `ml-framework` |
|---------|-------|------------|-----------|----------------|
| E1 One-liner | REQ | REQ | REQ | REQ |
| E2 Value prop | REC | REC | REC | REQ |
| E3 Badges ≥ 4 | REQ | REQ | REQ | REQ |
| E4 Install ≤ 3 steps | REQ | REQ | REQ | REQ |
| E5 Example with output | REQ | REQ | REQ | REQ |
| E6 Freshness ≤ 90 days | REQ | REQ | REQ | REQ |
| E7 Requirements/MSRV | REC | REC | REQ | REQ |
| E8 Supported models | — | — | — | REQ |
| E9 ToC (if > 100 lines) | REQ | REQ | REQ | REQ |
| E10 License | REQ | REQ | REQ | REQ |
| E11 CI/test evidence | REC | REQ | REQ | REQ |
| E12 Documentation links | REC | REC | REQ | REQ |
| E13 Contributing | REC | REC | REC | REC |
| E14 Feature list | REC | REQ | REQ | REQ |
| E15 No anti-patterns | REQ | REQ | REQ | REQ |

REQ = required (FAIL if missing), REC = recommended (WARN if missing).

## Scoring (0-100)

The score is a weighted sum, not a simple count. Tier weights reflect research
findings that identity + proof elements correlate 3-5x more with adoption than
ecosystem elements.

### Tier Weights (sum = 100)

| Tier | Weight | Elements | Per-Element Weight | Rationale |
|------|--------|----------|--------------------|-----------|
| Identity (E1-E3) | 30 | 3 | 10.0 | "What does it do?" — highest adoption signal |
| Proof (E4-E6) | 35 | 3 | 11.7 | "Is this true?" — trust is the #1 user complaint |
| Orientation (E7-E9) | 20 | 3 | 6.7 | "Can I use it?" — practical adoption gate |
| Ecosystem (E10-E15) | 15 | 6 | 2.5 | Community signals — nice-to-have |

### Element Multiplier

| Status | Multiplier | Rationale |
|--------|-----------|-----------|
| Required + PASS | 1.0 | Full weight |
| Recommended + PASS | 0.5 | Half weight — nice but not blocking |
| FAIL or absent | 0.0 | No credit |

### Formula

```
element_weight(E, profile) = tier_weight[tier(E)] / elements_in_tier(E)
                              * (1.0 if required, 0.5 if recommended, 0.0 otherwise)

readme_score = round(sum(gate(E) * element_weight(E)) / sum(element_weight(E)) * 100)
```

### Grade Thresholds

| Grade | Score | Meaning |
|-------|-------|---------|
| A | >= 90 | Adoption-ready, all required gates pass |
| B | >= 80 | Minor gaps, 1-2 recommended elements missing |
| C | >= 70 | Usable but missing trust/orientation signals |
| D | >= 60 | Significant gaps — hard for new users |
| F | < 60 | Not adoption-ready |

## Contract Equations (6)

### Eq 1: `identity_complete`
```
forall README with profile P:
  one_liner_length(first_non_html_paragraph) <= 150 AND
  badge_count(first_20_lines) >= 4 AND
  install_steps(first_code_block_with_install) <= 3
```

### Eq 2: `proof_verifiable`
```
forall README with profile P:
  exists code_block CB:
    CB contains command_line AND CB contains expected_output AND
  freshness_days(readme_date, HEAD_date) <= 90 AND
  forall url U in README:
    http_head(U).status != 404
```

### Eq 3: `orientation_complete`
```
forall README with profile P:
  (line_count <= 100 OR toc_exists_with_anchors(count >= 3)) AND
  (P != "library" OR msrv_stated) AND
  (P != "ml-framework" OR supported_models_count >= 3)
```

### Eq 4: `trust_signals`
```
forall README with profile P:
  license_stated AND
  (P == "any" OR ci_badge_present AND test_count_stated) AND
  aspirational_claims_without_qualifier == 0
```

### Eq 5: `anti_patterns_zero`
```
forall README:
  dead_link_count == 0 AND
  stale_version_reference_count == 0 AND
  bare_code_fence_count == 0 AND
  heading_hierarchy_valid
```

### Eq 6: `readme_score`
```
tier_weight = {identity: 30, proof: 35, orientation: 20, ecosystem: 15}

element_weight(E) =
  tier_weight[tier(E)] / |elements_in_tier(E)|
  * (1.0 if E in required(profile), 0.5 if recommended, 0.0 otherwise)

readme_score = round(sum(gate(E) * element_weight(E) for E in E1..E15)
                    / sum(element_weight(E) for E in E1..E15) * 100)

verdict = A if score >= 90, B if >= 80, C if >= 70, D if >= 60, else F
```

## Falsification Tests (6)

| ID | Equation | Test | Automation |
|----|----------|------|------------|
| F-README-E001 | identity | Parse first non-HTML paragraph; assert ≤ 150 chars, no words from jargon blocklist | Regex + word count |
| F-README-E002 | proof | Find code block with both input command and output line (comment with `#` prefix showing result, `>` output, or output annotation) | Regex on fenced blocks |
| F-README-E003 | freshness | Extract dates/versions from README; compare to `git log -1 --format=%ci`; assert delta ≤ 90 days. Fallback: HTTP HEAD CI badge URL, assert 200 | Date parse + HTTP |
| F-README-E004 | orientation | If wc -l > 100, assert section with ≥ 3 `[text](#anchor)` links exists | Regex + line count |
| F-README-E005 | anti-patterns | HTTP HEAD all URLs; assert 0 return 404. Assert 0 bare ``` fences. Assert heading hierarchy valid (no H1→H3 skip) | HTTP + regex |
| F-README-E006 | readme_score | Compute weighted score across all 15 elements; assert >= 90 (grade A) | Python script |

## Aprender Falsification

### Run 1 — Pre-fix baseline (`3ebecee5c`)

| Gate | Profile `ml-framework` | Result | Evidence |
|------|----------------------|--------|----------|
| E1 One-liner | REQ | **PASS** | "complete ML framework built from scratch in Rust" (55 chars) |
| E2 Value prop | REQ | **PASS** | Framework Comparison tables (lines 149-191) |
| E3 Badges ≥ 4 | REQ | **PASS** | 4 badges (crates.io, docs.rs, CI, MIT) |
| E4 Install ≤ 3 | REQ | **PASS** | 3-line Quick Start |
| E5 Example+output | REQ | **FAIL** | 8 code blocks, 0 show expected output |
| E6 Freshness | REQ | **FAIL** | Zero dates/versions/timestamps |
| E7 MSRV | REQ | **FAIL** | Not stated anywhere |
| E8 Models | REQ | **FAIL** | No supported architectures table |
| E9 ToC | REQ | **FAIL** | 239 lines, no ToC |
| E10 License | REQ | **PASS** | Line 238 + badge |
| E11 CI/test | REQ | **WARN** | CI badge exists, test count in "Numbers" but no coverage badge |
| E12 Docs links | REQ | **FAIL** | docs.rs badge only, no Documentation section |
| E13 Contributing | REC | **PASS** | Section at lines 228-234 |
| E14 Feature list | REQ | **PASS** | Command table (lines 33-43) |
| E15 Anti-patterns | REQ | **WARN** | Heading hierarchy OK; 1 bare code fence |

**Score: 7/14 required = 50% → FAIL**

### Run 2 — Post-fix (2026-04-09)

| Gate | Profile `ml-framework` | Result | Evidence |
|------|----------------------|--------|----------|
| E1 One-liner | REQ | **PASS** | "complete ML framework built from scratch in Rust" |
| E2 Value prop | REQ | **PASS** | Framework Comparison tables |
| E3 Badges ≥ 4 | REQ | **PASS** | 6 badges (crates.io, docs.rs, CI, MIT, downloads, MSRV) |
| E4 Install ≤ 3 | REQ | **PASS** | 3-line Quick Start in first 50 lines |
| E5 Example+output | REQ | **PASS** | Quick Start shows `# => 2 + 2 = 4.` |
| E6 Freshness | REQ | **PASS** | "Last verified: 2026-04-09" (delta 0 days) |
| E7 MSRV | REQ | **PASS** | MSRV-1.86 badge |
| E8 Models | REQ | **PASS** | 8 architectures in Supported Architectures table |
| E9 ToC | REQ | **PASS** | 287 lines, 13 ToC anchor links |
| E10 License | REQ | **PASS** | Line 287 + badge |
| E11 CI/test | REQ | **PASS** | CI badge + "25,391 tests" in Numbers section |
| E12 Docs links | REQ | **PASS** | Documentation section with 5 links |
| E13 Contributing | REC | **PASS** | Section present |
| E14 Feature list | REQ | **PASS** | Command table (8 categories) |
| E15 Anti-patterns | REQ | **PASS** | 0 bare opening fences, heading hierarchy valid |

**Score: 14/14 required = 100% → PASS**

## Relationship to Existing Contracts

| Contract | Layer | Scope |
|----------|-------|-------|
| `document-integrity-v1` | Structural | Markdown syntax, heading hierarchy, link safety, code fence tags |
| `crate-readme-v1` | Existence | Every crate has README ≥ 5 lines, links to monorepo |
| `apr-docs-v1` | Correctness | Install command matches binary, no stale repo references |
| **`readme-entity-v1` (NEW)** | **Semantic content** | **Value clarity, proof/trust, freshness, orientation, anti-patterns** |

No overlap — each contract operates at a different validation layer.

## Generalization: Applying to ANY Project

The contract is parameterized by `profile`. To apply to a new project:

```yaml
# In the target project's contract YAML or CI config:
readme_entity:
  profile: library    # or: any, cli-tool, ml-framework
  readme_path: README.md
  freshness_days: 90  # configurable
  min_badges: 4       # configurable
  max_oneliner: 150   # configurable
```

**Example applications:**

| Project | Profile | Required Elements |
|---------|---------|-------------------|
| `trueno` (SIMD compute) | `library` | E1-E7, E9-E12, E14-E15 |
| `apr` CLI | `cli-tool` | E1-E6, E9-E11, E14-E15 |
| Any GitHub project | `any` | E1, E3-E6, E9-E10, E15 |
| aprender | `ml-framework` | All 15 elements |

The `any` profile is deliberately minimal: one-liner, badges, install, example with
output, freshness, ToC, license, no anti-patterns. These 8 elements are universal to
every well-maintained open source project regardless of domain.

## Implementation Priority

| Priority | Item | Effort | Impact |
|----------|------|--------|--------|
| P0 | Add expected output to Quick Start code block | 5 min | Fixes E5 (highest-impact single element) |
| P0 | Add freshness section ("Last verified: YYYY-MM-DD") | 5 min | Fixes E6 |
| P1 | Add MSRV badge and requirements | 10 min | Fixes E7 |
| P1 | Add Supported Architectures table | 15 min | Fixes E8 |
| P1 | Add Table of Contents | 10 min | Fixes E9 |
| P1 | Add Documentation section | 5 min | Fixes E12 |
| P2 | Add coverage badge | 5 min | Fixes E11 WARN → PASS |
| P2 | CI job to verify freshness date ≤ 90 days | 30 min | Automated enforcement |
