<!-- PCU: ch14b-entity-contracts | contract: contracts/readme-entity-v1.yaml -->
<!-- Example: none (documentation-only chapter) -->
<!-- Status: enforced -->
# Entity Contracts: Scoring Repository Artifacts 0-100

Entity contracts are a new category of provable contract that scores **repository
artifacts** — the files, configurations, and documentation that make up a project.
Unlike runtime contracts (Chapter 14) which verify behavior during execution,
entity contracts are static: they can be checked by parsing files, counting
elements, and making HTTP requests.

## Why Entity Contracts?

User feedback on aprender consistently reported:

- "Hard to understand what the project does"
- "Can't tell if claims are true"
- "Don't know if it's maintained"

These are **artifact quality** problems, not code bugs. A missing README section,
stale benchmark numbers, or absent `SECURITY.md` erode trust just as much as a
failing test. Entity contracts make this measurable.

## The Scoring System

Every entity contract:

1. Defines **elements** — discrete, measurable properties (e.g., "README has a ToC")
2. Groups elements into **tiers** with weights (identity=30, proof=35, etc.)
3. Applies a **profile** (`any`, `cli-tool`, `library`, `ml-framework`)
4. Computes a **weighted score** from 0-100 with letter grades

```text
element_weight(E) = tier_weight[tier(E)] / elements_in_tier(E)
                    * (1.0 if required, 0.5 if recommended)

score = round(sum(gate(E) * element_weight(E)) / sum(element_weight(E)) * 100)
```

| Grade | Score | Meaning |
|-------|-------|---------|
| A | >= 90 | Production-ready |
| B | >= 80 | Minor gaps |
| C | >= 70 | Usable but missing signals |
| D | >= 60 | Significant gaps |
| F | < 60 | Not ready |

## The 14 Contracts

### Repo-Level (9 contracts, 105 elements)

| Contract | What It Scores | Key Elements |
|----------|---------------|--------------|
| `readme-entity-v1` | README.md quality | One-liner, install, expected output, freshness, ToC, architectures |
| `repo-entity-v1` | Repository structure | workspace.package, SECURITY.md, deny.toml, CI workflows |
| `crate-entity-v1` | Per-crate Cargo.toml | description, keywords, categories, docs.rs metadata |
| `ci-entity-v1` | GitHub Actions | SHA-pinned actions, permissions, timeouts, lint/test/audit jobs |
| `changelog-entity-v1` | CHANGELOG.md | Keep-a-Changelog format, dates, PR links, version-tag alignment |
| `dep-entity-v1` | Dependency health | deny.toml sections, dependabot, no wildcard deps |
| `spec-entity-v1` | Specification docs | Status/Version/Date fields, equations, TOC linkage |
| `api-entity-v1` | Public API docs | missing_docs lint, doc examples, docs.rs config |
| `error-entity-v1` | CLI error messages | Exit codes, stderr, no silent failures |

### Domain-Specific (5 contracts, 50 elements)

| Contract | What It Scores | Key Elements |
|----------|---------------|--------------|
| `subcommand-entity-v1` | Each CLI command (57) | --help, exit codes, --json valid, no phantom commands |
| `model-entity-v1` | Model files | Magic bytes, not truncated, architecture known, no NaN |
| `template-entity-v1` | Chat templates | Valid Jinja2, roles, multi-turn, no injection |
| `binary-entity-v1` | The `apr` binary | Version hash, completions, man page, SIGINT handling |
| `server-entity-v1` | `apr serve` | /health, OpenAI compat, SSE streaming, graceful shutdown |

## Profiles

Profiles determine which elements are required. They inherit:

```text
ml-framework ⊃ library ⊃ cli-tool ⊃ any
```

| Profile | Use Case | Typical Required % |
|---------|----------|-------------------|
| `any` | Any Rust workspace | 40-50% |
| `cli-tool` | CLI applications | 60-70% |
| `library` | Published crates | 70-80% |
| `ml-framework` | ML/AI projects (aprender) | 80-95% |

## Calibration

Every contract was **falsified against polars** (28-crate Rust monorepo) to prevent
false failures. If polars — a well-maintained top-tier project — would fail a gate,
the gate is miscalibrated.

Example: The initial taxonomy scored polars at 67%. Investigation revealed 6
miscalibrated elements (MSRV required for nightly projects, deny.toml required
when only 35% of projects use it). After calibration, polars scores 92 (Grade A).

## Applying to Your Project

Add a profile declaration to your project:

```yaml
# In your contract YAML or CI config
entity_contracts:
  profile: library    # or: any, cli-tool, ml-framework
  readme_path: README.md
  freshness_days: 90
```

The `any` profile requires only universal elements: one-liner description, badges,
install command, example with expected output, freshness signal, ToC (if >100 lines),
license, and no anti-patterns. These 8 elements are table stakes for any maintained
open source project.

## Running the Scores

Each contract includes a Python scoring script in its falsification tests. Run them:

```bash
# Score your README
python3 -c "$(grep -A100 'F-README-E006' contracts/readme-entity-v1.yaml | \
  sed -n '/python3/,/^  if_fails/p')"

# Score your repo structure
python3 -c "$(grep -A100 'F-REPO-E006' contracts/repo-entity-v1.yaml | \
  sed -n '/python3/,/^  if_fails/p')"
```

## Current Aprender Scores

| Contract | Score | Grade |
|----------|-------|-------|
| readme-entity-v1 | 100 | A |
| repo-entity-v1 | 100 | A |
| crate-entity-v1 | 100 | A |
| ci-entity-v1 | 100 | A |
| changelog-entity-v1 | 100 | A |
| dep-entity-v1 | 100 | A |
| spec-entity-v1 | 100 | A |
| api-entity-v1 | 100 | A |
| error-entity-v1 | 100 | A |
| subcommand-entity-v1 | 100 | A |
| model-entity-v1 | 100 | A |
| template-entity-v1 | 100 | A |
| binary-entity-v1 | 100 | A |
| server-entity-v1 | 100 | A |
| **Total** | **14/14 Grade A** | |

## Relationship to Runtime Contracts

Entity contracts complement, not replace, runtime contracts:

| Layer | Contract Type | Example |
|-------|-------------|---------|
| **Artifact quality** | Entity contracts | README has expected output |
| **Structural validity** | Document integrity | Heading hierarchy valid |
| **File existence** | Crate hygiene | Every crate has README |
| **Runtime behavior** | Provable contracts (Ch. 14) | `apr validate` exits non-zero on bad input |
| **Mathematical proof** | Kernel contracts | softmax output sums to 1.0 |

Each layer catches different classes of defects. Entity contracts catch the
"invisible" quality erosion that users feel but can't articulate.
