# 35. Commit-Level Entity Enforcement

> How PMAT enforces provable-contracts at `git commit` time.

## Problem

Contract verification was a CI concern — minutes after commit, developer
has context-switched. AWS Cedar (ICSE 2025) documents "proof brittleness":
proofs break from unrelated commits. The fix: shift-left to commit time.

Two parallel contract systems — `pmat work` (DbC v5.0 `contract.json`)
and provable-contracts (YAML) — never merged at the git boundary. Non-code
assets (README, Dockerfile, SVG) had no contracts at all.

## Architecture: Three Entity Types

Entity enforcement operates on three entity types, all unified under a
single pre-commit pipeline:

| Entity Type | Examples | Enforcement Mechanism |
|-------------|----------|----------------------|
| **Code entities** | Functions with provable-contracts bindings | CB-1338..1343, CB-1350..1351 |
| **Work entities** | Active work items with `.pmat-work/*/contract.json` | CB-1331, CB-1352..1353 |
| **Asset entities** | README, Dockerfile, SVG, CHANGELOG, forjar.yaml, mdBook | CB-1320..1326 |

All three feed into the verification ladder (L0–L5), creating an
integrated compliance surface at the commit boundary.

## O(1) Firm Requirement

**Every pre-commit check MUST complete in < 30ms from cached data.** No
check may invoke `cargo build`, `cargo test`, `pv lint` (cold), or network
calls. All verification data pre-computed during development and cached in
`.pmat/`:

| Cache File | Content | Written By |
|------------|---------|------------|
| `.pmat/contract-cache.json` | Work item status, obligation counts | `pmat comply refresh-bindings` |
| `.pmat/verification-levels.json` | L-level per binding | `pmat comply refresh-bindings` |
| `.pmat/asset-layout-cache.json` | Asset validation results | `pmat comply refresh-bindings` |
| `.pmat/binding-index.json` | File→binding reverse index | `pmat comply refresh-bindings` |

Staleness policy: >7 days = warning (CB-1332), >30 days = error.
Emergency bypass: `git commit --no-verify` (logged in
`.pmat-metrics/bypass-log.jsonl`).

## Eight-Phase Pipeline

### Phase 0: Cache Infrastructure (CB-1332)

Check `.pmat/` cache file ages. Gate all subsequent phases on freshness.

### Phase 1: Work Contract Validity (CB-1331)

Validate `contract.json` in `.pmat-work/` directories for required fields
(`work_item_id`, `claims`, `require`, `ensure`). Accepts both v4
(`work_item_id` only) and v5 (full DbC) contract formats.

**Bridge to provable-contracts:** `require[]` → preconditions,
`ensure[]` → postconditions, `invariant[]` → invariants,
`falsification_method` → `falsification_tests[].method`.

### Phase 2: Verification Level Monotonicity Ratchet (CB-1330)

Each binding carries a verification level (L0–L5). The ratchet ensures
levels **never decrease**, preventing deletion of Kani harnesses or Lean
proofs.

| Transition | Allowed? | Rationale |
|-----------|----------|-----------|
| L1→L2, L2→L3, L3→L4 | Yes | Progress |
| L3→L2, L4→L3 | **BLOCKED** | Regression |
| Any→L0 | **BLOCKED** | Total regression |

Escape hatch: `pmat comply ratchet-override --binding <fn> --from L4
--to L2 --reason "..."` writes signed entry expiring in 14 days.

**Basis:** Agent Behavioral Contracts (arXiv:2602.22302) Drift Bounds
Theorem; Gradual Verification (Bader et al., TOPLAS).

### Phase 3: Asset Layout Contracts (CB-1320..1326)

Non-code assets use the **grid protocol model** (derived from rmedia's
16×9 cell grid). Every asset is a **container** of named **slots** with
mathematical placement constraints. Content never exists without a
placement contract — the rmedia invariant that every cell in the grid
must be occupied by exactly one allocation, with zero overlap and full
coverage.

#### The Grid Protocol Paradigm

rmedia proves the concept: a 16-column × 9-row grid (1920×1080 canvas,
120px cells) enforces that content placement is a constraint satisfaction
problem, not a free-form layout task. Key mathematical properties:

| Property | Invariant | Enforcement |
|----------|-----------|-------------|
| **Cell uniqueness** | No cell occupied by >1 allocation | `HashSet<(col, row)>` overlap check |
| **Bounds compliance** | `col_end < 16, row_end < 9` | Allocation rejects out-of-bounds |
| **Full coverage** | `occupied_cells / 144 ≥ 1.0` | Spacer nodes fill empty regions |
| **Slot ordering** | Title at row 0, footer at row 8, content rows 1–7 | Layout engine enforces position |
| **Text width budget** | Label width ≤ content area (0.6em estimate) | Validation check 12 |

The same paradigm applies to markdown documents: a README is a 1-column
grid where sections are slots with required ordering, and the "full
coverage" invariant becomes "all required sections present."

Layout is a constraint satisfaction problem (CSP). SMT-Layout (arXiv
2411.12271) proves that real-world GUI layout can be encoded as MaxSMT
with hard constraints (invariants) and soft constraints (preferences),
solved in milliseconds. ORCSolver (arXiv 2002.09925) generalizes this
to OR-constraints unifying grid and flow layouts under a single
specification. LACE (arXiv 2402.04754) demonstrates that layout constraints
— alignment, non-overlap, spacing — are differentiable mathematical
functions enforceable during generation.

#### Seven Asset Types

| CB | Asset | Slots | Key Constraints |
|----|-------|-------|-----------------|
| CB-1320 | README.md | 10 (7 required) | Ordering, section headings, accuracy vs `pmat --version` |
| CB-1321 | Dockerfile | 4 instruction blocks | No `:latest`, no `curl\|bash`, non-root USER, pinned deps |
| CB-1322 | SVG | viewBox grid cells | Cell uniqueness, accessibility (`<title>`/`aria-label`), element budget |
| CB-1323 | forjar.yaml | DAG nodes | Acyclicity, template resolution, no plaintext secrets |
| CB-1324 | mdBook | SUMMARY chapters | Link integrity, code block compilation, cross-refs |
| CB-1325 | CHANGELOG | Version entries | Keep-a-Changelog format, semver ordering |
| CB-1326 | Badges | Badge row | Required set (CI, version, license), URLs live, header placement |

#### README as a Slot Grid (CB-1320)

The README contract defines a 1-column ordered slot grid:

```
Slot 0: title        [REQUIRED]  # heading
Slot 1: badges       [REQUIRED]  shields.io / img references
Slot 2: description  [REQUIRED]  project summary paragraph
Slot 3: installation [REQUIRED]  ## Install / Setup / Getting Started
Slot 4: usage        [REQUIRED]  ## Usage / Examples
Slot 5: benchmarks   [optional]  ## Benchmarks
Slot 6: architecture [optional]  ## Architecture
Slot 7: api          [optional]  ## API
Slot 8: contributing [optional]  ## Contributing
Slot 9: license      [REQUIRED]  ## License
Slot 10: footer      [optional]  closing div/sub element
```

**Current enforcement (substring matching):** Searches `content.to_lowercase()`
for heading patterns like `## install`, `## usage`, `## license`. Detects
badges via `shields.io` / `![` / `[![` substring presence.

**Limitation:** Substring matching has no structural awareness — it matches
headings inside code fences, HTML comments, and block quotes. It cannot
enforce slot ordering (§3 before §4) because it doesn't parse the heading
hierarchy.

#### SVG as a Cell Grid (CB-1322)

SVG assets map directly to the rmedia grid protocol. The contract checks:
1. `viewBox` attribute present (defines the coordinate system)
2. Accessibility: `<title>` or `aria-label` for screen readers
3. Element budget: reasonable number of top-level elements

SVG structure is inherently amenable to formal specification — SVGenius
(arXiv 2506.03139) demonstrates that SVG documents have verifiable
structural properties across 24 application domains. The rmedia validator
enforces 20 checks including cell uniqueness, bounds compliance, viewBox
parity (`width/height` matches `viewBox="0 0 1920 1080"`), color hint
validity, badge sequence ordering, and arrowhead marker presence.

#### Markdown Structure as Formal Grammar

Grammar-Aligned Decoding (arXiv 2405.21047, NeurIPS 2024) proves that
structured document output can be generated with mathematical guarantees
of grammar conformance — the formal grammar acts as a contract and the
algorithm provides a proof that all outputs satisfy it. This is the
theoretical foundation for treating markdown heading hierarchy as a
verifiable specification: the document structure is a context-free grammar
where headings define a tree, and the contract asserts properties of
that tree (ordering, completeness, depth bounds).

#### Accessibility as Contract Taxonomy

WCAG compliance is implicitly a contract-checking problem. AccessGuru
(arXiv 2507.19549) introduces a three-category taxonomy of violations:

| Category | Contract Analogy | Example |
|----------|-----------------|---------|
| Syntactic | Precondition failure | Missing `alt` attribute on `<img>` |
| Semantic | Postcondition failure | `alt=""` present but meaningless |
| Layout | Invariant breach | Insufficient color contrast ratio |

PDF accessibility benchmarking (arXiv 2509.18965) formalizes seven
criteria (alt text, reading order, semantic tagging, table structure,
hyperlinks, contrast, font readability) as a contract specification.
Only 3.2% of scholarly PDFs satisfy all criteria — demonstrating why
formal enforcement is needed, not just guidelines.

### Phase 4: Differential Obligation Verification (CB-1350, CB-1351)

Full contract verification is expensive. At commit time, only obligations
whose bound functions were modified need re-checking.

**Mechanism:**
1. `git diff --cached --name-only` → staged files
2. Lookup in `.pmat/binding-index.json` (file→binding reverse index)
3. Check cached verdicts for affected obligations
4. PASS/FAIL from cache — no cold verification

CB-1351 gates on binding index freshness (>7d warn, >30d error).

**Basis:** Mugnier et al. (OOPSLA 2025) proof brittleness from
whole-contract reverification; AWS Cedar (ICSE 2025) targeted verification.

### Phase 5: Assume-Guarantee Chains (CB-1352, CB-1353)

When multiple work items touch overlapping code, one commit can break
another's assumptions. Work contracts declare dependencies:

- **`assumes`**: references another contract's obligation (dependency)
- **`guarantees`**: obligations this item ensures (promise)

**Pre-commit validation:**
1. Load active `.pmat-work/*/contract.json`
2. Build dependency DAG from assumes→guarantees edges
3. For each modified file, find affected guarantees
4. Block if another work item assumes a broken guarantee (CB-1352)
5. DFS cycle detection on the DAG — reject cycles (CB-1353)

**Example:** PMAT-500 guarantees `safe_alloc()` contracts. PMAT-501
assumes `safe_alloc()`. Modifying `safe_alloc()` from PMAT-500 triggers
a warning that PMAT-501's assumptions may be invalidated.

**Basis:** Pacti (ACM TCPS 2025) algebraic A/G operations; Dewes &
Dimitrova (AAAI 2025) quantitative A/G for multi-agent coordination;
Dardik & Kang (2025) compositional inductive invariant inference.

### Phase 6: Contract Query Readiness (CB-1354)

Validates infrastructure for `pmat query --contracts` enrichment. Scores
readiness 0–4 based on: binding-index.json exists, contracts/YAML
directory exists, binding.yaml exists, pv CLI available.

### Phase 7: Hook Subsystem Consolidation (CB-1333..1337)

Root cause: 170 hook-related commits in pmat with ~38 bug-fixes. Six
independent codepaths write `.git/hooks/` with no coordination. Five
design rules:

| Rule | CB | Requirement |
|------|-----|-------------|
| Single Writer | CB-1333 | All writes through `HookRegistry` |
| Atomic Writes | CB-1334 | Write-then-rename, never direct `fs::write()` |
| Deterministic | CB-1335 | No timestamps, no HashMap iteration |
| No Injection | CB-1336 | Shell-escape all template substitution |
| Performance | CB-1337 | Pre-commit p95 < 45ms |

### Phase 8: Falsify Leak Remediation (CB-1338..1343)

Contracts that don't catch bugs because YAML→codegen→binding→call-site
is a 4-step pipeline where each stage leaks. Seven leak classes:

| Leak | CB | Rule |
|------|-----|------|
| Ghost bindings (97% ghosts in PMAT-106) | CB-1338 | `pv infer` verified against AST |
| Placeholder preconditions (`!is_empty()`) | CB-1339 | Zero placeholder ratio |
| Zero enforcement (<1% penetration) | CB-1340 | ≥10% call-site penetration |
| Spec number inflation | CB-1341 | Numbers from `pv status --json` |
| Codegen doesn't compile | CB-1342 | `pv codegen --check` dry-run |
| Assertion placement before guards | CB-1343 | Preconditions AFTER validation |

## Pre-Commit Latency Budget

| Phase | Max Latency | Data Source |
|-------|------------|------------|
| 0 Cache staleness | < 1ms | File mtime |
| 1 Work contracts | < 5ms | `.pmat/contract-cache.json` |
| 2 L-level ratchet | < 3ms | `.pmat/verification-levels.json` |
| 3 Asset layouts | < 10ms | `.pmat/asset-layout-cache.json` |
| 4 Differential obligations | < 5ms | `.pmat/binding-index.json` |
| 5 A/G chains | < 7ms | `.pmat-work/` |
| 6 Query readiness | < 50ms | Lazy-load binding-index |
| **Total** | **< 45ms** | **All from cache** |

## CB Check Summary

| CB Check | Phase | Severity | Description |
|----------|-------|----------|-------------|
| CB-1320 | 3 | Error | README layout slots, ordering, accuracy |
| CB-1321 | 3 | Error | Dockerfile security, layers, pinning |
| CB-1322 | 3 | Error | SVG viewBox, palette, accessibility |
| CB-1323 | 3 | Error | forjar DAG, templates, secrets |
| CB-1324 | 3 | Error | mdBook SUMMARY, code blocks, cross-refs |
| CB-1325 | 3 | Warning | CHANGELOG format, version ordering |
| CB-1326 | 3 | Warning | Badge URLs, required set, placement |
| CB-1330 | 2 | Error | L-level regression (ratchet) |
| CB-1331 | 1 | Error | Work contract validity |
| CB-1332 | 0 | Warning | Cache staleness (7d warn, 30d error) |
| CB-1333 | 7 | Error | Hook single writer |
| CB-1334 | 7 | Error | Hook atomic writes |
| CB-1335 | 7 | Error | Hook deterministic content |
| CB-1336 | 7 | Error | Hook no shell injection |
| CB-1337 | 7 | Error | Hook performance (p95 < 45ms) |
| CB-1338 | 8 | Error | No ghost bindings |
| CB-1339 | 8 | Error | No placeholder preconditions |
| CB-1340 | 8 | Error | Enforcement penetration ≥10% |
| CB-1341 | 8 | Error | Spec numbers from tooling |
| CB-1342 | 8 | Error | Codegen compiles |
| CB-1343 | 8 | Warning | Assertion placement after guards |
| CB-1350 | 4 | Warning | Differential obligation verification |
| CB-1351 | 4 | Error | Binding index freshness |
| CB-1352 | 5 | Warning | A/G chain validation |
| CB-1353 | 5 | Error | A/G cycle detection |
| CB-1354 | 6 | Warning | Contract query readiness |

## Relationship to provable-contracts

PMAT is the **enforcement engine**; provable-contracts is the **contract
definition language**. The relationship:

```
provable-contracts (YAML)     PMAT (enforcement)
─────────────────────────     ───────────────────
contracts/*.yaml              pmat comply check (CB-1320..1354)
binding.yaml                  .pmat/binding-index.json (reverse index)
pv lint / pv score            CB-1342 (pv codegen --check dry-run)
L0–L5 verification ladder    CB-1330 (monotonicity ratchet)
pv codegen macros             CB-1338..1343 (leak remediation)
```

`pv` defines what contracts exist and what they require. `pmat` enforces
that developers don't commit code that violates those contracts, using
pre-computed caches for O(1) commit-time checks.

## Implementation Status (2026-04-05)

**Detection layer (complete):** 29 CB checks, 98 tests, dogfooded on 4
repos (pmat, aprender, trueno, realizar).

**Infrastructure layer (in progress):** 4/14 artifacts remain missing.

| Phase | Status | Notes |
|-------|--------|-------|
| 0 Cache | Complete | 3 caches via `refresh-bindings` |
| 1 Work→YAML | Check done | YAML generation from work items missing |
| 2 Ratchet | Complete | `verification-levels.json` generated |
| 3 Assets | Caches done | `asset_validator/` service missing |
| 4 Diff Obligations | Complete | `binding-index.json` via `refresh-bindings` |
| 5 A/G Chains | Complete | Reads `.pmat-work/` directly |
| 6 Query Enrich | Check done | 5 of 6 query flags missing |
| 7 Hooks | Checks done | `HookRegistry` singleton missing |
| 8 Falsify Leaks | Complete | CB-1342 wired and passing |

## Five Improvements to Asset Layout Enforcement

### I-1: AST-Based Markdown Parsing (Replace Substring Matching)

**Problem:** CB-1320 uses `content.to_lowercase().contains("## install")`
which matches inside code fences, HTML comments, and block quotes. A
README with `## Installation` inside a ``````` block would pass even
though there's no real Installation section. Slot ordering is not checked.

**Fix:** Parse markdown into an AST via `pulldown-cmark` (already a Rust
crate, zero C deps). Walk only top-level `Heading` events at the expected
nesting depth. This enables:
- **Structural slot matching** — only real headings, not fenced content
- **Ordering enforcement** — verify slot N appears before slot N+1 in
  the AST node sequence, matching the rmedia grid protocol's row ordering
- **Heading hierarchy validation** — detect `## ` under `#### ` (inverted
  nesting), which Grammar-Aligned Decoding (arXiv 2405.21047) identifies
  as a grammar violation

**rmedia analogy:** rmedia's `GridValidator` check #1 (Cell Uniqueness)
rejects overlapping allocations by walking the allocation list, not by
substring-searching the SVG output. The same principle: validate the
structure, not the serialized text.

### I-2: Accuracy Cross-Referencing (Spec Claims vs Reality)

**Problem:** The spec claims CB-1320 should verify "accuracy via regex
matching against `pmat --version`, `cargo test` output, and coverage
data." The implementation does **zero accuracy checks**. A README
claiming "99.66% coverage" or "21,200+ tests" could be 6 months stale
and CB-1320 would pass. This is spec number inflation (L-5 leak class).

**Fix:** Extract claimed metrics from README via regex:
- Test count: `(\d[\d,]*)\+?\s*(?:passing|tests)`
- Coverage: `(\d+\.?\d*)%`  (near `coverage` keyword)
- Version: `v?(\d+\.\d+\.\d+)` (near crate name)

Compare against `.pmat/baseline.json` (already generated by
`pmat tdg baseline create`). Flag drift > 5% as Warning, > 20% as Error.
CB-1341 (Spec Number Accuracy) already exists for code specs — extend
the same principle to README claims.

**Basis:** AI Transparency Atlas (arXiv 2512.12443) defines an 8-section
documentation scoring framework where accuracy is a first-class
dimension. PDF accessibility benchmarking (arXiv 2509.18965) shows that
96.8% of documents fail structural criteria — accuracy enforcement
prevents the same fate for README metrics.

### I-3: Verdict Caching with Content Hashing

**Problem:** `asset-layout-cache.json` currently stores only boolean
existence flags (`{"readme": true, "changelog": true}`). Every
`pmat comply check` re-reads and re-parses every markdown file. Phase 3
does not actually achieve O(1) from cache — it is O(n) in file size on
every invocation, violating the < 10ms latency budget.

**Fix:** Cache the full verdict per asset:

```json
{
  "readme": {
    "content_sha256": "a1b2c3...",
    "verdict": "pass",
    "issues": [],
    "checked_at": "2026-04-05T14:30:00Z"
  }
}
```

On pre-commit: compute SHA-256 of `README.md` (< 1ms for typical files),
compare to cached hash. If unchanged → return cached verdict in < 1ms.
If changed → re-parse, update cache. This makes the O(1) claim honest.

**rmedia analogy:** rmedia's `Grid` tracks occupancy via `HashSet<(u32, u32)>`
for O(1) cell lookup. The same principle: pre-compute the occupancy
map, don't re-walk the allocation list on every query.

### I-4: Recursive Cross-Reference Link Validation

**Problem:** CB-1324 checks only `book/src/SUMMARY.md` links one level
deep. CB-1320 checks no cross-references. Neither validates links in
arbitrary `.md` files. A README containing `[CONTRIBUTING.md](CONTRIBUTING.md)`
pointing to a deleted file passes silently.

**Fix:** Generic markdown link validator across all `.md` files:
1. Extract all `[text](path)` where path is local (not `http`, not `#`)
2. Verify target file exists on disk
3. For anchor links (`#section-name`), verify the heading exists in the
   target file's AST (requires I-1 AST parsing)
4. Wire into CB-1320 (README), CB-1324 (mdBook), and a new **CB-1327**
   for general `.md` cross-reference integrity

This is what `rumdl` (already listed in the PMAT spec's tools) does.
The WCAG-EM methodology (arXiv 2511.03471) uses graph-based page
sampling to ensure verification covers the linkage structure —
the same principle applied to markdown cross-references.

**rmedia analogy:** rmedia's validation check #4 (Edge References Valid)
verifies that every `edge.from` and `edge.to` references an existing
node ID. Broken links in markdown are the document equivalent of
dangling edge references in a concept graph.

### I-5: YAML-Driven Asset Contracts (Declarative, Not Hardcoded)

**Problem:** Every asset check is a handwritten Rust function with
hardcoded rules. Adding a new asset type (e.g., `SECURITY.md`,
`CODE_OF_CONDUCT.md`, `.github/PULL_REQUEST_TEMPLATE.md`) requires a
new Rust function, recompilation, and a new pmat release.

**Fix:** Define asset layout contracts in YAML using the provable-contracts
schema with `surface: asset-layout`, then interpret via a generic
`validate_asset_contract(path, contract_yaml)` function:

```yaml
# contracts/assets/readme-layout-v1.yaml
metadata:
  id: asset-readme-layout-v1
  surface: asset-layout
  asset_type: markdown
  grid: { columns: 1, rows: 11 }

slots:
  - id: title
    row: 0
    pattern: "^# .+"
    required: true
  - id: badges
    row: 1
    pattern: "\\[!\\[|shields\\.io|badge"
    required: true
  - id: installation
    row: 3
    heading_match: [installation, install, getting started, setup]
    required: true
  - id: usage
    row: 4
    heading_match: [usage, examples, quick start]
    required: true
  - id: license
    row: 9
    heading_match: [license]
    required: true

constraints:
  - type: ordering
    rule: "slot[i].row < slot[i+1].row for all adjacent required slots"
  - type: accuracy
    fields: [test_count, coverage_pct, version]
    source: ".pmat/baseline.json"
    max_drift_pct: 5
  - type: cross_references
    validate_local_links: true
```

This makes asset enforcement declarative and extensible without code
changes — the same way kernel contracts work for code entities.
NL2Contract (arXiv 2510.12702) demonstrates that natural language
specifications ("headings must be hierarchical," "images must have
alt text") can be formalized into precondition/postcondition contracts.
The YAML schema is the intermediate representation between the natural
language intent and the enforcement engine.

**rmedia analogy:** rmedia's four layout engines (Pipeline, Radial,
Comparison, Tree) are pluggable implementations of the `LayoutEngine`
trait. The YAML-driven approach is the equivalent: a single
`AssetValidator` trait with YAML-driven implementations, not one
hardcoded function per asset type.

### Improvement Summary

| # | Improvement | Fixes | Basis |
|---|-------------|-------|-------|
| I-1 | AST markdown parsing | False positives, no ordering | Grammar-Aligned Decoding |
| I-2 | Accuracy cross-referencing | Stale README metrics | AI Transparency Atlas |
| I-3 | Verdict caching + SHA-256 | O(n) → O(1) for real | rmedia HashSet occupancy |
| I-4 | Recursive link validation | Broken cross-refs | WCAG-EM graph sampling |
| I-5 | YAML-driven asset contracts | Hardcoded, non-extensible | NL2Contract, rmedia LayoutEngine |

## Key PMAT Source Files

| File | Role |
|------|------|
| `src/cli/handlers/comply_handlers/check_handlers/check_commit_enforcement.rs` | CB-1320..1354 implementation |
| `src/cli/handlers/comply_handlers/check_handlers/check.rs` | Check dispatcher |
| `src/cli/commands/misc_commands_comply.rs` | CLI: `refresh-bindings` subcommand |
| `src/cli/handlers/hooks_command_handlers/tdg_hooks.rs` | TDG hook install (atomic) |

## References

### Commit-Level Enforcement

- **Mugnier et al. (OOPSLA 2025).** Proof brittleness in Dafny-verified codebases. [ACM DL](https://dl.acm.org/doi/10.1145/3763181)
- **Chakarov et al. (ICSE 2025).** Cedar: formally verified authorization at 1B req/sec. [ACM DL](https://dl.acm.org/doi/10.1109/ICSE55347.2025.00166)
- **Incer et al. (ACM TCPS 2025).** Pacti: assume-guarantee contract algebra. [ACM DL](https://dl.acm.org/doi/10.1145/3704736)
- **Dewes & Dimitrova (AAAI 2025).** Quantitative A/G for multi-agent coordination. [arXiv:2412.13114](https://arxiv.org/abs/2412.13114)
- **Bhardwaj (arXiv 2026).** Agent Behavioral Contracts. [arXiv:2602.22302](https://arxiv.org/abs/2602.22302)
- **Bader et al. (TOPLAS).** Gradual Verification.
- **Nagappan et al. (IEEE TSE 2006).** File size and churn correlate with defects.

### Layout as Provable Contract

- **SMT-Layout (2024).** MaxSMT-based GUI layout: encodes layout as hard constraints (invariants) + soft constraints (preferences), solved in milliseconds. Proves layout is a CSP with provable placement or proof of unsatisfiability. [arXiv:2411.12271](https://arxiv.org/abs/2411.12271)
- **ORCSolver (Zeidler et al., 2020).** OR-constraints unify grid and flow layout under single CSP specification. A single spec verified to produce valid layouts across all device configurations — a contract with universal quantification over dimensions. [arXiv:2002.09925](https://arxiv.org/abs/2002.09925)
- **LACE (Chen et al., 2024).** Differentiable aesthetic constraint functions (alignment, non-overlap, spacing) enforced during layout generation. Layout constraints are differentiable mathematical functions. [arXiv:2402.04754](https://arxiv.org/abs/2402.04754)
- **Grammar-Aligned Decoding (Park et al., NeurIPS 2024).** Proves structured document output can be generated with mathematical guarantees of grammar conformance. The formal grammar acts as a contract; the algorithm proves all outputs satisfy it. [arXiv:2405.21047](https://arxiv.org/abs/2405.21047)

### Accessibility as Formal Verification

- **AccessGuru (2025).** Three-category taxonomy of web accessibility violations: Syntactic (precondition), Semantic (postcondition), Layout (invariant). WCAG guidelines formalized as checkable properties over HTML structure. [arXiv:2507.19549](https://arxiv.org/abs/2507.19549)
- **WCAG-EM Scalable Audit (2025).** GRASP graph-based page sampling ensures verification covers textual, visual layout, and linkage relationships. [arXiv:2511.03471](https://arxiv.org/abs/2511.03471)
- **PDF Accessibility Benchmark (2025).** Seven criteria (alt text, reading order, semantic tagging, table structure, hyperlinks, contrast, font readability) formalized as contract specification. 96.8% of scholarly PDFs fail — demonstrates need for formal enforcement. [arXiv:2509.18965](https://arxiv.org/abs/2509.18965)

### SVG and Document Structure

- **SVGenius (2025).** Comprehensive SVG benchmark: 2,377 queries across understanding, editing, generation. SVG structure has verifiable properties across 24 domains. Bug-fixing tasks implicitly define correctness contracts. [arXiv:2506.03139](https://arxiv.org/abs/2506.03139)
- **NL2Contract (Richter & Wehrheim, 2025).** LLMs translate natural language specifications into formal pre/postcondition pairs. Directly applicable to formalizing layout rules ("headings must be hierarchical") into enforceable contracts. [arXiv:2510.12702](https://arxiv.org/abs/2510.12702)

### Implementation Reference

- **rmedia** — Grid Protocol implementation: 16×9 cell grid (1920×1080, 120px cells), 4 layout engines, 20-point validation checklist, full coverage guarantee. [github.com/paiml/rmedia](https://github.com/paiml/rmedia)
