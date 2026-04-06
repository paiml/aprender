# `batuta stack`

PAIML Stack dependency orchestration commands.

## Synopsis

```bash
batuta stack <COMMAND>
```

## Commands

| Command | Description |
|---------|-------------|
| `check` | Check dependency health across the PAIML stack |
| `comply` | Enforce cross-project consistency using MinHash+LSH |
| `drift` | Detect version drift across published stack crates |
| `gate` | Enforce A- quality threshold for all components |
| `publish-status` | Check which crates need publishing (O(1) cached) |
| `quality` | Analyze quality metrics across the PAIML stack |
| `release` | Coordinate releases across the PAIML stack |
| `status` | Show stack health status dashboard |
| `sync` | Synchronize dependencies across the stack |
| `tree` | Display hierarchical tree of PAIML stack components |
| `versions` | Check latest versions from crates.io |

---

## `batuta stack tree`

Display a visual hierarchical tree of all 21 PAIML stack components.

### Usage

```bash
batuta stack tree [OPTIONS]
```

### Options

| Option | Description |
|--------|-------------|
| `--format <FORMAT>` | Output format: `ascii` (default), `json`, `dot` |
| `--health` | Show health status and version information |
| `--filter <LAYER>` | Filter by layer name |

### Layers

| Layer | Components |
|-------|------------|
| `core` | trueno, trueno-viz, trueno-db, trueno-graph, trueno-rag |
| `ml` | aprender, aprender-shell, aprender-tsp |
| `inference` | realizar, renacer, alimentar, entrenar |
| `orchestration` | batuta, certeza, presentar, pacha |
| `distributed` | repartir |
| `transpilation` | ruchy, decy, depyler |
| `docs` | sovereign-ai-stack-book |

### Examples

```bash
# ASCII tree (default)
batuta stack tree

# Output:
# PAIML Stack (21 crates)
# ├── core
# │   ├── trueno
# │   ├── trueno-viz
# │   └── ...
# ├── ml
# │   └── ...

# JSON output for tooling
batuta stack tree --format json

# Graphviz DOT for visualization
batuta stack tree --format dot | dot -Tpng -o stack.png

# Filter to specific layer
batuta stack tree --filter core

# Show health status
batuta stack tree --health
```

---

## `batuta stack check`

Analyze dependency health across the PAIML ecosystem.

### Usage

```bash
batuta stack check [OPTIONS]
```

### Options

| Option | Description |
|--------|-------------|
| `--project <NAME>` | Specific project to check (default: all) |
| `--format <FORMAT>` | Output format: `text`, `json`, `markdown` |
| `--strict` | Fail on any warnings |
| `--verify-published` | Verify crates.io versions exist |
| `--workspace <PATH>` | Path to workspace root |

### Examples

```bash
# Check all projects
batuta stack check

# Check specific project with strict mode
batuta stack check --project trueno --strict

# JSON output for CI
batuta stack check --format json --verify-published
```

---

## `batuta stack comply`

Enforce cross-project consistency using MinHash+LSH code duplication detection and rule-based compliance checks.

### Usage

```bash
batuta stack comply [OPTIONS]
```

### Options

| Option | Description |
|--------|-------------|
| `--rule <RULE>` | Run specific rule only (e.g., `makefile-targets`) |
| `--fix` | Attempt to auto-fix violations |
| `--format <FORMAT>` | Output format: `text` (default), `json`, `html` |
| `--workspace <PATH>` | Path to workspace root |

### Available Rules

| Rule ID | Description | Points |
|---------|-------------|--------|
| `makefile-targets` | Ensures Makefile target consistency across projects | 25 |
| `cargo-toml-consistency` | Validates Cargo.toml parity (metadata, editions) | 25 |
| `ci-workflow-parity` | Checks GitHub Actions workflow alignment | 25 |
| `code-duplication` | Detects duplicates via MinHash+LSH (85% threshold) | 25 |

### MinHash+LSH Code Duplication

The `code-duplication` rule uses locality-sensitive hashing to detect near-duplicate code across projects:

- **MinHash**: Generates compact signatures from code shingles
- **LSH**: Efficiently finds candidates above 85% similarity threshold
- **Band optimization**: 20 bands × 5 rows for optimal precision/recall

### Examples

```bash
# Run all compliance checks
batuta stack comply

# Output:
# ═══════════════════════════════════════════════════════════
#  Stack Compliance Report
# ═══════════════════════════════════════════════════════════
#
# ✓ makefile-targets          PASS  (25/25)
# ✗ cargo-toml-consistency    FAIL  (20/25)
#     - trueno: missing homepage field
#     - aprender: edition mismatch (2021 vs 2024)
# ✓ ci-workflow-parity        PASS  (25/25)
# ✓ code-duplication          PASS  (23/25)
#     - Warning: 87% similarity detected between:
#       batuta/src/backend.rs:42-68
#       realizar/src/dispatch.rs:15-41
#
# ═══════════════════════════════════════════════════════════
# Pass Rate: 93.0%  (93/100 points)
# ═══════════════════════════════════════════════════════════

# Run specific rule
batuta stack comply --rule code-duplication

# Attempt auto-fix for violations
batuta stack comply --fix

# JSON output for CI
batuta stack comply --format json
```

### Run the Demo

```bash
# Run the Stack Comply demo
cargo run --example stack_comply_demo

# Output demonstrates:
# - Creating compliance engine
# - Listing available rules
# - Discovering projects in workspace
# - Running compliance checks
# - Displaying formatted report
```

### Programmatic API

```rust
use batuta::comply::{ComplyConfig, ComplyReportFormat, StackComplyEngine};
use std::path::Path;

// Create engine with default config
let config = ComplyConfig::default();
let mut engine = StackComplyEngine::new(config);

// Discover projects
let projects = engine.discover_projects(Path::new("."))?;

// Run compliance checks
let report = engine.check_all();

// Display results
println!("Pass rate: {:.1}%", report.summary.pass_rate);
println!("{}", report.format(ComplyReportFormat::Text));
```

---

## `batuta stack drift`

Ecosystem-wide drift detection for stack maintainers. Checks ALL published PAIML crates for stale inter-dependencies.

> **Note:** The startup drift warning only checks **batuta's own** dependencies. Use this command to audit the full ecosystem.

### Usage

```bash
batuta stack drift [OPTIONS]
```

### Options

| Option | Description |
|--------|-------------|
| `--fix` | Generate fix commands for drift issues |
| `--workspace <PATH>` | Workspace root containing stack crates |
| `--format <FORMAT>` | Output format: `text` (default), `json` |
| `--quiet, -q` | Only output if drift detected |

### Startup Self-Drift Check

Batuta checks **its own** published dependencies at startup. If batuta itself depends on stale PAIML crates, it shows a concise warning:

```bash
# Running any command when batuta has outdated deps:
batuta analyze .

# Output:
# ⚠️  batuta 0.7.2 has outdated dependencies
#
#    trueno ^0.15 → 0.16.0
#    aprender ^0.26 → 0.27.0
#
# Update: cargo install batuta
```

This warning appears once per hour and never blocks. It only reports on **batuta itself** — not on other ecosystem crates.

To enforce blocking (recommended for CI):
```bash
batuta --strict analyze .
# or: BATUTA_STRICT=1 batuta analyze .
```

To suppress warnings entirely:
```bash
batuta --allow-drift analyze .
```

### Drift Severity

| Severity | Example | Impact |
|----------|---------|--------|
| `MAJOR` | 0.6 → 0.11 | Likely breaking changes |
| `MINOR` | 0.10.1 → 0.11.0 | New features, possible deprecations |
| `PATCH` | 0.11.0 → 0.11.1 | Bug fixes only |

### Examples

```bash
# Check for drift across published crates
batuta stack drift

# Output:
# 📦 Stack Drift Analysis
# ════════════════════════════════════════════════════════════
#
# trueno-rag 0.1.5:
#   └─ trueno: 0.10.1 → 0.11.0 (MINOR)
#
# entrenar 0.5.0:
#   └─ aprender: 0.21 → 0.23 (MINOR)
#
# repartir 2.0.0:
#   └─ trueno: 0.6 → 0.11.0 (MAJOR)
#
# ⚠️ 3 crates with drift detected

# Generate fix commands
batuta stack drift --fix --workspace ~/src

# Output:
# cd ~/src/trueno-rag && sed -i 's/trueno = "0.10"/trueno = "0.11"/' Cargo.toml
# cd ~/src/entrenar && sed -i 's/aprender = "0.21"/aprender = "0.23"/' Cargo.toml
# cd ~/src/repartir && sed -i 's/trueno = "0.6"/trueno = "0.11"/' Cargo.toml

# JSON output for CI/tooling
batuta stack drift --format json
```

### CI Integration

Add to your CI pipeline to catch drift early:

```yaml
- name: Check Stack Drift
  run: cargo run --quiet -- stack drift --quiet
  # Exits 0 if no drift, 1 if drift detected
```

---

## `batuta stack gate`

Enforce A- quality threshold across all PAIML stack components. This command is designed for CI/CD pipelines and pre-commit hooks to block releases or commits when any component falls below the quality threshold.

### Usage

```bash
batuta stack gate [OPTIONS]
```

### Options

| Option | Description |
|--------|-------------|
| `--workspace <PATH>` | Path to workspace root (default: parent of current directory) |
| `--quiet, -q` | Quiet mode - only output on failure |

### Quality Threshold

The quality gate enforces an **A- minimum** (SQI ≥ 85) for all stack components. Components below this threshold are **blocked** and will cause the gate to fail.

| Grade | SQI Range | Gate Status |
|-------|-----------|-------------|
| A+ | 95-100% | PASS |
| A | 90-94% | PASS |
| A- | 85-89% | PASS |
| B+ | 80-84% | BLOCKED |
| B | 70-79% | BLOCKED |
| C | 60-69% | BLOCKED |
| D | 50-59% | BLOCKED |
| F | 0-49% | BLOCKED |

### Enforcement Points

The quality gate is enforced at multiple points in the development workflow:

| Point | Trigger | Action |
|-------|---------|--------|
| Pre-commit | `git push` | Blocks push if any component < A- |
| Release | `batuta stack release` | Blocks release by default (use `--no-verify` to skip) |
| CI Pipeline | Pull request | Blocks PR merge if quality gate fails |
| Manual | `make stack-gate` | Returns exit code 1 if failed |

### Examples

```bash
# Run quality gate check
batuta stack gate

# Output:
# ╔════════════════════════════════════════════════════╗
# ║  Stack Quality Gate - A- Enforcement               ║
# ╚════════════════════════════════════════════════════╝
#
# trueno           SQI: 95.9  Grade: A+  ✅ PASS
# aprender         SQI: 96.2  Grade: A+  ✅ PASS
# batuta           SQI: 94.1  Grade: A   ✅ PASS
# ...
#
# ✅ All 21 components meet A- quality threshold

# Quiet mode for CI (only outputs on failure)
batuta stack gate --quiet

# Check specific workspace
batuta stack gate --workspace /path/to/paiml
```

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | All components pass the quality gate |
| 1 | One or more components are below A- threshold |

### Pre-commit Hook Configuration

Add to `.pre-commit-config.yaml`:

```yaml
- repo: local
  hooks:
    - id: stack-quality-gate
      name: Stack Quality Gate (A- enforcement)
      entry: cargo run --quiet -- stack gate
      language: system
      pass_filenames: false
      stages: [push]
```

### Makefile Targets

```makefile
stack-gate:  ## Quality gate enforcement
	@cargo run --quiet -- stack gate

stack-quality:  ## Show detailed quality matrix
	@cargo run --quiet -- stack quality
```

---

## `batuta stack quality`

Analyze quality metrics across the PAIML stack using PMAT integration.

This command evaluates each stack component against the Stack Quality Matrix, which includes:
- **Rust Project Score** (0-114): Code quality, testing, documentation
- **Repository Score** (0-110): CI/CD, security, community health
- **README Score** (0-20): Documentation completeness
- **Hero Image**: Visual branding presence

### Usage

```bash
batuta stack quality [OPTIONS] [COMPONENT]
```

### Options

| Option | Description |
|--------|-------------|
| `--strict` | Require A+ grade for all components |
| `--format <FORMAT>` | Output format: `text` (default), `json` |
| `--verify-hero` | Verify hero image exists and meets requirements |
| `--verbose` | Show detailed scoring breakdown |
| `--workspace <PATH>` | Path to workspace root |

### Quality Grades

| Grade | SQI Range | Description |
|-------|-----------|-------------|
| A+ | 95-100% | Exceptional quality |
| A | 90-94% | Excellent quality |
| A- | 85-89% | Very good quality |
| B+ | 80-84% | Good quality |
| B | 70-79% | Acceptable quality |
| C | 60-69% | Needs improvement |
| D | 50-59% | Poor quality |
| F | 0-49% | Failing quality |

### Stack Quality Index (SQI)

The SQI is calculated as a weighted composite:

```
SQI = 0.40 × Rust Score + 0.30 × Repo Score + 0.20 × README Score + 0.10 × Hero Score
```

### Examples

```bash
# Check quality of all stack components
batuta stack quality

# Output:
# Stack Quality Report
# ====================
#
# trueno          A+  (SQI: 97.2%)
# aprender        A   (SQI: 92.1%)
# batuta          A+  (SQI: 96.8%)
# ...
#
# Summary: 18/25 components at A+ grade
# Overall Stack Grade: A

# Check specific component with verbose output
batuta stack quality trueno --verbose

# Strict mode for CI (fails if any component below A+)
batuta stack quality --strict

# JSON output for tooling
batuta stack quality --format json

# Verify hero images exist
batuta stack quality --verify-hero
```

### Hero Image Requirements

A hero image is required for A+ grade and must be:
- Located at `docs/hero.svg` (preferred) or `docs/hero.png`
- Can also be referenced as first image in README.md
- SVG format preferred for scalability and crisp rendering
- If using PNG: minimum dimensions 1280x640 pixels

---

## `batuta stack release`

Coordinate releases with automatic dependency ordering.

### Usage

```bash
batuta stack release [OPTIONS] [CRATE_NAME]
```

### Options

| Option | Description |
|--------|-------------|
| `--all` | Release all crates with changes |
| `--dry-run` | Show what would be released |
| `--bump <TYPE>` | Version bump: `patch`, `minor`, `major` |
| `--no-verify` | Skip quality gate verification |
| `--yes` | Skip interactive confirmation |
| `--publish` | Publish to crates.io |

### Examples

```bash
# Dry run to see release plan
batuta stack release --all --dry-run

# Release specific crate (and its dependencies)
batuta stack release trueno --bump patch

# Full release with publish
batuta stack release --all --bump minor --publish --yes
```

---

## `batuta stack status`

Show health dashboard for the entire stack.

### Usage

```bash
batuta stack status [OPTIONS]
```

### Options

| Option | Description |
|--------|-------------|
| `--simple` | Simple text output (no TUI) |
| `--format <FORMAT>` | Output format: `text`, `json`, `markdown` |
| `--tree` | Show dependency tree |

---

## `batuta stack sync`

Synchronize dependency versions across the stack.

### Usage

```bash
batuta stack sync [OPTIONS] [CRATE_NAME]
```

### Options

| Option | Description |
|--------|-------------|
| `--all` | Sync all crates |
| `--dry-run` | Show what would change |
| `--align <DEP=VER>` | Align specific dependency version |

### Examples

```bash
# Sync all crates
batuta stack sync --all --dry-run

# Align arrow version across stack
batuta stack sync --all --align "arrow=54.0"
```

---

## `batuta stack versions`

Check latest versions of PAIML stack crates from crates.io.

### Usage

```bash
batuta stack versions [OPTIONS]
```

### Options

| Option | Description |
|--------|-------------|
| `--outdated` | Only show crates with newer versions available |
| `--format <FORMAT>` | Output format: `text` (default), `json` |
| `--offline` | Skip network requests (use cached data only) |
| `--include-prerelease` | Include pre-release versions |

### Examples

```bash
# Check all stack versions
batuta stack versions

# Output:
# 📦 PAIML Stack Versions
# ════════════════════════════════════════════════════════════
# Crate                      Latest    Downloads Description
# ────────────────────────────────────────────────────────────
# trueno                      0.8.8         6.3K High-performance SIMD...
# aprender                   0.19.0         5.5K Next-generation ML...
# ...

# JSON output for scripting
batuta stack versions --format json

# Only outdated
batuta stack versions --outdated
```

---

## `batuta stack publish-status`

Check publish status of all PAIML stack repos with O(1) caching.

This command scans the local workspace for PAIML crates and shows which need publishing. It uses content-addressable caching for O(1) lookups on unchanged repos.

### Usage

```bash
batuta stack publish-status [OPTIONS]
```

### Options

| Option | Description |
|--------|-------------|
| `--format <FORMAT>` | Output format: `text` (default), `json` |
| `--workspace <PATH>` | Workspace root (parent directory containing stack crates) |
| `--clear-cache` | Clear cache and force refresh |

### Performance

The publish-status command uses intelligent caching for fast repeated queries:

| Scenario | Time | Description |
|----------|------|-------------|
| Cold cache | ~7s | First run, fetches all data from crates.io |
| Warm cache | <100ms | Subsequent runs, O(1) hash-based lookups |

### Cache Invalidation

The cache is automatically invalidated when:
- `Cargo.toml` content changes
- Git HEAD moves (new commit)
- crates.io TTL expires (15 minutes)

Cache is stored at `~/.cache/batuta/publish-status.json`.

### Actions

| Symbol | Action | Description |
|--------|--------|-------------|
| ✓ | up to date | Local matches crates.io, repo is clean |
| 📝 | commit | Has uncommitted changes |
| 📦 | PUBLISH | Local version higher than crates.io |
| 🆕 | new | Not yet published to crates.io |
| ⚠️ | behind | Local version behind crates.io (unusual) |
| ❌ | error | Error checking status |

### Examples

```bash
# Check publish status (fast with warm cache)
batuta stack publish-status

# Output:
# 📦 PAIML Stack Publish Status
# ═════════════════════════════════════════════════════════════════
# Crate                     Local  crates.io        Git       Action
# ─────────────────────────────────────────────────────────────────
# trueno                    0.8.8      0.8.8      clean ✓ up to date
# pacha                     0.2.0      0.2.0     clean ✓ up to date
# depyler                  3.21.0     3.20.0     33M 8? 📝 commit
# certeza                   0.1.0          -      clean 🆕 new
# ─────────────────────────────────────────────────────────────────
# 📊 20 crates: 1 publish, 12 commit, 6 up-to-date
# ⚡ 78ms (cache: 20 hits, 0 misses)

# Force cache refresh
batuta stack publish-status --clear-cache

# JSON output for CI/tooling
batuta stack publish-status --format json
```

### Makefile Targets

```makefile
stack-publish-status:  ## Check which crates need publishing (O(1) cached)
	@cargo run --quiet -- stack publish-status

stack-publish-status-refresh:  ## Force refresh publish status cache
	@cargo run --quiet -- stack publish-status --clear-cache
```

---

## Toyota Way Principles

The stack commands embody Toyota Way principles:

| Principle | Implementation |
|-----------|----------------|
| **Jidoka** | Pre-flight checks stop broken releases |
| **Just-in-Time** | Pull-based release ordering |
| **Heijunka** | Version alignment across stack |
| **Genchi Genbutsu** | Real-time crates.io verification |
| **Visual Management** | Tree view with health indicators |
