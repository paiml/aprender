# Sub-spec: CLI Reference

**Parent:** [pv-spec.md](../pv-spec.md) Section 5

---

## 1. Installation

```bash
cargo install --path crates/provable-contracts-cli
# or from workspace:
cargo build --release -p provable-contracts-cli
```

The binary is named `pv`.

---

## 2. Existing Commands (16)

### pv validate

Parse and validate a YAML contract against the schema.

```bash
pv validate contracts/softmax-kernel-v1.yaml
```

Reports missing fields, invalid cross-references, unreachable tests,
duplicate IDs. Exit code 0 = valid, 1 = errors.

### pv scaffold

Generate Rust trait definition + failing test stubs to stdout.

```bash
pv scaffold contracts/softmax-kernel-v1.yaml > src/softmax/trait.rs
```

### pv kani

Generate `#[kani::proof]` harnesses to stdout.

```bash
pv kani contracts/softmax-kernel-v1.yaml > src/softmax/kani_proofs.rs
```

Strategies: exhaustive, stub_float, compositional, bounded_int.

### pv probar

Generate probar property tests to stdout.

```bash
pv probar contracts/softmax-kernel-v1.yaml
pv probar contracts/softmax-kernel-v1.yaml --binding contracts/aprender/binding.yaml
```

With `--binding`, generates wired tests calling real functions.

### pv status

Display contract summary: equations, obligations, tests, harnesses.

```bash
pv status contracts/softmax-kernel-v1.yaml
```

### pv audit

Run traceability audit: paper -> equation -> obligation -> test -> proof.

```bash
pv audit contracts/softmax-kernel-v1.yaml
pv audit contracts/softmax-kernel-v1.yaml --binding contracts/aprender/binding.yaml
```

Reports gaps in the derivation chain.

### pv diff

Compare two contract versions. Suggest semver bump.

```bash
pv diff contracts/softmax-kernel-v1.yaml contracts/softmax-kernel-v2.yaml
```

Output: added/removed equations, obligations, tests. Bump suggestion:
major, minor, or patch.

### pv coverage

Cross-contract obligation coverage report.

```bash
pv coverage contracts/
pv coverage contracts/ --binding contracts/aprender/binding.yaml
```

Shows per-contract and aggregate coverage percentages.

### pv generate

Write all codegen artifacts to disk.

```bash
pv generate contracts/softmax-kernel-v1.yaml -o generated/
pv generate contracts/softmax-kernel-v1.yaml -o generated/ --binding binding.yaml
```

Produces: `{stem}_scaffold.rs`, `{stem}_kani.rs`, `{stem}_probar.rs`,
`{stem}_book.md`. With `--binding`: also `{stem}_wired_probar.rs`.

### pv graph

Render contract dependency DAG.

```bash
pv graph contracts/                        # text (default)
pv graph contracts/ --format dot           # Graphviz DOT
pv graph contracts/ --format json          # JSON
pv graph contracts/ --format mermaid       # Mermaid diagram
```

Shows which contracts depend on others. Reports cycles.

### pv equations

Render contract equations in multiple formats.

```bash
pv equations contracts/softmax-kernel-v1.yaml                # text
pv equations contracts/softmax-kernel-v1.yaml --format latex # LaTeX
pv equations contracts/softmax-kernel-v1.yaml --format ptx   # PTX stub
pv equations contracts/softmax-kernel-v1.yaml --format asm   # x86-64
```

### pv lean

Generate Lean 4 definition and theorem files.

```bash
pv lean contracts/softmax-kernel-v1.yaml
pv lean contracts/softmax-kernel-v1.yaml --output-dir lean/
```

### pv lean-status

Report Lean proof status across contracts.

```bash
pv lean-status contracts/
pv lean-status contracts/softmax-kernel-v1.yaml
```

### pv proof-status

Hierarchical proof level (L1-L5) report. Walks the contract directory
recursively. Use `--kind` to narrow by contract kind.

```bash
pv proof-status contracts/
pv proof-status contracts/ --binding contracts/aprender/binding.yaml
pv proof-status contracts/ --format json
pv proof-status contracts/ --kind pattern
pv proof-status contracts/ --kind registry --format json
```

Shows kernel equivalence classes (A-E) when multiple contracts present.
With `--kind pattern|registry|model-family|schema`, only the matching
contracts are reported; totals reflect the filtered set.

### pv lint

Run all contract quality gates (validate + audit + score) in one pass.

```bash
pv lint contracts/
pv lint contracts/ --min-score 0.60
pv lint contracts/ --binding contracts/aprender/binding.yaml --min-score 0.75
pv lint contracts/ -f json
pv lint contracts/ -f sarif                          # SARIF v2.1.0 output
pv lint contracts/ --diff main                       # Only lint changed contracts
pv lint contracts/ --baseline .pv/baseline.sarif     # Suppress known findings
pv lint contracts/ --severity error                  # Only errors (no warnings)
pv lint contracts/ --suppress SM-INV-001,KANI-SM-002 # Suppress specific findings
pv lint contracts/ --suggest                         # Auto-fix suggestions
pv lint contracts/ --watch                           # Re-lint on file change
pv lint contracts/ --trend                           # Show quality trend
```

Exit code 0 = all gates pass, 1 = any gate fails. See
[scoring.md](scoring.md) Section 5 for full gate definitions. See
[lint.md](lint.md) for the full quality-gate sub-spec including SARIF
output, diff-aware mode, suppression, and CI integration patterns.

#### Flags

| Flag | Description |
|---|---|
| `--min-score <f64>` | Minimum acceptable composite score |
| `--binding <path>` | Binding registry path |
| `--exit-code` | Exit 1 on any gate failure |
| `-f, --format <fmt>` | Output: text, json, markdown, **sarif**, **github** |
| `--diff <base_ref>` | Only lint contracts changed since base ref |
| `--baseline <sarif>` | Suppress findings present in baseline SARIF |
| `--severity <level>` | Minimum severity: error, warning, info |
| `--strict` | Promote warnings to errors |
| `--suppress <ids>` | Suppress specific finding IDs (comma-separated) |
| `--suppress-rule <rule>` | Suppress all findings for a rule |
| `--suppress-file <path>` | Suppress all findings in a file |
| `--suggest` | Show auto-fix suggestions (dry run) |
| `--fix` | Apply deterministic auto-fixes |
| `--watch` | Re-lint on file change (inotify) |
| `--trend` | Record snapshot to `.pv/trend/` |
| `--trend --show` | Display quality trend history |
| `--no-cache` | Bypass lint cache |
| `--cache-stats` | Show cache hit/miss statistics |
| `--config <path>` | Path to `.pv.toml` config file |
| `--rule <id>=<level>` | Override rule severity for this run |

### pv book

Generate mdBook contract pages.

```bash
pv book contracts/ -o book/src/contracts/
pv book contracts/ -o book/src/contracts/ --update-summary --summary-path book/src/SUMMARY.md
```

---

## 3. New Commands

### pv score

Score a contract or codebase. See [scoring.md](scoring.md) for full
methodology.

```bash
# Score a single contract
pv score contracts/softmax-kernel-v1.yaml

# Score a codebase
pv score ~/src/aprender --binding contracts/aprender/binding.yaml

# Score all contracts in a directory
pv score contracts/ --summary

# JSON output for CI integration
pv score contracts/softmax-kernel-v1.yaml -f json

# Fail CI if score below threshold
pv score contracts/ --min-score 0.75 --exit-code
```

#### Flags

| Flag | Description |
|---|---|
| `--binding <path>` | Binding registry for codebase scoring |
| `--min-score <f64>` | Minimum acceptable composite score |
| `--exit-code` | Exit 1 if any contract below `--min-score` |
| `-f, --format <fmt>` | Output format: text (default), json, markdown |
| `--summary` | Aggregate summary only (no per-contract detail) |
| `--top-gaps <n>` | Show top N gaps by impact (default: 5) |
| `--weights <json>` | Custom dimension weights (JSON object) |

#### CI Integration

```yaml
# .github/workflows/contract-score.yml
- name: Contract quality gate
  run: pv score contracts/ --min-score 0.75 --exit-code
```

### pv query

O(1) semantic search across contracts with automatic cross-project
discovery. See [query.md](query.md) for full architecture.

```bash
# Semantic search (auto-includes cross-project results)
pv query "softmax numerical stability"

# Regex / literal
pv query --regex "SM-INV-\d+"
pv query --literal "kani::proof"

# Filters
pv query --obligation invariant
pv query --min-score 0.8
pv query --min-level L4
pv query --depends-on softmax-kernel-v1
pv query --unproven

# Cross-project (automatic — no flag needed)
pv query "softmax" --call-sites        # Show all #[contract] annotations
pv query --violations                  # Contracts violated in any project
pv query --coverage-map                # Coverage matrix across stack
pv query "rmsnorm" --project aprender  # Filter to one project
pv query --binding-gaps --all-projects # Gaps across entire stack
pv query "rope" --include-project ../custom  # Explicit project path

# Enrichment
pv query "attention" --score --proof-status --binding

# Output formats
pv query "rope" -f json
pv query "rope" -f markdown
```

#### Flags

| Flag | Description |
|---|---|
| `--limit <n>` | Max results (default: 10) |
| `--regex` | Regex pattern match |
| `--literal` | Exact string match |
| `--obligation <type>` | Filter by obligation type |
| `--min-score <f64>` | Min contract score |
| `--min-level <L1-L5>` | Min proof level |
| `--depends-on <stem>` | Contracts depending on stem |
| `--depended-by <stem>` | Contracts that stem depends on |
| `--unproven` | Obligations at L2 or below |
| `--binding-gaps` | Unimplemented bindings |
| `--binding <path>` | Binding registry path |
| `--score` | Show scores inline |
| `--proof-status` | Show L1-L5 breakdown |
| `--graph` | Show dependency context |
| `--paper` | Show paper references |
| `--call-sites` | Show cross-project `#[contract]` annotations |
| `--violations` | Show contracts violated in consumer projects |
| `--coverage-map` | Cross-project coverage matrix |
| `--project <name>` | Filter to named project (aprender, trueno, etc.) |
| `--include-project <path>` | Add explicit project path to scan |
| `--all-projects` | Force full cross-project scan |
| `--tier <n>` | Filter by contract tier (1-7) |
| `--class <A-E>` | Filter by kernel equivalence class |
| `-f, --format <fmt>` | text, json, markdown |
| `-p, --contracts-dir <dir>` | Contracts directory (default: contracts/) |

#### Index Management

```bash
# Build/rebuild contract + cross-project index
pv query --rebuild-index

# Index cached at .pv/contracts.idx + .pv/cross-project.idx
# Auto-rebuilds when contracts/ or sibling project mtime changes
```

---

## 4. Global Flags

| Flag | Description |
|---|---|
| `--help` | Show help |
| `--version` | Show version |
| `-q, --quiet` | Suppress non-essential output |
| `-v, --verbose` | Verbose output |
