# Sub-spec: Scoring System

Version: 1.0
Status: proposed
Date: 2026-04-09

**Parent:** [pv-spec.md](../pv-spec.md) Section 7

---

## 1. Design Philosophy

Scoring answers two questions:

1. **How good is this contract?** — Is the specification deep? Are
   obligations falsified? Are proofs in place?
2. **How well does this codebase use contracts?** — What fraction of
   kernels are contracted? Are bindings complete? Are contracts fresh?

Scores are quantitative (0.0-1.0), composable, and actionable. Every
gap in the score maps to a concrete improvement action.

---

## 2. Contract Score

### Input

A single `Contract` struct (parsed YAML) plus optional `BindingRegistry`.

### Dimensions

#### D1: Specification Depth (weight: 20%)

Measures how completely the math is captured in the contract.

| Component | Weight | Criterion |
|---|---|---|
| equations present | 0.30 | At least one equation with formula |
| domains specified | 0.15 | domain + codomain on all equations |
| invariants listed | 0.15 | At least one invariant per equation |
| kernel_structure | 0.15 | Phase decomposition present |
| tolerances derived | 0.10 | tolerance field on obligations |
| paper references | 0.10 | At least one arXiv/DOI reference |
| depends_on | 0.05 | Dependencies declared (if applicable) |

```
spec_depth = sum(component_weight * present_indicator)
```

#### D2: Falsification Coverage (weight: 25%)

Ratio of proof obligations with at least one falsification test.

```
fc = count(obligations_with_matching_test) / count(total_obligations)
```

A falsification test "matches" an obligation if its `rule` field
references the obligation's property or ID.

#### D3: Kani Proof Coverage (weight: 25%)

Ratio of obligations with Kani harnesses, weighted by strategy quality.

| Strategy | Weight |
|---|---|
| exhaustive | 1.00 |
| bounded_int | 0.90 |
| stub_float | 0.80 |
| compositional | 0.70 |

```
kc = sum(best_strategy_weight per obligation) / count(total_obligations)
```

An obligation with multiple harnesses uses the best strategy weight.

#### D4: Lean Proof Coverage (weight: 10%)

```
lc = count(obligations with lean.status == "proved")
   / count(obligations with lean.status != "not-applicable")
```

If no obligations have Lean metadata, lc = 0.

#### D5: Binding Coverage (weight: 20%)

```
bc = (implemented + 0.5 * partial) / total_bindings_for_this_contract
```

If no binding registry provided, bc = 0.

### Composite Score

```
composite = D1*0.20 + D2*0.25 + D3*0.25 + D4*0.10 + D5*0.20
```

### Grade Mapping

| Grade | Range | Description |
|---|---|---|
| A | >= 0.90 | Exemplary |
| B | >= 0.75 | Strong |
| C | >= 0.60 | Adequate |
| D | >= 0.40 | Weak |
| F | < 0.40 | Deficient |

---

## 3. Codebase Score

### Input

A directory path, a set of parsed contracts, and a `BindingRegistry`.

### Dimensions

#### CD1: Contract Coverage (weight: 30%)

What fraction of the codebase's kernel functions have corresponding
contracts?

```
cc = count(functions_with_contract_annotation) / count(total_kernel_functions)
```

"Kernel functions" are identified by `#[contract(...)]` annotations.
Functions without annotations are uncontracted.

Alternatively (binding-based):
```
cc = count(unique_contracts_in_binding) / count(total_contracts_available)
```

#### CD2: Binding Completeness (weight: 20%)

Of contracts that are bound, how complete are the bindings?

```
bc = count(status == "implemented")
   / count(all_bindings_for_bound_contracts)
```

#### CD3: Mean Contract Score (weight: 20%)

Average composite score across all bound contracts.

```
mcs = mean(score_contract(c) for c in bound_contracts)
```

#### CD4: Proof Depth Distribution (weight: 15%)

How high on the verification ladder are obligations?

```
pd = (L1*0.1 + L2*0.2 + L3*0.4 + L4*0.8 + L5*1.0) / count(obligations)
```

Higher concentration at L4/L5 = higher score.

#### CD5: Drift Detection (weight: 15%) [IMPLEMENTED]

Are contracts current with the code?

```
drift = 1.0 - (stale_contracts / total_bound_contracts)
```

A contract is "stale" if its git commit timestamp is more recent than
the binding file's git commit timestamp. Uses `git log -1 --format=%ct`
for timestamp comparison. Implementation in `scoring::drift` module.

### Composite

```
codebase_composite = CD1*0.30 + CD2*0.20 + CD3*0.20 + CD4*0.15 + CD5*0.15
```

Same grade thresholds as contract score.

---

## 4. Gap Analysis [IMPLEMENTED]

`pv score` reports the top gaps sorted by impact. Impact is computed as:

```
impact = (1.0 - obligation_coverage) * dependency_fanout * tier_weight
```

Where:
- `obligation_coverage` = fraction of obligations proven for this contract
- `dependency_fanout` = number of contracts that depend on this one
- `tier_weight` = {tier1: 1.0, tier2: 0.9, tier3: 0.7, tier4: 0.5, ...}

### Gap Actions

Each gap maps to a concrete action:

| Gap Type | Action |
|---|---|
| No falsification test for obligation | Write a probar property test |
| No Kani harness for obligation | Write a `#[kani::proof]` harness |
| Binding status = partial | Complete the implementation |
| Binding status = not_implemented | Implement the equation |
| Stale contract | Run `pv diff` and update contract |
| No contract for kernel function | Write a new YAML contract |

---

## 5. Quality Gate (`pv lint`) [IMPLEMENTED]

`pv lint` is a single-command quality gate that runs validation, audit,
and scoring across all contracts in a directory. It combines the checks
that would otherwise require separate `pv validate`, `pv audit`, and
`pv score` invocations into one pass/fail gate suitable for CI and
pre-commit hooks.

### Gates

`pv lint` executes three gates sequentially. Each gate produces a
pass/fail result with timing. If any gate fails, the overall result
is FAIL and exit code is 1.

| Gate | What it checks | Pass condition |
|------|---------------|----------------|
| **validate** | Schema completeness (SCHEMA-001..013, PROVABILITY-001) | 0 errors across all contracts |
| **audit** | Traceability chain (paper→equation→obligation→test→proof) | 0 audit findings across all contracts |
| **score** | 5-dimension quality score | All contracts >= `--min-score` threshold |

### CLI

```bash
# Lint all contracts (default threshold: 0.0 = no score gate)
pv lint contracts/

# Lint with minimum score gate
pv lint contracts/ --min-score 0.60

# Include binding registry for D5 (binding coverage) scoring
pv lint contracts/ --binding contracts/aprender/binding.yaml --min-score 0.75

# JSON output for CI artifact collection
pv lint contracts/ -f json
```

#### Flags

| Flag | Description |
|---|---|
| `--min-score <f64>` | Minimum composite score (default: 0.0 = skip score gate) |
| `--binding <path>` | Binding registry YAML for binding coverage scoring |
| `-f, --format <fmt>` | Output format: text (default), json |

### Output Format (text)

```
pv lint — contract quality gate
================================

Gate 1: validate ............... PASS (107 contracts, 0 errors, 12 warnings) [42ms]
Gate 2: audit .................. PASS (107 contracts, 0 findings) [38ms]
Gate 3: score .................. PASS (107 contracts, min=0.27, mean=0.50, threshold=0.00) [15ms]

Result: PASS (3/3 gates passed) [95ms]
```

On failure:

```
Gate 1: validate ............... FAIL (107 contracts, 3 errors, 12 warnings) [42ms]
  [ERROR] SCHEMA-001: metadata.references must not be empty (new-kernel-v1.yaml)
  [ERROR] PROVABILITY-001: missing proof_obligations (new-kernel-v1.yaml)
  [ERROR] SCHEMA-003: equations must contain at least one equation (stub-v1.yaml)

Gate 2: audit .................. SKIP (validation failed)
Gate 3: score .................. SKIP (validation failed)

Result: FAIL (0/3 gates passed) [42ms]
```

### Output Format (json)

```json
{
  "passed": true,
  "gates": [
    {"name": "validate", "passed": true, "contracts": 107, "errors": 0, "warnings": 12, "duration_ms": 42},
    {"name": "audit", "passed": true, "contracts": 107, "findings": 0, "duration_ms": 38},
    {"name": "score", "passed": true, "contracts": 107, "min_score": 0.27, "mean_score": 0.50, "threshold": 0.0, "duration_ms": 15}
  ],
  "total_duration_ms": 95
}
```

### Makefile Integration

```makefile
lint: clippy deny pv-lint

pv-lint:
	cargo run --bin pv -- lint contracts/
```

### CI Integration

```yaml
# .github/workflows/contract-quality.yml
name: Contract Quality Gate
on: [push, pull_request]
jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install pv
        run: cargo install --path crates/provable-contracts-cli
      - name: Contract quality gate
        run: pv lint contracts/ --min-score 0.60
```

### Relationship to pmat quality-gate

`pv lint` is the **contract-layer** quality gate, analogous to how
`pmat quality-gate` is the **code-layer** quality gate:

| Concern | Tool | Gates |
|---------|------|-------|
| Rust code quality | `pmat quality-gate` | clippy, tests, coverage, complexity |
| Contract quality | `pv lint` | validate, audit, score |

A complete CI pipeline runs both:

```bash
pmat quality-gate          # Code: clippy + tests + coverage + complexity
pv lint contracts/ --min-score 0.60  # Contracts: validate + audit + score
```

### Trend Tracking

```bash
# Output JSON for trend tracking
pv lint contracts/ -f json > lint-report.json

# Score-only JSON for dashboards
pv score contracts/ -f json > scores.json
```

---

## 6. Custom Weights [IMPLEMENTED]

Default weights can be overridden:

```bash
pv score contracts/softmax-kernel-v1.yaml \
  --weights '{"spec_depth": 0.10, "falsification": 0.30, "kani": 0.30, "lean": 0.10, "binding": 0.20}'
```

All weights must sum to 1.0. If they don't, they are normalized.
