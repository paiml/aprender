# Data Component Specification

**Parent**: [apr-spec.md](../apr-spec.md) §13
**Status**: Active
**CLI**: `apr data`
**Implementation**: `crates/apr-cli/src/commands/data.rs`
**Library**: `alimentar`

---

## 1. Overview

Data quality pipeline for training data preparation. Supports loading,
validation, transformation, statistics, splitting, and balancing.

## 2. CLI Interface

```
apr data load <FILE> [--format <jsonl|csv|parquet>]
apr data validate <FILE> [--schema <YAML>]
apr data split <FILE> --train 0.8 --val 0.1 --test 0.1
apr data balance <FILE> --strategy <oversample|undersample>
apr data stats <FILE>
apr data evolve --seed-data <FILE> --model <MODEL> [--rounds <N>]
apr data filter-pii --input <FILE> --output <FILE>
```

## 3. Supported Formats

| Format | Extension | Description |
|--------|-----------|-------------|
| JSONL | `.jsonl` | One JSON object per line (default for training) |
| CSV | `.csv` | Comma-separated values |
| Parquet | `.parquet` | Columnar format via alimentar |

## 4. Planned Features (GH-453)

- **Synthetic data generation**: EvolKit-style instruction evolution
- **PII filtering**: Automatic PII detection and redaction
- **Domain filtering**: Quality scoring and relevance filtering

---

## Provable Contracts

### Contract: `data-quality-v1.yaml`

```yaml
metadata:
  description: "Data quality pipeline — validation, splitting, balancing"
  depends_on: []

equations:
  split_partition:
    formula: "train ∪ val ∪ test = dataset, train ∩ val = ∅, train ∩ test = ∅, val ∩ test = ∅"
    invariants:
      - "No sample appears in multiple splits"
      - "|train| + |val| + |test| == |dataset|"
      - "Ratios within 1% of requested"

  balance_oversample:
    formula: "∀ class c: count(c, balanced) >= count(majority_class, original)"
    invariants:
      - "All classes have equal representation after oversampling"
      - "Original samples preserved (only duplicates added)"
      - "No new synthetic data created (just repetition)"

  jsonl_validation:
    formula: "∀ line l: parse_json(l) succeeds AND has_required_fields(l)"
    invariants:
      - "Every line is valid JSON"
      - "Required fields present per schema"
      - "Empty lines rejected"

proof_obligations:
  - type: invariant
    property: "Split disjointness"
    formal: "train ∩ val == ∅ AND train ∩ test == ∅ AND val ∩ test == ∅"
  - type: conservation
    property: "Split completeness"
    formal: "|train| + |val| + |test| == |dataset|"
  - type: invariant
    property: "Oversample preservation"
    formal: "∀ sample in original: sample in balanced"
  - type: invariant
    property: "JSONL well-formedness"
    formal: "invalid JSON on any line → error (not skip)"

falsification_tests:
  - id: FALSIFY-DATA-001
    rule: "Split disjointness"
    prediction: "no sample appears in both train and test"
    if_fails: "Shuffle or partition logic has off-by-one"
  - id: FALSIFY-DATA-002
    rule: "Split completeness"
    prediction: "sum of split sizes equals original dataset size"
    if_fails: "Samples dropped during splitting"
  - id: FALSIFY-DATA-003
    rule: "Oversample preservation"
    prediction: "all original samples present after balancing"
    if_fails: "Oversampling replaces instead of augments"
  - id: FALSIFY-DATA-004
    rule: "Invalid JSONL rejection"
    prediction: "malformed JSON line produces error, not silent skip"
    if_fails: "Error swallowed in parsing loop"
  - id: FALSIFY-DATA-005
    rule: "Deterministic split"
    prediction: "same seed → same split assignment"
    if_fails: "RNG not seeded or shuffle non-deterministic"

kani_harnesses:
  - id: KANI-DATA-001
    obligation: "Split completeness"
    property: "sum of partitions equals total for N<=16"
    bound: 16
    strategy: exhaustive
```

### Binding Requirements

```yaml
  - contract: data-quality-v1.yaml
    equation: split_partition
    module_path: "aprender::data"
    function: split_dataset
    status: implemented

  - contract: data-quality-v1.yaml
    equation: jsonl_validation
    module_path: "aprender::data"
    function: validate_jsonl
    status: implemented
```
