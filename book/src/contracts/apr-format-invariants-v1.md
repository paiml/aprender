# apr-format-invariants-v1

**Version:** 1.0.0

APR format invariants — serialization roundtrip, schema validation, and report formatting for model QA evidence

## References

- apr-model-qa-playbook — production model quality assurance pipeline
- Apache Arrow IPC format specification

## Equations

### detect_regression

```
detect_regression: (MqsResult, MqsResult) -> Vec<Regression>
  Compares current vs baseline MQS results.
  Regression = dimension score decreased beyond tolerance.

```

**Domain:** $Two valid MqsResult values (current, baseline)$

**Codomain:** `Vec<Regression> (may be empty)`

**Invariants:**

- $No regressions when current >= baseline for all dimensions$
- $Regression detected when any dimension drops > tolerance$

### format_report

```
format_mqs_report: MqsResult -> String
  Renders human-readable MQS report with dimension breakdown.

```

**Domain:** $MqsResult with valid scores$

**Codomain:** $String (non-empty)$

**Invariants:**

- $Report contains all 6 dimension scores$
- $Report contains overall grade$

### parse_playbook

```
parse_qa_playbook: Path -> Result<QaPlaybook, ParseError>
  Parses YAML playbook defining checks, thresholds, and model configs.

```

**Domain:** $Path to a YAML file$

**Codomain:** $Result<QaPlaybook, ParseError>$

**Invariants:**

- $Valid YAML with correct schema parses successfully$
- $Missing required fields produce descriptive ParseError$

### serialize_roundtrip

```
serialize_model_evidence: ModelEvidence -> Result<Bytes, SerError>
  Serializes evidence to a deterministic binary format.
  Inverse: deserialize(serialize(e)) == e for all valid evidence e.

```

**Domain:** $ModelEvidence with all required fields populated$

**Codomain:** $Result<Bytes, SerError>$

**Invariants:**

- `Roundtrip: deserialize(serialize(e)) == e`
- $Output size proportional to evidence complexity$

### validate_schema

```
validate_evidence_schema: Bytes -> Result<ModelEvidence, ValidationError>
  Validates binary evidence against expected schema.
  Rejects unknown fields, missing required fields, type mismatches.

```

**Domain:** $Bytes (arbitrary byte stream)$

**Codomain:** $Result<ModelEvidence, ValidationError>$

**Invariants:**

- $Valid evidence always passes validation$
- $Truncated input produces ValidationError, never panics$

## Proof Obligations

| # | Type | Property | Formal |
|---|------|----------|--------|
| 1 | equivalence | Serialization roundtrip | `deserialize(serialize(e)) == e` |
| 2 | invariant | Schema validation soundness | $valid evidence always passes; invalid never passes$ |
| 3 | postcondition | Report completeness | $format_report output contains all 6 dimension names and scores$ |

## Falsification Tests

| ID | Rule | Prediction | If Fails |
|----|------|------------|----------|
| FALSIFY-APR-001 | Roundtrip identity | deserialize(serialize(e)) == e for random valid evidence | Serialization loses precision or field |
| FALSIFY-APR-002 | Truncated input rejection | Truncated bytes produce ValidationError, not panic | Missing bounds check in deserializer |
| FALSIFY-APR-003 | Regression detection | Identical baselines produce zero regressions | Floating point comparison not using tolerance |

## Kani Harnesses

| ID | Obligation | Bound | Strategy |
|----|------------|-------|----------|
| KANI-APR-001 | Serialization roundtrip | 8 | exhaustive |
| KANI-APR-002 | Schema validation soundness | 16 | exhaustive |
| KANI-APR_FO-003 | Report completeness | 8 | exhaustive |

## QA Gate

**apr-format-invariants-v1 Contract** (F-AFIV-001)

Quality gate for APR format invariants — serialization roundtrip, schema vali

**Checks:** validation, falsification

