# Data Quality Pipeline for Fine-Tuning (GH-453)

Three-stage data quality pipeline for preparing fine-tuning datasets:
1. **PII Filtering** — detect and redact sensitive information
2. **Quality Scoring** — score and filter by length, diversity, repetition, structure
3. **Instruction Evolution** — EvolKit-style complexity enhancement

## Running the Examples

```bash
cargo run --example pii_filtering           # PII detection + redaction
cargo run --example data_quality_pipeline   # Full 3-stage pipeline
```

## Stage 1: PII Filtering

Detects and redacts 5 PII types from text and JSONL data.

| Type | Pattern | Redaction Tag |
|------|---------|---------------|
| Email | `user@domain.tld` | `[EMAIL]` |
| Phone | `(XXX) XXX-XXXX`, `XXX-XXX-XXXX` | `[PHONE]` |
| SSN | `XXX-XX-XXXX` | `[SSN]` |
| Credit Card | 13-19 digits (with spaces/dashes) | `[CREDIT_CARD]` |
| IP Address | `X.X.X.X` (0-255 octets) | `[IP_ADDRESS]` |

```rust
use aprender::data::pii::{scan_pii, filter_pii, filter_pii_jsonl};

// Scan and redact
let filtered = filter_pii("Contact user@example.com, SSN: 123-45-6789");
assert_eq!(filtered, "Contact [EMAIL], SSN: [SSN]");

// JSONL pipeline
let (output, count) = filter_pii_jsonl(r#"{"text": "user@example.com"}"#);
assert_eq!(count, 1);
```

## Stage 2: Quality Scoring and Filtering

Scores text on 5 quality dimensions and filters by configurable threshold.

| Dimension | What it measures |
|-----------|-----------------|
| Length | Text within acceptable character range |
| Diversity | Unique/total word ratio (vocabulary richness) |
| Repetition | Repeated 3-grams (detects copy-paste) |
| Structure | Capitalization, punctuation, sentence form |
| Language | ASCII ratio (proxy for language consistency) |

```rust
use aprender::data::quality_filter::{score_quality, filter_by_quality, QualityConfig};

let config = QualityConfig::default().with_min_quality(0.5);
let report = score_quality("Well-written text with diverse vocabulary.", &config);
assert!(report.passed);

// Batch filtering
let texts = vec!["Good text.".into(), "x".into()];
let (passed, rejected) = filter_by_quality(&texts, &config);
assert_eq!(rejected, 1);
```

## Stage 3: Instruction Evolution

WizardLM-style Evol-Instruct for making training instructions more complex.

| Strategy | Effect |
|----------|--------|
| `AddConstraints` | Adds specificity ("without using X") |
| `DeepenReasoning` | Requires step-by-step explanation |
| `Concretize` | Grounds in specific domain context |
| `IncreaseComplexity` | Extends with additional requirements |
| `BreadthMutation` | Rephrases as different task type |

```rust
use aprender::data::evolve::{evolve_instruction, evolve_batch, EvolConfig, EvolStrategy};

let config = EvolConfig::default()
    .with_rounds(2)
    .with_strategies(vec![EvolStrategy::AddConstraints, EvolStrategy::DeepenReasoning]);

let evolved = evolve_instruction("Implement a sorting algorithm", &config);
assert_eq!(evolved.len(), 2);  // One per round
assert!(evolved[1].instruction.len() > evolved[0].instruction.len());
```

## Full Pipeline

```rust
use aprender::data::pii::filter_pii_jsonl;
use aprender::data::quality_filter::{filter_jsonl_quality, QualityConfig};
use aprender::data::evolve::{evolve_batch, EvolConfig};

// Stage 1: PII redaction
let (pii_clean, pii_count) = filter_pii_jsonl(raw_jsonl);

// Stage 2: Quality filtering
let config = QualityConfig::default().with_min_quality(0.5);
let (quality_clean, stats) = filter_jsonl_quality(&pii_clean, &config);

// Stage 3: Instruction evolution (for instruction-tuning datasets)
let evol_config = EvolConfig::default().with_rounds(3);
let evolved = evolve_batch(&instructions, &evol_config);
```

## Contract Reference

See `contracts/data-quality-v1.yaml` for formal specifications:
- **FALSIFY-PII-001**: All 5 PII types detected in mixed text
- **FALSIFY-PII-002**: Redaction preserves non-PII text
- **FALSIFY-PII-003**: JSONL structure preserved after filtering
- **FALSIFY-EVOL-001**: Evolution always produces longer output
- **FALSIFY-EVOL-002**: Batch evolution preserves all inputs

## Design Notes

- Zero-regex PII: hand-rolled byte-level scanners for performance
- Overlapping PII matches deduplicated (longer match wins)
- Quality scoring is heuristic-based (no model required)
- Instruction evolution is deterministic with fixed seed
- All stages are composable in any order
