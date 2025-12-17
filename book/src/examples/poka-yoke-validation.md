# Case Study: Poka-Yoke Validation (APR-POKA-001)

Poka-yoke (ポカヨケ, "mistake-proofing") is a Toyota Way concept that builds quality in at the source, not at inspection. The APR-POKA-001 specification brings this principle to ML model serialization.

## Overview

The Poka-yoke validation system provides:

- **Gate**: Individual validation check with pass/fail and actionable error message
- **PokaYokeResult**: Collection of gates with score (0-100) and letter grade (A+ to F)
- **PokaYoke trait**: Extensible validation per model type
- **Jidoka gate**: Save is REFUSED if quality_score=0 (stop the line)

## Core Concepts

### Gates: Atomic Validation Checks

Each gate validates one specific aspect of the model:

```rust
use aprender::format::validation::Gate;

// A passing gate
let gate = Gate::pass("filterbank_present", 20);
assert!(gate.passed);
assert_eq!(gate.points, 20);

// A failing gate with actionable error
let gate = Gate::fail(
    "filterbank_normalized",
    30,
    "Fix: Apply 2.0/bandwidth normalization (max=0.5, expected <0.1)"
);
assert!(!gate.passed);
assert_eq!(gate.points, 0);
assert!(gate.error.is_some());
```

**Key principle**: Error messages must be **actionable**. Tell the user exactly how to fix the issue, not just that it's wrong.

### PokaYokeResult: Aggregated Validation

```rust
use aprender::format::validation::{Gate, PokaYokeResult};

// Method 1: Add gates incrementally
let mut result = PokaYokeResult::new();
result.add_gate(Gate::pass("filterbank_present", 20));
result.add_gate(Gate::pass("filterbank_normalized", 30));
result.add_gate(Gate::fail("encoder_layers", 25, "Fix: Need ≥4 layers"));
result.add_gate(Gate::pass("vocabulary_size", 25));

// Method 2: Bulk construction with from_gates (v0.19+)
let gates = vec![
    Gate::pass("filterbank_present", 20),
    Gate::pass("filterbank_normalized", 30),
    Gate::fail("encoder_layers", 25, "Fix: Need ≥4 layers"),
    Gate::pass("vocabulary_size", 25),
];
let result = PokaYokeResult::from_gates(gates);

// Score and grade
println!("Score: {}/100", result.score);        // 75/100
println!("Grade: {}", result.grade());          // C+
println!("Passed: {}", result.passed());        // true (score >= 60)

// Failed gates and errors
for gate in result.failed_gates() {
    println!("{}: {}", gate.name, gate.error.as_ref().unwrap());
}
```

### Grading Scale

| Score Range | Grade | Status |
|-------------|-------|--------|
| 95-100      | A+    | Excellent |
| 90-94       | A     | Very Good |
| 85-89       | B+    | Good |
| 80-84       | B     | Above Average |
| 75-79       | C+    | Average |
| 70-74       | C     | Below Average |
| 60-69       | D     | Passing |
| 0-59        | F     | Failing |

**Passing threshold**: Score ≥ 60 (Grade D or better)

## Implementing PokaYoke Trait

```rust
use aprender::format::validation::{Gate, PokaYoke, PokaYokeResult};

struct WhisperModel {
    has_filterbank: bool,
    filterbank_max: f32,
    encoder_layers: usize,
    vocab_size: usize,
}

impl PokaYoke for WhisperModel {
    fn poka_yoke_validate(&self) -> PokaYokeResult {
        let mut result = PokaYokeResult::new();

        // Gate 1: Filterbank must be embedded (20 points)
        if self.has_filterbank {
            result.add_gate(Gate::pass("filterbank_present", 20));
        } else {
            result.add_gate(Gate::fail(
                "filterbank_present",
                20,
                "Fix: Embed Slaney-normalized filterbank via MelFilterbankData::mel_80()"
            ));
        }

        // Gate 2: Filterbank must be Slaney-normalized (30 points)
        if self.has_filterbank && self.filterbank_max < 0.1 {
            result.add_gate(Gate::pass("filterbank_normalized", 30));
        } else if self.has_filterbank {
            result.add_gate(Gate::fail(
                "filterbank_normalized",
                30,
                format!("Fix: Apply 2.0/bandwidth normalization (max={:.4}, expected <0.1)",
                        self.filterbank_max)
            ));
        }

        // Gate 3: Encoder layers (25 points)
        if self.encoder_layers >= 4 {
            result.add_gate(Gate::pass("encoder_layers", 25));
        } else {
            result.add_gate(Gate::fail(
                "encoder_layers",
                25,
                format!("Fix: Model needs ≥4 encoder layers (has {})", self.encoder_layers)
            ));
        }

        // Gate 4: Vocabulary (25 points)
        if self.vocab_size > 0 {
            result.add_gate(Gate::pass("vocabulary_size", 25));
        } else {
            result.add_gate(Gate::fail(
                "vocabulary_size",
                25,
                "Fix: Set vocabulary size > 0 for tokenization"
            ));
        }

        result
    }
}
```

## Integration with SaveOptions

The quality score is embedded in the APR header (byte 22):

```rust
use aprender::format::{save, ModelType, SaveOptions};
use aprender::format::validation::PokaYoke;

let model = WhisperModel { /* ... */ };
let result = model.poka_yoke_validate();

// Method 1: Use PokaYokeResult directly
let options = SaveOptions::new()
    .with_name("whisper-tiny")
    .with_poka_yoke_result(&result);

// Method 2: Set score manually
let options = SaveOptions::new()
    .with_quality_score(85);

// Save model (quality_score embedded in header)
save(&model, ModelType::LinearRegression, "model.apr", options)?;
```

## Jidoka: Stop the Line

Jidoka (自働化) is the Toyota principle of "automation with a human touch" - machines stop automatically when defects are detected.

**Critical behavior**: `save()` REFUSES to write if `quality_score == Some(0)`:

```rust
let broken_model = WhisperModel::new(); // Fails all validation
let result = broken_model.poka_yoke_validate();
assert_eq!(result.score, 0);

let options = SaveOptions::new()
    .with_poka_yoke_result(&result);  // score = 0

// This FAILS with ValidationError
match save(&broken_model, ModelType::LinearRegression, "bad.apr", options) {
    Err(AprenderError::ValidationError { message }) => {
        println!("Jidoka triggered: {}", message);
        // "Jidoka: Refusing to save model with quality_score=0.
        //  Fix validation errors or use score=None to skip validation."
    }
    _ => unreachable!()
}
```

### Bypass Options

If you need to save a model without validation:

```rust
// Option 1: Skip validation entirely (score=None, stored as 0 in file)
let options = SaveOptions::new();  // No quality_score set

// Option 2: Acknowledge low quality (score > 0 but < 60)
let options = SaveOptions::new()
    .with_quality_score(1);  // Allows save, but marks as F grade
```

## APR Header Format

The quality score is stored in byte 22 of the 32-byte APR header:

| Offset | Size | Field |
|--------|------|-------|
| 0-3    | 4    | Magic ("APRN") |
| 4-5    | 2    | Version (major, minor) |
| 6-7    | 2    | Model type |
| 8-11   | 4    | Metadata size |
| 12-15  | 4    | Payload size |
| 16-19  | 4    | Uncompressed size |
| 20     | 1    | Compression |
| 21     | 1    | Flags |
| **22** | 1    | **Quality score (0-100)** |
| 23-31  | 9    | Reserved |

## API Reference

### Gate

| Method | Description |
|--------|-------------|
| `Gate::pass(name, points)` | Create passing gate with awarded points |
| `Gate::fail(name, max_points, error)` | Create failing gate with actionable error |
| `gate.passed` | Whether gate passed |
| `gate.points` | Points awarded (0 if failed) |
| `gate.max_points` | Maximum possible points |
| `gate.error` | Error message (if failed) |

### PokaYokeResult

| Method | Description |
|--------|-------------|
| `PokaYokeResult::new()` | Create empty result |
| `PokaYokeResult::from_gates(gates)` | Create from vector of gates (bulk) |
| `result.add_gate(gate)` | Add gate and recalculate score |
| `result.score` | Total score (0-100) |
| `result.max_score` | Maximum possible score |
| `result.grade()` | Letter grade (A+ to F) |
| `result.passed()` | Whether validation passed (score ≥ 60) |
| `result.failed_gates()` | Get all failed gates |
| `result.error_summary()` | Formatted error messages |

### Helper Functions

| Function | Description |
|----------|-------------|
| `fail_no_validation_rules()` | Create failing result for unvalidated models |

### SaveOptions

| Method | Description |
|--------|-------------|
| `with_quality_score(score)` | Set quality score directly |
| `with_poka_yoke_result(&result)` | Set score from validation result |

## Running the Example

```bash
cargo run --example poka_yoke_validation
```

Output demonstrates:
1. **Perfect model (A+)**: All gates pass, saved successfully
2. **Partial model (C)**: Some gates fail, saved with warnings
3. **Failing model (F)**: All gates fail, Jidoka refuses save
4. **Gate inspection**: Detailed view of individual gate results

## Toyota Way Principles

| Principle | Application |
|-----------|-------------|
| **Poka-yoke** | Validation gates prevent shipping broken models |
| **Jidoka** | Automatic stop when quality_score=0 |
| **Genchi Genbutsu** | Actionable errors tell exactly what's wrong |
| **Kaizen** | Incremental validation improvements per model type |

## Best Practices

1. **Actionable errors**: Every `Gate::fail()` must explain HOW to fix the issue
2. **Weighted gates**: Assign more points to critical validations
3. **Implement per model type**: Each model type has unique validation rules
4. **Test your validation**: Write tests for both pass and fail cases
5. **Don't bypass Jidoka**: If save fails, fix the model instead of skipping validation

## See Also

- [APR Format Specification](../tools/apr-spec.md)
- [Case Study: APR 100-Point Quality Scoring](./apr-scoring.md)
- [Toyota Way: Jidoka](../toyota-way/jidoka.md)
- [Case Study: Pipeline Verification](./pipeline-verification.md)
