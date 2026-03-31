# Data Quality Pipeline

Demonstrates the full data quality pipeline for fine-tuning (GH-453):

1. **PII filtering** -- detection and redaction of emails, IPs, etc.
2. **Quality scoring and filtering** -- length, diversity, repetition, structure
3. **EvolKit-style instruction evolution** -- complexity enhancement

## Run

```bash
cargo run --example data_quality_pipeline
```

## Source

```rust,ignore
{{#include ../../../examples/data_quality_pipeline.rs}}
```
