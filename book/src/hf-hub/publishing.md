# Publishing to HuggingFace Hub

alimentar is the **only Rust crate** with native HuggingFace Hub upload support. The official `hf-hub` crate only supports downloads.

## Critical Warning: Data Quality Before Publishing

> **WARNING: Publishing low-quality datasets to HuggingFace is HARMFUL to the ML community.**
>
> Poor quality training data leads to:
> - Models that learn incorrect patterns
> - Wasted compute resources on garbage training
> - Propagation of errors across downstream models
> - Reduced trust in the dataset ecosystem
>
> **ALWAYS validate your data quality before publishing.**

## Data Quality Checklist

Before uploading ANY dataset to HuggingFace, verify:

### 1. Run Quality Score

```bash
# Check quality score - MINIMUM Grade B (85%) required
alimentar quality score my_dataset.parquet

# Example output:
# Quality Score: 92.3% (Grade A)
# - Completeness: 98% (no null values in critical columns)
# - Uniqueness: 95% (low duplicate rate)
# - Consistency: 89% (format validation passed)
# - Schema: 100% (all types valid)
```

### 2. Quality Grade Requirements

| Grade | Score | Recommendation |
|-------|-------|----------------|
| A | 95%+ | Excellent - safe to publish |
| B | 85-94% | Good - review warnings before publishing |
| C | 70-84% | **DO NOT PUBLISH** - fix issues first |
| D | <70% | **REJECTED** - major quality problems |

### 3. Use Quality Profiles

```bash
# Apply domain-specific quality rules
alimentar quality score --profile ml-training data.parquet
alimentar quality score --profile doctest-corpus doctests.parquet
alimentar quality score --profile code-translation code.parquet
```

## Improving Data Quality

### Recipe 1: Clean with aprender

[aprender](https://github.com/paiml/aprender) provides ML-focused data cleaning:

```bash
# Install aprender
cargo install aprender

# Clean dataset with ML-aware transforms
aprender clean input.parquet --output cleaned.parquet \
    --remove-nulls \
    --deduplicate \
    --validate-types \
    --normalize-text

# Verify improvement
alimentar quality score cleaned.parquet
```

### Recipe 2: Augment with entrenar

[entrenar](https://github.com/paiml/entrenar) provides training-focused transforms:

```bash
# Install entrenar
cargo install entrenar

# Augment dataset for training
entrenar augment input.parquet --output augmented.parquet \
    --balance-classes \
    --add-noise 0.1 \
    --synthetic-samples 1000

# Verify quality maintained
alimentar quality score augmented.parquet
```

### Recipe 3: Full Pipeline

```bash
#!/bin/bash
# quality_pipeline.sh - MANDATORY before HF Hub publishing

set -e  # Exit on any error

INPUT="$1"
OUTPUT="$2"
REPO="$3"

echo "=== Step 1: Initial Quality Check ==="
INITIAL=$(alimentar quality score "$INPUT" --json | jq '.score')
echo "Initial quality: $INITIAL%"

if (( $(echo "$INITIAL < 70" | bc -l) )); then
    echo "ERROR: Initial quality too low. Cleaning required."

    echo "=== Step 2: Clean with aprender ==="
    aprender clean "$INPUT" --output /tmp/cleaned.parquet

    echo "=== Step 3: Validate cleaning ==="
    CLEANED=$(alimentar quality score /tmp/cleaned.parquet --json | jq '.score')
    echo "After cleaning: $CLEANED%"

    INPUT="/tmp/cleaned.parquet"
fi

echo "=== Step 4: Final Quality Gate ==="
FINAL=$(alimentar quality score "$INPUT" --json | jq '.score')

if (( $(echo "$FINAL < 85" | bc -l) )); then
    echo "FATAL: Quality score $FINAL% below 85% threshold"
    echo "DO NOT PUBLISH - fix data quality issues first"
    exit 1
fi

echo "=== Step 5: Publish to HuggingFace ==="
alimentar hub push "$INPUT" "$REPO" \
    --readme /tmp/readme.md \
    --message "Quality-validated upload (score: $FINAL%)"

echo "SUCCESS: Published with quality score $FINAL%"
```

## CLI Usage

### Basic Upload

```bash
# Set your HuggingFace token
export HF_TOKEN="hf_xxxxx"

# Upload parquet file
alimentar hub push data.parquet paiml/my-dataset

# Upload with custom path
alimentar hub push train.parquet paiml/my-dataset \
    --path-in-repo data/train.parquet

# Upload with README
alimentar hub push data.parquet paiml/my-dataset \
    --readme README.md \
    --message "Initial upload with doctest corpus"
```

### With Quality Enforcement

```bash
# Recommended: Check quality before publishing
alimentar quality score data.parquet && \
alimentar hub push data.parquet paiml/my-dataset
```

## API Usage

```rust
use alimentar::hf_hub::HfPublisher;

// Create publisher
let publisher = HfPublisher::new("paiml/my-dataset")
    .with_token(std::env::var("HF_TOKEN").unwrap())
    .with_commit_message("Upload quality-validated corpus");

// Upload parquet (uses LFS for binary files)
publisher.upload_parquet_file_sync(
    Path::new("data.parquet"),
    "data/train.parquet"
)?;

// Upload README (validates dataset card)
publisher.upload_readme_validated_sync(&readme_content)?;
```

## Technical Details

### File Type Detection

| File Type | Method | API |
|-----------|--------|-----|
| Text (.md, .json, .csv) | Direct NDJSON | `/api/datasets/{repo}/commit/main` |
| Binary (.parquet, .arrow, .png) | LFS Batch | `/datasets/{repo}.git/info/lfs/objects/batch` |

### LFS Upload Flow

1. Compute SHA256 hash of file content
2. POST to LFS batch API with object OID
3. Extract presigned S3 URL from response
4. PUT binary content to S3
5. POST NDJSON commit with `lfsFile` reference

## Common Issues

### "Quality score below threshold"

```
ERROR: Quality score 72% below 85% threshold

Fix: Run aprender clean to address issues:
  - Remove null values: aprender clean --remove-nulls
  - Fix duplicates: aprender clean --deduplicate
  - Validate types: aprender clean --validate-types
```

### "Invalid task_categories"

```
ERROR: Invalid 'task_categories': 'text2text-generation' is not valid

Fix: Use valid HuggingFace task category:
  - text-generation
  - translation
  - text-classification
```

See [Dataset Card Validation](./api-reference.md#dataset-card-validation) for valid categories.
