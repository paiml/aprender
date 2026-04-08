# Shell Safety Classifier Training

Trains an MLP classifier to predict shell script safety using the
bashrs corpus (17,942 entries) merged with adversarial data (~8,000
entries for minority classes).

The model classifies scripts into 5 safety categories:

- **safe**: passes all checks (lint, deterministic, idempotent)
- **needs-quoting**: variable quoting issues
- **non-deterministic**: contains `$RANDOM`, `$$`, timestamps
- **non-idempotent**: missing `-p`/`-f` flags
- **unsafe**: security rule violations

## Prerequisites

```bash
# Export corpus + generate adversarial data
cd /path/to/bashrs
cargo run -- corpus export-dataset --format classification -o /tmp/corpus.jsonl
cargo run -- generate-adversarial --verify -o /tmp/adversarial.jsonl
{ cat /tmp/corpus.jsonl; echo; cat /tmp/adversarial.jsonl; } > /tmp/combined.jsonl
```

## Run

```bash
cargo run --example shell_safety_training -- /tmp/combined.jsonl
```

## Source

```rust,ignore
// Run this example:
//   cargo run --example shell_safety_training
//
// See the CLI reference and source code in crates/ for implementation details.
```
