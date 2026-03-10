# Case Study: Per-Layer Merge Granularity

**Ticket**: GH-452
**Module**: `aprender::online::per_layer_merge`

## Overview

Provides fine-grained per-tensor merge strategy overrides via YAML configuration. Different layers (attention, MLP, embeddings) can use different strategies and weights.

## Key Components

- **`LayerMergeConfig`** — Ordered rules with default fallback
- **`LayerRule`** — Pattern + strategy + optional weights/scale
- **`MergeYamlConfig`** — Top-level YAML config (models, output, layers)
- **`match_layer_rule`** — First-match pattern lookup (substring + `*` wildcards)
- **`parse_merge_yaml`** — State-machine YAML parser (no serde_yaml dependency)
- **`validate_merge_config`** — Model count, strategy names, weight finiteness

## Run

```bash
cargo run --example per_layer_merge
```

## Falsification Tests

| ID | Property | Status |
|----|----------|--------|
| FALSIFY-PERLAYER-001 | Empty rules match nothing | Falsified (holds) |
| FALSIFY-PERLAYER-002 | Valid YAML always parses | Falsified (holds) |
| FALSIFY-PERLAYER-003 | Invalid configs rejected | Falsified (holds) |
