# Case Study: Tokenizer Surgery

**Ticket**: GH-447
**Module**: `aprender::online::tokenizer_surgery`

## Overview

Transplants embedding rows when adapting a pre-trained model to a new tokenizer. Supports direct copy, nearest-neighbor, and average-pool strategies for unmatched tokens.

## Key Components

- **`TokenizerSurgeryConfig`** — Vocab sizes, overlap threshold, surgery method
- **`SurgeryMethod`** — DirectCopy, NearestNeighbor, AveragePool
- **`VocabMapping`** — Bidirectional source/target index mapping
- **`compute_vocab_overlap`** — O(n+m) vocabulary intersection
- **`transplant_embeddings`** — Row-by-row embedding transfer
- **`validate_surgery`** — Quality gate on overlap ratio

## Run

```bash
cargo run --example tokenizer_surgery
```

## Falsification Tests

| ID | Property | Status |
|----|----------|--------|
| FALSIFY-SURGERY-001 | Overlap ratio in [0, 1] | Falsified (holds) |
| FALSIFY-SURGERY-002 | Transplant preserves dimensions | Falsified (holds) |
| FALSIFY-SURGERY-003 | Identical vocabs yield identity | Falsified (holds) |
