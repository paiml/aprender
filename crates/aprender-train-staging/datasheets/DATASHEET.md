# Datasheet: entrenar Training Data Pipeline

Following Gebru et al. (2021) "Datasheets for Datasets"

## Motivation

- **Purpose**: Train transformer language models for code completion
- **Creator**: PAIML sovereign AI stack
- **Funding**: Self-funded research

## Composition

- **Format**: Pre-tokenized Parquet files (token IDs as i32 arrays)
- **Sequence length**: 2048 tokens per sample
- **Tokenizer**: ByteLevel BPE (albor-tokenizer-v2, 32768 vocab)
- **Content**: Python source code from public repositories
- **Size**: ~22K training sequences, ~814 validation sequences
- **Splits**: train/ and val/ directories (non-overlapping)

## Collection Process

1. Source code collected from public repositories
2. Filtered by alimentar pipeline (quality, dedup, length)
3. Tokenized with ByteLevel BPE tokenizer
4. Packed into fixed-length 2048-token sequences
5. Stored as Parquet with Arrow columnar format

## Preprocessing

- Exact deduplication (alimentar dedup)
- Quality filtering (alnum ratio, line length, dup lines, entropy)
- Optional FIM augmentation (50% PSM rate)
- Optional curriculum learning (stage-based data mixing)

## Distribution

- Local filesystem only (sovereign data governance)
- No cloud storage or external API access during training
- Content-addressable via BLAKE3 hashing

## Maintenance

- Data versioned via training_state.json (step, epoch, batch_index)
- Provenance tracked in JSONL experiment logs
- Config hash ensures reproducibility

## Legal & Ethical

- Public source code under permissive licenses
- No PII (filtered by quality pipeline)
- Contamination detection against eval benchmarks
- Right to erasure supported via model unlearning
