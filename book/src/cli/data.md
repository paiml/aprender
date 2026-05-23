<!-- PCU: cli-data | contract: contracts/apr-page-cli-data-v1.yaml -->

# apr data

Data quality pipeline (audit, split, balance) — powered by alimentar

**Category**: Training

## Synopsis

```text
apr data [OPTIONS]
```

## Example

```bash
apr data inspect train.jsonl
```

## What this does

`apr data` is the training-data hygiene suite — audit for quality issues,
stratified split, contamination check against benchmarks, and class balancing.
A finetune run is only as good as its dataset; this command catches the
mundane issues (label imbalance, near-duplicates, benchmark leakage) before
they ruin a 12-hour GPU run.

## Key subcommands

| Subcommand | What it does | Example |
|-----------|-------------|---------|
| `data audit` | Quality scan (duplicates, label balance, length stats) | `apr data audit train.jsonl` |
| `data split` | Stratified train/val/test split | `apr data split train.jsonl --ratios 0.8,0.1,0.1` |
| `data decontaminate` | N-gram overlap against benchmark sets | `apr data decontaminate train.jsonl --against humaneval` |
| `data balance` | Resample to fix class imbalance | `apr data balance train.jsonl --strategy oversample` |

## Common workflows

**Pre-finetune hygiene pass.**

```bash
apr data audit ./data/train.jsonl --json | jq '.warnings'
apr data decontaminate ./data/train.jsonl --against humaneval --against mbpp
apr data split ./data/train.jsonl --ratios 0.9,0.05,0.05 -o ./data/splits/
```

**Fix imbalanced classification dataset.**

```bash
apr data audit ./data/labels.jsonl --json | jq '.class_distribution'
apr data balance ./data/labels.jsonl --strategy oversample -o ./data/balanced.jsonl
```

## Troubleshooting

- **`decontaminate` flags too aggressively** — default n-gram threshold is
  strict. Tune via `--ngram-size 13 --min-overlap 0.5` for less-strict
  matching.
- **`split` produces a tiny val set** — small datasets + stratification can
  yield val sets too small to be useful. Use a held-out file instead.
- **`balance --strategy oversample` blows up dataset size** — major class
  imbalance. Try `undersample` for the majority class, or use
  `apr finetune --oversample` (in-loop oversampling) instead.

## See also

- Source: [`crates/apr-cli/src/commands/data.rs`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/commands/data.rs)
- Contract: [`contracts/apr-page-cli-data-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-page-cli-data-v1.yaml)

