<!-- PCU: cli-prune | contract: contracts/apr-page-cli-prune-v1.yaml -->

# apr prune

Prune model (structured/unstructured pruning) (GH-247)

**Category**: Model Transform

## Synopsis

```text
apr prune [OPTIONS]
```

## Example

```bash
apr prune model.apr --ratio 0.5 -o pruned.apr
```

## What this does

`apr prune` removes weights that contribute least to model output. `magnitude`
zeros the smallest-absolute-value weights; `structured` removes whole channels;
`depth` removes whole transformer blocks; `wanda` / `sparsegpt` use calibration
data for a better quality/sparsity trade. Output is a smaller, faster model
(after a sparse-aware kernel pass) or the same size with zeros (for sparsity
experiments).

## Key flags

| Flag | What it does | Example |
|------|-------------|---------|
| `-m, --method M` | `magnitude`, `structured`, `depth`, `width`, `wanda`, `sparsegpt` | `--method wanda` |
| `--target-ratio R` | Fraction to prune (default 0.5) | `--target-ratio 0.3` |
| `--remove-layers RANGE` | Layer range for depth pruning | `--remove-layers 20-24` |
| `--calibration FILE` | Calibration JSONL for Wanda / SparseGPT | `--calibration calib.jsonl` |
| `--analyze` | Analyze pruning opportunities (no write) | `--analyze` |
| `--plan` | Estimate only | `--plan` |

## Common workflows

**Magnitude-prune a 1.5B model to 50% sparsity.**

```bash
apr prune qwen2.5-coder-1.5b.apr --method magnitude --target-ratio 0.5 \
    -o qwen-pruned.apr
apr eval qwen-pruned.apr --dataset wikitext-2 --threshold 25
```

**Wanda-prune with calibration data for better quality.**

```bash
apr prune qwen2.5-coder-1.5b.apr --method wanda --target-ratio 0.5 \
    --calibration calib.jsonl -o qwen-wanda.apr
apr eval qwen-wanda.apr --dataset wikitext-2     # Wanda typically beats magnitude
```

## Troubleshooting

- **Pruned model is slower than the original** — sparse kernels only speed
  things up at 70%+ structured sparsity. At 50% magnitude (unstructured),
  expect the same speed but smaller disk after compression.
- **PPL doubles after pruning** — too aggressive a ratio or no calibration.
  Drop to 0.3 ratio or switch to `wanda` with calibration data.
- **`--remove-layers` produces a broken model** — depth pruning is risky;
  the deleted layers' residuals propagate. Always `apr check` afterward.

## See also

- Source: [`crates/apr-cli/src/commands/prune.rs`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/commands/prune.rs)
- Contract: [`contracts/apr-page-cli-prune-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-page-cli-prune-v1.yaml)

