<!-- PCU: cli-merge | contract: contracts/apr-page-cli-merge-v1.yaml -->

# apr merge

Merge multiple models

**Category**: Model Transform

## Synopsis

```text
apr merge [OPTIONS]
```

## Example

```bash
apr merge model1.apr model2.apr --strategy weighted --weights 0.7,0.3 -o merged.apr
```

## What this does

`apr merge` combines two or more compatible models into one — same architecture,
same vocab, same shape. Supports the standard model-merging strategies:
`average`, `weighted`, `slerp`, `ties` (trim by task-vector magnitude), `dare`
(drop and rescale). Used to fold finetune deltas back into the base model, or
to assemble a "best of N" checkpoint across runs.

## Key flags

| Flag | What it does | Example |
|------|-------------|---------|
| `--strategy S` | `average`, `weighted`, `slerp`, `ties`, `dare` | `--strategy slerp` |
| `--weights LIST` | Weights for `weighted` | `--weights 0.7,0.3` |
| `--base-model M` | Required for TIES / DARE | `--base-model qwen2.5-coder-0.5b.apr` |
| `--drop-rate R` | DARE drop probability (default 0.9) | `--drop-rate 0.7` |
| `--density D` | TIES trim density (default 0.2) | `--density 0.3` |
| `--seed N` | RNG seed for DARE | `--seed 42` |
| `--plan` | Validate inputs + show plan | `--plan` |

## Common workflows

**Linearly blend two finetunes weighted 70/30.**

```bash
apr merge qwen-ft-rust.apr qwen-ft-python.apr \
    --strategy weighted --weights 0.7,0.3 \
    -o qwen-ft-blend.apr
```

**TIES-merge two task finetunes against the base.**

```bash
apr merge qwen-ft-code.apr qwen-ft-chat.apr \
    --strategy ties --base-model qwen2.5-coder-1.5b.apr --density 0.2 \
    -o qwen-ft-ties.apr
```

## Troubleshooting

- **"shape mismatch"** — all input models must share the exact architecture
  and quantization scheme. Cross-check with `apr inspect` on each. F32 + Q4K
  cannot merge.
- **TIES / DARE require `--base-model`** — the strategy computes task vectors
  as deltas from a base. Pass it explicitly.
- **Merged model fails `apr qa`** — merge can degrade quality if weight
  spaces aren't aligned. Try `slerp` for two models or fall back to weighted
  averaging.

## See also

- Source: [`crates/apr-cli/src/commands/merge.rs`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/commands/merge.rs)
- Contract: [`contracts/apr-page-cli-merge-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-page-cli-merge-v1.yaml)

