<!-- PCU: cli-tensors | contract: contracts/apr-page-cli-tensors-v1.yaml -->

# apr tensors

List tensor names and shapes

**Category**: Inspection

## Synopsis

```text
apr tensors [OPTIONS]
```

## Example

```bash
apr tensors qwen2.5-coder-1.5b-instruct-q4_k_m.gguf --json | jq length
```

## What this does

`apr tensors` lists every tensor in a model with name, shape, dtype, and offset.
With `--stats` it adds mean/std/min/max — the fastest way to spot a divergent
finetune or a quantization step that introduced NaN. Filter by name pattern with
`--filter` to scope to one layer or projection.

## Key flags

| Flag | What it does | Example |
|------|-------------|---------|
| `--stats` | Compute mean/std/min/max per tensor | `--stats` |
| `--filter PAT` | Substring filter on tensor name | `--filter "layers.0"` |
| `--limit N` | Cap the number of rows | `--limit 20` |
| `--json` | JSON output for scripting | `--json` |

## Common workflows

**Confirm a finetune actually changed the weights.**

```bash
apr tensors qwen2.5-coder-0.5b.apr      --stats --json > before.json
apr tensors qwen2.5-coder-0.5b-ft.apr   --stats --json > after.json
jq -s '.[0] - .[1]' before.json after.json    # diffs cleanly when stats are identical
```

**Find the LM head and check its shape.**

```bash
apr tensors qwen2.5-coder-1.5b.apr --filter "lm_head"
# Expect [vocab_size, hidden_size] for tied embeddings
```

## Troubleshooting

- **Tensor count is half of the source** — `apr tensors` doesn't deduplicate
  shared (tied) tensors, but a converter might. Check `apr inspect --json | jq
  .tensor_count` against the source's reported count.
- **Stats show `nan` / `inf`** — bad weight. Trace with `apr trace --layer
  <name>` to find where corruption entered.
- **Long output for 30B models** — pipe through `--limit 50` or `--filter` to
  scope. The full list is 600+ rows for Qwen3 30B.

## See also

- Source: [`crates/apr-cli/src/commands/tensors.rs`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/commands/tensors.rs)
- Contract: [`contracts/apr-page-cli-tensors-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-page-cli-tensors-v1.yaml)

