<!-- PCU: cli-inspect | contract: contracts/apr-page-cli-inspect-v1.yaml -->

# apr inspect

Inspect model metadata, vocab, and structure

**Category**: Inspection

## Synopsis

```text
apr inspect [OPTIONS]
```

## Example

```bash
apr inspect qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
```

## What this does

`apr inspect` reads the model header (APR, GGUF, or SafeTensors) and prints
architecture, vocab size, tensor count, license, and HF identity metadata.
With `--quality` it emits a 0-100 score covering physics (no NaN/Inf), structural
completeness, provenance, and tokenizer presence — the SHIP-TWO-001 §84
ship-gate rubric. This is the first command to run on any model you didn't build
yourself.

## Key flags

| Flag | What it does | Example |
|------|-------------|---------|
| `--quality` | Print 0-100 SHIP-TWO score block | `--quality` |
| `--weights` | Tensor statistics (mean, std, min, max) | `--weights` |
| `--vocab` | Tokenizer / vocab details | `--vocab` |
| `--filters` | Filter/security checks | `--filters` |
| `--json` | Machine-readable output | `--json | jq .arch` |

## Common workflows

**Pre-ship gate: confirm score >= 90 before publishing to HF.**

```bash
apr inspect qwen2.5-coder-1.5b.apr --quality --json | jq '.score'
# Must be >= 90 per AC-SHIP2-007
```

**Diff two checkpoints' metadata after a finetune.**

```bash
apr inspect qwen2.5-coder-0.5b.apr      --json > before.json
apr inspect qwen2.5-coder-0.5b-ft.apr   --json > after.json
diff <(jq -S . before.json) <(jq -S . after.json)
```

## Troubleshooting

- **Quality score < 90** — drill into the breakdown. Missing `data_source` or
  `data_license` is the most common cause (provenance is 20 points). Fix with
  `apr stamp <model> --field data_source=...`.
- **`hf_architecture` field missing** — common on models converted before PMAT-690.
  Re-convert from the original SafeTensors with current `apr convert` to get the
  stamp.
- **NaN/Inf detected in weights** — almost always a training divergence or a
  bad quantization step. Trace back with `apr tensors --stats --filter <name>`.

## See also

- Source: [`crates/apr-cli/src/commands/inspect.rs`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/commands/inspect.rs)
- Contract: [`contracts/apr-page-cli-inspect-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-page-cli-inspect-v1.yaml)

