<!-- PCU: cli-eval | contract: contracts/apr-page-cli-eval-v1.yaml -->

# apr eval

Evaluate model perplexity (spec H13: PPL <= 20) or classification metrics

**Category**: Quality & Evaluation

## Synopsis

```text
apr eval [OPTIONS]
```

## Example

```bash
apr eval qwen2.5-coder-1.5b-instruct-q4_k_m.gguf --suite humaneval --limit 5
```

## What this does

`apr eval` runs perplexity (PPL) on wikitext-2 or lambada by default, with a
default H13 threshold of PPL <= 20. With `--task classify` it runs F1 / accuracy
on a JSONL test set for fine-tuned classifiers. pass@k for code benchmarks
(humaneval, mbpp) is gated through `--samples N --temperature 0.8`.

## Key flags

| Flag | What it does | Example |
|------|-------------|---------|
| `--dataset NAME` | `wikitext-2`, `lambada`, or `custom` | `--dataset lambada` |
| `--threshold N` | PPL fail threshold (default 20) | `--threshold 15` |
| `--task TASK` | Omit for PPL, `classify` for F1/accuracy | `--task classify` |
| `--samples N` | pass@k samples per problem | `--samples 5` |
| `--temperature T` | Sampling temperature for pass@k | `--temperature 0.8` |
| `--device DEV` | `cpu` or `cuda` | `--device cuda` |
| `--generate-card` | Write HF model card on success | `--generate-card` |

## Common workflows

**Pre-publish PPL gate.**

```bash
apr eval qwen2.5-coder-1.5b.apr --dataset wikitext-2 --threshold 15 --json
# exit 0 = ship; non-zero = block
```

**pass@1 on HumanEval for a code model.**

```bash
apr eval qwen2.5-coder-7b.apr --dataset humaneval --samples 1 --temperature 0.0 \
    --device cuda --json
```

## Troubleshooting

- **PPL > 20 on a known-good base model** — confirm the chat template wasn't
  accidentally applied. PPL is for base models; chat-templated text inflates
  PPL by 2-5x.
- **HumanEval pass@1 is 0%** — the prompt-extraction heuristic may be missing
  (see [`§70 RC3 fix`](https://github.com/paiml/aprender/pull/1641)).
  Enable `APR_EVAL_DEBUG=1` to see the actual prompt sent to the model.
- **OOM on CUDA for 7B** — drop `--samples 1`, or use `--device cpu` (slower
  but bounded by RAM not VRAM).

## See also

- Source: [`crates/apr-cli/src/commands/eval.rs`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/commands/eval.rs)
- Contract: [`contracts/apr-page-cli-eval-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-page-cli-eval-v1.yaml)

