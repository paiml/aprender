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

## Full help

Run `apr eval --help` for the complete option list.

## See also

- Source: [`crates/apr-cli/src/commands/eval.rs`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/commands/eval.rs)
- Contract: [`contracts/apr-page-cli-eval-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-page-cli-eval-v1.yaml)

