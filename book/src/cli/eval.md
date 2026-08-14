<!-- PCU: cli-eval | contract: contracts/apr-page-cli-eval-v1.yaml -->

# apr eval

Evaluate model perplexity (spec H13: PPL <= 20) or classification metrics

**Category**: Quality & Evaluation

## Synopsis

```text
apr eval [OPTIONS]
```

## Example

<!-- example-cost: model-required model: qwen2.5-coder-1.5b-instruct-q4_k_m.gguf -->
```bash
apr eval --help
```

## Full help

Run `apr eval --help` for the complete option list.

## See also

- Source: [`crates/apr-cli/src/commands/eval/mod.rs`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/commands/eval/mod.rs)
- Contract: [`contracts/apr-page-cli-eval-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-page-cli-eval-v1.yaml)
