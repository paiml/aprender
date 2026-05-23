<!-- PCU: cli-tokenize | contract: contracts/apr-page-cli-tokenize-v1.yaml -->

# apr tokenize

Tokenizer training pipeline (plan/apply) — BPE vocabulary learning

**Category**: Training

## Synopsis

```text
apr tokenize [OPTIONS]
```

## Example

<!-- example-cost: model-required model: qwen2.5-coder-1.5b -->
```bash
apr tokenize "Hello world" --tokenizer qwen2.5-coder-1.5b
```

## Full help

Run `apr tokenize --help` for the complete option list.

## See also

- Source: [`crates/apr-cli/src/commands/tokenize.rs`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/commands/tokenize.rs)
- Contract: [`contracts/apr-page-cli-tokenize-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-page-cli-tokenize-v1.yaml)
