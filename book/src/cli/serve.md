<!-- PCU: cli-serve | contract: contracts/apr-page-cli-serve-v1.yaml -->

# apr serve

Inference server (plan/run)

**Category**: Inference

## Synopsis

```text
apr serve [OPTIONS]
```

## Example

<!-- example-cost: model-required model: qwen2.5-coder-1.5b -->
```bash
apr serve run qwen2.5-coder-1.5b --port 8080
```

## Full help

Run `apr serve --help` for the complete option list.

## See also

- Source: [`crates/apr-cli/src/commands/serve/mod.rs`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/commands/serve/mod.rs)
- Contract: [`contracts/apr-page-cli-serve-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-page-cli-serve-v1.yaml)
