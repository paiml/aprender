<!-- PCU: cli-run | contract: contracts/apr-page-cli-run-v1.yaml -->

# apr run

Run model directly (auto-download, cache, execute)

**Category**: Inference

## Synopsis

```text
apr run [OPTIONS]
```

## Example

```bash
apr run qwen2.5-coder-1.5b "What is 2+2?" --max-tokens 16
```

## Full help

Run `apr run --help` for the complete option list.

## See also

- Source: [`crates/apr-cli/src/commands/run.rs`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/commands/run.rs)
- Contract: [`contracts/apr-page-cli-run-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-page-cli-run-v1.yaml)

