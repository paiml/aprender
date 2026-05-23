<!-- PCU: cli-validate | contract: contracts/apr-page-cli-validate-v1.yaml -->

# apr validate

Validate model integrity and quality

**Category**: Inspection

## Synopsis

```text
apr validate [OPTIONS]
```

## Example

<!-- example-cost: model-required model: qwen2.5-coder-1.5b-instruct-q4_k_m.gguf -->
```bash
apr validate qwen2.5-coder-1.5b-instruct-q4_k_m.gguf --quality
```

## Full help

Run `apr validate --help` for the complete option list.

## See also

- Source: [`crates/apr-cli/src/commands/validate.rs`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/commands/validate.rs)
- Contract: [`contracts/apr-page-cli-validate-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-page-cli-validate-v1.yaml)
