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

## Full help

Run `apr tensors --help` for the complete option list.

## See also

- Source: [`crates/apr-cli/src/commands/tensors.rs`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/commands/tensors.rs)
- Contract: [`contracts/apr-page-cli-tensors-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-page-cli-tensors-v1.yaml)

<!-- TODO: walkthrough -->
