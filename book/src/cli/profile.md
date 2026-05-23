<!-- PCU: cli-profile | contract: contracts/apr-page-cli-profile-v1.yaml -->

# apr profile

Deep profiling with Roofline analysis

**Category**: Quality & Evaluation

## Synopsis

```text
apr profile [OPTIONS]
```

## Example

<!-- example-cost: model-required model: qwen2.5-coder-1.5b-instruct-q4_k_m.gguf -->
```bash
apr profile qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
```

## Full help

Run `apr profile --help` for the complete option list.

## See also

- Source: [`crates/apr-cli/src/commands/profile.rs`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/commands/profile.rs)
- Contract: [`contracts/apr-page-cli-profile-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-page-cli-profile-v1.yaml)
