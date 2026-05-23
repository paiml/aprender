<!-- PCU: cli-hex | contract: contracts/apr-page-cli-hex-v1.yaml -->

# apr hex

Format-aware binary forensics (10X better than xxd)

**Category**: Inspection

## Synopsis

```text
apr hex [OPTIONS]
```

## Example

<!-- example-cost: model-required model: qwen2.5-coder-1.5b-instruct-q4_k_m.gguf -->
```bash
apr hex qwen2.5-coder-1.5b-instruct-q4_k_m.gguf | head -20
```

## Full help

Run `apr hex --help` for the complete option list.

## See also

- Source: [`crates/apr-cli/src/commands/hex.rs`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/commands/hex.rs)
- Contract: [`contracts/apr-page-cli-hex-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-page-cli-hex-v1.yaml)
