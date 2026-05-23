<!-- PCU: cli-parity | contract: contracts/apr-page-cli-parity-v1.yaml -->

# apr parity

GPU/CPU parity check (PMAT-232: genchi genbutsu — see where GPU diverges)

**Category**: Quality & Evaluation

## Synopsis

```text
apr parity [OPTIONS]
```

## Example

<!-- example-cost: model-required model: model.gguf -->
```bash
apr parity model.gguf --backends cpu,gpu
```

## Full help

Run `apr parity --help` for the complete option list.

## See also

- Source: [`crates/apr-cli/src/commands/parity.rs`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/commands/parity.rs)
- Contract: [`contracts/apr-page-cli-parity-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-page-cli-parity-v1.yaml)
