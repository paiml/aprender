<!-- PCU: cli-quantize | contract: contracts/apr-page-cli-quantize-v1.yaml -->

# apr quantize

Quantize model weights (GH-243)

**Category**: Model Transform

## Synopsis

```text
apr quantize [OPTIONS]
```

## Example

```bash
apr quantize model.apr --to q4_k -o model-q4k.apr
```

## Full help

Run `apr quantize --help` for the complete option list.

## See also

- Source: [`crates/apr-cli/src/commands/quantize.rs`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/commands/quantize.rs)
- Contract: [`contracts/apr-page-cli-quantize-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-page-cli-quantize-v1.yaml)

<!-- TODO: walkthrough -->
