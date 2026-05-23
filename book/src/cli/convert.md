<!-- PCU: cli-convert | contract: contracts/apr-page-cli-convert-v1.yaml -->

# apr convert

Convert/optimize model

**Category**: Model Transform

## Synopsis

```text
apr convert [OPTIONS]
```

## Example

<!-- example-cost: model-required model: model.safetensors -->
```bash
apr convert model.safetensors --quantize q4_k -o model-q4k.apr
```

## Full help

Run `apr convert --help` for the complete option list.

## See also

- Source: [`crates/apr-cli/src/commands/convert.rs`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/commands/convert.rs)
- Contract: [`contracts/apr-page-cli-convert-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-page-cli-convert-v1.yaml)
