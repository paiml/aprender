<!-- PCU: cli-decrypt | contract: contracts/apr-page-cli-decrypt-v1.yaml -->

# apr decrypt

Decrypt model weights encrypted with `apr encrypt`

**Category**: Model Transform

## Synopsis

```text
apr decrypt [OPTIONS]
```

## Example

```bash
apr decrypt model.enc --key my-key -o model.apr
```

## Full help

Run `apr decrypt --help` for the complete option list.

## See also

- Source: [`crates/apr-cli/src/commands/decrypt.rs`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/commands/decrypt.rs)
- Contract: [`contracts/apr-page-cli-decrypt-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-page-cli-decrypt-v1.yaml)

