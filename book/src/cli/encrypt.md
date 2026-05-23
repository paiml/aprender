<!-- PCU: cli-encrypt | contract: contracts/apr-page-cli-encrypt-v1.yaml -->

# apr encrypt

Encrypt model weights at rest using BLAKE3-derived keystream + MAC.

**Category**: Model Transform

## Synopsis

```text
apr encrypt [OPTIONS]
```

## Example

<!-- example-cost: destructive -->
```bash
apr encrypt model.apr --key my-key -o model.enc
```

## Full help

Run `apr encrypt --help` for the complete option list.

## See also

- Source: [`crates/apr-cli/src/commands/encrypt.rs`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/commands/encrypt.rs)
- Contract: [`contracts/apr-page-cli-encrypt-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-page-cli-encrypt-v1.yaml)
