<!-- PCU: cli-compile | contract: contracts/apr-page-cli-compile-v1.yaml -->

# apr compile

Compile model into standalone executable (APR-SPEC §4.16)

**Category**: Model Transform

## Synopsis

```text
apr compile [OPTIONS]
```

## Example

```bash
apr compile model.apr --target cuda -o compiled.apr
```

## Full help

Run `apr compile --help` for the complete option list.

## See also

- Source: [`crates/apr-cli/src/commands/compile.rs`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/commands/compile.rs)
- Contract: [`contracts/apr-page-cli-compile-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-page-cli-compile-v1.yaml)

<!-- TODO: walkthrough -->
