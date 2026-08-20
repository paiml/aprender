<!-- PCU: cli-zram | contract: contracts/apr-page-cli-zram-v1.yaml -->

# apr zram

zram device management.

**Category**: Tools

This command surface previously shipped only as the standalone `trueno-zram` binary.
Its command enum lived in a `main.rs`, which is importable by nothing, so `apr`
had no route to it at all. `apr zram` and `trueno-zram` now call the same
`dispatch` function, so the two surfaces cannot drift.

## Synopsis

```text
apr zram <COMMAND>
```

## Subcommands

| Path |
|------|
| `apr zram create` |
| `apr zram remove` |
| `apr zram status` |
| `apr zram benchmark` |

Every one of these is locked by `FALSIFY-CLI-006`: the list above and the built
binary are asserted to agree in both directions.

## Example

<!-- example-cost: trivial -->
```bash
apr zram --help
```

## Full help

Run `apr zram --help` for the complete option list.

## See also

- Source: [`crates/aprender-zram-cli/src/lib.rs`](https://github.com/paiml/aprender/blob/main/crates/aprender-zram-cli/src/lib.rs)
- Registry: [`contracts/apr-cli-commands-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-cli-commands-v1.yaml)
