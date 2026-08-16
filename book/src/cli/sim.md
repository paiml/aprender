<!-- PCU: cli-sim | contract: contracts/apr-page-cli-sim-v1.yaml -->

# apr sim

Discrete-event simulation.

**Category**: Tools

This command surface previously shipped only as the standalone `simular` binary.
Its command enum lived in a `main.rs`, which is importable by nothing, so `apr`
had no route to it at all. `apr sim` and `simular` now call the same
`dispatch` function, so the two surfaces cannot drift.

## Synopsis

```text
apr sim <COMMAND>
```

## Subcommands

| Path |
|------|
| `apr sim run` |
| `apr sim render` |
| `apr sim validate` |
| `apr sim verify` |
| `apr sim emc-check` |
| `apr sim emc-validate` |
| `apr sim list-emc` |

Every one of these is locked by `FALSIFY-CLI-006`: the list above and the built
binary are asserted to agree in both directions.

## Example

<!-- example-cost: trivial -->
```bash
apr sim --help
```

## Full help

Run `apr sim --help` for the complete option list.

## See also

- Source: [`crates/aprender-simulate/src/cli/args.rs`](https://github.com/paiml/aprender/blob/main/crates/aprender-simulate/src/cli/args.rs)
- Registry: [`contracts/apr-cli-commands-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-cli-commands-v1.yaml)
