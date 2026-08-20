<!-- PCU: cli-cgp | contract: contracts/apr-page-cli-cgp-v1.yaml -->

# apr cgp

Compute-graph profiling: profile, bench, roofline.

**Category**: Tools

This command surface previously shipped only as the standalone `aprender-cgp` binary.
Its command enum lived in a `main.rs`, which is importable by nothing, so `apr`
had no route to it at all. `apr cgp` and `aprender-cgp` now call the same
`dispatch` function, so the two surfaces cannot drift.

## Synopsis

```text
apr cgp <COMMAND>
```

## Subcommands

| Path |
|------|
| `apr cgp profile` |
| `apr cgp bench` |
| `apr cgp roofline` |
| `apr cgp diff` |
| `apr cgp contract` |
| `apr cgp trace` |
| `apr cgp explain` |
| `apr cgp tui` |
| `apr cgp baseline` |
| `apr cgp doctor` |
| `apr cgp compete` |

Every one of these is locked by `FALSIFY-CLI-006`: the list above and the built
binary are asserted to agree in both directions.

## Example

<!-- example-cost: trivial -->
```bash
apr cgp --help
```

## Full help

Run `apr cgp --help` for the complete option list.

## See also

- Source: [`crates/aprender-cgp/src/cli.rs`](https://github.com/paiml/aprender/blob/main/crates/aprender-cgp/src/cli.rs)
- Registry: [`contracts/apr-cli-commands-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-cli-commands-v1.yaml)
