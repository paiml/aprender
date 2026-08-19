<!-- PCU: cli-pv | contract: contracts/apr-page-cli-pv-v1.yaml -->

# apr pv

Provable-contracts: validate, lint, score, kani.

**Category**: Tools

This command surface previously shipped only as the standalone `pv` binary.
Its command enum lived in a `main.rs`, which is importable by nothing, so `apr`
had no route to it at all. `apr pv` and `pv` now call the same
`dispatch` function, so the two surfaces cannot drift.

## Synopsis

```text
apr pv <COMMAND>
```

## Subcommands

| Path |
|------|
| `apr pv explain` |
| `apr pv validate` |
| `apr pv check-parity` |
| `apr pv scaffold` |
| `apr pv extract-pytorch` |
| `apr pv codegen` |
| `apr pv kani` |
| `apr pv probar` |
| `apr pv status` |
| `apr pv audit` |
| `apr pv diff` |
| `apr pv coverage` |
| `apr pv generate` |
| `apr pv graph` |
| `apr pv equations` |
| `apr pv lean` |
| `apr pv lean-status` |
| `apr pv proof-status` |
| `apr pv lint` |
| `apr pv score` |
| `apr pv query` |
| `apr pv invariants` |
| `apr pv coq` |
| `apr pv fuzz` |
| `apr pv mirai` |
| `apr pv flux` |
| `apr pv tla` |
| `apr pv book` |
| `apr pv infer` |
| `apr pv unlock` |
| `apr pv roofline` |
| `apr pv pipeline` |
| `apr pv kaizen` |
| `apr pv certify` |
| `apr pv verify-structure` |
| `apr pv verify-pipeline` |
| `apr pv verify-bindings` |
| `apr pv migrate` |

Every one of these is locked by `FALSIFY-CLI-006`: the list above and the built
binary are asserted to agree in both directions.

## Example

<!-- example-cost: trivial -->
```bash
apr pv --help
```

## Full help

Run `apr pv --help` for the complete option list.

## See also

- Source: [`crates/aprender-contracts-cli/src/lib.rs`](https://github.com/paiml/aprender/blob/main/crates/aprender-contracts-cli/src/lib.rs)
- Registry: [`contracts/apr-cli-commands-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-cli-commands-v1.yaml)
