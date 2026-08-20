<!-- PCU: cli-rag | contract: contracts/apr-page-cli-rag-v1.yaml -->

# apr rag

RAG pipeline: index, query, transcribe.

**Category**: Pipeline

This command surface previously shipped only as the standalone `trueno-rag` binary.
Its command enum lived in a `main.rs`, which is importable by nothing, so `apr`
had no route to it at all. `apr rag` and `trueno-rag` now call the same
`dispatch` function, so the two surfaces cannot drift.

## Synopsis

```text
apr rag <COMMAND>
```

## Subcommands

| Path |
|------|
| `apr rag demo` |
| `apr rag index` |
| `apr rag query` |
| `apr rag transcribe` |
| `apr rag extract-frames` |
| `apr rag info` |

Every one of these is locked by `FALSIFY-CLI-006`: the list above and the built
binary are asserted to agree in both directions.

## Example

<!-- example-cost: trivial -->
```bash
apr rag --help
```

## Full help

Run `apr rag --help` for the complete option list.

## See also

- Source: [`crates/aprender-rag-cli/src/lib.rs`](https://github.com/paiml/aprender/blob/main/crates/aprender-rag-cli/src/lib.rs)
- Registry: [`contracts/apr-cli-commands-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-cli-commands-v1.yaml)
