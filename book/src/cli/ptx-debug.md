# apr ptx-debug

PTX falsification analysis and FKR test generation (was the `aprender-ptx-debug` binary)

**Category**: Hardware

Pure Rust: unlike `apr ptx`, it needs no build feature.

## Synopsis

```text
apr ptx-debug analyze <FILE> [--falsify] [--min-score N] [--html FILE] [--json]
apr ptx-debug gen-fkr <FILE> [-o FILE]
apr ptx-debug version
```

- `analyze` scores a PTX file against the falsification framework and fails when the score is below `--min-score` (default 70).
- `gen-fkr` generates FKR tests for jugar-probar from a PTX file (stdout unless `-o`).

## Example

<!-- example-cost: trivial -->
```bash
apr ptx-debug version
```

## Full help

Run `apr ptx-debug --help` for the complete option list.

## See also

- Source: [`crates/aprender-ptx-debug/src/cli.rs`](https://github.com/paiml/aprender/blob/main/crates/aprender-ptx-debug/src/cli.rs)
- [apr ptx](./ptx.md)
