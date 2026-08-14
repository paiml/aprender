<!-- PCU: cli-ptx-debug | contract: contracts/apr-cli-commands-v1.yaml -->

# apr ptx-debug

Pure-Rust PTX static analysis: score a PTX file against the 100-point Popperian
falsification framework, and generate FKR regression tests from it.

**Category**: Analysis

This was the standalone `trueno-ptx-debug` binary. That binary is gone; the
capability is reached as `apr ptx-debug`.

## Three PTX commands, three jobs

| Command | Job |
|---------|-----|
| [`apr ptx`](./ptx.md) | Register pressure, memory and roofline analysis of a PTX file or named kernel |
| [`apr ptx-map`](./ptx-map.md) | Which kernel a model's layers and steps dispatch to |
| `apr ptx-debug` | Falsification **score** for a PTX file, plus FKR test generation |

## Synopsis

```text
apr ptx-debug analyze <FILE> [--falsify] [--min-score N] [--html FILE] [--json]
apr ptx-debug gen-fkr <FILE> [-o FILE]
```

## Arguments

### `analyze`

| Flag | Default | Meaning |
|------|---------|---------|
| `<FILE>` | required | Path to the PTX source file to analyze |
| `--falsify` | off | Run the full 100-point framework. Accepted for command-line compatibility: the full framework is always evaluated, so this selects behaviour that is already the default |
| `--min-score <N>` | `70` | Report failure (exit 2) when the score is below N |
| `--html <FILE>` | none | Also write a standalone HTML report to this path |
| `--json` | off | Emit the report as JSON instead of human-readable text |

### `gen-fkr`

| Flag | Default | Meaning |
|------|---------|---------|
| `<FILE>` | required | Path to the PTX source file to generate tests from |
| `-o`, `--output <FILE>` | stdout | Write the generated tests here |

`--output` is a new long alias for the standalone binary's `-o`; `-o` itself is
unchanged.

## Exit codes

| Code | Meaning |
|------|---------|
| `0` | Score is 90 or above |
| `1` | Score is at or above `--min-score` but below 90 |
| `2` | Score is below `--min-score` |
| `3` | A critical bug was detected (outranks 1 and 2) |
| `3` (apr) | The PTX file was not found — apr's not-found code |
| `4` (apr) | The PTX source did not parse |

This table is the standalone binary's, passed through unchanged, because it is
the interface CI gates were written against. Its old `--help` also advertised
codes `10` (parse error) and `11` (I/O error); those were never emitted — every
error exited `1` — so they are not listed.

Note the analyzer is deliberately permissive: it does not refuse a file for
lacking `.version` / `.target`. That is the falsification framework's job, as
tests F001/F002/F003, and it shows up in the score rather than as a parse error.

## Examples

<!-- example-cost: trivial -->
```bash
apr ptx-debug analyze kernel.ptx --falsify
```

<!-- example-cost: trivial -->
```bash
apr ptx-debug analyze kernel.ptx --min-score 90 --html /tmp/report.html
```

<!-- example-cost: trivial -->
```bash
apr ptx-debug gen-fkr kernel.ptx -o tests/kernel_fkr.rs
```

## Full help

Run `apr ptx-debug --help`, or `apr ptx-debug <SUBCOMMAND> --help`, for the
complete option list.

## See also

- [`apr ptx`](./ptx.md), [`apr ptx-map`](./ptx-map.md)
- Source: [`crates/aprender-ptx-debug/src/cli.rs`](https://github.com/paiml/aprender/blob/main/crates/aprender-ptx-debug/src/cli.rs)
