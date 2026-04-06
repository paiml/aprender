# `batuta transpile`

Transpile source code to Rust using detected external transpilers (Phase 2: Transpilation).

## Synopsis

```bash
batuta transpile [OPTIONS]
```

## Description

The transpile command invokes external transpiler tools (Depyler for Python, Decy for C/C++, Bashrs for Shell) to convert source code to Rust. It supports incremental transpilation, caching, and an interactive Ruchy REPL for exploratory conversion.

This is Phase 2 of the 5-phase pipeline. It requires Phase 1 (Analysis) to be completed first.

## Options

| Option | Description |
|--------|-------------|
| `--incremental` | Enable incremental transpilation (only changed files) |
| `--cache` | Cache unchanged files to speed up re-runs |
| `--modules <MODULES>` | Transpile specific modules only |
| `--ruchy` | Generate Ruchy (gradual typing) instead of pure Rust |
| `--repl` | Start interactive Ruchy REPL after transpilation |
| `-v, --verbose` | Enable verbose output |
| `-h, --help` | Print help |

## External Transpilers

Batuta auto-detects transpilers in your PATH:

| Tool | Source Language | Install |
|------|---------------|---------|
| Depyler | Python | `cargo install depyler` |
| Decy | C/C++ | `cargo install decy` |
| Bashrs | Shell | `cargo install bashrs` |
| Ruchy | Gradual typing | `cargo install ruchy` |

## Examples

### Standard Transpilation

```bash
$ batuta transpile

🔄 Transpiling source code...
  Tool: depyler (Python → Rust)
  Source: ./src
  Output: ./rust-output

✅ Transpilation completed successfully!
```

### Incremental with Caching

```bash
$ batuta transpile --incremental --cache
```

### Ruchy Mode with REPL

```bash
$ batuta transpile --ruchy --repl

# After transpilation, drops into interactive REPL:
# ruchy> let x = 42
# ruchy> println!("{}", x)
```

### Specific Modules

```bash
$ batuta transpile --modules "auth,database,api"
```

## See Also

- [Phase 2: Transpilation](../part2/phase2-transpilation.md)
- [Tool Selection](../part2/tool-selection.md)
- [`batuta optimize`](./cli-optimize.md) - Next phase

---

**Previous:** [`batuta analyze`](./cli-analyze.md)
**Next:** [`batuta optimize`](./cli-optimize.md)
