# `batuta analyze`

Analyze source codebase for languages, dependencies, and technical debt (Phase 1: Analysis).

## Synopsis

```bash
batuta analyze [OPTIONS] [PATH]
```

## Description

The analyze command performs deep codebase analysis including language detection, dependency mapping, and Technical Debt Grade (TDG) scoring. This is Phase 1 of the transpilation pipeline.

## Arguments

| Argument | Description |
|----------|-------------|
| `[PATH]` | Project path to analyze (default: `.`) |

## Options

| Option | Description |
|--------|-------------|
| `--tdg` | Generate Technical Debt Grade score |
| `--languages` | Detect and report programming languages |
| `--dependencies` | Analyze project dependencies |
| `-v, --verbose` | Enable verbose output |
| `-h, --help` | Print help |

## Examples

### Full Analysis

```bash
$ batuta analyze --languages --tdg .

📊 Analyzing project...

Languages:
  Python: 42 files (8,521 lines)
  Shell:  12 files (1,234 lines)
  C:       3 files (567 lines)

Technical Debt Grade: B (78.5/100)
  Complexity: 12.3 avg cyclomatic
  SATD: 8 comments
  Dead code: 3.2%
```

### TDG Score Only

```bash
$ batuta analyze --tdg .

📊 Analysis Results
  Files: 508 total, 184,673 lines
  Languages: Rust (95%), TOML (3%), Markdown (2%)
  TDG Score: 98.4 (Grade: A+)
```

Note: `--tdg` automatically detects languages and counts files. You don't need to pass `--languages` separately.

### Language Detection Only

```bash
$ batuta analyze --languages
```

### Dependency Analysis

```bash
$ batuta analyze --dependencies
```

## See Also

- [Phase 1: Analysis](../part2/phase1-analysis.md)
- [TDG Scoring](../part2/tdg-scoring.md)
- [`batuta transpile`](./cli-transpile.md) - Next phase

---

**Previous:** [`batuta init`](./cli-init.md)
**Next:** [`batuta transpile`](./cli-transpile.md)
