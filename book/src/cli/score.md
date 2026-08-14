# apr score

Quality-score a Rust TUI crate across six weighted dimensions and print a
report, JSON, or YAML.

**Category**: Quality & Verification

## Synopsis

```text
apr score [OPTIONS] [PATH]
```

## Where this came from

`apr score` is the whole of the former `score` binary
(`aprender-present-terminal`, `--features score`), the SPEC-024 §18.10 TUI
quality scorer.

`score` is a bare English word. Published, it installed as `~/.cargo/bin/score`
and collided with anything else that wanted the name. The capability is
unchanged; only the address is.

Not to be confused with [`apr present score`](./present.md), which scores a
Presentar YAML **manifest** rather than a Rust crate.

## Dimensions

| Dimension | Points | What it measures |
|-----------|--------|------------------|
| Performance | 25 | SIMD patterns, `ComputeBlock` usage, zero-allocation types, benchmarks |
| Testing | 20 | `#[test]` density, proptest, golden/pixel tests, assertion count |
| Widget Reuse | 15 | `presentar_terminal::` imports, `impl Widget`/`impl Brick` |
| Code Coverage | 15 | `cargo llvm-cov` line coverage, estimated from test count if unavailable |
| Quality Metrics | 15 | clippy warning count, `cargo fmt --check`, doc-comment density |
| Falsifiability | 10 | `F-XXX-000` identifiers, "fails if" criteria, benchmark assertions |

Grades: `A` ≥ 90, `B` ≥ 80, `C` ≥ 70, `D` ≥ 60, otherwise `F`.

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `[PATH]` | `.` | Crate root to analyse; must contain a `Cargo.toml` |

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `-o, --output <FORMAT>` | `text` | Report format: `text`, `json`, or `yaml` |
| `--ci` | off | Exit **1** if the score is below `--threshold` |
| `--threshold <N>` | `80` | Minimum passing score, 0-100 |
| `--no-color` | off | Disable ANSI colour in the text report |
| `--config <PATH>` | none | YAML file overriding the six dimension weights |
| `-v, --verbose` | off | Print the raw metrics behind each dimension |
| `-q, --quiet` | off | Print only the final score, one number |

`-v` and `-q` are apr's global flags. The `score` binary declared its own
identically-named, identically-behaved pair; they are now inherited rather
than redeclared. The global `--json` is equivalent to `--output json`.

## Exit codes

| Code | Meaning |
|------|---------|
| 0 | Report produced (and, under `--ci`, the threshold was met) |
| 1 | `--ci` was given and the score is below `--threshold` (F-PMAT-007/008) |
| 5 | `PATH` is not a Rust crate — no `Cargo.toml` found (F-PMAT-017) |

## Examples

```bash
apr score crates/aprender-present-terminal
```

```bash
# Machine-readable, for a dashboard.
apr score crates/apr-cli --output json
```

```bash
# Gate a build. Exits 1 below 85.
apr score . --ci --threshold 85
```

```bash
# Just the number.
apr score . --quiet
```

## Full help

Run `apr score --help` for the complete option list.

## See also

- [`apr qualify`](./qualify.md) — model qualification gates
- [`apr present score`](./present.md) — quality score for a Presentar manifest
- Source: [`crates/apr-cli/src/commands/present.rs`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/commands/present.rs)
- Implementation: [`crates/aprender-present-terminal/src/tools/score.rs`](https://github.com/paiml/aprender/blob/main/crates/aprender-present-terminal/src/tools/score.rs)
