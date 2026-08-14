<!-- PCU: cli-compute | contract: contracts/apr-cli-commands-v1.yaml -->

# apr compute

Load-test, monitor and benchmark the compute backends themselves — SIMD, wgpu
and CUDA — rather than a model.

**Category**: Analysis

This was the standalone `cbtop` binary ("Compute Block Top"). That binary is
gone; `cargo install aprender` now places exactly one program in `~/.cargo/bin`.

## Not to be confused with `apr cbtop`

`apr cbtop` is a *different* tool that happens to share the acronym:
"ComputeBrick Top", which profiles the brick pipeline of an LLM inference run
against a model file (`--model-path`, `--brick-score`, `--speculative`, …).
`apr compute` has none of those arguments and takes no model. The two were
given separate namespaces precisely so neither had to give up arguments.

## Synopsis

```text
apr compute top [-r MS] [-d IDX] [-b BACKEND] [-l PROFILE] [-w WORKLOAD]
                [-s N] [-t N] [--deterministic] [--show-fps] [-c FILE]
                [--headless] [--format json|text] [--duration SECS] [-o FILE]

apr compute bench [-b BACKEND] [-w WORKLOAD] [-s N] [-d SECS]
                  [-f json|text] [-o FILE] [--baseline FILE]
                  [--fail-on-regression PCT] [--compare a,b,c]

apr compute optimize baseline [-o FILE] [--quick] [-d SECS]
apr compute optimize analyze  [-b FILE] [-f text|json] [-o FILE]
apr compute optimize check    [-b FILE] [-t PCT] [--quick] [-f text|json]
```

Every flag, short form and default is the one the standalone binary used.
`apr compute top` is its no-subcommand mode: a TUI, or a single headless
benchmark under `--headless`.

## Arguments

### `top`

| Flag | Default | Meaning |
|------|---------|---------|
| `-r`, `--refresh <MS>` | `100` | TUI refresh rate in milliseconds |
| `-d`, `--device <IDX>` | `0` | GPU device index |
| `-b`, `--backend <B>` | `all` | `simd`, `wgpu`, `cuda`, `all` |
| `-l`, `--load <P>` | `idle` | `idle`, `light`, `medium`, `heavy`, `stress` |
| `-w`, `--workload <W>` | `gemm` | `gemm`, `conv`, `attention`, `bandwidth`, `elementwise`, `reduction`, `all` |
| `-s`, `--size <N>` | `1048576` | Problem size in elements |
| `-t`, `--threads <N>` | available parallelism | Thread count for SIMD |
| `--deterministic` | off | Deterministic mode, for testing |
| `--show-fps` | off | Show frame timing statistics |
| `-c`, `--config <FILE>` | none | Config file path |
| `--headless` | off | No TUI — run one benchmark and print it |
| `--format <F>` | `text` | `json` or `text` (headless mode) |
| `--duration <SECS>` | `5` | Benchmark duration (headless mode) |
| `-o`, `--output <FILE>` | stdout | Where to write the report (headless mode) |

`--format`, `--duration` and `-o` describe the headless run; in TUI mode they
are accepted and unused, exactly as the standalone binary accepted them.

### `bench`

| Flag | Default | Meaning |
|------|---------|---------|
| `-b`, `--backend <B>` | `simd` | Backend to benchmark |
| `-w`, `--workload <W>` | `gemm` | `gemm`, `dot`, `elementwise`, `reduction` |
| `-s`, `--size <N>` | `1048576` | Problem size in elements |
| `-d`, `--duration <SECS>` | `5` | Benchmark duration |
| `-f`, `--format <F>` | `json` | `json` or `text` |
| `-o`, `--output <FILE>` | stdout | Where to write the report |
| `--baseline <FILE>` | none | Compare against a saved result; **exit 1 on regression** |
| `--fail-on-regression <PCT>` | `5.0` | Regression threshold, in percent |
| `--compare <a,b,c>` | none | Benchmark several backends and print a comparison |

Note `bench`'s defaults differ from `top`'s on purpose: `simd` rather than
`all`, and `json` rather than `text`.

A `--baseline` that does not exist is **refused**, not treated as "no baseline,
therefore no regression" — that would turn the gate green precisely when its
input went missing.

### `optimize`

`baseline` collects measurements across the configuration matrix and saves them;
`analyze` reports bottlenecks from a saved baseline; `check` re-measures and
compares, exiting non-zero when a regression exceeds `--threshold`.

## Examples

<!-- example-cost: trivial -->
```bash
apr compute bench --backend simd --workload gemm --size 65536 --duration 1
```

<!-- example-cost: trivial -->
```bash
apr compute top --headless --format json --duration 1
```

<!-- example-cost: moderate -->
```bash
apr compute optimize baseline --quick -o /tmp/baseline.json
apr compute optimize check -b /tmp/baseline.json --quick --threshold 5
```

## Exit codes

`0` on success. `apr compute bench --baseline` and `apr compute optimize check`
exit `1` when a regression is detected — the codes the standalone binary used,
passed through unchanged so existing CI gates keep working.

## Full help

Run `apr compute --help`, or `apr compute <SUBCOMMAND> --help`, for the
complete option list.

## See also

- Different tool, similar name: [`apr cbtop`](./cbtop.md)
- Kernel and backend profiling: [`apr perf`](./perf.md)
- Source: [`crates/aprender-cbtop/src/cli.rs`](https://github.com/paiml/aprender/blob/main/crates/aprender-cbtop/src/cli.rs)
